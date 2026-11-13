from __future__ import annotations

import time
from collections import deque
from typing import Any, Literal

from openai import OpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field

from casino.core.config import Settings, resolve_api_key
import uuid 

CallKind = Literal["decision", "phrase", "note"]


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: str


class LLMResult(BaseModel):
    # chocar con el namespace reservado de Pydantic.
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    kind: CallKind
    model_requested: str
    model_returned: str | None = None
    latency_ms: int = Field(ge=0)
    finish_reason: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    reasoning_present: bool = False
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMContractError(ValueError):
    """El cliente recibió una petición que viola el contrato del experimento."""


class ReasoningLeakError(RuntimeError):
    """El modelo devolvió razonamiento cuando la config exigía no razonamiento."""


class RateLimitPenalizado(RuntimeError):
    """NIM mantuvo el 429 tras los cooldowns: la key está en penalty box."""


class RateLimiter:
    """Ventana deslizante: nunca más de `rpm` llamadas en `window_s`.

    `min_interval_s` añade un espaciado mínimo entre llamadas. Una ventana
    pura permite ráfagas (N llamadas en un segundo y luego silencio), y NIM
    penaliza justo ese patrón además del volumen total.
    """

    def __init__(
        self,
        rpm: int,
        window_s: float = 60.0,
        min_interval_s: float | None = None,
    ) -> None:
        self._capacity = rpm
        self._window = window_s
        self._min_interval = min_interval_s
        self._stamps: deque[float] = deque()

    def acquire(self) -> None:
        self._prune()
        if len(self._stamps) >= self._capacity:
            espera = self._window - (time.monotonic() - self._stamps[0])
            if espera > 0:
                time.sleep(espera)
            self._prune()
        if self._min_interval is not None and self._stamps:
            espera = self._min_interval - (time.monotonic() - self._stamps[-1])
            if espera > 0:
                time.sleep(espera)
        self._stamps.append(time.monotonic())

    def cooldown(self, segundos: float) -> None:
        """Espera larga tras un 429 y reinicia la ventana.

        NIM no manda `Retry-After` y cada petición hecha durante el bloqueo
        reinicia el contador: limpiar los stamps evita soltar una ráfaga nueva
        en cuanto termina la espera.
        """
        if segundos > 0:
            time.sleep(segundos)
        self._stamps.clear()

    def _prune(self) -> None:
        ahora = time.monotonic()
        while self._stamps and ahora - self._stamps[0] >= self._window:
            self._stamps.popleft()


class LLMClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings

        if client is None:
            secret = resolve_api_key(settings)
            client = OpenAI(
                api_key=secret.get_secret_value(),
                base_url=settings.modelo.base_url,
                timeout=settings.modelo.timeout_s,
                max_retries=settings.modelo.max_retries,
            )

        self._client = client
        self._limiter = RateLimiter(
            settings.guardas.max_rpm,
            min_interval_s=60.0 / settings.guardas.max_rpm,
        )

    def decision(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        return self.complete("decision", messages, tools=tools)

    def phrase(self, messages: list[dict[str, Any]]) -> LLMResult:
        return self.complete("phrase", messages)

    def note(self, messages: list[dict[str, Any]]) -> LLMResult:
        return self.complete("note", messages)

    def complete(
        self,
        kind: CallKind,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        if kind not in ("decision", "phrase", "note"):
            raise LLMContractError(f"tipo de llamada inválido: {kind!r}")

        modelo = self._settings.modelo
        experimento = self._settings.experimento

        max_tokens = {
            "decision": modelo.max_tokens_decision,
            "phrase": experimento.max_tokens_frase,
            "note": experimento.max_tokens_nota,
        }[kind]

        kwargs: dict[str, Any] = {
            "model": modelo.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": modelo.temperature,
            "top_p": modelo.top_p,
            "seed": modelo.model_seed,
        }

        if kind == "decision":
            if not tools:
                raise LLMContractError("la llamada de decisión requiere tools")
            kwargs["tools"] = tools
            kwargs["tool_choice"] = modelo.tool_choice

        # Los 429 los gestionamos nosotros: el SDK, sin `Retry-After` (NIM no lo
        # manda), reintenta a 0.5/1/2 s, peticiones reales que el limitador no ve
        # y que además reinician el penalty box de NVIDIA.
        guardas = self._settings.guardas
        response = None
        inicio = 0.0
        ultimo_429: RateLimitError | None = None
        for intento in range(1 + guardas.max_reintentos_429):
            self._limiter.acquire()
            inicio = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except RateLimitError as exc:
                ultimo_429 = exc
                if intento < guardas.max_reintentos_429:
                    self._limiter.cooldown(guardas.cooldown_429_s * (2**intento))

        if response is None:
            raise RateLimitPenalizado(
                f"429 persistente tras {guardas.max_reintentos_429} cooldowns "
                f"(base {guardas.cooldown_429_s:g}s). NVIDIA NIM suele bloquear la "
                "key entre ~30 y 60 min y cada nueva petición reinicia el timer: "
                "espera antes de relanzar, prueba otra key o revisa la cuenta."
            ) from ultimo_429
        latency_ms = round((time.perf_counter() - inicio) * 1000)

        if not response.choices:
            raise LLMContractError("la respuesta no trae choices")

        choice = response.choices[0]
        message = choice.message

        reasoning_present = bool(
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        if reasoning_present and not modelo.enable_thinking:
            raise ReasoningLeakError(
                "el modelo devolvió razonamiento pero enable_thinking=false"
            )

        tool_calls = [
            ToolCallRecord(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in (getattr(message, "tool_calls", None) or [])
        ]

        usage = (
            response.usage.model_dump(mode="json")
            if getattr(response, "usage", None) is not None
            else None
        )
        raw = (
            response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else {}
        )

        return LLMResult(
            kind=kind,
            model_requested=modelo.model,
            model_returned=getattr(response, "model", None),
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None),
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            reasoning_present=reasoning_present,
            usage=usage,
            raw=raw,
        )