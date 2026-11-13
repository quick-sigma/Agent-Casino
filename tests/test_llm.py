import copy
import time
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from casino.core import llm
from casino.core.config import Settings
from casino.core.llm import RateLimiter


def make_settings(enable_thinking: bool = False) -> Settings:
    return Settings.model_validate(
        {
            "run": {
                "run_id": "test-llm",
                "run_seed": "test-seed",
                "output_dir": "runs/test-llm",
                "guardar_api_cruda": True,
            },
            "modelo": {
                "model": "test-model",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "TEST_API_KEY",
                "temperature": 1.0,
                "top_p": 1.0,
                "model_seed": 1234,
                "enable_thinking": enable_thinking,
                "max_tokens_decision": 256,
                "tool_choice": "required",
                "timeout_s": 60,
                "max_retries": 3,
            },
            "experimento": {
                "dinero_inicial": 2000,
                "turnos_por_generacion": 5,
                "generaciones": 2,
                "hereda_dinero": False,
                "hereda_nota": True,
                "max_tokens_frase": 100,
                "max_tokens_nota": 300,
                "visibilidad_tabla": "completa",
            },
            "prompts": {
                "prompts_dir": "prompts",
                "prompt_frase": "Escribe una frase.",
                "prompt_nota": "Deja una nota para tu sucesor.",
                "plantilla_turno": "Turno {turno}. Saldo: {saldo}.",
            },
            "guardas": {
                "max_llamadas_api": 200,
                "max_segundos": 900,
                "max_reintentos_malformed": 2,
                # Alto para que el espaciado mínimo no duerma en los tests unitarios.
                "max_rpm": 1000,
            },
        }
    )


class FakeUsage:
    def model_dump(self, mode=None):
        return {"total_tokens": 7}


class FakeResponse:
    def __init__(
        self,
        content=None,
        tool_calls=None,
        finish_reason="stop",
        reasoning_content=None,
        model="test-model-returned",
    ):
        self.model = model
        self.usage = FakeUsage()
        message = SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        self.choices = [
            SimpleNamespace(message=message, finish_reason=finish_reason)
        ]

    def model_dump(self, mode=None):
        return {
            "model": self.model,
            "finish_reason": self.choices[0].finish_reason,
        }


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def fake_tool_call(
    id="call_1",
    name="apostar",
    arguments='{"bet": 1}',
):
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_rate_limiter_espera_al_llenar_ventana():
    lim = RateLimiter(rpm=2, window_s=0.3)
    inicio = time.monotonic()
    lim.acquire()
    lim.acquire()
    assert time.monotonic() - inicio < 0.05
    lim.acquire()
    assert time.monotonic() - inicio >= 0.25


def test_rate_limiter_espacia_llamadas(monkeypatch):
    dormidos: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: dormidos.append(s))
    lim = RateLimiter(rpm=2, window_s=60.0, min_interval_s=5.0)
    lim.acquire()
    lim.acquire()
    assert len(dormidos) == 1
    assert 0 < dormidos[0] <= 5.0


class _Fake429(Exception):
    """Sustituye a openai.RateLimitError para no depender de httpx2 en los tests."""


def test_429_aplica_cooldown_largo_y_no_reintento_corto(monkeypatch):
    monkeypatch.setattr(llm, "RateLimitError", _Fake429)
    dormidos: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: dormidos.append(s))

    class Completions:
        def __init__(self) -> None:
            self.llamadas = 0

        def create(self, **kwargs):
            self.llamadas += 1
            if self.llamadas == 1:
                raise _Fake429("429")
            return FakeResponse(content="ok")

    completions = Completions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = llm.LLMClient(make_settings(), client=fake)

    result = client.phrase([{"role": "user", "content": "hola"}])

    assert completions.llamadas == 2
    assert dormidos == [60.0]  # cooldown largo, no los 0.5/1/2 s del SDK
    assert result.content == "ok"
    assert result.latency_ms >= 0


def test_429_persistente_aborta_con_mensaje_claro(monkeypatch):
    monkeypatch.setattr(llm, "RateLimitError", _Fake429)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    class Siempre429:
        def create(self, **kwargs):
            raise _Fake429("429")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Siempre429()))
    client = llm.LLMClient(make_settings(), client=fake)

    with pytest.raises(llm.RateLimitPenalizado, match="429 persistente"):
        client.phrase([{"role": "user", "content": "hola"}])


def test_decision_sends_tools_and_required():
    settings = make_settings()
    response = FakeResponse(
        content=None,
        tool_calls=[fake_tool_call()],
        finish_reason="tool_calls",
    )
    fake = FakeClient(response)
    client = llm.LLMClient(settings, client=fake)

    tools = [{"type": "function", "function": {"name": "apostar"}}]
    messages = [{"role": "user", "content": "Turno 1. Saldo: 2000."}]

    result = client.decision(messages, tools)
    kwargs = fake.chat.completions.calls[0]

    assert kwargs["tools"] is tools
    assert kwargs["tool_choice"] == "required"
    assert kwargs["max_tokens"] == 256
    assert kwargs["seed"] == 1234
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 1.0

    assert result.kind == "decision"
    assert result.finish_reason == "tool_calls"
    assert result.content is None
    assert result.model_returned == "test-model-returned"
    assert result.tool_calls[0].arguments == '{"bet": 1}'


def test_phrase_omits_tools_and_uses_cap():
    settings = make_settings()
    fake = FakeClient(FakeResponse(content="frase"))
    client = llm.LLMClient(settings, client=fake)

    messages = [{"role": "user", "content": "Escribe una frase."}]
    result = client.phrase(messages)
    kwargs = fake.chat.completions.calls[0]

    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert kwargs["max_tokens"] == 100
    assert result.content == "frase"
    assert result.kind == "phrase"


def test_note_omits_tools_and_uses_cap():
    settings = make_settings()
    fake = FakeClient(FakeResponse(content="nota"))
    client = llm.LLMClient(settings, client=fake)

    messages = [{"role": "user", "content": "Deja una nota para tu sucesor."}]
    result = client.note(messages)
    kwargs = fake.chat.completions.calls[0]

    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert kwargs["max_tokens"] == 300
    assert result.content == "nota"
    assert result.kind == "note"


def test_phrase_ignores_tools_argument():
    settings = make_settings()
    fake = FakeClient(FakeResponse(content="frase"))
    client = llm.LLMClient(settings, client=fake)

    messages = [{"role": "user", "content": "Escribe una frase."}]
    client.complete("phrase", messages, tools=[{"type": "function"}])
    kwargs = fake.chat.completions.calls[0]

    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_decision_requires_tools():
    settings = make_settings()
    fake = FakeClient(FakeResponse())
    client = llm.LLMClient(settings, client=fake)

    with pytest.raises(llm.LLMContractError):
        client.decision([{"role": "user", "content": "hola"}], tools=None)


def test_invalid_kind_raises():
    settings = make_settings()
    fake = FakeClient(FakeResponse())
    client = llm.LLMClient(settings, client=fake)

    with pytest.raises(llm.LLMContractError):
        client.complete("bad", [{"role": "user", "content": "hola"}])


def test_reasoning_leak_raises_when_disabled():
    settings = make_settings(enable_thinking=False)
    response = FakeResponse(content="frase", reasoning_content="hidden")
    client = llm.LLMClient(settings, client=FakeClient(response))

    with pytest.raises(llm.ReasoningLeakError):
        client.phrase([{"role": "user", "content": "Escribe una frase."}])


def test_reasoning_allowed_when_enabled_not_in_content():
    settings = make_settings(enable_thinking=True)
    response = FakeResponse(content="frase", reasoning_content="hidden")
    client = llm.LLMClient(settings, client=FakeClient(response))

    result = client.phrase([{"role": "user", "content": "Escribe una frase."}])

    assert result.reasoning_present is True
    assert result.content == "frase"
    assert "hidden" not in result.content


def test_messages_not_mutated():
    settings = make_settings()
    fake = FakeClient(FakeResponse(content="frase"))
    client = llm.LLMClient(settings, client=fake)

    messages = [{"role": "user", "content": "Escribe una frase."}]
    before = copy.deepcopy(messages)

    client.phrase(messages)

    assert messages == before


def test_empty_phrase_is_returned_not_raised():
    settings = make_settings()
    response = FakeResponse(content="", finish_reason="length")
    client = llm.LLMClient(settings, client=FakeClient(response))

    result = client.phrase([{"role": "user", "content": "Escribe una frase."}])

    assert result.content == ""
    assert result.finish_reason == "length"


def test_raw_and_usage_captured():
    settings = make_settings()
    response = FakeResponse(content="frase")
    client = llm.LLMClient(settings, client=FakeClient(response))

    result = client.phrase([{"role": "user", "content": "Escribe una frase."}])

    assert result.raw == response.model_dump()
    assert result.usage == {"total_tokens": 7}


def test_latency_nonnegative():
    settings = make_settings()
    fake = FakeClient(FakeResponse(content="frase"))
    client = llm.LLMClient(settings, client=fake)

    result = client.phrase([{"role": "user", "content": "Escribe una frase."}])

    assert result.latency_ms >= 0


def test_no_factory_when_client_injected(monkeypatch):
    settings = make_settings()

    def boom(*args, **kwargs):
        raise AssertionError("no debe resolver API key ni crear OpenAI")

    monkeypatch.setattr(llm, "resolve_api_key", boom)
    monkeypatch.setattr(llm, "OpenAI", boom)

    fake = FakeClient(FakeResponse(content="ok"))
    llm.LLMClient(settings, client=fake)


def test_factory_uses_timeout_and_retries(monkeypatch):
    settings = make_settings()
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        llm,
        "resolve_api_key",
        lambda settings: SecretStr("x"),
    )

    llm.LLMClient(settings)

    assert captured["api_key"] == "x"
    assert captured["base_url"] == settings.modelo.base_url
    assert captured["timeout"] == settings.modelo.timeout_s
    assert captured["max_retries"] == settings.modelo.max_retries