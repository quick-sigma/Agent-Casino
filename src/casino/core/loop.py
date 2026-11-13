from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from casino.core.config import Settings
from casino.core.eventos import EventLog
from casino.core.llm import LLMClient
from casino.core.observacion import observacion_agente, vista_auditoria
from casino.core.tools import Apuesta, herramientas, tools_hash
from casino.slots.engine import spin
from casino.slots.models import SpinRequest


class GuardiaExcedida(RuntimeError):
    pass


class _Presupuesto:
    def __init__(self, settings: Settings) -> None:
        self._max_llamadas = settings.guardas.max_llamadas_api
        self._max_segundos = settings.guardas.max_segundos
        self._inicio = time.monotonic()
        self.llamadas = 0

    def consumir(self) -> None:
        self.llamadas += 1
        if self.llamadas > self._max_llamadas:
            raise GuardiaExcedida("max_llamadas_api excedido")
        if time.monotonic() - self._inicio > self._max_segundos:
            raise GuardiaExcedida("max_segundos excedido")


def _hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def ejecutar(
    settings: Settings,
    config_path: Path,
    cliente: LLMClient | None = None,
) -> dict[str, Any]:
    salida = Path(settings.run.output_dir)
    if salida.exists():
        raise FileExistsError(
            f"{salida} ya existe: cambia run_id o borra el directorio. "
            "Un run no se sobreescribe."
        )
    salida.mkdir(parents=True)

    archivo_sistema = (
        Path(settings.prompts.prompts_dir)
        / f"system_{settings.experimento.visibilidad_tabla}.md"
    )
    sistema = archivo_sistema.read_text(encoding="utf-8")
    tools = herramientas()

    shutil.copy(config_path, salida / "config.resuelta.toml")
    (salida / "prompts").mkdir()
    shutil.copy(archivo_sistema, salida / "prompts" / archivo_sistema.name)

    hashes = {
        "config_hash": settings.config_hash(),
        "system_prompt_hash": _hash_texto(sistema),
        "tools_hash": tools_hash(),
    }
    (salida / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": settings.run.run_id,
                "model_requested": settings.modelo.model,
                "started_at": datetime.now(timezone.utc).isoformat(),
                **hashes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cruda_dir = salida / "api_cruda"
    if settings.run.guardar_api_cruda:
        cruda_dir.mkdir()

    def guardar_cruda(nombre: str, resultado: Any) -> None:
        if settings.run.guardar_api_cruda:
            (cruda_dir / f"{nombre}.json").write_text(
                json.dumps(resultado.raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    exp = settings.experimento
    presupuesto = _Presupuesto(settings)
    cliente = cliente if cliente is not None else LLMClient(settings)

    saldo = exp.dinero_inicial
    nota_heredada: str | None = None
    motivo = "completo"
    generaciones_completadas = 0

    with EventLog(salida / "eventos.jsonl", settings.run.run_id) as log:
        log.append("run_start", model_requested=settings.modelo.model, **hashes)
        try:
            for gen in range(exp.generaciones):
                if not exp.hereda_dinero:
                    saldo = exp.dinero_inicial
                if saldo <= 0:
                    motivo = "ruina"
                    break

                mensajes: list[dict[str, Any]] = [
                    {"role": "system", "content": sistema}
                ]
                if exp.hereda_nota and nota_heredada:
                    mensajes.append({"role": "user", "content": nota_heredada})

                log.append(
                    "generation_start",
                    generation=gen,
                    saldo_inicial=saldo,
                    # El log refleja la realidad: con hereda_nota=False la nota
                    # existe pero NO se inyecta en el contexto del sucesor.
                    nota_heredada=nota_heredada if exp.hereda_nota else None,
                )

                ruina = False
                for turn in range(exp.turnos_por_generacion):
                    mensajes.append(
                        {
                            "role": "user",
                            "content": settings.prompts.plantilla_turno.format(
                                turno=turn + 1, saldo=saldo
                            ),
                        }
                    )

                    apuesta: int | None = None
                    tool_call = None
                    argumentos = ""
                    for intento in range(1 + settings.guardas.max_reintentos_malformed):
                        presupuesto.consumir()
                        res = cliente.decision(mensajes, tools)
                        guardar_cruda(f"g{gen:02d}-t{turn:02d}-decision-{intento}", res)
                        log.append(
                            "decision",
                            generation=gen,
                            turn=turn,
                            intento=intento,
                            latency_ms=res.latency_ms,
                            finish_reason=res.finish_reason,
                            model_returned=res.model_returned,
                            tool_calls=[tc.model_dump() for tc in res.tool_calls],
                        )
                        # El contexto NO se toca entre reintentos: nada de errores
                        # inyectados que le enseñen el schema a mitad de generación.
                        if not res.tool_calls:
                            log.append(
                                "tool_rejected",
                                generation=gen,
                                turn=turn,
                                intento=intento,
                                motivo="sin_tool_call",
                            )
                            continue
                        tool_call = res.tool_calls[0]
                        argumentos = tool_call.arguments
                        try:
                            candidata = Apuesta.model_validate_json(argumentos).bet
                        except ValidationError as exc:
                            log.append(
                                "tool_rejected",
                                generation=gen,
                                turn=turn,
                                intento=intento,
                                motivo=exc.errors()[0]["type"],
                                argumentos_crudos=argumentos,
                            )
                            continue
                        if candidata > saldo:
                            log.append(
                                "tool_rejected",
                                generation=gen,
                                turn=turn,
                                intento=intento,
                                motivo="apuesta_excede_saldo",
                                argumentos_crudos=argumentos,
                            )
                            continue
                        apuesta = candidata
                        break

                    if apuesta is None or tool_call is None:
                        log.append(
                            "turno_fallido", generation=gen, turn=turn, saldo=saldo
                        )
                        continue

                    resultado = spin(
                        SpinRequest(
                            run_seed=settings.run.run_seed,
                            generation=gen,
                            turn=turn,
                            bet=apuesta,
                        )
                    )
                    saldo = saldo - apuesta + resultado.payout
                    log.append("spin", **vista_auditoria(resultado, saldo))

                    mensajes.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": "apostar",
                                        "arguments": argumentos,
                                    },
                                }
                            ],
                        }
                    )
                    mensajes.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": observacion_agente(resultado, saldo),
                        }
                    )

                    if saldo < 0:
                        raise RuntimeError("saldo_invalido: el orquestador dejó saldo negativo")
                    if saldo == 0:
                        ruina = True
                        break

                    mensajes.append(
                        {"role": "user", "content": settings.prompts.prompt_frase}
                    )
                    presupuesto.consumir()
                    fres = cliente.phrase(mensajes)
                    guardar_cruda(f"g{gen:02d}-t{turn:02d}-frase", fres)
                    log.append(
                        "phrase",
                        generation=gen,
                        turn=turn,
                        contenido=fres.content,
                        finish_reason=fres.finish_reason,
                        truncada=fres.finish_reason == "length",
                        latency_ms=fres.latency_ms,
                    )
                    if fres.content:
                        mensajes.append({"role": "assistant", "content": fres.content})

                if ruina and exp.hereda_dinero:
                    # Herencia ACTIVADA: la quiebra es terminal (la siguiente
                    # generación arrancaría con saldo 0). Aun así el agente deja
                    # un último mensaje con el MISMO prompt de las frases.
                    mensajes.append(
                        {"role": "user", "content": settings.prompts.prompt_frase}
                    )
                    presupuesto.consumir()
                    fres = cliente.phrase(mensajes)
                    guardar_cruda(f"g{gen:02d}-cierre", fres)
                    log.append(
                        "phrase",
                        generation=gen,
                        turn=turn,
                        contenido=fres.content,
                        finish_reason=fres.finish_reason,
                        truncada=fres.finish_reason == "length",
                        latency_ms=fres.latency_ms,
                    )
                    nota_heredada = None
                    log.append(
                        "generation_end",
                        generation=gen,
                        saldo_final=saldo,
                        motivo="ruina",
                    )
                    motivo = "ruina"
                    break

                # Cierre normal, o ruina con hereda_dinero=False: en ambos casos
                # el agente deja una nota. Con la herencia DESACTIVADA la
                # siguiente generación reinicia el saldo, así que el experimento
                # continúa en vez de detenerse.
                mensajes.append(
                    {"role": "user", "content": settings.prompts.prompt_nota}
                )
                presupuesto.consumir()
                nres = cliente.note(mensajes)
                guardar_cruda(f"g{gen:02d}-nota", nres)
                log.append(
                    "note",
                    generation=gen,
                    contenido=nres.content,
                    finish_reason=nres.finish_reason,
                    truncada=nres.finish_reason == "length",
                    latency_ms=nres.latency_ms,
                )
                nota_heredada = nres.content or None

                if ruina:
                    # Se registró la nota; solo se inyectará en el sucesor si
                    # hereda_nota está activo (ver arriba). El run NO se detiene.
                    log.append(
                        "generation_end",
                        generation=gen,
                        saldo_final=saldo,
                        motivo="ruina",
                    )
                    continue

                generaciones_completadas += 1
                log.append(
                    "generation_end",
                    generation=gen,
                    saldo_final=saldo,
                    motivo="completo",
                )
        except GuardiaExcedida as exc:
            motivo = "abortado"
            log.append("run_abort", motivo=str(exc), llamadas_api=presupuesto.llamadas)
        except Exception as exc:
            motivo = "abortado"
            log.append(
                "run_abort",
                motivo=f"{type(exc).__name__}: {exc}",
                llamadas_api=presupuesto.llamadas,
            )
            raise
        finally:
            log.append(
                "run_end",
                motivo=motivo,
                llamadas_api=presupuesto.llamadas,
                saldo_final=saldo,
                generaciones_completadas=generaciones_completadas,
            )

    return {
        "run_id": settings.run.run_id,
        "motivo": motivo,
        "llamadas_api": presupuesto.llamadas,
        "saldo_final": saldo,
        "generaciones_completadas": generaciones_completadas,
    }