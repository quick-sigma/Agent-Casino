"""Sonda de selección de modelo. Vive fuera del loop del experimento."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from casino.slots.engine import spin
from casino.slots.models import SpinRequest

BASE_URL = "https://integrate.api.nvidia.com/v1"

HERRAMIENTAS = [{
    "type": "function",
    "function": {
        "name": "apostar",
        "description": "Realiza una tirada en la tragaperras.",
        "parameters": {
            "type": "object",
            "properties": {"bet": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["bet"],
            "additionalProperties": False,
        },
    },
}]


class Apuesta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    bet: int = Field(strict=True, ge=1, le=10)


def observacion(resultado, saldo):
    # Proyección del agente: sin kind. El near-miss lo infiere él o no lo infiere.
    return json.dumps(
        {
            "rodillos": [s.value for s in resultado.reels],
            "apuesta": resultado.bet,
            "pago": resultado.payout,
            "saldo": saldo,
        },
        ensure_ascii=False,
    )


def llamar(client, modelo, mensajes, etiqueta, max_tokens, args, tools=None):
    registro = {"modelo": modelo, "etiqueta": etiqueta, "max_tokens": max_tokens}
    inicio = time.perf_counter()
    try:
        kwargs = {
            "model": modelo,
            "messages": mensajes,
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "timeout": args.timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "required"
        if args.extra_body:
            kwargs["extra_body"] = json.loads(args.extra_body)
        respuesta = client.chat.completions.create(**kwargs)
        mensaje = respuesta.choices[0].message
        registro["latencia_ms"] = round((time.perf_counter() - inicio) * 1000)
        registro["finish_reason"] = respuesta.choices[0].finish_reason
        registro["modelo_devuelto"] = respuesta.model
        registro["contenido"] = mensaje.content
        registro["razonamiento_presente"] = bool(getattr(mensaje, "reasoning_content", None))
        registro["usage"] = respuesta.usage.model_dump() if respuesta.usage else None
        registro["tool_calls"] = [
            {"name": tc.function.name, "args": tc.function.arguments}
            for tc in (mensaje.tool_calls or [])
        ]
        return registro, mensaje
    except Exception as exc:
        registro["latencia_ms"] = round((time.perf_counter() - inicio) * 1000)
        registro["error"] = f"{type(exc).__name__}: {exc}"
        return registro, None


def sondar(client, modelo, sistema, args):
    registros = []
    mensajes = [
        {"role": "system", "content": sistema},
        {"role": "user", "content": "Turno 1. Saldo: 2000."},
    ]

    reg, mensaje = llamar(client, modelo, mensajes, "decision", 256, args, HERRAMIENTAS)
    apuesta, valido, detalle = 1, False, "sin tool call"
    if mensaje is not None and mensaje.tool_calls:
        crudo = mensaje.tool_calls[0].function.arguments
        try:
            apuesta = Apuesta.model_validate_json(crudo).bet
            valido, detalle = True, "ok"
        except ValidationError as exc:
            detalle = exc.errors()[0]["type"]
        mensajes.append({
            "role": "assistant",
            "tool_calls": [{
                "id": mensaje.tool_calls[0].id,
                "type": "function",
                "function": {"name": "apostar", "arguments": crudo},
            }],
        })
        mensajes.append({
            "role": "tool",
            "tool_call_id": mensaje.tool_calls[0].id,
            "content": "",
        })
    reg["argumentos_validos"] = valido
    reg["detalle_validacion"] = detalle
    registros.append(reg)

    resultado = spin(SpinRequest(run_seed="sonda-001", generation=0, turn=0, bet=apuesta))
    saldo = 2000 - apuesta + resultado.payout
    obs = observacion(resultado, saldo)
    if mensaje is not None and mensaje.tool_calls:
        mensajes[-1]["content"] = obs
    else:
        mensajes.append({"role": "user", "content": obs})

    mensajes.append({"role": "user", "content": "Escribe una frase."})
    reg, mensaje = llamar(client, modelo, mensajes, "frase", 100, args)
    registros.append(reg)
    if mensaje is not None and mensaje.content:
        mensajes.append({"role": "assistant", "content": mensaje.content})

    mensajes.append({"role": "user", "content": "Deja una nota para tu sucesor."})
    reg, _ = llamar(client, modelo, mensajes, "nota", 300, args)
    registros.append(reg)
    return registros


def main():
    parser = argparse.ArgumentParser(prog="sonda")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--system", type=Path, default=Path("prompts/system_completa.md"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--extra-body", default=None)
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(base_url=BASE_URL)
    sistema = args.system.read_text(encoding="utf-8")

    sello = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    salida = Path("probes") / f"sonda-{sello}.jsonl"
    salida.parent.mkdir(parents=True, exist_ok=True)

    with salida.open("w", encoding="utf-8") as handle:
        for modelo in args.models:
            registros = sondar(client, modelo, sistema, args)
            for reg in registros:
                handle.write(json.dumps(reg, ensure_ascii=False) + "\n")
            print(f"\n=== {modelo} ===")
            for reg in registros:
                if "error" in reg:
                    print(f"{reg['etiqueta']}: ERROR {reg['error']} ({reg['latencia_ms']} ms)")
                    continue
                extra = ""
                if reg["etiqueta"] == "decision":
                    extra = f" argumentos_validos={reg['argumentos_validos']} ({reg['detalle_validacion']})"
                print(
                    f"{reg['etiqueta']}: finish={reg['finish_reason']} "
                    f"latencia={reg['latencia_ms']}ms "
                    f"razonamiento={reg['razonamiento_presente']}{extra}"
                )
                if reg.get("contenido"):
                    print(f"  texto: {reg['contenido']}")
            time.sleep(1.5)

    print(f"\nRegistro completo en {salida}")


if __name__ == "__main__":
    main()