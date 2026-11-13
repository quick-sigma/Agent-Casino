"""Diagnostica un 429 de NVIDIA NIM con UNA sola petición.

Uso:  uv run python scripts/sonda_429.py

IMPORTANTE: NIM extiende el bloqueo con cada petición que recibe estando en
429. Ejecuta esto una sola vez. Si la respuesta es 429, deja la key quieta
30-60 min antes de volver a probar o de relanzar un run: bajar `max_rpm` no
sirve si la cuenta ya está en "penalty box".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3.5-bolt-e4b"


def main() -> int:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Falta OPENAI_API_KEY en .env o en el entorno.", file=sys.stderr)
        return 2

    # max_retries=0: que el SDK no dispare los 0.5/1/2 s que reinician el timer.
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=30, max_retries=0)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Responde solo: OK"}],
            max_tokens=8,
        )
        print("HTTP 200 → la key responde; el 429 fue por volumen/ráfagas.")
        print("modelo:", getattr(resp, "model", "?"))
        uso = getattr(resp, "usage", None)
        print("uso:", uso.model_dump() if uso is not None else "(sin usage)")
        return 0
    except RateLimitError as exc:
        print("HTTP 429 → la key está siendo limitada por NIM.")
        print("mensaje:", str(exc)[:300])
        response = getattr(exc, "response", None)
        if response is not None:
            headers = {
                k: v
                for k, v in response.headers.items()
                if "retry" in k.lower() or "ratelimit" in k.lower() or k.lower().startswith("x-")
            }
            print("headers de rate limit:", json.dumps(headers, ensure_ascii=False))
        print()
        print("Si no hay Retry-After ni X-RateLimit-*, es el comportamiento habitual")
        print("del free tier: bloqueo por cuenta, no una ventana limpia de 40 RPM.")
        print("Espera 30-60 min SIN llamar y vuelve a probar una sola vez.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
