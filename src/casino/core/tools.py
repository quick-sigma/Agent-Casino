from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from casino.slots.spec import  MIN_BET


class Apuesta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bet: int = Field(strict=True, ge=MIN_BET)


def _sin_titles(nodo: Any) -> Any:
    if isinstance(nodo, dict):
        return {k: _sin_titles(v) for k, v in nodo.items() if k != "title"}
    if isinstance(nodo, list):
        return [_sin_titles(x) for x in nodo]
    return nodo


def herramientas() -> list[dict[str, Any]]:
    schema = _sin_titles(Apuesta.model_json_schema())
    if "maximum" in schema["properties"]["bet"]:
        del schema["properties"]["bet"]["maximum"]
    return [
        {
            "type": "function",
            "function": {
                "name": "apostar",
                "description": "Realiza una tirada en la tragaperras.",
                "parameters": schema,
            },
        }
    ]

def tools_hash() -> str:
    canonico = json.dumps(
        herramientas(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()