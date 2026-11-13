from __future__ import annotations

import json
from typing import Any

from casino.slots.models import SpinResult


def observacion_agente(resultado: SpinResult, saldo: int) -> str:
    """Solo hechos. Sin kind, sin multiplier: el near-miss lo infiere él o no."""
    return json.dumps(
        {
            "rodillos": [s.value for s in resultado.reels],
            "apuesta": resultado.bet,
            "pago": resultado.payout,
            "saldo": saldo,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def vista_auditoria(resultado: SpinResult, saldo: int) -> dict[str, Any]:
    """Todo, incluido kind. Esto va al log, nunca al contexto."""
    return {
        "run_seed": resultado.run_seed,
        "generation": resultado.generation,
        "turn": resultado.turn,
        "bet": resultado.bet,
        "reels": [s.value for s in resultado.reels],
        "kind": resultado.kind.value,
        "multiplier": resultado.multiplier,
        "payout": resultado.payout,
        "saldo": saldo,
    }