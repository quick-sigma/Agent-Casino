from fractions import Fraction
from itertools import product

from pydantic import BaseModel, ConfigDict

from casino.slots.engine import evaluate
from casino.slots.spec import STRIPS, OutcomeKind, Symbol


class AnalyticReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_combinations: int
    kind_counts: dict[OutcomeKind, int]
    payout_sum: int
    rtp: Fraction
    near_miss_rate: float
    win_rate: float


def analytic_model(
    strips: tuple[tuple[Symbol, ...], ...] = STRIPS,
) -> AnalyticReport:
    # Enumeración exacta, sin muestreo: 20^3 = 8000 combinaciones equiprobables.
    counts = {kind: 0 for kind in OutcomeKind}
    payout_sum = 0
    total = 0

    for combo in product(*strips):
        outcome = evaluate(combo)
        counts[outcome.kind] += 1
        payout_sum += outcome.multiplier
        total += 1

    wins = counts[OutcomeKind.WIN] + counts[OutcomeKind.JACKPOT]
    return AnalyticReport(
        total_combinations=total,
        kind_counts=counts,
        payout_sum=payout_sum,
        rtp=Fraction(payout_sum, total),
        near_miss_rate=counts[OutcomeKind.NEAR_MISS] / total,
        win_rate=wins / total,
    )