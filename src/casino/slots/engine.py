import hashlib
import random

from casino.slots.models import Outcome, SpinRequest, SpinResult
from casino.slots.spec import (
    GAME_NAME,
    PAYTABLE,
    STRIPS,
    TWO_CHERRY_MULTIPLIER,
    OutcomeKind,
    Symbol,
)

Reels = tuple[Symbol, Symbol, Symbol]


def derive_rng(run_seed: str, generation: int, turn: int) -> random.Random:
    # hashlib, nunca hash(): hash() va salado por proceso y rompe el replay.
    # La apuesta no entra acá: eso garantiza que apostar más no cambia los rodillos.
    material = f"{run_seed}|{generation}|{turn}|{GAME_NAME}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def draw_reels(rng: random.Random) -> Reels:
    drawn = [strip[rng.randrange(len(strip))] for strip in STRIPS]
    return (drawn[0], drawn[1], drawn[2])


def evaluate(reels: Reels) -> Outcome:
    a, b, c = reels

    if a == b == c:
        kind = OutcomeKind.JACKPOT if a == Symbol.SEVEN else OutcomeKind.WIN
        return Outcome(kind=kind, multiplier=PAYTABLE[a])

    # Near-miss posicional: SEVEN en 1 y 2, fallo en 3. Paga cero.
    if a == Symbol.SEVEN and b == Symbol.SEVEN and c != Symbol.SEVEN:
        return Outcome(kind=OutcomeKind.NEAR_MISS, multiplier=0)

    if sum(s == Symbol.CHERRY for s in reels) == 2:
        return Outcome(kind=OutcomeKind.WIN, multiplier=TWO_CHERRY_MULTIPLIER)

    return Outcome(kind=OutcomeKind.LOSS, multiplier=0)


def settle(bet: int, outcome: Outcome) -> int:
    return bet * outcome.multiplier


def spin(request: SpinRequest) -> SpinResult:
    rng = derive_rng(request.run_seed, request.generation, request.turn)
    reels = draw_reels(rng)
    outcome = evaluate(reels)
    return SpinResult(
        **request.model_dump(),
        reels=reels,
        kind=outcome.kind,
        multiplier=outcome.multiplier,
        payout=settle(request.bet, outcome),
    )