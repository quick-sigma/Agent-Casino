from enum import StrEnum


class Symbol(StrEnum):
    CHERRY = "CHERRY"
    LEMON = "LEMON"
    GRAPE = "GRAPE"
    BELL = "BELL"
    SEVEN = "SEVEN"
    DIAMOND = "DIAMOND"


class OutcomeKind(StrEnum):
    JACKPOT = "JACKPOT"
    WIN = "WIN"
    NEAR_MISS = "NEAR_MISS"
    LOSS = "LOSS"


GAME_NAME = "slots"

_REEL_1 = (
    (Symbol.CHERRY,) * 5
    + (Symbol.LEMON,) * 4
    + (Symbol.GRAPE,) * 3
    + (Symbol.BELL,) * 2
    + (Symbol.SEVEN,) * 5
    + (Symbol.DIAMOND,) * 1
)
_REEL_2 = _REEL_1
_REEL_3 = (
    (Symbol.CHERRY,) * 6
    + (Symbol.LEMON,) * 5
    + (Symbol.GRAPE,) * 4
    + (Symbol.BELL,) * 3
    + (Symbol.SEVEN,) * 1
    + (Symbol.DIAMOND,) * 1
)

STRIPS: tuple[tuple[Symbol, ...], ...] = (_REEL_1, _REEL_2, _REEL_3)

PAYTABLE: dict[Symbol, int] = {
    Symbol.CHERRY: 5,
    Symbol.LEMON: 8,
    Symbol.GRAPE: 12,
    Symbol.BELL: 20,
    Symbol.DIAMOND: 40,
    Symbol.SEVEN: 60,
}

TWO_CHERRY_MULTIPLIER = 3

MIN_BET = 1