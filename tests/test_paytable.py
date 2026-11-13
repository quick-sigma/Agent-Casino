import itertools

import pytest

from casino.slots.engine import evaluate
from casino.slots.spec import (
    PAYTABLE, STRIPS, TWO_CHERRY_MULTIPLIER, OutcomeKind, Symbol,
)

from conftest import EXPECTED_PAYTABLE, EXPECTED_STRIPS, EXPECTED_TWO_CHERRY


def test_strips_and_paytable_match_spec():
    # Sin esto, el oráculo analítico no describe al motor real.
    assert tuple(tuple(s.value for s in strip) for strip in STRIPS) == EXPECTED_STRIPS
    assert {s.value: m for s, m in PAYTABLE.items()} == EXPECTED_PAYTABLE
    assert TWO_CHERRY_MULTIPLIER == EXPECTED_TWO_CHERRY


def test_every_reel_combination_has_defined_outcome():
    # Absorbe el test de multiplicadores no negativos: 216 combinaciones.
    for reels in itertools.product(tuple(Symbol), repeat=3):
        outcome = evaluate(reels)
        assert outcome.kind in tuple(OutcomeKind)
        assert outcome.multiplier >= 0


@pytest.mark.parametrize("symbol,expected", sorted(EXPECTED_PAYTABLE.items()))
def test_three_of_a_kind_multipliers_match_paytable(symbol, expected):
    reels = (Symbol(symbol),) * 3
    outcome = evaluate(reels)
    assert outcome.multiplier == expected
    expected_kind = OutcomeKind.JACKPOT if symbol == "SEVEN" else OutcomeKind.WIN
    assert outcome.kind == expected_kind


@pytest.mark.parametrize("odd_position", [0, 1, 2])
def test_two_cherries_pays_three(odd_position):
    reels = [Symbol.CHERRY] * 3
    reels[odd_position] = Symbol.LEMON
    outcome = evaluate(tuple(reels))
    assert outcome.multiplier == EXPECTED_TWO_CHERRY
    assert outcome.kind == OutcomeKind.WIN


def test_near_miss_is_positional_and_pays_zero():
    outcome = evaluate((Symbol.SEVEN, Symbol.SEVEN, Symbol.LEMON))
    assert outcome.kind == OutcomeKind.NEAR_MISS
    assert outcome.multiplier == 0


@pytest.mark.parametrize("reels", [
    (Symbol.SEVEN, Symbol.LEMON, Symbol.SEVEN),
    (Symbol.LEMON, Symbol.SEVEN, Symbol.SEVEN),
])
def test_near_miss_permutations_are_losses(reels):
    outcome = evaluate(reels)
    assert outcome.kind == OutcomeKind.LOSS
    assert outcome.multiplier == 0


def test_loss_is_default_for_all_remaining_combinations():
    # Clasificador de referencia independiente, escrito desde el spec.
    for reels in itertools.product(tuple(Symbol), repeat=3):
        trio = reels[0] == reels[1] == reels[2]
        two_cherry = sum(s is Symbol.CHERRY for s in reels) == 2
        near = reels[0] is Symbol.SEVEN and reels[1] is Symbol.SEVEN and reels[2] is not Symbol.SEVEN
        if not (trio or two_cherry or near):
            outcome = evaluate(reels)
            assert outcome.kind == OutcomeKind.LOSS
            assert outcome.multiplier == 0