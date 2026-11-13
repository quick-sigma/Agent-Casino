import inspect

import pytest

from casino.slots.engine import derive_rng, spin
from casino.slots.models import SpinRequest


@pytest.mark.parametrize("bet", range(1, 11))
def test_bet_does_not_change_reels(valid_request_kwargs, bet):
    baseline = spin(SpinRequest(**valid_request_kwargs)).reels
    actual = spin(SpinRequest(**{**valid_request_kwargs, "bet": bet})).reels
    assert actual == baseline


def test_bet_not_in_seed_material():
    assert "bet" not in inspect.signature(derive_rng).parameters