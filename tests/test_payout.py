from hypothesis import given, strategies as st

from casino.slots.engine import evaluate, settle
from casino.slots.spec import Symbol

REELS = st.tuples(*(st.sampled_from(tuple(Symbol)) for _ in range(3)))
BETS = st.integers(min_value=1, max_value=1000)

@given(bet=BETS, reels=REELS)
def test_payout_equals_bet_times_multiplier(bet, reels):
    outcome = evaluate(reels)
    assert settle(bet, outcome) == bet * outcome.multiplier


@given(bet=BETS, reels=REELS)
def test_payout_is_integer_and_non_negative(bet, reels):
    payout = settle(bet, evaluate(reels))
    assert isinstance(payout, int)
    assert payout >= 0


def test_payout_scales_linearly_with_bet():
    outcome = evaluate((Symbol.CHERRY,) * 3)
    assert [settle(b, outcome) for b in (1, 5, 10)] == [5, 25, 50]