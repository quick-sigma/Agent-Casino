import pytest

# Fuente de verdad independiente del paquete: si el motor cambia los strips,
# el oráculo analítico deja de ser válido y estos tests lo detectan.
REEL_1 = (
    ("CHERRY",) * 5 + ("LEMON",) * 4 + ("GRAPE",) * 3
    + ("BELL",) * 2 + ("SEVEN",) * 5 + ("DIAMOND",) * 1
)
REEL_2 = REEL_1
REEL_3 = (
    ("CHERRY",) * 6 + ("LEMON",) * 5 + ("GRAPE",) * 4
    + ("BELL",) * 3 + ("SEVEN",) * 1 + ("DIAMOND",) * 1
)
EXPECTED_STRIPS = (REEL_1, REEL_2, REEL_3)

EXPECTED_PAYTABLE = {
    "CHERRY": 5, "LEMON": 8, "GRAPE": 12,
    "BELL": 20, "DIAMOND": 40, "SEVEN": 60,
}
EXPECTED_TWO_CHERRY = 3


@pytest.fixture
def valid_request_kwargs():
    return {
        "run_seed": "piloto-001",
        "generation": 0,
        "turn": 0,
        "bet": 1,
    }


@pytest.fixture
def valid_result_kwargs(valid_request_kwargs):
    return {
        **valid_request_kwargs,
        "reels": ("CHERRY", "CHERRY", "CHERRY"),
        "kind": "WIN",
        "multiplier": 5,
        "payout": 5,
    }