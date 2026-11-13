import pytest
from pydantic import ValidationError

from casino.slots.models import SpinRequest, SpinResult


@pytest.mark.parametrize("bet", [0, -1, -100])
def test_spin_request_rejects_non_positive_bet(valid_request_kwargs, bet):
    with pytest.raises(ValidationError):
        SpinRequest(**{**valid_request_kwargs, "bet": bet})


@pytest.mark.parametrize("bet", [1, 2000])
def test_spin_request_accepts_bet_bounds(valid_request_kwargs, bet):
    assert SpinRequest(**{**valid_request_kwargs, "bet": bet}).bet == bet

@pytest.mark.parametrize("bet", [2.5, "3", None])
def test_spin_request_rejects_non_integer_bet(valid_request_kwargs, bet):
    # strict=True: nada de coerción silenciosa de tipos.
    with pytest.raises(ValidationError):
        SpinRequest(**{**valid_request_kwargs, "bet": bet})


def test_spin_request_forbids_extra_fields(valid_request_kwargs):
    with pytest.raises(ValidationError) as exc:
        SpinRequest(**{**valid_request_kwargs, "lucky": True})
    assert exc.value.errors()[0]["type"] == "extra_forbidden"



def test_spin_result_roundtrips_json(valid_result_kwargs):
    original = SpinResult(**valid_result_kwargs)
    restored = SpinResult.model_validate_json(original.model_dump_json())
    assert restored == original


def test_models_are_immutable(valid_result_kwargs):
    result = SpinResult(**valid_result_kwargs)
    with pytest.raises(ValidationError):
        result.payout = 999