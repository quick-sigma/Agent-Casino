import os
import subprocess
import sys

from casino.slots.engine import spin
from casino.slots.models import SpinRequest


def test_same_inputs_produce_identical_result(valid_request_kwargs):
    request = SpinRequest(**valid_request_kwargs)
    assert spin(request) == spin(request)


def test_determinism_across_processes_with_different_hashseed(valid_request_kwargs):
    # Caza la derivación de seed con hash() de Python, que va salada por proceso.
    local = spin(SpinRequest(**valid_request_kwargs)).model_dump_json()
    code = (
        "from casino.slots.models import SpinRequest;"
        "from casino.slots.engine import spin;"
        f"print(spin(SpinRequest(**{valid_request_kwargs!r})).model_dump_json())"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, check=True,
    )
    assert completed.stdout.strip() == local


def test_different_turn_changes_outcome(valid_request_kwargs):
    results = {
        spin(SpinRequest(**{**valid_request_kwargs, "turn": t})).reels
        for t in range(200)
    }
    assert len(results) > 1


def test_different_run_seed_changes_sequence(valid_request_kwargs):
    a = [spin(SpinRequest(**{**valid_request_kwargs, "turn": t})).reels for t in range(20)]
    b = [spin(SpinRequest(**{**valid_request_kwargs, "run_seed": "piloto-002", "turn": t})).reels for t in range(20)]
    assert a != b


