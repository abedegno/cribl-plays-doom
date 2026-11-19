"""Cribl Decision 2: the Doom prompt, the IWAD option, and MLX vs the reference.

    .venv/bin/python -m pytest -q tests/test_decision2.py                         # fast
    CRIBL_MODEL_TESTS=1 .venv/bin/python -m pytest -q tests/test_decision2.py     # + both models, both backends
"""
from __future__ import annotations

import gc, os

import pytest

import cribl_decision2 as c2
import doom_agent as da

needs_model = pytest.mark.skipif(os.environ.get("CRIBL_MODEL_TESTS") != "1",
                                 reason="set CRIBL_MODEL_TESTS=1 to load the models")
NOTE = ("bearing 0 = dead ahead. POSITIVE bearing means the monster is to your LEFT "
        "(turn left to face it); NEGATIVE means to your RIGHT.")
OPTIONS = {k: da.CRITERIA_RULE[k] for k in da.OPTION_ORDER}
BEARINGS = [0, 4, -7, 9, 15, 60, -25, -120, 170, -9]


def state(b, dist=300):
    return {"health": 100, "ammo": 21, "armor": 0,
            "nearest_monster": {"distance": dist, "bearing_deg": b},
            "monsters_in_view": 1, "monsters_total": 4, "previous_action": "TURN_LEFT", "note": NOTE}


def cached(size):
    from huggingface_hub import try_to_load_from_cache
    repo, rev = c2.MODELS[size]
    return isinstance(try_to_load_from_cache(repo, "decision.py", revision=rev), str)


def test_confidence_formula():
    a = c2.make_answer({"ATTACK": 0.6, "TURN_LEFT": 0.3, "TURN_RIGHT": 0.1})
    assert a.choice == "ATTACK" and a.p_selected == 0.6
    assert abs(a.confidence - (0.6 - 1 / 3) / (1 - 1 / 3)) < 1e-12


def test_iwad_default_is_freedoom(monkeypatch):
    monkeypatch.delenv("DOOM_IWAD", raising=False)
    assert da.resolve_iwad(None) is None
    assert da.resolve_iwad("freedoom") is None
    with pytest.raises(FileNotFoundError):
        da.resolve_iwad("/nonexistent/doom2.wad")


@pytest.mark.skipif(not cached("4b"), reason="Decision 2 4B snapshot not downloaded")
def test_doom_prompt_has_one_readout_and_one_marker_per_button():
    shell = c2.prompt_shell(c2.snapshot("4b"))
    ref = c2.reference_module(c2.snapshot("4b"))
    text, labels = shell.prompt(ref.Question(state(5), da.INSTRUCTIONS["default"], OPTIONS))
    ids = shell.tokenizer(text)["input_ids"]
    assert labels == da.OPTION_ORDER
    assert ids.count(shell.readout_id) == 1
    assert ids.count(shell.choice_end_id) == 3
    # options first, state last, then the readout (decision_config "options_first")
    assert text.index("ATTACK — Fire.") < text.index('"bearing_deg": 5') < text.index("Answer:")


@needs_model
@pytest.mark.parametrize("size", ["4b", "mini"])
def test_mlx_matches_reference(size):
    """MLX port vs Cribl's decision.py on PyTorch: same decisions, close
    probabilities (bf16 on different kernels, so not bit for bit)."""
    q = da.INSTRUCTIONS["default"]
    tm = c2.load(size, "torch")
    ref = [tm.decide(state(b, 200 + 37 * i), q, OPTIONS)[0] for i, b in enumerate(BEARINGS)]
    del tm; gc.collect()
    mm = c2.load(size, "mlx")
    got = [mm.decide(state(b, 200 + 37 * i), q, OPTIONS)[0] for i, b in enumerate(BEARINGS)]
    worst = 0.0
    for r, g in zip(ref, got):
        worst = max(worst, *(abs(r.probabilities[k] - g.probabilities[k]) for k in OPTIONS))
        top2 = sorted(r.probabilities.values())[-2:]
        if top2[1] - top2[0] > 0.05:
            assert r.choice == g.choice
    print(f"{size}: max |p_torch - p_mlx| = {worst:.4f}")
    assert worst < 0.05


@needs_model
def test_mlx_is_deterministic():
    m = c2.load("mini", "mlx")
    a = [m.decide(state(b), da.INSTRUCTIONS["default"], OPTIONS)[0].probabilities for b in BEARINGS]
    b = [m.decide(state(b), da.INSTRUCTIONS["default"], OPTIONS)[0].probabilities for b in BEARINGS]
    assert a == b
