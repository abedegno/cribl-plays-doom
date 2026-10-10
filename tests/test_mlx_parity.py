"""The MLX backend (merged LoRA) must give the same decisions as the
reference PyTorch + PEFT path (unmerged LoRA) on identical prompts.

    CRIBL_MODEL_TESTS=1 .venv/bin/python -m pytest -q tests/test_mlx_parity.py
"""
from __future__ import annotations

import gc, os

import pytest

needs_model = pytest.mark.skipif(os.environ.get("CRIBL_MODEL_TESTS") != "1",
                                 reason="set CRIBL_MODEL_TESTS=1 to load the 4B model")

BEARINGS = [0, 4, -7, 15, 60, -25, -120, 170]


@needs_model
def test_mlx_matches_torch_peft():
    from tests.test_cribl_readout import DOOM_Q, doom_state
    import cribl_decision as cd
    import torch

    tm = cd.CriblDecisionModel(fmt="plain_letter")
    ref = [tm.choose(doom_state(b, 200 + 10 * i), DOOM_Q)[0] for i, b in enumerate(BEARINGS)]
    del tm; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    from cribl_mlx import CriblDecisionModelMLX
    mm = CriblDecisionModelMLX(fmt="plain_letter")
    got = [mm.choose(doom_state(b, 200 + 10 * i), DOOM_Q)[0] for i, b in enumerate(BEARINGS)]

    worst = 0.0
    for r, g in zip(ref, got):
        for k in r.probabilities:
            worst = max(worst, abs(r.probabilities[k] - g.probabilities[k]))
        # decisions must agree unless the reference itself is a near tie
        top2 = sorted(r.probabilities.values())[-2:]
        if top2[1] - top2[0] > 0.1:
            assert r.choice == g.choice
    print(f"max |p_torch - p_mlx| = {worst:.4f}")
    assert worst < 0.08


@needs_model
def test_mlx_prefix_cache_is_stable_and_matches_uncached():
    """The cached static prefix must not be mutated by later decisions, and
    cached answers must match a full forward pass."""
    from tests.test_cribl_readout import DOOM_Q, doom_state
    from cribl_mlx import CriblDecisionModelMLX
    m = CriblDecisionModelMLX(fmt="plain_letter_qfirst")
    m.prefix_cache = False
    full = [m.choose(doom_state(b), DOOM_Q)[0].probabilities for b in BEARINGS]
    m.prefix_cache = True
    first = [m.choose(doom_state(b), DOOM_Q)[0].probabilities for b in BEARINGS]
    again = [m.choose(doom_state(b), DOOM_Q)[0].probabilities for b in BEARINGS]
    assert first == again                                   # snapshot not mutated
    for f, c in zip(full, first):
        for k in f:
            assert abs(f[k] - c[k]) < 0.05
