"""Tests for the Cribl decision readout and action selection.

Fast tests need only the tokenizer (cached by `huggingface_hub`). Tests that
load the 4B model are marked `model` and run when CRIBL_MODEL_TESTS=1:

    .venv/bin/python -m pytest -q                        # fast
    CRIBL_MODEL_TESTS=1 .venv/bin/python -m pytest -q    # everything
"""
from __future__ import annotations

import math, os, types

import pytest

import cribl_decision as cd
import doom_agent as da

needs_model = pytest.mark.skipif(os.environ.get("CRIBL_MODEL_TESTS") != "1",
                                 reason="set CRIBL_MODEL_TESTS=1 to load the 4B model")

DOOM_Q = cd.Question("choice", da.INSTRUCTIONS["default"],
                     {k: da.CRITERIA_RULE[k] for k in da.OPTION_ORDER})


def doom_state(bearing, distance=300):
    return {"health": 100, "ammo": 20, "facing_deg": 0,
            "monsters_nearest_first": [{"kind": "Demon", "distance": distance,
                                        "bearing_deg": bearing}],
            "note": "bearing 0 = dead ahead. POSITIVE bearing means the monster is to "
                    "your LEFT (turn left to face it); NEGATIVE means to your RIGHT."}


# --- pure maths -------------------------------------------------------------

def test_softmax_normalises_and_is_shift_invariant():
    p = cd.softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(p), 1.0)
    assert p == pytest.approx(cd.softmax([101.0, 102.0, 103.0]))
    assert p[2] > p[1] > p[0]


def test_choice_confidence_matches_system_one_formula():
    assert cd.choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert cd.choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cd.choice_confidence([0.5, 0.25, 0.25]) == pytest.approx((0.5 - 1 / 3) / (2 / 3))


def test_make_answer_types():
    a = cd.make_answer(DOOM_Q, ["ATTACK", "TURN_LEFT", "TURN_RIGHT"], [0.0, 2.0, 1.0], 0.9)
    assert a.choice == "TURN_LEFT" and a.p_selected == max(a.probabilities.values())
    assert math.isclose(sum(a.probabilities.values()), 1.0)
    n = cd.make_answer(cd.Question("noul", "x?"), ["true", "false"], [2.0, 0.0], 1.0)
    assert n.noul == pytest.approx(1 / (1 + math.exp(-2)))
    s = cd.make_answer(cd.Question("score", "x?", ["lo", "mid", "hi"]), ["0", "1", "2"],
                       [0.0, 0.0, 50.0], 1.0)
    assert s.score == pytest.approx(2.0, abs=1e-6)


def test_question_validation():
    with pytest.raises(ValueError):
        cd.Question("choice", "x", {}).candidates()
    with pytest.raises(ValueError):
        cd.Question("score", "x", ["only one"]).candidates()
    with pytest.raises(ValueError):
        cd.Question("bogus", "x").candidates()


# --- prompt formats (tokenizer only) ----------------------------------------

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cd.BASE_ID, revision=cd.BASE_REVISION)


@pytest.mark.parametrize("fmt", ["plain_letter", "chat_letter"])
def test_letter_rows_are_single_distinct_tokens(tok, fmt):
    f = cd.FORMATS[fmt]
    prompt = f.prompt(tok, doom_state(3), DOOM_Q)
    rows = f.candidate_rows(tok, prompt, DOOM_Q)
    assert all(len(r) == 1 for r in rows)
    assert len({r[0] for r in rows}) == 3
    want = [" A", " B", " C"] if f.template == "plain" else ["A", "B", "C"]
    assert [tok.decode(r) for r in rows] == want


def test_plain_prompt_ends_at_decision_slot_and_lists_all_candidates(tok):
    f = cd.FORMATS["plain_letter"]
    p = f.prompt(tok, doom_state(3), DOOM_Q)
    assert p.endswith("\nAnswer:")
    for i, k in enumerate(da.OPTION_ORDER):
        assert f"{'ABC'[i]}. {k}: " in p
    assert '"bearing_deg": 3' in p


def test_name_labels_reject_shared_first_token(tok):
    """TURN_LEFT and TURN_RIGHT both start with the token 'TURN', so scoring
    names by first token cannot tell them apart. That must fail loudly."""
    f = cd.FORMATS["plain_name"]
    with pytest.raises(ValueError, match="share a scoring token"):
        f.candidate_rows(tok, f.prompt(tok, doom_state(0), DOOM_Q), DOOM_Q)


def test_candidate_order_changes_letters_not_names(tok):
    f = cd.FORMATS["plain_letter"]
    q2 = cd.Question("choice", DOOM_Q.instructions,
                     {k: DOOM_Q.criteria[k] for k in ["TURN_RIGHT", "ATTACK", "TURN_LEFT"]})
    p = f.prompt(tok, doom_state(0), q2)
    assert "A. TURN_RIGHT" in p and "B. ATTACK" in p and "C. TURN_LEFT" in p


# --- action selection with a fake model -------------------------------------

class FakeModel:
    """Stands in for CriblDecisionModel: returns fixed logits."""
    def __init__(self, logits, fail=False):
        self.logits, self.fail, self.calls = logits, fail, 0
        self.fmt = cd.FORMATS["plain_letter"]

    def ask(self, state, questions):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        out = {}
        for n, q in questions.items():
            keys = [k for k, _ in q.candidates()]
            out[n] = cd.make_answer(q, keys, [self.logits.get(k, 0.0) for k in keys], 1.0)
        return out, {"total_ms": 1.0, "tokens": 10}


def brain_with(model, **kw):
    b = da.CriblDecisionBrain.__new__(da.CriblDecisionBrain)
    b.model = model
    b.question = cd.Question("choice", da.INSTRUCTIONS["default"],
                             {k: da.CRITERIA_RULE[k] for k in da.OPTION_ORDER})
    b.danger = None
    b.conf_floor = kw.get("conf_floor", 0.0)
    b.on_error = kw.get("on_error", "raise")
    return b


def test_brain_returns_argmax_with_probabilities():
    b = brain_with(FakeModel({"ATTACK": 3.0, "TURN_LEFT": 0.0, "TURN_RIGHT": 0.0}))
    d = b.decide(doom_state(-50))
    assert d.action == "ATTACK" and d.fallback is None
    assert set(d.probabilities) == set(da.BUTTONS)
    assert d.p_selected == pytest.approx(math.exp(3) / (math.exp(3) + 2))


def test_low_confidence_fallback_is_explicit():
    b = brain_with(FakeModel({"ATTACK": 0.1, "TURN_LEFT": 0.0, "TURN_RIGHT": 0.0}), conf_floor=0.5)
    d = b.decide(doom_state(-50))
    assert d.fallback == "low_confidence"
    assert d.model_action == "ATTACK" and d.action == "TURN_RIGHT"   # heuristic's pick


def test_model_errors_raise_by_default_and_are_flagged_otherwise():
    with pytest.raises(RuntimeError):
        brain_with(FakeModel({}, fail=True)).decide(doom_state(0))
    d = brain_with(FakeModel({}, fail=True), on_error="heuristic").decide(doom_state(0))
    assert d.fallback == "error:RuntimeError" and d.action == "ATTACK"


def test_describe_bearing_sign():
    def obj(name, x, y):
        return types.SimpleNamespace(name=name, position_x=x, position_y=y)
    # player at origin facing east (angle 0); monster north = to the LEFT
    st = types.SimpleNamespace(game_variables=[100, 26, 0, 0, 0, 0, 0, 0, 26],
                               objects=[obj("DoomPlayer", 0, 0), obj("Demon", 0, 100),
                                        obj("MarineChainsawVzd", 50, 0)])
    d = da.describe(st)
    assert [m["kind"] for m in d["monsters_nearest_first"]] == ["Demon"]
    assert d["monsters_nearest_first"][0]["bearing_deg"] == 90
    assert da.heuristic(d) == "TURN_LEFT"
    r = da.describe(st, ["ATTACK"], rich=True)
    assert r["monsters_in_view"] == 0 and r["previous_actions"] == ["ATTACK"]


def test_episode_loop_uses_brain_decisions():
    """A full lock-step episode with the fake model always picking ATTACK:
    every decision is the model's, and it is carried out (shots are fired)."""
    fake = FakeModel({"ATTACK": 5.0})
    b = brain_with(fake)
    g = da.make_game()
    try:
        r = da.run_episode(g, b, 1, 1234, 4, False)
    finally:
        g.close()
    assert r.decisions == fake.calls > 0 and r.fallbacks == 0
    assert r.shots > 0 and r.survival_tics > 0


# --- the real model ---------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return cd.CriblDecisionModel(fmt="plain_letter")


@needs_model
def test_every_adapter_tensor_is_applied(model):
    assert model.adapter_report["tensors"] == 304
    assert model.adapter_report["max_abs_diff"] < 1e-2


@needs_model
def test_readout_is_lm_head_rows_at_decision_slot(model):
    """candidate_logits == candidate_weights @ hidden_state at the slot."""
    import torch
    prompt = model.fmt.prompt(model.tokenizer, doom_state(30), DOOM_Q)
    ids = torch.tensor([model.tokenizer(prompt, add_special_tokens=False)["input_ids"]],
                       device=model.device)
    with torch.inference_mode():
        out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0, -1]
        rows = [r[0] for r in model.fmt.candidate_rows(model.tokenizer, prompt, DOOM_Q)]
        manual = (model.lm_head[rows] @ h).float()
    a, _ = model.choose(doom_state(30), DOOM_Q)
    assert torch.allclose(manual.cpu(), torch.tensor(a.logits), atol=0.25)
    assert [k for k, _ in DOOM_Q.candidates()][int(manual.argmax())] == a.choice


@needs_model
def test_batched_questions_match_single(model):
    """Left padding through the DeltaNet layers must not change answers."""
    short = cd.Question("noul", "Is a monster close enough to hurt you within the next second?")
    both, _ = model.ask(doom_state(-40), {"action": DOOM_Q, "danger": short})
    one, _ = model.ask(doom_state(-40), {"action": DOOM_Q})
    for k in one["action"].probabilities:
        assert both["action"].probabilities[k] == pytest.approx(one["action"].probabilities[k], abs=0.02)


@needs_model
def test_deterministic(model):
    a1, _ = model.choose(doom_state(12), DOOM_Q)
    a2, _ = model.choose(doom_state(12), DOOM_Q)
    assert a1.probabilities == a2.probabilities


@needs_model
@pytest.mark.parametrize("bearing,want", [(60, "TURN_LEFT"), (-60, "TURN_RIGHT")])
def test_turns_towards_monster(model, bearing, want):
    a, _ = model.choose(doom_state(bearing), DOOM_Q)
    assert a.choice == want and a.p_selected > 0.5


def test_compact_state_carries_nearest_monster_only():
    def obj(name, x, y):
        return types.SimpleNamespace(name=name, position_x=x, position_y=y)
    st = types.SimpleNamespace(game_variables=[90, 20, 0, 0, 0, 0, 0, 0, 20],
                               objects=[obj("DoomPlayer", 0, 0), obj("Demon", 0, -300),
                                        obj("Demon", 100, 10), obj("Demon", -500, 0)])
    d = da.describe(st, ["TURN_LEFT"], rich="compact")
    assert d["nearest_monster"] == {"distance": 100, "bearing_deg": 6}
    assert d["monsters_in_view"] == 1 and d["monsters_total"] == 3
    assert d["previous_action"] == "TURN_LEFT"
    assert "monsters_nearest_first" not in d
    assert da.heuristic(d) == "ATTACK"
