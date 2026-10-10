# Technical notes

This document covers how the agent works, what about the readout is verified, how to install and run everything, and the limits. The results and findings are in the [README](../README.md), and every table is in [results/REPORT.md](../results/REPORT.md).

## How it works

```
 ViZDoom (headless)         doom_agent.py                cribl_decision.py / cribl_mlx.py
┌──────────────────┐ objects, ┌───────────────┐  state  ┌──────────────────────────────────┐
│ defend_the_      ├─ health ►│ describe()    ├────────►│ prompt: question + lettered      │
│ center.wad       │   ammo   │  → bearings,  │         │ candidates + state + "Answer:"   │
│                  │          │    distances  │         │ one forward pass                 │
│                  │◄─────────┤ Decision      │◄────────┤ logits at the decision slot,     │
└──────────────────┘ button × │ (action, p,   │ choice, │ rows " A" " B" " C" of lm_head,  │
                     N tics   │  latency)     │ probs   │ softmax over the 3 candidates    │
                              └───────────────┘         └──────────────────────────────────┘
```

The model receives JSON such as this (`--state compact`):

```json
{"health": 100, "ammo": 21, "armor": 0,
 "nearest_monster": {"distance": 412, "bearing_deg": 7},
 "monsters_in_view": 1, "monsters_total": 4, "previous_action": "TURN_LEFT",
 "note": "bearing 0 = dead ahead. POSITIVE bearing means the monster is to your LEFT (turn left to face it); NEGATIVE means to your RIGHT."}
```

`--state jev` (the default) is exactly the original project's state: health,
ammo, facing, and up to six monsters with kind, distance and bearing, nearest
first. Both are computed from the same ViZDoom object positions the original
Jev agent used. Neither gives the model anything the Jev agent did not have.

It is asked one typed `choice` question:

```
Question (choice): You are fighting for your life in Doom, surrounded by monsters. Pick the single best button to press right now.

Candidates:
A. ATTACK: Fire. Correct whenever the nearest monster's bearing is between -8 and +8, ...
B. TURN_LEFT: Rotate left. Correct only when the nearest monster's bearing is GREATER than +8. ...
C. TURN_RIGHT: Rotate right. Correct only when the nearest monster's bearing is LESS than -8. ...

Answer with the letter of the single best candidate.

State:
{...}
Answer:
```

and returns:

```python
Decision(action="ATTACK",
         probabilities={"ATTACK": 0.91, "TURN_LEFT": 0.06, "TURN_RIGHT": 0.03},
         confidence=0.87,      # (max p − 1/n) / (1 − 1/n), the system-one formula
         p_selected=0.91,
         model_ms=171.4,
         fallback=None)        # or "low_confidence" / "error:...", never silent
```

## The structured readout, and what is verified

The model card describes the readout as *hidden state at the decision slot →
native Qwen `lm_head` candidate logits → normalisation over the supplied
candidates*. It refers to "the accompanying candidate-scoring implementation",
but that has not been published: there is none in the Hugging Face repository
and none on Cribl's GitHub as of 2026-10-10. So this repository implements a
compatible readout and states clearly which parts are verified and which are
inferred.

**Verified, with tests ([`tests/`](../tests/)):**

| claim | how |
|---|---|
| Base model is `Qwen/Qwen3.5-4B-Base` at the pinned revision `1001bb4d…` | `cribl_decision.BASE_REVISION`; the adapter is pinned too (`75cb0dd9…`) |
| **Every one of the 304 adapter tensors is applied** | `_verify_adapter()` compares the live model with `adapter_model.safetensors` and refuses to run otherwise. See the pitfall below. |
| Probabilities come from the native `lm_head` rows at the decision slot | `test_readout_is_lm_head_rows_at_decision_slot`: `candidate_weights @ hidden_state` computed by hand equals the readout |
| Softmax over the supplied candidates only; never generated text | no `generate()` anywhere; one `forward()` per question |
| Several questions in one padded forward pass equal the same questions asked singly | `test_batched_questions_match_single` |
| MLX (merged LoRA) agrees with PyTorch + PEFT (unmerged LoRA) | `test_mlx_matches_torch_peft`: max \|Δp\| 0.020 over 8 Doom states, same decisions |
| The MLX prefix cache does not change answers or drift | `test_mlx_prefix_cache_is_stable_and_matches_uncached` |
| Deterministic | `test_deterministic` |

**Inferred, not verified against Cribl's code:** the prompt text, the
position of the decision slot, and which vocabulary rows score each
candidate. `probe_formats.py` tested seven layouts on 27 questions with known
answers (customer routing, telemetry event types, log severity, sentiment,
arithmetic, knowledge, tool choice, threshold rules, Doom), each asked in two
candidate orders, with the adapter on and off. The decisive signal is
probability mass at the slot:

| layout | adapter: slot mass on the candidate tokens | bare base | adapter acc | base acc |
|---|---|---|---|---|
| plain text, letters, question first, ends `Answer:` (**used**) | **0.95** | 0.35 | 0.85 | 0.72 |
| plain text, letters, state first | **0.99** | 0.57 | 0.89 | 0.85 |
| plain text, candidate names | **0.95** | 0.25 | 1.00* | 0.85* |
| Qwen chat template, letters | 0.39 | 0.36 | 0.87 | 0.85 |
| Qwen chat template, question first | 0.54 | 0.39 | 0.83 | 0.76 |

<sub>*Name scoring cannot run the Doom question because `TURN_LEFT` and
`TURN_RIGHT` share their first token, so those formats were scored on fewer items.
The code refuses such candidate sets rather than guessing.</sub>

The adapter clearly learned to put a candidate label at a plain-text
`Answer:` slot (95 to 99% of the full-vocabulary probability, against 25 to 57% for
the base), and not at a chat-template slot. Of the plain layouts,
question-first shows the largest accuracy gain from the adapter (+0.13). It is
also the only one whose static part can be cached, and it played Doom far
better (4.15 kills against 0.55 for state-first, `results/REPORT.md`). The exact
training prompt is still unknown, so absolute scores may be lower than
Cribl's own readout would achieve.

**Pitfall worth knowing:** the model card's own snippet loads the base with
`AutoModelForCausalLM`. With transformers 5.x this resolves to the text-only
`Qwen3_5ForCausalLM`, whose modules are named `model.layers.*`. The adapter was
saved against `Qwen3_5ForConditionalGeneration` (`model.language_model.layers.*`).
PEFT only *warns* about the mismatch and then runs the bare base model. This
repository loads `Qwen3_5ForConditionalGeneration` and checks every tensor.

## Install

Tested on a Mac mini, Apple M4 Pro (24 GB), macOS 26.7, Python 3.12.

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[cribl,mlx,record,test]"   # drop mlx off Apple Silicon
.venv/bin/python selfcheck.py
```

The first `--brain cribl` run downloads the pinned base model (9.3 GB) and the
adapter (58 MB) from Hugging Face. On Apple Silicon with `mlx` installed, the
LoRA is merged once into the base in float32, cast back to bfloat16, and cached
in `~/.cache/cribl-doom/` (another 9.3 GB). `--backend torch` uses the
unmerged PEFT reference path: CUDA, then MPS, then CPU.

| backend | device | precision | decision latency (Doom, jev state) |
|---|---|---|---|
| `mlx` (default on Apple Silicon) | M4 Pro GPU | bfloat16 weights, LoRA merged in float32 | **~175 ms** with prefix cache, ~410 ms without |
| `torch` | M4 Pro via MPS | bfloat16, LoRA unmerged | ~940 to 1050 ms (transformers' reference DeltaNet kernels on MPS) |
| `torch` | CUDA | bfloat16 | not measured here; install `flash-linear-attention` and `causal-conv1d` for the fast kernels |
| `torch` | CPU | `--dtype float32` recommended | works; not measured here |

## Commands

```bash
# play
.venv/bin/python doom_agent.py --brain cribl --episodes 20 --seed 1234
.venv/bin/python doom_agent.py --brain cribl --state compact --episodes 20 --seed 1234 --trace results/traces/mine.csv
.venv/bin/python doom_agent.py --brain heuristic --episodes 20 --seed 1234
.venv/bin/python doom_agent.py --brain random --episodes 20 --seed 1234
.venv/bin/python doom_agent.py --brain jev --episodes 20 --seed 1234        # needs TYPESAFE_API_KEY and --extra jev
.venv/bin/python doom_agent.py --brain cribl --state compact --live --tics 8  # real time

# benchmark every brain and experiment, then the report
.venv/bin/python benchmark.py --suite all --episodes 20 --seed 1234
.venv/bin/python order_probe.py --trace results/traces/cribl.csv --n 150
.venv/bin/python batch_latency.py --backend mlx; .venv/bin/python batch_latency.py --backend torch
.venv/bin/python probe_formats.py
.venv/bin/python report.py                     # → results/REPORT.md

# watch / record
.venv/bin/python record.py --brain cribl --state compact --seconds 60 --out docs/cribl-doom.mp4
.venv/bin/python record.py --brain cribl --state compact --live --tics 8 --seconds 30 --out docs/cribl-doom-live.mp4
.venv/bin/python doom_agent.py --brain cribl --state compact --window --episodes 1   # ViZDoom window
.venv/bin/python play.py                       # play the same seeds yourself

# tests
.venv/bin/python -m pytest -q                          # fast, tokenizer only
CRIBL_MODEL_TESTS=1 .venv/bin/python -m pytest -q      # + loads the model (both backends)
```

| flag | default | |
|---|---|---|
| `--brain` | `heuristic` | `heuristic`, `random`, `jev`, `cribl` |
| `--state` | `jev` | `jev` (original), `rich` (+ armour, in-view count, last 3 actions), `compact` (nearest monster as its own field) |
| `--criteria` | `rule` | `rule`, `intent`, `minimal`, `goal` |
| `--format` | `plain_letter_qfirst` | prompt layout, see `cribl_decision.FORMATS` |
| `--order` | `ATTACK,TURN_LEFT,TURN_RIGHT` | candidate order shown to the model |
| `--tics` | `4` | game tics each decision is held for |
| `--live` | off | game runs at 35 tics/s of wall clock while the model thinks |
| `--backend` | `auto` | `mlx` or `torch` |
| `--device` / `--dtype` | `auto` / `bfloat16` | torch backend only |
| `--conf-floor` | `0.0` | below it, substitute the heuristic (counted as `low_confidence` fallback) |
| `--on-error` | `raise` | `raise`, or substitute `heuristic` / `noop` (counted as `error:` fallback) |
| `--ask-danger` | off | also ask a `noul` "in danger?" question every decision |
| `--seed` | `1234` | episode *i* uses `seed + i`; `-1` randomises |
| `--trace` / `--json` | none | per-decision CSV / run summary |

## Measurements, and what they mean

**Game time vs wall-clock time.** By default the loop is *lock-step*: ViZDoom
waits for every decision, so kills and survival do not depend on how slow the
model is. Wall-clock cost is measured separately and reported as latency,
decisions per second and *game speed* (game seconds per wall second). `--live`
removes the lock-step. The game advances at 35 tics/s of wall clock while a
worker thread runs the model on the most recent state snapshot. The model can
only ever see current or stale state, never future state, and the staleness of
every decision is traced.

**Latency** is the full decision: prompt building, tokenisation, forward pass
and readout. Raw forward-pass time on MLX is about 86 ms for a 16-token
input, rising by about 1.25 ms per token. A prompt of about 300 tokens therefore
costs about 410 ms. Caching the static question-and-candidates prefix leaves only
the state (about 100 to 150 tokens) per decision, which gives the ~175 ms.

**Multi-token prediction** (mlx-lm PR 990) does not apply. It speeds up
*generation* by drafting several tokens per forward pass. This readout
generates nothing: one prefill and one read per decision. The MTP head is also
not part of what the LoRA adapted.

**Batching** two or four questions into one padded forward pass is supported
on the torch backend (and tested equal to asking them singly). On this machine
it is *slower* than asking sequentially (1.79 s against 1.42 s for two questions on
MPS), because prefill is compute-bound. MLX asks questions sequentially, at about
168 ms each.

## Files

```
doom_agent.py      state, brains (random, heuristic, Jev, Cribl), lock-step and live loops, CLI
cribl_decision.py  typed questions, prompt formats, PyTorch + PEFT model and readout
cribl_mlx.py       LoRA merge, MLX model, same readout, prefix cache
benchmark.py       every brain and experiment on identical seeds → results/benchmarks, results/traces
report.py          results/REPORT.md from those files
probe_formats.py   which prompt layout the adapter responds to
order_probe.py     candidate order and wording on constant recorded states
batch_latency.py   cost of several questions per decision
record.py          mp4 with decision overlay (lock-step or live)
play.py            play the same seeds yourself
selfcheck.py       install check without loading a model
tests/             readout, action selection, adapter integrity, backend parity
results/           REPORT.md, benchmark JSON, decision traces (CSV), probe outputs
docs/              recordings; docs/jev/ holds the original project's README and footage
```

## Limitations

* **The readout format is inferred.** It is not Cribl's published code, because
  none exists yet. Results may change once the reference implementation is
  released. Swap the layout in `cribl_decision.FORMATS`.
* **There is one scenario.** `defend_the_center` has three buttons and no movement.
  Nothing here says how the model handles navigation or richer action sets.
* **Results are zero-shot.** There was no fine-tuning, and the prompt was not
  tuned against the benchmark seeds, but the wording and state variants were
  chosen with the benchmark in mind. Treat the best configuration as a
  favourable one, not a typical one.
* **Twenty episodes is a small sample.** Per-episode standard deviations are
  1 to 2 kills, so differences under about 1 kill between configurations are noise.
* **The MLX path changes rounding.** It runs merged weights
  (bf16(W + BA) rather than bf16(W) + BA), so probabilities differ from the
  PEFT path by up to about 0.02. Decisions agreed on every tested state that was
  not a near tie.
* **Latency depends on hardware.** All latencies are from one M4 Pro. A CUDA GPU
  with the fused DeltaNet kernels should be much faster, but was not measured.
* **The state comes from ViZDoom's object list, like the original project's.**
  That list includes monsters outside the field of view, which the original Jev
  agent was also given. `monsters_in_view` is computed from bearings and a 90°
  field of view; it is not a line-of-sight test.

## License

MIT, as the original project. See [LICENSE](../LICENSE). The Cribl adapter is
Apache-2.0 and the Qwen base model has its own license; neither is
redistributed here.
