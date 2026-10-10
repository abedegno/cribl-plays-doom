"""Benchmark every brain and experiment on identical seeds.

    python benchmark.py --suite baseline --episodes 20      # random, heuristic, cribl (+jev if key)
    python benchmark.py --suite all --episodes 20           # baseline + experiments A-E
    python benchmark.py --only cribl_tics8                  # one configuration

Each configuration writes results/benchmarks/<name>.json (summary + per
episode) and results/traces/<name>.csv (every decision). The Cribl model is
loaded once and shared by every configuration. `report.py` turns the JSON into
results/REPORT.md.

Episode i of every configuration uses seed `--seed + i`, so all brains face
the same starting states.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from types import SimpleNamespace

import doom_agent as da

DEFAULTS = dict(brain="cribl", criteria="rule", format="plain_letter_qfirst", order="",
                tics=4, live=False, conf_floor=0.0, state="jev", on_error="raise",
                ask_danger=False, backend="auto", device="auto", dtype="bfloat16",
                jev_model="jev-latest")

PERMS = ["ATTACK,TURN_LEFT,TURN_RIGHT", "TURN_LEFT,ATTACK,TURN_RIGHT",
         "TURN_RIGHT,TURN_LEFT,ATTACK"]

SUITES = {
    "baseline": [
        ("random", dict(brain="random")),
        ("heuristic", dict(brain="heuristic")),
        ("jev", dict(brain="jev")),
        ("cribl", dict()),
    ],
    # A. prompt engineering: what the three buttons are described as
    "prompts": [
        ("cribl_minimal", dict(criteria="minimal")),
        ("cribl_intent", dict(criteria="intent")),
        ("cribl_goal", dict(criteria="goal")),
    ],
    # B. candidate order (gameplay); the constant-state analysis is order_probe.py
    "order": [(f"cribl_order_{p.replace(',', '-')}", dict(order=p)) for p in PERMS[1:]],
    # C. decision interval
    "tics": [(f"cribl_tics{t}", dict(tics=t)) for t in (2, 8, 16)],
    # D. confidence floor: below it the heuristic is used, counted as fallback
    "confidence": [("cribl_conf0.4", dict(conf_floor=0.4))],
    # E. two typed questions in one request
    "batched": [("cribl_with_danger", dict(ask_danger=True))],
    # prompt format actually played (state first, no prefix cache possible)
    "format": [("cribl_state_first", dict(format="plain_letter"))],
    # state representation: same information, restructured
    "state": [("cribl_state_compact", dict(state="compact")),
              ("cribl_state_rich", dict(state="rich")),
              ("heuristic_state_compact", dict(brain="heuristic", state="compact")),
              ("cribl_compact_tics8", dict(state="compact", tics=8)),
              ("cribl_compact_live", dict(state="compact", live=True)),
              ("cribl_compact_live_tics8", dict(state="compact", live=True, tics=8))],
    # real time: the game runs at 35 tics/s of wall clock while the model thinks
    "live": [("heuristic_live", dict(brain="heuristic", live=True)),
             ("cribl_live", dict(live=True)),
             ("cribl_live_tics8", dict(live=True, tics=8))],
}
SUITES["all"] = [c for k in ("baseline", "prompts", "order", "tics", "confidence",
                             "batched", "format", "state", "live") for c in SUITES[k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="baseline", choices=list(SUITES))
    ap.add_argument("--only", default="", help="comma-separated configuration names")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    configs = SUITES["all"] if a.only else SUITES[a.suite]
    if a.only:
        want = set(a.only.split(","))
        configs = [c for c in configs if c[0] in want]
    model = None
    for name, over in configs:
        path = os.path.join(a.out, "benchmarks", f"{name}.json")
        if a.skip_existing and os.path.exists(path):
            print(f"[{name}] exists, skipping"); continue
        cfg = SimpleNamespace(**{**DEFAULTS, "backend": a.backend, **over,
                                 "seed": a.seed, "episodes": a.episodes})
        if cfg.brain == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
            print(f"[{name}] skipped: TYPESAFE_API_KEY not set")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"name": name, "skipped": "TYPESAFE_API_KEY not set"}, open(path, "w"))
            continue
        if cfg.brain == "cribl" and model is None:
            model = da.load_cribl_model(cfg.backend, cfg.format)
            print(f"loaded cribl on {model.device} in {model.load_seconds:.1f}s", flush=True)
        brain = da.build_brain(cfg, model=model if cfg.brain == "cribl" else None)
        print(f"[{name}] {over or 'defaults'}", flush=True)
        t0 = time.monotonic()
        s = da.play(brain, a.episodes, cfg.tics, False, a.seed, da.state_mode(cfg),
                    os.path.join(a.out, "traces", f"{name}.csv"), cfg.live, name)
        cfg_out = {k: v for k, v in vars(cfg).items()}
        if cfg.brain == "cribl":
            cfg_out.update(model_device=model.device,
                           model_dtype=getattr(model, "dtype_name", cfg.dtype),
                           model_load_s=model.load_seconds)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"name": name, "config": cfg_out, "summary": s,
                   "elapsed_s": time.monotonic() - t0}, open(path, "w"), indent=1)
        acc = s["accuracy"]
        print(f"[{name}] kills {s['kills_mean']:.2f}±{s['kills_sd']:.2f} reward {s['reward_mean']:.2f} "
              f"survival {s['survival_s_mean']:.1f}s acc {acc if acc is None else round(acc, 3)} "
              f"decisions {s['decisions']} fallbacks {s['fallbacks']} "
              f"lat p50 {s['decision_ms_p50']:.1f}ms p95 {s['decision_ms_p95']:.1f}ms\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
