"""Experiment E: cost of asking several typed questions per decision.

PyTorch backend: all questions go through ONE forward pass as a left-padded
batch (one sequence per question; Qwen3.5's DeltaNet layers honour the
padding mask -- tests/test_cribl_readout.py checks batched == single).
Compared with asking the same questions one forward pass at a time.

MLX backend: mlx_lm has no padding mask, so questions run one after another,
each reusing its own cached static prefix.

    python batch_latency.py --backend mlx --trace results/traces/cribl.csv
    python batch_latency.py --backend torch
"""
from __future__ import annotations

import argparse, csv, json, os, statistics, time

import doom_agent as da
from cribl_decision import Question

QUESTIONS = {
    "action": Question("choice", da.INSTRUCTIONS["default"],
                       {k: da.CRITERIA_RULE[k] for k in da.OPTION_ORDER}),
    "in_danger": Question("noul", "Is a monster close enough to hurt you within the next second?"),
    "threat": Question("score", "How dangerous is the current situation?",
                       ["safe: no monster within 300 units", "some danger: a monster within 300 units",
                        "high danger: a monster within 120 units", "critical: several monsters within 120 units"]),
    "low_ammo": Question("noul", "Is ammunition running low (fewer than 8 bullets)?"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--trace", default="results/traces/cribl.csv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="results/batch_latency.json")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.trace)) if r["state"]][:: 7][: a.n]
    states = [json.loads(r["state"]) for r in rows]
    m = da.load_cribl_model(a.backend, "plain_letter_qfirst")
    names = list(QUESTIONS)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"rows": []}
    res["rows"] = [x for x in res["rows"] if x["backend"] != m.device]
    for k in (1, 2, 4):
        qs = {n: QUESTIONS[n] for n in names[:k]}
        modes = ["one pass" if a.backend == "torch" else "sequential, prefix cached"]
        if a.backend == "torch" and k > 1:
            modes.append("sequential")
        for mode in modes:
            m.ask(states[0], qs)                          # warm
            ts = []
            for s in states:
                t0 = time.monotonic()
                if mode == "sequential":
                    for n, q in qs.items():
                        m.ask(s, {n: q})
                else:
                    m.ask(s, qs)
                ts.append((time.monotonic() - t0) * 1000)
            ts.sort()
            x = {"backend": m.device, "questions": k, "mode": mode,
                 "mean_ms": statistics.mean(ts), "p95_ms": ts[int(.95 * (len(ts) - 1))]}
            res["rows"].append(x)
            print(x, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
