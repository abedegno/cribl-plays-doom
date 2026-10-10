"""Candidate order and wording sensitivity on CONSTANT states (experiment B,
and the wording half of A).

States are sampled from a recorded gameplay trace, so they are real game
situations. Each state is scored under all 6 orderings of the three buttons
and under each criteria wording, with nothing else changed. Reported:

  agree_all     share of states where every ordering picks the same button
  flip_rate     share of (state, ordering) pairs whose pick differs from the
                canonical ordering's
  mean_tv       mean total-variation distance from the canonical distribution
  p_first_slot  mean probability given to whichever button is listed first
                (1/3 means no positional preference)
  oracle_agree  agreement with the 8-degree rule (the action the `rule`
                criteria describe), per ordering / wording

    python order_probe.py --trace results/traces/cribl.csv --n 150
"""
from __future__ import annotations

import argparse, csv, itertools, json, os, random, statistics

import doom_agent as da
from cribl_decision import Question

CANON = da.OPTION_ORDER


def tv(p, q):
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="results/traces/cribl.csv")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--format", default="plain_letter_qfirst")
    ap.add_argument("--out", default="results/order_probe.json")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.trace)))
    states = [json.loads(r["state"]) for r in rows if r["state"]]
    states = [s for s in states if s["monsters_nearest_first"]]
    rng = random.Random(a.seed)
    # stratify by the oracle action so aligned states are not drowned out
    by = {}
    for s in states:
        by.setdefault(da.heuristic(s), []).append(s)
    sample = []
    for k, v in sorted(by.items()):
        sample += rng.sample(v, min(len(v), a.n // 3))
    print(f"{len(sample)} states: " + ", ".join(f"{k} {min(len(v), a.n // 3)}" for k, v in sorted(by.items())))

    model = da.load_cribl_model(a.backend, a.format)
    out = {"format": a.format, "n_states": len(sample), "orderings": {}, "wordings": {}}

    def run(criteria, order):
        crit = da.CRITERIA[criteria]
        q = Question("choice", da.INSTRUCTIONS[da.CRITERIA_INSTRUCTIONS[criteria]],
                     {k: crit[k] for k in order})
        return [model.choose(s, q)[0] for s in sample]

    oracle = [da.heuristic(s) for s in sample]
    canon = run("rule", CANON)
    picks = {}
    for perm in itertools.permutations(CANON):
        ans = canon if list(perm) == CANON else run("rule", list(perm))
        name = ",".join(perm)
        picks[name] = [x.choice for x in ans]
        out["orderings"][name] = {
            "flip_rate": statistics.mean(x.choice != c.choice for x, c in zip(ans, canon)),
            "mean_tv": statistics.mean(tv(x.probabilities, c.probabilities) for x, c in zip(ans, canon)),
            "p_first_slot": statistics.mean(x.probabilities[perm[0]] for x in ans),
            "p_last_slot": statistics.mean(x.probabilities[perm[-1]] for x in ans),
            "oracle_agree": statistics.mean(x.choice == o for x, o in zip(ans, oracle)),
            "attack_rate": statistics.mean(x.choice == "ATTACK" for x in ans),
        }
        print("order", name, {k: round(v, 3) for k, v in out["orderings"][name].items()}, flush=True)
    out["agree_all_orderings"] = statistics.mean(
        len({picks[p][i] for p in picks}) == 1 for i in range(len(sample)))

    for crit in da.CRITERIA:
        ans = canon if crit == "rule" else run(crit, CANON)
        out["wordings"][crit] = {
            "agree_with_rule_wording": statistics.mean(x.choice == c.choice for x, c in zip(ans, canon)),
            "oracle_agree": statistics.mean(x.choice == o for x, o in zip(ans, oracle)),
            "attack_rate": statistics.mean(x.choice == "ATTACK" for x in ans),
            "p_selected_mean": statistics.mean(x.p_selected for x in ans),
            "confidence_mean": statistics.mean(x.confidence for x in ans),
            # oracle agreement split by what the rule says should happen
            **{f"oracle_agree_when_{k}": statistics.mean(
                x.choice == o for x, o in zip(ans, oracle) if o == k) for k in sorted(set(oracle))},
        }
        print("wording", crit, {k: round(v, 3) for k, v in out["wordings"][crit].items()}, flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nall 6 orderings agree on {out['agree_all_orderings']:.1%} of states · wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
