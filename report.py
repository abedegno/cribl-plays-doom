"""Build results/REPORT.md from results/benchmarks/*.json, the decision
traces, and (if present) the probe outputs. Every number in the report is
read from those files; nothing is typed in by hand.

    python report.py
"""
from __future__ import annotations

import csv, glob, json, math, os, platform, statistics, subprocess

import doom_agent as da

R = "results"


def load(name):
    p = os.path.join(R, "benchmarks", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=2, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}{unit}"


def pearson(xs, ys):
    if len(xs) < 3 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


def row(name, label=None):
    b = load(name)
    if b is None:
        return None
    if "skipped" in b:
        return f"| {label or name} | skipped: {b['skipped']} |" + " |" * 10
    s = b["summary"]
    lat = (f"{f(s['decision_ms_p50'], 0)} / {f(s['decision_ms_mean'], 0)} / {f(s['decision_ms_p95'], 0)}"
           if s["decision_ms_mean"] and s["decision_ms_mean"] > 0.5 else "<1 / <1 / <1")
    return (f"| {label or name} | {f(s['kills_mean'])} ± {f(s['kills_sd'])} | {f(s['reward_mean'])} | "
            f"{s['deaths']}/{s['episodes']} | {f(s['survival_s_mean'], 1)} s | {f(s['accuracy'])} | "
            f"{s['decisions']} | {lat} | {f(s['p_selected_mean'])} | {s['fallbacks']} | "
            f"{f(s['decisions_per_wall_s'], 1)} | {f(s['game_s_per_wall_s'], 2)}x |")


HEAD = ("| config | kills (mean ± sd) | reward | deaths | survival | hit rate | decisions | "
        "latency p50 / mean / p95 (ms) | mean p(selected) | fallbacks | decisions/s (wall) | game speed |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|")


def table(names):
    rows = [row(n, lbl) for n, lbl in names]
    return HEAD + "\n" + "\n".join(r for r in rows if r)


def trace(name):
    p = os.path.join(R, "traces", f"{name}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def confidence_section(name="cribl"):
    rows = [r for r in trace(name) if r["p_selected"]]
    if not rows:
        return "_No Cribl trace yet._"
    out = []
    # decision level: does a confident pick agree more often with the rule?
    bins = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]
    out.append("| p(selected) | decisions | agrees with 8° rule | ATTACK share |\n|---|---|---|---|")
    for lo, hi in bins:
        sel = [r for r in rows if lo <= float(r["p_selected"]) < hi]
        if not sel:
            continue
        agree = statistics.mean(r["action"] == da.heuristic(json.loads(r["state"])) for r in sel)
        att = statistics.mean(r["action"] == "ATTACK" for r in sel)
        out.append(f"| {lo:.1f}–{min(hi, 1):.1f} | {len(sel)} | {agree:.2f} | {att:.2f} |")
    # episode level
    eps = {}
    for r in rows:
        eps.setdefault(int(r["episode"]), []).append(float(r["p_selected"]))
    b = load(name)
    kills = {e["episode"]: e["kills"] for e in b["summary"]["per_episode"]} if b else {}
    xs = [statistics.mean(v) for k, v in sorted(eps.items()) if k in kills]
    ys = [kills[k] for k in sorted(eps) if k in kills]
    r_ep = pearson(xs, ys)
    out.append(f"\nEpisode level: Pearson r between an episode's mean p(selected) and its kills "
               f"= **{f(r_ep)}** over {len(xs)} episodes.")
    # aligned states: does it fire?
    al = [r for r in rows if r["nearest_bearing"] != "" and abs(int(r["nearest_bearing"])) <= 8]
    if al:
        out.append(f"\nWhen the nearest monster is within ±8° ({len(al)} decisions) the model picks "
                   f"ATTACK {statistics.mean(r['action'] == 'ATTACK' for r in al):.0%} of the time, "
                   f"mean p(ATTACK) {statistics.mean(float(r['p_ATTACK']) for r in al):.2f}.")
    far = [r for r in rows if r["nearest_bearing"] != "" and abs(int(r["nearest_bearing"])) > 30]
    if far:
        good = statistics.mean(r["action"] == ("TURN_LEFT" if int(r["nearest_bearing"]) > 0 else "TURN_RIGHT")
                               for r in far)
        out.append(f"When it is more than 30° off ({len(far)} decisions) it turns the correct way "
                   f"{good:.0%} of the time.")
    return "\n".join(out)


def env():
    import importlib.metadata as md
    def v(p):
        try: return md.version(p)
        except Exception: return "not installed"
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip() or platform.processor()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    mem = f"{int(mem) / 2**30:.0f} GB" if mem.isdigit() else "?"
    return (f"{chip}, {mem} unified memory, {platform.system()} {platform.mac_ver()[0] or platform.release()}, "
            f"Python {platform.python_version()}; vizdoom {v('vizdoom')}, mlx {v('mlx')}, "
            f"mlx-lm {v('mlx-lm')}, torch {v('torch')}, transformers {v('transformers')}, peft {v('peft')}")


def main() -> int:
    md = ["# Cribl Decision 1.0 plays Doom: benchmark report", "",
          "_Generated by `report.py` from `results/benchmarks/*.json` and `results/traces/*.csv`._", "",
          f"**Hardware / software:** {env()}", ""]
    c = load("cribl")
    if c:
        cfg = c["config"]
        md += [f"**Cribl configuration:** backend `{cfg.get('model_device')}`, "
               f"dtype `{cfg.get('model_dtype')}`, format `{cfg['format']}`, criteria `{cfg['criteria']}`, "
               f"{cfg['tics']} tics per decision, seeds {cfg['seed']}…{cfg['seed'] + c['summary']['episodes'] - 1}.", ""]
    md += ["Kills come from the engine's KILLCOUNT. Reward is the scenario's score: +1 per kill and "
           "−1 for dying, which is the number the original project's table calls kills. Hit rate is "
           "HITCOUNT over shots fired. Latency is wall-clock time per decision, including prompt building. "
           "Game speed is seconds of game time per second of wall clock (1.0 = real time).", "",
           "## Baseline (lock-step: the game waits for every decision)", "",
           table([("random", "random"), ("heuristic", "hand-coded heuristic"),
                  ("jev", "Jev (TypeSafe API)"), ("cribl", "**Cribl Decision 1.0**")]), "",
           "## A. Prompt engineering (button descriptions)", "",
           table([("cribl", "rule: explicit ±8° tactical criteria"),
                  ("cribl_intent", "intent: what each button does"),
                  ("cribl_minimal", "minimal: names only"),
                  ("cribl_goal", "goal-oriented instructions")]), "",
           "## B. Candidate order (gameplay)", "",
           table([("cribl", "ATTACK, TURN_LEFT, TURN_RIGHT")] +
                 [(n, n.replace("cribl_order_", "").replace("-", ", "))
                  for n in ("cribl_order_TURN_LEFT-ATTACK-TURN_RIGHT",
                            "cribl_order_TURN_RIGHT-TURN_LEFT-ATTACK")]), ""]
    op = os.path.join(R, "order_probe.json")
    if os.path.exists(op):
        o = json.load(open(op))
        md += [f"Constant-state probe (`order_probe.py`, {o['n_states']} recorded game states, "
               f"format `{o['format']}`). All six orderings choose the same button on "
               f"**{o['agree_all_orderings']:.0%}** of states.", "",
               "| ordering | picks changed vs canonical | mean TV distance | p(first listed) | p(last listed) | agrees with rule | ATTACK rate |",
               "|---|---|---|---|---|---|---|"]
        for k, v in o["orderings"].items():
            md.append(f"| {k} | {v['flip_rate']:.2f} | {v['mean_tv']:.3f} | {v['p_first_slot']:.2f} | "
                      f"{v['p_last_slot']:.2f} | {v['oracle_agree']:.2f} | {v['attack_rate']:.2f} |")
        md += ["", "Wording on the same states:", "",
               "| criteria | same pick as `rule` | agrees with rule | …when rule says ATTACK | …TURN_LEFT | …TURN_RIGHT | ATTACK rate | mean p(selected) |",
               "|---|---|---|---|---|---|---|---|"]
        for k, v in o["wordings"].items():
            md.append(f"| {k} | {v['agree_with_rule_wording']:.2f} | {v['oracle_agree']:.2f} | "
                      f"{v.get('oracle_agree_when_ATTACK', float('nan')):.2f} | "
                      f"{v.get('oracle_agree_when_TURN_LEFT', float('nan')):.2f} | "
                      f"{v.get('oracle_agree_when_TURN_RIGHT', float('nan')):.2f} | "
                      f"{v['attack_rate']:.2f} | {v['p_selected_mean']:.2f} |")
        md.append("")
    md += ["## C. Decision interval", "",
           table([("cribl_tics2", "2 tics (57 ms of game time)"), ("cribl", "4 tics (114 ms)"),
                  ("cribl_tics8", "8 tics (229 ms)"), ("cribl_tics16", "16 tics (457 ms)")]), "",
           "## D. Confidence", "",
           table([("cribl", "no floor"), ("cribl_conf0.4", "confidence < 0.4 → heuristic (counted as fallback)")]), "",
           confidence_section("cribl"), "",
           "## E. Two typed questions per decision", "",
           table([("cribl", "action only"), ("cribl_with_danger", "action + in_danger (noul)")]), ""]
    fp = os.path.join(R, "batch_latency.json")
    if os.path.exists(fp):
        bl = json.load(open(fp))
        md += ["Forward-pass cost of asking several questions (`batch_latency.py`):", "",
               "| backend | questions | mode | mean ms | p95 ms |", "|---|---|---|---|---|"]
        for x in bl["rows"]:
            md.append(f"| {x['backend']} | {x['questions']} | {x['mode']} | {x['mean_ms']:.0f} | {x['p95_ms']:.0f} |")
        md.append("")
    md += ["## F. State representation (same information, restructured)", "",
           table([("cribl", "Cribl, jev state (6 monsters, nearest first)"),
                  ("cribl_state_rich", "Cribl, rich (+ armour, in-view count, last 3 actions)"),
                  ("cribl_state_compact", "Cribl, compact (nearest monster as its own field)"),
                  ("heuristic", "heuristic, jev state"),
                  ("heuristic_state_compact", "heuristic, compact state (sanity check: identical)")]), "",
           confidence_section("cribl_state_compact") if trace("cribl_state_compact") else "", ""]
    md += ["## Prompt format played", "",
           table([("cribl", "plain, question first, state last (prefix cached)"),
                  ("cribl_state_first", "plain, state first (no cache possible)")]), "",
           "## Real time (`--live`: the game runs at 35 tics/s while the model thinks)", "",
           table([("heuristic_live", "heuristic, live"), ("cribl_live", "Cribl, live, act for 4 tics"),
                  ("cribl_live_tics8", "Cribl, live, act for 8 tics"),
                  ("cribl_state_compact", "Cribl compact state, lock-step, 4 tics (reference)"),
                  ("cribl_compact_tics8", "Cribl compact state, lock-step, 8 tics (reference)"),
                  ("cribl_compact_live", "Cribl compact state, live, 4 tics"),
                  ("cribl_compact_live_tics8", "Cribl compact state, live, 8 tics")]), ""]
    live = [r for r in trace("cribl_live") if r["stale_tics"]]
    if live:
        st = [int(r["stale_tics"]) for r in live]
        md.append(f"In live play the state a decision was made from was on average "
                  f"**{statistics.mean(st):.1f} tics** old when the action was applied "
                  f"(p95 {sorted(st)[int(.95 * (len(st) - 1))]}).")
    fpp = os.path.join(R, "format_probe.json")
    if os.path.exists(fpp):
        S = json.load(open(fpp))["summary"]
        q = os.path.join(R, "format_probe_qfirst.json")
        if os.path.exists(q):
            S.update(json.load(open(q))["summary"])
        md += ["", "## Readout format probe (`probe_formats.py`, PyTorch + PEFT, adapter on vs off)", "",
               "| format | adapter acc | base acc | adapter p(true) | base p(true) | adapter slot mass on candidates | base mass | adapter Doom probe acc |",
               "|---|---|---|---|---|---|---|---|"]
        for k in sorted({k.split("/")[0] for k in S}):
            on, off = S.get(f"{k}/adapter"), S.get(f"{k}/base")
            if not on or not off:
                continue
            md.append(f"| {k} | {f(on['acc'])} | {f(off['acc'])} | {f(on['p_true'])} | {f(off['p_true'])} | "
                      f"{f(on['mass'])} | {f(off['mass'])} | {f(on['doom_acc'])} |")
    os.makedirs(R, exist_ok=True)
    open(os.path.join(R, "REPORT.md"), "w").write("\n".join(md) + "\n")
    print("wrote results/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
