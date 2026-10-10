"""Which prompt format does the adapter actually respond to?

Cribl did not publish the prompt text or decision-slot layout used in
training. A LoRA trained on one format should help most on that format, so
each candidate format is scored on questions with known answers, with the
adapter on and off:

    acc       fraction where the argmax candidate is correct
    p_true    mean probability on the correct candidate
    mass      share of the full-vocab softmax on the candidate rows at the slot
    uplift    adapter minus base, for acc and p_true

Every item is asked twice, with candidates in original and reversed order,
so position bias shows up as lost accuracy rather than hiding.

    python probe_formats.py --out results/format_probe.json
"""
from __future__ import annotations

import argparse, json, os, statistics, time

from cribl_decision import FORMATS, CriblDecisionModel, Question

# (state, instructions, {candidate: description}, correct candidate)
# Deliberately spread across task families named on the model card --
# classification, routing, telemetry, reasoning, knowledge -- plus the Doom
# rule, so the winner is not just whichever format suits Doom.
PROBES: list[tuple] = [
    ("The package arrived crushed and the screen is cracked. I want my money back.",
     "Which team should handle this message?",
     {"returns": "refunds and returns", "shipping": "delivery tracking and delays",
      "billing": "invoices and payment methods"}, "returns"),
    ("Where is my parcel? Tracking hasn't updated in six days.",
     "Which team should handle this message?",
     {"returns": "refunds and returns", "shipping": "delivery tracking and delays",
      "billing": "invoices and payment methods"}, "shipping"),
    ("My card was charged twice for the same invoice.",
     "Which team should handle this message?",
     {"returns": "refunds and returns", "shipping": "delivery tracking and delays",
      "billing": "invoices and payment methods"}, "billing"),
    ({"event": "connect", "src_ip": "10.0.0.4", "dst_ip": "93.184.216.34", "dst_port": 443, "proto": "tcp"},
     "Which event type is this?",
     {"network": "network activity", "file": "file activity", "process": "process activity"}, "network"),
    ({"event": "open", "path": "/etc/shadow", "mode": "r", "uid": 0},
     "Which event type is this?",
     {"network": "network activity", "file": "file activity", "process": "process activity"}, "file"),
    ({"event": "exec", "image": "/usr/bin/curl", "parent": "bash", "pid": 4411},
     "Which event type is this?",
     {"network": "network activity", "file": "file activity", "process": "process activity"}, "process"),
    ('Failed password for invalid user admin from 185.220.101.7 port 52114 ssh2',
     "What is the severity of this log line?",
     {"info": "routine, no action", "warning": "suspicious, worth reviewing", "critical": "outage or confirmed compromise"},
     "warning"),
    ('kernel: Out of memory: Killed process 1234 (postgres); database unavailable',
     "What is the severity of this log line?",
     {"info": "routine, no action", "warning": "suspicious, worth reviewing", "critical": "outage or confirmed compromise"},
     "critical"),
    ('GET /healthz 200 2ms',
     "What is the severity of this log line?",
     {"info": "routine, no action", "warning": "suspicious, worth reviewing", "critical": "outage or confirmed compromise"},
     "info"),
    ("This film was a joy from start to finish; I'd watch it again tomorrow.",
     "What is the sentiment of the review?",
     {"positive": "the reviewer liked it", "negative": "the reviewer disliked it"}, "positive"),
    ("Two hours of my life I will never get back. Dreadful acting.",
     "What is the sentiment of the review?",
     {"positive": "the reviewer liked it", "negative": "the reviewer disliked it"}, "negative"),
    ({"a": 17, "b": 25}, "What is a + b?",
     {"42": None, "41": None, "32": None, "52": None}, "42"),
    ({"a": 9, "b": 7}, "What is a times b?",
     {"56": None, "63": None, "72": None, "16": None}, "63"),
    ({"list": [4, 18, 7, 2]}, "Which value is the largest in the list?",
     {"4": None, "18": None, "7": None, "2": None}, "18"),
    ("", "What is the capital of France?",
     {"Lyon": None, "Paris": None, "Marseille": None}, "Paris"),
    ("", "Which planet is closest to the Sun?",
     {"Venus": None, "Earth": None, "Mercury": None, "Mars": None}, "Mercury"),
    ({"user_request": "Book me a table for two at 7pm tonight"},
     "Which tool should the agent call?",
     {"search_web": "look something up online", "make_reservation": "book restaurants or hotels",
      "send_email": "send an email"}, "make_reservation"),
    ({"user_request": "Email Sam the slides from today"},
     "Which tool should the agent call?",
     {"search_web": "look something up online", "make_reservation": "book restaurants or hotels",
      "send_email": "send an email"}, "send_email"),
    ({"temperature_c": 31, "rule": "alert if temperature_c > 30"},
     "Should an alert be raised under the rule?",
     {"alert": "the rule fires", "no_alert": "the rule does not fire"}, "alert"),
    ({"temperature_c": 22, "rule": "alert if temperature_c > 30"},
     "Should an alert be raised under the rule?",
     {"alert": "the rule fires", "no_alert": "the rule does not fire"}, "no_alert"),
]

DOOM_Q = ("Pick the single best button to press right now.", {
    "ATTACK": "Fire. Correct when the nearest monster's bearing is between -8 and +8.",
    "TURN_LEFT": "Rotate left. Correct only when the nearest monster's bearing is greater than +8.",
    "TURN_RIGHT": "Rotate right. Correct only when the nearest monster's bearing is less than -8."})
for bearing, want in [(0, "ATTACK"), (5, "ATTACK"), (-6, "ATTACK"), (40, "TURN_LEFT"),
                      (120, "TURN_LEFT"), (-35, "TURN_RIGHT"), (-150, "TURN_RIGHT")]:
    PROBES.append(({"health": 100, "ammo": 20,
                    "monsters_nearest_first": [{"kind": "Demon", "distance": 300, "bearing_deg": bearing}],
                    "note": "bearing 0 = dead ahead. POSITIVE means LEFT, NEGATIVE means RIGHT."},
                   DOOM_Q[0], DOOM_Q[1], want))


def items():
    for i, (state, instr, crit, want) in enumerate(PROBES):
        for order in ("fwd", "rev"):
            keys = list(crit) if order == "fwd" else list(reversed(crit))
            yield i, order, state, Question("choice", instr, {k: crit[k] for k in keys}), want


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--out", default="results/format_probe.json")
    a = ap.parse_args()
    m = CriblDecisionModel()
    rows, summary = [], {}
    for fname in a.formats.split(","):
        m.fmt = FORMATS[fname]; m._row_cache.clear()
        for use in (True, False):
            m.use_adapter = use
            tag = f"{fname}/{'adapter' if use else 'base'}"
            accs, ptrue, mass, skipped = [], [], [], 0
            t0 = time.monotonic()
            for i, order, state, q, want in items():
                try:
                    ans, _ = m.choose(state, q)
                except ValueError as e:      # e.g. names sharing a first token
                    skipped += 1; continue
                accs.append(ans.choice == want); ptrue.append(ans.probabilities[want])
                mass.append(ans.candidate_mass)
                rows.append({"format": fname, "adapter": use, "item": i, "order": order,
                             "want": want, "choice": ans.choice, "p_true": ans.probabilities[want],
                             "mass": ans.candidate_mass, "probabilities": ans.probabilities,
                             "doom": i >= len(PROBES) - 7})
            doom = [r for r in rows if r["format"] == fname and r["adapter"] == use and r["doom"]]
            summary[tag] = {
                "n": len(accs), "skipped": skipped, "acc": statistics.mean(accs) if accs else None,
                "p_true": statistics.mean(ptrue) if ptrue else None,
                "mass": statistics.mean(mass) if mass and mass[0] == mass[0] else None,
                "doom_acc": statistics.mean(r["choice"] == r["want"] for r in doom) if doom else None,
                "seconds": round(time.monotonic() - t0, 1)}
            print(tag, summary[tag], flush=True)
    for fname in a.formats.split(","):
        on, off = summary[f"{fname}/adapter"], summary[f"{fname}/base"]
        if on["acc"] is not None:
            on["uplift_acc"] = on["acc"] - off["acc"]
            on["uplift_p_true"] = on["p_true"] - off["p_true"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print("\nformat              acc   p_true  mass   doom   uplift_acc uplift_p")
    for fname in a.formats.split(","):
        s = summary[f"{fname}/adapter"]
        if s["acc"] is None:
            print(f"{fname:18s} skipped ({s['skipped']})"); continue
        print(f"{fname:18s} {s['acc']:.2f}  {s['p_true']:.2f}   "
              f"{(s['mass'] or float('nan')):.2f}   {(s['doom_acc'] if s['doom_acc'] is not None else float('nan')):.2f}   "
              f"{s['uplift_acc']:+.2f}      {s['uplift_p_true']:+.2f}   (skipped {s['skipped']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
