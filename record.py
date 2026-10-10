"""Record a brain playing, with a live decision overlay, to mp4.

    python record.py --brain cribl --out docs/cribl-doom.mp4 --seconds 60
    python record.py --brain cribl --live --out docs/cribl-doom-live.mp4

Frames are captured every tic (35 per second), so playback runs at game
speed. The overlay shows the decision actually being executed: the chosen
button, every candidate's probability, confidence, the model's latency, health,
ammo and kills. Every label comes from the Decision object returned by the
brain; nothing on screen is computed by the recorder.

Default (lock-step): the game waits for each decision, so the video shows the
policy at full quality regardless of latency. --live: the game runs at 35
tics/s of wall clock while the model thinks, so the video shows what real-time
play looks like on this machine (actions are applied late; the overlay shows
how many tics stale the acted-on state was).
"""
from __future__ import annotations

import argparse, os, time

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw, ImageFont

import doom_agent as da

COLORS = {"ATTACK": (255, 110, 90), "TURN_LEFT": (120, 200, 255), "TURN_RIGHT": (150, 255, 160)}


def _font(size: int):
    """The PIL default font is ~11px and unreadable at 640x480."""
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


BIG, MID, SMALL = _font(20), _font(15), _font(12)


def overlay(frame: np.ndarray, label: str, d: da.Decision | None, health: int, ammo: int,
            kills: int, mean_ms: float, stale: int | None = None) -> np.ndarray:
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    dr.rectangle([0, 0, w, 112], fill=(0, 0, 0, 185))
    dr.text((12, 5), label, fill=(200, 200, 200), font=SMALL)
    if d is not None:
        dr.text((12, 24), d.action, fill=COLORS.get(d.action, (255, 255, 255)), font=BIG)
        if d.fallback:
            dr.text((12, 52), f"FALLBACK: {d.fallback}", fill=(255, 80, 80), font=SMALL)
        elif d.confidence is not None:
            dr.text((12, 52), f"confidence {d.confidence:.2f} · p {d.p_selected:.2f}",
                    fill=(200, 200, 200), font=SMALL)
        if d.model_ms:
            dr.text((12, 94), f"decision latency {d.model_ms:.0f} ms (mean {mean_ms:.0f})"
                    + (f" · acted {stale} tics late" if stale is not None else ""),
                    fill=(150, 150, 150), font=SMALL)
        if d.probabilities:                      # one bar per candidate
            x0, bw = 236, 150
            for i, k in enumerate(da.BUTTONS):
                p = d.probabilities.get(k, 0.0)
                y = 28 + i * 21
                dr.text((x0, y), k, fill=(220, 220, 220), font=SMALL)
                dr.rectangle([x0 + 84, y + 2, x0 + 84 + bw, y + 14], outline=(90, 90, 90))
                dr.rectangle([x0 + 84, y + 2, x0 + 84 + int(bw * p), y + 14],
                             fill=COLORS[k] + (230,))
                dr.text((x0 + 90 + bw, y), f"{p:.2f}", fill=(220, 220, 220), font=SMALL)
    dr.text((w - 100, 24), f"kills {kills}", fill=(255, 210, 120), font=BIG)
    dr.text((w - 100, 52), f"health {health}", fill=(255, 140, 140), font=SMALL)
    dr.text((w - 100, 70), f"ammo {ammo}", fill=(200, 200, 120), font=SMALL)
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser()
    da.add_args(ap)
    ap.add_argument("--out", default="docs/cribl-doom.mp4")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--trace", default="", help="also write the decision trace CSV")
    ap.set_defaults(brain="cribl")
    a = ap.parse_args()

    import imageio
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    brain = da.build_brain(a)
    label = {"cribl": "CRIBL DECISION 1.0 (local, " +
                      getattr(getattr(brain, "model", None), "device", "") + ")",
             "jev": "JEV (TypeSafe API)", "heuristic": "HAND-CODED HEURISTIC",
             "random": "RANDOM"}[a.brain] + (" · LIVE" if a.live else " · lock-step")
    g = da.make_game(render=True)
    da.warmup(brain, g)
    tracer = da.Tracer(a.trace or None, a.brain)
    want = int(a.seconds * da.TICS_PER_SEC)
    frames, ms = [0], []
    writer = imageio.get_writer(a.out, fps=da.TICS_PER_SEC, codec="libx264",
                                quality=8, macro_block_size=None)
    last = {"id": None}

    def on_tic(game, d, desc):
        if frames[0] >= want:
            return
        st = game.get_state()
        if st is None or st.screen_buffer is None:
            return
        if id(d) != last["id"]:
            last["id"] = id(d)
            if d.model_ms:
                ms.append(d.model_ms)
        gv = st.game_variables
        writer.append_data(overlay(st.screen_buffer, label, d, int(gv[0]), int(gv[1]),
                                   int(gv[6]), sum(ms) / len(ms) if ms else 0.0,
                                   d.extra.get("stale_tics") if a.live else None))
        frames[0] += 1

    episodes, ep = [], 0
    try:
        while frames[0] < want:
            ep += 1
            seed = None if a.seed < 0 else a.seed + ep - 1
            fn = da.run_episode_live if a.live else da.run_episode
            r = fn(g, brain, ep, seed, a.tics, da.state_mode(a), tracer, on_tic=on_tic)
            episodes.append(r)
            print(f"  episode {ep}: {r.kills} kills, {r.decisions} decisions, "
                  f"{r.fallbacks} fallbacks", flush=True)
    finally:
        writer.close(); tracer.close(); g.close()
    n = sum(r.decisions for r in episodes)
    print(f"\n  wrote {a.out} · {frames[0]} frames ≈ {frames[0]/da.TICS_PER_SEC:.0f}s · "
          f"{ep} episodes · {n} decisions · {sum(ms)/max(len(ms),1):.0f} ms mean model time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
