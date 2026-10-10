"""Build the README chart and the social media assets from the result files.

    python tools/make_media.py

Writes:
  docs/kills-chart.png       bar chart of mean kills per configuration
  docs/share-card.png        1200x675 still for a post or comment
  docs/share-clip.mp4        1080x1080, 24 s gameplay clip with a caption band
  docs/share-clip.gif        480x480, 6 s loop of the same clip

Every number is read from results/benchmarks/*.json; the gameplay is
docs/cribl-doom.mp4, recorded by record.py from real model decisions.
"""
from __future__ import annotations

import json, os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SRC_VIDEO = os.path.join(DOCS, "cribl-doom.mp4")
STILL_FRAME = 958          # ATTACK at p 0.97, muzzle flash on a demon
CLIP_START_S, CLIP_LEN_S = 0, 24
GIF_START_S, GIF_LEN_S = 8, 6       # within the clip

# Palette: the validated categorical slots 1 and 2 (light surface), and the
# dark card colours.
CRIBL, BASE = (42, 120, 214), (235, 104, 52)
INK, INK2, INK3, RULE, PANEL = (21, 23, 26), (74, 79, 87), (115, 121, 131), (221, 224, 229), (255, 255, 255)
DARK_BG, DARK_INK, DARK_INK2, DARK_ACCENT = (18, 20, 22), (236, 238, 240), (170, 176, 185), (127, 176, 236)


def font(name: str, size: int, index: int = 0):
    paths = {
        "sans": ["/System/Library/Fonts/Avenir Next.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "sans_bold": ["/System/Library/Fonts/Avenir Next.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "num": ["/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "mono": ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    }[name]
    idx = {"sans": 7, "sans_bold": 0}.get(name, 0) if index == 0 else index
    for p in paths:
        try:
            return ImageFont.truetype(p, size, index=idx if p.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def summary(name):
    return json.load(open(os.path.join(ROOT, "results", "benchmarks", f"{name}.json")))["summary"]


ROWS = [  # (benchmark, label, mode, is_cribl)
    ("heuristic", "Hand-coded aiming rule", "lock-step", False),
    ("cribl_state_compact", "Cribl, nearest monster as its own field", "lock-step", True),
    ("cribl_compact_tics8", "Cribl, nearest monster, 8 tics per decision", "lock-step", True),
    ("cribl_state_rich", "Cribl, original state plus extras", "lock-step", True),
    ("cribl_compact_live_tics8", "Cribl, nearest monster, 8 tics per decision", "live", True),
    ("cribl", "Cribl, original Jev state", "lock-step", True),
    ("random", "Random buttons", "lock-step", False),
    ("cribl_compact_live", "Cribl, nearest monster, 4 tics per decision", "live", True),
]


def kills_chart(path: str, scale: int = 2):
    s = scale
    W, row_h, top, left = 800 * s, 46 * s, 92 * s, 330 * s
    H = top + row_h * len(ROWS) + 64 * s
    img = Image.new("RGB", (W, H), PANEL)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, W - 2, H - 2], radius=10 * s, outline=RULE, width=s)
    d.text((24 * s, 20 * s), "Average kills per episode", fill=INK, font=font("sans_bold", 22 * s))
    d.text((24 * s, 52 * s), "defend_the_center, 20 episodes per row, same seeds for every row",
           fill=INK3, font=font("sans", 14 * s))
    lx = W - 300 * s                                            # legend
    for i, (c, t) in enumerate([(CRIBL, "Cribl Decision 1.0"), (BASE, "Baseline")]):
        x = lx + i * 150 * s
        d.rounded_rectangle([x, 26 * s, x + 12 * s, 38 * s], radius=3 * s, fill=c)
        d.text((x + 18 * s, 23 * s), t, fill=INK2, font=font("sans", 14 * s))
    plot_w, vmax = W - left - 90 * s, 10.0
    X = lambda v: left + plot_w * max(0.0, min(v, vmax)) / vmax   # noqa: E731
    for t in range(0, 11, 2):                                   # recessive grid
        x = X(t)
        d.line([x, top - 6 * s, x, top + row_h * len(ROWS)], fill=(238, 240, 243), width=s)
        d.text((x, top + row_h * len(ROWS) + 8 * s), str(t), fill=INK3, font=font("mono", 12 * s), anchor="ma")
    for i, (name, label, mode, cribl) in enumerate(ROWS):
        sm = summary(name)
        k, sd = sm["kills_mean"], sm["kills_sd"]
        y = top + i * row_h
        d.text((24 * s, y + 4 * s), label, fill=INK, font=font("sans", 14 * s))
        d.text((24 * s, y + 22 * s), mode, fill=INK3, font=font("sans", 12 * s))
        by = y + 10 * s
        d.rounded_rectangle([X(0), by, max(X(k), X(0) + 4 * s), by + 18 * s], radius=4 * s,
                            fill=CRIBL if cribl else BASE)
        d.rectangle([X(0), by, X(0) + 4 * s, by + 18 * s], fill=CRIBL if cribl else BASE)  # square baseline end
        d.line([X(k - sd), by + 9 * s, X(k + sd), by + 9 * s], fill=INK3, width=2 * s)
        d.text((X(k + sd) + 8 * s, by + 1 * s), f"{k:.2f}", fill=INK, font=font("mono", 13 * s))
    d.line([X(0), top - 6 * s, X(0), top + row_h * len(ROWS)], fill=INK3, width=s)
    d.text((24 * s, H - 28 * s), "The thin line marks one standard deviation either side of the mean. "
           "Lock-step means the game waits for each decision.", fill=INK3, font=font("sans", 12 * s))
    img.save(path, optimize=True)


def read_frame(index: int) -> Image.Image:
    import imageio.v2 as iio
    r = iio.get_reader(SRC_VIDEO)
    try:
        return Image.fromarray(r.get_data(index))
    finally:
        r.close()


def share_card(path: str):
    W, H = 1200, 675
    img = Image.new("RGB", (W, H), DARK_BG)
    d = ImageDraw.Draw(img)
    still = read_frame(STILL_FRAME).resize((640, 480), Image.LANCZOS)
    mask = Image.new("L", still.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, 639, 479], radius=14, fill=255)
    img.paste(still, (40, 98), mask)
    x = 720
    d.text((40, 36), "VIZDOOM  DEFEND_THE_CENTER  ZERO-SHOT", fill=DARK_INK2, font=font("mono", 15))
    d.text((x, 92), "Cribl Decision 1.0", fill=DARK_INK, font=font("sans_bold", 40))
    d.text((x, 140), "plays Doom", fill=DARK_INK, font=font("sans_bold", 40))
    d.text((x, 200), "A 4B open decision model on a Mac mini reads\nthe game state as JSON and picks a button.",
           fill=DARK_INK2, font=font("sans", 19), spacing=6)
    c, h = summary("cribl_state_compact"), summary("heuristic")
    for i, (val, line1, col) in enumerate([
            (f"{c['kills_mean']:.2f}", "kills per episode\nCribl Decision 1.0", DARK_ACCENT),
            (f"{h['kills_mean']:.2f}", "kills per episode\nhand-coded aiming rule", (240, 140, 100))]):
        y = 290 + i * 112
        d.text((x, y), val, fill=col, font=font("num", 84))
        d.multiline_text((x + 150, y + 16), line1, fill=DARK_INK2, font=font("sans", 18), spacing=6)
    lat = c["decision_ms_p50"]
    d.text((x, 520), f"{lat:.0f} ms per decision on a Mac mini (M4 Pro).\nSame 20 episodes for both. No pixels, no fine-tuning.",
           fill=DARK_INK2, font=font("sans", 17), spacing=6)
    d.text((40, 612), "github.com/abedegno/cribl-plays-doom", fill=DARK_INK2, font=font("mono", 17))
    img.save(path, optimize=True)


def _band(width: int, height: int) -> Image.Image:
    c, h = summary("cribl_state_compact"), summary("heuristic")
    band = Image.new("RGB", (width, height), DARK_BG)
    d = ImageDraw.Draw(band)
    sc = width / 1080
    d.text((40 * sc, 34 * sc), "Cribl Decision 1.0 plays Doom", fill=DARK_INK, font=font("sans_bold", int(52 * sc)))
    d.text((40 * sc, 104 * sc), "Local 4B decision model. JSON game state in, one button out.",
           fill=DARK_INK2, font=font("sans", int(28 * sc)))
    d.text((40 * sc, 158 * sc), f"{c['kills_mean']:.2f}", fill=DARK_ACCENT, font=font("num", int(70 * sc)))
    d.text((190 * sc, 176 * sc), "Cribl Decision 1.0", fill=DARK_INK2, font=font("sans", int(26 * sc)))
    d.text((520 * sc, 158 * sc), f"{h['kills_mean']:.2f}", fill=(240, 140, 100), font=font("num", int(70 * sc)))
    d.text((670 * sc, 176 * sc), "hand-coded rule", fill=DARK_INK2, font=font("sans", int(26 * sc)))
    d.text((40 * sc, 232 * sc), "mean kills per episode over the same 20 episodes", fill=DARK_INK2,
           font=font("sans", int(22 * sc)))
    return band


def share_clip(mp4: str, gif: str):
    import imageio.v2 as iio
    r = iio.get_reader(SRC_VIDEO)
    fps = r.get_meta_data().get("fps", 35)
    size, band_h = 1080, 270
    band = np.asarray(_band(size, band_h))
    w = iio.get_writer(mp4, fps=fps, codec="libx264", quality=None, macro_block_size=None,
                       pixelformat="yuv420p",
                       ffmpeg_params=["-crf", "24", "-preset", "slow",
                                      "-movflags", "+faststart"])
    try:
        for i, f in enumerate(r):
            t = i / fps
            if t >= CLIP_START_S + CLIP_LEN_S:
                break
            game = Image.fromarray(f)
            if CLIP_START_S <= t < CLIP_START_S + CLIP_LEN_S:
                frame = np.vstack([band, np.asarray(game.resize((size, size - band_h), Image.LANCZOS))])
                w.append_data(frame)
    finally:
        w.close(); r.close()
    # GIF from the finished clip, with one palette built for the segment
    import subprocess, tempfile, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    seg = ["-ss", str(GIF_START_S), "-t", str(GIF_LEN_S), "-i", mp4]
    with tempfile.TemporaryDirectory() as tmp:
        pal = os.path.join(tmp, "pal.png")
        subprocess.run([ff, "-y", "-loglevel", "error", *seg, "-vf",
                        "fps=10,scale=480:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff", pal],
                       check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", *seg, "-i", pal, "-lavfi",
                        "fps=10,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                        gif], check=True)


def main():
    kills_chart(os.path.join(DOCS, "kills-chart.png"))
    share_card(os.path.join(DOCS, "share-card.png"))
    share_clip(os.path.join(DOCS, "share-clip.mp4"), os.path.join(DOCS, "share-clip.gif"))
    for n in ("kills-chart.png", "share-card.png", "share-clip.mp4", "share-clip.gif"):
        p = os.path.join(DOCS, n)
        print(f"  {n:18s} {os.path.getsize(p) / 1e6:5.2f} MB")


if __name__ == "__main__":
    main()
