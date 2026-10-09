#!/usr/bin/env python3
"""
Speculative decoding schematic -- ICLR-style flat vector, 7:1 banner.

Renders
    figures/speculative_decoding.pdf    vector, for \\includegraphics
    figures/speculative_decoding.png    300 dpi preview

Two things drove this rewrite:

1. Print size.  A 7:1 figure placed at \\textwidth is only ~0.79 in tall, so the
   ratio of font size to canvas height -- not the font size in points -- decides
   legibility.  The figure is therefore authored at 7 x 1 in and the script
   prints the effective point size after the \\textwidth downscale.

2. Machine-checkable layout.  Nobody eyeballs this file, so text is measured with
   the renderer *before* the boxes are sized, zones are packed to fill the width
   exactly, and ``verify()`` reports out-of-canvas artists and cross-group
   overlaps.  Plus a crude ink histogram, which is the closest thing to a look.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh-mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# --------------------------------------------------------------------- style
INK = "#333333"
MUTED = "#6B6B6B"
FAINT = "#9A9A9A"
BLUE = "#AFCBE3"
BLUE_L = "#E4EEF7"
AMBER = "#E8C99B"
GREEN = "#AFD8AC"
GREEN_F = "#DCEEDA"
RED = "#E8AFAF"
RED_F = "#F7E4E4"

STROKE = 1.0          # uniform stroke weight, the house rule
DARK_GREEN = "#3B6E35"
DARK_RED = "#9E3838"

# Authored at the printed size: 5.5 in == \textwidth in a single-column ICLR
# paper.  Nothing is downscaled, so the point sizes below are the printed sizes.
# That matters: a 7:1 figure at \textwidth is only 0.786 in tall, and any
# \includegraphics scaling would shrink the labels with it.
FIG_W = 5.5
FIG_H = FIG_W / 7.0
TEXTWIDTH_IN = FIG_W

# fonts (points, and therefore points on the page)
F_TITLE, F_SUB, F_TOK, F_NOTE, F_PHASE, F_TINY = 9.0, 7.5, 8.5, 8.0, 7.5, 7.0

# vertical layout (inches).  FIG_H is 0.786, so each row is explicitly budgeted:
#   phase 0.654-0.758 | bars 0.546-0.630 | row 0.256-0.560
#   notes 0.125-0.236 | repeat 0.002-0.099
BOX_Y0, BOX_Y1 = 0.256, 0.560      # the two model boxes
SQ_Y0, SQ_Y1 = 0.276, 0.540        # token squares
LAB_Y = 0.408                      # draft token label
D_TOK_Y = 0.472                    # verified token label
MARK_Y = 0.350                     # check / cross
BAR_Y, BAR_MAX = 0.546, 0.630      # target-probability bars
PHASE_Y = 0.706
NOTE_Y = 0.180
REPEAT_Y = 0.050
LOOP_Y = 0.050
MARGIN = 0.06
MIN_GAP = 0.07

PARTS = []


def reg(name, artist, group, kind):
    PARTS.append(dict(name=name, artist=artist, group=group, kind=kind))
    return artist


fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
FIG_DPI = fig.get_dpi()


# ---------------------------------------------------------------- measuring
_MCACHE = {}


def measure(s, fs, weight="normal"):
    """Return (width, height) of a string in inches."""
    key = (s, fs, weight)
    if key in _MCACHE:
        return _MCACHE[key]
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    val = (bb.width / FIG_DPI, bb.height / FIG_DPI)
    _MCACHE[key] = val
    return val


def tw(s, fs, weight="normal"):
    return measure(s, fs, weight)[0]


# ------------------------------------------------------------ zone A: draft
A_LABELS = ["Draft Model", "small", "$q(x)$ draft"]
A_PAD = 0.10
A_W = tw("Draft Model", F_TITLE, "bold") + A_PAD

# --------------------------------------------------------- zone B / D: row
SQ_W = max(0.20, tw("$t_1$", F_TOK) + 0.04)
SQ_H = SQ_Y1 - SQ_Y0
N_TOK = 4
B_PAD = 0.07
GAP_TOK = SQ_W * 0.40
ROW_W = N_TOK * SQ_W + (N_TOK - 1) * GAP_TOK
B_W = ROW_W + 2 * B_PAD
D_W = B_W

# ----------------------------------------------------------- zone C: target
C_W = tw("Target Model", F_TITLE, "bold") + 0.12

# ------------------------------------------------------- zone E: resampling
# the zone is widened to fit its caption; the square itself stays centre-aligned
E_W = max(SQ_W, tw("resampled", F_SUB))

# ------------------------------------------------------ phases and packing
PHASES = [
    ("1  Draft, serial", B_W, None),
    ("2  Verify, one pass", C_W + D_W + 0.0, None),
    ("3  Accept / resample", E_W, None),
]

ZONE_W = [A_W, B_W, C_W, D_W, E_W]
usable = FIG_W - 2 * MARGIN
gap = (usable - sum(ZONE_W)) / (len(ZONE_W) - 1)
if gap < 0:
    raise SystemExit(
        f"zones need {sum(ZONE_W):.2f} in but only {usable:.2f} in is usable"
    )
if gap < MIN_GAP:
    print(f"WARNING: computed gap {gap:.3f} in < MIN_GAP {MIN_GAP}")

x = MARGIN
X = []
for w in ZONE_W:
    X.append(x)
    x += w + gap
X_END = X[-1] + ZONE_W[-1]
if X_END > FIG_W - MARGIN + 1e-9:
    raise SystemExit("packing overflowed")


# ------------------------------------------------------------------ helpers
def rbox(px, py, w, h, fc, ec=INK, lw=STROKE, ls="-", name=None, group=None, z=2):
    p = FancyBboxPatch((px, py), w, h,
                       boxstyle="round,pad=0,rounding_size=0.035",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    return reg(name or f"box({px:.3f},{py:.3f})", p, group, "box")


def sq(px, py, w, h, fc, ec=INK, lw=STROKE, ls="-", name=None, group=None, z=3):
    p = Rectangle((px, py), w, h, linewidth=lw, edgecolor=ec, facecolor=fc,
                  linestyle=ls, zorder=z)
    ax.add_patch(p)
    return reg(name or f"sq({px:.3f},{py:.3f})", p, group, "box")


def arr(x0, x1, py, lw=STROKE, ms=5, color=INK, name=None, group=None, z=5):
    a = FancyArrowPatch((x0, py), (x1, py), arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)
    return reg(name or f"arr({x0:.3f}->{x1:.3f})", a, group, "arrow")


def loopback(x0, x1, py, name="repeat-arrow", group="R"):
    a = FancyArrowPatch((x0, py), (x1, py), arrowstyle="-|>", mutation_scale=6,
                        linewidth=STROKE, color=MUTED, shrinkA=0, shrinkB=0,
                        zorder=5)
    ax.add_patch(a)
    return reg(name, a, group, "arrow")


def txt(px, py, s, fs=F_NOTE, color=INK, ha="center", va="center",
        weight="normal", name=None, group=None, z=6, mask=False):
    # mask paints a white plate behind the label so a line under it stays broken
    t = ax.text(px, py, s, fontsize=fs, color=color, ha=ha, va=va,
                fontweight=weight, zorder=z,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0)
                if mask else None)
    return reg(name or f"txt('{s}')", t, group, "text")


LINK_Y = (SQ_Y0 + SQ_Y1) / 2

# --------------------------------------------------------------- draw: A
rbox(X[0], BOX_Y0, A_W, BOX_Y1 - BOX_Y0, BLUE, name="draft-model", group="A")
txt(X[0] + A_W / 2, 0.470, "Draft Model", fs=F_TITLE, weight="bold",
    group="A", name="draft-title")
txt(X[0] + A_W / 2, 0.350, "small", fs=F_SUB, color=MUTED, group="A",
    name="draft-sub")
txt(X[0] + A_W / 2, NOTE_Y, "$q(x)$ draft", fs=F_NOTE, color=MUTED,
    group="A", name="draft-note")

arr(X[0] + A_W + 0.015, X[1] - 0.005, LINK_Y, name="arr-A-B", group="LAB")

# --------------------------------------------------------------- draw: B
rbox(X[1], BOX_Y0 + 0.012, B_W, (BOX_Y1 - BOX_Y0) - 0.024, "none", ec=BLUE,
     ls=(0, (2.5, 2.0)), name="draft-group", group="B")
for i in range(N_TOK):
    tx = X[1] + B_PAD + i * (SQ_W + GAP_TOK)
    sq(tx, SQ_Y0, SQ_W, SQ_H, BLUE_L, ec=BLUE, name=f"draft-tok-{i+1}", group="B")
    txt(tx + SQ_W / 2, LAB_Y, f"$t_{i+1}$", fs=F_TOK, group="B",
        name=f"draft-tok-{i+1}-lab")
    if i < N_TOK - 1:
        arr(tx + SQ_W, tx + SQ_W + GAP_TOK, LINK_Y, ms=4,
            name=f"arr-serial-{i+1}", group="B")

txt(X[1] + B_W / 2, NOTE_Y, "K tokens, serial", fs=F_NOTE, color=MUTED,
    group="B", name="b-note")

arr(X[1] + B_W + 0.02, X[2] - 0.01, LINK_Y, lw=2.0, ms=7,
    name="arr-B-C", group="LBC")

# --------------------------------------------------------------- draw: C
rbox(X[2], BOX_Y0 - 0.015, C_W, (BOX_Y1 - BOX_Y0) + 0.03, AMBER,
     name="target-model", group="C")
txt(X[2] + C_W / 2, 0.470, "Target Model", fs=F_TITLE, weight="bold",
    group="C", name="target-title")
txt(X[2] + C_W / 2, 0.350, "one pass", fs=F_SUB, color=MUTED, group="C",
    name="target-sub")
txt(X[2] + C_W / 2, NOTE_Y, "$p(x)$ target", fs=F_NOTE, color=MUTED,
    group="C", name="target-note")

arr(X[2] + C_W + 0.015, X[3] - 0.005, LINK_Y, name="arr-C-D", group="LCD")

# --------------------------------------------------------------- draw: D
PROBS = [0.95, 0.82, 0.70, 0.16]
ACCEPT = [True, True, True, False]
for i in range(N_TOK):
    tx = X[3] + B_PAD + i * (SQ_W + GAP_TOK)
    ok = ACCEPT[i]
    sq(tx, SQ_Y0, SQ_W, SQ_H, GREEN_F if ok else RED_F,
       ec=GREEN if ok else RED, lw=1.3, name=f"ver-tok-{i+1}", group="D")
    txt(tx + SQ_W / 2, D_TOK_Y, f"$t_{i+1}$", fs=F_SUB, group="D",
        name=f"ver-tok-{i+1}-lab")
    txt(tx + SQ_W / 2, MARK_Y, "$\\checkmark$" if ok else "$\\times$", fs=F_TOK,
        color=DARK_GREEN if ok else DARK_RED, group="D",
        name=f"ver-mark-{i+1}")
    bw = SQ_W * 0.45
    sq(tx + (SQ_W - bw) / 2, BAR_Y, bw, PROBS[i] * (BAR_MAX - BAR_Y),
       GREEN if ok else RED, ec=INK, lw=0.6, name=f"ver-bar-{i+1}", group="D")

d_tok = [X[3] + B_PAD + i * (SQ_W + GAP_TOK) for i in range(N_TOK)]
txt((d_tok[0] + d_tok[2] + SQ_W) / 2, NOTE_Y, "accepted", fs=F_NOTE,
    color=DARK_GREEN, group="D", name="d-accepted")
txt(d_tok[3] + SQ_W / 2, NOTE_Y, "rejected", fs=F_NOTE, color=DARK_RED,
    group="D", name="d-rejected")

arr(X[3] + D_W + 0.02, X[4] + (E_W - SQ_W) / 2 - 0.01, LINK_Y,
    name="arr-D-E", group="LDE")

# --------------------------------------------------------------- draw: E
E_SQX = X[4] + (E_W - SQ_W) / 2
sq(E_SQX, SQ_Y0, SQ_W, SQ_H, GREEN_F, ec=GREEN, lw=1.3, name="resample-tok",
   group="E")
txt(E_SQX + SQ_W / 2, LAB_Y, "$\\tilde{t}$", fs=9, group="E", name="resample-lab")
txt(X[4] + E_W / 2, NOTE_Y, "resampled", fs=F_NOTE, color=DARK_GREEN,
    group="E", name="e-note")

# ------------------------------------------------------- phases (top row)
def center_clamped(cx, s, fs):
    """Centre a label on cx but keep it inside the canvas."""
    w = tw(s, fs)
    return min(max(cx, 0.05 + w / 2), FIG_W - 0.05 - w / 2)


for cx, s, nm in [
    (X[1] + B_W / 2, "1  Draft", "phase-1"),
    ((X[2] + X[3] + D_W) / 2, "2  Verify", "phase-2"),
    ((X[3] + X[4] + E_W) / 2, "3  Accept / resample", "phase-3"),
]:
    txt(center_clamped(cx, s, F_PHASE), PHASE_Y, s, fs=F_PHASE, color=MUTED,
        name=nm, group="P")

# ------------------------------------------------------ bottom repeat loop
loopback(X[4] + E_W / 2, X[0] + A_W / 2, LOOP_Y)
txt((X[0] + A_W / 2 + X[4] + E_W / 2) / 2, REPEAT_Y, "repeat", fs=F_TINY,
    color=MUTED, name="repeat-lab", group="R", mask=True)


# ------------------------------------------------------------- verification
def verify():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wpx, Hpx = fig.canvas.get_width_height()
    errors, notes = [], []

    for p in PARTS:
        b = p["artist"].get_window_extent(r)
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > Wpx + 0.5 or b.y1 > Hpx + 0.5:
            errors.append(f"OUT OF CANVAS  {p['name']:<20} "
                          f"x[{b.x0:.0f},{b.x1:.0f}] y[{b.y0:.0f},{b.y1:.0f}] "
                          f"canvas {Wpx}x{Hpx}")

    for i in range(len(PARTS)):
        for j in range(i + 1, len(PARTS)):
            a, b = PARTS[i], PARTS[j]
            ba, bb = a["artist"].get_window_extent(r), b["artist"].get_window_extent(r)
            if not ba.overlaps(bb):
                continue
            ov_w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            ov_h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if ov_w < 1.5 or ov_h < 1.5:
                continue
            msg = f"OVERLAP {ov_w:5.1f}x{ov_h:5.1f}px  {a['name']} <-> {b['name']}"
            (notes if a["group"] == b["group"] else errors).append(msg)

    print("=" * 74)
    print(f"authored {FIG_W}x{FIG_H} in ({FIG_W/FIG_H:.2f}:1), {Wpx}x{Hpx}px, "
          f"{len(PARTS)} artists")
    print(f"zone widths: " + "  ".join(f"{n}={w:.2f}in"
          for n, w in zip("ABCDE", ZONE_W)))
    print(f"packed gap: {gap:.3f} in, right edge {X_END:.3f} in")
    print("-" * 74)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print("  ", e)
    else:
        print("ERRORS: none -- everything inside the canvas, no cross-group overlap")
    if notes:
        print(f"intentional nesting ({len(notes)}):")
        for n in notes:
            print("  ", n)
    print("-" * 74)
    scale = TEXTWIDTH_IN / FIG_W
    print(f"placed with width=\\textwidth ({TEXTWIDTH_IN} in) -> scale {scale:.3f}")
    for fs in sorted({F_TITLE, F_SUB, F_TOK, F_NOTE, F_PHASE, F_TINY}, reverse=True):
        eff = fs * scale
        flag = "" if eff >= 7.0 else "   <-- BELOW 7pt, reconsider"
        print(f"    authored {fs:>4.1f}pt -> printed {eff:>4.1f}pt{flag}")
    print("=" * 74)
    return not errors


def ink_balance(path, bands=10):
    from matplotlib.image import imread

    img = imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    ink = (img < 0.92).any(axis=-1)
    h, w = ink.shape
    cells = [ink[:, k * w // bands:(k + 1) * w // bands].mean() for k in range(bands)]
    print(f"ink balance, {bands} bands of {w//bands}px:")
    print("  [" + "".join(" .:-=+*#%@"[min(9, int(v * 30))] for v in cells) + "]")
    print("  " + "  ".join(f"{v*100:4.1f}%" for v in cells))
    empty = [k for k, v in enumerate(cells) if v < 0.004]
    print("  WARNING near-empty bands: " + str(empty) if empty
          else "  no empty band -- content spans the full width")
    return cells


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    pdf = os.path.join(here, "speculative_decoding.pdf")
    png = os.path.join(here, "speculative_decoding.png")

    ok = verify()
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    ink_balance(png)
    print("RESULT:", "CLEAN" if ok else "LAYOUT ERRORS PRESENT")
