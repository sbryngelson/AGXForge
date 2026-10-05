#!/usr/bin/env python3
"""The showcase's figures (docs/figures/showcase-*.svg), drawn from the committed evidence files.

The chart reads its numbers from evidence/ directly, so a re-measured file moves the figure with it; the two
diagrams encode the execution paths and one tensor-register example from the technical reference (section 10.5,
table "Three slots in canonical coordinates"). Every figure has a white background so it reads on GitHub's dark
theme as well as its light one.

    python3 tools/g17showcasefigs.py            write all three figures
    python3 tools/g17showcasefigs.py --check    exit 1 if a committed figure differs from what would be written
"""
import io, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
DECODE = os.path.join(ROOT, "evidence", "g17-matched-study-v1", "decode-shared-history.json")

AGXFORGE = "#2f6db5"       # this project's parts
AGXFORGE_LIGHT = "#dbe7f6"
APPLE = "#8a8f98"       # Apple's parts and baselines
APPLE_LIGHT = "#eceef1"
INK = "#1f2328"
MUTED = "#57606a"
HIGHLIGHT = "#d4761c"

plt.rcParams.update({
    "svg.fonttype": "none",          # text stays text: smaller files, selectable, searchable
    "svg.hashsalt": "agxforge",         # deterministic ids, so --check compares bytes
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "text.color": INK,
})


def save(fig, name):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.15,
                metadata={"Date": None, "Creator": "tools/g17showcasefigs.py"})
    plt.close(fig)
    return name, buf.getvalue()


def box(ax, x, y, w, h, text, owner, size=9.5, bold=False):
    face, edge = (AGXFORGE_LIGHT, AGXFORGE) if owner == "agxforge" else (APPLE_LIGHT, APPLE)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=0, shrinkB=0))


def paths():
    """Two routes for the same native code: through Metal (performance results), below Metal (native applications)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(-0.1, 10.3)
    ax.set_ylim(-0.05, 4.45)
    ax.axis("off")
    # inputs
    box(ax, 0.0, 3.35, 2.0, 0.75, "Python model-kernel\nbuilders", "agxforge", 9)
    box(ax, 0.0, 2.3, 2.0, 0.75, "Metal source\n→ xcrun metal → AIR", "apple", 9)
    # compiler and code
    box(ax, 2.6, 2.75, 1.9, 1.0, "AGXForge compiler\nselection, allocation,\nencoder, image", "agxforge", 9)
    arrow(ax, 2.0, 3.72, 2.6, 3.4)
    arrow(ax, 2.0, 2.68, 2.6, 3.05)
    box(ax, 5.05, 2.85, 1.55, 0.8, "native G17\nmachine code", "agxforge", 9.5, bold=True)
    arrow(ax, 4.5, 3.25, 5.05, 3.25)
    # the two routes; the Metal route's arrow runs down the right edge, clear of the runtime box
    box(ax, 7.2, 3.55, 3.0, 0.85, "Metal: pipelines, command buffers\n(every performance result)", "apple", 9)
    box(ax, 7.2, 2.05, 2.45, 0.85, "AGXForge runtime: resources,\nlaunch state, Submit records\n(MiniLM, Qwen; no Metal)", "agxforge", 8.8)
    arrow(ax, 6.6, 3.4, 7.2, 3.9)
    arrow(ax, 6.6, 3.1, 7.2, 2.5)
    # shared Apple layers and the device
    box(ax, 2.6, 0.75, 7.6, 0.6, "Apple IOGPU framework, kernel driver and firmware", "apple", 9.5)
    box(ax, 2.6, 0.0, 7.6, 0.5, "M5 GPU cores and their tensor units", "apple", 9.5, bold=True)
    arrow(ax, 9.95, 3.55, 9.95, 1.35)
    arrow(ax, 8.42, 2.05, 8.42, 1.35)
    arrow(ax, 6.4, 0.75, 6.4, 0.5)
    # legend
    for i, (owner, label) in enumerate([("agxforge", "this project"), ("apple", "Apple")]):
        face, edge = (AGXFORGE_LIGHT, AGXFORGE) if owner == "agxforge" else (APPLE_LIGHT, APPLE)
        ax.add_patch(FancyBboxPatch((0.05, 1.05 - 0.45 * i), 0.3, 0.25, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(0.45, 1.175 - 0.45 * i, label, va="center", fontsize=9)
    return save(fig, "showcase-paths.svg")


def decode():
    """Three implementations decoding one 128-token history: median and the four repetitions."""
    d = json.load(open(DECODE))["tok_s"]
    arms = [("mlx", "mlx-lm generate_step", APPLE),
            ("graph_forced", "AGXForge kernels, AGXForge compiler", AGXFORGE),
            ("graph_twin_all_forced", "same kernels, Apple compiler", "#6f9fd8")]
    base = d["mlx"]["median"]
    fig, ax = plt.subplots(figsize=(7.6, 2.3))
    for i, (key, label, color) in enumerate(arms):
        y = len(arms) - 1 - i
        med = d[key]["median"]
        ax.barh(y, med, height=0.62, color=color)
        ax.scatter(d[key]["per_rep"], [y] * len(d[key]["per_rep"]), s=10, color="white", edgecolors=INK,
                   linewidths=0.6, zorder=3)
        ax.text(max(d[key]["per_rep"]) + 4, y, "%.1f tok/s  ·  %.2fx" % (med, med / base), va="center", fontsize=9.5,
                fontweight="bold" if key != "mlx" else "normal")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[1] for a in reversed(arms)])
    ax.set_xlim(0, 270)
    ax.set_xlabel("median tokens per second (dots: the four repetitions)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("InternLM2.5-1.8B, 4-bit weights, 1,792-token context, one shared 128-token history, through Metal",
                 fontsize=9, color=MUTED, loc="left")
    return save(fig, "showcase-decode.svg")


def packing():
    """Lane 0's eight accumulator slots, read by the next MMA as B: which D row each packing puts in slot 4."""
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.set_xlim(-3.3, 8.2)
    ax.set_ylim(0.35, 4.3)
    ax.axis("off")
    rows = [
        ("hardware: canonical D element", ["(%d,%d)" % (j >> 2, j & 3) for j in range(8)], None),
        ("next MMA reads it as B element", ["(%d,%d)" % (8 * (j >> 2), j & 3) for j in range(8)], None),
        ("canonical packing puts D row", [str(j >> 2) for j in range(8)], "apple"),
        ("AGXForge's packing puts D row", [str(8 * (j >> 2)) for j in range(8)], "agxforge"),
    ]
    for j in range(8):
        ax.text(j + 0.5, 4.05, "slot %d" % j, ha="center", fontsize=8.5,
                color=HIGHLIGHT if j == 4 else MUTED, fontweight="bold" if j == 4 else "normal")
    for r, (label, cells, owner) in enumerate(rows):
        y = 3.0 - r * 0.85
        ax.text(-0.15, y + 0.33, label, ha="right", va="center", fontsize=9)
        for j, c in enumerate(cells):
            face = AGXFORGE_LIGHT if owner == "agxforge" else APPLE_LIGHT if owner == "apple" else "white"
            ax.add_patch(plt.Rectangle((j + 0.04, y), 0.92, 0.66, facecolor=face,
                                       edgecolor=HIGHLIGHT if j == 4 else "#c9ced6",
                                       linewidth=1.8 if j == 4 else 0.8))
            ax.text(j + 0.5, y + 0.33, c, ha="center", va="center", fontsize=9,
                    fontweight="bold" if j == 4 else "normal")
    ax.text(8.15, 3.0 - 2 * 0.85 + 0.33, "B row 8 gets D row 1:\nrotated, must be undone", ha="left", va="center",
            fontsize=8.5, color=MUTED)
    ax.text(8.15, 3.0 - 3 * 0.85 + 0.33, "B row 8 gets D row 8:\nused directly", ha="left", va="center",
            fontsize=8.5, color=AGXFORGE, fontweight="bold")
    ax.set_title("Lane 0 of a 16 x 16 accumulator tile, fed unchanged to the next MMA as its B operand",
                 fontsize=9, color=MUTED, loc="left")
    return save(fig, "showcase-packing.svg")


FIGURES = [paths, decode, packing]


def main(argv):
    check = "--check" in argv
    stale = []
    os.makedirs(OUT, exist_ok=True)
    for make in FIGURES:
        name, svg = make()
        path = os.path.join(OUT, name)
        if check:
            if not os.path.exists(path) or open(path).read() != svg:
                stale.append(name)
        else:
            with open(path, "w") as f:
                f.write(svg)
            print("wrote", os.path.relpath(path, ROOT))
    if stale:
        print("stale figures (run tools/g17showcasefigs.py):", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
