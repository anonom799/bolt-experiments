"""Show why PCO64 does not separate methods: its table has a flat top.

Two panels over the feasible rows of each instance, both on a "% of that
instance's own optimum" axis so the three tables are comparable despite their
different throughput scales:

  left   the top of the table by rank -- how fast throughput falls off as you
         step away from the best configuration.
  right  how many configurations sit within a given % of the optimum -- the
         same fact as a count, which is what a search actually runs into.

Usage:
    python -m bolt_exp.figures.plot_pco_plateau --out pics/pco/pco_plateau.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bolt_exp import REPO_ROOT

# The PCO candidate tables (the same data the `bolt` problems download), kept
# in the repo so the figure builds offline.
DATA_ROOT = REPO_ROOT / "data"

# Categorical slots 1-3 of the validated reference palette, keyed by instance so
# pco32/pco64 keep the colours they had before pco16 was added.
PALETTE = {"pco32": "#2a78d6", "pco64": "#eb6834", "pco16": "#1baf7a"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

# Value most methods stall at on pco64 (every cts trial): the dp16/tp1/pp4 plateau.
# Annotated because it is the answer the search settles for (ckg stops here in
# about half its trials; the rest reach the optimum or stall lower).
PLATEAU = {"pco64": 0.265064}

TOP_N = 60
PCT_MAX = 12.0


def load(name):
    d = pd.read_parquet(DATA_ROOT / name / "data.parquet")
    # Feasible = finite throughput (an OOMed run records none); the current
    # tables have no `feasible` column.
    y = np.sort(d.loc[np.isfinite(d.throughput_mean), "throughput_mean"].values)[::-1]
    return y


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("pics/pco/pco_plateau.pdf"))
    args = p.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 2.9), facecolor=SURFACE)

    for name in ("pco16", "pco32", "pco64"):
        color = PALETTE[name]
        y = load(name)
        opt = y[0]
        pct = 100.0 * y / opt

        # left: rank profile
        ax1.plot(
            np.arange(1, TOP_N + 1), pct[:TOP_N],
            color=color, linewidth=2.0, label=f"{name}  ({len(y)} feasible rows)",
        )

        # right: how many rows within x% of the optimum
        xs = np.linspace(0, PCT_MAX, 400)
        counts = [(y >= (1 - x / 100) * opt).sum() for x in xs]
        ax2.plot(xs, counts, color=color, linewidth=2.0, label=name)

        if name in PLATEAU:
            v = 100.0 * PLATEAU[name] / opt
            ax1.axhline(v, color=color, linewidth=1.0, linestyle=(0, (2, 2)))
            ax1.annotate(
                "plateau most\nmethods stall at",
                xy=(TOP_N * 0.80, v), xytext=(TOP_N * 0.55, 79.0),
                color=MUTED, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
            )

    ax1.set_xlabel("Configuration rank (feasible rows, best first)", color=TEXT, fontsize=10)
    ax1.set_ylabel("Throughput (% of optimum)", color=TEXT, fontsize=10)
    ax1.set_xlim(1, TOP_N)
    ax1.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left")

    ax2.set_xlabel("Within x% of the optimum", color=TEXT, fontsize=10)
    ax2.set_ylabel("Feasible configurations", color=TEXT, fontsize=10)
    ax2.set_xlim(0, PCT_MAX)
    ax2.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")

    for ax in (ax1, ax2):
        style(ax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
