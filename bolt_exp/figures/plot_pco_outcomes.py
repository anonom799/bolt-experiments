"""PCO views that suit a discrete search space better than mean log regret.

PCO's candidate set is a small table, so a trial's simple regret only takes a
handful of values (the optimum, the runner-up, a plateau, ...). Mean log regret
then mostly counts how many trials hit regret 0 (clipped to 1e-8), and methods
with the same count draw the same curve. Two plots that show that directly:

  {out}_success.pdf   fraction of trials whose recommendation is a top-k
                      configuration, vs number of observations (default k=1,
                      i.e. has found the optimum).
  {out}_success_heatmap.pdf
                      the same success rate as one row per method (colour =
                      share of trials), so methods on the same level do not
                      overlap as they do in the curves.
  {out}_rank_heatmap.pdf
                      mean rank of each method against the others (ranked
                      within every seed and step, ties averaged, 1 = best),
                      the same quantity as plot_results.py --mean-rank.
  {out}_outcomes.pdf  stacked bar per method of where each trial ends up: the
                      optimum, the 2nd best configuration, or else by the
                      throughput gap of its final recommendation to the
                      optimum (<=2.5%, 2.5-10%, >10%). Its legend is saved
                      separately as {out}_outcomes_legend.pdf.

Method labels, colours and linestyles come from the same --config YAML as
plot_results.py, so the curves match the existing legend.

Usage (from the repository root):
    python -m bolt_exp.figures.plot_pco_outcomes results/pco32/pco32_*_10trials_50iterations*.json \
        --data data/pco32/data.parquet --config plot_configs/pco32_okabe_ito.yaml \
        --out pics/pco/pco32
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from bolt_exp.plot_results import _method_label

LINESTYLES = {
    "solid": (0, ()),
    "dashed": (0, (5, 2)),
    "dotted": (0, (1, 2)),
    "dashdot": (0, (5, 2, 1, 2)),
}

# Final-outcome tiers. The top two configurations by name, then the rest by
# throughput gap to the optimum, so the tiers mean the same loss on every
# instance (table ranks do not: rank 3 is 8% below the optimum on pco32 but 2%
# on pco64, where ranks 3-4 are the dp16/tp1/pp4 plateau). The 2.5% cut keeps
# that plateau (2.08% below) in one tier with its near neighbours (up to 2.43%);
# 2% would fall just short of it. Each tier is
# (min rank, max rank, min gap, max gap, label); gaps are fractions.
TIERS = [
    (1, 1, 0.0, np.inf, "Optimum"),
    (2, 2, 0.0, np.inf, "2nd best"),
    (3, np.inf, 0.0, 0.025, "Other, ≤2.5% below"),
    (3, np.inf, 0.025, 0.10, "2.5–10% below"),
    (3, np.inf, 0.10, np.inf, ">10% below"),
]
# Colours are seaborn's perceptually uniform "mako" map throughout: dark = good
# (optimum found, rank 1), light = bad. The outcome tiers take evenly spaced
# steps of it, and alternate tiers add a hatch so adjacent segments also differ
# in texture (grayscale print, colour-vision deficiency).
CMAP = "mako"
TIER_HATCHES = [None, "////", None, "\\\\", None]
# Opaque: PDF viewers render a translucent hatch pattern inconsistently.
HATCH_INK = "white"

# Regret within this of a table value is treated as that value, so float noise
# in the stored regrets does not shift a rank.
ATOL = 1e-9


def load_config(path):
    cfg = yaml.safe_load(open(path)) or {}
    entries = cfg.get("methods", []) if isinstance(cfg, dict) else cfg
    return cfg if isinstance(cfg, dict) else {}, entries


def load_runs(files, table):
    """{label: (n_trials, n_steps) array of the rank of each step's recommendation}."""
    runs = {}
    for f in files:
        d = json.load(open(f))
        opt = d["optimal_value"]
        trials = sorted(d["trials"], key=lambda t: t["seed"])
        regrets = np.array([t["simple_regret_all"] for t in trials])
        values = opt - regrets
        # Rank = 1 + number of feasible table rows strictly better than the
        # recommendation; equal values share the better rank.
        ranks = 1 + np.searchsorted(-table, -(values + ATOL), side="left")
        runs[_method_label(d)] = ranks
    return runs


def plot_success(runs, order, colors, styles, k, figsize, out):
    fig, ax = plt.subplots(figsize=figsize)
    for m in order:
        r = runs[m]
        frac = (r <= k).mean(axis=0)
        ax.step(np.arange(len(frac)), frac, where="post", color=colors[m],
                linestyle=styles[m], linewidth=2, label=m)
    ax.set_xlim(0, max(r.shape[1] for r in runs.values()) - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Found optimum" if k == 1 else f"Found top-{k}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(True, axis="y", color="#e6e5e1", linewidth=0.8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


def _strip_heatmap(grid, rows, final_labels, cmap, vmin, vmax, cbar_label, cbar_ticks,
                   figsize, out, invert_cbar=False, cbar_fmt=None):
    """One row per method over observations, with the final value printed at the end."""
    n_steps = grid.shape[1]
    fig, ax = plt.subplots(figsize=(figsize[0] * 1.15, 0.32 * len(rows) + 0.9))
    sns.heatmap(
        pd.DataFrame(grid, index=rows), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        xticklabels=False, yticklabels=True,
        # Rasterized mesh: vector cells leave hairline seams between them in PDF.
        rasterized=True,
        cbar_kws={"label": cbar_label, "ticks": cbar_ticks, "format": cbar_fmt,
                  "fraction": 0.04, "pad": 0.09},
    )
    ticks = np.arange(0, n_steps, 10)
    ax.set_xticks(ticks + 0.5, ticks, rotation=0)
    ax.set_xlabel("Number of observations")
    ax.tick_params(axis="y", length=0)
    # White gaps between rows, so each method reads as its own strip.
    for r in range(1, len(rows)):
        ax.axhline(r, color="white", linewidth=2)
    # Final value as a direct label, so the exact level never rests on colour.
    for i, label in enumerate(final_labels):
        ax.text(n_steps + n_steps * 0.015, i + 0.5, label,
                va="center", ha="left", fontsize="small", color="#52514e")
    cb = ax.collections[0].colorbar
    cb.outline.set_visible(False)
    if invert_cbar:
        cb.ax.invert_yaxis()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


def plot_success_heatmap(runs, order, k, figsize, out):
    frac = {m: (runs[m] <= k).mean(axis=0) for m in order}
    # Best first: highest final success, then earliest (area under the curve).
    rows = sorted(order, key=lambda m: (frac[m][-1], frac[m].mean()), reverse=True)
    n_steps = max(len(f) for f in frac.values())
    grid = np.full((len(rows), n_steps), np.nan)
    for i, m in enumerate(rows):
        grid[i, : len(frac[m])] = frac[m]
    # One colour per attainable level: with n trials the rate moves in 1/n
    # steps, and n+1 evenly split bins over [0, 1] put each level in its own.
    n_trials = runs[rows[0]].shape[0]
    _strip_heatmap(
        grid, rows, [f"{frac[m][-1]:.0%}" for m in rows],
        cmap=sns.color_palette(CMAP + "_r", n_trials + 1), vmin=0, vmax=1,
        cbar_label="Found optimum" if k == 1 else f"Found top-{k}",
        cbar_ticks=[0, 0.5, 1], cbar_fmt=matplotlib.ticker.PercentFormatter(1.0, decimals=0),
        figsize=figsize, out=out,
    )


def plot_rank_heatmap(runs, order, figsize, out):
    # (methods, seeds, steps) of table ranks; ranking those per (seed, step) is
    # the same as ranking the regrets, since the table rank is monotone in it.
    n_steps = min(runs[m].shape[1] for m in order)
    stack = np.stack([runs[m][:, :n_steps] for m in order])
    ranks = pd.DataFrame(stack.reshape(len(order), -1).T).rank(axis=1, method="average")
    mean_rank = ranks.values.T.reshape(stack.shape).mean(axis=1)
    mean = dict(zip(order, mean_rank))
    rows = sorted(order, key=lambda m: (mean[m][-1], mean[m].mean()))
    _strip_heatmap(
        np.array([mean[m] for m in rows]), rows, [f"{mean[m][-1]:.1f}" for m in rows],
        cmap=CMAP, vmin=1, vmax=len(order), cbar_label="Mean rank (1 = best)",
        cbar_ticks=[1, (1 + len(order)) / 2, len(order)], figsize=figsize, out=out,
        invert_cbar=True,
    )


def plot_outcomes(runs, order, table, figsize, out):
    final = {m: runs[m][:, -1] for m in order}
    # A rank points at its table row, so the gap is read off the table.
    gap = {m: 1 - table[final[m] - 1] / table[0] for m in order}
    shares = pd.DataFrame(
        {name: [np.mean((final[m] >= r0) & (final[m] <= r1) & (gap[m] >= g0) & (gap[m] < g1))
                for m in order]
         for r0, r1, g0, g1, name in TIERS},
        index=order,
    )
    # Best first: most optima, then most runner-ups, ...
    shares = shares.sort_values([t[-1] for t in TIERS], ascending=False)

    plt.rcParams["hatch.linewidth"] = 1.0
    fig, ax = plt.subplots(figsize=(figsize[0], 0.29 * len(shares) + 0.9))
    y = np.arange(len(shares))[::-1]
    left = np.zeros(len(shares))
    handles = []
    colors = sns.color_palette(CMAP, len(TIERS))
    for (*_, name), color, hatch in zip(TIERS, colors, TIER_HATCHES):
        w = shares[name].values
        # White edges give the surface gap between stacked segments.
        ax.barh(y, w, left=left, height=0.8, color=color, edgecolor="white", linewidth=1.5)
        if hatch:
            # Hatch drawn as a second, edgeless layer so its ink colour does not
            # replace the white gap.
            ax.barh(y, w, left=left, height=0.8, facecolor="none", edgecolor=HATCH_INK,
                    hatch=hatch, linewidth=0)
        handles.append(matplotlib.patches.Patch(facecolor=color, edgecolor=HATCH_INK,
                                                hatch=hatch, linewidth=0, label=name))
        left += w
    ax.set_yticks(y, shares.index)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    # Hug the bars: the default 5% y margin leaves an empty band above the top row.
    ax.set_ylim(-0.5, len(shares) - 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of trials (final recommendation)")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")

    # Legend in its own file, one row, so it can be placed freely in the paper.
    legend_fig = plt.figure(figsize=(0.1, 0.1))
    legend_fig.legend(handles=handles, ncol=len(TIERS), loc="center", frameon=False,
                      fontsize="small", handlelength=1.0, columnspacing=1.0)
    legend_out = Path(out).with_name(Path(out).stem + "_legend" + Path(out).suffix)
    legend_fig.savefig(legend_out, bbox_inches="tight", pad_inches=0.05)
    plt.close(legend_fig)
    print(f"Saved legend to {legend_out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="PCO result JSON files")
    p.add_argument("--data", required=True, type=Path, help="Instance data.parquet (candidate table)")
    p.add_argument("--config", required=True, help="plot_results.py style YAML (labels, colours, linestyles)")
    p.add_argument("--out", required=True, help="Output prefix; writes {out}_success.pdf and {out}_outcomes.pdf")
    p.add_argument("--top-k", type=int, default=1, help="Success = recommendation ranks <= k (default 1)")
    args = p.parse_args()

    d = pd.read_parquet(args.data)
    # Feasible = finite throughput (an OOMed run records none); the current
    # tables have no `feasible` column.
    table = np.sort(d.loc[np.isfinite(d.throughput_mean), "throughput_mean"].values)[::-1]

    cfg, entries = load_config(args.config)
    runs = load_runs(args.files, table)
    order = [e["label"] for e in entries if e["label"] in runs]
    if not cfg.get("skip_unlisted"):
        order += [m for m in runs if m not in order]
    colors = {e["label"]: e.get("color") for e in entries}
    styles = {e["label"]: LINESTYLES.get(e.get("linestyle", "solid"), e.get("linestyle")) for e in entries}
    figsize = cfg.get("figsize", [5, 2.8])

    sns.set_theme(style="white", context="paper", font_scale=1.8)
    plot_success(runs, order, colors, styles, args.top_k, figsize, f"{args.out}_success.pdf")
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plot_success_heatmap(runs, order, args.top_k, figsize, f"{args.out}_success_heatmap.pdf")
    plot_rank_heatmap(runs, order, figsize, f"{args.out}_rank_heatmap.pdf")
    plot_outcomes(runs, order, table, figsize, f"{args.out}_outcomes.pdf")


if __name__ == "__main__":
    main()
