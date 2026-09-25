"""Compare LLM-in-the-loop real-task results with their emulator counterparts.

Works for any problem family (DM, HPO, ...) that logs `best_y_all` per
trial; the family-specific bits (result directories, matched method ->
filename mapping, optimisation direction, source labels) come from a
"methods config" YAML rather than being hardcoded here. See
plot_configs/dm_real_vs_emulated.yaml and plot_configs/hpo_real_vs_emulated.yaml
for examples.

The sources use different run lengths, so all are truncated to the initial
N_ITERATIONS BO iterations (from the methods config). Curves are normalised
against the best/worst `best_y_all` value found by any method within that
same window (matching YAHPO Gym's normalized-regret convention), not the
raw noisy observation pool.

Usage:
    python -m bolt_exp.figures.plot_real_vs_emulated \
        --methods-config plot_configs/dm_real_vs_emulated.yaml \
        --style-config plot_configs/dm_okabe_ito.yaml \
        --out pics/dm_real_vs_emulated_normalized_regret.pdf
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr

from bolt_exp.plot_results import mean_rank_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods-config", type=Path, required=True,
                         help="YAML describing sources, matched methods, n_iterations, "
                              "and optimisation direction (see plot_configs/*_real_vs_emulated.yaml).")
    parser.add_argument("--style-config", type=Path, required=True,
                         help="YAML with figsize/method colors, as used by plot_results.py.")
    parser.add_argument("--out", type=Path, required=True,
                         help="Base output path; suffixes are appended per source/panel type.")
    parser.add_argument("--shared-scale", action="store_true",
                         help="Normalize real and emulator runs against one pooled bound "
                              "instead of a bound per source (per-source is the convention "
                              "used by YAHPO Gym's real-vs-surrogate comparisons).")
    parser.add_argument("--shared-axis", action="store_true",
                         help="Share the y-axis range across source panels for the regret "
                              "plots, independent of how values were normalized. Only the "
                              "'real' panel keeps its y-tick labels; other panels share the "
                              "same scale without drawing their own ticks.")
    parser.add_argument("--combined", action="store_true",
                         help="Also save one combined grid figure (rows = regret / mean "
                              "rank, columns = sources) with the legend inside it, so the "
                              "document needs a single \\includegraphics and one caption "
                              "instead of a subfigure per panel.")
    parser.add_argument("--no-title", action="store_true",
                         help="Omit panel/figure titles (for figures captioned by the "
                              "document they are embedded in). Per-seed subplot labels "
                              "are kept, since they identify the subplot.")
    return parser.parse_args()


def _load(path: Path, method: str, source: str, n_iterations: int) -> list[dict]:
    with path.open() as f:
        result = json.load(f)

    rows = []
    for i, trial in enumerate(result["trials"]):
        seed = trial.get("seed", i)
        values = trial["best_y_all"][: n_iterations + 1]
        if len(values) != n_iterations + 1:
            raise ValueError(f"{path} has fewer than {n_iterations} BO iterations")
        rows.extend(
            {
                "method": f"{method} ({source})",
                "seed": seed,
                "step": step,
                "value": value,
            }
            for step, value in enumerate(values)
        )
    return rows


def _regret_ylim(data: pd.DataFrame, aggregated: bool) -> tuple[float, float]:
    """Shared y limits covering exactly what the regret panels draw.

    `aggregated` panels draw a per-(source, method, step) mean with a 95% CI
    band, so the extremes of the raw per-seed values never appear and would
    only squash the curves; bound those panels by mean +/- 1.96 * sem (a close
    stand-in for seaborn's bootstrap CI). Per-seed panels draw the raw values,
    so bound those by the values themselves. Either way pad by 5% of the range
    rather than a fixed amount, so the padding scales with the data.
    """
    if aggregated:
        grouped = data.groupby(["source", "method_base", "step"])["value"]
        mean, sem = grouped.mean(), grouped.sem().fillna(0.0)
        lo, hi = (mean - 1.96 * sem).min(), (mean + 1.96 * sem).max()
    else:
        lo, hi = data["value"].min(), data["value"].max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    return lo - pad, hi + pad


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved to {out_path}")


def _hide_y_labels(ax) -> None:
    """Blank the y tick labels and ylabel without freeing their margin.

    Setting `labelleft=False` would let tight_layout reclaim the space the
    labels occupied and widen the axes, so a panel with hidden labels would
    end up with a different axes box than the panel that keeps them — the
    same y-range drawn at a different aspect ratio reads as a flatter curve.
    Colouring the text "none" keeps its extent, so every panel in a
    shared-scale set gets an identically sized axes box.
    """
    for label in ax.get_yticklabels():
        label.set_color("none")
    ax.yaxis.label.set_color("none")


def load_regret_frame(methods_config: dict, shared_scale: bool = False) -> pd.DataFrame:
    """Long-form (method_base, source, seed, step, value) normalised regret.

    Shared with plot_spearman_grid.py, so a cross-family figure is built from
    exactly the numbers the per-family figures plot.
    """
    n_iterations = methods_config["n_iterations"]
    sources = tuple(methods_config["sources"].keys())
    source_dirs = {name: Path(path) for name, path in methods_config["sources"].items()}

    rows = []
    for entry in methods_config["methods"]:
        method = entry["label"]
        for source in sources:
            filename = entry["files"][source]
            rows.extend(_load(source_dirs[source] / filename, method, source, n_iterations))

    df = pd.DataFrame(rows)
    source_pattern = "|".join(sources)
    df["source"] = df["method"].str.extract(rf"\(({source_pattern})\)$", expand=False)
    df["method_base"] = df["method"].str.replace(rf" \(({source_pattern})\)$", "", regex=True)

    # Real runs only cover a subset of seeds; restrict every source to that
    # subset so real vs. emulated comparisons are seed-matched throughout.
    real_seeds = set(df.loc[df["source"] == "real", "seed"])
    dropped = sorted(set(df["seed"]) - real_seeds)
    if dropped:
        print(f"Restricting to real's seeds {sorted(real_seeds)}; dropping seeds {dropped} "
              f"from emulated sources.")
    df = df[df["seed"].isin(real_seeds)].reset_index(drop=True)

    # normalized_regret_all convention (see plot_results.py:normalize_regret):
    # 0 means best value found by any method, 1 means worst; lower is better.
    # Bounds come from best_y_all itself (the range of values as found by any
    # method, matching YAHPO Gym's convention), not the raw observation pool.
    if shared_scale:
        y_max, y_min = df["value"].max(), df["value"].min()
        df["value"] = (y_max - df["value"]) / (y_max - y_min)
        print(f"Normalisation bounds through iteration {n_iterations} (shared across sources): "
              f"y_max={y_max:.6f}, y_min={y_min:.6f}")
    else:
        bounds = df.groupby("source")["value"].agg(y_max="max", y_min="min")
        df = df.merge(bounds, left_on="source", right_index=True)
        df["value"] = (df["y_max"] - df["value"]) / (df["y_max"] - df["y_min"])
        df = df.drop(columns=["y_max", "y_min"])
        for source, row in bounds.iterrows():
            print(f"Normalisation bounds through iteration {n_iterations} ({source}): "
                  f"y_max={row['y_max']:.6f}, y_min={row['y_min']:.6f}")
    return df


def rank_frame(df: pd.DataFrame, sources) -> pd.DataFrame:
    """Per-(step, seed) method ranks, ranked within each source separately."""
    frames = []
    for source in sources:
        sub = df[df["source"] == source].copy()
        sub["method"] = sub["method_base"]
        ranked = mean_rank_frame(sub, "normalized_regret_all")
        ranked["source"] = source
        ranked["method_base"] = ranked["method"]
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def mean_rank_matrix(rank_data: pd.DataFrame, source: str, steps, methods) -> np.ndarray:
    """(step, method) mean rank over seeds for one source."""
    sub = rank_data[rank_data["source"] == source]
    wide = (sub.groupby(["step", "method_base"])["value"].mean()
               .unstack("method_base")
               .reindex(index=steps, columns=list(methods)))
    return wide.to_numpy()


def rho_per_step(real_mean: np.ndarray, other_mean: np.ndarray) -> np.ndarray:
    """Spearman rho across methods, per step, for two (step, method) arrays.

    Spearman is Pearson on the ranks, so ranking each row and correlating
    gives the same number as calling spearmanr per step, vectorised over
    steps. A step whose ranks are constant on either side has no defined
    correlation and comes back as NaN rather than raising.
    """
    a = rankdata(real_mean, axis=1)
    b = rankdata(other_mean, axis=1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def main() -> None:
    args = parse_args()

    with args.methods_config.open() as f:
        methods_config = yaml.safe_load(f)

    n_iterations = methods_config["n_iterations"]
    title = methods_config.get("title", "Real vs Emulated")
    sources = tuple(methods_config["sources"].keys())
    source_labels = methods_config.get("source_labels", {name: name.title() for name in sources})
    matched_methods = methods_config["methods"]
    out = args.out

    df = load_regret_frame(methods_config, args.shared_scale)

    with args.style_config.open() as f:
        style = yaml.safe_load(f)
    entries = {entry["label"]: entry for entry in style["methods"]}

    methods = [entry["label"] for entry in matched_methods]
    aliases = {"solid": (1, 0), "dashed": (5, 2), "dotted": (1, 2),
               "dashdot": (5, 2, 1, 2)}
    color_map = {method: entries[method]["color"] for method in methods}
    dashes_map = {method: aliases.get(entries[method].get("linestyle", "solid"), (1, 0))
                  for method in methods}

    sns.set_theme(style="white", context="paper", font_scale=1.8)

    def save_legend(out):
        handles = [Line2D([0], [0], color=color_map[m], linewidth=2,
                           linestyle=(0, dashes_map[m])) for m in methods]
        legend_fig = plt.figure(figsize=(style["figsize"][0], 1.0))
        legend_ax = legend_fig.add_axes([0, 0, 1, 1])
        legend_ax.set_axis_off()
        legend_ax.legend(handles, methods, loc="center", ncol=len(methods),
                          framealpha=0.9, handlelength=2.4, edgecolor="0.7", fontsize="small")
        legend_path = out.with_name(out.stem + "_legend" + out.suffix)
        legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight", pad_inches=0.05,
                            facecolor="white")
        plt.close(legend_fig)
        print(f"Saved legend to {legend_path}")

    def plot_panels(data, ylabel, panel_title, out, rank=False):
        # Rank plots always share a y axis across sources; regret plots only
        # do so when --shared-scale is set (which also hides ticks on the
        # non-real panels so they read as sharing the real panel's axis).
        share_axis = rank or args.shared_axis
        shared_ylim = None
        if rank:
            shared_ylim = (data["value"].min() - 0.2, data["value"].max() + 0.2)
        elif share_axis:
            shared_ylim = _regret_ylim(data, aggregated=True)

        for source in sources:
            fig, ax = plt.subplots(figsize=style["figsize"])
            sub = data[data["source"] == source]
            for method in methods:
                method_sub = sub[sub["method_base"] == method]
                sns.lineplot(data=method_sub, x="step", y="value", color=color_map[method],
                             errorbar="ci", linewidth=2, ax=ax, label=method,
                             linestyle=(0, dashes_map[method]))
            if ax.get_legend():
                ax.get_legend().remove()
            if not args.no_title:
                ax.set_title(f"{panel_title} ({source_labels[source]})")
            ax.set_xlabel("BO Iteration")
            ax.set_xlim(0, n_iterations)
            ax.set_ylabel(ylabel)
            if rank:
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            if share_axis:
                ax.set_ylim(shared_ylim)
            hidden = args.shared_axis and source != "real"
            if hidden:
                _hide_y_labels(ax)
            fig.tight_layout(pad=0.4)
            out_path = out.with_name(out.stem + f"_{source}" + out.suffix)
            _save(fig, out_path)

        save_legend(out)

    def plot_grid(regret_data, rank_data, out):
        """One figure: rows = metric, columns = source, single shared legend.

        Column headers and row y-labels carry what the per-panel subcaptions
        used to say, so the document embeds one graphic with one caption.
        """
        n_cols = len(sources)
        panel_w, _panel_h = style["figsize"]
        row_scale = 1.2
        # Row height uses a fixed reference panel_h (HPO's, 3.0) rather than
        # each family's own style figsize height: dm_okabe_ito.yaml's is 4,
        # and deriving row height from that made the DM grid's subplots
        # taller than HPO's even at the same row_scale.
        reference_panel_h = 3.0
        fig, axes = plt.subplots(
            2, n_cols, squeeze=False, sharex=True, sharey="row",
            figsize=(panel_w * 0.45 * n_cols + 0.5, reference_panel_h * 0.62 * row_scale * 2),
        )

        rows = [
            (regret_data, "Norm. Reg.", False),
            (rank_data, "Mean Rank", True),
        ]
        for r, (data, ylabel, rank) in enumerate(rows):
            if rank:
                ylim = (data["value"].min() - 0.2, data["value"].max() + 0.2)
            else:
                ylim = _regret_ylim(data, aggregated=True)
            for c, source in enumerate(sources):
                ax = axes[r][c]
                sub = data[data["source"] == source]
                for method in methods:
                    method_sub = sub[sub["method_base"] == method]
                    sns.lineplot(data=method_sub, x="step", y="value",
                                 color=color_map[method], errorbar="ci", linewidth=2,
                                 ax=ax, label=method, linestyle=(0, dashes_map[method]))
                if ax.get_legend():
                    ax.get_legend().remove()
                ax.set_xlim(0, n_iterations)
                ax.set_ylim(ylim)
                if rank:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.set_xlabel("BO Iteration" if r == len(rows) - 1 else "")
                ax.set_ylabel(ylabel if c == 0 else "")
                # Smaller ticks than the axis labels: panels sit close
                # together here, and full-size tick text would collide across
                # the seam between adjacent columns.
                ax.tick_params(labelsize="small")
                if r == 0:
                    ax.set_title(source_labels[source], fontsize="small")

        handles = [Line2D([0], [0], color=color_map[m], linewidth=2,
                           linestyle=(0, dashes_map[m])) for m in methods]
        fig.legend(handles, methods, loc="upper center", ncol=len(methods),
                   frameon=False, handlelength=1.8, fontsize="x-small",
                   bbox_to_anchor=(0.5, 1.0))
        if not args.no_title:
            fig.suptitle(title, y=1.06)
        # Top strip a bit taller than plain 0.94 to leave more air between the
        # legend and the top row of panels.
        fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.8, rect=(0, 0, 1, 0.92))
        _save(fig, out.with_name(out.stem + "_grid" + out.suffix))

    def plot_spearman(rank_data, out):
        """Rank agreement between real and each emulator, per BO iteration.

        rho correlates the seed-averaged (mean) rank of the methods under
        `real` with their mean rank under the emulator. Seeds are averaged
        first on purpose: a real seed and an emulator seed are independent
        draws of the same distribution, so pairing them by index and
        correlating per-seed rankings would feed both sides' seed noise into
        the statistic and attenuate rho toward 0. The mean ranks are also what
        the mean-rank panels plot, so this is the agreement the figure claims.

        Drawn as a bare curve, with no error band. rho here is a statistic of
        the whole seed sample rather than a per-seed observation, so seaborn's
        usual errorbar="ci" has nothing to resample.
        """
        other_sources = [s for s in sources if s != "real"]
        if not other_sources:
            print("Skipping spearman plot: no non-real source to compare against.")
            return

        steps = np.sort(rank_data["step"].unique())
        real_mean = mean_rank_matrix(rank_data, "real", steps, methods)

        fig, ax = plt.subplots(figsize=style["figsize"])
        palette = sns.color_palette(n_colors=len(other_sources))
        for color, source in zip(palette, other_sources):
            rho = rho_per_step(real_mean,
                               mean_rank_matrix(rank_data, source, steps, methods))
            ax.plot(steps, rho, color=color, linewidth=2, label=source_labels[source])
            print(f"Spearman rho ({source_labels[source]} vs real) at iteration "
                  f"{steps[-1]}: {rho[-1]:.3f}")
        ax.legend(title="Comparison", frameon=True)
        ax.set_xlabel("BO Iteration")
        ax.set_ylabel("Spearman's rho")
        if not args.no_title:
            ax.set_title(f"{title} Method Ranking (Spearman)")
        ax.set_xlim(0, n_iterations)
        fig.tight_layout(pad=0.4)
        _save(fig, out.with_name(out.stem + "_spearman" + out.suffix))

    def plot_seed_panels(data, ylabel, panel_title, out, rank=False):
        # One subplot per seed within each source figure — no CI since each
        # subplot is a single seed. Within a figure, seeds already share a y
        # axis via sharey=True; share_axis additionally aligns the scale
        # across source figures and hides ticks on non-real ones.
        seeds = sorted(data["seed"].unique())
        share_axis = rank or args.shared_axis
        shared_ylim = None
        if rank:
            shared_ylim = (data["value"].min() - 0.2, data["value"].max() + 0.2)
        elif share_axis:
            shared_ylim = _regret_ylim(data, aggregated=False)

        for source in sources:
            sub_source = data[data["source"] == source]
            fig, axes = plt.subplots(
                1, len(seeds),
                figsize=(style["figsize"][0] * len(seeds) * 0.55, style["figsize"][1]),
                sharey=True,
            )
            axes = [axes] if len(seeds) == 1 else list(axes)
            for ax, seed in zip(axes, seeds):
                sub = sub_source[sub_source["seed"] == seed]
                for method in methods:
                    method_sub = sub[sub["method_base"] == method]
                    sns.lineplot(data=method_sub, x="step", y="value", color=color_map[method],
                                 linewidth=2, ax=ax, label=method, errorbar=None,
                                 linestyle=(0, dashes_map[method]))
                if ax.get_legend():
                    ax.get_legend().remove()
                ax.set_title(f"seed {seed}")
                ax.set_xlabel("BO Iteration")
                ax.set_xlim(0, n_iterations)
                if rank:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                if share_axis:
                    ax.set_ylim(shared_ylim)
            axes[0].set_ylabel(ylabel)
            hidden = args.shared_axis and source != "real"
            if hidden:
                _hide_y_labels(axes[0])
            rect = (0, 0, 1, 1)
            if not args.no_title:
                fig.suptitle(f"{panel_title} ({source_labels[source]})")
                rect = (0, 0, 1, 0.94)
            fig.tight_layout(pad=0.4, rect=rect)
            out_path = out.with_name(out.stem + f"_{source}" + out.suffix)
            _save(fig, out_path)

        save_legend(out)

    def plot_transfer_scatter(data, out):
        # One point per (method, seed) at the final iteration: does relative
        # performance in the emulator carry over to the real objective?
        other_sources = [s for s in sources if s != "real"]
        if not other_sources:
            print("Skipping transfer scatter: no non-real source to compare against.")
            return
        final = data[data["step"] == n_iterations]
        real_final = final[final["source"] == "real"].set_index(["method_base", "seed"])["value"]

        fig, axes = plt.subplots(1, len(other_sources),
                                  figsize=(style["figsize"][0] * 0.9 * len(other_sources) + 0.9,
                                           style["figsize"][1]),
                                  sharex=True, sharey=True)
        axes = [axes] if len(other_sources) == 1 else list(axes)
        for ax, source in zip(axes, other_sources):
            other_final = final[final["source"] == source].set_index(["method_base", "seed"])["value"]
            paired = pd.concat([real_final, other_final], axis=1, join="inner",
                                keys=["real", source]).reset_index()
            for method in methods:
                sub = paired[paired["method_base"] == method]
                ax.scatter(sub["real"], sub[source], color=color_map[method], label=method, s=50)
            lo = min(paired["real"].min(), paired[source].min())
            hi = max(paired["real"].max(), paired[source].max())
            ax.plot([lo, hi], [lo, hi], color="0.5", linestyle="--", linewidth=1, zorder=0)
            r, _ = pearsonr(paired["real"], paired[source])
            rho, _ = spearmanr(paired["real"], paired[source])
            ax.set_title(f"{source_labels[source]} (Pearson r={r:.2f}, Spearman rho={rho:.2f})")
            ax.set_xlabel("Real: Normalised Regret")
            ax.set_aspect("equal", adjustable="box")
        axes[0].set_ylabel("Emulated: Normalised Regret")
        # The per-axes titles carry the correlation values, so they stay even
        # under --no-title; only the figure-level caption is dropped.
        rect = (0, 0, 1, 1)
        if not args.no_title:
            fig.suptitle(f"{title} Relative Performance (iteration {n_iterations})")
            rect = (0, 0, 1, 0.92)
        fig.tight_layout(pad=0.4, rect=rect)
        out_path = out.with_name(out.stem + "_transfer_scatter" + out.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"Saved to {out_path}")

    plot_panels(df, "Normalised Regret", title, out)

    rank_df = rank_frame(df, sources)
    plot_panels(rank_df, "Mean Rank", f"Mean Rank ({title})",
                out.with_name(out.stem + "_rank" + out.suffix), rank=True)
    plot_spearman(rank_df, out.with_name(out.stem + "_rank" + out.suffix))

    if args.combined:
        plot_grid(df, rank_df, out)

    plot_seed_panels(df, "Normalised Regret", f"{title}, per seed",
                      out.with_name(out.stem + "_by_seed" + out.suffix))
    plot_seed_panels(rank_df, "Mean Rank", f"Mean Rank ({title}), per seed",
                      out.with_name(out.stem + "_rank_by_seed" + out.suffix), rank=True)

    plot_transfer_scatter(df, out)


if __name__ == "__main__":
    main()
