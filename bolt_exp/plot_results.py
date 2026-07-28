"""Plot best-so-far score or hypervolume vs budget across result JSON files.

Handles HPO (best_value_all), DM single-objective (best_y_all), and
DM multi-objective (hv_all) outputs automatically.

Usage:
    python plot_results.py hpo_*.json --out hpo.png
    python plot_results.py results/dm/*.json --out dm.png
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


YLABEL_MAP = {
    "log_best_hv_diff_true_all": "Log HV Difference",
    "log_simple_regret_all": "Log Simple Regret",
    "log_best_inference_hv_regret_all": "Log Inference HV Regret",
    "log_inference_hv_regret_all": "Log Inference HV Regret (per step)",
    "log_best_inference_regret_all": "Log Inference Regret",
    "best_inference_regret_all": "Inference Regret",
    "log_inference_regret_all": "Log Inference Regret (per step)",
    "inference_regret_all": "Inference Regret (per step)",
    "log_hv_diff_true_all": "Log True HV Difference (per step)",
    "log_best_obs_regret_all": "Log Regret (per step)",
    "best_obs_regret_all": "Best-Observed Regret",
    "best_y_all": "Best Score",
    "best_value_all": "Best Value",
}


def _method_label(d: dict, show_cost_scale: bool = False) -> str:
    acq = d.get("acq_fn", d.get("method", "unknown"))

    label = acq.upper()

    # rename some labels
    # if label[0] == "Q":
    #     label = "q" + label[1:]

    if acq.lower() == "ei":
        label = "LogEI"
    elif acq.lower() == "qnei":
        label = "LogNEI"
    elif acq.lower() == "qparego":
        label = "ParEGO"
    elif acq.lower() == "qnehvi":
        label = "NEHVI"
    elif acq.lower() == "mfmes":
        label = "MF-MES"
    elif acq.lower() == "mfgibbon":
        label = "MF-GIBBON"

    # add details
    if (
        acq.lower() == "ucb"
        and d.get("ucb_beta") is not None
        and d["ucb_beta"] != "null"
    ):
        label += f"(beta={d['ucb_beta']})"

    if d.get("mlhgp"):
        em = d.get("mlhgp_em_iter")
        label += f"+MLHGP(em={em})" if em else "+MLHGP"

    if d.get("known_noise"):
        label += "+KN"

    if d.get("msr"):
        label = "MSR+" + label
    elif d.get("raasp"):
        label = "RAASP+" + label
    elif d.get("mle_scaled_init"):
        label = "MLESI+" + label

    if d.get("saasbo"):
        label = "SAASBO+" + label
    elif d.get("baxus"):
        label = "dBAxUS+" + label
    elif d.get("turbo"):
        label = "dTuRBO+" + label

    bs = d.get("batch_size", 1)
    if bs > 1:
        label += f"(q={bs})"

    # for nsga2
    pop = d.get("pop_size")
    if pop and pop > 1:
        label += f"(pop_size={pop})"

    if show_cost_scale and d.get("cost_scale"):
        label += f"(cost_scale={d.get('cost_scale')})"

    return label


def load_results(
    path: Path,
    metric: str | None = None,
    by_iter: bool = False,
    use_budget: bool = False,
    use_wall_clock: bool = False,
    show_cost_scale: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, metric_key) for any supported result JSON."""
    with open(path) as f:
        d = json.load(f)

    label = _method_label(d, show_cost_scale=show_cost_scale)
    first = d["trials"][0]

    is_random = d.get("acq_fn", d.get("method", "")).lower() == "random"

    if metric is not None:
        if metric not in first:
            raise KeyError(
                f"Key '{metric}' not found in {path}. Available keys: {list(first)}"
            )
        key = metric
    elif not is_random and "log_best_inference_hv_regret_all" in first:
        key = "log_best_inference_hv_regret_all"
    elif not is_random and "log_inference_hv_regret_all" in first:
        key = "log_inference_hv_regret_all"
    elif not is_random and "log_best_inference_regret_all" in first:
        key = "log_best_inference_regret_all"
    elif not is_random and "log_inference_regret_all" in first:
        key = "log_inference_regret_all"
    elif not is_random and "inference_regret_all" in first:
        key = "inference_regret_all"
    elif "log_best_hv_diff_true_all" in first:
        key = "log_best_hv_diff_true_all"
    elif "log_hv_diff_true_all" in first:
        key = "log_hv_diff_true_all"
    elif "log_best_obs_regret_all" in first:
        key = "log_best_obs_regret_all"
    elif "best_obs_regret_all" in first:
        key = "best_obs_regret_all"
    elif "best_y_all" in first:
        key = "best_y_all"
    else:
        key = "best_value_all"

    if use_wall_clock:
        times = [t["time_seconds"] for t in d["trials"] if "time_seconds" in t]
        if not times:
            raise KeyError(f"'time_seconds' not found in trials of {path}")
        avg_time_per_step = np.mean(times) / len(d["trials"][0][key])

    rows = []
    for trial in d["trials"]:
        vals = trial[key]
        if use_budget:
            budget_all = trial["budget_all"]
            # metric arrays with an initial pre-BO entry are one longer than budget_all
            if len(vals) == len(budget_all) + 1:
                budgets = [0.0] + budget_all
            else:
                budgets = budget_all
            budgets = np.array(budgets)
            vals_arr = np.array(vals)
            # interpolate to every integer budget step via step-function (forward-fill)
            int_steps = np.arange(1, int(budgets[-1]) + 1)
            idx = np.searchsorted(budgets, int_steps, side="right") - 1
            idx = np.clip(idx, 0, len(vals_arr) - 1)
            x_vals = int_steps.tolist()
            vals = vals_arr[idx].tolist()
        elif use_wall_clock:
            x_vals = [step * avg_time_per_step for step in range(len(vals))]
        else:
            q = d.get("batch_size") or d.get("pop_size", 1)
            x_vals = [
                step if by_iter else int(step * q) for step in range(len(vals))
            ]
        rows.extend({"method": label, "step": x, "value": v} for x, v in zip(x_vals, vals))
    return pd.DataFrame(rows), key


def parse_args():
    parser = argparse.ArgumentParser(description="Plot BO results (HPO or DM)")
    parser.add_argument("files", nargs="+", help="Result JSON files to plot")
    parser.add_argument("--out", default=None, help="Output image path (default: show)")
    parser.add_argument("--xlabel", default=None, help="X-axis label override")
    parser.add_argument("--title", default=None, help="Plot title override")
    parser.add_argument(
        "--no-title",
        action="store_true",
        default=False,
        help="Suppress the plot title entirely",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Force a specific JSON key instead of auto-detecting (e.g. log_inference_regret_all)",
    )
    parser.add_argument(
        "--bo-iter",
        action="store_true",
        default=False,
        help="Plot vs BO iteration instead of number of observations",
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        default=False,
        help="Use cumulative budget (budget_all) as x axis (for hpo_fd_step / hpo_fd_model)",
    )
    parser.add_argument(
        "--wall-clock",
        action="store_true",
        default=False,
        help="Use wall clock time as x axis (scaled from total time_seconds / n_steps)",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Maximum x-axis value",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Minimum y-axis value",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Maximum y-axis value",
    )
    parser.add_argument(
        "--errorbar-every",
        type=int,
        default=None,
        metavar="N",
        help="Show error bars every N steps instead of shaded CI (e.g. --errorbar-every 5)",
    )
    parser.add_argument(
        "--cost-scale-label",
        action="store_true",
        default=False,
        help="Append cost_scale to method labels",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help=(
            "YAML file to control label order, colors, and linestyles. "
            "Format: list of entries with keys 'label' (required), 'color' (optional), "
            "'linestyle' (optional: solid/dashed/dotted/dashdot or a dash tuple). "
            "Labels not listed are appended after in default order."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.files)

    dfs, metrics = [], []
    for f in args.files:
        p = Path(f)

        if not p.exists():
            print(f"Warning: {f} not found, skipping")
            continue

        try:
            df, metric = load_results(p, metric=args.metric, by_iter=args.bo_iter, use_budget=args.budget, use_wall_clock=args.wall_clock, show_cost_scale=args.cost_scale_label)
        except KeyError as e:
            print(f"Warning: skipping {f} — {e}")
            continue
        dfs.append(df)
        metrics.append(metric)

    if not dfs:
        print("No data loaded.")
        return

    metric = max(set(metrics), key=metrics.count)
    if len(set(metrics)) > 1:
        print(
            f"Warning: mixing result types {set(metrics)}; using majority metric '{metric}'"
        )

    df = pd.concat(dfs, ignore_index=True)

    max_step = df["step"].max()
    last = (
        df[df["step"] == max_step]
        .groupby("method")["value"]
        .agg(["mean", "std", "count"])
    )
    last.columns = ["mean", "std", "n_trials"]
    last.index.name = "method"
    print(f"\n--- Values at max iteration (step={max_step}) ---")
    print(last.sort_values("mean").to_string())
    print()

    first_d = json.load(open(args.files[0]))
    if args.xlabel:
        xlabel = args.xlabel
    elif args.wall_clock:
        xlabel = "Wall clock time (s)"
    elif args.budget:
        xlabel = "Cumulative budget"
    elif args.bo_iter:
        xlabel = "BO Iteration"
    elif "initial_random_samples" in first_d:
        init = first_d["initial_random_samples"]
        # xlabel = f"Number of observations (after {init} random init)"
        xlabel = f"Number of observations"
    else:
        xlabel = "Number of observations"

    ylabel = YLABEL_MAP.get(metric, metric)
    default_title = f"{ylabel} vs Budget"

    DASH_PATTERNS = [(1, 0), (5, 2), (1, 2), (5, 2, 1, 2)]
    LINESTYLE_ALIASES = {
        "solid": (1, 0),
        "dashed": (5, 2),
        "dotted": (1, 2),
        "dashdot": (5, 2, 1, 2),
    }

    cfg_entries = []
    cfg_title = None
    cfg_figsize = None
    cfg_save_legend = False
    cfg_skip_unlisted = False
    cfg_ymin = None
    cfg_ymax = None
    if args.config:
        with open(args.config) as f:
            cfg_raw = yaml.safe_load(f) or []
        if isinstance(cfg_raw, dict):
            cfg_title = cfg_raw.get("title")
            cfg_entries = cfg_raw.get("methods", [])
            cfg_figsize = cfg_raw.get("figsize")
            cfg_save_legend = cfg_raw.get("save_legend", False)
            cfg_skip_unlisted = cfg_raw.get("skip_unlisted", False)
            cfg_ymin = cfg_raw.get("ymin")
            cfg_ymax = cfg_raw.get("ymax")
        else:
            cfg_entries = cfg_raw

    cfg_order = [e["label"] for e in cfg_entries]
    cfg_colors = {e["label"]: e["color"] for e in cfg_entries if "color" in e}
    cfg_dashes = {}

    for e in cfg_entries:
        if "linestyle" in e:
            ls = e["linestyle"]

            if isinstance(ls, str):
                ls = LINESTYLE_ALIASES.get(ls.lower(), (1, 0))
            else:
                ls = tuple(ls)

            cfg_dashes[e["label"]] = ls

    all_methods = list(df["method"].unique())
    listed = [m for m in cfg_order if m in all_methods]
    unlisted = [m for m in sorted(all_methods) if m not in set(cfg_order)]
    methods = listed if cfg_skip_unlisted else listed + unlisted
    if cfg_skip_unlisted and unlisted:
        print(f"Skipping unlisted methods: {unlisted}")
        df = df[df["method"].isin(methods)]

    print("UNLISTED", unlisted)

    sns.set_theme(style="white", context="paper", font_scale=1.8)
    n_colors = 10
    palette = sns.color_palette("deep", n_colors=n_colors)
    default_color_map = {m: palette[i % n_colors] for i, m in enumerate(methods)}
    default_dashes_map = {
        m: DASH_PATTERNS[(i // n_colors) % len(DASH_PATTERNS)]
        for i, m in enumerate(methods)
    }
    color_map = {m: cfg_colors.get(m, default_color_map[m]) for m in methods}
    dashes_map = {m: cfg_dashes.get(m, default_dashes_map[m]) for m in methods}

    figsize = tuple(cfg_figsize) if cfg_figsize else (7, 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    all_steps = sorted(df["step"].unique())

    if args.errorbar_every is not None:
        n = args.errorbar_every
        step_range = all_steps[-1] - all_steps[0]
        suggested = max(1, round(step_range / 10))
        if n == 0:
            n = suggested
            print(f"Auto-selected --errorbar-every {n} (1/10 of x range)")

        sns.lineplot(
            data=df,
            x="step",
            y="value",
            hue="method",
            style="method",
            hue_order=methods,
            style_order=methods,
            palette=color_map,
            dashes=dashes_map,
            errorbar=None,
            linewidth=2,
            ax=ax,
        )
        checkpoints = [s for s in all_steps if (s - all_steps[0]) % n == 0]
        for method in methods:
            sub = df[df["method"] == method]
            ck = sub[sub["step"].isin(checkpoints)].groupby("step")["value"]
            ax.errorbar(
                ck.mean().index,
                ck.mean().values,
                yerr=(ck.sem() * 1.96).values,
                fmt="none",
                color=color_map[method],
                capsize=3,
                linewidth=1.2,
            )
    else:
        sns.lineplot(
            data=df,
            x="step",
            y="value",
            hue="method",
            style="method",
            hue_order=methods,
            style_order=methods,
            palette=color_map,
            dashes=dashes_map,
            errorbar="ci",
            err_kws={"alpha": 0.15},
            linewidth=2,
            ax=ax,
        )

    xmin = df["step"].min()
    xmax = args.xmax if args.xmax is not None else df["step"].max()
    ax.set_xlim(left=xmin, right=xmax)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ymin = args.ymin if args.ymin is not None else cfg_ymin
    ymax = args.ymax if args.ymax is not None else cfg_ymax
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not args.no_title:
        ax.set_title(args.title or cfg_title or default_title)

    handles, labels = ax.get_legend_handles_labels()

    if cfg_save_legend:
        if ax.get_legend():
            ax.get_legend().remove()
    else:
        ax.legend(title="Method", framealpha=0.9, handlelength=2.4, edgecolor="0.7")

    plt.tight_layout(pad=0.4)

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.08)
        print(f"Saved to {args.out}")
        if cfg_save_legend:
            n = len(labels)
            fig_w = figsize[0]
            ncols_start = max(1, min(n, max(1, round(fig_w / 1.8))))

            legend_fig = plt.figure(figsize=(fig_w, 2.0))
            legend_ax = legend_fig.add_axes([0, 0, 1, 1])
            legend_ax.set_axis_off()

            legend_kwargs = dict(
                loc="center",
                frameon=True,
                fontsize="small",
                handlelength=2.2,
                columnspacing=1.0,
                handletextpad=0.5,
                edgecolor="0.7",
                framealpha=0.9,
            )

            def _make_legend(nc):
                leg = legend_ax.legend(handles, labels, ncol=nc, **legend_kwargs)
                for h in leg.legend_handles:
                    h.set_linewidth(2)
                return leg

            # Increase ncols until legend fills ~90% of figure width or all items in one row
            ncols = ncols_start
            while ncols < n:
                leg = _make_legend(ncols)
                legend_fig.canvas.draw()
                leg_w = leg.get_window_extent().width
                fig_w_px = legend_fig.get_window_extent().width
                if leg_w >= fig_w_px * 0.88:
                    break
                ncols += 1
            else:
                leg = _make_legend(ncols)
                legend_fig.canvas.draw()

            nrows = math.ceil(n / ncols)
            legend_fig.set_size_inches(fig_w, max(0.4, nrows * 0.22))
            legend_fig.canvas.draw()

            out_path = Path(args.out)
            legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
            legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
            print(f"Saved legend to {legend_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
