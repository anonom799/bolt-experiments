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
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml

from bolt_exp import dedupe_results, load_result
from bolt_exp.hpo_flops import flops as query_flops


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
    "normalized_regret_all": "Normalised Regret",
    "best_y_all": "Best Score",
    "best_obs_true_all": "Best Observed Value",
    "best_value_all": "Best Value",
}


def _method_label(d: dict, show_cost_scale: bool = False) -> str:
    acq = d.get("acq_fn", d.get("method", "unknown"))

    label = acq.upper()

    # rename some labels
    # if label[0] == "Q":
    #     label = "q" + label[1:]

    # A constrained run records the recommender's feasibility threshold; the two
    # EI variants are then built with `constraints=` (EIC of Gardner et al. 2014
    # and the constrained qLogNEI of Ament et al. 2023), so they are different
    # acquisitions from their unconstrained namesakes and are named as such.
    constrained = d.get("pf_delta") is not None

    if acq.lower() == "ei":
        label = "LogEIC" if constrained else "LogEI"
    elif acq.lower() == "qnei":
        label = "LogNEIC" if constrained else "LogNEI"
    elif acq.lower() == "qparego":
        label = "ParEGO"
    elif acq.lower() == "qnehvi":
        label = "NEHVI"
    elif acq.lower() == "nsga2":
        label = "NSGA-II"
    elif acq.lower() == "nsga3":
        label = "NSGA-III"
    elif acq.lower() == "mfmes":
        label = "MF-MES"
    elif acq.lower() == "mfgibbon":
        label = "MF-GIBBON"
    # PCO constrained acquisitions: the leading "c" is a lowercase modifier
    # ("constrained"), so it is not upper-cased with the rest of the name.
    elif acq.lower() == "ckg":
        label = "cKG"
    elif acq.lower() == "cts":
        label = "cTS"
    elif acq.lower() == "ucb_c":
        label = "UCB-C"
    elif acq.lower() == "cmes_ibo":
        label = "CMES-IBO"
    elif acq.lower() == "scbo":
        label = "dSCBO"

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

    # PCO's physics-informed GP means: an analytic communication-cost model for
    # throughput and a peak-memory model for the constraint, in place of a
    # constant mean. A model change, not a different acquisition, so it reads as
    # a suffix on whichever acquisition carries it.
    if d.get("prior_mean"):
        label += "+prior"

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

    # PCO sampling without replacement (--no_repeats): same acquisition, with
    # already-evaluated configurations removed from the candidate pool
    if d.get("no_repeats"):
        label += "(no repeats)"

    # for nsga2
    pop = d.get("pop_size")
    if pop and pop > 1:
        label += f"(pop_size={pop})"

    if show_cost_scale and d.get("cost_scale"):
        label += f"(cost_scale={d.get('cost_scale')})"

    return label


# ------------------------------------------------------------------
# HPO-B style normalised regret
# ------------------------------------------------------------------
# Synthetic metric key: it is NOT stored in the results JSONs, because the
# normalisation pool is the set of files being plotted. Two figures over
# different method subsets get different constants, so the value is a property
# of the figure rather than of the run and has to be computed on demand.
NORMALIZED_METRIC = "normalized_regret_all"


def seen_y_bounds(d: dict) -> tuple[float, float]:
    """(max, min) over every noisy observation of every trial in one file."""
    vals = [
        float(v[0]) if isinstance(v, list) else float(v)
        for trial in d["trials"]
        for v in trial["seen_y"]
    ]
    return max(vals), min(vals)


def normalize_regret(
    df: pd.DataFrame, bounds: list[tuple[float, float]]
) -> pd.DataFrame:
    """Rescale best-so-far noisy y onto [0, 1] over the pooled file bounds.

        normalized_regret[t] = (y_max - best_y_all[t]) / (y_max - y_min)

    y_max / y_min are the best / worst single noisy observation across all
    seeds and all methods in `bounds`. Lower is better; 0 means that run found
    the best value anyone in the pool found.
    """
    y_max = max(b[0] for b in bounds)
    y_min = min(b[1] for b in bounds)
    rng = y_max - y_min
    if rng <= 0:
        raise ValueError(f"Degenerate normalisation range: y_max == y_min == {y_max}")
    print(
        f"Normalisation pool ({len(bounds)} files): "
        f"y_max={y_max:.6f} y_min={y_min:.6f} range={rng:.6f}"
    )
    out = df.copy()
    out["value"] = (y_max - out["value"]) / rng
    return out


# Metrics where a larger value is better; everything else is a regret/error
# and is ranked ascending.
HIGHER_IS_BETTER = {
    "best_y_all",
    "best_obs_true_all",
    "best_value_all",
    "hv_all",
    "hv_true_all",
    "best_hv_true_all",
    "inf_hv_all",
    "best_inf_hv_all",
}


def mean_rank_frame(
    df: pd.DataFrame, metric: str, group_fn: Callable[[str], str] | None = None
) -> pd.DataFrame:
    """Rank methods against each other within every (step, seed).

    Rank 1 is the best method for that seed at that step; ties share the
    average rank. Seeds are paired across methods, so only seeds present for
    every method are used — otherwise the rank scale would differ per seed.
    Averaging over seeds is left to the caller's plotting (same as any other
    metric), so the CI band is over seeds.

    ``group_fn`` maps a method label to a comparison group; methods are then
    ranked only against the others in their own group (e.g. group by batch
    size so the 4 acquisitions are ranked against each other at fixed q,
    rather than every method-and-q curve sharing one rank scale). The group
    is returned in a "group" column for faceting. Default: one global group.
    """
    per_method = df.groupby("method")["seed"].agg(set)
    common = set.intersection(*per_method) if len(per_method) else set()
    if not common:
        raise ValueError(
            "No seed is shared by every method; cannot rank them against "
            f"each other. Seeds per method: {per_method.to_dict()}"
        )
    dropped = set().union(*per_method) - common
    if dropped:
        print(f"Mean rank: using {len(common)} shared seed(s), dropping {sorted(dropped)}")

    out = df[df["seed"].isin(common)].copy()
    out["group"] = out["method"].map(group_fn) if group_fn else ""
    out["value"] = out.groupby(["group", "step", "seed"])["value"].rank(
        ascending=metric not in HIGHER_IS_BETTER, method="average"
    )
    return out


def infer_variant(path: Path, results: dict) -> str:
    """Which HPO problem a result file came from."""
    problem = results.get("problem", "")
    for name in ("hpo_fd_step", "hpo_fd_model"):
        if problem == name or name in path.stem:
            return name
    return "hpo"


def n_skip_for(results: dict) -> int:
    """Leading candidates excluded from the cost axis, matching add_budget_all.py."""
    method = results.get("method", results.get("acq_fn", ""))
    if method.lower() == "asha":
        return 1
    return results.get("initial_random_samples", 0)


def flops_budget(trial: dict, variant: str, n_skip: int, units: float) -> list[float]:
    """Cumulative training FLOPs (in `units`) after each query of one trial.

    Same accounting as the `budget_all` cost axis -- the initial design is
    excluded -- but each query is priced by the FLOPs formula in `hpo_flops.py`
    instead of by the problem's normalized `cost()`.
    """
    X = torch.tensor(trial["candidates"], dtype=torch.double)
    per_query = query_flops(X, variant).numpy()[n_skip:]
    return (np.cumsum(per_query) / units).tolist()


def load_results(
    path: Path,
    metric: str | None = None,
    by_iter: bool = False,
    use_budget: bool = False,
    use_flops: bool = False,
    flops_units: float = 1e17,
    use_wall_clock: bool = False,
    show_cost_scale: bool = False,
    bounds_out: list[tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, metric_key) for any supported result JSON."""
    d = load_result(path)

    label = _method_label(d, show_cost_scale=show_cost_scale)
    first = d["trials"][0]

    is_random = d.get("acq_fn", d.get("method", "")).lower() == "random"

    # normalised regret reads raw best-so-far values here; normalize_regret()
    # rescales them in main() once every file's bounds are known
    metric_name = metric
    if metric == NORMALIZED_METRIC:
        key = "best_y_all"
        if bounds_out is not None:
            bounds_out.append(seen_y_bounds(d))
    elif metric is not None:
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

    if use_flops:
        variant = infer_variant(path, d)
        n_skip = n_skip_for(d)
        if "candidates" not in first:
            raise KeyError(f"'candidates' not found in trials of {path}")

    rows = []
    for i, trial in enumerate(d["trials"]):
        seed = trial.get("seed", i)
        vals = trial[key]
        if use_budget or use_flops:
            if use_flops:
                budget_all = flops_budget(trial, variant, n_skip, flops_units)
            else:
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
        rows.extend(
            {"method": label, "seed": seed, "step": x, "value": v}
            for x, v in zip(x_vals, vals)
        )
    return pd.DataFrame(rows), metric_name or key


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
        help=(
            "Force a specific JSON key instead of auto-detecting "
            "(e.g. log_inference_regret_all), or the synthetic key "
            f"{NORMALIZED_METRIC} for HPO-B style normalised regret "
            "computed over the plotted files."
        ),
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
        "--flops",
        action="store_true",
        default=False,
        help=(
            "Use cumulative training FLOPs as x axis (HPO problems only). "
            "Each query is priced from its hyperparameters by the formula in "
            "hpo_flops.py, so model size, training tokens and LoRA depth all "
            "change how far a query moves the axis."
        ),
    )
    parser.add_argument(
        "--flops-units",
        type=float,
        default=1e17,
        help="FLOPs per unit of the --flops x axis (default 1e17)",
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
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help=(
            "Figure size in inches, overriding the config's figsize. With "
            "--facet-by-base, W is the width of each subplot."
        ),
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
        "--mean-rank",
        action="store_true",
        default=False,
        help=(
            "Additionally plot mean rank across methods for the same metric, "
            "saved alongside --out as {stem}_rank{suffix}. Methods are ranked "
            "within each (step, seed), rank 1 = best."
        ),
    )
    parser.add_argument(
        "--rank-within-q",
        action="store_true",
        default=False,
        help=(
            "Make --mean-rank rank methods only against the others sharing "
            "their variant (batch size q, or w/ repeats vs no repeats), with "
            "one subplot per variant. Answers whether the ordering of the "
            "acquisitions is preserved across variants, instead of putting "
            "every method-and-variant curve on one rank scale where a method "
            "effect and a variant effect are indistinguishable."
        ),
    )
    parser.add_argument(
        "--facet-by-base",
        action="store_true",
        default=False,
        help=(
            "One subplot per base method (label with any trailing '(q=N)' or "
            "'(no repeats)' stripped), sharing a y axis, with the variant "
            "encoded as linestyle (from --config) inside each subplot instead "
            "of overlaying all variants of all methods on one axes."
        ),
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


def draw_figure(
    df: pd.DataFrame,
    methods: list[str],
    color_map: dict,
    dashes_map: dict,
    figsize: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str | None,
    out: str | Path | None,
    errorbar_every: int | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    integer_yticks: bool = True,
    save_legend: bool = False,
    write_legend: bool = True,
    ci_alpha: float = 0.15,
) -> None:
    """Draw and save one <metric> vs <step> figure.

    `df` needs columns method/step/value; the styling maps are shared by every
    figure of a run so the main plot and its mean-rank companion stay visually
    consistent. `title=None` suppresses the title. With `out=None` nothing is
    written and the caller is expected to plt.show().
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_steps = sorted(df["step"].unique())

    if errorbar_every is not None:
        n = errorbar_every
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
            # ci_alpha 0 drops the band (config `ci_alpha`).
            errorbar="ci" if ci_alpha else None,
            err_kws={"alpha": ci_alpha},
            linewidth=2,
            ax=ax,
        )

    ax.set_xlim(left=df["step"].min(), right=xmax if xmax is not None else df["step"].max())
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    if integer_yticks:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()

    if save_legend:
        if ax.get_legend():
            ax.get_legend().remove()
    else:
        ax.legend(title="Method", framealpha=0.9, handlelength=2.4, edgecolor="0.7")

    plt.tight_layout(pad=0.4)

    if not out:
        return

    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out}")
    if not (save_legend and write_legend):
        return

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

    out_path = Path(out)
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print(f"Saved legend to {legend_path}")


# A variant suffix marks one run setting compared across the same acquisition:
# batch size ('(q=N)') or PCO sampling without replacement ('(no repeats)').
VARIANT_SUFFIX_RE = re.compile(r"\((q=\d+|no repeats)\)$")
NO_REPEATS = "no repeats"
WITH_REPEATS = "w/ repeats"


def base_method(label: str) -> str:
    """Strip a trailing variant suffix, e.g. 'LogNEI(q=5)' -> 'LogNEI'."""
    return VARIANT_SUFFIX_RE.sub("", label)


def variant_fn(methods: list[str]) -> Callable[[str], str]:
    """Map a label to the variant it encodes, e.g. 'LogNEI(q=5)' -> 'q=5'.

    A label with no suffix is the default setting of whichever axis the plotted
    methods vary along: sequential (q=1) for a batch-size comparison, or
    sampling with repeats when any method is a no-repeats run.
    """
    default = WITH_REPEATS if any(m.endswith(f"({NO_REPEATS})") for m in methods) else "q=1"

    def variant_of(label: str) -> str:
        match = VARIANT_SUFFIX_RE.search(label)
        return match.group(1) if match else default

    return variant_of


def variant_sort_key(variant: str) -> tuple[int, int]:
    """Order variants as q=1 < q=5 < q=10, and w/ repeats before no repeats."""
    if variant.startswith("q="):
        return (0, int(variant.removeprefix("q=")))
    return (1, variant == NO_REPEATS)


def draw_faceted_figure(
    df: pd.DataFrame,
    methods: list[str],
    color_map: dict,
    dashes_map: dict,
    figsize: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str | None,
    out: str | Path | None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    integer_yticks: bool = True,
    save_legend: bool = False,
    group_fn: Callable[[str], str] = base_method,
    group_order: list[str] | None = None,
) -> None:
    """One subplot per method group, the group's members overlaid within it.

    Declutters an all-methods-and-all-q-values overlay by giving each group
    its own axes. Grouping by base method (the default) puts one acquisition
    per axes with q encoded by linestyle; grouping by batch size instead puts
    one q per axes so the acquisitions are compared at a fixed batch size.
    """
    bases = group_order or list(dict.fromkeys(group_fn(m) for m in methods))
    n = len(bases)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), sharey=True, sharex=True)
    axes = [axes] if n == 1 else list(axes)

    for ax, base in zip(axes, bases):
        sub_methods = [m for m in methods if group_fn(m) == base]
        sub_df = df[df["method"].isin(sub_methods)]
        sns.lineplot(
            data=sub_df,
            x="step",
            y="value",
            hue="method",
            style="method",
            hue_order=sub_methods,
            style_order=sub_methods,
            palette=color_map,
            dashes=dashes_map,
            errorbar="ci",
            err_kws={"alpha": 0.15},
            linewidth=2,
            ax=ax,
            legend=False,
        )
        ax.set_title(base)
        ax.set_xlabel("")
        ax.set_xlim(left=df["step"].min(), right=xmax if xmax is not None else df["step"].max())
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        if integer_yticks:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    axes[0].set_ylabel(ylabel)
    for ax in axes[1:]:
        ax.set_ylabel("")
        # sharey=True aligns the scale but still draws tick labels per axes
        ax.tick_params(labelleft=False)

    if title:
        fig.suptitle(title)

    plt.tight_layout(pad=0.4, rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(wspace=0.35)
    # one shared x-label centered under all subplots, instead of one per axes;
    # tight_layout's rect reserves the bottom strip so it clears the tick labels
    fig.supxlabel(xlabel, y=0.02)

    if not out:
        return

    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out}")

    if not save_legend:
        return

    # One shared legend for the variant (linestyle), independent of per-method color.
    variant_of = variant_fn(methods)
    variant_dashes = {}
    for m in methods:
        variant_dashes.setdefault(variant_of(m), dashes_map[m])
    variant_order = sorted(variant_dashes, key=variant_sort_key)

    handles = [
        plt.Line2D([0], [0], color="black", dashes=variant_dashes[v], linewidth=2, label=v)
        for v in variant_order
    ]
    legend_fig = plt.figure(figsize=(1.2 * len(variant_order), 0.6))
    legend_ax = legend_fig.add_axes([0, 0, 1, 1])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=handles,
        ncol=len(variant_order),
        loc="center",
        frameon=True,
        fontsize="small",
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.5,
        edgecolor="0.7",
        framealpha=0.9,
    )
    out_path = Path(out)
    legend_path = out_path.with_name(out_path.stem + "_legend" + out_path.suffix)
    legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print(f"Saved legend to {legend_path}")


def main():
    args = parse_args()
    args.files = [str(p) for p in dedupe_results(args.files)]
    print(args.files)

    dfs, metrics = [], []
    bounds: list[tuple[float, float]] = []
    for f in args.files:
        p = Path(f)

        if not p.exists() and not p.with_name(p.name + ".gz").exists():
            print(f"Warning: {f} not found, skipping")
            continue

        try:
            df, metric = load_results(p, metric=args.metric, by_iter=args.bo_iter, use_budget=args.budget, use_flops=args.flops, flops_units=args.flops_units, use_wall_clock=args.wall_clock, show_cost_scale=args.cost_scale_label, bounds_out=bounds)
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

    if metric == NORMALIZED_METRIC:
        df = normalize_regret(df, bounds)

    if args.flops:
        # Methods spend different compute per query, so they run out at different
        # x; past this point a curve is backed by only some of the files.
        common = df.groupby(["method", "seed"])["step"].max().min()
        print(f"All methods reach x={common:.0f}; consider --xmax {common:.0f}")

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

    first_d = load_result(args.files[0])
    if args.xlabel:
        xlabel = args.xlabel
    elif args.wall_clock:
        xlabel = "Wall clock time (s)"
    elif args.flops:
        exponent = int(round(math.log10(args.flops_units)))
        unit = "FLOPs" if exponent == 0 else rf"$10^{{{exponent}}}$ FLOPs"
        xlabel = f"Estimated compute ({unit})"
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
    cfg_ci_alpha = 0.15
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
            cfg_ci_alpha = cfg_raw.get("ci_alpha", 0.15)
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

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.figsize:
        figsize = tuple(args.figsize)
    elif cfg_figsize:
        figsize = tuple(cfg_figsize)
    else:
        figsize = (7, 4.5)
    style = dict(
        methods=methods,
        color_map=color_map,
        dashes_map=dashes_map,
        figsize=figsize,
        xlabel=xlabel,
        errorbar_every=args.errorbar_every,
        xmax=args.xmax,
        save_legend=cfg_save_legend,
        ci_alpha=cfg_ci_alpha,
    )

    if args.facet_by_base:
        draw_faceted_figure(
            df,
            methods=methods,
            color_map=color_map,
            dashes_map=dashes_map,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            title=None if args.no_title else (args.title or cfg_title or default_title),
            out=args.out,
            xmax=args.xmax,
            ymin=args.ymin if args.ymin is not None else cfg_ymin,
            ymax=args.ymax if args.ymax is not None else cfg_ymax,
            integer_yticks=metric != NORMALIZED_METRIC,
            save_legend=cfg_save_legend,
        )
    else:
        draw_figure(
            df,
            ylabel=ylabel,
            title=None if args.no_title else (args.title or cfg_title or default_title),
            out=args.out,
            ymin=args.ymin if args.ymin is not None else cfg_ymin,
            ymax=args.ymax if args.ymax is not None else cfg_ymax,
            integer_yticks=metric != NORMALIZED_METRIC,
            **style,
        )

    if args.mean_rank:
        rank_df = mean_rank_frame(
            df, metric, group_fn=variant_fn(methods) if args.rank_within_q else None
        )
        rank_out = None
        if args.out:
            op = Path(args.out)
            rank_out = op.with_name(op.stem + "_rank" + op.suffix)
        if args.rank_within_q:
            variant_of = variant_fn(methods)
            variant_order = sorted(
                dict.fromkeys(variant_of(m) for m in methods), key=variant_sort_key
            )
            draw_faceted_figure(
                rank_df,
                methods=methods,
                color_map=color_map,
                dashes_map=dashes_map,
                figsize=figsize,
                xlabel=xlabel,
                ylabel="Mean Rank",
                title=None
                if args.no_title
                else f"Mean Rank ({args.title or cfg_title or ylabel})",
                out=rank_out,
                xmax=args.xmax,
                group_fn=variant_of,
                group_order=variant_order,
            )
            if not args.out:
                plt.show()
            return
        rank_style = dict(style)
        if args.facet_by_base:
            # style["figsize"] is a per-subplot width for the faceted main
            # plot; the rank plot is a single (non-faceted) axes, so scale
            # back up to the overall width the facets together span.
            n_bases = len(dict.fromkeys(base_method(m) for m in methods))
            rank_style["figsize"] = (figsize[0] * n_bases, figsize[1])
        draw_figure(
            rank_df,
            ylabel="Mean Rank",
            title=None
            if args.no_title
            else f"Mean Rank ({args.title or cfg_title or ylabel})",
            out=rank_out,
            write_legend=False,
            **rank_style,
        )

    if not args.out:
        plt.show()


if __name__ == "__main__":
    main()
