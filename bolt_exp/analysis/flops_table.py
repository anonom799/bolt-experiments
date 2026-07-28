"""Tabulate a metric at a fixed compute budget from the FLOPs-costed HPO results.

Reads the result copies written by `compute_hpo_flops.py` (whose `budget_all` is
cumulative FLOPs) and reports, per method, the metric value once a given amount
of compute has been spent, plus how many BO iterations that took.

The value at a budget is the last observation at or before it, matching the
forward-fill that `plot_results.load_results` uses to build the x axis, so the
numbers here are the ones the figures show.

Usage:
    python flops_table.py results_flops/hpo/hpo_{random,tpe,qnei,mes}_*.json \
        --budget 400 --out flops/hpo_flops_table.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from bolt_exp.plot_results import _method_label

from bolt_exp import REPO_ROOT


def value_at_budget(trial: dict, key: str, budget: float) -> tuple[float, int]:
    """Metric value once `budget` is spent, and the BO iteration that took.

    Index 0 of the metric array is the initial design, so the index doubles as
    the number of completed BO iterations.
    """
    vals = np.asarray(trial[key], dtype=float)
    budgets = np.asarray([0.0] + trial["budget_all"], dtype=float)
    if len(vals) != len(budgets):
        budgets = budgets[: len(vals)]
    if budget > budgets[-1]:
        raise ValueError(
            f"budget {budget:g} exceeds this trial's total of {budgets[-1]:.0f}"
        )
    i = int(np.searchsorted(budgets, budget, side="right") - 1)
    return float(vals[i]), i


def summarise(path: Path, key: str, budget: float, cost_from: Path | None) -> dict:
    """Summarise one result file at `budget`.

    If `cost_from` is given, the last column reports the original abstract cost
    budget (`fidelity * 0.9 + 0.1`, cumulated) spent at the same point, read from
    the untouched results in that directory. That is the x axis of the
    multi-fidelity figures in `plot_figs.sh`, so the two are directly comparable.
    """
    with open(path) as f:
        results = json.load(f)

    original = None
    if cost_from is not None:
        with open(cost_from / path.name) as f:
            original = json.load(f)["trials"]

    values, iters, costs = [], [], []
    for j, trial in enumerate(results["trials"]):
        v, i = value_at_budget(trial, key, budget)
        values.append(v)
        iters.append(i)
        if original is not None:
            # budgets[i] in value_at_budget is [0.0] + budget_all, so index i-1.
            costs.append(0.0 if i == 0 else original[j]["budget_all"][i - 1])

    values = np.asarray(values)
    n = len(values)
    # t-based 95% CI on the mean over trials; n is small (5), so not normal-based.
    half_width = (
        stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    )
    return {
        "method": _method_label(results),
        "mean": values.mean(),
        "half_width": half_width,
        "n_trials": n,
        "last": float(np.mean(costs)) if costs else float(np.mean(iters)),
    }


def to_markdown(
    rows: list[dict],
    budget: float,
    units: float,
    key: str,
    label: str | None = None,
    display_exp: int = 19,
    last_column: str = "Avg iteration",
) -> str:
    pretty = key.replace("log_", "log ").replace("_all", "").replace("_", " ")
    title = f"**{label}**\n\n" if label else ""
    shown = budget * units / 10**display_exp
    header = title + (
        f"| Method | {pretty} at {shown:g} × 10^{display_exp} FLOPs "
        f"[95% CI] | {last_column} |\n"
        "| --- | --- | --- |\n"
    )
    body = "".join(
        f"| {r['method']} | {r['mean']:.2f} [{r['mean'] - r['half_width']:.2f}, "
        f"{r['mean'] + r['half_width']:.2f}] | {r['last']:.1f} |\n"
        for r in rows
    )
    return header + body


def main(args):
    cost_from = Path(args.cost_from) if args.cost_from else None
    rows = [summarise(Path(f), args.metric, args.budget, cost_from) for f in args.files]
    if args.sort:
        rows.sort(key=lambda r: r["mean"])

    last_column = "Avg cumulative cost" if cost_from else "Avg iteration"
    table = to_markdown(
        rows,
        args.budget,
        args.units,
        args.metric,
        args.label,
        args.display_exp,
        last_column,
    )
    print(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(f"Wrote {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tabulate a metric at a fixed compute budget"
    )
    parser.add_argument("files", nargs="+", help="FLOPs-costed result JSON files")
    parser.add_argument(
        "--budget",
        type=float,
        default=400,
        help="Compute budget, in the units budget_all was written in",
    )
    parser.add_argument(
        "--units",
        type=float,
        default=1e17,
        help="FLOPs per unit of budget_all, for the column header",
    )
    parser.add_argument("--metric", default="log_simple_regret_all")
    parser.add_argument("--out", default=str(REPO_ROOT / "flops" / "hpo_flops_table.md"))
    parser.add_argument("--label", default=None, help="Bold heading above the table")
    parser.add_argument(
        "--cost_from",
        default=None,
        help="Directory of the original results; report their cumulative cost "
        "budget instead of the iteration count (multi-fidelity only)",
    )
    parser.add_argument(
        "--display_exp",
        type=int,
        default=19,
        help="Power of ten the budget is quoted in, in the column header",
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort rows by the metric, best first"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
