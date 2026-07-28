"""Plot baxus_target_dim_all, baxus_ncand_all, or turbo_ncand_all vs BO iteration.

Detects the metric to plot from the filename: files containing "baxus" produce
two plots (baxus_target_dim_all and baxus_ncand_all); files containing "turbo"
use turbo_ncand_all.

Usage:
    python plot_baxus_turbo.py results/po_bak5_mc/po128_qnei_baxus*.json --out baxus_dims.png
    python plot_baxus_turbo.py results/po_bak5_turbo/po128_*turbo*.json --out turbo_k.png
    python plot_baxus_turbo.py results/po_bak5_mc/*.json --out mixed.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bolt_exp.plot_results import _method_label


# Each kind maps to a list of (metric_key, ylabel) pairs to plot
METRICS_BY_TYPE = {
    "baxus": [
        ("baxus_target_dim_all", "Target Dimensionality"),
        ("baxus_ncand_all", "Number of BAxUS Candidates"),
    ],
    "turbo": [
        ("turbo_ncand_all", "Number of TuRBO Candidates"),
    ],
}


def detect_type(path: Path) -> str | None:
    name = path.name.lower()
    if "baxus" in name:
        return "baxus"
    if "turbo" in name:
        return "turbo"
    return None


def load_file(path: Path) -> tuple[dict[str, pd.DataFrame], str, str] | None:
    """Return (metric -> DataFrame, kind, problem) or None if file should be skipped."""
    kind = detect_type(path)
    if kind is None:
        print(f"Skipping {path.name}: no 'baxus' or 'turbo' in filename")
        return None

    with open(path) as f:
        d = json.load(f)

    first = d["trials"][0]
    problem = d.get("problem", "unknown")
    label = _method_label(d)

    dfs: dict[str, pd.DataFrame] = {}
    for metric, _ylabel in METRICS_BY_TYPE[kind]:
        if metric not in first:
            print(f"Skipping metric '{metric}' in {path.name}: key not found")
            continue
        rows = [
            {"method": label, "step": step, "value": val}
            for trial in d["trials"]
            for step, val in enumerate(trial[metric])
        ]
        dfs[metric] = pd.DataFrame(rows)

    if not dfs:
        print(f"Skipping {path.name}: no recognised metrics found")
        return None

    return dfs, kind, problem


def parse_args():
    parser = argparse.ArgumentParser(description="Plot BAxUS target dims or TuRBO k vs iteration")
    parser.add_argument("files", nargs="+", help="Result JSON files to plot")
    parser.add_argument("--out", default=None, help="Output image path (default: show)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    return parser.parse_args()


def plot_problem(df: pd.DataFrame, ylabel: str, problem: str, args, out_path: Path | None):
    methods = list(df["method"].unique())

    n_colors = 10
    palette = sns.color_palette("deep", n_colors=n_colors)
    DASH_PATTERNS = [(1, 0), (5, 2), (1, 2), (5, 2, 1, 2)]
    color_map = {m: palette[i % n_colors] for i, m in enumerate(methods)}
    dashes_map = {m: DASH_PATTERNS[(i // n_colors) % len(DASH_PATTERNS)] for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(7, 4.5))

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
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(args.title or f"{problem} — {ylabel} vs BO Iteration")
    ax.legend(title="Method", framealpha=0.8, handlelength=2.4)

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    # group by (problem, metric) -> list of DataFrames
    by_problem_metric: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"Warning: {f} not found, skipping")
            continue
        result = load_file(p)
        if result is None:
            continue
        dfs_by_metric, _kind, problem = result
        for metric, df in dfs_by_metric.items():
            by_problem_metric.setdefault((problem, metric), []).append(df)

    if not by_problem_metric:
        print("No data loaded.")
        return

    sns.set_theme(style="white", context="paper", font_scale=1.2)

    # build ylabel lookup: metric -> ylabel
    ylabel_map: dict[str, str] = {}
    for metrics in METRICS_BY_TYPE.values():
        for metric, ylabel in metrics:
            ylabel_map[metric] = ylabel

    problems = sorted({p for p, _ in by_problem_metric})
    for problem in problems:
        # collect all metrics present for this problem
        metrics_for_problem = sorted({m for (pr, m) in by_problem_metric if pr == problem})
        for metric in metrics_for_problem:
            df = pd.concat(by_problem_metric[(problem, metric)], ignore_index=True)

            if args.out:
                out_path = Path(args.out)
                # embed metric suffix so baxus produces separate files per metric
                metric_suffix = metric.replace("_all", "").replace("baxus_", "baxus").replace("turbo_", "turbo")
                stem = out_path.stem
                suffix = out_path.suffix
                new_name = f"{stem}_{metric_suffix}{suffix}"
                if len(problems) > 1:
                    new_name = f"{problem}_{new_name}"
                out_path = out_path.with_name(new_name)
            else:
                out_path = None

            ylabel = ylabel_map.get(metric, metric)
            plot_problem(df, ylabel, problem, args, out_path)


if __name__ == "__main__":
    main()
