"""Plot cost-scale comparison as a 1×N grid: one subplot per method.

Each subplot shows all cost_scale variants (solid/dashed/dotted) for that method.

Usage:
    python plot_cost_scale_grid.py \
        $(ls results/hpo/hpo_fd_step_*_50iterations*cost*.json) \
        --config plot_configs/hpo_step_cost_scale_okabe_ito.yaml \
        --out hpo_step_cost_simple_grid.png \
        --metric log_simple_regret_all --budget --cost-scale-label
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from bolt_exp import dedupe_results
from bolt_exp.plot_results import load_results, YLABEL_MAP


def _parse_args():
    p = argparse.ArgumentParser(description="1×N cost-scale grid plot")
    p.add_argument("files", nargs="+", help="Result JSON files")
    p.add_argument("--config", required=True, metavar="YAML")
    p.add_argument("--out", default=None, help="Output image path")
    p.add_argument("--metric", default=None)
    p.add_argument("--budget", action="store_true", default=False)
    p.add_argument("--wall-clock", action="store_true", default=False)
    p.add_argument("--bo-iter", action="store_true", default=False)
    p.add_argument("--cost-scale-label", action="store_true", default=False)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--subplot-width", type=float, default=4.0)
    p.add_argument("--height", type=float, default=4.0)
    return p.parse_args()


def _strip_cost_scale(label: str) -> str:
    return re.sub(r"\(cost_scale=[^)]+\)", "", label).strip()


def main():
    args = _parse_args()

    with open(args.config) as f:
        cfg_raw = yaml.safe_load(f) or {}
    cfg_entries = cfg_raw.get("methods", []) if isinstance(cfg_raw, dict) else cfg_raw
    cfg_ymin = cfg_raw.get("ymin") if isinstance(cfg_raw, dict) else None
    cfg_ymax = cfg_raw.get("ymax") if isinstance(cfg_raw, dict) else None

    cfg_colors = {e["label"]: e["color"] for e in cfg_entries if "color" in e}
    cfg_linestyles = {e["label"]: e.get("linestyle", "solid") for e in cfg_entries}
    cfg_order = [e["label"] for e in cfg_entries]

    dfs, metrics = [], []
    for f in dedupe_results(args.files):
        path = Path(f)
        if not path.exists() and not path.with_name(path.name + ".gz").exists():
            print(f"Warning: {f} not found, skipping")
            continue
        try:
            df, metric = load_results(
                path,
                metric=args.metric,
                by_iter=args.bo_iter,
                use_budget=args.budget,
                use_wall_clock=args.wall_clock,
                show_cost_scale=args.cost_scale_label,
            )
        except KeyError as e:
            print(f"Warning: skipping {f} — {e}")
            continue
        dfs.append(df)
        metrics.append(metric)

    if not dfs:
        print("No data loaded.")
        return

    metric = max(set(metrics), key=metrics.count)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[df_all["method"].isin(cfg_order)]

    # unique base methods in config order
    seen: set[str] = set()
    base_methods: list[str] = []
    for label in cfg_order:
        base = _strip_cost_scale(label)
        if base not in seen:
            seen.add(base)
            base_methods.append(base)

    n_methods = len(base_methods)
    sns.set_theme(style="white", context="paper", font_scale=1.5)

    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(args.subplot_width * n_methods, args.height),
        sharey=False,
    )
    if n_methods == 1:
        axes = [axes]

    ylabel = YLABEL_MAP.get(metric, metric)
    ymin = args.ymin if args.ymin is not None else cfg_ymin
    ymax = args.ymax if args.ymax is not None else cfg_ymax

    for ax, base in zip(axes, base_methods):
        sub_labels = [lbl for lbl in cfg_order if _strip_cost_scale(lbl) == base]
        sub_df = df_all[df_all["method"].isin(sub_labels)]

        if sub_df.empty:
            ax.set_visible(False)
            continue

        for lbl in sub_labels:
            short = re.sub(r"^[^(]+", "", lbl)  # "(cost_scale=X)"
            color = cfg_colors.get(lbl, "#333333")
            linestyle = cfg_linestyles.get(lbl, "solid")
            sns.lineplot(
                data=sub_df[sub_df["method"] == lbl],
                x="step",
                y="value",
                color=color,
                linestyle=linestyle,
                errorbar="ci",
                err_kws={"alpha": 0.15},
                linewidth=2,
                label=short,
                ax=ax,
            )

        xmin_val = sub_df["step"].min()
        xmax_val = args.xmax if args.xmax is not None else sub_df["step"].max()
        ax.set_xlim(left=xmin_val, right=xmax_val)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        ax.set_title(base)
        ax.set_xlabel("Cumulative budget" if args.budget else "Number of observations")
        ax.set_ylabel(ylabel)

        sns.move_legend(ax, "best", title="cost_scale", framealpha=0.9, edgecolor="0.7")

    plt.tight_layout(pad=0.5)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.08)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
