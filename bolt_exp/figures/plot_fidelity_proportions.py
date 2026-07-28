"""Plot fidelity value distributions from multi-fidelity HPO result JSONs.

For discrete fidelity (hpo_fd_model): stacked bar of proportion per fidelity
level, split by iteration window, plus an overall summary bar.

For continuous fidelity (hpo_fd_step): KDE of fidelity values and a rolling
mean ± std of fidelity vs query index.

Fidelity is assumed to be the last column of each candidate vector.

Usage:
    python plot_fidelity_proportions.py hpo_fd_step_*.json --config plot_configs/hpo_step_okabe_ito.yaml --out fid_step.png
    python plot_fidelity_proportions.py hpo_fd_model_*.json --config plot_configs/hpo_model_okabe_ito.yaml --out fid_model.png
    python plot_fidelity_proportions.py hpo_fd_*.json --out fid_all.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from bolt_exp import REPO_ROOT, dedupe_results, load_result


LINESTYLE_ALIASES = {
    "solid": (1, 0),
    "dashed": (4, 2),
    "dotted": (1, 2),
    "dashdot": (4, 2, 1, 2),
}

# ── config helpers ────────────────────────────────────────────────────────────

def _load_config(path: Path | None) -> tuple[list[str], dict, dict]:
    """Return (order, color_map, dashes_map) from a YAML plot config."""
    if path is None:
        return [], {}, {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    entries = raw.get("methods", raw) if isinstance(raw, dict) else raw
    order = [e["label"] for e in entries]
    color_map = {e["label"]: e["color"] for e in entries if "color" in e}
    dashes_map = {}
    for e in entries:
        if "linestyle" in e:
            ls = e["linestyle"]
            dashes_map[e["label"]] = LINESTYLE_ALIASES.get(ls.lower(), (1, 0)) if isinstance(ls, str) else tuple(ls)
    return order, color_map, dashes_map


def _sort_records(records: list[dict], cfg_order: list[str]) -> list[dict]:
    """Reorder records to match config order; unlisted labels go at the end."""
    order_map = {lab: i for i, lab in enumerate(cfg_order)}
    return sorted(records, key=lambda r: order_map.get(r["label"], len(cfg_order)))


def _style_maps(records: list[dict], cfg_colors: dict, cfg_dashes: dict) -> tuple[dict, dict]:
    """Build per-label color and linestyle maps, falling back to defaults."""
    labels = [r["label"] for r in records]
    palette = sns.color_palette("deep", max(len(labels), 1))
    color_map = {lab: cfg_colors.get(lab, palette[i % len(palette)]) for i, lab in enumerate(labels)}
    dashes_map = {lab: cfg_dashes.get(lab, (1, 0)) for lab in labels}
    return color_map, dashes_map


# ── data helpers ──────────────────────────────────────────────────────────────

def _method_label(d: dict) -> str:
    acq = d.get("acq_fn", d.get("method", "unknown"))
    label = acq.upper()
    
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
    return label


def _load_fidelities(path: Path) -> tuple[str, str, list[np.ndarray]]:
    """Return (label, problem, list-of-per-trial fidelity arrays)."""
    d = load_result(path)
    label = _method_label(d)
    problem = d.get("problem", "unknown")
    per_trial = []
    for t in d["trials"]:
        cands = t.get("candidates", [])
        if cands:
            fids = np.array([c[-1] for c in cands], dtype=float)
            per_trial.append(fids)
    return label, problem, per_trial


def _infer_type(problem: str, fids_flat: np.ndarray) -> str:
    if "model" in problem:
        return "discrete"
    if "step" in problem:
        return "continuous"
    return "discrete" if len(np.unique(fids_flat)) <= 5 else "continuous"


# ── plot helpers ──────────────────────────────────────────────────────────────

_SCATTER_KEYWORDS = {"bohb", "hyperband"}


def _is_scatter(label: str, extra: frozenset[str] = frozenset()) -> bool:
    return any(kw in label.lower() for kw in _SCATTER_KEYWORDS | extra)


def _subplot_grid(n: int, subplot_kw: dict | None = None) -> tuple[plt.Figure, np.ndarray]:
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3.2 * nrows),
        sharey=True,
        squeeze=False,
        **(subplot_kw or {}),
    )
    return fig, axes.flatten()


# ── plot functions ────────────────────────────────────────────────────────────

def plot_discrete(records: list[dict], out: Path, cfg_order: list[str], cfg_colors: dict, cfg_dashes: dict) -> None:
    """One subplot per method: stacked bars (proportion per fidelity level × window).
    Scatter methods (BOHB, ASHA, …) show raw fidelity values vs query index instead."""
    records = _sort_records(records, cfg_order)
    all_fids = sorted({v for r in records for v in np.unique(r["fids_flat"])})
    color_map, _ = _style_maps(records, cfg_colors, cfg_dashes)
    _OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    _HATCHES   = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo", "***"]
    bar_colors = _OKABE_ITO[: len(all_fids)]
    bar_hatches = _HATCHES[: len(all_fids)]
    fid_color = dict(zip(all_fids, bar_colors))

    n = len(records)
    n_windows = 5
    fig, axes = _subplot_grid(n)

    for ax, r in zip(axes, records):
        label = r["label"]
        color = color_map[label]

        if _is_scatter(label):
            # raw scatter: x normalised to 0–100 so axis matches other subplots
            rng = np.random.default_rng(0)
            for trial_fids in r["per_trial"]:
                nt = len(trial_fids)
                xs = np.arange(nt) * 100 / max(nt, 1)
                jitter = rng.uniform(-0.06, 0.06, size=nt)
                ax.scatter(xs, trial_fids + jitter, color=color, alpha=0.35, s=8, linewidths=0)
            ax.set_xlim(0, 100)
            ax.set_yticks(all_fids)
            ax.set_yticklabels([str(int(f)) for f in all_fids])
        else:
            # stacked bar: proportion at each fidelity per window, averaged over trials
            rows = []
            for trial_fids in r["per_trial"]:
                nt = len(trial_fids)
                window_ids = np.arange(nt) * n_windows // max(nt, 1)
                for w, f in zip(window_ids, trial_fids):
                    rows.append({"window": w, "fidelity": f})
            df = pd.DataFrame(rows)

            bottom = np.zeros(n_windows)
            for fid, bcolor, hatch in zip(all_fids, bar_colors, bar_hatches):
                props = [
                    (df[df["window"] == w]["fidelity"] == fid).mean() if (df["window"] == w).any() else 0.0
                    for w in range(n_windows)
                ]
                ax.bar(range(n_windows), props, bottom=bottom, color=bcolor,
                       hatch=hatch, edgecolor="white", label=f"fid={int(fid)}", width=0.7)
                bottom += np.array(props)
            ax.set_xticks(range(n_windows))
            ax.set_xticklabels([f"{w * 100 // n_windows}%" for w in range(n_windows)])
            ax.set_ylim(0, 1)

        ax.set_xlabel("Query progress (%)")
        ax.set_ylabel("Proportion of queries")
        ax.set_title(label)

    # shared fidelity legend from bar plots (attach to figure)
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h, edgecolor="white", label=f"fid={int(f)}")
        for f, c, h in zip(all_fids, bar_colors, bar_hatches)
    ]
    fig.legend(handles=bar_handles, loc="lower center", ncol=len(all_fids),
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    for ax in axes[n:]:
        ax.set_visible(False)

    sns.despine()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def plot_continuous(records: list[dict], out: Path, cfg_order: list[str], cfg_colors: dict, cfg_dashes: dict) -> None:
    """One subplot per method: mean ± sd fidelity over query-progress windows.
    Scatter methods (BOHB, ASHA, …) show raw fidelity values vs query index instead."""
    records = _sort_records(records, cfg_order)
    color_map, _ = _style_maps(records, cfg_colors, cfg_dashes)

    n = len(records)
    n_windows = 10
    fig, axes = _subplot_grid(n)

    for ax, r in zip(axes, records):
        label = r["label"]
        color = color_map[label]

        for trial_fids in r["per_trial"]:
            nt = len(trial_fids)
            xs = np.arange(nt) * 100 / max(nt, 1)
            ax.scatter(xs, trial_fids, color=color, alpha=0.3, s=8, linewidths=0)
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])

        ax.set_xlabel("Query progress (%)")
        ax.set_title(label)
        ax.set_ylabel("Fidelity")
        ax.set_ylim(-0.05, 1.05)

    for ax in axes[n:]:
        ax.set_visible(False)

    sns.despine()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "fidelity_proportions.png")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML plot config (colors, linestyles) — same format as plot_results.py")
    parser.add_argument(
        "--type",
        choices=["discrete", "continuous", "auto"],
        default="auto",
        help="Fidelity type (default: auto-detect from problem name / values)",
    )
    args = parser.parse_args()

    cfg_order, cfg_colors, cfg_dashes = _load_config(args.config)

    records = []
    for path in dedupe_results(args.files):
        label, problem, per_trial = _load_fidelities(path)
        if not per_trial:
            print(f"  Skipping {path.name}: no candidates found")
            continue
        fids_flat = np.concatenate(per_trial)
        ftype = args.type if args.type != "auto" else _infer_type(problem, fids_flat)
        records.append(
            {"label": label, "problem": problem, "per_trial": per_trial, "fids_flat": fids_flat, "type": ftype}
        )
        print(f"  {path.name}: label={label}, problem={problem}, type={ftype}, "
              f"n_queries={len(fids_flat)}, n_queries_hf={sum(f == 1 for f in fids_flat)}")

    if not records:
        print("No valid result files found.")
        return

    disc = [r for r in records if r["type"] == "discrete"]
    cont = [r for r in records if r["type"] == "continuous"]

    if disc and cont:
        stem, suffix, parent = args.out.stem, args.out.suffix, args.out.parent
        plot_discrete(disc, parent / f"{stem}_discrete{suffix}", cfg_order, cfg_colors, cfg_dashes)
        plot_continuous(cont, parent / f"{stem}_continuous{suffix}", cfg_order, cfg_colors, cfg_dashes)
    elif disc:
        plot_discrete(disc, args.out, cfg_order, cfg_colors, cfg_dashes)
    else:
        plot_continuous(cont, args.out, cfg_order, cfg_colors, cfg_dashes)


if __name__ == "__main__":
    main()
