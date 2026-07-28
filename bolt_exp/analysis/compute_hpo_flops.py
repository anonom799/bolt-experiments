"""Estimate the training FLOPs behind every HPO benchmark query, then rewrite the
results with FLOPs as the cost axis.

The FLOPs formula and its justification live in `hpo_flops.py`. This script
applies it to the recorded BO results and writes, in order:

1. `flops/hpo_flops_per_query.csv`   -- one row per query, with the decoded
   hyperparameters, the per-query FLOPs and the running total.
2. `flops/hpo_flops_summary.csv`     -- per result file: n_queries, mean/min/max
   FLOPs per query, total FLOPs. Includes a `dmo_reference` row.
3. `results_flops/hpo/*.json`        -- copies of the input results whose
   `budget_all` is the cumulative FLOPs (in units of `--units`), so the existing
   `plot_results.py --budget` can plot them unchanged.

The input result JSONs are opened read-only and never modified.

Usage:
    python compute_hpo_flops.py results/hpo/hpo_*_200iterations*.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bolt_exp.hpo_flops import (
    DMO_ANCHOR_FLOPS,
    DMO_TOKENS,
    LORA_LAYERS_ALL,
    QWEN3_4B,
    SEQ_LEN,
    decode,
    flops,
    run_flops,
)
from bolt_exp.plot_results import _method_label

from bolt_exp import REPO_ROOT, load_result

# Categorical column 6, in the order used to one-hot encode it (params.yaml).
LORA_TARGET_MODULES = [
    "q_proj,v_proj",
    "q_proj,v_proj,k_proj,o_proj",
    "gate_proj,up_proj,down_proj",
    "all-linear",
]


def infer_variant(path: Path, results: dict) -> str:
    """Which HPO problem a result file came from."""
    problem = results.get("problem", "")
    for name in ("hpo_fd_step", "hpo_fd_model"):
        if problem == name or name in path.stem:
            return name
    return "hpo"


def n_skip_for(results: dict) -> int:
    """Leading candidates excluded from the cost axis (the initial design)."""
    method = results.get("method", results.get("acq_fn", ""))
    if method.lower() == "asha":
        return 1
    return results.get("initial_random_samples", 0)


def describe_queries(X: torch.Tensor, variant: str) -> pd.DataFrame:
    """Decode raw X rows into a readable table of hyperparameters plus FLOPs."""
    decoded = decode(X, variant)
    lora_layers = decoded["lora_layers"].numpy()
    df = pd.DataFrame(
        {
            "learning_rate": 1e-6 + X[:, 0].numpy() * (1e-3 - 1e-6),
            "batch_size": 2 ** X[:, 1].round().numpy().astype(int),
            "lora_r": 2 ** X[:, 2].round().numpy().astype(int),
            "lora_alpha": 2 ** X[:, 3].round().numpy().astype(int),
            "lora_dropout": X[:, 4].numpy() * 0.1,
            "lora_layers": [
                "all" if v == LORA_LAYERS_ALL else f"last_{int(v)}" for v in lora_layers
            ],
            "lora_target_modules": [
                LORA_TARGET_MODULES[int(i)] for i in X[:, 6].round().numpy()
            ],
            "model": decoded["model_name"],
            "tokens": decoded["tokens"].numpy(),
            "trainable_layers": decoded["trainable_layers"].numpy().astype(int),
            "flops": flops(X, variant).numpy(),
        }
    )
    return df


def process_file(path: Path, units: float) -> tuple[pd.DataFrame, dict]:
    """Return the per-query table for one result file and its FLOPs-costed copy."""
    results = load_result(path)

    variant = infer_variant(path, results)
    label = _method_label(results)
    n_skip = n_skip_for(results)

    frames = []
    for trial in results["trials"]:
        X = torch.tensor(trial["candidates"], dtype=torch.double)
        df = describe_queries(X, variant)
        df.insert(0, "query", np.arange(len(df)))
        df.insert(0, "trial", trial["trial"])
        df.insert(0, "variant", variant)
        df.insert(0, "method", label)
        df.insert(0, "file", path.name)

        # The cost axis starts after the initial design.
        counted = df["flops"].to_numpy().copy()
        counted[:n_skip] = 0.0
        df["cum_flops"] = np.cumsum(counted)
        frames.append(df)

        trial["budget_all"] = (np.cumsum(df["flops"].to_numpy()[n_skip:]) / units).tolist()

    return pd.concat(frames, ignore_index=True), results


def summarise(per_query: pd.DataFrame) -> pd.DataFrame:
    """Per-file FLOPs summary, with the data mixture reference appended."""
    grouped = per_query.groupby(["file", "method", "variant"], sort=False)
    summary = grouped["flops"].agg(["count", "mean", "min", "max"]).reset_index()
    summary = summary.rename(
        columns={
            "count": "n_queries",
            "mean": "mean_flops_per_query",
            "min": "min_flops_per_query",
            "max": "max_flops_per_query",
        }
    )
    # Per-trial totals, averaged over trials, so these are per-run compute budgets.
    # "total" counts every query; "plotted" excludes the initial design, matching
    # the cost axis of the figures.
    summary["total_flops_per_trial"] = grouped.apply(
        lambda g: g.groupby("trial")["flops"].sum().mean(), include_groups=False
    ).to_numpy()
    summary["plotted_flops_per_trial"] = grouped.apply(
        lambda g: g.groupby("trial")["cum_flops"].max().mean(), include_groups=False
    ).to_numpy()

    dmo = run_flops(QWEN3_4B, DMO_TOKENS, QWEN3_4B["num_hidden_layers"]).item()
    reference = pd.DataFrame(
        [
            {
                "file": "dmo_reference",
                "method": "data mixture (Qwen3-4B, all layers, 1e7 tokens)",
                "variant": "dm_curriculum",
                "n_queries": 1,
                "mean_flops_per_query": dmo,
                "min_flops_per_query": dmo,
                "max_flops_per_query": dmo,
                "total_flops_per_trial": DMO_ANCHOR_FLOPS,
                "plotted_flops_per_trial": DMO_ANCHOR_FLOPS,
            }
        ]
    )
    return pd.concat([summary, reference], ignore_index=True)


def common_xmax(paths: list[Path], units: float) -> float:
    """Largest compute budget every trial of every given result file reaches.

    Methods spend different amounts of compute per query, so they run out at
    different x. Beyond this point a curve is backed by only some of the trials,
    so it is where a like-for-like comparison stops.
    """
    smallest = np.inf
    for path in paths:
        results = load_result(path)
        variant = infer_variant(path, results)
        n_skip = n_skip_for(results)
        for trial in results["trials"]:
            X = torch.tensor(trial["candidates"], dtype=torch.double)
            total = flops(X, variant).numpy()[n_skip:].sum()
            smallest = min(smallest, total / units)
    return smallest


def main(args):
    if args.xmax_only:
        print(f"{common_xmax([Path(f) for f in args.files], args.units):.0f}")
        return

    out_dir = Path(args.out_dir)
    flops_dir = out_dir / "flops"
    results_dir = out_dir / "results_flops" / "hpo"
    flops_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in args.files:
        path = Path(name)
        per_query, costed = process_file(path, args.units)
        frames.append(per_query)
        with open(results_dir / path.name, "w") as f:
            json.dump(costed, f, indent=4)
        print(
            f"{path.name}: {per_query['flops'].mean():.3g} FLOPs/query "
            f"(min {per_query['flops'].min():.3g}, max {per_query['flops'].max():.3g})"
        )

    per_query = pd.concat(frames, ignore_index=True)
    per_query.to_csv(flops_dir / "hpo_flops_per_query.csv", index=False)

    summary = summarise(per_query)
    summary.to_csv(flops_dir / "hpo_flops_summary.csv", index=False)

    print(f"\nseq_len={SEQ_LEN}, budget_all written in units of {args.units:g} FLOPs")
    print(f"Wrote {flops_dir / 'hpo_flops_per_query.csv'} ({len(per_query)} rows)")
    print(f"Wrote {flops_dir / 'hpo_flops_summary.csv'} ({len(summary)} rows)")
    print(f"Wrote {len(args.files)} FLOPs-costed result file(s) to {results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate per-query training FLOPs for HPO results and "
        "write FLOPs-costed copies of them."
    )
    parser.add_argument("files", nargs="+", help="HPO result JSON files")
    parser.add_argument(
        "--out_dir",
        default=str(REPO_ROOT),
        help="Directory to write flops/ and results_flops/ into",
    )
    parser.add_argument(
        "--units",
        type=float,
        default=1e17,
        help="FLOPs per unit of budget_all in the costed result copies",
    )
    parser.add_argument(
        "--xmax_only",
        action="store_true",
        help="Print the compute budget every trial reaches (in --units) and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
