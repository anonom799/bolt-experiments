"""Print the x-axis limit for the HPO training-FLOPs figures.

Methods spend different amounts of compute per query, so they run out at
different x. This prints the largest compute budget (in `--units` FLOPs) that
every trial of every given result file reaches -- beyond it a curve is backed by
only some of the trials, so it is where a like-for-like comparison stops.

The FLOPs formula lives in `hpo_flops.py`; `plot_results.py --flops` applies it
to draw the curves.

Usage:
    python -m bolt_exp.analysis.compute_hpo_flops results/hpo/hpo_*_200iterations*.json
"""

import argparse
from pathlib import Path

import numpy as np

from bolt_exp import load_result
from bolt_exp.plot_results import flops_budget, infer_variant, n_skip_for


def common_xmax(paths: list[Path], units: float) -> float:
    """Largest compute budget every trial of every given result file reaches."""
    smallest = np.inf
    for path in paths:
        results = load_result(path)
        variant = infer_variant(path, results)
        n_skip = n_skip_for(results)
        for trial in results["trials"]:
            smallest = min(smallest, flops_budget(trial, variant, n_skip, units)[-1])
    return smallest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the compute budget every trial of the given HPO "
        "results reaches, for the --xmax of the FLOPs figures."
    )
    parser.add_argument("files", nargs="+", help="HPO result JSON files")
    parser.add_argument(
        "--units",
        type=float,
        default=1e17,
        help="FLOPs per unit of the printed budget",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"{common_xmax([Path(f) for f in args.files], args.units):.0f}")
