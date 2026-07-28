"""Experiment, analysis and plotting code for the `bolt` benchmark paper.

`REPO_ROOT` is the anchor every module uses to locate `results/`,
`plot_configs/`, `flops/` and `tables/`, so scripts behave the same however
deep in the package they live and whatever the working directory is.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["REPO_ROOT"]
