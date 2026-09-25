"""Experiment, analysis and plotting code for the `bolt` benchmark paper.

`REPO_ROOT` is the anchor every module uses to locate `results/`,
`plot_configs/` and `tables/`, so scripts behave the same however
deep in the package they live and whatever the working directory is.
"""

import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_result(path: str | Path) -> dict:
    """Load a result JSON, transparently handling the gzipped PO copies.

    The large PO results ship as `*.json.gz`. A `.json` path whose file is
    absent falls back to the `.json.gz` beside it, so callers work on a fresh
    clone whether or not it has been decompressed.
    """
    path = Path(path)
    if not path.exists() and path.suffix != ".gz":
        gz = path.with_name(path.name + ".gz")
        if gz.exists():
            path = gz
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def dedupe_results(paths) -> list[Path]:
    """Drop `*.json.gz` entries whose decompressed `*.json` is also listed.

    `gunzip -k` leaves both copies in place, and a shell glob of `*.json*` then
    yields the same run twice, which would double its trial count.
    """
    seen: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        key = p.name[:-3] if p.name.endswith(".gz") else p.name
        if key not in seen or seen[key].name.endswith(".gz"):
            seen[key] = p
    return list(seen.values())


def result_files(pattern: str, root: Path | None = None) -> list[Path]:
    """Glob result files, matching both plain and gzipped copies."""
    root = Path(".") if root is None else Path(root)
    matches = list(root.glob(pattern)) + list(root.glob(pattern + ".gz"))
    return sorted(dedupe_results(matches), key=lambda p: p.name)


__all__ = ["REPO_ROOT", "load_result", "dedupe_results", "result_files"]
