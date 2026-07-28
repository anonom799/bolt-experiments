"""Collate final log_simple_regret_all for PO results that appear in the plot.

Matches the file selection and method labelling from plot_figs.sh / plot_results.py.
Files: po{128,256,512,768}_*_200iterations*.json (excluding mlesi)
Methods: only those listed (non-commented) in plot_configs/po128_okabe_ito.yaml
Metric: log_simple_regret_all (last value of each trial)
"""

import json
import re

import numpy as np
import pandas as pd

from bolt_exp import REPO_ROOT, dedupe_results, load_result

RESULTS_DIR = REPO_ROOT / "results" / "po"
OUT_CSV = REPO_ROOT / "results" / "po_final_regret.csv"
METRIC = "log_simple_regret_all"

PLOT_METHODS = {
    "RANDOM",
    "TS",
    "LogEI",
    "LogNEI",
    "LogNEI(q=5)",
    "UCB",
    "GIBBON",
    "dTuRBO+LogNEI",
    "dTuRBO+LogNEI(q=5)",
    "dBAxUS+LogNEI",
    "dBAxUS+LogNEI(q=5)",
}


def method_label(d: dict) -> str:
    acq = d.get("acq_fn", d.get("method", "unknown"))
    label = acq.upper()

    if acq.lower() == "ei":
        label = "LogEI"
    elif acq.lower() == "qnei":
        label = "LogNEI"

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

    return label


def main() -> None:
    # mirror the shell glob + grep -v mlesi from plot_figs.sh
    # `.json.gz` matches too: the large PO results ship compressed
    pattern = re.compile(r"^po(128|256|512|768)_.*_200iterations.*\.json(\.gz)?$")

    if not RESULTS_DIR.is_dir():
        raise SystemExit(
            f"No PO results at {RESULTS_DIR}. Run scripts/run_botorch_po.sh first."
        )

    rows = []
    candidates = [p for p in RESULTS_DIR.iterdir() if pattern.match(p.name)]
    for path in sorted(dedupe_results(candidates), key=lambda p: p.name):
        if "mlesi" in path.name:
            continue

        d = load_result(path)

        label = method_label(d)
        if label not in PLOT_METHODS:
            continue

        trials = d["trials"]
        meta = {
            "problem": d.get("problem", path.name.split("_")[0]),
            "method": label,
            "last_iteration": d.get("iterations", len(np.array(trials[0][METRIC])) - 1),
            "num_trials": len(trials),
            "batch_size": d.get("batch_size", 1),
            "acq_fn": d.get("acq_fn", ""),
            "turbo": bool(d.get("turbo")),
            "baxus": bool(d.get("baxus")),
        }
        for t in trials:
            rows.append({**meta, "seed": t["seed"], METRIC: np.array(t[METRIC])[-1]})

    df = pd.DataFrame(rows).sort_values(["problem", "method", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print(df.to_string())


if __name__ == "__main__":
    main()
