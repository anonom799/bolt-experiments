"""Generate LaTeX wall-clock-time tables.

Table 1: HPO + DMO side-by-side subtables.
Table 2: PO — methods as rows, PO-128/256/512/768 as columns.
Table 3: PCO — methods as rows, PCO-16/32/64 as columns.

File selection and method ordering mirror plot_figs.sh / plot_configs/.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml

from bolt_exp.plot_results import _method_label

from bolt_exp import REPO_ROOT, load_result, result_files

CONFIG_DIR = REPO_ROOT / "plot_configs"


# ── file helpers ──────────────────────────────────────────────────────────────

def _load_files(pattern: str, excludes: list[str]) -> list[Path]:
    files = result_files(pattern)  # matches the gzipped PO copies too
    for ex in excludes:
        files = [f for f in files if not re.search(ex, f.name)]
    return files


def _file_stats(path: Path) -> tuple[str, float, int]:
    d = load_result(path)
    iters = d["iterations"]
    per_step = [t["time_seconds"] / iters for t in d["trials"]]
    return _method_label(d), float(np.mean(per_step)), len(d["trials"])


def _config_order(config_name: str) -> list[str]:
    """Return method labels in the order they appear in a plot config."""
    path = CONFIG_DIR / config_name
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return [m["label"] for m in cfg.get("methods", [])]


def _sort_methods(
    method_times: dict[str, float], config_name: str
) -> list[tuple[str, float]]:
    order = _config_order(config_name)
    ordered = [(m, method_times[m]) for m in order if m in method_times]
    rest = sorted(
        [(m, t) for m, t in method_times.items() if m not in order],
        key=lambda x: x[0],
    )
    return ordered + rest


# ── problem definitions ───────────────────────────────────────────────────────

# (problem_display, glob_pattern, excludes, config_yaml)
HPO_PROBS = [
    (
        "HPO",
        "results/hpo/hpo_*_200iterations*.json",
        [r"_fd_"],
        "hpo_okabe_ito.yaml",
    ),
    (
        "HPO-FD-Step",
        "results/hpo/hpo_fd_step_*_200iterations*.json",
        [r"mfgibbon_5trials_200iterations_cost0\.005_", r"beta", r"_ucb_5trials_200iterations_cost0\.0[15]_"],
        "hpo_step_okabe_ito.yaml",
    ),
    (
        "HPO-FD-Model",
        "results/hpo/hpo_fd_model_*_200iterations*.json",
        [r"beta"],
        "hpo_model_okabe_ito.yaml",
    ),
]

DMO_PROBS = [
    (
        "DM",
        "results/dm/*200iterations*.json",
        [r"_mo_", r"ucb_beta[015]\.[0-9]", r"ucb_beta[13]0\.0", r"hetero"],
        "dm_okabe_ito.yaml",
    ),
    (
        "DM-MO",
        "results/dm/*_mo_*_200iterations*.json",
        [r"hetero"],
        "dm_mo_okabe_ito.yaml",
    ),
    (
        "DM-Het",
        "results/dm/dm_curriculum_hetero*_200iterations*.json",
        [r"_em[123]"],
        "dm_het_okabe_ito.yaml",
    ),
]

# (prob_display, glob_pattern, excludes, config_yaml)
PO_PROBS = [
    ("PO-128", "results/po/po128_*_200iterations*.json", [r"mlesi"], "po128_okabe_ito.yaml"),
    ("PO-256", "results/po/po256_*_200iterations*.json", [r"mlesi"], "po256_okabe_ito.yaml"),
    ("PO-512", "results/po/po512_*_200iterations*.json", [r"mlesi"], "po512_okabe_ito.yaml"),
    ("PO-768", "results/po/po768_*_200iterations*.json", [r"mlesi"], "po768_okabe_ito.yaml"),
]

PCO_PROBS = [
    ("PCO-16", "results/pco16/pco16_*_10trials_100iterations*.json", [], "pco16_okabe_ito.yaml"),
    ("PCO-32", "results/pco32/pco32_*_10trials_50iterations*.json", [], "pco32_okabe_ito.yaml"),
    ("PCO-64", "results/pco64/pco64_*_10trials_100iterations*.json", [], "pco64_okabe_ito.yaml"),
]


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt(mean: float) -> str:
    if mean >= 60:
        return rf"{mean / 60:.3g}\,min"
    return rf"{mean:.3g}\,s"


def _escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ── subtable builder (for HPO / DMO) ─────────────────────────────────────────

def _collect_prob_data(
    probs: list[tuple],
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Return [(prob_display, [(method, mean), ...]), ...]."""
    result = []
    for prob_display, pattern, excludes, config in probs:
        files = _load_files(pattern, excludes)
        if not files:
            print(f"  WARNING: no files for {prob_display}")
            continue
        method_times: dict[str, float] = {}
        method_trials: dict[str, int] = {}
        for f in files:
            method, mean, n_trials = _file_stats(f)
            method_times[method] = mean
            method_trials[method] = n_trials
        rows = _sort_methods(method_times, config)
        print(f"  [{prob_display}]")
        for method, _ in rows:
            print(f"    {method}: {method_trials[method]} trials")
        result.append((prob_display, rows, method_trials))
    return result


def _build_subtable(title: str, probs: list[tuple]) -> list[str]:
    prob_data = _collect_prob_data(probs)

    lines = [
        r"  \begin{subtable}[t]{0.48\textwidth}",
        r"    \centering",
        r"    \footnotesize",
        rf"    \caption{{\textbf{{{title}}}}}",
        r"    \begin{tabular}{lr}",
        r"      \toprule",
        r"      \textbf{Method} & \textbf{Time\,/\,step} \\",
        r"      \midrule",
    ]

    first_prob = True
    for prob_display, rows, _trials in prob_data:
        if not first_prob:
            lines.append(r"      \midrule")
        first_prob = False
        lines.append(
            rf"      \multicolumn{{2}}{{l}}{{\textit{{{prob_display}}}}} \\"
        )
        lines.append(r"      \cmidrule{1-2}")
        for method, mean in rows:
            lines.append(rf"      {_escape(method)} & {_fmt(mean)} \\")

    lines += [
        r"      \bottomrule",
        r"    \end{tabular}",
        r"  \end{subtable}",
    ]
    return lines


# ── column table builder (for PO / PCO) ──────────────────────────────────────

def _build_column_table(probs: list[tuple], caption: str, label: str) -> list[str]:
    # Collect per-problem method->mean dicts
    col_data: list[tuple[str, dict[str, float]]] = []
    for prob_display, pattern, excludes, config in probs:
        files = _load_files(pattern, excludes)
        if not files:
            print(f"  WARNING: no files for {prob_display}")
            col_data.append((prob_display, {}))
            continue
        method_times: dict[str, float] = {}
        method_trials: dict[str, int] = {}
        for f in files:
            method, mean, n_trials = _file_stats(f)
            method_times[method] = mean
            method_trials[method] = n_trials
        print(f"  [{prob_display}]")
        for method in method_times:
            print(f"    {method}: {method_trials[method]} trials")
        col_data.append((prob_display, method_times))

    # Build unified method order from the first config, then later ones for extras
    all_methods: set[str] = set()
    for _, mt in col_data:
        all_methods.update(mt.keys())

    combined_order: list[str] = []
    for *_, config in probs:
        combined_order += [m for m in _config_order(config) if m not in combined_order]
    ordered_methods = [m for m in combined_order if m in all_methods]
    ordered_methods += sorted(all_methods - set(combined_order))

    col_headers = [d for d, _ in col_data]
    n_cols = len(col_headers)
    col_spec = "l" + "r" * n_cols

    header_cells = " & ".join(
        rf"\textbf{{\texttt{{{h}}}}}" for h in col_headers
    )

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    \textbf{{Method}} & {header_cells} \\",
        r"    \midrule",
    ]

    for method in ordered_methods:
        cells = [_escape(method)]
        for _, mt in col_data:
            cells.append(_fmt(mt[method]) if method in mt else "---")
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return lines


# ── top-level builders ────────────────────────────────────────────────────────

def build_hpo_dmo_table() -> list[str]:
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Mean wall-clock time per BO step --- HPO and data mixture problems.}",
        r"  \label{tab:wall_clock_hpo_dmo}",
    ]
    lines += _build_subtable("HPO", HPO_PROBS)
    lines.append(r"  \hfill")
    lines += _build_subtable("DMO", DMO_PROBS)
    lines.append(r"\end{table}")
    return lines


def build_po_table() -> list[str]:
    return _build_column_table(
        PO_PROBS,
        "Mean wall-clock time per BO step --- prompt optimisation problems.",
        "tab:wall_clock_po",
    )


def build_pco_table() -> list[str]:
    return _build_column_table(
        PCO_PROBS,
        "Mean wall-clock time per BO step --- parallelism configuration problems.",
        "tab:wall_clock_pco",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "tables",
        help="Output directory (default: tables/, where the paper's copies live)",
    )
    parser.add_argument(
        "--tables",
        default="hpo_dmo,po,pco",
        help="Comma-separated subset of tables to build (hpo_dmo, po, pco)",
    )
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)

    builders = {
        "hpo_dmo": build_hpo_dmo_table,
        "po": build_po_table,
        "pco": build_pco_table,
    }
    for name in args.tables.split(","):
        tex = "\n".join(builders[name]())
        out = args.out_dir / f"wall_clock_{name}.tex"
        out.write_text(tex + "\n")
        print(tex)
        print(f"\nSaved to {out}\n")


if __name__ == "__main__":
    main()
