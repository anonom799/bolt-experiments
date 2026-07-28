"""Generate a LaTeX table of paired deltas vs LogNEI baseline for PO problems.

For each enhancement, delta[seed] = method_final[seed] - baseline_final[seed]
is computed per trial (paired by seed), then mean ± std reported.
This is valid because all methods share seeds 0-4.

Enhancements shown (all relative to LogNEI):
  1. +q=5        : LogNEI(q=5)
  2. +TuRBO      : dTuRBO+LogNEI
  3. +TuRBO+q=5  : dTuRBO+LogNEI(q=5)
  4. +BAxUS      : dBAxUS+LogNEI
  5. +BAxUS+q=5  : dBAxUS+LogNEI(q=5)
"""


import numpy as np
import pandas as pd

from bolt_exp import REPO_ROOT

CSV_PATH = REPO_ROOT / "results" / "po_final_regret.csv"
OUT_TEX = REPO_ROOT / "tables" / "po_delta_table.tex"
METRIC = "log_simple_regret_all"
PROBLEMS = ["po128", "po256", "po512", "po768"]

ENHANCEMENTS = [
    (r"\texttt{q=5}",       "LogNEI(q=5)"),
    (r"\texttt{TuRBO}",     "dTuRBO+LogNEI"),
    (r"\texttt{TuRBO+q=5}", "dTuRBO+LogNEI(q=5)"),
    (r"\texttt{BAxUS}",     "dBAxUS+LogNEI"),
    (r"\texttt{BAxUS+q=5}", "dBAxUS+LogNEI(q=5)"),
]

BASELINE_METHOD = "LogNEI"


def get_trials(df: pd.DataFrame, problem: str, method: str) -> pd.Series | None:
    """Return seed-indexed Series of final metric values, or None if not found."""
    sub = df[(df["problem"] == problem) & (df["method"] == method)]
    if sub.empty:
        return None
    return sub.set_index("seed")[METRIC]


def fmt(mean: float, std: float) -> str:
    return rf"${mean:+.2f} \pm {std:.2f}$"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    col_header = " & ".join([r"\textbf{Enhancement}"] + [rf"\textbf{{{p.upper()}}}" for p in PROBLEMS])
    rows_tex = []

    for enh_label, enh_method in ENHANCEMENTS:
        cells = [enh_label]
        for problem in PROBLEMS:
            base = get_trials(df, problem, BASELINE_METHOD)
            enh = get_trials(df, problem, enh_method)

            if base is None or enh is None:
                missing = "baseline" if base is None else enh_method
                print(f"  Missing {missing} for {problem} / {enh_label}")
                cells.append("--")
                continue

            shared = sorted(set(base.index) & set(enh.index))
            deltas = np.array([enh[s] - base[s] for s in shared])
            enh_vals = enh[shared]
            FLOOR = -18.420680743952367
            all_floored = np.allclose(enh_vals, FLOOR)
            cell = fmt(deltas.mean(), deltas.std(ddof=1))
            if all_floored:
                cell += r"$^\dagger$"
            cells.append(cell)

        rows_tex.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Paired delta in \texttt{log\_simple\_regret\_all} at iteration 200 vs LogNEI baseline "
        r"(mean $\pm$ std over 5 seeds). Negative = improvement (lower log regret).}",
        r"\label{tab:po_deltas}",
        r"\begin{tabular}{l" + "r" * len(PROBLEMS) + "}",
        r"\toprule",
        col_header + r" \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"{\footnotesize $^\dagger$ All 5 seeds hit the log-regret floor ($\approx -18.42$); "
        r"deltas coincide with those of \texttt{TuRBO+q=5} as both methods fully floor out.}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    OUT_TEX.write_text(tex)
    print(tex)
    print(f"\nSaved to {OUT_TEX}")


if __name__ == "__main__":
    main()
