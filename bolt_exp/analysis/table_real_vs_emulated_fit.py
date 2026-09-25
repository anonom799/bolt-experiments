"""Emulator fit on the points the real (LLM-in-the-loop) BBO runs visited.

For every point a real run evaluated, compare the observed real score against
the noise-free emulator prediction at the same x, and report R^2
(1 - SS_res / SS_tot, emulator as the predictor of the real score), Spearman's
rho, bias, and RMSE, with i.i.d. bootstrap 95% CIs on R^2 and rho.

Families and their real result files come from the same methods configs that
plot_real_vs_emulated.py uses (plot_configs/*_real_vs_emulated.yaml, `real`
source only), so the method set stays in sync with the real-vs-emulated plots.
Points are truncated to the initial design plus `n_iterations` BO iterations,
methods in `--exclude` are dropped, and repeated candidates are kept once (the
first occurrence; mostly the shared initial design, which every method on a seed
replays).

DMO is also broken down per objective. The results JSONs only store the
3-task mean, so the raw IFEval / MATH-500 / MBPP+ scores come from
results/dm_real/all_trials.parquet (the real-run tracker
output), joined by position and checked against `candidates` / `seen_y`.

Emulators are evaluated through the installed `bolt`, whose `hf_revision` pins
select the emulator version; `--emulator-revision` asserts that pin so the table
can't silently come from an unpinned copy.

Usage:
    python -m bolt_exp.analysis.table_real_vs_emulated_fit
    python -m bolt_exp.analysis.table_real_vs_emulated_fit --exclude  # keep every method
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

import bolt
from bolt import HPO, DMCurriculum, DMCurriculumMO
from bolt_exp import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "plot_configs"
DM_TRIALS = REPO_ROOT / "results" / "dm_real" / "all_trials.parquet"
DM_SCORE_COLS = ["score_if", "score_math", "score_code"]  # DMCurriculumMO output order

# (key, label, methods config, emulator class, objective labels)
FAMILIES = [
    ("dm", "DMO", CONFIG_DIR / "dm_real_vs_emulated.yaml", DMCurriculumMO,
     ["IFEval", "MATH-500", "MBPP+"]),
    ("hpo", "HPO", CONFIG_DIR / "hpo_real_vs_emulated.yaml", HPO, ["MATH-500"]),
]


# ── loading ───────────────────────────────────────────────────────────────────

def _dm_scores(trials: pd.DataFrame, method: str, seed: int, n: int) -> pd.DataFrame:
    """The n parquet rows for one trial, in the order the tracker exported it:
    the seed's shared initial design, then the method's own chain."""
    ok = trials[trials["status"] == "ok"]
    seq = pd.concat([
        ok[(ok["method"] == "init") & (ok["seed"] == seed)].sort_values("iteration"),
        ok[(ok["method"] == method) & (ok["seed"] == seed)].sort_values("iteration"),
    ])
    if len(seq) < n:
        raise ValueError(f"{DM_TRIALS} has {len(seq)} rows for {method} seed {seed}, need {n}")
    return seq.iloc[:n]


def load_real(key: str, methods_config: dict, exclude: set[str]) -> tuple:
    """Deduplicated (X, Y, methods) for one family. Y is (N, n_objectives)."""
    real_dir = Path(methods_config["sources"]["real"])
    trials = pd.read_parquet(DM_TRIALS) if key == "dm" else None

    X, Y, used, seen = [], [], [], set()
    for entry in methods_config["methods"]:
        result = json.loads((real_dir / entry["files"]["real"]).read_text())
        method = result["acq_fn"]
        if method in exclude:
            continue
        used.append(entry["label"])
        n = result["initial_random_samples"] + methods_config["n_iterations"]

        for i, trial in enumerate(result["trials"]):
            cand = np.asarray(trial["candidates"][:n], dtype=float)
            y = np.asarray(trial["seen_y"][:n], dtype=float)
            if len(cand) != n or len(y) != n:
                raise ValueError(f"{entry['files']['real']} trial {i} has fewer than {n} points")

            if trials is None:
                ys = y[:, None]
            else:
                seq = _dm_scores(trials, method, trial.get("seed", i), n)
                ys = seq[DM_SCORE_COLS].to_numpy(float)
                xs = seq[[f"x{j}" for j in range(cand.shape[1])]].to_numpy(float)
                if not (np.allclose(xs, cand) and np.allclose(seq["y"], y)
                        and np.allclose(ys.mean(axis=1), y)):
                    raise ValueError(f"{DM_TRIALS} out of sync with {entry['files']['real']} "
                                     f"(method {method}, seed {trial.get('seed', i)})")

            for x_row, y_row in zip(cand, ys):
                k = tuple(np.round(x_row, 8))
                if k not in seen:
                    seen.add(k)
                    X.append(x_row)
                    Y.append(y_row)

    return np.array(X), np.array(Y), used


# ── emulator ──────────────────────────────────────────────────────────────────

def load_emulator(cls, revision: str):
    if cls.hf_revision != revision:
        raise RuntimeError(f"{cls.__name__}.hf_revision is {cls.hf_revision!r}, expected "
                           f"{revision!r} (bolt imported from {bolt.__file__})")
    prob = cls(noise_std=None)

    # guard against a changed forward pass: the pinned emulator must reproduce
    # the single-objective optimum recorded in bolt (DMCurriculum for the MO
    # class, whose objective is the mean of the three outputs)
    ref = DMCurriculum if cls is DMCurriculumMO else cls
    x_opt = torch.tensor([ref._optimizers[0]], dtype=torch.double)
    with torch.no_grad():
        y_opt = prob.evaluate_true(x_opt).mean().item()
    if not np.isclose(y_opt, ref._optimal_value, atol=1e-4):
        raise RuntimeError(f"{cls.__name__} gives {y_opt:.5f} at the recorded optimizer, "
                           f"expected {ref._optimal_value}")

    # HF cache layout: .../snapshots/<commit>/model.safetensors
    commit = Path(prob.model_path).parent.name[:7]
    return prob, commit


def predict(prob, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = prob.evaluate_true(torch.tensor(X, dtype=torch.double)).numpy()
    return out.reshape(len(X), -1)


# ── metrics ───────────────────────────────────────────────────────────────────

def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_stats(y: np.ndarray, pred: np.ndarray, n_boot: int, rng) -> dict:
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boot.append((_r2(y[i], pred[i]), spearmanr(y[i], pred[i])[0]))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "n": len(y),
        "r2": _r2(y, pred), "r2_ci": (lo[0], hi[0]),
        "rho": spearmanr(y, pred)[0], "rho_ci": (lo[1], hi[1]),
        "bias": np.mean(pred - y),
        "rmse": np.sqrt(np.mean((pred - y) ** 2)),
        "real_sd": y.std(),
    }


# ── output ────────────────────────────────────────────────────────────────────

def to_markdown(rows: list[dict], provenance: list[str], settings: str, tex: str) -> str:
    lines = [
        "# Emulator fit on real BBO observations",
        "",
        "Generated by `python -m bolt_exp.analysis.table_real_vs_emulated_fit`; do not edit by hand.",
        "",
        settings,
        "",
        *[f"- {p}" for p in provenance],
        "",
        "| Family | Objective | n | R² [95% CI] | Spearman ρ [95% CI] | Bias (emu − real) | RMSE | Real SD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["stats"]
        lines.append(
            f"| {r['family']} | {r['objective']} | {s['n']} "
            f"| {s['r2']:.3f} [{s['r2_ci'][0]:.3f}, {s['r2_ci'][1]:.3f}] "
            f"| {s['rho']:.3f} [{s['rho_ci'][0]:.3f}, {s['rho_ci'][1]:.3f}] "
            f"| {s['bias']:+.4f} | {s['rmse']:.4f} | {s['real_sd']:.4f} |"
        )
    lines += [
        "",
        "CIs are i.i.d. bootstrap over points. BO points cluster within a chain, so",
        "these are narrower than a seed-level bootstrap would give.",
        "",
        "## LaTeX",
        "",
        "```latex",
        tex,
        "```",
        "",
    ]
    return "\n".join(lines)


def to_latex(rows: list[dict], methods_note: str) -> str:
    def ci(v, lo, hi):
        return rf"${v:.2f}$ {{\scriptsize$[{lo:.2f}, {hi:.2f}]$}}"

    body, prev = [], None
    for r in rows:
        s = r["stats"]
        if prev is not None and r["family"] != prev:
            body.append(r"\midrule")
        body.append(" & ".join([
            r["family"] if r["family"] != prev else "",
            r["objective"],
            str(s["n"]),
            ci(s["r2"], *s["r2_ci"]),
            ci(s["rho"], *s["rho_ci"]),
            f"${s['rmse']:.3f}$",
        ]) + r" \\")
        prev = r["family"]

    return "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Emulator fit on the points visited by the real BBO runs "
        rf"({methods_note}; repeated points counted once). "
        r"$R^2$ treats the noise-free emulator as the predictor of the observed real score; "
        r"brackets are 95\% bootstrap CIs.}",
        r"\label{tab:real_vs_emulated_fit}",
        r"\begin{tabular}{llrccr}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Objective} & $n$ & $R^2$ & Spearman $\rho$ & RMSE \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exclude", nargs="*", default=["random"],
                        help="acq_fn values to drop (default: random). Pass with no "
                             "values to keep every method.")
    parser.add_argument("--emulator-revision", default="v0.2.0",
                        help="Required hf_revision pin on the bolt problem classes.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "tables" / "real_vs_emulated_fit.md")
    parser.add_argument("--tex-out", type=Path, default=None,
                        help="Also write the LaTeX table on its own, for \\input{}.")
    args = parser.parse_args()

    exclude = set(args.exclude)
    rng = np.random.default_rng(args.seed)
    rows, provenance, all_used = [], [], []

    for key, label, config_path, cls, objectives in FAMILIES:
        methods_config = yaml.safe_load(config_path.read_text())
        X, Y, used = load_real(key, methods_config, exclude)
        prob, commit = load_emulator(cls, args.emulator_revision)
        pred = predict(prob, X)

        provenance.append(
            f"**{label}**: `{cls.__name__}` → `{cls.hf_repo}@{cls.hf_revision}` "
            f"(`{commit}`); real runs from `{methods_config['sources']['real']}` "
            f"({', '.join(used)})"
        )
        all_used.append(f"{label}: {', '.join(used)}")

        for j, obj in enumerate(objectives):
            rows.append({"family": label, "objective": obj,
                         "stats": fit_stats(Y[:, j], pred[:, j], args.n_boot, rng)})
        if len(objectives) > 1:
            rows.append({"family": label, "objective": "Mean",
                         "stats": fit_stats(Y.mean(axis=1), pred.mean(axis=1), args.n_boot, rng)})

    excl = ", ".join(sorted(exclude)) or "none"
    settings = (f"Points: initial design + first `n_iterations` BO iterations per trial, "
                f"deduplicated across methods; excluded methods: {excl}. "
                f"Bootstrap: {args.n_boot} resamples, seed {args.seed}.")
    methods_note = "; ".join(all_used).replace("_", r"\_")
    tex = to_latex(rows, methods_note)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(rows, provenance, settings, tex))
    print(args.out.read_text())
    print(f"Saved to {args.out}")

    if args.tex_out is not None:
        args.tex_out.write_text(tex + "\n")
        print(f"Saved to {args.tex_out}")


if __name__ == "__main__":
    main()
