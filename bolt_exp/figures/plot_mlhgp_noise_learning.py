"""One figure: the MLHGP noise model becoming accurate as BO collects data.

Left  -- Spearman(learned, oracle sigma^2) vs BO iteration, 10 seeds, mean +/- 95% CI
         (reads data/mlhgp/mlhgp_noise_vs_bo_iter.json, from analysis.mlhgp_noise_vs_bo_iter).
Right -- learned vs oracle noise variance on the held-out test set at three budgets,
         pooled over seeds. A perfect model lies on y = x; the homoscedastic GP can
         only ever be the horizontal line drawn for reference.

The scatter panels need per-point predictions, which the sweep JSON does not store,
so this script refits MLHGP at those budgets (cached in data/mlhgp/*.npz, which
ships with the repo).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bolt_exp import REPO_ROOT

SURFACE = "#ffffff"
INK, INK2, GRID = "#1a1a19", "#5c5c56", "#e2e2dd"
C_MLHGP, C_HOMO = "#2a78d6", "#eb6834"


def build_cache(path, budgets, n_test, seeds, em_iter):
    """Refit MLHGP at each budget and store per-point predictions on the test set."""
    import torch
    from bolt import DMCurriculumHet
    from bolt_exp.mlhgp import fit_mlhgp
    from bolt_exp.analysis.mlhgp_noise_vs_bo_iter import homo_noise_var, load_trial, test_design

    DT = torch.double
    prob = DMCurriculumHet()
    bounds = prob.bounds.to(DT)
    x_test = test_design(prob, n_test)
    true_var = (prob.evaluate_noise(x_test).clamp(min=1e-6).reshape(-1, 1).to(DT) ** 2)

    pred = {n: [] for n in budgets}
    homo = {n: [] for n in budgets}
    for seed in range(seeds):
        trial = load_trial(seed)
        X = torch.tensor(trial["candidates"], dtype=DT)
        Y = torch.tensor(trial["seen_y"], dtype=DT).reshape(-1, 1)
        for n in budgets:
            torch.manual_seed(seed)
            _, m = fit_mlhgp(X[:n], Y[:n], bounds, n_em_iter=em_iter)
            pred[n].append(m.predict_noise_var(x_test).flatten().numpy())
            torch.manual_seed(seed)
            homo[n].append(float(homo_noise_var(X[:n], Y[:n], bounds).item()))
            print(f"seed{seed} n={n} done", flush=True)
    np.savez(path, true=true_var.flatten().numpy(),
             **{f"pred{n}": np.array(pred[n]) for n in budgets},
             **{f"homo{n}": np.array(homo[n]) for n in budgets})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default=str(REPO_ROOT / "data" / "mlhgp" / "mlhgp_noise_vs_bo_iter.json"))
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "mlhgp" / "mlhgp_noise_learning_cache.npz"))
    ap.add_argument("--budgets", type=int, nargs=4, default=[20, 60, 110, 210])
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--em_iter", type=int, default=5)
    ap.add_argument("--out", default="pics/mlhgp/mlhgp_noise_learning.png")
    args = ap.parse_args()

    if not Path(args.cache).exists():
        build_cache(args.cache, args.budgets, args.n_test, args.seeds, args.em_iter)
    z = np.load(args.cache)

    runs = [r for r in json.loads(Path(args.curve).read_text())["runs"].values()
            if r.get("arm", "mlhgp") == "mlhgp"]
    ns = sorted({r["n"] for r in runs})
    x = np.array(ns) - ns[0]
    a = np.array([[r["heldout"]["spearman"] for r in runs if r["n"] == n] for n in ns]).T
    mu, ci = np.nanmean(a, 0), 1.96 * np.nanstd(a, 0, ddof=1) / np.sqrt(a.shape[0])

    fig = plt.figure(figsize=(15.4, 3.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.45, 1, 1, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURFACE)
    ax.plot(x, mu, lw=2.2, color=C_MLHGP, marker="o", ms=4, zorder=3)
    ax.fill_between(x, mu - ci, mu + ci, color=C_MLHGP, alpha=0.15, lw=0, zorder=2)
    for n in args.budgets:
        ax.axvline(n - ns[0], color=GRID, lw=1.2, zorder=1)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("BO iteration", fontsize=10, color=INK2)
    ax.set_ylabel("Spearman ρ  (learned vs oracle σ²)", fontsize=10.5, color=INK)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

    true = z["true"]
    lo = min([true.min()] + [z[f"pred{n}"].min() for n in args.budgets]
              + [np.mean(z[f"homo{n}"]) for n in args.budgets])
    lim = (lo / 1.5, true.max() * 2.5)
    for j, n in enumerate(args.budgets):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.set_facecolor(SURFACE)
        p = z[f"pred{n}"]
        ax.scatter(np.tile(true, p.shape[0]), p.flatten(), s=2, alpha=0.05,
                   color=C_MLHGP, lw=0, rasterized=True)
        # binned median of the predictions -- the trend the scatter is too dense to show
        edges = np.quantile(true, np.linspace(0, 1, 9))
        ctr, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (true >= lo) & (true <= hi)
            ctr.append(np.sqrt(lo * hi))
            med.append(np.median(p[:, m]))
        ax.plot(ctr, med, color="#184f95", lw=2, marker="o", ms=3.5, zorder=6,
                label="binned median")
        ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=4)
        ax.axhline(np.mean(z[f"homo{n}"]), color=C_HOMO, lw=1.6, zorder=5)
        if j == 0:
            ax.annotate("homoscedastic GP\n(one level)", xy=(lim[0] * 1.2, np.mean(z[f"homo{n}"])),
                        xytext=(0, 5), textcoords="offset points", fontsize=8,
                        color=C_HOMO, style="italic")
            ax.set_ylabel("learned noise variance", fontsize=10.5, color=INK)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel("oracle σ(x)²", fontsize=10, color=INK2)
        ax.set_title(f"BO iteration {n - ns[0]}", fontsize=10, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)

    # fig.suptitle("MLHGP learns the noise surface over the course of BO  —  DMCurriculumHet, "
    #              "10 seeds, held-out test set (dashed = perfect)", fontsize=11.5, color=INK, y=1.02)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
