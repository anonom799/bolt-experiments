"""Does the MLHGP noise model improve as BO collects data?

Replays the design actually visited by the LogNEI+MLHGP runs on DMCurriculumHet
(`candidates` / `seen_y` in the results JSONs -- 10 initial points + 200 BO
iterations) and refits MLHGP at a series of checkpoints. At each checkpoint the
learned noise is scored against the oracle sigma(x)^2 two ways:

  held-out  -- noise_gp.predict_noise_var on a FIXED Dirichlet test set. Comparable
               across checkpoints, and the quantity BO actually uses when it
               evaluates the acquisition at unseen x.
  in-sample -- learned_noise_var at the n training points, i.e. the train_Yvar fed
               to the signal GP. The point set grows with n, so read the trend, not
               the level.

Arms (--arms):
  mlhgp  -- MLHGP at the shipped settings, the model under study.
  homo   -- the homoscedastic baseline: a plain SingleTaskGP on the same data, whose
            single learned noise level is the "noise model" LogNEI uses. Constant in
            x, so its Spearman is undefined (recorded as NaN) and only the log-RMSE /
            bias against sigma(x)^2 is meaningful -- it is the floor the het model
            has to beat.
  Known-noise (LogNEI+KN) is not an arm: it is handed prob.evaluate_noise(x)^2 as
  train_Yvar, so in-sample it equals the oracle exactly (rho = 1, RMSE = 0) at every
  iteration, and at held-out x a fixed-noise likelihood carries no noise model at all.

Writes data/mlhgp/mlhgp_noise_vs_bo_iter.json, which figures.plot_mlhgp_noise_learning reads.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import spearmanr

from bolt import DMCurriculumHet
from bolt_exp import REPO_ROOT, load_result
from bolt_exp.mlhgp import fit_mlhgp

DTYPE = torch.double
SIMPLEX_GROUPS = [[0, 1, 2], [3, 4, 5]]
RUN = REPO_ROOT / "results" / "dm" / "dm_curriculum_heteroscedastic_qnei_mlhgp_em5_10trials_200iterations_results.json"


def load_trial(seed):
    """The LogNEI+MLHGP trial run with `seed`."""
    return next(t for t in load_result(RUN)["trials"] if t["seed"] == seed)


def test_design(prob, n, seed=12345):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, prob.dim))
    for group in SIMPLEX_GROUPS:
        s = rng.dirichlet(np.ones(len(group)), size=n)
        for k, idx in enumerate(group):
            x[:, idx] = s[:, k]
    return torch.tensor(x, dtype=DTYPE)


def homo_noise_var(X, Y, bounds):
    """The single noise level a plain (homoscedastic) SingleTaskGP learns, raw units."""
    gp = SingleTaskGP(X, Y, input_transform=Normalize(d=X.shape[-1], bounds=bounds))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp).to(X))
    with torch.no_grad():
        var = gp.likelihood.noise.mean().reshape(1, 1).to(X)
        ot = getattr(gp, "outcome_transform", None)
        stdvs = getattr(ot, "stdvs", None) if ot is not None else None
        if stdvs is not None:                       # noise is learned in standardized space
            var = var * stdvs.to(var).pow(2).reshape(1, 1)
    return var


def score(est_var, true_var):
    ev = np.broadcast_to(est_var.flatten().cpu().numpy(), true_var.numel())
    tv = true_var.flatten().cpu().numpy()
    # a constant estimate (the homoscedastic arm) has no ranking to score
    rho = float("nan") if np.ptp(ev) == 0 else float(spearmanr(ev, tv).statistic)
    return {
        "spearman": rho,
        "rmse_log10": float(np.sqrt(np.mean((np.log10(ev) - np.log10(tv)) ** 2))),
        "bias_log10": float(np.mean(np.log10(ev) - np.log10(tv))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--em_iter", type=int, default=5)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[10, 15, 20, 30, 40, 60, 80, 110, 140, 170, 210])
    ap.add_argument("--arms", nargs="+", default=["mlhgp", "homo"],
                    choices=["mlhgp", "homo"])
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "mlhgp" / "mlhgp_noise_vs_bo_iter.json"))
    args = ap.parse_args()

    prob = DMCurriculumHet()
    bounds = prob.bounds.to(DTYPE)

    x_test = test_design(prob, args.n_test)
    true_var_test = prob.evaluate_noise(x_test).clamp(min=1e-6).reshape(-1, 1).to(DTYPE) ** 2

    out = {"config": vars(args), "runs": {}}
    if Path(args.out).exists():                      # merge, so arms can be added later
        out["runs"] = json.loads(Path(args.out).read_text()).get("runs", {})
    t0 = time.time()

    for seed in range(args.seeds):
        trial = load_trial(seed)
        X = torch.tensor(trial["candidates"], dtype=DTYPE)
        Y = torch.tensor(trial["seen_y"], dtype=DTYPE).reshape(-1, 1)
        true_var_train = prob.evaluate_noise(X).clamp(min=1e-6).reshape(-1, 1).to(DTYPE) ** 2

        for n in args.checkpoints:
            if n > X.shape[0]:
                continue
            for arm in args.arms:
                torch.manual_seed(seed)
                if arm == "mlhgp":
                    _, m = fit_mlhgp(X[:n], Y[:n], bounds, n_em_iter=args.em_iter)
                    var_test, var_train = m.predict_noise_var(x_test), m.learned_noise_var
                    floor = int((var_train <= 1.001e-6).sum().item())
                else:
                    var_test = var_train = homo_noise_var(X[:n], Y[:n], bounds)
                    floor = 0
                rec = {
                    "seed": seed, "n": n, "bo_iter": n - args.checkpoints[0], "arm": arm,
                    "heldout": score(var_test, true_var_test),
                    "insample": score(var_train, true_var_train[:n]),
                    "n_at_clamp_floor": floor,
                }
                key = f"seed{seed}_n{n}" if arm == "mlhgp" else f"seed{seed}_n{n}_{arm}"
                out["runs"][key] = rec
                print(f"seed{seed} n={n:3d} {arm:5s} heldout rho={rec['heldout']['spearman']:+.3f} "
                      f"rmse={rec['heldout']['rmse_log10']:.3f} | "
                      f"insample rho={rec['insample']['spearman']:+.3f} "
                      f"rmse={rec['insample']['rmse_log10']:.3f} [{time.time()-t0:.0f}s]",
                      flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
