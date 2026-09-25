# Multi-objective baseline methods (TSEMO, NSGA-II, NSGA-III) for dm_curriculum_mo.
#
# Usage:
#   python scripts/test_w_baselines_mo.py --method tsemo
#   python scripts/test_w_baselines_mo.py --method nsga2 --pop_size 50 --iterations 10
#   python scripts/test_w_baselines_mo.py --method nsga3 --iterations 10 --trials 3
#   python scripts/test_w_baselines_mo.py --method tsemo --iterations 50 --trials 3 --verbose
#
# Arguments:
#   --problem         {dm_curriculum_mo}                                          (default: dm_curriculum_mo)
#   --noise_std       Emulator noise std                              (default: problem's own)
#   --method          {tsemo, nsga2, nsga3}                                      (default: tsemo)
#   --iterations      TSEMO: BO iterations. NSGA: number of generations.         (default: 100)
#   --pop_size        NSGA only: population size per generation.                  (default: 50)
#   --initial_random_samples  TSEMO: initial Dirichlet samples.                  (default: 10)
#   --trials          Number of independent runs                                  (default: 1)
#   --trial_offset    Seed offset; merges with existing <offset>trials file.      (default: 0)
#   --verbose         Print per-iteration logs
#
# Output:
#   results/dm/<problem>_<method>_<N>trials_<iterations>iterations_results.json
#
# JSON structure (per trial) mirrors test_w_botorch_dm.py MO format:
#   hv_all, log_hv_diff_all, hv_true_all, log_hv_diff_true_all
#   best_hv_true_all, log_best_hv_diff_true_all
#   inf_hv_all, best_inf_hv_all, log_inference_hv_regret_all, log_best_inference_hv_regret_all
#   pareto_x_best_hv, pareto_y_best_hv, pareto_x_best_inf_hv, pareto_y_best_inf_hv
#   pareto_x_inf_hv, pareto_y_inf_hv, candidates, seen_y, ref_point, trial, seed, time_seconds
#
# Notes:
#   TSEMO uses BoTorch ModelListGP (not GPy) for surrogate fitting. Thompson sampling
#   selects the candidate with the highest greedy HV improvement over the current Pareto front.
#   NSGA-II/III use pymoo with Dirichlet sampling and a simplex repair operator.
#   NSGA methods do not fit a surrogate, so inf_hv_all / best_inf_hv_all are empty.
#   NSGA hv_all has T+1 entries: index 0 = initial population only, indices 1..T = after each generation.
#   TSEMO hv_all has one entry per BO iteration (plus one for the initial samples).

import argparse
import json
import math
from pathlib import Path
import time

from bolt_exp import emulator_version
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.ref_dirs import get_reference_directions
from rich import print

from bolt import DMCurriculumMO

from bolt_exp import REPO_ROOT


SIMPLEX_GROUPS = [[0, 1, 2], [3, 4, 5]]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _dirichlet_sample(n: int, rng) -> np.ndarray:
    x = np.zeros((n, 6))
    for group in SIMPLEX_GROUPS:
        s = rng.dirichlet(np.ones(len(group)), size=n)
        for k, idx in enumerate(group):
            x[:, idx] = s[:, k]
    return x


def compute_hypervolume(y: torch.Tensor, ref_point: list) -> float:
    pareto_mask = is_non_dominated(y)
    pareto_y = y[pareto_mask]
    hv_obj = Hypervolume(torch.tensor(ref_point, dtype=y.dtype))
    return hv_obj.compute(pareto_y)


def _make_dirichlet_candidates(n: int, rng, dtype, device) -> torch.Tensor:
    return torch.tensor(_dirichlet_sample(n, rng), dtype=dtype, device=device)


def initialize_mo_model(train_x, train_y, bounds):
    models = [
        SingleTaskGP(
            train_x,
            train_y[:, i : i + 1],
            input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        )
        for i in range(train_y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model).to(train_x)
    return mll, model


def compute_inference_hv(model, prob, ref_point, bounds, rng, dtype, device, n_cand=512):
    """HV of true values at the posterior-mean Pareto front."""
    cand = _make_dirichlet_candidates(n_cand, rng, dtype, device)
    with torch.no_grad():
        mean = model.posterior(cand).mean  # (n_cand, m)
    pareto_mask = is_non_dominated(mean)
    rec_x = cand[pareto_mask]
    rec_true = prob.evaluate_true(rec_x)
    hv = compute_hypervolume(rec_true.cpu(), ref_point)
    return hv, rec_x, rec_true


# ---------------------------------------------------------------------------
# TSEMO
# ---------------------------------------------------------------------------

def _tsemo_select(model, train_y, ref_point, bounds, rng, dtype, device, n_cand=1000):
    """Thompson sampling + greedy HV improvement candidate selection."""
    cand = _make_dirichlet_candidates(n_cand, rng, dtype, device)
    with torch.no_grad():
        ts_sample = model.posterior(cand).rsample(torch.Size([1])).squeeze(0)  # (n_cand, m)

    pareto_mask = is_non_dominated(ts_sample)
    pareto_x = cand[pareto_mask]
    pareto_ts = ts_sample[pareto_mask]

    if pareto_x.shape[0] == 0:
        idx = int(rng.integers(n_cand))
        return cand[idx : idx + 1]

    obs_pareto_mask = is_non_dominated(train_y)
    obs_pareto_y = train_y[obs_pareto_mask].cpu()
    ref_tensor = torch.tensor(ref_point, dtype=train_y.dtype)
    hv_base = Hypervolume(ref_tensor).compute(obs_pareto_y)

    best_gain = -float("inf")
    best_idx = 0
    for i in range(pareto_ts.shape[0]):
        combined = torch.cat([obs_pareto_y, pareto_ts[i : i + 1].cpu()], dim=0)
        combined_pareto = combined[is_non_dominated(combined)]
        gain = Hypervolume(ref_tensor).compute(combined_pareto) - hv_base
        if gain > best_gain:
            best_gain = gain
            best_idx = i

    return pareto_x[best_idx : best_idx + 1]


def run_tsemo(prob, rng, trial, seed, args, ref_point, bounds, dtype, device) -> dict:
    init_x_np = _dirichlet_sample(args.initial_random_samples, rng)
    train_x = torch.tensor(init_x_np, dtype=dtype, device=device)
    train_y = prob(train_x)

    # Initial metrics
    hv0 = compute_hypervolume(train_y.cpu(), ref_point)
    hv_all = [hv0]
    log_hv_diff_all = [math.log(max(prob._max_hv - hv0, 1e-8))]

    true_y0 = prob.evaluate_true(train_x)
    hv0_true = compute_hypervolume(true_y0.cpu(), ref_point)
    hv_true_all = [hv0_true]
    best_hv_true_running = hv0_true
    best_hv_true_all = [hv0_true]
    log_hv_diff_true_all = [math.log(max(prob._max_hv - hv0_true, 1e-8))]
    log_best_hv_diff_true_all = [math.log(max(prob._max_hv - best_hv_true_running, 1e-8))]

    pm0 = is_non_dominated(true_y0.cpu())
    best_hv_true_pareto_x = train_x[pm0].cpu().tolist()
    best_hv_true_pareto_y = true_y0.cpu()[pm0].tolist()

    seen_y = train_y.cpu().tolist()
    candidates_np = init_x_np.copy()

    mll, model = initialize_mo_model(train_x, train_y, bounds)
    fit_gpytorch_mll(mll)

    inf_hv0, inf_rec_x0, inf_rec_true0 = compute_inference_hv(
        model, prob, ref_point, bounds, rng, dtype, device
    )
    inf_hv_all = [inf_hv0]
    best_inf_hv_running = inf_hv0
    best_inf_hv_all = [inf_hv0]
    log_inference_hv_regret_all = [math.log(max(prob._max_hv - inf_hv0, 1e-8))]
    log_best_inference_hv_regret_all = [math.log(max(prob._max_hv - best_inf_hv_running, 1e-8))]
    best_inf_hv_pareto_x = inf_rec_x0.cpu().tolist()
    best_inf_hv_pareto_y = inf_rec_true0.cpu().tolist()
    inf_hv_pareto_x = inf_rec_x0.cpu().tolist()
    inf_hv_pareto_y = inf_rec_true0.cpu().tolist()

    t0 = time.time()

    for itr in range(args.iterations):
        new_x = _tsemo_select(model, train_y, ref_point, bounds, rng, dtype, device)
        new_y = prob(new_x)

        train_x = torch.vstack([train_x, new_x])
        train_y = torch.vstack([train_y, new_y])
        seen_y.extend(new_y.cpu().tolist())
        candidates_np = np.concatenate([candidates_np, new_x.cpu().numpy()], axis=0)

        hv = compute_hypervolume(train_y.cpu(), ref_point)
        hv_all.append(hv)
        log_hv_diff_all.append(math.log(max(prob._max_hv - hv, 1e-8)))

        true_y = prob.evaluate_true(train_x)
        hv_true = compute_hypervolume(true_y.cpu(), ref_point)
        hv_true_all.append(hv_true)
        log_hv_diff_true_all.append(math.log(max(prob._max_hv - hv_true, 1e-8)))

        if hv_true > best_hv_true_running:
            best_hv_true_running = hv_true
            pm = is_non_dominated(true_y.cpu())
            best_hv_true_pareto_x = train_x[pm].cpu().tolist()
            best_hv_true_pareto_y = true_y.cpu()[pm].tolist()
        best_hv_true_all.append(best_hv_true_running)
        log_best_hv_diff_true_all.append(math.log(max(prob._max_hv - best_hv_true_running, 1e-8)))

        mll, model = initialize_mo_model(train_x, train_y, bounds)
        fit_gpytorch_mll(mll)

        inf_hv, inf_rec_x, inf_rec_true = compute_inference_hv(
            model, prob, ref_point, bounds, rng, dtype, device
        )
        inf_hv_all.append(inf_hv)
        inf_hv_pareto_x = inf_rec_x.cpu().tolist()
        inf_hv_pareto_y = inf_rec_true.cpu().tolist()
        if inf_hv > best_inf_hv_running:
            best_inf_hv_running = inf_hv
            best_inf_hv_pareto_x = inf_rec_x.cpu().tolist()
            best_inf_hv_pareto_y = inf_rec_true.cpu().tolist()
        best_inf_hv_all.append(best_inf_hv_running)
        log_inference_hv_regret_all.append(math.log(max(prob._max_hv - inf_hv, 1e-8)))
        log_best_inference_hv_regret_all.append(math.log(max(prob._max_hv - best_inf_hv_running, 1e-8)))

        if args.verbose:
            print(
                f"  itr {itr + 1}: hv={hv:.4f}  hv_true={hv_true:.4f}  inf_hv={inf_hv:.4f}"
            )
        else:
            print(".", end="", flush=True)

    t1 = time.time()

    return {
        "trial": trial,
        "seed": seed,
        "time_seconds": t1 - t0,
        "ref_point": ref_point,
        "hv_all": hv_all,
        "log_hv_diff_all": log_hv_diff_all,
        "hv_true_all": hv_true_all,
        "log_hv_diff_true_all": log_hv_diff_true_all,
        "best_hv_true_all": best_hv_true_all,
        "log_best_hv_diff_true_all": log_best_hv_diff_true_all,
        "inf_hv_all": inf_hv_all,
        "best_inf_hv_all": best_inf_hv_all,
        "log_inference_hv_regret_all": log_inference_hv_regret_all,
        "log_best_inference_hv_regret_all": log_best_inference_hv_regret_all,
        "pareto_x_best_hv": best_hv_true_pareto_x,
        "pareto_y_best_hv": best_hv_true_pareto_y,
        "pareto_x_best_inf_hv": best_inf_hv_pareto_x,
        "pareto_y_best_inf_hv": best_inf_hv_pareto_y,
        "pareto_x_inf_hv": inf_hv_pareto_x,
        "pareto_y_inf_hv": inf_hv_pareto_y,
        "candidates": candidates_np.tolist(),
        "seen_y": seen_y,
    }


# ---------------------------------------------------------------------------
# NSGA-II / NSGA-III (pymoo)
# ---------------------------------------------------------------------------

class _BoLTProblem(Problem):
    def __init__(self, bolt_prob, dtype, device):
        super().__init__(n_var=6, n_obj=3, xl=0.0, xu=1.0)
        self.bolt_prob = bolt_prob
        self.dtype = dtype
        self.device = device
        self.X_batches: list[np.ndarray] = []
        self.Y_batches: list[torch.Tensor] = []

    def _evaluate(self, x, out, *args, **kwargs):
        x_t = torch.tensor(x, dtype=self.dtype, device=self.device)
        y = self.bolt_prob(x_t).detach().cpu()  # (n, 3), maximise
        self.X_batches.append(x.copy())
        self.Y_batches.append(y)
        out["F"] = -y.numpy()  # pymoo minimises


class _SimplexRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X = np.clip(X, 0.0, None)
        for group in SIMPLEX_GROUPS:
            s = X[:, group].sum(axis=1, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            X[:, group] = X[:, group] / s
        return X


class _DirichletSampling(Sampling):
    def __init__(self, rng):
        super().__init__()
        self.rng = rng

    def _do(self, problem, n_samples, **kwargs):
        return _dirichlet_sample(n_samples, self.rng)


class _HVCallback(Callback):
    """Tracks cumulative HV metrics after each NSGA generation."""

    def __init__(self, bolt_prob, ref_point, max_hv, dtype, device):
        super().__init__()
        self.bolt_prob = bolt_prob
        self.ref_point = ref_point
        self.max_hv = max_hv
        self.dtype = dtype
        self.device = device

        self.hv_all: list[float] = []
        self.log_hv_diff_all: list[float] = []
        self.hv_true_all: list[float] = []
        self.log_hv_diff_true_all: list[float] = []
        self.best_hv_true_all: list[float] = []
        self.log_best_hv_diff_true_all: list[float] = []

        self._best_hv_true_running = -float("inf")
        self._best_hv_true_pareto_x: list = []
        self._best_hv_true_pareto_y: list = []

        self._Y_true_cumulative: torch.Tensor | None = None
        self._Y_obs_cumulative: torch.Tensor | None = None
        self._X_cumulative: np.ndarray | None = None
        self._n_batches_seen = 0

    def _record_batch(self, X_np: np.ndarray, Y: torch.Tensor, Y_true: torch.Tensor):
        """Append metrics for one batch of observations."""
        if self._Y_true_cumulative is None:
            self._Y_true_cumulative = Y_true
            self._X_cumulative = X_np
            self._Y_obs_cumulative = Y
        else:
            self._Y_true_cumulative = torch.cat([self._Y_true_cumulative, Y_true], dim=0)
            self._X_cumulative = np.concatenate([self._X_cumulative, X_np], axis=0)
            self._Y_obs_cumulative = torch.cat([self._Y_obs_cumulative, Y], dim=0)

        hv = compute_hypervolume(self._Y_obs_cumulative, self.ref_point)
        self.hv_all.append(hv)
        self.log_hv_diff_all.append(math.log(max(self.max_hv - hv, 1e-8)))

        hv_true = compute_hypervolume(self._Y_true_cumulative, self.ref_point)
        self.hv_true_all.append(hv_true)
        self.log_hv_diff_true_all.append(math.log(max(self.max_hv - hv_true, 1e-8)))

        if hv_true > self._best_hv_true_running:
            self._best_hv_true_running = hv_true
            pm = is_non_dominated(self._Y_true_cumulative)
            self._best_hv_true_pareto_x = self._X_cumulative[pm.numpy()].tolist()
            self._best_hv_true_pareto_y = self._Y_true_cumulative[pm].tolist()

        self.best_hv_true_all.append(self._best_hv_true_running)
        self.log_best_hv_diff_true_all.append(
            math.log(max(self.max_hv - self._best_hv_true_running, 1e-8))
        )

    def notify(self, algorithm):
        pw: _BoLTProblem = algorithm.problem

        # Process any new batches since last notify.
        # Batch 0 (pymoo's re-evaluation of the initial population) is skipped because
        # _n_batches_seen is pre-seeded to 1 before the run starts.
        n_new = len(pw.X_batches) - self._n_batches_seen
        for i in range(n_new):
            idx = self._n_batches_seen + i
            x_np = pw.X_batches[idx]
            x_t = torch.tensor(x_np, dtype=self.dtype, device=self.device)
            y_true = self.bolt_prob.evaluate_true(x_t).detach().cpu()
            self._record_batch(x_np, pw.Y_batches[idx], y_true)
        self._n_batches_seen = len(pw.X_batches)


def _nsga3_n_partitions(pop_size: int, n_obj: int = 3) -> int:
    p = 1
    while math.comb(p + n_obj - 1, n_obj - 1) < pop_size:
        p += 1
    return p


def run_nsga(prob, rng, trial, seed, args, ref_point, bounds, dtype, device, method) -> dict:
    pop_size = args.pop_size
    # pymoo counts the initial population as gen 0, so n_gen=T gives T-1 offspring
    # generations. Use T+1 to get exactly T offspring generations (T+1 total results).
    n_gen = args.iterations + 1

    # Evaluate initial population once to produce index-0 metrics.
    init_x_np = _dirichlet_sample(pop_size, rng)
    init_x_t = torch.tensor(init_x_np, dtype=dtype, device=device)
    init_y = prob(init_x_t).detach().cpu()
    init_y_true = prob.evaluate_true(init_x_t).detach().cpu()

    pymoo_prob = _BoLTProblem(prob, dtype, device)
    callback = _HVCallback(prob, ref_point, prob._max_hv, dtype, device)

    # Seed callback with index-0 (initial population only) before the run.
    callback._record_batch(init_x_np, init_y, init_y_true)
    # pymoo will re-evaluate init_x_np as batch 0; skip it in notify.
    callback._n_batches_seen = 1

    repair = _SimplexRepair()

    if method == "nsga2":
        algorithm = NSGA2(pop_size=pop_size, sampling=init_x_np, repair=repair)
    else:  # nsga3
        n_part = _nsga3_n_partitions(pop_size)
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_part)
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=init_x_np,
            repair=repair,
        )

    t0 = time.time()
    pymoo_minimize(
        pymoo_prob,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=args.verbose,
        save_history=False,
    )
    t1 = time.time()

    # Use the manually evaluated init_y (not pymoo's re-eval at batch 0) for consistency.
    X_offspring = np.concatenate(pymoo_prob.X_batches[1:], axis=0) if len(pymoo_prob.X_batches) > 1 else np.zeros((0, 6))
    Y_offspring = torch.cat(pymoo_prob.Y_batches[1:], dim=0) if len(pymoo_prob.Y_batches) > 1 else torch.zeros((0, 3))
    X_all = np.concatenate([init_x_np, X_offspring], axis=0)
    Y_all = torch.cat([init_y, Y_offspring], dim=0)

    return {
        "trial": trial,
        "seed": seed,
        "time_seconds": t1 - t0,
        "ref_point": ref_point,
        "hv_all": callback.hv_all,
        "log_hv_diff_all": callback.log_hv_diff_all,
        "hv_true_all": callback.hv_true_all,
        "log_hv_diff_true_all": callback.log_hv_diff_true_all,
        "best_hv_true_all": callback.best_hv_true_all,
        "log_best_hv_diff_true_all": callback.log_best_hv_diff_true_all,
        "inf_hv_all": [],
        "best_inf_hv_all": [],
        "log_inference_hv_regret_all": [],
        "log_best_inference_hv_regret_all": [],
        "pareto_x_best_hv": callback._best_hv_true_pareto_x,
        "pareto_y_best_hv": callback._best_hv_true_pareto_y,
        "pareto_x_best_inf_hv": [],
        "pareto_y_best_inf_hv": [],
        "pareto_x_inf_hv": [],
        "pareto_y_inf_hv": [],
        "candidates": X_all.tolist(),
        "seen_y": Y_all.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"problem: {args.problem}")
    print(f"method: {args.method}")
    print(f"iterations: {args.iterations}")
    print(f"trials: {args.trials}")
    if args.method == "tsemo":
        print(f"initial_random_samples: {args.initial_random_samples}")
    else:
        print(f"pop_size: {args.pop_size}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float32 if device in ("mps", "cpu") else torch.double

    # --noise_std unset: let DMCurriculumMO use its own measured per-objective default
    if args.noise_std is None:
        prob = DMCurriculumMO(negate=False)
        noise_std = DMCurriculumMO._measured_std
    else:
        noise_std = args.noise_std
        prob = DMCurriculumMO(noise_std=noise_std, negate=False)
    prob.to(dtype=dtype, device=device)

    bounds = torch.tensor(prob._bounds, dtype=dtype).T.to(device)
    ref_point = prob._ref_point
    print(f"reference point: {ref_point}")

    all_trial_results = []

    for trial in range(args.trials):
        seed = trial + args.trial_offset
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.trials}  (seed={seed})")
        print(f"{'='*60}")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if args.method == "tsemo":
            result = run_tsemo(prob, rng, trial, seed, args, ref_point, bounds, dtype, device)
        else:
            result = run_nsga(
                prob, rng, trial, seed, args, ref_point, bounds, dtype, device, args.method
            )

        all_trial_results.append(result)
        print(
            f"\nTrial {trial + 1} done in {result['time_seconds']:.1f}s — "
            f"final hv={result['hv_all'][-1]:.4f}"
        )

    results_dir = REPO_ROOT / "results" / f"dm{args.folder_prefix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    total_trials = args.trial_offset + args.trials

    config = {
        "problem": args.problem,
        "method": args.method,
        "iterations": args.iterations,
        "pop_size": args.pop_size if args.method in ("nsga2", "nsga3") else None,
        "initial_random_samples": args.initial_random_samples if args.method == "tsemo" else None,
        "noise_std": noise_std,
        "emulator_versions": emulator_version.emulator_versions_for(prob),
    }

    def _make_filename(n_trials: int) -> Path:
        return results_dir / (
            f"{args.problem}_{args.method}_{n_trials}trials_{args.iterations}iterations_results.json"
        )

    if args.trial_offset > 0:
        prev_file = _make_filename(args.trial_offset)
        if not prev_file.exists():
            raise FileNotFoundError(
                f"--trial_offset={args.trial_offset} but expected prior results file not found: {prev_file}"
            )
        with open(prev_file) as f:
            existing = json.load(f)
        _check_keys = ["problem", "method", "iterations"]
        mismatches = {
            k: (existing.get(k), config[k])
            for k in _check_keys
            if existing.get(k) != config[k]
        }
        if mismatches:
            raise ValueError(
                "Config mismatch with "
                + prev_file.name
                + ":\n"
                + "\n".join(
                    f"  {k}: existing={old!r}, new={new!r}"
                    for k, (old, new) in mismatches.items()
                )
            )
        existing_seeds = {t["seed"] for t in existing["trials"]}
        new_seeds = {t["seed"] for t in all_trial_results}
        if existing_seeds & new_seeds:
            raise ValueError(
                f"Duplicate seeds {existing_seeds & new_seeds} — adjust --trial_offset."
            )
        merged_trials = existing["trials"] + all_trial_results
        results = {**config, "num_trials": len(merged_trials), "trials": merged_trials}
        print(f"\nMerging {args.trial_offset} existing + {args.trials} new trials")
    else:
        results = {**config, "num_trials": total_trials, "trials": all_trial_results}

    output_file = _make_filename(total_trials)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline MO methods (TSEMO, NSGA-II, NSGA-III) for dm_curriculum_mo"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="dm_curriculum_mo",
        choices=["dm_curriculum_mo"],
        help="Problem to optimize. Default: dm_curriculum_mo",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="Observation noise std for the emulator. Default: the problem class's own default",
    )
    parser.add_argument(
        "--folder_prefix",
        type=str,
        default="",
        help="Suffix appended to the results subfolder name, e.g. 'dm{folder_prefix}'. Default: '' (results/dm)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsemo",
        choices=["tsemo", "nsga2", "nsga3"],
        help="Optimization method. Default: tsemo",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="TSEMO: BO iterations (total obs = initial_random_samples + iterations). "
             "NSGA: number of generations after initialization. Default: 100",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=50,
        help="NSGA only: population size. Default: 50",
    )
    parser.add_argument(
        "--initial_random_samples",
        type=int,
        default=10,
        help="TSEMO only: number of initial Dirichlet samples. Default: 10",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent trials (each uses a different random seed). Default: 1",
    )
    parser.add_argument(
        "--trial_offset",
        type=int,
        default=0,
        help="Seed offset; also used to extend existing results. Default: 0",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration logs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
