# Bayesian optimization baselines for the PO (prompt optimization) problem.
#
# Usage:
#   python test_w_botorch_po.py
#   python test_w_botorch_po.py --problem po256 --acq_fn ei
#   python test_w_botorch_po.py --acq_fn ucb --iterations 50 --trials 3
#   python test_w_botorch_po.py --acq_fn pes --verbose
#   python test_w_botorch_po.py --acq_fn ts --turbo --verbose
#   python test_w_botorch_po.py --acq_fn qnei --raasp
#   python test_w_botorch_po.py --acq_fn qnei --msr
#   python test_w_botorch_po.py --acq_fn qnei --mle_scaled_init
#   python test_w_botorch_po.py --acq_fn qnei --turbo --raasp
#   python test_w_botorch_po.py --acq_fn ts --baxus --verbose
#   python test_w_botorch_po.py --acq_fn qnei --saasbo --verbose
#
# Arguments:
#   --problem       Problem variant: po128, po256, po512, po768          (default: po128)
#   --acq_fn        Acquisition function: ei, qnei, ucb, kg, mes, gibbon, pes, jes, ts, random  (default: qnei)
#   --iterations    Number of BO iterations; total observations = iterations * batch_size  (default: 100)
#   --batch_size    Candidates per BO iteration; ei/ucb require 1        (default: 1)
#   --ucb_beta      Fixed beta for UCB (None = Srinivas schedule)        (default: None)
#   --initial_random_samples  Number of initial random samples           (default: 10)
#   --trials        Number of independent runs with different seeds      (default: 1)
#   --trial_offset  Offset added to trial index to get seed             (default: 0)
#   --turbo         Use TuRBO trust-region filtering (mutually exclusive with --baxus/--saasbo)
#   --turbo_pm_center  Use posterior mean argmax as TuRBO center instead of noisy obs argmax (requires --turbo)
#   --raasp         Use RAASP candidate generation (combinable with --turbo); no lengthscale changes
#   --mle_scaled_init  Use sqrt(d)/10 lengthscale init before MLE, without RAASP
#   --msr           Use MSR = RAASP + sqrt(d)/10 lengthscale init (Hvarfner et al., ICML 2025)
#   --baxus         Use BAxUS: random subspace embedding that expands on stagnation (mutually exclusive with --turbo/--saasbo)
#   --saasbo        Use SAASBO: fully Bayesian GP with SAAS horseshoe prior via NUTS MCMC (mutually exclusive with --turbo/--baxus)
#   --verbose       Print per-iteration logs
#
# Output:
#   results/<problem>/<problem>_<acq_fn>[_turbo|_baxus|_saasbo][_raasp|_msr][_mlesi][_beta<B>][_q<Q>]_<trials>trials_<iterations>iterations_results.json
#
# The PO problem is a nearest-neighbour lookup over 5014 tabular prompt embeddings
# (128-dim) evaluated on MATH500 0-shot accuracy.  The search space is the finite
# discrete set of table rows, so optimize_acqf_discrete is used to enumerate the
# full candidate set at each iteration.

import argparse
from dataclasses import dataclass, field
import json
import math
import time

import numpy as np
from rich import print
import torch

from botorch.acquisition import (
    LogExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy, qMaxValueEntropy
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bolt.problems.prompt_opt import PO, PO128, PO256, PO512, PO768

from bolt_exp import REPO_ROOT

MC_SAMPLES = 128

_PO_CLASSES = {
    "po128": PO128,
    "po256": PO256,
    "po512": PO512,
    "po768": PO768,
}

@dataclass
class TurboState:
    dim: int
    batch_size: int
    n_cand_max: int           # Total candidate set size (len(all_X))
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = field(init=False)
    n_cand: int = field(init=False)   # Current trust-region size (# L2-nearest neighbors kept)
    n_cand_init: int = field(init=False)
    n_cand_min: int = field(init=False)
    restart_triggered: bool = False

    def __post_init__(self):
        self.n_cand_init = self.n_cand_max // 2
        self.n_cand = self.n_cand_init
        self.n_cand_min = max(self.batch_size * 4, 20)
        # dim/batch_size is calibrated for continuous sobol generation and is
        # far too large for a fixed discrete candidate set — use a small fixed value
        self.failure_tolerance = 10


def update_turbo_state(state: TurboState, new_y: torch.Tensor, best_y: float) -> TurboState:
    improved = new_y.max().item() > best_y + 1e-3 * abs(best_y)
    if improved:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.failure_counter += 1
        state.success_counter = 0

    if state.success_counter >= state.success_tolerance:
        state.n_cand = min(state.n_cand * 2, state.n_cand_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.n_cand = max(state.n_cand // 2, state.n_cand_min)
        state.failure_counter = 0

    state.restart_triggered = state.n_cand <= state.n_cand_min
    if state.restart_triggered:
        state.n_cand = state.n_cand_init
    return state



@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    n_cand_max: int
    batch_size: int
    target_dim: int = field(init=False)
    n_splits: int = field(init=False)
    d_init: int = field(init=False)
    n_cand: int = field(init=False)
    n_cand_init: int = field(init=False)
    n_cand_min: int = field(init=False)
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    expand_triggered: bool = False

    def __post_init__(self):
        self.d_init = 8
        self.target_dim = self.d_init
        self.n_splits = round(math.log2(self.dim / self.d_init))
        self.n_cand_init = self.n_cand_max // 2
        self.n_cand = self.n_cand_init
        self.n_cand_min = max(self.batch_size * 4, 20)

    @property
    def split_budget(self) -> int:
        # Geometric allocation (paper Sec. 3.4, b=1):
        #   m^s_i = round(b * mD * d_i / (d_init * ((b+1)^(n+1) - 1)))
        # With b=1: denominator = d_init * (2^(n_splits+1) - 1)
        return round(self.eval_budget * self.target_dim / (self.d_init * (2 ** (self.n_splits + 1) - 1)))

    @property
    def failure_tolerance(self) -> int:
        # k = halvings of discrete TR from n_cand_init to n_cand_min (mirrors
        # the continuous TR's log(length_min/length_init, 0.5) in the paper)
        k = max(1, math.floor(math.log2(self.n_cand_init / max(self.n_cand_min, 1))))
        return max(1, min(math.floor(self.split_budget / k), self.target_dim))


def get_baxus_embedding(d: int, t: int, dtype, device) -> torch.Tensor:
    """Sparse ±1 BAxUS embedding S ∈ R^(t × d) via random permutation + even bin split.

    Each input dim maps to exactly one target bucket; bins are balanced so no
    bucket is empty (avoids the constant-zero column / Normalize divide-by-zero).
    """
    if t >= d:
        return torch.eye(d, dtype=dtype, device=device)
    perm = torch.randperm(d, device=device)
    bins = torch.tensor_split(perm, t)
    S = torch.zeros(t, d, dtype=dtype, device=device)
    for i, b in enumerate(bins):
        signs = (torch.randint(2, (len(b),), device=device).to(dtype) * 2 - 1)
        S[i, b] = signs
    return S


def expand_baxus_embedding(S: torch.Tensor, dtype, device) -> torch.Tensor:
    """Double target_dim by splitting each row's input dims into two sub-rows.

    Rows with only one input dim assigned cannot be split and are kept whole;
    in that case a fresh random row is appended to still double the size.
    This preserves the nested-subspace property: each new row covers a strict
    subset of the input dims its parent row covered.
    """
    d = S.shape[1]
    new_rows = []
    for i in range(S.shape[0]):
        nz = S[i].nonzero(as_tuple=True)[0]
        if len(nz) <= 1:
            new_rows.append(S[i].clone())
            # pad with a fresh random singleton from remaining dims if possible
            free = (S.abs().sum(0) == 0).nonzero(as_tuple=True)[0]
            if len(free) > 0:
                row = torch.zeros(d, dtype=dtype, device=device)
                chosen = free[torch.randint(len(free), (1,), device=device)]
                row[chosen] = (torch.randint(2, (1,), device=device).to(dtype) * 2 - 1)
                new_rows.append(row)
            else:
                new_rows.append(S[i].clone())
        else:
            half = len(nz) // 2
            for sub in [nz[:half], nz[half:]]:
                row = torch.zeros(d, dtype=dtype, device=device)
                row[sub] = S[i, sub]
                new_rows.append(row)
    return torch.stack(new_rows)


def update_baxus_state(state: BaxusState, new_y: torch.Tensor, best_y: float) -> BaxusState:
    improved = new_y.max().item() > best_y + 1e-3 * abs(best_y)
    if improved:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.failure_counter += 1
        state.success_counter = 0

    if state.success_counter >= state.success_tolerance:
        state.n_cand = min(state.n_cand * 2, state.n_cand_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.n_cand = max(state.n_cand // 2, state.n_cand_min)
        state.failure_counter = 0

    # TR collapsed to minimum → expand to next subspace and reset TR
    if state.n_cand <= state.n_cand_min:
        if state.target_dim < state.dim:
            state.target_dim = min(state.target_dim * 2, state.dim)
            state.expand_triggered = True
        else:
            state.expand_triggered = False
        state.n_cand = state.n_cand_init
        state.failure_counter = 0
        state.success_counter = 0
    else:
        state.expand_triggered = False
    return state


def initialize_saasbo_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
) -> SaasFullyBayesianSingleTaskGP:
    return SaasFullyBayesianSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )


def generate_initial_data(prob, choices: torch.Tensor, rng, n: int = 10, dtype=torch.double) -> tuple:
    """Sample n random rows from the table as initial observations."""
    idx = rng.choice(len(choices), size=n, replace=False)
    train_x = choices[idx]
    train_y = prob(train_x.to(dtype=dtype))
    return train_x, train_y


def initialize_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    state_dict=None,
    covar_module=None,
    scaled_ls_init: bool = False,
):
    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=covar_module,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
    )
    if scaled_ls_init:
        # Scaled lengthscale init: set length scales to sqrt(d)/10 before MLE.
        # GPyTorch's default (~0.65) causes vanishing MLL gradients in high-d
        # because inter-point distances scale as sqrt(d). The property setter
        # writes directly to raw_lengthscale (BoTorch uses GreaterThan, not
        # softplus, so raw == actual lengthscale — no inverse transform needed).
        d = train_x.shape[-1]
        init_ls = math.sqrt(d) / 10.0
        ls_val = torch.full((1, d), init_ls, dtype=train_x.dtype, device=train_x.device)
        # default RBF: covar_module IS the kernel; --matern: ScaleKernel wraps it
        kern = (
            model.covar_module.base_kernel
            if hasattr(model.covar_module, "base_kernel")
            else model.covar_module
        )
        kern.lengthscale = ls_val

    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


def get_raasp_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    pool_X: torch.Tensor,
    n_perturb: int = 500,
) -> torch.Tensor:
    """Return a locally-perturbed candidate set for RAASP (Hvarfner et al. 2025).

    Takes the top-5% observed points (elite set), generates n_perturb perturbations
    by randomly replacing each dimension with probability min(1, 20/d) using values
    drawn from pool_X, then snaps each perturbation to its nearest row in pool_X.
    When combined with TuRBO, pool_X is already the trust-region-filtered subset,
    so perturbations stay local to both the incumbent and the trust region.
    """
    d = train_x.shape[-1]
    # sparse replacement probability: each dim is perturbed with prob 20/d,
    # so on average 20 dims change regardless of dimensionality
    replace_prob = min(1.0, 20.0 / d)
    n_elite = max(1, math.ceil(0.05 * len(train_y)))
    elite_x = train_x[train_y.flatten().topk(n_elite).indices]  # (n_elite, d)

    # tile elite points and draw random replacement rows from the pool
    n_each = max(1, n_perturb // n_elite)
    perturbs = elite_x.repeat_interleave(n_each, dim=0)          # (n_perturb, d)
    rand_rows = pool_X[torch.randint(len(pool_X), (len(perturbs),), device=pool_X.device)]
    mask = torch.rand_like(perturbs) < replace_prob
    perturbs = torch.where(mask, rand_rows, perturbs)

    # snap to nearest valid embedding row so candidates stay on-table
    nn_idx = torch.cdist(perturbs, pool_X).argmin(dim=1).unique()
    return pool_X[nn_idx]


def get_beta_t(n_step: int, n_var_dim: int) -> float:
    return 2.0 * np.log(n_var_dim * (n_step + 1) ** 2 * np.pi**2 / 6.0 / 0.1)


def get_optimal_inputs_discrete(
    model,
    choices: torch.Tensor,
    num_samples: int = 64,
) -> torch.Tensor:
    """Draw Thompson samples from the table candidates for PES."""
    thompson_sampler = MaxPosteriorSampling(model=model, replacement=False)
    return thompson_sampler(choices, num_samples=num_samples)


def build_so_acqf(
    name: str,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    beta: float,
    choices: torch.Tensor,
):
    if name == "qnei":
        return qLogNoisyExpectedImprovement(
            model=model, X_baseline=train_x, prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )
    elif name == "ei":
        return LogExpectedImprovement(model=model, best_f=train_y.max())
    elif name == "ucb":
        return UpperConfidenceBound(model=model, beta=beta)
    elif name == "kg":
        return qKnowledgeGradient(model=model, num_fantasies=8, raw_samples=MC_SAMPLES)
    elif name == "mes":
        return qMaxValueEntropy(model=model, candidate_set=choices)
    elif name == "gibbon":
        return qLowerBoundMaxValueEntropy(model=model, candidate_set=choices)
    elif name == "pes":
        optimal_inputs = get_optimal_inputs_discrete(model, choices)
        return qPredictiveEntropySearch(
            model=model, optimal_inputs=optimal_inputs, maximize=True
        )
    elif name == "jes":
        optimal_inputs = get_optimal_inputs_discrete(model, choices)
        with torch.no_grad():
            optimal_outputs = model.posterior(optimal_inputs).mean
        return qJointEntropySearch(
            model=model, optimal_inputs=optimal_inputs, optimal_outputs=optimal_outputs,
            num_samples=MC_SAMPLES,
        )
    else:
        raise ValueError(f"Unknown acq_fn: {name}")


def main(args):
    print("problem:", args.problem)
    print("acq_fn:", args.acq_fn)
    print("number of BO iterations:", args.iterations)
    print("total observations:", args.iterations * args.batch_size)
    print("batch_size:", args.batch_size)

    if args.acq_fn == "ucb":
        beta_desc = f"fixed ({args.ucb_beta})" if args.ucb_beta is not None else "Srinivas schedule"
        print(f"ucb_beta: {beta_desc}")

    print("initial_random_samples:", args.initial_random_samples)
    print("trials:", args.trials)
    print("kernel:", "matern-2.5" if args.matern else "default (rbf)")
    if args.turbo:
        print("turbo: enabled")
    if args.turbo_pm_center:
        print("turbo_pm_center: enabled (posterior mean used to select trust region center)")
    if args.baxus:
        print("baxus: enabled (random subspace, expands on stagnation)")
    if args.saasbo:
        print("saasbo: enabled (fully Bayesian GP via NUTS)")
    if args.raasp:
        print("raasp: enabled")
    if args.mle_scaled_init:
        print("mle_scaled_init: enabled (scaled lengthscale init only, no raasp)")
    if args.msr:
        print("msr: enabled (raasp + scaled lengthscale init)")
        args.raasp = True  # msr implies raasp

    verbose = args.verbose
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    # NUTS (used by SAASBO) relies on Pyro ops not available on MPS; force CPU
    if args.saasbo and dev == "mps":
        dev = "cpu"
    dtype = torch.float32 if dev in ("mps", "cpu") else torch.double

    prob = _PO_CLASSES[args.problem](noise_std=0.001, negate=False)
    prob.to(dtype=dtype, device=dev)

    optimal_value = prob.obj_func.ys.max().item()

    # Full discrete candidate set loaded from the table once
    all_X = prob.obj_func.Xs.to(device=dev, dtype=dtype)  # (N, 128)
    bounds_tensor = torch.tensor(prob._bounds, dtype=dtype).T.to(dev)  # (2, 128)

    covar_module = (
        ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=prob.dim)) if args.matern else None
    )

    all_trial_results = []

    for trial in range(args.trials):
        seed = trial + args.trial_offset
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.trials}  (seed={seed})")
        print(f"{'='*60}")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed) 

        print("generating initial data...")
        train_x, train_y = generate_initial_data(
            prob, all_X, rng, n=args.initial_random_samples, dtype=dtype
        )
        train_x = train_x.to(device=dev, dtype=dtype)
        train_y = train_y.to(device=dev, dtype=dtype)

        turbo_state = TurboState(dim=prob.dim, batch_size=args.batch_size, n_cand_max=len(all_X)) if args.turbo else None
        turbo_ncand_all = []

        baxus_state = BaxusState(dim=prob.dim, eval_budget=args.iterations, n_cand_max=len(all_X), batch_size=args.batch_size) if args.baxus else None
        baxus_S = get_baxus_embedding(prob.dim, baxus_state.d_init, dtype, dev) if args.baxus else None
        baxus_target_dim_all = []
        baxus_ncand_all = []

        best_y_all = [train_y.max().item()]
        seen_y = train_y.cpu().tolist()
        candidates = train_x.cpu().numpy()

        rec_x_all, rec_true_all, best_rec_true_all = [], [], []
        inference_regret_all, log_inference_regret_all = [], []
        best_inference_regret_all, log_best_inference_regret_all = [], []

        _init_best_x = train_x[train_y.flatten().argmax()].detach()
        _init_best_true = prob.evaluate_true(_init_best_x.unsqueeze(0)).item()
        best_obs_x_all = [_init_best_x.cpu().tolist()]
        best_obs_true_all = [_init_best_true]
        best_obs_regret_all = [max(optimal_value - _init_best_true, 0.0)]
        log_best_obs_regret_all = [math.log(max(optimal_value - _init_best_true, 1e-8))]

        _init_true_vals = prob.evaluate_true(train_x).flatten()
        _best_simple_true_running = _init_true_vals.max().item()
        simple_regret_all = [max(optimal_value - _best_simple_true_running, 0.0)]
        log_simple_regret_all = [math.log(max(optimal_value - _best_simple_true_running, 1e-8))]

        # ---- initial model fit ----
        if args.acq_fn != "random":
            if args.saasbo:
                model = initialize_saasbo_model(train_x, train_y, bounds_tensor)
                fit_fully_bayesian_model_nuts(
                    model, warmup_steps=256, num_samples=128, thinning=16, disable_progbar=not verbose
                )
                if dev == "cuda":
                    torch.cuda.empty_cache()
                mll = None
            elif args.baxus:
                Z_all = all_X @ baxus_S.T  # (N, t)
                train_x_proj = train_x @ baxus_S.T  # (n_init, t)
                proj_bounds = torch.stack([Z_all.min(0).values, Z_all.max(0).values])
                baxus_covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=baxus_state.target_dim)) if args.matern else None
                mll, model = initialize_model(train_x_proj, train_y, proj_bounds, covar_module=baxus_covar, scaled_ls_init=args.msr or args.mle_scaled_init)
                fit_gpytorch_mll(mll)
            else:
                mll, model = initialize_model(train_x, train_y, bounds_tensor, covar_module=covar_module, scaled_ls_init=args.msr or args.mle_scaled_init)
                fit_gpytorch_mll(mll)

            # initial recommendation
            if args.baxus:
                rec_cand_proj, _ = optimize_acqf_discrete(PosteriorMean(model), choices=Z_all, q=1)
                rec_idx = torch.cdist(rec_cand_proj, Z_all).argmin(dim=1).item()
                rec_x_init = all_X[rec_idx].detach()
            else:
                rec_cand, _ = optimize_acqf_discrete(
                    PosteriorMean(model), choices=all_X, q=1,
                    max_batch_size=256,
                )
                rec_x_init = rec_cand.squeeze(0).detach()

            rec_true_init = prob.evaluate_true(rec_x_init.unsqueeze(0)).item()
            rec_x_all.append(rec_x_init.cpu().tolist())
            rec_true_all.append(rec_true_init)
            _best_rec_true_running = rec_true_init
            best_rec_true_all.append(_best_rec_true_running)

            _ir_init = max(optimal_value - rec_true_init, 0.0)
            inference_regret_all.append(_ir_init)
            log_inference_regret_all.append(math.log(max(_ir_init, 1e-8)))
            best_inference_regret_all.append(max(optimal_value - _best_rec_true_running, 0.0))
            log_best_inference_regret_all.append(math.log(max(optimal_value - _best_rec_true_running, 1e-8)))

        num_bo_iters = args.iterations
        t0 = time.time()

        for itr in range(num_bo_iters):

            if args.acq_fn == "random":
                idx = rng.choice(len(all_X), size=args.batch_size, replace=False)
                new_x = all_X[idx].to(dtype=dtype)
                acq_value = torch.zeros(1)

            elif args.saasbo:
                # free previous model's 128 MCMC-sample tensors before re-fitting to avoid OOM
                del model
                if dev == "cuda":
                    torch.cuda.empty_cache()
                elif dev == "mps":
                    torch.mps.empty_cache()

                # refit fully Bayesian model each iteration (no warm-start for NUTS)
                model = initialize_saasbo_model(train_x, train_y, bounds_tensor)
                fit_fully_bayesian_model_nuts(
                    model, warmup_steps=256, num_samples=128, thinning=16, disable_progbar=not verbose
                )
                if dev == "cuda":
                    torch.cuda.empty_cache()
                elif dev == "mps":
                    torch.mps.empty_cache()

                rec_cand, _ = optimize_acqf_discrete(
                    PosteriorMean(model), choices=all_X, q=1, max_batch_size=256
                )
                rec_x = rec_cand.squeeze(0).detach()
                rec_true = prob.evaluate_true(rec_x.unsqueeze(0)).item()
                rec_x_all.append(rec_x.cpu().tolist())
                rec_true_all.append(rec_true)
                _best_rec_true_running = max(_best_rec_true_running, rec_true)
                best_rec_true_all.append(_best_rec_true_running)

                _ir = max(optimal_value - rec_true, 0.0)
                inference_regret_all.append(_ir)
                log_inference_regret_all.append(math.log(max(_ir, 1e-8)))
                best_inference_regret_all.append(max(optimal_value - _best_rec_true_running, 0.0))
                log_best_inference_regret_all.append(math.log(max(optimal_value - _best_rec_true_running, 1e-8)))

                cand_X = all_X
                if args.raasp:
                    cand_X = get_raasp_candidates(train_x, train_y, cand_X)

                if args.acq_fn == "ts":
                    new_x = MaxPosteriorSampling(model=model, replacement=False)(
                        cand_X, num_samples=args.batch_size
                    ).detach().to(dtype=dtype)
                    acq_value = torch.zeros(1)
                else:
                    acq = qLogNoisyExpectedImprovement(
                        model=model, X_baseline=train_x, prune_baseline=True,
                        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
                    )
                    candidate, acq_value = optimize_acqf_discrete(
                        acq_function=acq, choices=cand_X, q=args.batch_size, max_batch_size=256
                    )
                    new_x = candidate.detach().to(dtype=dtype)

            elif args.baxus:
                # expand subspace if TR collapsed in previous iteration
                if baxus_state.expand_triggered:
                    baxus_S = expand_baxus_embedding(baxus_S, dtype, dev)
                    baxus_state.expand_triggered = False

                # project training data and full candidate set into current subspace
                Z_all = all_X @ baxus_S.T  # (N, t)
                train_x_proj = train_x @ baxus_S.T  # (n_obs, t)
                proj_bounds = torch.stack([Z_all.min(0).values, Z_all.max(0).values])

                baxus_covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=baxus_state.target_dim)) if args.matern else None
                mll, model = initialize_model(train_x_proj, train_y, proj_bounds, covar_module=baxus_covar, scaled_ls_init=args.msr or args.mle_scaled_init)
                fit_gpytorch_mll(mll)

                # recommendation: optimize PosteriorMean in projected space, map back to original
                rec_cand_proj, _ = optimize_acqf_discrete(PosteriorMean(model), choices=Z_all, q=1)
                rec_idx = torch.cdist(rec_cand_proj, Z_all).argmin(dim=1).item()
                rec_x = all_X[rec_idx].detach()
                rec_true = prob.evaluate_true(rec_x.unsqueeze(0)).item()
                rec_x_all.append(rec_x.cpu().tolist())
                rec_true_all.append(rec_true)
                _best_rec_true_running = max(_best_rec_true_running, rec_true)
                best_rec_true_all.append(_best_rec_true_running)

                _ir = max(optimal_value - rec_true, 0.0)
                inference_regret_all.append(_ir)
                log_inference_regret_all.append(math.log(max(_ir, 1e-8)))
                best_inference_regret_all.append(max(optimal_value - _best_rec_true_running, 0.0))
                log_best_inference_regret_all.append(math.log(max(optimal_value - _best_rec_true_running, 1e-8)))

                # TR narrowing: restrict to n_cand nearest neighbors of the
                # best observed point in the projected space (discrete analog
                # of the continuous TR used in the BAxUS paper)
                best_proj = train_x_proj[train_y.flatten().argmax()]
                l2_dists_proj = (Z_all - best_proj).norm(dim=-1)
                k_tr = min(baxus_state.n_cand, len(Z_all))
                tr_idx = l2_dists_proj.topk(k_tr, largest=False).indices
                cand_X_proj = Z_all[tr_idx]
                if args.raasp:
                    cand_X_proj = get_raasp_candidates(train_x_proj, train_y, cand_X_proj)

                if args.acq_fn == "ts":
                    cand_proj = MaxPosteriorSampling(model=model, replacement=False)(
                        cand_X_proj, num_samples=args.batch_size
                    ).detach()
                    acq_value = torch.zeros(1)
                else:
                    beta = args.ucb_beta if args.ucb_beta is not None else get_beta_t(itr, baxus_state.target_dim)
                    acq = build_so_acqf(args.acq_fn, model, train_x_proj, train_y, beta, cand_X_proj)
                    cand_proj, acq_value = optimize_acqf_discrete(
                        acq_function=acq, choices=cand_X_proj, q=args.batch_size
                    )
                    cand_proj = cand_proj.detach()

                # map projected candidate back to original embedding rows
                new_x_idx = torch.cdist(cand_proj, Z_all).argmin(dim=1)
                new_x = all_X[new_x_idx].to(dtype=dtype)

            else:
                # standard GP: warm-start MLE fit
                fit_gpytorch_mll(mll)

                rec_cand, _ = optimize_acqf_discrete(PosteriorMean(model), choices=all_X, q=1)
                rec_x = rec_cand.squeeze(0).detach()
                rec_true = prob.evaluate_true(rec_x.unsqueeze(0)).item()
                rec_x_all.append(rec_x.cpu().tolist())
                rec_true_all.append(rec_true)
                _best_rec_true_running = max(_best_rec_true_running, rec_true)
                best_rec_true_all.append(_best_rec_true_running)

                _ir = max(optimal_value - rec_true, 0.0)
                inference_regret_all.append(_ir)
                log_inference_regret_all.append(math.log(max(_ir, 1e-8)))
                best_inference_regret_all.append(max(optimal_value - _best_rec_true_running, 0.0))
                log_best_inference_regret_all.append(math.log(max(optimal_value - _best_rec_true_running, 1e-8)))

                # build candidate set: TuRBO narrows to L2-nearest neighbours of
                # the incumbent; RAASP then perturbs elite points within that pool
                if turbo_state is not None:
                    if args.turbo_pm_center:
                        with torch.no_grad():
                            pm = model.posterior(train_x).mean.flatten()
                        center = train_x[pm.argmax()]
                    else:
                        center = train_x[train_y.flatten().argmax()]
                    l2_dists = (all_X - center).norm(dim=-1)
                    k = min(turbo_state.n_cand, len(all_X))
                    topk_idx = l2_dists.topk(k, largest=False).indices
                    cand_X = all_X[topk_idx]
                else:
                    cand_X = all_X

                if args.raasp:
                    cand_X = get_raasp_candidates(train_x, train_y, cand_X)

                if args.acq_fn == "ts":
                    new_x = MaxPosteriorSampling(model=model, replacement=False)(
                        cand_X, num_samples=args.batch_size
                    ).detach().to(dtype=dtype)
                    acq_value = torch.zeros(1)
                else:
                    beta = args.ucb_beta if args.ucb_beta is not None else get_beta_t(itr, train_x.shape[-1])
                    acq = build_so_acqf(args.acq_fn, model, train_x, train_y, beta, cand_X)
                    candidate, acq_value = optimize_acqf_discrete(
                        acq_function=acq,
                        choices=cand_X,
                        q=args.batch_size,
                    )
                    new_x = candidate.detach().to(dtype=dtype)

            new_y = prob(new_x)

            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))

            best_y = train_y.max().item()
            best_y_all.append(best_y)
            seen_y.extend(new_y.cpu().tolist())
            candidates = np.concatenate((candidates, new_x.cpu().numpy()), axis=0)

            best_obs_x = train_x[train_y.flatten().argmax()].detach()
            best_obs_true = prob.evaluate_true(best_obs_x.unsqueeze(0)).item()
            best_obs_x_all.append(best_obs_x.cpu().tolist())
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(max(optimal_value - best_obs_true, 0.0))
            log_best_obs_regret_all.append(math.log(max(optimal_value - best_obs_true, 1e-8)))

            new_true_vals = prob.evaluate_true(new_x).flatten()
            _best_simple_true_running = max(_best_simple_true_running, new_true_vals.max().item())
            simple_regret_all.append(max(optimal_value - _best_simple_true_running, 0.0))
            log_simple_regret_all.append(math.log(max(optimal_value - _best_simple_true_running, 1e-8)))

            if turbo_state is not None:
                turbo_state = update_turbo_state(turbo_state, new_y, best_y_all[-2])
                turbo_ncand_all.append(turbo_state.n_cand)

            if baxus_state is not None:
                baxus_state = update_baxus_state(baxus_state, new_y, best_y_all[-2])
                baxus_target_dim_all.append(baxus_state.target_dim)
                baxus_ncand_all.append(baxus_state.n_cand)

            # warm-start model reinit for next iteration (skip for saasbo/baxus: they reinit at top)
            if not args.saasbo and not args.baxus and args.acq_fn != "random":
                mll, model = initialize_model(train_x, train_y, bounds_tensor, covar_module=covar_module, scaled_ls_init=args.msr or args.mle_scaled_init)

            if verbose:
                inf_str = f"  best_inf_regret={best_inference_regret_all[-1]:.4f}" if args.acq_fn != "random" else ""
                turbo_str = f"  tr_ncand={turbo_state.n_cand}" if turbo_state is not None else ""
                baxus_str = f"  baxus_t={baxus_state.target_dim}  baxus_ncand={baxus_state.n_cand}" if baxus_state is not None else ""
                print(
                    f"  itr {itr + 1}: acq={acq_value.detach().cpu().item():.4f}  "
                    f"best_y={best_y:.4f}  "
                    f"best_obs_regret={best_obs_regret_all[-1]:.4f}"
                    f"{inf_str}{turbo_str}{baxus_str}"
                )
            else:
                print(".", end="", flush=True)

        t1 = time.time()
        print(
            f"\nTrial {trial + 1} done in {t1 - t0:.1f}s — "
            f"final best_y={best_y_all[-1]:.4f}"
        )

        trial_result = {
            "trial": trial,
            "seed": seed,
            "time_seconds": t1 - t0,
            "best_y_all": best_y_all,
            "rec_x_all": rec_x_all,
            "rec_true_all": rec_true_all,
            "best_rec_true_all": best_rec_true_all,
            "inference_regret_all": inference_regret_all,
            "log_inference_regret_all": log_inference_regret_all,
            "best_inference_regret_all": best_inference_regret_all,
            "log_best_inference_regret_all": log_best_inference_regret_all,
            "best_obs_x_all": best_obs_x_all,
            "best_obs_true_all": best_obs_true_all,
            "best_obs_regret_all": best_obs_regret_all,
            "log_best_obs_regret_all": log_best_obs_regret_all,
            "simple_regret_all": simple_regret_all,
            "log_simple_regret_all": log_simple_regret_all,
            "candidates": candidates.tolist(),
            "seen_y": seen_y,
        }
        if args.turbo:
            trial_result["turbo_ncand_all"] = turbo_ncand_all
        if args.baxus:
            trial_result["baxus_target_dim_all"] = baxus_target_dim_all
            trial_result["baxus_ncand_all"] = baxus_ncand_all
        all_trial_results.append(trial_result)

    ucb_beta_fixed = args.ucb_beta if args.acq_fn == "ucb" else None
    results = {
        "problem": args.problem,
        "acq_fn": args.acq_fn,
        "turbo": args.turbo,
        "baxus": args.baxus,
        "saasbo": args.saasbo,
        "raasp": args.raasp,
        "msr": args.msr,
        "mle_scaled_init": args.mle_scaled_init,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "ucb_beta": ucb_beta_fixed,
        "initial_random_samples": args.initial_random_samples,
        "num_trials": args.trials,
        "trials": all_trial_results,
    }

    ucb_beta_tag = f"_beta{ucb_beta_fixed}" if ucb_beta_fixed is not None else ""
    batch_tag = f"_q{args.batch_size}" if args.batch_size > 1 else ""
    # surrogate tag: turbo / baxus / saasbo are mutually exclusive
    turbo_pm_tag = "_pmc" if args.turbo_pm_center else ""
    surrogate_tag = f"_turbo{turbo_pm_tag}" if args.turbo else ("_baxus" if args.baxus else ("_saasbo" if args.saasbo else ""))
    # msr tag takes priority over raasp; mlesi is independent (scaled init only, no raasp)
    raasp_tag = "_msr" if args.msr else ("_raasp" if args.raasp else "")
    scaled_init_tag = "_mlesi" if args.mle_scaled_init and not args.msr else ""

    results_dir = REPO_ROOT / "results" / "po"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        results_dir
        / f"{args.problem}_{args.acq_fn}{surrogate_tag}{raasp_tag}{scaled_init_tag}{ucb_beta_tag}{batch_tag}_{args.trials}trials_{args.iterations}iterations_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="BO baselines for PO problem")
    parser.add_argument(
        "--problem",
        type=str,
        default="po128",
        choices=["po128", "po256", "po512", "po768"],
        help="PO problem variant. Default: po128",
    )
    parser.add_argument(
        "--acq_fn",
        type=str,
        default="qnei",
        choices=["qnei", "ei", "ucb", "kg", "mes", "gibbon", "pes", "jes", "ts", "random"],
        help="Acquisition function: ei, qnei, ucb, kg, mes, gibbon, pes, jes, ts, random. Default: qnei",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of BO iterations (acqf optimizations). Total observations = iterations * batch_size. Default: 100",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Number of candidates queried per BO iteration (q). "
            "Note: analytic acq_fns (ei, ucb) only support batch_size=1. Default: 1"
        ),
    )
    parser.add_argument(
        "--ucb_beta",
        type=float,
        default=None,
        help=(
            "Fixed beta for UCB acquisition function. "
            "Suggested values: 0.1, 0.5, 1.0, 2.0. "
            "Default: None (uses Srinivas schedule via get_beta_t)."
        ),
    )
    parser.add_argument(
        "--initial_random_samples",
        type=int,
        default=10,
        help="Number of initial random samples. Default: 10",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent trials. Default: 1",
    )
    parser.add_argument(
        "--trial_offset",
        type=int,
        default=0,
        help="Offset added to trial index to get seed. Default: 0",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-iteration logs.",
    )
    parser.add_argument(
        "--matern",
        action="store_true",
        help="Use Matérn-2.5 kernel instead of the default RBF.",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Use TuRBO trust-region filtering (combinable with any acq_fn).",
    )
    parser.add_argument(
        "--turbo_pm_center",
        action="store_true",
        help="Use posterior mean argmax (instead of noisy observation argmax) as TuRBO trust-region center. Requires --turbo.",
    )
    parser.add_argument(
        "--raasp",
        action="store_true",
        help="Use RAASP candidate generation (combinable with --turbo). No lengthscale changes.",
    )
    parser.add_argument(
        "--mle_scaled_init",
        action="store_true",
        help="Use sqrt(d)/10 lengthscale init before MLE, without RAASP.",
    )
    parser.add_argument(
        "--msr",
        action="store_true",
        help="Use MSR: RAASP candidate generation + sqrt(d)/10 lengthscale init (Hvarfner et al. 2025).",
    )
    parser.add_argument(
        "--baxus",
        action="store_true",
        help=(
            "Use BAxUS: fit the GP in a random low-dimensional subspace that doubles on stagnation. "
            "Mutually exclusive with --turbo and --saasbo."
        ),
    )
    parser.add_argument(
        "--saasbo",
        action="store_true",
        help=(
            "Use SAASBO: fully Bayesian GP with SAAS horseshoe prior, fit via NUTS MCMC. "
            "Defaults to qnei acquisition; acqf flag is ignored for other analytic forms. "
            "Mutually exclusive with --turbo and --baxus."
        ),
    )
    args = parser.parse_args()

    if args.ucb_beta is not None and args.acq_fn != "ucb":
        parser.error("--ucb_beta is only applicable when --acq_fn ucb")
    
    surrogate_flags = sum([args.turbo, args.baxus, args.saasbo])

    if surrogate_flags > 1:
        parser.error("--turbo, --baxus, and --saasbo are mutually exclusive")
    
    if args.saasbo and (args.mle_scaled_init or args.msr):
        parser.error("--mle_scaled_init / --msr are not applicable with --saasbo (NUTS handles lengthscale uncertainty)")

    if args.turbo_pm_center and not args.turbo:
        parser.error("--turbo_pm_center requires --turbo")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
