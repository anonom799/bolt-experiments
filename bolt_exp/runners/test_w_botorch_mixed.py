import argparse
import json
import time
import math
import warnings

from bolt_exp import emulator_version

warnings.filterwarnings("ignore")

MAX_ACQF_RETRIES = 10

import numpy as np
import pandas as pd
import torch
import yaml
from functools import partial
from bolt import (
    DMCurriculum,
    HPO,
    HPOMultiFidelityModel,
    HPOMultiFidelityToken,
    LLMTestProblem,
)
from botorch.acquisition import (
    AcquisitionFunction,
    LogExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import expand_trace_observations, project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.exceptions.errors import OptimizationGradientError
from botorch.optim import optimize_acqf_mixed, optimize_acqf_mixed_alternating
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from huggingface_hub import hf_hub_download
from rich import print

from bolt_exp import REPO_ROOT

MC_SAMPLES = 128


def device_info(dev: str) -> dict:
    """Hardware the run executed on, recorded so timings stay comparable across machines."""
    if dev == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    elif dev == "mps":
        gpu_name = "mps"
    else:
        gpu_name = None
    return {"device": dev, "gpu_name": gpu_name}


def get_discrete_dims_dict(
    bounds: list[list[float]],
    discrete_inds: list[int],
    categorical_inds: list[int],
) -> dict[int, list[int]]:
    """Build discrete_dims dict for optimize_acqf_mixed_alternating.

    Maps each discrete/categorical dimension index to its list of allowed
    integer values derived from the original bounds.

    Args:
        bounds: Original bounds list, e.g. [[0,1],[2,4],[0,3],...].
        discrete_inds: Indices of discrete (ordinal) dimensions.
        categorical_inds: Indices of categorical dimensions.

    Returns:
        Dict mapping dimension index to list of allowed values.
    """
    discrete_dims = {}
    for i in list(discrete_inds) + list(categorical_inds):
        b_min, b_max = bounds[i]
        discrete_dims[i] = list(range(int(b_min), int(b_max) + 1))
    return discrete_dims


def generate_initial_data(prob: LLMTestProblem, rng, n=3, noise_std: float = 1e-2, device="cpu"):
    train_x = rng.random((n, prob.dim))

    print(prob._bounds)

    for i, (b_min, b_max) in enumerate(prob._bounds):
        train_x[:, i] = (train_x[:, i] * (b_max - b_min)) + b_min

        if i in prob.discrete_inds or i in prob.categorical_inds:
            train_x[:, i] = np.round(train_x[:, i])

    train_y = prob(torch.Tensor(train_x).to(device))

    return train_x, train_y


def initialize_model(
    prob: LLMTestProblem, train_x: torch.Tensor, train_y: torch.Tensor, state_dict=None,
    covar_module=None,
):
    if prob.categorical_inds:
        cont_kernel_factory = (
            (lambda batch_shape, ard_num_dims, active_dims: ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape, active_dims=active_dims),
                batch_shape=batch_shape,
            ))
            if covar_module is not None
            else None
        )
        model = MixedSingleTaskGP(
            train_x,
            train_y,
            cat_dims=prob.categorical_inds,
            cont_kernel_factory=cont_kernel_factory,
        ).to(train_x.device)
    else:
        model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=covar_module,
            input_transform=Normalize(
                d=train_x.shape[-1], bounds=torch.Tensor(prob._bounds).T
            ),
        ).to(train_x.device)

    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def observe_with_noise(
    prob, X: torch.Tensor, rng=None, noise_std: float = 1e-2
) -> torch.Tensor:
    y = prob(X)
    return y


def get_beta_t(n_step: int, n_var_dim: int) -> float:
    """Return beta_t for the current step according to UCB-GP

    Args:
        n_step (int): current step of optimization
        n_var_dim (int): number of variable dimensions

    Returns:
        float: beta_t
    """

    # beta = 50.0 * np.log(n_var_dim * (n_step + 1) ** 2 * np.pi**2 / 6.0 / 0.1) / 15.0
    # Loosely based on Srinivas et al. (2010), with edits: their Theorem 1 uses
    # |D|, the candidate set size, where this passes the input dimension. delta=0.1.
    beta = 2.0 * np.log(n_var_dim * (n_step + 1) ** 2 * np.pi**2 / 6.0 / 0.1)

    return beta


def convert_discrete_dims_to_fixed_feature_list(discrete_dims):
    """
    fixed_features_list (list[dict[int, float]] | None) – A list of maps {feature_index: value}. The i-th item represents the fixed_feature for the i-th optimization. If fixed_features_list is provided, optimize_acqf_mixed is invoked.

    discrete_dims (Mapping[int, Sequence[float]] | None) – A dictionary mapping indices of discrete and binary dimensions to a list of allowed values for that dimension.
    """

    fixed_features_list = []
    if discrete_dims is not None:
        from itertools import product

        discrete_indices = list(discrete_dims.keys())
        discrete_values = [discrete_dims[idx] for idx in discrete_indices]

        for combination in product(*discrete_values):
            fixed_feature = {
                idx: val for idx, val in zip(discrete_indices, combination)
            }
            fixed_features_list.append(fixed_feature)
    return fixed_features_list


# cost per fidelity (batched)
# for scaling acquisition functions
def cost_fn(X, scale):
    fidelity = X[..., -1]          # assuming last column is fidelity

    return fidelity * scale + 1


class CostScaledLogEI(AcquisitionFunction):
    def __init__(self, model, best_f, cost_fn, cost_scale = 1.0):
        # cost_scale: A hyperparameter scale for the fidelity cost function.
        #         Cost c(x) is computed as fidelity + cost_scale.
        #         In cost-aware multi-fidelity BO, the acquisition value of a candidate is a(x)/c(x).
        #         The larger the cost_scale, it implies that the acquisition value prefers lower fidelity candidates.
        #         See https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo for more details.

        super().__init__(model)
        self.log_ei = LogExpectedImprovement(model=model, best_f=best_f)
        self.cost_fn = cost_fn  # cost_fn: X -> cost
        self.cost_scale = cost_scale

    def forward(self, X):
        logei_val = self.log_ei(X)  # log(EI)
        cost = self.cost_fn(X, self.cost_scale).squeeze(-1)
        return logei_val - torch.log(cost)  # log(EI / cost)


class CostScaledUCB(AcquisitionFunction):
    def __init__(self, model, beta, cost_fn, cost_scale = 1.0):
        super().__init__(model)
        self.ucb = UpperConfidenceBound(model=model, beta=beta)
        self.cost_fn = cost_fn  # X -> cost
        self.cost_scale = cost_scale

    def forward(self, X):
        """
        X: batch_shape x q x d
        """
        ucb_val = self.ucb(X)  # shape: batch_shape
        cost = self.cost_fn(X, self.cost_scale).squeeze(-1)  # shape: batch_shape

        return ucb_val / cost


class CostScaledLogNEI(AcquisitionFunction):
    def __init__(self, model, X_baseline, cost_fn, cost_scale=1.0):
        super().__init__(model)
        self.log_nei = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )
        self.cost_fn = cost_fn
        self.cost_scale = cost_scale

    def forward(self, X):
        lognei_val = self.log_nei(X)  # log(NEI)
        cost = self.cost_fn(X, self.cost_scale).squeeze(-1)
        return lognei_val - torch.log(cost)  # log(NEI / cost)


class CostScaledPES(AcquisitionFunction):
    def __init__(self, model, optimal_inputs, maximize, cost_fn, cost_scale = 1.0):
        super().__init__(model)
        self.pes = qPredictiveEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            maximize=maximize,
        )
        self.cost_fn = cost_fn
        self.cost_scale = cost_scale

    def forward(self, X):
        pes_val = self.pes(X)
        cost = self.cost_fn(X, self.cost_scale).squeeze(-1)
        return pes_val / cost


def _make_candidate_set(
    bounds: torch.Tensor, discrete_dims: dict, num_samples: int = 512
) -> torch.Tensor:
    """Generate feasible candidates (Sobol + discrete rounding) for entropy-search acq fns."""
    dim = bounds.shape[1]
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True)

    X = sobol.draw(num_samples).to(bounds.device, dtype=bounds.dtype)
    X = bounds[0] + (bounds[1] - bounds[0]) * X

    for j, vals in discrete_dims.items():
        X[:, j] = torch.round(X[:, j]).clamp(min(vals), max(vals))

    return X


def thompson_sampling_candidate(
    model, bounds: torch.Tensor, discrete_dims: dict, num_candidates: int = 2048
) -> torch.Tensor:
    """Select next candidate via Thompson Sampling over a discrete candidate set.

    Draws one posterior sample function and returns its argmax over a Sobol
    candidate set with discrete dimensions rounded to valid values.

    Returns:
        Tensor of shape (1, d).
    """
    X_cand = _make_candidate_set(bounds, discrete_dims, num_samples=num_candidates)
    ts = MaxPosteriorSampling(model=model, replacement=False)
    candidate = ts(X_cand, num_samples=1)  # (1, d)
    return candidate


def get_optimal_inputs_outputs(
    model, bounds: torch.Tensor, discrete_dims: dict,
    num_samples: int = 64, raw_samples: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (optimal_inputs, optimal_outputs) for qJointEntropySearch.

    For each of num_samples posterior draws, returns the argmax location and its
    sample value. Both are consistent with the same draw, as required by JES.

    Returns:
        optimal_inputs: (num_samples, d)
        optimal_outputs: (num_samples, 1)
    """
    X_cand = _make_candidate_set(bounds, discrete_dims, num_samples=raw_samples)

    with torch.no_grad():
        post_samples = model.posterior(X_cand).rsample(torch.Size([num_samples]))  # (num_samples, raw_samples, 1)

    argmax_indices = post_samples.squeeze(-1).argmax(dim=1)  # (num_samples,)
    optimal_inputs = X_cand[argmax_indices]               # (num_samples, d)
    optimal_outputs = post_samples[torch.arange(num_samples), argmax_indices]  # (num_samples, 1)

    return optimal_inputs, optimal_outputs


def get_optimal_inputs(
    model, bounds, discrete_dims, num_samples=64, num_restarts=5, raw_samples=256
):
    """Draw Thompson samples and optimize them to get approximate optimal inputs.

    Args:
        model: Fitted GP model.
        bounds: (2, d) bounds tensor.
        discrete_dims: Dict mapping discrete dim index to allowed values.
        num_samples: Number of Thompson samples to draw.
        num_restarts: Restarts for optimizing each sample.
        raw_samples: Raw samples for initialization.

    Returns:
        Tensor of shape (num_samples, d) with approximate optimal inputs.
    """
    X_cand = _make_candidate_set(bounds, discrete_dims, num_samples=raw_samples)

    thompson_sampler = MaxPosteriorSampling(model=model, replacement=False)
    optimal_inputs = thompson_sampler(X_cand, num_samples=num_samples)
    return optimal_inputs


def build_acquisition_function(
    acq_fn_name: str,
    problem_name: str,
    model: SingleTaskGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    beta: float,
    cost_fn: callable = None,
    bounds: torch.Tensor = None,
    discrete_dims: dict = None,
):
    """Build the acquisition function from its name.

    Args:
        acq_fn_name: One of 'ucb', 'ei', 'qnei', 'mes', 'gibbon', 'kg', 'pes'.
        problem_name: Name of the problem. This indicates if multi-fidelity acq is needed.
        model: The fitted GP model.
        train_x: Training inputs (n, d). Required for qnei.
        train_y: Training targets (used to derive best_f for EI).
        beta: Exploration parameter for UCB.
        cost_fn: Callable X -> cost, e.g. prob.cost. Required for multi-fidelity problems.
        bounds: (2, d) bounds tensor. Required for PES, MES, GIBBON.
        discrete_dims: Discrete dims dict. Required for PES, MES, GIBBON.

    Returns:
        A BoTorch AcquisitionFunction instance.
    """
    is_mf = problem_name in ["hpo_fd_step", "hpo_fd_model"]
    if acq_fn_name == "ucb":
        if is_mf:
            return CostScaledUCB(
                model=model, beta=beta, cost_fn=cost_fn, cost_scale=args.cost_scale
            )
        return UpperConfidenceBound(model=model, beta=beta)
    elif acq_fn_name == "ei":
        if is_mf:
            return CostScaledLogEI(
                model=model, best_f=train_y.max(), cost_fn=cost_fn, cost_scale=args.cost_scale
            )
        return LogExpectedImprovement(model=model, best_f=train_y.max())
    elif acq_fn_name == "qnei":
        if is_mf:
            return CostScaledLogNEI(
                model=model, X_baseline=train_x, cost_fn=cost_fn, cost_scale=args.cost_scale
            )
        return qLogNoisyExpectedImprovement(
            model=model, X_baseline=train_x, prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )
    elif acq_fn_name == "mes":
        candidate_set = _make_candidate_set(bounds, discrete_dims or {}, num_samples=512)
        return qMaxValueEntropy(model=model, candidate_set=candidate_set)
    elif acq_fn_name == "gibbon":
        candidate_set = _make_candidate_set(bounds, discrete_dims or {}, num_samples=512)
        return qLowerBoundMaxValueEntropy(model=model, candidate_set=candidate_set)
    elif acq_fn_name == "kg":
        if is_mf:
            raise NotImplementedError("Cost-scaled KG is not implemented yet.")
        return qKnowledgeGradient(model=model, num_fantasies=8)
    elif acq_fn_name in ("mfkg", "mfmes", "mfgibbon"):
        if not is_mf:
            raise ValueError(f"{acq_fn_name} requires a multi-fidelity problem (hpo_fd_step or hpo_fd_model).")
        fidelity_dim = train_x.shape[-1] - 1
        target_fidelities = {fidelity_dim: 1.0}
        cost_model = AffineFidelityCostModel(
            fidelity_weights={fidelity_dim: args.cost_scale}, fixed_cost=1.0
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        project = partial(
            project_to_target_fidelity, target_fidelities=target_fidelities, d=train_x.shape[-1]
        )
        if acq_fn_name == "mfkg":
            expand = partial(expand_trace_observations, fidelity_dims=[fidelity_dim], num_trace_obs=4)
            return qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=8,
                target_fidelities=target_fidelities,
                cost_aware_utility=cost_aware_utility,
                project=project,
                expand=expand,
            )
        candidate_set = _make_candidate_set(bounds, discrete_dims or {}, num_samples=512)
        mf_entropy_cls = qMultiFidelityMaxValueEntropy if acq_fn_name == "mfmes" else qMultiFidelityLowerBoundMaxValueEntropy
        return mf_entropy_cls(
            model=model,
            candidate_set=candidate_set,
            project=project,
            cost_aware_utility=cost_aware_utility,
        )
    elif acq_fn_name == "jes":
        optimal_inputs, optimal_outputs = get_optimal_inputs_outputs(model, bounds, discrete_dims or {})
        return qJointEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            num_samples=MC_SAMPLES,
        )
    elif acq_fn_name == "pes":
        optimal_inputs = get_optimal_inputs(model, bounds, discrete_dims or {})
        if is_mf:
            return CostScaledPES(
                model=model,
                optimal_inputs=optimal_inputs,
                maximize=True,
                cost_fn=cost_fn,
                cost_scale=args.cost_scale
            )
        return qPredictiveEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            maximize=True,
        )
    elif acq_fn_name == "ts":
        raise ValueError("Thompson Sampling does not use an acquisition function object; handle it directly in the BO loop.")
    else:
        raise ValueError(f"Unknown acquisition function: {acq_fn_name}")


def main(args):

    print("hpo problem type:", args.problem)
    print("using acquisition function: ", args.acq_fn)
    print("number of iterations: ", args.iterations)
    print("initial random samples: ", args.initial_random_samples)
    print("trials: ", args.trials)
    print("kernel:", "matern-2.5" if args.matern else "default (rbf)")

    verbose = args.verbose
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    dev_info = device_info(dev)
    print("device:", dev_info["device"], "|", dev_info["gpu_name"])

    dtype = torch.float32 if dev in ("mps", "cpu") else torch.double
    torch.set_default_dtype(dtype)

    # --noise_std unset: let each problem class use its own default noise std
    noise_kwargs = {} if args.noise_std is None else {"noise_std": args.noise_std}

    if args.problem == "hpo":
        prob = HPO(negate=False, **noise_kwargs)
    elif args.problem == "hpo_fd_step":
        prob = HPOMultiFidelityToken(negate=False, **noise_kwargs)
    elif args.problem == "hpo_fd_model":
        prob = HPOMultiFidelityModel(negate=False, **noise_kwargs)
    elif args.problem == "dm_curriculum":
        prob = DMCurriculum(negate=False, **noise_kwargs)
    else:
        raise ValueError(f"Unknown problem {args.problem}")

    noise_std = prob.noise_std
    print("noise std:", noise_std)

    prob.to(device=dev, dtype=dtype)

    covar_module = (
        ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=prob.dim)) if args.matern else None
    )

    discrete_dims = get_discrete_dims_dict(
        prob._bounds, prob.discrete_inds, prob.categorical_inds
    )
    print("discrete dims: ", discrete_dims)

    BO_iterations = args.iterations
    all_trial_results = []

    # --seed_offset lets seeds be run as separate parallel processes (see --help)
    seed_base = 0 if args.seed_offset is None else args.seed_offset

    for trial in range(args.trials):
        seed = trial + seed_base
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.trials}  (seed={seed})")
        print(f"{'='*60}")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        print("generating initial data...")
        train_x, train_y = generate_initial_data(
            prob, rng, n=args.initial_random_samples, device=dev
        )

        best_y_all = [max(train_y).item()]
        candidates = train_x
        seen_y = train_y.flatten().tolist()

        train_x = torch.Tensor(train_x).to(device=dev, dtype=dtype)
        train_y = torch.Tensor(train_y).to(device=dev, dtype=dtype)

        _init_true_vals = prob.evaluate_true(train_x).flatten()
        _best_simple_true_running = _init_true_vals.max().item()
        simple_regret_all = [max(prob._optimal_value - _best_simple_true_running, 0.0)]
        log_simple_regret_all = [math.log(max(prob._optimal_value - _best_simple_true_running, 1e-8))]

        _init_best_i = train_y.flatten().argmax()
        _init_best_x = train_x[_init_best_i].detach()
        _init_best_true = prob.evaluate_true(_init_best_x.unsqueeze(0)).item()
        best_obs_x_all = [_init_best_x.cpu().tolist()]
        best_obs_true_all = [_init_best_true]
        best_obs_regret_all = [max(prob._optimal_value - _init_best_true, 0.0)]
        log_best_obs_regret_all = [math.log(max(prob._optimal_value - _init_best_true, 1e-8))]
        rec_x_all, rec_true_all, best_rec_true_all = [], [], []
        inference_regret_all, log_inference_regret_all = [], []
        best_inference_regret_all, log_best_inference_regret_all = [], []
        _best_rec_true_running = -float("inf")

        mll, model = initialize_model(prob, train_x, train_y, covar_module=covar_module)

        bounds = torch.Tensor(prob._bounds).T.to(device=train_x.device, dtype=dtype)

        is_multifidelity = args.problem in ["hpo_fd_step", "hpo_fd_model"]

        # Initial recommendation (based on max posterior mean) before the BO loop (index 0)
        fit_gpytorch_mll(mll)
        _rec_x_init, _ = optimize_acqf_mixed_alternating(
            PosteriorMean(model), bounds=bounds, q=1,
            num_restarts=5, raw_samples=256,
            discrete_dims=discrete_dims,
        )
        _rec_x_init = _rec_x_init.squeeze(0).detach()

        for j in discrete_dims:
            _rec_x_init[j] = torch.round(_rec_x_init[j])

        _rec_true_init = prob.evaluate_true(_rec_x_init.unsqueeze(0)).item()

        rec_x_all.append(_rec_x_init.cpu().tolist())
        rec_true_all.append(_rec_true_init)

        # inference regret is based on recommended point following posterior mean
        inference_regret_all.append(prob._optimal_value - _rec_true_init)
        log_inference_regret_all.append(math.log(max(prob._optimal_value - _rec_true_init, 1e-8)))
        _best_rec_true_running = _rec_true_init
        best_rec_true_all.append(_best_rec_true_running)
        best_inference_regret_all.append(max(prob._optimal_value - _best_rec_true_running, 0.0))
        log_best_inference_regret_all.append(math.log(max(prob._optimal_value - _best_rec_true_running, 1e-8)))

        t0 = time.time()
        budget = 0.0
        budget_all = []
        itr = 0

        while budget < BO_iterations:
            fit_gpytorch_mll(mll)

            # max of posterior mean
            rec_x_cand, _ = optimize_acqf_mixed_alternating(
                PosteriorMean(model), bounds=bounds, q=1,
                num_restarts=5, raw_samples=256,
                discrete_dims=discrete_dims,
            )
            rec_x_cand = rec_x_cand.squeeze(0).detach()
            for j in discrete_dims:
                rec_x_cand[j] = torch.round(rec_x_cand[j])

            rec_true = prob.evaluate_true(rec_x_cand.unsqueeze(0)).item()

            rec_x_all.append(rec_x_cand.cpu().tolist())
            rec_true_all.append(rec_true)

            # calculate inference regret
            inference_regret_all.append(prob._optimal_value - rec_true)
            log_inference_regret_all.append(math.log(max(prob._optimal_value - rec_true, 1e-8)))
            _best_rec_true_running = max(_best_rec_true_running, rec_true)
            best_rec_true_all.append(_best_rec_true_running)
            best_inference_regret_all.append(max(prob._optimal_value - _best_rec_true_running, 0.0))
            log_best_inference_regret_all.append(math.log(max(prob._optimal_value - _best_rec_true_running, 1e-8)))

            if args.acq_fn == "ts":
                candidate = thompson_sampling_candidate(model, bounds, discrete_dims, num_candidates=args.ts_num_candidates)
                acq_value = torch.zeros(1, device=bounds.device, dtype=bounds.dtype)
            else:
                beta = args.ucb_beta if args.ucb_beta is not None else get_beta_t(itr, prob.dim)
                acq = build_acquisition_function(
                    args.acq_fn,
                    args.problem,
                    model,
                    train_x,
                    train_y,
                    beta,
                    cost_fn=cost_fn,
                    bounds=bounds,
                    discrete_dims=discrete_dims,
                )
                if args.acq_fn == "ucb":
                    print(f"beta at iteration {itr}: ", beta)

                if args.acq_fn in ("kg", "mfkg"):
                    fixed_features_list = convert_discrete_dims_to_fixed_feature_list(
                        discrete_dims
                    )

                    candidate, acq_value = optimize_acqf_mixed(
                        acq,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=50,
                        fixed_features_list=fixed_features_list,
                    )
                else:
                    for _retry in range(MAX_ACQF_RETRIES):
                        try:
                            candidate, acq_value = optimize_acqf_mixed_alternating(
                                acq,
                                bounds=bounds,
                                q=1,
                                num_restarts=10,
                                raw_samples=50,
                                discrete_dims=discrete_dims,
                            )
                            break
                        except OptimizationGradientError as e:
                            print(f"optimize_acqf_mixed_alternating attempt {_retry + 1}/{MAX_ACQF_RETRIES} failed with gradient error: {e}")
                            if _retry == MAX_ACQF_RETRIES - 1:
                                raise

            if args.problem in ["hpo_fd_step", "hpo_fd_model"]:
                print(
                    "fidelity value of the candidate: ", candidate[..., -1].item()
                )

                # Compare acquisition values at low/high fidelity for the same base candidate.
                # Assumes fidelity is the last dimension in X and bounded in prob._bounds[-1].
                cand_low = candidate.detach().clone()
                cand_high = candidate.detach().clone()

                fid_low = float(prob._bounds[-1][0])
                fid_high = float(prob._bounds[-1][1])

                cand_low[..., -1] = fid_low
                cand_high[..., -1] = fid_high

                with torch.no_grad():
                    acq_low = acq(cand_low).detach().cpu().item()
                    acq_high = acq(cand_high).detach().cpu().item()

                print(
                    f"acq(low fid={fid_low:.0f})={acq_low:.6f}, "
                    f"acq(high fid={fid_high:.0f})={acq_high:.6f}"
                )

            new_x = candidate.detach()

            for j in discrete_dims:
                new_x[:, j] = torch.round(new_x[:, j])

            if is_multifidelity:
                budget_cost = prob.cost(new_x).item()
            else:
                budget_cost = 1.0
            budget += budget_cost
            if is_multifidelity:
                budget_all.append(budget)

            new_y = prob(new_x)

            _new_true = prob.evaluate_true(new_x).flatten().max().item()
            _best_simple_true_running = max(_best_simple_true_running, _new_true)
            simple_regret_all.append(max(prob._optimal_value - _best_simple_true_running, 0.0))
            log_simple_regret_all.append(math.log(max(prob._optimal_value - _best_simple_true_running, 1e-8)))

            new_x = new_x.to(train_x.device, dtype=train_x.dtype)
            new_y = torch.Tensor(new_y).to(train_y.device, dtype=train_y.dtype)

            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))

            # get best x based on noisy y
            best_i = torch.argmax(train_y)
            best_value = train_y[best_i].detach().cpu().item()
            best_x = train_x[best_i].detach().cpu().numpy()

            best_obs_x_t = train_x[best_i].detach()
            best_obs_true = prob.evaluate_true(best_obs_x_t.unsqueeze(0)).item()
            best_obs_x_all.append(best_obs_x_t.cpu().tolist())
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(max(prob._optimal_value - best_obs_true, 0.0))
            log_best_obs_regret_all.append(math.log(max(prob._optimal_value - best_obs_true, 1e-8)))

            mll, model = initialize_model(prob, train_x, train_y, covar_module=covar_module)

            if verbose:
                print(
                    f"  itr {itr + 1} (budget={budget:.2f}/{BO_iterations}): candidate={np.array2string(new_x.detach().cpu().numpy(), precision=2, suppress_small=True)}  acq={acq_value.detach().cpu().item():.4f}"
                )
                print(
                    f"    best y={best_value:.3g}  best x={np.array2string(best_x, precision=2, suppress_small=True)}"
                )
                print("-" * 50)
            else:
                print(".", end="")

            best_y_all.append(float(best_value))
            candidates = np.concatenate(
                (candidates, new_x.detach().cpu().numpy()), axis=0
            )
            seen_y.append(float(new_y.detach().cpu().item()))
            itr += 1

        t1 = time.time()
        trial_best_value = best_y_all[-1]
        print(
            f"\nTrial {trial + 1} done in {t1 - t0:.1f}s — "
            f"best standardized={trial_best_value:.3g}  "
        )

        if is_multifidelity:
            fid_values = candidates[:, -1]
            n_low = int(np.sum(np.abs(fid_values) < 0.05))
            n_high = int(np.sum(np.abs(fid_values - 1.0) < 0.05))
            n_mid = len(fid_values) - n_low - n_high
            print(
                f"  Fidelity breakdown: low(0)={n_low}, mid=(0,1)={n_mid}, high(1)={n_high}  "
                f"(total={len(fid_values)})"
            )

        all_trial_results.append(
            {
                "trial": trial,
                "seed": seed,
                "time_seconds": t1 - t0,
                **dev_info,
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
                **({"budget_all": budget_all} if is_multifidelity else {}),
            }
        )

    results = {
        "problem": args.problem,
        "acq_fn": args.acq_fn,
        "iterations": BO_iterations,
        "initial_random_samples": args.initial_random_samples,
        "noise_std": noise_std,
        "num_trials": args.trials,
        **dev_info,
        "emulator_versions": emulator_version.emulator_versions_for(prob),
        "trials": all_trial_results,
    }
    if args.problem in ("hpo_fd_step", "hpo_fd_model"):
        results["cost_scale"] = args.cost_scale
    if args.acq_fn == "ucb" and args.ucb_beta is not None:
        results["ucb_beta"] = args.ucb_beta
    info_tag = f"_{args.info}" if args.info else ""
    cost_tag = f"_cost{args.cost_scale:g}" if args.problem in ("hpo_fd_step", "hpo_fd_model") else ""
    beta_tag = f"_beta{args.ucb_beta:g}" if args.acq_fn == "ucb" and args.ucb_beta is not None else ""
    # --results_folder names the subfolder outright; otherwise it is hpo{--folder_prefix}
    folder = args.results_folder or f"hpo{args.folder_prefix}"
    results_dir = REPO_ROOT / "results" / folder
    results_dir.mkdir(parents=True, exist_ok=True)

    # per-seed shards get their own filename so parallel runs never collide
    seed_tag = "" if args.seed_offset is None else f"_seed{args.seed_offset}"
    output_file = (
        results_dir
        / f"{args.problem}_{args.acq_fn}_{args.trials}trials_{args.iterations}iterations{cost_tag}{beta_tag}{info_tag}{seed_tag}_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="BO with mixed search space")
    parser.add_argument(
        "--problem",
        type=str,
        default="hpo",
        choices=["hpo", "hpo_fd_step", "hpo_fd_model", "dm_curriculum"],
        help="HPO problem to set. Default: hpo",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=None,
        help=(
            "Seed for the first trial. Writes a separate _seed<N>_results.json shard so "
            "seeds can be run as parallel processes and combined afterwards with "
            "scripts/merge_seed_runs.py. Default: 0"
        ),
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
        help="Suffix appended to the results subfolder name, e.g. 'hpo{folder_prefix}'. Default: '' (results/hpo)",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default=None,
        help="Results subfolder name, used verbatim (e.g. 'dm_mean_std' -> results/dm_mean_std). "
        "Overrides --folder_prefix. Default: None (use hpo{folder_prefix})",
    )
    parser.add_argument(
        "--acq_fn",
        type=str,
        default="ucb",
        choices=["ucb", "ei", "qnei", "mes", "gibbon", "kg", "jes", "pes", "mfkg", "mfmes", "mfgibbon", "ts"],
        help="Acquisition function to use: ucb (Upper Confidence Bound), "
        "ei (Expected Improvement), qnei (qLogNoisyExpectedImprovement), "
        "mes (Max Value Entropy Search), gibbon (GIBBON/LB-MES), "
        "kg (Knowledge Gradient), jes (Joint Entropy Search), "
        "pes (Predictive Entropy Search), "
        "mfkg (qMultiFidelityKnowledgeGradient), mfmes (qMultiFidelityMaxValueEntropy), "
        "mfgibbon (qMultiFidelityLowerBoundMaxValueEntropy), "
        "ts (Thompson Sampling). Default: ucb",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of BO iterations to run. Default: 100",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print detailed logs during optimization.",
    )

    parser.add_argument(
        "--initial_random_samples",
        type=int,
        default=10,
        help="Number of initial random samples to generate before starting BO iterations. Default: 10",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent trials to run. Each trial uses a different random seed (0, 1, ...). Default: 1",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="Optional tag appended to the output filename for identification.",
    )
    parser.add_argument(
        "--cost_scale",
        type=float,
        default=1.0,
        help="Additive cost scale for multi-fidelity problems. "
        "Cost = fidelity + cost_scale. Recommend 100 for hpo_fd_step, 10 for hpo_fd_model.",
    )
    parser.add_argument(
        "--matern",
        action="store_true",
        help="Use Matérn-2.5 kernel instead of the default RBF.",
    )
    parser.add_argument(
        "--ucb_beta",
        type=float,
        default=None,
        help="Fixed beta for UCB. If not set, uses the Srinivas schedule (get_beta_t).",
    )
    parser.add_argument(
        "--ts_num_candidates",
        type=int,
        default=2048,
        help="Num candidates for TS. Default is 2048.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
