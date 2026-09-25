# Multi-objective and single-objective Bayesian optimization for DM curriculum problems.
#
# Usage:
#   python test_w_botorch_dm.py
#   python test_w_botorch_dm.py --problem dm_curriculum --acq_fn ei
#   python test_w_botorch_dm.py --problem dm_curriculum --acq_fn ucb --ucb_beta 0.5
#   python test_w_botorch_dm.py --problem dm_curriculum_mo --acq_fn qnehvi
#   python test_w_botorch_dm.py --problem dm_curriculum_mo --iterations 50 --trials 5 --verbose
#   python test_w_botorch_dm.py --problem dm_curriculum_mo --trials 2 --trial_offset 5
#
# Arguments:
#   --problem         {dm_curriculum, dm_curriculum_mo, dm_curriculum_heteroscedastic}  (default: dm_curriculum_mo)
#   --noise_std       Observation noise std for the emulator; ignored for heteroscedastic problems  (default: problem class default)
#   --acq_fn          Acquisition function (SO): ei, qnei, ucb, kg, mes, gibbon, pes, jes, ts;
#                     (MO): qnehvi, qparego, qhvkg, jes_mo, mes_mo, pes_mo; (both): random. Default: auto
#   --ucb_beta        Fixed beta for UCB (SO only). Options: 0.1, 0.5, 1.0, 2.0.
#                     Default: None (uses Srinivas schedule via get_beta_t)
#   --known_noise     Use SingleTaskGP with known noise variances (dm_curriculum_heteroscedastic only);
#                     conditions on known noise variances from the emulator. Default: SingleTaskGP (inferred noise)
#   --iterations      Number of BO iterations; total observations = iterations * batch_size (default: 100)
#   --batch_size      Candidates per BO iteration; analytic acq_fns (ei, ucb) require 1 (default: 1)
#   --initial_random_samples  Number of initial Dirichlet samples         (default: 10)
#   --trials          Number of independent runs with different seeds     (default: 1)
#   --trial_offset    Seed offset; also used to extend existing results. When > 0, loads the
#                     <trial_offset>trials results file, validates config matches, and writes a
#                     new merged file with <trial_offset + trials> total trials. (default: 0)
#   --verbose         Print per-iteration logs
#
# Output:
#   <problem>_<acq_fn>_<N>trials_<iterations>iterations_results.json
#   where N = trial_offset + trials (merged file written fresh; original file untouched)
#
# Heteroscedastic baselines (beyond --hetero_gp):
#   - SAASBO (sparse axis-aligned subspace BO): effective in high-D, low-data regimes with
#     heteroscedastic noise; use SaasBOModel from botorch.models.fully_bayesian.
#   - Input warping (Kumaraswamy): warp inputs before GP fitting to capture
#     non-stationary lengthscale variation correlated with noise; see
#     botorch.models.transforms.input.Warp.
#   - Noise-weighted EI / WNEHVI: weight acquisition by inverse noise variance to
#     focus queries on low-noise regions; implement via a custom MCAcquisitionFunction.
#   - Random baseline: uniform Dirichlet sampling (no model) for calibration.

import argparse
import json
import math
from pathlib import Path
import time

from bolt_exp import emulator_version

from botorch.exceptions.errors import CandidateGenerationError
from botorch.acquisition import (
    LogExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy, qMaxValueEntropy
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qHypervolumeKnowledgeGradient
from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
from botorch.acquisition.multi_objective.max_value_entropy_search import qLowerBoundMultiObjectiveMaxValueEntropySearch
from botorch.acquisition.multi_objective.predictive_entropy_search import qMultiObjectivePredictiveEntropySearch
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import numpy as np
from rich import print
import torch

from bolt import (
    DMCurriculum,
    DMCurriculumMO,
    DMCurriculumHet,
)
from bolt_exp.mlhgp import fit_mlhgp

from bolt_exp import REPO_ROOT

def device_info(dev: str) -> dict:
    """Hardware the run executed on, recorded so timings stay comparable across machines."""
    if dev == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    elif dev == "mps":
        gpu_name = "mps"
    else:
        gpu_name = None
    return {"device": dev, "gpu_name": gpu_name}


MO_PROBLEMS = {"dm_curriculum_mo"}
SO_PROBLEMS = {"dm_curriculum", "dm_curriculum_heteroscedastic"}

MC_SAMPLES = 128


def fit_mlhgp_so_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    n_em_iter: int = 5,
    initial_noise_var: torch.Tensor | None = None,
    covar_module=None,
) -> tuple:
    """Fit an MLHGP for a single-objective problem and return (ExactMarginalLogLikelihood, SingleTaskGP)."""
    gp, _ = fit_mlhgp(train_x, train_y, bounds, n_em_iter=n_em_iter, warm_start_noise_var=initial_noise_var, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(train_x)
    return mll, gp


def fit_mlhgp_mo_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    n_em_iter: int = 5,
    initial_noise_var: torch.Tensor | None = None,
    covar_module=None,
) -> tuple:
    """Fit one MLHGP per objective and return a (SumMarginalLogLikelihood, ModelListGP)."""

    signal_gps = []
    for i in range(train_y.shape[-1]):
        warm = initial_noise_var[:, i : i + 1] if initial_noise_var is not None else None

        gp, _ = fit_mlhgp(train_x, train_y[:, i : i + 1], bounds, n_em_iter=n_em_iter, warm_start_noise_var=warm, covar_module=covar_module)
        
        signal_gps.append(gp)

    model = ModelListGP(*signal_gps)
    mll = SumMarginalLogLikelihood(model.likelihood, model).to(train_x)

    return mll, model


_ENTROPY_SEARCH_ACQFNS = {"jes_mo", "mes_mo"}
MAX_ACQF_RETRIES = 3


def get_simplex_equality_constraints(
    groups: list[list[int]],
    dtype=torch.double,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Build equality constraints so each simplex group sums to 1.

    Args:
        groups: List of index groups where each group must sum to 1.
                E.g. [[0, 1, 2], [3, 4, 5]] for two simplex groups.

    Returns:
        List of (indices, coefficients, rhs) tuples for botorch.
    """
    constraints = []
    for group in groups:
        indices = torch.tensor(group, dtype=torch.long)
        coefficients = torch.ones(len(group), dtype=dtype)
        constraints.append((indices, coefficients, 1.0))

    return constraints


def generate_initial_data(
    prob,
    rng,
    n: int = 10,
    simplex_groups: list[list[int]] | None = None,
    device: str = "cpu",
    dtype = torch.double,
):
    """Generate initial data using Dirichlet sampling for simplex constraints.

    Args:
        prob: The problem instance.
        rng: NumPy random generator.
        n: Number of initial samples.
        simplex_groups: List of index groups for simplex constraints.
        device: Torch device string.

    Returns:
        Tuple of (train_x as np.ndarray of shape (n, d), train_y as torch.Tensor of shape (n, m)).
    """
    dim = prob.dim
    train_x = np.zeros((n, dim))

    simplex_indices: set[int] = set()
    if simplex_groups is not None:
        simplex_indices = {idx for group in simplex_groups for idx in group}

        for group in simplex_groups:
            alpha = np.ones(len(group))
            samples = rng.dirichlet(alpha, size=n)

            for k, idx in enumerate(group):
                train_x[:, idx] = samples[:, k]

    for i, (b_min, b_max) in enumerate(prob._bounds):
        if i not in simplex_indices:
            train_x[:, i] = rng.uniform(b_min, b_max, size=n)

    train_y = prob(torch.tensor(train_x, dtype=dtype).to(device))
    return train_x, train_y


def initialize_so_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    state_dict=None,
    covar_module=None,
):
    """Initialize a SingleTaskGP for single-objective problems.

    Args:
        train_x: Input tensor of shape (n, d).
        train_y: Output tensor of shape (n, 1).
        bounds: Bounds tensor of shape (2, d).
        state_dict: Optional state dict to load.

    Returns:
        Tuple of (ExactMarginalLogLikelihood, SingleTaskGP).
    """
    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=covar_module,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


def initialize_hetero_so_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor,
    bounds: torch.Tensor,
    state_dict=None,
    covar_module=None,
):
    """Initialize a SingleTaskGP (known noise) for single-objective problems.

    Args:
        train_x: Input tensor of shape (n, d).
        train_y: Output tensor of shape (n, 1).
        train_yvar: Noise variance tensor of shape (n, 1).
        bounds: Bounds tensor of shape (2, d).
        state_dict: Optional state dict to load.

    Returns:
        Tuple of (ExactMarginalLogLikelihood, SingleTaskGP).
    """
    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        covar_module=covar_module,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


def initialize_mo_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    state_dict=None,
    covar_module=None,
):
    """Initialize a ModelListGP with one SingleTaskGP per objective.

    Args:
        train_x: Input tensor of shape (n, d).
        train_y: Output tensor of shape (n, m).
        bounds: Bounds tensor of shape (2, d).
        state_dict: Optional state dict to load.

    Returns:
        Tuple of (SumMarginalLogLikelihood, ModelListGP).
    """
    models = [
        SingleTaskGP(
            train_x,
            train_y[:, i : i + 1],
            covar_module=covar_module,
            input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        )
        for i in range(train_y.shape[-1])
    ]

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model).to(train_x)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


def initialize_hetero_mo_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor,
    bounds: torch.Tensor,
    state_dict=None,
    covar_module=None,
):
    """Initialize a ModelListGP with one SingleTaskGP (known noise) per objective.

    Uses known input-dependent noise variances from the problem's noise emulator.

    Args:
        train_x: Input tensor of shape (n, d).
        train_y: Output tensor of shape (n, m).
        train_yvar: Noise variance tensor of shape (n, m) — noise_std squared, clamped positive.
        bounds: Bounds tensor of shape (2, d).
        state_dict: Optional state dict to load.

    Returns:
        Tuple of (SumMarginalLogLikelihood, ModelListGP).
    """
    models = [
        SingleTaskGP(
            train_x,
            train_y[:, i : i + 1],
            train_Yvar=train_yvar[:, i : i + 1],
            covar_module=covar_module,
            input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        )
        for i in range(train_y.shape[-1])
    ]

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model).to(train_x)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


def compute_ref_point(train_y: torch.Tensor, slack: float = 0.1) -> list[float]:
    """Compute a reference point slightly below the worst observed values.

    Args:
        train_y: Observed objectives of shape (n, m).
        slack: Fraction of the objective range to subtract from the minimum.

    Returns:
        Reference point as a list of floats of length m.
    """
    y_min = train_y.min(dim=0).values
    y_max = train_y.max(dim=0).values
    ref = y_min - slack * (y_max - y_min)

    return ref.tolist()


def get_beta_t(n_step: int, n_var_dim: int) -> float:
    # Loosely based on Srinivas et al. (2010), with edits: their Theorem 1 uses
    # |D|, the candidate set size, where this passes the input dimension. delta=0.1.
    return 2.0 * np.log(n_var_dim * (n_step + 1) ** 2 * np.pi**2 / 6.0 / 0.1)

def get_optimal_inputs_simplex(
    model,
    simplex_groups: list[list[int]],
    bounds: torch.Tensor,
    num_samples: int = 64,
    raw_samples: int = 256,
    rng=None,
) -> torch.Tensor:
    """Draw Thompson samples from feasible simplex candidates to approximate optimal inputs for PES."""
    if rng is None:
        rng = np.random.default_rng()

    dim = bounds.shape[1]
    X_cand = np.zeros((raw_samples, dim))
    simplex_indices = {idx for group in simplex_groups for idx in group}

    for group in simplex_groups:
        # Uniform Dirichlet: symmetric alpha=1 gives uniform coverage of the simplex face
        samples = rng.dirichlet(np.ones(len(group)), size=raw_samples)
        for k, idx in enumerate(group):
            X_cand[:, idx] = samples[:, k]

    for i in range(dim):
        if i not in simplex_indices:
            # Box-uniform for free (non-simplex) dimensions
            X_cand[:, i] = rng.uniform(bounds[0, i].item(), bounds[1, i].item(), size=raw_samples)

    X_cand_t = torch.tensor(X_cand, dtype=bounds.dtype, device=bounds.device)
    # Select num_samples candidates with highest posterior sample values (Thompson sampling)
    thompson_sampler = MaxPosteriorSampling(model=model, replacement=False)
    optimal_inputs = thompson_sampler(X_cand_t, num_samples=num_samples)

    return optimal_inputs


def get_optimal_inputs_outputs_simplex(
    model,
    simplex_groups: list[list[int]],
    bounds: torch.Tensor,
    num_samples: int = 64,
    raw_samples: int = 256,
    rng=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (optimal_inputs, optimal_outputs) for qJointEntropySearch.

    For each of num_samples posterior draws, returns the argmax location and its
    sample value (not the posterior mean). Both outputs are consistent with the
    same draw, which is required for JES to be well-specified.

    Returns:
        optimal_inputs: (num_samples, d)
        optimal_outputs: (num_samples, 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = bounds.shape[1]
    X_cand = np.zeros((raw_samples, dim))
    simplex_indices = {idx for group in simplex_groups for idx in group}

    for group in simplex_groups:
        samples = rng.dirichlet(np.ones(len(group)), size=raw_samples)
        for k, idx in enumerate(group):
            X_cand[:, idx] = samples[:, k]

    for i in range(dim):
        if i not in simplex_indices:
            X_cand[:, i] = rng.uniform(bounds[0, i].item(), bounds[1, i].item(), size=raw_samples)

    X_cand_t = torch.tensor(X_cand, dtype=bounds.dtype, device=bounds.device)

    with torch.no_grad():
        post_samples = model.posterior(X_cand_t).rsample(torch.Size([num_samples]))  # (num_samples, raw_samples, 1)

    argmax_indices = post_samples.squeeze(-1).argmax(dim=1)  # (num_samples,)
    optimal_inputs = X_cand_t[argmax_indices]  # (num_samples, d)
    optimal_outputs = post_samples[torch.arange(num_samples), argmax_indices]  # (num_samples, 1)

    return optimal_inputs, optimal_outputs


def sample_simplex(m: int, device, dtype) -> torch.Tensor:
    """Sample a random weight vector from the unit simplex via exponential trick."""
    u = torch.zeros(m, device=device, dtype=dtype).exponential_(1.0)
    return u / u.sum()


def sample_pareto_optimal(
    model,
    bounds: torch.Tensor,
    simplex_groups: list[list[int]],
    rng,
    num_pareto_samples: int = 16,
    num_pareto_points: int = 10,
    raw_samples: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample approximate Pareto optimal sets and fronts for MO entropy search methods.

    Draws posterior samples over feasible simplex candidates and identifies non-dominated
    solutions per sample to produce (pareto_sets, pareto_fronts) of fixed shape.

    Returns:
        pareto_sets: (num_pareto_samples, num_pareto_points, d)
        pareto_fronts: (num_pareto_samples, num_pareto_points, m)
    """
    dim = bounds.shape[1]
    simplex_indices = {idx for group in simplex_groups for idx in group}

    # sample candidates
    X_cand = np.zeros((raw_samples, dim))

    # handle simplex constraints
    for group in simplex_groups:
        samples = rng.dirichlet(np.ones(len(group)), size=raw_samples)
        for k, idx in enumerate(group):
            X_cand[:, idx] = samples[:, k]

    for i in range(dim):
        if i not in simplex_indices:
            X_cand[:, i] = rng.uniform(bounds[0, i].item(), bounds[1, i].item(), size=raw_samples)

    X_cand_t = torch.tensor(X_cand, dtype=bounds.dtype, device=bounds.device)

    with torch.no_grad():
        posterior = model.posterior(X_cand_t)
        # (num_pareto_samples, raw_samples, m)
        post_samples = posterior.rsample(torch.Size([num_pareto_samples]))

    pareto_sets_list = []
    pareto_fronts_list = []

    for s in range(num_pareto_samples):
        y_s = post_samples[s]  # (raw_samples, m)
        mask = is_non_dominated(y_s)
        px = X_cand_t[mask]
        py = y_s[mask]

        # Guarantee at least one point
        if px.shape[0] == 0:
            px = X_cand_t[:1]
            py = y_s[:1]

        # Subsample evenly if Pareto front exceeds num_pareto_points
        if px.shape[0] > num_pareto_points:
            idx = torch.linspace(0, px.shape[0] - 1, num_pareto_points).long()
            px = px[idx]
            py = py[idx]
        pareto_sets_list.append(px)
        pareto_fronts_list.append(py)

    # Pad each sample to num_pareto_points by repeating last row
    def _pad(t: torch.Tensor, target: int) -> torch.Tensor:
        if t.shape[0] >= target:
            return t[:target]
        repeats = target - t.shape[0]
        return torch.cat([t, t[-1:].expand(repeats, -1)], dim=0)

    pareto_sets_t = torch.stack([_pad(p, num_pareto_points) for p in pareto_sets_list])
    pareto_fronts_t = torch.stack([_pad(p, num_pareto_points) for p in pareto_fronts_list])

    return pareto_sets_t, pareto_fronts_t


def _make_simplex_candidate_set(
    simplex_groups: list[list[int]],
    bounds: torch.Tensor,
    num_samples: int = 512,
    rng=None,
) -> torch.Tensor:
    """Generate feasible simplex candidates for entropy-search acquisition functions."""
    if rng is None:
        rng = np.random.default_rng()

    dim = bounds.shape[1]
    simplex_indices = {idx for group in simplex_groups for idx in group}
    X = np.zeros((num_samples, dim))

    for group in simplex_groups:
        samples = rng.dirichlet(np.ones(len(group)), size=num_samples)
        for k, idx in enumerate(group):
            X[:, idx] = samples[:, k]

    for i in range(dim):
        if i not in simplex_indices:
            X[:, i] = rng.uniform(bounds[0, i].item(), bounds[1, i].item(), size=num_samples)

    return torch.tensor(X, dtype=bounds.dtype, device=bounds.device)


def build_so_acqf(
    name: str,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    beta: float,
    simplex_groups: list[list[int]] | None = None,
    bounds: torch.Tensor | None = None,
    rng=None,
):
    """Build a single-objective acquisition function by name.

    Args:
        name: One of 'ei', 'ucb', 'kg', 'mes', 'pes', 'qnei'.
        model: Fitted GP model.
        train_x: Training inputs (n, d).
        train_y: Training targets (n, 1).
        beta: UCB exploration parameter.
        simplex_groups: Simplex index groups (required for 'mes' and 'pes').
        bounds: (2, d) bounds tensor (required for 'mes' and 'pes').
        rng: NumPy RNG (passed through to candidate generation).
    """
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
        return qKnowledgeGradient(model=model, num_fantasies=8)
    elif name == "mes":
        candidate_set = _make_simplex_candidate_set(
            simplex_groups or [], bounds, num_samples=512, rng=rng
        )
        return qMaxValueEntropy(
            model=model, candidate_set=candidate_set
        )
    elif name == "gibbon":
        candidate_set = _make_simplex_candidate_set(
            simplex_groups or [], bounds, num_samples=512, rng=rng
        )
        return qLowerBoundMaxValueEntropy(
            model=model, candidate_set=candidate_set
        )
    elif name == "pes":
        optimal_inputs = get_optimal_inputs_simplex(
            model, simplex_groups or [], bounds, rng=rng
        )
        return qPredictiveEntropySearch(
            model=model, optimal_inputs=optimal_inputs, maximize=True
        )
    elif name == "jes":
        optimal_inputs, optimal_outputs = get_optimal_inputs_outputs_simplex(
            model, simplex_groups or [], bounds, rng=rng
        )
        return qJointEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            num_samples=MC_SAMPLES,
        )
    else:
        raise ValueError(f"Unknown SO acq_fn: {name}")


def build_mo_acqf(
    name: str,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    ref_point: list[float],
    bounds: torch.Tensor,
    covar_module=None,
    simplex_groups: list[list[int]] | None = None,
    rng=None,
):
    """Build a multi-objective acquisition function by name.

    Args:
        name: One of 'qnehvi', 'qparego', 'qhvkg', 'jes_mo', 'mes_mo', 'pes_mo'.
        model: Fitted MO GP model (ModelListGP).
        train_x: Training inputs (n, d).
        train_y: Training targets (n, m).
        ref_point: Reference point for hypervolume (length m).
        bounds: (2, d) bounds tensor.
        simplex_groups: Simplex index groups (required for entropy search methods).
        rng: NumPy RNG (required for entropy search methods).

    Returns:
        Acquisition function. For 'qparego', also returns a freshly fitted SO model
        (as part of the acqfn internals); the original MO model is unused.
    """
    ref_point_tensor = torch.tensor(ref_point, dtype=train_y.dtype, device=train_y.device)

    if name == "qnehvi":
        return qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_x,
            prune_baseline=True,
            cache_root=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )
    elif name == "qparego":
        # Random Chebyshev scalarization + qNEI on a fresh SO model.
        # A new random weight vector is sampled each iteration.
        weights = sample_simplex(train_y.shape[-1], device=train_y.device, dtype=train_y.dtype)
        scalarize = get_chebyshev_scalarization(weights=weights, Y=train_y)
        train_y_scalar = scalarize(train_y).unsqueeze(-1)
        mll_so, model_so = initialize_so_model(train_x, train_y_scalar, bounds, covar_module=covar_module)

        fit_gpytorch_mll(mll_so)

        return qLogNoisyExpectedImprovement(
            model=model_so, X_baseline=train_x, prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
        )
    elif name == "qhvkg":
        return qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=ref_point_tensor,
            num_fantasies=8,
            raw_samples=MC_SAMPLES,
        )
    elif name in ("jes_mo", "mes_mo", "pes_mo"):
        pareto_sets, pareto_fronts = sample_pareto_optimal(
            model, bounds, simplex_groups or [], rng,
            num_pareto_samples=16, num_pareto_points=(20 if "pes_mo" else 50),
        )
        if name == "pes_mo":
            return qMultiObjectivePredictiveEntropySearch(
                model=model,
                pareto_sets=pareto_sets,
            )
        # Compute per-sample hypercell bounds for JES/MES: (num_pareto_samples, 2, J, m)
        hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)
        if name == "jes_mo":
            return qLowerBoundMultiObjectiveJointEntropySearch(
                model=model,
                pareto_sets=pareto_sets,
                pareto_fronts=pareto_fronts,
                hypercell_bounds=hypercell_bounds,
                num_samples=MC_SAMPLES,
            )
        else:  # mes_mo
            return qLowerBoundMultiObjectiveMaxValueEntropySearch(
                model=model,
                hypercell_bounds=hypercell_bounds,
                num_samples=MC_SAMPLES,
            )
    else:
        raise ValueError(f"Unknown MO acq_fn: {name}")


def get_next_candidate(
    acq_fn: str,
    model,
    mll,
    is_mo: bool,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds_dev: torch.Tensor,
    eq_constraints_dev,
    simplex_groups: list[list[int]],
    ref_point: list[float] | None,
    itr: int,
    trial: int,
    rng,
    dtype,
    dev: str,
    ucb_beta: float | None,
    covar_module=None,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (new_x of shape (batch_size, d), acq_value scalar tensor).

    Handles random and ts without building an acqf; all others go through
    optimize_acqf with NaN-retry logic for MO entropy search methods.
    Note: analytic acq_fns (ei, ucb) only support batch_size=1.
    """
    # --- model-free baselines: no GP fitting needed ---
    if acq_fn == "random":
        # sample each simplex group independently from a uniform Dirichlet
        dim = bounds_dev.shape[1]
        new_x_np = np.zeros((batch_size, dim))
        for group in simplex_groups:
            s = rng.dirichlet(np.ones(len(group)), size=batch_size)
            for k, idx in enumerate(group):
                new_x_np[:, idx] = s[:, k]
        return torch.tensor(new_x_np, dtype=dtype, device=dev), torch.zeros(1)

    fit_gpytorch_mll(mll)

    if acq_fn == "ts":
        # Thompson sampling: argmax of batch_size posterior samples over feasible candidates
        cand_set = _make_simplex_candidate_set(simplex_groups, bounds_dev, num_samples=512, rng=rng)
        new_x = MaxPosteriorSampling(model=model, replacement=False)(cand_set, num_samples=batch_size)
        return new_x.to(dtype=dtype), torch.zeros(1)

    # --- gradient-based acquisition function optimization ---
    if is_mo:
        acq = build_mo_acqf(
            acq_fn, model, train_x, train_y, ref_point, bounds_dev,
            covar_module=covar_module, simplex_groups=simplex_groups, rng=rng,
        )
    else:
        beta = ucb_beta if ucb_beta is not None else get_beta_t(itr, train_x.shape[-1])
        acq = build_so_acqf(
            acq_fn, model, train_x, train_y, beta,
            simplex_groups=simplex_groups, bounds=bounds_dev, rng=rng,
        )

    # MES/GIBBON forward only accepts q=1; qNEHVI joint batch optimization is very slow
    # for q>1 — all use sequential greedy (optimize one point at a time with X_pending)
    _sequential_acqfns = {"mes", "gibbon", "ei", "ucb"}
    use_sequential = acq_fn in _sequential_acqfns and batch_size > 1

    # CandidateGenerationError is raised when optimize_acqf returns a candidate that
    # violates the simplex equality constraints and BoTorch's SLSQP repair projection
    # (optimize.py -> project_to_feasible_space_via_slsqp) then fails to converge.
    # It derives from BotorchError, NOT RuntimeError, so it must be named explicitly.
    # Resampling is a legitimate retry: qParEGO redraws its scalarisation weights each
    # call anyway, so a fresh draw is the same method, not a fudge.
    for attempt in range(MAX_ACQF_RETRIES):
        try:
            candidate, acq_value = optimize_acqf(
                acq, bounds=bounds_dev, q=batch_size,
                num_restarts=10, raw_samples=50,
                equality_constraints=eq_constraints_dev,
                sequential=use_sequential,
            )
            return candidate.detach().to(dtype=dtype), acq_value
        except (RuntimeError, CandidateGenerationError) as e:
            # Acquisition function occasionally produces NaN/inf values; resample and retry
            is_nan_inf_error = "nan" in str(e).lower() or "inf" in str(e).lower()
            is_infeasible_error = isinstance(e, CandidateGenerationError)
            if attempt < MAX_ACQF_RETRIES - 1 and (is_nan_inf_error or is_infeasible_error):
                reason = "infeasible candidate" if is_infeasible_error else "NaN/inf"
                print(f"[trial {trial}, iter {itr}] {reason} in optimize_acqf (attempt {attempt + 1}), resampling acqf...")
                if is_mo:
                    acq = build_mo_acqf(
                        acq_fn, model, train_x, train_y, ref_point, bounds_dev,
                        covar_module=covar_module, simplex_groups=simplex_groups, rng=rng,
                    )
                else:
                    beta = ucb_beta if ucb_beta is not None else get_beta_t(itr, train_x.shape[-1])
                    acq = build_so_acqf(
                        acq_fn, model, train_x, train_y, beta,
                        simplex_groups=simplex_groups, bounds=bounds_dev, rng=rng,
                    )
            else:
                raise


def compute_mo_inference_hv(
    model,
    prob,
    ref_point: list[float],
    simplex_groups: list[list[int]],
    bounds: torch.Tensor,
    rng,
    num_cands: int = 512,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute inference HV: HV of true values at the posterior-mean Pareto front.

    Returns:
        (hv, rec_x, rec_true): hypervolume, recommended inputs, true objective values.
    """
    # feasible candidates respecting simplex constraints
    cand_set = _make_simplex_candidate_set(
        simplex_groups, bounds, num_samples=num_cands, rng=rng
    )

    with torch.no_grad():
        mean = model.posterior(cand_set).mean  # (num_cands, m)

    # recommend the non-dominated set under the posterior mean
    pareto_mask = is_non_dominated(mean)
    rec_x = cand_set[pareto_mask]

    # evaluate noiseless true function to get unbiased HV estimate
    rec_true = prob.evaluate_true(rec_x)

    return compute_hypervolume(rec_true.cpu(), ref_point), rec_x, rec_true


def compute_hypervolume(train_y: torch.Tensor, ref_point: list[float]) -> float:
    """Compute the dominated hypervolume of the current Pareto front.

    Args:
        train_y: Observed objectives of shape (n, m).
        ref_point: Reference point as a list of floats of length m.

    Returns:
        Hypervolume as a float.
    """
    pareto_mask = is_non_dominated(train_y)
    pareto_y = train_y[pareto_mask]
    hv = Hypervolume(torch.tensor(ref_point, dtype=train_y.dtype))

    return hv.compute(pareto_y)


def main(args):
    # Overridable so a single stubborn seed can be given more resampling attempts
    # without changing behaviour for any other run (see --max_acqf_retries).
    global MAX_ACQF_RETRIES
    MAX_ACQF_RETRIES = args.max_acqf_retries

    print("problem:", args.problem)
    print("acq_fn:", args.acq_fn)
    if args.max_acqf_retries != 3:
        print("max_acqf_retries:", args.max_acqf_retries)
    if args.acq_fn == "ucb" and not args.problem in MO_PROBLEMS:
        beta_desc = f"fixed ({args.ucb_beta})" if args.ucb_beta is not None else "Srinivas schedule"
        print(f"ucb_beta: {beta_desc}")
    print("number of BO iterations:", args.iterations)
    print("total observations:", args.iterations * args.batch_size)
    print("batch_size:", args.batch_size)
    print("initial random samples:", args.initial_random_samples)
    print("trials:", args.trials)

    if args.known_noise:
        print("known noise GP: enabled")

    if args.mlhgp:
        warm = " (oracle warm-start)" if args.known_noise else ""
        print(f"MLHGP: enabled, em_iter={args.mlhgp_em_iter}{warm}")

    print("kernel:", "matern-2.5" if args.matern else "default (rbf)")

    verbose = args.verbose
    is_mo = args.problem in MO_PROBLEMS

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    dev_info = device_info(dev)
    print("device:", dev_info["device"], "|", dev_info["gpu_name"])

    dtype = torch.float32 if dev in ("mps", "cpu") else torch.double

    # heteroscedastic problems draw noise from their own emulator; --noise_std does not apply.
    # --noise_std unset: let each problem class use its own default noise std.
    noise_kwargs = (
        {}
        if args.noise_std is None or "heteroscedastic" in args.problem
        else {"noise_std": args.noise_std}
    )

    if args.problem == "dm_curriculum":
        prob = DMCurriculum(negate=False, **noise_kwargs)
    elif args.problem == "dm_curriculum_mo":
        prob = DMCurriculumMO(negate=False, **noise_kwargs)
    elif args.problem == "dm_curriculum_heteroscedastic":
        prob = DMCurriculumHet(negate=False)
    else:
        raise ValueError(f"Unknown problem {args.problem}")

    noise_std = prob.noise_std
    print("noise std:", noise_std)

    prob.to(dtype=dtype, device=dev)

    covar_module = (
        ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=prob.dim)) if args.matern else None
    )

    use_known_noise = args.known_noise
    use_mlhgp = args.mlhgp

    # DM curriculum problems: two simplex groups over 6 parameters
    simplex_groups = [[0, 1, 2], [3, 4, 5]]

    bounds_tensor = torch.tensor(prob._bounds, dtype=dtype).T  # (2, d)
    equality_constraints = get_simplex_equality_constraints(simplex_groups, dtype=dtype)

    num_bo_iters = args.iterations
    all_trial_results = []

    # --seed_offset sets the seed independently of --trial_offset, so seeds can be run
    # as separate parallel processes without triggering the merge-with-existing-file path.
    seed_base = args.trial_offset if args.seed_offset is None else args.seed_offset

    for trial in range(args.trials):
        seed = trial + seed_base
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{args.trials}  (seed={seed})")
        print(f"{'='*60}")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed) 

        print("generating initial data...")
        train_x_np, train_y = generate_initial_data(
            prob, rng, n=args.initial_random_samples, simplex_groups=simplex_groups, device=dev, dtype=dtype
        )

        train_x = torch.tensor(train_x_np, dtype=dtype, device=dev)
        train_y = train_y.to(device=dev, dtype=dtype)

        bounds_dev = bounds_tensor.to(dev)
        eq_constraints_dev = [
            (idx.to(dev), coef.to(dev), rhs)
            for idx, coef, rhs in equality_constraints
        ]

        if is_mo:
            # Reference point fixed from initial data for the duration of this trial
            ref_point = prob._ref_point
            if trial == 0:
                print("reference point:", ref_point)

            hv0 = compute_hypervolume(train_y.cpu(), ref_point)
            hv_all = [hv0]
            log_hv_diff_all = [math.log(max(prob._max_hv - hv0, 1e-8))]

            _true_y0 = prob.evaluate_true(train_x)
            hv0_true = compute_hypervolume(_true_y0.cpu(), ref_point)
            hv_true_all = [hv0_true]
            _best_hv_true_running = hv0_true
            best_hv_true_all = [hv0_true]
            log_hv_diff_true_all = [math.log(max(prob._max_hv - hv0_true, 1e-8))]
            log_best_hv_diff_true_all = [math.log(max(prob._max_hv - _best_hv_true_running, 1e-8))]

            _true_pareto_mask0 = is_non_dominated(_true_y0.cpu())
            _best_hv_true_pareto_x = train_x[_true_pareto_mask0].cpu().tolist()
            _best_hv_true_pareto_y = _true_y0.cpu()[_true_pareto_mask0].tolist()

            log_inference_hv_regret_all = []
            log_best_inference_hv_regret_all = []
            inf_hv_all = []
            best_inf_hv_all = []
            _best_inf_hv_running = -float("inf")
            _best_inf_hv_pareto_x: list = []
            _best_inf_hv_pareto_y: list = []
            _inf_hv_pareto_x: list = []
            _inf_hv_pareto_y: list = []

            seen_y = train_y.cpu().tolist()
            candidates = train_x_np

            if use_mlhgp:
                train_x_d = train_x.to(dtype=dtype)
                warm_noise = (
                    (prob.evaluate_noise(train_x_d).clamp(min=1e-6) ** 2).to(device=dev, dtype=dtype)
                    if use_known_noise
                    else None
                )

                mll, model = fit_mlhgp_mo_model(
                    train_x, train_y, bounds_dev,
                    n_em_iter=args.mlhgp_em_iter,
                    initial_noise_var=warm_noise,
                    covar_module=covar_module,
                )

                if use_known_noise:
                    train_yvar = warm_noise

            elif use_known_noise:
                train_x_d = train_x.to(dtype=dtype)
                train_yvar = (
                    prob.evaluate_noise(train_x_d).clamp(min=1e-6) ** 2
                ).to(device=dev, dtype=dtype)
                mll, model = initialize_hetero_mo_model(train_x, train_y, train_yvar, bounds_dev, covar_module=covar_module)

            else:
                mll, model = initialize_mo_model(train_x, train_y, bounds_dev, covar_module=covar_module)

            if args.acq_fn != "random":
                
                if not use_mlhgp:
                    fit_gpytorch_mll(mll)

                inf_hv_init, inf_rec_x_init, inf_rec_true_init = compute_mo_inference_hv(model, prob, ref_point, simplex_groups, bounds_dev, rng)
                inf_hv_all.append(inf_hv_init)
                _best_inf_hv_running = inf_hv_init
                best_inf_hv_all.append(inf_hv_init)
                log_inference_hv_regret_all.append(math.log(max(prob._max_hv - inf_hv_init, 1e-8)))
                log_best_inference_hv_regret_all.append(math.log(max(prob._max_hv - _best_inf_hv_running, 1e-8)))
                _best_inf_hv_pareto_x = inf_rec_x_init.cpu().tolist()
                _best_inf_hv_pareto_y = inf_rec_true_init.cpu().tolist()
                _inf_hv_pareto_x = inf_rec_x_init.cpu().tolist()
                _inf_hv_pareto_y = inf_rec_true_init.cpu().tolist()
        else:
            best_y_all = [train_y.max().item()]
            seen_y = train_y.cpu().tolist()
            candidates = train_x_np

            rec_x_all, rec_true_all, best_rec_true_all = [], [], []
            inference_regret_all, log_inference_regret_all = [], []
            best_inference_regret_all, log_best_inference_regret_all = [], []
            _init_best_x = train_x[train_y.flatten().argmax()].detach()
            _init_best_true = prob.evaluate_true(_init_best_x.unsqueeze(0)).item()

            best_obs_x_all = [_init_best_x.cpu().tolist()]
            best_obs_true_all = [_init_best_true]
            best_obs_regret_all = [max(prob._optimal_value - _init_best_true, 0.0)]
            log_best_obs_regret_all = [math.log(max(prob._optimal_value - _init_best_true, 1e-8))]

            _init_true_vals = prob.evaluate_true(train_x).flatten()
            _best_simple_true_running = _init_true_vals.max().item()
            simple_regret_all = [max(prob._optimal_value - _best_simple_true_running, 0.0)]
            log_simple_regret_all = [math.log(max(prob._optimal_value - _best_simple_true_running, 1e-8))]

            if use_mlhgp:
                mll, model = fit_mlhgp_so_model(train_x, train_y, bounds_dev, n_em_iter=args.mlhgp_em_iter, covar_module=covar_module)
            elif use_known_noise:
                train_x_d = train_x.to(dtype=dtype)
                train_yvar = (
                    prob.evaluate_noise(train_x_d).clamp(min=1e-6) ** 2
                ).to(device=dev, dtype=dtype)
                mll, model = initialize_hetero_so_model(train_x, train_y, train_yvar, bounds_dev, covar_module=covar_module)
            else:
                mll, model = initialize_so_model(train_x, train_y, bounds_dev, covar_module=covar_module)

            if args.acq_fn != "random":
                if not use_mlhgp:
                    fit_gpytorch_mll(mll)

                rec_x_init, _ = optimize_acqf(
                    PosteriorMean(model), bounds=bounds_dev, q=1,
                    num_restarts=5, raw_samples=256,
                    equality_constraints=eq_constraints_dev,
                )

                rec_x_init = rec_x_init.squeeze(0).detach()
                rec_true_init = prob.evaluate_true(rec_x_init.unsqueeze(0)).item()
                rec_x_all.append(rec_x_init.cpu().tolist())
                rec_true_all.append(rec_true_init)
                inference_regret_all.append(prob._optimal_value - rec_true_init)
                log_inference_regret_all.append(math.log(max(prob._optimal_value - rec_true_init, 1e-8)))
                _best_rec_true_running = rec_true_init
                best_rec_true_all.append(_best_rec_true_running)
                best_inference_regret_all.append(max(prob._optimal_value - _best_rec_true_running, 0.0))
                log_best_inference_regret_all.append(math.log(max(prob._optimal_value - _best_rec_true_running, 1e-8)))

        t0 = time.time()
        for itr in range(num_bo_iters):
            new_x, acq_value = get_next_candidate(
                acq_fn=args.acq_fn,
                model=model,
                mll=mll,
                is_mo=is_mo,
                train_x=train_x,
                train_y=train_y,
                bounds_dev=bounds_dev,
                eq_constraints_dev=eq_constraints_dev,
                simplex_groups=simplex_groups,
                ref_point=ref_point if is_mo else None,
                itr=itr,
                trial=trial,
                rng=rng,
                dtype=dtype,
                dev=dev,
                ucb_beta=args.ucb_beta,
                covar_module=covar_module,
                batch_size=args.batch_size,
            )
            new_y = prob(new_x)

            train_x = torch.vstack((train_x, new_x))
            train_y = torch.vstack((train_y, new_y))

            seen_y.extend(new_y.cpu().tolist())
            candidates = np.concatenate((candidates, new_x.cpu().numpy()), axis=0)

            if is_mo:
                hv = compute_hypervolume(train_y.cpu(), ref_point)
                hv_all.append(hv)
                log_hv_diff_all.append(math.log(max(prob._max_hv - hv, 1e-8)))

                _true_y = prob.evaluate_true(train_x)
                hv_true = compute_hypervolume(_true_y.cpu(), ref_point)
                hv_true_all.append(hv_true)
                log_hv_diff_true_all.append(math.log(max(prob._max_hv - hv_true, 1e-8)))
                if hv_true > _best_hv_true_running:
                    _best_hv_true_running = hv_true
                    _true_pareto_mask = is_non_dominated(_true_y.cpu())
                    _best_hv_true_pareto_x = train_x[_true_pareto_mask].cpu().tolist()
                    _best_hv_true_pareto_y = _true_y.cpu()[_true_pareto_mask].tolist()
                best_hv_true_all.append(_best_hv_true_running)
                log_best_hv_diff_true_all.append(math.log(max(prob._max_hv - _best_hv_true_running, 1e-8)))

                if args.acq_fn != "random":
                    inf_hv, inf_rec_x, inf_rec_true = compute_mo_inference_hv(model, prob, ref_point, simplex_groups, bounds_dev, rng)
                    inf_hv_all.append(inf_hv)
                    if inf_hv > _best_inf_hv_running:
                        _best_inf_hv_running = inf_hv
                        _best_inf_hv_pareto_x = inf_rec_x.cpu().tolist()
                        _best_inf_hv_pareto_y = inf_rec_true.cpu().tolist()
                    best_inf_hv_all.append(_best_inf_hv_running)
                    log_inference_hv_regret_all.append(math.log(max(prob._max_hv - inf_hv, 1e-8)))
                    log_best_inference_hv_regret_all.append(math.log(max(prob._max_hv - _best_inf_hv_running, 1e-8)))
                    _inf_hv_pareto_x = inf_rec_x.cpu().tolist()
                    _inf_hv_pareto_y = inf_rec_true.cpu().tolist()

                if use_mlhgp:
                    warm_noise = None

                    if use_known_noise:
                        new_x_d = new_x.to(dtype=dtype)
                        new_yvar = (
                            prob.evaluate_noise(new_x_d).clamp(min=1e-6) ** 2
                        ).to(device=dev, dtype=dtype)
                        train_yvar = torch.vstack((train_yvar, new_yvar))
                        warm_noise = train_yvar

                    mll, model = fit_mlhgp_mo_model(
                        train_x, train_y, bounds_dev,
                        n_em_iter=args.mlhgp_em_iter,
                        initial_noise_var=warm_noise,
                        covar_module=covar_module,
                    )
                elif use_known_noise:
                    new_x_d = new_x.to(dtype=dtype)
                    new_yvar = (
                        prob.evaluate_noise(new_x_d).clamp(min=1e-6) ** 2
                    ).to(device=dev, dtype=dtype)
                    train_yvar = torch.vstack((train_yvar, new_yvar))
                    mll, model = initialize_hetero_mo_model(
                        train_x, train_y, train_yvar, bounds_dev, covar_module=covar_module
                    )
                else:
                    mll, model = initialize_mo_model(train_x, train_y, bounds_dev, covar_module=covar_module)

                if verbose:
                    pareto_size = is_non_dominated(train_y).sum().item()
                    inf_regret_str = (
                        f"  log_inf_hv_regret={log_inference_hv_regret_all[-1]:.4f}"
                        if log_inference_hv_regret_all else ""
                    )
                    print(
                        f"  itr {itr + 1}: acq={acq_value.detach().cpu().item():.4f}  "
                        f"hv={hv:.4f}  pareto_size={pareto_size}{inf_regret_str}"
                    )
                    print(
                        f"    new_x={np.array2string(new_x.cpu().numpy(), precision=3, suppress_small=True)}"
                    )
                    print("-" * 50)
                else:
                    print(".", end="")
            else:
                best_y = train_y.max().item()
                best_y_all.append(best_y)

                if args.acq_fn != "random":
                    # recommend max of posterior mean
                    rec_x, _ = optimize_acqf(
                        PosteriorMean(model), bounds=bounds_dev, q=1,
                        num_restarts=5, raw_samples=256,
                        equality_constraints=eq_constraints_dev,
                    )
                    rec_x = rec_x.squeeze(0).detach()
                    rec_true = prob.evaluate_true(rec_x.unsqueeze(0)).item()
                    rec_x_all.append(rec_x.cpu().tolist())
                    rec_true_all.append(rec_true)

                    # inf regret
                    inference_regret_all.append(prob._optimal_value - rec_true)
                    log_inference_regret_all.append(math.log(max(prob._optimal_value - rec_true, 1e-8)))
                    _best_rec_true_running = max(_best_rec_true_running, rec_true)
                    best_rec_true_all.append(_best_rec_true_running)
                    best_inference_regret_all.append(max(prob._optimal_value - _best_rec_true_running, 0.0))
                    log_best_inference_regret_all.append(math.log(max(prob._optimal_value - _best_rec_true_running, 1e-8)))

                # alternative best point is best observed x
                best_obs_x = train_x[train_y.flatten().argmax()].detach()
                best_obs_true = prob.evaluate_true(best_obs_x.unsqueeze(0)).item()
                best_obs_x_all.append(best_obs_x.cpu().tolist())
                best_obs_true_all.append(best_obs_true)
                best_obs_regret_all.append(max(prob._optimal_value - best_obs_true, 0.0))
                log_best_obs_regret_all.append(math.log(max(prob._optimal_value - best_obs_true, 1e-8)))

                _new_true = prob.evaluate_true(new_x).flatten().max().item()
                _best_simple_true_running = max(_best_simple_true_running, _new_true)
                simple_regret_all.append(max(prob._optimal_value - _best_simple_true_running, 0.0))
                log_simple_regret_all.append(math.log(max(prob._optimal_value - _best_simple_true_running, 1e-8)))

                if use_mlhgp:
                    mll, model = fit_mlhgp_so_model(train_x, train_y, bounds_dev, n_em_iter=args.mlhgp_em_iter, covar_module=covar_module)
                elif use_known_noise:
                    new_x_d = new_x.to(dtype=dtype)
                    new_yvar = (
                        prob.evaluate_noise(new_x_d).clamp(min=1e-6) ** 2
                    ).to(device=dev, dtype=dtype)
                    train_yvar = torch.vstack((train_yvar, new_yvar))
                    mll, model = initialize_hetero_so_model(
                        train_x, train_y, train_yvar, bounds_dev, covar_module=covar_module
                    )
                else:
                    mll, model = initialize_so_model(train_x, train_y, bounds_dev, covar_module=covar_module)

                if verbose:
                    inf_str = f"  best_inf_regret={best_inference_regret_all[-1]:.4f}" if args.acq_fn != "random" else ""
                    print(
                        f"  itr {itr + 1}: acq={acq_value.detach().cpu().item():.4f}  "
                        f"best_y={best_y:.4f}{inf_str}  "
                        f"best_obs_regret={best_obs_regret_all[-1]:.4f}"
                    )
                    print(
                        f"    new_x={np.array2string(new_x.cpu().numpy(), precision=3, suppress_small=True)}"
                    )
                    print("-" * 50)
                else:
                    print(".", end="")

        t1 = time.time()

        if is_mo:
            print(
                f"\nTrial {trial + 1} done in {t1 - t0:.1f}s — "
                f"final hv={hv_all[-1]:.4f}"
            )
            trial_result = {
                "trial": trial,
                "seed": seed,
                "time_seconds": t1 - t0,
                **dev_info,
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
                "pareto_x_best_hv": _best_hv_true_pareto_x,
                "pareto_y_best_hv": _best_hv_true_pareto_y,
                "pareto_x_best_inf_hv": _best_inf_hv_pareto_x,
                "pareto_y_best_inf_hv": _best_inf_hv_pareto_y,
                "pareto_x_inf_hv": _inf_hv_pareto_x,
                "pareto_y_inf_hv": _inf_hv_pareto_y,
                "candidates": candidates.tolist(),
                "seen_y": seen_y,
            }
        else:
            print(
                f"\nTrial {trial + 1} done in {t1 - t0:.1f}s — "
                f"final best_y={best_y_all[-1]:.4f}"
            )
            trial_result = {
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
            }

        all_trial_results.append(trial_result)

    ucb_beta_fixed = args.ucb_beta if (args.acq_fn == "ucb" and not is_mo) else None
    results_dir = REPO_ROOT / "results" / f"dm{args.folder_prefix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    ucb_beta_tag = f"_beta{ucb_beta_fixed}" if ucb_beta_fixed is not None else ""
    batch_tag = f"_q{args.batch_size}" if args.batch_size > 1 else ""

    # per-seed shards get their own filename so parallel runs never collide
    seed_tag = "" if args.seed_offset is None else f"_seed{args.seed_offset}"

    def _make_filename(n_trials: int) -> Path:
        mlhgp_tag = f"_mlhgp_em{args.mlhgp_em_iter}" if use_mlhgp else ""
        return results_dir / (
            f"{args.problem}_{args.acq_fn}{ucb_beta_tag}"
            f"{mlhgp_tag}"
            f"{'_knownnoise' if use_known_noise else ''}"
            f"{batch_tag}"
            f"_{n_trials}trials_{args.iterations}iterations{seed_tag}_results.json"
        )

    _config_keys = ["problem", "acq_fn", "ucb_beta", "known_noise", "mlhgp", "mlhgp_em_iter", "iterations", "batch_size", "initial_random_samples", "noise_std"]
    new_config = {
        "problem": args.problem,
        "acq_fn": args.acq_fn,
        "ucb_beta": ucb_beta_fixed,
        "noise_std": noise_std,
        "known_noise": use_known_noise,
        "mlhgp": use_mlhgp,
        "mlhgp_em_iter": args.mlhgp_em_iter if use_mlhgp else None,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "initial_random_samples": args.initial_random_samples,
        # not part of _config_keys: merging trials run on different hardware is allowed,
        # per-trial gpu_name keeps the record straight.
        **dev_info,
        "emulator_versions": emulator_version.emulator_versions_for(prob),
    }

    if args.trial_offset > 0:
        prev_file = _make_filename(args.trial_offset)
        if not prev_file.exists():
            raise FileNotFoundError(
                f"--trial_offset={args.trial_offset} but expected prior results file not found: {prev_file}"
            )
        with open(prev_file) as f:
            existing = json.load(f)
        mismatches = {k: (existing.get(k), new_config[k]) for k in _config_keys if existing.get(k) != new_config[k]}
        if mismatches:
            raise ValueError(
                f"Config mismatch with {prev_file.name}:\n"
                + "\n".join(f"  {k}: existing={old!r}, new={new!r}" for k, (old, new) in mismatches.items())
            )
        existing_seeds = {t["seed"] for t in existing["trials"]}
        new_seeds = {t["seed"] for t in all_trial_results}
        duplicates = existing_seeds & new_seeds
        if duplicates:
            raise ValueError(f"Duplicate seeds {duplicates} — adjust --trial_offset.")
        merged_trials = existing["trials"] + all_trial_results
        results = {**new_config, "num_trials": len(merged_trials), "trials": merged_trials}
        print(f"\nMerging {args.trial_offset} existing + {args.trials} new trials")
    else:
        results = {**new_config, "num_trials": args.trials, "trials": all_trial_results}

    total_trials = results["num_trials"]
    output_file = _make_filename(total_trials)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="BO for DM curriculum problems")
    parser.add_argument(
        "--problem",
        type=str,
        default="dm_curriculum_mo",
        choices=["dm_curriculum", "dm_curriculum_mo", "dm_curriculum_heteroscedastic"],
        help="Problem to run. Default: dm_curriculum_mo",
    )
    parser.add_argument(
        "--acq_fn",
        type=str,
        default=None,
        choices=["qnei", "ei", "ucb", "kg", "mes", "gibbon", "pes", "jes", "ts", "random", "qnehvi", "qparego", "qhvkg", "jes_mo", "mes_mo", "pes_mo"],
        help="Acquisition function. SO: ei, ucb, kg, mes, gibbon, pes, jes, ts, qnei. MO: qnehvi, qparego, qhvkg, jes_mo, mes_mo, pes_mo. Both: random. Default: auto",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help=(
            "Observation noise std for the emulator. Ignored for heteroscedastic problems, "
            "which draw input-dependent noise from their own emulator. "
            "Default: the problem class's own default"
        ),
    )
    parser.add_argument(
        "--folder_prefix",
        type=str,
        default="",
        help="Suffix appended to the results subfolder name, e.g. 'dm{folder_prefix}'. Default: '' (results/dm)",
    )
    parser.add_argument(
        "--known_noise",
        action="store_true",
        help=(
            "Use SingleTaskGP with known noise variances, conditioning on noise variances from the emulator. "
            "Only applies to heteroscedastic problems. Default: SingleTaskGP with inferred noise."
        ),
    )
    parser.add_argument(
        "--mlhgp",
        action="store_true",
        help=(
            "Use Most Likely Heteroscedastic GP (Kersting et al., 2007) to learn input-dependent noise from residuals. "
            "Applies to all problems. For dm_curriculum_heteroscedastic, combine with --known_noise for oracle warm-start."
        ),
    )
    parser.add_argument(
        "--mlhgp_em_iter",
        type=int,
        default=5,
        help="Number of EM iterations for MLHGP. Default: 5",
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
        "--verbose",
        action="store_true",
        help="Print detailed logs during optimization.",
    )
    parser.add_argument(
        "--initial_random_samples",
        type=int,
        default=10,
        help="Number of initial random samples before BO starts. Default: 10",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent trials (each uses a different random seed). Default: 1",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=None,
        help=(
            "Seed for the first trial, independent of --trial_offset. Unlike --trial_offset "
            "this does NOT merge with an existing results file; it writes a separate "
            "_seed<N>_results.json shard, so seeds can be run as parallel processes and "
            "combined afterwards with scripts/merge_seed_runs.py. Default: use --trial_offset"
        ),
    )
    parser.add_argument(
        "--trial_offset",
        type=int,
        default=0,
        help="Offset added to trial index to get seed (use to extend existing runs). Default: 0",
    )
    parser.add_argument(
        "--ucb_beta",
        type=float,
        default=None,
        help=(
            "Fixed beta for UCB acquisition function (SO only). "
            "Suggested values: 0.1, 0.5, 1.0, 2.0. "
            "Default: None (uses Srinivas schedule via get_beta_t)."
        ),
    )
    parser.add_argument(
        "--max_acqf_retries",
        type=int,
        default=3,
        help=(
            "Attempts at optimize_acqf per BO step before giving up. Each retry "
            "resamples the acquisition function, which usually clears transient "
            "NaN gradients / infeasible candidates. Default: 3"
        ),
    )
    parser.add_argument(
        "--matern",
        action="store_true",
        help="Use Matérn-2.5 kernel instead of the default RBF.",
    )

    args = parser.parse_args()

    BOTH_ACQ_FNS = {"random"}
    SO_ACQ_FNS = {"qnei", "ei", "ucb", "kg", "mes", "gibbon", "pes", "jes", "ts"} | BOTH_ACQ_FNS
    MO_ACQ_FNS = {"qnehvi", "qparego", "qhvkg", "jes_mo", "mes_mo", "pes_mo"} | BOTH_ACQ_FNS

    # Auto-select acq_fn if not specified
    if args.acq_fn is None:
        args.acq_fn = "qnehvi" if args.problem in MO_PROBLEMS else "qnei"

    # Validate acq_fn is compatible with problem type
    if args.problem in MO_PROBLEMS and args.acq_fn not in MO_ACQ_FNS:
        parser.error(f"--acq_fn {args.acq_fn} is not supported for MO problems; use one of {MO_ACQ_FNS}")
    if args.problem in SO_PROBLEMS and args.acq_fn not in SO_ACQ_FNS:
        parser.error(f"--acq_fn {args.acq_fn} is not supported for SO problems; use one of {SO_ACQ_FNS}")

    if args.known_noise and "heteroscedastic" not in args.problem:
        parser.error("--known_noise is only applicable to dm_curriculum_heteroscedastic")
    if args.ucb_beta is not None:
        if args.problem in MO_PROBLEMS:
            parser.error("--ucb_beta is only applicable to SO problems (dm_curriculum)")
        if args.acq_fn != "ucb":
            parser.error("--ucb_beta is only applicable when --acq_fn ucb")


    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
