# Constrained Bayesian optimization baselines for the PCO problems.
#
# Usage:
#   python -m bolt_exp.runners.test_w_botorch_pco
#   python -m bolt_exp.runners.test_w_botorch_pco --problem pco64 --acq_fn ei
#   python -m bolt_exp.runners.test_w_botorch_pco --acq_fn ucb_c --iterations 50 --trials 3
#   python -m bolt_exp.runners.test_w_botorch_pco --acq_fn scbo --verbose
#
# Arguments:
#   --problem       Problem variant: pco16, pco32, pco64                 (default: pco32)
#   --noise_std     Observation noise std                                 (default: none, noiseless)
#   --acq_fn        Acquisition function: qnei, ei, ucb_c, cts, scbo, cmes_ibo,
#                   ckg, admmbo, random, tpe, cmaes                       (default: qnei)
#   --iterations    Number of BO iterations; one observation per iteration  (default: 100)
#   --ucb_beta      Fixed beta for UCB-C (None = Srinivas schedule)        (default: None)
#   --initial_random_samples  Number of initial random samples            (default: 10)
#   --trials        Number of independent runs with different seeds       (default: 1)
#   --trial_offset  Offset added to trial index to get seed               (default: 0)
#   --seed_offset   Seed base, independent of --trial_offset              (default: --trial_offset)
#   --pf_delta      Recommend only where P(feasible) >= 1 - pf_delta   (default: 0.05)
#   --no_repeats    Remove already-evaluated configurations from the candidate pool
#   --prior_mean    Fit both GPs on a physics-informed mean instead of a constant
#   --folder_prefix Suffix appended to the results subfolder name
#   --verbose       Print per-iteration logs
#
# Output:
#   results/<problem>/<problem>_<acq_fn>[_beta<B>]_<trials>trials_<iterations>iterations_results.json
#
# PCO is a *hidden constraint* problem: an infeasible strategy runs out of
# GPU memory, so its throughput is never measured. The problem imputes those
# rows itself, so the imputation is part of the benchmark and identical for
# every method here. That shapes the loop:
#
#   * the objective GP is fit on every observation, imputed values included;
#   * the constraint GP is fit on the memory margin;
#   * the acquisition weighs improvement by the probability of feasibility,
#     which is the only thing keeping the loop off the high imputed values;
#   * feasibility comes from the slack, and incumbents and regret count
#     feasible points only (`true_objective`): a failed run realises no
#     throughput.
#
# The search space is the finite set of table rows -- the parallelism degrees
# must multiply to the GPU count -- so every acquisition is maximised over that
# candidate set with optimize_acqf_discrete rather than over the bounding box.
# UCB-C, CMES-IBO and cKG have no BoTorch implementation; all three are written
# against the AcquisitionFunction interface in bolt_exp/constrained_acqf.py so
# they use that same selector. Two exceptions: constrained Thompson sampling,
# which has no per-candidate acquisition value, and ADMMBO, which is a stateful
# policy alternating between two subproblems rather than one acquisition.
#
# `cts` and `scbo` differ only in the candidate pool: both sample the same way,
# but `scbo` first narrows the table to a trust region. That region is what SCBO
# adds over plain constrained TS, so the pair measures what it is worth on a
# table this small. `scbo` logs its size as `scbo_ncand_all`.
#
# `tpe` and `cmaes` are the non-BO baselines (Optuna's TPESampler and
# CmaEsSampler). They fit no GP, so like `random` they log no recommendation or
# inference regret; see OptunaPolicy for how they meet the table and the constraint.

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from rich import print
import torch

from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.exceptions.errors import ModelFittingError
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.constraints import GreaterThan
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from bolt import PCO16, PCO32, PCO64
from bolt_exp import REPO_ROOT
from bolt_exp.parallel_prior import ParallelismPrior
from bolt_exp.constrained_acqf import (
    ADMMBOPolicy,
    ConstrainedKnowledgeGradient,
    ConstrainedMaxValueEntropySearch,
    ConstrainedUpperConfidenceBound,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


_PROBLEMS = {"pco16": PCO16, "pco32": PCO32, "pco64": PCO64}

# Minimum feasible observations needed before a GP can be fit to the objective.
MIN_FEASIBLE_INIT = 2

# A recommendation must be feasible with probability >= 1 - PF_DELTA, following
# Gelbart, Snoek & Adams (UAI 2014). A coin-flip threshold would hand back runs
# the model thinks are as likely to crash as not.
PF_DELTA = 0.05

# Number of sampled constrained maxima f* used by CMES-IBO.
CMES_IBO_SAMPLES = 10

# cKG outcomes of observing a candidate: the Cartesian product of Gaussian
# quantiles for the objective and the constraint (Ungredda & Branke, App. D).
# The paper's objective set is 0.1, ..., 0.9; it leaves the constraint's open.
CKG_Y_QUANTILES = 9
CKG_C_QUANTILES = 5

# Rows each outcome contributes to cKG's discretisation: a runner-up, since
# the paper's lone argmax collapses `X_d` to one row and zeroes the
# acquisition. It costs O(|X_d|^2), paid for by the lower `CKG_C_QUANTILES`.
CKG_BEST_PER_OUTCOME = 2

# Two cKG values within this are a tie, broken at random.
CKG_TIE_TOL = 1e-12

# Candidates per cKG batch: each scores every table row under every outcome,
# so the pool is chunked well below optimize_acqf_discrete's default.
CKG_BATCH = 32

# Non-BO baselines, run through Optuna rather than fit GPs.
OPTUNA_METHODS = ("tpe", "cmaes")

# Value CMA-ES is told at an OOM: below any throughput, which is non-negative.
CMAES_OOM_VALUE = -1.0


@dataclass
class ScboState:
    """SCBO's trust region, sized in table rows rather than in side length.

    A hypercube of side `L` routinely contains no row of a few-hundred-row
    table, so the region is the `n_cand` nearest rows to the centre and
    `n_cand` halves and doubles where `L` would -- always non-empty, same number
    of shrinks before a restart. Mirrors the discrete TuRBO state in
    `test_w_botorch_po.py`.

    `best_*` track the centre's own observation, which a success is judged
    against. A restart resets the region and keeps going rather than ending the
    run: the budget is fixed and shared with every other method.
    """

    n_cand_max: int
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    # dim/batch_size is calibrated for continuous Sobol generation, far too
    # large for a fixed discrete candidate set.
    failure_tolerance: int = 10
    best_value: float = -float("inf")
    best_violation: float = float("inf")
    best_feasible: bool = False
    restart_triggered: bool = False
    n_cand: int = field(init=False)
    n_cand_init: int = field(init=False)
    n_cand_min: int = field(init=False)

    def __post_init__(self):
        self.n_cand_init = max(self.n_cand_max // 2, 1)
        self.n_cand = self.n_cand_init
        # Strictly below n_cand_init, or a restart would re-trigger immediately.
        self.n_cand_min = max(1, min(10, self.n_cand_init // 2))


def update_scbo_state(state: ScboState, new_y: torch.Tensor, new_c: torch.Tensor) -> ScboState:
    """Advance the trust region on one observation.

    The centre may be infeasible, so unlike TuRBO a success is a feasible point
    beating the centre by the usual 1e-3 relative margin, or -- while nothing
    has run yet -- one that violates less. The first feasible point always
    counts, whatever its objective.
    """
    y = float(torch.nan_to_num(new_y.reshape(-1)[0], nan=-math.inf))
    slack = float(new_c.reshape(-1)[0])
    # Feasibility is the sign of the slack, as everywhere else here, so a zero
    # margin fails. It is tracked separately rather than read off
    # `best_violation`, which a zero margin also reports as zero.
    feasible = slack > 0
    violation = 0.0 if feasible else max(-slack, 0.0)

    if feasible:
        # Same relative threshold as the PO loop and BoTorch's SCBO tutorial:
        # without it the region expands on noise-level gains.
        beat = y > state.best_value + 1e-3 * abs(state.best_value)
        success = not state.best_feasible or beat
        if success:
            state.best_value, state.best_violation = y, 0.0
            state.best_feasible = True
    else:
        # Once anything has run, no failed run is progress, however near it came.
        success = not state.best_feasible and violation < state.best_violation
        if success:
            state.best_value, state.best_violation = y, violation

    if success:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

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


def scbo_best_index(train_y: torch.Tensor, train_c: torch.Tensor) -> int:
    """The incumbent row: best feasible, else least infeasible.

    Every OOM reports the same constant pad, so all violations tie and the
    paper's objective tie-break is what actually orders the infeasible rows.
    """
    score = torch.nan_to_num(train_y.flatten(), nan=-math.inf).clone()
    feasible = feasible_mask(train_c)
    if bool(feasible.any()):
        score[~feasible] = -math.inf
    else:
        violation = (-train_c.flatten()).clamp(min=0)
        score[violation > violation.min()] = -math.inf
    return int(score.argmax())


def scbo_center(train_x: torch.Tensor, train_y: torch.Tensor, train_c: torch.Tensor):
    """The trust region's centre."""
    return train_x[scbo_best_index(train_y, train_c)]


def seed_scbo_state(state: ScboState, train_y: torch.Tensor, train_c: torch.Tensor) -> ScboState:
    """Point the state at the initial design's incumbent.

    The centre is read from all the data, so a state left at `-inf` would score
    the first observation a success however far below the design it falls, and
    expand the region on it.
    """
    i = scbo_best_index(train_y, train_c)
    state.best_value = float(torch.nan_to_num(train_y.flatten(), nan=-math.inf)[i])
    state.best_feasible = bool(feasible_mask(train_c)[i])
    state.best_violation = (
        0.0 if state.best_feasible else float((-train_c.flatten()).clamp(min=0)[i])
    )
    return state


def objective_lengthscales(obj_model) -> torch.Tensor | None:
    """ARD lengthscales of the fitted objective GP, on the `Normalize`d scale."""
    kernel = getattr(obj_model.covar_module, "base_kernel", obj_model.covar_module)
    ls = getattr(kernel, "lengthscale", None)
    return None if ls is None else ls.detach().flatten()


def trust_region_pool(
    pool: torch.Tensor,
    center: torch.Tensor,
    bounds: torch.Tensor,
    k: int,
    lengthscales: torch.Tensor | None = None,
):
    """The `k` rows of `pool` nearest `center`, measured in the unit cube.

    SCBO runs on `[0, 1]^d`; the knobs are log2 steps with different spans, so
    raw distances would let the widest decide the region.

    With `lengthscales`, the distance is weighted as SCBO shapes its box: each
    dimension divided by its lengthscale, normalised to geometric mean 1, so the
    region stretches along knobs the GP finds irrelevant.
    """
    lo, hi = bounds[0].to(pool), bounds[1].to(pool)
    scaled = (pool - lo) / (hi - lo).clamp_min(1e-12)
    center_scaled = (center.to(pool) - lo) / (hi - lo).clamp_min(1e-12)
    delta = scaled - center_scaled
    if lengthscales is not None:
        w = lengthscales.to(pool).clamp_min(1e-12)
        w = w / w.log().mean().exp()
        delta = delta / w
    k = min(k, len(pool))
    idx = delta.norm(dim=-1).topk(k, largest=False).indices
    return pool[idx]


def get_beta_t(n_step: int, n_var_dim: int) -> float:
    # Loosely based on Srinivas et al. (2010), with edits: their Theorem 1 uses
    # |D|, the candidate set size, where this passes the input dimension. delta=0.1.
    return 2.0 * np.log(n_var_dim * (n_step + 1) ** 2 * np.pi**2 / 6.0 / 0.1)


def observe(prob, X: torch.Tensor, noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the objective and the feasibility slack at `X`.

    Args:
        prob: the PCO problem.
        X: `(n, d)`-dim tensor of configurations.
        noise: whether to add observation noise to the objective.

    Returns:
        `(n, 1)` objective (imputed where the run OOMed) and `(n, 1)` slack.
    """
    y = prob(X, noise=noise)
    c = prob.evaluate_slack(X)
    return y, c


def true_objective(prob, X: torch.Tensor) -> torch.Tensor:
    """Noiseless throughput at `X`, NaN where the configuration OOMs.

    Regret is scored on this: `prob(X)` imputes a throughput at an OOM, but the
    run realises none, so infeasible points are ignored. `noise=False` keeps the
    ground truth clean when `--noise_std` is set.

    Args:
        prob: the PCO problem.
        X: `(n, d)`-dim tensor of configurations.

    Returns:
        `(n,)`-dim tensor of throughputs.
    """
    y = prob(X, noise=False).flatten()
    return torch.where(prob.is_feasible(X), y, torch.full_like(y, float("nan")))


def feasible_mask(train_c: torch.Tensor) -> torch.Tensor:
    """Which observations ran, from the sign of the slack.

    The objective is imputed at an OOM, so it is finite everywhere and cannot
    say; the slack is what carries feasibility.
    """
    return (train_c > 0).flatten()


def generate_initial_data(prob, choices: torch.Tensor, rng, n: int, dtype=torch.double):
    """Sample `n` random rows, extending the sample until the objective is fittable.

    A draw that happens to be all-infeasible leaves the objective GP with no
    training data at all, so rows are added until MIN_FEASIBLE_INIT of them ran.
    The extra draws are counted as part of the initial budget.
    """
    idx = list(rng.choice(len(choices), size=n, replace=False))
    remaining = [i for i in range(len(choices)) if i not in set(idx)]
    rng.shuffle(remaining)

    while True:
        train_x = choices[idx].to(dtype=dtype)
        train_y, train_c = observe(prob, train_x)
        if int(feasible_mask(train_c).sum()) >= MIN_FEASIBLE_INIT or not remaining:
            return train_x, train_y, train_c, idx
        idx.append(remaining.pop())


def floor_lengthscales(model, bounds: torch.Tensor) -> None:
    """Bound each ARD lengthscale below by one lattice step.

    Every input is an ordinal knob on a log2 lattice with 2-5 levels, so
    `Normalize` puts adjacent levels `1 / span` apart. Below that the likelihood
    is flat -- no design on the lattice can tell a lengthscale of half a step
    from one of a hundredth -- and the fitted value is whatever the optimiser
    happened to land on. Those unidentified values are not harmless: read as
    relevance weights they make a knob look decisive when the data never said so.

    Args:
        model: a GP whose kernel has ARD lengthscales.
        bounds: `(2, d)`-dim tensor of the problem bounds, in log2 units.
    """
    kernel = getattr(model.covar_module, "base_kernel", model.covar_module)
    if not hasattr(kernel, "raw_lengthscale"):
        return
    span = (bounds[1] - bounds[0]).to(kernel.raw_lengthscale).clamp_min(1.0)
    lower = (1.0 / span).unsqueeze(0)
    kernel.register_constraint("raw_lengthscale", GreaterThan(lower))
    with torch.no_grad():
        kernel.lengthscale = kernel.lengthscale.clamp_min(lower)


# Fit attempts before a GP is given up on. BoTorch's default is 5; the physics
# prior makes the constraint MLL badly scaled -- its parameters see gradients
# around 1e-3 to 1e-6 where the kernel's see 1e2, so L-BFGS-B's single step
# length suits neither and the line search reports ABNORMAL -- and each attempt
# restarts from randomised hyperparameters, so more draws get past it.
FIT_ATTEMPTS = 10

# Iterations whose constraint fit fell back to a constant mean, per trial.
PRIOR_FALLBACKS: list[int] = []


def fit_with_fallback(model, bounds, ls_floor: bool, rebuild, step: int | None = None):
    """Fit `model`, falling back to a constant mean if the prior mean will not fit.

    The prior-mean fit fails outright on some pco64 data: every one of BoTorch's
    attempts ends in an L-BFGS-B line-search failure and it raises
    `ModelFittingError`, which would end a ten-trial run partway through. Rather
    than lose the run, that iteration is fitted with `rebuild()` -- the same model
    with a constant mean -- and the fallback is counted, so the result says how
    often the prior was actually in play.

    Args:
        model: the GP to fit.
        bounds: problem bounds, for the lengthscale floor.
        ls_floor: whether to bound lengthscales below by one lattice step.
        rebuild: callable returning a constant-mean version of the same model.
        step: BO iteration, recorded when a fallback happens.

    Returns:
        The fitted model: `model`, or the constant-mean rebuild.
    """
    if ls_floor:
        floor_lengthscales(model, bounds)
    try:
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(model.likelihood, model),
            max_attempts=FIT_ATTEMPTS,
        )
        return model
    except ModelFittingError:
        fallback = rebuild()
        if ls_floor:
            floor_lengthscales(fallback, bounds)
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(fallback.likelihood, fallback),
            max_attempts=FIT_ATTEMPTS,
        )
        if step is not None:
            PRIOR_FALLBACKS.append(step)
        return fallback


def initialize_models(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_c: torch.Tensor,
    bounds: torch.Tensor,
    prior=None,
    step: int = 0,
    ls_floor: bool = False,
):
    """Fit the objective GP and the constraint GP, both on every observation.

    A failure is not a hole in the objective data: it is imputed, and held away
    from the incumbent by the constraint model rather than by omission.

    With `prior`, each GP takes an analytic model of distributed training as its
    mean instead of a constant, and `fit_gpytorch_mll` learns that model's
    parameters alongside the kernel's. The means answer in physical units while
    the GPs standardise their targets, so the transform is built here and handed
    to both; `step` is the BO iteration, which sets how much weight the
    throughput prior still carries.

    Returns:
        (objective model, constraint model, ModelListGP over both).
    """
    d = train_x.shape[-1]
    fitted = ~torch.isnan(train_y).flatten()

    obj_transform = Standardize(m=1)
    obj_model = SingleTaskGP(
        train_x[fitted],
        train_y[fitted],
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=obj_transform,
        mean_module=(
            None if prior is None else prior.throughput_mean(obj_transform, step=step)
        ),
    )
    obj_model = fit_with_fallback(
        obj_model,
        bounds,
        ls_floor,
        rebuild=lambda: SingleTaskGP(
            train_x[fitted],
            train_y[fitted],
            input_transform=Normalize(d=d, bounds=bounds),
            outcome_transform=Standardize(m=1),
        ),
        step=step,
    )

    # Regression on the memory margin, a constant pad where the run OOMed. Those
    # entries are censored, so the fitted boundary is pulled toward the pad; a
    # censored likelihood would be the cleaner model.
    #
    # Stored negated, in BoTorch's convention that a constraint is feasible where
    # c(x) <= 0. The problem reports slack the other way round (feasible >= 0),
    # and every constrained acquisition below expects BoTorch's sign.
    con_transform = Standardize(m=1)
    con_model = SingleTaskGP(
        train_x,
        -train_c,
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=con_transform,
        mean_module=None if prior is None else prior.memory_mean(con_transform),
    )
    con_model = fit_with_fallback(
        con_model,
        bounds,
        ls_floor,
        rebuild=lambda: SingleTaskGP(
            train_x,
            -train_c,
            input_transform=Normalize(d=d, bounds=bounds),
            outcome_transform=Standardize(m=1),
        ),
        step=step,
    )

    return obj_model, con_model, ModelListGP(obj_model, con_model)


def constraint_posterior(con_model, X: torch.Tensor):
    """Posterior mean and std of the (negated) constraint at `(n, d)`-dim `X`."""
    with torch.no_grad():
        post = con_model.posterior(X.unsqueeze(-2))
        mean = post.mean.squeeze(-1).squeeze(-1)
        std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
    return mean, std


def prob_feasible(con_model, X: torch.Tensor) -> torch.Tensor:
    """P(c(x) <= 0), i.e. the probability the configuration fits in memory."""
    mean, std = constraint_posterior(con_model, X)
    normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
    return normal.cdf(-mean / std)


def build_acqf(acq_fn: str, models, train_x, train_y, train_c, beta: float):
    """Build a constrained EI acquisition function.

    `ucb_c`, `scbo`, `cmes_ibo` and `random` pick candidates by their own rules
    in `select`, so only the two EI variants are constructed here.
    """
    obj_model, con_model, model_list = models
    feasible = feasible_mask(train_c)
    best_f = train_y[feasible].max()

    if acq_fn == "ei":
        # EIC (Gardner et al., 2014): EI weighted by the probability of
        # feasibility. Output 0 is the objective, output 1 the constraint,
        # feasible on (-inf, 0].
        return LogConstrainedExpectedImprovement(
            model=model_list,
            best_f=best_f,
            objective_index=0,
            constraints={1: (None, 0.0)},
        )

    if acq_fn == "qnei":
        # Constrained noisy EI: qLogNEI (Ament et al., 2023) with `constraints`,
        # which weights each MC sample's improvement by a smoothed sigmoid
        # approximation to 1[c(x) <= 0] rather than an exact feasibility CDF.
        return qLogNoisyExpectedImprovement(
            model=model_list,
            X_baseline=train_x[feasible],
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            objective=GenericMCObjective(lambda Z, X=None: Z[..., 0]),
            constraints=[lambda Z: Z[..., 1]],
            prune_baseline=True,
        )

    raise ValueError(f"no acquisition function to build for {acq_fn!r}")


class OptunaPolicy:
    """TPE or CMA-ES, the two non-BO baselines, driven through Optuna's ask/tell.

    Both samplers search the integer bounding box, but most of the box is not a
    valid configuration -- the degrees must multiply to the GPU count -- so each
    proposal is repaired to the nearest row of the pool, measured in the unit
    cube as in ADMMBO. The repair is Baldwinian: the sampler is told the repaired
    row's value against the point it proposed, which is how a box-constrained
    optimiser meets a lattice it cannot represent.

    The constraint is handled natively where the sampler has a way to:

      * `tpe`: constrained TPE (Watanabe & Hutter, 2023) through
        `constraints_func`, splitting good and bad on feasibility first. It sees
        the imputed objective at an OOM, as the GP methods do.
      * `cmaes`: CmaEsSampler takes no constraints, so an OOM is told the death
        penalty `CMAES_OOM_VALUE`. The slack at an OOM is a fixed pad, so there
        is no violation magnitude to grade a softer penalty by, and CMA-ES only
        reads ranks. `with_margin` keeps integer dimensions from collapsing.

    The initial design is added as completed trials, so TPE starts past its
    random startup phase. CmaEsSampler ignores trials it did not generate, so it
    is instead centred on the best feasible initial row.
    """

    def __init__(self, acq_fn, bounds, train_x, train_y, train_c, seed):
        self.acq_fn = acq_fn
        self.lo = bounds[0]
        self.span = (bounds[1] - bounds[0]).clamp_min(1.0)
        self.names = [f"x{i}" for i in range(bounds.shape[1])]
        self.dists = {
            name: optuna.distributions.IntDistribution(int(lo), int(hi))
            for name, lo, hi in zip(self.names, bounds[0].tolist(), bounds[1].tolist())
        }
        feasible = feasible_mask(train_c)

        if acq_fn == "tpe":
            sampler = optuna.samplers.TPESampler(
                seed=seed, constraints_func=_optuna_constraints
            )
        else:
            best = train_x[feasible][train_y[feasible].argmax()]
            sampler = optuna.samplers.CmaEsSampler(
                x0=self._params(best),
                seed=seed,
                with_margin=True,
                warn_independent_sampling=False,
            )
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        for x, y, c in zip(train_x, train_y.flatten(), train_c.flatten()):
            self.study.add_trial(
                optuna.trial.create_trial(
                    params=self._params(x),
                    distributions=self.dists,
                    value=self._value(float(y), float(c)),
                    user_attrs={"slack": float(c)},
                    system_attrs={_CONSTRAINTS_KEY: [-float(c)]},
                )
            )
        self._trial = None

    def _params(self, x: torch.Tensor) -> dict:
        return {name: int(round(float(v))) for name, v in zip(self.names, x)}

    def _value(self, y: float, c: float) -> float:
        if self.acq_fn == "cmaes" and c <= 0:
            return CMAES_OOM_VALUE
        return y

    def next_query(self, pool: torch.Tensor) -> torch.Tensor:
        """Ask for a point in the box and repair it to the nearest pool row."""
        self._trial = self.study.ask(self.dists)
        x = torch.tensor(
            [self._trial.params[name] for name in self.names], dtype=pool.dtype
        )
        dists = torch.cdist(
            ((x - self.lo) / self.span).unsqueeze(0), (pool - self.lo) / self.span
        )
        return pool[dists.argmin()].unsqueeze(0)

    def tell(self, new_y: torch.Tensor, new_c: torch.Tensor) -> None:
        y, c = float(new_y.flatten()[0]), float(new_c.flatten()[0])
        self._trial.set_user_attr("slack", c)
        self.study.tell(self._trial, self._value(y, c))


def _optuna_constraints(trial) -> list[float]:
    """Optuna's convention is feasible iff every value <= 0; the slack's is > 0."""
    return [-trial.user_attrs["slack"]]


def select(
    acq_fn, models, train_x, train_y, train_c, pool, beta, rng, policy=None, table=None
):
    """Pick one configuration from `pool`, and report the acquisition value.

    `table` is the whole domain, which cKG's lookahead ranges over even when
    `pool` is narrower; it defaults to `pool`.

    Every method is maximised over the discrete candidate set: the space is a
    table of valid strategies, not a box.
    """
    if acq_fn == "random":
        # No models are fit for the random baseline, so this branch comes first.
        idx = rng.choice(len(pool), size=1, replace=False)
        return pool[idx], torch.tensor(float("nan"))

    if acq_fn in OPTUNA_METHODS:
        # No models either: the sampler keeps its own state across iterations.
        return policy.next_query(pool), torch.tensor(float("nan"))

    obj_model, con_model, model_list = models

    if acq_fn == "ucb_c":
        # UCB-C (Nguyen et al., ICLR 2024, Alg. 2): maximise the objective's
        # upper confidence bound over the *optimistic* feasible region
        #   O_t = {x : the constraint's confidence bound still admits feasibility}.
        # Coupled queries: both functions are then observed at x_t.
        acqf = ConstrainedUpperConfidenceBound(obj_model, con_model, beta=beta)
        new_x, acq_value = optimize_acqf_discrete(acqf, q=1, choices=pool)
        if not torch.isfinite(acq_value).all():
            # O_t empty: the acquisition is -inf everywhere. The paper's remedy
            # is to set the constraint GP's prior mean to the threshold; the
            # equivalent action on a fixed candidate set is to query wherever
            # feasibility is least ruled out.
            lcb_c = acqf.constraint_lcb(pool)
            top = torch.topk(-lcb_c, k=1).indices
            return pool[top], lcb_c[top].min()
        return new_x, acq_value.max()

    if acq_fn == "cmes_ibo":
        # CMES-IBO (Takeno et al., ICML 2022, Eq. 6). The sampled maxima f* are
        # drawn once, at construction, and shared by every candidate.
        acqf = ConstrainedMaxValueEntropySearch(
            obj_model, con_model, pool, num_mv_samples=CMES_IBO_SAMPLES
        )
        new_x, acq_value = optimize_acqf_discrete(acqf, q=1, choices=pool)
        return new_x, acq_value.max()

    if acq_fn == "ckg":
        # cKG: expected gain in the value of the recommendation made now.
        # Paper's Alg. 1 over the whole table, chunked over the pool.
        acqf = ConstrainedKnowledgeGradient(
            obj_model,
            con_model,
            pool if table is None else table,
            num_y_quantiles=CKG_Y_QUANTILES,
            num_c_quantiles=CKG_C_QUANTILES,
            best_per_outcome=CKG_BEST_PER_OUTCOME,
        )
        # Scored here because optimize_acqf_discrete breaks a tie on the first
        # row of the pool, and cKG can still be flat at zero -- no observation
        # would change the recommendation -- which would then spend the rest of
        # the budget on that one row.
        with torch.no_grad():
            vals = torch.cat(
                [
                    acqf(pool[i : i + CKG_BATCH].unsqueeze(-2))
                    for i in range(0, len(pool), CKG_BATCH)
                ]
            )
        tied = (vals >= vals.max() - CKG_TIE_TOL).nonzero(as_tuple=True)[0]
        pick = tied[torch.randint(len(tied), (1,))]
        return pool[pick], vals[pick].max()

    if acq_fn == "admmbo":
        # ADMMBO carries state across iterations, so the policy is built once in
        # main and driven here.
        new_x = policy.next_query(
            obj_model, con_model, pool, train_x, train_y, feasible_mask(train_c)
        )
        return new_x, torch.tensor(float("nan"))

    if acq_fn in ("cts", "scbo"):
        # Constrained Thompson sampling from SCBO (Eriksson & Poloczek, 2021),
        # via BoTorch's sampler: draw joint posterior samples of objective and
        # constraint, and take the best candidate that the sample says is
        # feasible. `cts` gets the whole table, `scbo` the trust region the
        # main loop narrowed it to.
        sampler = ConstrainedMaxPosteriorSampling(
            model=obj_model,
            constraint_model=ModelListGP(con_model),
            replacement=False,
        )
        with torch.no_grad():
            new_x = sampler(pool.unsqueeze(0), num_samples=1)
        return new_x.squeeze(0), torch.tensor(float("nan"))

    acqf = build_acqf(acq_fn, models, train_x, train_y, train_c, beta)
    new_x, acq_value = optimize_acqf_discrete(acqf, q=1, choices=pool)
    return new_x, acq_value.max()


def recommend(models, candidates, prob, train_x, train_y, train_c, pf_delta=PF_DELTA):
    """Best configuration the models believe in: posterior-mean argmax among
    candidates the constraint model calls feasible with confidence `1 - pf_delta`,
    falling back to the best feasible observation when none clears the bar.
    """
    obj_model, con_model, _ = models
    pf = prob_feasible(con_model, candidates)
    believed = pf >= 1.0 - pf_delta
    if not bool(believed.any()):
        feasible = feasible_mask(train_c)
        return train_x[feasible][train_y[feasible].argmax()].detach()
    with torch.no_grad():
        mean = obj_model.posterior(candidates.unsqueeze(-2)).mean.squeeze(-1).squeeze(-1)
    mean = mean.masked_fill(~believed, -float("inf"))
    return candidates[mean.argmax()].detach()


def main(args):
    dtype = torch.double
    dev = torch.device("cpu")

    noise_kwargs = {} if args.noise_std is None else {"noise_std": args.noise_std}
    prob = _PROBLEMS[args.problem](
        negate=False,
        **noise_kwargs,
    )
    prob.to(dtype=dtype, device=dev)
    noise_std = prob.noise_std
    print(f"problem: {prob.name}  noise std: {noise_std}")

    optimal_value = prob._optimal_value
    all_X = prob.candidates(dtype=dtype, device=dev)
    bounds_tensor = torch.tensor(prob._bounds, dtype=dtype).T.to(dev)
    n_feasible = int(prob.feasible.sum())
    print(
        f"candidate set: {len(all_X)} configurations, {n_feasible} feasible "
        f"({100 * (1 - n_feasible / len(all_X)):.0f}% OOM)  optimum {optimal_value:.5f}"
    )

    # Stateless: it only holds the instance constants and the bounds needed to
    # decode a normalised input. The mean modules themselves are rebuilt, with
    # fresh parameters, on every model fit.
    prior = ParallelismPrior(prob, bounds_tensor) if args.prior_mean else None
    if prior is not None:
        print("prior mean: analytic communication-cost and peak-memory models")

    all_trial_results = []
    seed_base = args.trial_offset if args.seed_offset is None else args.seed_offset

    for trial in range(args.trials):
        seed = trial + seed_base
        print(f"\n{'=' * 60}\nTrial {trial + 1}/{args.trials}  (seed={seed})\n{'=' * 60}")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        t0 = time.time()

        train_x, train_y, train_c, seen_idx = generate_initial_data(
            prob, all_X, rng, n=args.initial_random_samples, dtype=dtype
        )
        if len(seen_idx) > args.initial_random_samples:
            print(
                f"  initial sample extended to {len(seen_idx)} rows to reach "
                f"{MIN_FEASIBLE_INIT} feasible observations"
            )
        seen_idx = set(seen_idx)

        best_y_all, rec_true_all, rec_x_all, best_obs_x_all = [], [], [], []
        best_obs_true_all, best_obs_regret_all, log_best_obs_regret_all = [], [], []
        inference_regret_all, log_inference_regret_all = [], []
        best_rec_true_all, best_inference_regret_all = [], []
        log_best_inference_regret_all = []
        simple_regret_all, log_simple_regret_all = [], []
        n_infeasible_all, feasible_frac_all = [], []

        # Running bests, over feasible points only: an OOM realises no
        # throughput. Simple regret counts every evaluation the trial spent, the
        # initial design included -- a lucky random draw has still found it.
        _init_true = true_objective(prob, train_x)
        _init_true = _init_true[~torch.isnan(_init_true)]
        _best_simple_true = (
            float(_init_true.max()) if _init_true.numel() else -float("inf")
        )
        _best_rec_true = -float("inf")

        if args.acq_fn == "admmbo":
            policy = ADMMBOPolicy(bounds_tensor, rng)
        elif args.acq_fn in OPTUNA_METHODS:
            policy = OptunaPolicy(
                args.acq_fn, bounds_tensor, train_x, train_y, train_c, seed
            )
        else:
            policy = None
        scbo_state = ScboState(n_cand_max=len(all_X)) if args.acq_fn == "scbo" else None
        if scbo_state is not None:
            seed_scbo_state(scbo_state, train_y, train_c)
        scbo_ncand_all = []
        PRIOR_FALLBACKS.clear()
        models = None
        for itr in range(args.iterations):
            feasible = feasible_mask(train_c)

            if args.acq_fn not in ("random", *OPTUNA_METHODS):
                models = initialize_models(
                    train_x,
                    train_y,
                    train_c,
                    bounds_tensor,
                    prior=prior,
                    step=itr,
                    ls_floor=args.ls_floor,
                )

            # The full table is offered every iteration, as in the PO loop. A
            # repeat is not pathological: the constraint GP has already seen the
            # failure, so P(feasible) there has collapsed and the acquisition
            # moves on by itself, while re-querying a feasible point draws fresh
            # observation noise. --no_repeats enforces sampling without
            # replacement instead, which suits an exhaustive-search protocol.
            if args.no_repeats:
                mask = torch.ones(len(all_X), dtype=torch.bool)
                mask[list(seen_idx)] = False
                pool = all_X[mask]
                if len(pool) == 0:
                    print("  candidate set exhausted")
                    break
            else:
                pool = all_X

            # SCBO confines sampling to a trust region; `cts` gets the table.
            if scbo_state is not None:
                pool = trust_region_pool(
                    pool,
                    scbo_center(train_x, train_y, train_c),
                    bounds_tensor,
                    scbo_state.n_cand,
                    lengthscales=objective_lengthscales(models[0]),
                )

            beta = (
                args.ucb_beta
                if args.ucb_beta is not None
                else get_beta_t(itr, prob.dim)
            )
            new_x, acq_value = select(
                args.acq_fn,
                models,
                train_x,
                train_y,
                train_c,
                pool,
                beta,
                rng,
                policy=policy,
                table=all_X,
            )
            new_y, new_c = observe(prob, new_x)
            if isinstance(policy, OptunaPolicy):
                policy.tell(new_y, new_c)

            if scbo_state is not None:
                scbo_state = update_scbo_state(scbo_state, new_y, new_c)
                scbo_ncand_all.append(scbo_state.n_cand)

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])
            train_c = torch.cat([train_c, new_c])
            for row in new_x:
                seen_idx.add(int(torch.cdist(row.unsqueeze(0), all_X).argmin()))

            feasible = feasible_mask(train_c)
            n_infeasible_all.append(int((~feasible).sum()))
            feasible_frac_all.append(float(feasible.float().mean()))

            # Best feasible observation so far; the noisy value and the true one.
            best_y = float(train_y[feasible].max())
            best_y_all.append(best_y)
            best_obs_x = train_x[feasible][train_y[feasible].argmax()].detach()
            best_obs_true = float(true_objective(prob, best_obs_x.unsqueeze(0)))
            best_obs_x_all.append(best_obs_x.cpu().tolist())
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(max(optimal_value - best_obs_true, 0.0))
            log_best_obs_regret_all.append(
                math.log(max(optimal_value - best_obs_true, 1e-8))
            )

            new_true = true_objective(prob, new_x)
            new_true = new_true[~torch.isnan(new_true)]
            if new_true.numel():
                _best_simple_true = max(_best_simple_true, float(new_true.max()))
            simple_regret_all.append(max(optimal_value - _best_simple_true, 0.0))
            log_simple_regret_all.append(
                math.log(max(optimal_value - _best_simple_true, 1e-8))
            )

            if models is not None:
                rec_x = recommend(
                    models, all_X, prob, train_x, train_y, train_c, args.pf_delta
                )
                rec_true = float(true_objective(prob, rec_x.unsqueeze(0)))
                if math.isnan(rec_true):
                    # Recommended an infeasible configuration: no throughput is
                    # realised, so it earns the worst possible regret.
                    rec_regret = optimal_value
                else:
                    rec_regret = max(optimal_value - rec_true, 0.0)
                rec_x_all.append(rec_x.cpu().tolist())
                rec_true_all.append(rec_true)
                inference_regret_all.append(rec_regret)
                log_inference_regret_all.append(math.log(max(rec_regret, 1e-8)))

                if not math.isnan(rec_true):
                    _best_rec_true = max(_best_rec_true, rec_true)
                best_rec_true_all.append(_best_rec_true)
                best_inf_regret = max(optimal_value - _best_rec_true, 0.0)
                best_inference_regret_all.append(best_inf_regret)
                log_best_inference_regret_all.append(
                    math.log(max(best_inf_regret, 1e-8))
                )

            if args.verbose:
                inf_str = (
                    f"  inf_regret={inference_regret_all[-1]:.4f}"
                    if inference_regret_all
                    else ""
                )
                tr_str = (
                    f"  tr_ncand={scbo_state.n_cand}" if scbo_state is not None else ""
                )
                print(
                    f"  itr {itr + 1}: acq={float(acq_value):.4f}  "
                    f"best_y={best_y:.4f}  "
                    f"best_obs_regret={best_obs_regret_all[-1]:.4f}"
                    f"{inf_str}  n_oom={n_infeasible_all[-1]}{tr_str}"
                )
            else:
                print(".", end="", flush=True)

        t1 = time.time()
        print(
            f"\nTrial {trial + 1} done in {t1 - t0:.1f}s — "
            f"final best_y={best_y_all[-1]:.4f}, "
            f"{n_infeasible_all[-1]} of {len(train_y)} evaluations OOMed"
        )

        all_trial_results.append(
            {
                "trial": trial,
                "seed": seed,
                "time_seconds": t1 - t0,
                "best_y_all": best_y_all,
                "rec_x_all": rec_x_all,
                "rec_true_all": rec_true_all,
                "inference_regret_all": inference_regret_all,
                "log_inference_regret_all": log_inference_regret_all,
                "best_rec_true_all": best_rec_true_all,
                "best_inference_regret_all": best_inference_regret_all,
                "log_best_inference_regret_all": log_best_inference_regret_all,
                "simple_regret_all": simple_regret_all,
                "log_simple_regret_all": log_simple_regret_all,
                "best_obs_x_all": best_obs_x_all,
                "best_obs_true_all": best_obs_true_all,
                "best_obs_regret_all": best_obs_regret_all,
                "log_best_obs_regret_all": log_best_obs_regret_all,
                "n_infeasible_all": n_infeasible_all,
                "feasible_frac_all": feasible_frac_all,
                "candidates": train_x.cpu().tolist(),
                "seen_y": train_y.flatten().cpu().tolist(),
                "seen_c": train_c.flatten().cpu().tolist(),
            }
        )
        if scbo_state is not None:
            all_trial_results[-1]["scbo_ncand_all"] = scbo_ncand_all
        if prior is not None:
            all_trial_results[-1]["prior_fallback_iters"] = list(PRIOR_FALLBACKS)

    ucb_beta_fixed = args.ucb_beta if args.acq_fn == "ucb_c" else None
    results = {
        "problem": args.problem,
        "acq_fn": args.acq_fn,
        "iterations": args.iterations,
        "ucb_beta": ucb_beta_fixed,
        "initial_random_samples": args.initial_random_samples,
        "noise_std": noise_std,
        "no_repeats": args.no_repeats,
        "pf_delta": args.pf_delta,
        "prior_mean": args.prior_mean,
        "ls_floor": args.ls_floor,
        "optimal_value": optimal_value,
        "num_trials": args.trials,
        "trials": all_trial_results,
    }

    beta_tag = f"_beta{ucb_beta_fixed}" if ucb_beta_fixed is not None else ""
    prior_tag = "_prior" if args.prior_mean else ""
    lsf_tag = "_lsfloor" if args.ls_floor else ""
    out_dir = REPO_ROOT / "results" / f"{args.problem}{args.folder_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{args.problem}_{args.acq_fn}{beta_tag}{prior_tag}{lsf_tag}"
        f"_{args.trials}trials_{args.iterations}iterations_results.json"
    )
    out_path.write_text(json.dumps(results, indent=4))
    print(f"\nSaved results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Constrained BO baselines for the PCO problems."
    )
    parser.add_argument("--problem", type=str, default="pco32", choices=list(_PROBLEMS))
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument(
        "--acq_fn",
        type=str,
        default="qnei",
        choices=[
            "qnei", "ei", "ucb_c", "cts", "scbo", "cmes_ibo", "ckg", "admmbo", "random",
            "tpe", "cmaes",
        ],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ucb_beta", type=float, default=None)
    parser.add_argument("--initial_random_samples", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--seed_offset", type=int, default=None)
    parser.add_argument(
        "--no_repeats",
        action="store_true",
        help="Remove already-evaluated configurations from the candidate pool. "
        "Off by default: the full table is offered every iteration, matching the "
        "PO loop and standard constrained BO, where a repeat is discouraged by "
        "the feasibility-weighted acquisition rather than forbidden outright.",
    )
    parser.add_argument(
        "--ls_floor",
        action="store_true",
        help="Bound each ARD lengthscale below by one lattice step (1 / span). "
        "Below that spacing the likelihood is flat, so the fitted value is "
        "unidentified -- and it still feeds SCBO's region shape and every "
        "acquisition's posterior.",
    )
    parser.add_argument(
        "--prior_mean",
        action="store_true",
        help="Give both GPs a physics-informed prior mean -- an analytic "
        "communication-cost model for throughput and a peak-memory model for the "
        "constraint -- whose parameters are fit jointly with the kernel, instead "
        "of the default constant mean.",
    )
    parser.add_argument(
        "--pf_delta",
        type=float,
        default=PF_DELTA,
        help="A recommendation must be feasible with probability >= 1 - pf_delta. "
        "0.5 reproduces the coin-flip threshold used before.",
    )
    parser.add_argument("--folder_prefix", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
