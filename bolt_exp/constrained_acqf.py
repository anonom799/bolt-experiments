"""Acquisition functions for coupled constrained BO over a finite candidate set.

BoTorch ships constrained EI (`LogConstrainedExpectedImprovement`, and the
`constraints=` weighting on the MC acquisitions) but nothing for UCB-C,
CMES-IBO or cKG, so these are implemented here against the standard
`AcquisitionFunction` interface. That is what lets `optimize_acqf_discrete`
select candidates for every model-based method in the benchmark, instead of
each rule hand-rolling its own scan over the pool. ADMMBO carries state across
iterations, so it is a policy the outer loop drives rather than an acquisition.

All four take two independent single-output GPs: an objective model fit on
every observation with a measured objective value, and a constraint model fit
on every observation. All follow BoTorch's convention that the constraint is
satisfied where c(x) <= 0, so the caller negates the problem's slack before
fitting.
"""

import math

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


def _expected_improvement(best: Tensor, mean: Tensor, sigma: Tensor) -> Tensor:
    """EI for a minimisation problem with Gaussian `mean`, `sigma` and incumbent `best`."""
    sigma = sigma.clamp_min(1e-12)
    z = (best - mean) / sigma
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    return sigma * (z * normal.cdf(z) + torch.exp(normal.log_prob(z)))


def _mean_sigma(model: Model, X: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior mean and std of a single-output model at `(b, 1, d)`-dim `X`."""
    post = model.posterior(X)
    mean = post.mean.squeeze(-1).squeeze(-1)
    sigma = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
    return mean, sigma


def _expected_max_of_lines(a: Tensor, s: Tensor) -> Tensor:
    r"""`E[max_i a_i + s_i Z]` for `Z ~ N(0, 1)`, over the last dim of `a`, `s`.

    Scott, Frazier & Powell (2011), Alg. 1, vectorised: the maximum is convex
    and piecewise linear in `Z`, so each line owns one (possibly empty) interval
    of it, and integrating the line against the normal density over that
    interval is closed-form. The interval of line `i` is bounded below by its
    crossings with every shallower line and above by those with every steeper
    one. Among lines of equal slope only the highest survives, the lowest
    index breaking exact ties, so duplicate lines are counted once.

    `O(L^2)` in the number of lines, against the `O(L log L)` of the sorted
    sweep, in exchange for batching over every leading dim.
    """
    ds = s.unsqueeze(-1) - s.unsqueeze(-2)  # [i, j] = s_i - s_j
    da = a.unsqueeze(-2) - a.unsqueeze(-1)  # [i, j] = a_j - a_i
    # Line i is above line j where (s_i - s_j) z >= a_j - a_i.
    cross = da / torch.where(ds == 0, torch.ones_like(ds), ds)
    lo = torch.where(ds > 0, cross, torch.full_like(cross, -math.inf)).amax(-1)
    hi = torch.where(ds < 0, cross, torch.full_like(cross, math.inf)).amin(-1)

    idx = torch.arange(a.shape[-1], device=a.device)
    earlier = idx.unsqueeze(0) < idx.unsqueeze(1)  # [i, j] = j < i
    beaten = (ds == 0) & ((da > 0) | ((da == 0) & earlier))
    empty = beaten.any(-1) | (lo >= hi)

    normal = torch.distributions.Normal(torch.zeros_like(a), torch.ones_like(a))
    pdf_lo, pdf_hi = torch.exp(normal.log_prob(lo)), torch.exp(normal.log_prob(hi))
    mass = a * (normal.cdf(hi) - normal.cdf(lo)) + s * (pdf_lo - pdf_hi)
    return torch.where(empty, torch.zeros_like(mass), mass).sum(-1)


def _gaussian_quantiles(n: int) -> Tensor:
    r"""`n` evenly spaced standard-normal quantiles, `Phi^-1(k / (n + 1))`.

    `n = 9` gives `Phi^-1(0.1), ..., Phi^-1(0.9)`, Pearce et al. (2020)'s set.
    """
    levels = torch.arange(1, n + 1, dtype=torch.double) / (n + 1)
    return torch.distributions.Normal(0.0, 1.0).icdf(levels)


def _one_step_lookahead(
    model: Model, table: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""What one observation at each row of `X` does to the posterior on `table`.

    For `(N, d)`-dim `table` and `(b, d)`-dim `X`, returns the current mean and
    variance over `table`, both `(N,)`-dim, and the `(b, N)`-dim sensitivity

        sigma~(x', x) = k_n(x', x) / sqrt(k_n(x, x) + noise(x)),

    so that after observing `x`, `mu_{n+1}(x') = mu_n(x') + sigma~(x', x) Z`
    with `Z ~ N(0, 1)`, and `k_{n+1}(x', x') = k_n(x', x') - sigma~(x', x)^2`.
    """
    N = table.shape[0]
    post = model.posterior(torch.cat([table, X], dim=0))
    mean = post.mean.squeeze(-1)[:N]
    cov = post.distribution.covariance_matrix
    var = cov.diagonal()[:N]
    noisy_var = model.posterior(X.unsqueeze(-2), observation_noise=True).variance
    scale = noisy_var.reshape(-1, 1).clamp_min(1e-12).sqrt()
    return mean, var, cov[N:, :N] / scale



class ConstrainedUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""UCB-C: UCB of the objective over the optimistic feasible region.

    Nguyen et al., ICLR 2024, Alg. 2. At iteration `t` the optimistic feasible
    region is

        O_t = {x : lcb_c(x) <= 0},   lcb_c(x) = mu_c(x) - sqrt(beta) sigma_c(x),

    i.e. every input whose constraint confidence bound still admits feasibility.
    The acquisition is the objective's upper confidence bound restricted to that
    region, and `-inf` outside it.

    An all-`-inf` result means `O_t` is empty: feasibility has been ruled out
    everywhere the acquisition can see. That is a policy decision for the outer
    loop (the paper's remedy is to set the constraint GP's prior mean to the
    threshold), not an acquisition value, so it is signalled rather than patched
    over here -- the caller checks for a non-finite best value.

    Example:
        >>> acqf = ConstrainedUpperConfidenceBound(obj_model, con_model, beta=4.0)
        >>> cand, val = optimize_acqf_discrete(acqf, q=1, choices=pool)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Model,
        beta: float,
        maximize: bool = True,
    ) -> None:
        r"""Constrained upper confidence bound.

        Args:
            model: single-output GP over the objective, fit on every
                observation with a measured objective value.
            constraint_model: single-output GP over the negated slack, fit on
                all data. Feasible where its value is <= 0.
            beta: exploration weight; the bound uses `sqrt(beta)`.
            maximize: if False, the objective is minimised.
        """
        super().__init__(model=model)
        self.constraint_model = constraint_model
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate UCB-C on `(b, 1, d)`-dim `X`, returning a `(b,)`-dim tensor."""
        root_beta = self.beta.to(X).sqrt()
        f_mean, f_sigma = _mean_sigma(self.model, X)
        c_mean, c_sigma = _mean_sigma(self.constraint_model, X)

        ucb_f = f_mean + root_beta * f_sigma
        if not self.maximize:
            ucb_f = -f_mean + root_beta * f_sigma
        lcb_c = c_mean - root_beta * c_sigma

        neg_inf = torch.full_like(ucb_f, -float("inf"))
        return torch.where(lcb_c <= 0.0, ucb_f, neg_inf)

    def constraint_lcb(self, X: Tensor) -> Tensor:
        r"""Lower confidence bound on the constraint at `(b, d)`-dim `X`.

        Exposed for the empty-`O_t` fallback: the outer loop queries where
        feasibility is least ruled out, i.e. at the smallest `lcb_c`.
        """
        c_mean, c_sigma = _mean_sigma(self.constraint_model, X.unsqueeze(-2))
        return c_mean - self.beta.to(X).sqrt() * c_sigma


class ConstrainedMaxValueEntropySearch(AnalyticAcquisitionFunction):
    r"""CMES-IBO: constrained max-value entropy via an information lower bound.

    Takeno et al., ICML 2022, Eq. 6:

        alpha(x) = -(1/K) sum_k log Zbar_k(x),
        Zbar_k(x) = 1 - P(f(x) > f*_k) P(x feasible).

    The two probabilities factorise because the objective and constraint models
    are independent. `Zbar <= 1` makes every term non-negative, which is the
    property a naive constrained extension of MES loses.

    The sampled constrained maxima `f*_k` are drawn once at construction and
    held as a buffer, following `qMaxValueEntropy`. They must be shared by every
    candidate: `optimize_acqf_discrete` evaluates the pool in chunks, so drawing
    them inside `forward` would score different chunks against different `f*`
    and make the values incomparable.

    On a finite candidate set `f*` is sampled directly rather than through the
    random-Fourier-feature sample paths CMES-IBO needs on a continuous domain:
    draw joint posterior samples of objective and constraint over every
    candidate, and take the largest objective sample among the candidates that
    sample calls feasible.

    Example:
        >>> acqf = ConstrainedMaxValueEntropySearch(obj_model, con_model, pool)
        >>> cand, val = optimize_acqf_discrete(acqf, q=1, choices=pool)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Model,
        candidate_set: Tensor,
        num_mv_samples: int = 10,
    ) -> None:
        r"""Constrained max-value entropy search.

        Args:
            model: single-output GP over the objective, fit on every
                observation with a measured objective value.
            constraint_model: single-output GP over the negated slack, fit on
                all data. Feasible where its value is <= 0.
            candidate_set: `(N, d)`-dim tensor of candidates to sample `f*` over.
            num_mv_samples: number of sampled constrained maxima `K`.
        """
        super().__init__(model=model)
        self.constraint_model = constraint_model
        self.num_mv_samples = num_mv_samples
        self.register_buffer(
            "max_values", self._sample_max_values(candidate_set, num_mv_samples)
        )

    def _sample_max_values(self, candidate_set: Tensor, num_samples: int) -> Tensor:
        r"""Sample the constrained maximum over `candidate_set`.

        Returns:
            `(K,)`-dim tensor; `-inf` for a sample in which no candidate came
            out feasible.
        """
        shape = torch.Size([num_samples])
        with torch.no_grad():
            f_s = (
                self.model.posterior(candidate_set.unsqueeze(0))
                .rsample(shape)
                .squeeze(-1)
                .squeeze(1)
            )
            c_s = (
                self.constraint_model.posterior(candidate_set.unsqueeze(0))
                .rsample(shape)
                .squeeze(-1)
                .squeeze(1)
            )
        feasible = c_s <= 0.0
        return f_s.masked_fill(~feasible, -float("inf")).max(dim=-1).values

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate CMES-IBO on `(b, 1, d)`-dim `X`, returning a `(b,)`-dim tensor."""
        f_mean, f_sigma = _mean_sigma(self.model, X)
        c_mean, c_sigma = _mean_sigma(self.constraint_model, X)

        normal = torch.distributions.Normal(
            torch.zeros(1, dtype=f_mean.dtype, device=f_mean.device),
            torch.ones(1, dtype=f_mean.dtype, device=f_mean.device),
        )
        # P(c(x) <= 0): the probability the configuration is feasible.
        pf = normal.cdf(-c_mean / c_sigma)

        f_stars = self.max_values.to(f_mean)
        # gamma_k(x) = (f*_k - mu(x)) / sigma(x); P(f(x) > f*_k) = 1 - Phi(gamma).
        gamma = (f_stars.unsqueeze(-1) - f_mean.unsqueeze(0)) / f_sigma.unsqueeze(0)
        p_improve = 1.0 - normal.cdf(gamma)
        # f* = -inf (no feasible candidate in that sample) makes improvement
        # certain, reducing the term to -log(1 - P(feasible)): pure feasibility
        # seeking, the sensible behaviour when nothing is known to work.
        p_improve = torch.nan_to_num(p_improve, nan=1.0)

        z_bar = (1.0 - p_improve * pf.unsqueeze(0)).clamp_min(1e-12)
        return (-z_bar.log()).mean(dim=0)


class ConstrainedKnowledgeGradient(AnalyticAcquisitionFunction):
    r"""cKG: expected gain in the value of the recommendation.

    Ungredda & Branke (2021), "Bayesian Optimisation for Constrained
    Problems", arXiv:2105.13245, Alg. 1 and App. D, on a finite table. A state
    is worth what a recommendation made from it would be worth, the
    feasibility-weighted posterior mean `mu_f PF`, and the acquisition is the
    expected gain from observing (f, c) at `x` over today's recommendation
    `x_r = argmax mu_f^n PF^n` (eq. 9):

        cKG(x) = E[ max_{x'} mu_f^{n+1}(x') PF^{n+1}(x')
                    - mu_f^n(x_r) PF^{n+1}(x_r) ].

    Computed as in the paper, for each `x`:

    1. The outcomes of observing `x` are the Cartesian product of Gaussian
       quantiles `Z_y` for the objective and `Z_c` for the constraint (App. D).
    2. Under each outcome, the best point `argmax (mu_f^n + sigma~_f Z_y)
       PF^{n+1}` joins the discretisation `X_d` (Alg. 1).
    3. For each `Z_c`, `PF^{n+1}` is fixed, and the objective's outcome is
       integrated out over `X_d` in closed form with the discrete KG of Scott
       et al. (2011) (eq. 10); cKG averages over `Z_c` (eq. 11).

    `Z_y` must include 0, so its count is odd: then, for each `Z_c`, `X_d`
    holds the maximiser of `mu_f^n PF^{n+1}`, which is at least as good as
    `x_r`, and each term is non-negative (Lemma 1) without clipping.

    Departures from the paper, all from the finite domain:

    * each best point in step 2 is an exact argmax over the table, where the
      paper runs L-BFGS;
    * every candidate is scored, where the paper searches for `x` by a Latin
      hypercube refined with L-BFGS;
    * the paper fixes `Z_y` at `Phi^-1(0.1), ..., Phi^-1(0.9)` but leaves the
      number of `Z_c` quantiles open; it is `num_c_quantiles` here;
    * each outcome contributes its `best_per_outcome` best rows to `X_d`,
      not just the argmax, which on a table can repeat until `X_d` holds one
      row and the acquisition is identically zero. `E[max]` over a subset
      lower-bounds the inner maximum, so the extra rows only tighten it.

    The candidates scored must be rows of `table`, as the argmax in step 2 is
    over the table alone.

    Note the product rewards a low `PF` wherever `mu_f` is negative, so it reads
    as a value only for a positive objective. It is also not the rule
    `recommend()` applies, so cKG optimises a slightly different quantity than
    inference regret measures.

    The models are independent, so the objective and constraint outcomes are
    independent; the problem is coupled, so every outcome updates both. A
    fantasy supposes an objective observation at `x` whatever its feasibility,
    which holds only when OOM objectives are imputed. With NaN objectives the
    fantasy is optimistic there.

    Example:
        >>> acqf = ConstrainedKnowledgeGradient(obj_model, con_model, table)
        >>> cand, val = optimize_acqf_discrete(acqf, q=1, choices=pool,
        ...                                    max_batch_size=32)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Model,
        table: Tensor,
        num_y_quantiles: int = 9,
        num_c_quantiles: int = 5,
        best_per_outcome: int = 2,
        maximize: bool = True,
    ) -> None:
        r"""Constrained knowledge gradient over a finite table.

        Args:
            model: single-output GP over the objective.
            constraint_model: single-output GP over the negated slack, feasible
                where its value is <= 0.
            table: `(N, d)`-dim domain, over which the best point under each
                outcome is found.
            num_y_quantiles: number of objective quantiles `Z_y`; odd, so that
                it includes 0. The paper's is 9.
            num_c_quantiles: number of constraint quantiles `Z_c`, which the
                paper does not fix.
            best_per_outcome: rows each outcome contributes to `X_d`. The
                paper's is 1, which on a table lets `X_d` collapse to one row
                and zero the acquisition, so the default keeps a runner-up.
            maximize: if False, the objective is minimised.
        """
        if num_y_quantiles % 2 == 0:
            raise ValueError(
                f"num_y_quantiles must be odd so Z_y includes 0, got {num_y_quantiles}"
            )
        if best_per_outcome < 1:
            raise ValueError(f"best_per_outcome must be >= 1, got {best_per_outcome}")
        super().__init__(model=model)
        self.constraint_model = constraint_model
        self.maximize = maximize
        self.best_per_outcome = min(int(best_per_outcome), table.shape[0])
        self.register_buffer("table", table.clone())
        self.register_buffer("z_y", _gaussian_quantiles(num_y_quantiles).to(table))
        self.register_buffer("z_c", _gaussian_quantiles(num_c_quantiles).to(table))
        with torch.no_grad():
            f_mean, pf = self._posterior_stats(table)
            self.xr_index = int((f_mean * pf).argmax())

    def _posterior_stats(self, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Objective mean and feasibility probability at `(n, d)`-dim `X`."""
        f_mean, _ = _mean_sigma(self.model, X.unsqueeze(-2))
        c_mean, c_sigma = _mean_sigma(self.constraint_model, X.unsqueeze(-2))
        if not self.maximize:
            f_mean = -f_mean
        normal = torch.distributions.Normal(
            torch.zeros_like(c_mean), torch.ones_like(c_sigma)
        )
        return f_mean, normal.cdf(-c_mean / c_sigma)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate cKG on `(b, 1, d)`-dim `X`, returning a `(b,)`-dim tensor."""
        X = X.squeeze(-2)
        table = self.table.to(X)
        f_mean, _, f_sens = _one_step_lookahead(self.model, table, X)
        c_mean, c_var, c_sens = _one_step_lookahead(self.constraint_model, table, X)
        if not self.maximize:
            f_mean, f_sens = -f_mean, -f_sens

        # (n_c, b, N): PF over the table after each constraint outcome at `x`.
        z_c = self.z_c.to(X).view(-1, 1, 1)
        c_sigma_next = (c_var - c_sens**2).clamp_min(1e-12).sqrt()
        normal = torch.distributions.Normal(0.0, 1.0)
        pf = normal.cdf(-(c_mean + c_sens * z_c) / c_sigma_next)

        # Alg. 1: `X_d` is the best `best_per_outcome` rows under each
        # (Z_y, Z_c) outcome. The paper's single argmax is a different point
        # each time only because L-BFGS starts it elsewhere; an exact argmax
        # over a table repeats, and one row winning every outcome leaves a
        # single line whose expected maximum is its own intercept -- the
        # baseline -- so the acquisition would be identically zero.
        z_y = self.z_y.to(X).view(-1, 1, 1, 1)
        best = ((f_mean + f_sens * z_y) * pf).topk(self.best_per_outcome, dim=-1).indices
        X_d = best.permute(2, 0, 1, 3).reshape(X.shape[0], -1)

        # Eq. 10 for each Z_c: `mu_{n+1} PF_{n+1}` is a line in Z_y on `X_d`.
        idx = X_d.expand(len(self.z_c), *X_d.shape)
        a = (f_mean * pf).gather(-1, idx)
        s = (f_sens * pf).gather(-1, idx)
        baseline = f_mean[self.xr_index] * pf[..., self.xr_index]

        # Eq. 11: average over the constraint outcomes.
        return (_expected_max_of_lines(a, s) - baseline).mean(dim=0)


class ADMMBOPolicy:
    r"""ADMMBO: constrained BO by alternating direction method of multipliers.

    Ariafar, Coll-Font, Brooks & Dy (JMLR 2019), "ADMMBO: Bayesian
    Optimization with Unknown Constraints using ADMM", adapted to a finite
    candidate set and a coupled evaluation.

    The constrained problem is split over two copies of the decision variable,

        min_x f(x) + h(z)  s.t.  x = z,    h(z) = M * 1[c(z) > 0],

    and solved by alternating three steps on the scaled augmented Lagrangian:

        OP:   x <- argmin_x  f(x) + (rho/2) ||x - z + u||^2
        FP:   z <- argmin_z  M P(c(z) > 0) + (rho/2) ||x - z + u||^2
        dual: u <- u + (x - z)

    The subproblems hold different variables fixed, so the quadratic is centred
    on `z - u` in OP and on `x + u` in FP; sharing one centre would leave `z`
    chasing its own previous value.

    Each is solved by EI on a model of its augmented objective, not on a GP
    fitted to augmented values (section 3.1). The quadratic is deterministic,
    so OP shifts the objective GP by it (eq. 10-11); FP's objective is
    two-valued, giving EI in closed form (eq. 13):

        EI(z) = (1 - p) max(0, best - quad) + p max(0, best - quad - M).

    Neither runs to convergence: each gets a fixed budget, and its solution is
    the argmin of the augmented objective over the observed data (OPT and FEAS
    line 10), since the last EI query is exploratory rather than a minimiser.

    `rho` and `penalty_M` are in units of the objective's observed range, the
    scale Proposition 1 sets `M` from. The paper's objectives are errors on
    `[0, 1]`; this one is throughput, and `SingleTaskGP` untransforms its
    posterior, so a constant penalty would vanish against the raw values.

    Distances are measured in the unit cube, or the widest log2 range would
    dominate the quadratic. The Alg. 3.1 line 12 stopping rule is not
    implemented: the budget is fixed and shared, so the policy keeps sweeping.

    Not an `AcquisitionFunction`: it carries state across iterations, so the
    outer loop drives it rather than `optimize_acqf_discrete`.

    Example:
        >>> policy = ADMMBOPolicy(bounds, rng)
        >>> x = policy.next_query(obj_model, con_model, pool, train_x, train_y, train_c)
    """

    def __init__(
        self,
        bounds: Tensor,
        rng,
        rho: float = 0.1,
        penalty_M: float = 20.0,
        steps_per_subproblem: int = 2,
        first_sweep_steps: int = 10,
        maximize: bool = True,
    ) -> None:
        r"""ADMM policy over a finite candidate set.

        Defaults follow section 5.1: `rho = 0.1`, `M` in `{20, 50}`, and a
        larger opening sweep (`{10, 20, 50}` queries) than the rest (`{2, 5}`).

        Args:
            bounds: `(2, d)`-dim tensor used to normalise distances.
            rng: numpy Generator, used only to seed the copies.
            rho: augmented-Lagrangian penalty weight, in units of the
                objective's observed range.
            penalty_M: cost charged to an infeasible point, in units of the
                objective's observed range. Proposition 1 needs more than one;
                the paper uses 20 to 50.
            steps_per_subproblem: queries spent on OP, then on FP, per sweep.
            first_sweep_steps: the same for the opening sweep, which the paper
                gives the larger budget.
            maximize: if True, the problem's objective is maximised.
        """
        self.bounds = bounds
        self.rng = rng
        self.rho = float(rho)
        self.penalty_M = float(penalty_M)
        self.steps = max(1, int(steps_per_subproblem))
        self.first_steps = max(1, int(first_sweep_steps))
        self.maximize = maximize

        self.z = None  # constraint copy, normalised
        self.u = None  # scaled dual
        self._x = None  # objective copy, normalised
        self._phase = "op"
        self._first = True  # the opening sweep draws the larger budget
        self._left = self.first_steps

    def _norm(self, X: Tensor) -> Tensor:
        r"""Map `(n, d)`-dim `X` into the unit cube."""
        lo, hi = self.bounds[0].to(X), self.bounds[1].to(X)
        return (X - lo) / (hi - lo).clamp_min(1e-12)

    def _anchor(self) -> Tensor:
        r"""Centre of the quadratic penalty for the phase in progress.

        `||x - z + u||^2` is `||x - (z - u)||^2` in `x` and `||z - (x + u)||^2`
        in `z`.
        """
        return self.z - self.u if self._phase == "op" else self._x + self.u

    def _scale(self, model: Model, obs: Tensor) -> float:
        r"""The objective's range, the unit `rho` and `penalty_M` are quoted in.

        Proposition 1 sets `M` from the range of `f`: above it, the penalised
        problem is equivalent to the constrained one. Falls back to the GP's own
        standardisation until two objectives have been seen.
        """
        vals = obs[torch.isfinite(obs)]
        if vals.numel() > 1:
            r = float(vals.max() - vals.min())
            if math.isfinite(r) and r > 0:
                return r
        transform = getattr(model, "outcome_transform", None)
        if transform is not None and bool(getattr(transform, "_is_trained", False)):
            s = float(transform.stdvs.reshape(-1)[0])
            if math.isfinite(s) and s > 0:
                return s
        return 1.0

    def _quad(self, X_norm: Tensor, rho: float) -> Tensor:
        r"""Quadratic penalty at normalised `X_norm`, for the current phase."""
        return 0.5 * rho * ((X_norm - self._anchor()) ** 2).sum(dim=-1)

    def _objective_mean(self, model: Model, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Objective posterior at `(n, d)`-dim `X`, on ADMM's minimisation sign."""
        mean, sigma = _mean_sigma(model, X.unsqueeze(-2))
        return (-mean if self.maximize else mean), sigma

    def _prob_infeasible(self, constraint_model: Model, X: Tensor) -> Tensor:
        r"""`P(c(x) > 0)` at `(n, d)`-dim `X`, under the negated-slack convention."""
        c_mean, c_sigma = _mean_sigma(constraint_model, X.unsqueeze(-2))
        normal = torch.distributions.Normal(
            torch.zeros_like(c_mean), torch.ones_like(c_sigma)
        )
        return 1.0 - normal.cdf(-c_mean / c_sigma)

    def _commit(self, solution: Tensor) -> None:
        r"""Take `solution` as the spent subproblem's answer and roll the phase."""
        if self._phase == "op":
            self._x = solution
            self._phase = "fp"
        else:
            self.z = solution
            self.u = self.u + (self._x - self.z)
            self._phase = "op"
            self._first = False  # a full sweep has closed
        self._left = self.first_steps if self._first else self.steps

    def _close_subproblem(
        self, train_x: Tensor, obs: Tensor, feasible: Tensor, rho: float, M: float
    ) -> None:
        r"""Solve the spent subproblem: the best point in the whole dataset.

        OPT and FEAS line 10 minimise over the accumulated data, not just the
        points this subproblem tried, so a sweep can return to an earlier point
        the moving anchor now favours. Both augmented objectives are observed --
        `f` is measured, `h` is `M` exactly when the run failed -- so no
        posterior is needed. The paper's two datasets coincide here, since every
        query reports both.
        """
        X = self._norm(train_x)
        quad = self._quad(X, rho)
        if self._phase == "op":
            value = obs + quad
        else:
            value = M * (~feasible).to(quad) + quad
        value = torch.nan_to_num(value, nan=math.inf)
        self._commit(X[int(value.argmin())])

    def next_query(
        self,
        model: Model,
        constraint_model: Model,
        pool: Tensor,
        train_x: Tensor,
        train_y: Tensor,
        feasible: Tensor,
    ) -> Tensor:
        r"""Pick the next configuration to evaluate.

        Args:
            model: objective GP.
            constraint_model: GP over the negated slack, feasible where <= 0.
            pool: `(N, d)`-dim candidate set.
            train_x: `(n, d)`-dim observed inputs.
            train_y: `(n, 1)`-dim observed objectives, possibly with NaN.
            feasible: `(n,)`-dim boolean mask of which observations ran.

        Returns:
            Tensor: `(1, d)`-dim chosen configuration.
        """
        P = self._norm(pool)
        obs = train_y.flatten()
        obs = -obs if self.maximize else obs  # ADMM minimises
        scale = self._scale(model, obs)
        rho, M = self.rho * scale, self.penalty_M * scale

        if self.z is None:
            # Start the copies at the best feasible observation, the dual at zero.
            start = pool[0]
            if bool(feasible.any()):
                seen = torch.nan_to_num(obs[feasible], nan=math.inf)
                start = train_x[feasible][int(seen.argmin())]
            self.z = self._norm(start.unsqueeze(0)).squeeze(0)
            self.u = torch.zeros_like(self.z)
            self._x = self.z.clone()

        if self._left == 0:
            # Budget ran out last call and its observation has now landed, so
            # the subproblem can close before the anchor moves.
            self._close_subproblem(train_x, obs, feasible, rho, M)

        quad = self._quad(P, rho)
        # The incumbent is on the augmented objective, not the raw one: the
        # anchor moves each phase, so the two are different functions.
        quad_obs = self._quad(self._norm(train_x), rho)

        if self._phase == "op":
            # EI on f(x) + quad(x), whose posterior is the objective GP shifted.
            f_mean, f_sigma = self._objective_mean(model, pool)
            aug_mean = f_mean + quad
            aug_obs = obs + quad_obs
            aug_obs = aug_obs[torch.isfinite(aug_obs)]
            best = aug_obs.min() if aug_obs.numel() else aug_mean.min()
            score = _expected_improvement(best, aug_mean, f_sigma)
        else:
            # FP: M * 1[c(z) > 0] + quad is two-valued, so EI is the exact
            # two-atom expectation of eq. (13), in objective units rather than
            # the paper's, which divides by M. A Gaussian moment match would
            # zero the variance wherever feasibility is settled, and with it the
            # EI of the candidates the quadratic likes best.
            p_infeasible = self._prob_infeasible(constraint_model, pool)
            aug_obs = M * (~feasible).to(quad_obs) + quad_obs
            best = aug_obs.min() if aug_obs.numel() else (p_infeasible * M + quad).min()
            gap = best - quad
            score = (1.0 - p_infeasible) * gap.clamp_min(0.0) + p_infeasible * (
                gap - M
            ).clamp_min(0.0)
            if not bool((score > 0).any()):
                # Nothing can improve on the incumbent once the anchor sits on
                # the feasible row already minimising the quadratic. EI is flat
                # at zero, so argmax would pick arbitrarily: rank by the
                # augmented objective instead.
                score = -(M * p_infeasible + quad)

        idx = int(score.argmax())
        self._left -= 1
        return pool[idx].unsqueeze(0)
