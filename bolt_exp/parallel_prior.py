"""Physics-informed GP prior means for the PCO problems.

The method is a prior mean and nothing else: an analytic model of distributed
training stands in for the GP's constant mean, on both the objective GP and the
constraint GP, and its parameters are learned jointly with the kernel by
`fit_gpytorch_mll`. It changes no acquisition, so it composes with every
`--acq_fn` in `test_w_botorch_pco.py`.

    ParallelCommCostMean  throughput as 1 / (communication + computation),
                          from a ring-allreduce model for the data- and
                          tensor-parallel collectives plus a point-to-point
                          model for the pipeline stages.
    MaxMemoryMean         peak memory as parameters/(pp*tp) + activations +
                          overhead, returned as the normalised margin.

Both are ported from an existing analytic cost model of parallel training; the
cost algebra is unchanged. What is new here is the bridge to this benchmark, which the original
left to its own codebase:

Input decoding
    A `mean_module` inside a `SingleTaskGP` sees whatever `input_transform`
    produced, so with `Normalize(d, bounds)` it gets the unit cube, not the log2
    encoding the cost model reads. `_Decoder` inverts the normalisation and then
    the encoding, and supplies `num_gpus`/`num_hosts` from the problem instance
    -- constants of the sweep rather than search dimensions, so they are not
    columns of `X` at all.

ZeRO stage
    The cost model branches on the raw stage (0, 2, 3); `PCO` encodes it as the
    ordinal 0, 1, 2. `o * (5 - o) / 2` maps one to the other, exactly on the
    three levels and smoothly between them.

Gradient accumulation
    `grad_accum` scales both means and is read from the sweep's
    `log2_grad_accum_steps`, as in the original. PCO16 searches it, so it is a
    column of `X`; PCO32/PCO64 fix it (at 1), so it is taken from the table as
    an instance constant, like `num_gpus`.

Outcome standardisation
    `SingleTaskGP` standardises `train_Y` by default, so the mean module lives
    in standardised space while both cost models return physical units
    (throughput, and a memory margin). `StandardizedMean` applies the fitted
    transform's own affine to the prior's output. The transform must therefore
    be constructed here and handed to both the mean and the GP.
"""

import numpy as np
import torch
from gpytorch.means import Mean
from torch import nn


# Margin substituted for a run that OOMed, matching `bolt.problems.
# parallelism_config.OOM_MARGIN_PAD`: the process dies before reporting how far
# over it went, so the overflow has no magnitude. The memory mean is squashed
# towards this value so it cannot predict an unboundedly bad configuration.
OOM_ADD_FACTOR = 0.2

# Per-GPU HBM, as `bolt.problems.parallelism_config.MEM_CAPACITY_GB`. The
# instance's own `prob.mem_capacity_gb` overrides it.
MEM_CAPACITY_GB = 96.0

# `PCO.input_cols` names the decoder reads. Looked up by name, not position:
# the instances do not share a column order (PCO16 drops `log2_cp_size` and
# searches `log2_grad_accum_steps` instead).
DP, TP, PP, ZERO, CHUNKS = (
    "log2_dp_size",
    "log2_tp_size",
    "log2_pp_size",
    "zero_stage",
    "log2_num_model_chunks",
)
GRAD_ACCUM = "log2_grad_accum_steps"

# Iterations over which the throughput prior decays to a learnable constant;
# `prior_scale` follows a quarter-cosine and reaches zero at HALF_LIFE / 2.
PRIOR_HALF_LIFE = 40.0


def _decay(step: int) -> float:
    """Weight on the cost model at BO iteration `step`, 1 at the start, 0 by 20."""
    return float(np.clip(np.cos(step / PRIOR_HALF_LIFE * np.pi), 0.0, None))


class _Decoder(nn.Module):
    """Normalised `PCO` inputs -> the physical quantities the cost models read.

    Holds the instance constants and the bounds `Normalize` was built with, so
    it can undo both the unit-cube scaling and the log2 encoding.
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        input_cols: list[str],
        num_gpus: int,
        num_hosts: int,
        log2_grad_accum: float,
    ):
        super().__init__()
        self.col = {name: i for i, name in enumerate(input_cols)}
        self.register_buffer("lo", bounds[0].clone())
        self.register_buffer("span", (bounds[1] - bounds[0]).clone())
        self.num_gpus = float(num_gpus)
        self.num_hosts = float(num_hosts)
        # Used only where the accumulation count is not a column of `X`.
        self.grad_accum = 2.0 ** float(log2_grad_accum)

    def forward(self, x: torch.Tensor) -> dict:
        """Decode `(..., d)`-dim normalised inputs into named `(...)`-dim tensors."""
        raw = self.lo.to(x) + x * self.span.to(x)
        col = self.col

        p_dp = raw[..., col[DP]].exp2()
        p_tp = raw[..., col[TP]].exp2()
        p_pp = raw[..., col[PP]].exp2()
        chunks = raw[..., col[CHUNKS]].exp2()

        # Ordinal 0/1/2 -> ZeRO stage 0/2/3, which is what the cost model's
        # thresholds at 1.5 and 2.5 are written against.
        zero_ord = raw[..., col[ZERO]]
        zero_stage = zero_ord * (5.0 - zero_ord) / 2.0

        ones = torch.ones_like(p_dp)
        n_gpus = ones * self.num_gpus
        num_hosts = ones * self.num_hosts

        if GRAD_ACCUM in col:
            grad_accum = raw[..., col[GRAD_ACCUM]].exp2()
        else:
            grad_accum = ones * self.grad_accum

        return {
            "n_gpus": n_gpus,
            "num_hosts": num_hosts,
            "p_dp": p_dp,
            "p_tp": p_tp,
            "p_pp": p_pp,
            "zero_stage": zero_stage,
            "chunks": chunks,
            "grad_accum": grad_accum,
            # The original fixes the number of microbatches in flight at the
            # pipeline depth rather than reading it from the configuration.
            "microbatches": p_pp,
        }


def _allreduce_cost(
    p: torch.Tensor, n: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    r"""Ring-allreduce time for `n` bytes over `p` ranks.

    `2 (p - 1) alpha + ((p - 1) / p) n beta`: one latency term per hop, and a
    bandwidth term over the share of the data each rank forwards. Zero at
    `p = 1`, where the formula already vanishes.

    Args:
        p: `(...)`-dim tensor of participant counts.
        n: `(...)`-dim tensor of message sizes.
        alpha: per-hop latency.
        beta: per-unit-of-data cost.

    Returns:
        torch.Tensor: `(...)`-dim tensor of costs.
    """
    p = p.clamp_min(1.0)
    n = n.clamp_min(1e-6)
    cost = 2.0 * (p - 1.0) * alpha + ((p - 1.0) / p.clamp_min(1.0 + 1e-6)) * n * beta
    return torch.where(p > 1.0001, cost, torch.zeros_like(cost))


class ParallelCommCostMean(Mean):
    r"""Prior mean for the throughput GP: `1 / (communication + computation)`.

    Three communication terms, each split into an intra-host and an inter-host
    regime because crossing hosts changes both latency and bandwidth:

    - **Data parallel**, a gradient allreduce over `M / tp` bytes. Hierarchical
      when the data-parallel group spans hosts, flat otherwise. ZeRO adds a
      second collective, sized by the learnable `zero_div_factor` and active
      only at stage 3.
    - **Tensor parallel**, an activation allreduce repeated
      `tp_ops_per_microbatch` times per microbatch.
    - **Pipeline parallel**, two point-to-point transfers per microbatch across
      each of the `pp - 1` stage boundaries.

    Computation is one microbatch on a tensor-sharded stage, times the
    microbatches in flight plus the `(pp - 1) / chunks` bubble slots, divided
    across the data-parallel ranks.

    Every parameter is stored as a log and exponentiated in `forward`, so the
    unconstrained optimiser `fit_gpytorch_mll` runs cannot drive a cost
    negative.

    The cost model is blended against a learnable constant by `prior_scale`,
    which decays to zero over the first `PRIOR_HALF_LIFE / 2` BO iterations: the
    prior informs the early fits, when there is too little data to learn the
    surface, and steps aside once there is not.
    """

    def __init__(self, decoder: _Decoder, step: int = 0):
        r"""
        Args:
            decoder: maps normalised inputs to physical quantities.
            step: BO iteration, which sets how much weight the cost model keeps.
        """
        super().__init__()
        self.decoder = decoder
        self.prior_scale = _decay(step)

        # Log-scale initial values, drawn in the original's order. The alphas
        # share one draw and the betas another, as there: each group starts
        # from a common value and is separated only by the fit.
        M, M_act_tp, M_act_pp, tp_ops, comp, alpha, beta = np.random.randn(7)

        def _log_param(value: float) -> nn.Parameter:
            return nn.Parameter(torch.tensor(float(value)))

        self.constant = _log_param(np.random.randn())
        self.M = _log_param(M)
        self.comp = _log_param(comp)

        self.M_act_tp_log = _log_param(M_act_tp)
        self.tp_ops_per_microbatch_log = _log_param(tp_ops)
        self.M_act_pp_log = _log_param(M_act_pp)
        self.zero_div_factor = nn.Parameter(torch.tensor(0.0))

        self.alpha_intra = _log_param(alpha)
        self.beta_dp_intra = _log_param(beta)
        self.beta_dp_zero3_intra = _log_param(beta)
        self.beta_tp_intra = _log_param(beta)
        self.beta_pp_intra = _log_param(beta)

        self.alpha_inter = _log_param(alpha)
        self.beta_dp_inter = _log_param(beta)
        self.beta_tp_inter = _log_param(beta)
        self.beta_pp_inter = _log_param(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Predicted throughput at `(..., 8)`-dim normalised inputs.

        Returns:
            torch.Tensor: `(...)`-dim tensor of throughputs.
        """
        c = self.decoder(x)
        n_gpus, num_hosts = c["n_gpus"], c["num_hosts"]
        p_dp, p_tp, p_pp = c["p_dp"], c["p_tp"], c["p_pp"]
        grad_accum, microbatches, chunks = c["grad_accum"], c["microbatches"], c["chunks"]

        one = torch.ones_like(p_dp)
        p_tp_clamped = p_tp.clamp_min(1.0)
        p_dp_clamped = p_dp.clamp_min(1.0)
        safe_p_pp = p_pp.clamp_min(1.0)
        safe_chunks = chunks.clamp_min(1.0)

        # Gradient bytes each data-parallel rank reduces: the model, sharded by
        # the tensor-parallel degree.
        n_dp = (self.M.exp() * one / p_tp_clamped).clamp_min(1e-6)
        # Stage 3 shards optimizer state across the data-parallel group as well;
        # stages 0 and 2 leave it replicated.
        p_zero3 = torch.where(c["zero_stage"] < 2.5, one, p_dp)

        gpus_per_host = (n_gpus / num_hosts.clamp_min(1.0)).clamp_min(1.0)
        is_multi_host = num_hosts > 1.0001

        m_act_tp = self.M_act_tp_log.exp() * one / grad_accum
        tp_ops = self.tp_ops_per_microbatch_log.exp() / grad_accum
        m_act_pp = self.M_act_pp_log.exp() * one / grad_accum

        a_intra, a_inter = self.alpha_intra.exp(), self.alpha_inter.exp()

        # === Data-parallel gradient allreduce ===
        is_hierarchical = is_multi_host & (p_dp > gpus_per_host + 1e-4)
        is_flat_inter = is_multi_host & ~is_hierarchical

        hierarchical = _allreduce_cost(
            gpus_per_host, n_dp, a_intra, self.beta_dp_intra.exp()
        ) + _allreduce_cost(
            torch.ceil(p_dp / gpus_per_host), n_dp, a_inter, self.beta_dp_inter.exp()
        )
        flat_inter = _allreduce_cost(p_dp, n_dp, a_inter, self.beta_dp_inter.exp())
        flat_intra = _allreduce_cost(
            p_dp, n_dp, a_intra, self.beta_dp_zero3_intra.exp()
        )

        cost_dp = torch.where(
            is_hierarchical, hierarchical, torch.where(is_flat_inter, flat_inter, flat_intra)
        )
        cost_dp = torch.where(p_dp > 1.0001, cost_dp, torch.zeros_like(cost_dp))

        zero_div = self.zero_div_factor.exp()
        cost_dp = cost_dp + zero_div * _allreduce_cost(
            p_zero3, n_dp / zero_div, a_inter, self.beta_dp_zero3_intra.exp()
        )

        # === Tensor-parallel activation allreduce ===
        tp_inter = is_multi_host & (p_tp > gpus_per_host + 1e-4)
        cost_tp_one_op = torch.where(
            tp_inter,
            _allreduce_cost(p_tp, m_act_tp, a_inter, self.beta_tp_inter.exp()),
            _allreduce_cost(p_tp, m_act_tp, a_intra, self.beta_tp_intra.exp()),
        )
        cost_tp = tp_ops * microbatches * cost_tp_one_op
        cost_tp = torch.where(p_tp > 1.0001, cost_tp, torch.zeros_like(cost_tp))

        # === Pipeline-parallel point-to-point ===
        # One activation forward and one gradient back per microbatch, across
        # each stage boundary.
        transfers = (p_pp - 1.0) * 2.0 * microbatches
        cost_pp = transfers * torch.where(
            is_multi_host,
            a_inter + m_act_pp * self.beta_pp_inter.exp(),
            a_intra + m_act_pp * self.beta_pp_intra.exp(),
        )
        cost_pp = torch.where(p_pp > 1.0001, cost_pp, torch.zeros_like(cost_pp))

        cost = cost_dp + cost_tp + cost_pp
        cost = (1.0 - self.prior_scale) * self.constant.exp() + self.prior_scale * cost

        # === Computation ===
        bubble = ((safe_p_pp - 1.0) / safe_chunks).clamp_min(0.0)
        per_stage = self.comp.exp() / p_tp_clamped
        cost = cost + per_stage * (microbatches + bubble) / p_dp_clamped

        return 1.0 / (grad_accum * cost.clamp_min(1e-7))


class MaxMemoryMean(Mean):
    r"""Prior mean for the constraint GP: the normalised peak-memory margin.

    Peak memory per GPU as three learnable terms -- parameters and optimizer
    state, sharded by the pipeline and tensor degrees; activations, shared out
    over the cluster and the microbatches; and a constant overhead -- reported
    as `(peak - capacity) / capacity`, so it is negative exactly while the run
    fits. That is the quantity the constraint GP regresses, in BoTorch's
    convention that `c(x) <= 0` is feasible.

    Above the boundary the output is squashed towards `OOM_ADD_FACTOR` by a
    reflected softplus, matching the constant the benchmark records for a run
    that died: an OOM reports no memory figure, so there is no magnitude for the
    prior to predict and it should not claim one.
    """

    def __init__(self, decoder: _Decoder, max_mem_GB: float = MEM_CAPACITY_GB):
        r"""
        Args:
            decoder: maps normalised inputs to physical quantities.
            max_mem_GB: per-GPU capacity the margin is taken against.
        """
        super().__init__()
        self.decoder = decoder
        self.max_mem_GB = max_mem_GB

        self.m1 = nn.Parameter(torch.tensor(1.0).log())
        self.m2 = nn.Parameter(torch.tensor(1.0).log())
        self.m3 = nn.Parameter(torch.tensor(1.0).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Predicted memory margin at `(..., 8)`-dim normalised inputs.

        Returns:
            torch.Tensor: `(...)`-dim tensor of margins, `< 0` where the run fits.
        """
        c = self.decoder(x)
        p_tp, p_pp = c["p_tp"], c["p_pp"]
        n_gpus, microbatches, grad_accum = c["n_gpus"], c["microbatches"], c["grad_accum"]

        params = self.m1.exp() / (p_pp * p_tp).clamp_min(1.0)
        activations = self.m2.exp() / (n_gpus * microbatches * grad_accum).clamp_min(1.0)
        overhead = self.m3.exp() * torch.ones_like(params)

        margin = (params + activations + overhead - self.max_mem_GB) / self.max_mem_GB
        return OOM_ADD_FACTOR - torch.nn.functional.softplus(
            OOM_ADD_FACTOR - margin, beta=20.0, threshold=0.5
        )


class StandardizedMean(Mean):
    """Wraps a mean in physical units for a GP whose targets are standardised.

    `SingleTaskGP` standardises `train_Y`, so the mean module has to answer on
    that scale. The transform is held outside the module tree: the GP owns it,
    and registering it here again would duplicate its buffers in the state dict.
    """

    def __init__(self, base: Mean, transform):
        r"""
        Args:
            base: mean returning the outcome in its physical units.
            transform: the `Standardize` this GP was built with, already fitted.
        """
        super().__init__()
        self.base = base
        self._transform = (transform,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.base(x)
        tf = self._transform[0]
        if not hasattr(tf, "means"):
            return value
        means = tf.means.to(value).reshape(())
        stdvs = tf.stdvs.to(value).reshape(())
        return (value - means) / stdvs


def _fixed_log2_grad_accum(prob) -> float:
    """The table's `log2_grad_accum_steps` where it is constant, else 0 (unused)."""
    if GRAD_ACCUM in prob.input_cols:
        return 0.0
    levels = np.unique(np.asarray(prob.obj_func.ds[GRAD_ACCUM]))
    if len(levels) != 1:
        raise ValueError(
            f"{prob.name}: {GRAD_ACCUM} varies ({levels.tolist()}) but is not an input"
        )
    return float(levels[0])


class ParallelismPrior:
    """Builds the prior means for one `PCO` instance.

    Args:
        prob: the `PCO` problem, for the cluster constants and, where it is not
            searched, the table's fixed accumulation count.
        bounds: `(2, d)`-dim tensor `Normalize` was built with.
    """

    def __init__(self, prob, bounds: torch.Tensor):
        self.decoder = _Decoder(
            bounds=bounds,
            input_cols=prob.input_cols,
            num_gpus=prob.num_gpus,
            num_hosts=prob.num_hosts,
            log2_grad_accum=_fixed_log2_grad_accum(prob),
        )
        self.mem_capacity_gb = getattr(prob, "mem_capacity_gb", MEM_CAPACITY_GB)

    def throughput_mean(self, transform, step: int = 0) -> Mean:
        """Objective-GP mean, weighted by the decay schedule at `step`."""
        return StandardizedMean(ParallelCommCostMean(self.decoder, step=step), transform)

    def memory_mean(self, transform) -> Mean:
        """Constraint-GP mean. It does not decay: memory is modelled throughout."""
        return StandardizedMean(
            MaxMemoryMean(self.decoder, max_mem_GB=self.mem_capacity_gb), transform
        )
