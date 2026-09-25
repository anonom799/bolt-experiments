"""Most Likely Heteroscedastic Gaussian Process (Kersting et al., 2007).

Iteratively learns input-dependent noise variance from residuals using an auxiliary
noise GP, avoiding the need for oracle noise estimates.
"""

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.posteriors import Posterior
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class MLHGP(Model):
    """Most Likely Heteroscedastic GP for a single output.

    Runs an EM loop: alternates between fitting a signal GP with current noise
    estimates and fitting an auxiliary noise GP on log-squared residuals.

    Args:
        train_X: Training inputs of shape (n, d).
        train_Y: Training outputs of shape (n, 1).
        n_em_iter: Maximum number of EM iterations.
        em_tol: Early-stop when ||noise_var_new - noise_var_old|| < em_tol. Set
            to None to always run all iterations.
        em_damping: Step size in (0, 1]. At 1.0 (default) updates are undamped;
            smaller values blend old and new noise estimates as
            ``(1 - em_damping) * prev + em_damping * new``, reducing oscillation.
        n_mc_samples: Number of posterior-predictive samples per point used to
            estimate the noise levels in the E-step.
        noise_gp_var: If given, fix the auxiliary GP's observation variance to this
            instead of fitting it. Useful because the targets ``z`` are noisy
            log-variances whose spread does not shrink with ``n_mc_samples``.
        noise_pred: "mode" (default) predicts exp(E[log r]); "mean" predicts the
            lognormal mean exp(E[log r] + Var[log r] / 2).
        noise_covar_module: Optional separate kernel for the auxiliary noise GP.
            Defaults to a copy of ``covar_module``.
        input_transform: Input transform applied to both GPs. A fresh copy is
            used for each new GP instance created in the loop.
        warm_start_noise_var: Optional (n, 1) initial noise variance. When given,
            the first EM step fixes these variances; otherwise it fits a standard
            GP with learned homoscedastic noise.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_em_iter: int = 5,
        em_tol: Optional[float] = 1e-4,
        em_damping: float = 1.0,
        n_mc_samples: int = 100,
        input_transform: Optional[Normalize] = None,
        warm_start_noise_var: Optional[Tensor] = None,
        covar_module: Optional[Kernel] = None,
        noise_gp_var: Optional[float] = None,
        noise_pred: str = "mode",
        noise_covar_module: Optional[Kernel] = None,
    ) -> None:
        super().__init__()
        assert train_Y.shape[-1] == 1, "MLHGP expects single-output (n, 1) train_Y"
        if not (0 < em_damping <= 1.0):
            raise ValueError(f"em_damping must be in (0, 1], got {em_damping}")

        if n_mc_samples < 1:
            raise ValueError(f"n_mc_samples must be >= 1, got {n_mc_samples}")
        if noise_pred not in ("mode", "mean"):
            raise ValueError(f"noise_pred must be 'mode' or 'mean', got {noise_pred!r}")

        # None means the first signal GP is a standard GP that learns its own
        # homoscedastic noise; a warm start replaces it with fixed variances.
        noise_var = warm_start_noise_var

        # store intermediate noise deltas, and the noise vector after each step
        em_noise_deltas = []
        em_noise_history = []

        for i in range(n_em_iter):
            prev_noise_var = None if noise_var is None else noise_var.detach().clone()

            # --- M-step: fit signal GP with current noise estimates ---
            signal_gp = SingleTaskGP(
                train_X,
                train_Y,
                train_Yvar=noise_var,
                covar_module=deepcopy(covar_module),
                input_transform=deepcopy(input_transform),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(signal_gp.likelihood, signal_gp).to(train_X))

            # --- E-step: estimate noise levels from posterior-predictive draws ---
            #   z_i = log( mean_j 0.5 * (y_i - t_ij)^2 ),  t_ij ~ predictive at x_i
            # Averaging inside the log makes exp(z) a variance estimate that cannot
            # collapse to zero when the posterior mean interpolates the data.
            # observation_noise=True would add the *mean* training noise, discarding
            # the per-point structure, so once noise_var is known it is added by hand.
            with torch.no_grad():
                f_post = signal_gp.posterior(train_X, observation_noise=noise_var is None)
                t = f_post.rsample(torch.Size([n_mc_samples]))  # (s, n, 1)
                if noise_var is not None:
                    t = t + noise_var.sqrt() * torch.randn_like(t)
                z = (
                    (0.5 * (train_Y.unsqueeze(0) - t).pow(2))
                    .mean(dim=0)
                    .clamp(min=1e-12)
                    .log()
                )

            # --- Fit auxiliary noise GP on log-residuals ---
            noise_gp = SingleTaskGP(
                train_X,
                z,
                train_Yvar=(
                    None if noise_gp_var is None
                    else torch.full_like(z, noise_gp_var)
                ),
                covar_module=deepcopy(
                    covar_module if noise_covar_module is None else noise_covar_module
                ),
                input_transform=deepcopy(input_transform),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(noise_gp.likelihood, noise_gp).to(train_X))

            # --- Update noise variance estimate ---
            with torch.no_grad():
                z_post = noise_gp.posterior(train_X)
                # "mode": exp(E[log r]), the most likely noise level.
                # "mean": the lognormal mean exp(mu + var / 2).
                log_r = z_post.mean + (0.5 * z_post.variance if noise_pred == "mean" else 0.0)
                new_noise_var = log_r.exp().clamp(min=1e-6, max=1e3)

                # add damping for stability
                noise_var = (
                    new_noise_var if prev_noise_var is None
                    else (1 - em_damping) * prev_noise_var + em_damping * new_noise_var
                )

            delta = (
                float("inf") if prev_noise_var is None
                else (noise_var.detach() - prev_noise_var).norm().item()
            )
            em_noise_deltas.append(delta)
            em_noise_history.append(noise_var.detach().clone())

            # early stop
            if em_tol is not None and i > 0 and delta < em_tol:
                break

        self.signal_gp = signal_gp
        self.noise_gp = noise_gp
        self.learned_noise_var = noise_var
        self._noise_pred = noise_pred
        self._em_noise_deltas = em_noise_deltas
        self._em_noise_history = em_noise_history

    @property
    def num_outputs(self) -> int:
        return 1

    def predict_noise_var(self, X: Tensor) -> Tensor:
        """Noise variance predicted by the auxiliary noise GP at arbitrary ``X``."""
        with torch.no_grad():
            post = self.noise_gp.posterior(X)
            log_r = post.mean + (
                0.5 * post.variance if self._noise_pred == "mean" else 0.0
            )
            return log_r.exp().clamp(min=1e-6, max=1e3)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs,
    ) -> Posterior:
        # Observation noise at new points comes from the noise GP. Deferring to the
        # signal GP instead would apply the mean training noise everywhere.
        if observation_noise is True:
            noise_var = self.predict_noise_var(X)
            # A tensor observation_noise is interpreted in the signal GP's
            # standardized outcome space, so rescale the raw-space variances.
            ot = getattr(self.signal_gp, "outcome_transform", None)
            stdvs = getattr(ot, "stdvs", None) if ot is not None else None
            if stdvs is not None:
                noise_var = noise_var / stdvs.to(noise_var).pow(2)
            observation_noise = noise_var
        return self.signal_gp.posterior(
            X, observation_noise=observation_noise, **kwargs
        )


def fit_mlhgp(
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    n_em_iter: int = 5,
    em_tol: Optional[float] = 1e-4,
    em_damping: float = 1.0,
    n_mc_samples: int = 100,
    warm_start_noise_var: Optional[Tensor] = None,
    covar_module: Optional[Kernel] = None,
    noise_gp_var: Optional[float] = None,
    noise_pred: str = "mode",
    noise_covar_module: Optional[Kernel] = None,
) -> Tuple[SingleTaskGP, "MLHGP"]:
    """Fit a single MLHGP and return the final GP with its learned noise variances.

    Args:
        train_X: Training inputs of shape (n, d).
        train_Y: Training outputs of shape (n, 1).
        bounds: Bounds of shape (2, d).
        n_em_iter: Maximum number of EM iterations.
        em_tol: Early-stop tolerance on noise_var change norm. None to disable.
        em_damping: Step size in (0, 1] for EM updates; see MLHGP docstring.
        n_mc_samples: Posterior-predictive samples per E-step; see MLHGP docstring.
        warm_start_noise_var: Optional (n, 1) initial noise variance.
        covar_module: Optional covariance module for both signal and noise GPs.
        noise_gp_var: Optional fixed observation variance for the noise GP.
        noise_pred: "mode" or "mean"; see MLHGP docstring.
        noise_covar_module: Optional separate kernel for the noise GP.

    Returns:
        Tuple of (SingleTaskGP fitted with learned noise, MLHGP).
        The MLHGP object exposes `learned_noise_var` and `_em_noise_deltas`.
    """
    d = train_X.shape[-1]
    mlhgp = MLHGP(
        train_X,
        train_Y,
        n_em_iter=n_em_iter,
        em_tol=em_tol,
        em_damping=em_damping,
        n_mc_samples=n_mc_samples,
        noise_gp_var=noise_gp_var,
        noise_pred=noise_pred,
        noise_covar_module=noise_covar_module,
        input_transform=Normalize(d=d, bounds=bounds),
        warm_start_noise_var=warm_start_noise_var,
        covar_module=covar_module,
    )
    final_gp = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=mlhgp.learned_noise_var,
        covar_module=deepcopy(covar_module),
        input_transform=Normalize(d=d, bounds=bounds),
    )
    return final_gp, mlhgp
