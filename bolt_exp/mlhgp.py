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
        input_transform: Input transform applied to both GPs. A fresh copy is
            used for each new GP instance created in the loop.
        warm_start_noise_var: Optional (n, 1) initial noise variance. When given,
            the first EM step uses this instead of a small fixed constant.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_em_iter: int = 5,
        em_tol: Optional[float] = 1e-4,
        em_damping: float = 1.0,
        input_transform: Optional[Normalize] = None,
        warm_start_noise_var: Optional[Tensor] = None,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        super().__init__()
        assert train_Y.shape[-1] == 1, "MLHGP expects single-output (n, 1) train_Y"
        if not (0 < em_damping <= 1.0):
            raise ValueError(f"em_damping must be in (0, 1], got {em_damping}")

        noise_var = (
            warm_start_noise_var
            if warm_start_noise_var is not None
            else torch.full_like(train_Y, 1e-3)
        )

        # store intermediate noise deltas
        em_noise_deltas = []

        for i in range(n_em_iter):
            prev_noise_var = noise_var.detach().clone()

            # --- M-step: fit signal GP with current noise estimates ---
            signal_gp = SingleTaskGP(
                train_X,
                train_Y,
                train_Yvar=noise_var,
                covar_module=deepcopy(covar_module),
                input_transform=deepcopy(input_transform),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(signal_gp.likelihood, signal_gp).to(train_X))

            # --- E-step: compute log-squared residuals as noise GP targets ---
            with torch.no_grad():
                resid_sq = (train_Y - signal_gp.posterior(train_X).mean).pow(2)
                z = resid_sq.clamp(min=1e-6).log()

            # --- Fit auxiliary noise GP on log-residuals ---
            noise_gp = SingleTaskGP(
                train_X,
                z,
                covar_module=deepcopy(covar_module),
                input_transform=deepcopy(input_transform),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(noise_gp.likelihood, noise_gp).to(train_X))

            # --- Update noise variance estimate ---
            with torch.no_grad():
                new_noise_var = (
                    noise_gp.posterior(train_X).mean.exp().clamp(min=1e-6, max=1e3)
                )

                # add damping for stability
                noise_var = (
                    (1 - em_damping) * prev_noise_var + em_damping * new_noise_var
                )

            delta = (noise_var.detach() - prev_noise_var).norm().item()
            em_noise_deltas.append(delta)

            # early stop
            if em_tol is not None and i > 0 and delta < em_tol:
                break

        self.signal_gp = signal_gp
        self.noise_gp = noise_gp
        self.learned_noise_var = noise_var
        self._em_noise_deltas = em_noise_deltas

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs,
    ) -> Posterior:
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
    warm_start_noise_var: Optional[Tensor] = None,
    covar_module: Optional[Kernel] = None,
) -> Tuple[SingleTaskGP, "MLHGP"]:
    """Fit a single MLHGP and return the final GP with its learned noise variances.

    Args:
        train_X: Training inputs of shape (n, d).
        train_Y: Training outputs of shape (n, 1).
        bounds: Bounds of shape (2, d).
        n_em_iter: Maximum number of EM iterations.
        em_tol: Early-stop tolerance on noise_var change norm. None to disable.
        em_damping: Step size in (0, 1] for EM updates; see MLHGP docstring.
        warm_start_noise_var: Optional (n, 1) initial noise variance.
        covar_module: Optional covariance module for both signal and noise GPs.

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
