"""Find the optimal value for a BoLT problem via grid search + local optimization.

Grid phase: enumerate all discrete/categorical combos with random continuous samples.
  - For simplex-constrained dims, samples from Dirichlet(1,...,1) (uniform on simplex).
Refinement phase: scipy L-BFGS-B (unconstrained) or SLSQP (simplex-constrained).

Usage:
    python find_optimal.py --problem hpo_multifidelity_step --fidelity-ind 7
    python find_optimal.py --problem hpo --n-top 200 --n-grad-steps 500
    python find_optimal.py --problem dm --simplex-groups 0,1,2 3,4,5
"""

import argparse
import itertools

import numpy as np
import torch
from scipy.optimize import minimize

import bolt

PROBLEM_REGISTRY = {
    "hpo": bolt.HPO,
    "hpo_multifidelity_step": bolt.HPOMultiFidelityToken,
    "hpo_multifidelity_model": bolt.HPOMultiFidelityModel,
    "dm": bolt.DMCurriculum,
    "dm_mo": bolt.DMCurriculumMO,
    "dm_het": bolt.DMCurriculumHet,
    "po128": bolt.PO128,
    "po256": bolt.PO256,
    "po512": bolt.PO512,
    "po768": bolt.PO768,
}


def sample_continuous(
    n: int,
    cont_inds: list[int],
    bounds: list,
    simplex_groups: list[list[int]],
    fidelity_ind: int | None,
    dtype,
    device,
) -> torch.Tensor:
    """Sample n points over continuous dims, respecting simplex groups and fidelity."""
    samples = torch.rand(n, len(cont_inds), dtype=dtype, device=device)
    # scale non-simplex dims to their bounds
    for j, i in enumerate(cont_inds):
        samples[:, j] = samples[:, j] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    # overwrite simplex dims with Dirichlet samples (uniform on simplex)
    simplex_flat = {idx for grp in simplex_groups for idx in grp}
    for grp in simplex_groups:
        local = [cont_inds.index(i) for i in grp]
        # Dirichlet(1,...,1) = uniform on simplex
        dirichlet = torch.distributions.Dirichlet(torch.ones(len(grp), dtype=dtype))
        samples[:, local] = dirichlet.sample((n,))

    # fix fidelity at upper bound
    if fidelity_ind is not None and fidelity_ind in cont_inds:
        fid_local = cont_inds.index(fidelity_ind)
        samples[:, fid_local] = bounds[fidelity_ind][1]

    return samples


def build_grid(
    prob,
    n_rand_cont: int,
    fidelity_ind: int | None,
    simplex_groups: list[list[int]],
    dtype,
    device,
):
    """Return all discrete/cat combos crossed with random continuous samples."""
    bounds = prob._bounds
    disc_cat_inds = sorted(prob.discrete_inds + prob.categorical_inds)
    cont_inds = sorted(prob.continuous_inds)

    disc_cat_values = [
        list(range(int(bounds[i][0]), int(bounds[i][1]) + 1)) for i in disc_cat_inds
    ]
    disc_cat_combos = list(itertools.product(*disc_cat_values)) or [()]
    n_combos = len(disc_cat_combos)

    cont_samples = sample_continuous(
        n_combos * n_rand_cont, cont_inds, bounds, simplex_groups, fidelity_ind, dtype, device
    )

    X = torch.zeros(n_combos * n_rand_cont, prob.dim, dtype=dtype, device=device)
    if disc_cat_inds:
        dc_tensor = torch.tensor(disc_cat_combos, dtype=dtype, device=device)
        dc_tensor = dc_tensor.repeat_interleave(n_rand_cont, dim=0)
        for j, i in enumerate(disc_cat_inds):
            X[:, i] = dc_tensor[:, j]
    for j, i in enumerate(cont_inds):
        X[:, i] = cont_samples[:, j]

    return X


def eval_in_batches(prob, X: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """Evaluate _evaluate_true in batches, no grad."""
    outs = []
    for start in range(0, len(X), batch_size):
        with torch.no_grad():
            y = prob._evaluate_true(X[start : start + batch_size])
        outs.append(y)
    return torch.cat(outs, dim=0)  # (N, n_obj)


def local_optimize(
    prob,
    X_init: torch.Tensor,
    opt_inds: list[int],
    simplex_groups: list[list[int]],
    n_steps: int,
    dtype,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine each candidate in X_init with scipy on opt_inds.

    Uses SLSQP (with simplex equality constraints) when simplex_groups is
    non-empty, otherwise L-BFGS-B.

    Returns:
        X_best: (N, dim)
        y_best: (N,) scalar objective (first obj for MO)
    """
    bounds_list = prob._bounds
    scipy_bounds = [(bounds_list[i][0], bounds_list[i][1]) for i in opt_inds]

    # build equality constraints for simplex groups (expressed over opt_inds positions)
    constraints = []
    for grp in simplex_groups:
        local_positions = [opt_inds.index(i) for i in grp if i in opt_inds]
        if len(local_positions) == len(grp):  # all members are being optimized
            constraints.append({
                "type": "eq",
                "fun": lambda x, pos=local_positions: np.sum(x[pos]) - 1.0,
            })

    method = "SLSQP" if constraints else "L-BFGS-B"
    slsqp_opts = {"maxiter": n_steps, "ftol": 1e-12}
    lbfgsb_opts = {"maxiter": n_steps, "ftol": 1e-12, "gtol": 1e-8}
    options = slsqp_opts if constraints else lbfgsb_opts

    N = X_init.shape[0]
    X_best = X_init.clone()
    y_best = torch.full((N,), -torch.inf, dtype=dtype, device=device)

    def neg_obj(cont_vals, x_fixed):
        x = x_fixed.clone()
        x[opt_inds] = torch.tensor(cont_vals, dtype=dtype, device=device)
        # normalize simplex groups in case SLSQP passes slightly infeasible points
        for grp in simplex_groups:
            s = x[grp].sum()
            if s > 0:
                x[grp] = x[grp] / s
        with torch.no_grad():
            y = prob._evaluate_true(x.unsqueeze(0))  # (1, n_obj)
        return -y[0, 0].item()

    for i in range(N):
        x0 = X_init[i].clone()
        x0_cont = x0[opt_inds].cpu().numpy()

        res = minimize(
            neg_obj,
            x0_cont,
            args=(x0,),
            method=method,
            bounds=scipy_bounds,
            constraints=constraints,
            options=options,
        )

        x_opt = x0.clone()
        x_opt[opt_inds] = torch.tensor(res.x, dtype=dtype, device=device)
        y_opt_val = -res.fun

        with torch.no_grad():
            y_init = prob._evaluate_true(x0.unsqueeze(0))[0, 0].item()

        if y_opt_val >= y_init:
            X_best[i] = x_opt
            y_best[i] = y_opt_val
        else:
            X_best[i] = x0
            y_best[i] = y_init

        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  {i+1}/{N}  running best={y_best[:i+1].max().item():.6f}")

    return X_best, y_best


def main():
    parser = argparse.ArgumentParser(description="Find optimal value for a BoLT problem.")
    parser.add_argument(
        "--problem", required=True, choices=list(PROBLEM_REGISTRY),
        help="Problem name",
    )
    parser.add_argument(
        "--n-rand-cont", type=int, default=5,
        help="Random continuous samples per discrete/categorical combo in grid phase",
    )
    parser.add_argument(
        "--n-top", type=int, default=200,
        help="Number of top grid candidates to refine with L-BFGS-B",
    )
    parser.add_argument(
        "--n-grad-steps", type=int, default=500,
        help="Max L-BFGS-B iterations per candidate",
    )
    parser.add_argument(
        "--fidelity-ind", type=int, default=None,
        help="Index of fidelity dim — fixed at upper bound (not optimized)",
    )
    parser.add_argument(
        "--simplex-groups", nargs="*", default=[],
        metavar="I,J,K",
        help="Comma-separated dim indices for each simplex group, e.g. --simplex-groups 0,1,2 3,4,5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    simplex_groups = [
        [int(x) for x in grp.split(",")] for grp in args.simplex_groups
    ]

    torch.manual_seed(args.seed)
    dtype = torch.double
    device = torch.device(args.device)

    print(f"Loading problem: {args.problem}")
    prob = PROBLEM_REGISTRY[args.problem](noise_std=None, negate=False, dtype=dtype)
    prob = prob.to(device)

    # dims to optimize over (continuous, excluding fidelity)
    cont_inds = sorted(prob.continuous_inds)
    opt_inds = [i for i in cont_inds if i != args.fidelity_ind]
    print(f"  Continuous dims: {cont_inds}")
    print(f"  Optimizing over: {opt_inds}" + (f"  (fidelity dim {args.fidelity_ind} fixed at upper bound)" if args.fidelity_ind is not None else ""))
    if simplex_groups:
        print(f"  Simplex groups : {simplex_groups}")

    # --- Grid phase ---
    print(f"\n[Grid phase] Building grid (n_rand_cont={args.n_rand_cont})...")
    X_grid = build_grid(prob, args.n_rand_cont, args.fidelity_ind, simplex_groups, dtype, device)
    print(f"  Evaluating {len(X_grid):,} candidates...")
    Y_grid = eval_in_batches(prob, X_grid)
    obj_grid = Y_grid[:, 0] if Y_grid.shape[1] > 1 else Y_grid.squeeze(-1)

    n_top = min(args.n_top, len(X_grid))
    top_vals, top_inds = obj_grid.topk(n_top)
    X_top = X_grid[top_inds]
    print(f"  Grid best  : {top_vals[0].item():.6f}")
    print(f"  Grid top-{n_top} range: [{top_vals[-1].item():.6f}, {top_vals[0].item():.6f}]")

    # --- L-BFGS-B refinement phase ---
    print(f"\n[Refinement phase] Refining {n_top} candidates (max {args.n_grad_steps} iters each)...")
    X_opt, y_opt = local_optimize(prob, X_top, opt_inds, simplex_groups, args.n_grad_steps, dtype, device)

    best_idx = y_opt.argmax()
    best_val = y_opt[best_idx].item()
    best_x = X_opt[best_idx]

    print(f"\n=== Result ===")
    print(f"Optimal value : {best_val:.6f}")
    print(f"Optimizer     : {best_x.tolist()}")
    if hasattr(prob, "_optimal_value") and prob._optimal_value is not None:
        print(f"Known optimal : {prob._optimal_value}")


if __name__ == "__main__":
    main()
