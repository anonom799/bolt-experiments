"""Baseline comparison for HPO benchmark problems (Random, TPE, CMA-ES, BOHB, ASHA).

Supports three problem variants:
  hpo            — single-fidelity 7D mixed search space
  hpo_fd_step    — continuous fidelity (training tokens as fidelity)
  hpo_fd_model   — discrete 2-level fidelity (8B vs 4B model size)

Baseline methods and compatible problems:
  random   — all problems (random search, samples fidelity uniformly for MF variants)
  tpe      — hpo only (Optuna TPE)
  cmaes    — hpo only (Optuna CMA-ES with_margin; lora_target treated as categorical)
  bohb     — hpo_fd_step only (SMAC3 BOHB, continuous fidelity)
  asha     — hpo_fd_model only (Optuna Successive Halving, discrete 2-level fidelity)

Output is saved to JSON in the same directory as this script, named:
  hpo_{method}_{trials}trials_{iterations}iterations_results.json
  (or hpo_{problem}_{method}_... for non-default problems)

Usage examples:
  # Random search, 3 trials, 10 iterations on single-fidelity HPO
  python test_w_baselines_hpo.py --method random --trials 3 --iterations 10

  # TPE on single-fidelity HPO, matching 10 init samples from BO script
  python test_w_baselines_hpo.py --method tpe --trials 5 --iterations 50 --initial_random_samples 10

  # CMA-ES on single-fidelity HPO
  python test_w_baselines_hpo.py --method cmaes --trials 5 --iterations 50 --initial_random_samples 10

  # BOHB on continuous-fidelity HPO
  python test_w_baselines_hpo.py --problem hpo_fd_step --method bohb --trials 3 --iterations 30

  # ASHA on discrete-fidelity HPO
  python test_w_baselines_hpo.py --problem hpo_fd_model --method asha --trials 3 --iterations 30 --verbose
"""

import argparse
import json
import math
import time
import warnings

from bolt_exp import emulator_version

warnings.filterwarnings("ignore")

import numpy as np
import torch
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from bolt import HPO, HPOMultiFidelityModel, HPOMultiFidelityToken

from bolt_exp import REPO_ROOT

if not torch.cuda.is_available():
    _DEV = "mps"
else:
    _DEV = "cuda"

_DTYPE = torch.float32 if _DEV == "mps" else torch.double

# HP parameter definitions: (name, type, lo, hi)
HP_PARAM_DEFS = [
    ("lr", "float", 0.0, 1.0),
    ("batch", "int", 2, 4),
    ("lora_rank", "int", 2, 5),
    ("lora_alpha", "int", 2, 5),
    ("lora_dropout", "float", 0.0, 1.0),
    ("lora_layers", "int", 1, 30),
    ("lora_target", "int", 0, 3),
]


def _sample_config_random(rng):
    config = {}
    for name, ptype, lo, hi in HP_PARAM_DEFS:
        if ptype == "float":
            config[name] = float(rng.uniform(lo, hi))
        else:
            config[name] = int(rng.integers(lo, hi + 1))
    return config


def _suggest_config_optuna(trial):
    config = {}
    for name, ptype, lo, hi in HP_PARAM_DEFS:
        if ptype == "float":
            config[name] = trial.suggest_float(name, lo, hi)
        elif name == "lora_target":
            config[name] = trial.suggest_categorical(name, list(range(lo, hi + 1)))
        else:
            config[name] = trial.suggest_int(name, lo, hi)
    return config


def _suggest_config_cmaes(trial):
    config = {}
    for name, ptype, lo, hi in HP_PARAM_DEFS:
        if ptype == "float":
            config[name] = trial.suggest_float(name, lo, hi)
        elif name == "lora_target":
            # nominal categorical: CmaEsSampler falls back to random for this dim
            config[name] = trial.suggest_categorical(name, list(range(lo, hi + 1)))
        else:
            config[name] = trial.suggest_int(name, lo, hi)
    return config


def _evaluate(prob, config_dict, fidelity=None):
    x_vals = [
        config_dict["lr"],
        float(config_dict["batch"]),
        float(config_dict["lora_rank"]),
        float(config_dict["lora_alpha"]),
        config_dict["lora_dropout"],
        float(config_dict["lora_layers"]),
        float(config_dict["lora_target"]),
    ]
    if fidelity is not None:
        x_vals.append(float(fidelity))
    X = torch.tensor([x_vals], dtype=_DTYPE)
    return prob(X).item()


def _config_to_x(config_dict, fidelity=None):
    x = [
        config_dict["lr"],
        config_dict["batch"],
        config_dict["lora_rank"],
        config_dict["lora_alpha"],
        config_dict["lora_dropout"],
        config_dict["lora_layers"],
        config_dict["lora_target"],
    ]
    if fidelity is not None:
        x.append(fidelity)
    return x


def _run_init_phase(prob, n_init, rng, problem_name=None):
    """Sample n_init random points without consuming budget.

    Mirrors generate_initial_data in test_w_botorch_mixed.py exactly: draws a
    single (n_init, dim) matrix via rng.random(), scales to bounds, then rounds
    discrete/categorical dims.
    """
    best_value = -np.inf
    best_x = None
    best_y_all, best_obs_x_all = [], []
    best_obs_true_all, best_obs_regret_all, log_best_obs_regret_all = [], [], []
    seen_y, candidates = [], []

    if n_init > 0:
        train_x = rng.random((n_init, prob.dim))
        for i, (b_min, b_max) in enumerate(prob._bounds):
            train_x[:, i] = train_x[:, i] * (b_max - b_min) + b_min
            if i in prob.discrete_inds or i in prob.categorical_inds:
                train_x[:, i] = np.round(train_x[:, i])

        train_y = prob(torch.tensor(train_x, dtype=_DTYPE)).flatten()

        for j in range(n_init):
            x = train_x[j].tolist()
            y = train_y[j].item()
            candidates.append(x)
            seen_y.append(y)
            if y > best_value:
                best_value = y
                best_x = x

    if n_init > 0:
        # save 1 best obs from all the initial samples
        best_y_all.append(best_value)
        best_obs_x_all.append(list(best_x))

        best_obs_true = prob.evaluate_true(
            torch.tensor([best_x], dtype=_DTYPE)
        ).item()
        best_obs_true_all.append(best_obs_true)
        best_obs_regret_all.append(
            max(prob._optimal_value - best_obs_true, 0.0)
        )
        log_best_obs_regret_all.append(
            math.log(max(prob._optimal_value - best_obs_true, 1e-8))
        )

        init_X = torch.tensor(candidates, dtype=_DTYPE)
        init_true = prob.evaluate_true(init_X).flatten()
        _best_simple_true_running = init_true.max().item()
        simple_regret_all = [
            max(prob._optimal_value - _best_simple_true_running, 0.0)
        ]
        log_simple_regret_all = [
            math.log(max(prob._optimal_value - _best_simple_true_running, 1e-8))
        ]
    else:
        _best_simple_true_running = -np.inf
        simple_regret_all = []
        log_simple_regret_all = []

    return (
        best_value,
        best_x,
        best_y_all,
        best_obs_x_all,
        best_obs_true_all,
        best_obs_regret_all,
        log_best_obs_regret_all,
        seen_y,
        candidates,
        simple_regret_all,
        log_simple_regret_all,
        _best_simple_true_running,
    )


def _record_eval(
    y,
    x,
    fidelity,
    candidates,
    seen_y,
    best_value,
    best_x,
    best_value_all,
    best_x_all,
):
    """Update tracking lists in-place. Returns updated best_value, best_x."""
    candidates.append(x)
    seen_y.append(y)

    if y > best_value:
        best_value = y
        best_x = list(x)

    best_value_all.append(best_value)
    best_x_all.append(list(best_x))

    return best_value, best_x


def run_random(
    prob,
    problem_name,
    iterations,
    num_trials,
    verbose,
    initial_random_samples=0,
):
    """Random search.

    For HPO: evaluates at max fidelity (no fidelity parameter).
    For MF problems: randomly samples fidelity — budget-matched baseline for fair
    comparison with BOHB/ASHA.
    """
    is_mf_step = problem_name == "hpo_fd_step"
    is_mf_model = problem_name == "hpo_fd_model"

    all_trial_results = []
    for trial_idx in range(num_trials):
        rng = np.random.default_rng(trial_idx)
        torch.manual_seed(trial_idx)
        t0 = time.time()

        (
            best_value,
            best_x,
            best_y_all,
            best_obs_x_all,
            best_obs_true_all,
            best_obs_regret_all,
            log_best_obs_regret_all,
            seen_y,
            candidates,
            simple_regret_all,
            log_simple_regret_all,
            _best_simple_true_running,
        ) = _run_init_phase(prob, initial_random_samples, rng, problem_name)
        budget = 0.0
        budget_all = []

        while budget < iterations:
            config = _sample_config_random(rng)

            if is_mf_model:
                fidelity = float(rng.integers(0, 2))  # 0 or 1
            elif is_mf_step:
                fidelity = float(rng.uniform(0.0, 1.0))
            else:
                fidelity = None

            y = _evaluate(prob, config, fidelity)
            x = _config_to_x(config, fidelity)
            best_value, best_x = _record_eval(
                y,
                x,
                fidelity,
                candidates,
                seen_y,
                best_value,
                best_x,
                best_y_all,
                best_obs_x_all,
            )

            best_obs_true = prob.evaluate_true(
                torch.tensor([best_x], dtype=_DTYPE)
            ).item()
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(
                max(prob._optimal_value - best_obs_true, 0.0)
            )
            log_best_obs_regret_all.append(
                math.log(max(prob._optimal_value - best_obs_true, 1e-8))
            )

            x_true = prob.evaluate_true(torch.tensor([x], dtype=_DTYPE)).item()
            _best_simple_true_running = max(_best_simple_true_running, x_true)
            simple_regret_all.append(
                max(prob._optimal_value - _best_simple_true_running, 0.0)
            )
            log_simple_regret_all.append(
                math.log(
                    max(prob._optimal_value - _best_simple_true_running, 1e-8)
                )
            )

            cost = (
                prob.cost(torch.tensor([x], dtype=_DTYPE)).item()
                if fidelity is not None
                else 1.0
            )
            budget += cost
            if fidelity is not None:
                budget_all.append(budget)

            if verbose:
                print(
                    f"  budget={budget:.2f}/{iterations}  y={y:.4f}  best={best_value:.4f}"
                )

        t1 = time.time()
        print(
            f"Trial {trial_idx + 1}/{num_trials} done in {t1 - t0:.1f}s — best={best_value:.3g}"
        )
        all_trial_results.append(
            {
                "trial": trial_idx,
                "seed": trial_idx,
                "time_seconds": t1 - t0,
                "best_y_all": best_y_all,
                "best_obs_x_all": best_obs_x_all,
                "best_obs_true_all": best_obs_true_all,
                "best_obs_regret_all": best_obs_regret_all,
                "log_best_obs_regret_all": log_best_obs_regret_all,
                "simple_regret_all": simple_regret_all,
                "log_simple_regret_all": log_simple_regret_all,
                "candidates": candidates,
                "seen_y": seen_y,
                **({"budget_all": budget_all} if (is_mf_step or is_mf_model) else {}),
            }
        )

    return all_trial_results


def run_tpe(prob, iterations, num_trials, verbose, initial_random_samples=0):
    """Optuna TPE for single-fidelity HPO (no fidelity dimension).

    Standard BO baseline: Bayesian optimization with a KDE surrogate and
    Tree-structured Parzen Estimator. Applied to HPO (no multi-fidelity) for a
    clean single-fidelity comparison.
    """
    all_trial_results = []

    for trial_idx in range(num_trials):
        torch.manual_seed(trial_idx)
        sampler = optuna.samplers.TPESampler(seed=trial_idx)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        t0 = time.time()

        (
            best_value,
            best_x,
            best_y_all,
            best_obs_x_all,
            best_obs_true_all,
            best_obs_regret_all,
            log_best_obs_regret_all,
            seen_y,
            candidates,
            simple_regret_all,
            log_simple_regret_all,
            _best_simple_true_running,
        ) = _run_init_phase(
            prob,
            initial_random_samples,
            np.random.default_rng(trial_idx),
            "hpo",
        )

        # Warm-start Optuna study with init observations so TPE uses them
        for xi, yi in zip(candidates, seen_y):
            frozen = optuna.trial.create_trial(
                params={
                    name: (float(xi[i]) if ptype == "float" else int(xi[i]))
                    for i, (name, ptype, _, _) in enumerate(HP_PARAM_DEFS)
                },
                distributions={
                    name: (
                        optuna.distributions.FloatDistribution(lo, hi)
                        if ptype == "float"
                        else optuna.distributions.CategoricalDistribution(
                            choices=tuple(range(lo, hi + 1))
                        )
                        if name == "lora_target"
                        else optuna.distributions.IntDistribution(lo, hi)
                    )
                    for name, ptype, lo, hi in HP_PARAM_DEFS
                },
                value=yi,
            )
            study.add_trial(frozen)

        budget = 0.0

        while budget < iterations:
            trial = study.ask()
            config = _suggest_config_optuna(trial)
            y = _evaluate(prob, config, fidelity=None)
            study.tell(trial, y)

            x = _config_to_x(config, fidelity=None)
            best_value, best_x = _record_eval(
                y,
                x,
                None,
                candidates,
                seen_y,
                best_value,
                best_x,
                best_y_all,
                best_obs_x_all,
            )

            best_obs_true = prob.evaluate_true(
                torch.tensor([best_x], dtype=_DTYPE)
            ).item()
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(
                max(prob._optimal_value - best_obs_true, 0.0)
            )
            log_best_obs_regret_all.append(
                math.log(max(prob._optimal_value - best_obs_true, 1e-8))
            )

            x_true = prob.evaluate_true(torch.tensor([x], dtype=_DTYPE)).item()
            _best_simple_true_running = max(_best_simple_true_running, x_true)
            simple_regret_all.append(
                max(prob._optimal_value - _best_simple_true_running, 0.0)
            )
            log_simple_regret_all.append(
                math.log(
                    max(prob._optimal_value - _best_simple_true_running, 1e-8)
                )
            )

            budget += 1.0  # single-fidelity: cost = 1 per eval

            if verbose:
                print(
                    f"  budget={budget:.2f}/{iterations}  y={y:.4f}  best={best_value:.4f}"
                )

        t1 = time.time()
        print(
            f"Trial {trial_idx + 1}/{num_trials} done in {t1 - t0:.1f}s — best={best_value:.3g}"
        )
        all_trial_results.append(
            {
                "trial": trial_idx,
                "seed": trial_idx,
                "time_seconds": t1 - t0,
                "best_y_all": best_y_all,
                "best_obs_x_all": best_obs_x_all,
                "best_obs_true_all": best_obs_true_all,
                "best_obs_regret_all": best_obs_regret_all,
                "log_best_obs_regret_all": log_best_obs_regret_all,
                "simple_regret_all": simple_regret_all,
                "log_simple_regret_all": log_simple_regret_all,
                "candidates": candidates,
                "seen_y": seen_y,
            }
        )

    return all_trial_results


def run_cmaes(prob, iterations, num_trials, verbose, initial_random_samples=0):
    """Optuna CMA-ES (with_margin) for single-fidelity HPO.

    Mixed-integer CMA-ES: continuous and ordinal-integer dims are handled by
    with_margin=True; lora_target is declared as suggest_categorical so
    CmaEsSampler falls back to random sampling for that dim independently,
    avoiding the false assumption that its integer codes are ordered.
    """
    all_trial_results = []

    for trial_idx in range(num_trials):
        torch.manual_seed(trial_idx)
        sampler = optuna.samplers.CmaEsSampler(
            seed=trial_idx, with_margin=True, warn_independent_sampling=False
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        t0 = time.time()

        (
            best_value,
            best_x,
            best_y_all,
            best_obs_x_all,
            best_obs_true_all,
            best_obs_regret_all,
            log_best_obs_regret_all,
            seen_y,
            candidates,
            simple_regret_all,
            log_simple_regret_all,
            _best_simple_true_running,
        ) = _run_init_phase(
            prob,
            initial_random_samples,
            np.random.default_rng(trial_idx),
            "hpo",
        )

        for xi, yi in zip(candidates, seen_y):
            frozen = optuna.trial.create_trial(
                params={
                    name: (
                        float(xi[i]) if ptype == "float"
                        else int(xi[i])
                    )
                    for i, (name, ptype, _, _) in enumerate(HP_PARAM_DEFS)
                },
                distributions={
                    name: (
                        optuna.distributions.FloatDistribution(lo, hi)
                        if ptype == "float"
                        else optuna.distributions.CategoricalDistribution(
                            choices=tuple(range(lo, hi + 1))
                        )
                        if name == "lora_target"
                        else optuna.distributions.IntDistribution(lo, hi)
                    )
                    for name, ptype, lo, hi in HP_PARAM_DEFS
                },
                value=yi,
            )
            study.add_trial(frozen)

        budget = 0.0

        while budget < iterations:
            trial = study.ask()
            config = _suggest_config_cmaes(trial)
            y = _evaluate(prob, config, fidelity=None)
            study.tell(trial, y)

            x = _config_to_x(config, fidelity=None)
            best_value, best_x = _record_eval(
                y,
                x,
                None,
                candidates,
                seen_y,
                best_value,
                best_x,
                best_y_all,
                best_obs_x_all,
            )

            best_obs_true = prob.evaluate_true(
                torch.tensor([best_x], dtype=_DTYPE)
            ).item()
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(
                max(prob._optimal_value - best_obs_true, 0.0)
            )
            log_best_obs_regret_all.append(
                math.log(max(prob._optimal_value - best_obs_true, 1e-8))
            )

            x_true = prob.evaluate_true(torch.tensor([x], dtype=_DTYPE)).item()
            _best_simple_true_running = max(_best_simple_true_running, x_true)
            simple_regret_all.append(
                max(prob._optimal_value - _best_simple_true_running, 0.0)
            )
            log_simple_regret_all.append(
                math.log(
                    max(prob._optimal_value - _best_simple_true_running, 1e-8)
                )
            )

            budget += 1.0

            if verbose:
                print(
                    f"  budget={budget:.2f}/{iterations}  y={y:.4f}  best={best_value:.4f}"
                )

        t1 = time.time()
        print(
            f"Trial {trial_idx + 1}/{num_trials} done in {t1 - t0:.1f}s — best={best_value:.3g}"
        )
        all_trial_results.append(
            {
                "trial": trial_idx,
                "seed": trial_idx,
                "time_seconds": t1 - t0,
                "best_y_all": best_y_all,
                "best_obs_x_all": best_obs_x_all,
                "best_obs_true_all": best_obs_true_all,
                "best_obs_regret_all": best_obs_regret_all,
                "log_best_obs_regret_all": log_best_obs_regret_all,
                "simple_regret_all": simple_regret_all,
                "log_simple_regret_all": log_simple_regret_all,
                "candidates": candidates,
                "seen_y": seen_y,
            }
        )

    return all_trial_results


def run_bohb(prob, iterations, num_trials, verbose, initial_random_samples=0):
    """SMAC3 BOHB for HPOMultiFidelityToken (continuous fidelity).

    BOHB combines HyperBand's multi-fidelity scheduling with a KDE surrogate
    (similar to TPE). SMAC's MultiFidelityFacade maps bolt's fidelity ∈ [0, 1]
    directly to the budget parameter.

    Budget rungs with min_budget=0.1, max_budget=1.0, eta=3:
        rung 0: budget ≈ 0.100
        rung 1: budget ≈ 0.316
        rung 2: budget = 1.000
    """
    try:
        from smac import MultiFidelityFacade, Scenario
        from smac.runhistory.dataclasses import TrialValue
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace import Float as CSFloat, Integer as CSInteger
    except ImportError:
        raise ImportError("SMAC3 not installed. Run: pip install smac")

    all_trial_results = []

    for trial_idx in range(num_trials):
        torch.manual_seed(trial_idx)
        cs = ConfigurationSpace(seed=trial_idx)
        cs.add(
            [
                CSFloat("lr", (0.0, 1.0)),
                CSInteger("batch", (2, 4)),
                CSInteger("lora_rank", (2, 5)),
                CSInteger("lora_alpha", (2, 5)),
                CSFloat("lora_dropout", (0.0, 1.0)),
                CSInteger("lora_layers", (1, 30)),
                CSInteger("lora_target", (0, 3)),
            ]
        )
        scenario = Scenario(
            configspace=cs,
            n_trials=100000,  # large; we stop by budget
            seed=trial_idx,
            min_budget=0.1,
            max_budget=1.0,
            name=f"bohb_trial_{trial_idx}",
        )

        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=lambda config, seed=0, budget=1.0: 0.0,  # unused with ask/tell
            overwrite=True,
        )

        t0 = time.time()
        (
            best_value,
            best_x,
            best_y_all,
            best_obs_x_all,
            best_obs_true_all,
            best_obs_regret_all,
            log_best_obs_regret_all,
            seen_y,
            candidates,
            simple_regret_all,
            log_simple_regret_all,
            _best_simple_true_running,
        ) = _run_init_phase(
            prob,
            initial_random_samples,
            np.random.default_rng(trial_idx),
            "hpo_fd_step",
        )
        budget = 0.0
        budget_all = []

        while budget < iterations:
            trial_info = smac.ask()
            fidelity = trial_info.budget  # SMAC budget IS the bolt fidelity
            config_dict = dict(trial_info.config)

            y = _evaluate(prob, config_dict, fidelity=fidelity)

            # SMAC minimizes cost, so pass negative y
            smac.tell(trial_info, TrialValue(cost=-y, time=0.01))

            x = _config_to_x(config_dict, fidelity=fidelity)
            best_value, best_x = _record_eval(
                y,
                x,
                fidelity,
                candidates,
                seen_y,
                best_value,
                best_x,
                best_y_all,
                best_obs_x_all,
            )

            best_obs_true = prob.evaluate_true(
                torch.tensor([best_x], dtype=_DTYPE)
            ).item()
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(
                max(prob._optimal_value - best_obs_true, 0.0)
            )
            log_best_obs_regret_all.append(
                math.log(max(prob._optimal_value - best_obs_true, 1e-8))
            )

            x_true = prob.evaluate_true(torch.tensor([x], dtype=_DTYPE)).item()
            _best_simple_true_running = max(_best_simple_true_running, x_true)
            simple_regret_all.append(
                max(prob._optimal_value - _best_simple_true_running, 0.0)
            )
            log_simple_regret_all.append(
                math.log(
                    max(prob._optimal_value - _best_simple_true_running, 1e-8)
                )
            )

            budget += prob.cost(torch.tensor([x], dtype=_DTYPE)).item()
            budget_all.append(budget)

            if verbose:
                print(
                    f"  budget={budget:.2f}/{iterations}  fidelity={fidelity:.3f}"
                    f"  y={y:.4f}  best={best_value:.4f}"
                )

        t1 = time.time()
        print(
            f"Trial {trial_idx + 1}/{num_trials} done in {t1 - t0:.1f}s — best={best_value:.3g}"
        )
        all_trial_results.append(
            {
                "trial": trial_idx,
                "seed": trial_idx,
                "time_seconds": t1 - t0,
                "best_y_all": best_y_all,
                "best_obs_x_all": best_obs_x_all,
                "best_obs_true_all": best_obs_true_all,
                "best_obs_regret_all": best_obs_regret_all,
                "log_best_obs_regret_all": log_best_obs_regret_all,
                "simple_regret_all": simple_regret_all,
                "log_simple_regret_all": log_simple_regret_all,
                "candidates": candidates,
                "seen_y": seen_y,
                "budget_all": budget_all,
            }
        )

    return all_trial_results


def run_asha(prob, iterations, num_trials, verbose, initial_random_samples=0):
    """Optuna ASHA (Asynchronous Successive Halving) for HPOMultiFidelityModel.

    Discrete 2-level fidelity: {0=4B model (cheap, low fidelity), 1=8B model (expensive, high fidelity)}.
    Evaluates each config at fidelity=0 first; poorly-performing configs are
    pruned (budget cost = 0.1), top configs are promoted to fidelity=1
    (additional budget cost = 1.0).

    Uses SuccessiveHalvingPruner (the underlying algorithm for ASHA) with
    reduction_factor=2: roughly top 50% of configs at fidelity=0 are promoted.
    """
    all_trial_results = []

    for trial_idx in range(num_trials):
        torch.manual_seed(trial_idx)
        sampler = optuna.samplers.TPESampler(seed=trial_idx)
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=2,
            min_early_stopping_rate=0,
        )
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        t0 = time.time()
        best_value = -np.inf
        best_x = None
        best_y_all, best_obs_x_all = [], []
        best_obs_true_all, best_obs_regret_all, log_best_obs_regret_all = (
            [],
            [],
            [],
        )
        seen_y, candidates = [], []
        simple_regret_all, log_simple_regret_all = [], []
        _best_simple_true_running = -np.inf

        # One free step (not counted in budget) to produce a step-0 entry,
        # matching the +1 structure of other methods' init phase.
        _trial0 = study.ask()
        _config0 = _suggest_config_optuna(_trial0)
        _y0_low = _evaluate(prob, _config0, fidelity=0)
        _trial0.report(_y0_low, step=1)
        _x0_low = _config_to_x(_config0, fidelity=0)

        if _trial0.should_prune():
            study.tell(_trial0, state=optuna.trial.TrialState.PRUNED)
            _y0, _x0 = _y0_low, _x0_low
        else:
            _y0_high = _evaluate(prob, _config0, fidelity=1)
            _trial0.report(_y0_high, step=2)
            study.tell(_trial0, _y0_high)
            _y0, _x0 = _y0_high, _config_to_x(_config0, fidelity=1)

        candidates.append(_x0)
        seen_y.append(_y0)
        best_value, best_x = _y0, list(_x0)
        best_y_all.append(best_value)
        best_obs_x_all.append(list(best_x))

        _best_obs_true0 = prob.evaluate_true(
            torch.tensor([best_x], dtype=_DTYPE)
        ).item()
        best_obs_true_all.append(_best_obs_true0)
        best_obs_regret_all.append(
            max(prob._optimal_value - _best_obs_true0, 0.0)
        )
        log_best_obs_regret_all.append(
            math.log(max(prob._optimal_value - _best_obs_true0, 1e-8))
        )

        _best_simple_true_running = prob.evaluate_true(
            torch.tensor([_x0], dtype=_DTYPE)
        ).item()
        simple_regret_all.append(
            max(prob._optimal_value - _best_simple_true_running, 0.0)
        )
        log_simple_regret_all.append(
            math.log(max(prob._optimal_value - _best_simple_true_running, 1e-8))
        )

        budget = 0.0
        budget_all = []

        while budget < iterations:
            trial = study.ask()
            config = _suggest_config_optuna(trial)

            # --- Evaluate at fidelity=0 (cheap: 4B model, low fidelity, cost=0.1) ---
            y_low = _evaluate(prob, config, fidelity=0)
            trial.report(y_low, step=1)

            x_low = _config_to_x(config, fidelity=0)
            best_value, best_x = _record_eval(
                y_low,
                x_low,
                0,
                candidates,
                seen_y,
                best_value,
                best_x,
                best_y_all,
                best_obs_x_all,
            )

            best_obs_true = prob.evaluate_true(
                torch.tensor([best_x], dtype=_DTYPE)
            ).item()
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(
                max(prob._optimal_value - best_obs_true, 0.0)
            )
            log_best_obs_regret_all.append(
                math.log(max(prob._optimal_value - best_obs_true, 1e-8))
            )

            x_low_true = prob.evaluate_true(
                torch.tensor([x_low], dtype=_DTYPE)
            ).item()
            _best_simple_true_running = max(
                _best_simple_true_running, x_low_true
            )

            simple_regret_all.append(
                max(prob._optimal_value - _best_simple_true_running, 0.0)
            )
            log_simple_regret_all.append(
                math.log(
                    max(prob._optimal_value - _best_simple_true_running, 1e-8)
                )
            )

            budget += prob.cost(
                torch.tensor([x_low], dtype=_DTYPE)
            ).item()  # fidelity=0 → cost=0.1
            budget_all.append(budget)

            if budget >= iterations:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                break

            if trial.should_prune():
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                if verbose:
                    print(
                        f"  budget={budget:.2f}/{iterations}  PRUNED"
                        f"  y_low={y_low:.4f}  best={best_value:.4f}"
                    )
                continue

            # --- Promoted: evaluate at fidelity=1 (expensive: 8B model, high fidelity, cost=1.0) ---
            y_high = _evaluate(prob, config, fidelity=1)
            trial.report(y_high, step=2)
            study.tell(trial, y_high)

            x_high = _config_to_x(config, fidelity=1)
            best_value, best_x = _record_eval(
                y_high,
                x_high,
                1,
                candidates,
                seen_y,
                best_value,
                best_x,
                best_y_all,
                best_obs_x_all,
            )
            best_obs_true = prob.evaluate_true(
                torch.tensor([best_x], dtype=_DTYPE)
            ).item()
            best_obs_true_all.append(best_obs_true)
            best_obs_regret_all.append(
                max(prob._optimal_value - best_obs_true, 0.0)
            )
            log_best_obs_regret_all.append(
                math.log(max(prob._optimal_value - best_obs_true, 1e-8))
            )
            x_high_true = prob.evaluate_true(
                torch.tensor([x_high], dtype=_DTYPE)
            ).item()
            _best_simple_true_running = max(
                _best_simple_true_running, x_high_true
            )
            simple_regret_all.append(
                max(prob._optimal_value - _best_simple_true_running, 0.0)
            )
            log_simple_regret_all.append(
                math.log(
                    max(prob._optimal_value - _best_simple_true_running, 1e-8)
                )
            )
            budget += prob.cost(
                torch.tensor([x_high], dtype=_DTYPE)
            ).item()  # fidelity=1 → cost=1.0
            budget_all.append(budget)

            if verbose:
                print(
                    f"  budget={budget:.2f}/{iterations}"
                    f"  y_low={y_low:.4f}  y_high={y_high:.4f}  best={best_value:.4f}"
                )

        t1 = time.time()
        print(
            f"Trial {trial_idx + 1}/{num_trials} done in {t1 - t0:.1f}s — best={best_value:.3g}"
        )
        all_trial_results.append(
            {
                "trial": trial_idx,
                "seed": trial_idx,
                "time_seconds": t1 - t0,
                "best_y_all": best_y_all,
                "best_obs_x_all": best_obs_x_all,
                "best_obs_true_all": best_obs_true_all,
                "best_obs_regret_all": best_obs_regret_all,
                "log_best_obs_regret_all": log_best_obs_regret_all,
                "simple_regret_all": simple_regret_all,
                "log_simple_regret_all": log_simple_regret_all,
                "candidates": candidates,
                "seen_y": seen_y,
                "budget_all": budget_all,
            }
        )

    return all_trial_results


def main(args):
    print(f"problem: {args.problem}")
    print(f"method:  {args.method}")
    print(f"iterations: {args.iterations}")
    print(f"initial_random_samples: {args.initial_random_samples}")
    print(f"trials: {args.trials}")

    # --noise_std unset: let each problem class use its own default noise std
    noise_kwargs = {} if args.noise_std is None else {"noise_std": args.noise_std}

    if args.problem == "hpo":
        prob = HPO(negate=False, **noise_kwargs)
    elif args.problem == "hpo_fd_step":
        prob = HPOMultiFidelityToken(negate=False, **noise_kwargs)
    elif args.problem == "hpo_fd_model":
        prob = HPOMultiFidelityModel(negate=False, **noise_kwargs)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")

    noise_std = prob.noise_std
    print("noise std:", noise_std)

    method = args.method
    problem_name = args.problem

    # Validate method/problem combinations
    if method in ("tpe", "cmaes") and problem_name != "hpo":
        raise ValueError(
            f"{method.upper()} is a single-fidelity baseline and should be run on 'hpo', "
            f"not '{problem_name}'. Use 'random' or 'bohb'/'asha' for MF problems."
        )
    if method == "bohb" and problem_name != "hpo_fd_step":
        raise ValueError(
            "BOHB (SMAC3) targets continuous fidelity and should be run on "
            f"'hpo_fd_step', not '{problem_name}'."
        )
    if method == "asha" and problem_name != "hpo_fd_model":
        raise ValueError(
            "ASHA (Successive Halving) targets discrete 2-level fidelity and "
            f"should be run on 'hpo_fd_model', not '{problem_name}'."
        )

    if method == "random":
        all_trial_results = run_random(
            prob,
            problem_name,
            args.iterations,
            args.trials,
            args.verbose,
            initial_random_samples=args.initial_random_samples,
        )
    elif method == "tpe":
        all_trial_results = run_tpe(
            prob,
            args.iterations,
            args.trials,
            args.verbose,
            initial_random_samples=args.initial_random_samples,
        )
    elif method == "cmaes":
        all_trial_results = run_cmaes(
            prob,
            args.iterations,
            args.trials,
            args.verbose,
            initial_random_samples=args.initial_random_samples,
        )
    elif method == "bohb":
        all_trial_results = run_bohb(
            prob,
            args.iterations,
            args.trials,
            args.verbose,
            initial_random_samples=args.initial_random_samples,
        )
    elif method == "asha":
        all_trial_results = run_asha(
            prob,
            args.iterations,
            args.trials,
            args.verbose,
            initial_random_samples=args.initial_random_samples,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    results = {
        "method": method,
        "problem": problem_name,
        "iterations": args.iterations,
        "initial_random_samples": args.initial_random_samples,
        "noise_std": noise_std,
        "num_trials": args.trials,
        "emulator_versions": emulator_version.emulator_versions_for(prob),
        "trials": all_trial_results,
    }
    info_tag = f"_{args.info}" if args.info else ""
    results_dir = REPO_ROOT / "results" / f"hpo{args.folder_prefix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        results_dir
        / f"{problem_name}_{method}_{args.trials}trials_{args.iterations}iterations{info_tag}_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HPO baseline comparison (Random, TPE, BOHB, ASHA)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="hpo",
        choices=["hpo", "hpo_fd_step", "hpo_fd_model"],
        help=(
            "Problem to optimize. "
            "'hpo': single-fidelity (7D); "
            "'hpo_fd_step': continuous fidelity (training tokens); "
            "'hpo_fd_model': discrete 2-level fidelity (model size). "
            "Default: hpo"
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
        "--method",
        type=str,
        default="random",
        choices=["random", "tpe", "cmaes", "bohb", "asha"],
        help=(
            "Baseline method. "
            "'random': random search (all problems); "
            "'tpe': Optuna TPE, single-fidelity (hpo only); "
            "'cmaes': Optuna CMA-ES with_margin, single-fidelity (hpo only); "
            "'bohb': SMAC3 BOHB, continuous fidelity (hpo_fd_step only); "
            "'asha': Optuna Successive Halving, discrete 2-level (hpo_fd_model only). "
            "Default: random"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Budget limit (same unit as BO script). Default: 100",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent trials (random seeds 0, 1, ...). Default: 1",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="Optional tag appended to the output filename.",
    )
    parser.add_argument(
        "--initial_random_samples",
        type=int,
        default=10,
        help=(
            "Number of initial random evaluations before the optimizer starts "
            "(not counted in budget). Set to match --initial_random_samples from "
            "test_w_botorch_mixed.py for fair comparison. Default: 10"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-evaluation logs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
