# Bolt Experiments

Experiment and analysis scripts reproducing the paper's results on [`bolt`](https://github.com/anonom799/bolt), a benchmark suite for Bayesian optimization of expensive LLM tasks.

Every runner writes a JSON file in a shared output format, so results are directly comparable across methods and can be plotted by the same `bolt_exp.plot_results`.

Related repos:

- [`bolt`](https://github.com/anonom799/bolt) — the benchmark suite (surrogate-backed problems)
- [`bolt-data`](https://github.com/anonom799/bolt-data) — data collection and surrogate training

---

## Install

```bash
pip install -r requirements.txt
```

`bolt-bench` pulls its surrogates from the HuggingFace Hub on first use; no local data is needed to run the optimizers.

`figures.plot_emulators` plots surrogate diagnostics against the raw eval data those surrogates were fit to. That data (held-out val sets plus the DM Pareto front and noise targets) is committed under `data/`, along with the PCO candidate tables and the cached MLHGP noise-learning sweep, so every figure script runs without extra setup. Override the defaults with the `--*_path` flags to point at your own collection runs.

---

## Layout

Everything is invoked as a module from the **repository root**:
`python -m bolt_exp.<subpackage>.<module>`.

```
bolt_exp/
├── plot_results.py      plotting library + CLI, shared by the figure scripts
├── hpo_flops.py         the training-FLOPs formula and its derivation
├── mlhgp.py             Most Likely Heteroscedastic GP (Kersting et al., 2007)
├── constrained_acqf.py  constrained acquisitions for PCO (UCB-C, CMES-IBO, cKG)
├── parallel_prior.py    physics-informed GP prior means for PCO (--prior_mean)
├── emulator_version.py  records which emulator version a run used
├── runners/             BO runners; each writes a result JSON
├── analysis/            analyses, FLOPs accounting, LaTeX tables
└── figures/             paper figures
scripts/                 shell drivers for the full sweeps and figure sets
plot_configs/            per-figure plot styling (colors, labels, ordering)
results/                 experiment output JSONs (large ones ship gzipped; runners write here)
flops/                   per-query and per-method FLOPs accounting (CSV)
tables/                  generated LaTeX / Markdown tables used in the paper
data/                    raw eval data behind the surrogates, PCO tables, MLHGP cache
```

`bolt_exp.REPO_ROOT` anchors `results/`, `plot_configs/`, `flops/`, `tables/` and `data/`,
so modules find them regardless of the working directory. `bolt_exp.load_result` /
`result_files` / `dedupe_results` are the shared readers every plot and analysis
goes through; they hide whether a result is stored plain or gzipped.

### Results

The raw result JSONs behind every number and figure in the paper are committed under `results/` — all 250 of them, plus `results/po_final_regret.csv` (the collated PO regrets that `analysis.po_latex_table` reads) and `results/dm_real/all_trials.parquet` (per-objective scores of the real DM runs).

The 37 largest PO files are stored **gzipped** (`*.json.gz`) to stay under GitHub's per-file size limits. No decompression step is needed: every loader goes through `bolt_exp.load_result`, which reads `.json.gz` transparently and falls back from a `.json` path to the `.json.gz` beside it, and the file-collecting analyses (`analysis.collate_po_regret`, `analysis.table_wall_clock`) glob both forms. A fresh clone plots and tabulates the complete set.

To get plain JSON anyway — for inspection or for other tools — decompress at the repository root:

```bash
find results -name '*.json.gz' -exec gunzip {} +
```

That expands the tree from 2.0 GB to ~5 GB. Either layout works, including a half-decompressed one: `gunzip -k` leaves both copies in place and the loaders keep the plain `.json`, so a `*.json*` shell glob can't double-count a run.

| Directory | Files | Trials × iterations | As committed |
|---|---|---|---|
| `results/hpo/` | 33 | 10 × 200 | 86 MB |
| `results/hpo_fd_{step,model}_costsens/` | 18 + 18 | 10 × 50 | 34 MB |
| `results/dm/` | 50 | 10 × 200 | 212 MB |
| `results/po/` | 78 (37 gzipped) | 5 × 200 | 1.6 GB |
| `results/pco16/`, `results/pco64/` | 15 + 15 | 10 × 100 | 34 MB |
| `results/pco32/` | 15 | 10 × 50 | 9 MB |
| `results/{hpo,dm}_real/` | 4 + 4 | 5 × 50 | 0.7 MB |

`hpo_fd_*` runs carry a `cost{c}` tag: each acquisition's main result uses the fidelity-cost scale that did best in the 50-iteration sensitivity sweep (`*_costsens`, 6 acquisitions × 3 scales per problem). The UCB β sweeps (0.5–30) sit beside the main results in `results/hpo/` and `results/dm/`. The `*_real` directories hold BO runs on the *real* tasks — actual LLM fine-tuning, not the emulator — which the emulator-validation figures compare against.

Every file carries the headline metric — `simple_regret_all` / `log_simple_regret_all` for single-objective, `log_best_hv_diff_true_all` for multi-objective — so all of them plot directly with `bolt_exp.plot_results`. Runs of `runners.test_w_baselines_hpo` are model-free and therefore omit the `rec_*` / `*inference_regret*` families; see [Result JSON structure](#result-json-structure).

The PO files dominate the total because each stores the full `(n0 + iterations) × D` candidate matrix at up to 768 dims. They compress well — roughly 6×, which is why only those needed gzipping.

To regenerate rather than reuse: run the `scripts/run_*.sh` drivers, then `scripts/plot_figs.sh`. Runners can also be sharded one seed per process (`--trials 1 --seed_offset <s>`) and the shards joined with `python -m bolt_exp.analysis.merge_seed_runs <dir>`, which is how the 10-trial files were produced.

---

## Problems

### HPO

| Problem | Fidelity | Dims | Description |
|---|---|---|---|
| `hpo` | None (single-fidelity) | 10 | Qwen3-8B LoRA fine-tuning HPO at full training budget |
| `hpo_fd_step` | Continuous ∈ [0, 1] | 11 | Same task; fidelity = normalised training token count (1e5–9e6 tokens) |
| `hpo_fd_model` | Discrete ∈ {0, 1} | 11 | Same task; fidelity = model size (0 = Qwen3-8B, 1 = Qwen3-3B) |

Budget cost per evaluation: `0.1 + 0.9 × fidelity` (fidelity = 1.0 for single-fidelity).

All HPO and DM problems are noisy by default, at the noise level measured across repeat training runs of the real task (`--noise_std` overrides it).

### Data Mixture (DM)

| Problem | Type | Dims | Description |
|---|---|---|---|
| `dm_curriculum` | Single-objective | 6 | Data mixture selection; two 3-simplex groups; maximise a scalar LLM metric |
| `dm_curriculum_mo` | Multi-objective | 6 | Same; three objectives evaluated jointly |
| `dm_curriculum_heteroscedastic` | Single-objective, heteroscedastic | 6 | Same with input-dependent observation noise from a learned noise emulator |

### Prompt Optimization (PO)

| Problem | Type | Dims | Description |
|---|---|---|---|
| `po128` / `po256` / `po512` / `po768` | Discrete candidate set | 128–768 | Nearest-neighbour lookup over 5014 tabular prompt embeddings, scored by MATH500 0-shot accuracy on Qwen3-14B |

### Parallelism Configuration Optimization (PCO)

| Problem | Type | Dims | Description |
|---|---|---|---|
| `pco16` | Black-box constraint, discrete candidate set | 8 | Parallelism configuration of distributed LLM training on 16 GPUs; 919 configurations |
| `pco32` | Black-box constraint, discrete candidate set | 8 | 40-layer model on 32 GPUs; 379 configurations, 339 feasible |
| `pco64` | Black-box constraint, discrete candidate set | 8 | 64-layer model on 64 GPUs; 780 configurations, 560 feasible |

Maximise training throughput subject to a *hidden* constraint: an infeasible configuration runs out of GPU memory, so its throughput is never measured. The problem imputes those rows from their nearest feasible neighbours, so the objective model is fit on every observation; the constraint model is fit on the memory margin, in BoTorch's sign convention (feasible where `c(x) <= 0`). The search space is the finite table of valid configurations — the parallelism degrees must multiply to the GPU count — so acquisitions are maximised over `prob.candidates()`, not over the bounding box. The tables are small, which is why the budgets are 100 iterations (50 on `pco32`).

---

## Methods by Problem

### DM curriculum — single-objective (`dm_curriculum`)

| Script | Method flag | Algorithm |
|---|---|---|
| `runners.test_w_botorch_dm` | `--acq_fn ei/ucb/kg/mes/gibbon/pes/jes/ts/qnei` | GP-BO (BoTorch); simplex equality constraints via `optimize_acqf` |
| `runners.test_w_botorch_dm` | `--acq_fn random` | Uniform Dirichlet baseline |

### DM curriculum — multi-objective (`dm_curriculum_mo`)

| Script | Method flag | Algorithm |
|---|---|---|
| `runners.test_w_botorch_dm` | `--acq_fn qnehvi/qparego/qhvkg/jes_mo/mes_mo/pes_mo` | MO GP-BO (BoTorch); ModelListGP with one GP per objective |
| `runners.test_w_botorch_dm` | `--acq_fn random` | Uniform Dirichlet baseline (no model) |
| `runners.test_w_baselines_mo` | `--method tsemo` | TSEMO — Thompson-sampling MO baseline |
| `runners.test_w_baselines_mo` | `--method nsga2/nsga3` | Evolutionary MO baselines (pymoo), simplex-repaired |

### DM curriculum — heteroscedastic (`dm_curriculum_heteroscedastic`)

| Script | Method flag | Algorithm |
|---|---|---|
| `runners.test_w_botorch_dm` | `--known_noise` | Known-noise GP; conditions on the emulator's noise variances |
| `runners.test_w_botorch_dm` | `--mlhgp` | MLHGP (Kersting et al.); learns input-dependent noise via EM |

### Single-fidelity HPO (`hpo`)

| Script | Method flag | Algorithm | Description |
|---|---|---|---|
| `runners.test_w_botorch_mixed` | `--acq_fn ei/qnei/ucb/mes/gibbon/pes/jes/ts` | GP-BO (BoTorch) | Gaussian process surrogate with acquisition function optimization. Handles mixed discrete/continuous space via `optimize_acqf_mixed`. |
| `runners.test_w_baselines_hpo` | `--method random` | Random Search | Uniform random sampling over the full search space. |
| `runners.test_w_baselines_hpo` | `--method tpe` | TPE (Optuna) | Tree-structured Parzen Estimator; models good/bad regions independently and samples from the good region. |
| `runners.test_w_baselines_hpo` | `--method cmaes` | CMA-ES (Optuna) | Covariance matrix adaptation evolution strategy (with margin, for the integer dims). |

### Multi-fidelity HPO — continuous fidelity (`hpo_fd_step`)

| Script | Method flag | Algorithm | Description |
|---|---|---|---|
| `runners.test_w_botorch_mixed` | `--acq_fn ei/qnei/ucb/pes/mfmes/mfgibbon` | GP-BO (BoTorch) | GP surrogate including fidelity as a continuous input; `--cost_scale` sets the fidelity-cost model. |
| `runners.test_w_baselines_hpo` | `--method random` | Random Search | Randomly samples fidelity alongside HPs; budget-matched baseline for fair comparison with BOHB. |
| `runners.test_w_baselines_hpo` | `--method bohb` | BOHB (SMAC3) | Bayesian optimization and HyperBand. Uses a KDE surrogate with HyperBand scheduling across continuous budgets (rungs at ~10 %, 33 %, 100 % of max fidelity). |

### Multi-fidelity HPO — discrete fidelity (`hpo_fd_model`)

| Script | Method flag | Algorithm | Description |
|---|---|---|---|
| `runners.test_w_botorch_mixed` | `--acq_fn ei/qnei/ucb/pes/mfmes/mfgibbon` | GP-BO (BoTorch) | GP surrogate with discrete fidelity as an additional dimension. |
| `runners.test_w_baselines_hpo` | `--method random` | Random Search | Randomly samples fidelity ∈ {0, 1}; budget-matched baseline for fair comparison with ASHA. |
| `runners.test_w_baselines_hpo` | `--method asha` | ASHA (Optuna) | Asynchronous Successive Halving. Evaluates all configs at fidelity 0 (cheap), promotes survivors to fidelity 1 (expensive). Uses `SuccessiveHalvingPruner`. |

### Prompt optimization (`po128` … `po768`)

| Script | Method flag | Algorithm |
|---|---|---|
| `runners.test_w_botorch_po` | `--acq_fn ei/qnei/ucb/kg/mes/gibbon/pes/jes/ts/random` | GP-BO over the discrete candidate set via `optimize_acqf_discrete` |
| `runners.test_w_botorch_po` | `--turbo` | TuRBO trust-region filtering |
| `runners.test_w_botorch_po` | `--baxus` | BAxUS random subspace embedding, expanding on stagnation |
| `runners.test_w_botorch_po` | `--saasbo` | SAASBO — fully Bayesian GP with SAAS horseshoe prior (NUTS) |
| `runners.test_w_botorch_po` | `--raasp` / `--msr` / `--mle_scaled_init` | High-dimensional candidate generation and lengthscale-init variants |

### PCO (`pco16`, `pco32`, `pco64`)

All methods are published constrained-BO algorithms, maximised over the discrete candidate set.

| Script | Method flag | Algorithm |
|---|---|---|
| `runners.test_w_botorch_pco` | `--acq_fn ei` | EIC — EI weighted by probability of feasibility (Gardner et al., 2014) |
| `runners.test_w_botorch_pco` | `--acq_fn qnei` | Noisy constrained EI (Letham et al., 2019) |
| `runners.test_w_botorch_pco` | `--acq_fn ucb_c` | UCB-C — objective UCB over the optimistic feasible region (Nguyen et al., 2024, Alg. 2) |
| `runners.test_w_botorch_pco` | `--acq_fn cts` | Constrained Thompson sampling: SCBO's sampling rule over the whole table, without its trust region |
| `runners.test_w_botorch_pco` | `--acq_fn scbo` | SCBO (Eriksson & Poloczek, 2021) — the same rule confined to a trust region over the table; run with `--ls_floor` |
| `runners.test_w_botorch_pco` | `--acq_fn cmes_ibo` | CMES-IBO — constrained max-value entropy search (Takeno et al., 2022) |
| `runners.test_w_botorch_pco` | `--acq_fn ckg` | cKG — constrained knowledge gradient (Ungredda & Branke, 2021, Alg. 1) |
| `runners.test_w_botorch_pco` | `--acq_fn admmbo` | ADMMBO — ADMM splitting into objective and feasibility subproblems (Ariafar et al., 2019) |
| `runners.test_w_botorch_pco` | `--acq_fn tpe/cmaes` | TPE / CMA-ES (Optuna), with the constraint passed to the sampler |
| `runners.test_w_botorch_pco` | `--acq_fn random` | Uniform sampling over the candidate set |

`--prior_mean` is orthogonal to the model-based methods: it swaps both GPs' constant mean for an analytic model of distributed training (`bolt_exp/parallel_prior.py`) — throughput as `1 / (communication + computation)` and peak memory as sharded parameters plus activations — whose parameters are learned jointly with the kernel. Results are tagged `_prior`. `--no_repeats` removes already-evaluated configurations from the pool; by default a repeat is discouraged by the acquisition rather than forbidden. Implementation notes on each method are in the header of `runners/test_w_botorch_pco.py`.

---

## Usage Examples

```bash
# DM — single-objective
python -m bolt_exp.runners.test_w_botorch_dm --problem dm_curriculum --acq_fn ei --iterations 100 --trials 3

# DM — multi-objective
python -m bolt_exp.runners.test_w_botorch_dm --problem dm_curriculum_mo --acq_fn qnehvi --iterations 100 --trials 3

# HPO — GP-BO (BoTorch)
python -m bolt_exp.runners.test_w_botorch_mixed --problem hpo --acq_fn ucb --iterations 100 --trials 3
python -m bolt_exp.runners.test_w_botorch_mixed --problem hpo_fd_step --acq_fn ucb --iterations 100 --trials 3
python -m bolt_exp.runners.test_w_botorch_mixed --problem hpo_fd_model --acq_fn ucb --iterations 100 --trials 3

# HPO — baselines
python -m bolt_exp.runners.test_w_baselines_hpo --problem hpo --method tpe --iterations 100 --trials 3
python -m bolt_exp.runners.test_w_baselines_hpo --problem hpo_fd_step --method bohb --iterations 100 --trials 3
python -m bolt_exp.runners.test_w_baselines_hpo --problem hpo_fd_model --method asha --iterations 100 --trials 3

# PO
python -m bolt_exp.runners.test_w_botorch_po --problem po128 --acq_fn qnei --iterations 100 --trials 3
python -m bolt_exp.runners.test_w_botorch_po --problem po768 --acq_fn ts --turbo --iterations 100 --trials 3

# PCO
python -m bolt_exp.runners.test_w_botorch_pco --problem pco32 --acq_fn ucb_c --iterations 50 --trials 3
python -m bolt_exp.runners.test_w_botorch_pco --problem pco64 --acq_fn qnei --prior_mean --iterations 100 --trials 3

# Full sweeps (as used in the paper)
bash scripts/run_botorch_dm.sh
bash scripts/run_botorch_mixed.sh
bash scripts/run_cost_scales.sh
bash scripts/run_ucb_betas.sh
bash scripts/run_baselines.sh
bash scripts/run_botorch_po.sh
bash scripts/run_botorch_pco.sh
```

Results are saved to `results/{hpo,dm,po,pco16,pco32,pco64}/` (plus `--folder_prefix`, if given). Every filename carries `{problem}_{method}` and `{trials}trials_{iterations}iterations`, with method-variant tags in between or after depending on the runner:

| Runner | Pattern |
|---|---|
| `test_w_botorch_po` | `{problem}_{acq_fn}[_turbo\|_baxus\|_saasbo][_raasp\|_msr][_mlesi][_beta{β}][_q{Q}]_{trials}trials_{iterations}iterations_results.json` |
| `test_w_botorch_dm` | `{problem}_{acq_fn}[_beta{β}][_mlhgp_em{N}][_knownnoise][_q{Q}]_{trials}trials_{iterations}iterations_results.json` |
| `test_w_botorch_mixed` | `{problem}_{acq_fn}_{trials}trials_{iterations}iterations[_cost{c}][_beta{β}]_results.json` |
| `test_w_botorch_pco` | `{problem}_{acq_fn}[_beta{β}][_prior][_lsfloor]_{trials}trials_{iterations}iterations_results.json` |
| `test_w_baselines_hpo` / `_mo` | `{problem}_{method}_{trials}trials_{iterations}iterations_results.json` |

`mlesi` is `--mle_scaled_init`; `q{Q}` appears only for batch size > 1; `cost{c}` is the multi-fidelity cost scale. Examples: `po128_qnei_turbo_mlesi_q5_5trials_200iterations_results.json`, `dm_curriculum_ucb_beta1.0_10trials_200iterations_results.json`, `hpo_fd_step_mfmes_10trials_50iterations_cost0.01_results.json`, `pco32_scbo_lsfloor_10trials_50iterations_results.json`.

---

## Plotting and analysis

```bash
# Regenerate all paper figures into pics/<family>/
bash scripts/plot_figs.sh

# Only some families (hpo, hpo_mf, dm, po, pco, emulator, mlhgp) or single targets
bash scripts/plot_figs.sh pco,dm_batch

# A single figure
python -m bolt_exp.plot_results results/dm/*_10trials_200iterations*.json \
    --out dm_simple.pdf --metric log_simple_regret_all \
    --bo-iter --config plot_configs/dm_okabe_ito.yaml --mean-rank
```

Headline metrics: `log_simple_regret_all` for single-objective problems, `log_best_hv_diff_true_all` for multi-objective. `--mean-rank` also writes a `*_rank` figure of each method's mean rank over time; `--flops` puts the HPO runs on a training-FLOPs x axis.

| Script | Purpose |
|---|---|
| `bolt_exp.plot_results` | Main regret / hypervolume curves, mean-rank plots, FLOPs axis |
| `figures.plot_emulators` | Surrogate diagnostics (landscape, predicted-vs-actual, Pareto front) |
| `figures.plot_fidelity_proportions` | Fidelity usage over the multi-fidelity runs |
| `figures.plot_baxus_turbo`, `figures.plot_cost_scale_grid` | PO ablations, HPO cost-scale grid |
| `figures.plot_pco_plateau`, `figures.plot_pco_outcomes` | PCO throughput landscape; success rates, rank heatmaps and final outcomes |
| `figures.plot_real_vs_emulated`, `figures.plot_spearman_grid`, `analysis.table_real_vs_emulated_fit` | Emulator validation: BO on the real tasks vs on the emulators, and emulator fit on the points the real runs visited |
| `analysis.mlhgp_noise_vs_bo_iter`, `figures.plot_mlhgp_noise_learning` | How well MLHGP's learned noise tracks the true noise as BO collects data |
| `analysis.merge_seed_runs` | Join per-seed result shards into one multi-trial file |
| `analysis.compute_hpo_flops`, `bolt_exp.hpo_flops`, `analysis.flops_table` | FLOPs accounting; writes `flops/` |
| `analysis.collate_po_regret`, `analysis.po_latex_table`, `analysis.table_wall_clock` | `analysis.collate_po_regret` writes `po_final_regret.csv`, which `analysis.po_latex_table` turns into `tables/po_delta_table.tex`; `analysis.table_wall_clock` writes the wall-clock tables (HPO/DM, PO, PCO) |
| `analysis.find_optimal` | Locate a problem's optimum by grid search + local refinement (how the reference optima behind the regret metrics were obtained) |
| `analysis.analyse_po_pca`, `analysis.analyse_dm_curriculum` | Analyses reported in the appendix |


---

## Result JSON structure

**Common top-level keys:** `acq_fn` / `method`, `iterations`, `initial_random_samples`, `noise_std`, `num_trials`, `trials`; the emulator-backed runners also record `emulator_versions` (`{hf_repo: version}`), and per-trial `device` / `gpu_name`. `problem` is present in every runner's output except `runners.test_w_botorch_mixed`. `runners.test_w_botorch_dm` additionally includes: `batch_size`, `ucb_beta` (SO UCB only, else `null`), `known_noise` (bool), `mlhgp` (bool), `mlhgp_em_iter` (`null` unless `--mlhgp`). `runners.test_w_baselines_mo` includes `pop_size` for the evolutionary baselines.

Below, `T` = number of BO iterations, `n0` = initial random samples, `D` = input dims, `m` = number of objectives.

### Single-objective (HPO, DM SO)

| Key | Length | Description |
|---|---|---|
| `trial`, `seed`, `time_seconds` | scalar | Metadata |
| `best_y_all` | `T + 1` | Best noisy objective seen so far (index 0 = initial data) |
| `candidates` | `(n0 + T) × D` | All evaluated points in order |
| `seen_y` | `n0 + T` | All noisy observations in order |
| `rec_x_all` | `(T + 1) × D` | Posterior mean argmax; index 0 = initial fit before first BO step |
| `rec_true_all` | `T + 1` | Noiseless `f_true` at `rec_x` |
| `best_rec_true_all` | `T + 1` | Cumulative max of `rec_true_all` (running best recommendation) |
| `inference_regret_all` | `T + 1` | `optimal − rec_true` — raw per-step inference regret (non-monotonic) |
| `log_inference_regret_all` | `T + 1` | `log(max(optimal − rec_true, 1e-8))` |
| `best_inference_regret_all` | `T + 1` | `max(optimal − best_rec_true, 0)` — monotonically non-increasing |
| `log_best_inference_regret_all` | `T + 1` | `log(max(optimal − best_rec_true, 1e-8))` — monotonically non-increasing |
| `best_obs_x_all` | `T + 1` | Best-seen point (argmax of noisy `train_y`) at each step |
| `best_obs_true_all` | `T + 1` | Noiseless `f_true` at `best_obs_x`; the point is selected by noisy `train_y` |
| `best_obs_regret_all` | `T + 1` | `max(optimal − best_obs_true, 0)` — **not** guaranteed monotonic |
| `log_best_obs_regret_all` | `T + 1` | `log(max(optimal − best_obs_true, 1e-8))` — **not** guaranteed monotonic |
| `simple_regret_all` | `T + 1` | `max(f* − max_{i≤t} f_true(x_i), 0)` — noiseless simple regret over all observed inputs; monotonically non-increasing. **Present for all methods.** |
| `log_simple_regret_all` | `T + 1` | `log(max(f* − max_{i≤t} f_true(x_i), 1e-8))`. **Present for all methods.** |

The `rec_*` / `*inference_regret*` families are absent for the baseline runs (`runners.test_w_baselines_hpo`: `random`, `tpe`, `cmaes`, `bohb`, `asha`), which fit no surrogate. `--acq_fn random` in `runners.test_w_botorch_dm` *does* carry them — it fits a GP for the recommendation even though it samples uniformly.

### DM multi-objective

| Key | Length | Description |
|---|---|---|
| `trial`, `seed`, `time_seconds` | scalar | Metadata |
| `ref_point` | `m` | Fixed reference point for hypervolume computation |
| `hv_all` | `T + 1` | Dominated hypervolume of noisy observations at each step (index 0 = initial data) |
| `log_hv_diff_all` | `T + 1` | `log(max_hv − hv)` — log HV regret on noisy observations |
| `hv_true_all` | `T + 1` | Dominated hypervolume of noiseless `f_true` at all evaluated points (raw, non-monotonic) |
| `log_hv_diff_true_all` | `T + 1` | `log(max_hv − hv_true)` (non-monotonic) |
| `best_hv_true_all` | `T + 1` | Running max of `hv_true_all` — monotonically non-decreasing |
| `log_best_hv_diff_true_all` | `T + 1` | `log(max_hv − best_hv_true)` — monotonically non-increasing |
| `inf_hv_all` | `T + 1` | Raw inference HV: HV of noiseless `f_true` at the posterior-mean Pareto front. |
| `best_inf_hv_all` | `T + 1` | Running max of `inf_hv_all`. |
| `log_inference_hv_regret_all` | `T + 1` | `log(max_hv − inf_hv)` (non-monotonic). |
| `log_best_inference_hv_regret_all` | `T + 1` | `log(max_hv − best_inf_hv)` — monotonically non-increasing. |
| `pareto_x_best_hv` / `pareto_y_best_hv` | `P₁ × D` / `P₁ × m` | Non-dominated set (under `f_true`) at the iteration achieving `best_hv_true` |
| `pareto_x_best_inf_hv` / `pareto_y_best_inf_hv` | `P₂ × D` / `P₂ × m` | Posterior-mean Pareto set at the iteration achieving `best_inf_hv`. |
| `pareto_x_inf_hv` / `pareto_y_inf_hv` | `P₃ × D` / `P₃ × m` | Posterior-mean Pareto set at the final step. |
| `candidates` | `(n0 + T) × D` | All evaluated points in order |
| `seen_y` | `(n0 + T) × m` | All noisy observations in order |

### Prompt optimization

Top-level keys: `problem`, `acq_fn`, `turbo` (bool), `iterations`, `batch_size`, `ucb_beta`, `initial_random_samples`, `num_trials`, `trials`.

Here `T` = `iterations // batch_size` and total evaluations = `n0 + iterations`. Per-trial keys match the single-objective table above (`candidates` is `(n0 + iterations) × D`, `seen_y` is `n0 + iterations`), with one addition:

| Key | Length | Description |
|---|---|---|
| `turbo_length_all` | `T` | TuRBO trust-region length after each iteration. **Present only when `--turbo`.** |

### PCO

Top-level keys add `no_repeats`, `pf_delta`, `prior_mean`, `ls_floor` and `optimal_value` (the best feasible throughput in the table). Per-trial keys match the single-objective table above — regret counts feasible points only, since a run that ran out of memory realises no throughput — plus:

| Key | Length | Description |
|---|---|---|
| `seen_c` | `n0 + T` | Observed constraint value (memory margin; `<= 0` is feasible) |
| `n_infeasible_all` / `feasible_frac_all` | `T + 1` | Infeasible evaluations so far, and the feasible fraction |
| `scbo_ncand_all` | `T` | SCBO trust-region size in table rows. **Present only for `scbo`.** |
| `prior_fallback_iters` | — | Iterations where the `--prior_mean` fit failed and a constant mean was used. **Present only with `--prior_mean`.** |

---

## License

MIT — see [LICENSE](LICENSE).
