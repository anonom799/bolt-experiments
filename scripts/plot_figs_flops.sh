#!/bin/bash
# Re-plot the three main HPO figures of plot_figs.sh against estimated training
# FLOPs instead of BO iterations / abstract cost budget.
#
# The FLOPs formula and its justification are in hpo_flops.py. Nothing under
# results/ is modified: compute_hpo_flops.py writes FLOPs-costed copies into
# results_flops/hpo/ whose budget_all is cumulative FLOPs in units of 1e17, which
# plot_results.py --budget then plots unchanged.

set -e

METRIC=log_simple_regret_all
XLABEL='Compute (10^17 FLOPs)'
OUTDIR=pics/flops

mkdir -p "$OUTDIR"

# same file selections as plot_figs.sh
HPO_FILES=$(ls results/hpo/hpo_*_200iterations*.json | grep -v _fd_ | grep -v _beta1_ | grep -v '_beta[2,5,3]')
HPO_STEP_FILES=$(ls results/hpo/hpo_fd_step_*_200iterations*.json | grep -v _mfgibbon_5trials_200iterations_cost0.005_ | grep -v beta | grep -v '_ucb_5trials_200iterations_cost0.0[1,5]_')
HPO_MODEL_FILES=$(ls results/hpo/hpo_fd_model_*_200iterations*.json | grep -v beta)

# 1. the formula's self-check (data mixture anchor + per-config FLOPs table)
python -m bolt_exp.hpo_flops

# 2. per-query FLOPs tables + FLOPs-costed copies of the results
python -m bolt_exp.analysis.compute_hpo_flops $HPO_FILES $HPO_STEP_FILES $HPO_MODEL_FILES

# 3. the figures. Methods spend different compute per query and so run out at
#    different x; --xmax trims each panel to the budget every trial reaches, so
#    the whole panel is a like-for-like comparison.
to_costed() { for f in $@; do echo "results_flops/hpo/$(basename $f)"; done; }

python -m bolt_exp.plot_results $(to_costed $HPO_FILES) \
    --out ${OUTDIR}/hpo_flops.pdf --metric $METRIC \
    --config plot_configs/hpo_okabe_ito.yaml --no-title \
    --budget --xlabel "$XLABEL" \
    --xmax $(python -m bolt_exp.analysis.compute_hpo_flops --xmax_only $HPO_FILES)

python -m bolt_exp.plot_results $(to_costed $HPO_STEP_FILES) \
    --out ${OUTDIR}/hpo_step_flops.pdf --metric $METRIC \
    --config plot_configs/hpo_step_okabe_ito.yaml --no-title \
    --budget --xlabel "$XLABEL" \
    --xmax $(python -m bolt_exp.analysis.compute_hpo_flops --xmax_only $HPO_STEP_FILES)

python -m bolt_exp.plot_results $(to_costed $HPO_MODEL_FILES) \
    --out ${OUTDIR}/hpo_model_flops.pdf --metric $METRIC \
    --config plot_configs/hpo_model_okabe_ito.yaml --no-title \
    --budget --xlabel "$XLABEL" \
    --xmax $(python -m bolt_exp.analysis.compute_hpo_flops --xmax_only $HPO_MODEL_FILES)
