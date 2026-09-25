#!/bin/bash

# Run from the repository root.
#
# usage: bash scripts/plot_figs.sh [targets] [folder_prefix]
#   targets        comma-separated functions below to run (default: every family),
#                  either a whole family (hpo, hpo_mf, dm, po, pco, emulator,
#                  mlhgp) or one subset (e.g. hpo_flops, dm_inf_inst, pco_legend)
#                  e.g. bash scripts/plot_figs.sh emulator    bash scripts/plot_figs.sh dm,pco_inf_inst
#   folder_prefix  defaults to "pics"; output goes under ${PICS}/<figure family>/
TARGETS="${1:-hpo,hpo_mf,dm,po,pco,emulator,mlhgp}"
PICS="${2:-pics}"

# One directory per figure family; filenames are unchanged, so a figure keeps a
# unique name if the tree is ever flattened again.
HPO_DIR="$PICS/hpo"        # single-fidelity HPO
HPO_MF_DIR="$PICS/hpo_mf"  # multi-fidelity HPO (step / model fidelity)
DM_DIR="$PICS/dm"          # data mixture
PO_DIR="$PICS/po"          # prompt optimization
PCO_DIR="$PICS/pco"        # parallelism configuration
EMU_DIR="$PICS/emulator"   # emulator validation (real vs emulated)
LAND_DIR="$PICS/landscape" # emulator landscape slices
MLHGP_DIR="$PICS/mlhgp"    # MLHGP noise model learning

# metrics
# log_simple_regret_all  base metric
# log_best_inference_regret_all   based on posterior
# log_best_obs_regret_all   best obs with no oracle
# log_inference_regret_all   best posterior with no oracle
SIMPLE=log_simple_regret_all
SIMPLE_MO=log_best_hv_diff_true_all
INF_INST=log_inference_regret_all
SIMPLE_INST=log_best_obs_regret_all

# ------------------------------------------------------------------
# result file sets
# ------------------------------------------------------------------

hpo_files() { ls results/hpo/hpo_*_10trials_200iterations*.json | grep -v _fd_ | grep -v _beta0.5_ | grep -v _beta1_ | grep -v _beta5_ | grep -v _beta10_ | grep -v _beta30_; }
hpo_step_files() { ls results/hpo/hpo_fd_step_*_10trials_200iterations*.json | grep -v _mfgibbon_5trials_200iterations_cost0.005_ | grep -v beta | grep -v '_ucb_5trials_200iterations_cost0.0[1,5]_'; }
hpo_model_files() { ls results/hpo/hpo_fd_model_*_10trials_200iterations*.json | grep -v beta; }
dm_files() { ls results/dm/*_10trials_200iterations*.json | grep -v _mo_ | grep -v 'ucb_beta[0,1,5]\.[0-9]' | grep -v 'ucb_beta[1,3]0.0' | grep -v hetero | grep -v '_q[0-9]*_'; }
dm_mo_files() { ls results/dm/*_mo_*_10trials_200iterations*.json | grep -v hetero | grep -v '_q[0-9]*_'; }
dm_het_files() { ls results/dm/dm_curriculum_hetero*_10trials_200iterations*.json | grep -v "_em[123]"; }
po_files() { ls results/po/$1_*_200iterations*.json* | grep -v mlesi; }

# ------------------------------------------------------------------
# hpo
# ------------------------------------------------------------------

hpo_simple() {
    python -m bolt_exp.plot_results $(hpo_files) --out ${HPO_DIR}/hpo_simple.pdf --metric $SIMPLE --config plot_configs/hpo_okabe_ito.yaml --no-title --mean-rank
}

# hpo on a training-FLOPs x axis (same curves, compute instead of query count).
# --xmax (from compute_hpo_flops) is the largest budget every trial reaches, so no curve is drawn
# past the point where it is backed by all the trials.
hpo_flops() {
    python -m bolt_exp.plot_results $(hpo_files) --out ${HPO_DIR}/hpo_simple_flops.pdf --figsize 5 4.5 --metric $SIMPLE --flops --xmax $(python -m bolt_exp.analysis.compute_hpo_flops $(hpo_files)) --config plot_configs/hpo_okabe_ito.yaml --no-title --mean-rank
}

hpo_inf_inst() {
    python -m bolt_exp.plot_results $(hpo_files) --out ${HPO_DIR}/hpo_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --config plot_configs/hpo_okabe_ito.yaml --no-title --mean-rank
}

hpo_simple_inst() {
    python -m bolt_exp.plot_results $(hpo_files) --out ${HPO_DIR}/hpo_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --config plot_configs/hpo_okabe_ito.yaml --no-title --mean-rank
}

hpo_legend() {
    python -m bolt_exp.plot_results $(ls results/hpo/hpo_*_10trials_200iterations*.json | grep -v _fd_ | grep -v '_beta[0,1,5,3]') --out ${HPO_DIR}/hpo_long.pdf --metric $SIMPLE --config plot_configs/hpo_legend_okabe_ito.yaml --no-title
    rm ${HPO_DIR}/hpo_long.pdf
}

hpo_ucb() {
    python -m bolt_exp.plot_results \
        $(ls results/hpo/hpo_ucb_10trials_200iterations*.json | grep -v _fd_) \
        --out ${HPO_DIR}/hpo_ucb_simple.pdf  \
        --metric $SIMPLE  \
        --config plot_configs/dm_ucb_okabe_ito.yaml \
        --no-title --mean-rank
}

hpo() { hpo_simple; hpo_flops; hpo_inf_inst; hpo_simple_inst; hpo_legend; hpo_ucb; }

# ------------------------------------------------------------------
# hpo multi-fidelity
# ------------------------------------------------------------------

hpo_mf_simple() {
    python -m bolt_exp.plot_results $(hpo_step_files) --out ${HPO_MF_DIR}/hpo_step_simple.pdf --metric $SIMPLE --budget --config plot_configs/hpo_step_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results $(hpo_model_files) --out ${HPO_MF_DIR}/hpo_model_simple.pdf --metric $SIMPLE --budget --config plot_configs/hpo_model_okabe_ito.yaml --no-title --mean-rank
}

hpo_mf_flops() {
    python -m bolt_exp.plot_results $(hpo_step_files) --out ${HPO_MF_DIR}/hpo_step_simple_flops.pdf --figsize 5 4.5 --metric $SIMPLE --flops --xmax $(python -m bolt_exp.analysis.compute_hpo_flops $(hpo_step_files)) --config plot_configs/hpo_step_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results $(hpo_model_files) --out ${HPO_MF_DIR}/hpo_model_simple_flops.pdf --figsize 5 4.5 --metric $SIMPLE --flops --xmax $(python -m bolt_exp.analysis.compute_hpo_flops $(hpo_model_files)) --config plot_configs/hpo_model_okabe_ito.yaml --no-title --mean-rank
}

hpo_mf_inf_inst() {
    python -m bolt_exp.plot_results $(hpo_step_files) --out ${HPO_MF_DIR}/hpo_step_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --budget --config plot_configs/hpo_step_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results $(hpo_model_files) --out ${HPO_MF_DIR}/hpo_model_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --budget --config plot_configs/hpo_model_okabe_ito.yaml --no-title --mean-rank
}

hpo_mf_simple_inst() {
    python -m bolt_exp.plot_results $(hpo_step_files) --out ${HPO_MF_DIR}/hpo_step_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --budget --config plot_configs/hpo_step_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results $(hpo_model_files) --out ${HPO_MF_DIR}/hpo_model_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --budget --config plot_configs/hpo_model_okabe_ito.yaml --no-title --mean-rank
}

hpo_mf_legend() {
    python -m bolt_exp.plot_results $(hpo_step_files) --out ${HPO_MF_DIR}/hpo_step_long.pdf --metric $SIMPLE --budget --config plot_configs/hpo_step_legend_okabe_ito.yaml --no-title
    rm ${HPO_MF_DIR}/hpo_step_long.pdf

    python -m bolt_exp.plot_results $(hpo_model_files) --out ${HPO_MF_DIR}/hpo_model_long.pdf --metric $SIMPLE --budget --config plot_configs/hpo_model_legend_okabe_ito.yaml --no-title
    rm ${HPO_MF_DIR}/hpo_model_long.pdf
}

hpo_mf_cost_scale() {
    python -m bolt_exp.figures.plot_cost_scale_grid \
        $(ls results/hpo_fd_step_costsens/hpo_fd_step_*_10trials_50iterations*cost*.json) \
        --config plot_configs/hpo_step_cost_scale_okabe_ito.yaml \
        --out ${HPO_MF_DIR}/hpo_step_cost_simple_grid.pdf \
        --metric $SIMPLE --budget --cost-scale-label

    python -m bolt_exp.figures.plot_cost_scale_grid  \
        $(ls results/hpo_fd_model_costsens/hpo_fd_model_*_10trials_50iterations*cost*.json) \
        --config plot_configs/hpo_step_cost_scale_okabe_ito.yaml \
        --out ${HPO_MF_DIR}/hpo_model_cost_simple_grid.pdf \
        --metric $SIMPLE --budget --cost-scale-label
}

hpo_mf_fidelity() {
    python -m bolt_exp.figures.plot_fidelity_proportions $(hpo_step_files) --config plot_configs/hpo_step_okabe_ito.yaml --out ${HPO_MF_DIR}/fid_step.pdf

    python -m bolt_exp.figures.plot_fidelity_proportions $(hpo_model_files) --config plot_configs/hpo_model_okabe_ito.yaml --out ${HPO_MF_DIR}/fid_model.pdf
}

hpo_mf() { hpo_mf_simple; hpo_mf_flops; hpo_mf_inf_inst; hpo_mf_simple_inst; hpo_mf_legend; hpo_mf_cost_scale; hpo_mf_fidelity; }

# ------------------------------------------------------------------
# dm
# ------------------------------------------------------------------

dm_simple() {
    python -m bolt_exp.plot_results $(dm_files) --out ${DM_DIR}/dm_simple.pdf --metric $SIMPLE --no-title --bo-iter --config plot_configs/dm_okabe_ito.yaml --mean-rank

    python -m bolt_exp.plot_results $(dm_mo_files) --out ${DM_DIR}/dm_mo_simple.pdf --metric $SIMPLE_MO --no-title --bo-iter --config plot_configs/dm_mo_okabe_ito.yaml --mean-rank

    python -m bolt_exp.plot_results $(dm_het_files) --out ${DM_DIR}/dm_het_simple.pdf --metric $SIMPLE --no-title --bo-iter --config plot_configs/dm_het_okabe_ito.yaml --mean-rank
}

dm_inf_inst() {
    python -m bolt_exp.plot_results $(dm_files) --out ${DM_DIR}/dm_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --no-title --bo-iter --config plot_configs/dm_okabe_ito.yaml --mean-rank

    python -m bolt_exp.plot_results $(dm_het_files) --out ${DM_DIR}/dm_het_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --no-title --bo-iter --config plot_configs/dm_het_okabe_ito.yaml --mean-rank
}

dm_simple_inst() {
    python -m bolt_exp.plot_results $(dm_files) --out ${DM_DIR}/dm_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --no-title --bo-iter --config plot_configs/dm_okabe_ito.yaml --mean-rank

    python -m bolt_exp.plot_results $(dm_het_files) --out ${DM_DIR}/dm_het_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --no-title --bo-iter --config plot_configs/dm_het_okabe_ito.yaml --mean-rank
}

# batch size grids. The SO y-limits are the mean +/- 95% CI spans over the 12
# files (data [-7.77, -4.47]), measured over the VISIBLE region only: x is step*q
# (plot_results.load_results), so --xmax 200 cuts the q=5 curves at step 40 and
# the q=10 curves at step 20, and the tails beyond that are never drawn.
dm_batch() {
    # dm batch size (q=1 vs q=5 vs q=10)
    python -m bolt_exp.plot_results \
        $(ls results/dm/dm_curriculum_{random,qnei,mes,gibbon}_10trials_200iterations_results.json) \
        $(ls results/dm/dm_curriculum_{random,qnei,mes,gibbon}_q5_10trials_200iterations_results.json) \
        $(ls results/dm/dm_curriculum_{random,qnei,mes,gibbon}_q10_10trials_200iterations_results.json) \
        --out ${DM_DIR}/dm_batch_simple.pdf --metric $SIMPLE --no-title --xmax 200 --config plot_configs/dm_batch_okabe_ito.yaml --facet-by-base --mean-rank --rank-within-q --ymin -7.9 --ymax -4.3

    # dm mo batch size (q=1 vs q=5)
    python -m bolt_exp.plot_results \
        $(ls results/dm/dm_curriculum_mo_{random,qparego,qnehvi,mes_mo}_10trials_200iterations_results.json) \
        $(ls results/dm/dm_curriculum_mo_{random,qparego,qnehvi,mes_mo}_q5_10trials_200iterations_results.json) \
        --out ${DM_DIR}/dm_mo_batch_simple.pdf --metric $SIMPLE_MO --no-title --xmax 200 --config plot_configs/dm_mo_batch_okabe_ito.yaml --facet-by-base --mean-rank --rank-within-q
}

dm_legend() {
    python -m bolt_exp.plot_results $(dm_files) --out ${DM_DIR}/dm_long.pdf --metric $SIMPLE --no-title --bo-iter --config plot_configs/dm_legend_okabe_ito.yaml
    rm ${DM_DIR}/dm_long.pdf

    python -m bolt_exp.plot_results $(dm_het_files) --out ${DM_DIR}/dm_het_long.pdf --metric $SIMPLE --no-title --bo-iter --config plot_configs/dm_het_legend_okabe_ito.yaml
    rm ${DM_DIR}/dm_het_long.pdf
}

dm_ucb() {
    python -m bolt_exp.plot_results \
        $(ls results/dm/*_ucb*_10trials_200iterations*.json | grep -v _mo_ | grep -v _heteroscedastic_) \
        --out ${DM_DIR}/dm_ucb_simple.pdf  \
        --metric $SIMPLE  \
        --config plot_configs/dm_ucb_okabe_ito.yaml \
        --no-title --mean-rank
}

dm() { dm_simple; dm_inf_inst; dm_simple_inst; dm_batch; dm_legend; dm_ucb; }

# ------------------------------------------------------------------
# po -- every instance shares po128_legend's legend, so the per-plot legends
# are removed
# ------------------------------------------------------------------

po_simple() {
    for n in 128 256 512 768; do
        python -m bolt_exp.plot_results $(po_files po$n) --out ${PO_DIR}/po${n}_simple.pdf --metric $SIMPLE --config plot_configs/po128_okabe_ito.yaml --no-title --bo-iter --mean-rank
    done
    rm ${PO_DIR}/po*_simple_legend.pdf
}

po_inf_inst() {
    for n in 128 256 512 768; do
        python -m bolt_exp.plot_results $(po_files po$n) --out ${PO_DIR}/po${n}_inf_inst.pdf --figsize 3.75 4 --metric $INF_INST --config plot_configs/po128_okabe_ito.yaml --no-title --bo-iter --mean-rank
    done
    rm ${PO_DIR}/po*_inf_inst_legend.pdf
}

po_simple_inst() {
    for n in 128 256 512 768; do
        python -m bolt_exp.plot_results $(po_files po$n) --out ${PO_DIR}/po${n}_simple_inst.pdf --figsize 3.75 4 --metric $SIMPLE_INST --config plot_configs/po128_okabe_ito.yaml --no-title --bo-iter --mean-rank
    done
    rm ${PO_DIR}/po*_simple_inst_legend.pdf
}

po_legend() {
    python -m bolt_exp.plot_results results/po/po128_*_200iterations*.json* --out ${PO_DIR}/po128_long.pdf --metric $SIMPLE --bo-iter --config plot_configs/po128_legend_okabe_ito.yaml
    rm ${PO_DIR}/po128_long.pdf
}

po_pca() {
    python -m bolt_exp.analysis.analyse_po_pca --save --no_show --out_dir "$PO_DIR"
}

po() { po_simple; po_inf_inst; po_simple_inst; po_legend; po_pca; }

# ------------------------------------------------------------------
# pco -- constrained, one observation per iteration, so the x axis is
# observations as for hpo rather than BO iterations. The instances carry
# different budgets (100 iterations on pco16 and pco64, 50 on pco32), so they are
# plotted separately rather than faceted.
# ------------------------------------------------------------------

pco_simple() {
    python -m bolt_exp.plot_results results/pco16/pco16_*_10trials_100iterations*.json --out ${PCO_DIR}/pco16_simple.pdf --metric $SIMPLE --config plot_configs/pco16_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results results/pco32/pco32_*_10trials_50iterations*.json --out ${PCO_DIR}/pco32_simple.pdf --metric $SIMPLE --config plot_configs/pco32_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results results/pco64/pco64_*_10trials_100iterations*.json --out ${PCO_DIR}/pco64_simple.pdf --metric $SIMPLE --config plot_configs/pco64_okabe_ito.yaml --no-title --mean-rank
}

pco_inf_inst() {
    python -m bolt_exp.plot_results results/pco16/pco16_*_10trials_100iterations*.json --out ${PCO_DIR}/pco16_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --config plot_configs/pco16_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results results/pco32/pco32_*_10trials_50iterations*.json --out ${PCO_DIR}/pco32_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --config plot_configs/pco32_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results results/pco64/pco64_*_10trials_100iterations*.json --out ${PCO_DIR}/pco64_inf_inst.pdf --figsize 5 4.5 --metric $INF_INST --config plot_configs/pco64_okabe_ito.yaml --no-title --mean-rank
}

pco_simple_inst() {
    python -m bolt_exp.plot_results results/pco16/pco16_*_10trials_100iterations*.json --out ${PCO_DIR}/pco16_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --config plot_configs/pco16_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results results/pco32/pco32_*_10trials_50iterations*.json --out ${PCO_DIR}/pco32_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --config plot_configs/pco32_okabe_ito.yaml --no-title --mean-rank

    python -m bolt_exp.plot_results results/pco64/pco64_*_10trials_100iterations*.json --out ${PCO_DIR}/pco64_simple_inst.pdf --figsize 5 4.5 --metric $SIMPLE_INST --config plot_configs/pco64_okabe_ito.yaml --no-title --mean-rank
}

# pco legend -- one row of all nine methods; the config is pco32's with a
# wider figure, which is what pushes the legend into a single row.
pco_legend() {
    python -m bolt_exp.plot_results results/pco32/pco32_*_10trials_50iterations*.json --out ${PCO_DIR}/pco_long.pdf --metric $SIMPLE --config plot_configs/pco_legend_okabe_ito.yaml --no-title
    rm ${PCO_DIR}/pco_long.pdf
}

# pco plateau -- reads the candidate tables in data/pco{16,32,64}/data.parquet
pco_plateau() {
    python -m bolt_exp.figures.plot_pco_plateau --out "$PCO_DIR"/pco_plateau.pdf
}

# pco outcomes -- success curves, success/rank heatmaps and final-outcome bars
# (plus its legend); ranks come from the same candidate tables.
pco_outcomes() {
    python -m bolt_exp.figures.plot_pco_outcomes results/pco16/pco16_*_10trials_100iterations*.json --data data/pco16/data.parquet --config plot_configs/pco16_okabe_ito.yaml --out "$PCO_DIR"/pco16
    python -m bolt_exp.figures.plot_pco_outcomes results/pco32/pco32_*_10trials_50iterations*.json --data data/pco32/data.parquet --config plot_configs/pco32_okabe_ito.yaml --out "$PCO_DIR"/pco32
    python -m bolt_exp.figures.plot_pco_outcomes results/pco64/pco64_*_10trials_100iterations*.json --data data/pco64/data.parquet --config plot_configs/pco64_okabe_ito.yaml --out "$PCO_DIR"/pco64
}

pco() { pco_simple; pco_inf_inst; pco_simple_inst; pco_legend; pco_plateau; pco_outcomes; }

# ------------------------------------------------------------------
# emulator
# ------------------------------------------------------------------

# real vs emulated comparison
# --combined also writes <out>_grid.pdf (regret + rank panels in one figure),
# which is the version the paper includes
emulator_real_vs_emulated() {
    python -m bolt_exp.figures.plot_real_vs_emulated \
        --methods-config plot_configs/dm_real_vs_emulated.yaml \
        --style-config plot_configs/dm_okabe_ito.yaml \
        --out ${EMU_DIR}/dm_real_vs_emulated_normalized_regret.pdf \
        --shared-axis --no-title --combined

    python -m bolt_exp.figures.plot_real_vs_emulated \
        --methods-config plot_configs/hpo_real_vs_emulated.yaml \
        --style-config plot_configs/hpo_okabe_ito.yaml \
        --out ${EMU_DIR}/hpo_real_vs_emulated_normalized_regret.pdf \
        --shared-axis --no-title --combined
}

# emulator R^2 / spearman on the points the real runs visited (md + LaTeX).
# needs bolt with the v0.2.0 hf_revision pins installed; writes tables/real_vs_emulated_fit.md
emulator_fit_table() {
    python -m bolt_exp.analysis.table_real_vs_emulated_fit
}

# both families' spearman curves stacked in one figure (HPO on top, DMO below)
emulator_spearman() {
    python -m bolt_exp.figures.plot_spearman_grid \
        --panel plot_configs/hpo_real_vs_emulated.yaml:plot_configs/hpo_okabe_ito.yaml:HPO \
        --panel plot_configs/dm_real_vs_emulated.yaml:plot_configs/dm_okabe_ito.yaml:DMO \
        --row-height 0.5 \
        --out ${EMU_DIR}/real_vs_emulated_normalized_regret_rank.pdf
}

# emulator landscape slices + validation figures
# --paper_landscape_only keeps the landscape slices to the four panels the paper
# includes (mixtures pinned in the script rather than re-picked from the val
# set). Those and the emulator's DM Pareto front go to LAND_DIR; the rank-rank
# scatter against the held-out val points, the multi-fidelity step error /
# 4B-vs-8B histogram and the noise model diagnostics go to EMU_DIR.
#
# The Pareto front panel reads data/data_mixture/out_qwen4b/pareto_front.csv,
# the DM emulator's front precomputed on a dense grid of mixtures.
emulator_landscape() {
    python -m bolt_exp.figures.plot_emulators --paper_landscape_only --out_dir ${LAND_DIR} --validation_dir ${EMU_DIR}
}

emulator() { emulator_real_vs_emulated; emulator_fit_table; emulator_spearman; emulator_landscape; }

# ------------------------------------------------------------------
# mlhgp noise model learning over the course of BO
# ------------------------------------------------------------------

mlhgp() {
    python -m bolt_exp.figures.plot_mlhgp_noise_learning \
        --curve data/mlhgp/mlhgp_noise_vs_bo_iter.json \
        --cache data/mlhgp/mlhgp_noise_learning_cache.npz \
        --out ${MLHGP_DIR}/mlhgp_noise_learning.png
}

# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------

for t in ${TARGETS//,/ }; do
    if ! declare -F "$t" > /dev/null; then
        echo "unknown target '$t'" >&2
        exit 1
    fi
done

mkdir -p "$HPO_DIR" "$HPO_MF_DIR" "$DM_DIR" "$PO_DIR" "$PCO_DIR" "$EMU_DIR" "$LAND_DIR" "$MLHGP_DIR"

for t in ${TARGETS//,/ }; do
    "$t"
done
