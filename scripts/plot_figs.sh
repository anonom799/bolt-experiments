#!/bin/bash

mkdir -p pics

# log_simple_regret_all  base metric
# log_best_inference_regret_all   based on posterior
# log_best_obs_regret_all   best obs with no oracle
# log_inference_regret_all   best posterior with no oracle

for pair in \
    "log_simple_regret_all log_best_hv_diff_true_all _simple" ; do
    # "log_best_inference_regret_all log_best_inference_hv_regret_all _inf" \
    # "log_best_obs_regret_all log_hv_diff_true_all _simple_inst" \
    # "log_inference_regret_all log_inference_hv_regret_all _inf_inst"; do    
    read -r METRIC METRIC_MO SUFFIX <<< "$pair"
    OUTDIR="pics/${SUFFIX#_}"
    mkdir -p "$OUTDIR"

    # hpo
    python -m bolt_exp.plot_results $(ls results/hpo/hpo_*_200iterations*.json | grep -v _fd_ | grep -v _beta1_ | grep -v '_beta[2,5,3]') --out ${OUTDIR}/hpo${SUFFIX}.pdf --metric $METRIC --config plot_configs/hpo_okabe_ito.yaml --no-title

    python -m bolt_exp.plot_results $(ls results/hpo/hpo_fd_step_*_200iterations*.json | grep -v _mfgibbon_5trials_200iterations_cost0.005_ | grep -v beta | grep -v '_ucb_5trials_200iterations_cost0.0[1,5]_') --out ${OUTDIR}/hpo_step${SUFFIX}.pdf --metric $METRIC --budget --config plot_configs/hpo_step_okabe_ito.yaml --no-title

    python -m bolt_exp.plot_results $(ls results/hpo/hpo_fd_model_*_200iterations*.json | grep -v beta) --out ${OUTDIR}/hpo_model${SUFFIX}.pdf --metric $METRIC --budget --config plot_configs/hpo_model_okabe_ito.yaml --no-title

    # dm
    python -m bolt_exp.plot_results $(ls results/dm/*200iterations*.json | grep -v _mo_ | grep -v 'ucb_beta[0,1,5]\.[0-9]' | grep -v 'ucb_beta[1,3]0.0' | grep -v hetero) --out ${OUTDIR}/dm${SUFFIX}.pdf --metric $METRIC --no-title --bo-iter --config plot_configs/dm_okabe_ito.yaml

    python -m bolt_exp.plot_results $(ls results/dm/*_mo_*_200iterations*.json | grep -v hetero) --out ${OUTDIR}/dm_mo${SUFFIX}.pdf --metric $METRIC_MO --no-title --bo-iter --config plot_configs/dm_mo_okabe_ito.yaml

    python -m bolt_exp.plot_results $(ls results/dm/dm_curriculum_hetero*_200iterations*.json | grep -v "_em[123]") --out ${OUTDIR}/dm_het${SUFFIX}.pdf --metric $METRIC --no-title --bo-iter --config plot_configs/dm_het_okabe_ito.yaml

    # po
    python -m bolt_exp.plot_results $(ls results/po/po128_*_200iterations*.json | grep -v mlesi) --out ${OUTDIR}/po128${SUFFIX}.pdf --metric $METRIC --no-title --bo-iter --config plot_configs/po128_okabe_ito.yaml

    python -m bolt_exp.plot_results $(ls results/po/po256_*_200iterations*.json | grep -v mlesi) --out ${OUTDIR}/po256${SUFFIX}.pdf --metric $METRIC --config plot_configs/po128_okabe_ito.yaml --no-title --bo-iter

    python -m bolt_exp.plot_results $(ls results/po/po512_*_200iterations*.json | grep -v mlesi) --out ${OUTDIR}/po512${SUFFIX}.pdf --metric $METRIC --config plot_configs/po128_okabe_ito.yaml --no-title --bo-iter

    python -m bolt_exp.plot_results $(ls results/po/po768_*_200iterations*.json | grep -v mlesi) --out ${OUTDIR}/po768${SUFFIX}.pdf --metric $METRIC --config plot_configs/po128_okabe_ito.yaml --no-title --bo-iter

    # hpo legend
    python -m bolt_exp.plot_results $(ls results/hpo/hpo_*_200iterations*.json | grep -v _fd_ | grep -v _beta1_ | grep -v '_beta[2,5,3]') --out ${OUTDIR}/hpo_long.pdf --metric $METRIC --config plot_configs/hpo_legend_okabe_ito.yaml --no-title
    rm ${OUTDIR}/hpo_long.pdf

    # dm legend
    python -m bolt_exp.plot_results $(ls results/dm/*200iterations*.json | grep -v _mo_ | grep -v 'ucb_beta[0,1,5]\.[0-9]' | grep -v 'ucb_beta[1,3]0.0' | grep -v hetero) --out ${OUTDIR}/dm_long.pdf --metric $METRIC --no-title --bo-iter --config plot_configs/dm_legend_okabe_ito.yaml
    rm ${OUTDIR}/dm_long.pdf

    # dm het legend
    python -m bolt_exp.plot_results $(ls results/dm/dm_curriculum_hetero*_200iterations*.json | grep -v "_em[123]") --out ${OUTDIR}/dm_het_long.pdf --metric $METRIC --no-title --bo-iter --config plot_configs/dm_het_legend_okabe_ito.yaml
    rm ${OUTDIR}/dm_het_long.pdf

    # po legend
    python -m bolt_exp.plot_results results/po/po128_*_200iterations*.json --out ${OUTDIR}/po128_long.pdf --metric $METRIC --bo-iter --config plot_configs/po128_legend_okabe_ito.yaml
    rm ${OUTDIR}/po128_long.pdf ${OUTDIR}/po*${SUFFIX}_legend.pdf

done


# po pca

python -m bolt_exp.analysis.analyse_po_pca  --save --no_show
mv po_pca_variance_standalone.pdf pics/.

# for app

# hpo mf cost scale

python -m bolt_exp.figures.plot_cost_scale_grid \
    $(ls results/hpo/hpo_fd_step_*_50iterations*cost*.json) \
    --config plot_configs/hpo_step_cost_scale_okabe_ito.yaml \
    --out pics/hpo_step_cost_simple_grid.pdf \
    --metric log_simple_regret_all --budget --cost-scale-label

python -m bolt_exp.figures.plot_cost_scale_grid  \
    $(ls results/hpo/hpo_fd_model_*_50iterations*cost*.json) \
    --config plot_configs/hpo_step_cost_scale_okabe_ito.yaml \
    --out pics/hpo_model_cost_simple_grid.pdf \
    --metric log_simple_regret_all --budget --cost-scale-label

# ucb

python -m bolt_exp.plot_results \
    $(ls results/hpo/hpo_ucb_*_200iterations*.json | grep -v _fd_) \
    --out pics/hpo_ucb_simple.pdf  \
    --metric log_simple_regret_all  \
    --config plot_configs/dm_ucb_okabe_ito.yaml \
    --no-title 

python -m bolt_exp.plot_results \
    $(ls results/dm/*_ucb_*_200iterations*.json | grep -v _mo_) \
    --out pics/dm_ucb_simple.pdf  \
    --metric log_simple_regret_all  \
    --config plot_configs/dm_ucb_okabe_ito.yaml \
    --no-title 

# mf fidelity obs

python -m bolt_exp.figures.plot_fidelity_proportions $(ls results/hpo/hpo_fd_step_*_200iterations*.json | grep -v _mfgibbon_5trials_200iterations_cost0.005_ | grep -v beta | grep -v '_ucb_5trials_200iterations_cost0.0[1,5]_') --config plot_configs/hpo_step_okabe_ito.yaml --out pics/fid_step.pdf

python -m bolt_exp.figures.plot_fidelity_proportions $(ls results/hpo/hpo_fd_model_*_200iterations*.json | grep -v beta) --config plot_configs/hpo_model_okabe_ito.yaml --out pics/fid_model.pdf

# # dm inference reg

# python -m bolt_exp.plot_results $(ls results/dm/*200iterations*.json | grep -v _mo_ | grep -v 'ucb_beta[0,1,5]\.[0-9]' | grep -v 'ucb_beta[1,3]0.0' | grep -v hetero) --out pics/dm_inf.pdf  --metric log_best_inference_regret_all  --bo-iter --config plot_configs/dm_okabe_ito.yaml --no-title 
 
# python -m bolt_exp.plot_results $(ls results/dm/*_mo_*_200iterations*.json |  grep -v hetero) --out pics/dm_mo_simple.pdf --metric log_best_inference_hv_regret_all --no-title --bo-iter --config plot_configs/dm_mo_okabe_ito.yaml

# python -m bolt_exp.plot_results $(ls results/dm/dm_curriculum_hetero*_200iterations*.json | grep -v "_em[123]") --out pics/dm_het_inf.pdf --metric log_best_inference_regret_all --no-title --bo-iter --config plot_configs/dm_het_okabe_ito.yaml



# python -m bolt_exp.plot_results $(ls results/dm/*200iterations*.json | grep -v _mo_ | grep -v 'ucb_beta[0,1,5]\.[0-9]' | grep -v 'ucb_beta[1,3]0.0' | grep -v hetero) --out pics/dm_simple_inst.pdf  --metric log_best_obs_regret_all  --bo-iter --config plot_configs/dm_okabe_ito.yaml --no-title 

# python -m bolt_exp.plot_results $(ls results/dm/*200iterations*.json | grep -v _mo_ | grep -v 'ucb_beta[0,1,5]\.[0-9]' | grep -v 'ucb_beta[1,3]0.0' | grep -v hetero) --out pics/dm_inf_inst.pdf  --metric log_inference_regret_all  --bo-iter --config plot_configs/dm_okabe_ito.yaml --no-title 
