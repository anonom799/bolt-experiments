#!/usr/bin/env bash
# Fidelity-cost sensitivity sweep for the multi-fidelity HPO problems: every
# acquisition at each cost scale, on a 50-iteration budget. Its final regrets
# pick the per-acquisition cost scale run_botorch_mixed.sh uses, and it is the
# data behind the cost-scale grid figure (plot_figs.sh hpo_mf_cost_scale).
#
# Usage:
#   bash scripts/run_cost_scales.sh                          # both problems
#   bash scripts/run_cost_scales.sh --problem hpo_fd_step
#   bash scripts/run_cost_scales.sh --iterations 20 --trials 3
#
# Writes to results/<problem>_costsens/.

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_mixed"

ITERATIONS=50
TRIALS=10
INIT_SAMPLES=10
PROBLEM="all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)    PROBLEM="$2";    shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        --initial_random_samples) INIT_SAMPLES="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

ACQFNS=(ei ucb pes qnei mfmes mfgibbon)
COST_SCALES=(0.005 0.01 0.05)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_problem() {
    local prob="$1"
    for acq in "${ACQFNS[@]}"; do
        for cost in "${COST_SCALES[@]}"; do
            run "$SCRIPT" --problem "$prob" --acq_fn "$acq" --cost_scale "$cost" \
                --iterations "$ITERATIONS" --trials "$TRIALS" \
                --initial_random_samples "$INIT_SAMPLES" \
                --results_folder "${prob}_costsens" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
        done
    done
}

case "$PROBLEM" in
    hpo_fd_step)  run_problem hpo_fd_step ;;
    hpo_fd_model) run_problem hpo_fd_model ;;
    all)          run_problem hpo_fd_step; run_problem hpo_fd_model ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: hpo_fd_step, hpo_fd_model, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All cost-scale runs complete."
