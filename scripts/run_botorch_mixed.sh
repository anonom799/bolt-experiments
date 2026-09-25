#!/usr/bin/env bash
# Run the GP-BO configurations for the HPO problems, as used in the paper.
#
# Usage:
#   bash scripts/run_botorch_mixed.sh                         # all problems
#   bash scripts/run_botorch_mixed.sh --problem hpo           # hpo only
#   bash scripts/run_botorch_mixed.sh --problem hpo_fd_step   # hpo_fd_step only
#   bash scripts/run_botorch_mixed.sh --problem hpo_fd_model  # hpo_fd_model only
#   bash scripts/run_botorch_mixed.sh --iterations 50 --trials 3
#   bash scripts/run_botorch_mixed.sh --initial_random_samples 20 --noise_std 0.01
#   bash scripts/run_botorch_mixed.sh --folder_prefix _v2     # writes to results/hpo_v2
#
# The non-BO baselines (random, TPE, CMA-ES, BOHB, ASHA) are in run_baselines.sh,
# the UCB beta sweep in run_ucb_betas.sh and the cost-scale sweep in
# run_cost_scales.sh.

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_mixed"

ITERATIONS=200
TRIALS=10
INIT_SAMPLES=10
NOISE_STD=""   # empty: let the problem class use its own default noise std
PROBLEM="all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)    PROBLEM="$2";    shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        --initial_random_samples) INIT_SAMPLES="$2"; shift 2 ;;
        --noise_std) NOISE_STD="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

NOISE_ARGS=()
if [[ -n "$NOISE_STD" ]]; then
    NOISE_ARGS=(--noise_std "$NOISE_STD")
fi
EXTRA_ARGS=(--initial_random_samples "$INIT_SAMPLES" ${NOISE_ARGS[@]+"${NOISE_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

HPO_ACQFNS=(ei qnei ucb mes gibbon pes jes ts)

# Multi-fidelity: each acquisition runs at its own fidelity-cost scale, the one
# with the lowest final normalised regret in the 50-iteration cost-scale sweep
# (results/hpo_fd_{step,model}_costsens, run_cost_scales.sh). "acq:cost_scale".
STEP_CONFIGS=(ucb:0.05 pes:0.005 qnei:0.005 mfgibbon:0.005 ei:0.01 mfmes:0.05)
MODEL_CONFIGS=(pes:0.005 ucb:0.01 qnei:0.01 ei:0.05 mfgibbon:0.005 mfmes:0.01)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_hpo() {
    for acq in "${HPO_ACQFNS[@]}"; do
        run "$SCRIPT" --problem hpo --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

run_mf() {
    local prob="$1"; shift
    for cfg in "$@"; do
        run "$SCRIPT" --problem "$prob" --acq_fn "${cfg%%:*}" --cost_scale "${cfg##*:}" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

case "$PROBLEM" in
    hpo)          run_hpo ;;
    hpo_fd_step)  run_mf hpo_fd_step "${STEP_CONFIGS[@]}" ;;
    hpo_fd_model) run_mf hpo_fd_model "${MODEL_CONFIGS[@]}" ;;
    all)
        run_hpo
        run_mf hpo_fd_step "${STEP_CONFIGS[@]}"
        run_mf hpo_fd_model "${MODEL_CONFIGS[@]}"
        ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: hpo, hpo_fd_step, hpo_fd_model, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All runs complete."
