#!/usr/bin/env bash
# Run all acqfn/problem configurations for HPO mixed search space problems.
#
# Usage:
#   bash scripts/run_botorch_mixed.sh                          # all problems
#   bash scripts/run_botorch_mixed.sh --problem hpo            # HPO only
#   bash scripts/run_botorch_mixed.sh --problem hpo_fd_step    # multi-fidelity step only
#   bash scripts/run_botorch_mixed.sh --problem hpo_fd_model   # multi-fidelity model only
#   bash scripts/run_botorch_mixed.sh --iterations 50 --trials 1

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_mixed"

ITERATIONS=100
TRIALS=5
PROBLEM="all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)    PROBLEM="$2";    shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HPO_ACQFNS=(ei qnei ucb mes gibbon pes kg)
MF_ACQFNS=(ei ucb pes mfkg mfmes mfgibbon)

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

run_hpo_fd_step() {
    for acq in "${MF_ACQFNS[@]}"; do
        run "$SCRIPT" --problem hpo_fd_step --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

run_hpo_fd_model() {
    for acq in "${MF_ACQFNS[@]}"; do
        run "$SCRIPT" --problem hpo_fd_model --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

case "$PROBLEM" in
    hpo)          run_hpo ;;
    hpo_fd_step)  run_hpo_fd_step ;;
    hpo_fd_model) run_hpo_fd_model ;;
    all)          run_hpo; run_hpo_fd_step; run_hpo_fd_model ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: hpo, hpo_fd_step, hpo_fd_model, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All runs complete."
