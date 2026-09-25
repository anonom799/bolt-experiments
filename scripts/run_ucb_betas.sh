#!/usr/bin/env bash
# Run UCB with fixed beta values on hpo and dm_curriculum (SO).
#
# Usage:
#   bash scripts/run_ucb_betas.sh                        # both problems
#   bash scripts/run_ucb_betas.sh --problem hpo
#   bash scripts/run_ucb_betas.sh --problem dm
#   bash scripts/run_ucb_betas.sh --iterations 50 --trials 3

set -euo pipefail

ITERATIONS=200
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

BETAS=(0.5 1.0 2.0 5.0 10.0 30.0)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_betas() {
    local script="$1" prob="$2"
    for beta in "${BETAS[@]}"; do
        run "$script" --problem "$prob" --acq_fn ucb --ucb_beta "$beta" \
            --iterations "$ITERATIONS" --trials "$TRIALS" \
            --initial_random_samples "$INIT_SAMPLES" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
}

case "$PROBLEM" in
    hpo) run_betas bolt_exp.runners.test_w_botorch_mixed hpo ;;
    dm)  run_betas bolt_exp.runners.test_w_botorch_dm dm_curriculum ;;
    all)
        run_betas bolt_exp.runners.test_w_botorch_mixed hpo
        run_betas bolt_exp.runners.test_w_botorch_dm dm_curriculum
        ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: hpo, dm, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All UCB beta runs complete."
