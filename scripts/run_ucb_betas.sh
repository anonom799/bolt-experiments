#!/usr/bin/env bash
# Run UCB with fixed beta values for dm_curriculum (SO).
#
# Usage:
#   bash scripts/run_ucb_betas.sh
#   bash scripts/run_ucb_betas.sh --iterations 50 --trials 3

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_dm"

ITERATIONS=200
TRIALS=5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

BETAS=(0.1 0.5 1.0 2.0)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

for beta in "${BETAS[@]}"; do
    run "$SCRIPT" --problem dm_curriculum --acq_fn ucb --ucb_beta "$beta" \
        --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
done

echo ""
echo "All UCB beta runs complete."
