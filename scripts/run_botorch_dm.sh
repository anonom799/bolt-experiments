#!/usr/bin/env bash
# Run all acqfn/model configurations for DM curriculum problems.
#
# Usage:
#   bash scripts/run_botorch_dm.sh                          # all problems
#   bash scripts/run_botorch_dm.sh --problem so             # dm_curriculum only
#   bash scripts/run_botorch_dm.sh --problem mo             # dm_curriculum_mo only
#   bash scripts/run_botorch_dm.sh --problem hetero         # dm_curriculum_heteroscedastic only
#   bash scripts/run_botorch_dm.sh --iterations 50 --trials 1

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_dm"

ITERATIONS=200
TRIALS=5
PROBLEM="all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)   PROBLEM="$2";   shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --trials)    TRIALS="$2";    shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

SO_ACQFNS=(ei qnei ucb mes gibbon pes jes kg ts random)
MO_ACQFNS=(qnehvi qparego qhvkg jes_mo mes_mo pes_mo random)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_so() {
    for acq in "${SO_ACQFNS[@]}"; do
        run "$SCRIPT" --problem dm_curriculum --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

run_mo() {
    for acq in "${MO_ACQFNS[@]}"; do
        run "$SCRIPT" --problem dm_curriculum_mo --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

run_hetero_so() {
    for acq in "${SO_ACQFNS[@]}"; do
        run "$SCRIPT" --problem dm_curriculum_heteroscedastic --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}


case "$PROBLEM" in
    so)        run_so ;;
    mo)        run_mo ;;
    hetero)    run_hetero_so ;;
    all)       run_so; run_mo; run_hetero_so ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: so, mo, hetero, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All runs complete."
