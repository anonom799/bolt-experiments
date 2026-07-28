#!/usr/bin/env bash
# Run all acqfn/problem configurations for PO (prompt optimization) problems.
#
# Usage:
#   bash scripts/run_botorch_po.sh                          # all problems
#   bash scripts/run_botorch_po.sh --problem po128          # po128 only
#   bash scripts/run_botorch_po.sh --problem po256          # po256 only
#   bash scripts/run_botorch_po.sh --problem po512          # po512 only
#   bash scripts/run_botorch_po.sh --problem po768          # po768 only
#   bash scripts/run_botorch_po.sh --iterations 50 --trials 3

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_po"

ITERATIONS=200
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

ACQFNS=(ei qnei ucb mes gibbon pes jes kg ts random)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_problem() {
    local prob="$1"
    for acq in "${ACQFNS[@]}"; do
        run "$SCRIPT" --problem "$prob" --acq_fn "$acq" \
            --iterations "$ITERATIONS" --trials "$TRIALS" "${EXTRA_ARGS[@]}"
    done
}

case "$PROBLEM" in
    po128)  run_problem po128 ;;
    po256)  run_problem po256 ;;
    po512)  run_problem po512 ;;
    po768)  run_problem po768 ;;
    all)
        run_problem po128
        run_problem po256
        run_problem po512
        run_problem po768
        ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: po128, po256, po512, po768, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All runs complete."
