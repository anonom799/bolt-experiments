#!/usr/bin/env bash
# Run the constrained-BO sweep for the PCO (parallelism configuration) problems,
# as used in the paper.
#
# Usage:
#   bash scripts/run_botorch_pco.sh                       # all instances
#   bash scripts/run_botorch_pco.sh --problem pco32
#   bash scripts/run_botorch_pco.sh --trials 3
#
# The instances get different budgets: pco32 is the smallest table (379 rows)
# and converges sooner, so it runs 50 iterations against 100 on pco16 (919
# rows) and pco64 (780 rows). On top of the main sweep, the model-based
# methods are rerun with the physics-informed prior mean (--prior_mean, files
# tagged _prior).
#
# A PCO evaluation is a table lookup -- no GPU is used -- so runs are serial on
# 8 BLAS threads each.

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_pco"

TRIALS=10
INIT_SAMPLES=10
NOISE_STD=0.001
PROBLEM="all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)    PROBLEM="$2";    shift 2 ;;
        --trials)     TRIALS="$2";     shift 2 ;;
        --initial_random_samples) INIT_SAMPLES="$2"; shift 2 ;;
        --noise_std)  NOISE_STD="$2";  shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ACQFNS=(random ei qnei ucb_c cts scbo cmes_ibo ckg admmbo tpe cmaes)
PRIOR_ACQFNS=(qnei ucb_c ckg admmbo)

iterations_for() {
    case "$1" in
        pco32) echo 50 ;;
        *)     echo 100 ;;
    esac
}

# SCBO reads the objective GP's ARD lengthscales to shape its trust region, and
# on this lattice a lengthscale below one level is unidentifiable -- left
# unbounded it collapses a dimension out of the region and costs a seed badly.
# No other acquisition reads the lengthscales directly.
acq_extra_args() {
    case "$1" in
        scbo) echo "--ls_floor" ;;
        *)    echo "" ;;
    esac
}

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_pco() {
    local prob="$1" acq="$2"; shift 2
    local extra; read -r -a extra <<< "$(acq_extra_args "$acq")"
    run "$SCRIPT" --problem "$prob" --acq_fn "$acq" \
        --iterations "$(iterations_for "$prob")" --trials "$TRIALS" \
        --initial_random_samples "$INIT_SAMPLES" --noise_std "$NOISE_STD" \
        ${extra[@]+"${extra[@]}"} "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

run_problem() {
    local prob="$1"
    for acq in "${ACQFNS[@]}"; do
        run_pco "$prob" "$acq"
    done
    for acq in "${PRIOR_ACQFNS[@]}"; do
        run_pco "$prob" "$acq" --prior_mean
    done
}

case "$PROBLEM" in
    pco16|pco32|pco64) run_problem "$PROBLEM" ;;
    all)
        run_problem pco16
        run_problem pco32
        run_problem pco64
        ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: pco16, pco32, pco64, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All runs complete."
