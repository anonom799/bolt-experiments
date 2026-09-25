#!/usr/bin/env bash
# Run the GP-BO configurations for the DM curriculum problems, as used in the paper.
#
# Usage:
#   bash scripts/run_botorch_dm.sh                          # all problems
#   bash scripts/run_botorch_dm.sh --problem so             # dm_curriculum only
#   bash scripts/run_botorch_dm.sh --problem mo             # dm_curriculum_mo only
#   bash scripts/run_botorch_dm.sh --problem hetero         # dm_curriculum_heteroscedastic only
#   bash scripts/run_botorch_dm.sh --iterations 50 --trials 1
#   bash scripts/run_botorch_dm.sh --initial_random_samples 20 --noise_std 0.01
#   bash scripts/run_botorch_dm.sh --folder_prefix _v2      # writes to results/dm_v2
#
# The MO baselines (TSEMO, NSGA-II) are in run_baselines.sh and the UCB beta
# sweep in run_ucb_betas.sh.

set -euo pipefail

SCRIPT="bolt_exp.runners.test_w_botorch_dm"

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

SO_ACQFNS=(random ts ei qnei ucb mes gibbon pes jes kg)
SO_BATCH_ACQFNS=(random qnei mes gibbon)   # also run at q=5 and q=10
MO_ACQFNS=(random qnehvi qparego jes_mo mes_mo pes_mo)
MO_BATCH_ACQFNS=(random qnehvi qparego mes_mo)   # also run at q=5
HET_ACQFNS=(random ts ei qnei ucb mes gibbon pes jes kg)

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_dm() {
    run "$SCRIPT" --iterations "$ITERATIONS" --trials "$TRIALS" "$@" "${EXTRA_ARGS[@]}"
}

run_so() {
    for acq in "${SO_ACQFNS[@]}"; do
        run_dm --problem dm_curriculum --acq_fn "$acq"
    done
    for q in 5 10; do
        for acq in "${SO_BATCH_ACQFNS[@]}"; do
            run_dm --problem dm_curriculum --acq_fn "$acq" --batch_size "$q"
        done
    done
}

run_mo() {
    for acq in "${MO_ACQFNS[@]}"; do
        run_dm --problem dm_curriculum_mo --acq_fn "$acq"
    done
    for acq in "${MO_BATCH_ACQFNS[@]}"; do
        run_dm --problem dm_curriculum_mo --acq_fn "$acq" --batch_size 5
    done
}

run_hetero() {
    for acq in "${HET_ACQFNS[@]}"; do
        run_dm --problem dm_curriculum_heteroscedastic --acq_fn "$acq"
    done
    for beta in 1.0 2.0; do
        run_dm --problem dm_curriculum_heteroscedastic --acq_fn ucb --ucb_beta "$beta"
    done
    # LogNEI with the oracle noise variances, and with noise learned by MLHGP
    run_dm --problem dm_curriculum_heteroscedastic --acq_fn qnei --known_noise
    run_dm --problem dm_curriculum_heteroscedastic --acq_fn qnei --mlhgp --mlhgp_em_iter 5
}

case "$PROBLEM" in
    so)        run_so ;;
    mo)        run_mo ;;
    hetero)    run_hetero ;;
    all)       run_so; run_mo; run_hetero ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: so, mo, hetero, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All runs complete."
