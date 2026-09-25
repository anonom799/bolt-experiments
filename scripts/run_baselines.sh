#!/usr/bin/env bash
# Run the non-GP baselines, as used in the paper: random search, TPE and CMA-ES on
# hpo; random + BOHB on hpo_fd_step; random + ASHA on hpo_fd_model; TSEMO and
# NSGA-II on dm_curriculum_mo. (Random search on the DM and PO problems is an
# --acq_fn of the GP runners instead.)
#
# Usage:
#   bash scripts/run_baselines.sh                       # everything
#   bash scripts/run_baselines.sh --problem hpo         # hpo / hpo_fd_step / hpo_fd_model / dm_mo
#   bash scripts/run_baselines.sh --iterations 50 --trials 3

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

run() {
    echo ""
    echo ">>> python -m $*"
    python -m "$@"
}

run_methods() {
    local script="$1" prob="$2"; shift 2
    for method in "$@"; do
        run "$script" --problem "$prob" --method "$method" \
            --iterations "$ITERATIONS" --trials "$TRIALS" \
            --initial_random_samples "$INIT_SAMPLES" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
}

HPO=bolt_exp.runners.test_w_baselines_hpo
MO=bolt_exp.runners.test_w_baselines_mo

case "$PROBLEM" in
    hpo)          run_methods $HPO hpo random tpe cmaes ;;
    hpo_fd_step)  run_methods $HPO hpo_fd_step random bohb ;;
    hpo_fd_model) run_methods $HPO hpo_fd_model random asha ;;
    dm_mo)        run_methods $MO dm_curriculum_mo tsemo nsga2 ;;
    all)
        run_methods $HPO hpo random tpe cmaes
        run_methods $HPO hpo_fd_step random bohb
        run_methods $HPO hpo_fd_model random asha
        run_methods $MO dm_curriculum_mo tsemo nsga2
        ;;
    *)
        echo "Unknown --problem '$PROBLEM'. Use: hpo, hpo_fd_step, hpo_fd_model, dm_mo, all" >&2
        exit 1
        ;;
esac

echo ""
echo "All baseline runs complete."
