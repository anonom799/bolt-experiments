"""Analyse best points found for DMCurriculum (single-objective) BO results."""

import argparse
import json
from pathlib import Path

import numpy as np

from bolt_exp import REPO_ROOT

DIMS = ["IF_1", "Math_1", "Code_1", "IF_2", "Math_2", "Code_2"]

RESULTS_DIR = REPO_ROOT / "results" / "dm"


def _acq_name_from_path(path: Path) -> str:
    """Extract a short acq fn label from a result filename.

    e.g. dm_curriculum_qnei_5trials_200iterations_results.json -> qNEI
    """
    stem = path.stem  # strip .json
    # remove known prefix/suffix fragments
    for fragment in ("dm_curriculum_", "_results"):
        stem = stem.replace(fragment, "")
    # strip trailing _Ntrials_Miterations
    parts = stem.split("_")
    acq_parts = []
    for p in parts:
        if p.endswith("trials") or p.endswith("iterations") or p.isdigit():
            break
        acq_parts.append(p)
    return "_".join(acq_parts)


def discover_files(results_dir: Path) -> list[tuple[str, Path]]:
    """Return (acq_name, path) for all non-MO dm_curriculum result files."""
    paths = sorted(results_dir.glob("dm_curriculum_*.json"))
    return [
        (_acq_name_from_path(p), p)
        for p in paths
        if "_mo_" not in p.name
    ]


def load_results(results_dir: Path) -> dict:
    """Load all non-MO result files and return per-acq data."""
    data = {}
    for acq, path in discover_files(results_dir):
        with open(path) as f:
            d = json.load(f)
        trials = []
        for t in d["trials"]:
            rec_true_all = t.get("rec_true_all") or []
            if rec_true_all:
                best_rec_idx = int(np.argmax(rec_true_all))
                best_rec_y = t["best_rec_true_all"][-1]
                best_rec_x = t["rec_x_all"][best_rec_idx]
            else:
                best_rec_y = t["best_obs_true_all"][-1]
                best_rec_x = t["best_obs_x_all"][-1]
            trials.append({
                "trial": t["trial"],
                "best_rec_y": best_rec_y,
                "best_rec_x": best_rec_x,
                "best_obs_y": t["best_obs_true_all"][-1],
                "best_obs_x": t["best_obs_x_all"][-1],
                "best_y_all": t["best_y_all"],
            })
        data[acq] = trials
    return data


def print_per_trial_table(data: dict) -> None:
    print("Per-trial best points (true values):\n")
    for label, y_key, x_key in [
        ("Rec", "best_rec_y", "best_rec_x"),
        ("Obs", "best_obs_y", "best_obs_x"),
    ]:
        print(f"  [{label}]")
        print(f"  {'Acq':<10}  {'Trial':>5}  {'Best Y':>10}  Best X")
        print("  " + "-" * 108)
        for acq, trials in data.items():
            for t in trials:
                x_str = "[" + ", ".join(f"{v:.4f}" for v in t[x_key]) + "]"
                print(f"  {acq:<10}  {t['trial']:>5}  {t[y_key]:>10.6f}  {x_str}")
            bests = [t[y_key] for t in trials]
            print(
                f"  {'':10}  {'mean':>5}  {np.mean(bests):>10.6f}"
                f"  std={np.std(bests):.6f}  max={max(bests):.6f}"
            )
            print()


def print_acq_summary(data: dict) -> None:
    print("Acquisition function summary (mean best true-y ± std over trials):\n")
    for label, y_key in [("Rec", "best_rec_y"), ("Obs", "best_obs_y")]:
        print(f"  [{label}]  {'Acq':<10}  {'Mean Best Y':>12}  {'Std':>10}  {'Max':>10}")
        print("  " + "-" * 50)
        rows = []
        for acq, trials in data.items():
            bests = [t[y_key] for t in trials]
            rows.append((acq, np.mean(bests), np.std(bests), max(bests)))
        for acq, mean_b, std_b, max_b in sorted(rows, key=lambda r: -r[1]):
            print(f"  {acq:<10}  {mean_b:>12.6f}  {std_b:>10.6f}  {max_b:>10.6f}")
        print()


def collect_all_best(data: dict):
    """Flatten best rec and obs (x, y) pairs across acq fns and trials."""
    rec_xs, rec_ys, obs_xs, obs_ys = [], [], [], []
    for trials in data.values():
        for t in trials:
            rec_xs.append(t["best_rec_x"])
            rec_ys.append(t["best_rec_y"])
            obs_xs.append(t["best_obs_x"])
            obs_ys.append(t["best_obs_y"])
    return np.array(rec_xs), np.array(rec_ys), np.array(obs_xs), np.array(obs_ys)


def print_top_k(rec_xs, rec_ys, obs_xs, obs_ys, k: int = 5) -> None:
    for label, xs, ys in [("Rec", rec_xs, rec_ys), ("Obs", obs_xs, obs_ys)]:
        print(f"\nTop {k} best [{label}] evaluations across all methods/trials:\n")
        idxs = np.argsort(ys)[::-1][:k]
        for rank, i in enumerate(idxs):
            x = xs[i]
            print(
                f"  #{rank+1}  y={ys[i]:.6f}"
                f"  P1=[IF={x[0]:.3f}, Math={x[1]:.3f}, Code={x[2]:.3f}]"
                f"  P2=[IF={x[3]:.3f}, Math={x[4]:.3f}, Code={x[5]:.3f}]"
            )


def print_dim_stats(xs: np.ndarray, ys: np.ndarray) -> None:
    print("\nPer-dimension statistics over all best points:\n")
    print(f"  {'Dim':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 46)
    for i, d in enumerate(DIMS):
        vals = xs[:, i]
        print(f"  {d:<10} {vals.mean():>8.4f} {vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

    print()
    mean_x = xs.mean(axis=0)
    print(f"  Mean best X: {dict(zip(DIMS, [round(float(v), 4) for v in mean_x]))}")
    print(f"  Phase 1 sum: {mean_x[:3].sum():.4f},  Phase 2 sum: {mean_x[3:].sum():.4f}")

    print()
    n = len(xs)
    code1_near_zero = (xs[:, 2] < 0.05).sum()
    code2_small = (xs[:, 5] < 0.15).sum()
    print(f"  Code_1 < 0.05 in {code1_near_zero}/{n} best points ({100*code1_near_zero/n:.0f}%)")
    print(f"  Code_2 < 0.15 in {code2_small}/{n} best points ({100*code2_small/n:.0f}%)")


def main(results_dir: Path, top_k: int) -> None:
    print(f"Loading results from: {results_dir}")
    files = discover_files(results_dir)
    if not files:
        print("No dm_curriculum (non-MO) result files found.")
        return
    print(f"Found {len(files)} file(s): {', '.join(acq for acq, _ in files)}\n")
    data = load_results(results_dir)
    if not data:
        print("No result files found.")
        return

    print_per_trial_table(data)
    print_acq_summary(data)

    rec_xs, rec_ys, obs_xs, obs_ys = collect_all_best(data)
    print_top_k(rec_xs, rec_ys, obs_xs, obs_ys, k=top_k)
    print("\n--- Dim stats: Rec ---")
    print_dim_stats(rec_xs, rec_ys)
    print("\n--- Dim stats: Obs ---")
    print_dim_stats(obs_xs, obs_ys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse DMCurriculum BO results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing result JSON files.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top points to display.")
    args = parser.parse_args()
    main(args.results_dir, args.top_k)
