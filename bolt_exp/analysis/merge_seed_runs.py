#!/usr/bin/env python
"""Merge per-seed result shards written by --seed_offset into one results file.

Runs launched with --seed_offset write `..._1trials_<N>iterations_seed<S>_results.json`
shards instead of the canonical `..._<T>trials_<N>iterations_results.json`. This
combines the shards back into that canonical file, validating that every shard shares
the same config and that no seed appears twice.

Usage:
  # merge every shard group found in a results dir
  python -m bolt_exp.analysis.merge_seed_runs results/dm_v2

  # merge one explicit group
  python -m bolt_exp.analysis.merge_seed_runs results/dm_v2 --pattern 'dm_curriculum_ei_*'

  # check what would be written without writing it
  python -m bolt_exp.analysis.merge_seed_runs results/dm_v2 --dry_run

Shards are left in place; pass --clean to delete them once the merged file is written.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# config fields that must agree across shards for a merge to be valid. Hardware fields
# are deliberately excluded: seeds spread over different GPUs is the normal case, and
# each trial keeps its own gpu_name/hostname.
CONFIG_KEYS = [
    "problem", "acq_fn", "ucb_beta", "noise_std", "known_noise", "mlhgp",
    "mlhgp_em_iter", "iterations", "batch_size", "initial_random_samples",
    "cost_scale",
]
HARDWARE_KEYS = {"device", "gpu_name", "hostname"}

SHARD_RE = re.compile(r"^(?P<stem>.+?)_(?P<ntrials>\d+)trials_(?P<iters>\d+)iterations(?P<tail>.*)_seed(?P<seed>\d+)_results\.json$")


def group_shards(results_dir: Path, pattern: str):
    """Group shard files by everything except their seed."""
    groups = defaultdict(list)
    for path in sorted(results_dir.glob(pattern)):
        m = SHARD_RE.match(path.name)
        if m is None:
            continue
        key = f"{m['stem']}_{{n}}trials_{m['iters']}iterations{m['tail']}_results.json"
        groups[key].append((int(m["seed"]), path))
    return groups


def merge_group(out_template: str, shards, results_dir: Path, dry_run: bool, clean: bool) -> bool:
    shards = sorted(shards)
    loaded = [(seed, path, json.load(open(path))) for seed, path in shards]

    ref_seed, ref_path, ref = loaded[0]
    for seed, path, doc in loaded[1:]:
        mismatches = {
            k: (ref.get(k), doc.get(k))
            for k in CONFIG_KEYS
            if k in ref or k in doc
            if ref.get(k) != doc.get(k)
        }
        if mismatches:
            print(f"  SKIP {out_template}: config mismatch between {ref_path.name} and {path.name}")
            for k, (a, b) in mismatches.items():
                print(f"    {k}: {a!r} vs {b!r}")
            return False

    trials = []
    seen = {}
    for seed, path, doc in loaded:
        for t in doc["trials"]:
            s = t["seed"]
            if s in seen:
                print(f"  SKIP {out_template}: seed {s} appears in both {seen[s]} and {path.name}")
                return False
            seen[s] = path.name
            trials.append(t)

    trials.sort(key=lambda t: t["seed"])
    # renumber trial indices so the merged file reads like a single sequential run
    for i, t in enumerate(trials):
        t["trial"] = i

    config = {k: v for k, v in ref.items() if k not in ("trials", "num_trials") and k not in HARDWARE_KEYS}
    merged = {**config, "num_trials": len(trials), "trials": trials}

    out_path = results_dir / out_template.format(n=len(trials))
    seeds = [t["seed"] for t in trials]
    print(f"  {out_path.name}  <- {len(loaded)} shards, {len(trials)} trials, seeds={seeds}")
    if dry_run:
        return True

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)
    if clean:
        for _, path, _ in loaded:
            path.unlink()
        print(f"    removed {len(loaded)} shard files")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="Directory holding the _seed<N>_results.json shards")
    ap.add_argument("--pattern", default="*_seed*_results.json", help="Glob to restrict which shards are merged")
    ap.add_argument("--dry_run", action="store_true", help="Report what would be written without writing")
    ap.add_argument("--clean", action="store_true", help="Delete shard files after a successful merge")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"not a directory: {args.results_dir}")

    groups = group_shards(args.results_dir, args.pattern)
    if not groups:
        raise SystemExit(f"no shards matching {args.pattern!r} in {args.results_dir}")

    print(f"{len(groups)} shard group(s) in {args.results_dir}:")
    ok = sum(merge_group(t, s, args.results_dir, args.dry_run, args.clean) for t, s in sorted(groups.items()))
    print(f"\n{ok}/{len(groups)} merged" + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
