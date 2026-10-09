#!/usr/bin/env python3
"""Dataset Pruner for Air-Written Character Templates.

Identifies and cleans:
  1. Redundant intra-class duplicates (near-identical strokes of the same character).
  2. Inter-class boundary polluters (templates misclassified by their dataset neighbors).
  3. Over-capacity samples (caps max templates per character under --max-per-class).

Usage:
    python prune.py                          # dry-run scan on default user 'me'
    python prune.py --apply                  # delete flagged .npz files from disk
    python prune.py --sim-threshold 4.5      # set custom duplicate distance threshold
    python prune.py --max-per-class 10       # cap max templates per character
"""

import argparse
import os
import sys
import numpy as np
from collections import Counter

from airpen_common import (
    DATA_DIR,
    VARIANTS,
    MAG_VARIANTS,
    MAG_WEIGHT,
    list_samples,
    load_sample,
    sample_has_mag,
    derive,
    features,
    vec,
)
from dtw_eval import dtw


def prune(args):
    v = args.variant.upper()
    if v not in VARIANTS:
        sys.exit(f"Unknown variant '{v}'. Choose from {', '.join(VARIANTS)}.")

    paths = list_samples(args.data, args.user)
    if not paths:
        sys.exit(f"No samples found in {args.data}/{args.user}.")

    need_mag = v in MAG_VARIANTS
    samples = []
    skipped_mag = 0
    for p in paths:
        s = load_sample(p)
        if need_mag and not sample_has_mag(s):
            skipped_mag += 1
            continue
        samples.append(s)

    if not samples:
        sys.exit("No usable samples loaded for pruning.")

    band = max(1, int(round(args.band * args.len)))

    print(f"Loading features for {len(samples)} samples (User: '{args.user}', Variant {v})...")
    feats = []
    labels = []
    file_paths = []

    for s in samples:
        d = derive(s, use_mag=need_mag)
        feats.append(vec(features(d, v, args.mag_weight), d["fs"], args.len))
        labels.append(s["label"])
        file_paths.append(s["path"])

    N = len(feats)
    labels = np.array(labels)
    to_remove = {}  # idx -> reason string

    # --- 1. Compute Pairwise DTW Matrix ---
    print(f"Computing DTW matrix across {N} templates...")
    D = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(i + 1, N):
            dist = dtw(feats[i], feats[j], band)
            D[i, j] = D[j, i] = dist

    # --- 2. Intra-Class Redundancy (Duplicates) ---
    for i in range(N):
        if i in to_remove:
            continue
        for j in range(i + 1, N):
            if j in to_remove:
                continue
            if labels[i] == labels[j]:
                if D[i, j] < args.sim_threshold:
                    to_remove[j] = (
                        f"Duplicate of {os.path.basename(file_paths[i])} "
                        f"(dist: {D[i, j]:.2f} < threshold {args.sim_threshold})"
                    )

    # --- 3. Inter-Class Boundary Polluters (Edited 1-NN) ---
    for i in range(N):
        if i in to_remove:
            continue
        valid_indices = [j for j in range(N) if j != i and j not in to_remove]
        if not valid_indices:
            continue
        dists = D[i, valid_indices]
        nearest_idx = valid_indices[np.argmin(dists)]
        nearest_dist = np.min(dists)
        nearest_label = labels[nearest_idx]

        if nearest_label != labels[i]:
            to_remove[i] = (
                f"Polluter: Class '{labels[i]}' misclassified as '{nearest_label}' "
                f"by neighbor {os.path.basename(file_paths[nearest_idx])} (dist: {nearest_dist:.2f})"
            )

    # --- 4. Class Capacity Capping ---
    if args.max_per_class > 0:
        remaining_indices = [i for i in range(N) if i not in to_remove]
        class_groups = {}
        for idx in remaining_indices:
            class_groups.setdefault(labels[idx], []).append(idx)

        for label, idxs in class_groups.items():
            if len(idxs) > args.max_per_class:
                sub_D = D[np.ix_(idxs, idxs)]
                mean_dists = sub_D.mean(axis=1)
                sorted_by_centrality = [idxs[k] for k in np.argsort(mean_dists)]
                num_to_trim = len(idxs) - args.max_per_class
                for trim_idx in sorted_by_centrality[:num_to_trim]:
                    to_remove[trim_idx] = (
                        f"Capacity Cap: Exceeds max {args.max_per_class} templates for class '{label}'"
                    )

    # --- Reporting & Execution ---
    print(f"\nScan complete. Total samples: {N} | Flagged for removal: {len(to_remove)}")
    if not to_remove:
        print("Dataset is clean! No redundant or polluting templates found.")
        return

    print("\nFlagged files:")
    for idx, reason in to_remove.items():
        fname = os.path.basename(file_paths[idx])
        lbl = labels[idx]
        print(f"  [{lbl}] {fname} -> {reason}")

    if args.apply:
        print(f"\nDeleting {len(to_remove)} files from disk...")
        deleted = 0
        for idx in to_remove:
            try:
                os.remove(file_paths[idx])
                deleted += 1
            except OSError as e:
                print(f"  Error removing {file_paths[idx]}: {e}")
        print(f"Done. Successfully deleted {deleted} files.")
    else:
        print("\n[DRY RUN] No files deleted. Run with `--apply` to prune files from disk.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--user", default="me")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--variant", default="B", help="Feature variant (A-G)")
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=4.5,
        help="DTW distance threshold below which samples are considered duplicates",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Max templates to keep per character class (0 = unlimited)",
    )
    ap.add_argument(
        "--band",
        type=float,
        default=0.15,
        help="Sakoe-Chiba band as a fraction of vector length",
    )
    ap.add_argument("--len", type=int, default=100, help="Resampled vector length")
    ap.add_argument(
        "--mag-weight",
        type=float,
        default=MAG_WEIGHT,
        help="Weight of magnetometer channels in variant E",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete flagged files from disk (default is dry-run)",
    )
    args = ap.parse_args()
    prune(args)


if __name__ == "__main__":
    main()
