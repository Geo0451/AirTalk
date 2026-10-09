#!/usr/bin/env python3
"""Air-stylus sample collector with real-time DTW quality verification.

    python collect.py --labels digits --reps 5

Features:
  * Automatically evaluates strokes against existing templates before saving.
  * Warns if a stroke misclassifies as another character (overmatching).
  * Auto-skips near-identical duplicates (intra-class redundancy) to keep dataset lean.
  * Touch = write. Quick TAP (< 0.25 s) = undo last sample.
"""

import argparse
import os
import random
import string
from collections import deque
from datetime import datetime

import numpy as np
from airpen_common import (
    DATA_DIR,
    MAG_VARIANTS,
    MAG_WEIGHT,
    VARIANTS,
    PenStream,
    build_sample,
    count_label,
    derive,
    features,
    load_templates,
    mag_field_dev,
    mag_present,
    meta_has_mag,
    prepare_meta,
    save_sample,
    vec,
)
from dtw_eval import dtw

TAP_S = 0.25  # touches shorter than this = "undo"
MAX_S = 8.0  # longer than this = discarded as a stuck touch
MIN_MOTION_DPS = 5.0  # mean |gyro| threshold
CLIP_BAND_DPS = (245.0, 258.0)
MAG_DISTURBED = 0.25

KEYWORDS = {
    "digits": string.digits,
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
}


def parse_labels(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        for ch in KEYWORDS.get(tok.lower(), tok):
            if ch not in out:
                out.append(ch)
    return out


def build_queue(labels, need, rng):
    q = []
    for r in range(max(need.values(), default=0)):
        rnd = [l for l in labels if need[l] > r]
        rng.shuffle(rnd)
        q += rnd
    return deque(q)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", default="digits")
    ap.add_argument(
        "--reps", type=int, default=5, help="target samples per label (tops up)"
    )
    ap.add_argument("--user", default="me")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument(
        "--variant", default="B", help="Feature variant used for live DTW checks (A-G)"
    )
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=4.5,
        help="DTW distance threshold to skip redundant duplicate templates",
    )
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="Disable real-time DTW quality verification gates",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="force the full calibration wizard (IMU axis + magnetometer)",
    )
    ap.add_argument(
        "--calibrate-mag",
        action="store_true",
        help="redo only the magnetometer calibration",
    )
    ap.add_argument(
        "--no-rebias", action="store_true", help="skip the 2 s gyro re-zero at start"
    )
    ap.add_argument("--release-ms", type=float, default=80.0)
    ap.add_argument("--beep", action="store_true", help="terminal bell on each save")
    args = ap.parse_args()

    labels = parse_labels(args.labels)
    stream = PenStream(args.port)
    meta = prepare_meta(
        stream, args.data, args.calibrate, not args.no_rebias, args.calibrate_mag
    )
    session = datetime.now().strftime("%Y%m%d_%H%M%S")

    v = args.variant.upper()
    need_mag = v in MAG_VARIANTS
    band = 15  # Sakoe-Chiba band window (len=100 * 0.15)

    # Load initial in-memory template pool for real-time verification
    X_templates, y_templates, _ = load_templates(
        args.data, args.user, v, L=100, mag_weight=MAG_WEIGHT
    )
    y_templates = list(y_templates) if y_templates is not None else []

    have = {l: count_label(args.data, args.user, l) for l in labels}
    need = {l: max(0, args.reps - have[l]) for l in labels}
    queue = build_queue(labels, need, random.Random())
    total = len(queue)
    if not total:
        print(
            f"Every label already has >= {args.reps} samples. Raise --reps to collect more."
        )
        return

    print(f"\nUser '{args.user}', session {session}: {total} samples to collect.")
    print("Touch = write.  Quick tap = undo last.  Ctrl+C = stop.\n")

    saved = []  # (path, label, vector) stack for undo
    warned_clip = warned_nomag = False
    try:
        while queue:
            label = queue[0]
            print(
                f"[{total - len(queue) + 1}/{total}]  WRITE:   {label}     (have {have[label]}/{args.reps})"
            )
            cap = stream.next_touch(release_ms=args.release_ms)
            dur = cap["duration"]

            if dur < TAP_S:  # tap = undo
                if saved:
                    path, lab, vec_item = saved.pop()
                    if os.path.exists(path):
                        os.remove(path)
                    have[lab] -= 1
                    queue.appendleft(lab)

                    # Remove from in-memory verification pool
                    if X_templates is not None and len(X_templates) > 0:
                        X_templates = X_templates[:-1]
                        y_templates.pop()

                    print(f"  ↩ undid last '{lab}'. Write it again.")
                else:
                    print("  (tap ignored; nothing to undo)")
                continue

            if dur > MAX_S:
                print(f"  ✗ touch lasted {dur:.1f} s. Discarded, try again.")
                continue

            gyro = cap["raw"][cap["start"] : cap["end"], 3:] - meta["gyro_bias_dps"]
            if np.linalg.norm(gyro, axis=1).mean() < MIN_MOTION_DPS:
                print("  ✗ barely any motion. Discarded, try again.")
                continue

            peak = float(np.abs(gyro).max())
            if not warned_clip and CLIP_BAND_DPS[0] <= peak <= CLIP_BAND_DPS[1]:
                print(f"  ! gyro peaked at {peak:.0f} dps (possible clipping).")
                warned_clip = True

            s = build_sample(cap, meta, label, args.user, session)
            d = derive(s, use_mag=need_mag)

            if need_mag and not d["has_mag"]:
                print("  ! no magnetometer data in capture, try again.")
                continue

            # Compute feature vector for DTW check
            x_vec = vec(features(d, v, MAG_WEIGHT), d["fs"], 100)

            # --- REAL-TIME QUALITY VERIFICATION GATES ---
            if not args.no_check and X_templates is not None and len(X_templates) > 0:
                dists = [dtw(x_vec, t, band) for t in X_templates]
                best_idx = int(np.argmin(dists))
                best_dist = dists[best_idx]
                predicted_label = y_templates[best_idx]

                # GATE 1: Inter-Class Overmatching Warning
                if predicted_label != label:
                    print(
                        f"  ⚠️  WARNING: Stroke looks closer to '{predicted_label}' than target '{label}' (dist: {best_dist:.2f})!"
                    )
                    choice = input("      Save anyway? [y/N]: ").strip().lower()
                    if choice != "y":
                        print("  --> Discarded corrupted stroke. Write it again.")
                        continue

                # GATE 2: Intra-Class Redundancy Check
                same_class_dists = [
                    dists[i] for i, l in enumerate(y_templates) if l == label
                ]
                if same_class_dists and min(same_class_dists) < args.sim_threshold:
                    print(
                        f"  ℹ️  REDUNDANT: Matches existing '{label}' template closely (dist: {min(same_class_dists):.2f} < {args.sim_threshold})."
                    )
                    choice = (
                        input("      Save duplicate anyway? [y/N]: ").strip().lower()
                    )
                    if choice != "y":
                        print("  --> Skipped redundant template to keep dataset lean.")
                        continue

            if meta_has_mag(meta):
                if not mag_present(cap["mag"]):
                    if not warned_nomag:
                        print("  ! no magnetometer data in this capture.")
                        warned_nomag = True
                elif (dev := mag_field_dev(s)) is not None and dev > MAG_DISTURBED:
                    print(
                        f"  ! magnetic interference while writing (|B| off by {100 * dev:.0f}%)."
                    )

            path = save_sample(args.data, s)
            saved.append((path, label, x_vec))

            # Update live in-memory evaluation cache
            if X_templates is None or len(X_templates) == 0:
                X_templates = x_vec[np.newaxis, ...]
            else:
                X_templates = np.vstack([X_templates, x_vec[np.newaxis, ...]])
            y_templates.append(label)

            have[label] += 1
            queue.popleft()
            print(f"  ✓ saved ({dur:.2f} s){chr(7) if args.beep else ''}")

    except KeyboardInterrupt:
        print("\nStopped.")

    print("\nSamples on disk for this user:")
    print(
        "  " + "  ".join(f"{l}:{count_label(args.data, args.user, l)}" for l in labels)
    )


if __name__ == "__main__":
    main()
