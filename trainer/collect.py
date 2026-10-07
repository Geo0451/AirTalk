#!/usr/bin/env python3
"""Air-stylus sample collector. No keyboard needed once it's running.

    python collect.py --labels digits --reps 5

  * Touch the pen pad, write the prompted character, release. It saves and shows the next one.
  * A quick TAP (< 0.25 s) undoes the last sample and re-asks for it.
  * Re-running the same command only tops up what's missing (counts what's already on disk),
    so "again and again" is just: run it, write, Ctrl+C.
  * Prompts go round-robin through the labels (shuffled each round), which gives more realistic
    variation than writing the same character five times in a row.

--labels accepts characters and/or keywords: "digits", "upper", "lower", e.g. "digits,upper" or "ABC".
First run launches a one-time calibration that figures out the pen axis by itself and then
calibrates the magnetometer (tumble the pen through every orientation for 20 s).
    --calibrate      redo the whole wizard (IMU axis + magnetometer)
    --calibrate-mag  redo only the magnetometer (e.g. after moving the battery/wiring near it)
Samples store the raw magnetometer plus the calibration they were captured under.
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
    PenStream,
    build_sample,
    count_label,
    mag_field_dev,
    mag_present,
    meta_has_mag,
    prepare_meta,
    save_sample,
)

TAP_S = 0.25  # touches shorter than this = "undo"
MAX_S = 8.0  # longer than this = discarded as a stuck touch
MIN_MOTION_DPS = (
    5.0  # mean |gyro| during the touch; below this the pen basically didn't move
)
CLIP_BAND_DPS = (245.0, 258.0)  # a gyro peak in here = sign of the old +-250 dps firmware range
MAG_DISTURBED = 0.25  # |B| this far (relative) from the calibrated field = interference

KEYWORDS = {
    "digits": string.digits,
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
}


def parse_labels(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        for ch in KEYWORDS.get(tok.lower(), tok):  # type: ignore
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
        "--calibrate",
        action="store_true",
        help="force the full calibration wizard (IMU axis + magnetometer)",
    )
    ap.add_argument(
        "--calibrate-mag",
        action="store_true",
        help="redo only the magnetometer calibration (keeps the IMU axis calibration)",
    )
    ap.add_argument(
        "--no-rebias", action="store_true", help="skip the 2 s gyro re-zero at start"
    )
    ap.add_argument(
        "--release-ms",
        type=float,
        default=80.0,
        help="touch debounce (raise if characters get split)",
    )
    ap.add_argument("--beep", action="store_true", help="terminal bell on each save")
    args = ap.parse_args()

    labels = parse_labels(args.labels)
    stream = PenStream(args.port)
    meta = prepare_meta(
        stream, args.data, args.calibrate, not args.no_rebias, args.calibrate_mag
    )
    session = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    saved = []  # (path, label) stack for undo
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
                    path, lab = saved.pop()
                    os.remove(path)
                    have[lab] -= 1
                    queue.appendleft(lab)
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
                print(
                    f"  ! gyro peaked at {peak:.0f} dps, which looks like +-250 dps clipping. "
                    "Check GYRO_FS_REG / GYRO_LSB_PER_DPS in the firmware."
                )
                warned_clip = True

            s = build_sample(cap, meta, label, args.user, session)
            if meta_has_mag(meta):
                if not mag_present(cap["mag"]):
                    if not warned_nomag:
                        print("  ! no magnetometer data in this capture (saved, but E-G can't use it).")
                        warned_nomag = True
                elif (dev := mag_field_dev(s)) is not None and dev > MAG_DISTURBED:
                    print(
                        f"  ! magnetic interference while writing (|B| off by {100 * dev:.0f}%). "
                        "Saved, but consider redoing it away from monitors/phones/metal."
                    )
            path = save_sample(args.data, s)
            saved.append((path, label))
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
