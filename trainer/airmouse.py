#!/usr/bin/env python3
"""Wayland Air Mouse Controller with acceleration ballistics and high responsiveness."""

import argparse
import math
import subprocess
import time
import numpy as np
from airpen_common import DATA_DIR, PenStream, prepare_meta


def ydo_action(cmd_args):
    try:
        subprocess.run(["ydotool"] + cmd_args, check=True)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"  [error] ydotool failed: {e}")


def main():
    ap = argparse.ArgumentParser(
        description=("Control mouse movement on Wayland with acceleration ballistics.")
    )
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--transport", choices=["auto", "udp", "ble"], default="auto")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument(
        "--hold-threshold",
        type=float,
        default=0.5,
        help="Seconds for long hold (Right Click)",
    )
    ap.add_argument(
        "--double-gap",
        type=float,
        default=0.30,
        help="Max gap for double click (Middle Click)",
    )
    ap.add_argument(
        "--smoothing",
        type=float,
        default=0.75,
        help="Smoothing factor",
    )
    ap.add_argument(
        "--sensitivity",
        type=float,
        default=0.3,
        help="Base sensitivity multiplier",
    )
    ap.add_argument("--deadzone", type=float, default=0.5, help="Deadzone threshold")
    args = ap.parse_args()

    stream = PenStream(args.port, transport=args.transport)
    meta = prepare_meta(stream, args.data, force_calibrate=False, rebias=False)
    stream.drain()

    print(
        "\nFast Air Mouse active. Tap=Left, Double tap=Middle, Hold=Right."
        " Ctrl+C to quit.\n"
    )

    prev_touch = False
    t_down = 0.0
    last_release_t = 0.0
    tap_count = 0

    bias = np.asarray(meta["gyro_bias_dps"], float)
    ax = np.asarray(meta["pen_axis"], float)
    ax = ax / np.linalg.norm(ax)
    acc_scale = float(meta["acc_scale"])

    g = None
    forward_vec = None
    prev_t = None
    fX = 0.0
    fY = 0.0

    try:
        while True:
            p = stream._read()
            if p is None:
                if tap_count == 1 and (time.time() - last_release_t) > args.double_gap:
                    ydo_action(["click", "0xC0"])
                    print("  [action] Left Click (Single Tap)")
                    tap_count = 0
                continue

            t, touched = p[0], p[1]
            dt = 0.01 if prev_t is None else min(max(t - prev_t, 0.0), 0.05)
            prev_t = t

            a = np.array(p[2:5], float) / acc_scale
            w = np.array(p[5:8], float) - bias

            if g is None:
                g = a.copy()
            else:
                g += (1.0 - math.exp(-dt / 0.4)) * (a - g)

            u = g / max(float(np.linalg.norm(g)), 1e-6)

            f = ax - (ax @ u) * u
            nf = float(np.linalg.norm(f))
            if nf > 0.25:
                forward_vec = f / nf
            if forward_vec is None:
                temp = (
                    np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
                )
                forward_vec = np.cross(u, temp)
                forward_vec /= np.linalg.norm(forward_vec)

            f_proj = forward_vec - (forward_vec @ u) * u
            f_proj /= max(float(np.linalg.norm(f_proj)), 1e-6)
            l = np.cross(u, f_proj)

            yaw = float(w @ np.cross(ax, l))
            pitch = float(w @ l)

            gyroX = -yaw
            gyroY = pitch

            # Exponential smoothing
            fX = fX * args.smoothing + gyroX * (1.0 - args.smoothing)
            fY = fY * args.smoothing + gyroY * (1.0 - args.smoothing)

            # Base scaling
            moveX = fX * args.sensitivity
            moveY = fY * args.sensitivity

            # Deadzone filtering
            if abs(moveX) < args.deadzone:
                moveX = 0
            if abs(moveY) < args.deadzone:
                moveY = 0

            # Mouse Ballistics / Acceleration:
            # Fast movements get a non-linear multiplier to hit screen edges instantly
            speed = math.hypot(moveX, moveY)
            if speed > 0:
                boost = 1.0 + (speed / 8.0)
                moveX *= boost
                moveY *= boost

            mouseX = int(moveX)
            mouseY = int(moveY)

            if mouseX != 0 or mouseY != 0:
                ydo_action(["mousemove", "--", str(mouseX), str(mouseY)])

            if touched and not prev_touch:
                t_down = time.time()
            elif not touched and prev_touch:
                t_up = time.time()
                duration = t_up - t_down

                if duration >= args.hold_threshold:
                    tap_count = 0
                    ydo_action(["click", "0xC2"])
                    print("  [action] Right Click (Long Hold)")
                else:
                    if (t_up - last_release_t) <= args.double_gap:
                        tap_count += 1
                    else:
                        tap_count = 1

                    if tap_count == 2:
                        tap_count = 0
                        last_release_t = 0.0
                        ydo_action(["click", "0xC1"])
                        print("  [action] Middle Click (Double Tap)")
                    else:
                        last_release_t = t_up

            prev_touch = touched

            if tap_count == 1 and (time.time() - last_release_t) > args.double_gap:
                ydo_action(["click", "0xC0"])
                print("  [action] Left Click (Single Tap)")
                tap_count = 0

    except KeyboardInterrupt:
        print("\nExiting air mouse.")


if __name__ == "__main__":
    main()
