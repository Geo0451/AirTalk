#!/usr/bin/env python3
"""DTW 1-NN baseline for air-written characters.

    python dtw_eval.py                     # compare feature variants A-D on your collected data
    python dtw_eval.py --by session        # group by collection run instead of by day
    python dtw_eval.py --live --variant B  # recognise characters as you write them
    python dtw_eval.py --mag-only          # evaluate A-G on just the samples that have a calibrated mag

Variants A-D use only the IMU. E-G also need the calibrated magnetometer (collect.py --calibrate-mag,
then collect samples); they are skipped automatically while any sample lacks it, unless --mag-only.

Metrics per variant:
  LOO         leave-one-out 1-NN over every sample (needs no grouping)
  same-day kN random N templates per class within one day, tested on the rest (avg over days)
  cross-day   all templates from one day, tested on every other day (avg over days)
"""

import argparse
import sys
import time
from collections import Counter

import numpy as np
from airpen_common import (
    DATA_DIR,
    MAG_VARIANTS,
    MAG_WEIGHT,
    VARIANTS,
    PenStream,
    build_sample,
    derive,
    features,
    list_samples,
    load_sample,
    load_templates,
    meta_has_mag,
    prepare_meta,
    sample_has_mag,
    vec,
)

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(f=None, **kw):
        return f if f else (lambda g: g)


@njit(cache=True)
def dtw(a, b, band):
    """DTW with Sakoe-Chiba band; local cost = Euclidean distance across channels."""
    n, m = a.shape[0], b.shape[0]
    INF = 1e18
    prev = np.full(m + 1, INF)
    cur = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur[:] = INF
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = 0.0
            for k in range(a.shape[1]):
                d = a[i - 1, k] - b[j - 1, k]
                c += d * d
            best = prev[j - 1]
            best = min(best, prev[j])
            best = min(best, cur[j - 1])
            cur[j] = np.sqrt(c) + best
        prev, cur = cur, prev
    return prev[m]


def dist_matrix(X, band):
    N = len(X)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = dtw(X[i], X[j], band)
    return D


# ---- metrics --------------------------------------------------------------------------------
def loo(D, y):
    D2 = D + np.diag(np.full(len(D), np.inf))
    pred = y[np.argmin(D2, 1)]
    return float((pred == y).mean()), pred


def _split_acc(D, y, k, rng):
    gal, test = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        if len(idx) < 2:
            gal += list(idx)
            continue
        m = min(k, len(idx) - 1)
        gal += list(idx[:m])
        test += list(idx[m:])
    if not test:
        return None
    gal, test = np.array(gal), np.array(test)
    pred = y[gal[np.argmin(D[np.ix_(test, gal)], 1)]]
    return float((pred == y[test]).mean())


def within_group(D, y, groups, k, rng, trials=100):
    res = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 4:
            continue
        accs = [
            a
            for a in (
                _split_acc(D[np.ix_(idx, idx)], y[idx], k, rng) for _ in range(trials)
            )
            if a is not None
        ]
        if accs:
            res.append(np.mean(accs))
    return float(np.mean(res)) if res else float("nan")


def cross_group(D, y, groups):
    res, n_classes = [], len(np.unique(y))
    if len(np.unique(groups)) < 2:
        return float("nan")
    for g in np.unique(groups):
        gal = np.where(groups == g)[0]
        if len(np.unique(y[gal])) < 0.5 * n_classes:
            continue
        test = np.where((groups != g) & np.isin(y, y[gal]))[0]
        if len(test) == 0:
            continue
        pred = y[gal[np.argmin(D[np.ix_(test, gal)], 1)]]
        res.append((pred == y[test]).mean())
    return float(np.mean(res)) if res else float("nan")


def pct(x):
    return "   -  " if x != x else f"{100 * x:5.1f}%"


# ---- offline evaluation -----------------------------------------------------------------------
def evaluate(args):
    paths = list_samples(args.data, args.user)
    if not paths:
        sys.exit(f"No samples in {args.data}/{args.user}. Run collect.py first.")
    samples = [load_sample(p) for p in paths]
    n_nomag = sum(not sample_has_mag(s) for s in samples)
    if args.mag_only:
        samples = [s for s in samples if sample_has_mag(s)]
        if not samples:
            sys.exit(
                "--mag-only: no samples with a calibrated magnetometer. "
                "Run collect.py --calibrate-mag and collect some."
            )
        print(
            f"--mag-only: using {len(samples)} samples with mag, ignoring {n_nomag} without"
        )
        n_nomag = 0
    mag_ok = n_nomag == 0
    variants = {v: d for v, d in VARIANTS.items() if mag_ok or v not in MAG_VARIANTS}
    if not mag_ok:
        print(
            f"note: {n_nomag} of {len(samples)} samples have no calibrated magnetometer data, so variants "
            f"{'/'.join(sorted(MAG_VARIANTS))} are skipped.\n"
            "      (use --mag-only to compare all variants on just the samples that have it)"
        )

    feats = {v: [] for v in variants}
    y, groups, bad_heading = [], [], 0
    for s in samples:
        d = derive(s, use_mag=mag_ok)
        bad_heading += not d["heading_ok"]
        for v in variants:
            feats[v].append(vec(features(d, v, args.mag_weight), d["fs"], args.len))
        y.append(s["label"])
        groups.append(s["session"][:8] if args.by == "day" else s["session"])
    paths = [s["path"] for s in samples]
    y, groups = np.array(y), np.array(groups)
    counts = Counter(y)
    print(
        f"{len(y)} samples | {len(counts)} classes | per class {min(counts.values())}-{max(counts.values())} "
        f"| {len(set(groups))} {args.by}(s) | numba: {'yes' if HAVE_NUMBA else 'NO (pip install numba for ~100x speed)'}"
    )
    if bad_heading:
        print(
            f"note: {bad_heading} samples started with the pen nearly vertical, so heading alignment was skipped for them"
        )
    if min(counts.values()) < 2:
        print("warning: some classes have a single sample and can't be tested")

    band = max(1, int(round(args.band * args.len)))
    ks = [k for k in (1, 2, 3, 5) if k < max(counts.values())] or [1]
    rng = np.random.default_rng(0)
    head = (
        f"{'variant':<50}{'LOO':>8}"
        + "".join(f"{'same-' + args.by + ' k=' + str(k):>16}" for k in ks)
        + f"{'cross-' + args.by:>14}"
    )
    print("\n" + head + "\n" + "-" * len(head))
    res = {}
    for v, desc in variants.items():
        X = np.ascontiguousarray(np.stack(feats[v]))
        D = dist_matrix(X, band)
        acc, pred = loo(D, y)
        res[v] = (acc, pred)
        row = f"{v}  {desc:<47}{pct(acc):>8}"
        row += "".join(f"{pct(within_group(D, y, groups, k, rng)):>16}" for k in ks)
        row += f"{pct(cross_group(D, y, groups)):>14}"
        print(row)

    best = max(res, key=lambda v: res[v][0])
    _, pred = res[best]
    print(f"\nBest by LOO: variant {best}. Where it goes wrong:")
    wrong = np.where(pred != y)[0]
    if len(wrong) == 0:
        print("  nothing misclassified")
        return
    for (t, p), n in Counter(zip(y[wrong], pred[wrong])).most_common(5):
        print(f"  '{t}' read as '{p}': {n}x")
    print("  suspicious samples (delete if the writing was bad):")
    for i in wrong[:10]:
        print(f"    {paths[i]}   ('{y[i]}' -> '{pred[i]}')")


# ---- live recognition -----------------------------------------------------------------------
# Capslock is server-side only: the .ino keeps sending raw touch events unchanged. A quick
# touch (<TAP_S) is a "tap". Two taps landing within DOUBLE_TAP_GAP_S of each other toggle
# caps. A tap that isn't followed by a second one in time is just discarded as noise -- there's
# no separate timer thread; the next incoming touch (tap or real write) resolves it.
TAP_S = 0.25
DOUBLE_TAP_GAP_S = 0.4


def live(args):
    v = args.variant.upper()
    if v not in VARIANTS:
        sys.exit(f"Unknown variant '{v}'. Choose from {', '.join(VARIANTS)}.")
    need_mag = v in MAG_VARIANTS
    X, y, skipped = load_templates(args.data, args.user, v, args.len, args.mag_weight)
    if X is None:
        sys.exit(
            f"No usable templates in {args.data}/{args.user} for variant {v}."
            + (
                " (variant needs samples with a calibrated magnetometer; run collect.py --calibrate-mag)"
                if need_mag
                else " Run collect.py first."
            )
        )
    if skipped:
        print(
            f"note: {skipped} templates have no calibrated mag and are ignored for variant {v}."
        )
    band = max(1, int(round(args.band * args.len)))

    stream = PenStream(args.port)
    meta = prepare_meta(stream, args.data, False, not args.no_rebias)
    if need_mag and not meta_has_mag(meta):
        sys.exit(
            f"Variant {v} needs a magnetometer calibration, but calib.json has none. "
            "Run `python collect.py --calibrate-mag` first."
        )
    caps = False
    pending_tap_t = None  # wall-clock time of an unresolved first tap
    print(
        f"\nLoaded {len(y)} templates ({len(set(y))} classes), variant {v}. "
        f"Write a character, or double-tap to toggle caps. Ctrl+C to quit.\n"
    )
    print(f"  [caps: {'ON ' if caps else 'off'}]")
    try:
        while True:
            cap = stream.next_touch(release_ms=args.release_ms)
            now = time.monotonic()

            if cap["duration"] < TAP_S:
                if (
                    pending_tap_t is not None
                    and now - pending_tap_t <= DOUBLE_TAP_GAP_S
                ):
                    caps = not caps
                    pending_tap_t = None
                    print(f"  [caps: {'ON ' if caps else 'off'}]")
                else:
                    pending_tap_t = now  # wait to see if a second tap follows
                continue
            pending_tap_t = None  # a real write cancels any dangling single tap

            if cap["duration"] < 0.4:
                continue
            d = derive(build_sample(cap, meta), use_mag=need_mag)
            if need_mag and not d["has_mag"]:
                print("  (no magnetometer data in this capture, skipped)")
                continue
            x = vec(features(d, v, args.mag_weight), d["fs"], args.len)
            dist = np.array([dtw(x, t, band) for t in X])
            ranked = sorted((dist[y == c].min(), c) for c in set(y))
            best, second = ranked[0], ranked[1] if len(ranked) > 1 else (np.inf, "-")
            top3 = " | ".join(f"{c}: {dd:.1f}" for dd, c in ranked[:3])
            out = best[1].upper() if caps else best[1].lower()
            print(
                f"  -> {out}    (x{second[0] / max(best[0], 1e-9):.2f} margin)   [{top3}]"
            )
    except KeyboardInterrupt:
        print("\nbye")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--user", default="me")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--by", choices=["day", "session"], default="day")
    ap.add_argument(
        "--band",
        type=float,
        default=0.15,
        help="Sakoe-Chiba band as a fraction of length",
    )
    ap.add_argument("--len", type=int, default=100, help="resampled length")
    ap.add_argument("--live", action="store_true")
    ap.add_argument(
        "--variant",
        default="B",
        help="feature variant for --live (A-G; E-G need the magnetometer)",
    )
    ap.add_argument(
        "--mag-only",
        action="store_true",
        help="evaluate on only the samples that have a calibrated magnetometer (enables E-G)",
    )
    ap.add_argument(
        "--mag-weight",
        type=float,
        default=MAG_WEIGHT,
        help="weight of the magnetometer channels in variant E (1.0 = same as accel/gyro groups)",
    )
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--no-rebias", action="store_true")
    ap.add_argument("--release-ms", type=float, default=80.0)
    args = ap.parse_args()
    live(args) if args.live else evaluate(args)


if __name__ == "__main__":
    main()
