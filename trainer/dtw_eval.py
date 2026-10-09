#!/usr/bin/env python3
"""DTW 1-NN baseline for air-written characters.

    python dtw_eval.py                     # compare feature variants A-D on your collected data
    python dtw_eval.py --by session        # group by collection run instead of by day
    python dtw_eval.py --live --variant B  # recognise characters as you write them
    python dtw_eval.py --mag-only          # evaluate A-G on just the samples that have a calibrated mag
    python dtw_eval.py --active --variant A                 # active training on built-in sentences
    python dtw_eval.py --active --variant A --text "hello world"   # ...or on your own sentence(s)

Variants A-D use only the IMU. E-G also need the calibrated magnetometer (collect.py --calibrate-mag,
then collect samples); they are skipped automatically while any sample lacks it, unless --mag-only.

--active: you are prompted one character at a time from a sentence. The recognizer reads each one
with the templates it has *so far*; a wrong read is fixed by saving that very sample under the
intended label (it becomes a template at once and is written to disk). Quick tap = undo.

Metrics per variant:
  LOO         leave-one-out 1-NN over every sample (needs no grouping)
  same-day kN random N templates per class within one day, tested on the rest (avg over days)
  cross-day   all templates from one day, tested on every other day (avg over days)
"""

import argparse
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

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
    save_sample,
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


# ---- active training ------------------------------------------------------------------------
# "Test, then train": for each prompted character the recognizer first reads the pen with the
# templates it has so far, and only then learns from it. A wrong read is corrected by saving that
# very sample under the intended label (immediately usable, and written to disk like collect.py).
# The first-try accuracy is therefore an honest running score, not a fit to its own training data.
# Case is ignored here: caps is output-only, there is one canonical stroke shape per letter.
ACTIVE_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "two driven jocks help fax my big quiz",
    "call 081 7365 4290 before 6",
    "room 204 on floor 13 gate 9 seat 57",
]
LIVE_MIN_S = 0.4  # live mode ignores touches shorter than this, so the trainer does too


def _label_for(ch, classes):
    """Template class for a sentence character (exact match first, then the other case)."""
    for c in (ch, ch.upper(), ch.lower()):
        if c in classes:
            return c
    return None


def _rank(x, Xs, ys, band):
    """[(distance, class)] nearest-first, one entry per class (its best template)."""
    dist = np.array([dtw(x, t, band) for t in Xs])
    arr = np.array(ys)
    return sorted((float(dist[arr == c].min()), c) for c in set(ys))


def _case_like(pred, ch):
    return pred.upper() if ch.isupper() else pred.lower() if ch.isalpha() else pred


def _sentence_summary(si, sent, tasks, history):
    es = {e["idx"]: e for e in history if e["si"] == si}
    got = [" " if c.isspace() else "_" for c in sent]
    for i, e in es.items():
        got[i] = _case_like(e["pred"], sent[i])
    ok = sum(e["ok"] for e in es.values())
    print(f"  wrote: {sent}\n  read : {''.join(got)}")
    print(f"  sentence {si + 1}: {ok}/{len(tasks)} right first try\n")


def _active_summary(history, data_dir="data"):
    n = len(history)
    if not n:
        print("Nothing written.")
        return
    ok = sum(e["ok"] for e in history)
    print(
        f"\nFirst-try accuracy: {ok}/{n} = {100 * ok / n:.1f}%  (each read was made before learning from it)"
    )
    per = {}
    for e in history:
        per.setdefault(e["si"], []).append(e["ok"])
    if len(per) > 1:
        print(
            "  per sentence: "
            + "  ".join(f"s{k + 1} {sum(v)}/{len(v)}" for k, v in sorted(per.items()))
        )
    learned = [e for e in history if e["path"]]
    wrong = sum(not e["ok"] for e in learned)
    print(
        f"Templates added: {len(learned)} ({wrong} after wrong reads, {len(learned) - wrong} low-margin)"
        + (
            "  -> "
            + " ".join(
                f"{c}+{k}"
                for c, k in sorted(Counter(e["label"] for e in learned).items())
            )
            if learned
            else ""
        )
    )
    conf = Counter((e["label"], e["pred"]) for e in history if not e["ok"])
    if conf:
        print("Misreads:")
        for (t, p), k in conf.most_common(8):
            note = (
                "  <- keeps recurring despite corrections: possibly a same-shape pair"
                if k >= 3
                else ""
            )
            print(f"  '{t}' read as '{p}': {k}x{note}")
    print(
        f"Added samples are tagged with a session ending in '_active' (ls {data_dir}/*/*/*_active_*.npz)"
    )


def active(args):
    from collect import MAX_S, MIN_MOTION_DPS  # same discard rules as collect.py

    v = args.variant.upper()
    if v not in VARIANTS:
        sys.exit(f"Unknown variant '{v}'. Choose from {', '.join(VARIANTS)}.")
    need_mag = v in MAG_VARIANTS
    X, y, skipped = load_templates(args.data, args.user, v, args.len, args.mag_weight)
    if X is None:
        sys.exit(
            f"No usable templates in {args.data}/{args.user} for variant {v}."
            " Active training refines an existing set: run collect.py first."
        )
    if skipped:
        print(
            f"note: {skipped} templates have no calibrated mag and are ignored for variant {v}."
        )
    Xs, ys = list(X), [str(c) for c in y]
    classes = set(ys)
    band = max(1, int(round(args.band * args.len)))

    def tasks_of(sent):
        out = []
        for i, c in enumerate(sent):
            lab = None if c.isspace() else _label_for(c, classes)
            if lab:
                out.append((i, lab))
        return out

    rng = random.Random()
    sents = list(args.text or [])
    if sents and not any(tasks_of(s) for s in sents):
        sys.exit(
            f"None of the characters to write have templates (classes: {' '.join(sorted(classes))})."
        )

    def coverage(sent):  # share of the sentence's characters that have a template class
        n = sum(not c.isspace() for c in sent)
        return len(tasks_of(sent)) / max(n, 1)

    pool = [s for s in ACTIVE_SENTENCES if coverage(s) >= 0.8]
    if not args.text and not pool:
        print(
            "note: the built-in sentences need characters you haven't collected; using random groups of your classes."
        )

    def get_sentence(si):
        while si >= len(sents):
            if args.text:
                return None  # your own sentences are written once
            if pool:
                batch = list(pool)
                rng.shuffle(batch)
            else:
                cl = sorted(classes)
                batch = [
                    " ".join(
                        "".join(rng.choice(cl) for _ in range(4)) for _ in range(4)
                    )
                    for _ in range(3)
                ]
            sents.extend(batch)
        return sents[si]

    stream = PenStream(args.port)
    meta = prepare_meta(stream, args.data, False, not args.no_rebias)
    if need_mag and not meta_has_mag(meta):
        sys.exit(
            f"Variant {v} needs a magnetometer calibration, but calib.json has none. "
            "Run `python collect.py --calibrate-mag` first."
        )
    session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_active"
    print(
        f"\nLoaded {len(ys)} templates ({len(classes)} classes), variant {v}.\n"
        "Write each prompted character (case ignored). A wrong read is fixed by saving your sample as the\n"
        "intended character. Quick tap = undo the last one (removes what it learned). Ctrl+C = stop.\n"
        + (
            f"Also learning from correct reads with margin below x{args.learn_margin:.2f}.\n"
            if args.learn_margin > 0
            else ""
        )
    )

    history = []  # one dict per written character; undo pops it
    si = ci = 0
    shown = -1
    try:
        while True:
            sent = get_sentence(si)
            if sent is None:
                break
            tasks = tasks_of(sent)
            if ci >= len(tasks):
                if tasks:
                    _sentence_summary(si, sent, tasks, history)
                si, ci = si + 1, 0
                continue
            if shown != si:
                shown = si
                miss = sorted(
                    {c for c in sent if not c.isspace() and not _label_for(c, classes)}
                )
                print(
                    f"--- sentence {si + 1}: {sent}"
                    + (
                        f"   (no templates for {' '.join(miss)}, skipped)"
                        if miss
                        else ""
                    )
                )
            idx, label = tasks[ci]
            print(
                f"[{ci + 1}/{len(tasks)}]  WRITE: {sent[idx]}     {sent[:idx]}[{sent[idx]}]{sent[idx + 1 :]}"
            )

            cap = stream.next_touch(release_ms=args.release_ms)
            dur = cap["duration"]
            if dur < TAP_S:  # tap = undo
                if not history:
                    print("  (tap ignored; nothing to undo)")
                    continue
                e = history.pop()
                if e["path"]:
                    Xs.pop()
                    ys.pop()
                    try:
                        os.remove(e["path"])
                    except FileNotFoundError:
                        pass
                si, ci = e["si"], e["ci"]
                print(
                    f"  ↩ undid '{e['char']}'"
                    + (" and removed the template it added" if e["path"] else "")
                    + ". Write it again."
                )
                continue
            if dur < LIVE_MIN_S:
                print(
                    f"  ✗ touch lasted {dur:.2f} s; live mode ignores touches under {LIVE_MIN_S} s. Try again."
                )
                continue
            if dur > MAX_S:
                print(f"  ✗ touch lasted {dur:.1f} s. Discarded, try again.")
                continue
            gyro = cap["raw"][cap["start"] : cap["end"], 3:] - meta["gyro_bias_dps"]
            if np.linalg.norm(gyro, axis=1).mean() < MIN_MOTION_DPS:
                print("  ✗ barely any motion. Discarded, try again.")
                continue

            s = build_sample(cap, meta, label, args.user, session)
            d = derive(s, use_mag=need_mag)
            if need_mag and not d["has_mag"]:
                print("  (no magnetometer data in this capture, skipped)")
                continue
            x = vec(features(d, v, args.mag_weight), d["fs"], args.len)

            ranked = _rank(x, Xs, ys, band)
            pred = ranked[0][1]
            ok = pred == label
            best = max(ranked[0][0], 1e-9)
            margin = (
                ranked[1][0] / best if len(ranked) > 1 else float("inf")
            )  # only meaningful if ok
            learn = (not ok) or (args.learn_margin > 0 and margin < args.learn_margin)
            path = None
            if learn:
                path = save_sample(args.data, s)
                Xs.append(x)
                ys.append(label)
            history.append(
                dict(
                    si=si,
                    ci=ci,
                    idx=idx,
                    char=sent[idx],
                    label=label,
                    pred=pred,
                    ok=ok,
                    path=path,
                )
            )
            n_ok, n = sum(e["ok"] for e in history), len(history)

            if ok and not learn:
                print(f"  ✓ {pred}   (x{margin:.2f} margin)   running {n_ok}/{n}")
            elif ok:
                print(
                    f"  ✓ {pred} but only x{margin:.2f} margin -> added as template #{ys.count(label)} for '{label}'   running {n_ok}/{n}"
                )
            else:
                names = [c for _, c in ranked]
                gap = dict((c, dd) for dd, c in ranked)[label] / best
                print(
                    f"  ✗ read '{pred}', wanted '{label}' ('{label}' was #{names.index(label) + 1}, {gap:.2f}x farther)"
                    f" -> added as template #{ys.count(label)} for '{label}'   running {n_ok}/{n}"
                )
            ci += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    _active_summary(history, args.data)


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
    ap.add_argument(
        "--active",
        action="store_true",
        help="active training: write prompted sentences; wrong reads are corrected with your sample",
    )
    ap.add_argument(
        "--text",
        action="append",
        help="sentence for --active (repeatable; written once each). Default: built-in sentences, endless",
    )
    ap.add_argument(
        "--learn-margin",
        type=float,
        default=0.0,
        help="--active: also add correct reads whose margin (2nd/1st distance) is below this, e.g. 1.15",
    )
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--no-rebias", action="store_true")
    ap.add_argument("--release-ms", type=float, default=80.0)
    args = ap.parse_args()
    if args.active and args.live:
        ap.error("--active and --live are mutually exclusive")
    if args.active:
        active(args)
    elif args.live:
        live(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
