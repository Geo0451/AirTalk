#!/usr/bin/env python3
"""DTW 1-NN baseline for air-written characters, with a live WebSocket feed for the
airtalk_live.html demo frontend.

    python airtalk_server.py                      # same offline eval as dtw_eval.py
    python airtalk_server.py --by session          # group by collection run instead of by day
    python airtalk_server.py --live --variant B    # recognise live + drive the frontend
    python airtalk_server.py --live --no-frontend  # plain console --live, no websocket server

Offline metrics are identical to dtw_eval.py (LOO / same-day / cross-day, variants A-G, --mag-only,
--mag-weight) and are reused directly from it. What's new here is --live: it now also streams every
raw packet to any connected frontend (for the live "drawing" sketch) plus touch/char/caps events,
in addition to printing to the console exactly like dtw_eval.py did.

Websocket messages (JSON): touch_start, touch_end, clear, caps {on}, char {char}, and per packet
imu {touched, gx, gy, gz, x, y, mx, my, mz}. mx/my/mz are the calibrated body-frame magnetic field in
Gauss, or null when the pen has no magnetometer data or calib.json has no mag calibration.

Requires `pip install websockets` for the frontend link (falls back to console-only --live
without it, same as before).
"""

import argparse
import math
import sys
import time

import numpy as np

from airpen_common import (
    DATA_DIR,
    MAG_VARIANTS,
    MAG_WEIGHT,
    VARIANTS,
    PenStream,
    apply_mag_cal,
    build_sample,
    derive,
    features,
    load_templates,
    meta_has_mag,
    prepare_meta,
    vec,
)

# Reuse the offline eval, the DTW routine and the tap/caps timing as-is.
from dtw_eval import (  # noqa: F401  (HAVE_NUMBA re-exported for parity)
    DOUBLE_TAP_GAP_S,
    HAVE_NUMBA,
    TAP_S,
    dtw,
    evaluate,
)

try:
    import asyncio
    import json
    import threading

    import websockets

    HAVE_WEBSOCKETS = True
except ImportError:
    HAVE_WEBSOCKETS = False


# =============================================================================================
# WebSocket broadcaster -- runs its own asyncio loop on a background thread, so the existing
# synchronous PenStream loop doesn't need to change shape. `send()` is safe to call from the
# main thread at any point.
# =============================================================================================
class Broadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(5):
            print(
                "[frontend] websocket server didn't come up in time; continuing without it."
            )

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except OSError as e:
            print(
                f"[frontend] couldn't start websocket server on {self.host}:{self.port}: {e}"
            )
            self._ready.set()

    async def _serve(self):
        async def handler(ws, *_):
            self.clients.add(ws)
            try:
                async for _msg in ws:
                    pass  # frontend doesn't send anything back; just keep the socket open
            finally:
                self.clients.discard(ws)

        async with websockets.serve(handler, self.host, self.port):
            self._ready.set()
            await asyncio.Future()  # run forever

    def send(self, msg):
        if self.loop is None:
            return
        data = json.dumps(msg)
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)
        except RuntimeError:
            pass  # loop already shutting down

    async def _broadcast(self, data):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# =============================================================================================
# Per-packet streaming to the frontend.
#
# PenStream.next_touch() has hooks (on_start / on_packet / on_end), so the capture logic itself
# (thresholds, debounce, pre-touch buffer) is exactly the one dtw_eval.py / collect.py use and can't
# drift out of sync. This class turns those hooks into websocket messages plus a running 2D
# dead-reckoned trace (integrated gyro -> x,y) so the frontend can sketch the stroke live as it's
# written. The trace is for visual flavour only -- it's not what the recognizer uses; DTW below
# still runs on the same derived features as dtw_eval.py.
# =============================================================================================
def _num(v):
    """JSON-safe number: NaN/inf -> None (json.dumps would emit invalid `NaN`)."""
    return float(v) if math.isfinite(v) else None


class LiveFeed:
    def __init__(self, broadcaster, meta, trace_scale=1.0):
        self.bc = broadcaster
        self.meta = meta
        self.scale = trace_scale
        self.tx = self.ty = 0.0
        self.prev_t = None

    def on_start(self, p):
        self.tx = self.ty = 0.0
        self.prev_t = p[0]
        self.bc.send({"type": "touch_start"})

    def on_packet(self, p):
        t, touched, ax, ay, az, gx, gy, gz, mx, my, mz = p
        dt = max(0.0, t - self.prev_t) if self.prev_t is not None else 0.0
        self.prev_t = t
        if touched:
            self.tx += gy * dt * self.scale
            self.ty += -gx * dt * self.scale
        msg = {
            "type": "imu",
            "touched": bool(touched),
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "x": self.tx,
            "y": self.ty,
            "mx": None,
            "my": None,
            "mz": None,
        }
        raw = np.array([[mx, my, mz]], float)
        if meta_has_mag(self.meta) and np.isfinite(raw).all() and np.abs(raw).max() > 0:
            m = apply_mag_cal(
                raw,
                self.meta["mag_offset"],
                self.meta["mag_scale"],
                self.meta["mag_align"],
            )[0]
            msg.update(mx=_num(m[0]), my=_num(m[1]), mz=_num(m[2]))
        self.bc.send(msg)

    def on_end(self):
        self.bc.send({"type": "touch_end"})


# =============================================================================================
# Live recognition (same tap/caps/DTW logic as dtw_eval.py, plus frontend events)
# =============================================================================================
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

    broadcaster = None
    if not args.no_frontend:
        if not HAVE_WEBSOCKETS:
            print(
                "note: `pip install websockets` to drive the frontend; "
                "continuing with console-only live mode."
            )
        else:
            broadcaster = Broadcaster(args.ws_host, args.ws_port)
            broadcaster.start()
            print(
                f"Frontend server listening on ws://{args.ws_host}:{args.ws_port} "
                f"-- open airtalk_live.html, point it at that address, and connect."
            )

    stream = PenStream(args.port)
    meta = prepare_meta(stream, args.data, False, not args.no_rebias)
    if need_mag and not meta_has_mag(meta):
        sys.exit(
            f"Variant {v} needs a magnetometer calibration, but calib.json has none. "
            "Run `python collect.py --calibrate-mag` first."
        )
    feed = LiveFeed(broadcaster, meta) if broadcaster else None
    caps = False
    pending_tap_t = None
    print(
        f"\nLoaded {len(y)} templates ({len(set(y))} classes), variant {v}. "
        f"Write a character, or double-tap to toggle caps. Ctrl+C to quit.\n"
    )
    print(f"  [caps: {'ON ' if caps else 'off'}]")
    if broadcaster:
        broadcaster.send({"type": "clear"})
        broadcaster.send({"type": "caps", "on": caps})

    try:
        while True:
            if feed:
                cap = stream.next_touch(
                    release_ms=args.release_ms,
                    on_start=feed.on_start,
                    on_packet=feed.on_packet,
                    on_end=feed.on_end,
                )
            else:
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
                    if broadcaster:
                        broadcaster.send({"type": "caps", "on": caps})
                else:
                    pending_tap_t = now
                continue
            pending_tap_t = None

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
            if broadcaster:
                broadcaster.send({"type": "char", "char": out})
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
        help="offline eval on only the samples that have a calibrated magnetometer (enables E-G)",
    )
    ap.add_argument(
        "--mag-weight",
        type=float,
        default=MAG_WEIGHT,
        help="weight of the magnetometer channels in variant E (1.0 = same as accel/gyro groups)",
    )
    ap.add_argument("--port", type=int, default=5005, help="UDP port for the pen")
    ap.add_argument("--no-rebias", action="store_true")
    ap.add_argument("--release-ms", type=float, default=80.0)
    ap.add_argument(
        "--ws-host", default="0.0.0.0", help="websocket host for the frontend"
    )
    ap.add_argument(
        "--ws-port", type=int, default=8765, help="websocket port for the frontend"
    )
    ap.add_argument(
        "--no-frontend",
        action="store_true",
        help="skip the websocket server; behaves like plain dtw_eval.py --live",
    )
    args = ap.parse_args()
    live(args) if args.live else evaluate(args)


if __name__ == "__main__":
    main()
