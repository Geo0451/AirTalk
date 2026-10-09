#!/usr/bin/env python3
"""DTW 1-NN baseline for air-written characters, with a live WebSocket feed for the
airtalk_live.html frontend, and a gesture mode for editing.

    python airtalk_server.py                      # same offline eval as dtw_eval.py
    python airtalk_server.py --by session          # group by collection run instead of by day
    python airtalk_server.py --live --variant B    # recognise live + drive the frontend
    python airtalk_server.py --live --no-frontend  # console only, no websocket server
    python airtalk_server.py --live --no-gestures  # characters only, no gesture mode

Offline metrics are identical to dtw_eval.py (LOO / same-day / cross-day, variants A-G, --mag-only,
--mag-weight) and are reused directly from it.

--live has two modes, both driven by the single touch pad:

  WRITE mode (default)  touch, write a character, release -> it is recognised and typed.
  GESTURE mode          quick TAP, release, then touch again quickly and HOLD (the second touch
                        must start within --gesture-gap of the tap's release and last past
                        --gesture-confirm). You stay in gesture mode until you let go, and can do
                        as many gestures as you like during that one hold:

      flick up      -> UPPERCASE on           flick left   -> backspace (delete char before cursor)
      flick down    -> lowercase on           flick right  -> insert a space
      thrust forward-> Enter (newline)
      twist anticlockwise -> cursor back      twist clockwise -> cursor forward   (a step per
                                                --twist-deg degrees, so a long twist moves several)

  Up/down only SET the case (flicking up twice is the same as once). Case applies to the characters
  typed afterwards. This replaces the old double-tap caps toggle.

Gesture detection is deliberately simple and tunable (everything is a flag): rates are measured in a
pen-relative frame built from gravity and the calibrated pen axis -- "forward" is where the tip
points, "up" is world up -- so it works at any wrist angle. After each gesture there is a short
lock-out (until the pen is quiet again) so the return stroke of a flick isn't read as the opposite
gesture; twists likewise ignore the untwist for ~0.5 s. First time on real hardware, watch the
live gesture gauges in the frontend and check the signs (--flip-ud / --flip-lr / --twist-sign) and
thresholds (--flick-dps / --thrust-g / --twist-min-dps).

Websocket messages (JSON), server -> frontend:
  touch_start, touch_end, clear, caps {on}, char {char},
  imu {touched, gx, gy, gz, x, y, mx, my, mz}   (WRITE mode only; mx/my/mz = calibrated field in
                                                 Gauss or null)
  mode {mode: "write"|"gesture"}
  gest {ud, lr, fw, tw, tw_deg, tw_step, state}  (GESTURE mode, every packet: live gesture scores,
                                                  1.0 = detection threshold; ud>0 = up, lr>0 = right,
                                                  fw>0 = forward, tw>0 = clockwise)
  gesture {name, label, score}                   (a gesture fired)
  text {text, cursor}                            (authoritative text buffer; sent after every change
                                                  and, with caps/mode, to each newly connected client)

Requires `pip install websockets` for the frontend link (falls back to console-only --live
without it).
"""

import argparse
import math
import sys

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

# Reuse the offline eval, the DTW routine and the tap timing as-is.
from dtw_eval import (  # noqa: F401  (HAVE_NUMBA re-exported for parity)
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
# main thread at any point. `snapshot` (optional callable -> list of messages) is sent to every
# client right after it connects, so a page opened/reloaded mid-session shows the current text,
# case and mode.
# =============================================================================================
class Broadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.snapshot = None
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
                if self.snapshot:
                    for m in self.snapshot():
                        await ws.send(json.dumps(m))
                async for _msg in ws:
                    pass  # frontend doesn't send anything back; just keep the socket open
            except Exception:
                pass
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


class NoBroadcaster:
    """Stand-in when there is no frontend, so the rest of the code never has to check."""

    snapshot = None

    def send(self, msg):
        pass


# =============================================================================================
# Text buffer (server-owned so console-only mode works and the frontend is just a view)
# =============================================================================================
class Typist:
    def __init__(self):
        self.chars = []
        self.cur = 0
        self.caps = False

    def put(self, s):
        for ch in s:
            self.chars.insert(self.cur, ch)
            self.cur += 1

    def backspace(self):
        if self.cur == 0:
            return False
        del self.chars[self.cur - 1]
        self.cur -= 1
        return True

    def move(self, d):
        new = min(max(self.cur + d, 0), len(self.chars))
        moved = new != self.cur
        self.cur = new
        return moved

    def snapshot(self):
        return {"type": "text", "text": "".join(self.chars), "cursor": self.cur}

    def show(self):
        s = "".join(self.chars).replace("\n", "\u23ce")
        return f"  text: {s[: self.cur]}|{s[self.cur :]}"


# =============================================================================================
# Gesture detector
# =============================================================================================
GESTURE_LABELS = {
    "flick_up": "UPPERCASE",
    "flick_down": "lowercase",
    "flick_left": "BACKSPACE",
    "flick_right": "SPACE",
    "thrust": "ENTER",
    "twist_ccw": "CURSOR \u2190",
    "twist_cw": "CURSOR \u2192",
}


class GestureDetector:
    """Turns the raw IMU stream into discrete gesture events.

    Frame (rebuilt every packet): u = world up in body coords (low-passed accel, frozen while the
    pen is moving hard), p = calibrated pen axis (toward the tip), f = horizontal direction the
    tip points, l = u x f (the writer's left). (p, l, p x l) is an orthonormal basis, so the three
    rotations below don't leak into each other however the pen is tilted:
        yaw   = gyro . (p x l)  (+ = tip swings left)     -> flick left / right
        pitch = gyro . l        (+ = tip pitches down)    -> flick down / up
        roll  = gyro . p        (+ = clockwise seen from behind the pen, looking toward the tip;
                                 rotating about p doesn't move the tip at all)
        thrust= (accel - gravity) . normalize(f + p)      -> forward push
    Flicks and thrust use a short decision window (WIN_S) after the first threshold crossing, then the
    strongest channel wins if it clearly beats the runner-up (DOM). A lock-out follows until the pen
    is quiet again. Twist is a separate dial: roll angle is integrated while the twist is fast and
    dominant, one cursor step per `twist_deg`; the opposite direction is ignored for TW_OPP_S after a
    step so the untwist doesn't undo it.
    """

    WIN_S = 0.05
    DOM = 1.3
    QUIET_DPS = 70.0
    QUIET_G = 0.35
    QUIET_S = 0.08
    MIN_LOCK_S = 0.30
    MAX_LOCK_S = 1.0
    TW_DOM = 1.6
    TW_OPP_S = 0.5
    ARM_LOCK_S = 0.10

    def __init__(self, meta, args):
        self.cfg = args
        self.axis = np.asarray(meta["pen_axis"], float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.bias = np.asarray(meta["gyro_bias_dps"], float)
        self.acc_scale = float(meta["acc_scale"])
        self.ud = -1.0 if args.flip_ud else 1.0
        self.lr = -1.0 if args.flip_lr else 1.0
        self.reset([])

    # -- lifecycle ----------------------------------------------------------------------------
    def reset(self, seed):
        """Start tracking (events stay off until arm()). `seed` = recent packets for gravity."""
        acc = [np.array(q[2:5], float) / self.acc_scale for q in seed]
        calm = [
            a
            for a, q in zip(acc, seed)
            if np.linalg.norm(np.array(q[5:8]) - self.bias) < 100.0
        ]
        use = calm or acc
        self.g = np.mean(use, 0) if use else None
        self.f = None
        self.prev_t = None
        self.active_from = math.inf
        self.state = "lockout"
        self.lock_start = self.lock_until = 0.0
        self.quiet_since = None
        self.win_end = 0.0
        self.peaks = {}
        self.tw_acc = 0.0
        self.tw_last_ok = 0.0
        self.tw_dir = 0
        self.tw_opp_until = 0.0
        self.tw_block_until = 0.0
        self.flick_block_until = 0.0

    def arm(self, t):
        self.active_from = t
        self.state = "lockout"
        self.lock_start = t
        self.lock_until = t + self.ARM_LOCK_S
        self.quiet_since = None
        self.tw_acc = 0.0

    def _lock(self, t):
        self.state = "lockout"
        self.lock_start = t
        self.lock_until = t + self.MIN_LOCK_S
        self.quiet_since = None
        self.tw_acc = 0.0
        self.tw_block_until = t + 0.35

    # -- per packet ---------------------------------------------------------------------------
    def feed(self, p):
        """p = (t, touched, ax..gz, mx..mz). Returns ([(name, score)], debug dict)."""
        cfg = self.cfg
        t = p[0]
        a = np.array(p[2:5], float) / self.acc_scale
        w = np.array(p[5:8], float) - self.bias
        dt = 0.01 if self.prev_t is None else min(max(t - self.prev_t, 0.0), 0.05)
        self.prev_t = t
        wn = float(np.linalg.norm(w))
        if self.g is None:
            self.g = a.copy()
        if wn < 150.0 and abs(np.linalg.norm(a) - 1.0) < 0.15:
            self.g += (1.0 - math.exp(-dt / 0.4)) * (a - self.g)
        u = self.g / max(float(np.linalg.norm(self.g)), 1e-6)
        lin = a - self.g

        ax = self.axis
        f = ax - (ax @ u) * u
        nf = float(np.linalg.norm(f))
        if nf > 0.25:  # pen not (nearly) vertical -> heading is defined
            self.f = f / nf
        if self.f is None:
            f = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
            self.f = f / np.linalg.norm(f)
        f = self.f - (self.f @ u) * u
        f /= max(float(np.linalg.norm(f)), 1e-6)
        l = np.cross(u, f)
        yaw, pitch, roll = float(w @ np.cross(ax, l)), float(w @ l), float(w @ ax)
        d = f + ax
        d /= max(float(np.linalg.norm(d)), 1e-6)
        thrust = float(lin @ d)

        up_rate = -pitch * self.ud  # + = up
        right_rate = -yaw * self.lr  # + = right
        tw_rate = roll * cfg.twist_sign  # + = clockwise
        T, Tt = cfg.flick_dps, cfg.thrust_g
        sc = {
            "flick_up": max(0.0, up_rate) / T,
            "flick_down": max(0.0, -up_rate) / T,
            "flick_left": max(0.0, -right_rate) / T,
            "flick_right": max(0.0, right_rate) / T,
            "thrust": max(0.0, thrust) / Tt,
        }

        events = []
        if t >= self.active_from:
            if self.state == "lockout":
                if wn < self.QUIET_DPS and float(np.linalg.norm(lin)) < self.QUIET_G:
                    if self.quiet_since is None:
                        self.quiet_since = t
                else:
                    self.quiet_since = None
                quiet_ok = (
                    self.quiet_since is not None
                    and t - self.quiet_since >= self.QUIET_S
                )
                if t >= self.lock_until and (
                    quiet_ok or t - self.lock_start >= self.MAX_LOCK_S
                ):
                    self.state = "armed"
                    self.tw_acc = 0.0
            elif self.state == "armed":
                if t >= self.flick_block_until and max(sc.values()) >= 1.0:
                    self.state = "window"
                    self.win_end = t + self.WIN_S
                    self.peaks = dict(sc)
            if self.state == "window":
                for k, v in sc.items():
                    self.peaks[k] = max(self.peaks[k], v)
                if t >= self.win_end:
                    (n1, s1), (_, s2) = sorted(
                        self.peaks.items(), key=lambda kv: -kv[1]
                    )[:2]
                    if s1 >= 1.0 and s1 >= self.DOM * s2:
                        events.append((n1, s1))
                    self._lock(t)
            if self.state == "armed" and t >= self.tw_block_until:
                self._twist(t, dt, tw_rate, yaw, pitch, events)
            else:
                self.tw_acc = 0.0

        dbg = {
            "type": "gest",
            "ud": round(up_rate / T, 3),
            "lr": round(right_rate / T, 3),
            "fw": round(thrust / Tt, 3),
            "tw": round(tw_rate / cfg.twist_min_dps, 3),
            "tw_deg": round(self.tw_acc, 1),
            "tw_step": cfg.twist_deg,
            "state": self.state if t >= self.active_from else "warmup",
        }
        return events, dbg

    def _twist(self, t, dt, tw_rate, yaw, pitch, events):
        cfg = self.cfg
        other = math.hypot(
            yaw, pitch
        )  # everything that is NOT a roll about the pen axis
        s = 1 if tw_rate > 0 else -1
        ok = abs(tw_rate) >= cfg.twist_min_dps and abs(tw_rate) >= self.TW_DOM * other
        if ok and self.tw_dir and s != self.tw_dir and t < self.tw_opp_until:
            ok = False  # the untwist after a step: ignore
        if ok:
            if s != (1 if self.tw_acc > 0 else -1 if self.tw_acc < 0 else s):
                self.tw_acc = 0.0
            self.tw_acc += tw_rate * dt
            self.tw_last_ok = t
            if abs(self.tw_acc) >= cfg.twist_deg:
                step_dir = 1 if self.tw_acc > 0 else -1
                events.append(
                    (
                        "twist_cw" if step_dir > 0 else "twist_ccw",
                        abs(self.tw_acc) / cfg.twist_deg,
                    )
                )
                self.tw_acc -= step_dir * cfg.twist_deg
                self.tw_dir = step_dir
                self.tw_opp_until = t + self.TW_OPP_S
                self.flick_block_until = t + 0.25
            elif self.tw_dir == s:
                self.tw_opp_until = max(self.tw_opp_until, t + self.TW_OPP_S)
        elif t - self.tw_last_ok > 0.15:
            self.tw_acc = 0.0


# =============================================================================================
# One touch, from touch-down to release (driven by PenStream.next_touch's hooks, so the capture
# logic is exactly the one dtw_eval.py / collect.py use).
#
# At touch-down we know whether a TAP just happened (begin(tap_end)); if the new touch starts within
# --gesture-gap of that tap it is a *candidate* gesture session. It only becomes one after being held
# --gesture-confirm, so a plain double-tap doesn't flash gesture mode; until then its messages are
# buffered (and flushed as an ordinary touch if it's released early). Otherwise the touch is a normal
# stroke: stream imu + a dead-reckoned 2D trace to the frontend (visual only; recognition uses DTW
# on the same derived features as dtw_eval.py).
# =============================================================================================
def _num(v):
    """JSON-safe number: NaN/inf -> None (json.dumps would emit invalid `NaN`)."""
    return float(v) if math.isfinite(v) else None


class Session:
    def __init__(self, bc, meta, stream, typist, det, args, trace_scale=1.0):
        self.bc, self.meta, self.stream, self.typist = bc, meta, stream, typist
        self.det, self.args, self.scale = det, args, trace_scale
        self.active = False
        self.begin(None)

    def begin(self, tap_end):
        self.tap_end = tap_end
        self.active = False
        self.cand = False
        self.buf = []
        self.start_t = None

    def _send(self, msg):
        if self.cand and not self.active:
            self.buf.append(msg)
        else:
            self.bc.send(msg)

    def _imu(self, p):
        t, touched, ax, ay, az, gx, gy, gz, mx, my, mz = p
        dt = max(0.0, t - self.prev_t)
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
        return msg

    # -- hooks --------------------------------------------------------------------------------
    def on_start(self, p):
        self.start_t = p[0]
        self.tx = self.ty = 0.0
        self.prev_t = p[0]
        a = self.args
        self.cand = bool(
            a.gestures
            and self.tap_end is not None
            and p[0] - self.tap_end <= a.gesture_gap
        )
        if self.cand:
            self.det.reset(list(self.stream.ring)[-30:])
        self._send({"type": "touch_start"})

    def on_packet(self, p):
        if self.cand:
            events, dbg = self.det.feed(p)
            # Confirm only while the pad is actually touched: next_touch keeps delivering packets
            # during the release debounce, which must not count as "held".
            if (
                not self.active
                and p[1]
                and p[0] - self.start_t >= self.args.gesture_confirm
            ):
                self.active = True
                self.buf.clear()
                self.det.arm(p[0])
                self.bc.send({"type": "mode", "mode": "gesture"})
                print(
                    "  [GESTURE MODE on]  flick up/down = case, left = backspace, right = space, "
                    "thrust = enter, twist = cursor. Release to go back to writing."
                )
            if self.active:
                self.bc.send(dbg)
                if p[
                    1
                ]:  # ignore anything that "happens" after the finger has left the pad
                    for name, score in events:
                        self.handle(name, score)
            else:
                self._send(self._imu(p))
            return
        self.bc.send(self._imu(p))

    def on_end(self):
        if self.active:
            self.bc.send({"type": "mode", "mode": "write"})
            print("  [gesture mode off]")
        else:
            for m in self.buf:
                self.bc.send(m)
            self.buf = []
            self.bc.send({"type": "touch_end"})

    # -- gesture actions ----------------------------------------------------------------------
    def handle(self, name, score):
        t, bc = self.typist, self.bc
        label = GESTURE_LABELS[name]
        note = ""
        if name == "flick_up":
            t.caps = True
            bc.send({"type": "caps", "on": True})
        elif name == "flick_down":
            t.caps = False
            bc.send({"type": "caps", "on": False})
        elif name == "flick_left":
            note = "" if t.backspace() else "  (nothing to delete)"
            bc.send(t.snapshot())
        elif name == "flick_right":
            t.put(" ")
            bc.send(t.snapshot())
        elif name == "thrust":
            t.put("\n")
            bc.send(t.snapshot())
        elif name == "twist_ccw":
            note = "" if t.move(-1) else "  (at start)"
            bc.send(t.snapshot())
        elif name == "twist_cw":
            note = "" if t.move(1) else "  (at end)"
            bc.send(t.snapshot())
        bc.send(
            {"type": "gesture", "name": name, "label": label, "score": round(score, 2)}
        )
        extra = (
            f"   [caps: {'ON ' if t.caps else 'off'}]"
            if name in ("flick_up", "flick_down")
            else ""
        )
        print(f"  [gesture] {label:<10} (x{score:.1f}){note}{extra}")
        if name not in ("flick_up", "flick_down"):
            print(t.show())


# =============================================================================================
# Live recognition
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

    broadcaster = NoBroadcaster()
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

    typist = Typist()
    det = GestureDetector(meta, args)
    sess = Session(broadcaster, meta, stream, typist, det, args)
    broadcaster.snapshot = lambda: [
        {"type": "caps", "on": typist.caps},
        {"type": "mode", "mode": "gesture" if sess.active else "write"},
        typist.snapshot(),
    ]

    print(
        f"\nLoaded {len(y)} templates ({len(set(y))} classes), variant {v}. Write a character."
        + (
            "\nGesture mode: tap, release, then touch again quickly and hold. Ctrl+C to quit.\n"
            if args.gestures
            else " (gestures disabled) Ctrl+C to quit.\n"
        )
    )
    print(f"  [caps: {'ON ' if typist.caps else 'off'}]")
    broadcaster.send({"type": "clear"})
    broadcaster.send({"type": "caps", "on": typist.caps})
    broadcaster.send({"type": "mode", "mode": "write"})
    broadcaster.send(typist.snapshot())

    tap_end = None  # pen-time at which the last tap was released
    try:
        while True:
            sess.begin(tap_end)
            cap = stream.next_touch(
                release_ms=args.release_ms,
                max_s=120.0,
                on_start=sess.on_start,
                on_packet=sess.on_packet,
                on_end=sess.on_end,
            )
            if sess.active:  # that whole touch was a gesture session
                tap_end = None
                continue

            dur = cap["duration"]
            if dur < TAP_S:  # a tap: may be the first half of tap-and-hold
                tap_end = sess.start_t + dur
                continue
            tap_end = None

            if dur < 0.4:
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
            out = best[1].upper() if typist.caps else best[1].lower()
            print(
                f"  -> {out}    (x{second[0] / max(best[0], 1e-9):.2f} margin)   [{top3}]"
            )
            typist.put(out)
            broadcaster.send({"type": "char", "char": out})
            broadcaster.send(typist.snapshot())
            print(typist.show())
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

    g = ap.add_argument_group("gesture mode (--live)")
    g.add_argument(
        "--no-gestures",
        dest="gestures",
        action="store_false",
        help="disable gesture mode (every touch is a character)",
    )
    g.add_argument(
        "--gesture-gap",
        type=float,
        default=0.4,
        help="max seconds between the tap's release and the touch that holds (default 0.4)",
    )
    g.add_argument(
        "--gesture-confirm",
        type=float,
        default=0.15,
        help="seconds the second touch must be held before gesture mode starts (default 0.15)",
    )
    g.add_argument(
        "--flick-dps",
        type=float,
        default=250.0,
        help="angular rate that counts as a flick (default 250 dps; lower = more sensitive)",
    )
    g.add_argument(
        "--thrust-g",
        type=float,
        default=0.6,
        help="forward acceleration that counts as a thrust (default 0.6 g)",
    )
    g.add_argument(
        "--twist-deg",
        type=float,
        default=40.0,
        help="degrees of twist per cursor step (default 40)",
    )
    g.add_argument(
        "--twist-min-dps",
        type=float,
        default=70.0,
        help="a twist must be at least this fast to count (default 70 dps; keeps the "
        "slow untwist from registering)",
    )
    g.add_argument(
        "--twist-sign",
        type=int,
        choices=[1, -1],
        default=1,
        help="-1 swaps clockwise/anticlockwise if the cursor moves the wrong way",
    )
    g.add_argument("--flip-ud", action="store_true", help="swap flick up / down")
    g.add_argument("--flip-lr", action="store_true", help="swap flick left / right")
    args = ap.parse_args()
    live(args) if args.live else evaluate(args)


if __name__ == "__main__":
    main()
