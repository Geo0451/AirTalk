"""Shared helpers for the air-stylus tools: UDP stream, calibration, sample I/O, preprocessing.

Raw sensor data is what gets saved to disk (IMU + raw magnetometer, plus the calibration the
sample was captured under). Everything derived (orientation, gravity removal, heading alignment,
features) is recomputed from raw at load time, so you can change the preprocessing later without
re-collecting anything.

Magnetometer (firmware: 41-byte packets, mx my mz in Gauss, raw):
  * `PenStream` decodes 41-byte packets, legacy 29-byte packets (mag = NaN) and text lines.
  * `--calibrate` / `--calibrate-mag` runs a mag calibration: hard-iron offset + per-axis soft-iron
    scale (axis-aligned ellipsoid fit), and a check of the mag->body axis alignment against
    gravity (the dip angle must stay constant while the pen rotates). Stored in calib.json.
  * Samples store the raw mag plus the calibration used, so old samples stay interpretable.
  * derive() additionally runs a Madgwick MARG filter (gyro + accel + calibrated mag) which gives
    a drift-free, magnetic-north-referenced orientation; feature variants E-G use it.
"""

import asyncio
import atexit
import glob
import itertools
import json
import os
import select
import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

# ---- packet format --------------------------------------------------------------------------
PKT_FMT_V1 = (
    "<IB6f"  # legacy, 29 bytes: u32 timestamp_us, u8 touched, ax ay az gx gy gz
)
PKT_FMT_V2 = "<IB9f"  # current, 41 bytes: ... + mx my mz (Gauss, raw, uncalibrated)
PKT_SIZE_V1 = struct.calcsize(PKT_FMT_V1)
PKT_SIZE_V2 = struct.calcsize(PKT_FMT_V2)
PKT_FMT, PKT_SIZE = PKT_FMT_V2, PKT_SIZE_V2  # what the current firmware sends
# Text lines `ts,ax,ay,az,gx,gy,gz,touched[,mx,my,mz]`: multiply `ts` by this to get microseconds.
TEXT_TS_TO_US = 1.0
NAN3 = (float("nan"),) * 3
DATA_DIR = "data"
MAG_WEIGHT = (
    1.0  # relative weight of the mag channel group in variant E (others are 1.0)
)
# How long grab() waits for the first/next packet before giving up. BLE needs time to scan,
# connect, discover services and subscribe, so it gets a much longer leash than UDP.
NO_DATA_LIMIT_UDP_S = 5.0
NO_DATA_LIMIT_BLE_S = 30.0


def _looks_like_text(data):
    return len(data) > 0 and data.count(b",") >= 7 and max(data) < 128


# =============================================================================================
# UDP stream
# =============================================================================================
# ---- Bluetooth LE pen (firmware built with USE_BLE 1) ---------------------------------------
# Same 41-byte packet as the UDP stream, one per GATT notification. Needs `pip install bleak`
# (BlueZ on Linux). The UUIDs/name must match AirTalk3.ino.
BLE_NAME = "AirStylus"
BLE_SERVICE_UUID = "4f3c5a10-7b2e-4d6a-9a51-3c8e0b1a0001"
BLE_TELEMETRY_UUID = "4f3c5a11-7b2e-4d6a-9a51-3c8e0b1a0001"


class _BleSource:
    """Background thread: find the pen, connect, subscribe, and queue every notification.

    Reconnects forever. `should_scan()` lets the owner pause scanning (auto mode does not scan while
    the WiFi/UDP pen is talking, so Bluetooth can't interfere with it). Data is handed over through
    `q` plus one byte on a socketpair, so the reader can wait on UDP and BLE with a single select().

    `close()` (registered with atexit) stops the thread and lets bleak disconnect cleanly. Without
    that, a process that dies mid-connection can leave BlueZ holding the pen's single link, and the
    ESP32 stops advertising, so the next run never finds it."""

    def __init__(self, wake_w, should_scan):
        self.q = deque(
            maxlen=1000
        )  # ~10 s at 100 Hz; if the reader stalls, oldest data goes first
        self.wake_w, self.should_scan = wake_w, should_scan
        self.connected = False
        self.bad = 0
        self._last_msg = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ble-pen", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def close(self):
        """Stop the BLE thread and wait (briefly) for the clean disconnect."""
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(4.0)

    def _say_once(self, msg):
        if msg != self._last_msg:
            self._last_msg = msg
            print(f"[ble] {msg}")

    def _on_data(self, _char, data):
        data = bytes(data)
        if len(data) not in (PKT_SIZE_V2, PKT_SIZE_V1):
            self.bad += 1
            if self.bad in (1, 200):
                print(
                    f"[ble] got a {len(data)}-byte notification (expected {PKT_SIZE_V2}): the ATT MTU "
                    f"is probably too small (needs >= {PKT_SIZE_V2 + 3}). Dropping those."
                )
            return
        self.q.append(data)
        try:
            self.wake_w.send(b"\x01")
        except (
            OSError
        ):  # wake buffer full: the reader is behind and will find the data in q anyway
            pass

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:  # never take the host down because Bluetooth misbehaved
            print(f"[ble] stopped: {e}")

    async def _main(self):
        from bleak import BleakClient, BleakScanner

        def is_pen(_dev, ad):
            uuids = [u.lower() for u in (ad.service_uuids or [])]
            return BLE_SERVICE_UUID in uuids or ad.local_name == BLE_NAME

        while not self._stop.is_set():
            if not self.should_scan():
                await asyncio.sleep(1.0)
                continue
            try:
                dev = await BleakScanner.find_device_by_filter(is_pen, timeout=3.0)
            except Exception as e:
                self._say_once(
                    f"scan failed: {e}. Is Bluetooth on? (`rfkill list`, `systemctl status bluetooth`)"
                )
                await asyncio.sleep(3.0)
                continue
            if dev is None:
                self._say_once(
                    f"looking for '{BLE_NAME}' (is the pen on, flashed with USE_BLE 1? "
                    "If a previous run died, `bluetoothctl disconnect <addr>` and reset the pen)"
                )
                continue
            try:
                gone = asyncio.Event()
                async with BleakClient(
                    dev, disconnected_callback=lambda _c: gone.set(), timeout=10.0
                ) as cli:
                    await cli.start_notify(BLE_TELEMETRY_UUID, self._on_data)
                    self.connected = True
                    self._last_msg = ""
                    print(
                        f"[ble] connected to {dev.name or BLE_NAME} ({dev.address}), "
                        f"MTU {getattr(cli, 'mtu_size', '?')}, notifications on"
                    )
                    while not gone.is_set() and not self._stop.is_set():
                        try:
                            await asyncio.wait_for(gone.wait(), 0.5)
                        except asyncio.TimeoutError:
                            pass
                # leaving the `async with` disconnects the client
            except Exception as e:
                self._say_once(f"connection problem: {e}")
            if self.connected:
                print("[ble] disconnected; searching again")
            self.connected = False
            await asyncio.sleep(1.0)


class _StreamStats:
    """--stream-stats: every few seconds print the packet rate, timestamp gaps (lost/late packets)
    and how much later than the best case packets reach the host loop. Works on UDP and BLE alike, so
    the two can be compared. The delay is relative (the ESP's clock isn't synced to the host's), and
    includes whatever the host's main loop was busy with, which is the same for both transports."""

    WINDOW = 5.0

    def __init__(self, label):
        self.label, self.rows, self.t0 = label, [], None

    def reset(self):
        self.rows, self.t0 = [], None

    def add(self, now, st):
        if self.t0 is None:
            self.t0 = now
        self.rows.append((now, st))
        if now - self.t0 >= self.WINDOW:
            rows, self.rows, self.t0 = self.rows, [], now
            if len(rows) < 10:
                return
            span = rows[-1][1] - rows[0][1]
            rate = (len(rows) - 1) / span if span > 0 else 0.0
            gaps = sum(1 for a, b in zip(rows, rows[1:]) if b[1] - a[1] > 0.015)
            lag = np.array([n - t for n, t in rows])
            lag = (lag - lag.min()) * 1000.0
            p50, p95, p99, mx = np.percentile(lag, [50, 95, 99, 100])
            print(
                f"[stats {self.label() or '?':>3}] {rate:5.1f} pkt/s | gaps>15ms {gaps:3d} | "
                f"arrival delay p50 {p50:5.1f}  p95 {p95:5.1f}  p99 {p99:5.1f}  max {mx:5.1f} ms"
            )


class PenStream:
    """Reads the ESP32 packets, keeps a rolling pre-touch buffer, and cuts out touch windows.

    A packet is the tuple (t, touched, ax, ay, az, gx, gy, gz, mx, my, mz); mx..mz are NaN when
    the sender is a legacy 29-byte firmware.
    """

    def __init__(self, port=5005, pre_s=1.0, transport=None, stats=None):
        self.pre_s = pre_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", port))
        except OSError as e:
            raise SystemExit(
                f"Can't bind UDP port {port}: {e}\n"
                "Is the Godot visualizer (or another script) still running?"
            )
        self.sock.settimeout(0.25)
        self.ring = deque()
        self._pending = deque()
        self._t = 0.0
        self._last_ts = None
        self._prev_touch = (
            True  # conservative: require a release before the first touch-down
        )
        self._warned = False
        self._init_transport(transport, stats)

    # -- transport selection (BLE support) ----------------------------------------------------
    def _init_transport(self, transport, stats):
        """transport: "udp" (the original WiFi-AP stream, default), "ble", or "auto" (listen on UDP
        and look for the BLE pen at the same time; whichever one is talking is used). If None, the
        AIRPEN_TRANSPORT environment variable is used, so tools without a flag can opt in too."""
        t = (transport or os.environ.get("AIRPEN_TRANSPORT") or "udp").lower()
        if t not in ("udp", "ble", "auto"):
            raise SystemExit(f"Unknown transport '{t}' (use udp, ble or auto).")
        self.transport, self._ble, self.source, self._udp_last = t, None, None, 0.0
        want_stats = (
            stats
            if stats is not None
            else os.environ.get("AIRPEN_STREAM_STATS", "") not in ("", "0")
        )
        self._stats = (
            _StreamStats(
                lambda: self.source or ("udp" if self.transport == "udp" else None)
            )
            if want_stats
            else None
        )
        if t == "udp":
            return
        try:
            import bleak  # noqa: F401
        except ImportError:
            if t == "ble":
                raise SystemExit("BLE transport needs bleak:  pip install bleak")
            print(
                "[stream] 'bleak' isn't installed, so only the WiFi/UDP pen is supported (pip install bleak for BLE)."
            )
            self.transport = "udp"
            return
        self._wake_r, wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        wake_w.setblocking(False)
        # auto: only scan while no UDP data has arrived for 3 s, so Bluetooth scanning can never
        # disturb a working WiFi pen. ble: always.
        gate = (
            (lambda: True)
            if t == "ble"
            else (lambda: time.time() - self._udp_last > 3.0)
        )
        self._ble = _BleSource(wake_w, gate)
        print(
            "[stream] waiting for the pen over BLE"
            + (f" and UDP :{self.sock.getsockname()[1]}" if t == "auto" else "")
            + f" (BLE name '{BLE_NAME}')"
        )

    def _got(self, data, src):
        if src != self.source:
            self.source = src
            print(
                f"[stream] pen is talking over {'BLE' if src == 'ble' else 'UDP (WiFi AP)'}"
            )
        return data

    def _wait_data(self):
        """One raw datagram/notification from BLE and/or UDP, or None after 0.25 s of silence."""
        ble = self._ble
        if ble.q:
            return self._got(ble.q.popleft(), "ble")
        rl = [self._wake_r] + ([self.sock] if self.transport == "auto" else [])
        try:
            ready, _, _ = select.select(rl, [], [], 0.25)
        except (OSError, ValueError):
            return None
        if self.sock in ready:
            try:
                data, _ = self.sock.recvfrom(2048)
            except (socket.timeout, BlockingIOError):
                data = None
            if data:
                self._udp_last = time.time()
                return self._got(data, "udp")
        if self._wake_r in ready:
            try:
                self._wake_r.recv(4096)
            except BlockingIOError:
                pass
            if ble.q:
                return self._got(ble.q.popleft(), "ble")
        return None

    def _drain_multi(self):
        """drain() for ble/auto: discard queued BLE notifications and UDP datagrams (still decoded,
        so stream time stays continuous, exactly like the UDP-only drain)."""
        s, self._stats = self._stats, None
        try:
            while self._ble.q:
                self._decode(self._ble.q.popleft())
            try:
                while self._wake_r.recv(4096):
                    pass
            except BlockingIOError:
                pass
            if self.transport == "auto":
                self.sock.setblocking(False)
                try:
                    while True:
                        try:
                            data, _ = self.sock.recvfrom(2048)
                        except (BlockingIOError, InterruptedError):
                            break
                        self._decode(data)
                finally:
                    self.sock.settimeout(0.25)
        finally:
            self._stats = s
            if s:
                s.reset()
        self.ring.clear()
        self._pending.clear()
        self._prev_touch = True

    # -- low level --------------------------------------------------------------------------
    def _decode(self, data):
        """One UDP datagram -> list of packets. Handles 41-byte and 29-byte binary packets and
        text lines (8 fields = legacy, 11 fields = legacy + mx,my,mz)."""
        rows = []  # (ts_us, touched, [ax ay az gx gy gz mx my mz])
        if len(data) in (PKT_SIZE_V2, PKT_SIZE_V1) and not _looks_like_text(data):
            if len(data) == PKT_SIZE_V2:
                ts, touched, *vals = struct.unpack(PKT_FMT_V2, data)
            else:
                ts, touched, *vals = struct.unpack(PKT_FMT_V1, data)
                vals = vals + list(NAN3)
            rows.append((ts, touched, vals))
        elif _looks_like_text(data):
            for line in data.decode("ascii", "ignore").splitlines():
                f = line.strip().split(",")
                if len(f) not in (8, 11):
                    continue
                try:
                    n = [float(x) for x in f]
                    ts = int(n[0] * TEXT_TS_TO_US) & 0xFFFFFFFF
                except (ValueError, OverflowError):
                    continue
                rows.append(
                    (
                        ts,
                        int(n[7] != 0),
                        n[1:7] + (n[8:11] if len(f) == 11 else list(NAN3)),
                    )
                )
        elif not self._warned:
            print(
                f"[stream] ignoring packet of {len(data)} bytes "
                f"(expected {PKT_SIZE_V2}, legacy {PKT_SIZE_V1}, or a text line)"
            )
            self._warned = True

        out = []
        for ts, touched, vals in rows:
            if self._last_ts is not None:
                d = (ts - self._last_ts) & 0xFFFFFFFF  # survives u32 wraparound
                self._t += (
                    d if d < 5_000_000 else 10_000
                ) / 1e6  # huge jump = ESP reboot
            self._last_ts = ts
            out.append((self._t, bool(touched), *vals))
        if self._stats and out:
            self._stats.add(time.perf_counter(), out[-1][0])
        return out

    def _read(self):
        if self._pending:
            return self._pending.popleft()
        if self._ble is not None:  # ble / auto transport
            data = self._wait_data()
            if data is None:
                return None
            self._pending.extend(self._decode(data))
            return self._pending.popleft() if self._pending else None
        try:
            data, _ = self.sock.recvfrom(2048)
        except socket.timeout:
            return None
        self._pending.extend(self._decode(data))
        return self._pending.popleft() if self._pending else None

    def drain(self):
        """Throw away whatever is queued in the OS socket buffer (stale data)."""
        if self._ble is not None:
            return self._drain_multi()
        _st, self._stats = (
            self._stats,
            None,
        )  # drained packets shouldn't count in --stream-stats
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    data, _ = self.sock.recvfrom(2048)
                except (BlockingIOError, InterruptedError):
                    break
                self._decode(data)
        finally:
            self.sock.settimeout(0.25)
        self._stats = _st
        if _st:
            _st.reset()
        self.ring.clear()
        self._pending.clear()
        self._prev_touch = True

    def _no_data_limit(self):
        return NO_DATA_LIMIT_BLE_S if self._ble is not None else NO_DATA_LIMIT_UDP_S

    def _no_data_hint(self):
        if self._ble is not None:
            return (
                "Is the pen on? BLE: check `bluetoothctl devices`; a stale connection left by a "
                "previous run blocks advertising (`bluetoothctl disconnect <addr>`, then reset the "
                "pen). Don't connect it from the GNOME Bluetooth panel."
            )
        return "Is the ESP32 on and on the same network?"

    # -- fixed-duration capture (calibration) ------------------------------------------------
    def grab(self, seconds):
        """Return (N,9) raw [ax ay az gx gy gz mx my mz] covering `seconds` of stream time."""
        out, t0, last = [], None, time.time()
        while True:
            p = self._read()
            if p is None:
                limit = self._no_data_limit()
                if time.time() - last > limit:
                    raise SystemExit(
                        f"No packets for {limit:.0f} s. {self._no_data_hint()}"
                    )
                continue
            last = time.time()
            t0 = p[0] if t0 is None else t0
            out.append(p[2:])
            if p[0] - t0 >= seconds:
                break
        self.ring.clear()
        self._prev_touch = True
        return np.array(out)

    def wait_still(self, seconds=1.5, gyro_std_max=2.0, accel_rel_std_max=0.03):
        """Block until the pen has been still for `seconds`; return that (N,9) window."""
        self.drain()
        shown = False
        while True:
            a = self.grab(seconds)
            an = np.linalg.norm(a[:, :3], axis=1)
            if (
                a[:, 3:6].std(0).max() < gyro_std_max
                and an.std() / max(an.mean(), 1e-9) < accel_rel_std_max
            ):
                return a
            if not shown:
                print("  ...waiting for the pen to be still")
                shown = True

    # -- touch capture -----------------------------------------------------------------------
    def next_touch(
        self, release_ms=80.0, max_s=10.0, on_start=None, on_packet=None, on_end=None
    ):
        """Block until one full touch (down -> up) has been captured.

        The touch flag is debounced: it must stay low for `release_ms` to count as released.
        Returns dict(t, raw, mag, touch, start, end, duration). Data includes ~pre_s of pre-touch
        samples; [start:end] indexes the touched part.

        Optional hooks (used by the websocket server): on_start(p) on the touch-down packet,
        on_packet(p) for every later packet of the capture, on_end() just before returning.
        """
        cap, start, last_on = None, 0, 0
        last_pkt = time.time()
        while True:
            p = self._read()
            if p is None:
                if time.time() - last_pkt > 5:
                    print(f"[stream] no packets for 5 s. {self._no_data_hint()}")
                    last_pkt = time.time()
                continue
            last_pkt = time.time()
            rising = p[1] and not self._prev_touch
            self._prev_touch = p[1]
            if cap is None:
                self.ring.append(p)
                while self.ring[-1][0] - self.ring[0][0] > self.pre_s:
                    self.ring.popleft()
                if rising:
                    cap = list(self.ring)
                    start = last_on = len(cap) - 1
                    if on_start:
                        on_start(p)
                continue
            cap.append(p)
            if on_packet:
                on_packet(p)
            if p[1]:
                last_on = len(cap) - 1
            if (
                p[0] - cap[last_on][0] >= release_ms / 1000
                or p[0] - cap[start][0] > max_s
            ):
                tail = cap[-1][0]
                self.ring = deque(q for q in cap if q[0] >= tail - self.pre_s)
                if on_end:
                    on_end()
                return self._pack(cap[: last_on + 1], start)

    @staticmethod
    def _pack(cap, start):
        a = np.array(
            cap, dtype=np.float64
        )  # cols: t, touch, ax ay az gx gy gz, mx my mz
        t = a[:, 0] - a[0, 0]
        return {
            "t": t,
            "raw": a[:, 2:8].astype(np.float32),
            "mag": a[:, 8:11].astype(np.float32),
            "touch": a[:, 1].astype(np.uint8),
            "start": start,
            "end": len(cap),
            "duration": float(t[-1] - t[start]),
        }


# =============================================================================================
# Magnetometer calibration
# =============================================================================================
def mag_present(mag):
    """True if a (N,3) raw mag block holds real data (finite, not all zero)."""
    mag = np.asarray(mag, float)
    return (
        mag.size > 0 and bool(np.isfinite(mag).all()) and float(np.abs(mag).max()) > 0.0
    )


def meta_has_mag(meta):
    return "mag_offset" in meta and "mag_scale" in meta and "mag_align" in meta


def apply_mag_cal(mag, offset, scale, align):
    """Raw sensor-frame mag (N,3) -> calibrated, body-frame mag in Gauss."""
    m = (np.asarray(mag, float) - np.asarray(offset, float)) * np.asarray(scale, float)
    return m @ np.asarray(align, float).T


def _rotations24():
    """The 24 proper axis permutations/sign flips. body[i] = sign[i] * sensor[perm[i]] -- exactly
    the firmware's MAG_AXIS / MAG_SIGN convention."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = signs[i]
            if np.linalg.det(R) > 0:
                out.append((R, perm, signs))
    return out


def _fit_ellipsoid_axis_aligned(m):
    """Returns (center, radii, method). Axis-aligned ellipsoid LS fit, sphere fit as fallback."""
    x, y, z = m.T
    try:
        p, *_ = np.linalg.lstsq(
            np.stack([x * x, y * y, z * z, x, y, z], 1), np.ones(len(m)), rcond=None
        )
        A, B = p[:3], p[3:]
        if (A > 0).all():
            c = -B / (2 * A)
            k = 1.0 + float((A * c * c).sum())
            if k > 0:
                r = np.sqrt(k / A)
                if r.max() / r.min() < 2.5 and 0.05 < r.mean() < 5.0:
                    return c, r, "ellipsoid"
    except np.linalg.LinAlgError:
        pass
    p, *_ = np.linalg.lstsq(np.c_[2 * m, np.ones(len(m))], (m**2).sum(1), rcond=None)
    c = p[:3]
    r = float(np.sqrt(max(p[3] + c @ c, 1e-9)))
    return c, np.full(3, r), "sphere"


def fit_mag_calibration(mag, acc=None, gyr_dps=None, acc_scale=1.0):
    """Fit hard-iron offset + per-axis soft-iron scale from a tumble through all orientations, then
    (if accel is given) check/repair the mag->body axis alignment using the dip angle.

    Returns (cal, report). `cal` holds mag_offset / mag_scale / mag_align / mag_field (what goes in
    calib.json); `report["ok"]` says whether the result is trustworthy.
    """
    mag = np.asarray(mag, float)
    keep = np.isfinite(mag).all(1) & (np.abs(mag).max(1) > 0)
    m = mag[keep]
    n = len(m)
    rep = {"n": int(n), "ok": False}
    if n < 200:
        rep["why"] = f"only {n} usable samples"
        return None, rep

    c, r, method = _fit_ellipsoid_axis_aligned(m)
    field = float(r.mean())
    scale = field / r
    mc = (m - c) * scale
    nrm = np.linalg.norm(mc, axis=1)
    raw_n = np.linalg.norm(m, axis=1)
    rep.update(
        method=method,
        field_gauss=field,
        offset=c.tolist(),
        scale=scale.tolist(),
        sphericity_before=float(raw_n.std() / raw_n.mean()),
        sphericity_after=float(nrm.std() / nrm.mean()),
    )

    # coverage: the fit needs every axis direction (+-X, +-Y, +-Z) visited, i.e. the calibrated field
    # vector must have pointed to within ~50 deg of each of the six poles at some point
    u = mc / np.maximum(nrm[:, None], 1e-9)
    rep["coverage"] = int(
        sum((u[:, i] * sg > 0.6).any() for i in range(3) for sg in (1, -1))
    )

    # axis alignment vs gravity
    R_best, perm, signs = np.eye(3), (0, 1, 2), (1, 1, 1)
    rep.update(aligned=False, dip_std_identity=None, dip_std_final=None, dip_n=0)
    if acc is not None and gyr_dps is not None:
        acc = np.asarray(acc, float)[keep] / acc_scale
        gyr = np.asarray(gyr_dps, float)[keep]
        gate = (np.abs(np.linalg.norm(acc, axis=1) - 1.0) < 0.08) & (
            np.linalg.norm(gyr, axis=1) < 100.0
        )
        rep["dip_n"] = int(gate.sum())
        if gate.sum() >= 150:
            ah = acc[gate] / np.linalg.norm(acc[gate], axis=1, keepdims=True)
            mh = mc[gate] / np.linalg.norm(mc[gate], axis=1, keepdims=True)

            def dip_std(R):
                ct = np.clip(np.einsum("ni,ni->n", ah, mh @ R.T), -1, 1)
                return float(np.degrees(np.arccos(ct)).std())

            s_id = dip_std(np.eye(3))
            best = min(_rotations24(), key=lambda t: dip_std(t[0]))
            s_best = dip_std(best[0])
            rep["dip_std_identity"] = rep["dip_std_final"] = s_id
            if s_id > 4.0 and s_best < 0.7 * s_id:
                R_best, perm, signs = best
                rep.update(aligned=True, dip_std_final=s_best)
        else:
            rep["dip_note"] = "too little slow motion to check axis alignment"
    rep["perm"], rep["signs"] = [int(i) for i in perm], [int(i) for i in signs]

    why = []
    if rep["coverage"] < 6:
        why.append(
            f"only {rep['coverage']}/6 axis directions visited: rotate through more orientations "
            "(tip up AND down, both sides, flipped)"
        )
    if rep["sphericity_after"] > 0.08:
        why.append(
            f"field strength still varies {100 * rep['sphericity_after']:.1f}% after calibration "
            "(interference nearby, or not enough rotation)"
        )
    if rep["dip_std_final"] is not None and rep["dip_std_final"] > 8.0:
        why.append(
            f"dip angle varies by {rep['dip_std_final']:.1f} deg: mag and IMU axes disagree and "
            "no simple remap fixes it (or the pen was moved too fast)"
        )
    rep["why"] = "; ".join(why)
    rep["ok"] = not why

    cal = {
        "mag_offset": c.tolist(),
        "mag_scale": scale.tolist(),
        "mag_align": R_best.tolist(),
        "mag_field": field,
    }
    return cal, rep


def _print_mag_report(rep):
    print(
        f"  samples: {rep['n']} | fit: {rep.get('method', '-')} | field strength: "
        f"{rep.get('field_gauss', 0):.3f} G"
    )
    if "sphericity_before" in rep:
        print(
            f"  |B| variation through the tumble: {100 * rep['sphericity_before']:.1f}% raw -> "
            f"{100 * rep['sphericity_after']:.1f}% calibrated | axis directions visited {rep['coverage']}/6"
        )
    if rep.get("dip_std_identity") is not None:
        line = f"  dip angle (mag vs gravity) std: {rep['dip_std_identity']:.1f} deg with identity axes"
        if rep["aligned"]:
            p, sg = rep["perm"], rep["signs"]
            line += (
                f" -> {rep['dip_std_final']:.1f} deg after remapping.\n"
                f"  Mag axes are rotated vs the MPU6050; applied on the host. Firmware equivalent: "
                f"MAG_AXIS={{{p[0]},{p[1]},{p[2]}}}, MAG_SIGN={{{sg[0]}.0f,{sg[1]}.0f,{sg[2]}.0f}} "
                f"(then re-run this calibration)"
            )
        print(line)
    elif rep.get("dip_note"):
        print(f"  ({rep['dip_note']})")
    print("  " + ("OK" if rep["ok"] else f"NOT GOOD ENOUGH: {rep.get('why', '')}"))


def run_mag_calibration(stream, meta, seconds=20.0):
    """Interactive mag calibration. Returns the updated meta (unchanged if skipped/declined)."""
    print("\n=== Magnetometer calibration ===")
    print(
        "  Keep away from monitors, speakers, phones and big metal objects. Battery and wiring must\n"
        "  sit exactly as they will during use (they are part of what gets calibrated out)."
    )
    while True:
        input(
            f"  Press Enter, then slowly tumble the pen through EVERY orientation for {seconds:.0f} s\n"
            "  (tip up, tip down, on each side, twists, big figure-8s). Not too fast. "
        )
        stream.drain()
        print("  GO...")
        raw = stream.grab(seconds)
        mag = raw[:, 6:9]
        if not mag_present(mag) or float(np.ptp(mag, axis=0).max()) < 0.05:
            print(
                "  No usable magnetometer data (mx,my,mz are 0/NaN or never change). "
                "Skipping mag calibration; mag variants E-G will be unavailable."
            )
            return meta
        gyr = raw[:, 3:6] - np.asarray(meta["gyro_bias_dps"], float)
        cal, rep = fit_mag_calibration(mag, raw[:, :3], gyr, meta["acc_scale"])
        if cal is None:
            print(f"  Calibration failed: {rep.get('why')}")
        else:
            _print_mag_report(rep)
            if rep["ok"]:
                break
        if input("  Retry? [Y/n] ").strip().lower() in ("n", "no"):
            print(
                "  Keeping the previous mag calibration."
                if meta_has_mag(meta)
                else "  No mag calibration saved; mag variants E-G will be unavailable."
            )
            return meta
    meta.update(cal)
    keep = (
        "n",
        "method",
        "field_gauss",
        "sphericity_before",
        "sphericity_after",
        "coverage",
        "dip_std_identity",
        "dip_std_final",
        "aligned",
    )
    meta["mag_cal_report"] = {k: rep[k] for k in keep}
    meta["mag_cal_created"] = datetime.now().isoformat(timespec="seconds")
    return meta


# =============================================================================================
# Calibration (finds the pen axis for you, so you don't need to know how the IMU is mounted)
# =============================================================================================
def calib_path(data_dir):
    return os.path.join(data_dir, "calib.json")


def save_meta(data_dir, meta):
    os.makedirs(data_dir, exist_ok=True)
    with open(calib_path(data_dir), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {calib_path(data_dir)}\n")


def axis_name(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "XYZ"[i]


def run_calibration(stream, data_dir):
    print("\n=== One-time calibration ===")
    input("Step 1/4: lay the pen on the table and leave it. Press Enter... ")
    still = stream.wait_still(3.0)
    bias = still[:, 3:6].mean(0)
    scale = float(np.linalg.norm(still[:, :3], axis=1).mean())
    print(f"  gyro bias (dps): {np.round(bias, 3)} | 1 g = {scale:.3f} raw accel units")

    input(
        "Step 2/4: hold the pen. Press Enter, then twist it back and forth around its LONG axis\n"
        "         (like a screwdriver) for 5 seconds. "
    )
    stream.drain()
    print("  GO...")
    g = stream.grab(5.0)[:, 3:6] - bias
    w, V = np.linalg.eigh(g.T @ g)
    axis, purity = V[:, -1], float(w[-1] / w.sum())
    print(f"  dominant rotation axis purity: {purity:.2f}")
    if purity < 0.6:
        print(
            "  WARNING: rotation wasn't clean. Re-run with `--calibrate` and twist more purely."
        )

    input(
        "Step 3/4: point the pen TIP at the ceiling and hold it still. Press Enter... "
    )
    still2 = stream.wait_still(1.5, gyro_std_max=4.0)
    d = float((still2[:, :3].mean(0) / scale) @ axis)
    if abs(d) < 0.7:
        print("  WARNING: pen doesn't look vertical. Re-run with `--calibrate`.")
    axis = axis * (1.0 if d >= 0 else -1.0)
    print(
        f"  pen axis (points to tip) in sensor frame: {np.round(axis, 3)}  (~{axis_name(axis)})"
    )

    meta = {
        "pen_axis": axis.tolist(),
        "acc_scale": scale,
        "gyro_bias_dps": bias.tolist(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    print("Step 4/4: magnetometer.")
    meta = run_mag_calibration(stream, meta)
    save_meta(data_dir, meta)
    return meta


def prepare_meta(stream, data_dir, force_calibrate=False, rebias=True, force_mag=False):
    """Load calibration (running the wizard if needed) and re-zero the gyro bias for this session.
    `force_mag` redoes only the magnetometer calibration (keeps the IMU axis calibration)."""
    path = calib_path(data_dir)
    if force_calibrate or not os.path.exists(path):
        return run_calibration(stream, data_dir)  # already includes a fresh bias
    with open(path) as f:
        meta = json.load(f)
    if rebias:
        print("Set the pen down and keep it still for ~2 s (re-zeroing gyro)...")
        still = stream.wait_still(2.0)
        meta["gyro_bias_dps"] = still[:, 3:6].mean(0).tolist()
        meta["acc_scale"] = float(np.linalg.norm(still[:, :3], axis=1).mean())
        print("  done.")
    if force_mag:
        meta = run_mag_calibration(stream, meta)
        save_meta(data_dir, meta)
    if not meta_has_mag(meta):
        print(
            "note: calib.json has no magnetometer calibration -> samples are saved without usable mag\n"
            "      and variants E-G are unavailable. Run `collect.py --calibrate-mag` to add it."
        )
    return meta


# =============================================================================================
# Sample I/O
# =============================================================================================
def label_dir(label):
    """Folder name for a label (safe on case-insensitive filesystems)."""
    if len(label) == 1 and label.isascii() and label.isalnum():
        return label + "_lower" if label.islower() else label
    return "u" + "-".join(f"{ord(c):04x}" for c in label)


def build_sample(cap, meta, label="?", user="me", session=""):
    """Raw capture + the calibration it was captured under. Mag fields are always present;
    `mag_cal` says whether the calibration is real (0 -> mag is stored but not usable)."""
    return dict(
        t=cap["t"],
        raw=cap["raw"],
        mag=cap["mag"],
        touch=cap["touch"],
        start=cap["start"],
        end=cap["end"],
        bias=np.asarray(meta["gyro_bias_dps"], float),
        acc_scale=float(meta["acc_scale"]),
        pen_axis=np.asarray(meta["pen_axis"], float),
        mag_offset=np.asarray(meta.get("mag_offset", [0.0, 0.0, 0.0]), float),
        mag_scale=np.asarray(meta.get("mag_scale", [1.0, 1.0, 1.0]), float),
        mag_align=np.asarray(meta.get("mag_align", np.eye(3)), float),
        mag_field=float(meta.get("mag_field", 0.0)),
        mag_cal=np.uint8(meta_has_mag(meta)),
        label=label,
        user=user,
        session=session,
    )


def save_sample(data_dir, s):
    d = os.path.join(data_dir, s["user"], label_dir(s["label"]))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{s['session']}_{time.time_ns() // 1_000_000}.npz")
    np.savez_compressed(path, **s)
    return path


def load_sample(path):
    """Loads a sample; samples from before the magnetometer get NaN mag and mag_cal=False."""
    z = np.load(path, allow_pickle=False)
    s = {k: z[k] for k in z.files}
    for k in ("label", "user", "session"):
        s[k] = str(s[k])
    s["start"], s["end"], s["acc_scale"] = (
        int(s["start"]),
        int(s["end"]),
        float(s["acc_scale"]),
    )
    if "mag" not in s:
        s["mag"] = np.full((len(s["t"]), 3), np.nan, np.float32)
    s.setdefault("mag_offset", np.zeros(3))
    s.setdefault("mag_scale", np.ones(3))
    s.setdefault("mag_align", np.eye(3))
    s["mag_field"] = float(s["mag_field"]) if "mag_field" in s else 0.0
    s["mag_cal"] = bool(int(s["mag_cal"])) if "mag_cal" in s else False
    s["path"] = path
    return s


def sample_has_mag(s):
    """True if the sample carries real mag data AND a mag calibration."""
    return bool(s["mag_cal"]) and mag_present(s["mag"])


def mag_field_dev(s):
    """Mean relative deviation of |calibrated mag| from the calibrated field strength over the
    touch window (None if no mag). Large = magnetic interference while writing."""
    if not sample_has_mag(s) or s["mag_field"] <= 0:
        return None
    m = apply_mag_cal(
        s["mag"][s["start"] : s["end"]], s["mag_offset"], s["mag_scale"], s["mag_align"]
    )
    return float(
        np.mean(np.abs(np.linalg.norm(m, axis=1) - s["mag_field"])) / s["mag_field"]
    )


def list_samples(data_dir, user):
    return sorted(glob.glob(os.path.join(data_dir, user, "*", "*.npz")))


def count_label(data_dir, user, label):
    return len(glob.glob(os.path.join(data_dir, user, label_dir(label), "*.npz")))


# =============================================================================================
# Orientation + preprocessing
# =============================================================================================
def tilt_quat(a):
    """Quaternion (w,x,y,z), body->earth, from one accel reading (yaw = 0)."""
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([1.0, 0, 0, 0])
    a = a / n
    z = np.array([0, 0, 1.0])
    w = 1.0 + a @ z
    if w < 1e-6:
        return np.array([0.0, 1, 0, 0])
    q = np.array([w, *np.cross(a, z)])
    return q / np.linalg.norm(q)


def quat_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R):
    """Rotation matrix -> unit quaternion (w,x,y,z); inverse of quat_to_R."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ]
    q = np.array(q)
    return q / np.linalg.norm(q)


def marg_init_quat(a, m):
    """Body->earth quaternion from one accel + one calibrated mag reading. Earth frame: z up,
    x = horizontal component of the magnetic field (magnetic north), y = z cross x."""
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([1.0, 0, 0, 0])
    z = a / n
    hm = m - (m @ z) * z
    if np.linalg.norm(hm) < 1e-6:
        return tilt_quat(a)
    x = hm / np.linalg.norm(hm)
    return R_to_quat(
        np.stack([x, np.cross(z, x), z])
    )  # rows = earth axes expressed in body


def madgwick_step(q, g, a, dt, beta):
    """One Madgwick IMU update. g in rad/s, a in any units (normalised inside). beta=0 -> gyro only."""
    w, x, y, z = q
    qdot = 0.5 * np.array(
        [
            -x * g[0] - y * g[1] - z * g[2],
            w * g[0] + y * g[2] - z * g[1],
            w * g[1] - x * g[2] + z * g[0],
            w * g[2] + x * g[1] - y * g[0],
        ]
    )
    n = np.linalg.norm(a)
    if beta > 0 and n > 1e-6:
        a = a / n
        f = np.array(
            [
                2 * (x * z - w * y) - a[0],
                2 * (w * x + y * z) - a[1],
                2 * (0.5 - x * x - y * y) - a[2],
            ]
        )
        J = np.array(
            [
                [-2 * y, 2 * z, -2 * w, 2 * x],
                [2 * x, 2 * w, 2 * z, 2 * y],
                [0.0, -4 * x, -4 * y, 0.0],
            ]
        )
        grad = J.T @ f
        gn = np.linalg.norm(grad)
        if gn > 0:
            qdot = qdot - beta * grad / gn
    q = q + qdot * dt
    return q / np.linalg.norm(q)


def _rt_jac(q, v):
    """r = R(q)^T v (earth vector v seen from the body) and its 3x4 Jacobian wrt (w,x,y,z)."""
    w, x, y, z = q
    RT = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    dw = np.array([[0, 2 * z, -2 * y], [-2 * z, 0, 2 * x], [2 * y, -2 * x, 0]])
    dx = np.array([[0, 2 * y, 2 * z], [2 * y, -4 * x, 2 * w], [2 * z, -2 * w, -4 * x]])
    dy = np.array([[-4 * y, 2 * x, -2 * w], [2 * x, 0, 2 * z], [2 * w, 2 * z, -4 * y]])
    dz = np.array([[-4 * z, 2 * w, 2 * x], [-2 * w, -4 * z, 2 * y], [2 * x, 2 * y, 0]])
    v = np.asarray(v, float)
    return RT @ v, np.stack([dw @ v, dx @ v, dy @ v, dz @ v], 1)


def madgwick_marg_step(q, g, a, m, dt, beta_a, beta_m):
    """One Madgwick MARG update (gyro + accel + calibrated mag), body->earth quaternion, earth x =
    magnetic north, z = up. Accel and mag corrections are applied separately so they can be gated
    / weighted independently. The mag reference (bx, 0, bz) is re-derived from the current estimate
    each step (flux compensation), so the mag term only ever corrects yaw. beta=0 disables a term."""
    w, x, y, z = q
    qdot = 0.5 * np.array(
        [
            -x * g[0] - y * g[1] - z * g[2],
            w * g[0] + y * g[2] - z * g[1],
            w * g[1] - x * g[2] + z * g[0],
            w * g[2] + x * g[1] - y * g[0],
        ]
    )
    corr = np.zeros(4)
    na = np.linalg.norm(a)
    if beta_a > 0 and na > 1e-6:
        r, J = _rt_jac(q, (0.0, 0.0, 1.0))
        grad = J.T @ (r - a / na)
        gn = np.linalg.norm(grad)
        if gn > 0:
            corr += beta_a * grad / gn
    nm = np.linalg.norm(m)
    if beta_m > 0 and nm > 1e-9:
        mn = m / nm
        h = quat_to_R(q) @ mn
        r, J = _rt_jac(q, (np.hypot(h[0], h[1]), 0.0, h[2]))
        grad = J.T @ (r - mn)
        gn = np.linalg.norm(grad)
        if gn > 0:
            corr += beta_m * grad / gn
    q = q + (qdot - corr) * dt
    return q / np.linalg.norm(q)


def _yaw_rot(v, h):
    c, sn = np.cos(h), np.sin(h)
    return np.stack(
        [c * v[:, 0] + sn * v[:, 1], -sn * v[:, 0] + c * v[:, 1], v[:, 2]], 1
    )


def derive(
    s,
    trim_ms=50.0,
    beta_free=0.15,
    beta_pen=0.02,
    accel_gate=0.25,
    use_mag=True,
    mag_beta_free=0.10,
    mag_beta_pen=0.03,
    mag_gate=0.25,
):
    """Raw sample -> dict of signals for the touch window (trimmed at both ends).

    Orientation is initialised from gravity at the start of the pre-touch buffer, then tracked by
    Madgwick: strong accel correction while the pen is up, very weak while it's touching (accel
    then contains real motion). Accel is also ignored whenever |a| is far from 1 g.
    Body frame: as measured.  World frame: z up, gravity removed from accel.
    Heading-aligned ('_h'): x = horizontal direction the pen tip points at touch-down.

    If the sample has a calibrated magnetometer (and use_mag), a second, MARG-based orientation is
    tracked (init from accel+mag, mag ignored whenever |B| is >mag_gate off the calibrated field
    strength) and these are added:
      mag_b          calibrated mag, body frame (Gauss)
      lin_n, gyr_n   world frame with x = magnetic north (no pen-heading rotation)
      lin_hm, gyr_hm heading-aligned like lin_h/gyr_h, but when the pen is (nearly) vertical and
                     the pen heading is undefined, x stays at magnetic north instead of arbitrary
    """
    t = s["t"].astype(float)
    N = len(t)
    dt = np.diff(t, prepend=t[0])
    fs = float(1.0 / np.median(dt[1:])) if N > 2 else 100.0
    acc = s["raw"][:, :3].astype(float) / s["acc_scale"]
    gyr = np.deg2rad(s["raw"][:, 3:].astype(float) - s["bias"])

    q = tilt_quat(acc[: min(5, N)].mean(0))
    Rs = np.empty((N, 3, 3))
    for i in range(N):
        touching = s["start"] <= i < s["end"]
        beta = beta_pen if touching else beta_free
        if abs(np.linalg.norm(acc[i]) - 1.0) > accel_gate:
            beta = 0.0
        q = madgwick_step(q, gyr[i], acc[i], dt[i], beta)
        Rs[i] = quat_to_R(q)

    k = int(round(trim_ms / 1000 * fs))
    lo, hi = s["start"] + k, s["end"] - k
    if hi - lo < 10:
        lo, hi = s["start"], s["end"]
    A, G, R = acc[lo:hi], gyr[lo:hi], Rs[lo:hi]
    up = np.array([0, 0, 1.0])
    lin_w = np.einsum("nij,nj->ni", R, A) - up
    gyr_w = np.einsum("nij,nj->ni", R, G)

    p_w = np.einsum("nij,j->ni", R[:5], s["pen_axis"]).mean(0)
    heading_ok = bool(
        np.hypot(p_w[0], p_w[1]) > 0.3
    )  # pen not pointing (nearly) straight up/down
    h = float(np.arctan2(p_w[1], p_w[0])) if heading_ok else 0.0

    out = dict(
        fs=fs,
        acc_b=A,
        gyr_b=G,
        lin_w=lin_w,
        gyr_w=gyr_w,
        lin_h=_yaw_rot(lin_w, h),
        gyr_h=_yaw_rot(gyr_w, h),
        heading_ok=heading_ok,
        has_mag=False,
    )

    if use_mag and sample_has_mag(s):
        mag = apply_mag_cal(
            s["mag"].astype(float), s["mag_offset"], s["mag_scale"], s["mag_align"]
        )
        field = float(s["mag_field"])
        k0 = min(5, N)
        qm = marg_init_quat(acc[:k0].mean(0), mag[:k0].mean(0))
        Rm = np.empty((N, 3, 3))
        for i in range(N):
            touching = s["start"] <= i < s["end"]
            ba = beta_pen if touching else beta_free
            if abs(np.linalg.norm(acc[i]) - 1.0) > accel_gate:
                ba = 0.0
            bm = mag_beta_pen if touching else mag_beta_free
            if field > 0 and abs(np.linalg.norm(mag[i]) - field) / field > mag_gate:
                bm = 0.0
            qm = madgwick_marg_step(qm, gyr[i], acc[i], mag[i], dt[i], ba, bm)
            Rm[i] = quat_to_R(qm)
        Rm = Rm[lo:hi]
        lin_n = np.einsum("nij,nj->ni", Rm, A) - up
        gyr_n = np.einsum("nij,nj->ni", Rm, G)
        pm = np.einsum("nij,j->ni", Rm[:5], s["pen_axis"]).mean(0)
        ok_m = bool(np.hypot(pm[0], pm[1]) > 0.3)
        hm = float(np.arctan2(pm[1], pm[0])) if ok_m else 0.0
        out.update(
            has_mag=True,
            mag_b=mag[lo:hi],
            lin_n=lin_n,
            gyr_n=gyr_n,
            lin_hm=_yaw_rot(lin_n, hm),
            gyr_hm=_yaw_rot(gyr_n, hm),
            heading_ok_m=ok_m,
        )
    return out


VARIANTS = {
    "A": "raw body frame (6ch)",
    "B": "gravity-removed, world + heading aligned (6ch)",
    "C": "B + |accel|, |gyro| magnitudes (8ch)",
    "D": "yaw-invariant: z + horizontal magnitude (4ch)",
    "E": "A + calibrated mag, body frame (9ch)",
    "F": "magnetic-north world frame, no gravity (6ch)",
    "G": "B; pen-vertical falls back to mag north (6ch)",
}
MAG_VARIANTS = {"E", "F", "G"}  # need a calibrated magnetometer in every sample


def _zh(v):  # (vertical, horizontal magnitude)
    return np.stack([v[:, 2], np.hypot(v[:, 0], v[:, 1])], 1)


def features(d, variant, mag_weight=MAG_WEIGHT):
    """Returns a list of channel groups; each group is normalised separately in vec().
    A group may be given as (array, weight) to scale it after normalisation."""
    if variant in MAG_VARIANTS and not d.get("has_mag"):
        raise ValueError(
            f"variant {variant} needs a calibrated magnetometer in the sample"
        )
    if variant == "A":
        return [d["acc_b"], d["gyr_b"]]
    if variant == "B":
        return [d["lin_h"], d["gyr_h"]]
    if variant == "C":
        return [
            d["lin_h"],
            d["gyr_h"],
            np.linalg.norm(d["lin_h"], axis=1, keepdims=True),
            np.linalg.norm(d["gyr_h"], axis=1, keepdims=True),
        ]
    if variant == "D":
        return [_zh(d["lin_w"]), _zh(d["gyr_w"])]
    if variant == "E":
        return [d["acc_b"], d["gyr_b"], (d["mag_b"], mag_weight)]
    if variant == "F":
        return [d["lin_n"], d["gyr_n"]]
    if variant == "G":
        return [d["lin_hm"], d["gyr_hm"]]
    raise ValueError(variant)


def _smooth(x, n):
    n = 2 * (n // 2) + 1
    if n <= 1:
        return x
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    xp = np.pad(x, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(xp[:, c], k, mode="valid") for c in range(x.shape[1])], 1
    )


def vec(groups, fs, L=100):
    """Smooth (~40 ms), remove mean, divide each group by its own RMS (keeps the relative
    amplitude between axes inside a group), apply the group's weight if given, then resample
    to L steps."""
    cols = []
    for g in groups:
        w = 1.0
        if isinstance(g, tuple):
            g, w = g
        g = _smooth(g, int(round(0.04 * fs)))
        g = g - g.mean(0)
        cols.append(w * g / (np.sqrt((g**2).mean()) + 1e-9))
    X = np.concatenate(cols, 1)
    src, dst = np.linspace(0, 1, len(X)), np.linspace(0, 1, L)
    return np.stack([np.interp(dst, src, X[:, c]) for c in range(X.shape[1])], 1)


def load_templates(data_dir, user, variant, L=100, mag_weight=MAG_WEIGHT):
    """Feature matrix + labels for live recognition. Mag variants only use samples that have a
    calibrated mag. Returns (X, y, n_skipped); X is None if nothing usable."""
    need = variant in MAG_VARIANTS
    X, y, skipped = [], [], 0
    for p in list_samples(data_dir, user):
        s = load_sample(p)
        if need and not sample_has_mag(s):
            skipped += 1
            continue
        d = derive(s, use_mag=need)
        X.append(vec(features(d, variant, mag_weight), d["fs"], L))
        y.append(s["label"])
    if not X:
        return None, None, skipped
    return np.ascontiguousarray(np.stack(X)), np.array(y), skipped
