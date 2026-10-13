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

      8 raw motions are detected: flick up/down/left/right, twist clockwise/anticlockwise, thrust
      forward/backward. Which action each one performs is a REMAPPABLE table, edited live from the
      frontend's settings panel (gear icon) -- nothing to pass on the command line for that. The
      shipped default (same behaviour as before remapping existed):

      flick up -> UPPERCASE (set)      flick left  -> backspace          twist ccw -> cursor back
      flick down -> lowercase (set)    flick right -> space              twist cw  -> cursor forward
      thrust forward -> Enter          thrust backward -> toggle case

  "Set" actions (UPPERCASE/lowercase) only ever set the state, not toggle it -- flicking up twice
  does nothing the second time. "Toggle case" flips whatever the current state is.

Gesture detection is deliberately simple and tunable (everything is a flag): rates are measured in a
pen-relative frame built from gravity and the calibrated pen axis -- "forward" is where the tip
points, "up" is world up -- so it works at any wrist angle. After each gesture there is a short
lock-out (until the pen is quiet again) so the return stroke of a flick isn't read as the opposite
gesture; twists likewise ignore the untwist for ~0.5 s. First time on real hardware, watch the
live gesture gauges in the frontend and check the signs (--flip-ud / --flip-lr / --twist-sign) and
thresholds (--flick-dps / --thrust-g / --twist-min-dps) -- or, easier, open the settings panel in
the frontend: every threshold and the whole action mapping can be changed live from there, for
any connected client, with no restart (see "live config" below).

Teach-by-demonstration: any gesture (the 8 factory motions OR brand-new ones you name) can be
(re)defined from the settings panel: press Teach, enter gesture mode, do the motion ~5 times with a
pause between reps. The server records the peak rotation/acceleration direction of each rep in the
SENSOR's own frame (so it doesn't depend on which way you face, or on `pen_axis`), averages them,
and from then on matches live motion against that learned direction. Taught gestures are saved to
<data>/gestures.json and reloaded at startup. Factory kinematics stay as the default for any
gesture you haven't taught; "Back to factory" removes a taught override.

Autocomplete + read-aloud: a local Ollama model (default qwen2.5:3b) predicts the rest of the
current word and the words after it; the new "Autocomplete" action (assign it to any gesture in
settings) inserts the first predicted word, and again for the next one. Every finished word (a
space/newline typed or autocompleted after it) is sent to the page as a `speak` message so the
browser reads it aloud. Gestures themselves are never spoken.

Voice: by default the SERVER speaks each finished word itself through the host's `spd-say` (or
`espeak-ng`), because browsers on some Linux setups can't reach speech-dispatcher. At startup it says
"voice ready" and prints every word it speaks, so you can see whether it works. The page's
settings > Voice can mute it or switch to the browser's own voice instead; exactly one of them
speaks. --no-speak turns the server voice off.

Twists are single moves now (one twist = one event), not a stepping dial.

Live config: the gesture->action mapping and every threshold live in a Config object on the server,
not just in argparse. The CLI flags below only set the STARTING values. Any connected frontend can
change them at any time (settings panel, gear icon); the change applies immediately to every
connected client and to detection on the next packet -- no restart, no reconnect.

Websocket messages (JSON), server -> frontend:
  touch_start, touch_end, clear, caps {on}, char {char},
  imu {touched, gx, gy, gz, x, y, mx, my, mz}   (WRITE mode only; mx/my/mz = calibrated field in
                                                 Gauss or null)
  mode {mode: "write"|"gesture"}
  gest {ud, lr, fw, tw, tw_deg, tw_step, state}  (GESTURE mode, every packet: live gesture scores,
                                                  1.0 = detection threshold; ud>0 = up, lr>0 = right,
                                                  fw>0 = forward (negative = backward), tw>0 = cw)
  gesture {name, action, gesture_label, action_label, score}  (a raw motion fired and was resolved
                                                  through the current mapping to an action)
  text {text, cursor}                            (authoritative text buffer; sent after every change
                                                  and, with caps/mode/config, to each newly connected
                                                  client)
  config {thresholds, mapping, gestures, actions} (the full live config: `gestures` and `actions` are
                                                  {id: label} dictionaries the frontend builds its
                                                  settings panel from, so the two stay in sync without
                                                  hardcoding the list twice; sent on connect and after
                                                  every change, to every client)

Websocket messages (JSON), frontend -> server (new: the socket used to be send-only):
  config_set {thresholds?, mapping?}   partial update; unknown keys/ids are ignored, numeric fields
                                        are clamped to a sane range server-side. The server always
                                        answers (to everyone) with the resulting full `config`, so a
                                        client never has to guess whether its change stuck.

Requires `pip install websockets` for the frontend link (falls back to console-only --live
without it).
"""

import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

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
# main thread (or from inside `on_message`) at any point. `snapshot` (optional callable -> list of
# messages) is sent to every client right after it connects, so a page opened/reloaded mid-session
# shows the current text, case, mode and config. `on_message` (optional callable(dict)), if set, is
# called -- on the broadcaster's own background thread -- for every JSON object a client sends;
# exceptions and non-dict/invalid-JSON payloads are swallowed so one bad client can't take the
# server down.
# =============================================================================================
class Broadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.snapshot = None
        self.on_message = None
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
                async for raw in ws:
                    if not self.on_message:
                        continue
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(msg, dict):
                        try:
                            self.on_message(msg)
                        except Exception as e:
                            print(f"[frontend] bad message from client, ignored: {e}")
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
    on_message = None

    def send(self, msg):
        pass


# =============================================================================================
# Text buffer (server-owned so console-only mode works and the frontend is just a view)
# =============================================================================================
class Typist:
    PUNCT = '.,;:!?"()[]{}\u2026\u2013\u2014\u201c\u201d'

    def __init__(self):
        self.chars = []
        self.cur = 0
        self.caps = False
        self.lock = threading.RLock()  # the LLM thread may insert text too
        self.on_change = (
            None  # callable(), after any edit or cursor move (called outside the lock)
        )
        self.on_word = (
            None  # callable(word), when a word is finished (whitespace typed after it)
        )

    def _fire(self, words=()):
        for w in words:
            if self.on_word:
                self.on_word(w)
        if self.on_change:
            self.on_change()

    def _word_before(self, i):
        j = i
        while j > 0 and not self.chars[j - 1].isspace():
            j -= 1
        w = "".join(self.chars[j:i]).strip(self.PUNCT)
        return w if any(c.isalnum() for c in w) else ""

    def put(self, s):
        words = []
        with self.lock:
            for ch in s:
                self.chars.insert(self.cur, ch)
                self.cur += 1
                if ch.isspace():
                    w = self._word_before(self.cur - 1)
                    if w:
                        words.append(w)
        self._fire(words)

    def backspace(self):
        with self.lock:
            if self.cur == 0:
                return False
            del self.chars[self.cur - 1]
            self.cur -= 1
        self._fire()
        return True

    def move(self, d):
        with self.lock:
            new = min(max(self.cur + d, 0), len(self.chars))
            moved = new != self.cur
            self.cur = new
        if moved:
            self._fire()
        return moved

    def context(self):
        """(text before the cursor, the character right after it or "")."""
        with self.lock:
            nxt = self.chars[self.cur] if self.cur < len(self.chars) else ""
            return "".join(self.chars[: self.cur]), nxt

    def snapshot(self):
        with self.lock:
            return {"type": "text", "text": "".join(self.chars), "cursor": self.cur}

    def show(self):
        with self.lock:
            s = "".join(self.chars).replace("\n", "\u23ce")
            return f"  text: {s[: self.cur]}|{s[self.cur :]}"


class Voice:
    """Speaks finished words through the host's speech-dispatcher (`spd-say`) or `espeak-ng`.
    Only does anything while `enabled` (the page switches it on when it picked the server route)."""

    def __init__(self, bc, enabled=True):
        self.bc, self.enabled, self.rate = bc, enabled, 1.0
        self.engine = (
            "spd-say"
            if shutil.which("spd-say")
            else "espeak-ng"
            if shutil.which("espeak-ng")
            else None
        )
        self.q = queue.Queue(maxsize=50)
        self._warned = False
        if self.engine:
            print(
                f"[voice] server voice: {self.engine}" + ("" if enabled else " (off)")
            )
            threading.Thread(target=self._run, daemon=True).start()
        else:
            print(
                "[voice] neither spd-say nor espeak-ng found; the server can't speak."
            )

    def say(self, word, force=False):
        if (self.enabled or force) and self.engine and word:
            try:
                self.q.put_nowait(word)
            except queue.Full:
                pass

    def _cmd(self, word):
        word = word.lstrip("-")  # never let a word look like an option
        if (
            self.engine == "spd-say"
        ):  # rate -100..100; speech-dispatcher queues messages in order itself
            r = int(max(-100, min(100, (self.rate - 1.0) * 100)))
            return ["spd-say", "-r", str(r), word]
        return ["espeak-ng", "-s", str(int(175 * self.rate)), word]

    def _run(self):
        while True:
            w = self.q.get()
            print(f"  [voice] say: {w}")
            try:
                r = subprocess.run(
                    self._cmd(w), timeout=15, capture_output=True, text=True
                )
                if r.returncode != 0:
                    print(
                        f"[voice] {self.engine} exited {r.returncode}: {(r.stderr or r.stdout).strip()}"
                    )
            except Exception as e:
                print(f"[voice] {self.engine} failed: {e}")

    def on_message(self, msg):
        k = msg.get("type")
        if k == "voice_set":
            self.enabled = bool(msg.get("server"))
            try:
                self.rate = min(max(float(msg.get("rate", 1.0)), 0.5), 2.0)
            except (TypeError, ValueError):
                pass
        elif k == "voice_test":
            self.say("Hello, this is AirTalk", force=True)

    def snapshot(self):
        return {
            "type": "voice",
            "engine": self.engine or "none",
            "server": self.enabled,
        }


# =============================================================================================
# Autocomplete: a local LLM (Ollama) predicts what comes next; one gesture accepts it word by word.
#
# `cont` is the model's predicted continuation of `key` (= the text before the cursor). Accepting
# inserts its first word (the rest of the word you're in the middle of, or else the next word) plus
# a space, and keeps the remainder, so a second accept needs no new model call. Typing the predicted
# letters consumes `cont` the same way; anything else throws it away and asks again. Requests are
# made on a background thread (latest wins; stale answers are dropped), so the pen loop never waits.
# If you accept before the answer is back, the accept is queued and applied the moment it arrives.
# =============================================================================================
class Autocomplete:
    N_PREDICT = 8  # tokens per request; trimmed to MAX_WORDS clean words
    MAX_WORDS = 2
    CONTEXT_CHARS = 400
    DEBOUNCE_S = 0.2
    PENDING_S = 4.0  # an early accept waits this long for the model
    # A few plain lines so even an instruct model continues in the same plain style (no markdown,
    # no punctuation) instead of writing a formatted answer.
    HEADER = (
        "plain text typed by one person using only letters numbers and spaces\n"
        "i am going to the store to buy some milk and bread\n"
        "can you call me when you get home tonight\n"
        "the meeting starts at 3 pm in room 12\n"
        "thank you so much for your help with the project\n"
        "we should leave at 7 so we are not late\n"
    )
    STOPS = ["\n", "*", "#", "`", "_", "(", ")", '"', ".", ",", "!", "?", ";", ":"]

    def __init__(self, bc, typist, url, model):
        self.bc, self.typist = bc, typist
        self.url = url.rstrip("/")
        self.model = model  # None = pick a qwen2.5 3b model from what Ollama has
        self.lock = threading.Lock()
        self.key, self.cont = "", ""
        self.want = (
            ""  # first job = warm the model up so your first real request isn't slow
        )
        self.warmed = False
        self.pending_until = 0.0
        self.status = "loading"
        self.wake = threading.Event()
        typist.on_change = self.on_text_change
        threading.Thread(target=self._worker, daemon=True).start()
        self.wake.set()

    # -- status / messages --------------------------------------------------------------------
    def _set_status(self, st):
        if st != self.status:
            self.status = st
            self.bc.send({"type": "llm", "status": st, "model": self.model or ""})

    def _send_suggest(self):
        with self.lock:
            msg = {"type": "suggest", "text": self.cont, "at": len(self.key or "")}
        self.bc.send(msg)

    def snapshot_msgs(self):
        with self.lock:
            sug = {"type": "suggest", "text": self.cont, "at": len(self.key or "")}
        return [{"type": "llm", "status": self.status, "model": self.model or ""}, sug]

    # -- typist hook --------------------------------------------------------------------------
    @staticmethod
    def _consume(cont, extra):
        """`extra` was just typed/inserted after the text `cont` continues; what's left of `cont`,
        or None if the text diverged from the prediction."""
        if cont.startswith(extra):
            return cont[len(extra) :]
        e = extra.rstrip()
        if (
            e != extra
            and e
            and cont.startswith(e)
            and (len(cont) == len(e) or cont[len(e)].isspace())
        ):
            return cont[len(e) :].lstrip()
        return None

    def on_text_change(self):
        before, nxt = self.typist.context()
        with self.lock:
            old, cont = self.key, self.cont
            if nxt.isalnum():  # cursor is inside an existing word: stay quiet
                self.key, self.cont, need = before, "", False
            else:
                left = None
                if old is not None and before.startswith(old):
                    left = self._consume(cont, before[len(old) :])
                self.key = before
                self.cont = left if left is not None else ""
                need = len(self.cont.split()) < 2
            if need:
                self.want = before
        if need:
            self.wake.set()
        self._send_suggest()

    # -- accept (gesture) ---------------------------------------------------------------------
    def accept(self):
        """Returns a short note for the console."""
        if self.status == "offline":
            return "  (autocomplete: Ollama not reachable)"
        before, _ = self.typist.context()
        with self.lock:
            cont = self.cont if self.key == before else ""
        if not cont.strip():
            with self.lock:
                self.pending_until = time.time() + self.PENDING_S
                self.want = before
            self.wake.set()
            return "  (autocomplete: waiting for the model...)"
        m = re.match(r"(\S*)\s*", cont)
        token = m.group(1)
        self.typist.put(
            token + " "
        )  # fires on_word (read aloud) and on_text_change (consumes cont)
        self.bc.send(self.typist.snapshot())
        return ""

    # -- model --------------------------------------------------------------------------------
    def _http(self, path, body=None, timeout=60):
        req = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _pick_model(self):
        names = [
            m.get("name", "")
            for m in self._http("/api/tags", timeout=5).get("models", [])
        ]
        if self.model and self.model in names:
            return
        if self.model:
            print(
                f"[llm] model '{self.model}' isn't in Ollama (have: {', '.join(names) or 'none'}). Run: ollama pull {self.model}"
            )
            return
        pick = next((n for n in names if n.startswith("qwen2.5:3b")), None) or next(
            (n for n in names if "qwen2.5" in n and "3b" in n), None
        )
        self.model = pick or "qwen2.5:3b"
        if pick:
            print(f"[llm] using Ollama model '{pick}'")
        else:
            print(
                f"[llm] no qwen2.5 3b model found in Ollama (have: {', '.join(names) or 'none'}). Run: ollama pull qwen2.5:3b"
            )

    def _ask(self, key, n=None):
        ends_space = key.endswith(" ")
        text = key[-self.CONTEXT_CHARS :]
        prompt = self.HEADER + (text.rstrip(" ") if ends_space else text)
        out = self._http(
            "/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "raw": True,  # plain completion; no chat template
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": n or self.N_PREDICT,
                    "temperature": 0.2,
                    "top_k": 20,
                    "num_ctx": 1024,
                    "stop": self.STOPS,
                },
            },
            timeout=90,
        )
        return self._clean(
            out.get("response", ""), key, ends_space, out.get("done_reason") == "length"
        )

    @classmethod
    def _clean(cls, r, key, ends_space, cut):
        """Model output -> at most MAX_WORDS words made only of letters/digits, joined by spaces."""
        r = r.split("\n")[0]
        if ends_space and r.startswith(" "):
            r = r[1:]
        if cut:  # generation stopped mid-token: the last word may be unfinished
            i = r.rfind(" ")
            if i > 0:
                r = r[:i]
        lead = " " if r.startswith(" ") else ""
        keep = []
        for w in r.split():
            m = re.match(r"[A-Za-z0-9]+", w)
            if not m:
                break
            keep.append(m.group())
            if m.end() < len(w):
                # sentence punctuation after the word: keep the word, stop. Anything else (an
                # apostrophe, hyphen, slash...) means we can't write it, so drop this word too.
                if w[m.end()] not in '.,;:!?)"':
                    keep.pop()
                break
        r = lead + " ".join(keep[: cls.MAX_WORDS]) if keep else ""
        lw = re.search(
            r"[A-Za-z0-9]+(?= *$)", key
        )  # the user is typing in capitals: match them
        if lw and len(lw.group()) >= 2 and lw.group().isupper():
            r = r.upper()
        return r

    def _worker(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            time.sleep(self.DEBOUNCE_S)
            with self.lock:
                key = self.want
            if key is None:
                continue
            warm = not self.warmed
            with self.lock:
                if not warm and key != self.key:
                    continue  # already stale
            self._set_status(
                "loading" if warm or self.status != "ready" else "thinking"
            )
            try:
                if self.model is None or warm:
                    self._pick_model()
                cont = self._ask(key, 1 if warm else None)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = json.loads(e.read().decode()).get("error", "")
                except Exception:
                    pass
                print(f"[llm] Ollama error {e.code}: {detail}")
                self._set_status("offline")
                continue
            except Exception as e:
                if self.status != "offline":
                    print(
                        f"[llm] can't reach Ollama at {self.url} ({e}); autocomplete is off until it answers."
                    )
                self._set_status("offline")
                continue
            self._set_status("ready")
            if warm:
                self.warmed = True
                print("[llm] ready")
                with self.lock:  # you may have written (or accepted) while it warmed up
                    redo = bool(self.key) or time.time() < self.pending_until
                    if redo:
                        self.want = self.key
                if redo:
                    self.wake.set()
                continue
            with self.lock:
                if key != self.key:
                    continue  # you kept writing; this answer is for old text
                self.cont = cont
                pend = time.time() < self.pending_until
                self.pending_until = 0.0
            self._send_suggest()
            if pend and cont.strip():
                self.accept()


# =============================================================================================
# Gesture <-> action mapping (remappable, live, from the frontend's settings panel)
#
# GESTURE_META / ACTION_META are the catalogue of raw motions and of things they can do -- sent to
# every client as part of `config` so the settings panel is built from this one list and can't drift
# out of sync with what the server actually understands. DEFAULT_MAPPING is only the starting point;
# the live mapping lives in `Config.mapping` and is changed by `config_set` messages.
# =============================================================================================
GESTURE_META = {
    "flick_up": "Flick \u2191 (tip up)",
    "flick_down": "Flick \u2193 (tip down)",
    "flick_left": "Flick \u2190 (tip left)",
    "flick_right": "Flick \u2192 (tip right)",
    "thrust_fwd": "Thrust forward",
    "thrust_back": "Thrust backward (pull in)",
    "twist_cw": "Twist clockwise \u21bb",
    "twist_ccw": "Twist anticlockwise \u21ba",
}
ACTION_META = {
    "none": "\u2014 (unassigned)",
    "backspace": "Backspace",
    "space": "Space",
    "enter": "Enter",
    "uppercase": "UPPERCASE (set)",
    "lowercase": "lowercase (set)",
    "caps_toggle": "Toggle case",
    "cursor_back": "Cursor \u2190 (back)",
    "cursor_fwd": "Cursor \u2192 (forward)",
    "autocomplete": "Autocomplete (word, then next)",
}
DEFAULT_MAPPING = {
    "flick_up": "uppercase",
    "flick_down": "lowercase",
    "flick_left": "backspace",
    "flick_right": "space",
    "thrust_fwd": "enter",
    "thrust_back": "caps_toggle",
    "twist_ccw": "cursor_back",
    "twist_cw": "cursor_fwd",
}
# (name, min, max, cast) -- the thresholds a client is allowed to change, and the range they're
# clamped to. These live as plain attributes on the shared `args` Namespace (the single source both
# GestureDetector and Session already read every packet/touch), so writing one here takes effect
# immediately with no restart -- see Config.apply().
THRESHOLD_FIELDS = {
    "flick_dps": (50.0, 600.0, float),
    "thrust_g": (0.15, 2.0, float),
    "twist_deg": (10.0, 120.0, float),
    "twist_min_dps": (20.0, 300.0, float),
    "gesture_gap": (0.1, 2.0, float),
    "gesture_confirm": (0.05, 1.0, float),
    "twist_sign": (None, None, lambda v: 1 if float(v) >= 0 else -1),
    "flip_ud": (None, None, bool),
    "flip_lr": (None, None, bool),
    "mag_assist": (None, None, bool),
    "mag_sweep_deg": (15.0, 120.0, float),
}


class Config:
    """The live, remappable part of the setup: thresholds (on the shared `args` Namespace) and the
    gesture->action mapping (its own dict, since `args` only ever held scalars). One instance per
    `live()` run, shared between the Session (reads, on the main thread) and the websocket handler
    (writes, on the broadcaster's background thread) -- `lock` only needs to guard `mapping` (a dict
    swap isn't atomic the way a single attribute set is under the GIL); the threshold attributes on
    `args` are written one at a time with plain `setattr`, each of which is itself atomic enough that
    a reader never sees a half-written value, so no lock is needed there.
    """

    def __init__(self, args):
        self.args = args
        self.mapping = dict(DEFAULT_MAPPING)
        self.lock = threading.Lock()
        self.learner = None  # set by live(); supplies custom gestures + taught state

    def labels(self):
        """Factory motions + any custom (taught) gestures, {id: label}."""
        d = dict(GESTURE_META)
        if self.learner:
            d.update(self.learner.custom_labels())
        return d

    def label(self, gid):
        return self.labels().get(gid, gid)

    def mapping_copy(self):
        with self.lock:
            return dict(self.mapping)

    def action_for(self, gesture_name):
        with self.lock:
            return self.mapping.get(gesture_name, "none")

    def apply(self, msg):
        """Apply a `config_set` message's `thresholds` / `mapping` (both optional). Unknown keys,
        unknown gesture/action ids, and values that don't cast are silently skipped rather than
        raising -- a malformed message from a client should never be able to crash the server."""
        for k, v in (msg.get("thresholds") or {}).items():
            spec = THRESHOLD_FIELDS.get(k)
            if spec is None:
                continue
            lo, hi, cast = spec
            try:
                v = cast(v)
            except (TypeError, ValueError):
                continue
            if lo is not None:
                v = min(max(v, lo), hi)
            setattr(self.args, k, v)
        mp = msg.get("mapping")
        if isinstance(mp, dict):
            known = self.labels()
            with self.lock:
                for g, a in mp.items():
                    if g in known and a in ACTION_META:
                        self.mapping[g] = a
            if (
                self.learner
            ):  # custom gestures' actions persist with them (outside our lock)
                self.learner.save()

    def snapshot(self):
        with self.lock:
            mapping = dict(self.mapping)
        return {
            "type": "config",
            "thresholds": {k: getattr(self.args, k) for k in THRESHOLD_FIELDS},
            "mapping": mapping,
            "gestures": self.labels(),
            "actions": ACTION_META,
            "learned": self.learner.summary() if self.learner else {},
            "teach": self.learner.teach_msg() if self.learner else None,
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
    MAG_SWEEP_S = 1.0  # a mag-assisted sweep must be completed within this long
    MAG_MIN_SPAN_S = 0.25  # ... and be watched for at least this long
    MAG_REFRACT_S = (
        1.2  # after any gesture, ignore sweeps this long (so the slow return isn't one)
    )

    def __init__(self, meta, args):
        self.cfg = args  # read live every packet, including by this field's own name: a `config_set`
        # (--flip-ud/--flip-lr included) takes effect on the very next packet, no restart needed.
        self.axis = np.asarray(meta["pen_axis"], float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.bias = np.asarray(meta["gyro_bias_dps"], float)
        self.acc_scale = float(meta["acc_scale"])
        # Calibrated magnetometer (hard-iron offset, soft-iron scale, axis alignment from calib.json).
        # Without it, mag assist is simply unavailable and everything runs on accel + gyro as before.
        self.mag_cal = None
        self.mag_field = 0.0
        if meta_has_mag(meta):
            self.mag_cal = (meta["mag_offset"], meta["mag_scale"], meta["mag_align"])
            self.mag_field = float(meta.get("mag_field") or 0.0)
        # Taught gestures (swapped in whole by Learner._publish, so a reader never sees a half-built
        # dict): {id: {"kind": "rot"|"lin", "dir": unit 3-vector, "thr": float}}; ids that override
        # a factory motion are in `overridden` and that motion's built-in detection is switched off.
        self.learned = {}
        self.overridden = frozenset()
        self.w_dps = np.zeros(
            3
        )  # latest bias-corrected gyro (deg/s, sensor frame), for the Learner
        self.lin_g = np.zeros(
            3
        )  # latest gravity-removed accel (g, heading frame, see feed())
        self.gyro_sign = 1.0  # +1 if the gyro follows the right-hand rule vs. the accel axes; learned live
        self._sign_ev = 0.0
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
        self._prev_ah = None
        self._mh = []  # mag sweep history: (t, unwrapped heading rad, cumulative gyro yaw rad)
        self._psi_raw = None
        self._psi_unw = 0.0
        self._yaw_cum = 0.0
        self._mag_block_until = 0.0
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
        self._mh = []

    def _lock(self, t):
        self._mh = []
        self._mag_block_until = t + self.MAG_REFRACT_S
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
        an = float(np.linalg.norm(a))
        w_rad = np.radians(w)
        # Gravity must follow the pen as it ROTATES, otherwise a flick's own rotation leaks gravity
        # into the "linear" accel and what you measure depends on how the pen happens to be oriented.
        # So: (1) work out whether the gyro axes are right-handed relative to the accel (sign learned
        # from the data: a fixed world vector changes by -w x v in body axes), (2) turn gravity with
        # the gyro every packet, (3) nudge it toward the accelerometer whenever that reads ~1 g.
        ah = a / an if an > 1e-6 else None
        if (
            ah is not None
            and self._prev_ah is not None
            and abs(an - 1.0) < 0.1
            and 40.0 < wn < 400.0
        ):
            pred = -np.cross(w_rad, self._prev_ah) * dt
            self._sign_ev = 0.995 * self._sign_ev + float((ah - self._prev_ah) @ pred)
            self.gyro_sign = 1.0 if self._sign_ev >= 0 else -1.0
        self._prev_ah = ah
        self.g = self.g - self.gyro_sign * np.cross(w_rad, self.g) * dt
        if abs(an - 1.0) < 0.15:
            tau = 0.4 if wn < 150.0 else 2.0
            self.g += (1.0 - math.exp(-dt / tau)) * (a - self.g)
        u = self.g / max(float(np.linalg.norm(self.g)), 1e-6)
        lin = a - self.g

        ax = self.axis
        f = ax - (ax @ u) * u
        nf = float(np.linalg.norm(f))
        head_ok = nf > 0.25
        if nf > 0.25:  # pen not (nearly) vertical -> heading is defined
            self.f = f / nf
        if self.f is None:
            f = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
            self.f = f / np.linalg.norm(f)
        f = self.f - (self.f @ u) * u
        f /= max(float(np.linalg.norm(f)), 1e-6)
        l = np.cross(u, f)
        yaw, pitch, roll = float(w @ np.cross(ax, l)), float(w @ l), float(w @ ax)
        # Heading frame (forward = where the pen points, horizontally; left; world up), built from
        # gravity and the pen's own heading. Taught gestures are recorded and matched in it, so they
        # don't care which way you face, how the pen is tilted, or how it is rolled in your hand.
        B = np.vstack([f, l, u])
        wp, lp = B @ w, B @ lin
        self.w_dps, self.lin_g = wp, lp
        d = f + ax
        d /= max(float(np.linalg.norm(d)), 1e-6)
        thrust = float(lin @ d)

        up_rate = -pitch * (-1.0 if cfg.flip_ud else 1.0)  # + = up
        right_rate = -yaw * (-1.0 if cfg.flip_lr else 1.0)  # + = right
        tw_rate = roll * cfg.twist_sign  # + = clockwise
        T, Tt = cfg.flick_dps, cfg.thrust_g
        sc = {
            "flick_up": max(0.0, up_rate) / T,
            "flick_down": max(0.0, -up_rate) / T,
            "flick_left": max(0.0, -right_rate) / T,
            "flick_right": max(0.0, right_rate) / T,
            "thrust_fwd": max(0.0, thrust) / Tt,
            "thrust_back": max(0.0, -thrust) / Tt,
        }
        for gid in self.overridden:
            sc.pop(gid, None)
        # Magnetometer-assisted slow sweeps (+ = swept left). Only ever ADDS a left/right score, and
        # only when the gyro-integrated sweep agrees (see _mag_sweep), so interference can't fire it.
        sweep, mag_info = self._mag_sweep(t, dt, p, u, f, l, float(wp[2]), head_ok)
        if cfg.flip_lr:
            sweep = -sweep
        if "flick_left" in sc:
            sc["flick_left"] = max(sc["flick_left"], max(0.0, sweep))
        if "flick_right" in sc:
            sc["flick_right"] = max(sc["flick_right"], max(0.0, -sweep))
        lg = {}
        for gid, sig in self.learned.items():
            v = wp if sig["kind"] == "rot" else lp
            n = float(np.linalg.norm(v))
            proj = float(v @ sig["dir"])
            # |v| * cos^2: both "how hard" and "how well aimed" count, so a big motion about some
            # other axis doesn't fire this gesture.
            lg[gid] = (
                max(0.0, proj) * (max(0.0, proj) / n) / sig["thr"] if n > 1e-6 else 0.0
            )
        sc.update(lg)

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
                    # Taught gestures get first say: you showed the pen exactly what you mean, so a
                    # clear winner among them beats the factory guesses (which, if the grip differs
                    # from the pen-axis assumption, would otherwise "see" the same motion as
                    # something else and veto it).
                    lp = sorted(
                        ((k, v) for k, v in self.peaks.items() if k in self.learned),
                        key=lambda kv: -kv[1],
                    )[:2]
                    if (
                        lp
                        and lp[0][1] >= 1.0
                        and (len(lp) < 2 or lp[0][1] >= self.DOM * lp[1][1])
                    ):
                        events.append(lp[0])
                    else:
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
        if lg:
            dbg["lg"] = {k: round(v, 3) for k, v in lg.items()}
        if mag_info is not None:
            dbg["mag"] = (
                mag_info  # {"ok": field trusted, "deg": signed sweep seen (+ = left)}
            )
        return events, dbg

    def _mag_sweep(self, t, dt, p, u, f, l, wu, head_ok):
        """Slow left/right sweeps (too slow for the rate threshold) measured as a change of
        tilt-compensated magnetic heading, which is drift-free and ignores hard-iron offsets.
        Returns (signed score where 1.0 = `mag_sweep_deg` swept, + = left; debug dict or None).

        Safe by construction: BOTH the magnetometer heading change and the gyro-integrated yaw over the
        same window must show the sweep (the score uses the smaller of the two, same direction), so a
        magnet or a laptop moving near the pen cannot create a gesture. The sample is ignored if the
        field strength is off its calibrated value, the horizontal field is tiny, or the pen is nearly
        vertical (heading undefined)."""
        cfg = self.cfg
        if not getattr(cfg, "mag_assist", False) or self.mag_cal is None:
            self._mh = []
            return 0.0, None
        self._yaw_cum += (
            math.radians(wu) * dt
        )  # gyro yaw (+ = left), accumulated for corroboration
        m = np.array(p[8:11], float) if len(p) >= 11 else None
        ok = (
            m is not None
            and bool(np.isfinite(m).all())
            and float(np.abs(m).max()) > 0.0
        )
        mh = None
        if ok:
            off, scl, al = self.mag_cal
            m = apply_mag_cal(m[None, :], off, scl, al)[0]
            mn = float(np.linalg.norm(m))
            mh = m - (m @ u) * u  # horizontal part of the field
            ok = (
                mn > 1e-6
                and (self.mag_field <= 0 or abs(mn / self.mag_field - 1.0) < 0.30)
                and float(np.linalg.norm(mh)) > 0.15 * mn
            )
        if not ok or not head_ok or t < self._mag_block_until:
            self._mh = []
            return 0.0, {"ok": bool(ok and head_ok), "deg": 0.0}
        # angle of magnetic north as seen from the pen's heading (toward the writer's left = +)
        psi = math.atan2(float(mh @ l), float(mh @ f))
        if self._psi_raw is not None and self._mh:
            self._psi_unw += (psi - self._psi_raw + math.pi) % (2 * math.pi) - math.pi
        else:
            self._psi_unw = psi
        self._psi_raw = psi
        self._mh.append((t, self._psi_unw, self._yaw_cum))
        while self._mh and t - self._mh[0][0] > self.MAG_SWEEP_S:
            self._mh.pop(0)
        t0, psi0, yaw0 = self._mh[0]
        if t - t0 < self.MAG_MIN_SPAN_S:
            return 0.0, {"ok": True, "deg": 0.0}
        mag_left = -math.degrees(
            self._psi_unw - psi0
        )  # pen turns left => north turns right in its frame
        gyr_left = math.degrees(self._yaw_cum - yaw0)
        if mag_left * gyr_left <= 0:
            return 0.0, {"ok": True, "deg": round(mag_left, 1)}
        deg = math.copysign(min(abs(mag_left), abs(gyr_left)), mag_left)
        return deg / max(float(cfg.mag_sweep_deg), 1.0), {
            "ok": True,
            "deg": round(deg, 1),
        }

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
                name = "twist_cw" if step_dir > 0 else "twist_ccw"
                if name not in self.overridden:  # a taught twist handles it instead
                    events.append((name, abs(self.tw_acc) / cfg.twist_deg))
                self.tw_dir = step_dir
                self.tw_opp_until = t + self.TW_OPP_S
                self.flick_block_until = t + 0.25
                self._lock(
                    t
                )  # one twist = one move: wait for the pen to settle before another
            elif self.tw_dir == s:
                self.tw_opp_until = max(self.tw_opp_until, t + self.TW_OPP_S)
        elif t - self.tw_last_ok > 0.15:
            self.tw_acc = 0.0


# =============================================================================================
# Learner: define gestures by demonstration.
#
# A rep is one motion between two calm moments. For each rep we keep the *peak vector of the first
# lobe* (the stroke itself, not the return stroke) of either the gyro (kind "rot") or the
# gravity-removed accel (kind "lin"), whichever is the bigger part of the motion, in the sensor's
# HEADING frame (see GestureDetector.feed). Across reps we average the unit directions: `consistency` is
# the length of that mean (1.0 = every rep pointed the exact same way). The saved signature is
# direction + typical peak.
#
# The heading frame is built from gravity and where the pen points, and gravity is tracked through rotations, so
# a taught gesture ignores which way you face, how the pen is tilted and how it is rolled in your hand.
# =============================================================================================
BUILTIN_IDS = frozenset(GESTURE_META)


def _slug(text):
    s = "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
    return "_".join(p for p in s.split("_") if p) or "gesture"


def _first_lobe_peak(n):
    """Index of the peak of the first contiguous stretch of `n` that is >= 60% of its max."""
    ok = n >= 0.6 * n.max()
    i0 = int(np.argmax(ok))
    j = i0
    while j + 1 < len(n) and ok[j + 1]:
        j += 1
    return i0 + int(np.argmax(n[i0 : j + 1]))


class Learner:
    TARGET = 5  # reps the UI asks for
    MIN_REPS = 3  # fewest reps that can be saved
    ONSET_DPS, ONSET_G = (
        90.0,
        0.30,
    )  # a rep starts when the pen moves at least this much
    QUIET_DPS, QUIET_G = (
        70.0,
        0.35,
    )  # ... and ends once it has been below this for QUIET_S
    QUIET_S, SETTLE_S, MAX_REP_S = 0.12, 0.15, 1.5
    REF_DPS, REF_G = 250.0, 0.6  # scale used only to decide rot vs lin
    MIN_ROT_DPS, MIN_LIN_G = 60.0, 0.20  # weaker reps are rejected as "too gentle"
    FRAC = 0.55  # detection threshold = FRAC * typical peak (before sensitivity)
    FLOOR_ROT, FLOOR_LIN = 50.0, 0.15  # ...but never lower than this
    MIN_CONSISTENCY = 0.75

    def __init__(self, bc, det, config, path):
        self.bc, self.det, self.config, self.path = bc, det, config, path
        self.lock = threading.RLock()
        self.sigs = {}  # id -> persisted signature (JSON-able)
        self.teach = None
        config.learner = self
        self._load()
        self._publish()

    # -- persistence --------------------------------------------------------------------------
    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            print(
                f"[learn] couldn't read {self.path} ({e}); starting with no taught gestures."
            )
            return
        if int(data.get("version", 1)) < 2:
            bak = self.path + ".v1.bak"
            try:
                if not os.path.exists(bak):
                    os.replace(self.path, bak)
            except OSError:
                pass
            print(
                f"[learn] {self.path} holds old sensor-frame gestures that depended on how the pen was "
                f"held; they can't be converted. Please re-teach them (old file kept as {bak})."
            )
            return
        actions = {}
        for gid, s in (data.get("gestures") or {}).items():
            try:
                d = [float(x) for x in s["dir"]]
                if (
                    s["kind"] not in ("rot", "lin")
                    or len(d) != 3
                    or float(s["ref"]) <= 0
                ):
                    raise ValueError("bad signature")
                if not all(math.isfinite(x) for x in d) or not any(d):
                    raise ValueError("bad direction")
                self.sigs[gid] = {
                    "label": str(s.get("label") or gid)[:24],
                    "custom": gid not in BUILTIN_IDS,
                    "kind": s["kind"],
                    "dir": d,
                    "ref": float(s["ref"]),
                    "sens": min(max(float(s.get("sens", 1.0)), 0.4), 2.5),
                    "reps": int(s.get("reps", 0)),
                    "consistency": float(s.get("consistency", 0.0)),
                }
                if s.get("action") in ACTION_META:
                    actions[gid] = s["action"]
            except (KeyError, TypeError, ValueError) as e:
                print(f"[learn] skipping bad entry '{gid}' in {self.path}: {e}")
        with self.config.lock:
            for gid, a in actions.items():
                if gid not in BUILTIN_IDS:
                    self.config.mapping[gid] = a
        if self.sigs:
            print(f"[learn] loaded {len(self.sigs)} taught gesture(s) from {self.path}")

    def save(self):
        mapping = (
            self.config.mapping_copy()
        )  # config lock first, never while holding ours
        with self.lock:
            out = {}
            for gid, s in self.sigs.items():
                e = dict(s)
                if not s["custom"]:
                    e.pop("custom", None)
                else:
                    e["action"] = mapping.get(gid, "none")
                out[gid] = e
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"version": 2, "gestures": out}, f, indent=2)
                os.replace(tmp, self.path)
            except OSError as e:
                print(f"[learn] couldn't save {self.path}: {e}")

    def _publish(self):
        """Push the signatures into the live detector."""
        with self.lock:
            rt = {}
            for gid, s in self.sigs.items():
                d = np.asarray(s["dir"], float)
                d = d / np.linalg.norm(d)
                floor = self.FLOOR_ROT if s["kind"] == "rot" else self.FLOOR_LIN
                rt[gid] = {
                    "kind": s["kind"],
                    "dir": d,
                    "thr": max(floor, self.FRAC * s["ref"]) / s["sens"],
                }
            self.det.learned = rt
            self.det.overridden = frozenset(g for g in rt if g in BUILTIN_IDS)

    # -- views --------------------------------------------------------------------------------
    def custom_labels(self):
        with self.lock:
            return {g: s["label"] for g, s in self.sigs.items() if s["custom"]}

    def summary(self):
        with self.lock:
            return {
                g: {
                    "label": s["label"],
                    "custom": s["custom"],
                    "kind": s["kind"],
                    "reps": s["reps"],
                    "consistency": round(s["consistency"], 2),
                    "sens": s["sens"],
                }
                for g, s in self.sigs.items()
            }

    def teach_msg(self, note="", error=""):
        with self.lock:
            tc = self.teach
            if tc is None:
                return None
            fit = self._fit(tc["reps"]) if tc["reps"] else None
            return {
                "type": "teach",
                "state": "recording" if tc["buf"] is not None else "waiting",
                "id": tc["id"],
                "label": tc["label"],
                "custom": tc["custom"],
                "reps": len(tc["reps"]),
                "target": self.TARGET,
                "min_reps": self.MIN_REPS,
                "consistency": round(fit[3], 2) if isinstance(fit, tuple) else None,
                "note": note,
                "error": error,
            }

    def _say(self, note="", error=""):
        m = self.teach_msg(note, error)
        self.bc.send(m if m else {"type": "teach", "state": "idle"})

    def _publish_all(self):
        self.bc.send(self.config.snapshot())

    # -- maths --------------------------------------------------------------------------------
    @staticmethod
    def _fit(reps):
        """reps = [(kind, vec)] -> (kind, unit dir, ref peak, consistency, n_used) or an error str."""
        kinds = [k for k, _ in reps]
        kind = max(set(kinds), key=kinds.count)
        use = [np.asarray(v, float) for k, v in reps if k == kind]
        U = [v / np.linalg.norm(v) for v in use]
        m = np.mean(U, 0)
        c = float(np.linalg.norm(m))
        if c < 1e-6:
            return "the reps cancel each other out (opposite directions?)"
        d = m / c
        peak = [max(0.0, v @ d) * max(0.0, (v @ d) / np.linalg.norm(v)) for v in use]
        return kind, d, float(np.median(peak)), c, len(use)

    def _analyse(self, buf):
        """One recorded rep -> (kind, peak vector) or an error string."""
        ts = np.array([b[0] for b in buf])
        keep = ts - ts[0] <= 0.35
        pick = {}
        for kind, idx in (("rot", 1), ("lin", 2)):
            V = np.array([b[idx] for b in buf])[keep]
            n = np.linalg.norm(V, axis=1)
            i = _first_lobe_peak(n)
            pick[kind] = (V[i], float(n[i]))
        kind = (
            "rot"
            if pick["rot"][1] / self.REF_DPS >= pick["lin"][1] / self.REF_G
            else "lin"
        )
        vec, mag = pick[kind]
        if mag < (self.MIN_ROT_DPS if kind == "rot" else self.MIN_LIN_G):
            return "too gentle, move a bit more firmly"
        return kind, vec

    # -- per packet (main thread, gesture mode only) ------------------------------------------
    def teaching(self):
        return self.teach is not None

    def feed(self, t, w, lin):
        """Returns True while a teach session is open (the caller then ignores normal gestures)."""
        with self.lock:
            tc = self.teach
            if tc is None:
                return False
            wn, ln = float(np.linalg.norm(w)), float(np.linalg.norm(lin))
            calm = wn < self.QUIET_DPS and ln < self.QUIET_G
            if tc["buf"] is None:
                # "settled" = the pen was calm for SETTLE_S *before* this packet (so this packet's own
                # movement can't cancel it)
                settled = (
                    tc["calm_since"] is not None
                    and t - tc["calm_since"] >= self.SETTLE_S
                )
                if calm:
                    if tc["calm_since"] is None:
                        tc["calm_since"] = t
                else:
                    tc["calm_since"] = None
                if settled and (wn > self.ONSET_DPS or ln > self.ONSET_G):
                    tc["buf"] = [(t, np.array(w, float), np.array(lin, float))]
                    tc["end_calm"] = None
                    self._say("recording...")
                return True
            tc["buf"].append((t, np.array(w, float), np.array(lin, float)))
            if calm:
                if tc["end_calm"] is None:
                    tc["end_calm"] = t
            else:
                tc["end_calm"] = None
            ended = tc["end_calm"] is not None and t - tc["end_calm"] >= self.QUIET_S
            if ended or t - tc["buf"][0][0] > self.MAX_REP_S:
                buf, tc["buf"], tc["calm_since"] = tc["buf"], None, None
                res = self._analyse(buf)
                if isinstance(res, str):
                    print(f"  [teach] rep rejected: {res}")
                    self._say(error=f"Rep not counted: {res}.")
                else:
                    tc["reps"].append(res)
                    print(f"  [teach] rep {len(tc['reps'])}/{self.TARGET} ({res[0]})")
                    self._say(note="Rep recorded. Pause, then do it again.")
            return True

    def interrupt(self):
        """Gesture mode ended (finger lifted): drop a half-recorded rep."""
        with self.lock:
            if self.teach is not None and self.teach["buf"] is not None:
                self.teach["buf"] = None
                self.teach["calm_since"] = None
                self._say(note="Interrupted: that rep was dropped.")

    # -- websocket messages (broadcaster thread) ----------------------------------------------
    def on_message(self, msg):
        kind = msg.get("type")
        with self.lock:
            if kind == "teach_start":
                self._start(msg)
            elif kind == "teach_cancel":
                self.teach = None
                self._say()
            elif kind == "teach_undo" and self.teach and self.teach["reps"]:
                self.teach["reps"].pop()
                self._say(note="Last rep removed.")
            elif kind == "teach_finish" and self.teach:
                self._finish()
            elif kind == "gesture_delete":
                self._delete(msg.get("id"))
            elif kind == "gesture_set":
                self._set(msg)

    def _start(self, msg):
        gid = msg.get("id")
        if gid in BUILTIN_IDS:
            label, custom = GESTURE_META[gid], False
        elif gid in self.sigs and self.sigs[gid]["custom"]:
            label, custom = self.sigs[gid]["label"], True
        else:
            label = " ".join(str(msg.get("label") or "").split())[:24]
            if not label:
                self.bc.send(
                    {
                        "type": "teach",
                        "state": "idle",
                        "error": "Give the new gesture a name first.",
                    }
                )
                return
            base = "custom_" + _slug(label)
            gid, i = base, 2
            while gid in self.sigs or gid in BUILTIN_IDS:
                gid, i = f"{base}_{i}", i + 1
            custom = True
        self.teach = {
            "id": gid,
            "label": label,
            "custom": custom,
            "reps": [],
            "buf": None,
            "calm_since": None,
            "end_calm": None,
        }
        print(
            f"  [teach] teaching '{label}' ({gid}); enter gesture mode and do the motion {self.TARGET}x"
        )
        self._say(note="Enter gesture mode, then do the motion.")

    def _finish(self):
        tc = self.teach
        if len(tc["reps"]) < self.MIN_REPS:
            self._say(
                error=f"Need at least {self.MIN_REPS} reps ({len(tc['reps'])} so far)."
            )
            return
        fit = self._fit(tc["reps"])
        if isinstance(fit, str):
            self._say(error=f"Can't learn that: {fit}.")
            return
        kind, d, ref, cons, n_used = fit
        if n_used < self.MIN_REPS:
            self._say(
                error="The reps were a mix of rotations and pushes. Undo the odd ones out, or redo it more consistently."
            )
            return
        if cons < self.MIN_CONSISTENCY:
            self._say(
                error=f"Reps point in different directions (consistency {cons:.0%}, need {self.MIN_CONSISTENCY:.0%}). Undo the odd ones or redo them."
            )
            return
        gid = tc["id"]
        old = self.sigs.get(gid)
        similar = [
            s["label"]
            for g, s in self.sigs.items()
            if g != gid
            and s["kind"] == kind
            and float(np.dot(d, s["dir"]) / np.linalg.norm(s["dir"])) > 0.9
        ]
        self.sigs[gid] = {
            "label": tc["label"],
            "custom": tc["custom"],
            "kind": kind,
            "dir": [float(x) for x in d],
            "ref": ref,
            "sens": old["sens"] if old else 1.0,
            "reps": n_used,
            "consistency": cons,
        }
        self.teach = None
        self._publish()
        done = {
            "type": "teach",
            "state": "done",
            "id": gid,
            "label": tc["label"],
            "reps": n_used,
            "consistency": round(cons, 2),
        }
        if similar:
            done["note"] = (
                "Heads up: very similar to "
                + ", ".join(similar)
                + "; they may get confused."
            )
        print(
            f"  [teach] saved '{tc['label']}' ({kind}, {n_used} reps, consistency {cons:.0%})"
        )
        threading.Thread(target=self._after_change, args=(done,), daemon=True).start()

    def _after_change(self, done=None):
        # outside our lock: save() takes the config lock first
        self.save()
        self._publish_all()
        if done:
            self.bc.send(done)

    def _delete(self, gid):
        if gid not in self.sigs:
            return
        s = self.sigs.pop(gid)
        self._publish()
        if s["custom"]:
            with self.config.lock:
                self.config.mapping.pop(gid, None)
        print(
            f"  [teach] {'deleted' if s['custom'] else 'reset to factory'}: {s['label']}"
        )
        threading.Thread(target=self._after_change, daemon=True).start()

    def _set(self, msg):
        gid = msg.get("id")
        try:
            sens = min(max(float(msg.get("sensitivity")), 0.4), 2.5)
        except (TypeError, ValueError):
            return
        if gid in self.sigs:
            self.sigs[gid]["sens"] = sens
            self._publish()
            threading.Thread(target=self._after_change, daemon=True).start()


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
    def __init__(
        self,
        bc,
        meta,
        stream,
        typist,
        det,
        args,
        config,
        trace_scale=1.0,
        learner=None,
        auto=None,
    ):
        self.bc, self.meta, self.stream, self.typist = bc, meta, stream, typist
        self.det, self.args, self.config, self.scale = det, args, config, trace_scale
        self.learner = learner
        self.auto = auto
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
                    "  [GESTURE MODE on]  (see the settings panel for the current mapping). "
                    "Release to go back to writing."
                )
            if self.active:
                self.bc.send(dbg)
                if p[
                    1
                ]:  # ignore anything that "happens" after the finger has left the pad
                    if self.learner and self.learner.feed(
                        p[0], self.det.w_dps, self.det.lin_g
                    ):
                        pass  # teaching: motions are being recorded, not acted on
                    else:
                        for name, score in events:
                            self.handle(name, score)
            else:
                self._send(self._imu(p))
            return
        self.bc.send(self._imu(p))

    def on_end(self):
        if self.active:
            if self.learner:
                self.learner.interrupt()
            self.bc.send({"type": "mode", "mode": "write"})
            print("  [gesture mode off]")
        else:
            for m in self.buf:
                self.bc.send(m)
            self.buf = []
            self.bc.send({"type": "touch_end"})

    # -- gesture actions ----------------------------------------------------------------------
    def handle(self, name, score):
        """`name` is the raw motion the detector saw (e.g. "flick_up"); it's resolved through the
        CURRENT live mapping (config.action_for) to decide what actually happens -- this is the one
        place remapping takes effect. If a motion is unmapped ("none"), it's still reported to the
        frontend (for the settings panel / gauges) but does nothing to the text."""
        t, bc = self.typist, self.bc
        action = self.config.action_for(name)
        note = ""
        if action == "uppercase":
            t.caps = True
            bc.send({"type": "caps", "on": True})
        elif action == "lowercase":
            t.caps = False
            bc.send({"type": "caps", "on": False})
        elif action == "caps_toggle":
            t.caps = not t.caps
            bc.send({"type": "caps", "on": t.caps})
        elif action == "backspace":
            note = "" if t.backspace() else "  (nothing to delete)"
            bc.send(t.snapshot())
        elif action == "space":
            t.put(" ")
            bc.send(t.snapshot())
        elif action == "enter":
            t.put("\n")
            bc.send(t.snapshot())
        elif action == "cursor_back":
            note = "" if t.move(-1) else "  (at start)"
            bc.send(t.snapshot())
        elif action == "cursor_fwd":
            note = "" if t.move(1) else "  (at end)"
            bc.send(t.snapshot())
        elif action == "autocomplete":
            note = (
                self.auto.accept() if self.auto else "  (autocomplete is off: --no-llm)"
            )
        # action == "none": nothing to do to the text.
        gesture_label, action_label = self.config.label(name), ACTION_META[action]
        bc.send(
            {
                "type": "gesture",
                "name": name,
                "action": action,
                "gesture_label": gesture_label,
                "action_label": action_label,
                "score": round(score, 2),
            }
        )
        extra = (
            f"   [caps: {'ON ' if t.caps else 'off'}]"
            if action in ("uppercase", "lowercase", "caps_toggle")
            else ""
        )
        print(
            f"  [gesture] {gesture_label} -> {action_label}  (x{score:.1f}){note}{extra}"
        )
        if action not in ("uppercase", "lowercase", "caps_toggle", "none"):
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

    stream = PenStream(args.port, transport=args.transport, stats=args.stream_stats)
    meta = prepare_meta(stream, args.data, False, not args.no_rebias)
    if need_mag and not meta_has_mag(meta):
        sys.exit(
            f"Variant {v} needs a magnetometer calibration, but calib.json has none. "
            "Run `python collect.py --calibrate-mag` first."
        )

    typist = Typist()
    config = Config(
        args
    )  # starting thresholds come from the CLI flags below; remappable live.
    det = GestureDetector(meta, args)
    if getattr(args, "mag_assist", False):
        print(
            "Mag-assisted sweeps: "
            + (
                "ON (needs gyro agreement)"
                if det.mag_cal is not None
                else "requested but no mag calibration in calib.json -> inactive (run --calibrate-mag)"
            )
        )
    learner = Learner(
        broadcaster, det, config, os.path.join(args.data, "gestures.json")
    )
    auto = None
    if not args.no_llm:
        auto = Autocomplete(broadcaster, typist, args.llm_url, args.llm_model)
    # every finished word (typed, written or autocompleted) is read aloud by the page
    voice = Voice(broadcaster, enabled=not args.no_speak)
    voice.say(
        "Voice ready", force=True
    )  # proves the server -> spd-say path works, before any page

    def on_word(w):
        broadcaster.send(
            {"type": "speak", "word": w}
        )  # the page speaks it if it chose the browser route
        voice.say(w)  # ...or we do, if it chose the server route

    typist.on_word = on_word
    sess = Session(
        broadcaster, meta, stream, typist, det, args, config, learner=learner, auto=auto
    )

    def on_ws_message(msg):
        if msg.get("type") == "config_set":
            config.apply(msg)
            broadcaster.send(
                config.snapshot()
            )  # echo the resulting config to every client
        else:
            voice.on_message(msg)  # voice_set / voice_test
            learner.on_message(msg)  # teach_* / gesture_* (ignores anything else)

    broadcaster.on_message = on_ws_message
    broadcaster.snapshot = lambda: [
        config.snapshot(),
        {"type": "caps", "on": typist.caps},
        {"type": "mode", "mode": "gesture" if sess.active else "write"},
        typist.snapshot(),
        *(auto.snapshot_msgs() if auto else []),
        voice.snapshot(),
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
    broadcaster.send(config.snapshot())
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
    ap.add_argument(
        "--transport",
        choices=["auto", "udp", "ble"],
        default="auto",
        help="how the pen connects: udp = WiFi AP (the original), ble = Bluetooth LE "
        "(firmware USE_BLE 1, needs `pip install bleak`), auto = listen for both and use "
        "whichever the pen speaks (default)",
    )
    ap.add_argument(
        "--stream-stats",
        action="store_true",
        help="every 5 s print pkt/s, timestamp gaps and arrival-delay spread (to compare UDP vs BLE)",
    )
    ap.add_argument("--no-rebias", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="turn off LLM autocomplete")
    ap.add_argument(
        "--no-speak",
        action="store_true",
        help="don't speak finished words from the server (spd-say/espeak-ng)",
    )
    ap.add_argument(
        "--llm-url", default="http://localhost:11434", help="Ollama address"
    )
    ap.add_argument(
        "--llm-model",
        default=None,
        help="Ollama model for autocomplete (default: the qwen2.5 3b model Ollama has)",
    )
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

    g = ap.add_argument_group(
        "gesture mode (--live)",
        "Starting values only -- every one of these (plus the whole gesture->action mapping) can "
        "also be changed live, per the running server, from the frontend's settings panel, with no "
        "restart. These flags just set what a freshly started server begins with.",
    )
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
        help="forward/backward acceleration that counts as a thrust (default 0.6 g)",
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
    g.add_argument(
        "--no-mag-gestures",
        dest="mag_assist",
        action="store_false",
        help="don't use the magnetometer for gestures (default: slow left/right sweeps are confirmed by "
        "the mag heading change; needs a mag calibration in calib.json, otherwise it's inactive). "
        "Also a checkbox in the frontend settings.",
    )
    g.add_argument(
        "--mag-sweep-deg",
        type=float,
        default=35.0,
        help="degrees of heading change (within 1 s) for a mag-assisted left/right sweep (default 35)",
    )
    g.add_argument("--flip-ud", action="store_true", help="swap flick up / down")
    g.add_argument("--flip-lr", action="store_true", help="swap flick left / right")
    args = ap.parse_args()
    live(args) if args.live else evaluate(args)


if __name__ == "__main__":
    main()
