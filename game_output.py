"""Game transport and independent watchdog. No EEG/model dependencies."""
from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from pathlib import Path

CONTROL_NAMES = {0: "left", 1: "right", 2: "forward", 3: "stop"}


class TCPJSONOutlet:
    """Explicit site profile only; the official JSON wire schema is not supplied.

    A successful send means locally submitted bytes, never game acknowledgement.
    After reconnect, only decisions made after that connection may drive the car.
    """
    def __init__(self, profile_path):
        self.profile = json.loads(Path(profile_path).read_text(encoding="utf-8-sig"))
        p = self.profile
        if p.get("confirmed_by_organizer") is not True:
            raise ValueError("TCP profile must be confirmed by the organizer before use")
        if p.get("role") != "client" or not p.get("host") or not isinstance(p.get("port"), int) or not 0 < p["port"] < 65536:
            raise ValueError("TCP profile requires role=client, host and port (1..65535)")
        if p.get("framing") not in {"newline", "length-prefix-be", "raw"}:
            raise ValueError("Confirm framing: newline, length-prefix-be or raw")
        messages = p.get("messages", {})
        if set(messages) != set(CONTROL_NAMES.values()) or not all(isinstance(v, dict) and v for v in messages.values()):
            raise ValueError("TCP messages must contain the four exact JSON objects from the organizer")
        self._socket = None
        self._next_connect = 0.0
        self._connected_at = float("inf")

    def encode(self, control):
        payload = json.dumps(self.profile["messages"][CONTROL_NAMES[control]], ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        framing = self.profile["framing"]
        if framing == "newline":
            return payload + b"\n"
        if framing == "length-prefix-be":
            return struct.pack("!I", len(payload)) + payload
        return payload

    def push(self, control, decision_at=0.0):
        now = time.perf_counter()
        if self._socket is None:
            if now < self._next_connect:
                raise ConnectionError("TCP reconnect pending")
            self._next_connect = now + 1.0
            self._socket = socket.create_connection((self.profile["host"], self.profile["port"]), timeout=0.3)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._connected_at = time.perf_counter()
        actual = control if decision_at >= self._connected_at else 3
        try:
            self._socket.sendall(self.encode(actual))
        except OSError:
            self.close()
            raise
        return actual

    def have_consumers(self):
        return self._socket is not None

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class ControlPublisher:
    """10 Hz output independent of blocking acquisition and inference.

    Only submit() renews a decision. Expired decisions become stop automatically.
    guard() additionally enforces live EEG freshness. Shutdown attempts three stops.
    """
    def __init__(self, outlet, rate=10.0, guard=lambda: True, on_send=lambda *args: None):
        if not math.isfinite(rate) or not 0 < rate <= 100:
            raise ValueError("Control rate must be finite and in (0,100]")
        self.outlet, self.rate, self.guard, self.on_send = outlet, rate, guard, on_send
        self._lock = threading.RLock()
        self._halt = threading.Event()
        self._thread = None
        self._decision = (3, 0.0, 0.0)
        self.last_error = None
        self.last_sent = None
        self.last_sent_at = None
        self.consumer_online = False

    def submit(self, control, ttl):
        if control not in CONTROL_NAMES or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Invalid control or decision lifetime")
        with self._lock:
            if not self._halt.is_set():
                now = time.perf_counter()
                self._decision = (control, now, now + ttl)

    def invalidate(self):
        with self._lock:
            self._decision = (3, 0.0, 0.0)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="game-output", daemon=True)
        self._thread.start()

    def _send(self, stopping=False):
        try:
            with self._lock:
                control, decided, expires = self._decision
                allowed = not stopping and self.guard() and time.perf_counter() < expires
                if not allowed:
                    control = 3
                    self._decision = (3, 0.0, 0.0)
                if isinstance(self.outlet, TCPJSONOutlet):
                    actual = self.outlet.push(control, decided)
                else:
                    self.outlet.push(control)
                    actual = control
                self.last_sent = actual
                self.last_sent_at = time.time()
                self.consumer_online = self.outlet.have_consumers()
                self.last_error = None
        except Exception as exc:
            self.invalidate()
            self.last_error = str(exc)
            self.consumer_online = False
            actual = None
        self.on_send(actual, self.last_error)

    def _run(self):
        while not self._halt.is_set():
            started = time.perf_counter()
            self._send()
            self._halt.wait(max(0.0, 1.0 / self.rate - (time.perf_counter() - started)))
        for _ in range(3):
            self._send(stopping=True)
            time.sleep(0.03)
        close = getattr(self.outlet, "close", None)
        if close:
            close()

    def stop(self):
        self._halt.set()
        self.invalidate()
        if self._thread:
            self._thread.join(timeout=2.0)
