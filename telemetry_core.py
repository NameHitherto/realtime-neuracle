from __future__ import annotations

import json
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryBus:
    """Thread-safe bounded telemetry store shared by inference and FastAPI."""

    history_size: int = 240
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _latest: dict[str, Any] = field(default_factory=dict, init=False)
    _history: deque[dict[str, Any]] = field(init=False)
    _events: deque[dict[str, Any]] = field(init=False)
    _seq: int = field(default=0, init=False)
    _subscribers: set[Any] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_size)
        self._events = deque(maxlen=300)
        self.update(
            status={
                "device_tcp": "idle",
                "inference": "idle",
                "lsl_outlet": "idle",
                "lsl_consumer": "unknown",
                "game_process": "unknown",
                "game_capture": "idle",
            },
            model={},
            eeg={},
            lsl={},
            game={"telemetry_level": "inferred"},
            health={"level": "yellow", "message": "等待启动"},
        )

    def update(self, **sections: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            now = time.time()
            for key, value in sections.items():
                if isinstance(value, dict) and isinstance(self._latest.get(key), dict):
                    self._latest[key] = {**self._latest[key], **value}
                else:
                    self._latest[key] = deepcopy(value)
            self._latest["seq"] = self._seq
            self._latest["ts"] = now
            snapshot = deepcopy(self._latest)
            self._history.append(snapshot)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(snapshot)
            except Exception:
                self.unsubscribe(callback)
        return snapshot

    def event(self, level: str, message: str, **details: Any) -> dict[str, Any]:
        item = {
            "ts": time.time(),
            "level": level,
            "message": message,
            "details": details,
        }
        with self._lock:
            self._events.appendleft(item)
        return item

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._latest)

    def history(self, limit: int = 120) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(list(self._history)[-max(1, min(limit, self.history_size)) :])

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(list(self._events)[: max(1, min(limit, 300))])

    def subscribe(self, callback: Any) -> None:
        with self._lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback: Any) -> None:
        with self._lock:
            self._subscribers.discard(callback)

    def export_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":"))
