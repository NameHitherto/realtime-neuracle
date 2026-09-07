from __future__ import annotations

import csv
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from realtime_common import DataBlock


class ExperimentLogger:
    """Crash-tolerant streaming logger for one realtime experiment run.

    Raw EEG is stored sample-major as little-endian float32. Triggers are
    stored in a parallel little-endian int32 file. CSV/JSON metadata describes
    how the binary files can be reconstructed without any extra dependency.
    """

    def __init__(
        self,
        root_dir: str | Path,
        experiment_id: str,
        model_name: str,
        stream_channels: list[str],
        model_channels: list[str],
        class_names: list[str],
        control_map: dict[int, int],
        sampling_rate: float,
        settings: dict[str, Any],
    ) -> None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_id.strip()) or "experiment"
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(root_dir).expanduser().resolve() / f"{stamp}_{safe_id}"
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self.stream_channels = list(stream_channels)
        self.eeg_channels = [name for name in self.stream_channels if name.upper() != "TRG"]
        self.has_trigger = bool(self.stream_channels and self.stream_channels[-1].upper() == "TRG")
        self.model_channels = list(model_channels)
        self.class_names = list(class_names)
        self.control_map = {int(key): int(value) for key, value in control_map.items()}
        self.sampling_rate = float(sampling_rate)
        self.started_at = time.time()
        self.total_samples = 0
        self.total_blocks = 0
        self.total_predictions = 0
        self.total_controls = 0
        self.state = "recording"
        self._closed = False
        self._lock = threading.RLock()

        # Binary files are unbuffered so data already received survives an
        # abrupt GUI/process stop. Text logs are line-buffered for the same reason.
        self._eeg_file = (self.session_dir / "eeg_raw.f32").open("wb", buffering=0)
        self._trigger_file = (self.session_dir / "triggers.i32").open("wb", buffering=0)
        self._blocks_file = (self.session_dir / "blocks.csv").open(
            "w", encoding="utf-8-sig", newline="", buffering=1
        )
        self._predictions_file = (self.session_dir / "predictions.csv").open(
            "w", encoding="utf-8-sig", newline="", buffering=1
        )
        self._controls_file = (self.session_dir / "lsl_controls.csv").open(
            "w", encoding="utf-8-sig", newline="", buffering=1
        )
        self._events_file = (self.session_dir / "events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )

        self._blocks_writer = csv.writer(self._blocks_file)
        self._blocks_writer.writerow(
            ["block_index", "received_unix_s", "sample_start", "sample_count", "last_trigger"]
        )
        probability_columns = [f"raw_{name}" for name in self.class_names] + [
            f"smooth_{name}" for name in self.class_names
        ]
        self._predictions_writer = csv.writer(self._predictions_file)
        self._predictions_writer.writerow(
            [
                "timestamp_iso",
                "unix_s",
                "sample_index",
                "predicted_index",
                "predicted_label",
                "confidence",
                "game_control",
                "game_control_name",
                "latest_trigger",
                *probability_columns,
            ]
        )
        self._controls_writer = csv.writer(self._controls_file)
        self._controls_writer.writerow(
            ["timestamp_iso", "unix_s", "game_control", "game_control_name", "source"]
        )

        self._metadata: dict[str, Any] = {
            "format_version": 1,
            "state": "recording",
            "experiment_id": experiment_id,
            "model_name": model_name,
            "started_at_iso": datetime.now().astimezone().isoformat(),
            "started_at_unix_s": self.started_at,
            "sampling_rate_hz": self.sampling_rate,
            "stream_channels": self.stream_channels,
            "eeg_channels": self.eeg_channels,
            "trigger_channel": "TRG" if self.has_trigger else None,
            "model_channels": self.model_channels,
            "class_names": self.class_names,
            "control_map": self.control_map,
            "binary_layout": {
                "eeg_raw.f32": {
                    "dtype": "<f4",
                    "shape": ["total_samples", len(self.eeg_channels)],
                    "order": "sample-major",
                    "unit": "device-native; realtime preprocessing auto-detects V versus uV",
                },
                "triggers.i32": {
                    "dtype": "<i4",
                    "shape": ["total_samples"],
                },
            },
            "settings": settings,
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        target = self.session_dir / "session.json"
        temporary = self.session_dir / "session.json.tmp"
        temporary.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(target)

    @staticmethod
    def _iso_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).astimezone().isoformat()

    def log_data(self, block: DataBlock) -> None:
        eeg = np.asarray(block.eeg, dtype=np.float32)
        if eeg.ndim != 2 or eeg.shape[0] != len(self.eeg_channels):
            raise ValueError(
                f"Logger expected ({len(self.eeg_channels)}, T) EEG block, got {eeg.shape}"
            )
        n_samples = int(eeg.shape[1])
        if n_samples == 0:
            return
        if block.triggers is None:
            triggers = np.zeros(n_samples, dtype=np.int32)
        else:
            triggers = np.asarray(block.triggers, dtype=np.int32).reshape(-1)
            if triggers.size != n_samples:
                raise ValueError(
                    f"Logger received {triggers.size} triggers for {n_samples} EEG samples"
                )

        # Transpose to sample-major before writing and force the documented
        # little-endian representation independently of the host platform.
        eeg_sample_major = np.ascontiguousarray(eeg.T, dtype="<f4")
        triggers_le = np.ascontiguousarray(triggers, dtype="<i4")
        with self._lock:
            if self._closed:
                return
            sample_start = self.total_samples
            self._eeg_file.write(eeg_sample_major.tobytes(order="C"))
            self._trigger_file.write(triggers_le.tobytes(order="C"))
            self._blocks_writer.writerow(
                [
                    self.total_blocks,
                    f"{block.received_at:.6f}",
                    sample_start,
                    n_samples,
                    int(triggers[-1]),
                ]
            )
            self.total_samples += n_samples
            self.total_blocks += 1

    def log_prediction(
        self,
        timestamp: float,
        sample_index: int,
        predicted_index: int,
        confidence: float,
        game_control: int,
        game_control_name: str,
        latest_trigger: int | None,
        raw_probabilities: np.ndarray,
        smooth_probabilities: np.ndarray,
    ) -> None:
        raw = np.asarray(raw_probabilities, dtype=float).reshape(-1)
        smooth = np.asarray(smooth_probabilities, dtype=float).reshape(-1)
        if raw.size != len(self.class_names) or smooth.size != len(self.class_names):
            raise ValueError("Prediction probability count does not match class_names")
        with self._lock:
            if self._closed:
                return
            self._predictions_writer.writerow(
                [
                    self._iso_timestamp(timestamp),
                    f"{timestamp:.6f}",
                    int(sample_index),
                    int(predicted_index),
                    self.class_names[int(predicted_index)],
                    f"{float(confidence):.8f}",
                    int(game_control),
                    game_control_name,
                    "" if latest_trigger is None else int(latest_trigger),
                    *[f"{value:.8f}" for value in raw],
                    *[f"{value:.8f}" for value in smooth],
                ]
            )
            self.total_predictions += 1

    def log_event(self, level: str, message: str, **details: Any) -> None:
        item = {
            "timestamp_iso": datetime.now().astimezone().isoformat(),
            "unix_s": time.time(),
            "level": level,
            "message": message,
            "details": details,
        }
        with self._lock:
            if self._closed:
                return
            self._events_file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def log_control(
        self,
        timestamp: float,
        game_control: int,
        game_control_name: str,
        source: str,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._controls_writer.writerow(
                [
                    self._iso_timestamp(timestamp),
                    f"{timestamp:.6f}",
                    int(game_control),
                    game_control_name,
                    source,
                ]
            )
            self.total_controls += 1

    def close(self, state: str = "completed", error: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self.state = state
            self._metadata.update(
                {
                    "state": state,
                    "ended_at_iso": datetime.now().astimezone().isoformat(),
                    "ended_at_unix_s": time.time(),
                    "total_samples": self.total_samples,
                    "total_blocks": self.total_blocks,
                    "total_predictions": self.total_predictions,
                    "total_controls": self.total_controls,
                    "duration_s": self.total_samples / self.sampling_rate,
                    "error": error,
                }
            )
            self._write_metadata()
            for handle in (
                self._blocks_file,
                self._predictions_file,
                self._controls_file,
                self._events_file,
                self._eeg_file,
                self._trigger_file,
            ):
                handle.close()
            self._closed = True
