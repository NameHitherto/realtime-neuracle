from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from telemetry_core import TelemetryBus
from realtime_common import (
    DataServer,
    LSLControlOutlet,
    RealTimePreprocessor,
    predict_window,
    select_model_channels,
    parse_channel_file,
    parse_channel_list,
    load_state_dict,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Importing the existing model entrypoints keeps their checkpoint/model
# definitions as the single source of truth.
sys.path.insert(0, str(SCRIPT_DIR))
from realtime_bcic2a_4class import (  # noqa: E402
    BCIC2A_CHANNELS,
    CLASS_NAMES as BCIC2A_CLASSES,
    CONTROL_MAP as BCIC2A_CONTROL_MAP,
    DEFAULT_CHECKPOINT as BCIC2A_CHECKPOINT,
    build_model as build_bcic2a,
)
from realtime_hgd_4class import (  # noqa: E402
    HGD_CHANNELS,
    CLASS_NAMES as HGD_CLASSES,
    CONTROL_MAP as HGD_CONTROL_MAP,
    DEFAULT_CHECKPOINT as HGD_CHECKPOINT,
    build_model as build_hgd,
)

CONTROL_NAMES = {0: "left", 1: "right", 2: "forward", 3: "stop"}
DISPLAY_CHANNELS = ["C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]
LOCAL_CHANNEL_DERIVATIONS = {
    "CPZ": ("CP1", "CP2"),
    "P1": ("PZ", "P3"),
    "P2": ("PZ", "P4"),
}


class LocalEEGSimulator:
    """Simulates EEG data stream from a local BDF file using mne."""

    def __init__(self, file_path: str, channel_names: list[str], sampling_rate: float, buffer_seconds: float):
        self.file_path = file_path
        self.channel_names = list(channel_names)
        self.sampling_rate = float(sampling_rate)
        self.eeg_channel_names = [ch for ch in self.channel_names if ch.upper() != "TRG"]
        self.n_chan_eeg = len(self.eeg_channel_names)
        from realtime_common import RingBuffer
        self.buffer = RingBuffer(self.n_chan_eeg, int(math.ceil(buffer_seconds * sampling_rate)))
        self._data: np.ndarray | None = None
        self._current_pos = 0
        self._total_samples = 0
        self._started_at = 0.0
        self._samples_emitted = 0
        self.derived_channels: dict[str, tuple[str, ...]] = {}

    def __enter__(self):
        import mne
        raw = mne.io.read_raw_bdf(self.file_path, preload=True, verbose=False)
        available_by_name = {name.strip().upper(): name for name in raw.ch_names}
        missing: list[str] = []
        channel_sources: list[tuple[str, ...]] = []
        for channel in self.eeg_channel_names:
            key = channel.strip().upper()
            if key in available_by_name:
                channel_sources.append((available_by_name[key],))
                continue
            source_keys = LOCAL_CHANNEL_DERIVATIONS.get(key)
            if source_keys and all(source in available_by_name for source in source_keys):
                sources = tuple(available_by_name[source] for source in source_keys)
                channel_sources.append(sources)
                self.derived_channels[channel] = sources
                continue
            missing.append(channel)
        if missing:
            raise ValueError(
                "BDF file is missing required model channels: "
                + ", ".join(missing)
                + ". Available channels: "
                + ", ".join(raw.ch_names)
            )
        file_sfreq = float(raw.info["sfreq"])
        if not math.isclose(file_sfreq, self.sampling_rate, rel_tol=0.0, abs_tol=1e-6):
            raw.resample(self.sampling_rate, verbose=False)
        # Preserve model order and average neighboring electrodes for absent
        # intermediate 10-10 positions in this 64-channel cap.
        rows = [raw.get_data(picks=list(sources)).mean(axis=0) for sources in channel_sources]
        # MNE exposes EEG in volts. The rest of this realtime stack and the
        # dashboard use microvolts, matching the Neuracle TCP stream.
        self._data = np.asarray(rows, dtype=np.float32) * 1_000_000.0
        self._total_samples = self._data.shape[1]
        self._current_pos = 0
        self._started_at = time.monotonic()
        self._samples_emitted = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        self._data = None

    def poll(self, max_samples: int | None = None):
        if self._data is None:
            raise RuntimeError("LocalEEGSimulator is not initialized")
        max_samples = max_samples or max(1, int(self.sampling_rate // 10))
        due = int((time.monotonic() - self._started_at) * self.sampling_rate) - self._samples_emitted
        if due <= 0:
            time.sleep(min(0.01, 1.0 / self.sampling_rate))
            return
        remaining = min(due, max_samples)
        chunks: list[np.ndarray] = []
        while remaining > 0:
            count = min(remaining, self._total_samples - self._current_pos)
            chunks.append(self._data[:, self._current_pos : self._current_pos + count])
            self._current_pos = (self._current_pos + count) % self._total_samples
            self._samples_emitted += count
            remaining -= count
        self.buffer.append(np.concatenate(chunks, axis=1))


class InferenceRuntime:
    def __init__(self, args: argparse.Namespace, bus: TelemetryBus):
        self.args = args
        self.bus = bus
        self.model_name = args.model
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        self.model_channels, self.class_names, self.control_map, default_checkpoint, self.build_model = self._model_spec()
        self.checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint
        self.model = self.build_model(self.device)
        load_state_dict(self.model, self.checkpoint, self.device)
        self.preprocessor = RealTimePreprocessor(
            input_sfreq=args.device_sfreq,
            model_sfreq=args.model_sfreq,
            window_seconds=args.window_sec,
            low_hz=args.filter_low,
            high_hz=args.filter_high,
            model_samples=1000,
            zscore_window=args.zscore_window,
            calibration_npz=args.calibration_npz,
        )
        self.outlet: LSLControlOutlet | None = None
        self.current_control = int(args.default_control)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_eeg_at = 0.0
        self._last_inference_at = 0.0
        self._smoother: deque[np.ndarray] = deque(maxlen=max(1, int(args.smooth_n)))

    def _model_spec(self):
        if self.model_name == "hgd":
            return HGD_CHANNELS, HGD_CLASSES, HGD_CONTROL_MAP, HGD_CHECKPOINT, build_hgd
        return BCIC2A_CHANNELS, BCIC2A_CLASSES, BCIC2A_CONTROL_MAP, BCIC2A_CHECKPOINT, build_bcic2a

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="inference-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.bus.update(status={"inference": "stopped", "device_tcp": "stopped"})

    def manual_control(self, control: int, duration_ms: int = 1000) -> None:
        if not 0 <= control <= 3:
            raise ValueError("control must be one of 0, 1, 2, 3")
        self.current_control = control
        if self.outlet:
            self.outlet.push(control)
        self.bus.event("info", "测试控制指令已发送", control=control, name=CONTROL_NAMES[control], duration_ms=duration_ms)
        self.bus.update(model={"control": control, "control_name": CONTROL_NAMES[control]}, lsl={"last_sent": control})

    def _publish(self) -> None:
        if self.outlet:
            self.outlet.push(self.current_control)
            self.bus.update(
                lsl={
                    "last_sent": self.current_control,
                    "last_sent_at": time.time(),
                    "consumer_online": self.outlet.have_consumers(),
                }
            )

    def _loop(self) -> None:
        if self.args.dry_run:
            self._dry_run()
            return
        try:
            self.outlet = LSLControlOutlet(
                stream_name=self.args.stream_name,
                stream_type=self.args.stream_type,
                sample_rate=self.args.lsl_rate,
                source_id=f"{self.model_name}_dashboard_control",
            )
            self.bus.update(
                status={"lsl_outlet": "running"},
                lsl={"stream_name": self.args.stream_name, "stream_type": self.args.stream_type, "sample_rate": self.args.lsl_rate},
            )
        except Exception as exc:
            self.bus.event("error", "LSL 输出初始化失败", error=str(exc))
            self.bus.update(status={"lsl_outlet": "error"}, health={"level": "red", "message": "pylsl/LSL 输出不可用"})
            return

        stream_channels = parse_channel_file(self.args.channel_list_file)
        if stream_channels is None:
            stream_channels = parse_channel_list(self.args.channel_list, self.model_channels + ["TRG"])
        eeg_stream_channels = stream_channels[:-1] if stream_channels and stream_channels[-1].upper() == "TRG" else stream_channels
        n_points = int(round(self.args.window_sec * self.args.device_sfreq))
        publish_interval = 1.0 / max(float(self.args.lsl_rate), 1e-6)
        next_publish = 0.0
        next_infer = time.time() + self.args.step_sec
        self.bus.update(
            status={"inference": "starting", "device_tcp": "connecting"},
            model={"name": self.model_name, "class_names": self.class_names, "device": str(self.device)},
            eeg={"sample_rate": self.args.device_sfreq, "channels": DISPLAY_CHANNELS, "display_rate": 50},
            game={"telemetry_level": "inferred"},
        )
        try:
            # Choose data source based on --eeg-source
            if self.args.eeg_source == "local":
                server = LocalEEGSimulator(
                    file_path=self.args.local_eeg_file,
                    channel_names=stream_channels,
                    sampling_rate=self.args.device_sfreq,
                    buffer_seconds=max(self.args.window_sec + 2.0, self.args.window_sec * 2),
                )
                self.bus.event("info", "使用本地 EEG 数据模拟", file=self.args.local_eeg_file)
            else:
                server = DataServer(
                    host=self.args.host,
                    port=self.args.port,
                    channel_names=stream_channels,
                    sampling_rate=self.args.device_sfreq,
                    buffer_seconds=max(self.args.window_sec + 2.0, self.args.window_sec * 2),
                )
                self.bus.event("info", "已连接 EEG TCP 数据流", host=self.args.host, port=self.args.port)
            
            with server:
                if isinstance(server, LocalEEGSimulator) and server.derived_channels:
                    self.bus.event(
                        "warning",
                        "本地 BDF 缺少部分模型导联，已使用邻近导联插值",
                        derived_channels=server.derived_channels,
                    )
                self.bus.update(status={"device_tcp": "connected", "inference": "running"})
                while not self._stop.is_set():
                    try:
                        server.poll()
                    except TimeoutError:
                        continue
                    except OSError as exc:
                        self.bus.event("error", "EEG 连接中断", error=str(exc))
                        self.bus.update(status={"device_tcp": "error", "inference": "error"}, health={"level": "red", "message": "EEG 连接中断"})
                        break
                    now = time.time()
                    if now >= next_publish:
                        self._publish()
                        next_publish = now + publish_interval
                    if now < next_infer:
                        continue
                    next_infer = now + self.args.step_sec
                    if server.buffer.n_updates < n_points:
                        continue
                    try:
                        raw_all = server.buffer.latest(n_points)
                        raw_model = select_model_channels(raw_all, eeg_stream_channels, self.model_channels)
                        window = self.preprocessor.transform(raw_model)
                        pred, conf, probs = predict_window(self.model, window, self.device, self._smoother)
                        self.current_control = self.control_map.get(pred, self.current_control) if conf >= self.args.min_confidence else self.args.default_control
                        self.last_prediction = pred
                        self._last_inference_at = now
                        display_indices = [eeg_stream_channels.index(ch) for ch in DISPLAY_CHANNELS if ch in eeg_stream_channels]
                        display = raw_all[display_indices, -min(raw_all.shape[-1], int(self.args.device_sfreq * 2)) :] if display_indices else raw_all[: min(4, raw_all.shape[0]), -int(self.args.device_sfreq * 2) :]
                        display = display - display.mean(axis=-1, keepdims=True)
                        display = self._downsample(display, 50)
                        self._last_eeg_at = now
                        self.bus.update(
                            eeg={"channels": [ch for ch in DISPLAY_CHANNELS if ch in eeg_stream_channels], "values": display.tolist(), "display_rate": 50, "last_data_at": now},
                            model={"name": self.model_name, "class_names": self.class_names, "probabilities": probs.tolist(), "prediction": self.class_names[pred], "confidence": conf, "control": self.current_control, "control_name": CONTROL_NAMES[self.current_control], "last_inference_at": now},
                            lsl={"last_sent": self.current_control, "consumer_online": self.outlet.have_consumers()},
                            status={"lsl_consumer": "connected" if self.outlet.have_consumers() else "unknown"},
                            health=self._health(),
                        )
                    except (RuntimeError, ValueError) as exc:
                        self.bus.event("warning", "推理窗口暂不可用", error=str(exc))
        except Exception as exc:
            self.bus.event("error", "实时服务启动失败", error=str(exc))
            self.bus.update(status={"device_tcp": "error", "inference": "error"}, health={"level": "red", "message": str(exc)})

    @staticmethod
    def _downsample(values: np.ndarray, target_rate: int) -> np.ndarray:
        if values.size == 0:
            return values
        stride = max(1, values.shape[-1] // max(1, int(target_rate * 2)))
        return values[..., ::stride]

    def _health(self) -> dict[str, str]:
        lsl_online = bool(self.outlet and self.outlet.have_consumers())
        if not self.outlet:
            return {"level": "red", "message": "LSL 输出未启动"}
        if time.time() - self._last_inference_at > max(3.0, self.args.step_sec * 4):
            return {"level": "yellow", "message": "等待有效 EEG 推理窗口"}
        if not lsl_online:
            return {"level": "yellow", "message": "LSL consumer 尚未发现"}
        return {"level": "green", "message": "脑电-模型-LSL 链路运行中"}

    def _dry_run(self) -> None:
        n_points = int(round(self.args.window_sec * self.args.device_sfreq))
        fake = np.random.randn(len(self.model_channels), n_points).astype(np.float32) * 20.0
        window = self.preprocessor.transform(fake)
        pred, conf, probs = predict_window(self.model, window, self.device, self._smoother)
        control = self.control_map[pred]
        now = time.time()
        self.bus.update(
            status={"device_tcp": "dry-run", "inference": "running", "lsl_outlet": "disabled"},
            model={"name": self.model_name, "class_names": self.class_names, "probabilities": probs.tolist(), "prediction": self.class_names[pred], "confidence": conf, "control": control, "control_name": CONTROL_NAMES[control], "device": str(self.device), "last_inference_at": now},
            eeg={"sample_rate": self.args.device_sfreq, "channels": self.model_channels[:4], "values": self._downsample(fake[:4], 50).tolist(), "display_rate": 50, "last_data_at": now},
            lsl={"stream_name": self.args.stream_name, "stream_type": self.args.stream_type, "consumer_online": False, "last_sent": control},
            game={"telemetry_level": "inferred"},
            health={"level": "yellow", "message": "离线 dry-run：未连接设备和游戏"},
        )
        self.bus.event("info", "dry-run 推理完成", prediction=self.class_names[pred], confidence=conf, control=control)

    def snapshot(self) -> dict[str, Any]:
        return self.bus.snapshot()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime BCI racing dashboard service")
    parser.add_argument("--model", choices=["bcic2a", "hgd"], default="bcic2a")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--device-sfreq", type=float, default=1000.0)
    parser.add_argument("--model-sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--channel-list", default=None)
    parser.add_argument("--channel-list-file", default=None)
    parser.add_argument("--calibration-npz", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--filter-low", type=float, default=4.0)
    parser.add_argument("--filter-high", type=float, default=38.0)
    parser.add_argument("--zscore-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-n", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-name", default="EEGback")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--lsl-rate", type=float, default=10.0)
    parser.add_argument("--default-control", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--eeg-source", choices=["neuracle", "local"], default="neuracle")
    parser.add_argument("--local-eeg-file", default=None)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.channel_list is None:
        if args.model == "hgd":
            args.channel_list = ",".join(HGD_CHANNELS + ["TRG"])
        else:
            args.channel_list = ",".join(BCIC2A_CHANNELS + ["TRG"])
    bus = TelemetryBus()
    runtime = InferenceRuntime(args, bus)
    runtime.start()
    from dashboard_api import create_app
    import uvicorn
    app = create_app(bus, runtime)
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("DASHBOARD_PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
