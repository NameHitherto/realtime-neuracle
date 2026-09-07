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

from experiment_logger import ExperimentLogger
from telemetry_core import TelemetryBus
from realtime_common import (
    DataBlock,
    DataServer,
    LSLControlOutlet,
    NEUSEN_W_64_STREAM_CHANNELS,
    predict_window,
    predict_window_details,
    select_model_channels,
    parse_channel_file,
    parse_channel_list,
    load_state_dict,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Importing the YHC entrypoint keeps its checkpoint/model definition as the
# single source of truth for the realtime application.
sys.path.insert(0, str(SCRIPT_DIR))
from realtime_yhc_4class import (  # noqa: E402
    YHC_CHANNELS,
    CLASS_NAMES as YHC_CLASSES,
    CLASS_ACTION_NAMES as YHC_CLASS_ACTION_NAMES,
    CONTROL_MAP as YHC_CONTROL_MAP,
    DEFAULT_CHECKPOINT as YHC_CHECKPOINT,
    build_model as build_yhc,
    build_preprocessor as build_yhc_preprocessor,
    validate_checkpoint_contract as validate_yhc_checkpoint,
)

CONTROL_NAMES = {0: "left", 1: "right", 2: "forward", 3: "stop"}
LOCAL_CHANNEL_DERIVATIONS = {
    "CPZ": ("CP1", "CP2"),
    "P1": ("PZ", "P3"),
    "P2": ("PZ", "P4"),
}


class StreamContractError(RuntimeError):
    """Raised when the TCP byte stream cannot match the configured EEG contract."""


class LocalEEGSimulator:
    """Simulates EEG data stream from a local BDF file using mne."""

    def __init__(self, file_path: str, channel_names: list[str], sampling_rate: float, buffer_seconds: float):
        self.file_path = file_path
        self.channel_names = list(channel_names)
        self.sampling_rate = float(sampling_rate)
        self.eeg_channel_names = [ch for ch in self.channel_names if ch.upper() != "TRG"]
        self.has_trigger = bool(self.channel_names and self.channel_names[-1].upper() == "TRG")
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
        eeg = np.concatenate(chunks, axis=1)
        self.buffer.append(eeg)
        triggers = np.zeros(eeg.shape[1], dtype=np.int32) if self.has_trigger else None
        return DataBlock(eeg=eeg, triggers=triggers, received_at=time.time())


class InferenceRuntime:
    def __init__(self, args: argparse.Namespace, bus: TelemetryBus):
        self.args = args
        self.bus = bus
        self.model_name = args.model
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        self.model_channels, self.class_names, self.control_map, default_checkpoint, self.build_model = self._model_spec()
        self.checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint
        self.model = self.build_model(self.device)
        checkpoint = load_state_dict(self.model, self.checkpoint, self.device)
        validate_yhc_checkpoint(checkpoint)
        self.preprocessor = build_yhc_preprocessor(args, checkpoint)
        self.outlet: LSLControlOutlet | None = None
        self.current_control = int(args.default_control)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_eeg_at = 0.0
        self._last_inference_at = 0.0
        self._last_display_at = 0.0
        self._last_calibration_update_at = 0.0
        self._stream_faulted = False
        self._calibrated = float(args.live_baseline_sec) <= 0.0
        self._baseline_start_updates = 0
        self._smoother: deque[np.ndarray] = deque(maxlen=max(1, int(args.smooth_n)))
        self.stream_channels = self._resolve_stream_channels()
        self.eeg_stream_channels = (
            self.stream_channels[:-1]
            if self.stream_channels and self.stream_channels[-1].upper() == "TRG"
            else self.stream_channels
        )
        self.missing_model_channels: list[str] = []
        self.model_input_available = False
        self._model_input_warning_emitted = False
        self._validate_stream_contract()
        stream_index = {name.upper(): name for name in self.eeg_stream_channels}
        if self.model_input_available:
            self.display_channels = [stream_index[name.upper()] for name in self.model_channels]
        else:
            self.display_channels = list(self.eeg_stream_channels)
        self.logger: ExperimentLogger | None = None
        if self.args.save_logs and not self.args.dry_run:
            settings = {**vars(self.args), "checkpoint_resolved": str(self.checkpoint.resolve())}
            self.logger = ExperimentLogger(
                root_dir=self.args.log_dir,
                experiment_id=self.args.experiment_id,
                model_name=self.model_name,
                stream_channels=self.stream_channels,
                model_channels=self.model_channels,
                class_names=self.class_names,
                control_map=self.control_map,
                sampling_rate=self.args.device_sfreq,
                settings=settings,
            )
            self.bus.update(
                recording={
                    "enabled": True,
                    "state": "recording",
                    "session_dir": str(self.logger.session_dir),
                    "samples_saved": 0,
                }
            )

    def _resolve_stream_channels(self) -> list[str]:
        stream_channels = parse_channel_file(self.args.channel_list_file)
        if stream_channels is None:
            stream_channels = parse_channel_list(
                self.args.channel_list,
                NEUSEN_W_64_STREAM_CHANNELS,
            )
        return stream_channels

    def _validate_stream_contract(self) -> None:
        if not self.stream_channels:
            raise ValueError("EEG stream channel list is empty")
        normalized = [name.upper() for name in self.stream_channels]
        if len(normalized) != len(set(normalized)):
            raise ValueError("EEG stream channel list contains duplicate names")
        if self.args.eeg_source == "neuracle":
            if len(self.stream_channels) < 2:
                raise ValueError("Neuracle TCP must contain at least one EEG field plus TRG")
            if normalized[-1] != "TRG":
                raise ValueError("Neuracle TCP last field must be TRG")
        self.missing_model_channels = [
            name for name in self.model_channels if name.upper() not in normalized
        ]
        self.model_input_available = not self.missing_model_channels

    def _event(self, level: str, message: str, **details: Any) -> None:
        self.bus.event(level, message, **details)
        if self.logger:
            self.logger.log_event(level, message, **details)

    def _close_logger(self, state: str, error: str | None = None) -> None:
        if not self.logger:
            return
        self.logger.close(state=state, error=error)
        self.bus.update(
            recording={
                "state": self.logger.state,
                "samples_saved": self.logger.total_samples,
                "predictions_saved": self.logger.total_predictions,
            }
        )

    def _model_spec(self):
        if self.model_name != "yhc":
            raise ValueError("This realtime application is locked to the YHC model")
        return YHC_CHANNELS, YHC_CLASSES, YHC_CONTROL_MAP, YHC_CHECKPOINT, build_yhc

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="inference-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if not self._thread or not self._thread.is_alive():
            self._close_logger("completed")
        self.current_control = int(self.args.default_control)
        if self.outlet is not None:
            try:
                self.outlet.push(self.current_control)
            except Exception:
                pass
        self.bus.update(
            status={
                "inference": "stopped",
                "device_tcp": "stopped",
                "lsl_outlet": "stopped",
                "lsl_consumer": "unknown",
            },
            model={
                "probabilities": [],
                "prediction": None,
                "confidence": 0.0,
                "control": self.current_control,
                "control_name": CONTROL_NAMES[self.current_control],
                "action_name": "stop",
                "last_inference_at": None,
            },
            lsl={
                "last_sent": self.current_control,
                "consumer_online": False,
            },
            health={"level": "yellow", "message": "模型服务已停止"},
        )

    def manual_control(self, control: int, duration_ms: int = 1000) -> None:
        if not 0 <= control <= 3:
            raise ValueError("control must be one of 0, 1, 2, 3")
        self.current_control = control
        now = time.time()
        if self.outlet:
            self.outlet.push(control)
            if self.logger:
                self.logger.log_control(now, control, CONTROL_NAMES[control], "manual")
        self._event("info", "测试控制指令已发送", control=control, name=CONTROL_NAMES[control], duration_ms=duration_ms)
        self.bus.update(
            model={
                "control": control,
                "control_name": CONTROL_NAMES[control],
                "action_name": CONTROL_NAMES[control],
            },
            lsl={"last_sent": control},
        )

    def _publish(self) -> None:
        if self.outlet:
            now = time.time()
            self.outlet.push(self.current_control)
            if self.logger:
                self.logger.log_control(
                    now,
                    self.current_control,
                    CONTROL_NAMES[self.current_control],
                    "lsl",
                )
            self.bus.update(
                lsl={
                    "last_sent": self.current_control,
                    "last_sent_at": now,
                    "consumer_online": self.outlet.have_consumers(),
                }
            )

    def _force_stop(self, message: str) -> None:
        first_fault = not self._stream_faulted
        self._stream_faulted = True
        self.current_control = int(self.args.default_control)
        self._publish()
        if first_fault:
            self._event("warning", message, control=self.current_control)
        self.bus.update(
            status={"device_tcp": "stale", "inference": "waiting"},
            model={
                "control": self.current_control,
                "control_name": CONTROL_NAMES[self.current_control],
                "action_name": "stop",
            },
            health={"level": "red", "message": message},
        )

    def _create_data_source(self, buffer_seconds: float):
        if self.args.eeg_source == "local":
            self._event("info", "使用本地 EEG 数据模拟", file=self.args.local_eeg_file)
            return LocalEEGSimulator(
                file_path=self.args.local_eeg_file,
                channel_names=self.stream_channels,
                sampling_rate=self.args.device_sfreq,
                buffer_seconds=buffer_seconds,
            )
        return DataServer(
            host=self.args.host,
            port=self.args.port,
            channel_names=self.stream_channels,
            sampling_rate=self.args.device_sfreq,
            buffer_seconds=buffer_seconds,
            socket_timeout_seconds=self.args.socket_timeout_sec,
        )

    def _latest_trigger(self, server: Any, block: DataBlock | None) -> int | None:
        latest_trigger = getattr(server, "last_trigger", None)
        if block is not None and block.triggers is not None and block.triggers.size:
            latest_trigger = int(block.triggers[-1])
        return latest_trigger

    def _update_eeg_display(
        self,
        server: Any,
        block: DataBlock | None,
        now: float,
    ) -> None:
        if now - self._last_display_at < float(self.args.display_update_sec):
            return
        available = min(
            int(server.buffer.n_updates),
            int(round(self.args.device_sfreq * 2.0)),
        )
        if available < max(32, int(self.args.device_sfreq * 0.1)):
            return
        raw_all = server.buffer.latest(available)
        if self.model_input_available:
            raw_display = select_model_channels(
                raw_all,
                self.eeg_stream_channels,
                self.model_channels,
            )
        else:
            raw_display = raw_all
        display = self.preprocessor.prepare_for_display(raw_display)
        display = self._downsample(display, 50)
        self._last_display_at = now
        self._last_eeg_at = now
        self.bus.update(
            eeg={
                "channels": self.display_channels,
                "values": display.tolist(),
                "display_rate": 50,
                "unit": "uV",
                "filtered": True,
                "source_unit": self.preprocessor.last_input_unit,
                "source_scale_to_uv": self.preprocessor.last_unit_scale,
                "last_data_at": now,
                "latest_trigger": self._latest_trigger(server, block),
            }
        )

    def _validate_receive_rate(
        self,
        received_samples: int,
        elapsed_seconds: float,
    ) -> bool:
        check_seconds = float(self.args.rate_check_sec)
        if elapsed_seconds < check_seconds:
            remaining = check_seconds - elapsed_seconds
            self.bus.update(
                status={"device_tcp": "validating", "inference": "waiting"},
                health={
                    "level": "yellow",
                    "message": f"正在验证 EEG 帧宽和数据率，还需 {remaining:.1f} 秒",
                },
            )
            return False

        actual_rate = received_samples / max(elapsed_seconds, 1e-6)
        expected_rate = float(self.args.device_sfreq)
        relative_error = abs(actual_rate - expected_rate) / expected_rate
        if relative_error > float(self.args.rate_tolerance):
            raise StreamContractError(
                "EEG TCP 数据率与当前帧宽不匹配："
                f"按 {len(self.stream_channels)} 字段解析得到 {actual_rate:.1f} samples/s，"
                f"期望 {expected_rate:.1f} samples/s。"
            )
        self._baseline_start_updates = received_samples
        self._event(
            "info",
            "EEG 帧宽和数据率验证通过",
            measured_sample_rate=actual_rate,
            expected_sample_rate=expected_rate,
        )
        self.bus.update(
            status={"device_tcp": "connected"},
            eeg={"received_sample_rate": actual_rate},
        )
        return True

    def _maybe_finish_live_baseline(self, server: Any, now: float) -> bool:
        if self._calibrated:
            return True
        baseline_points = int(round(self.args.live_baseline_sec * self.args.device_sfreq))
        received = max(0, int(server.buffer.n_updates) - self._baseline_start_updates)
        if received < baseline_points:
            remaining = max(0.0, (baseline_points - received) / self.args.device_sfreq)
            if now - self._last_calibration_update_at >= float(self.args.display_update_sec):
                self._last_calibration_update_at = now
                self.bus.update(
                    status={"inference": "calibrating"},
                    calibration={
                        "state": "collecting",
                        "required_seconds": self.args.live_baseline_sec,
                        "received_seconds": received / self.args.device_sfreq,
                        "remaining_seconds": remaining,
                    },
                    health={"level": "yellow", "message": f"静息基线采集中，还需 {remaining:.1f} 秒"},
                )
            return False

        raw_all = server.buffer.latest(baseline_points)
        raw_model = select_model_channels(
            raw_all,
            self.eeg_stream_channels,
            self.model_channels,
        )
        mean, std = self.preprocessor.fit_calibration(raw_model)
        self._calibrated = True
        self._smoother.clear()
        if self.logger:
            np.savez(
                self.logger.session_dir / "live_calibration.npz",
                mean=mean,
                std=std,
                channels=np.asarray(self.model_channels),
                sampling_rate=np.asarray(self.args.model_sfreq),
            )
        self._event(
            "info",
            f"{self.args.live_baseline_sec:g} 秒静息基线采集完成，实时推理开始",
            mean_uv=mean.tolist(),
            std_uv=std.tolist(),
        )
        self.bus.update(
            status={"inference": "running"},
            calibration={
                "state": "completed",
                "required_seconds": self.args.live_baseline_sec,
                "completed_at": now,
            },
            health={"level": "yellow", "message": "基线完成，等待首个推理窗口"},
        )
        return True

    def _run_connected_source(
        self,
        server: Any,
        n_points: int,
        publish_interval: float,
    ) -> None:
        next_publish = 0.0
        next_infer = time.time() + self.args.step_sec
        connected_at = time.time()
        rate_started_at = time.monotonic()
        rate_received_samples = 0
        rate_validated = False
        self._last_eeg_at = connected_at
        self._stream_faulted = False

        if self.args.eeg_source == "neuracle":
            self._event(
                "info",
                f"已连接 EEG TCP 数据流（{len(self.eeg_stream_channels)} EEG + TRG）",
                host=self.args.host,
                port=self.args.port,
            )
        if isinstance(server, LocalEEGSimulator) and server.derived_channels:
            self._event(
                "warning",
                "本地 BDF 缺少部分模型导联，已使用邻近导联插值",
                derived_channels=server.derived_channels,
            )
        self.bus.update(
            status={
                "device_tcp": "validating",
                "inference": "waiting",
            }
        )

        while not self._stop.is_set():
            try:
                block = server.poll()
            except TimeoutError:
                if time.time() - self._last_eeg_at >= float(self.args.data_timeout_sec):
                    self._force_stop("EEG 数据超时，已强制停车")
                continue

            now = time.time()
            if block is not None:
                self._last_eeg_at = now
                rate_received_samples += int(block.eeg.shape[1])
                if self._stream_faulted:
                    self._stream_faulted = False
                    self._event("info", "EEG 数据已恢复")
                    self.bus.update(status={"device_tcp": "connected"})
                if self.logger:
                    self.logger.log_data(block)

            if now >= next_publish:
                self._publish()
                next_publish = now + publish_interval

            if not rate_validated:
                rate_validated = self._validate_receive_rate(
                    rate_received_samples,
                    time.monotonic() - rate_started_at,
                )
                if not rate_validated:
                    continue
                self.bus.update(
                    status={
                        "device_tcp": "connected",
                        "inference": (
                            "unavailable"
                            if not self.model_input_available
                            else "running" if self._calibrated else "calibrating"
                        ),
                    }
                )

            try:
                self._update_eeg_display(server, block, now)
            except (RuntimeError, ValueError) as exc:
                self._event("warning", "EEG 显示窗口暂不可用", error=str(exc))

            if not self.model_input_available:
                if not self._model_input_warning_emitted:
                    self._model_input_warning_emitted = True
                    self._event(
                        "warning",
                        "EEG 波形已接收，但当前通道不满足模型输入",
                        stream_channels=self.eeg_stream_channels,
                        missing_model_channels=self.missing_model_channels,
                    )
                self.bus.update(
                    status={"device_tcp": "connected", "inference": "unavailable"},
                    health={
                        "level": "yellow",
                        "message": (
                            f"已直接接收 {len(self.eeg_stream_channels)} 路 EEG 波形；"
                            f"模型仍缺少 {len(self.missing_model_channels)} 个指定导联"
                        ),
                    },
                )
                continue

            if not self._maybe_finish_live_baseline(server, now):
                continue
            if now < next_infer:
                continue
            next_infer = now + self.args.step_sec
            if server.buffer.n_updates < n_points:
                continue

            try:
                raw_all = server.buffer.latest(n_points)
                raw_model = select_model_channels(
                    raw_all,
                    self.eeg_stream_channels,
                    self.model_channels,
                )
                window = self.preprocessor.transform(raw_model)
                pred, conf, raw_probs, probs = predict_window_details(
                    self.model,
                    window,
                    self.device,
                    self._smoother,
                )
                accepted = conf >= self.args.min_confidence
                self.current_control = (
                    self.control_map.get(pred, self.current_control)
                    if accepted
                    else self.args.default_control
                )
                action_name = YHC_CLASS_ACTION_NAMES[pred] if accepted else "stop"
                self.last_prediction = pred
                self._last_inference_at = now
                latest_trigger = self._latest_trigger(server, block)
                if self.logger:
                    self.logger.log_prediction(
                        timestamp=now,
                        sample_index=self.logger.total_samples,
                        predicted_index=pred,
                        confidence=conf,
                        game_control=self.current_control,
                        game_control_name=CONTROL_NAMES[self.current_control],
                        latest_trigger=latest_trigger,
                        raw_probabilities=raw_probs,
                        smooth_probabilities=probs,
                    )
                self.bus.update(
                    model={
                        "name": self.model_name,
                        "class_names": self.class_names,
                        "probabilities": probs.tolist(),
                        "prediction": self.class_names[pred],
                        "confidence": conf,
                        "control": self.current_control,
                        "control_name": CONTROL_NAMES[self.current_control],
                        "action_name": action_name,
                        "last_inference_at": now,
                    },
                    lsl={
                        "last_sent": self.current_control,
                        "consumer_online": self.outlet.have_consumers(),
                    },
                    recording={
                        "samples_saved": self.logger.total_samples,
                        "predictions_saved": self.logger.total_predictions,
                    }
                    if self.logger
                    else {"enabled": False},
                    status={
                        "inference": "running",
                        "lsl_consumer": "connected"
                        if self.outlet.have_consumers()
                        else "unknown",
                    },
                    health=self._health(),
                )
            except (RuntimeError, ValueError) as exc:
                self._event("warning", "推理窗口暂不可用", error=str(exc))

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
            self._event("error", "LSL 输出初始化失败", error=str(exc))
            self.bus.update(status={"lsl_outlet": "error"}, health={"level": "red", "message": "pylsl/LSL 输出不可用"})
            self._close_logger("failed", str(exc))
            return

        n_points = int(round(self.args.window_sec * self.args.device_sfreq))
        publish_interval = 1.0 / max(float(self.args.lsl_rate), 1e-6)
        buffer_seconds = max(
            self.args.window_sec + 2.0,
            self.args.window_sec * 2,
            self.args.live_baseline_sec + 2.0,
        )
        self.bus.update(
            status={"inference": "starting", "device_tcp": "connecting"},
            model={"name": self.model_name, "class_names": self.class_names, "device": str(self.device)},
            eeg={"sample_rate": self.args.device_sfreq, "channels": self.display_channels, "display_rate": 50, "unit": "uV", "filtered": True},
            game={"telemetry_level": "inferred"},
        )
        while not self._stop.is_set():
            try:
                server = self._create_data_source(buffer_seconds)
                with server:
                    self._run_connected_source(server, n_points, publish_interval)
            except StreamContractError as exc:
                self._force_stop("EEG 发送帧配置错误，已强制停车")
                self._event("error", "EEG 数据契约验证失败", error=str(exc))
                self.bus.update(
                    status={"device_tcp": "error", "inference": "error"},
                    health={"level": "red", "message": str(exc)},
                )
                self._close_logger("failed", str(exc))
                return
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._force_stop("EEG 连接中断，已强制停车")
                self._event("error", "EEG 连接中断", error=str(exc))
                if self.args.eeg_source == "local" or not self.args.auto_reconnect:
                    self.bus.update(
                        status={"device_tcp": "error", "inference": "error"},
                        health={"level": "red", "message": str(exc)},
                    )
                    self._close_logger("failed", str(exc))
                    return
                self.bus.update(
                    status={"device_tcp": "reconnecting", "inference": "waiting"},
                    health={
                        "level": "red",
                        "message": f"EEG 已断开，{self.args.reconnect_sec:.1f} 秒后重连",
                    },
                )
                self._stop.wait(float(self.args.reconnect_sec))
        self._close_logger("completed")

    @staticmethod
    def _downsample(values: np.ndarray, target_rate: int) -> np.ndarray:
        if values.size == 0:
            return values
        stride = max(1, values.shape[-1] // max(1, int(target_rate * 2)))
        return values[..., ::stride]

    def _health(self) -> dict[str, str]:
        lsl_online = bool(self.outlet and self.outlet.have_consumers())
        status = self.bus.snapshot().get("status", {})
        if status.get("device_tcp") in {"stale", "reconnecting", "error"}:
            return {"level": "red", "message": "EEG 数据不可用，当前已停车"}
        if status.get("inference") == "calibrating":
            return {"level": "yellow", "message": "正在采集静息基线"}
        if not self.outlet:
            return {"level": "red", "message": "LSL 输出未启动"}
        if time.time() - self._last_inference_at > max(3.0, self.args.step_sec * 4):
            return {"level": "yellow", "message": "等待有效 EEG 推理窗口"}
        if not lsl_online:
            return {"level": "yellow", "message": "LSL Receiver 尚未发现"}
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
            model={"name": self.model_name, "class_names": self.class_names, "probabilities": probs.tolist(), "prediction": self.class_names[pred], "confidence": conf, "control": control, "control_name": CONTROL_NAMES[control], "action_name": YHC_CLASS_ACTION_NAMES[pred], "device": str(self.device), "last_inference_at": now},
            eeg={"sample_rate": self.args.device_sfreq, "channels": self.display_channels, "values": self._downsample(fake, 50).tolist(), "display_rate": 50, "last_data_at": now, "latest_trigger": 0},
            lsl={"stream_name": self.args.stream_name, "stream_type": self.args.stream_type, "consumer_online": False, "last_sent": control},
            game={"telemetry_level": "inferred"},
            health={"level": "yellow", "message": "离线 dry-run：未连接设备和游戏"},
        )
        self._event("info", "dry-run 推理完成", prediction=self.class_names[pred], confidence=conf, control=control)

    def snapshot(self) -> dict[str, Any]:
        return self.bus.snapshot()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime BCI racing dashboard service")
    parser.add_argument("--model", choices=["yhc"], default="yhc")
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
    parser.add_argument("--zscore-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smooth-n", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-name", default="EEGback")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--lsl-rate", type=float, default=10.0)
    parser.add_argument("--default-control", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument(
        "--live-baseline-sec",
        type=float,
        default=30.0,
        help="Collect a resting baseline before inference; use 0 to reuse checkpoint statistics.",
    )
    parser.add_argument("--display-update-sec", type=float, default=0.2)
    parser.add_argument("--socket-timeout-sec", type=float, default=1.0)
    parser.add_argument("--data-timeout-sec", type=float, default=2.0)
    parser.add_argument("--rate-check-sec", type=float, default=5.0)
    parser.add_argument("--rate-tolerance", type=float, default=0.3)
    parser.add_argument("--reconnect-sec", type=float, default=2.0)
    parser.add_argument(
        "--auto-reconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eeg-source", choices=["neuracle", "local"], default="neuracle")
    parser.add_argument("--local-eeg-file", default=None)
    parser.add_argument(
        "--save-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw 64-channel EEG, TRG, predictions, LSL output and events.",
    )
    parser.add_argument("--log-dir", default=str(SCRIPT_DIR / "experiment_logs"))
    parser.add_argument("--experiment-id", default="yhc_realtime")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.live_baseline_sec < 0:
        parser.error("--live-baseline-sec must be >= 0")
    if args.display_update_sec <= 0 or args.socket_timeout_sec <= 0:
        parser.error("display and socket timeout values must be > 0")
    if args.data_timeout_sec < args.socket_timeout_sec:
        parser.error("--data-timeout-sec must be >= --socket-timeout-sec")
    if args.rate_check_sec <= 0 or not 0.0 < args.rate_tolerance < 1.0:
        parser.error("rate-check seconds must be > 0 and rate tolerance must be between 0 and 1")
    if args.reconnect_sec <= 0:
        parser.error("--reconnect-sec must be > 0")
    if not 0 <= args.default_control <= 3:
        parser.error("--default-control must be one of 0, 1, 2, 3")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")
    if args.channel_list is None:
        args.channel_list = ",".join(NEUSEN_W_64_STREAM_CHANNELS)
    bus = TelemetryBus()
    runtime = InferenceRuntime(args, bus)
    runtime.start()
    from dashboard_api import create_app
    import uvicorn
    server: uvicorn.Server | None = None

    def request_server_shutdown() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(bus, runtime, request_server_shutdown)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=int(os.getenv("DASHBOARD_PORT", "8000")),
        log_level="info",
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
