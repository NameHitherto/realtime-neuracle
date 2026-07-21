import argparse
import math
import socket
import struct
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt


class RingBuffer:
    def __init__(self, n_channels: int, n_points: int):
        self.n_channels = int(n_channels)
        self.n_points = int(n_points)
        self.buffer = np.zeros((self.n_channels, self.n_points), dtype=np.float32)
        self.current_ptr = 0
        self.n_updates = 0

    def append(self, data: np.ndarray):
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(f"Expected data shape ({self.n_channels}, T), got {data.shape}")
        n = data.shape[1]
        if n >= self.n_points:
            self.buffer[:] = data[:, -self.n_points :]
            self.current_ptr = 0
            self.n_updates += n
            return
        end = self.current_ptr + n
        if end <= self.n_points:
            self.buffer[:, self.current_ptr : end] = data
        else:
            first = self.n_points - self.current_ptr
            self.buffer[:, self.current_ptr :] = data[:, :first]
            self.buffer[:, : end % self.n_points] = data[:, first:]
        self.current_ptr = end % self.n_points
        self.n_updates += n

    def latest(self, n_points: int) -> np.ndarray:
        n_points = min(int(n_points), self.n_points)
        if self.n_updates < n_points:
            raise RuntimeError(f"Only {self.n_updates} samples received, need {n_points}")
        start = (self.current_ptr - n_points) % self.n_points
        if start < self.current_ptr:
            return self.buffer[:, start : self.current_ptr].copy()
        return np.concatenate((self.buffer[:, start:], self.buffer[:, : self.current_ptr]), axis=1)


class DataServer:
    def __init__(
        self,
        host: str,
        port: int,
        channel_names: list[str],
        sampling_rate: float,
        buffer_seconds: float,
        trigger_name: str = "TRG",
    ):
        self.host = host
        self.port = int(port)
        self.channel_names = list(channel_names)
        self.sampling_rate = float(sampling_rate)
        self.trigger_name = trigger_name
        self.has_trigger = self.channel_names[-1].upper() == trigger_name.upper()
        self.eeg_channel_names = self.channel_names[:-1] if self.has_trigger else self.channel_names
        self.n_chan_total = len(self.channel_names)
        self.n_chan_eeg = len(self.eeg_channel_names)
        self.buffer = RingBuffer(self.n_chan_eeg, int(math.ceil(buffer_seconds * sampling_rate)))
        if self.has_trigger:
            self.struct_format = "<" + "f" * (self.n_chan_total - 1) + "i"
        else:
            self.struct_format = "<" + "f" * self.n_chan_total
        self.bytes_per_sample = struct.calcsize(self.struct_format)
        self._byte_cache = b""
        self._socket = None

    def __enter__(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.connect((self.host, self.port))
        self._socket.settimeout(2.0)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def poll(self, max_samples: int | None = None):
        if self._socket is None:
            raise RuntimeError("DataServer is not connected")
        max_samples = max_samples or max(1, int(self.sampling_rate // 10))
        payload = self._byte_cache + self._socket.recv(self.bytes_per_sample * max_samples)
        usable = len(payload) - (len(payload) % self.bytes_per_sample)
        if usable <= 0:
            self._byte_cache = payload
            return
        self._byte_cache = payload[usable:]
        rows = list(struct.iter_unpack(self.struct_format, payload[:usable]))
        if not rows:
            return
        arr = np.asarray(rows, dtype=np.float32).T
        if self.has_trigger:
            arr = arr[:-1]
        self.buffer.append(arr)


class LSLControlOutlet:
    def __init__(
        self,
        stream_name: str = "EEGback",
        stream_type: str = "EEG",
        sample_rate: float = 10.0,
        source_id: str = "hyena_realtime_bci_control",
    ):
        try:
            from pylsl import StreamInfo, StreamOutlet
        except ImportError as exc:
            raise RuntimeError(
                "pylsl is required for racing-game control. Install it with: pip install pylsl"
            ) from exc

        info = StreamInfo(
            name=stream_name,
            type=stream_type,
            channel_count=1,
            nominal_srate=float(sample_rate),
            channel_format="float32",
            source_id=source_id,
        )
        self.outlet = StreamOutlet(info)
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.sample_rate = float(sample_rate)
        self.last_value: int | None = None
        self.last_push_at: float | None = None

    def push(self, value: int):
        self.outlet.push_sample([float(value)])
        self.last_value = int(value)
        self.last_push_at = time.time()

    def have_consumers(self) -> bool:
        """Return whether at least one LSL inlet is currently connected."""
        checker = getattr(self.outlet, "have_consumers", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:
            return False


class RealTimePreprocessor:
    def __init__(
        self,
        input_sfreq: float,
        model_sfreq: float,
        window_seconds: float,
        low_hz: float,
        high_hz: float,
        model_samples: int,
        zscore_window: bool,
        calibration_npz: str | None = None,
    ):
        self.input_sfreq = float(input_sfreq)
        self.model_sfreq = float(model_sfreq)
        self.window_seconds = float(window_seconds)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.model_samples = int(model_samples)
        self.zscore_window = bool(zscore_window)
        self.calibration_mean = None
        self.calibration_std = None
        if calibration_npz:
            stats = np.load(calibration_npz)
            self.calibration_mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1)
            self.calibration_std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1)
            self.calibration_std = np.maximum(self.calibration_std, 1e-6)

    def _maybe_to_microvolts(self, x: np.ndarray) -> np.ndarray:
        median_abs = float(np.nanmedian(np.abs(x)))
        if median_abs < 1e-3:
            return x * 1_000_000.0
        return x

    def _bandpass(self, x: np.ndarray, sfreq: float) -> np.ndarray:
        nyquist = sfreq / 2.0
        high = min(self.high_hz, nyquist - 1.0)
        low = max(0.1, min(self.low_hz, high * 0.5))
        sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
        return sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    def _resample(self, x: np.ndarray) -> np.ndarray:
        if abs(self.input_sfreq - self.model_sfreq) < 1e-6:
            return x
        input_i = int(round(self.input_sfreq))
        model_i = int(round(self.model_sfreq))
        divisor = math.gcd(input_i, model_i)
        return resample_poly(x, model_i // divisor, input_i // divisor, axis=-1).astype(np.float32)

    def _fix_length(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] > self.model_samples:
            return x[:, -self.model_samples :]
        if x.shape[-1] < self.model_samples:
            pad = self.model_samples - x.shape[-1]
            return np.pad(x, ((0, 0), (pad, 0)), mode="edge")
        return x

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.calibration_mean is not None:
            return (x - self.calibration_mean) / self.calibration_std
        if self.zscore_window:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            return (x - mean) / np.maximum(std, 1e-6)
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = self._maybe_to_microvolts(x)
        x = self._bandpass(x, self.input_sfreq)
        x = self._resample(x)
        x = self._fix_length(x)
        x = self._standardize(x)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def parse_channel_list(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_channel_file(path: str | None) -> list[str] | None:
    if path is None or str(path).strip() == "":
        return None
    content = Path(path).read_text(encoding="utf-8")
    channels: list[str] = []
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        channels.extend([item.strip() for item in line.split(",") if item.strip()])
    return channels


def select_model_channels(
    raw_window: np.ndarray,
    stream_channels: list[str],
    model_channels: list[str],
) -> np.ndarray:
    index_by_name = {name.upper(): idx for idx, name in enumerate(stream_channels)}
    missing = [name for name in model_channels if name.upper() not in index_by_name]
    if missing:
        raise ValueError(
            "Realtime stream is missing model channels: "
            + ", ".join(missing)
            + "\nCurrent stream channels: "
            + ", ".join(stream_channels)
        )
    indices = [index_by_name[name.upper()] for name in model_channels]
    return raw_window[indices]


def load_state_dict(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return checkpoint


@torch.no_grad()
def predict_window(
    model: torch.nn.Module,
    window: np.ndarray,
    device: torch.device,
    prob_smoother: deque[np.ndarray],
) -> tuple[int, float, np.ndarray]:
    tensor = torch.from_numpy(window[None]).float().to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    prob_smoother.append(probs)
    smooth_probs = np.mean(np.stack(list(prob_smoother), axis=0), axis=0)
    pred = int(np.argmax(smooth_probs))
    return pred, float(smooth_probs[pred]), smooth_probs


def add_common_args(parser: argparse.ArgumentParser, default_checkpoint: Path, default_channels: list[str]):
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--device-sfreq", type=float, default=1000.0)
    parser.add_argument("--model-sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--channel-list", default=",".join(default_channels))
    parser.add_argument("--channel-list-file", default=None)
    parser.add_argument("--calibration-npz", default=None)
    parser.add_argument("--smooth-n", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def add_game_lsl_args(parser: argparse.ArgumentParser):
    parser.add_argument("--game-lsl", action="store_true")
    parser.add_argument("--stream-name", default="EEGback")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--lsl-rate", type=float, default=10.0)
    parser.add_argument("--default-control", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser


def run_realtime_loop(
    model: torch.nn.Module,
    model_channels: list[str],
    class_names: list[str],
    args,
    preprocessor: RealTimePreprocessor,
    device: torch.device,
    control_mapper: dict[int, int] | None = None,
    control_names: dict[int, str] | None = None,
    game_outlet: LSLControlOutlet | None = None,
):
    stream_channels = parse_channel_file(getattr(args, "channel_list_file", None))
    if stream_channels is None:
        stream_channels = parse_channel_list(args.channel_list, model_channels + ["TRG"])
    if stream_channels[-1].upper() == "TRG":
        eeg_stream_channels = stream_channels[:-1]
    else:
        eeg_stream_channels = stream_channels
    n_points = int(round(args.window_sec * args.device_sfreq))
    smoother = deque(maxlen=max(1, int(args.smooth_n)))

    if args.dry_run:
        fake = np.random.randn(len(model_channels), n_points).astype(np.float32) * 20.0
        window = preprocessor.transform(fake)
        pred, conf, probs = predict_window(model, window, device, smoother)
        control = control_mapper.get(pred, -1) if control_mapper else None
        suffix = f" control={control}" if control is not None else ""
        print(f"[dry-run] pred={class_names[pred]} conf={conf:.3f}{suffix} probs={np.round(probs, 4).tolist()}")
        return

    print("Connecting Neuracle TCP data stream...")
    print(f"host={args.host} port={args.port} device_sfreq={args.device_sfreq}Hz")
    print(f"model_channels={len(model_channels)} window={args.window_sec:.2f}s step={args.step_sec:.2f}s")
    current_control = int(getattr(args, "default_control", 3))
    if game_outlet is not None:
        print(
            f"LSL control stream: name={game_outlet.stream_name} type={game_outlet.stream_type} "
            f"rate={game_outlet.sample_rate}Hz"
        )
        print("Game mapping: 0=left, 1=right, 2=forward, 3=stop")

    with DataServer(
        host=args.host,
        port=args.port,
        channel_names=stream_channels,
        sampling_rate=args.device_sfreq,
        buffer_seconds=max(args.window_sec + 2.0, args.window_sec * 2),
    ) as server:
        next_emit = time.time() + args.step_sec
        next_lsl_emit = time.time()
        lsl_interval = 1.0 / max(float(getattr(args, "lsl_rate", 10.0)), 1e-6)
        while True:
            server.poll()
            now = time.time()
            if game_outlet is not None and now >= next_lsl_emit:
                game_outlet.push(current_control)
                next_lsl_emit = now + lsl_interval
            if now < next_emit:
                continue
            next_emit = now + args.step_sec
            try:
                raw = server.buffer.latest(n_points)
                raw = select_model_channels(raw, eeg_stream_channels, model_channels)
                window = preprocessor.transform(raw)
                pred, conf, probs = predict_window(model, window, device, smoother)
            except RuntimeError:
                continue
            if control_mapper is not None:
                mapped_control = int(control_mapper.get(pred, current_control))
                if conf >= float(getattr(args, "min_confidence", 0.0)):
                    current_control = mapped_control
                else:
                    current_control = int(getattr(args, "default_control", 3))
            stamp = time.strftime("%H:%M:%S")
            prob_text = " ".join(
                f"{name}:{prob:.2f}" for name, prob in zip(class_names, probs)
            )
            if control_mapper is None:
                print(f"[{stamp}] pred={class_names[pred]} conf={conf:.3f} {prob_text}")
            else:
                action = control_names.get(current_control, str(current_control)) if control_names else str(current_control)
                print(
                    f"[{stamp}] pred={class_names[pred]} conf={conf:.3f} "
                    f"control={current_control}({action}) {prob_text}"
                )
