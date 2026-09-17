import csv
import json
import struct
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

from experiment_logger import ExperimentLogger
from realtime_common import (
    DataBlock,
    DataServer,
    NEUSEN_W_64_EEG_CHANNELS,
    NEUSEN_W_64_STREAM_CHANNELS,
    RealTimePreprocessor,
    predict_window_details,
    select_model_channels,
)
from realtime_yhc_4class import CLASS_ACTION_NAMES, CONTROL_MAP, YHC_CHANNELS


class _OneShotSocket:
    def __init__(self, payload: bytes):
        self.payload = payload

    def recv(self, _size: int) -> bytes:
        payload, self.payload = self.payload, b""
        return payload


class _FixedModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=x.device).repeat(x.shape[0], 1)


class RealtimeRecordingTests(unittest.TestCase):
    def test_yhc_classes_cover_all_four_official_states(self):
        self.assertEqual(CONTROL_MAP, {0: 3, 1: 2, 2: 0, 3: 1})
        self.assertEqual(CLASS_ACTION_NAMES, ["stop", "forward", "left", "right"])
        self.assertEqual(set(CONTROL_MAP.values()), {0, 1, 2, 3})

    def test_neusen_stream_is_64_eeg_plus_trigger(self):
        self.assertEqual(len(NEUSEN_W_64_EEG_CHANNELS), 64)
        self.assertEqual(len(NEUSEN_W_64_STREAM_CHANNELS), 65)
        self.assertEqual(NEUSEN_W_64_STREAM_CHANNELS[-1], "TRG")
        self.assertEqual(len({name.upper() for name in NEUSEN_W_64_STREAM_CHANNELS}), 65)
        self.assertEqual(
            NEUSEN_W_64_EEG_CHANNELS,
            [
                "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
                "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
                "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
                "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
                "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
                "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
                "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
                "Oz", "O1", "O2", "ECG", "HEOR", "HEOL", "VEOU", "VEOL",
            ],
        )

    def test_data_server_decodes_full_frames_and_preserves_triggers(self):
        frame_format = "<" + "f" * 64 + "i"
        first = [float(index) for index in range(64)]
        second = [float(index + 1000) for index in range(64)]
        payload = struct.pack(frame_format, *first, 11) + struct.pack(frame_format, *second, 22)

        server = DataServer(
            host="127.0.0.1",
            port=8712,
            channel_names=NEUSEN_W_64_STREAM_CHANNELS,
            sampling_rate=1000,
            buffer_seconds=4,
        )
        server._socket = _OneShotSocket(payload)
        block = server.poll(max_samples=2)

        self.assertIsNotNone(block)
        np.testing.assert_array_equal(block.triggers, np.array([11, 22], dtype=np.int32))
        self.assertEqual(block.eeg.shape, (64, 2))
        self.assertEqual(server.last_trigger, 22)
        selected = select_model_channels(
            block.eeg,
            NEUSEN_W_64_EEG_CHANNELS,
            YHC_CHANNELS,
        )
        self.assertEqual(selected.shape, (12, 2))
        fc3_index = NEUSEN_W_64_EEG_CHANNELS.index("FC3")
        self.assertEqual(float(selected[0, 0]), float(fc3_index))
        self.assertEqual(float(selected[0, 1]), float(fc3_index + 1000))

    def test_prediction_returns_raw_and_smoothed_probabilities(self):
        smoother = deque(maxlen=3)
        pred, confidence, raw, smooth = predict_window_details(
            _FixedModel(),
            np.zeros((12, 1000), dtype=np.float32),
            torch.device("cpu"),
            smoother,
        )
        self.assertEqual(pred, 3)
        self.assertAlmostEqual(confidence, float(smooth[3]))
        np.testing.assert_allclose(raw, smooth)
        self.assertAlmostEqual(float(raw.sum()), 1.0, places=6)

    def test_live_baseline_fits_twelve_channel_statistics(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0.0, 10.0, size=(12, 30_000)).astype(np.float32)
        preprocessor = RealTimePreprocessor(
            input_sfreq=1000,
            model_sfreq=250,
            window_seconds=4,
            low_hz=4,
            high_hz=38,
            model_samples=1000,
            zscore_window=False,
            common_average_reference=True,
        )

        mean, std = preprocessor.fit_calibration(baseline)
        window = preprocessor.transform(baseline[:, -4000:])

        self.assertEqual(mean.shape, (12,))
        self.assertEqual(std.shape, (12,))
        self.assertTrue(np.all(std > 0))
        self.assertEqual(window.shape, (12, 1000))
        self.assertTrue(np.isfinite(window).all())

    def test_device_units_are_converted_to_microvolts(self):
        preprocessor = RealTimePreprocessor(
            input_sfreq=1000,
            model_sfreq=250,
            window_seconds=4,
            low_hz=4,
            high_hz=38,
            model_samples=1000,
            zscore_window=False,
        )

        np.testing.assert_allclose(
            preprocessor._maybe_to_microvolts(np.array([[20e-6]], dtype=np.float32)),
            np.array([[20.0]], dtype=np.float32),
        )
        self.assertEqual(preprocessor.last_input_unit, "V")
        np.testing.assert_allclose(
            preprocessor._maybe_to_microvolts(np.array([[20.0]], dtype=np.float32)),
            np.array([[20.0]], dtype=np.float32),
        )
        self.assertEqual(preprocessor.last_input_unit, "uV")
        np.testing.assert_allclose(
            preprocessor._maybe_to_microvolts(np.array([[20_000.0]], dtype=np.float32)),
            np.array([[20.0]], dtype=np.float32),
        )
        self.assertEqual(preprocessor.last_input_unit, "nV")

    def test_experiment_logger_writes_reconstructable_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ExperimentLogger(
                root_dir=temp_dir,
                experiment_id="yhc_test",
                model_name="yhc",
                stream_channels=NEUSEN_W_64_STREAM_CHANNELS,
                model_channels=YHC_CHANNELS,
                class_names=["rest", "feet", "left_hand", "right_hand"],
                control_map=CONTROL_MAP,
                sampling_rate=1000,
                settings={"step_sec": 0.5},
            )
            eeg = np.arange(64 * 3, dtype=np.float32).reshape(64, 3)
            logger.log_data(
                DataBlock(
                    eeg=eeg,
                    triggers=np.array([0, 7, 7], dtype=np.int32),
                    received_at=123.5,
                )
            )
            logger.log_prediction(
                timestamp=1_700_000_124.0,
                sample_index=3,
                predicted_index=2,
                confidence=0.77,
                game_control=0,
                game_control_name="left",
                latest_trigger=7,
                raw_probabilities=np.array([0.1, 0.1, 0.7, 0.1]),
                smooth_probabilities=np.array([0.08, 0.07, 0.77, 0.08]),
            )
            logger.log_control(1_700_000_124.0, 0, "left", "lsl")
            logger.log_event("info", "test event", trigger=7)
            logger.close()

            session_dir = logger.session_dir
            metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["state"], "completed")
            self.assertEqual(metadata["total_samples"], 3)
            self.assertEqual(metadata["total_predictions"], 1)
            self.assertEqual(metadata["total_controls"], 1)
            self.assertEqual((session_dir / "eeg_raw.f32").stat().st_size, 3 * 64 * 4)
            self.assertEqual((session_dir / "triggers.i32").stat().st_size, 3 * 4)

            raw = np.fromfile(session_dir / "eeg_raw.f32", dtype="<f4").reshape(3, 64)
            np.testing.assert_array_equal(raw, eeg.T)
            triggers = np.fromfile(session_dir / "triggers.i32", dtype="<i4")
            np.testing.assert_array_equal(triggers, np.array([0, 7, 7], dtype=np.int32))
            with (session_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(predictions[0]["predicted_label"], "left_hand")
            self.assertEqual(predictions[0]["game_control_name"], "left")


if __name__ == "__main__":
    unittest.main()
