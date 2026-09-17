from pathlib import Path
import argparse
import sys

import numpy as np
import torch

from realtime_common import (
    LSLControlOutlet,
    NEUSEN_W_64_STREAM_CHANNELS,
    RealTimePreprocessor,
    add_common_args,
    add_game_lsl_args,
    load_state_dict,
    run_realtime_loop,
)


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "model_deps"))

from model_hyena_input_motor_rhythm_enhanced_classifier import (  # noqa: E402
    HyenaInputMotorRhythmEnhancedClassifierNet,
)


YHC_CHANNELS = [
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP4",
]

# Keep the checkpoint's output indices unchanged.
CLASS_NAMES = ["rest", "feet", "left_hand", "right_hand"]
CLASS_NAMES_ZH = ["静息", "双脚运动想象", "左手运动想象", "右手运动想象"]

# The compiled game uses 0=left, 1=right, 2=accelerator and 3=brake/stop.
# Rest supplies the fourth state: stop / charge in the acceleration region.
# A stop command is not a guarantee of zero speed in all official track regions.
CONTROL_MAP = {
    0: 3,  # rest -> stop / charge
    1: 2,  # feet -> forward
    2: 0,  # left_hand -> left
    3: 1,  # right_hand -> right
}
CONTROL_NAMES = {0: "left", 1: "right", 2: "forward", 3: "stop"}
CLASS_ACTION_NAMES = ["stop", "forward", "left", "right"]

DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "yhc_hyena_final_realtime.pth"

MODEL_KWARGS = {
    "chans": 12,
    "samples": 1000,
    "num_classes": 4,
    "F1": 9,
    "F2": 48,
    "time_kernel1": 75,
    "pool_kernels": (50, 100, 200),
    "hyena_layers": 1,
    "hyena_dropout": 0.1,
    "hyena_long_conv": "implicit",
    "hyena_long_kernel": 63,
    "hyena_min_scale": 0.005,
    "hyena_max_extra_scale": 0.05,
    "rhythm_modes": 40,
    "rhythm_dropout": 0.05,
    "rhythm_min_scale": 0.002,
    "rhythm_max_extra_scale": 0.03,
    "ssa_gate_scale": 0.5,
    "output_bins": (16, 10, 6),
    "locality": 1.25,
    "max_logit_offset": 2.0,
    "classifier_dropout": 0.1,
    "classifier_maxnorm": 1.0,
    "sampling_rate": 250.0,
    "rhythm_max_scale": 0.25,
}


def build_model(device: torch.device) -> HyenaInputMotorRhythmEnhancedClassifierNet:
    return HyenaInputMotorRhythmEnhancedClassifierNet(
        **MODEL_KWARGS,
        device=device,
    )


def validate_checkpoint_contract(checkpoint: dict) -> None:
    if checkpoint.get("model_name") != "hyena_input_motor_rhythm_enhanced_classifier":
        raise ValueError(f"Unexpected YHC model_name: {checkpoint.get('model_name')!r}")
    if list(checkpoint.get("channels", [])) != YHC_CHANNELS:
        raise ValueError("YHC checkpoint channel order does not match the realtime contract")
    if list(checkpoint.get("class_names", [])) != CLASS_NAMES:
        raise ValueError("YHC checkpoint class order does not match the control mapping contract")
    if set(CONTROL_MAP) != set(range(len(CLASS_NAMES))):
        raise ValueError("YHC model-to-game control map is incomplete")
    if not set(CONTROL_MAP.values()).issubset(CONTROL_NAMES):
        raise ValueError("YHC model-to-game control map contains an unknown game command")
    if len(CLASS_ACTION_NAMES) != len(CLASS_NAMES):
        raise ValueError("YHC class action names do not match the model classes")
    for key, expected in MODEL_KWARGS.items():
        actual = checkpoint.get("model_kwargs", {}).get(key)
        if isinstance(expected, tuple):
            actual = tuple(actual) if actual is not None else None
        if actual != expected:
            raise ValueError(
                f"YHC checkpoint model_kwargs mismatch for {key}: {actual!r} != {expected!r}"
            )
    for key in ("baseline_mean_uv", "baseline_std_uv"):
        values = np.asarray(checkpoint.get(key))
        if values.shape != (len(YHC_CHANNELS),):
            raise ValueError(f"YHC checkpoint {key} must contain 12 channel values")


def build_preprocessor(args, checkpoint: dict) -> RealTimePreprocessor:
    return RealTimePreprocessor(
        input_sfreq=args.device_sfreq,
        model_sfreq=args.model_sfreq,
        window_seconds=args.window_sec,
        low_hz=args.filter_low,
        high_hz=args.filter_high,
        model_samples=1000,
        zscore_window=args.zscore_window,
        calibration_npz=args.calibration_npz,
        calibration_mean=checkpoint["baseline_mean_uv"],
        calibration_std=checkpoint["baseline_std_uv"],
        common_average_reference=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YHC 4-class Hyena realtime inference for NeuSen W/Neuracle TCP."
    )
    add_common_args(parser, DEFAULT_CHECKPOINT, NEUSEN_W_64_STREAM_CHANNELS)
    add_game_lsl_args(parser)
    parser.add_argument("--filter-low", type=float, default=4.0)
    parser.add_argument("--filter-high", type=float, default=38.0)
    parser.add_argument(
        "--zscore-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fallback only; checkpoint baseline normalization takes precedence.",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(device)
    checkpoint = load_state_dict(model, args.checkpoint, device)
    validate_checkpoint_contract(checkpoint)
    preprocessor = build_preprocessor(args, checkpoint)

    game_outlet = None
    if args.game_lsl and not args.dry_run:
        game_outlet = LSLControlOutlet(
            stream_name=args.stream_name,
            stream_type=args.stream_type,
            sample_rate=args.lsl_rate,
            source_id="yhc_hyena_racing_control",
        )

    run_realtime_loop(
        model=model,
        model_channels=YHC_CHANNELS,
        class_names=CLASS_NAMES,
        args=args,
        preprocessor=preprocessor,
        device=device,
        control_mapper=CONTROL_MAP if args.game_lsl else None,
        control_names=CONTROL_NAMES,
        game_outlet=game_outlet,
    )


if __name__ == "__main__":
    main()
