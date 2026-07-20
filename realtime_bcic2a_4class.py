from pathlib import Path
import sys
import argparse

import torch

from realtime_common import (
    LSLControlOutlet,
    RealTimePreprocessor,
    add_common_args,
    add_game_lsl_args,
    load_state_dict,
    run_realtime_loop,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "model_deps"))
sys.path.insert(0, str(ROOT / "SST-DPN-Hyena"))

from model_hyena_input_motor_rhythm_enhanced_classifier import (  # noqa: E402
    HyenaInputMotorRhythmEnhancedClassifierNet,
)


BCIC2A_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

CLASS_NAMES = ["feet", "left_hand", "right_hand", "tongue"]
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "bcic2a_hyena_rhythm_subject03.pth"
CONTROL_MAP = {0: 2, 1: 0, 2: 1, 3: 3}
CONTROL_NAMES = {0: "left", 1: "right", 2: "forward", 3: "stop"}


def build_model(device: torch.device) -> HyenaInputMotorRhythmEnhancedClassifierNet:
    return HyenaInputMotorRhythmEnhancedClassifierNet(
        chans=22,
        samples=1000,
        num_classes=4,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=(50, 100, 200),
        device=device,
        hyena_layers=1,
        hyena_dropout=0.1,
        hyena_long_conv="implicit",
        hyena_long_kernel=63,
        hyena_min_scale=0.005,
        hyena_max_extra_scale=0.05,
        rhythm_modes=40,
        rhythm_dropout=0.05,
        rhythm_min_scale=0.002,
        rhythm_max_extra_scale=0.03,
        ssa_gate_scale=0.5,
        output_bins=(16, 10, 6),
        locality=1.25,
        max_logit_offset=2.0,
        classifier_dropout=0.1,
        classifier_maxnorm=1.0,
        sampling_rate=250.0,
        rhythm_max_scale=0.25,
    )


def main():
    parser = argparse.ArgumentParser(
        description="BCIC2a 4-class Hyena realtime inference for Neuracle TCP stream."
    )
    add_common_args(parser, DEFAULT_CHECKPOINT, BCIC2A_CHANNELS + ["TRG"])
    add_game_lsl_args(parser)
    parser.add_argument("--filter-low", type=float, default=4.0)
    parser.add_argument("--filter-high", type=float, default=38.0)
    parser.add_argument("--zscore-window", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(device)
    load_state_dict(model, args.checkpoint, device)
    preprocessor = RealTimePreprocessor(
        input_sfreq=args.device_sfreq,
        model_sfreq=args.model_sfreq,
        window_seconds=args.window_sec,
        low_hz=args.filter_low,
        high_hz=args.filter_high,
        model_samples=1000,
        zscore_window=args.zscore_window,
        calibration_npz=args.calibration_npz,
    )
    game_outlet = None
    if args.game_lsl and not args.dry_run:
        game_outlet = LSLControlOutlet(
            stream_name=args.stream_name,
            stream_type=args.stream_type,
            sample_rate=args.lsl_rate,
            source_id="bcic2a_hyena_racing_control",
        )
    run_realtime_loop(
        model=model,
        model_channels=BCIC2A_CHANNELS,
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
