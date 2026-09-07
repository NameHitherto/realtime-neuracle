from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "einops": "einops",
    "pylsl": "pylsl",
    "torch": "torch",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "Pillow": "PIL",
}


def import_package(distribution: str, module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return True, version
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def model_self_test() -> None:
    import numpy as np
    import torch

    from realtime_common import load_state_dict, predict_window_details
    from realtime_yhc_4class import (
        DEFAULT_CHECKPOINT,
        YHC_CHANNELS,
        build_model,
        build_preprocessor,
        validate_checkpoint_contract,
    )

    device = torch.device("cpu")
    model = build_model(device)
    checkpoint = load_state_dict(model, DEFAULT_CHECKPOINT, device)
    validate_checkpoint_contract(checkpoint)
    args = argparse.Namespace(
        device_sfreq=1000.0,
        model_sfreq=250.0,
        window_sec=4.0,
        filter_low=4.0,
        filter_high=38.0,
        zscore_window=False,
        calibration_npz=None,
    )
    preprocessor = build_preprocessor(args, checkpoint)
    fake = np.zeros((len(YHC_CHANNELS), 4000), dtype=np.float32)
    window = preprocessor.transform(fake)
    if window.shape != (12, 1000) or not np.isfinite(window).all():
        raise RuntimeError(f"Unexpected preprocessed shape/content: {window.shape}")
    _, _, raw_probs, _ = predict_window_details(
        model,
        window,
        device,
        deque(maxlen=3),
    )
    if raw_probs.shape != (4,) or not np.isclose(raw_probs.sum(), 1.0, atol=1e-5):
        raise RuntimeError("YHC model probability output is invalid")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    errors: list[str] = []
    if sys.version_info[:2] != (3, 10):
        errors.append("This project requires Python 3.10 to match the validated environment")

    for distribution, module in REQUIRED_PACKAGES.items():
        ok, detail = import_package(distribution, module)
        print(f"{distribution:>10}: {'OK' if ok else 'FAILED'} ({detail})")
        if not ok:
            errors.append(f"{distribution}: {detail}")

    required_files = [
        ROOT / "checkpoints" / "yhc_hyena_final_realtime.pth",
        ROOT / "configs" / "neusen_w_64_channels_template.txt",
        ROOT / "experiment_logger.py",
        ROOT / "model_deps" / "model_hyena_input_motor_rhythm_enhanced_classifier.py",
        ROOT / "GUI" / "package.json",
        WORKSPACE / "虚拟任务竞速赛06251432" / "虚拟任务竞速赛.exe",
        WORKSPACE
        / "虚拟任务竞速赛06251432"
        / "虚拟任务竞速赛_Data"
        / "StreamingAssets"
        / "LSLInletConfig.txt",
    ]
    for path in required_files:
        ok = path.exists()
        print(f"{path.name:>55}: {'OK' if ok else 'MISSING'}")
        if not ok:
            errors.append(str(path))

    lsl_config = required_files[-1]
    if lsl_config.exists():
        value = lsl_config.read_text(encoding="utf-8").strip()
        print(f"{'game LSL config':>55}: {value}")
        if value != "EEGback|EEG":
            errors.append(f"Unexpected game LSL config: {value!r}")

    if not errors:
        try:
            model_self_test()
            print(f"{'YHC model self-test':>55}: OK")
        except Exception as exc:
            print(f"{'YHC model self-test':>55}: FAILED ({exc})")
            errors.append(f"YHC model self-test: {exc}")

    if errors:
        print("\nEnvironment check failed:")
        for error in errors:
            print(f"- {error}")
        print("Recreate with: .\\00_setup_environment.ps1")
        return 1

    print("\nEnvironment check passed. YHC realtime runtime is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
