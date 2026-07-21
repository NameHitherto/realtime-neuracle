from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REQUIRED_PACKAGES = ["numpy", "scipy", "einops", "pylsl", "torch", "fastapi", "uvicorn", "PIL"]
ROOT = Path(__file__).resolve().parent


def package_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    missing = []
    for name in REQUIRED_PACKAGES:
        ok = package_ok(name)
        print(f"{name:>8}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(name)

    required_files = [
        ROOT / "checkpoints" / "bcic2a_hyena_rhythm_subject03.pth",
        ROOT / "checkpoints" / "hgd_cosine_temporal_subject003_phase2.pth",
        ROOT / "model_deps" / "model_hyena_input_motor_rhythm_enhanced_classifier.py",
        ROOT / "model_deps" / "model_hyena_cosine_readout.py",
    ]
    for path in required_files:
        print(f"{path.name:>55}: {'OK' if path.exists() else 'MISSING'}")
        if not path.exists():
            missing.append(str(path))

    if missing:
        print("\nEnvironment check failed.")
        print("Install dependencies with: pip install -r requirements_realtime.txt")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
