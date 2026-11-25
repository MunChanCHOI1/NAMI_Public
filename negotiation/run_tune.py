"""Launcher: python negotiation/run_tune.py plan --state-dir runs/negotiation/tuning/<label>

NOT EXECUTED ON A GPU as of 2026-11-18: this launcher and the tuning runner behind it were
verified on CPU only, with CPU reference backends and test doubles.

Commands: plan | run --stage {smoke,regress,calibrate,select,confirm,all} | fit-map | board |
pack --output ZIP | restore --input ZIP. Every command takes --state-dir and the usual backend
flags. Development and calibration cases only; the held-out split is never touched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.tune import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
