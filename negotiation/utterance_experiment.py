"""Launcher: python negotiation/utterance_experiment.py plan --state-dir runs/negotiation/utterance/v1

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only (CPU stand-in and test
doubles). Commands: plan, bench, fit, run, report, pack, unpack. See
negotiation/docs/UTTERANCE_AGENT_DESIGN.md, sections 13.7 and 13.9.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utterance_agent.experiment import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
