"""Launcher: python negotiation/run_utterance_probe.py --backend keyword --output runs/negotiation/utterance.json

Not executed on a GPU as of 2026-11-18; only CPU reference backends have been run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.utterance_probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
