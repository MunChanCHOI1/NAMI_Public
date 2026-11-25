"""Launcher: python negotiation/run_choice_probe.py --output runs/negotiation/choice.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.choice_probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
