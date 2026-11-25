"""Launcher: python negotiation/run_probe.py --backend heuristic --output runs/negotiation/probe.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
