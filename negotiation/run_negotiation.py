"""Launcher: python negotiation/run_negotiation.py --payment-sensitive --reveal --output runs/negotiation/demo.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.console import utf8_stdio  # noqa: E402
from negotiation_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    utf8_stdio()
    raise SystemExit(main())
