"""Launcher: python negotiation/run_scenarios.py --backend keyword --output-dir runs/negotiation/scenarios"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.console import utf8_stdio  # noqa: E402
from negotiation_agent.scenarios import main  # noqa: E402

if __name__ == "__main__":
    utf8_stdio()
    raise SystemExit(main())
