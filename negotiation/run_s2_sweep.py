"""Pick the s2 (approval card) seed with the backend that will be demonstrated. Rule: scenarios.py.

    python negotiation/run_s2_sweep.py --backend keyword --output runs/negotiation/s2_sweep_keyword.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.console import utf8_stdio  # noqa: E402
from negotiation_agent.scenarios import sweep_main  # noqa: E402

if __name__ == "__main__":
    utf8_stdio()
    raise SystemExit(sweep_main())
