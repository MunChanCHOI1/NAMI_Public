"""Launcher: python negotiation/run_reader_check.py --backend keyword --output-dir runs/negotiation/final_reader/<new>

Protocol final-reader-2026-11-19.v1 (negotiation/src/negotiation_agent/reader_check.py). Executed once on a
local GPU on 2026-11-19 with the bytes of commit fce9eef; negotiation/docs/PLAN.md section 16.1 has the results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.reader_check import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
