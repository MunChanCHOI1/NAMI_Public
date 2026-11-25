"""Launcher: python negotiation/run_evaluation.py evaluate --split dev --output runs/negotiation/eval_dev.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.evaluation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
