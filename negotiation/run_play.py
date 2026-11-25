"""Launcher for the negotiation playground: you play the supplier, the buyer agent answers.

    python negotiation/run_play.py                      # CPU keyword reference, opens /play
    python negotiation/run_play.py --backend zoro-http --url http://127.0.0.1:30001 --decision-temperature 1

Every option of run_web.py works here. Without --backend the keyword reference is used, which
reads a few fixed phrases and is NOT the Zoro model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.console import utf8_stdio  # noqa: E402
from negotiation_agent.web import main  # noqa: E402

if __name__ == "__main__":
    utf8_stdio()
    args = sys.argv[1:]
    if "--backend" not in args and not any(a.startswith("--backend=") for a in args):
        args = ["--backend", "keyword", *args]
    if "--open" not in args:
        args = [*args, "--open", "/play"]
    raise SystemExit(main(args))
