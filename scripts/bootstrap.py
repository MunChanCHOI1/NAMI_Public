#!/usr/bin/env python3
"""Create a local CPU environment, install locked dependencies, and run tests."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="Install only; test status remains unverified")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required; Python 3.12 is the validated version")
    root = Path(__file__).resolve().parents[1]
    env = root / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    pip = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True)
    if pip.returncode:
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True, cwd=root)
    subprocess.run([str(python), "-m", "pip", "install", "-c", "requirements-cpu.lock.txt", "-e", ".[dev]"], check=True, cwd=root)
    if not args.skip_tests:
        # A fresh project-local base avoids another session's Windows temp ACLs.
        # Never reuse --basetemp: pytest clears that directory on startup.
        test_cache = root / ".cache" / "pytest"
        test_cache.mkdir(parents=True, exist_ok=True)
        test_base = Path(tempfile.mkdtemp(prefix="bootstrap-", dir=test_cache)) / "tests"
        subprocess.run([str(python), "-m", "pytest", "-q",
                        "--basetemp", str(test_base)], check=True, cwd=root)
    print(f"CPU environment ready: {python}")
    print("No model weights, GPU packages, remote repository, or paid resources were provisioned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
