"""Launchers that print Korean must not crash when stdout is a cp1252 pipe.

GitHub's Windows runners pipe stdout with the cp1252 code page. Before `console.utf8_stdio()` the
System 1 probe, the scenarios, a single negotiation and the dashboard help all exited with
UnicodeEncodeError there, while a Korean (cp949) PC passed. PYTHONIOENCODING=cp1252 reproduces the
runner on any machine.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

NEGOTIATION = Path(__file__).resolve().parents[1]
CP1252 = {**os.environ, "PYTHONIOENCODING": "cp1252"}


def run(*arguments):
    return subprocess.run([sys.executable, *map(str, arguments)], capture_output=True, env=CP1252, timeout=600)


def failure(done):
    return done.stderr.decode("utf-8", "replace")[-1500:]


@pytest.mark.parametrize("launcher", sorted(path.name for path in NEGOTIATION.glob("run_*.py")))
def test_every_launcher_help_survives_a_cp1252_pipe(launcher):
    done = run(NEGOTIATION / launcher, "--help")
    assert done.returncode == 0, failure(done)


def test_a_single_negotiation_prints_korean_through_a_cp1252_pipe(tmp_path):
    done = run(NEGOTIATION / "run_negotiation.py", "--seed", "1000", "--payment-sensitive", "--reveal",
               "--output", tmp_path / "demo.json")
    assert done.returncode == 0, failure(done)
    assert "라운드".encode("utf-8") in done.stdout


def test_the_scenarios_print_korean_through_a_cp1252_pipe(tmp_path):
    done = run(NEGOTIATION / "run_scenarios.py", "--backend", "keyword", "--output-dir", tmp_path)
    assert done.returncode == 0, failure(done)
    assert "합의".encode("utf-8") in done.stdout
    assert {path.name for path in tmp_path.glob("*.json")} >= {
        "s1_renewal_auto.json", "s2_exact_approval.json", "s3_safe_under_errors.json"}
