#!/usr/bin/env python3
"""Install, start and stop the official Zoro-1 decision server on this machine.

The pinned Trida source serves decisions with ``serve_decide.py`` on stock
SGLang 0.5.22 ("pip install 'sglang==0.5.22' fastapi uvicorn"). SGLang brings
its own torch, so it gets a separate virtual environment and never replaces the
runtime's working CUDA PyTorch.

  python scripts/official_server.py install
  python scripts/official_server.py start      # waits until GET /health answers
  python scripts/official_server.py stop

The server keeps most of the GPU memory. Do not load the native adapter
(--backend zoro) on the same GPU while it runs.

Validation: command construction and readiness checks ran on CPU with a stand-in
server. SGLang 0.5.22 installed on Colab with a separate Python 3.12.3 environment;
the default Python 3.13.16 install failed on outlines_core 0.1.26. Installation
does not establish successful serving; see docs/VALIDATION.md for engine results.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / ".cache" / "trida-stack"
SERVER_DIR = SOURCE_DIR / "inference" / "zoro-1"
STATE_DIR = ROOT / "runs" / "official-server"
PACKAGES = ["sglang==0.5.22", "fastapi", "uvicorn"]


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_server_venv(venv: Path) -> Path:
    """Bootstrap pip, recovering an ensurepip failure without clearing the venv."""
    python = venv_python(venv)
    pip_check = [str(python), "-m", "pip", "--version"]
    try:
        if not python.is_file():
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run(pip_check, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise
        print("venv/pip bootstrap failed; recovering with uv while preserving existing files.", flush=True)
        # A prior attempt may have left a working interpreter without pip.
        interpreter = str(python) if python.is_file() else sys.executable
        subprocess.run(
            [uv, "venv", "--seed", "--allow-existing", "--python", interpreter,
             "--no-python-downloads", str(venv)],
            check=True, env={**os.environ, "UV_VENV_CLEAR": "false"},
        )
        subprocess.run(pip_check, check=True, capture_output=True, text=True)
    return python


def server_command(python: Path, checkpoint: Path, port: int, mem_fraction: float, context_length: int) -> list[str]:
    # Keep venv/bin/python symlinks: resolving that file can select system Python.
    return [str(python.absolute()), "serve_decide.py", str(checkpoint.resolve()), "--host", "127.0.0.1", "--port", str(port),
            "--mem-fraction", str(mem_fraction), "--context-length", str(context_length)]


def health(port: int, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            body = json.loads(response.read(10_000))
            return body if isinstance(body, dict) and body.get("ok") else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_until_healthy(port: int, is_running, *, timeout_seconds: float, interval: float = 5.0) -> dict | None:
    """Poll /health until it answers, the process exits, or the time is up."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_running():
            return None
        answer = health(port)
        if answer:
            return answer
        if not is_running():
            return None
        time.sleep(interval)
    return None


def stop_started_process(process) -> None:
    """Clean up only the child/group started by this invocation after a failure."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def process_identity(pid: int) -> dict | None:
    """Bind persisted ownership to a process birth time, not a reusable PID."""
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("Process identity checks need psutil from requirements-gpu.txt.") from error
    try:
        return {"pid": pid, "created_at": psutil.Process(pid).create_time()}
    except psutil.NoSuchProcess:
        return None


def verify_sources() -> None:
    spec = importlib.util.spec_from_file_location("_fetch_official_sources", ROOT / "scripts" / "fetch_official_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.fetch_sources(ROOT / "config" / "model_pins.json", SOURCE_DIR, with_server=True)


def default_checkpoint() -> Path | None:
    paths = ROOT / ".cache" / "gpu_paths.json"
    if paths.is_file():
        value = json.loads(paths.read_text(encoding="utf-8")).get("original_checkpoint")
        return Path(value) if value else None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("install", "start", "stop"))
    parser.add_argument("--venv", type=Path, default=ROOT / ".cache" / "venv-sglang")
    parser.add_argument("--checkpoint", type=Path, help="Original pinned snapshot (default: from .cache/gpu_paths.json)")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--wait-minutes", type=float, default=30.0)
    args = parser.parse_args(argv)
    args.venv = args.venv.resolve()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    pid_file, log_file = STATE_DIR / "server.pid", STATE_DIR / "server.log"
    identity_file = STATE_DIR / "server_process.json"

    if args.command == "install":
        python = ensure_server_venv(args.venv)
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", *PACKAGES], check=True)
        frozen = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True)
        (STATE_DIR / "sglang_environment.txt").write_text(frozen.stdout, encoding="utf-8")
        print(f"Installed into {args.venv}; versions saved to {STATE_DIR / 'sglang_environment.txt'}")
        return 0

    if args.command == "stop":
        if not pid_file.is_file():
            print("No recorded server process.")
            return 0
        pid = int(pid_file.read_text().strip())
        if not identity_file.is_file():
            print("The legacy PID has no process identity; refusing to signal an unverified process.", file=sys.stderr)
            return 1
        recorded = json.loads(identity_file.read_text(encoding="utf-8"))
        current = process_identity(pid)
        if current is None or current != recorded:
            print("The recorded process is gone or its PID was reused; no signal sent.")
            pid_file.unlink(missing_ok=True)
            identity_file.unlink(missing_ok=True)
            return 0
        try:
            os.killpg(pid, signal.SIGTERM) if os.name != "nt" else os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to server process group {pid}.")
        except ProcessLookupError:
            print("The recorded server process is already gone.")
        pid_file.unlink(missing_ok=True)
        identity_file.unlink(missing_ok=True)
        return 0

    checkpoint = args.checkpoint or default_checkpoint()
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
    if checkpoint is None or not (checkpoint / "jev_head.pt").is_file():
        print("No decision checkpoint with jev_head.pt. Run scripts/gpu_setup.py --steps model first, "
              "or pass --checkpoint.", file=sys.stderr)
        return 1
    if not venv_python(args.venv).is_file():
        print("Run 'python scripts/official_server.py install' first.", file=sys.stderr)
        return 1
    if pid_file.exists():
        print(f"A server launch is already recorded in {pid_file}; resolve it before starting again.", file=sys.stderr)
        return 1
    if health(args.port):
        print(f"A server already answers on port {args.port}; not starting another.", file=sys.stderr)
        return 1
    try:
        verify_sources()
    except (OSError, ValueError, KeyError) as error:
        print(f"Official source verification failed: {error}", file=sys.stderr)
        return 1
    process_identity(os.getpid())  # ensure identity inspection is available before launching
    command = server_command(venv_python(args.venv), checkpoint, args.port, args.mem_fraction, args.context_length)
    venv_bin = str(venv_python(args.venv).parent)
    # Engine build helpers such as ninja must come from this environment too.
    server_env = {**os.environ, "PATH": venv_bin + os.pathsep + os.environ.get("PATH", "")}
    print("$", " ".join(command), f"(cwd {SERVER_DIR}; tools {venv_bin})", flush=True)
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=SERVER_DIR, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt", env=server_env)
    pid_file.write_text(str(process.pid), encoding="utf-8")
    started = time.monotonic()
    try:
        identity_file.write_text(json.dumps(process_identity(process.pid)), encoding="utf-8")
        answer = wait_until_healthy(args.port, lambda: process.poll() is None, timeout_seconds=args.wait_minutes * 60)
    except BaseException:
        stop_started_process(process)
        pid_file.unlink(missing_ok=True)
        identity_file.unlink(missing_ok=True)
        raise
    if not answer:
        stop_started_process(process)
        pid_file.unlink(missing_ok=True)
        identity_file.unlink(missing_ok=True)
    record = {"command": command, "checkpoint": str(checkpoint), "port": args.port, "venv_bin": venv_bin,
              "SGLANG_MAMBA_CONV_DTYPE": server_env.get("SGLANG_MAMBA_CONV_DTYPE"),
              "seconds_until_ready": round(time.monotonic() - started, 1), "health": answer,
              "exit_code_if_stopped": process.poll()}
    (STATE_DIR / "start_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not answer:
        print(log_file.read_text(encoding="utf-8", errors="replace")[-4000:])
        print(f"\nThe server did not become ready. Full log: {log_file}", file=sys.stderr)
        return 1
    print(f"Ready after {record['seconds_until_ready']}s: {answer}")
    print(f"Next: python negotiation/run_probe.py --backend zoro-http --url http://127.0.0.1:{args.port} "
          "--decision-temperature 1 --request-timeout 180 --output runs/negotiation/probe_http.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
