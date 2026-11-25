#!/usr/bin/env python3
"""One command to prepare a GPU runtime (Colab, Kaggle, or an event machine) for Zoro-1.

Steps, in order (choose a subset with --steps):

  env      report GPU, VRAM, RAM and disk; stop when no CUDA GPU is attached
  install  keep the runtime's working torch, install requirements-gpu.txt and this package
  sources  fetch the pinned official Trida sources and verify their SHA256
  model    download the pinned Zoro-1 snapshot (needs an HF token with access to the gated model)
  export   run the official text-only exporter in a separate process
  probe    one real decision through the native adapter (sessionguard zoro-probe). The
           negotiation-only copy of this script does not include the SessionGuard CLI,
           so the default steps stop at export; use negotiation/run_probe.py instead.

Each step records what actually happened in <run-dir>/setup_report.json. Paths
needed by later commands are written to .cache/gpu_paths.json. The HF token is
read from HF_TOKEN, Colab secrets or Kaggle secrets and is never printed or saved.

Authoring validation: the env/sources steps and the helper functions ran on a
CPU-only Linux machine. The install, model, export and probe steps have NOT
been executed on Colab, Kaggle or any GPU by the author of this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("env", "install", "sources", "model", "export", "probe")
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "jev_head.pt"]
EXPORT_FILES = ("head.pt", "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
MARKER = "sessionguard_export_complete.json"


def load_pins() -> dict:
    return json.loads((ROOT / "config" / "model_pins.json").read_text(encoding="utf-8"))


def detect_platform() -> str:
    if Path("/kaggle/working").exists():
        return "kaggle"
    if Path("/content").exists() and importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "other"


def default_work_dir(kind: str) -> Path:
    """Weights and exports are large; on Kaggle keep them out of the saved /kaggle/working output."""
    return Path("/kaggle/temp/sessionguard") if kind == "kaggle" else ROOT / ".cache"


def environment_report() -> dict:
    report: dict = {
        "platform": detect_platform(), "python": platform.python_version(), "system": platform.platform(),
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2), "gpu_devices": [],
        "optional_compiled_packages": [p for p in ("fla", "causal_conv1d") if importlib.util.find_spec(p)],
    }
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines() if ":" in line)
        for key, name in (("MemTotal", "ram_total_gib"), ("MemAvailable", "ram_available_gib")):
            if key in values:
                report[name] = round(int(values[key].split()[0]) / 2**20, 2)
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        report["nvidia_smi"] = query.stdout.strip().splitlines() if query.returncode == 0 else query.stderr.strip()
    try:
        import torch
    except ImportError:
        report.update(torch=None, cuda_available=False)
        return report
    report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            capability = torch.cuda.get_device_capability(index)
            report["gpu_devices"].append({
                "index": index, "name": torch.cuda.get_device_name(index),
                "vram_total_gib": round(total / 2**30, 2), "vram_free_gib": round(free / 2**30, 2),
                "compute_capability": list(capability),
                # The adapter's "auto" uses the same rule: native BF16 needs capability >= 8.
                "adapter_auto_dtype": "bf16" if capability[0] >= 8 and torch.cuda.is_bf16_supported() else "fp16",
            })
    return report


def resolve_hf_token(environ: dict | None = None) -> tuple[str | None, str]:
    """Return (token, where it came from). The token itself must never be logged."""
    environ = os.environ if environ is None else environ
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if environ.get(name):
            return environ[name], f"environment:{name}"
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN")
        if token:
            return token, "colab_secret:HF_TOKEN"
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token, "kaggle_secret:HF_TOKEN"
    except Exception:
        pass
    return None, "huggingface_hub cached login, if any"


def export_inputs(original: Path, temperature: float, pins: dict) -> dict:
    return {
        "model_revision": pins["model_revision"], "source_revision": pins["source_revision"],
        "original_checkpoint": str(original.resolve()), "temperature": temperature,
        "exporter_sha256": pins["official_source_files"]["train/tools/export_kev_checkpoint.py"],
    }


def export_is_reusable(export_dir: Path, expected_inputs: dict) -> bool:
    """Reuse only a finished export of exactly these inputs whose files still have the recorded sizes."""
    try:
        previous = json.loads((export_dir / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if previous.get("inputs") != expected_inputs:
        return False
    sizes = previous.get("file_sizes", {})
    return all((export_dir / name).is_file() and (export_dir / name).stat().st_size == sizes.get(name)
               for name in EXPORT_FILES)


def zoro_options(paths: dict, dtype: str, temperature: float) -> list[str]:
    return ["--backend", "zoro", "--original-checkpoint", paths["original_checkpoint"],
            "--exported-checkpoint", paths["exported_checkpoint"],
            "--official-source-dir", paths["official_source_dir"], "--device", "cuda:0",
            "--dtype", dtype, "--decision-temperature", str(temperature),
            "--max-input-tokens", "2048", "--max-decisions", "5", "--max-inspections", "4"]


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), **kwargs)


def step_install(work: Path) -> dict:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("This runtime has no torch. Use a GPU runtime image that already provides CUDA PyTorch.") from error
    constraint = work / "keep-torch.txt"
    constraint.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    _run([sys.executable, "-m", "pip", "install", "--quiet", "-r", ROOT / "requirements-gpu.txt",
          "-c", constraint], check=True)
    _run([sys.executable, "-m", "pip", "install", "--quiet", "-c", ROOT / "requirements-cpu.lock.txt",
          "-c", constraint, "-e", ROOT], check=True)
    after = importlib.metadata.version("torch")
    if after != torch_version:
        raise RuntimeError(f"torch changed from {torch_version} to {after}; restart with a clean runtime.")
    versions = {}
    for package in ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "fastapi", "httpx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {"torch_kept": torch_version, "versions": versions,
            "note": "Restart the runtime if the platform asks for it, then rerun with --steps sources,model,export,probe."}


def step_sources(source_dir: Path, with_server: bool) -> dict:
    spec = importlib.util.spec_from_file_location("_fetch_official_sources", ROOT / "scripts" / "fetch_official_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fetch_sources(ROOT / "config" / "model_pins.json", source_dir, with_server=with_server)


def step_model(cache_dir: Path, pins: dict) -> tuple[Path, dict]:
    from huggingface_hub import hf_hub_download, snapshot_download

    token, token_source = resolve_hf_token()
    try:
        hf_hub_download(pins["model_id"], "config.json", revision=pins["model_revision"],
                        token=token, cache_dir=str(cache_dir))
    except Exception as error:
        raise RuntimeError(
            f"Cannot read the gated model ({type(error).__name__}). Accept the access terms on the model page "
            "with your own account and provide a read token as HF_TOKEN (Colab/Kaggle secret or environment)."
        ) from error
    started = time.perf_counter()
    original = Path(snapshot_download(repo_id=pins["model_id"], revision=pins["model_revision"], token=token,
                                      cache_dir=str(cache_dir), allow_patterns=ALLOW_PATTERNS))
    if not (original / "jev_head.pt").is_file():
        raise RuntimeError("The snapshot has no original decision head (jev_head.pt). Do not substitute another model.")
    if original.name != pins["model_revision"]:
        raise RuntimeError("The snapshot path does not end in the pinned revision.")
    size = sum(path.stat().st_size for path in original.rglob("*") if path.is_file())
    return original, {"token_source": token_source, "download_seconds": round(time.perf_counter() - started, 1),
                      "snapshot_bytes": size, "original_checkpoint": str(original)}


def _tree_rss(process) -> int:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except Exception:
            pass
    return total


def step_export(original: Path, export_dir: Path, source_dir: Path, pins: dict, log_path: Path) -> dict:
    relative = "train/tools/export_kev_checkpoint.py"
    exporter = source_dir / relative
    if hashlib.sha256(exporter.read_bytes()).hexdigest() != pins["official_source_files"][relative]:
        raise ValueError("Official exporter source does not match its pinned SHA256.")
    temperature_file = original / "temperature.json"
    temperature = float(json.loads(temperature_file.read_text()).get("text", 1.0)) if temperature_file.is_file() else 1.0
    if not 0 < temperature < float("inf"):
        raise ValueError("The checkpoint temperature must be finite and positive.")
    inputs = export_inputs(original, temperature, pins)
    if export_is_reusable(export_dir, inputs):
        return {"reused": True, "temperature": temperature, "exported_checkpoint": str(export_dir)}
    (export_dir / MARKER).unlink(missing_ok=True)
    command = [sys.executable, str(exporter), "--ckpt", str(original), "--out", str(export_dir),
               "--temperature", str(temperature)]
    print("$", " ".join(command), flush=True)
    started, peak = time.perf_counter(), None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            import psutil
            observed, peak = psutil.Process(process.pid), 0
            while process.poll() is None:
                peak = max(peak, _tree_rss(observed))
                time.sleep(0.2)
        except ImportError:
            pass
        code = process.wait()
    if code != 0:
        print(log_path.read_text(encoding="utf-8")[-3000:])
        raise RuntimeError(f"The official exporter failed with code {code}; no completion marker was written.")
    if not all((export_dir / name).is_file() and (export_dir / name).stat().st_size > 0 for name in EXPORT_FILES):
        raise RuntimeError("The export is missing required files.")
    completed = {"inputs": inputs, "file_sizes": {name: (export_dir / name).stat().st_size for name in EXPORT_FILES},
                 "sampled_process_tree_peak_rss_bytes": peak}
    temporary = export_dir / (MARKER + ".tmp")
    temporary.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    temporary.replace(export_dir / MARKER)
    return {"reused": False, "temperature": temperature, "exported_checkpoint": str(export_dir),
            "export_seconds": round(time.perf_counter() - started, 1),
            "sampled_process_tree_peak_rss_bytes": peak}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default=",".join(step for step in STEPS if step != "probe"), help="Comma-separated subset of: " + ", ".join(STEPS))
    parser.add_argument("--work-dir", type=Path, help="Where weights and the export go (default: .cache, or /kaggle/temp on Kaggle)")
    parser.add_argument("--run-dir", type=Path, help="Where the small result files go (default: runs/gpu-setup-<UTC time>)")
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--with-server-source", action="store_true",
                        help="Also fetch the pinned official decision server source")
    parser.add_argument("--allow-no-gpu", action="store_true", help="Do not stop in the env step without a CUDA GPU")
    args = parser.parse_args(argv)
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        parser.error(f"unknown steps: {unknown}")
    steps = [step for step in STEPS if step in steps]

    pins = load_pins()
    work = (args.work_dir or default_work_dir(detect_platform())).resolve()
    work.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir or ROOT / "runs" / ("gpu-setup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / ".cache" / "trida-stack"
    export_dir = work / ("zoro-text-" + pins["model_revision"][:12])
    paths_file = ROOT / ".cache" / "gpu_paths.json"
    paths = json.loads(paths_file.read_text(encoding="utf-8")) if paths_file.is_file() else {}
    paths["official_source_dir"] = str(source_dir)
    report: dict = {"schema_version": "1.0", "started_at": datetime.now(timezone.utc).isoformat(),
                    "model_id": pins["model_id"], "model_revision": pins["model_revision"],
                    "source_revision": pins["source_revision"], "work_dir": str(work), "steps": {}}

    def save() -> None:
        (run_dir / "setup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        paths_file.write_text(json.dumps(paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        started = time.perf_counter()
        try:
            if step == "env":
                detail = environment_report()
                print(json.dumps(detail, ensure_ascii=False, indent=2))
                if not detail["cuda_available"] and not args.allow_no_gpu:
                    raise RuntimeError("No CUDA GPU is attached. Select a GPU runtime and run this step again.")
                if detail["optional_compiled_packages"]:
                    print("Note: the native adapter refuses to run while these packages are installed:",
                          detail["optional_compiled_packages"])
            elif step == "install":
                detail = step_install(work)
            elif step == "sources":
                detail = step_sources(source_dir, args.with_server_source)
            elif step == "model":
                original, detail = step_model(work / "hf", pins)
                paths["original_checkpoint"] = str(original)
            elif step == "export":
                if "original_checkpoint" not in paths:
                    raise RuntimeError("Run the model step first; no snapshot path is recorded.")
                detail = step_export(Path(paths["original_checkpoint"]), export_dir, source_dir, pins,
                                     run_dir / "export_process.log")
                paths["exported_checkpoint"] = detail["exported_checkpoint"]
                paths["checkpoint_temperature"] = detail["temperature"]
            else:
                if not {"original_checkpoint", "exported_checkpoint"} <= set(paths):
                    raise RuntimeError("Run the model and export steps first.")
                output = run_dir / "first_probe.json"
                command = [sys.executable, "-m", "sessionguard", "zoro-probe",
                           "--original-checkpoint", paths["original_checkpoint"],
                           "--exported-checkpoint", paths["exported_checkpoint"],
                           "--official-source-dir", str(source_dir), "--device", "cuda:0",
                           "--dtype", args.dtype, "--max-input-tokens", "2048", "--output", str(output)]
                code = _run(command, cwd=ROOT).returncode
                if code != 0 or not output.is_file():
                    raise RuntimeError(f"The probe failed (exit {code}); see {output}.failure.json")
                metadata = json.loads(output.read_text(encoding="utf-8"))["metadata"]
                paths["resolved_dtype"] = metadata["dtype"]
                paths["decision_temperature"] = metadata["decision_temperature"]
                detail = {key: metadata.get(key) for key in (
                    "dtype", "gpu_name", "decision_temperature", "temperature_source", "input_tokens_per_question",
                    "backbone_forward_latency_ms", "peak_cuda_memory_allocated_bytes", "runtime_versions")}
                detail["output"] = str(output)
            report["steps"][step] = {"status": "ok", "seconds": round(time.perf_counter() - started, 1), **detail}
        except Exception as error:
            report["steps"][step] = {"status": "failed", "seconds": round(time.perf_counter() - started, 1),
                                     "error_type": type(error).__name__, "error": str(error)[:2000]}
            save()
            print(f"\nStep '{step}' failed: {type(error).__name__}: {error}", file=sys.stderr)
            print(f"Recorded in {run_dir / 'setup_report.json'}", file=sys.stderr)
            return 1
        save()

    print(f"\nReport: {run_dir / 'setup_report.json'}\nPaths:  {paths_file}")
    if {"original_checkpoint", "exported_checkpoint", "resolved_dtype", "decision_temperature"} <= set(paths):
        options = zoro_options(paths, paths["resolved_dtype"], paths["decision_temperature"])
        print("\nNext, the 16 development cases with the same explicit options:\n")
        print("python -m sessionguard evaluate --split dev " + " ".join(options) + " --output runs/dev_zoro.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
