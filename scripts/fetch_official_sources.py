#!/usr/bin/env python3
"""Fetch three pinned Trida Python files, verify SHA256, and stop on any mismatch.

``--with-server`` additionally fetches the pinned official decision server
(serve_decide.py) so that it sits next to the codec and engine it imports.

No model weights, account credentials, repositories, or pip packages are fetched.
Downloaded files stay in their official repository layout for the CPU contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPOSITORY = "trillion-labs/trida-stack"
EXPECTED_PATHS = {
    "train/tools/export_kev_checkpoint.py",
    "inference/zoro-1/decide_request.py",
    "inference/zoro-1/decide_engine.py",
}
EXPECTED_SERVER_PATHS = {"inference/zoro-1/serve_decide.py"}


def fetch_sources(pins_path: Path, output_dir: Path, *, verify_only: bool = False,
                  with_server: bool = False) -> dict:
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    repository = pins["source_repository"]
    revision = pins["source_revision"]
    files = pins["official_source_files"]
    if repository != OFFICIAL_REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected the official Trida repository and an immutable 40-character commit.")
    if set(files) != EXPECTED_PATHS or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in files.values()):
        raise ValueError("The pins must contain exactly the three supported sources and their SHA256 values.")
    if with_server:
        server_files = pins.get("official_server_files", {})
        if set(server_files) != EXPECTED_SERVER_PATHS or any(
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in server_files.values()
        ):
            raise ValueError("The pins must contain the official decision server file and its SHA256 value.")
        files = {**files, **server_files}
    checked, pending = {}, []
    for relative_path, expected in files.items():
        destination = output_dir / relative_path
        if destination.is_file():
            data = destination.read_bytes()
            origin = "existing_verified_file"
        else:
            if verify_only:
                raise FileNotFoundError(f"Missing pinned source: {destination}")
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{relative_path}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError(f"Unexpectedly large source response: {relative_path}")
            origin = "downloaded_and_verified"
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA256 mismatch for {relative_path}: expected {expected}, got {actual}. "
                "No replacement was accepted; inspect or remove the invalid local file before retrying."
            )
        checked[relative_path] = {"sha256": actual, "origin": origin}
        if not destination.is_file():
            pending.append((destination, data))
    # Verify every response before writing any newly downloaded file.
    for destination, data in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return {
        "status": "verified", "source_repository": repository, "source_revision": revision,
        "output_dir": str(output_dir.resolve()), "files": checked,
        "model_weights_downloaded": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, default=ROOT / "config/model_pins.json")
    parser.add_argument("--out", type=Path, default=ROOT / ".cache/trida-stack")
    parser.add_argument("--verify-only", action="store_true", help="Check local files without network requests.")
    parser.add_argument("--with-server", action="store_true",
                        help="Also fetch the pinned official decision server (serve_decide.py).")
    args = parser.parse_args(argv)
    try:
        result = fetch_sources(args.pins, args.out, verify_only=args.verify_only, with_server=args.with_server)
    except (OSError, ValueError, KeyError) as error:
        print(f"Official source verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
