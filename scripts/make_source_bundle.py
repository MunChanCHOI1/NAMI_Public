#!/usr/bin/env python3
"""Build the source ZIP that the Negotiation tuning notebook verifies and unpacks.

GPU status: this script has NOT been executed on a GPU machine as of 2026-11-18 and
needs none. It was verified on CPU only, with temporary git repositories and test
doubles. No bundle built by it has been consumed by a GPU run yet.

  python scripts/make_source_bundle.py [--out runs/bundles] [--allow-dirty]

What goes into the bundle: the tracked files of the HEAD commit, taken with
``git archive``. Ignored and untracked files (``.cache``, ``runs``, weights, tokens)
and uncommitted edits are therefore never included. A working tree that differs from
HEAD is refused; ``--allow-dirty`` builds the HEAD bundle anyway and records
``dirty: true`` together with the entries that were left out.

The archive must pass the same checks as the source cell of the executed notebook
``notebooks/negotiation_colab_quickstart.ipynb``: exactly one ``pyproject.toml``, at most
32 MiB of file content, no symbolic links and no unsafe member names.

``content_sha256`` reproduces that cell's formula exactly::

    sha256("".join(f"{name}\\0{sha256(data).hexdigest()}\\n" for name in sorted(names)).encode())

where ``name`` is the POSIX path relative to the folder that holds ``pyproject.toml``.
Directory entries are skipped, so the value depends neither on the ZIP container
(compression, member order, timestamps) nor on a top-level folder prefix.
``zip_sha256`` does depend on the container and on the git version; it is informational.

Output: ``negotiation-source-<commit12>.zip`` and
``negotiation-source-<commit12>.manifest.json``. The last line printed is the bare
content hash, to be pasted into the notebook's ``EXPECTED_CONTENT_SHA256`` field.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
PROJECT_MARKER = "pyproject.toml"
MAX_TOTAL_BYTES = 32 * 1024**2          # the limit of the executed notebook's source cell
BUNDLE_NAME = "negotiation-source-{commit12}"
PROVENANCE_MODULE = "negotiation/src/negotiation_agent/provenance.py"
# Defence in depth: these mirror the repository's .gitignore. A tracked file of this kind
# means something was committed that must not leave the machine inside a bundle.
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".gguf")
KEY_SUFFIXES = (".pem", ".key")
TOKEN_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{30,}")
NOTICE = ("Tracked files of the commit only; uncommitted changes are not included. "
          "content_sha256 identifies the source; zip_sha256 depends on the container and is informational. "
          "Not executed on a GPU as of 2026-11-18.")


class BundleError(RuntimeError):
    """The bundle was refused. Nothing is left in the output folder."""


def content_sha256(contents: Mapping[str, bytes]) -> str:
    """The content hash of the executed notebook's source cell, for names relative to the project root."""
    return hashlib.sha256("".join(
        f"{name}\0{hashlib.sha256(contents[name]).hexdigest()}\n" for name in sorted(contents)
    ).encode()).hexdigest()


def read_archive(path: Path) -> tuple[str, dict[str, bytes]]:
    """Read a source ZIP under the checks of the notebook's source cell.

    Returns the top-level prefix ("" for a flat archive) and the files keyed by their path
    relative to the single ``pyproject.toml``.
    """
    with zipfile.ZipFile(path) as bundle:
        members = [member for member in bundle.infolist() if not member.is_dir()]
        roots = [member.filename[:-len(PROJECT_MARKER)] for member in members
                 if member.filename == PROJECT_MARKER or member.filename.endswith("/" + PROJECT_MARKER)]
        if len(roots) != 1:
            raise BundleError(f"expected exactly one {PROJECT_MARKER} in the archive, found {len(roots)}")
        total = sum(member.file_size for member in members)
        if total > MAX_TOTAL_BYTES:
            raise BundleError(f"archive content is {total} bytes; the notebook accepts at most {MAX_TOTAL_BYTES}")
        prefix = roots[0]
        contents: dict[str, bytes] = {}
        for member in members:
            if not member.filename.startswith(prefix):
                raise BundleError(f"member outside the project folder: {member.filename!r}")
            name = member.filename[len(prefix):]
            pure = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or pure.is_absolute() or ".." in pure.parts
                    or name in contents or stat.S_ISLNK(member.external_attr >> 16)):
                raise BundleError(f"member name or type the notebook does not accept: {member.filename!r}")
            contents[name] = bundle.read(member)
    return prefix, contents


def _git(repo: Path, *arguments: str) -> bytes:
    # A caller's GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE would silently point git at another repository.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX")}
    try:
        done = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, env=environment)
    except FileNotFoundError as error:
        raise BundleError("git is required to build a source bundle") from error
    if done.returncode:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise BundleError(f"git {arguments[0]} failed: {detail or 'exit ' + str(done.returncode)}")
    return done.stdout


def head_commit(repo: Path) -> str:
    """HEAD as a full commit id; the repository path must be the top of its working tree."""
    top = Path(_git(repo, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != Path(repo).resolve():
        # `git archive` run below the top level would archive that subfolder only.
        raise BundleError(f"{repo} is not the top level of a git working tree ({top})")
    commit = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise BundleError(f"unexpected commit id: {commit!r}")
    return commit


def uncommitted_entries(repo: Path) -> list[str]:
    """`git status --porcelain` lines: modified, staged and untracked (not ignored) paths."""
    output = _git(repo, "status", "--porcelain", "--untracked-files=normal").decode("utf-8", "replace")
    return [line for line in output.splitlines() if line.strip()]


def tracked_files(repo: Path, commit: str) -> set[str]:
    """Paths of the commit's tracked files (blobs and links; submodule entries are not files)."""
    names = set()
    for entry in _git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if entry:
            header, _, name = entry.partition(b"\t")
            if header.split()[1] != b"commit":
                names.add(name.decode("utf-8"))
    return names


def write_archive(repo: Path, commit: str, destination: Path, prefix: str = "") -> None:
    """`git archive` of one commit. The content hash is the same for every ``prefix``."""
    # Line endings follow the committed .gitattributes, not this machine's autocrlf setting.
    _git(repo, "-c", "core.autocrlf=false", "archive", "--format=zip", f"--prefix={prefix}",
         "-o", str(destination), commit)


def forbidden_content(contents: Mapping[str, bytes]) -> list[str]:
    """Tracked paths that look like weights, key files or an access token. Values are never returned."""
    found = []
    for name in sorted(contents):
        base = PurePosixPath(name).name
        if base.endswith(WEIGHT_SUFFIXES):
            found.append(f"{name} (model weight file type)")
        elif base.endswith(KEY_SUFFIXES) or base == ".env" or (base.startswith(".env.") and base != ".env.example"):
            found.append(f"{name} (credential file type)")
        elif TOKEN_PATTERN.search(contents[name]):
            found.append(f"{name} (contains a Hugging Face token pattern)")
    return found


def bundle_input_hashes(contents: Mapping[str, bytes]) -> dict[str, str] | None:
    """`provenance.input_hashes()` as the bundled source itself computes it once unpacked.

    The archive is unpacked into a temporary folder and its own provenance module is asked,
    so the value describes the bundle even when the working tree differs from HEAD. Results
    produced from this bundle record the same mapping. None when the archive has no
    Negotiation provenance module.
    """
    if PROVENANCE_MODULE not in contents:
        return None
    with tempfile.TemporaryDirectory(prefix="negotiation-bundle-") as folder:
        root = Path(folder).resolve()
        for name, data in contents.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        module_name = "_bundled_negotiation_provenance_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, root / PROVENANCE_MODULE)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            hashes = dict(module.input_hashes())
        except Exception as error:
            raise BundleError(f"the bundled provenance module could not hash its inputs: "
                              f"{type(error).__name__}: {error}") from error
        finally:
            sys.modules.pop(module_name, None)
            sys.dont_write_bytecode = previous
    for name, digest in hashes.items():
        if name not in contents or hashlib.sha256(contents[name]).hexdigest() != digest:
            raise BundleError(f"input hash of {name!r} does not describe the archived file")
    return hashes


def build_bundle(repo: Path, out_dir: Path, *, allow_dirty: bool = False) -> dict:
    """Write the ZIP and its manifest. Returns {"zip": Path, "manifest_path": Path, "manifest": dict}."""
    repo = Path(repo)
    commit = head_commit(repo)
    entries = uncommitted_entries(repo)
    if entries and not allow_dirty:
        shown = "\n".join("  " + line for line in entries[:10])
        more = f"\n  ... and {len(entries) - 10} more" if len(entries) > 10 else ""
        raise BundleError("the working tree differs from HEAD; a bundle holds committed files only.\n"
                          f"{shown}{more}\n"
                          "Commit the changes, or pass --allow-dirty to bundle HEAD without them.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = BUNDLE_NAME.format(commit12=commit[:12])
    archive, manifest_path = out_dir / f"{stem}.zip", out_dir / f"{stem}.manifest.json"
    partial = out_dir / f".{stem}.{uuid.uuid4().hex[:8]}.partial"
    try:
        write_archive(repo, commit, partial)
        prefix, contents = read_archive(partial)
        tracked = tracked_files(repo, commit)
        if prefix or set(contents) != tracked:
            missing, extra = sorted(tracked - set(contents))[:5], sorted(set(contents) - tracked)[:5]
            raise BundleError("the archive is not exactly the tracked files of the commit "
                              f"(missing, e.g. export-ignore: {missing}; not tracked: {extra})")
        forbidden = forbidden_content(contents)
        if forbidden:
            raise BundleError("tracked files that must not be bundled: " + "; ".join(forbidden[:10]))
        manifest = {
            "commit": commit,
            "dirty": bool(entries),
            "file_count": len(contents),
            "total_bytes": sum(len(data) for data in contents.values()),
            "files": {name: hashlib.sha256(contents[name]).hexdigest() for name in sorted(contents)},
            "content_sha256": content_sha256(contents),
            "zip_sha256": hashlib.sha256(partial.read_bytes()).hexdigest(),
            "input_sha256": bundle_input_hashes(contents),
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "archive_name": archive.name,
            "uncommitted_entries_not_in_bundle": entries,
            "notice": NOTICE,
        }
        os.replace(partial, archive)
    finally:
        partial.unlink(missing_ok=True)
    pending = manifest_path.with_name(manifest_path.name + f".{uuid.uuid4().hex[:8]}.partial")
    try:
        pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, manifest_path)
    finally:
        pending.unlink(missing_ok=True)
    return {"zip": archive, "manifest_path": manifest_path, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path,
                        help="Output folder (default: runs/bundles in the repository, which git ignores)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Bundle HEAD although the working tree differs; the manifest then says dirty: true "
                             "and the uncommitted changes are NOT in the bundle")
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository to archive (default: this checkout)")
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else args.repo / "runs" / "bundles"
    try:
        built = build_bundle(args.repo, out_dir, allow_dirty=args.allow_dirty)
    except BundleError as error:
        print(f"Bundle refused: {error}", file=sys.stderr)
        return 2
    manifest = built["manifest"]
    print("Source bundle of the tracked files of HEAD (CPU tool; not executed on a GPU as of 2026-11-18)")
    print(f"  commit:   {manifest['commit']}")
    print(f"  dirty:    {str(manifest['dirty']).lower()}")
    print(f"  files:    {manifest['file_count']} ({manifest['total_bytes']} bytes)")
    print(f"  zip:      {built['zip']}")
    print(f"  manifest: {built['manifest_path']}")
    if manifest["dirty"]:
        print(f"WARNING: {len(manifest['uncommitted_entries_not_in_bundle'])} uncommitted or untracked "
              "entries are NOT in this bundle; it holds the committed files only.")
    print("Paste the next line into the tuning notebook's EXPECTED_CONTENT_SHA256 field:")
    print(manifest["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
