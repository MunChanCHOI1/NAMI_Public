"""Source bundle builder for the Negotiation tuning notebook.

GPU status: NOT executed on a GPU as of 2026-11-18. Everything here runs on CPU against
temporary git repositories; it shows that the bundle and its content hash are built
correctly, not that any model run used them.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import warnings
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "negotiation_colab_quickstart.ipynb"
EXECUTED_SOURCE_COMMIT = "2f57d074c039646dde96ad1afa9e90247b87037a"


def _load():
    spec = importlib.util.spec_from_file_location("_test_make_source_bundle", ROOT / "scripts" / "make_source_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle = _load()


EMPTY_GIT_CONFIG = None


@pytest.fixture(scope="module", autouse=True)
def _empty_git_config(tmp_path_factory):
    global EMPTY_GIT_CONFIG
    EMPTY_GIT_CONFIG = tmp_path_factory.mktemp("gitconfig") / "empty"
    EMPTY_GIT_CONFIG.write_text("", encoding="utf-8")


def git(repo, *arguments):
    """Fixture-side git, isolated from the user's configuration (signing, hooks, identity)."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL=str(EMPTY_GIT_CONFIG), GIT_CONFIG_NOSYSTEM="1")
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Bundle Test", "-c", "user.email=bundle@example.invalid",
         "-c", "commit.gpgsign=false", *arguments],
        capture_output=True, text=True, encoding="utf-8", env=environment)
    assert done.returncode == 0, done.stderr
    return done.stdout


def write(repo, name, data):
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def commit_all(repo, message="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


TRACKED = {
    ".gitignore": "runs/\n.cache/\n*.safetensors\n.env\n",
    "pyproject.toml": "[project]\nname = 'fixture'\n",
    "README.md": "# fixture\n",
    "src/pkg/__init__.py": "",
    "src/pkg/module.py": "VALUE = 1\n",
    "docs/계획.md": "한국어 문서\n",
    "data/blob.dat": bytes(range(256)) * 8,
}


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    folder = tmp_path / "repo"
    folder.mkdir()
    git(folder, "init", "-q")
    for name, data in TRACKED.items():
        write(folder, name, data)
    commit_all(folder, "initial")
    return folder


def independent_content_hash(files):
    """Written out separately from the script: name, NUL, hex digest of the bytes, newline; sorted by name."""
    outer = hashlib.sha256()
    for name in sorted(files):
        outer.update(name.encode("utf-8") + b"\x00" + hashlib.sha256(files[name]).hexdigest().encode("ascii") + b"\n")
    return outer.hexdigest()


def tracked_bytes():
    return {name: data if isinstance(data, bytes) else data.encode("utf-8") for name, data in TRACKED.items()}


def notebook_source_cell():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return next("".join(cell["source"]) for cell in cells
                if cell["cell_type"] == "code" and "content_hash = hashlib.sha256" in "".join(cell["source"]))


def notebook_formula():
    """The `content_hash = ...` expression of the executed notebook's source cell, compiled as written there."""
    tree = ast.parse(notebook_source_cell())
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and [getattr(target, "id", None) for target in node.targets] == ["content_hash"]]
    assert len(found) == 1
    code = compile(ast.Expression(found[0].value), "<notebook source cell>", "eval")
    return lambda contents: eval(code, {"hashlib": hashlib, "contents": contents})


def test_header_says_the_tool_was_not_executed_on_a_gpu():
    assert "NOT been executed on a GPU" in bundle.__doc__ and "2026-11-18" in bundle.__doc__


def test_bundle_holds_exactly_the_tracked_files_and_the_manifest_matches_the_archive(repo, tmp_path):
    commit = git(repo, "rev-parse", "HEAD").strip()
    built = bundle.build_bundle(repo, tmp_path / "out")
    manifest = built["manifest"]
    assert built["zip"].name == f"negotiation-source-{commit[:12]}.zip"
    assert built["manifest_path"].name == f"negotiation-source-{commit[:12]}.manifest.json"
    assert json.loads(built["manifest_path"].read_text(encoding="utf-8")) == manifest
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(
        [built["zip"].name, built["manifest_path"].name])

    with zipfile.ZipFile(built["zip"]) as archive:
        archived = {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
    assert archived == tracked_bytes()
    assert set(archived) == set(git(repo, "ls-files", "-z").split("\0")) - {""}
    assert manifest["files"] == {name: hashlib.sha256(data).hexdigest() for name, data in archived.items()}
    assert list(manifest["files"]) == sorted(manifest["files"])
    assert manifest["commit"] == commit and manifest["dirty"] is False
    assert manifest["file_count"] == len(archived)
    assert manifest["total_bytes"] == sum(len(data) for data in archived.values())
    assert manifest["zip_sha256"] == hashlib.sha256(built["zip"].read_bytes()).hexdigest()
    assert manifest["uncommitted_entries_not_in_bundle"] == []
    assert manifest["input_sha256"] is None            # this fixture has no Negotiation provenance module
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", manifest["created_at_utc"])
    assert {"commit", "dirty", "file_count", "total_bytes", "files", "content_sha256", "zip_sha256",
            "input_sha256", "created_at_utc"} <= set(manifest)


def test_content_hash_equals_an_independent_implementation_and_the_notebook_expression(repo, tmp_path):
    manifest = bundle.build_bundle(repo, tmp_path / "out")["manifest"]
    files = tracked_bytes()
    assert manifest["content_sha256"] == independent_content_hash(files)
    assert manifest["content_sha256"] == notebook_formula()(files)
    assert bundle.content_sha256(files) == manifest["content_sha256"]
    # Any change of a name or of one byte changes the hash.
    assert bundle.content_sha256({**files, "README.md": b"# fixture \n"}) != manifest["content_sha256"]
    renamed = {("README2.md" if name == "README.md" else name): data for name, data in files.items()}
    assert bundle.content_sha256(renamed) != manifest["content_sha256"]


def test_content_hash_is_the_same_for_flat_prefixed_and_repacked_archives(repo, tmp_path):
    commit = git(repo, "rev-parse", "HEAD").strip()
    expected = independent_content_hash(tracked_bytes())
    flat, prefixed = tmp_path / "flat.zip", tmp_path / "prefixed.zip"
    bundle.write_archive(repo, commit, flat)
    bundle.write_archive(repo, commit, prefixed, prefix=f"fixture-{commit}/")
    assert bundle.read_archive(flat)[0] == "" and bundle.read_archive(prefixed)[0] == f"fixture-{commit}/"
    assert bundle.read_archive(flat)[1] == bundle.read_archive(prefixed)[1] == tracked_bytes()

    # Another container: stored instead of deflated, reversed member order, other timestamps, deeper prefix.
    repacked = tmp_path / "repacked.zip"
    with zipfile.ZipFile(repacked, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("outer/inner/", b"")
        for name in sorted(tracked_bytes(), reverse=True):
            archive.writestr(zipfile.ZipInfo("outer/inner/" + name, (2001, 2, 3, 4, 5, 6)), tracked_bytes()[name])
    assert len({hashlib.sha256(path.read_bytes()).hexdigest() for path in (flat, prefixed, repacked)}) == 3
    for path in (flat, prefixed, repacked):
        assert bundle.content_sha256(bundle.read_archive(path)[1]) == expected


def test_dirty_tree_is_refused_and_nothing_is_written(repo, tmp_path):
    out = tmp_path / "out"
    write(repo, "src/pkg/module.py", "VALUE = 2\n")
    with pytest.raises(bundle.BundleError, match="differs from HEAD"):
        bundle.build_bundle(repo, out)
    git(repo, "checkout", "-q", "--", "src/pkg/module.py")

    write(repo, "src/pkg/new_module.py", "NEW = True\n")       # untracked, not ignored: would be missing
    with pytest.raises(bundle.BundleError, match="new_module.py"):
        bundle.build_bundle(repo, out)
    git(repo, "add", "src/pkg/new_module.py")                   # staged but not committed
    with pytest.raises(bundle.BundleError, match="differs from HEAD"):
        bundle.build_bundle(repo, out)
    assert not out.exists()


def test_allow_dirty_bundles_head_only_and_says_so(repo, tmp_path):
    clean = bundle.build_bundle(repo, tmp_path / "clean")["manifest"]
    write(repo, "src/pkg/module.py", "VALUE = 2\n")
    write(repo, "src/pkg/new_module.py", "NEW = True\n")
    built = bundle.build_bundle(repo, tmp_path / "out", allow_dirty=True)
    manifest = built["manifest"]
    assert manifest["dirty"] is True and manifest["commit"] == clean["commit"]
    assert manifest["content_sha256"] == clean["content_sha256"] and manifest["files"] == clean["files"]
    assert sorted(entry.split()[-1] for entry in manifest["uncommitted_entries_not_in_bundle"]) == [
        "src/pkg/module.py", "src/pkg/new_module.py"]
    with zipfile.ZipFile(built["zip"]) as archive:
        assert archive.read("src/pkg/module.py") == b"VALUE = 1\n"
        assert "src/pkg/new_module.py" not in archive.namelist()


def test_ignored_and_untracked_files_never_enter_the_bundle(repo, tmp_path):
    token_like = "hf_" + "A" * 34                               # built here so that no file holds the pattern
    write(repo, "runs/negotiation/result.json", "{}")
    write(repo, ".cache/zoro/model.safetensors", b"\x00" * 64)
    write(repo, "weights.safetensors", b"\x00" * 64)
    write(repo, ".env", f"HF_TOKEN={token_like}\n")
    built = bundle.build_bundle(repo, tmp_path / "out")          # ignored files do not make the tree dirty
    assert built["manifest"]["dirty"] is False
    write(repo, "scratch.txt", "untracked\n")
    dirty = bundle.build_bundle(repo, tmp_path / "dirty", allow_dirty=True)
    for result in (built, dirty):
        with zipfile.ZipFile(result["zip"]) as archive:
            names = {info.filename for info in archive.infolist() if not info.is_dir()}
            payload = b"".join(archive.read(name) for name in names)
        assert names == set(TRACKED) == set(result["manifest"]["files"])
        assert token_like.encode() not in payload
        assert not any(name.startswith(("runs/", ".cache/")) for name in names)


def test_second_pyproject_is_refused(repo, tmp_path):
    write(repo, "negotiation/pyproject.toml", "[project]\nname = 'second'\n")
    commit_all(repo)
    with pytest.raises(bundle.BundleError, match="exactly one pyproject.toml"):
        bundle.build_bundle(repo, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_missing_pyproject_is_refused(repo, tmp_path):
    git(repo, "rm", "-q", "pyproject.toml")
    commit_all(repo)
    with pytest.raises(bundle.BundleError, match="found 0"):
        bundle.build_bundle(repo, tmp_path / "out")


def test_size_limit_is_the_notebook_limit(repo, tmp_path, monkeypatch):
    assert bundle.MAX_TOTAL_BYTES == 32 * 1024**2
    assert "32 * 1024**2" in notebook_source_cell()
    total = sum(len(data) for data in tracked_bytes().values())
    monkeypatch.setattr(bundle, "MAX_TOTAL_BYTES", total)       # exactly at the limit is accepted, as in the cell
    assert bundle.build_bundle(repo, tmp_path / "at-limit")["manifest"]["total_bytes"] == total
    monkeypatch.setattr(bundle, "MAX_TOTAL_BYTES", total - 1)
    with pytest.raises(bundle.BundleError, match="at most"):
        bundle.build_bundle(repo, tmp_path / "over")
    assert list((tmp_path / "over").iterdir()) == []


def test_symbolic_link_is_refused(repo, tmp_path):
    try:
        os.symlink("README.md", repo / "link.md")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    commit_all(repo)
    with pytest.raises(bundle.BundleError, match="link.md"):
        bundle.build_bundle(repo, tmp_path / "out")


@pytest.mark.parametrize("names", [
    ["pyproject.toml", "../escape.py"],
    ["pyproject.toml", "pkg/../../escape.py"],
    # On Windows zipfile itself turns a backslash into "/" when a member is created or read, so such a
    # name cannot reach the check there; the refusal is exercised on POSIX.
    pytest.param(["pyproject.toml", "pkg\\module.py"],
                 marks=pytest.mark.skipif(os.sep == "\\", reason="zipfile normalizes backslashes on Windows")),
    ["pyproject.toml", "C:/module.py"],
    ["pyproject.toml", "module.py", "module.py"],
    ["top/pyproject.toml", "other/module.py"],
    ["top/pyproject.toml", "top//absolute.py"],
])
def test_unsafe_member_names_are_refused(tmp_path, names):
    path = tmp_path / "unsafe.zip"
    with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as archive:
        warnings.simplefilter("ignore")                         # zipfile warns about the duplicate name
        for name in names:
            archive.writestr(zipfile.ZipInfo(name), b"x")
    with pytest.raises(bundle.BundleError):
        bundle.read_archive(path)


def test_symlink_member_is_refused_by_the_reader(tmp_path):
    path = tmp_path / "link.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pyproject.toml", b"x")
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"pyproject.toml")
    with pytest.raises(bundle.BundleError, match="link"):
        bundle.read_archive(path)


def test_tracked_weights_key_files_and_tokens_are_refused_without_printing_the_value(repo, tmp_path):
    token_like = "hf_" + "B" * 34
    for name, data in (("model/head.pt", b"\x00"), ("deploy/server.key", b"k"),
                       ("notes.txt", f"token {token_like}\n".encode())):
        write(repo, name, data)
        git(repo, "add", "-f", name)
        git(repo, "commit", "-q", "-m", "mistake")
        with pytest.raises(bundle.BundleError) as refused:
            bundle.build_bundle(repo, tmp_path / "out")
        assert name in str(refused.value) and token_like not in str(refused.value)
        git(repo, "rm", "-q", name)
        git(repo, "commit", "-q", "-m", "undo")
    assert list((tmp_path / "out").iterdir()) == []
    assert bundle.build_bundle(repo, tmp_path / "out")["manifest"]["dirty"] is False


def test_tracked_file_left_out_by_export_ignore_is_refused(repo, tmp_path):
    write(repo, ".gitattributes", "docs/** export-ignore\n")
    commit_all(repo)
    with pytest.raises(bundle.BundleError, match="not exactly the tracked files"):
        bundle.build_bundle(repo, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_repository_path_must_be_the_top_level(repo, tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(bundle.BundleError, match="top level"):
        bundle.build_bundle(repo / "src", tmp_path / "out")
    with pytest.raises(bundle.BundleError):
        bundle.build_bundle(tmp_path / "plain", tmp_path / "out")   # not a git repository at all


def test_rebuilding_the_same_commit_gives_the_same_content_hash(repo, tmp_path):
    first = bundle.build_bundle(repo, tmp_path / "out")["manifest"]
    second = bundle.build_bundle(repo, tmp_path / "out")["manifest"]
    assert first["content_sha256"] == second["content_sha256"] and first["files"] == second["files"]
    assert len(list((tmp_path / "out").iterdir())) == 2
    write(repo, "README.md", "# changed\n")
    changed = bundle.build_bundle(repo.parent / "repo", tmp_path / "out", allow_dirty=True)["manifest"]
    assert changed["content_sha256"] == first["content_sha256"]  # uncommitted edit: same HEAD, same bundle
    commit_all(repo)
    committed = bundle.build_bundle(repo, tmp_path / "out")["manifest"]
    assert committed["content_sha256"] != first["content_sha256"] and committed["commit"] != first["commit"]
    assert len(list((tmp_path / "out").iterdir())) == 4


def test_input_hashes_are_those_of_the_bundled_source_not_of_the_working_tree(repo, tmp_path):
    provenance_source = (ROOT / bundle.PROVENANCE_MODULE).read_text(encoding="utf-8")
    shared = ast.literal_eval(next(
        node.value for node in ast.parse(provenance_source).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SHARED_INPUTS"))
    write(repo, bundle.PROVENANCE_MODULE, provenance_source)
    write(repo, "negotiation/src/negotiation_agent/other.py", "X = 1\n")
    write(repo, "negotiation/run_fixture.py", "print('fixture')\n")
    for name in shared:
        if name != "pyproject.toml":
            write(repo, name, f"# stand-in for {name}\n")
    commit_all(repo)
    committed = {name: (repo / name).read_bytes() for name in (
        bundle.PROVENANCE_MODULE, "negotiation/src/negotiation_agent/other.py", "negotiation/run_fixture.py", *shared)}

    # Uncommitted edits and an untracked module must not leak into the recorded input hashes.
    write(repo, "negotiation/src/negotiation_agent/other.py", "X = 2\n")
    write(repo, "negotiation/src/negotiation_agent/untracked.py", "Y = 1\n")
    manifest = bundle.build_bundle(repo, tmp_path / "out", allow_dirty=True)["manifest"]
    assert manifest["input_sha256"] == {name: hashlib.sha256(data).hexdigest() for name, data in committed.items()}
    assert list(manifest["input_sha256"]) == sorted(manifest["input_sha256"])
    assert all(manifest["files"][name] == digest for name, digest in manifest["input_sha256"].items())


def test_broken_bundled_provenance_module_is_refused_not_skipped(repo, tmp_path):
    write(repo, bundle.PROVENANCE_MODULE, "def input_hashes():\n    raise RuntimeError('broken')\n")
    commit_all(repo)
    with pytest.raises(bundle.BundleError, match="could not hash its inputs"):
        bundle.build_bundle(repo, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_command_line_prints_the_bare_content_hash_on_the_last_line(repo, tmp_path, capsys):
    out = tmp_path / "out"
    assert bundle.main(["--repo", str(repo), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    manifest = json.loads(next(out.glob("*.manifest.json")).read_text(encoding="utf-8"))
    assert lines[-1] == manifest["content_sha256"] == independent_content_hash(tracked_bytes())
    assert re.fullmatch(r"[0-9a-f]{64}", lines[-1])
    assert lines[-2] == "Paste the next line into the tuning notebook's EXPECTED_CONTENT_SHA256 field:"
    assert f"  commit:   {manifest['commit']}" in lines and "  dirty:    false" in lines
    assert not any(line.startswith("WARNING") for line in lines)

    write(repo, "README.md", "# edited\n")
    assert bundle.main(["--repo", str(repo), "--out", str(tmp_path / "refused")]) == 2
    refused = capsys.readouterr()
    assert refused.out == "" and "Bundle refused" in refused.err and "--allow-dirty" in refused.err
    assert not (tmp_path / "refused").exists()

    assert bundle.main(["--repo", str(repo), "--out", str(out), "--allow-dirty"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == manifest["content_sha256"] and "  dirty:    true" in lines
    assert any(line.startswith("WARNING") and "NOT in this bundle" in line for line in lines)


def test_default_output_folder_is_runs_bundles_inside_the_repository(repo, capsys):
    assert bundle.main(["--repo", str(repo)]) == 0               # the fixture ignores runs/, as the project does
    capsys.readouterr()
    assert len(list((repo / "runs" / "bundles").glob("negotiation-source-*.zip"))) == 1
    assert git(repo, "status", "--porcelain") == ""


def _repository_has(commit):
    """True when this checkout is itself the top of a git working tree that holds the commit."""
    try:
        top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, encoding="utf-8")
        if top.returncode or Path(top.stdout.strip()).resolve() != ROOT:
            return False
        done = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", commit + "^{commit}"], capture_output=True)
    except FileNotFoundError:
        return False
    return done.returncode == 0


def test_formula_reproduces_the_hash_pinned_in_the_executed_notebook(tmp_path):
    """The executed notebook pins the content hash of its source commit; `git archive` of it must give that hash."""
    if not _repository_has(EXECUTED_SOURCE_COMMIT):
        pytest.skip("the executed notebook's source commit is not in this checkout (for example a shallow clone)")
    cell = notebook_source_cell()
    assert f"NEG_SOURCE_COMMIT = '{EXECUTED_SOURCE_COMMIT}'" in cell
    pinned = re.search(r"NEG_CONTENT_HASH = '([0-9a-f]{64})'", cell).group(1)
    for prefix in ("", f"sessionguard-{EXECUTED_SOURCE_COMMIT}/"):
        archive = tmp_path / f"executed-{len(prefix)}.zip"
        bundle.write_archive(ROOT, EXECUTED_SOURCE_COMMIT, archive, prefix=prefix)
        assert bundle.content_sha256(bundle.read_archive(archive)[1]) == pinned


def test_this_checkout_at_head_passes_the_notebook_limits(tmp_path):
    if not _repository_has("HEAD"):
        pytest.skip("this checkout is not a git repository")
    built = bundle.build_bundle(ROOT, tmp_path / "out", allow_dirty=True)   # HEAD only; the tree is not touched
    manifest = built["manifest"]
    assert manifest["content_sha256"] == notebook_formula()(bundle.read_archive(built["zip"])[1])
    assert manifest["total_bytes"] <= 32 * 1024**2 and "pyproject.toml" in manifest["files"]
    assert not any(name.startswith(("runs/", ".cache/", ".venv/")) for name in manifest["files"])
    if manifest["input_sha256"] is not None:
        assert all(manifest["files"][name] == digest for name, digest in manifest["input_sha256"].items())
