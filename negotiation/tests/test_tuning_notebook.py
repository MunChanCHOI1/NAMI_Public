"""CPU tests of notebooks/negotiation_tuning_colab.ipynb.

NOT executed on a GPU as of 2026-11-18. The notebook's cells are executed here with a fake
`google.colab`, `/content` mapped into a temporary folder, and the CPU keyword reference backend
(or a test double built on it) in the model's place, on shortened rehearsal blocks. This checks
the control flow of the cells against the real command line of `negotiation/run_tune.py` and
`scripts/make_source_bundle.py`. No number in this file is model evidence, and passing these
tests does not show that the notebook works on Colab or on any GPU.
"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from negotiation_agent import tune
from negotiation_agent.backends import KeywordBackend

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "negotiation_tuning_colab.ipynb"
HEADER = "GPU 미실행 (2026-11-18 기준, CPU 검증만)"
REHEARSAL = "d0=3,select=6,confirm=8,cal=20"
SPLIT_LITERAL = "ev" + "al"            # spelled in two parts so this file can be scanned as well
# Joined at run time: this file is part of the bundle the tests unpack, and the joined value must not be in it.
SECRET = "-".join(["synthetic", "auth", "secret", "for", "the", "notebook", "test"])
IDENTITY = {"backend": "tuning_notebook_test_double", "note": "test double on the keyword reference; not a model"}
FORM_DEFAULTS = {
    1: {"EXPECTED_CONTENT_SHA256": ""},
    7: {"BUDGET_MINUTES": 45},
    8: {"RUN_UTTERANCE_PROBE": False},
    9: {"ACCEPT_FAILED_CANDIDATE": False},
    10: {"RUN_CONFIRM": False, "CONFIRM_BUDGET_MINUTES": 60},
    11: {"DOWNLOAD_LOGS": False},
}


def _load_bundle_module():
    spec = importlib.util.spec_from_file_location("_tuning_notebook_bundle", ROOT / "scripts" / "make_source_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle = _load_bundle_module()


# ---- reading the notebook ----------------------------------------------------------------------------------

def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources(kind):
    return ["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == kind]


def cell(number):
    """The code cell whose Colab title starts with this number."""
    found = [text for text in sources("code") if text.startswith(f"#@title {number}. ")]
    assert len(found) == 1, f"expected exactly one code cell titled {number}."
    return found[0]


def execute(number, namespace, **form):
    """Run one code cell as it is, except for its form values and the pathlib import (the test maps /content)."""
    tree = ast.parse(cell(number))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            continue
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in form):
            node.value = ast.Constant(form.pop(node.targets[0].id))
        body.append(node)
    assert not form, f"cell {number} has no form field named {sorted(form)}"
    tree.body = body
    ast.fix_missing_locations(tree)
    exec(compile(tree, f"tuning_notebook_cell_{number}", "exec"), namespace)


# ---- a fake Colab runtime ----------------------------------------------------------------------------------

class Colab:
    """`/content` inside a temporary folder, with the upload/download/display calls recorded."""

    def __init__(self, root, monkeypatch):
        self.root = Path(root)
        self.content = self.root / "content"
        self.content.mkdir(parents=True)
        self.uploads, self.upload_calls, self.downloads, self.displayed = None, 0, [], []
        monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
            files=SimpleNamespace(upload=self._upload, download=self.downloads.append),
            userdata=SimpleNamespace(get=lambda name: None)))
        monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(
            Markdown=lambda text: text, display=self.displayed.append))

    def _upload(self):
        self.upload_calls += 1
        assert self.uploads is not None, "the upload chooser must not open in this test"
        return dict(self.uploads)

    def path(self, value):
        value = str(value)
        if value == "/content" or value.startswith("/content/"):
            return self.root / value.lstrip("/")
        return Path(value)

    def namespace(self):
        return {"Path": self.path, "PurePosixPath": PurePosixPath}

    def source_folders(self):
        return sorted(self.content.glob("negotiation-tuning-src-*"))


@pytest.fixture
def colab(tmp_path, monkeypatch):
    return Colab(tmp_path / "runtime", monkeypatch)


def _git(repo, *arguments, config):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM="1")
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Notebook Test", "-c", "user.email=notebook@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *arguments],
        capture_output=True, env=environment)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    return done.stdout


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory):
    """A bundle of THIS working tree, built by scripts/make_source_bundle.py.

    The builder archives committed files only, so the tracked and the not-ignored untracked files of
    the working tree are copied into a temporary repository and committed there. The real repository
    is only read.
    """
    listing = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True)
    if listing.returncode != 0:
        pytest.skip("this checkout is not a git repository, so its file list cannot be bundled")
    folder = tmp_path_factory.mktemp("tuning-notebook-bundle")
    repo, config = folder / "repo", folder / "empty-gitconfig"
    config.write_text("", encoding="utf-8")
    repo.mkdir()
    for name in listing.stdout.decode("utf-8").split("\0"):
        if name and (ROOT / name).is_file():
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    _git(repo, "init", "-q", config=config)
    _git(repo, "add", "-A", config=config)
    _git(repo, "commit", "-q", "-m", "snapshot of the working tree for the notebook test", config=config)
    built = bundle.build_bundle(repo, folder / "out")
    return SimpleNamespace(zip=built["zip"], name=built["zip"].name, data=built["zip"].read_bytes(),
                           content_sha256=built["manifest"]["content_sha256"], manifest=built["manifest"],
                           repo=repo)


def prepared(colab, source_bundle, capsys):
    """Cell 1 on a runtime whose file panel holds the bundle. Returns the notebook namespace."""
    (colab.content / source_bundle.name).write_bytes(source_bundle.data)
    namespace = colab.namespace()
    execute(1, namespace, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    capsys.readouterr()
    return namespace


class Rehearsal:
    """The notebook's helper redirected to the CPU keyword backend and shortened blocks.

    It stands where cells 2 to 4 (install, login, model) leave the notebook: NEG_ARGS and NEG_READY
    are set, with `--backend keyword` instead of the model. Every command still goes through the
    notebook's own `neg_run` into a real subprocess of the unpacked bundle.
    """

    def __init__(self, namespace):
        self.real, self.calls, self.live_limit = namespace["neg_run"], [], None
        namespace.update(neg_run=self, NEG_ARGS=["--backend", "keyword"], NEG_READY=True)

    def __call__(self, *args, **kwargs):
        args = [str(a) for a in args]
        if args[:2] == ["negotiation/run_tune.py", "plan"]:
            args += ["--rehearsal-blocks", REHEARSAL]
        if self.live_limit is not None and args[:2] == ["negotiation/run_tune.py", "run"]:
            args += ["--max-live-requests", str(self.live_limit)]
        self.calls.append(args)
        return self.real(*args, **kwargs)

    def commands(self):
        """(script, command or stage) of every call so far; the list is emptied."""
        out = []
        for args in self.calls:
            if args[0] == "negotiation/run_tune.py":
                out.append(args[args.index("--stage") + 1] if args[1] == "run" else args[1])
            else:
                out.append(args[0])
        self.calls.clear()
        return out


class ModelDouble:
    """KeywordBackend values behind a declared-model test double. It is NOT a model."""

    kind, is_model, supports_final_offer = "tuning_notebook_test_double_model", True, True

    def __init__(self, fail_at=()):
        self.calls, self.fail_at, self.last_response = 0, set(fail_at), None
        self._base = KeywordBackend()

    def predict(self, state, candidates, request):
        self.last_response = None
        self.calls += 1
        if self.calls in self.fail_at:
            raise RuntimeError("test backend failure")
        out = self._base.predict(state, candidates, request)
        self.last_response = {"metadata": {"dtype": "test_only"}}
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": True, "test_double": True}


def tune_in_process(argv, double):
    return tune.main(argv, backend_factory=lambda: double, declared=(ModelDouble.kind, True, True),
                     real_model=False, store_requested=IDENTITY)


class InProcess:
    """The notebook's helper replaced by the runner in this process, so a failing double can be injected."""

    def __init__(self, namespace, double):
        self.double, self.calls = double, []
        namespace.update(neg_run=self, NEG_ARGS=["--backend", "keyword"], NEG_READY=True)

    def __call__(self, *args, check=True):
        args = [str(a) for a in args]
        assert args[0] == "negotiation/run_tune.py"
        if args[1] == "plan":
            args += ["--rehearsal-blocks", REHEARSAL]
        self.calls.append(args)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tune_in_process(args[1:], self.double)
        lines = (err.getvalue() + out.getvalue()).splitlines()
        print("\n".join(lines))
        if code and check:
            raise RuntimeError(f"exit={code}")
        return {"exit_code": code, "lines": lines, "log": "in-process"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- the file itself ---------------------------------------------------------------------------------------

def test_notebook_is_nbformat_4_without_outputs_or_execution_counts():
    raw = NOTEBOOK.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    document = json.loads(raw.decode("utf-8"))
    assert set(document) == {"cells", "metadata", "nbformat", "nbformat_minor"}
    assert document["nbformat"] == 4 and isinstance(document["nbformat_minor"], int)
    assert document["metadata"]["kernelspec"]["name"] == "python3" and document["metadata"]["accelerator"] == "GPU"
    identifiers = []
    for item in document["cells"]:
        if item["cell_type"] == "code":
            assert set(item) == {"cell_type", "execution_count", "metadata", "outputs", "source"}
            assert item["outputs"] == [] and item["execution_count"] is None
            assert item["metadata"].get("cellView") == "form"
        else:
            assert item["cell_type"] == "markdown" and set(item) == {"cell_type", "metadata", "source"}
        lines = item["source"]
        assert isinstance(lines, list) and lines and all(isinstance(line, str) for line in lines)
        assert all(line.endswith("\n") and line.count("\n") == 1 for line in lines[:-1])
        identifiers.append(item["metadata"]["id"])
    assert len(set(identifiers)) == len(identifiers)


def test_first_cell_says_not_executed_on_a_gpu_and_that_minutes_are_estimates():
    first = notebook()["cells"][0]
    text = "".join(first["source"])
    assert first["cell_type"] == "markdown" and HEADER in text
    assert "추정치" in text and "측정값이 아닙니다" in text
    assert "평가 분할은 이 노트에서 사용하지 않으며" in text
    assert "NOT executed on a GPU as of 2026-11-18" in cell(1)          # the first code cell says it as well


def test_code_cells_are_plain_python_numbered_in_order_with_safe_form_defaults():
    titles = []
    for text in sources("code"):
        tree = ast.parse(text)                      # no shell or line magics: every cell is plain Python
        compile(tree, "tuning_notebook", "exec")
        titles.append(int(re.match(r"#@title (\d+)\. ", text).group(1)))
        fields = {}
        for line in text.splitlines():
            found = re.match(r"(\w+) = (.+?)\s+#@param \{type:\"(string|number|boolean)\"\}$", line)
            if found:
                fields[found.group(1)] = ast.literal_eval(found.group(2))
        assert fields == FORM_DEFAULTS.get(titles[-1], {})
        assert text.count("#@param") == len(fields)
    assert titles == list(range(1, 12))
    # The confirm block and the optional probe do nothing unless a person ticks the box.
    assert FORM_DEFAULTS[10]["RUN_CONFIRM"] is False and FORM_DEFAULTS[8]["RUN_UTTERANCE_PROBE"] is False


def test_notebook_never_names_the_held_out_split_a_token_or_the_other_serving_paths():
    raw = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(rf"\b{SPLIT_LITERAL}\b", raw)
    for forbidden in ("--split", "heldout", "holdout", "--freeze-manifest", "--evaluation-mode", "evidence/",
                      "official_server", "--reasoning", "zoro-http", "systemone", "sglang", "SGLang 0.",
                      "2f57d07", "NEG_CONTENT_HASH", "--rehearsal-blocks"):
        assert forbidden not in raw, forbidden
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", raw) and not bundle.TOKEN_PATTERN.search(raw.encode("utf-8"))
    for text in sources("code"):
        constants = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not any("run_evaluation" in value for value in constants)       # no cell runs the evaluator
        for value in constants:
            if value.endswith(".py"):
                assert (ROOT / value).is_file(), value


def test_no_cell_prints_or_stores_the_environment_or_the_token():
    environment_uses = 0
    for text in sources("code"):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                environment_uses += 1
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in ("print", "dumps", "dump", "display", "write", "write_text"):
                    inside = {n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", None)
                              for argument in [*node.args, *[k.value for k in node.keywords]]
                              for n in ast.walk(argument)}
                    assert not inside & {"environ", "env", "NEG_ENV", "neg_token", "getenv"}, ast.unparse(node)
    # The subprocess environment in cell 1, and reading / setting HF_TOKEN in cell 3: nowhere else.
    assert environment_uses == 3
    assert cell(1).count("os.environ") == 1 and cell(3).count("os.environ") == 2


def _powershell_commands(text):
    blocks = re.findall(r"```powershell\n(.*?)```", text, flags=re.S)
    assert blocks
    joined = "\n".join(blocks)
    zoro = re.findall(r"'([^']*)'", re.search(r"\$zoro = @\((.*?)\)\n", joined, flags=re.S).group(1))
    state = re.search(r"\$state = '([^']*)'", joined)
    commands = []
    for line in joined.splitlines():
        if line.startswith("& $gpuPython -X utf8 "):
            tokens = []
            for token in line[len("& $gpuPython -X utf8 "):].split():
                tokens += zoro if token == "@zoro" else [state.group(1) if token == "$state" else token]
            commands.append(tokens)
    return zoro, commands


def test_powershell_commands_use_the_plan_document_flags_and_parse_with_the_real_runner(tmp_path):
    zoro, commands = _powershell_commands(sources("markdown")[-1])
    plan_zoro, _ = _powershell_commands((ROOT / "negotiation" / "docs" / "PLAN.md").read_text(encoding="utf-8"))
    assert zoro == plan_zoro
    pins = read(ROOT / "config" / "model_pins.json")
    assert Path(zoro[zoro.index("--original-checkpoint") + 1]).name == pins["model_revision"]
    seen = []
    for tokens in commands:
        assert (ROOT / tokens[0]).is_file()
        if tokens[0] != "negotiation/run_tune.py":
            assert tokens[0] == "negotiation/run_utterance_probe.py" and "--output" in tokens and "--backend" in tokens
            continue
        args = tune._parser().parse_args(tokens[1:])
        seen.append(args.stage if args.command == "run" else args.command)
        if args.command == "run":
            assert args.backend == "zoro" and args.decision_temperature == 1.0
            tune.Tuner(tmp_path / "unused", args).check_run_arguments()       # refused flags would raise here
    assert seen == ["plan", "smoke", "all", "board", "confirm", "pack", "restore", "board"]


# ---- cell 1: source and state ------------------------------------------------------------------------------

def test_source_cell_hash_formula_is_the_bundle_builders(colab, source_bundle, capsys):
    namespace = prepared(colab, source_bundle, capsys)
    digest, contents, comment = namespace["neg_source_contents"](source_bundle.data)
    independent = hashlib.sha256("".join(
        f"{name}\0{hashlib.sha256(contents[name]).hexdigest()}\n" for name in sorted(contents)).encode()).hexdigest()
    assert digest == source_bundle.content_sha256 == independent == bundle.content_sha256(contents)
    assert {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()} == source_bundle.manifest["files"]
    assert comment == source_bundle.manifest["commit"]        # git archive writes the commit id; informational
    # The same source under a top-level folder (a GitHub-style archive) has the same content hash.
    prefixed = colab.root / "prefixed.zip"
    bundle.write_archive(source_bundle.repo, source_bundle.manifest["commit"], prefixed, prefix="project-main/")
    assert namespace["neg_source_contents"](prefixed.read_bytes())[0] == source_bundle.content_sha256


def test_source_cell_accepts_the_panel_zip_and_keeps_state_outside_the_source(colab, source_bundle, capsys):
    (colab.content / source_bundle.name).write_bytes(source_bundle.data)
    namespace = colab.namespace()
    execute(1, namespace, EXPECTED_CONTENT_SHA256="  " + source_bundle.content_sha256.upper() + "\n")
    out = capsys.readouterr().out
    assert colab.upload_calls == 0                      # a ZIP in the file panel never opens the chooser
    repo, state, logs = namespace["NEG_REPO"], namespace["NEG_STATE"], namespace["NEG_LOGS"]
    assert colab.source_folders() == [repo]
    files = {p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()}
    assert files == set(source_bundle.manifest["files"])
    assert (repo / "negotiation" / "run_tune.py").read_bytes() == (ROOT / "negotiation" / "run_tune.py").read_bytes()
    assert state == colab.content / "negotiation-tuning-state" and not state.exists()
    assert repo not in state.parents and state not in repo.parents and repo not in logs.parents
    assert namespace["NEG_STATE_READY"] is True and namespace["NEG_READY"] is False and "NEG_ARGS" not in namespace
    record = read(logs / "source.json")
    assert record["content_sha256"] == record["expected_content_sha256"] == source_bundle.content_sha256
    assert record["zip_sha256"] == hashlib.sha256(source_bundle.data).hexdigest() and "contents" not in record
    assert "새 상태로 시작합니다" in out and source_bundle.content_sha256[:12] in out
    # A second run of the cell unpacks into another fresh folder and leaves the first one alone.
    execute(1, namespace, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    assert len(colab.source_folders()) == 2 and namespace["NEG_REPO"] != repo and repo.is_dir()


def test_source_cell_takes_one_uploaded_zip_when_the_panel_is_empty(colab, source_bundle, capsys):
    colab.uploads = {source_bundle.name: source_bundle.data}
    namespace = colab.namespace()
    execute(1, namespace, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    assert colab.upload_calls == 1 and (namespace["NEG_REPO"] / "pyproject.toml").is_file()
    colab.uploads = {"a.zip": source_bundle.data, "b.zip": source_bundle.data}
    with pytest.raises(ValueError, match="소스 ZIP 한 개"):
        execute(1, colab.namespace(), EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    assert len(colab.source_folders()) == 1


@pytest.mark.parametrize("expected", ["", "   ", "not-a-hash", "abc123", "g" * 64])
def test_source_cell_stops_without_a_usable_expected_hash(colab, source_bundle, expected):
    (colab.content / source_bundle.name).write_bytes(source_bundle.data)
    namespace = colab.namespace()
    with pytest.raises(ValueError, match="EXPECTED_CONTENT_SHA256"):
        execute(1, namespace, **({} if expected == "" else {"EXPECTED_CONTENT_SHA256": expected}))
    # Nothing was unpacked, nothing was asked for, and later cells stay blocked.
    assert colab.upload_calls == 0 and colab.source_folders() == []
    assert namespace["NEG_STATE_READY"] is False and "neg_run" not in namespace
    assert [p.name for p in colab.content.iterdir()] == [source_bundle.name]


def _rewritten(data, change):
    """The same archive with `change(name, bytes)` applied to every member (None drops it)."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as original, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as copy:
        for member in original.infolist():
            if member.is_dir():
                continue
            for name, content in change(member.filename, original.read(member)):
                copy.writestr(name, content)
    return out.getvalue()


@pytest.mark.parametrize("kind", ["one_byte_changed", "file_added", "file_removed", "second_project", "not_a_zip",
                                  "other_hash"])
def test_source_cell_rejects_an_archive_that_is_not_the_expected_source(colab, source_bundle, capsys, kind):
    expected = source_bundle.content_sha256
    if kind == "one_byte_changed":
        data = _rewritten(source_bundle.data, lambda n, d: [(n, d + b"\n# changed\n" if n == "negotiation/run_tune.py" else d)])
    elif kind == "file_added":
        data = _rewritten(source_bundle.data, lambda n, d: [(n, d)] + ([("extra.py", b"x = 1\n")] if n == "pyproject.toml" else []))
    elif kind == "file_removed":
        data = _rewritten(source_bundle.data, lambda n, d: [] if n == "negotiation/run_tune.py" else [(n, d)])
    elif kind == "second_project":
        data = _rewritten(source_bundle.data, lambda n, d: [(n, d)] + ([("copy/pyproject.toml", d)] if n == "pyproject.toml" else []))
    elif kind == "not_a_zip":
        data = b"this is not an archive"
    else:
        data, expected = source_bundle.data, hashlib.sha256(b"another source").hexdigest()
    (colab.content / source_bundle.name).write_bytes(data)
    namespace = colab.namespace()
    with pytest.raises(ValueError, match="내용이 같은 소스 ZIP이") as refusal:
        execute(1, namespace, EXPECTED_CONTENT_SHA256=expected)
    assert source_bundle.name in str(refusal.value)
    assert colab.source_folders() == [] and namespace["NEG_STATE_READY"] is False and "NEG_REPO" not in namespace
    assert not (colab.content / "negotiation-tuning-state").exists()
    # The refusal shows a short prefix only; the full hash of a wrong archive is never offered for pasting.
    actual = None
    if kind not in ("second_project", "not_a_zip"):
        actual = bundle.content_sha256(bundle.read_archive(colab.content / source_bundle.name)[1])
    assert actual is None or (actual not in str(refusal.value) and actual not in capsys.readouterr().out)


def _archive(members):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive, warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # a duplicate member name is one of the cases
        for name, data in members:                     # a name or a prepared ZipInfo
            archive.writestr(name, data)
    return out.getvalue()


def _link(name):
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


PROJECT = ("pyproject.toml", b"[project]\nname = 'fixture'\n")
LIMIT = 32 * 1024**2


@pytest.mark.parametrize("kind,members", [
    ("too_large", [PROJECT, ("big.bin", bytes(LIMIT))]),
    ("symbolic_link", [PROJECT, (_link("link"), b"pyproject.toml")]),
    ("parent_folder", [PROJECT, ("../outside.py", b"x = 1\n")]),
    ("drive_or_stream", [PROJECT, ("c:outside.py", b"x = 1\n")]),
    ("duplicate", [PROJECT, ("a.py", b"x = 1\n"), ("a.py", b"x = 2\n")]),
    ("outside_the_project_folder", [("project/pyproject.toml", PROJECT[1]), ("other/a.py", b"x = 1\n")]),
    ("no_project_file", [("a.py", b"x = 1\n")]),
    ("two_project_files", [PROJECT, ("copy/pyproject.toml", PROJECT[1])]),
])
def test_source_cell_applies_the_same_limits_as_the_bundle_builder(colab, source_bundle, capsys, tmp_path, kind, members):
    namespace = prepared(colab, source_bundle, capsys)
    data = _archive(members)
    (tmp_path / "candidate.zip").write_bytes(data)
    with pytest.raises(bundle.BundleError):
        bundle.read_archive(tmp_path / "candidate.zip")
    with pytest.raises(ValueError):
        namespace["neg_source_contents"](data)
    # In the cell such an archive is never unpacked, whatever hash is pasted.
    with pytest.raises(ValueError, match="candidate.zip: 읽을 수 없음"):
        namespace["neg_pick_source"]([("candidate.zip", data)], "0" * 64)


def test_source_cell_accepts_content_of_exactly_the_size_limit(colab, source_bundle, capsys, tmp_path):
    namespace = prepared(colab, source_bundle, capsys)
    data = _archive([PROJECT, ("big.bin", bytes(LIMIT - len(PROJECT[1])))])
    (tmp_path / "candidate.zip").write_bytes(data)
    digest, contents, _ = namespace["neg_source_contents"](data)
    assert sum(map(len, contents.values())) == LIMIT == bundle.MAX_TOTAL_BYTES
    assert digest == bundle.content_sha256(bundle.read_archive(tmp_path / "candidate.zip")[1])


def test_source_cell_picks_the_archive_that_matches_among_several(colab, source_bundle, capsys):
    changed = _rewritten(source_bundle.data, lambda n, d: [(n, d + b" " if n == "pyproject.toml" else d)])
    (colab.content / "negotiation-source-000000000000.zip").write_bytes(changed)
    (colab.content / "negotiation-source-zzzzzzzzzzzz.zip").write_bytes(source_bundle.data)
    (colab.content / "negotiation-source-folder.zip").mkdir()             # a folder with a matching name is ignored
    namespace = colab.namespace()
    execute(1, namespace, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    assert colab.upload_calls == 0
    assert read(namespace["NEG_LOGS"] / "source.json")["archive"] == "negotiation-source-zzzzzzzzzzzz.zip"
    assert (namespace["NEG_REPO"] / "pyproject.toml").read_bytes() == (source_bundle.repo / "pyproject.toml").read_bytes()


# ---- cells 2 to 4: install, login, model (stubs; nothing is installed or downloaded) -------------------------

def test_install_cell_clears_readiness_and_runs_the_setup_script(tmp_path):
    calls = []
    namespace = {"NEG_READY": True, "NEG_LOGS": tmp_path / "logs", "neg_run": lambda *args: calls.append(args)}
    execute(2, namespace)
    assert namespace["NEG_READY"] is False
    assert calls[0] == ("scripts/gpu_setup.py", "--steps", "env,install", "--run-dir", tmp_path / "logs" / "setup-install")
    assert calls[1] == ("-m", "pip", "install", "--quiet", "huggingface_hub==1.33.0") and len(calls) == 2


@pytest.mark.parametrize("auth_source", ["environment", "colab", "cached", "browser"])
def test_auth_cell_prints_one_fixed_sentence_and_never_the_token(monkeypatch, capsys, auth_source):
    environment = {"HF_TOKEN": SECRET} if auth_source == "environment" else {}
    cached = SECRET if auth_source == "cached" else None
    logins = []

    def login(**kwargs):
        nonlocal cached
        logins.append(kwargs)
        cached = SECRET

    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        userdata=SimpleNamespace(get=lambda _: SECRET if auth_source == "colab" else None)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: cached, login=login))
    namespace = {"os": SimpleNamespace(environ=environment), "NEG_READY": True}
    execute(3, namespace)
    assert capsys.readouterr().out == "인증 준비 완료. 모델 접근 권한은 다음 다운로드에서 확인됩니다.\n"
    assert logins == ([{"add_to_git_credential": False}] if auth_source == "browser" else [])
    assert environment == ({"HF_TOKEN": SECRET} if auth_source in ("environment", "colab") else {})
    assert namespace["neg_token"] is None and namespace["NEG_READY"] is False
    assert SECRET not in json.dumps({k: repr(v) for k, v in namespace.items() if k not in ("os", "__builtins__")})


def test_auth_cell_stops_when_the_web_login_was_not_completed(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(userdata=SimpleNamespace(get=lambda _: None)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: None, login=lambda **_: None))
    namespace = {"os": SimpleNamespace(environ={}), "NEG_READY": True}
    with pytest.raises(RuntimeError, match="웹 로그인"):
        execute(3, namespace)
    assert namespace["NEG_READY"] is False and capsys.readouterr().out == ""


def _model_cell(colab, tmp_path, complete=True):
    repo, logs, calls = tmp_path / "repo", tmp_path / "logs", []
    paths = {"original_checkpoint": tmp_path / "hf" / "snapshot", "exported_checkpoint": tmp_path / "export",
             "official_source_dir": repo / ".cache" / "trida-stack"}

    def neg_run(*args):
        calls.append(args)
        for key, folder in paths.items():
            if complete or key != "exported_checkpoint":
                folder.mkdir(parents=True, exist_ok=True)
        (repo / ".cache" / "gpu_paths.json").write_text(json.dumps({k: str(v) for k, v in paths.items()}), encoding="utf-8")

    namespace = {**colab.namespace(), "json": json, "NEG_REPO": repo, "NEG_LOGS": logs, "NEG_READY": True, "neg_run": neg_run}
    return namespace, paths, calls


def test_model_cell_derives_the_checkpoint_arguments_from_the_setup_script(colab, tmp_path):
    namespace, paths, calls = _model_cell(colab, tmp_path)
    execute(4, namespace)
    assert calls == [("scripts/gpu_setup.py", "--steps", "sources,model,export", "--dtype", "fp16",
                      "--work-dir", colab.content / "negotiation-model-cache", "--run-dir", tmp_path / "logs" / "setup-model")]
    assert namespace["NEG_ARGS"] == [
        "--backend", "zoro", "--original-checkpoint", str(paths["original_checkpoint"]),
        "--exported-checkpoint", str(paths["exported_checkpoint"]),
        "--official-source-dir", str(paths["official_source_dir"]),
        "--device", "cuda:0", "--dtype", "fp16", "--max-input-tokens", "2048", "--decision-temperature", "1"]
    assert namespace["NEG_READY"] is True
    # The real runner accepts exactly these flags, and its store identity holds the settings but no location.
    args = tune._parser().parse_args(["run", "--stage", "smoke", "--state-dir", str(tmp_path / "state"), *namespace["NEG_ARGS"]])
    tuner = tune.Tuner(tmp_path / "state", args)
    tuner.check_run_arguments()
    identity = tuner.requested_identity()
    assert (identity["backend"], identity["dtype"], identity["max_input_tokens"], identity["decision_temperature"]) == \
        ("zoro", "fp16", 2048, 1.0)
    assert str(tmp_path) not in json.dumps(identity)
    # An existing model cache of the earlier notebook is reused when it is there.
    (colab.content / "sessionguard" / ".cache").mkdir(parents=True)
    namespace, _, calls = _model_cell(colab, tmp_path / "again")
    execute(4, namespace)
    assert calls[0][calls[0].index("--work-dir") + 1] == colab.content / "sessionguard" / ".cache"


def test_model_cell_stays_not_ready_when_a_prepared_folder_is_missing(colab, tmp_path):
    namespace, _, _ = _model_cell(colab, tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="exported_checkpoint"):
        execute(4, namespace)
    assert namespace["NEG_READY"] is False and "NEG_ARGS" not in namespace


# ---- cells 5 to 11 on CPU: keyword backend, shortened blocks, real subprocesses ------------------------------

def test_cells_refuse_to_run_before_their_prerequisites(colab, source_bundle, capsys):
    namespace = prepared(colab, source_bundle, capsys)
    state = namespace["NEG_STATE"]
    calls = []
    real = namespace["neg_run"]
    namespace["neg_run"] = lambda *args, **kwargs: (calls.append(args), real(*args, **kwargs))[1]
    for number, form in ((6, {}), (7, {}), (8, {"RUN_UTTERANCE_PROBE": True}), (10, {"RUN_CONFIRM": True})):
        with pytest.raises(RuntimeError, match="모델 준비"):          # cells 2 to 4 have not run
            execute(number, namespace, **form)
    assert calls == [] and not state.exists()
    with pytest.raises(RuntimeError, match="내려받을 상태가 없습니다"):
        execute(11, namespace)
    assert colab.downloads == [] and calls == []
    # With a "model" but no plan and no selection: the optional probe and the confirm block still do nothing.
    rehearsal = Rehearsal(namespace)
    with pytest.raises(RuntimeError, match="5번"):
        execute(8, namespace, RUN_UTTERANCE_PROBE=True)
    with pytest.raises(RuntimeError, match="selection.json"):
        execute(10, namespace, RUN_CONFIRM=True)
    assert rehearsal.calls == [] and not state.exists()
    # Cell 1 must have finished: a namespace in which it stopped half-way runs no runner command.
    namespace["NEG_STATE_READY"] = False
    for number in (5, 6, 9, 11):
        with pytest.raises(RuntimeError, match="1번 셀"):
            execute(number, namespace)
    assert rehearsal.calls == [] and not state.exists()
    assert not re.search(rf"\b{SPLIT_LITERAL}\b", capsys.readouterr().out)


def test_run_cells_plan_smoke_all_board_confirm_and_pack_on_cpu(colab, source_bundle, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", SECRET)             # must reach the subprocess environment and nothing else
    namespace = prepared(colab, source_bundle, capsys)
    rehearsal = Rehearsal(namespace)
    state, packs, repo = namespace["NEG_STATE"], namespace["NEG_PACKS"], namespace["NEG_REPO"]
    printed = []

    def run(number, **form):
        execute(number, namespace, **form)
        printed.append(capsys.readouterr().out)
        return printed[-1]

    # 5. plan: the protocol is written before anything else exists, and no model command ran.
    out = run(5)
    protocol = read(state / "protocol.json")
    assert rehearsal.commands() == ["plan"] and sorted(p.name for p in state.iterdir()) == ["protocol.json"]
    assert protocol["protocol_version"] == tune.PROTOCOL_VERSION and protocol["rehearsal_blocks"] is not None
    assert "ESTIMATES, not measurements" in out and "추정치이며 측정값이 아닙니다" in out and "축소 블록" in out
    protocol_bytes = (state / "protocol.json").read_bytes()
    assert "already present and unchanged" in run(5) and (state / "protocol.json").read_bytes() == protocol_bytes
    rehearsal.commands()

    # 6. smoke: two passes, the second one without a live request. No pack for this small step.
    out = run(6)
    assert rehearsal.commands() == ["smoke"] and namespace["neg_smoke"]["status"] == "complete"
    assert "smoke passed" in out and "짧은 확인 통과" in out and not packs.exists()

    # 7. main cell stopped by its budget: a clean stop, a state ZIP, no selection, and a hint to run it again.
    rehearsal.live_limit = 5
    out = run(7, BUDGET_MINUTES=45)
    first_all = next(args for args in rehearsal.calls if "--stage" in args)
    assert first_all[first_all.index("--budget-minutes") + 1] == "45.0"
    assert rehearsal.commands() == ["all", "pack"]
    assert namespace["neg_all"]["status"] == "partial_budget" and namespace["neg_all"]["exit_code"] == 0
    assert "다시 실행하면 이어서 진행합니다" in out and not (state / "selection.json").exists()
    assert len(list(packs.glob("negotiation-tuning-state-*.zip"))) == 1
    # 9. the leaderboard of an unfinished run names the reason and selects nothing.
    out = run(9)
    assert rehearsal.commands() == ["board"] and namespace["neg_selection"] is None
    assert "선택 결과가 아직 없습니다" in out and "선택 보류" in colab.displayed[-1]
    assert not (state / "selection.json").exists()

    # 7 again: continues from the store and finishes; a third run asks the backend nothing.
    rehearsal.live_limit = None
    out = run(7)
    summary = namespace["neg_all"]
    assert (summary["status"], summary["selection"], summary["counts_as_model_evidence"]) == ("complete", "selected", "False")
    assert summary["live_requests"] > 0 and "모델 근거로 세지 않습니다" in out and "전체 실행이 끝났습니다" in out
    selection_bytes = (state / "selection.json").read_bytes()
    assert summary["selected"] == read(state / "selection.json")["selected"]["id"]
    run(7)
    assert namespace["neg_all"]["live_requests"] == 0 and namespace["neg_all"]["status"] == "complete"
    assert (state / "selection.json").read_bytes() == selection_bytes
    assert rehearsal.commands() == ["all", "pack", "all", "pack"]
    assert len({p.name for p in packs.glob("negotiation-tuning-state-*.zip")}) == 3      # a new name every time
    with pytest.raises(ValueError, match="BUDGET_MINUTES"):
        run(7, BUDGET_MINUTES=0)
    assert rehearsal.commands() == []

    # 8. the optional probe: off by default, and its result goes to a new folder under the state folder.
    assert "실행하지 않았습니다" in run(8) and rehearsal.commands() == [] and not (state / "utterance_probe").exists()
    out = run(8, RUN_UTTERANCE_PROBE=True)
    assert rehearsal.commands() == ["negotiation/run_utterance_probe.py"]
    (attempt,) = (state / "utterance_probe").iterdir()
    probe = read(attempt / "result.json")
    assert probe["complete"] is True and probe["verdict"]["counts_as_model_evidence"] is False
    assert probe["verdict"]["reading"] in out and "2026-11-18에 고정한 그대로" in out
    run(8, RUN_UTTERANCE_PROBE=True)
    assert len(list((state / "utterance_probe").iterdir())) == 2 and read(attempt / "result.json") == probe
    rehearsal.commands()

    # 9. leaderboard and selection, from the runner's own files.
    out = run(9)
    assert rehearsal.commands() == ["board"]
    assert colab.displayed[-1] == (state / "leaderboard.md").read_text(encoding="utf-8")
    assert namespace["neg_selection"]["selected"]["id"] == summary["selected"] and summary["selected"] in out
    assert "cpu_rehearsal_not_model_evidence" in out and (state / "selected_config.json").is_file()
    assert (state / "selection.json").read_bytes() == selection_bytes

    # 10. confirm: nothing without the tick; once with it; refused the second time.
    out = run(10)
    assert rehearsal.commands() == [] and "RUN_CONFIRM이 꺼져 있어" in out
    assert not (state / "confirm_first_use.json").exists() and not (state / "runs" / "confirm").exists()
    out = run(10, RUN_CONFIRM=True)
    confirm_call = rehearsal.calls[0]
    assert confirm_call[confirm_call.index("--budget-minutes") + 1] == "60.0"
    assert rehearsal.commands() == ["confirm", "pack"]
    board = read(state / "leaderboard.json")
    assert namespace["neg_confirm"]["confirm_verdict"] == board["confirm"]["verdict"] in tune.VERDICTS
    assert board["counts_as_model_evidence"] is False and "확인 구간 판정: " + board["confirm"]["verdict"] in out
    assert "판정" in colab.displayed[-1] and "리허설" in colab.displayed[-1]
    marker = (state / "confirm_first_use.json").read_bytes()
    with pytest.raises(RuntimeError, match="confirm: 끝나지 않았습니다"):
        run(10, RUN_CONFIRM=True)
    out = capsys.readouterr().out
    assert "거절 사유: CONFIRM was already used" in out and "status=refused" in out
    assert (state / "confirm_first_use.json").read_bytes() == marker and rehearsal.commands() == ["confirm", "pack"]
    assert read(state / "leaderboard.json")["confirm"]["verdict"] == board["confirm"]["verdict"]

    # 11. pack and download: a verified copy of the whole state folder, logs only when asked for.
    run(11)
    assert rehearsal.commands() == ["pack"] and len(colab.downloads) == 1
    state_zip = Path(colab.downloads[0])
    assert state_zip.parent == packs and state_zip.name.startswith("negotiation-tuning-state-")
    with zipfile.ZipFile(state_zip) as archive:
        manifest = json.loads(archive.read(tune.MANIFEST_NAME))
        packed = {entry["path"] for entry in manifest["files"]}
        assert set(archive.namelist()) == packed | {tune.MANIFEST_NAME}
    on_disk = {p.relative_to(state).as_posix() for p in state.rglob("*") if p.is_file()}
    assert packed == on_disk and {"protocol.json", "selection.json", "confirm_first_use.json"} <= packed
    assert any(name.startswith("utterance_probe/") for name in packed)
    run(11, DOWNLOAD_LOGS=True)
    assert len(colab.downloads) == 3 and Path(colab.downloads[1]).name != state_zip.name
    with zipfile.ZipFile(colab.downloads[2]) as archive:
        assert {"commands.jsonl", "source.json"} <= set(archive.namelist())

    # Housekeeping the notebook promises: the state is outside the source folder, the source folder gained
    # no result and no evidence folder, and the token appears in no output and no saved file.
    written = {p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert written == set(source_bundle.manifest["files"])
    assert not (repo / "negotiation" / "evidence").exists() and not (repo / "runs").exists()
    history = [json.loads(line) for line in (namespace["NEG_LOGS"] / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(entry["interrupted"] is False for entry in history) and [e["exit_code"] for e in history].count(1) == 1
    assert all(entry["argv"][0] == sys.executable and (namespace["NEG_LOGS"] / entry["log"]).is_file() for entry in history)
    everything = "".join(printed) + out + "".join(map(str, colab.displayed))
    assert SECRET not in everything and not re.search(rf"\b{SPLIT_LITERAL}\b", everything)
    for path in [p for p in colab.content.rglob("*") if p.is_file() and "__pycache__" not in p.parts]:
        data = path.read_bytes()
        assert SECRET.encode() not in data, path
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                assert all(SECRET.encode() not in archive.read(name) for name in archive.namelist()), path

    # A new runtime: the downloaded state ZIP in the file panel is verified and restored by cell 1, the
    # results can be viewed without a model, and the main cell continues without asking the backend again.
    later = Colab(tmp_path / "later-runtime", monkeypatch)
    (later.content / source_bundle.name).write_bytes(source_bundle.data)
    shutil.copyfile(state_zip, later.content / "negotiation-tuning-state-20261118T000000Z-abcd (1).zip")
    resumed = later.namespace()
    execute(1, resumed, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    out = capsys.readouterr().out
    restored = resumed["NEG_STATE"]
    assert "이전 상태를 복원했습니다" in out and "restored: written=" in out
    assert {p.relative_to(restored).as_posix() for p in restored.rglob("*") if p.is_file()} == packed
    assert (restored / "selection.json").read_bytes() == selection_bytes
    execute(9, resumed)                                          # no model in this runtime: viewing still works
    assert later.displayed[-1].startswith("# ") and resumed["neg_selection"]["selected"]["id"] == summary["selected"]
    with pytest.raises(RuntimeError, match="모델 준비"):
        execute(6, resumed)
    again = Rehearsal(resumed)
    execute(5, resumed)
    execute(7, resumed)
    assert resumed["neg_all"]["status"] == "complete" and resumed["neg_all"]["live_requests"] == 0
    assert again.commands() == ["plan", "all", "pack"]
    assert (restored / "selection.json").read_bytes() == selection_bytes
    with pytest.raises(RuntimeError, match="confirm: 끝나지 않았습니다"):      # the single use travels with the state
        execute(10, resumed, RUN_CONFIRM=True)
    # Running cell 1 again keeps the existing state folder and does not restore over it.
    execute(1, resumed, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    out = capsys.readouterr().out
    assert "기존 상태 폴더를 그대로 사용합니다" in out and "restored: written=" not in out
    assert (restored / "selection.json").read_bytes() == selection_bytes


def test_source_cell_refuses_several_or_tampered_state_zips_before_unpacking(colab, source_bundle, capsys, tmp_path):
    donor = tmp_path / "donor"
    assert tune.main(["plan", "--state-dir", str(donor), "--rehearsal-blocks", REHEARSAL]) == 0
    tune.pack_state(donor, tmp_path / "negotiation-tuning-state-a.zip")
    capsys.readouterr()
    (colab.content / source_bundle.name).write_bytes(source_bundle.data)
    for name in ("negotiation-tuning-state-a.zip", "negotiation-tuning-state-b.zip"):
        shutil.copyfile(tmp_path / "negotiation-tuning-state-a.zip", colab.content / name)
    namespace = colab.namespace()
    with pytest.raises(ValueError, match="상태 ZIP이 여러 개"):
        execute(1, namespace, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    assert colab.source_folders() == [] and not (colab.content / "negotiation-tuning-state").exists()
    # One archive whose protocol file was changed after packing: the runner refuses it, nothing is restored,
    # and the notebook does not mark the state as ready.
    (colab.content / "negotiation-tuning-state-b.zip").unlink()
    (colab.content / "negotiation-tuning-state-a.zip").write_bytes(_rewritten(
        (tmp_path / "negotiation-tuning-state-a.zip").read_bytes(),
        lambda n, d: [(n, d.replace(b"dev-1130", b"dev-1131") if n == "protocol.json" else d)]))
    namespace = colab.namespace()
    with pytest.raises(RuntimeError, match="실행 실패"):
        execute(1, namespace, EXPECTED_CONTENT_SHA256=source_bundle.content_sha256)
    assert "Refused: tampered archive member protocol.json" in capsys.readouterr().out
    assert namespace["NEG_STATE_READY"] is False
    state = colab.content / "negotiation-tuning-state"
    assert not state.exists() or not any(state.iterdir())
    with pytest.raises(RuntimeError, match="1번 셀"):
        execute(5, namespace)


# ---- a failed estimate waits for a person (runner in this process, with a failing test double) --------------

def test_failed_candidate_is_not_accepted_or_retried_by_viewing_the_leaderboard(colab, source_bundle, capsys, tmp_path):
    probe, counter = tmp_path / "probe", ModelDouble()
    assert tune_in_process(["plan", "--state-dir", str(probe), "--rehearsal-blocks", REHEARSAL], counter) == 0
    for stage in ("smoke", "regress", "calibrate"):
        assert tune_in_process(["run", "--stage", stage, "--state-dir", str(probe), "--backend", "keyword"], counter) == 0

    namespace = prepared(colab, source_bundle, capsys)
    runner = InProcess(namespace, ModelDouble(fail_at={counter.calls + 2}))      # fails inside M0 on SELECT
    state, packs = namespace["NEG_STATE"], namespace["NEG_PACKS"]
    execute(5, namespace)
    with pytest.raises(RuntimeError, match=r"all: 끝나지 않았습니다\(status=failed, exit=1\)"):
        execute(7, namespace)
    out = capsys.readouterr().out
    assert "자동으로 다시 시도하지" in out and "status=failed" in out
    assert len(list(packs.glob("*.zip"))) == 1                       # the state was packed although the run failed
    assert not (state / "selection.json").exists()
    failed_attempts = sorted((state / "runs" / "select").glob("M0-*/attempt-*"))
    assert len(failed_attempts) == 1 and read(failed_attempts[0] / "result.json")["status"] == "failed"

    # Viewing the leaderboard runs no `board` command, so the failure is not accepted behind the reader's back.
    runner.calls.clear()
    execute(9, namespace)
    out = capsys.readouterr().out
    assert runner.calls == [] and not (state / "selection.json").exists() and namespace["neg_selection"] is None
    assert "추정에 실패한 후보가 있어 선택을 기록하지 않았습니다: M0-" in out and "estimate_failed" in out
    assert "선택 보류" in colab.displayed[-1]
    # Running the main cell again does not retry the failed configuration: it stays failed with its one
    # attempt, and the run goes on with the other configurations (a person chose to continue).
    calls_before = runner.double.calls
    with pytest.raises(RuntimeError, match="status=failed"):
        execute(7, namespace)
    capsys.readouterr()
    assert runner.double.calls > calls_before and len(sorted((state / "runs" / "select").glob("M0-*/attempt-*"))) == 1
    calls_before = runner.double.calls
    with pytest.raises(RuntimeError, match="status=failed"):
        execute(7, namespace)                                   # nothing is left to run: no call at all
    capsys.readouterr()
    assert runner.double.calls == calls_before
    assert not (state / "selection.json").exists() and not (state / "confirm_first_use.json").exists()
    untouched = tmp_path / "before-decision"
    shutil.copytree(state, untouched)
    # The guard reads the attempt records, not the derived leaderboard: it holds without that view as well.
    (state / "leaderboard.json").unlink()
    assert len(namespace["neg_failed_waiting"]()) == 1 and namespace["neg_failed_waiting"]()[0].startswith("M0-")
    runner.calls.clear()
    execute(9, namespace)
    assert runner.calls == [] and not (state / "selection.json").exists()
    capsys.readouterr()

    # Decision 1, a person accepts the failure with the tick box: the selection is made among the others.
    execute(9, namespace, ACCEPT_FAILED_CANDIDATE=True)
    capsys.readouterr()
    selection = read(state / "selection.json")
    listed = {c["name"]: c for c in selection["candidates"]}
    assert (listed["M0"]["status"], listed["M0"]["eligible"]) == ("failed", False)
    assert selection["selected"]["name"] != "M0" and namespace["neg_selection"]["selected"]["id"] == selection["selected"]["id"]
    assert namespace["neg_failed_waiting"]() == []

    # Decision 2 (from the untouched copy), the retry written in the troubleshooting table.
    shutil.rmtree(state)
    shutil.copytree(untouched, state)
    snippets = re.findall(r"`(neg_stage\([^`]*'--retry-failed'\))`", sources("markdown")[-1])
    assert len(snippets) == 1
    runner.double = ModelDouble()
    exec(compile(ast.parse(snippets[0]), "troubleshooting_retry", "exec"), namespace)
    out = capsys.readouterr().out
    assert "status=complete" in out
    retried = read(sorted((state / "runs" / "select").glob("M0-*/attempt-*"))[-1] / "result.json")
    assert retried["complete"] is True and retried["retry_of"] == failed_attempts[0].name
    assert {c["name"]: c["eligible"] for c in read(state / "selection.json")["candidates"]}["M0"] is True


# ---- interruption --------------------------------------------------------------------------------------------

class FakeProcess:
    """A child that prints one line, is interrupted, and ends after `ends_after` waits."""

    def __init__(self, ends_after):
        self.ends_after, self.events, self.returncode = ends_after, [], None

    @property
    def stdout(self):
        yield "select M0 [1/6] dev-1130 agreed saving=1\n"
        raise KeyboardInterrupt

    def communicate(self, timeout=None):
        self.events.append(("wait", timeout))
        if sum(1 for event in self.events if event[0] == "wait") <= self.ends_after:
            raise subprocess.TimeoutExpired("cmd", timeout)
        self.returncode = 130
        return "tune run: stage=all status=interrupted live_requests_this_process=1 state_dir=x\n", None

    def send_signal(self, number):
        self.events.append(("signal", number))

    def terminate(self):
        self.events.append(("terminate",))

    def kill(self):
        self.events.append(("kill",))

    def poll(self):
        return self.returncode


@pytest.mark.parametrize("ends_after,expected", [
    (0, [("wait", 10)]),                                                     # it stopped by itself: no signal sent
    (1, [("wait", 10), ("signal", signal.SIGINT), ("wait", 30)]),
    (2, [("wait", 10), ("signal", signal.SIGINT), ("wait", 30), ("terminate",), ("wait", 15)]),
    (3, [("wait", 10), ("signal", signal.SIGINT), ("wait", 30), ("terminate",), ("wait", 15), ("kill",), ("wait", 15)]),
])
def test_interrupt_lets_the_command_finish_then_escalates_and_is_recorded(colab, source_bundle, capsys, ends_after, expected):
    namespace = prepared(colab, source_bundle, capsys)
    process = FakeProcess(ends_after)
    started = []

    def popen(command, **kwargs):
        started.append((command, kwargs))
        return process

    namespace["subprocess"] = SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                              TimeoutExpired=subprocess.TimeoutExpired)
    with pytest.raises(KeyboardInterrupt):
        namespace["neg_run"]("negotiation/run_tune.py", "run", "--stage", "all")
    assert process.events == expected
    out = capsys.readouterr().out
    assert "중단 요청을 받았습니다" in out and "status=interrupted" in out
    (command, kwargs), = started
    assert command == [sys.executable, "-u", "negotiation/run_tune.py", "run", "--stage", "all"]
    assert kwargs["cwd"] == namespace["NEG_REPO"] and kwargs["stderr"] == subprocess.STDOUT
    logs = namespace["NEG_LOGS"]
    (entry,) = [json.loads(line) for line in (logs / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entry["interrupted"] is True and entry["exit_code"] == 130 and entry["argv"] == command
    text = (logs / entry["log"]).read_text(encoding="utf-8")
    assert text.startswith("select M0 [1/6]") and text.rstrip().endswith("state_dir=x")


def test_an_interrupted_stage_still_packs_the_state_and_fills_in_nothing(colab, source_bundle, capsys):
    namespace = prepared(colab, source_bundle, capsys)
    state = namespace["NEG_STATE"]
    state.mkdir()
    (state / "protocol.json").write_text("{}", encoding="utf-8")
    calls = []

    def neg_run(*args, check=True):
        args = [str(a) for a in args]
        calls.append(args[1])
        if args[1] == "run":
            raise KeyboardInterrupt
        Path(args[args.index("--output") + 1]).write_bytes(b"packed")
        return {"exit_code": 0, "lines": [], "log": "stub"}

    namespace.update(neg_run=neg_run, NEG_ARGS=["--backend", "keyword"], NEG_READY=True)
    with pytest.raises(KeyboardInterrupt):
        execute(7, namespace)
    assert calls == ["run", "pack"] and "neg_all" not in namespace
    assert len(list(namespace["NEG_PACKS"].glob("negotiation-tuning-state-*.zip"))) == 1
    assert "전체 실행이 끝났습니다" not in capsys.readouterr().out


def test_status_lines_of_the_real_runner_are_what_the_notebook_parses(colab, source_bundle, capsys):
    namespace = prepared(colab, source_bundle, capsys)
    parse = namespace["neg_tune_summary"]
    explained = set(namespace["NEG_STATUS_TEXT"])
    # Every status the runner can print for `run` has a Korean explanation in the notebook.
    assert set(tune.STOPPING) | {"complete", "skipped_complete", "pending_maps", "failed", "smoke_failed"} <= explained
    summary = parse({"exit_code": 0, "log": "x", "lines": [
        "tune run: stage=confirm status=complete live_requests_this_process=23 state_dir=/content/s",
        "leaderboard: selection=selected selected=M4-9d85bb778cb5 confirm_verdict=not_confirmed "
        "counts_as_model_evidence=False"]})
    assert summary == {"exit_code": 0, "log": "x", "stage": "confirm", "status": "complete", "live_requests": 23,
                       "selection": "selected", "selected": "M4-9d85bb778cb5", "confirm_verdict": "not_confirmed",
                       "counts_as_model_evidence": "False", "refused": None}
    blocked = parse({"exit_code": 0, "log": "x", "lines": [
        "tune run: stage=all status=partial_budget live_requests_this_process=5 state_dir=s",
        "leaderboard: selection=blocked counts_as_model_evidence=True"]})
    assert (blocked["status"], blocked["selection"], blocked["selected"], blocked["confirm_verdict"],
            blocked["counts_as_model_evidence"]) == ("partial_budget", "blocked", None, None, "True")
    refused = parse({"exit_code": 1, "log": "x", "lines": ["Refused: something specific"]})
    assert (refused["status"], refused["refused"]) == ("refused", "something specific")
    unknown = parse({"exit_code": 2, "log": "x", "lines": ["usage: run_tune.py"]})
    assert unknown["status"] is None
    with pytest.raises(RuntimeError, match="exit=2"):
        namespace["neg_checked"](unknown, "계획")
