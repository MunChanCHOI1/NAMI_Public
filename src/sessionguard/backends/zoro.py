"""Custom text-only Transformers adapter for a Zoro-1 typed-decision GPU probe.

Validation so far: CPU contract mocks, then real pinned weights on one local
RTX 5060 Laptop GPU in BF16 on 2026-11-16 (artifacts/gpu/20261116/). T4/FP16,
Colab, Kaggle and every other GPU remain unverified.
Uses the pinned official exporter, request codec and original PointerHead. It never
calls generate(), the vocabulary output head, or an alternative model. A successful
import is not evidence of model execution; see each response's execution metadata.
This adapter is not the official SGLang DecideEngine. Numerical parity with that
engine on the real checkpoint has NOT been verified.

Prepare the checkpoint with the official export_kev_checkpoint.py before loading.
Keep the runtime's working torch installation; transformers==5.17.0 was used in
the one local GPU run above and is not a lock validated on other GPUs. Do not
install fla/causal-conv1d for this deliberately slow native-PyTorch path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sessionguard.contracts import InputOverflowError, ordered_json_sha256


MODEL_ID = "trillionlabs/Zoro-1"
MODEL_REVISION = "766c5522cfb7df47dc9546df20fe3ce6efe6067c"
TRIDA_REVISION = "3a7877868c63ff82af14f707a1018bc0baac0711"
# True since the 2026-11-16 local BF16 run. It describes this adapter's history,
# not the current process: see gpu_execution_performed in each response.
AUTHORING_GPU_TESTED = True
SOURCE_SHA256 = {
    "decide_request.py": "41b1734e112b5310231d7bacb72e92a8a933feb4504bcdf0ef1f8ad0d8131f56",
    "decide_engine.py": "41d75d4b04c990af50cdf6ae332dac0f0480db5e2ae1a3e5546946f6cf6ccc30",
    "export_kev_checkpoint.py": "dcb10a46d1a6495cbfe66974eaf50aacf5428ca091b993a36d40aec099548d80",
}
OFFICIAL_RELATIVE_PATHS = {
    "decide_request.py": "inference/zoro-1/decide_request.py",
    "decide_engine.py": "inference/zoro-1/decide_engine.py",
    "export_kev_checkpoint.py": "train/tools/export_kev_checkpoint.py",
}
_IMPORT_LOCK = threading.Lock()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_probabilities(values: list[float], expected_count: int) -> None:
    """Check full-precision probabilities BEFORE the official formatter rounds them."""
    if len(values) != expected_count or not values:
        raise ValueError("Probability vector has the wrong number of options.")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise FloatingPointError("Non-finite or out-of-range decision probabilities.")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-5):
        raise FloatingPointError("Decision probabilities do not sum to one.")


def _module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves the defining module here
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_official_helpers(source_dir: str | Path):
    """Verify a repo-layout or notebook-flat source directory; do not start SGLang."""
    source_dir = Path(source_dir)
    flat = all((source_dir / filename).is_file() for filename in SOURCE_SHA256)
    paths = {
        filename: source_dir / (filename if flat else OFFICIAL_RELATIVE_PATHS[filename])
        for filename in SOURCE_SHA256
    }
    for filename, expected in SOURCE_SHA256.items():
        actual = sha256_file(paths[filename])
        if actual != expected:
            raise ValueError(f"{filename} does not match pinned Trida source {TRIDA_REVISION}.")
    # decide_engine imports the bare name decide_request. Bind it only while importing
    # the official module, then restore any pre-existing module in the notebook.
    with _IMPORT_LOCK:
        codec = _module_from_file("_sessionguard_zoro_codec", paths["decide_request.py"])
        absent = object()
        previous = sys.modules.get("decide_request", absent)
        sys.modules["decide_request"] = codec
        try:
            engine_helpers = _module_from_file(
                "_sessionguard_zoro_head", paths["decide_engine.py"]
            )
        finally:
            if previous is absent:
                sys.modules.pop("decide_request", None)
            else:
                sys.modules["decide_request"] = previous
    return codec, engine_helpers.PointerHead, engine_helpers.serving_temperatures


class DecisionBackend:
    """One sequential, text-only native decision backend; no automatic CPU fallback.

    original_checkpoint must be the path returned by snapshot_download with the
    pinned revision. exported_checkpoint must be the official export of that path.
    CPU FP32 is available only by explicitly passing device='cpu'; a 4B FP32 load
    may exceed the RAM of a free Colab runtime.
    """

    name = "zoro"

    def __init__(
        self,
        original_checkpoint: str | Path,
        exported_checkpoint: str | Path,
        official_source_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "auto",
        max_input_tokens: int = 2048,
        decision_temperature: float | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self._lock = threading.Lock()
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("This probe supports CUDA, or explicitly requested CPU only.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is attached. Select a GPU runtime and retry.")
        if dtype == "auto":
            dtype = "fp32" if self.device.type == "cpu" else (
                "bf16" if torch.cuda.get_device_capability(self.device)[0] >= 8
                and torch.cuda.is_bf16_supported() else "fp16"
            )
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError("dtype must be auto, fp16, bf16 or fp32.")
        if self.device.type == "cpu" and dtype != "fp32":
            raise ValueError("Use FP32 for the explicit CPU reference probe.")
        if self.device.type == "cuda" and dtype == "bf16":
            if torch.cuda.get_device_capability(self.device)[0] < 8:
                raise ValueError("This probe requires native BF16 hardware for dtype=bf16; use fp16 on T4.")
        self.dtype_name = dtype
        # use_kernels=False disables Hub kernels, but Transformers can still select
        # an installed original FLA/causal-conv package. Avoid mislabelling that as
        # native PyTorch: require an environment without these optional packages.
        optional = [p for p in ("fla", "causal_conv1d") if importlib.util.find_spec(p) is not None]
        if optional:
            raise RuntimeError(
                "This native-PyTorch probe needs an environment without optional "
                f"compiled packages {optional}. Use a clean runtime; nothing was uninstalled."
            )
        original = Path(original_checkpoint).expanduser().absolute()
        exported = Path(exported_checkpoint).expanduser().absolute()
        if original.name != MODEL_REVISION or original.parent.name != "snapshots":
            raise ValueError(
                f"Use snapshot_download({MODEL_ID!r}, revision={MODEL_REVISION!r}) and pass its returned path."
            )
        self.codec, head_class, temperatures = load_official_helpers(official_source_dir)
        if not 1 <= max_input_tokens <= self.codec.MAX_BRANCH:
            raise ValueError(f"max_input_tokens must be in 1..{self.codec.MAX_BRANCH}.")
        self.max_input_tokens = int(max_input_tokens)
        temperature_file = original / "temperature.json"
        if decision_temperature is None:
            self.temperature = float(temperatures(str(original))[0])
            self.temperature_source = (
                "original_checkpoint/temperature.json:text" if temperature_file.exists()
                else "official_native_default:raw_temperature_1"
            )
        else:
            self.temperature = float(decision_temperature)
            self.temperature_source = "explicit_override; not independently calibrated"
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("decision_temperature must be finite and positive.")

        original_head = torch.load(original / "jev_head.pt", map_location="cpu", weights_only=True)
        exported_meta = torch.load(exported / "head.pt", map_location="cpu", weights_only=True)
        export_temperature = exported_meta.get("temperature")
        self.exported_temperature = None if export_temperature is None else float(export_temperature)
        if self.exported_temperature is not None and (
            not math.isfinite(self.exported_temperature) or self.exported_temperature <= 0
        ):
            raise ValueError("Export metadata contains an invalid temperature.")
        # Informational only: runtime temperature still comes from the original
        # checkpoint or the explicit override, never implicitly from this export.
        self.runtime_temperature_matches_export = (
            self.exported_temperature is not None and self.temperature == self.exported_temperature
        )
        if Path(exported_meta.get("exported_from", "")).resolve() != original.resolve():
            raise ValueError("Export metadata does not point to this pinned original checkpoint.")
        for key in ("q.weight", "q.bias", "k.weight", "k.bias"):
            if key not in original_head["head"] or key not in exported_meta["head"]:
                raise ValueError(f"Missing pointer-head parameter {key}.")
            if not torch.equal(original_head["head"][key].float(), exported_meta["head"][key].float()):
                raise ValueError(f"Exporter changed pointer-head parameter {key}.")
            if not torch.isfinite(original_head["head"][key]).all().item():
                raise FloatingPointError(f"Non-finite pointer-head parameter {key}.")
        self.hidden_size = int(original_head["head"]["q.weight"].shape[1])
        self.head = head_class(self.hidden_size, int(original_head["head_dim"])).float().eval()
        self.head.load_state_dict(original_head["head"], strict=True)
        self.head_sha256 = sha256_file(original / "jev_head.pt")
        self.tokenizer = AutoTokenizer.from_pretrained(str(exported), local_files_only=True)
        token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.codec.SPECIAL]
        if any(i is None or i == self.tokenizer.unk_token_id for i in token_ids) or len(set(token_ids)) != 5:
            raise ValueError("The official five decision delimiter tokens were not preserved.")
        self.backbone, info = AutoModel.from_pretrained(
            str(exported), dtype=dtype_map[dtype], attn_implementation="eager",
            device_map={"": str(self.device)}, local_files_only=True,
            use_kernels=False, output_loading_info=True,
        )
        self.backbone_loading_info = json.loads(json.dumps(
            info, default=lambda value: sorted(value) if isinstance(value, set) else str(value)
        ))
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Backbone did not load completely: {info}")
        if self.backbone.config.model_type != "qwen3_5_text":
            raise ValueError("Use the official text-only export, not a vision model or a different base.")
        if self.backbone.config.hidden_size != self.hidden_size:
            raise ValueError("Backbone hidden size does not match the original decision head.")
        self.backbone.eval()
        first_parameter = next(self.backbone.parameters())
        self.backbone_first_parameter_device = str(first_parameter.device)
        self.backbone_first_parameter_dtype = str(first_parameter.dtype)
        self.runtime_versions = {}
        for package in ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors"):
            try:
                self.runtime_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.runtime_versions[package] = "unavailable"
        self.gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None

    def _encode_without_truncation(self, request: dict[str, Any]):
        if request.get("image") is not None:
            raise ValueError("SessionGuard's probe is text-only.")
        parsed = self.codec.parse(request)
        # Use the pinned codec's exact escaping/tokenization BEFORE encode_rows
        # slices state or rejects an oversized branch. Delimiters count as tokens.
        state_tokens = 1 + len(self.codec._user(self.tokenizer, parsed.state))
        if state_tokens > self.codec.MAX_STATE or state_tokens > self.max_input_tokens:
            raise InputOverflowError(
                f"State has {state_tokens} tokens; no truncation is allowed "
                f"(adapter limit {self.max_input_tokens}, codec limit {self.codec.MAX_STATE})."
            )
        for question in parsed.questions:
            branch_tokens = 1 + len(self.codec._user(self.tokenizer, question.instr))
            branch_tokens += sum(2 + len(self.codec._user(self.tokenizer, option)) for option in question.options)
            branch_tokens += 1  # final decision delimiter
            row_tokens = state_tokens + branch_tokens
            if row_tokens > self.max_input_tokens or row_tokens > self.codec.MAX_BRANCH:
                raise InputOverflowError(
                    f"Question {question.qid} has {row_tokens} tokens; no truncation is allowed "
                    f"(adapter limit {self.max_input_tokens}, codec limit {self.codec.MAX_BRANCH})."
                )
        encoded = parsed.encode(self.tokenizer)
        if len(encoded.rows) != len(parsed.questions) or not encoded.rows:
            raise ValueError("Question/row count mismatch.")
        for question, row in zip(parsed.questions, encoded.rows):
            n = len(row["ids"])
            opts, decide = row["opts"], row["decide"]
            if n > self.max_input_tokens:
                raise InputOverflowError(f"Question {question.qid} has {n} tokens; limit is {self.max_input_tokens}.")
            if decide != n - 1 or len(opts) != len(question.keys) or not opts:
                raise ValueError("Malformed official option/decision indices.")
            if opts != sorted(set(opts)) or any(not 0 <= p < decide for p in opts):
                raise ValueError("Invalid or duplicate option positions.")
        return parsed, encoded

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return official formatted answers plus unrounded probabilities and measured metadata."""
        with self._lock:
            return self._decide_locked(request)

    def _decide_locked(self, request: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        started = time.perf_counter()
        parsed, encoded = self._encode_without_truncation(request)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        answers, raw_probabilities, raw_logits = {}, {}, {}
        forward_ms = 0.0
        model_ms = 0.0
        with torch.inference_mode():
            # One full causal row per question. No cross-question state sharing or
            # generation; owner indices are therefore all zero for each head call.
            for question, row in zip(parsed.questions, encoded.rows):
                ids = torch.tensor([row["ids"]], dtype=torch.long, device=self.device)
                t0 = time.perf_counter()
                output = self.backbone(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False, return_dict=True,
                )
                if cuda:
                    torch.cuda.synchronize(self.device)
                forward_ms += (time.perf_counter() - t0) * 1000
                hidden = output.last_hidden_state
                if tuple(hidden.shape) != (1, len(row["ids"]), self.hidden_size):
                    raise ValueError(f"Unexpected last_hidden_state shape {tuple(hidden.shape)}.")
                if not torch.isfinite(hidden).all().item():
                    raise FloatingPointError("Non-finite hidden states; do not use this FP16 run's decisions.")
                # Transfer only the required readout positions, not the full row.
                h_decide = hidden[0, row["decide"]].float().cpu().unsqueeze(0)
                h_opts = hidden[0, row["opts"]].float().cpu()
                owners = torch.zeros(len(row["opts"]), dtype=torch.long)
                logits = self.head(h_decide, h_opts, owners)
                if tuple(logits.shape) != (len(question.keys),) or not torch.isfinite(logits).all().item():
                    raise FloatingPointError("Invalid native decision-head logits.")
                p = torch.softmax(logits / self.temperature, dim=-1).tolist()
                validate_probabilities(p, len(question.keys))
                model_ms += (time.perf_counter() - t0) * 1000
                answers[question.qid] = self.codec.to_answer(question, p)
                raw_probabilities[question.qid] = dict(zip(question.keys, p))
                raw_logits[question.qid] = dict(zip(question.keys, logits.tolist()))
                del output, hidden, ids, h_decide, h_opts, owners, logits
        return {
            "answers": answers,
            "raw_probabilities": raw_probabilities,
            "raw_logits": raw_logits,
            "metadata": {
                "request_sha256": ordered_json_sha256(request),
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "source_revision": TRIDA_REVISION,
                "backend": "transformers_native_pointer_head",
                "official_sglang_parity_verified": False,
                "head_loaded": True, "head_file_sha256": self.head_sha256,
                "generation_used": False, "mock": False,
                "authoring_gpu_tested": AUTHORING_GPU_TESTED,
                "gpu_execution_performed": cuda,
                "native_head_forward_calls": len(encoded.rows),
                "dtype": self.dtype_name, "attention_backend": "eager",
                "gdn_backend": "native_pytorch_no_fla_or_causal_conv1d",
                "device": str(self.device), "gpu_name": self.gpu_name,
                "decision_temperature": self.temperature,
                "temperature_source": self.temperature_source,
                "exported_temperature": self.exported_temperature,
                "runtime_temperature_matches_export": self.runtime_temperature_matches_export,
                "backbone_loading_info": self.backbone_loading_info,
                "backbone_first_parameter_device": self.backbone_first_parameter_device,
                "backbone_first_parameter_dtype": self.backbone_first_parameter_dtype,
                "runtime_versions": dict(self.runtime_versions),
                "input_tokens_per_question": [len(r["ids"]) for r in encoded.rows],
                "question_option_order": {q.qid: list(q.keys) for q in parsed.questions},
                "backbone_forward_latency_ms": round(forward_ms, 3),
                "model_only_latency_ms": round(model_ms, 3),
                "backend_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else None,
                "probability_notice": "Probabilities rank allowed actions; they are not calibrated attack probabilities.",
            },
        }


def run_probe(
    *,
    original_checkpoint: str | Path,
    exported_checkpoint: str | Path,
    official_source_dir: str | Path,
    request: dict[str, Any],
    device: str = "cuda:0",
    dtype: str = "auto",
    max_input_tokens: int = 2048,
    decision_temperature: float | None = None,
) -> dict[str, Any]:
    """Load local artifacts and make one decision; never download or mutate sessions.

    Importing this module is lightweight. Torch/Transformers are needed only when
    this function constructs DecisionBackend. Successful CPU mock tests do not
    validate this real-checkpoint constructor or its GPU forward.
    Accept a bare request or the explicit {request: ..., reasoning: false} wrapper.
    """
    if not isinstance(request, dict):
        raise TypeError("request must be a decision request JSON object")
    if request.get("reasoning", False) is not False:
        raise ValueError("This native-head probe supports reasoning=false only.")
    if "request" in request:
        unsupported = set(request) - {"request", "reasoning"}
        if unsupported:
            raise ValueError(f"Unsupported probe wrapper fields: {sorted(unsupported)}")
        request = request["request"]
        if not isinstance(request, dict):
            raise TypeError("The wrapped request must be a JSON object.")
    started = time.perf_counter()
    backend = DecisionBackend(
        original_checkpoint=original_checkpoint,
        exported_checkpoint=exported_checkpoint,
        official_source_dir=official_source_dir,
        device=device,
        dtype=dtype,
        max_input_tokens=max_input_tokens,
        decision_temperature=decision_temperature,
    )
    load_seconds = time.perf_counter() - started
    response = backend.decide(request)
    response["metadata"].update(
        probe_mode="model_connection_only",
        session_mutation_performed=False,
        checkpoint_load_seconds=round(load_seconds, 6),
    )
    return response


def main(argv: list[str] | None = None) -> int:
    """Standalone probe; the project's main CLI can call run_probe directly."""
    import argparse

    parser = argparse.ArgumentParser(description="One local Zoro native-head decision; no automatic download.")
    parser.add_argument("--original-checkpoint", required=True)
    parser.add_argument("--exported-checkpoint", required=True)
    parser.add_argument("--official-source-dir", required=True)
    parser.add_argument("--request", required=True, help="Request JSON, or a JSON document containing request.")
    parser.add_argument("--output", help="Save JSON here; otherwise print it.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--decision-temperature", type=float)
    args = parser.parse_args(argv)
    body = json.loads(Path(args.request).read_text(encoding="utf-8"))
    result = run_probe(
        original_checkpoint=args.original_checkpoint,
        exported_checkpoint=args.exported_checkpoint,
        official_source_dir=args.official_source_dir,
        request=body,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        decision_temperature=args.decision_temperature,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(output)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
