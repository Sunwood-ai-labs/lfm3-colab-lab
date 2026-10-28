#!/usr/bin/env python3
"""Run pinned LFM3 text-tool and VL video/tool coverage on a Colab T4."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

MODEL_SPECS = {
    "text": {
        "model_id": "LiquidAI/LFM3-1.7B",
        "revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "subdir": "text_extra",
    },
    "vision": {
        "model_id": "LiquidAI/LFM3-VL-2B",
        "revision": "f83d2b0a2296aaa277f394990a9b8b949ddf9691",
        "subdir": "vision_extra",
    },
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers and return their sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer."},
                    "b": {"type": "integer", "description": "Second integer."},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str, error_type: str = "StageFailure"):
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "__dict__"):
        return {key: jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")
        stream.flush()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def input_facts(output_dir: Path) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for key, filename in (("scene", "scene.png"), ("sequence_video", "sequence_video.mp4")):
        path = output_dir / "inputs" / filename
        facts[key] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return facts


def add_numbers(a: int, b: int) -> int:
    if type(a) is not int or type(b) is not int:
        raise TypeError("add_numbers accepts integer arguments only")
    return a + b


ALLOWED_HANDLERS = {"add_numbers": add_numbers}


def parse_qwen3_coder_call(raw_text: str) -> dict[str, Any]:
    calls = list(re.finditer(r"<tool_call>(.*?)</tool_call>", raw_text, flags=re.DOTALL))
    if len(calls) != 1:
        raise ValueError(f"expected exactly one XML <tool_call>, found {len(calls)}")
    body = calls[0].group(1)
    function = re.fullmatch(
        r"\s*<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>\s*",
        body,
        flags=re.DOTALL,
    )
    if not function:
        raise ValueError("tool call did not match the documented <function=name> XML shape")
    name, payload = function.groups()
    arguments: dict[str, int] = {}
    cursor = 0
    pattern = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>", re.DOTALL)
    for match in pattern.finditer(payload):
        if payload[cursor : match.start()].strip():
            raise ValueError("unexpected content between XML parameters")
        key, value = match.groups()
        if key in arguments:
            raise ValueError(f"duplicate tool parameter: {key}")
        value = value.strip()
        if not re.fullmatch(r"[-+]?\d+", value):
            raise ValueError(f"parameter {key} must contain one integer")
        arguments[key] = int(value)
        cursor = match.end()
    if payload[cursor:].strip():
        raise ValueError("unexpected content after XML parameters")
    if set(arguments) != {"a", "b"}:
        raise ValueError(f"add_numbers requires exactly a and b, received {sorted(arguments)}")
    return {"name": name, "arguments": arguments, "raw_xml": calls[0].group(0)}


def answer_has_integer(answer: str, expected: int) -> bool:
    from decimal import Decimal
    values = re.findall(r"(?<![0-9.,+-])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9]|[.,][0-9])", answer)
    return any(Decimal(value.replace(",", "")) == Decimal(expected) for value in values)


def final_surface(answer: str) -> str:
    if "</think>" in answer:
        return answer.rsplit("</think>", 1)[-1].strip()
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


def colors_in_order(answer: str) -> bool:
    normalized = final_surface(answer).lower()
    red = [match.start() for match in re.finditer(r"赤|レッド|red", normalized)]
    blue = [match.start() for match in re.finditer(r"青|ブルー|blue", normalized)]
    return bool(red and blue and min(red) < min(blue))


def japanese_nonempty(answer: str) -> bool:
    surface = final_surface(answer)
    return bool(surface.strip() and re.search(r"[ぁ-んァ-ヶ一-龯]", surface))


def split_assistant_call(raw_text: str) -> tuple[str, str | None]:
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    visible = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    visible = re.sub(r"<tool_call>.*?</tool_call>", "", visible, flags=re.DOTALL).strip()
    return visible, think_match.group(1).strip() if think_match else None


def video_metadata_facts(inputs: Any) -> dict[str, Any]:
    metadata = inputs.get("video_metadata")
    if not metadata:
        raise StageFailure("video_metadata", "official processor output did not contain video_metadata")
    entries = metadata if isinstance(metadata, (list, tuple)) else [metadata]
    summary = []
    for item in entries:
        summary.append(
            {
                "fps": getattr(item, "fps", None),
                "total_num_frames": getattr(item, "total_num_frames", None),
                "duration": getattr(item, "duration", None),
                "height": getattr(item, "height", None),
                "width": getattr(item, "width", None),
                "frames_indices": jsonable(getattr(item, "frames_indices", None)),
            }
        )
    frames = inputs.get("pixel_values_videos")
    if frames is None:
        raise StageFailure("video_processor", "official processor did not return pixel_values_videos")
    return {
        "metadata": summary,
        "video_grid_thw": jsonable(inputs.get("video_grid_thw")),
        "pixel_values_videos_shape": list(frames.shape),
    }


class Runner:
    def __init__(self, mode: str, model: Any, processor: Any, torch: Any, generation_log: Path):
        self.mode = mode
        self.model = model
        self.processor = processor
        self.torch = torch
        self.generation_log = generation_log
        self.device = torch.device("cuda:0")
        self.seed_counter = 7000

    @property
    def tokenizer(self) -> Any:
        return self.processor if self.mode == "text" else self.processor.tokenizer

    def prepare(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, thinking: bool) -> Any:
        template_owner = self.processor
        options: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": thinking,
        }
        if tools:
            options["tools"] = tools
        if self.mode == "vision":
            # Keep these flat within processor_kwargs: Transformers 5.21 routes
            # known video options to videos_kwargs, but ProcessorMixin.__call__
            # retains the returned video_metadata only when its top-level
            # return_metadata kwarg is true.
            options["processor_kwargs"] = {"fps": 2.0, "return_metadata": True}
        try:
            result = template_owner.apply_chat_template(messages, **options)
        except Exception as exc:
            raise StageFailure("processor", str(exc), type(exc).__name__) from exc
        if not hasattr(result, "items"):
            raise StageFailure("processor", f"expected tokenized mapping, received {type(result).__name__}")
        moved = {}
        for key, value in result.items():
            moved[key] = value.to(self.device) if hasattr(value, "to") and hasattr(value, "shape") else value
        if "input_ids" not in moved:
            raise StageFailure("processor", "tokenized inputs did not include input_ids")
        return moved

    def generate(
        self,
        case_id: str,
        turn: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        thinking: bool,
        max_new_tokens: int,
        attempt: str,
    ) -> dict[str, Any]:
        inputs = self.prepare(messages, tools, thinking)
        video_facts = video_metadata_facts(inputs) if self.mode == "vision" and any(
            isinstance(item, dict) and item.get("type") == "video"
            for message in messages
            for item in (message.get("content") if isinstance(message.get("content"), list) else [])
        ) else None
        if video_facts is not None:
            inputs.pop("video_metadata", None)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        self.seed_counter += 1
        seed = self.seed_counter
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.torch.cuda.synchronize(self.device)
        allocated_before = int(self.torch.cuda.memory_allocated(self.device))
        self.torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.1,
                    top_k=50,
                    repetition_penalty=1.1,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            self.torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise StageFailure("generate", str(exc), type(exc).__name__) from exc
        elapsed = time.perf_counter() - started
        new_ids = generated[:, prompt_tokens:]
        raw_text = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
        generated_tokens = int(new_ids.shape[-1])
        measurement = {
            "case_id": case_id,
            "turn": turn,
            "attempt": attempt,
            "thinking_enabled": thinking,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "truncated_at_token_limit": generated_tokens >= max_new_tokens,
            "cuda_synchronized_wall_seconds": elapsed,
            "cuda_allocated_before_bytes": allocated_before,
            "cuda_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            "cuda_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved(self.device)),
            "video_processor": video_facts,
            "raw_text": raw_text,
            "captured_utc": utc_now(),
        }
        append_jsonl(self.generation_log, measurement)
        return measurement


def request_message(text: str, media: dict[str, Any] | None = None) -> dict[str, Any]:
    if media is None:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [media, {"type": "text", "text": text}]}


def stage_error(exc: Exception, fallback_stage: str) -> dict[str, str]:
    return {
        "stage": getattr(exc, "stage", fallback_stage),
        "type": getattr(exc, "error_type", type(exc).__name__),
        "message": str(exc),
    }


def is_infrastructure_error(error: dict[str, Any]) -> bool:
    return error.get("stage") in {"model_load", "gpu_check", "video_metadata", "video_processor", "processor", "generate", "input_fixture"}


def run_repeated_simple(
    runner: Runner,
    case_id: str,
    request: dict[str, Any],
    thinking: bool,
    max_new_tokens: int,
    repeats: int,
    scorer: Any,
    source_facts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    case: dict[str, Any] = {
        "case_id": case_id,
        "request": request,
        "source_facts": source_facts,
        "thinking_enabled": thinking,
        "max_new_tokens": max_new_tokens,
        "warmup": None,
        "repetitions": [],
    }
    infra_failed = False
    try:
        case["warmup"] = runner.generate(case_id, "answer", [request], None, thinking, max_new_tokens, "warmup")
    except Exception as exc:
        error = stage_error(exc, "warmup")
        case["warmup"] = {"error": error}
        infra_failed = is_infrastructure_error(error)
    for index in range(repeats):
        result: dict[str, Any] = {"repeat": index + 1}
        try:
            measured = runner.generate(case_id, "answer", [request], None, thinking, max_new_tokens, f"measure-{index + 1}")
            result["generation"] = measured
            result["score"] = scorer(measured["raw_text"])
        except Exception as exc:
            error = stage_error(exc, "measure")
            result["error"] = error
            result["score"] = False
            infra_failed = infra_failed or is_infrastructure_error(error)
        case["repetitions"].append(result)
    scores = [bool(row.get("score")) for row in case["repetitions"]]
    case["task_score"] = {"passed": sum(scores), "attempted": len(scores), "accuracy": (sum(scores) / len(scores)) if scores else 0.0}
    case["errors"] = [case["warmup"]["error"]] if isinstance(case["warmup"], dict) and "error" in case["warmup"] else []
    case["errors"].extend(row["error"] for row in case["repetitions"] if "error" in row)
    return case, infra_failed


def run_tool_case(
    runner: Runner,
    case_id: str,
    request: dict[str, Any],
    expected_args: dict[str, int],
    expected_result: int,
    repeats: int,
    source_facts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    case: dict[str, Any] = {
        "case_id": case_id,
        "request": request,
        "system_prompt": "算術計算では add_numbers ツールを呼び出し、ツール結果を受け取ってから答えてください。",
        "tool_schemas": TOOL_SCHEMAS,
        "source_facts": source_facts,
        "expected_tool_arguments_local_only": expected_args,
        "expected_result_local_only": expected_result,
        "warmup": None,
        "repetitions": [],
    }
    infra_failed = False

    def roundtrip(attempt: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": case["system_prompt"]},
            request,
        ]
        item: dict[str, Any] = {"attempt": attempt, "request_history": messages}
        first = runner.generate(case_id, "tool_call", messages, TOOL_SCHEMAS, False, 192, attempt)
        item["tool_call_generation"] = first
        try:
            parsed = parse_qwen3_coder_call(first["raw_text"])
        except ValueError as exc:
            item["parsed_call"] = None
            item["handler_execution_ledger"] = [{"status": "not_executed", "reason": str(exc)}]
            item["tool_result"] = None
            item["final_response"] = None
            item["task_score"] = 0
            item["error"] = {"stage": "parse_model_tool_call", "type": "InvalidToolCall", "message": str(exc)}
            return item
        call_id = f"{case_id}_{attempt}".replace("-", "_")
        item["parsed_call"] = {"id": call_id, "name": parsed["name"], "arguments": parsed["arguments"], "raw_xml": parsed["raw_xml"]}
        handler = ALLOWED_HANDLERS.get(parsed["name"])
        if handler is None:
            item["handler_execution_ledger"] = [{"status": "rejected", "reason": "tool name not in local allowlist", "name": parsed["name"]}]
            item["tool_result"] = None
            item["final_response"] = None
            item["task_score"] = 0
            item["error"] = {"stage": "tool_allowlist", "type": "UnknownTool", "message": parsed["name"]}
            return item
        if parsed["arguments"] != expected_args:
            item["handler_execution_ledger"] = [{"status": "rejected", "reason": "arguments did not match operands supplied by request", "name": parsed["name"], "arguments": parsed["arguments"]}]
            item["tool_result"] = None
            item["final_response"] = None
            item["task_score"] = 0
            item["error"] = {"stage": "validate_tool_arguments", "type": "InvalidArguments", "message": "parsed arguments did not match the request operands"}
            return item
        handler_started = time.perf_counter()
        tool_value = handler(**parsed["arguments"])
        handler_elapsed = time.perf_counter() - handler_started
        tool_result = str(tool_value)
        item["handler_execution_ledger"] = [
            {
                "status": "executed",
                "name": parsed["name"],
                "arguments": parsed["arguments"],
                "handler": "pure-local-add_numbers",
                "network_used": False,
                "shell_used": False,
                "eval_used": False,
                "elapsed_seconds": handler_elapsed,
                "result": tool_result,
            }
        ]
        item["tool_result"] = tool_result
        visible, thinking = split_assistant_call(first["raw_text"])
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": visible,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": parsed["name"], "arguments": parsed["arguments"]},
                }
            ],
        }
        if thinking:
            assistant_message["thinking"] = thinking
        messages.extend(
            [
                assistant_message,
                {"role": "tool", "tool_call_id": call_id, "content": tool_result},
            ]
        )
        item["request_history"] = messages
        try:
            second = runner.generate(case_id, "final_response", messages, TOOL_SCHEMAS, False, 128, attempt)
        except Exception as exc:
            item["final_response"] = None
            item["task_score"] = 0
            item["score_details"] = {
                "expected_tool_and_arguments": True,
                "handler_result_correct": tool_value == expected_result,
                "final_answer_contains_result": False,
            }
            item["error"] = stage_error(exc, "final_response")
            return item
        item["final_generation"] = second
        item["final_response"] = second["raw_text"]
        args_match = parsed["name"] == "add_numbers" and parsed["arguments"] == expected_args
        result_match = tool_value == expected_result
        final_match = answer_has_integer(final_surface(second["raw_text"]), expected_result)
        item["task_score"] = int(args_match and result_match and final_match)
        item["score_details"] = {"expected_tool_and_arguments": args_match, "handler_result_correct": result_match, "final_answer_contains_result": final_match}
        return item

    try:
        case["warmup"] = roundtrip("warmup")
        if case["warmup"].get("error"):
            infra_failed = is_infrastructure_error(case["warmup"]["error"])
    except Exception as exc:
        error = stage_error(exc, "warmup")
        case["warmup"] = {"error": error}
        infra_failed = is_infrastructure_error(error)
    for index in range(repeats):
        try:
            row = roundtrip(f"measure-{index + 1}")
            if row.get("error"):
                infra_failed = infra_failed or is_infrastructure_error(row["error"])
        except Exception as exc:
            error = stage_error(exc, "tool_roundtrip")
            row = {"attempt": f"measure-{index + 1}", "task_score": 0, "error": error}
            infra_failed = infra_failed or is_infrastructure_error(error)
        case["repetitions"].append(row)
    scores = [int(row.get("task_score", 0)) for row in case["repetitions"]]
    case["task_score"] = {"passed": sum(scores), "attempted": len(scores), "accuracy": (sum(scores) / len(scores)) if scores else 0.0}
    case["errors"] = []
    if isinstance(case["warmup"], dict) and "error" in case["warmup"]:
        case["errors"].append(case["warmup"]["error"])
    case["errors"].extend(row["error"] for row in case["repetitions"] if "error" in row)
    return case, infra_failed


def require_input(facts: dict[str, Any], name: str) -> Path:
    entry = facts[name]
    if not entry["exists"]:
        raise StageFailure("input_fixture", f"required input fixture is missing: {entry['path']}", "MissingInputFixture")
    return Path(entry["path"])


def input_failure_case(case_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "request": None,
        "warmup": None,
        "repetitions": [],
        "task_score": {"passed": 0, "attempted": 0, "accuracy": 0.0},
        "errors": [stage_error(exc, "input_fixture")],
    }


def run_suite(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[args.model]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / spec["subdir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = model_dir / f"run_{run_id}"
    run_dir.mkdir()
    generation_log = run_dir / "generations.jsonl"
    case_log = run_dir / "cases.jsonl"
    (run_dir / "runtime_errors.jsonl").write_text("", encoding="utf-8")
    facts = input_facts(output_dir)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_utc": utc_now(),
        "mode": args.model,
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "tokenizer_revision": spec["revision"],
        "processor_revision": spec["revision"] if args.model == "vision" else None,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "repeats": args.repeats,
        "dtype": "float16",
        "attention_implementation": "eager",
        "device_required": "NVIDIA Tesla T4, CUDA sm_75",
        "hf_token": False,
        "hf_implicit_token_lookup_disabled": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1",
        "inputs": facts,
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in ("torch", "torchvision", "transformers", "accelerate", "pillow", "av", "torchcodec")},
    }
    write_json(run_dir / "run_metadata.json", metadata)
    all_cases: list[dict[str, Any]] = []
    infra_failed = False
    try:
        import torch

        if not torch.cuda.is_available():
            raise StageFailure("gpu_check", "CUDA is unavailable; CPU fallback is disabled")
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        if "T4" not in device_name.upper() or capability != (7, 5):
            raise StageFailure("gpu_check", f"expected Tesla T4 sm_75, received {device_name} sm_{capability[0]}{capability[1]}")
        metadata["gpu"] = {
            "name": device_name,
            "compute_capability": list(capability),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "torch_cuda_version": torch.version.cuda,
        }
        if args.model == "text":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            processor = AutoTokenizer.from_pretrained(
                spec["model_id"], revision=spec["revision"], trust_remote_code=True, token=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                spec["model_id"],
                revision=spec["revision"],
                dtype=torch.float16,
                device_map="cuda:0",
                attn_implementation="eager",
                trust_remote_code=True,
                token=False,
            )
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                spec["model_id"], revision=spec["revision"], trust_remote_code=True, token=False
            )
            model = AutoModelForImageTextToText.from_pretrained(
                spec["model_id"],
                revision=spec["revision"],
                dtype=torch.float16,
                device_map="cuda:0",
                attn_implementation="eager",
                trust_remote_code=True,
                token=False,
            )
        model.eval()
        first_parameter = next(model.parameters())
        if first_parameter.device.type != "cuda" or first_parameter.dtype != torch.float16:
            raise StageFailure("model_load", f"model is not fully loaded on CUDA FP16: {first_parameter.device} {first_parameter.dtype}")
        metadata["model_runtime"] = {
            "parameter_device": str(first_parameter.device),
            "parameter_dtype": str(first_parameter.dtype),
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "transformers_version": package_version("transformers"),
        }
        write_json(run_dir / "run_metadata.json", metadata)
        runner = Runner(args.model, model, processor, torch, generation_log)

        if args.model == "text":
            request = request_message("12 と 30 を足してください。必ず加算ツールを使ってください。")
            tool_case, failed = run_tool_case(runner, "text_tool_add_numbers_no_think", request, {"a": 12, "b": 30}, 42, args.repeats)
            all_cases.append(tool_case)
            infra_failed |= failed
            append_jsonl(case_log, tool_case)
            japanese_request = request_message("天気のよい日に散歩をすると、気分転換になることがあります。これを短い日本語で言い換えてください。")
            case, failed = run_repeated_simple(runner, "text_japanese_rephrase_no_think", japanese_request, False, 96, args.repeats, japanese_nonempty)
            all_cases.append(case)
            infra_failed |= failed
            append_jsonl(case_log, case)
        else:
            try:
                video_path = require_input(facts, "sequence_video")
                video_request = request_message(
                    "動画の最初に表示される正方形と、最後に表示される正方形の色を順番に答えてください。",
                    {"type": "video", "path": str(video_path)},
                )
                for case_id, thinking, token_limit in (
                    ("vision_video_temporal_no_think", False, 128),
                    ("vision_video_temporal_thinking", True, 512),
                ):
                    case, failed = run_repeated_simple(runner, case_id, video_request, thinking, token_limit, args.repeats, colors_in_order, facts["sequence_video"])
                    all_cases.append(case)
                    infra_failed |= failed
                    append_jsonl(case_log, case)
            except StageFailure as exc:
                case = input_failure_case("vision_video_temporal", exc)
                all_cases.append(case)
                infra_failed = True
                append_jsonl(case_log, case)
                append_jsonl(run_dir / "runtime_errors.jsonl", {"captured_utc": utc_now(), **case["errors"][0]})
            try:
                scene_path = require_input(facts, "scene")
                image_request = request_message(
                    "画像の左から右へ、四角の中にある数字を読み取って足してください。計算には加算ツールを使ってください。",
                    {"type": "image", "path": str(scene_path)},
                )
                case, failed = run_tool_case(runner, "vision_image_addition_tool_no_think", image_request, {"a": 3, "b": 7}, 10, args.repeats, facts["scene"])
                all_cases.append(case)
                infra_failed |= failed
                append_jsonl(case_log, case)
            except StageFailure as exc:
                case = input_failure_case("vision_image_addition_tool_no_think", exc)
                all_cases.append(case)
                infra_failed = True
                append_jsonl(case_log, case)
                append_jsonl(run_dir / "runtime_errors.jsonl", {"captured_utc": utc_now(), **case["errors"][0]})
    except Exception as exc:
        error = stage_error(exc, "model_load")
        if isinstance(exc, StageFailure):
            error["stage"] = exc.stage
        append_jsonl(run_dir / "runtime_errors.jsonl", {"captured_utc": utc_now(), **error, "traceback": traceback.format_exc()})
        infra_failed = True
    metadata["finished_utc"] = utc_now()
    metadata["status"] = "completed_with_infrastructure_errors" if infra_failed else "completed"
    write_json(run_dir / "run_metadata.json", metadata)
    pass_count = sum(case.get("task_score", {}).get("passed", 0) for case in all_cases)
    attempt_count = sum(case.get("task_score", {}).get("attempted", 0) for case in all_cases)
    summary = {
        "run_id": run_id,
        "mode": args.model,
        "model_revision": spec["revision"],
        "status": metadata["status"],
        "infrastructure_failure": infra_failed,
        "task_score": {"passed": pass_count, "attempted": attempt_count, "accuracy": pass_count / attempt_count if attempt_count else 0.0},
        "cases": [{"case_id": case["case_id"], "task_score": case.get("task_score"), "errors": case.get("errors", [])} for case in all_cases],
        "artifacts": {
            "metadata": str(run_dir / "run_metadata.json"),
            "cases_jsonl": str(case_log),
            "generations_jsonl": str(generation_log),
            "runtime_errors_jsonl": str(run_dir / "runtime_errors.jsonl"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if infra_failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/lfm3-full-results"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run_suite(parse_args()))
