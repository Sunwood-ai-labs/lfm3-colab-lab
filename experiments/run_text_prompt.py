from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

from run_video_tools import MODEL_SPECS, Runner, request_message


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one ungraded prompt on the pinned LFM3-1.7B T4 model")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 20:
        parser.error("--repeats must be between 1 and 20")
    prompt_path = args.prompt_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip() or len(prompt) > 16000:
        parser.error("prompt must contain 1 to 16,000 non-whitespace characters")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise RuntimeError(f"Refusing to reuse a text playground run directory: {output_dir}")

    spec = MODEL_SPECS["text"]
    torch = None
    model = None
    processor = None
    try:
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_module
        if importlib.metadata.version("transformers") != "5.21.0":
            raise RuntimeError(f"transformers==5.21.0 is required, got {importlib.metadata.version('transformers')}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("A single Tesla T4 CUDA device is required; CPU fallback is disabled")
        device_name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        if "T4" not in device_name.upper() or capability != (7, 5):
            raise RuntimeError(f"Expected Tesla T4 sm75, got {device_name} {capability}")
        torch.cuda.empty_cache()
        processor = AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["revision"], trust_remote_code=True, token=False,
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
        model.eval()
        parameter = next(model.parameters())
        if parameter.device.type != "cuda" or parameter.dtype != torch.float16:
            raise RuntimeError(f"Model is not loaded on CUDA FP16: {parameter.device} {parameter.dtype}")

        generation_log = output_dir / "generations.jsonl"
        generation_log.write_text("", encoding="utf-8")
        runner = Runner("text", model, processor, torch, generation_log)
        samples = []
        for index in range(1, args.repeats + 1):
            result = runner.generate(
                case_id="user_text_playground",
                turn="answer",
                messages=request_message(prompt),
                tools=None,
                thinking=False,
                max_new_tokens=256,
                attempt=f"sample-{index}",
            )
            answer = str(result.get("raw_text") or "")
            sample = {
                "sample": index,
                "status": "generated" if answer.strip() else "functional_failure",
                "answer": answer,
                "elapsed_seconds": result.get("cuda_synchronized_wall_seconds"),
                "peak_allocated_gib": int(result.get("cuda_peak_allocated_bytes", 0)) / (1024 ** 3),
                "peak_reserved_gib": int(result.get("cuda_peak_reserved_bytes", 0)) / (1024 ** 3),
            }
            samples.append(sample)
            append_jsonl(output_dir / "samples.jsonl", sample)
            print(json.dumps({"sample": index, "status": sample["status"]}), flush=True)
        durations = [float(item["elapsed_seconds"]) for item in samples if item.get("elapsed_seconds") is not None]
        summary = {
            "schema_version": "lfm3-text-playground/v1",
            "status": "generated" if all(item["status"] == "generated" for item in samples) else "functional_failure",
            "graded": False,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "repeats": args.repeats,
            "median_seconds": statistics.median(durations) if durations else None,
            "samples": samples,
        }
        write_json(summary_path, summary)
        return 0
    except Exception as exc:
        write_json(summary_path, {
            "schema_version": "lfm3-text-playground/v1",
            "status": "infrastructure_error",
            "graded": False,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
