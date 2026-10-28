from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

OUT = Path("/content/lfm3-full-results")
AUDIO_SAMPLE_RATE = 24_000
MAX_TEXT_TOKENS = 160
MAX_AUDIO_TOKENS = 256
GENERATION_SEED = 42
GENERATION_PARAMETERS = {
    "audio_temperature": 1.0,
    "audio_top_k": 128,
    "audio_repetition_penalty": 1.1,
    "audio_repetition_window": 4096,
}


@dataclass(frozen=True)
class ModelSpec:
    key: Literal["audio", "omni"]
    repo_id: str
    revision: str
    expected_auto_model: str


@dataclass(frozen=True)
class TaskSpec:
    case_id: str
    task_type: Literal["asr", "tts", "voice_chat", "voice_tool", "translation", "audio_understanding", "image_audio", "video_audio"]
    max_new_tokens: int
    language: str | None = None
    prompt_text: str | None = None
    expected_answer: str | None = None
    synthetic_input: bool = False


MODELS: dict[str, ModelSpec] = {
    "audio": ModelSpec(
        "audio", "LiquidAI/LFM3-Audio-2B",
        "9d49701c372d6011d900f6abb40c6b296fb706dd", "AutoModelForMultimodalLM",
    ),
    "omni": ModelSpec(
        "omni", "LiquidAI/LFM3-Omni-2B",
        "ce2458cbce4ad970f92032ed94e3d18d30b04f9c", "AutoModelForMultimodalLM",
    ),
}

AUDIO_CASES = (
    TaskSpec("tts-ja-blue-question", "tts", MAX_AUDIO_TOKENS, "Japanese",
             "青い四角の中の数字は何ですか？", synthetic_input=True),
    TaskSpec("tts-en-short-phrase", "tts", MAX_AUDIO_TOKENS, "American English",
             "Hello. This is a short English speech synthesis check.", synthetic_input=True),
    TaskSpec("asr-ja-human-reference", "asr", MAX_TEXT_TOKENS, "Japanese", synthetic_input=False),
    TaskSpec("asr-en-synthetic-reference", "asr", MAX_TEXT_TOKENS, "English", synthetic_input=True),
    TaskSpec("translation-ja-en-human-reference", "translation", MAX_TEXT_TOKENS, "Japanese",
             expected_answer="Yachts are not necessarily needed on archipelagos and lakes."),
    TaskSpec("audio-understanding-tone", "audio_understanding", MAX_TEXT_TOKENS,
             expected_answer="An intermittent pure sine tone, with no speech."),
    TaskSpec("voice-chat-math", "voice_chat", MAX_AUDIO_TOKENS, "Japanese", expected_answer="5", synthetic_input=True),
    TaskSpec("voice-chat-math-en", "voice_chat", MAX_AUDIO_TOKENS, "American English", expected_answer="5", synthetic_input=True),
    TaskSpec("voice-tool-add", "voice_tool", MAX_AUDIO_TOKENS, "Japanese", expected_answer="42", synthetic_input=True),
    TaskSpec("voice-tool-add-en", "voice_tool", MAX_AUDIO_TOKENS, "American English", expected_answer="42", synthetic_input=True),
)

OMNI_CASES = (
    TaskSpec("image-audio-blue-digit", "image_audio", MAX_TEXT_TOKENS, expected_answer="7", synthetic_input=True),
    TaskSpec("image-audio-red-digit", "image_audio", MAX_TEXT_TOKENS, expected_answer="3", synthetic_input=True),
    TaskSpec("video-audio-blue-digit", "video_audio", MAX_TEXT_TOKENS, expected_answer="7", synthetic_input=True),
)

SYNTHETIC_FIXTURES = {
    "voice_question_red.wav": ("赤い四角の中の数字は何ですか？", "Japanese"),
    "voice_math_question.wav": ("2足す3はいくつですか？", "Japanese"),
    "voice_tool_request.wav": ("12に30を足してください。計算には利用可能な加算関数を使ってください。", "Japanese"),
    "voice_math_question_en.wav": ("What is two plus three?", "American English"),
    "voice_tool_request_en.wav": ("Please add twelve and thirty. Use the add function.", "American English"),
}

VOICE_CHAT_INPUTS = {
    "voice-chat-math": "voice_math_question.wav",
    "voice-chat-math-en": "voice_math_question_en.wav",
}
VOICE_TOOL_INPUTS = {
    "voice-tool-add": "voice_tool_request.wav",
    "voice-tool-add-en": "voice_tool_request_en.wav",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def relative_path(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_descriptor(path: Path, output_root: Path, provenance: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": relative_path(path, output_root), "exists": path.is_file()}
    if path.is_file():
        result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    else:
        result.update({"bytes": None, "sha256": None})
    if provenance:
        result["provenance"] = provenance
    return result


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "torch", "transformers", "accelerate", "huggingface-hub", "tokenizers",
        "safetensors", "Pillow", "librosa", "soundfile", "numpy", "scipy",
        "imageio-ffmpeg", "torchvision",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_info(torch: Any) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": package_versions().get("transformers"),
        "packages": package_versions(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": int(props.total_memory),
        "cuda_device_count": int(torch.cuda.device_count()),
    }


def verify_runtime(torch: Any) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A single CUDA device is required; CPU fallback is disabled")
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "T4" not in name.upper() or capability != (7, 5):
        raise RuntimeError(f"Expected Tesla T4 sm75, got {name} {capability}")
    version = importlib.metadata.version("transformers")
    if version != "5.21.0":
        raise RuntimeError(f"transformers==5.21.0 is required, got {version}")


def model_file_manifest(snapshot: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        item: dict[str, Any] = {"bytes": path.stat().st_size}
        if path.suffix in {".py", ".json", ".jinja", ".txt"}:
            item["sha256"] = sha256_file(path)
        files[path.relative_to(snapshot).as_posix()] = item
    return files


def load_model(spec: ModelSpec, output_root: Path, torch: Any) -> tuple[Any, Any, Any, Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    start = time.perf_counter()
    snapshot = Path(snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        token=False,
        allow_patterns=(
            "*.json", "*.py", "*.safetensors", "*.jinja", "*.txt", "*.model",
            "*.spm", "*.tiktoken", "*.vocab", "*.merges",
        ),
    ))
    download_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        str(snapshot),
        revision=spec.revision,
        code_revision=spec.revision,
        dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
        token=False,
    ).to("cuda:0", dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(
        str(snapshot),
        revision=spec.revision,
        code_revision=spec.revision,
        trust_remote_code=True,
        token=False,
    )
    codec = getattr(processor, "audio_tokenizer", None)
    if codec is None:
        raise RuntimeError("The pinned processor did not load audio_tokenizer")
    codec.to("cuda:0", dtype=torch.float16).eval()
    load_seconds = time.perf_counter() - start

    for label, module in (("model", model), ("audio_tokenizer", codec)):
        bad = [
            (str(parameter.device), str(parameter.dtype))
            for parameter in module.parameters()
            if parameter.is_floating_point()
            and (parameter.device.type != "cuda" or parameter.dtype != torch.float16)
        ]
        if bad:
            raise RuntimeError(f"{label} has non-CUDA-FP16 parameters: {bad[:3]}")

    load_facts = {
        "snapshot_path": str(snapshot),
        "download_seconds": download_seconds,
        "load_seconds": load_seconds,
        "model_source_files": model_file_manifest(snapshot),
        "dtype": "float16",
        "attention": "eager",
        "token": False,
        "trust_remote_code": True,
        "auto_model": spec.expected_auto_model,
    }
    return model, processor, codec, snapshot, load_facts


def move_batch(encoded: Any, torch: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to("cuda:0", dtype=torch.float16)
            else:
                moved[key] = value.to("cuda:0")
        else:
            moved[key] = value
    input_ids = moved.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Processor must return integer input_ids")
    for key, value in moved.items():
        if isinstance(value, torch.Tensor) and value.device.type != "cuda":
            raise RuntimeError(f"Encoded tensor {key} is not on CUDA")
    return moved


def tensor_shapes(encoded: dict[str, Any], torch: Any) -> dict[str, Any]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def add_memory_stats(record: dict[str, Any], torch: Any) -> None:
    torch.cuda.synchronize()
    record["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
    record["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))


def make_inputs(
    processor: Any,
    conversation: list[dict[str, Any]],
    torch: Any,
    *,
    continue_final_message: bool = False,
    tools: list[dict[str, Any]] | None = None,
    processor_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if continue_final_message:
        kwargs["continue_final_message"] = True
    else:
        kwargs["add_generation_prompt"] = True
    if tools is not None:
        kwargs["tools"] = tools
    if processor_kwargs:
        kwargs.update(processor_kwargs)
    encoded = processor.apply_chat_template(conversation, **kwargs)
    return move_batch(encoded, torch)


def sequence_text(outputs: Any, encoded: dict[str, Any], processor: Any) -> tuple[str, int]:
    sequence = outputs.sequences[0]
    start = int(encoded["input_ids"].shape[-1])
    lengths = getattr(outputs, "sequence_lengths", None)
    end = int(lengths[0]) if lengths is not None else int(sequence.shape[-1])
    end = max(start, min(end, int(sequence.shape[-1])))
    generated = sequence[start:end]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    return text, int(generated.numel())


def audio_codes(outputs: Any) -> Any:
    codes = getattr(outputs, "audio_codes", None)
    lengths = getattr(outputs, "audio_lengths", None)
    if codes is None or lengths is None:
        return None
    valid = int(lengths[0])
    if valid <= 0:
        return None
    return codes[:1, :, :valid]


def decode_full_audio(outputs: Any, codec: Any, torch: Any) -> tuple[Any | None, float, int]:
    codes = audio_codes(outputs)
    if codes is None:
        return None, 0.0, 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = codec.decode(codes).audio_values[0].squeeze()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return decoded.detach().float().cpu().numpy(), seconds, int(codes.shape[-1])


def write_wave(path: Path, waveform: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Generated audio must contain finite samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate)


def inspect_wave(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    finite = bool(np.isfinite(samples).all())
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    duration = float(samples.shape[0] / sample_rate) if sample_rate else 0.0
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(samples.shape[1]),
        "sample_count": int(samples.shape[0]),
        "duration_seconds": duration,
        "peak_amplitude": peak,
        "rms": rms,
        "finite": finite,
        "non_silent": bool(finite and peak > 1e-5 and rms > 1e-5 and duration > 0.05),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in value if not char.isspace())


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1]


def character_error_rate(hypothesis: str, reference: str) -> dict[str, Any]:
    expected = normalize_text(reference)
    actual = normalize_text(hypothesis)
    errors = edit_distance(actual, expected)
    return {
        "cer": errors / max(1, len(expected)),
        "character_errors": errors,
        "reference_characters": len(expected),
        "normalization": "NFKC, casefold, remove whitespace; punctuation retained",
    }


def load_reference(path: Path) -> tuple[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, str):
        return value, {"transcript_key": "<root-string>"}
    if not isinstance(value, dict):
        raise ValueError("speech_reference.json must be a string or object")
    for key in ("ground_truth", "ground_truth_text", "transcript", "transcription", "text", "sentence"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            provenance = value.get("provenance", value.get("source", value.get("dataset")))
            return candidate, {"transcript_key": key, "provenance": provenance}
    raise ValueError("speech_reference.json has no non-empty transcript field")


def numeric_answer_match(text: str, expected: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    numbers = re.findall(r"(?<![0-9.,+-])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9]|[.,][0-9])", normalized)
    if any(Decimal(number.replace(",", "")) == Decimal(expected) for number in numbers):
        return True
    aliases = {
        "3": ("three", "三"),
        "5": ("five", "五"),
        "7": ("seven", "七"),
        "42": ("forty-two", "forty two", "四十二"),
    }.get(expected, ())
    for alias in aliases:
        boundary = "A-Za-z" if re.search(r"[A-Za-z]", alias) else "〇零一二三四五六七八九十百千万億兆点0-9."
        if re.search(rf"(?<![{boundary}]){re.escape(alias)}(?![{boundary}])", normalized):
            return True
    return False


def unambiguous_digit_answer(text: str, expected: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    numbers = re.findall(r"(?<![0-9.,+-])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9]|[.,][0-9])", normalized)
    if any(Decimal(value.replace(",", "")) != Decimal(expected) for value in numbers):
        return False
    for other in ("3", "7"):
        if other != expected and numeric_answer_match(text, other):
            return False
    return numeric_answer_match(text, expected)


def parse_tool_calls(raw_text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw_text, re.DOTALL)
    if len(blocks) != 1 or raw_text.count("<tool_call>") != 1:
        raise ValueError("Exactly one complete tool call is required")
    function = re.fullmatch(r"\s*<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>\s*", blocks[0], re.DOTALL)
    if function is None:
        raise ValueError("Invalid function call shape")
    name, body = function.groups()
    arguments = {}
    cursor = 0
    for match in re.finditer(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
        key, value = match.groups()
        if body[cursor:match.start()].strip() or key in arguments or not re.fullmatch(r"[-+]?\d+", value.strip()):
            raise ValueError("Invalid or duplicate tool argument")
        arguments[key] = int(value.strip())
        cursor = match.end()
    if body[cursor:].strip() or set(arguments) != {"a", "b"}:
        raise ValueError("Tool requires exactly a and b")
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def generation_kwargs(spec: ModelSpec, max_new_tokens: int, torch: Any, **extra: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
    }
    if spec.key == "audio":
        kwargs.update(GENERATION_PARAMETERS)
    else:
        kwargs["generation_mode"] = "text"
    kwargs.update(extra)
    return kwargs


def generate(
    model: Any,
    spec: ModelSpec,
    encoded: dict[str, Any],
    max_new_tokens: int,
    torch: Any,
    **extra: Any,
) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            **generation_kwargs(spec, max_new_tokens, torch, **extra),
        )
    torch.cuda.synchronize()
    return outputs, time.perf_counter() - started


class CodecAudioStreamer:
    def __init__(self, codec: Any, torch: Any, generation_started: float) -> None:
        self.codec = codec
        self.torch = torch
        codec_module = importlib.import_module(codec.__class__.__module__)
        cache_type = getattr(codec_module, "LiquidCodecStreamingCache", None)
        if cache_type is None:
            raise RuntimeError("Pinned LiquidCodec source has no LiquidCodecStreamingCache")
        self.cache = cache_type(codec.config)
        self.generation_started = generation_started
        self.wave_chunks: list[Any] = []
        self.boundaries: list[dict[str, Any]] = []
        self.frame_count = 0
        self.first_chunk_latency: float | None = None
        self.codec_seconds = 0.0
        self.ended = False

    def put(self, value: Any) -> None:
        import numpy as np

        self.torch.cuda.synchronize()
        arrival = time.perf_counter()
        if self.first_chunk_latency is None:
            self.first_chunk_latency = arrival - self.generation_started
        codes = value if isinstance(value, self.torch.Tensor) else self.torch.as_tensor(value)
        if codes.ndim != 1 or codes.numel() != int(self.codec.config.n_q):
            raise ValueError(f"audio_streamer frame shape must be ({self.codec.config.n_q},), got {tuple(codes.shape)}")
        codes = codes.to(device="cuda:0", dtype=self.torch.long).reshape(1, -1, 1)
        started = time.perf_counter()
        with self.torch.inference_mode():
            decoded = self.codec.decode(codes, streaming_cache=self.cache, finalize=False).audio_values[0]
        self.torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        self.codec_seconds += decode_seconds
        samples = decoded.detach().float().cpu().numpy().reshape(-1)
        if samples.size:
            self.wave_chunks.append(samples)
        self.frame_count += 1
        self.boundaries.append({
            "frame_index": self.frame_count,
            "arrival_seconds": arrival - self.generation_started,
            "codec_decode_seconds": decode_seconds,
            "decoded_sample_count": int(samples.size),
        })

    def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        if not self.frame_count:
            return
        empty = self.torch.empty(
            (1, int(self.codec.config.n_q), 0), dtype=self.torch.long, device="cuda:0"
        )
        started = time.perf_counter()
        with self.torch.inference_mode():
            decoded = self.codec.decode(empty, streaming_cache=self.cache, finalize=True).audio_values[0]
        self.torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        self.codec_seconds += decode_seconds
        samples = decoded.detach().float().cpu().numpy().reshape(-1)
        if samples.size:
            self.wave_chunks.append(samples)
        self.boundaries.append({
            "frame_index": self.frame_count,
            "arrival_seconds": time.perf_counter() - self.generation_started,
            "codec_decode_seconds": decode_seconds,
            "decoded_sample_count": int(samples.size),
            "finalize": True,
        })

    def waveform(self) -> Any:
        import numpy as np

        return np.concatenate(self.wave_chunks) if self.wave_chunks else np.empty((0,), dtype=np.float32)


def base_record(
    spec: ModelSpec,
    script_sha: str,
    run_id: str,
    task: TaskSpec,
    repeat_index: int | None,
    status: str,
    output_root: Path,
    input_paths: list[tuple[Path, str | None]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": spec.key,
        "model_id": spec.repo_id,
        "revision": spec.revision,
        "code_revision": spec.revision,
        "case_id": task.case_id,
        "task_type": task.task_type,
        "repeat_index": repeat_index,
        "status": status,
        "expected": task.expected_answer if task.expected_answer is not None else task.prompt_text,
        "raw_output": None,
        "generated_tokens": None,
        "max_new_tokens": task.max_new_tokens,
        "input_files": [file_descriptor(path, output_root, provenance) for path, provenance in input_paths],
        "output_files": [],
        "input_tensor_shapes": {},
        "timing_seconds": {},
        "peak_vram_allocated_bytes": None,
        "peak_vram_reserved_bytes": None,
        "score": {},
        "failure_class": None,
        "error_stage": None,
        "error_type": None,
        "error_message": None,
        "script_sha256": script_sha,
    }


def add_output_file(record: dict[str, Any], path: Path, output_root: Path, details: dict[str, Any]) -> None:
    record["output_files"].append({
        "path": relative_path(path, output_root),
        **details,
    })


def copy_fixture(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def case_error(
    spec: ModelSpec,
    script_sha: str,
    run_id: str,
    task: TaskSpec,
    repeat_index: int | None,
    output_root: Path,
    input_paths: list[tuple[Path, str | None]],
    stage: str,
    error: Exception,
    *,
    functional: bool = False,
) -> dict[str, Any]:
    row = base_record(spec, script_sha, run_id, task, repeat_index, "failed" if functional else "error", output_root, input_paths)
    row.update({
        "failure_class": "functional" if functional else "infrastructure",
        "error_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
    })
    return row


def synthesize(
    model: Any,
    processor: Any,
    codec: Any,
    torch: Any,
    spec: ModelSpec,
    text: str,
    language: str,
    output_path: Path,
) -> tuple[Any, str, float, float, int]:
    conversation = [
        {"role": "system", "content": f"Perform TTS in {language} using a female voice."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "<|audio_start|>"},
    ]
    encoded = make_inputs(processor, conversation, torch, continue_final_message=True)
    outputs, generation_seconds = generate(
        model, spec, encoded, MAX_AUDIO_TOKENS, torch,
    )
    spoken, generated_tokens = sequence_text(outputs, encoded, processor)
    waveform, codec_seconds, frame_count = decode_full_audio(outputs, codec, torch)
    if waveform is None:
        raise ValueError("TTS returned no audio codes")
    write_wave(output_path, waveform, AUDIO_SAMPLE_RATE)
    return outputs, spoken, generation_seconds, codec_seconds, generated_tokens


def audio_case_input_paths(task: TaskSpec, output_root: Path, reference_info: dict[str, Any] | None = None) -> list[tuple[Path, str | None]]:
    inputs = output_root / "inputs"
    if task.case_id in {"asr-ja-human-reference", "translation-ja-en-human-reference"}:
        provenance = None if reference_info is None else json.dumps(reference_info, ensure_ascii=False)
        return [(inputs / "speech_reference.wav", provenance), (inputs / "speech_reference.json", provenance)]
    if task.case_id == "audio-understanding-tone":
        return [
            (inputs / "tone.wav", "synthetic_self_consistency: intermittent pure sine tone; no speech"),
            (inputs / "tone.json", "synthetic signal specification, if provided"),
        ]
    if task.case_id == "asr-en-synthetic-reference":
        return [(inputs / "voice_english_reference.wav", "synthetic_self_consistency: generated by LFM3-Audio-2B")]
    if task.case_id in VOICE_CHAT_INPUTS:
        return [(inputs / VOICE_CHAT_INPUTS[task.case_id], "synthetic_self_consistency: generated by LFM3-Audio-2B")]
    if task.case_id in VOICE_TOOL_INPUTS:
        return [(inputs / VOICE_TOOL_INPUTS[task.case_id], "synthetic_self_consistency: generated by LFM3-Audio-2B")]
    if task.case_id == "tts-ja-blue-question":
        return []
    if task.case_id == "tts-en-short-phrase":
        return []
    return []


def run_audio_task(
    task: TaskSpec,
    repeat_index: int,
    model: Any,
    processor: Any,
    codec: Any,
    torch: Any,
    spec: ModelSpec,
    output_root: Path,
    run_dir: Path,
    run_id: str,
    script_sha: str,
) -> dict[str, Any]:
    import soundfile as sf

    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generated_audio_path: Path | None = None
    reference_info: dict[str, Any] | None = None

    if task.case_id in {"asr-ja-human-reference", "translation-ja-en-human-reference"}:
        audio_path = output_root / "inputs" / "speech_reference.wav"
        reference_path = output_root / "inputs" / "speech_reference.json"
        input_paths = audio_case_input_paths(task, output_root)
        if not audio_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError("speech_reference.wav and speech_reference.json are required")
        reference_text, reference_info = load_reference(reference_path)
        input_paths = audio_case_input_paths(task, output_root, reference_info)
    elif task.case_id == "audio-understanding-tone":
        audio_path = output_root / "inputs" / "tone.wav"
        input_paths = audio_case_input_paths(task, output_root)
        if not audio_path.is_file():
            raise FileNotFoundError("Synthetic tone input is missing")
    elif task.case_id == "asr-en-synthetic-reference":
        audio_path = output_root / "inputs" / "voice_english_reference.wav"
        input_paths = audio_case_input_paths(task, output_root)
        if not audio_path.is_file():
            raise FileNotFoundError("English synthetic TTS input is missing")
        reference_text = "Hello. This is a short English speech synthesis check."
    elif task.case_id in VOICE_CHAT_INPUTS:
        audio_path = output_root / "inputs" / VOICE_CHAT_INPUTS[task.case_id]
        input_paths = audio_case_input_paths(task, output_root)
        if not audio_path.is_file():
            raise FileNotFoundError("Synthetic math question audio is missing")
    elif task.case_id in VOICE_TOOL_INPUTS:
        audio_path = output_root / "inputs" / VOICE_TOOL_INPUTS[task.case_id]
        input_paths = audio_case_input_paths(task, output_root)
        if not audio_path.is_file():
            raise FileNotFoundError("Synthetic tool request audio is missing")
    else:
        input_paths = audio_case_input_paths(task, output_root)

    record = base_record(spec, script_sha, run_id, task, repeat_index, "passed", output_root, input_paths)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    if task.task_type == "tts":
        assert task.prompt_text is not None and task.language is not None
        generated_audio_path = artifact_dir / f"{task.case_id}-r{repeat_index}.wav"
        _, spoken, generation_seconds, codec_seconds, generated_tokens = synthesize(
            model, processor, codec, torch, spec, task.prompt_text, task.language, generated_audio_path,
        )
        wave_info = inspect_wave(generated_audio_path)
        record["raw_output"] = spoken
        record["generated_tokens"] = generated_tokens
        record["score"] = {"waveform_valid": wave_info["finite"], "audible_nonzero_waveform": wave_info["non_silent"]}
        record["timing_seconds"] = {
            "model_generation": generation_seconds,
            "codec_decode": codec_seconds,
            "generation_plus_codec": generation_seconds + codec_seconds,
        }
        add_output_file(record, generated_audio_path, output_root, {"audio": wave_info, "sample_rate_hz": AUDIO_SAMPLE_RATE})
        if task.case_id == "tts-ja-blue-question" and repeat_index == 1:
            canonical = output_root / "inputs" / "voice_question.wav"
            copy_fixture(generated_audio_path, canonical)
            add_output_file(record, canonical, output_root, {
                "role": "synthetic_question_for_omni",
                "provenance": "synthetic_self_consistency: generated by LFM3-Audio-2B; not independent speech-quality evidence",
                **inspect_wave(canonical),
            })
        if task.case_id == "tts-en-short-phrase" and repeat_index == 1:
            canonical = output_root / "inputs" / "voice_english_reference.wav"
            copy_fixture(generated_audio_path, canonical)
            add_output_file(record, canonical, output_root, {
                "role": "synthetic_english_asr_input",
                "provenance": "synthetic_self_consistency: generated by LFM3-Audio-2B",
                **inspect_wave(canonical),
            })
        if not wave_info["non_silent"]:
            record.update({"status": "failed", "failure_class": "functional", "error_stage": "audio_quality", "error_type": "silent_or_invalid_audio", "error_message": "TTS waveform is empty, non-finite, too short, or silent"})
        add_memory_stats(record, torch)
        return record

    if task.task_type in {"asr", "translation", "audio_understanding"}:
        if task.task_type == "asr":
            system_prompt = f"Perform ASR in {task.language}."
            user_content: list[dict[str, Any]] = [{"type": "audio", "audio": str(audio_path)}]
        elif task.task_type == "translation":
            system_prompt = "Please translate the following spoken Japanese into English text."
            user_content = [{"type": "audio", "audio": str(audio_path)}]
        else:
            system_prompt = "Answer questions about the provided audio clip. Base every answer only on what can be heard."
            user_content = [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": "Describe the sounds you hear."},
            ]
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        encoded = make_inputs(processor, conversation, torch)
        outputs, generation_seconds = generate(model, spec, encoded, task.max_new_tokens, torch, suppress_mixed=True, suppress_audio_out=True)
        raw, generated_tokens = sequence_text(outputs, encoded, processor)
        if task.task_type == "asr":
            score = character_error_rate(raw, reference_text)
            score["evaluation_basis"] = "human_reference" if task.case_id == "asr-ja-human-reference" else "synthetic_self_consistency"
            record["expected"] = reference_text
        elif task.task_type == "translation":
            score = {
                "manual_review_required": True,
                "source_transcript": reference_text,
                "expected_meaning": task.expected_answer,
                "evaluation_basis": "human_reference",
            }
        else:
            score = {
                "manual_review_required": True,
                "expected_sound": task.expected_answer,
                "evaluation_basis": "synthetic_signal_specification",
            }
        record["raw_output"] = raw
        record["generated_tokens"] = generated_tokens
        record["score"] = score
        record["input_tensor_shapes"] = tensor_shapes(encoded, torch)
        record["timing_seconds"] = {"model_generation": generation_seconds, "codec_decode": 0.0, "generation_plus_codec": generation_seconds}
        record["status"] = "completed" if raw else "failed"
        if not raw:
            record.update({"failure_class": "functional", "error_stage": "audio_text_output", "error_type": "empty_output", "error_message": "Audio task generated no text"})
        add_memory_stats(record, torch)
        return record

    if task.case_id in VOICE_CHAT_INPUTS:
        conversation = [
            {"role": "system", "content": "Respond with interleaved text and audio. Answer the spoken arithmetic question directly, and speak the answer."},
            {"role": "user", "content": [{"type": "audio", "audio": str(audio_path)}]},
            {"role": "assistant", "content": "<|mixed_start|>"},
        ]
        encoded = make_inputs(processor, conversation, torch, continue_final_message=True)
        generation_started = time.perf_counter()
        streamer = CodecAudioStreamer(codec, torch, generation_started)
        outputs, generation_seconds = generate(model, spec, encoded, task.max_new_tokens, torch, audio_streamer=streamer)
        raw, generated_tokens = sequence_text(outputs, encoded, processor)
        waveform, codec_seconds, frame_count = decode_full_audio(outputs, codec, torch)
        if waveform is not None:
            generated_audio_path = artifact_dir / f"{task.case_id}-full-r{repeat_index}.wav"
            write_wave(generated_audio_path, waveform, AUDIO_SAMPLE_RATE)
            full_info = inspect_wave(generated_audio_path)
            add_output_file(record, generated_audio_path, output_root, {"audio": full_info, "sample_rate_hz": AUDIO_SAMPLE_RATE, "decode_path": "full_file"})
        else:
            full_info = {"non_silent": False, "finite": False}
        streamed = streamer.waveform()
        streamed_path = artifact_dir / f"{task.case_id}-streamed-r{repeat_index}.wav"
        if streamed.size:
            write_wave(streamed_path, streamed, AUDIO_SAMPLE_RATE)
            stream_info = inspect_wave(streamed_path)
            add_output_file(record, streamed_path, output_root, {"audio": stream_info, "sample_rate_hz": AUDIO_SAMPLE_RATE, "decode_path": "audio_streamer_plus_incremental_codec"})
        else:
            stream_info = {"non_silent": False, "finite": False}
        stream_equal: bool | None = None
        max_abs_diff: float | None = None
        waveform_length_matches: bool | None = None
        if waveform is not None and streamed.size:
            import numpy as np
            waveform_length_matches = len(waveform) == len(streamed)
            shared_length = min(len(waveform), len(streamed))
            if shared_length:
                max_abs_diff = float(np.max(np.abs(np.asarray(waveform[:shared_length]) - streamed[:shared_length])))
                stream_equal = bool(waveform_length_matches and max_abs_diff <= 1e-4)
        record["raw_output"] = raw
        record["generated_tokens"] = generated_tokens
        record["input_tensor_shapes"] = tensor_shapes(encoded, torch)
        record["timing_seconds"] = {
            "model_generation_wall_including_stream_decode": generation_seconds,
            "stream_codec_incremental_decode": streamer.codec_seconds,
            "full_codec_decode": codec_seconds,
            "generation_plus_full_codec": generation_seconds + codec_seconds,
            "first_audio_frame_latency_from_generate_start": streamer.first_chunk_latency,
        }
        record["streaming"] = {
            "api": "model.generate(audio_streamer=BaseStreamer)",
            "incremental_codec_decode": True,
            "frame_count": streamer.frame_count,
            "boundaries": streamer.boundaries,
            "streamed_waveform_non_silent": stream_info["non_silent"],
            "waveform_length_matches_full": waveform_length_matches,
            "full_vs_streamed_equal_within_1e-4": stream_equal,
            "full_vs_streamed_max_abs_diff_over_shared_length": max_abs_diff,
            "full_vs_streamed_comparison_note": "Waveforms are independently decoded in FP16; numeric difference is diagnostic and is not a semantic pass/fail criterion.",
        }
        answer_ok = numeric_answer_match(raw, task.expected_answer or "5")
        audio_ok = bool(
            full_info.get("finite") and full_info.get("non_silent") and full_info.get("sample_count", 0) > 0
            and stream_info.get("finite") and stream_info.get("non_silent") and stream_info.get("sample_count", 0) > 0
            and streamer.frame_count > 0 and frame_count > 0
        )
        record["score"] = {
            "numeric_answer_correct": answer_ok,
            "text_present": bool(raw),
            "spoken_audio_valid": audio_ok,
            "stream_decode_matches_full_within_1e-4": stream_equal,
            "stream_decode_length_matches_full": waveform_length_matches,
            "stream_decode_max_abs_diff_over_shared_length": max_abs_diff,
        }
        if not answer_ok or not raw or not audio_ok:
            record.update({"status": "failed", "failure_class": "functional", "error_stage": "voice_chat_score", "error_type": "answer_or_audio_check_failed", "error_message": "Voice chat must return the expected arithmetic answer and valid non-silent full and streamed speech"})
        add_memory_stats(record, torch)
        return record

    if task.case_id in VOICE_TOOL_INPUTS:
        tools = [{
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers and return the sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }]
        conversation = [
            {"role": "system", "content": "Respond with function calls, and interleaved text and audio.\n\nList of tools:\nAfter receiving tool results, answer with the result in both text and spoken audio."},
            {"role": "user", "content": [{"type": "audio", "audio": str(audio_path)}]},
        ]
        encoded1 = make_inputs(processor, conversation, torch, tools=tools)
        outputs1, seconds1 = generate(model, spec, encoded1, 128, torch)
        tool_call_text, tokens1 = sequence_text(outputs1, encoded1, processor)
        try:
            parsed_calls = parse_tool_calls(tool_call_text)
        except ValueError:
            parsed_calls = []
        if not parsed_calls:
            record["raw_output"] = tool_call_text
            record["generated_tokens"] = tokens1
            record["score"] = {"tool_call_emitted": False, "tool_executed": False}
            record["timing_seconds"] = {"model_generation_round_1": seconds1, "generation_plus_codec": seconds1}
            record.update({"status": "failed", "failure_class": "functional", "error_stage": "tool_call_parse", "error_type": "tool_call_missing", "error_message": "Model did not emit a parseable add function call"})
            add_memory_stats(record, torch)
            return record
        first = parsed_calls[0]["function"]
        args = first["arguments"]
        tool_ok = first["name"] == "add" and len(parsed_calls) == 1 and args == {"a": 12, "b": 30} and all(type(args[k]) is int for k in ("a", "b")) and all(-1000 <= args[k] <= 1000 for k in ("a", "b"))
        if tool_ok:
            tool_result = {"result": args["a"] + args["b"]}
            conversation.extend([
                {"role": "assistant", "tool_calls": parsed_calls},
                {"role": "tool", "content": json.dumps([tool_result], ensure_ascii=False)},
            ])
            encoded2 = make_inputs(processor, conversation, torch, tools=tools)
            outputs2, seconds2 = generate(model, spec, encoded2, task.max_new_tokens, torch)
            final_text, tokens2 = sequence_text(outputs2, encoded2, processor)
            waveform, codec_seconds, frame_count = decode_full_audio(outputs2, codec, torch)
            if waveform is not None:
                generated_audio_path = artifact_dir / f"{task.case_id}-r{repeat_index}.wav"
                write_wave(generated_audio_path, waveform, AUDIO_SAMPLE_RATE)
                wave_info = inspect_wave(generated_audio_path)
                add_output_file(record, generated_audio_path, output_root, {"audio": wave_info, "sample_rate_hz": AUDIO_SAMPLE_RATE})
            else:
                wave_info = {"non_silent": False, "finite": False}
            record["raw_output"] = {"tool_call": tool_call_text, "tool_result": tool_result, "final_text": final_text}
            record["generated_tokens"] = tokens1 + tokens2
            record["timing_seconds"] = {
                "model_generation_round_1": seconds1,
                "model_generation_round_2": seconds2,
                "codec_decode": codec_seconds,
                "generation_plus_codec": seconds1 + seconds2 + codec_seconds,
            }
            answer_ok = numeric_answer_match(final_text, task.expected_answer or "42")
            record["score"] = {
            "tool_call_emitted": True,
            "tool_name_and_arguments_correct": args == {"a": 12, "b": 30},
                "sandbox_result": tool_result,
                "next_response_contains_42": answer_ok,
                "spoken_audio_valid": bool(wave_info.get("non_silent")),
                "output_audio_frames": frame_count,
            }
            if args != {"a": 12, "b": 30} or not answer_ok or not final_text or not wave_info.get("non_silent"):
                record.update({"status": "failed", "failure_class": "functional", "error_stage": "tool_response_score", "error_type": "tool_response_check_failed", "error_message": "Tool request must execute add(12, 30), return 42, and produce spoken audio"})
        else:
            record["raw_output"] = tool_call_text
            record["generated_tokens"] = tokens1
            record["score"] = {"tool_call_emitted": True, "tool_name_and_arguments_correct": False, "parsed_tool_calls": parsed_calls, "tool_executed": False}
            record["timing_seconds"] = {"model_generation_round_1": seconds1, "generation_plus_codec": seconds1}
            record.update({"status": "failed", "failure_class": "functional", "error_stage": "tool_arguments", "error_type": "invalid_sandbox_request", "error_message": "Only one bounded add(a,b) request with integer arguments is accepted"})
        add_memory_stats(record, torch)
        return record

    raise ValueError(f"Unsupported audio task {task.case_id}")


def generate_audio_fixture(
    filename: str,
    text: str,
    language: str,
    model: Any,
    processor: Any,
    codec: Any,
    torch: Any,
    spec: ModelSpec,
    output_root: Path,
    run_dir: Path,
    run_id: str,
    script_sha: str,
) -> dict[str, Any]:
    task = TaskSpec(f"tts-fixture-{Path(filename).stem}", "tts", MAX_AUDIO_TOKENS, language, text, synthetic_input=True)
    artifact_path = run_dir / "artifacts" / filename
    record = base_record(spec, script_sha, run_id, task, 1, "passed", output_root, [])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    _, spoken, generation_seconds, codec_seconds, generated_tokens = synthesize(
        model, processor, codec, torch, spec, text, language, artifact_path,
    )
    wave_info = inspect_wave(artifact_path)
    canonical = output_root / "inputs" / filename
    copy_fixture(artifact_path, canonical)
    record["raw_output"] = spoken
    record["generated_tokens"] = generated_tokens
    record["score"] = {
        "waveform_valid": wave_info["finite"],
        "audible_nonzero_waveform": wave_info["non_silent"],
        "evaluation_basis": "synthetic_self_consistency",
    }
    record["timing_seconds"] = {
        "model_generation": generation_seconds,
        "codec_decode": codec_seconds,
        "generation_plus_codec": generation_seconds + codec_seconds,
    }
    add_output_file(record, artifact_path, output_root, {"audio": wave_info, "sample_rate_hz": AUDIO_SAMPLE_RATE})
    add_output_file(record, canonical, output_root, {
        "role": "synthetic_question_for_audio_suite",
        "provenance": "generated by LFM3-Audio-2B; not independent speech-quality evidence",
        **inspect_wave(canonical),
    })
    if not wave_info["non_silent"]:
        record.update({"status": "failed", "failure_class": "functional", "error_stage": "fixture_audio_quality", "error_type": "silent_or_invalid_audio", "error_message": "Generated fixture audio is empty, non-finite, too short, or silent"})
    add_memory_stats(record, torch)
    return record


def video_with_audio(video_path: Path, audio_path: Path, destination: Path) -> dict[str, Any]:
    import soundfile as sf
    import imageio_ffmpeg

    if not video_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError(f"video/audio fixture is missing: {video_path}, {audio_path}")
    duration = float(sf.info(audio_path).duration)
    duration = max(3.0, duration + 0.5)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-stream_loop", "-1", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-t", f"{duration:.3f}", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(destination),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return {
        "ffmpeg_path": ffmpeg,
        "source_video": str(video_path),
        "source_audio": str(audio_path),
        "mux_duration_seconds": duration,
        "video_and_audio_decode_validation": "passed",
        "probe_stderr": probe.stderr,
    }


def run_omni_task(
    task: TaskSpec,
    repeat_index: int,
    model: Any,
    processor: Any,
    torch: Any,
    spec: ModelSpec,
    output_root: Path,
    run_dir: Path,
    run_id: str,
    script_sha: str,
    video_path: Path | None,
    video_prep: dict[str, Any] | None,
) -> dict[str, Any]:
    inputs = output_root / "inputs"
    image_path = inputs / "scene.png"
    question_name = "voice_question_red.wav" if task.case_id == "image-audio-red-digit" else "voice_question.wav"
    question_path = inputs / question_name
    if task.task_type == "image_audio":
        media_path = image_path
        input_paths = [(image_path, "fixture: red 3 at left, blue 7 at right"), (question_path, "synthetic_self_consistency: generated by LFM3-Audio-2B")]
    else:
        media_path = video_path or (inputs / "scene_video.mp4")
        input_paths = [(media_path, "video input with generated spoken question muxed into the audio track"), (question_path, "synthetic_self_consistency: generated by LFM3-Audio-2B")]
        if video_prep is None:
            raise RuntimeError("Video+audio mux was not prepared")
    record = base_record(spec, script_sha, run_id, task, repeat_index, "passed", output_root, input_paths)
    if task.task_type == "image_audio":
        media = [
            {"type": "image", "image": str(media_path)},
            {"type": "audio", "audio": str(question_path)},
        ]
        system = "Answer the spoken question using both the image and audio. Respond with a concise text answer."
    else:
        media = [{"type": "video", "video": str(media_path)}]
        system = "Answer questions about the video using what you see and hear. Respond with a concise text answer."
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": media},
    ]
    processor_kwargs = None
    if task.task_type == "video_audio":
        processor_kwargs = {
            "load_audio_from_video": True,
            "processor_kwargs": {"use_audio_in_video": True},
        }
    encoded = make_inputs(processor, conversation, torch, processor_kwargs=processor_kwargs)
    expected_keys = ("pixel_values", "input_features") if task.task_type == "image_audio" else ("pixel_values_videos", "input_features")
    present = {key: key in encoded for key in expected_keys}
    if not all(present.values()):
        raise RuntimeError(f"Processor did not return both required modalities: {present}; keys={list(encoded)}")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    outputs, generation_seconds = generate(model, spec, encoded, task.max_new_tokens, torch)
    raw, generated_tokens = sequence_text(outputs, encoded, processor)
    expected_digit = task.expected_answer or "7"
    correct = unambiguous_digit_answer(raw, expected_digit)
    record["raw_output"] = raw
    record["generated_tokens"] = generated_tokens
    record["input_tensor_shapes"] = tensor_shapes(encoded, torch)
    record["observed_input_modalities"] = list(present)
    record["video_mux"] = video_prep if task.task_type == "video_audio" else None
    record["timing_seconds"] = {"model_generation": generation_seconds, "codec_decode": 0.0, "generation_plus_codec": generation_seconds}
    record["score"] = {"expected_digit": expected_digit, "expected_digit_present": numeric_answer_match(raw, expected_digit), "unambiguous_expected_digit": correct, "both_modalities_encoded": all(present.values())}
    if not raw or not correct:
        record.update({"status": "failed", "failure_class": "functional", "error_stage": "cross_modal_answer_score", "error_type": "wrong_or_ambiguous_digit", "error_message": f"Cross-modal answer must identify only the requested digit {expected_digit}, without competing answers"})
    add_memory_stats(record, torch)
    return record


def model_run(spec: ModelSpec, output_root: Path, repeats: int) -> int:
    import torch
    import numpy as np

    random.seed(GENERATION_SEED)
    np.random.seed(GENERATION_SEED)
    torch.manual_seed(GENERATION_SEED)
    torch.cuda.manual_seed_all(GENERATION_SEED)
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / spec.key
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = model_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    script_sha = sha256_file(Path(__file__).resolve())
    infrastructure_errors = 0
    functional_failures = 0
    rows: list[dict[str, Any]] = []
    model = processor = codec = None

    def emit(row: dict[str, Any]) -> None:
        nonlocal infrastructure_errors, functional_failures
        append_jsonl(results_path, row)
        rows.append(row)
        if row["failure_class"] == "infrastructure":
            infrastructure_errors += 1
        elif row["failure_class"] == "functional":
            functional_failures += 1
        print(json.dumps({
            "model": spec.key,
            "case_id": row["case_id"],
            "repeat_index": row["repeat_index"],
            "status": row["status"],
            "error_type": row["error_type"],
        }, ensure_ascii=False), flush=True)

    try:
        verify_runtime(torch)
        runtime = runtime_info(torch)
        model, processor, codec, snapshot, load_facts = load_model(spec, output_root, torch)
        manifest_inputs = [
            output_root / "inputs" / name
            for name in ("speech_reference.wav", "speech_reference.json", "scene.png", "scene_video.mp4")
        ]
        write_json(run_dir / "manifest.json", {
            "schema_version": 1,
            "run_id": run_id,
            "model": spec.key,
            "model_id": spec.repo_id,
            "revision": spec.revision,
            "code_revision": spec.revision,
            "script_sha256": script_sha,
            "output_root": str(output_root),
            "runtime": runtime,
            "load": load_facts,
            "snapshot_path": str(snapshot),
            "dtype": "float16",
            "attention": "eager",
            "generation_seed": GENERATION_SEED,
            "generation_parameters": GENERATION_PARAMETERS if spec.key == "audio" else {"generation_mode": "text"},
            "repeats": repeats,
            "input_hashes_at_start": [file_descriptor(path, output_root) for path in manifest_inputs],
            "environment": {
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"),
                "HF_HUB_DISABLE_PROGRESS_BARS": os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
            },
        })
    except Exception as exc:
        task = TaskSpec("__model_load__", "asr", MAX_TEXT_TOKENS)
        emit(case_error(spec, script_sha, run_id, task, None, output_root, [], "model_load_or_runtime", exc))
        write_json(run_dir / "summary.json", {
            "model": spec.key, "model_id": spec.repo_id, "revision": spec.revision,
            "code_revision": spec.revision, "status": "error",
            "infrastructure_error_count": infrastructure_errors,
            "functional_failure_count": functional_failures,
            "results_jsonl": relative_path(results_path, output_root),
            "script_sha256": script_sha,
        })
        return 2

    try:
        if spec.key == "audio":
            # Generate synthetic speech fixtures with the tested model and label them as self-consistency inputs.
            for filename, (text, language) in SYNTHETIC_FIXTURES.items():
                task = TaskSpec(f"tts-fixture-{Path(filename).stem}", "tts", MAX_AUDIO_TOKENS, language, text, synthetic_input=True)
                try:
                    emit(generate_audio_fixture(filename, text, language, model, processor, codec, torch, spec, output_root, run_dir, run_id, script_sha))
                except Exception as exc:
                    emit(case_error(spec, script_sha, run_id, task, 1, output_root, [], "synthetic_input_generation", exc))

            for task in AUDIO_CASES:
                task_repeats = 1 if task.task_type in {"translation", "audio_understanding"} else repeats
                for repeat_index in range(1, task_repeats + 1):
                    try:
                        emit(run_audio_task(task, repeat_index, model, processor, codec, torch, spec, output_root, run_dir, run_id, script_sha))
                    except Exception as exc:
                        emit(case_error(
                            spec, script_sha, run_id, task, repeat_index, output_root,
                            audio_case_input_paths(task, output_root), "inference_or_scoring", exc,
                        ))
        else:
            source_video = output_root / "inputs" / "scene_video.mp4"
            question_audio = output_root / "inputs" / "voice_question.wav"
            muxed_path = run_dir / "artifacts" / "scene_video_with_audio.mp4"
            video_prep: dict[str, Any] | None = None
            video_prep_error: Exception | None = None
            try:
                video_prep = video_with_audio(source_video, question_audio, muxed_path)
                video_prep["output_file"] = file_descriptor(muxed_path, output_root)
            except Exception as exc:
                video_prep_error = exc
            for task in OMNI_CASES:
                for repeat_index in range(1, repeats + 1):
                    try:
                        if task.task_type == "video_audio" and video_prep_error is not None:
                            raise video_prep_error
                        emit(run_omni_task(task, repeat_index, model, processor, torch, spec, output_root, run_dir, run_id, script_sha, muxed_path if task.task_type == "video_audio" else None, video_prep))
                    except Exception as exc:
                        media = (output_root / "inputs" / "scene.png") if task.task_type == "image_audio" else (muxed_path if video_prep_error is None else source_video)
                        emit(case_error(
                            spec, script_sha, run_id, task, repeat_index, output_root,
                            [(media, "visual input"), (question_audio, "synthetic_self_consistency: generated by LFM3-Audio-2B")],
                            "video_audio_mux" if task.task_type == "video_audio" and video_prep_error is not None else "inference_or_scoring",
                            exc,
                        ))

        timing_by_case: dict[str, dict[str, Any]] = {}
        for case_id in sorted({row["case_id"] for row in rows}):
            durations = [
                row["timing_seconds"].get("generation_plus_codec")
                for row in rows
                if row["case_id"] == case_id and row.get("timing_seconds", {}).get("generation_plus_codec") is not None
            ]
            if durations:
                timing_by_case[case_id] = {
                    "sample_count": len(durations),
                    "median_seconds": statistics.median(durations),
                    "min_seconds": min(durations),
                    "max_seconds": max(durations),
                }
        summary = {
            "model": spec.key,
            "model_id": spec.repo_id,
            "revision": spec.revision,
            "code_revision": spec.revision,
            "run_id": run_id,
            "status": "complete_with_errors" if infrastructure_errors else ("complete_with_functional_failures" if functional_failures else "complete"),
            "repeat_count": repeats,
            "result_count": len(rows),
            "passed_count": sum(row["status"] == "passed" for row in rows),
            "completed_count": sum(row["status"] == "completed" for row in rows),
            "failed_count": sum(row["status"] == "failed" for row in rows),
            "functional_failure_count": functional_failures,
            "infrastructure_error_count": infrastructure_errors,
            "timing_by_case": timing_by_case,
            "results_jsonl": relative_path(results_path, output_root),
            "manifest_json": relative_path(run_dir / "manifest.json", output_root),
            "script_sha256": script_sha,
        }
        write_json(run_dir / "summary.json", summary)
        write_json(model_dir / "latest-run.json", {
            "run_id": run_id,
            "summary_json": relative_path(run_dir / "summary.json", output_root),
            "results_jsonl": relative_path(results_path, output_root),
        })
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        if codec is not None:
            del codec
        torch.cuda.empty_cache()
    return 2 if infrastructure_errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LFM3 Audio or Omni functional coverage on a Tesla T4")
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 20:
        parser.error("--repeats must be between 1 and 20")
    return model_run(MODELS[args.model], args.output_dir, args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
