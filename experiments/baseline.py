from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

OUT = Path("/content/lfm3-t4-results")
MAX_NEW_TOKENS = 128
REPEATS = 5
FONT_COMMIT = "ddf386ba1b9fbc6b2c818f1ddcd9148f54ced94b"
FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/"
    f"{FONT_COMMIT}/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
)


@dataclass(frozen=True)
class ModelSpec:
    key: Literal["text", "vision"]
    repo_id: str
    revision: str
    kind: Literal["text", "vision"]


@dataclass(frozen=True)
class TextCase:
    case_id: str
    prompt: str
    expected: dict[str, object]


@dataclass(frozen=True)
class ImageCase:
    case_id: str
    prompt: str
    image_path: Path
    expected: dict[str, object]


MODELS = {
    "text": ModelSpec(
        "text", "LiquidAI/LFM3-1.7B",
        "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd", "text",
    ),
    "vision": ModelSpec(
        "vision", "LiquidAI/LFM3-VL-2B",
        "f83d2b0a2296aaa277f394990a9b8b949ddf9691", "vision",
    ),
}

TEXT_CASES = (
    TextCase(
        "text-01",
        "次の発注メモから注文番号と合計金額を抽出してください。"
        "JSON以外は出力しないでください。キーは注文番号、合計金額円です。\n"
        "発注メモ\n注文番号: TK-104\n合計: 2,400円",
        {"注文番号": "TK-104", "合計金額円": 2400},
    ),
    TextCase(
        "text-02",
        "次の受付票から氏名と受付日を抽出してください。"
        "受付日はYYYY-MM-DD形式にしてください。JSON以外は出力しないでください。"
        "キーは氏名、受付日です。\n氏名: 高橋 奈緒\n受付日: 2026年10月20日",
        {"氏名": "高橋 奈緒", "受付日": "2026-10-20"},
    ),
    TextCase(
        "text-03",
        "返品条件は購入から14日以内かつ未開封です。次の申請を判定し、"
        "申請番号と返品可否をJSONで返してください。JSON以外は出力しないでください。"
        "キーは申請番号、返品可です。\n申請番号: R-8831\n購入日: 2026-10-20"
        "\n申請日: 2026-10-28\n開封状態: 未開封",
        {"申請番号": "R-8831", "返品可": True},
    ),
)

INVOICES = (
    {
        "case_id": "invoice-01", "number": "INV-2026-041",
        "issue_jp": "2026年10月12日", "issue_iso": "2026-10-12",
        "due_jp": None, "amount": 12650,
    },
    {
        "case_id": "invoice-02", "number": "INV-2026-042",
        "issue_jp": "2026年10月15日", "issue_iso": "2026-10-15",
        "due_jp": "2026年10月31日", "amount": 8400,
    },
)
IMAGE_CASES = tuple(
    ImageCase(
        item["case_id"],
        "画像の請求書から請求書番号、発行日、合計金額円を抽出してください。"
        + ("支払期限も抽出してください。" if item["due_jp"] else "")
        + "日付はYYYY-MM-DD形式、JSON以外は出力しないでください。",
        Path("inputs") / f"{item['case_id']}.png",
        {
            "請求書番号": item["number"],
            "発行日": item["issue_iso"],
            **({"支払期限": item["due_jp"].replace("年", "-").replace("月", "-").replace("日", "")}
               if item["due_jp"] else {}),
            "合計金額円": item["amount"],
        },
    )
    for item in INVOICES
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strict_json(raw: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"非有限数値はJSONとして認めません: {value}")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重複キーです: {key}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"有限範囲を超える数値です: {value}")
        return number

    return json.loads(
        raw, parse_constant=reject_constant, parse_float=finite_float,
        object_pairs_hook=unique_pairs,
    )


def strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def invoice_strings(item: dict[str, object]) -> list[str]:
    amount = f"{item['amount']:,}円"
    parts = [
        "請求書", "青葉商事", "請求書番号", str(item["number"]),
        "発行日", str(item["issue_jp"]), "品目", "事務用品", "小計",
        f"{int(item['amount']) - 1200:,}円", "消費税", "1,200円", "合計", amount,
    ]
    if item["due_jp"]:
        parts += ["お支払期限", str(item["due_jp"])]
    return parts


def prepare(output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "inputs"
    asset_dir = output / "assets"
    image_dir.mkdir(exist_ok=True)
    asset_dir.mkdir(exist_ok=True)
    font_path = asset_dir / "NotoSansJP-variable.ttf"
    if not font_path.exists():
        with urlopen(FONT_URL, timeout=60) as response:
            font_bytes = response.read()
        if not font_bytes.startswith((b"\x00\x01\x00\x00", b"OTTO")):
            raise ValueError("Google Fontsから有効なTTFフォントを取得できません")
        font_path.write_bytes(font_bytes)
    font_hash = sha256_file(font_path)
    license_path = asset_dir / "NotoSansJP-OFL.txt"
    license_url = f"https://raw.githubusercontent.com/google/fonts/{FONT_COMMIT}/ofl/notosansjp/OFL.txt"
    with urlopen(license_url, timeout=60) as response:
        license_bytes = response.read()
    if b"SIL Open Font License" not in license_bytes:
        raise ValueError("フォントのOFLライセンスを取得できません")
    license_path.write_bytes(license_bytes)
    font_manifest = {
        "family": "Noto Sans JP", "source": "google/fonts",
        "commit": FONT_COMMIT, "url": FONT_URL, "sha256": font_hash,
        "bytes": font_path.stat().st_size,
        "license_url": license_url,
        "license_sha256": sha256_file(license_path),
    }
    regular = ImageFont.truetype(str(font_path), 28)
    title = ImageFont.truetype(str(font_path), 44)
    missing = regular.getmask("\u0378", mode="L")
    for item in INVOICES:
        strings = invoice_strings(item)
        unsupported = sorted({
            char for value in strings for char in value if not char.isspace()
            if regular.getmask(char, mode="L").size == missing.size
            and bytes(regular.getmask(char, mode="L")) == bytes(missing)
        })
        if unsupported:
            raise ValueError(f"日本語フォントに未収録の文字があります: {unsupported}")
        image = Image.new("RGB", (1280, 900), "white")
        draw = ImageDraw.Draw(image)
        draw.text((64, 48), "請 求 書", font=title, fill="#152c45")
        draw.text((64, 130), "青葉商事", font=regular, fill="#17212b")
        draw.line((64, 190, 1216, 190), fill="#57738e", width=3)
        rows = [
            ("請求書番号", str(item["number"])),
            ("発行日", str(item["issue_jp"])),
            ("品目", "事務用品"),
            ("小計", f"{int(item['amount']) - 1200:,}円"),
            ("消費税", "1,200円"),
        ]
        if item["due_jp"]:
            rows.append(("お支払期限", str(item["due_jp"])))
        y = 245
        for label, value in rows:
            draw.text((96, y), label, font=regular, fill="#374a5c")
            draw.text((430, y), value, font=regular, fill="#101820")
            y += 78
        draw.line((64, y + 10, 1216, y + 10), fill="#a8b5c2", width=2)
        draw.text((96, y + 40), "合計", font=title, fill="#152c45")
        draw.text((430, y + 40), f"{int(item['amount']):,}円", font=title, fill="#152c45")
        image.save(image_dir / f"{item['case_id']}.png", format="PNG", optimize=False)
    write_json(output / "font.json", font_manifest)
    registry = {
        "text": [asdict(case) for case in TEXT_CASES],
        "vision": [
            {**asdict(case), "image_path": str(case.image_path)} for case in IMAGE_CASES
        ],
    }
    write_json(output / "cases.json", registry)
    write_json(output / "input-hashes.json", {
        "text": {
            case.case_id: {"prompt_sha256": sha256_bytes(case.prompt.encode("utf-8"))}
            for case in TEXT_CASES
        },
        "vision": {
            case.case_id: {
                "path": case.image_path.as_posix(),
                "sha256": sha256_file(output / case.image_path),
                "prompt_sha256": sha256_bytes(case.prompt.encode("utf-8")),
            }
            for case in IMAGE_CASES
        },
    })


def runtime_info(torch: object) -> dict[str, object]:
    import importlib.metadata as metadata

    info: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": metadata.version("transformers"),
        "accelerate": metadata.version("accelerate"),
        "pillow": metadata.version("Pillow"),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    for package in ("huggingface-hub", "tokenizers", "safetensors", "nbformat"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    return info


def make_record(
    spec: ModelSpec, script_sha: str, case_id: str, status: str,
    expected: object = None, raw_output: str | None = None,
    parsed_output: object = None, exact_match: bool | None = None,
    prompt_tokens: int | None = None, generated_tokens: int | None = None,
    generate_seconds: float | None = None, peak_allocated: int | None = None,
    peak_reserved: int | None = None, repeat_index: int | None = None,
    is_warmup: bool = False, error_stage: str | None = None,
    error_type: str | None = None, error_message: str | None = None,
) -> dict[str, object]:
    return {
        "model_id": spec.repo_id, "revision": spec.revision,
        "code_revision": spec.revision, "kind": spec.kind,
        "case_id": case_id, "repeat_index": repeat_index, "is_warmup": is_warmup,
        "status": status, "expected": expected, "raw_output": raw_output,
        "parsed_output": parsed_output, "exact_match": exact_match,
        "prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
        "max_new_tokens": MAX_NEW_TOKENS, "generate_seconds": generate_seconds,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "error_stage": error_stage, "error_type": error_type,
        "error_message": error_message, "script_sha256": script_sha,
    }


def run_model(kind: str, output: Path, thinking: bool, expected_script_sha: str) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForCausalLM, AutoModelForImageTextToText,
        AutoProcessor, AutoTokenizer,
    )

    spec = MODELS[kind]
    model_dir = output / kind
    model_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("summary.json", "manifest.json", "results.jsonl", "warmups.jsonl"):
        (model_dir / stale).unlink(missing_ok=True)
    script_sha = sha256_file(Path(__file__).resolve())
    if script_sha != expected_script_sha:
        raise RuntimeError("埋め込みスクリプトのSHA-256が一致しません")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA GPUがありません。CPU推論には切り替えません")
    capability = torch.cuda.get_device_capability(0)
    if torch.cuda.get_device_name(0) != "Tesla T4" or capability != (7, 5):
        raise RuntimeError(
            f"Tesla T4 sm75ではありません: {torch.cuda.get_device_name(0)}, {capability}"
        )
    environment = runtime_info(torch)
    if environment["transformers"] != "5.21.0":
        raise RuntimeError(f"transformers==5.21.0が必要です: {environment['transformers']}")
    results_path = model_dir / "results.jsonl"
    warmups_path = model_dir / "warmups.jsonl"
    for path in (results_path, warmups_path):
        path.write_text("", encoding="utf-8")
    download_start = time.perf_counter()
    snapshot = Path(snapshot_download(
        repo_id=spec.repo_id, revision=spec.revision, token=False,
        allow_patterns=[
            "*.json", "*.py", "*.safetensors", "*.jinja", "*.model",
            "*.spm", "*.tiktoken", "*.txt", "*.vocab", "*.merges",
        ],
    ))
    download_seconds = time.perf_counter() - download_start
    hash_start = time.perf_counter()
    file_hashes = {
        str(path.relative_to(snapshot)): {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        }
        for path in sorted(snapshot.rglob("*")) if path.is_file()
    }
    hash_seconds = time.perf_counter() - hash_start
    model_cls = AutoModelForCausalLM if kind == "text" else AutoModelForImageTextToText
    load_start = time.perf_counter()
    model = model_cls.from_pretrained(
        str(snapshot), revision=spec.revision, code_revision=spec.revision,
        dtype=torch.float16, attn_implementation="eager",
        trust_remote_code=True, token=False,
    ).to("cuda:0", dtype=torch.float16)
    if any(p.device.type != "cuda" or p.dtype != torch.float16 for p in model.parameters()):
        raise RuntimeError("全パラメーターがCUDA FP16ではありません")
    model.eval()
    if kind == "text":
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), revision=spec.revision, trust_remote_code=True, token=False,
        )
        processor = None
        cases: tuple[TextCase | ImageCase, ...] = TEXT_CASES
    else:
        processor = AutoProcessor.from_pretrained(
            str(snapshot), revision=spec.revision, trust_remote_code=True, token=False,
        )
        tokenizer = processor.tokenizer
        cases = IMAGE_CASES
    load_seconds = time.perf_counter() - load_start
    manifest = {
        "model_id": spec.repo_id, "revision": spec.revision,
        "code_revision": spec.revision, "kind": kind,
        "download_seconds": download_seconds, "load_seconds": load_seconds,
        "file_hash_seconds": hash_seconds, "files": file_hashes,
        "runtime": environment, "script_sha256": script_sha,
        "dtype": "float16", "attention": "eager", "thinking": thinking,
        "max_new_tokens": MAX_NEW_TOKENS, "repeats": REPEATS,
        "inputs": json.loads((output / "input-hashes.json").read_text(encoding="utf-8"))[kind],
    }
    write_json(model_dir / "manifest.json", manifest)

    def append(path: Path, record: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    def infer(case: TextCase | ImageCase, warmup: bool, repeat_index: int | None) -> dict[str, object]:
        if kind == "text":
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": case.prompt}],
                add_generation_prompt=True, enable_thinking=thinking,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to("cuda:0")
        else:
            from PIL import Image

            assert isinstance(case, ImageCase)
            image = Image.open(output / case.image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": case.prompt},
            ]}]
            encoded = processor.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=thinking,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to("cuda:0", torch.float16)
        input_ids = encoded["input_ids"]
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"input_idsが整数型ではありません: {input_ids.dtype}")
        if any(t.device.type != "cuda" for t in encoded.values() if isinstance(t, torch.Tensor)):
            raise RuntimeError("入力テンソルがCUDAにありません")
        prompt_tokens = int(input_ids.shape[-1])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(
                **encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        generated = sequences[0, prompt_tokens:]
        generated_tokens = int(generated.numel())
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        status = "passed"
        parsed: object = None
        exact: bool | None = False
        error_type = error_message = None
        if not raw.strip():
            status, error_type, error_message = "failed", "empty_output", "生成結果が空です"
        elif generated_tokens >= MAX_NEW_TOKENS:
            status, error_type, error_message = "failed", "max_tokens_truncated", "最大生成トークン数に達しました"
        else:
            try:
                parsed = strict_json(raw)
                exact = strict_equal(parsed, case.expected)
                if not exact:
                    status, error_type, error_message = "failed", "exact_match_failed", "期待値とキーまたは値の型が一致しません"
            except (ValueError, json.JSONDecodeError) as exc:
                status, error_type, error_message = "failed", "strict_json_failed", str(exc)
        return make_record(
            spec, script_sha, case.case_id, "warmup" if warmup else status,
            expected=case.expected, raw_output=raw, parsed_output=parsed,
            exact_match=exact, prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens, generate_seconds=seconds,
            peak_allocated=peak_allocated, peak_reserved=peak_reserved,
            repeat_index=repeat_index, is_warmup=warmup,
            error_type=error_type, error_message=error_message,
        )

    infrastructure_errors = 0
    try:
        try:
            warm = infer(cases[0], True, None)
            append(warmups_path, warm)
        except Exception as exc:
            infrastructure_errors += 1
            append(warmups_path, make_record(
                spec, script_sha, cases[0].case_id, "error", expected=cases[0].expected,
                is_warmup=True, error_stage="warmup", error_type=type(exc).__name__,
                error_message=str(exc),
            ))
        for case in cases:
            for repeat_index in range(1, REPEATS + 1):
                try:
                    append(results_path, infer(case, False, repeat_index))
                except Exception as exc:
                    infrastructure_errors += 1
                    append(results_path, make_record(
                        spec, script_sha, case.case_id, "error", expected=case.expected,
                        repeat_index=repeat_index, error_stage="inference",
                        error_type=type(exc).__name__, error_message=str(exc),
                    ))
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        durations = [row["generate_seconds"] for row in rows if row["generate_seconds"] is not None]
        summary = {
            "model_id": spec.repo_id, "revision": spec.revision,
            "status": "complete" if infrastructure_errors == 0 else "error",
            "case_count": len(cases), "repeat_count": REPEATS,
            "sample_count": len(rows),
            "passed_count": sum(row["status"] == "passed" for row in rows),
            "answer_failed_count": sum(row["status"] == "failed" for row in rows),
            "infrastructure_error_count": infrastructure_errors,
            "generate_seconds_median": statistics.median(durations) if durations else None,
            "generate_seconds_min": min(durations) if durations else None,
            "generate_seconds_max": max(durations) if durations else None,
            "script_sha256": script_sha,
        }
        write_json(model_dir / "summary.json", summary)
    finally:
        del model
        torch.cuda.empty_cache()
    return 2 if infrastructure_errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--script-sha256", default="")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir)
        print(json.dumps({"prepared": True, "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return 0
    if not args.model:
        parser.error("runには--model textまたは--model visionが必要です")
    try:
        return run_model(args.model, args.output_dir, args.thinking, args.script_sha256)
    except Exception as exc:
        spec = MODELS[args.model]
        model_dir = args.output_dir / args.model
        model_dir.mkdir(parents=True, exist_ok=True)
        script_sha = sha256_file(Path(__file__).resolve())
        error = make_record(
            spec, script_sha, "__model__", "error", error_stage="model_run",
            error_type=type(exc).__name__, error_message=str(exc),
        )
        with (model_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(model_dir / "summary.json", {
            "model_id": spec.repo_id, "revision": spec.revision,
            "status": "error", "infrastructure_error_count": 1,
            "script_sha256": script_sha, "traceback": traceback.format_exc(),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
