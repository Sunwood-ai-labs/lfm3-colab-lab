from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SuiteSpec:
    suite_key: str
    model_key: str
    runner: str
    model_dir: str
    family: str
    group: str
    label: str
    coverage: tuple[str, ...]


SUITES = (
    SuiteSpec("text_extra", "text", "run_video_tools.py", "text_extra", "video", "full", "LFM3-1.7B", ("日本語テキスト", "実ツール呼出し")),
    SuiteSpec("vision_extra", "vision", "run_video_tools.py", "vision_extra", "video", "full", "LFM3-VL-2B", ("動画", "画像と実ツール呼出し")),
    SuiteSpec("audio", "audio", "run_audio.py", "audio", "audio", "full", "LFM3-Audio-2B", ("日本語音声認識", "音声合成", "音声理解", "音声対話")),
    SuiteSpec("omni", "omni", "run_audio.py", "omni", "audio", "full", "LFM3-Omni-2B", ("画像と音声", "動画と音声")),
    SuiteSpec("d1", "d1", "run_decisions.py", "d1", "decision", "full", "d1-2B", ("テキスト・画像の型付き判断",)),
    SuiteSpec("d1_omni", "d1_omni", "run_decisions.py", "d1_omni", "decision", "full", "d1-Omni-2B", ("テキスト・画像・音声の型付き判断",)),
    SuiteSpec("baseline_text", "text", "baseline.py", "text", "baseline", "baseline", "LFM3-1.7B baseline", ("日本語JSON抽出",)),
    SuiteSpec("baseline_vision", "vision", "baseline.py", "vision", "baseline", "baseline", "LFM3-VL-2B baseline", ("請求書画像JSON抽出",)),
)

CSV_COLUMNS = (
    "group", "suite_key", "model_key", "model", "model_id", "revision",
    "run_id", "status", "passed", "attempted", "functional_failures",
    "infrastructure_errors", "format_failures", "process_exit_code",
    "process_elapsed_seconds", "script_sha256", "summary_path", "manifest_path",
)
PERFORMANCE_COLUMNS = (
    "suite_key", "model_key", "model", "run_id", "script_sha256", "artifact_dir",
    "task_id", "stage", "measurement_type",
    "evidence_current", "sample_count", "median_seconds", "median_tokens_per_second",
    "median_rtf", "rtf_basis", "median_first_audio_chunk_seconds",
    "peak_allocated_gib", "peak_reserved_gib", "timing_scope",
)
GIB = 1024 ** 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append({"path": str(path), "line": str(line_number), "error": str(exc)})
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError as exc:
        errors.append({"path": str(path), "error": str(exc)})
    return rows


def module_assignment(script_path: Path, name: str) -> ast.expr:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise ValueError(f"{name} assignment was not found in {script_path.name}")


def model_specs(script_path: Path) -> dict[str, dict[str, Any]]:
    assignment = module_assignment(script_path, "MODELS" if script_path.name in {"run_audio.py", "baseline.py"} else "MODEL_SPECS")
    if not isinstance(assignment, ast.Dict):
        raise ValueError(f"model registry in {script_path.name} is not a dictionary")
    specs: dict[str, dict[str, Any]] = {}
    for key_node, value_node in zip(assignment.keys, assignment.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        if isinstance(value_node, ast.Dict):
            raw = ast.literal_eval(value_node)
            model_id = raw.get("model_id", raw.get("repo_id"))
            revision = raw.get("revision")
            extra = {key: value for key, value in raw.items() if key not in {"model_id", "repo_id", "revision"}}
        elif isinstance(value_node, ast.Call):
            args = [node.value if isinstance(node, ast.Constant) else None for node in value_node.args]
            if len(args) < 3 or not all(isinstance(item, str) for item in args[:3]):
                continue
            _, model_id, revision, *rest = args
            extra = {"constructor_args": rest}
        else:
            continue
        specs[key_node.value] = {"model_id": model_id, "revision": revision, **extra}
    return specs


def process_record(root: Path, suite_key: str) -> dict[str, Any] | None:
    latest = newest(list((root / "process_logs").glob(f"*/{suite_key}.json")))
    if latest is not None:
        return read_json(latest)
    return read_json(root / "process_logs" / f"{suite_key}.json")


def newest(paths: list[Path]) -> Path | None:
    return max((path for path in paths if path.is_file()), key=lambda path: path.stat().st_mtime_ns, default=None)


def artifact_candidates(root: Path, spec: SuiteSpec) -> list[Path]:
    model_root = root / spec.model_dir
    if spec.group == "baseline":
        return list((root / "baseline-runs").glob(f"*/{spec.model_key}/summary.json"))
    if spec.family == "audio":
        return list((model_root / "runs").glob("*/summary.json"))
    if spec.family == "decision":
        return list(model_root.glob("*/results.json"))
    return list(model_root.glob("*/summary.json"))


def candidate_files(spec: SuiteSpec, artifact_path: Path) -> tuple[Path, Path | None, Path, list[Path]]:
    run_dir = artifact_path.parent
    if spec.family == "decision":
        return artifact_path, run_dir / "manifest.json", run_dir, [run_dir / "tasks.jsonl"]
    if spec.family == "video":
        return artifact_path, run_dir / "run_metadata.json", run_dir, [run_dir / "cases.jsonl", run_dir / "generations.jsonl"]
    if spec.family == "audio":
        return artifact_path, run_dir / "manifest.json", run_dir, [run_dir / "results.jsonl"]
    return artifact_path, run_dir / "manifest.json", run_dir, [run_dir / "results.jsonl", run_dir / "warmups.jsonl"]


def artifact_evidence(spec: SuiteSpec, artifact_path: Path) -> tuple[str | None, str | None, str | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    summary = read_json(artifact_path) if artifact_path.name == "summary.json" else None
    result_doc = read_json(artifact_path) if spec.family == "decision" else None
    run_dir = artifact_path.parent
    manifest_path = run_dir / ("run_metadata.json" if spec.family == "video" else "manifest.json")
    manifest = read_json(manifest_path)
    metadata = manifest if spec.family == "video" else None
    source_hash = (
        (summary or {}).get("script_sha256")
        or (result_doc or {}).get("script_sha256")
        or (manifest or {}).get("script_sha256")
    )
    model_id = (
        (summary or {}).get("model_id")
        or (result_doc or {}).get("model_id")
        or (manifest or {}).get("model_id")
    )
    revision = (
        (summary or {}).get("revision")
        or (summary or {}).get("model_revision")
        or (result_doc or {}).get("model_sha")
        or (manifest or {}).get("revision")
        or (manifest or {}).get("model_revision")
        or (manifest or {}).get("model_sha")
    )
    return source_hash, model_id, revision, summary, manifest, result_doc


def current_input_match(root: Path, spec: SuiteSpec, run_dir: Path, manifest: dict[str, Any] | None) -> bool | None:
    inputs = root / "inputs"
    if spec.family == "video":
        recorded = (manifest or {}).get("inputs")
        if not isinstance(recorded, dict):
            return None
        pairs = {"scene": "scene.png", "sequence_video": "sequence_video.mp4"}
        for key, filename in pairs.items():
            fact = recorded.get(key)
            path = inputs / filename
            if not isinstance(fact, dict) or not path.is_file():
                return False
            if fact.get("sha256") != sha256_file(path) or fact.get("bytes") != path.stat().st_size:
                return False
        return True
    if spec.family == "audio":
        recorded = (manifest or {}).get("input_hashes_at_start")
        if not isinstance(recorded, list):
            return None
        expected = {"speech_reference.wav", "speech_reference.json", "scene.png", "scene_video.mp4"}
        found: set[str] = set()
        for item in recorded:
            if not isinstance(item, dict):
                return False
            filename = Path(str(item.get("path", ""))).name
            if filename not in expected:
                continue
            found.add(filename)
            path = inputs / filename
            if item.get("exists") != path.is_file():
                return False
            if path.is_file() and (item.get("sha256") != sha256_file(path) or item.get("bytes") != path.stat().st_size):
                return False
        if found != expected:
            return False
        # The audio manifests contain suite-wide base inputs. Per-task input descriptors
        # additionally bind cross-modal outputs such as Omni's generated voice question.
        if spec.model_key in {"audio", "omni"}:
            records = read_jsonl(run_dir / "results.jsonl", [])
            if not records:
                return False
            for record in records:
                descriptors = record.get("input_files")
                if not isinstance(descriptors, list):
                    return False
                for descriptor in descriptors:
                    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
                        return False
                    relative = Path(descriptor["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        return False
                    path = inputs.parent / relative
                    try:
                        path.resolve().relative_to(inputs.parent.resolve())
                    except ValueError:
                        return False
                    exists = path.is_file()
                    if descriptor.get("exists") is not exists:
                        return False
                    if exists and (descriptor.get("sha256") != sha256_file(path) or descriptor.get("bytes") != path.stat().st_size):
                        return False
        return True
    if spec.family == "decision":
        recorded = (manifest or {}).get("input_hashes")
        if not isinstance(recorded, dict):
            return None
        names = ["scene.png"]
        if spec.model_key == "d1_omni":
            names.extend(["speech_reference.wav", "speech_reference.json", "voice_question.wav"])
        for filename in names:
            path = inputs / filename
            fact = recorded.get(filename)
            if not isinstance(fact, dict):
                return False
            if path.is_file():
                if fact.get("sha256") != sha256_file(path) or fact.get("bytes") != path.stat().st_size:
                    return False
            elif fact.get("missing") is not True:
                return False
        return True
    batch_id = run_dir.parent.name
    process = read_json(root / "process_logs" / batch_id / f"baseline_{spec.model_key}.json")
    recorded = process.get("input_hashes") if process else None
    if not isinstance(recorded, dict):
        return None
    baseline_inputs = run_dir.parent / "inputs"
    current = {
        str(path.relative_to(baseline_inputs)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(baseline_inputs.rglob("*")) if path.is_file()
    } if baseline_inputs.is_dir() else {}
    return recorded == current


def locate_artifacts(root: Path, spec: SuiteSpec, expected_source_hash: str, expected_model_id: str | None, expected_revision: str | None) -> tuple[Path | None, Path | None, Path | None, list[Path], bool, str | None]:
    candidates = sorted(artifact_candidates(root, spec), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    if not candidates:
        return None, None, None, [], False, None
    latest = candidates[0]
    for artifact_path in candidates:
        source_hash, model_id, revision, summary, manifest, result_doc = artifact_evidence(spec, artifact_path)
        if source_hash != expected_source_hash:
            continue
        if expected_model_id and model_id != expected_model_id:
            continue
        if expected_revision and revision != expected_revision:
            continue
        run_dir = artifact_path.parent
        input_match = current_input_match(root, spec, run_dir, manifest)
        if input_match is False:
            continue
        if input_match is None:
            has_infrastructure_error = bool(
                (summary or {}).get("infrastructure_error_count")
                or (summary or {}).get("infrastructure_failure")
                or (manifest or {}).get("infrastructure_errors")
                or (manifest or {}).get("status") in {"error", "infrastructure_error", "complete_with_errors"}
            )
            has_scored_results = bool(
                (summary or {}).get("result_count")
                or (summary or {}).get("sample_count")
                or (summary or {}).get("case_count")
                or (summary or {}).get("task_score", {}).get("attempted")
                or (result_doc or {}).get("tasks")
            )
            if not has_infrastructure_error or has_scored_results:
                continue
        _, manifest_path, _, result_paths = candidate_files(spec, artifact_path)
        return artifact_path, manifest_path, run_dir, result_paths, True, "matched" if input_match else "not_declared"
    _, manifest_path, run_dir, result_paths = candidate_files(spec, latest)
    return latest, manifest_path, run_dir, result_paths, False, "stale_source_pin_or_input"


def score_fields(spec: SuiteSpec, summary: dict[str, Any] | None, manifest: dict[str, Any] | None, result_doc: dict[str, Any] | None) -> tuple[int, int, int, int, int, str]:
    passed = attempted = functional_failures = infrastructure_errors = format_failures = 0
    status = "artifact_present_no_scores"
    if spec.family == "decision" and result_doc:
        tasks = result_doc.get("tasks") or []
        if isinstance(tasks, list):
            attempted = len(tasks)
            passed = sum(isinstance(task, dict) and task.get("status") == "passed" for task in tasks)
            functional_failures = attempted - passed
        infrastructure_errors = len((manifest or {}).get("infrastructure_errors") or result_doc.get("infrastructure_errors") or [])
        format_failures = int(((manifest or {}).get("task_summary") or {}).get("format_failures") or 0)
        if result_doc.get("status") and result_doc["status"] == "infrastructure_error":
            infrastructure_errors = max(1, infrastructure_errors)
    elif summary:
        score = summary.get("task_score")
        if isinstance(score, dict):
            passed = int(score.get("passed") or 0)
            attempted = int(score.get("attempted") or 0)
            functional_failures = max(0, attempted - passed)
        elif spec.family == "audio":
            passed = int(summary.get("passed_count") or 0)
            attempted = int(summary.get("result_count") or 0)
            functional_failures = int(summary.get("failed_count") or 0)
        else:
            passed = int(summary.get("passed_count") or 0)
            attempted = int(summary.get("sample_count") or summary.get("case_count") or 0)
            functional_failures = int(summary.get("answer_failed_count") or 0)
        infrastructure_errors = int(summary.get("infrastructure_error_count") or 0)
        if summary.get("infrastructure_failure") or summary.get("status") in {
            "error", "infrastructure_error", "complete_with_errors", "completed_with_infrastructure_errors",
        }:
            infrastructure_errors = max(1, infrastructure_errors)
        if spec.family == "decision" and manifest:
            infrastructure_errors = max(infrastructure_errors, len(manifest.get("infrastructure_errors") or []))
            format_failures = int((manifest.get("task_summary") or {}).get("format_failures") or 0)
    else:
        return 0, 0, 0, 0, 0, status
    if infrastructure_errors:
        status = "infrastructure_error"
    elif attempted and functional_failures:
        status = "completed_with_failures"
    elif attempted:
        status = "completed"
    return passed, attempted, functional_failures, infrastructure_errors, format_failures, status


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and number < float("inf") else None


def median_or_none(values: list[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def performance_row(
    spec: SuiteSpec,
    *,
    task_id: str,
    stage: str,
    measurement_type: str,
    samples: list[dict[str, Any]],
    is_current: bool,
    timing_scope: str,
) -> dict[str, Any] | None:
    durations: list[float] = []
    token_rates: list[float] = []
    rtfs: list[float] = []
    first_chunks: list[float] = []
    allocated: list[float] = []
    reserved: list[float] = []
    for sample in samples:
        seconds = finite_number(sample.get("seconds"))
        if seconds is not None and seconds > 0:
            durations.append(seconds)
        token_count = finite_number(sample.get("generated_tokens"))
        if seconds is not None and seconds > 0 and token_count is not None:
            token_rates.append(token_count / seconds)
        rtf = finite_number(sample.get("rtf"))
        if rtf is not None:
            rtfs.append(rtf)
        first_chunk = finite_number(sample.get("first_audio_chunk_seconds"))
        if first_chunk is not None:
            first_chunks.append(first_chunk)
        allocated_bytes = finite_number(sample.get("peak_allocated_bytes"))
        reserved_bytes = finite_number(sample.get("peak_reserved_bytes"))
        if allocated_bytes is not None:
            allocated.append(allocated_bytes)
        if reserved_bytes is not None:
            reserved.append(reserved_bytes)
    if not any((durations, token_rates, rtfs, first_chunks, allocated, reserved)):
        return None
    return {
        "suite_key": spec.suite_key,
        "model_key": spec.model_key,
        "model": spec.label,
        "task_id": task_id,
        "stage": stage,
        "measurement_type": measurement_type,
        "evidence_current": is_current,
        "sample_count": len(durations),
        "median_seconds": median_or_none(durations),
        "median_tokens_per_second": median_or_none(token_rates),
        "median_rtf": median_or_none(rtfs),
        "rtf_basis": next((str(sample["rtf_basis"]) for sample in samples if sample.get("rtf_basis")), None),
        "median_first_audio_chunk_seconds": median_or_none(first_chunks),
        "peak_allocated_gib": max(allocated) / GIB if allocated else None,
        "peak_reserved_gib": max(reserved) / GIB if reserved else None,
        "timing_scope": timing_scope,
    }


def extract_performance(
    spec: SuiteSpec,
    run_dir: Path,
    is_current: bool,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Normalize per-task measurements while keeping runner timing scopes separate."""
    records: list[dict[str, Any]] = []
    if spec.family == "video":
        generations = read_jsonl(run_dir / "generations.jsonl", errors)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in generations:
            if not str(item.get("attempt", "")).startswith("measure-"):
                continue
            key = (str(item.get("case_id", "unknown")), str(item.get("turn", "generation")))
            groups.setdefault(key, []).append({
                "seconds": item.get("cuda_synchronized_wall_seconds"),
                "generated_tokens": item.get("generated_tokens"),
                "peak_allocated_bytes": item.get("cuda_peak_allocated_bytes"),
                "peak_reserved_bytes": item.get("cuda_peak_reserved_bytes"),
            })
        for (case_id, turn), samples in sorted(groups.items()):
            row = performance_row(
                spec, task_id=case_id, stage=turn, measurement_type="text_vl_generation",
                samples=samples, is_current=is_current,
                timing_scope="CUDA同期済みのmodel.generate区間。caseとturnごと。warmup試行を除外。モデルロードとprocessor前処理を含まない。",
            )
            if row:
                records.append(row)
        return records

    if spec.family == "audio":
        results = read_jsonl(run_dir / "results.jsonl", errors)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in results:
            timings = item.get("timing_seconds")
            if not isinstance(timings, dict):
                continue
            generation_rounds = sorted(
                (key for key in timings if key.startswith("model_generation_round_") and finite_number(timings.get(key)) is not None),
                key=lambda key: int(key.rsplit("_", 1)[1]),
            )
            if generation_rounds:
                generation_seconds = sum(float(timings[key]) for key in generation_rounds)
            else:
                generation_key = next((key for key in ("model_generation", "model_generation_wall_including_stream_decode") if finite_number(timings.get(key)) is not None), None)
                if generation_key is None:
                    continue
                generation_seconds = timings[generation_key]
            task_id = str(item.get("case_id", "unknown"))
            task_type = str(item.get("task_type", "audio"))
            output_files = item.get("output_files") if isinstance(item.get("output_files"), list) else []
            audio_outputs = [entry for entry in output_files if isinstance(entry, dict) and isinstance(entry.get("audio"), dict)]
            full_audio = next((entry for entry in audio_outputs if entry.get("decode_path") == "full_file"), None)
            audio_output = full_audio or (audio_outputs[0] if audio_outputs else None)
            output_audio = audio_output.get("audio") if audio_output else {}
            audio_duration = finite_number(output_audio.get("duration_seconds")) if isinstance(output_audio, dict) else None
            rtf_basis = next((key for key in ("generation_plus_full_codec", "generation_plus_codec") if finite_number(timings.get(key)) is not None), None)
            rtf_value = None
            if rtf_basis and audio_duration is not None and audio_duration > 0:
                rtf_value = float(timings[rtf_basis]) / audio_duration
            groups.setdefault((task_id, task_type), []).append({
                "seconds": generation_seconds,
                "peak_allocated_bytes": item.get("peak_vram_allocated_bytes"),
                "peak_reserved_bytes": item.get("peak_vram_reserved_bytes"),
                "rtf": rtf_value,
                "rtf_basis": rtf_basis if rtf_value is not None else None,
                "first_audio_chunk_seconds": timings.get("first_audio_frame_latency_from_generate_start"),
            })
        for (task_id, task_type), samples in sorted(groups.items()):
            row = performance_row(
                spec, task_id=task_id, stage=task_type, measurement_type="audio_generation_or_inference",
                samples=samples, is_current=is_current,
                timing_scope="runner記録のmodel generation wall time。複数generation roundは合算。RTFは音声出力がある場合のgeneration+codec時間/出力音声秒数。audio runnerにwarmup試行はない。モデルロードを含まない。",
            )
            if row:
                records.append(row)
        return records

    if spec.family == "decision":
        result_doc = read_json(run_dir / "results.json")
        tasks = result_doc.get("tasks") if result_doc else None
        if not isinstance(tasks, list):
            return records
        for task in tasks:
            if not isinstance(task, dict):
                continue
            measured = task.get("measured_repeats")
            if not isinstance(measured, list):
                continue
            samples: list[dict[str, Any]] = []
            for item in measured:
                if not isinstance(item, dict) or item.get("mode") != "measured" or item.get("status") == "infrastructure_error":
                    continue
                memory = item.get("memory") if isinstance(item.get("memory"), dict) else {}
                samples.append({
                    "seconds": (finite_number(item.get("elapsed_ms")) / 1000) if finite_number(item.get("elapsed_ms")) is not None else None,
                    "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
                    "peak_reserved_bytes": memory.get("peak_reserved_bytes"),
                })
            row = performance_row(
                spec, task_id=str(task.get("task_id", "unknown")), stage="system_one",
                measurement_type="d1_system_one", samples=samples, is_current=is_current,
                timing_scope="System One 1呼出しのelapsed_msを秒へ換算。measured_repeatsのみを集計しwarmupを除外。モデルロードを含まない。",
            )
            if row:
                records.append(row)
        return records
    return records


def discover_rows(root: Path, scripts_dir: Path, include_baseline: bool, errors: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for filename in {spec.runner for spec in SUITES}:
        path = scripts_dir / filename
        try:
            parsed[filename] = model_specs(path)
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append({"stage": "model_registry", "path": str(path), "error": str(exc)})
            parsed[filename] = {}
    rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for spec in SUITES:
        if spec.group == "baseline" and not include_baseline:
            continue
        spec_data = parsed.get(spec.runner, {}).get(spec.model_key, {})
        model_id = spec_data.get("model_id")
        revision = spec_data.get("revision")
        expected_source_hash = sha256_file(scripts_dir / spec.runner) if (scripts_dir / spec.runner).is_file() else ""
        summary_path, manifest_path, run_dir, result_paths, is_current, currentness = locate_artifacts(
            root, spec, expected_source_hash, model_id, revision,
        )
        summary = read_json(summary_path) if summary_path and summary_path.name == "summary.json" else None
        manifest = read_json(manifest_path) if manifest_path and manifest_path.is_file() else None
        result_doc = read_json(summary_path) if spec.family == "decision" and summary_path else None
        process = process_record(root, spec.suite_key)
        passed, attempted, functional_failures, infrastructure_errors, format_failures, status = score_fields(spec, summary, manifest, result_doc)
        if summary_path is None:
            status = "not_recorded" if process and process.get("returncode") not in (0, None) else "not_run"
        historical_metrics = {
            "passed": passed,
            "attempted": attempted,
            "completed_without_pass_grade": int((summary or {}).get("completed_count") or 0) if spec.family == "audio" else 0,
            "functional_failures": functional_failures,
            "infrastructure_errors": infrastructure_errors,
            "format_failures": format_failures,
        }
        if summary_path is not None and not is_current:
            status = "stale_evidence"
            passed = attempted = functional_failures = infrastructure_errors = format_failures = 0
        run_id = None
        if summary:
            run_id = summary.get("run_id")
        if manifest:
            run_id = run_id or manifest.get("run_id")
        row = {
            "group": spec.group,
            "suite_key": spec.suite_key,
            "model_key": spec.model_key,
            "model": spec.label,
            "model_id": model_id,
            "revision": revision,
            "coverage": list(spec.coverage),
            "run_id": run_id,
            "status": status,
            "passed": passed,
            "attempted": attempted,
            "completed_without_pass_grade": int((summary or {}).get("completed_count") or 0) if spec.family == "audio" and is_current else 0,
            "functional_failures": functional_failures,
            "infrastructure_errors": infrastructure_errors,
            "format_failures": format_failures,
            "process_exit_code": process.get("returncode") if process else None,
            "process_elapsed_seconds": process.get("elapsed_seconds") if process else None,
            "script_sha256": artifact_evidence(spec, summary_path)[0] if summary_path else None,
            "expected_script_sha256": expected_source_hash or None,
            "is_current_evidence": is_current if summary_path else False,
            "currentness": currentness,
            "historical_metrics": historical_metrics if summary_path and not is_current else None,
            "summary_path": str(summary_path.relative_to(root)) if summary_path else None,
            "manifest_path": str(manifest_path.relative_to(root)) if manifest_path and manifest_path.is_file() else None,
        }
        if spec.family == "decision" and result_doc:
            row["status"] = status
            row["result_schema"] = result_doc.get("schema_version")
        if summary:
            row["suite_summary"] = summary
        if manifest:
            row["run_manifest"] = manifest
        rows.append(row)
        if run_dir is not None and summary_path is not None and spec.group == "full":
            run_performance = extract_performance(spec, run_dir, is_current, errors)
            for measurement in run_performance:
                measurement.update({
                    "run_id": run_id,
                    "script_sha256": row.get("script_sha256"),
                    "artifact_dir": str(run_dir.relative_to(root)),
                })
            performance.extend(run_performance)
        for path in result_paths:
            if path.is_file():
                records = read_jsonl(path, errors) if path.suffix == ".jsonl" else []
                for record in records:
                    raw_files.append({
                        "suite_key": spec.suite_key,
                        "model": spec.label,
                        "run_id": run_id,
                        "is_current_evidence": is_current,
                        "source_file": str(path.relative_to(root)),
                        "record": record,
                    })
    return rows, raw_files, performance


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def aggregate(root: Path, scripts_dir: Path, include_baseline: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    rows, raw_records, performance = discover_rows(root, scripts_dir, include_baseline, errors)
    environment = read_json(root / "environment.json")
    input_manifest = read_json(root / "inputs" / "manifest.json")
    source_manifest = read_json(root / "source_manifest.json")
    output = {
        "schema_version": "lfm3-full-colab-results/v1",
        "gpu_claim": "Tesla T4 only. The notebook aborts before inference on any other device.",
        "scoring_note": "Scores stay within their own suites. Latency fields are not combined across inference paths.",
        "full_model_registry_count": 6,
        "baseline_registry_count": 2 if include_baseline else 0,
        "models": rows,
        "performance_measurements": performance,
        "environment": environment,
        "inputs": input_manifest,
        "source_manifest": source_manifest,
        "raw_answer_file": "raw_answers.jsonl",
        "artifact_parse_errors": errors,
    }
    json_write(root / "summary.json", output)
    with (root / "models.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (root / "performance.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PERFORMANCE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(performance)
    with (root / "raw_answers.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    json_write(root / "aggregation-errors.json", {"errors": errors})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate LFM3 all-model Colab results without mixing suite metrics.")
    parser.add_argument("--root", type=Path, default=Path("/content/lfm3-full-results"))
    parser.add_argument("--scripts-dir", type=Path, default=Path("/content/lfm3-full/scripts"))
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()
    output = aggregate(args.root, args.scripts_dir, args.include_baseline)
    print(json.dumps({
        "summary": str(args.root / "summary.json"),
        "models": len(output["models"]),
        "performance_measurements": len(output["performance_measurements"]),
        "raw_answers": str(args.root / output["raw_answer_file"]),
        "artifact_parse_errors": len(output["artifact_parse_errors"]),
    }, ensure_ascii=False, indent=2))
    return 1 if output["artifact_parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
