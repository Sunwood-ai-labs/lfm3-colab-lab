from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .models import AnswerExample, RunResult, SUITES, SuiteSpec, TaskMetric, TextAnswer, TextSample


DEFAULT_OUTPUT_DIR = Path(os.environ.get("LFM3_RESULTS_DIR", "/content/lfm3-results"))
DEFAULT_CACHE_DIR = Path(os.environ.get("LFM3_CACHE_DIR", "/content/lfm3-cache"))
_ANSWER_KEYS = ("final_response", "raw_text", "raw_output", "raw_result", "answer", "response")
_MEDIA_SUFFIXES = {".wav", ".mp3", ".mp4", ".png", ".jpg", ".jpeg", ".webp"}
_PROGRESS_LABELS = {
    "text": "テキスト（LFM3-1.7B）",
    "vision": "画像・動画（LFM3-VL-2B）",
    "audio": "音声（LFM3-Audio-2B）",
    "omni": "音声・動画（LFM3-Omni-2B）",
    "d1": "推論（d1-2B）",
    "d1_omni": "音声・推論（d1-Omni-2B）",
}


class LabError(RuntimeError):
    """An invalid lab request or missing prerequisite."""


class InfrastructureError(LabError):
    """A runner could not complete because its runtime or files failed."""


@dataclass(frozen=True)
class _Invocation:
    returncode: int
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path
    record_path: Path


def _utc_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfrastructureError(f"Could not read result JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InfrastructureError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _short_tail(path: Path, line_count: int = 8) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


class Lab:
    """Run the six fixed LFM3 suites and keep their artifacts under one result root."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
        *,
        cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.repo_root = Path(__file__).resolve().parents[2]
        package_experiments = Path(__file__).resolve().parent / "experiments"
        self.experiments_dir = (
            package_experiments if package_experiments.is_dir() else self.repo_root / "experiments"
        )
        self._aggregate_module: Any | None = None

    def prepare(self) -> dict[str, Any]:
        """Build the fixed image, audio, and video inputs and return their manifest."""
        self._ensure_source_tree()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_source_manifest()
        invocation = self._run_process(
            "prepare",
            [
                sys.executable,
                str(self.experiments_dir / "prepare_inputs.py"),
                "--output-dir", str(self.output_dir),
                "--cache-dir", str(self.cache_dir),
            ],
            display_label="固定入力の準備",
        )
        self._require_success(invocation, "input preparation")
        manifest = _read_json(self.output_dir / "inputs" / "manifest.json")
        required = ("scene.png", "tone.wav", "sequence_video.mp4", "speech_reference.wav", "speech_reference.json")
        missing = [name for name in required if not (self.output_dir / "inputs" / name).is_file()]
        if missing:
            raise InfrastructureError(f"Input preparation omitted required files: {', '.join(missing)}")
        return manifest

    def run(self, suite: str, repeats: int = 3) -> RunResult:
        """Run one fixed suite in a new run directory and preserve all earlier runs."""
        spec = self._suite(suite)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= spec.max_repeats:
            raise LabError(f"repeats for {suite!r} must be between 1 and {spec.max_repeats}")
        self._ensure_source_tree()
        self._require_prepared_inputs(spec)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prior_result_paths = self._existing_result_paths(self.output_dir / spec.aggregate_key)
        self._write_source_manifest()
        invocation = self._run_process(
            spec.aggregate_key,
            [
                sys.executable,
                str(self.experiments_dir / spec.runner),
                "--model", spec.key,
                "--output-dir", str(self.output_dir),
                "--repeats", str(repeats),
            ],
            display_label=_PROGRESS_LABELS[spec.key],
        )
        summary = self._aggregate()
        row = next((model for model in summary.get("models", []) if model.get("suite_key") == spec.aggregate_key), None)
        if row is None or not row.get("summary_path"):
            details = _short_tail(invocation.stderr_path) or _short_tail(invocation.stdout_path)
            raise InfrastructureError(
                f"{suite} runner exited {invocation.returncode} without a result summary. "
                f"Logs: {invocation.stdout_path}, {invocation.stderr_path}. {details}"
            )
        summary_path = (self.output_dir / str(row["summary_path"])).resolve()
        if summary_path in prior_result_paths:
            raise InfrastructureError(
                f"{suite} の今回の実行で新しい結果が作成されず、以前の結果が集計されました。"
                f"古い結果は採用しません。ログを確認し、原因を修正してから再実行してください。"
                f"標準出力ログ: {invocation.stdout_path}。標準エラーログ: {invocation.stderr_path}。"
            )
        if not row.get("is_current_evidence"):
            raise InfrastructureError(
                f"{suite} result does not match the checked runner and model pins: "
                f"{row.get('currentness') or row.get('status')}. "
                f"Logs: {invocation.stdout_path}, {invocation.stderr_path}"
            )
        infra = _integer(row.get("infrastructure_errors"))
        if infra:
            detail = _short_tail(invocation.stderr_path) or _short_tail(invocation.stdout_path)
            raise InfrastructureError(
                f"{suite} recorded {infra} infrastructure error(s). "
                f"Run summary: {self.output_dir / row['summary_path']}. "
                f"Logs: {invocation.stdout_path}, {invocation.stderr_path}. {detail}"
            )
        if invocation.returncode and not (_integer(row.get("functional_failures")) or _integer(row.get("format_failures"))):
            detail = _short_tail(invocation.stderr_path) or _short_tail(invocation.stdout_path)
            raise InfrastructureError(
                f"{suite} runner exited {invocation.returncode} without a recorded functional failure. "
                f"Logs: {invocation.stdout_path}, {invocation.stderr_path}. {detail}"
            )
        return RunResult(
            suite=spec.key,
            label=spec.label,
            model_id=spec.model_id,
            revision=spec.revision,
            status=str(row.get("status", "unknown")),
            path=summary_path.parent,
            summary_path=summary_path,
            run_id=row.get("run_id"),
            passed=_integer(row.get("passed")),
            attempted=_integer(row.get("attempted")),
            functional_failures=_integer(row.get("functional_failures")),
            infrastructure_errors=infra,
            format_failures=_integer(row.get("format_failures")),
            ungraded_count=_integer(row.get("completed_without_pass_grade")),
            task_metrics=self._task_metrics(summary, spec),
            examples=self._answer_examples(spec, row.get("run_id")),
            media_paths=self._media_paths(spec, row.get("run_id")),
            stdout_path=invocation.stdout_path,
            stderr_path=invocation.stderr_path,
        )

    def run_text(self, prompt: str, repeats: int = 1) -> TextAnswer:
        """Run an editable, ungraded text prompt against the pinned LFM3-1.7B model."""
        spec = self._suite("text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise LabError("prompt must be a nonempty string")
        if len(prompt) > 16000:
            raise LabError("prompt must be 16,000 characters or fewer")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= spec.max_repeats:
            raise LabError(f"repeats must be between 1 and {spec.max_repeats}")
        self._ensure_source_tree()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_source_manifest()
        run_dir = self.output_dir / "text_prompt" / _utc_id()
        run_dir.mkdir(parents=True, exist_ok=False)
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        invocation = self._run_process(
            "text-playground",
            [
                sys.executable,
                str(self.experiments_dir / "run_text_prompt.py"),
                "--prompt-file", str(prompt_path),
                "--output-dir", str(run_dir),
                "--repeats", str(repeats),
            ],
            display_label="テキスト自由入力",
        )
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            detail = _short_tail(invocation.stderr_path) or _short_tail(invocation.stdout_path)
            raise InfrastructureError(
                f"Text playground did not write a summary. Logs: {invocation.stdout_path}, "
                f"{invocation.stderr_path}. {detail}"
            )
        summary = _read_json(summary_path)
        if invocation.returncode or summary.get("status") == "infrastructure_error":
            detail = _short_tail(invocation.stderr_path) or _short_tail(invocation.stdout_path)
            raise InfrastructureError(
                f"Text playground could not run on the pinned T4 setup. "
                f"Logs: {invocation.stdout_path}, {invocation.stderr_path}. {detail}"
            )
        samples = tuple(
            TextSample(
                answer=str(item.get("answer", ""))[:4000],
                elapsed_seconds=_number(item.get("elapsed_seconds")),
                peak_allocated_gib=_number(item.get("peak_allocated_gib")),
                peak_reserved_gib=_number(item.get("peak_reserved_gib")),
            )
            for item in summary.get("samples", [])
            if isinstance(item, dict)
        )
        return TextAnswer(
            prompt=prompt,
            samples=samples,
            elapsed_seconds=_number(summary.get("median_seconds")),
            model_id=spec.model_id,
            revision=spec.revision,
            status=str(summary.get("status", "unknown")),
            path=run_dir,
            stdout_path=invocation.stdout_path,
            stderr_path=invocation.stderr_path,
        )

    def summarize(self) -> list[dict[str, Any]]:
        """Return six compact suite rows without raw answers or nested logs."""
        summary = self._aggregate()
        columns = (
            "suite_key", "model", "status", "passed", "attempted",
            "functional_failures", "infrastructure_errors", "format_failures", "completed_without_pass_grade",
        )
        api_key_by_aggregate = {suite.aggregate_key: suite.key for suite in SUITES.values()}
        compact = []
        for row in summary.get("models", []):
            item = {key: row.get(key) for key in columns}
            item["suite_key"] = api_key_by_aggregate.get(str(row.get("suite_key")), row.get("suite_key"))
            item["ungraded_count"] = item.pop("completed_without_pass_grade", 0)
            compact.append(item)
        return compact

    def export(self) -> Path:
        """Create a ZIP of prepared inputs, run artifacts, logs, and compact summaries."""
        self._aggregate()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        archive = self.output_dir / f"lfm3-results-export-{_utc_id()}.zip"
        generated_archives = set(self.output_dir.glob("lfm3-results-export-*.zip"))
        generated_archives.add(archive)
        try:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
                for path in sorted(self.output_dir.rglob("*")):
                    if not path.is_file() or path in generated_archives:
                        continue
                    bundle.write(path, path.relative_to(self.output_dir).as_posix())
        except OSError as exc:
            archive.unlink(missing_ok=True)
            raise InfrastructureError(f"Could not create result export {archive}: {exc}") from exc
        return archive

    def show(self, result: RunResult) -> None:
        """Display a bounded result table, selected answers, and one relevant media artifact."""
        from .display import show_result

        show_result(result)

    def show_inputs(self) -> dict[str, Any]:
        """Display the prepared scene and audio/video examples and return the manifest."""
        manifest_path = self.output_dir / "inputs" / "manifest.json"
        if not manifest_path.is_file():
            raise LabError("先に `lab.prepare()` を実行してください。")
        manifest = _read_json(manifest_path)
        from .display import show_inputs

        show_inputs(self.output_dir / "inputs", manifest)
        return manifest

    def _suite(self, key: str) -> SuiteSpec:
        try:
            return SUITES[key]
        except KeyError as exc:
            choices = ", ".join(SUITES)
            raise LabError(f"Unknown suite {key!r}. Choose one of: {choices}") from exc

    def _require_prepared_inputs(self, spec: SuiteSpec) -> None:
        inputs = self.output_dir / "inputs"
        if not (inputs / "manifest.json").is_file():
            raise LabError("先に `lab.prepare()` を実行してください。")
        if spec.key in {"omni", "d1_omni"} and not (inputs / "voice_question.wav").is_file():
            raise LabError("先に音声の章を実行してください（`lab.run('audio')`）。")
        required = {
            "text": ("scene.png",),
            "vision": ("scene.png", "sequence_video.mp4"),
            "audio": ("speech_reference.wav", "speech_reference.json", "scene.png", "sequence_video.mp4"),
            "omni": ("scene.png", "sequence_video.mp4", "voice_question.wav"),
            "d1": ("scene.png",),
            "d1_omni": ("scene.png", "speech_reference.wav", "speech_reference.json", "voice_question.wav"),
        }[spec.key]
        missing = [name for name in required if not (inputs / name).is_file()]
        if missing:
            raise LabError(f"Missing prepared input(s) for {spec.key}: {', '.join(missing)}. Run `lab.prepare()`.")

    @staticmethod
    def _existing_result_paths(suite_root: Path) -> set[Path]:
        """Snapshot prior per-run result files so aggregation cannot reuse them as new output."""
        if not suite_root.is_dir():
            return set()
        return {
            path.resolve()
            for path in suite_root.rglob("*")
            if path.is_file() and path.name in {"summary.json", "results.json"}
        }

    def _ensure_source_tree(self) -> None:
        required = {
            "run_video_tools.py", "run_audio.py", "run_decisions.py",
            "prepare_inputs.py", "aggregate_results.py", "baseline.py", "run_text_prompt.py",
        }
        missing = sorted(name for name in required if not (self.experiments_dir / name).is_file())
        if missing:
            raise InfrastructureError(f"Public experiment source is incomplete: {', '.join(missing)}")
        self._validate_registry()

    def _load_aggregator(self) -> Any:
        if self._aggregate_module is not None:
            return self._aggregate_module
        path = self.experiments_dir / "aggregate_results.py"
        module_name = f"_lfm3_lab_aggregate_{id(self)}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise InfrastructureError(f"Could not load result aggregator {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise InfrastructureError(f"Could not load result aggregator {path}: {exc}") from exc
        self._aggregate_module = module
        return module

    def _validate_registry(self) -> None:
        module = self._load_aggregator()
        for suite in SUITES.values():
            try:
                models = module.model_specs(self.experiments_dir / suite.runner)
                actual = models[suite.key]
            except (OSError, SyntaxError, ValueError, KeyError) as exc:
                raise InfrastructureError(f"Could not validate fixed model registry for {suite.key}: {exc}") from exc
            if (actual.get("model_id"), actual.get("revision")) != (suite.model_id, suite.revision):
                raise InfrastructureError(
                    f"Fixed model registry mismatch for {suite.key}: "
                    f"expected {suite.model_id}@{suite.revision}, got "
                    f"{actual.get('model_id')}@{actual.get('revision')}"
                )
        full_suites = {item.suite_key for item in module.SUITES if item.group == "full"}
        if full_suites != {suite.aggregate_key for suite in SUITES.values()}:
            raise InfrastructureError("The six-suite public registry no longer matches the audited aggregator")

    def _write_source_manifest(self) -> None:
        self._validate_registry()
        sources = []
        for path in sorted(self.experiments_dir.glob("*.py")):
            sources.append({"path": f"experiments/{path.name}", "sha256": _sha256(path), "bytes": path.stat().st_size})
        _write_json(self.output_dir / "source_manifest.json", {
            "schema_version": "lfm3-colab-lab-source/v1",
            "source_location": "experiments/",
            "suites": [
                {"key": suite.key, "model_id": suite.model_id, "revision": suite.revision, "runner": suite.runner}
                for suite in SUITES.values()
            ],
            "sources": sources,
        })

    def _run_process(
        self,
        label: str,
        command: Sequence[str],
        *,
        display_label: str | None = None,
        announce_start: bool = True,
    ) -> _Invocation:
        invocation_id = _utc_id()
        log_dir = self.output_dir / "process_logs" / invocation_id
        log_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = log_dir / f"{label}.stdout.log"
        stderr_path = log_dir / f"{label}.stderr.log"
        record_path = log_dir / f"{label}.json"
        env = os.environ.copy()
        env.update({
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTHONUNBUFFERED": "1",
        })
        started = time.perf_counter()
        start_time = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            process = subprocess.Popen(
                list(command),
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            _write_json(record_path, {
                "command": list(command), "returncode": 127, "elapsed_seconds": 0.0,
                "stdout_path": stdout_path.relative_to(self.output_dir).as_posix(),
                "stderr_path": stderr_path.relative_to(self.output_dir).as_posix(),
                "spawn_error": f"{type(exc).__name__}: {exc}", "started_at_utc": start_time,
            })
            raise InfrastructureError(f"Could not start {label}: {exc}. Logs: {stdout_path}, {stderr_path}") from exc

        def drain(stream: Any, path: Path) -> None:
            with stream, path.open("w", encoding="utf-8", newline="\n") as handle:
                for line in iter(stream.readline, ""):
                    handle.write(line)
                handle.flush()

        stdout_thread = threading.Thread(target=drain, args=(process.stdout, stdout_path))
        stderr_thread = threading.Thread(target=drain, args=(process.stderr, stderr_path))
        progress_label = display_label or label
        if announce_start:
            print(f"{progress_label}を開始しました。実行ログを保存しています。", flush=True)
        stdout_thread.start()
        stderr_thread.start()
        while True:
            try:
                returncode = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = self._elapsed_japanese(time.perf_counter() - started)
                print(f"{progress_label}を実行中です（経過時間：{elapsed}）。", flush=True)
        stdout_thread.join()
        stderr_thread.join()
        elapsed = time.perf_counter() - started
        _write_json(record_path, {
            "command": list(command),
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "started_at_utc": start_time,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stdout_path": stdout_path.relative_to(self.output_dir).as_posix(),
            "stderr_path": stderr_path.relative_to(self.output_dir).as_posix(),
        })
        return _Invocation(returncode, elapsed, stdout_path, stderr_path, record_path)

    @staticmethod
    def _elapsed_japanese(seconds: float) -> str:
        elapsed = max(0, int(seconds))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}時間{minutes}分"
        if minutes:
            return f"{minutes}分{seconds}秒" if seconds else f"{minutes}分"
        return f"{seconds}秒"

    def _require_success(self, invocation: _Invocation, action: str) -> None:
        if invocation.returncode == 0:
            return
        detail = _short_tail(invocation.stderr_path) or _short_tail(invocation.stdout_path)
        raise InfrastructureError(
            f"{action} exited {invocation.returncode}. Logs: {invocation.stdout_path}, "
            f"{invocation.stderr_path}. {detail}"
        )

    def _aggregate(self) -> dict[str, Any]:
        self._ensure_source_tree()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        invocation = self._run_process(
            "aggregate",
            [
                sys.executable,
                str(self.experiments_dir / "aggregate_results.py"),
                "--root", str(self.output_dir),
                "--scripts-dir", str(self.experiments_dir),
            ],
            display_label="結果の集計",
            announce_start=False,
        )
        self._require_success(invocation, "result aggregation")
        summary = _read_json(self.output_dir / "summary.json")
        errors = summary.get("artifact_parse_errors", [])
        if errors:
            sample = errors[0]
            raise InfrastructureError(
                f"Result aggregation found {len(errors)} artifact error(s); first: "
                f"{sample.get('path') or sample.get('stage')}: {sample.get('error')}"
            )
        return summary

    def _task_metrics(self, summary: dict[str, Any], spec: SuiteSpec) -> tuple[TaskMetric, ...]:
        row = next((model for model in summary.get("models", []) if model.get("suite_key") == spec.aggregate_key), {})
        run_id = row.get("run_id")
        output = []
        for item in summary.get("performance_measurements", []):
            if item.get("suite_key") != spec.aggregate_key or (run_id and item.get("run_id") != run_id):
                continue
            output.append(TaskMetric(
                task_id=str(item.get("task_id") or "unknown"),
                stage=str(item.get("stage") or item.get("measurement_type") or "inference"),
                sample_count=_integer(item.get("sample_count")) or None,
                median_seconds=_number(item.get("median_seconds")),
                median_tokens_per_second=_number(item.get("median_tokens_per_second")),
                median_rtf=_number(item.get("median_rtf")),
                peak_allocated_gib=_number(item.get("peak_allocated_gib")),
                peak_reserved_gib=_number(item.get("peak_reserved_gib")),
                timing_scope=str(item.get("timing_scope")) if item.get("timing_scope") is not None else None,
            ))
        return tuple(output)

    def _current_records(self, suite: SuiteSpec, run_id: str | None) -> list[dict[str, Any]]:
        path = self.output_dir / "raw_answers.jsonl"
        if not path.is_file():
            return []
        records = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("suite_key") != suite.aggregate_key or not item.get("is_current_evidence"):
                        continue
                    if run_id and item.get("run_id") != run_id:
                        continue
                    record = item.get("record")
                    if isinstance(record, dict):
                        records.append(record)
        except OSError as exc:
            raise InfrastructureError(f"Could not read aggregated raw answers {path}: {exc}") from exc
        return records

    def _answer_examples(self, suite: SuiteSpec, run_id: str | None) -> tuple[AnswerExample, ...]:
        passed: AnswerExample | None = None
        failed: AnswerExample | None = None
        for record in self._current_records(suite, run_id):
            task_id = str(record.get("case_id") or record.get("task_id") or "task")
            score = record.get("task_score")
            repetitions = record.get("repetitions")
            if isinstance(score, dict) and isinstance(repetitions, list):
                for repetition in repetitions:
                    if not isinstance(repetition, dict):
                        continue
                    outcome = self._repetition_outcome(repetition)
                    answer = self._answer_text(repetition)
                    if not answer:
                        continue
                    example = AnswerExample(task_id, outcome or "ungraded", answer[:1200])
                    if example.outcome == "passed" and passed is None:
                        passed = example
                    if example.outcome == "failed" and failed is None:
                        failed = example
                if (passed is None or failed is None) and self._answer_text(record):
                    value = self._answer_text(record)
                    attempted = _integer(score.get("attempted"))
                    count = _integer(score.get("passed"))
                    outcome = "passed" if count > 0 and count == attempted else "failed" if count < attempted else "ungraded"
                    if outcome == "passed" and passed is None:
                        passed = AnswerExample(task_id, outcome, value[:1200])
                    elif outcome == "failed" and failed is None:
                        failed = AnswerExample(task_id, outcome, value[:1200])
                continue
            outcome = self._repetition_outcome(record)
            if outcome is None:
                continue
            answer = self._answer_text(record)
            if not answer:
                continue
            example = AnswerExample(task_id, outcome, answer[:1200])
            if outcome == "passed" and passed is None:
                passed = example
            elif outcome == "failed" and failed is None:
                failed = example
            if passed is not None and failed is not None:
                break
        return tuple(example for example in (passed, failed) if example is not None)

    @staticmethod
    def _repetition_outcome(record: dict[str, Any]) -> str | None:
        status = str(record.get("status", "")).casefold()
        if status == "passed":
            return "passed"
        if status in {"failed", "format_error", "complete_with_functional_failures", "completed_with_failures"}:
            return "failed"
        if status in {"complete", "completed"}:
            return "ungraded"
        score = record.get("task_score")
        if isinstance(score, bool):
            return "passed" if score else "failed"
        if isinstance(score, (int, float)):
            return "passed" if score > 0 else "failed"
        score_value = record.get("score")
        if isinstance(score_value, bool):
            return "passed" if score_value else "failed"
        if isinstance(score, dict) and "passed" in score and "attempted" in score:
            attempted = _integer(score.get("attempted"))
            passed = _integer(score.get("passed"))
            return "passed" if attempted and passed == attempted else "failed" if attempted else None
        for key in ("all_correct", "correct", "exact_match"):
            value = record.get(key)
            if isinstance(value, bool):
                return "passed" if value else "failed"
        return None

    @staticmethod
    def _answer_text(value: Any, depth: int = 0) -> str:
        if depth > 8:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in _ANSWER_KEYS:
                if key in value:
                    candidate = value[key]
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                    if isinstance(candidate, (dict, list)):
                        return json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            for key in ("generation", "final_generation", "tool_call_generation", "repetitions", "measured_repeats", "warmup"):
                if key not in value:
                    continue
                answer = Lab._answer_text(value[key], depth + 1)
                if answer:
                    return answer
        elif isinstance(value, list):
            for candidate in value:
                answer = Lab._answer_text(candidate, depth + 1)
                if answer:
                    return answer
        return ""

    def _media_paths(self, suite: SuiteSpec, run_id: str | None) -> tuple[Path, ...]:
        found: list[Path] = []
        records = self._current_records(suite, run_id)
        records.sort(key=lambda record: 0 if str(record.get("case_id", "")).startswith("tts-ja-blue-question") else 1)
        for record in records:
            for path_value in self._nested_media_paths(record):
                candidate = Path(path_value)
                if not candidate.is_absolute():
                    candidate = self.output_dir / candidate
                try:
                    candidate = candidate.resolve()
                    candidate.relative_to(self.output_dir)
                except (OSError, ValueError):
                    continue
                if candidate.suffix.casefold() in _MEDIA_SUFFIXES and candidate.is_file() and candidate not in found:
                    found.append(candidate)
                    if len(found) >= 2:
                        return tuple(found)
        input_names = {
            "text": ("scene.png",),
            "vision": ("sequence_video.mp4", "scene.png"),
            "audio": ("speech_reference.wav",),
            "omni": ("scene_video.mp4", "voice_question.wav"),
            "d1": ("scene.png",),
            "d1_omni": ("speech_reference.wav", "scene.png"),
        }[suite.key]
        inputs = self.output_dir / "inputs"
        for name in input_names:
            candidate = inputs / name
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
                if len(found) >= 2:
                    break
        return tuple(found)

    @staticmethod
    def _nested_media_paths(value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            path_value = value.get("path")
            if isinstance(path_value, str) and Path(path_value).suffix.casefold() in _MEDIA_SUFFIXES:
                found.append(path_value)
            for item in value.values():
                found.extend(Lab._nested_media_paths(item))
        elif isinstance(value, list):
            for item in value:
                found.extend(Lab._nested_media_paths(item))
        return found
