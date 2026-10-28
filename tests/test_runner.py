from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3_lab import InfrastructureError, Lab, LabError, SUITES  # noqa: E402
from lfm3_lab.display import show_result  # noqa: E402
from lfm3_lab.models import AnswerExample, RunResult, TaskMetric  # noqa: E402


def summary_for_text(*, infrastructure_errors: int = 0) -> dict[str, object]:
    return {
        "models": [{
            "suite_key": "text_extra",
            "model": "LFM3-1.7B",
            "status": "completed_with_failures" if not infrastructure_errors else "completed_with_errors",
            "passed": 1,
            "attempted": 2,
            "functional_failures": 1 if not infrastructure_errors else 0,
            "infrastructure_errors": infrastructure_errors,
            "format_failures": 0,
            "is_current_evidence": True,
            "summary_path": "text_extra/run-1/summary.json",
            "run_id": "run-1",
        }],
        "performance_measurements": [{
            "suite_key": "text_extra",
            "run_id": "run-1",
            "task_id": "text-example",
            "stage": "answer",
            "sample_count": 2,
            "median_seconds": 0.4,
            "median_tokens_per_second": 12.5,
            "peak_allocated_gib": 2.5,
            "peak_reserved_gib": 2.8,
            "timing_scope": "generation only",
        }],
    }


class LabRunnerTests(unittest.TestCase):
    def test_registry_has_six_fixed_models_and_matches_runner_pins(self) -> None:
        self.assertEqual(tuple(SUITES), ("text", "vision", "audio", "omni", "d1", "d1_omni"))
        self.assertEqual(len({(item.model_id, item.revision) for item in SUITES.values()}), 6)
        Lab(ROOT / "unused-results")._ensure_source_tree()

    def test_omni_suites_require_the_audio_cell_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            inputs = output / "inputs"
            inputs.mkdir()
            (inputs / "manifest.json").write_text("{}", encoding="utf-8")
            lab = Lab(output)
            for suite in ("omni", "d1_omni"):
                with self.subTest(suite=suite), self.assertRaisesRegex(LabError, "先に音声の章を実行してください"):
                    lab._require_prepared_inputs(SUITES[suite])

    def test_run_returns_functional_failures_as_result_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            inputs = output / "inputs"
            inputs.mkdir()
            (inputs / "manifest.json").write_text("{}", encoding="utf-8")
            (inputs / "scene.png").write_bytes(b"image fixture")
            stdout_path = output / "runner.stdout.log"
            stderr_path = output / "runner.stderr.log"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            invocation = SimpleNamespace(returncode=0, stdout_path=stdout_path, stderr_path=stderr_path)
            lab = Lab(output)
            with patch.object(lab, "_run_process", return_value=invocation), patch.object(
                lab, "_aggregate", return_value=summary_for_text()
            ):
                result = lab.run("text", repeats=2)
            self.assertEqual(result.status, "completed_with_failures")
            self.assertEqual((result.passed, result.attempted, result.functional_failures), (1, 2, 1))
            self.assertEqual(result.infrastructure_errors, 0)
            self.assertEqual(result.task_metrics[0].task_id, "text-example")

    def test_run_raises_for_infrastructure_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            inputs = output / "inputs"
            inputs.mkdir()
            (inputs / "manifest.json").write_text("{}", encoding="utf-8")
            (inputs / "scene.png").write_bytes(b"image fixture")
            stdout_path = output / "runner.stdout.log"
            stderr_path = output / "runner.stderr.log"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            invocation = SimpleNamespace(returncode=1, stdout_path=stdout_path, stderr_path=stderr_path)
            lab = Lab(output)
            with patch.object(lab, "_run_process", return_value=invocation), patch.object(
                lab, "_aggregate", return_value=summary_for_text(infrastructure_errors=1)
            ):
                with self.assertRaisesRegex(InfrastructureError, "recorded 1 infrastructure error"):
                    lab.run("text")

    def test_run_rejects_prior_summary_when_current_runner_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            inputs = output / "inputs"
            inputs.mkdir()
            (inputs / "manifest.json").write_text("{}", encoding="utf-8")
            (inputs / "scene.png").write_bytes(b"image fixture")
            old_summary = output / "text_extra" / "run-old" / "summary.json"
            old_summary.parent.mkdir(parents=True)
            old_summary.write_text('{"run_id":"run-old"}', encoding="utf-8")
            stdout_path = output / "runner.stdout.log"
            stderr_path = output / "runner.stderr.log"
            stdout_path.write_text("runner crashed before writing results\n", encoding="utf-8")
            stderr_path.write_text("runtime error\n", encoding="utf-8")
            invocation = SimpleNamespace(returncode=1, stdout_path=stdout_path, stderr_path=stderr_path)
            stale_row = summary_for_text()["models"][0]
            stale_row["summary_path"] = "text_extra/run-old/summary.json"
            stale_row["run_id"] = "run-old"
            lab = Lab(output)
            with patch.object(lab, "_run_process", return_value=invocation), patch.object(
                lab, "_aggregate", return_value={"models": [stale_row], "performance_measurements": []}
            ):
                with self.assertRaisesRegex(InfrastructureError, "以前の結果が集計されました") as raised:
                    lab.run("text")
            self.assertIn(str(stdout_path), str(raised.exception))
            self.assertIn(str(stderr_path), str(raised.exception))

    def test_show_result_escapes_model_output_and_limits_answer_size(self) -> None:
        answer = "<script>alert('x')</script>" + ("x" * 1500)
        result = RunResult(
            suite="text",
            label="LFM3-1.7B",
            model_id=SUITES["text"].model_id,
            revision=SUITES["text"].revision,
            status="completed_with_failures",
            path=Path("run"),
            summary_path=Path("run/summary.json"),
            run_id="run-1",
            passed=1,
            attempted=2,
            functional_failures=1,
            infrastructure_errors=0,
            format_failures=0,
            ungraded_count=0,
            task_metrics=(TaskMetric("task", "answer", 2, 0.2, None, None, 2.0, 2.2, "generation"),),
            examples=(AnswerExample("text_tool_add_numbers_no_think", "failed", answer[:1200]),),
            media_paths=(),
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
        )
        with patch("lfm3_lab.display._display_html") as show_html:
            show_result(result)
        visible_html = show_html.call_args_list[0].args[0]
        detail_html = show_html.call_args_list[1].args[0]
        self.assertIn("&lt;script&gt;", visible_html)
        self.assertNotIn("<script>", visible_html)
        self.assertIn("テキスト加算ツール", visible_html)
        self.assertNotIn("text_tool_add_numbers_no_think", visible_html)
        self.assertIn("text_tool_add_numbers_no_think", detail_html)

    def test_ungraded_audio_count_is_preserved_for_display(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            inputs = output / "inputs"
            inputs.mkdir()
            for name in ("manifest.json", "scene.png", "tone.wav", "sequence_video.mp4", "speech_reference.wav", "speech_reference.json"):
                (inputs / name).write_bytes(b"input fixture")
            stdout_path = output / "runner.stdout.log"
            stderr_path = output / "runner.stderr.log"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            invocation = SimpleNamespace(returncode=0, stdout_path=stdout_path, stderr_path=stderr_path)
            summary = {
                "models": [{
                    "suite_key": "audio", "model": "LFM3-Audio-2B", "status": "completed_with_failures",
                    "passed": 3, "attempted": 4, "functional_failures": 1,
                    "infrastructure_errors": 0, "format_failures": 0,
                    "completed_without_pass_grade": 8, "is_current_evidence": True,
                    "summary_path": "audio/runs/run-1/summary.json", "run_id": "run-1",
                }],
                "performance_measurements": [],
            }
            lab = Lab(output)
            with patch.object(lab, "_run_process", return_value=invocation), patch.object(
                lab, "_aggregate", return_value=summary
            ):
                result = lab.run("audio")
            self.assertEqual(result.ungraded_count, 8)

    def test_export_contains_results_and_does_not_recurse_into_prior_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "summary.json").write_text("{}", encoding="utf-8")
            (output / "process_logs").mkdir()
            (output / "process_logs" / "stdout.log").write_text("runner output", encoding="utf-8")
            lab = Lab(output)
            with patch.object(lab, "_aggregate", return_value={"models": []}):
                first = lab.export()
                second = lab.export()
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            with zipfile.ZipFile(second) as archive:
                names = archive.namelist()
            self.assertIn("summary.json", names)
            self.assertIn("process_logs/stdout.log", names)
            self.assertFalse(any(name.endswith(first.name) for name in names))

    def test_cli_help_lists_the_supported_commands(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-m", "lfm3_lab", "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("prepare", "run", "summarize", "export"):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
