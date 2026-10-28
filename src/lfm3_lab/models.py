from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SuiteSpec:
    key: str
    label: str
    model_id: str
    revision: str
    runner: str
    aggregate_key: str
    max_repeats: int


SUITES: dict[str, SuiteSpec] = {
    "text": SuiteSpec(
        "text", "LFM3-1.7B", "LiquidAI/LFM3-1.7B",
        "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "run_video_tools.py", "text_extra", 20,
    ),
    "vision": SuiteSpec(
        "vision", "LFM3-VL-2B", "LiquidAI/LFM3-VL-2B",
        "f83d2b0a2296aaa277f394990a9b8b949ddf9691",
        "run_video_tools.py", "vision_extra", 20,
    ),
    "audio": SuiteSpec(
        "audio", "LFM3-Audio-2B", "LiquidAI/LFM3-Audio-2B",
        "9d49701c372d6011d900f6abb40c6b296fb706dd",
        "run_audio.py", "audio", 20,
    ),
    "omni": SuiteSpec(
        "omni", "LFM3-Omni-2B", "LiquidAI/LFM3-Omni-2B",
        "ce2458cbce4ad970f92032ed94e3d18d30b04f9c",
        "run_audio.py", "omni", 20,
    ),
    "d1": SuiteSpec(
        "d1", "d1-2B", "LiquidAI/d1-2B",
        "cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41",
        "run_decisions.py", "d1", 10,
    ),
    "d1_omni": SuiteSpec(
        "d1_omni", "d1-Omni-2B", "LiquidAI/d1-Omni-2B",
        "c8d48c08dada0191c6b362820f055cfb49c99426",
        "run_decisions.py", "d1_omni", 10,
    ),
}


@dataclass(frozen=True)
class TaskMetric:
    task_id: str
    stage: str
    sample_count: int | None
    median_seconds: float | None
    median_tokens_per_second: float | None
    median_rtf: float | None
    peak_allocated_gib: float | None
    peak_reserved_gib: float | None
    timing_scope: str | None


@dataclass(frozen=True)
class AnswerExample:
    task_id: str
    outcome: Literal["passed", "failed", "ungraded"]
    answer: str


@dataclass(frozen=True)
class RunResult:
    suite: str
    label: str
    model_id: str
    revision: str
    status: str
    path: Path
    summary_path: Path
    run_id: str | None
    passed: int
    attempted: int
    functional_failures: int
    infrastructure_errors: int
    format_failures: int
    ungraded_count: int
    task_metrics: tuple[TaskMetric, ...]
    examples: tuple[AnswerExample, ...]
    media_paths: tuple[Path, ...]
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class TextSample:
    answer: str
    elapsed_seconds: float | None
    peak_allocated_gib: float | None
    peak_reserved_gib: float | None


@dataclass(frozen=True)
class TextAnswer:
    prompt: str
    samples: tuple[TextSample, ...]
    elapsed_seconds: float | None
    model_id: str
    revision: str
    status: str
    path: Path
    stdout_path: Path
    stderr_path: Path

    @property
    def answer(self) -> str:
        return self.samples[0].answer if self.samples else ""
