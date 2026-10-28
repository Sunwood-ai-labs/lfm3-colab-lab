#!/usr/bin/env python3
"""Run representative, typed System One decisions on the pinned D1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import tempfile
import time
import unicodedata
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "d1": {
        "repo_id": "LiquidAI/d1-2B",
        "revision": "cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41",
    },
    "d1_omni": {
        "repo_id": "LiquidAI/d1-Omni-2B",
        "revision": "c8d48c08dada0191c6b362820f055cfb49c99426",
    },
}

TEXT_STATE = (
    "Support ticket R-314. The customer writes: 'Our credit card was charged twice for the same order. "
    "Please refund one charge today. I also cannot sign in to my account and need the invoice today. "
    "The parcel arrived normally.'\n\n"
    "Invoice issued by ACME Logistics, Inc. Invoice 2026-118. Total due: USD 125.75. "
    "Pay before: 2026-11-15. No VAT amount is listed.\n\n"
    "The support queue currently has exactly 7 open service tickets."
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    state: Any
    questions: dict[str, dict[str, Any]]
    expected: dict[str, dict[str, Any]]
    media: list[Any]
    input_files: tuple[str, ...]
    preparation_errors: tuple[dict[str, str], ...] = ()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_digest() -> str:
    return digest_file(Path(__file__).resolve())


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False) + "\n")


def error_record(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error)[:4000],
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def within_probability(value: Any) -> float | None:
    result = numeric(value)
    if result is None or not 0.0 <= result <= 1.0:
        return None
    return result


def normalized_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in text if not character.isspace())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def distribution_errors(
    probabilities: Any,
    keys: list[str],
    *,
    require_sum: bool,
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    if not isinstance(probabilities, dict):
        return {}, ["probabilities must be an object"]
    if set(probabilities) != set(keys):
        errors.append("probability keys do not match the question criteria")
    values: dict[str, float] = {}
    for key in keys:
        value = within_probability(probabilities.get(key))
        if value is None:
            errors.append(f"probability {key!r} is not finite and within [0, 1]")
        else:
            values[key] = value
    if require_sum and len(values) == len(keys) and abs(sum(values.values()) - 1.0) > 1e-3:
        errors.append("categorical probabilities do not sum to 1 within 0.001")
    return values, errors


def validation_result(
    *, format_errors: list[str], correct: bool, details: dict[str, Any]
) -> dict[str, Any]:
    return {
        "format_ok": not format_errors,
        "correct": bool(correct) and not format_errors,
        "format_errors": format_errors,
        "details": details,
    }


def validate_noul(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return validation_result(format_errors=["answer type must be noul"], correct=False, details={})
    probability = within_probability(answer.get("noul"))
    if probability is None:
        errors.append("noul probability is not finite and within [0, 1]")
    return validation_result(
        format_errors=errors,
        correct=probability is not None and (probability >= 0.5) is expected["yes"],
        details={"probability_yes": probability, "threshold": 0.5, "expected_yes": expected["yes"]},
    )


def validate_choice(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return validation_result(format_errors=["answer type must be choice"], correct=False, details={})
    criteria = list(question["criteria"])
    probabilities, errors = distribution_errors(answer.get("probabilities"), criteria, require_sum=True)
    choice = answer.get("choice")
    if choice not in criteria:
        errors.append("choice is not one of the question criteria")
    if probabilities and choice in probabilities:
        top_choice = max(criteria, key=lambda key: probabilities.get(key, -1.0))
        if choice != top_choice:
            errors.append("choice does not match the maximum-probability criterion")
    confidence = within_probability(answer.get("confidence"))
    if confidence is None:
        errors.append("confidence is not finite and within [0, 1]")
    elif choice in probabilities and abs(confidence - probabilities[choice]) > 1e-5:
        errors.append("confidence does not match the selected criterion probability")
    return validation_result(
        format_errors=errors,
        correct=choice == expected["choice"],
        details={"choice": choice, "expected_choice": expected["choice"], "probabilities": probabilities},
    )


def validate_multiple_choice(
    question: dict[str, Any], expected: dict[str, Any], answer: Any
) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return validation_result(format_errors=["answer type must be choice"], correct=False, details={})
    criteria = list(question["criteria"])
    probabilities, errors = distribution_errors(answer.get("probabilities"), criteria, require_sum=False)
    choices = answer.get("choices")
    if not isinstance(choices, list) or any(item not in criteria for item in choices):
        errors.append("choices must be a list of criterion names")
        choices = []
    derived = [name for name in criteria if name in probabilities and probabilities[name] >= 0.5]
    if choices != derived:
        errors.append("choices do not match criteria with probability at least 0.5")
    return validation_result(
        format_errors=errors,
        correct=set(choices) == set(expected["choices"]),
        details={
            "choices": choices,
            "expected_choices": expected["choices"],
            "probabilities_independent_yes": probabilities,
        },
    )


def validate_score(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "score":
        return validation_result(format_errors=["answer type must be score"], correct=False, details={})
    criteria = list(question["criteria"])
    keys = [str(index) for index in range(len(criteria))]
    probabilities, errors = distribution_errors(answer.get("probabilities"), keys, require_sum=True)
    legend = answer.get("legend")
    if not isinstance(legend, dict) or [legend.get(key) for key in keys] != criteria:
        errors.append("score legend does not match the ordered criteria")
    score = numeric(answer.get("score"))
    label = None
    if score is None or not score.is_integer() or not 0 <= int(score) < len(criteria):
        errors.append("score must be an integer index into the ordered criteria")
    else:
        label = criteria[int(score)]
        if probabilities:
            cumulative = 0.0
            expected_median = len(criteria) - 1
            for index, key in enumerate(keys):
                cumulative += probabilities.get(key, 0.0)
                if cumulative >= 0.5:
                    expected_median = index
                    break
            if int(score) != expected_median:
                errors.append("score does not match the documented median-level readout")
    confidence = within_probability(answer.get("confidence"))
    if confidence is None:
        errors.append("confidence is not finite and within [0, 1]")
    return validation_result(
        format_errors=errors,
        correct=label == expected["label"],
        details={"score_index": score, "label": label, "expected_label": expected["label"], "probabilities": probabilities},
    )


def validate_read(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    kind = expected["kind"]
    answer_type = {"read_text": "text", "read_number": "number", "read_date": "date", "read_absent": question["type"]}[kind]
    if not isinstance(answer, dict) or answer.get("type") != answer_type:
        return validation_result(format_errors=[f"answer type must be {answer_type}"], correct=False, details={})
    errors: list[str] = []
    if "found" not in answer:
        errors.append("found field is missing")
    if "span" not in answer:
        errors.append("span field is missing")
    found = within_probability(answer.get("found"))
    if found is None:
        errors.append("found is not finite and within [0, 1]")
    span = answer.get("span")
    field = expected.get("field", question["type"])
    value = answer.get(field)
    if kind == "read_absent":
        correct = found is not None and found < 0.5 and span is None and value is None
    else:
        if found is not None and found >= 0.5 and (span is None or value is None):
            errors.append("found is at least 0.5 but the extracted value or span is missing")
        if found is not None and found < 0.5 and (span is not None or value is not None):
            errors.append("found is below 0.5 but a value or span is present")
        if kind == "read_text":
            correct = normalized_text(value) == normalized_text(expected["value"])
        elif kind == "read_number":
            actual = numeric(value)
            correct = actual is not None and abs(actual - expected["value"]) <= expected.get("tolerance", 0.01)
        elif kind == "read_date":
            correct = value == expected["value"]
        else:
            correct = False
    details: dict[str, Any] = {
        "field": field,
        "value": value,
        "span": span,
        "found": found,
        "expected_value": expected.get("value"),
        "expected_found": kind != "read_absent",
    }
    if kind == "read_text" and expected.get("transcription"):
        hypothesis = normalized_text(value)
        reference = normalized_text(expected["value"])
        if hypothesis is not None and reference is not None:
            errors_count = edit_distance(reference, hypothesis)
            details["character_error_rate"] = errors_count / max(1, len(reference))
            details["exact_match"] = errors_count == 0
            correct = errors_count == 0
    if kind != "read_absent" and (found is None or found < 0.5):
        correct = False
    return validation_result(format_errors=errors, correct=correct, details=details)


def validate_estimate(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "number":
        return validation_result(format_errors=["answer type must be number"], correct=False, details={})
    errors: list[str] = []
    value = numeric(answer.get("number"))
    interval = answer.get("interval")
    low, high = question["range"]
    if value is None or not low <= value <= high:
        errors.append("estimate is not finite and within the requested range")
    if not isinstance(interval, list) or len(interval) != 2:
        errors.append("estimate interval must be a two-value list")
        bounds = None
    else:
        lower, upper = numeric(interval[0]), numeric(interval[1])
        if lower is None or upper is None or not low <= lower <= upper <= high:
            errors.append("estimate interval is not finite, ordered, and within the requested range")
            bounds = None
        else:
            bounds = [lower, upper]
            if value is not None and not lower <= value <= upper:
                errors.append("median estimate is outside its p10/p90 interval")
    tolerance = expected.get("tolerance", 0.5)
    correct = value is not None and abs(value - expected["value"]) <= tolerance
    return validation_result(
        format_errors=errors,
        correct=correct,
        details={"number": value, "interval_p10_p90": bounds, "expected_number": expected["value"], "tolerance": tolerance},
    )


def normalized_box(pixel_box: list[int], image_size: list[int]) -> dict[str, float]:
    x1, y1, x2, y2 = pixel_box
    width, height = image_size
    return {"top": y1 / height, "bottom": y2 / height, "left": x1 / width, "right": x2 / width}


def valid_box(box: Any) -> bool:
    if not isinstance(box, dict):
        return False
    values = {key: numeric(box.get(key)) for key in ("top", "bottom", "left", "right")}
    return (
        all(value is not None and 0.0 <= value <= 1.0 for value in values.values())
        and values["top"] < values["bottom"]
        and values["left"] < values["right"]
    )


def box_iou(left: dict[str, float], right: dict[str, float]) -> float:
    intersection = (
        max(0.0, min(left["bottom"], right["bottom"]) - max(left["top"], right["top"]))
        * max(0.0, min(left["right"], right["right"]) - max(left["left"], right["left"]))
    )
    area_left = (left["bottom"] - left["top"]) * (left["right"] - left["left"])
    area_right = (right["bottom"] - right["top"]) * (right["right"] - right["left"])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def validate_box(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "box":
        return validation_result(format_errors=["answer type must be box"], correct=False, details={})
    errors: list[str] = []
    found = within_probability(answer.get("found"))
    confidence = within_probability(answer.get("confidence"))
    if found is None:
        errors.append("box found is not finite and within [0, 1]")
    if confidence is None:
        errors.append("box confidence is not finite and within [0, 1]")
    predicted = answer.get("box")
    if found is not None and found >= 0.5 and not valid_box(predicted):
        errors.append("present box does not contain normalized top/bottom/left/right coordinates")
    if found is not None and found < 0.5 and predicted is not None:
        errors.append("box must be None when found is below 0.5")
    target = normalized_box(expected["pixel_box"], expected["image_size"])
    iou = box_iou(predicted, target) if valid_box(predicted) else None
    return validation_result(
        format_errors=errors,
        correct=found is not None and found >= 0.5 and iou is not None and iou >= 0.5,
        details={"box_normalized": predicted, "target_normalized": target, "found": found, "confidence": confidence, "box_iou": iou, "iou_threshold": 0.5},
    )


def validate_boxes(question: dict[str, Any], expected: dict[str, Any], answer: Any) -> dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != "boxes":
        return validation_result(format_errors=["answer type must be boxes"], correct=False, details={})
    errors: list[str] = []
    predicted_items = answer.get("boxes")
    if not isinstance(predicted_items, list):
        errors.append("boxes must be a list")
        predicted_items = []
    predicted = []
    for index, item in enumerate(predicted_items):
        if not isinstance(item, dict) or not valid_box(item.get("box")):
            errors.append(f"boxes[{index}] has invalid normalized geometry")
            continue
        if within_probability(item.get("found")) is None:
            errors.append(f"boxes[{index}].found is not finite and within [0, 1]")
        elif within_probability(item.get("found")) < 0.5:
            errors.append(f"boxes[{index}] must not be returned when found is below 0.5")
        if within_probability(item.get("confidence")) is None:
            errors.append(f"boxes[{index}].confidence is not finite and within [0, 1]")
        predicted.append(item["box"])
    count = answer.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(predicted_items):
        errors.append("count must be the number of returned boxes")
    presence = answer.get("presence")
    if not isinstance(presence, list) or len(presence) != question["max"]:
        errors.append("presence must contain one probability per requested box block")
    else:
        present_values = [within_probability(value) for value in presence]
        if any(value is None for value in present_values):
            errors.append("presence values must be finite and within [0, 1]")
        elif any(present_values[index] < present_values[index + 1] for index in range(len(present_values) - 1)):
            errors.append("presence probabilities must be nonincreasing")
    if not isinstance(answer.get("at_least"), bool):
        errors.append("at_least must be a boolean")
    targets = [normalized_box(box, expected["image_size"]) for box in expected["pixel_boxes"]]
    if len(predicted) == len(targets):
        assignment = max(
            itertools.permutations(range(len(predicted))),
            key=lambda order: sum(box_iou(predicted[order[index]], targets[index]) for index in range(len(targets))),
            default=(),
        )
        ious = [box_iou(predicted[assignment[index]], targets[index]) for index in range(len(targets))] if assignment else []
    else:
        ious = [max((box_iou(box, target) for box in predicted), default=0.0) for target in targets]
    expected_count = len(targets)
    correct_count = isinstance(count, int) and not isinstance(count, bool) and count == expected_count
    correct_boxes = len(predicted) == expected_count and len(ious) == expected_count and all(value >= 0.5 for value in ious)
    return validation_result(
        format_errors=errors,
        correct=correct_count and correct_boxes,
        details={
            "count": count,
            "expected_count": expected_count,
            "boxes_normalized": predicted,
            "targets_normalized": targets,
            "matched_box_ious": ious,
            "iou_threshold": 0.5,
            "at_least": answer.get("at_least"),
            "presence": presence,
        },
    )


ANSWER_VALIDATORS = {
    "noul": validate_noul,
    "choice": validate_choice,
    "multiple_choice": validate_multiple_choice,
    "score": validate_score,
    "read_text": validate_read,
    "read_number": validate_read,
    "read_date": validate_read,
    "read_absent": validate_read,
    "estimate": validate_estimate,
    "box": validate_box,
    "boxes": validate_boxes,
}


def evaluate_output(task: TaskSpec, raw: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return {"format_ok": False, "all_correct": False, "format_errors": ["system_one result must be an object"], "answers": {}}
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        return {"format_ok": False, "all_correct": False, "format_errors": ["result.answers must be an object"], "answers": {}}
    if set(answers) != set(task.expected):
        errors.append("answer names do not match the named questions")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        errors.append("result.usage must be an object")
    else:
        tokens = usage.get("input_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            errors.append("usage.input_tokens must be a nonnegative integer")
        if usage.get("output_tokens") != 0:
            errors.append("usage.output_tokens must be 0 for System One")
    scored: dict[str, Any] = {}
    for name, expected in task.expected.items():
        if name not in answers:
            scored[name] = validation_result(
                format_errors=["named answer is missing"], correct=False, details={}
            )
            continue
        validator = ANSWER_VALIDATORS[expected["kind"]]
        scored[name] = validator(task.questions[name], expected, answers[name])
    format_ok = not errors and all(result["format_ok"] for result in scored.values())
    all_correct = format_ok and all(result["correct"] for result in scored.values())
    return {"format_ok": format_ok, "all_correct": all_correct, "format_errors": errors, "answers": scored}


def text_task() -> TaskSpec:
    questions = {
        "refund": {"type": "noul", "instructions": "Is the customer asking for one of the duplicate card charges to be refunded?"},
        "routing": {
            "type": "choice",
            "instructions": "Which team should handle the duplicate card charge?",
            "criteria": {"billing": "Charges, refunds, and invoices", "delivery": "Shipping and parcels", "technical": "App or site faults"},
        },
        "issue_tags": {
            "type": "choice",
            "multiple": True,
            "instructions": "Which support issues does the customer report?",
            "criteria": {"billing": "A duplicate charge or refund request", "account_access": "Cannot sign in to the account", "delivery": "A parcel or delivery problem"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does the customer need help?",
            "criteria": ["Can wait", "Soon", "Needs attention today"],
        },
        "issuer": {"type": "text", "instructions": "Who issued the invoice?"},
        "total": {"type": "number", "instructions": "What is the invoice total in USD?", "unit": "USD"},
        "due_date": {"type": "date", "instructions": "What is the payment deadline?"},
        "vat": {"type": "number", "instructions": "How much VAT is charged?", "unit": "USD"},
        "open_tickets": {
            "type": "number",
            "instructions": "How many open service tickets are in the queue?",
            "range": [0, 10],
            "step": 1,
        },
    }
    expected = {
        "refund": {"kind": "noul", "yes": True},
        "routing": {"kind": "choice", "choice": "billing"},
        "issue_tags": {"kind": "multiple_choice", "choices": ["billing", "account_access"]},
        "urgency": {"kind": "score", "label": "Needs attention today"},
        "issuer": {"kind": "read_text", "field": "text", "value": "ACME Logistics, Inc."},
        "total": {"kind": "read_number", "field": "number", "value": 125.75, "tolerance": 0.01},
        "due_date": {"kind": "read_date", "field": "date", "value": "2026-11-15"},
        "vat": {"kind": "read_absent", "field": "number"},
        "open_tickets": {"kind": "estimate", "value": 7, "tolerance": 0.5},
    }
    return TaskSpec(
        task_id="text_typed_decisions",
        category="text_and_typed_decisions",
        state=TEXT_STATE,
        questions=questions,
        expected=expected,
        media=[],
        input_files=(),
    )


def image_task(scene: Any | None, scene_error: dict[str, str] | None) -> TaskSpec:
    questions = {
        "blue_square": {"type": "box", "instructions": "Locate the full outer boundary of the blue square, including its digit."},
        "all_squares": {"type": "boxes", "instructions": "Locate every full colored square in the picture.", "max": 2},
        "square_count": {
            "type": "number",
            "instructions": "How many filled colored squares are in the picture?",
            "range": [0, 4],
            "step": 1,
        },
    }
    expected = {
        "blue_square": {"kind": "box", "pixel_box": [340, 80, 540, 280], "image_size": [600, 360]},
        "all_squares": {
            "kind": "boxes",
            "pixel_boxes": [[40, 80, 240, 280], [340, 80, 540, 280]],
            "image_size": [600, 360],
        },
        "square_count": {"kind": "estimate", "value": 2, "tolerance": 0.5},
    }
    return TaskSpec(
        task_id="image_boxes_and_count",
        category="image_detection_and_counting",
        state=None,
        questions=questions,
        expected=expected,
        media=[] if scene is None else [scene],
        input_files=("scene.png",),
        preparation_errors=() if scene_error is None else (scene_error,),
    )


def audio_reference_task(
    audio_path: Path,
    audio_error: dict[str, str] | None,
    transcript: str | None,
    transcript_error: dict[str, str] | None,
    transcript_source: str | None,
) -> TaskSpec:
    questions = {
        "speech_present": {
            "type": "noul",
            "instructions": "Does this recording contain intelligible Japanese speech?",
        },
        "topic": {
            "type": "choice",
            "instructions": "What is the main topic of the spoken recording?",
            "criteria": {
                "waterway_transport": "Why yachts or boats are needed to travel among islands or across lakes",
                "aircraft_transport": "Why aircraft are needed for transportation",
                "other_topic": "A different main topic",
            },
        },
        "mentions_aircraft": {
            "type": "noul",
            "instructions": "Does the speaker mention an airplane, aircraft, or helicopter?",
        },
    }
    issues = tuple(error for error in (audio_error, transcript_error) if error is not None)
    expected = {
        "speech_present": {
            "kind": "noul",
            "yes": bool(transcript and transcript.strip()),
            "ground_truth_source": f"speech_reference.json:{transcript_source}",
        },
        "topic": {
            "kind": "choice",
            "choice": "waterway_transport",
            "ground_truth_source": f"speech_reference.json:{transcript_source}",
            "reference_label_note": "The pinned human FLEURS sample describes the need for yachts or boats among islands and lakes.",
        },
        "mentions_aircraft": {
            "kind": "noul",
            "yes": False,
            "ground_truth_source": f"speech_reference.json:{transcript_source}",
        },
    }
    return TaskSpec(
        task_id="public_audio_decisions",
        category="audio_typed_decisions",
        state=None,
        questions=questions,
        expected=expected,
        media=[] if audio_error is not None else [{"type": "audio", "path": str(audio_path.resolve())}],
        input_files=("speech_reference.wav", "speech_reference.json"),
        preparation_errors=issues,
    )


def spoken_image_question_task(
    scene: Any | None,
    scene_error: dict[str, str] | None,
    voice_path: Path,
    voice_error: dict[str, str] | None,
) -> TaskSpec:
    questions = {
        "blue_digit": {
            "type": "number",
            "instructions": {"type": "audio", "path": str(voice_path.resolve())},
            "range": [0, 10],
            "step": 1,
        }
    }
    return TaskSpec(
        task_id="spoken_question_with_image",
        category="audio_image_typed_decision",
        state=None,
        questions=questions,
        expected={"blue_digit": {"kind": "estimate", "value": 7, "tolerance": 0.5}},
        media=[] if scene is None else [scene],
        input_files=("scene.png", "voice_question.wav"),
        preparation_errors=tuple(error for error in (scene_error, voice_error) if error is not None),
    )


def load_scene(path: Path) -> tuple[Any | None, dict[str, str] | None, dict[str, Any] | None]:
    try:
        if not path.is_file():
            raise FileNotFoundError(f"required image is missing: {path}")
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("scene.png exceeds the 16 MiB input bound")
        from PIL import Image

        with Image.open(path) as image:
            if image.size != (600, 360):
                raise ValueError(f"scene.png must be 600x360 pixels, got {image.size}")
            scene = image.convert("RGB").copy()
        return scene, None, {"width": 600, "height": 360, "mode": "RGB"}
    except Exception as error:
        return None, error_record("input_validation", error), None


def validate_audio(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        if not path.is_file():
            raise FileNotFoundError(f"required audio file is missing: {path}")
        size = path.stat().st_size
        if size > 32 * 1024 * 1024:
            raise ValueError(f"audio input exceeds the 32 MiB bound: {path.name}")
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            duration = frames / sample_rate if sample_rate else 0
            compression = audio.getcomptype()
        if compression != "NONE" or sample_width != 2 or channels not in (1, 2):
            raise ValueError("audio must be uncompressed 16-bit mono or stereo PCM WAV")
        if sample_rate <= 0 or not 0 < duration <= 120:
            raise ValueError(f"audio duration must be in (0, 120] seconds, got {duration}")
        return {
            "bytes": size,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "sample_rate_hz": sample_rate,
            "duration_seconds": duration,
            "format": "PCM_16_WAV",
        }, None
    except Exception as error:
        return None, error_record("input_validation", error)


def load_transcript(path: Path) -> tuple[str | None, str | None, dict[str, str] | None, dict[str, Any] | None]:
    try:
        if not path.is_file():
            raise FileNotFoundError(f"required transcript metadata is missing: {path}")
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("speech_reference.json exceeds the 1 MiB input bound")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("speech_reference.json must contain an object")
        for field in ("text", "transcription", "raw_transcription"):
            value = data.get(field)
            if isinstance(value, str) and value.strip():
                metadata = {
                    key: data.get(key)
                    for key in ("source", "config", "split", "revision", "id", "url", "synthetic")
                    if key in data
                }
                return value, field, None, metadata
        raise ValueError("speech_reference.json has no nonempty text, transcription, or raw_transcription")
    except Exception as error:
        return None, None, error_record("input_validation", error), None


def input_provenance(input_dir: Path, model_key: str) -> dict[str, Any]:
    filenames = ["scene.png"]
    if model_key == "d1_omni":
        filenames.extend(["speech_reference.wav", "speech_reference.json", "voice_question.wav"])
    output: dict[str, Any] = {}
    for filename in filenames:
        path = input_dir / filename
        if path.is_file():
            output[filename] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
        else:
            output[filename] = {"missing": True}
    return output


def make_tasks(
    model_key: str,
    input_dir: Path,
    scene: Any | None,
    scene_error: dict[str, str] | None,
) -> tuple[list[TaskSpec], dict[str, Any]]:
    tasks = [text_task(), image_task(scene, scene_error)]
    audio_meta: dict[str, Any] = {}
    if model_key == "d1_omni":
        reference_path = input_dir / "speech_reference.wav"
        reference_meta, reference_error = validate_audio(reference_path)
        transcript, transcript_field, transcript_error, transcript_metadata = load_transcript(
            input_dir / "speech_reference.json"
        )
        tasks.append(audio_reference_task(reference_path, reference_error, transcript, transcript_error, transcript_field))
        voice_path = input_dir / "voice_question.wav"
        voice_meta, voice_error = validate_audio(voice_path)
        tasks.append(spoken_image_question_task(scene, scene_error, voice_path, voice_error))
        audio_meta = {
            "speech_reference": reference_meta,
            "speech_reference_errors": [error for error in (reference_error, transcript_error) if error is not None],
            "speech_reference_transcript_field": transcript_field,
            "speech_reference_metadata": transcript_metadata,
            "voice_question": voice_meta,
            "voice_question_error": voice_error,
        }
    return tasks, audio_meta


def validate_gpu(torch: Any) -> tuple[Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this runner does not fall back to CPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    name = torch.cuda.get_device_name(device)
    capability = list(torch.cuda.get_device_capability(device))
    if "t4" not in name.casefold():
        raise RuntimeError(f"expected a Tesla T4, found {name!r}")
    if capability != [7, 5]:
        raise RuntimeError(f"expected Tesla T4 compute capability [7, 5], found {capability}")
    return device, {
        "cuda_available": True,
        "device_name": name,
        "device_index": 0,
        "compute_capability": capability,
        "total_vram_bytes": properties.total_memory,
    }


def load_model(torch: Any, AutoModel: Any, model_key: str, device: Any) -> tuple[Any, dict[str, Any]]:
    spec = MODEL_SPECS[model_key]
    model = AutoModel.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        trust_remote_code=True,
        token=False,
        dtype=torch.float16,
        attn_implementation="d1",
    ).to(device)
    model.eval()
    torch.cuda.synchronize(device)
    config_revision = getattr(model.config, "_commit_hash", None)
    loaded_revision = config_revision or getattr(model, "d1_revision", None)
    if loaded_revision != spec["revision"]:
        raise RuntimeError(f"loaded revision {loaded_revision!r} does not match pin {spec['revision']}")
    if getattr(model.config, "_attn_implementation", None) != "d1":
        raise RuntimeError("pinned System One model did not load its required d1 attention implementation")
    parameter_dtypes: dict[str, int] = {}
    parameter_devices: dict[str, int] = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        location = str(parameter.device)
        parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + 1
        parameter_devices[location] = parameter_devices.get(location, 0) + 1
    if set(parameter_dtypes) != {"torch.float16"}:
        raise RuntimeError(f"model parameters are not all FP16: {parameter_dtypes}")
    if set(parameter_devices) != {str(device)}:
        raise RuntimeError(f"model parameters are not all on {device}: {parameter_devices}")
    d1_mode = getattr(model.config, "d1_attention", None)
    if d1_mode != "set":
        raise RuntimeError(f"pinned checkpoint d1_attention must be 'set', got {d1_mode!r}")
    return model, {
        "loaded_revision": loaded_revision,
        "config_attn_implementation": getattr(model.config, "_attn_implementation", None),
        "d1_attention_mode": d1_mode,
        "parameter_dtypes": parameter_dtypes,
        "parameter_devices": parameter_devices,
        "dtype": "float16",
        "device": str(device),
        "attention_execution": {
            "implementation": "d1",
            "d1_mask_mode": d1_mode,
            "model_native_attention_preserved": True,
            "t4_fused_flash_path": False,
            "attention_call": "torch.nn.functional.scaled_dot_product_attention",
            "backend_dispatch": "PyTorch SDPA selects the available backend",
            "literal_transformers_eager": False,
            "reason": "Pinned LFM3 System One overrides attn_implementation to d1 and needs it for state attention; its T4 path uses SDPA.",
        },
        "model_load_completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def call_system_one(
    torch: Any,
    model: Any,
    task: TaskSpec,
    *,
    repeat_index: int | None,
    mode: str,
) -> dict[str, Any]:
    torch.cuda.synchronize(0)
    baseline_allocated = torch.cuda.memory_allocated(0)
    baseline_reserved = torch.cuda.memory_reserved(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        raw = model.system_one(
            task.state,
            task.questions,
            media=task.media,
            score_readout="median",
            set_size=16,
        )
        torch.cuda.synchronize(0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        evaluation = evaluate_output(task, raw)
        status = "passed" if evaluation["all_correct"] else "failed"
        if not evaluation["format_ok"]:
            status = "format_error"
        return {
            "mode": mode,
            "repeat": repeat_index,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "peak_allocated_bytes": peak_allocated,
                "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
                "baseline_reserved_bytes": baseline_reserved,
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
            },
            "raw_result": json_safe(raw),
            "evaluation": evaluation,
        }
    except Exception as error:
        try:
            torch.cuda.synchronize(0)
        except Exception:
            pass
        try:
            failed_peak_allocated = torch.cuda.max_memory_allocated(0)
            failed_peak_reserved = torch.cuda.max_memory_reserved(0)
        except Exception:
            failed_peak_allocated = None
            failed_peak_reserved = None
        return {
            "mode": mode,
            "repeat": repeat_index,
            "status": "infrastructure_error",
            "error": error_record("system_one", error),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": failed_peak_allocated,
                "peak_reserved_bytes": failed_peak_reserved,
            },
        }


def task_record(task: TaskSpec, torch: Any, model: Any, repeats: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "category": task.category,
        "state": task.state,
        "questions": task.questions,
        "expected": task.expected,
        "input_files": list(task.input_files),
    }
    if task.preparation_errors:
        record.update({
            "status": "infrastructure_error",
            "preparation_errors": list(task.preparation_errors),
            "warmup": None,
            "measured_repeats": [],
        })
        return record
    warmup = call_system_one(torch, model, task, repeat_index=None, mode="warmup")
    measured = [
        call_system_one(torch, model, task, repeat_index=index + 1, mode="measured")
        for index in range(repeats)
    ]
    trials = [warmup, *measured]
    if any(trial["status"] == "infrastructure_error" for trial in trials):
        status = "infrastructure_error"
    elif any(trial["status"] == "format_error" for trial in trials):
        status = "format_error"
    elif any(trial["status"] == "failed" for trial in trials):
        status = "failed"
    else:
        status = "passed"
    record.update({"status": status, "warmup": warmup, "measured_repeats": measured})
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/lfm3-full-results"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 10:
        parser.error("--repeats must be between 1 and 10")

    spec = MODEL_SPECS[args.model]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = args.output_dir / args.model / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_dir = args.output_dir / "inputs"
    manifest: dict[str, Any] = {
        "schema_version": "lfm3-system-one-run/v1",
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_key": args.model,
        "model_id": spec["repo_id"],
        "model_sha": spec["revision"],
        "script_sha256": script_digest(),
        "output_dir": str(run_dir.resolve()),
        "inputs_dir": str(input_dir.resolve()),
        "input_hashes": input_provenance(input_dir, args.model),
        "config": {
            "requested_repeats": args.repeats,
            "warmups_per_task": 1,
            "device_required": "Tesla T4 / CUDA:0",
            "dtype_required": "float16",
            "system_one_attention": "d1",
            "d1_attention_mode_required": "set",
            "set_size": 16,
            "score_readout": "median",
            "probability_note": "System One probabilities reflect checkpoint calibration when available; this representative suite does not estimate empirical calibration.",
            "hf_token": False,
            "implicit_hf_token_disabled": True,
            "maximum_image_bytes": 16 * 1024 * 1024,
            "maximum_audio_bytes": 32 * 1024 * 1024,
            "maximum_audio_seconds": 120,
        },
        "environment": None,
        "model_runtime": None,
        "audio_inputs": None,
        "tasks": [],
        "infrastructure_errors": [],
    }
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.json"
    tasks_path = run_dir / "tasks.jsonl"
    write_json(manifest_path, manifest)
    write_json(results_path, {"schema_version": "lfm3-system-one-results/v1", "tasks": []})

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        import numpy
        import torch
        import transformers
        import PIL
        from transformers import AutoModel
    except Exception as error:
        manifest["status"] = "infrastructure_error"
        manifest["infrastructure_errors"].append(error_record("dependency_import", error))
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        return 1

    manifest["environment"] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": numpy.__version__,
        "pillow_version": PIL.__version__,
        "accelerate_version": package_version("accelerate"),
        "torch_cuda_version": torch.version.cuda,
        "hf_implicit_token_disabled": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1",
    }
    try:
        device, gpu_record = validate_gpu(torch)
        manifest["environment"].update(gpu_record)
    except Exception as error:
        manifest["status"] = "infrastructure_error"
        manifest["infrastructure_errors"].append(error_record("gpu_validation", error))
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        return 1

    scene, scene_error, scene_meta = load_scene(input_dir / "scene.png")
    tasks, audio_meta = make_tasks(args.model, input_dir, scene, scene_error)
    manifest["scene_input"] = scene_meta
    manifest["audio_inputs"] = audio_meta or None
    manifest["task_count"] = len(tasks)
    write_json(manifest_path, manifest)

    try:
        model, model_runtime = load_model(torch, AutoModel, args.model, device)
        manifest["model_runtime"] = model_runtime
    except Exception as error:
        manifest["status"] = "infrastructure_error"
        manifest["infrastructure_errors"].append(error_record("model_load", error))
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        write_json(results_path, {
            "schema_version": "lfm3-system-one-results/v1",
            "model_key": args.model,
            "model_id": spec["repo_id"],
            "model_sha": spec["revision"],
            "tasks": [],
            "infrastructure_errors": manifest["infrastructure_errors"],
        })
        return 1

    task_results: list[dict[str, Any]] = []
    format_failures = 0
    for task in tasks:
        result = task_record(task, torch, model, args.repeats)
        task_results.append(result)
        append_jsonl(tasks_path, result)
        if result["status"] == "infrastructure_error":
            errors = result.get("preparation_errors", []) or [
                trial.get("error") for trial in [result.get("warmup"), *result.get("measured_repeats", [])]
                if trial and trial.get("error")
            ]
            manifest["infrastructure_errors"].extend(error for error in errors if error)
        if result["status"] == "format_error":
            format_failures += 1
        manifest["tasks"] = [
            {"task_id": item["task_id"], "category": item["category"], "status": item["status"]}
            for item in task_results
        ]
        write_json(manifest_path, manifest)
        write_json(results_path, {
            "schema_version": "lfm3-system-one-results/v1",
            "model_key": args.model,
            "model_id": spec["repo_id"],
            "model_sha": spec["revision"],
            "script_sha256": manifest["script_sha256"],
            "tasks": task_results,
        })

    task_failures = sum(item["status"] != "passed" for item in task_results)
    manifest["status"] = "completed" if task_failures == 0 else "completed_with_failures"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["task_summary"] = {
        "total": len(task_results),
        "passed": sum(item["status"] == "passed" for item in task_results),
        "failed_or_invalid": task_failures,
        "format_failures": format_failures,
    }
    write_json(manifest_path, manifest)
    write_json(results_path, {
        "schema_version": "lfm3-system-one-results/v1",
        "model_key": args.model,
        "model_id": spec["repo_id"],
        "model_sha": spec["revision"],
        "script_sha256": manifest["script_sha256"],
        "status": manifest["status"],
        "tasks": task_results,
        "infrastructure_errors": manifest["infrastructure_errors"],
    })
    print(json.dumps({
        "run_dir": str(run_dir.resolve()),
        "model_key": args.model,
        "status": manifest["status"],
        "task_summary": manifest["task_summary"],
        "infrastructure_errors": len(manifest["infrastructure_errors"]),
    }, ensure_ascii=False))
    return 1 if manifest["infrastructure_errors"] or format_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
