from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .models import RunResult


_TASK_LABELS = {
    "text_tool_add_numbers_no_think": "テキスト加算ツール",
    "text_japanese_rephrase_no_think": "日本語の言い換え",
    "vision_video_temporal_no_think": "動画の順序理解",
    "vision_video_temporal_thinking": "動画の順序理解（thinking）",
    "vision_image_addition_tool_no_think": "画像の加算ツール",
    "tts-ja-blue-question": "日本語TTS",
    "tts-en-short-phrase": "英語TTS",
    "asr-ja-human-reference": "日本語ASR",
    "asr-en-synthetic-reference": "英語ASR",
    "translation-ja-en-human-reference": "日英翻訳",
    "audio-understanding-tone": "音声の内容理解",
    "voice-chat-math": "日本語音声対話",
    "voice-chat-math-en": "英語音声対話",
    "voice-tool-add": "日本語音声ツール",
    "voice-tool-add-en": "英語音声ツール",
    "image-audio-blue-digit": "青の数字（画像と音声）",
    "image-audio-red-digit": "赤の数字（画像と音声）",
    "video-audio-blue-digit": "動画と音声の質問",
    "text_typed_decisions": "テキストの型付き判断",
    "image_boxes_and_count": "画像の位置・個数",
    "public_audio_decisions": "音声の型付き判断",
    "spoken_question_with_image": "音声質問と画像",
}


def _display(value: Any) -> bool:
    try:
        from IPython.display import display
    except ImportError:
        return False
    display(value)
    return True


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<table style="border-collapse:collapse"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _fmt(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "未計測"
    return f"{value:.2f}{suffix}"


def show_result(result: RunResult) -> None:
    status = _status_label(result)
    incorrect = result.functional_failures + result.format_failures
    ungraded = str(result.ungraded_count) if result.ungraded_count else "なし"
    parts = [
        f"<h3>{html.escape(result.label)}の実験結果</h3>",
        _table(
            ("結果", "通過/試行", "誤り/形式", "未採点"),
            [(status, f"{result.passed}/{result.attempted}", str(incorrect), ungraded)],
        ),
    ]
    representative = next(
        (metric for metric in result.task_metrics if metric.median_seconds is not None),
        result.task_metrics[0] if result.task_metrics else None,
    )
    if representative is not None:
        parts.extend(("<h4>代表計測</h4>", _table(
            ("項目", "計測値"),
            [
                ("タスク", _task_label(representative.task_id)),
                (
                    "中央値・標本数",
                    f"{_fmt(representative.median_seconds, ' 秒')}、{representative.sample_count or '不明'}標本",
                ),
                (
                    "最大VRAM",
                    f"allocated {_fmt(representative.peak_allocated_gib, ' GiB')}、"
                    f"reserved {_fmt(representative.peak_reserved_gib, ' GiB')}",
                ),
            ],
        )))
        parts.append("<p>代表タスクの時間とVRAMを表示しています。回答の正誤とは別の計測値です。</p>")
    else:
        parts.append("<p>表示できる実行計測はありません。</p>")
    if result.examples:
        parts.append("<h4>モデルの回答例</h4>")
        for example in result.examples[:2]:
            label = {"passed": "正解した回答例", "failed": "誤答例", "ungraded": "未採点の出力"}.get(example.outcome, example.outcome)
            parts.append(
                f"<p><strong>{html.escape(label)}: {html.escape(_task_label(example.task_id))}</strong></p>"
                f"<pre style=\"white-space:pre-wrap\">{html.escape(example.answer[:1200])}</pre>"
            )
    else:
        parts.append("<p>表示できる採点対象の回答はありません。</p>")
    detailed_rows = []
    for metric in result.task_metrics:
        detailed_rows.append((
            metric.task_id,
            metric.stage,
            metric.sample_count if metric.sample_count is not None else "未計測",
            _fmt(metric.median_seconds, " 秒"),
            _fmt(metric.median_tokens_per_second, " token/秒"),
            _fmt(metric.median_rtf),
            _fmt(metric.peak_allocated_gib, " GiB"),
            _fmt(metric.peak_reserved_gib, " GiB"),
        ))
    details = ["<details><summary>詳しい計測値・実行ログ</summary>"]
    if detailed_rows:
        details.append(_table(
            ("task_id", "stage", "標本数", "中央値", "生成速度", "RTF", "最大allocated", "最大reserved"),
            detailed_rows,
        ))
    if result.examples:
        details.append("<p>回答例の元タスクID: ")
        details.append("、".join(
            f"<code>{html.escape(example.task_id)}</code> ({html.escape(example.outcome)})"
            for example in result.examples[:2]
        ))
        details.append("</p>")
    details.append(
        f"<p>結果ファイル: <code>{html.escape(str(result.summary_path))}</code><br>"
        f"標準出力: <code>{html.escape(str(result.stdout_path))}</code><br>"
        f"標準エラー: <code>{html.escape(str(result.stderr_path))}</code></p></details>"
    )
    _display_html("".join(parts))
    for path in result.media_paths[:1]:
        _display_media(path, _audio_note(path) if path.suffix.casefold() in {".wav", ".mp3"} else None)
    _display_html("".join(details))


def _status_label(result: RunResult) -> str:
    if result.infrastructure_errors:
        return "基盤エラー"
    if result.functional_failures or result.format_failures:
        return "回答に誤りあり"
    status = result.status.casefold()
    return {
        "completed": "完了",
        "complete": "完了",
        "completed_with_failures": "回答に誤りあり",
        "complete_with_functional_failures": "回答に誤りあり",
        "completed_with_errors": "基盤エラー",
        "complete_with_errors": "基盤エラー",
        "not_run": "未実行",
        "not_recorded": "記録なし",
        "stale_evidence": "条件外の記録",
    }.get(status, "実行結果を確認")


def _task_label(task_id: str) -> str:
    return _TASK_LABELS.get(task_id, task_id)


def show_inputs(inputs_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    rows = [
        (name, facts.get("sha256", "未計測")[:12], facts.get("bytes", "未計測"))
        for name, facts in files.items()
        if isinstance(facts, dict)
    ][:12]
    reference_path = inputs_dir / "speech_reference.json"
    reference = {}
    if reference_path.is_file():
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reference = {}
    transcript = reference.get("text") or reference.get("transcription") or reference.get("raw_transcription")
    parts = ["<h3>固定入力</h3>"]
    if rows:
        parts.append(_table(("入力", "SHA-256先頭12桁", "バイト"), rows))
    if transcript:
        parts.append(f"<p>公開参照音声の書き起こし: {html.escape(str(transcript)[:400])}</p>")
    _display_html("".join(parts))
    for name in ("scene.png", "speech_reference.wav", "sequence_video.mp4"):
        path = inputs_dir / name
        if path.is_file():
            note = _audio_note(path) if path.suffix.casefold() == ".wav" else None
            _display_media(path, note)


def _audio_note(path: Path) -> str:
    if path.name.startswith("tts-"):
        return "TTS音声の波形とファイル形式を表示しています。発話内容は未採点です。"
    if path.name == "speech_reference.wav":
        return "公開参照音声を再生しています。ASRや音声理解の採点結果は実験表に示します。"
    if path.name == "voice_question.wav":
        return "合成質問音声の波形と再生です。音声から答えを理解できたかは実験表で確認します。"
    return "音声の波形とファイル形式を表示しています。波形確認だけでは意味内容を評価しません。"


def _display_html(value: str) -> None:
    try:
        from IPython.display import HTML
    except ImportError:
        print(value)
        return
    _display(HTML(value))


def _display_media(path: Path, audio_note: str | None = None) -> None:
    try:
        from IPython.display import Audio, Image, Markdown, SVG, Video
    except ImportError:
        print(f"Media: {path}")
        if audio_note:
            print(audio_note)
        return
    suffix = path.suffix.casefold()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        _display(Markdown(f"**画像** `{html.escape(path.name)}`"))
        _display(Image(filename=str(path)))
    elif suffix in {".mp4"}:
        _display(Markdown(f"**動画** `{html.escape(path.name)}`"))
        _display(Video(filename=str(path), embed=True))
    elif suffix in {".wav", ".mp3"}:
        _display(Markdown(f"**音声** `{html.escape(path.name)}`"))
        if audio_note:
            _display(Markdown(html.escape(audio_note)))
        waveform = _waveform_svg(path)
        if waveform:
            _display(SVG(waveform))
        _display(Audio(filename=str(path)))


def _waveform_svg(path: Path, *, width: int = 640, height: int = 150) -> str | None:
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(str(path))
        frame_limit = min(info.frames, info.samplerate * 20)
        signal, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True, frames=frame_limit,
        )
        if signal.size == 0 or sample_rate <= 0:
            return None
        mono = signal.mean(axis=1)
        chunks = np.array_split(mono, min(width, len(mono)))
        lows = [float(np.min(chunk)) for chunk in chunks if len(chunk)]
        highs = [float(np.max(chunk)) for chunk in chunks if len(chunk)]
        if not lows or not highs:
            return None
        center = height / 2
        amplitude = height * 0.42
        points = []
        for index, (low, high) in enumerate(zip(lows, highs)):
            x = index if len(lows) > 1 else width / 2
            top = center - max(-1.0, min(1.0, high)) * amplitude
            bottom = center - max(-1.0, min(1.0, low)) * amplitude
            points.append(f"M{x:.1f},{top:.1f} L{x:.1f},{bottom:.1f}")
        duration = len(mono) / sample_rate
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="Audio waveform, {duration:.2f} seconds">'
            '<rect width="100%" height="100%" fill="#f7fafc"/>'
            f'<line x1="0" y1="{center:.1f}" x2="{width}" y2="{center:.1f}" stroke="#9ca3af" stroke-width="1"/>'
            f'<path d="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="1"/>'
            f'<text x="8" y="16" font-size="12" fill="#374151">{duration:.2f} s · {sample_rate} Hz</text>'
            '</svg>'
        )
    except (OSError, RuntimeError, ValueError, ImportError):
        return None
