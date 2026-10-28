import argparse
import hashlib
import io
import json
import subprocess
import urllib.request
from pathlib import Path

DATA_REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
DATA_URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{DATA_REVISION}/ja_jp/test/0000.parquet"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    import imageio_ffmpeg
    import pyarrow.parquet as pq
    import soundfile as sf
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/content/lfm3-full-results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/lfm3-full-cache"))
    args = parser.parse_args()
    inputs = args.output_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    scene = Image.new("RGB", (600, 360), "white")
    draw = ImageDraw.Draw(scene)
    font = ImageFont.load_default(size=80)
    draw.rectangle((40, 80, 240, 280), fill=(220, 25, 25))
    draw.rectangle((340, 80, 540, 280), fill=(25, 70, 220))
    draw.text((108, 130), "3", font=font, fill="white")
    draw.text((408, 130), "7", font=font, fill="white")
    scene.save(inputs / "scene.png")
    tone_time = np.arange(32000, dtype=np.float32) / 16000
    tone = 0.15 * np.sin(2 * np.pi * 880 * tone_time)
    tone *= ((tone_time % 0.5) < 0.25)
    sf.write(inputs / "tone.wav", tone, 16000, subtype="PCM_16")
    frames = inputs / "video_frames"
    frames.mkdir(exist_ok=True)
    for index in range(12):
        frame = Image.new("RGB", (600, 360), "white")
        frame_draw = ImageDraw.Draw(frame)
        frame_draw.rectangle((190, 80, 390, 280), fill=(220, 25, 25) if index < 6 else (25, 70, 220))
        frame.save(frames / f"{index:03d}.png")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4",
        "-i", str(frames / "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(inputs / "sequence_video.mp4"),
    ], check=True)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
        "-i", str(inputs / "scene.png"), "-t", "3", "-r", "4", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(inputs / "scene_video.mp4"),
    ], check=True)
    parquet_path = args.cache_dir / "fleurs-ja-test.parquet"
    if not parquet_path.exists():
        print("Downloading pinned public Japanese FLEURS audio dataset", flush=True)
        urllib.request.urlretrieve(DATA_URL, parquet_path)
    table = pq.ParquetFile(parquet_path)
    names = table.schema_arrow.names
    print("FLEURS schema", names, flush=True)
    columns = [name for name in ("id", "audio", "transcription", "raw_transcription", "language") if name in names]
    selected = None
    inspected = 0
    for batch in table.iter_batches(batch_size=16, columns=columns):
        for row in batch.to_pylist():
            inspected += 1
            audio = row["audio"]
            if not isinstance(audio, dict) or not audio.get("bytes"):
                continue
            with sf.SoundFile(io.BytesIO(audio["bytes"])) as handle:
                duration = len(handle) / handle.samplerate
                if not 1.5 <= duration <= 7.0:
                    continue
                waveform = handle.read(dtype="float32")
                rate = handle.samplerate
            text = row.get("transcription") or row.get("raw_transcription")
            if not text:
                continue
            sf.write(inputs / "speech_reference.wav", waveform, rate, subtype="PCM_16")
            selected = {
                "text": text, "transcription": text,
                "raw_transcription": row.get("raw_transcription"),
                "id": row.get("id"), "sample_rate": rate, "duration_seconds": duration,
                "source": "google/fleurs", "config": "ja_jp", "split": "test",
                "revision": DATA_REVISION, "url": DATA_URL,
                "synthetic": False, "parquet_sha256": digest(parquet_path),
                "license": "CC-BY-4.0", "attribution": "FLEURS, Google and contributors",
                "dataset_url": "https://huggingface.co/datasets/google/fleurs",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "waveform_sha256": digest(inputs / "speech_reference.wav"),
            }
            break
        if selected or inspected >= 128:
            break
    if not selected:
        raise RuntimeError("No short public Japanese reference audio found; no hidden synthetic fallback")
    (inputs / "speech_reference.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "scene": {
            "size": [600, 360],
            "objects": [
                {"color": "red", "digit": 3, "box_xyxy": [40, 80, 240, 280]},
                {"color": "blue", "digit": 7, "box_xyxy": [340, 80, 540, 280]},
            ],
        },
        "sequence_video": {"fps": 4, "duration_seconds": 3, "first_color": "red", "last_color": "blue"},
        "speech_reference": selected,
        "files": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                  for path in sorted(inputs.iterdir()) if path.is_file()},
    }
    (inputs / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "output": str(inputs), "human_audio_text": selected["text"],
                      "human_audio_seconds": selected["duration_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
