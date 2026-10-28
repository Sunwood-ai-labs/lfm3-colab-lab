from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_installed_wheel_prepares_inputs_from_a_local_cache_fixture(self) -> None:
        """Use a synthetic cached Parquet row so preparation needs no dataset download."""
        uv = shutil.which("uv")
        if uv is None:
            self.fail("the wheel distribution test requires uv")
        with tempfile.TemporaryDirectory(prefix="lfm3-wheel-smoke-") as temporary:
            root = Path(temporary)
            wheels = root / "wheels"
            wheels.mkdir()
            subprocess.run(
                [uv, "build", "--wheel", "--out-dir", str(wheels), "--project", str(ROOT)],
                cwd=root,
                check=True,
            )
            wheel, = wheels.glob("*.whl")

            environment = root / "venv"
            base_python = getattr(sys, "_base_executable", sys.executable)
            subprocess.run([uv, "venv", str(environment), "--python", base_python], cwd=root, check=True)
            interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run(
                [uv, "pip", "install", "--python", str(interpreter), "--no-deps", str(wheel)],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    uv, "pip", "install", "--python", str(interpreter),
                    "numpy", "Pillow", "soundfile", "imageio-ffmpeg", "pyarrow",
                ],
                cwd=root,
                check=True,
            )

            output_dir = root / "results"
            cache_dir = root / "cache"
            cache_dir.mkdir()
            fixture_writer = """
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

cache = Path(sys.argv[1])
audio = io.BytesIO()
sf.write(audio, np.zeros(32000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
table = pa.Table.from_pylist([{
    "id": "wheel-smoke-fixture",
    "audio": {"bytes": audio.getvalue()},
    "transcription": "wheel smoke fixture",
}])
pq.write_table(table, cache / "fleurs-ja-test.parquet")
"""
            subprocess.run(
                [str(interpreter), "-c", fixture_writer, str(cache_dir)],
                cwd=root,
                check=True,
            )
            environment_vars = os.environ.copy()
            environment_vars.pop("PYTHONPATH", None)
            environment_vars.pop("PYTHONHOME", None)
            environment_vars["PYTHONNOUSERSITE"] = "1"
            prepare = subprocess.run(
                [
                    str(interpreter), "-m", "lfm3_lab", "prepare",
                    "--output-dir", str(output_dir),
                    "--cache-dir", str(cache_dir),
                ],
                cwd=root,
                env=environment_vars,
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stdout + prepare.stderr)

            source_manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8"))
            expected_sources = sorted((ROOT / "experiments").glob("*.py"))
            self.assertEqual(
                [item["path"] for item in source_manifest["sources"]],
                [f"experiments/{path.name}" for path in expected_sources],
            )
            for item, source in zip(source_manifest["sources"], expected_sources, strict=True):
                self.assertEqual(item["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(
                [suite["key"] for suite in source_manifest["suites"]],
                ["text", "vision", "audio", "omni", "d1", "d1_omni"],
            )
            inputs = output_dir / "inputs"
            for name in (
                "scene.png", "tone.wav", "sequence_video.mp4", "scene_video.mp4",
                "speech_reference.wav", "speech_reference.json", "manifest.json",
            ):
                self.assertTrue((inputs / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
