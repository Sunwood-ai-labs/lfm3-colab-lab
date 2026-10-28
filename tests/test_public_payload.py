from __future__ import annotations

import importlib.util
from pathlib import Path


def checker():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_public_payload.py"
    spec = importlib.util.spec_from_file_location("payload_checker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rejects_credentials_without_echoing_the_value(tmp_path):
    secret = "gh" + "p_" + "x" * 36
    (tmp_path / "sample.py").write_text(secret, encoding="utf-8")
    errors = checker().inspect(tmp_path, ["sample.py"])
    assert errors and secret not in " ".join(errors)


def test_rejects_auth_state_and_model_archives(tmp_path):
    names = ["colab-session-local.json", "model.safetensors", "results.zip"]
    for name in names:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert len(checker().inspect(tmp_path, names)) == len(names)


def test_accepts_colab_paths_and_public_model_pins(tmp_path):
    (tmp_path / "example.py").write_text(
        "OUTPUT = '/content/lfm3-results'\nREVISION = '" + "a" * 40 + "'\n",
        encoding="utf-8",
    )
    assert checker().inspect(tmp_path, ["example.py"]) == []
