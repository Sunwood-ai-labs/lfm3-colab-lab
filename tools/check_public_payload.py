"""Check the files selected for public distribution without printing secrets."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath


def inspect(root: Path, names: list[str]) -> list[str]:
    errors = []
    secrets = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|Bearer\s+[A-Za-z0-9._-]{30,}|AIza[A-Za-z0-9_-]{30,})")
    local_paths = re.compile(r"(?:[A-Z]:[/\\](?:Users|Prj)[/\\]|/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/(?:Users|Prj)/)")
    forbidden_names = {"credentials.json", "application_default_credentials.json", "token.json", ".env"}
    forbidden_suffixes = {".safetensors", ".pt", ".pth", ".ckpt", ".zip", ".parquet"}
    for name in names:
        relative = PurePosixPath(name)
        path = root / name
        if not path.is_file():
            continue
        if relative.name in forbidden_names or relative.name.startswith("colab-session") or relative.suffix in forbidden_suffixes:
            errors.append(f"File is not intended for public source distribution: {name}")
        if path.stat().st_size > 2 * 1024 * 1024:
            errors.append(f"File exceeds the public-source size budget: {name}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if secrets.search(content):
            errors.append(f"Possible credential in {name}; matching value omitted")
        if local_paths.search(content):
            errors.append(f"Machine-specific local path in {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = ["git", "ls-files", "-z"]
    if args.staged:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    names = subprocess.check_output(command, cwd=root).decode("utf-8").split("\0")
    names = [name for name in names if name]
    if not names:
        raise SystemExit("No files selected; this is not a completed payload check")
    errors = inspect(root, names)
    for error in errors:
        print(error)
    print(f"Checked {len(names)} public files; {len(errors)} problems")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
