from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "LFM3_Colab_T4.ipynb"
PENDING_CODE_REF = "PENDING_PUBLIC_COMMIT_SHA"
CODE_REF_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
MODEL_KEYS = ("text", "vision", "audio", "omni", "d1", "d1_omni")
MAX_CODE_CELL_CHARS = 3000
MAX_CODE_CELL_LINES = 30
MAX_NOTEBOOK_CODE_CHARS = 10000
LOCAL_PATH_PATTERN = re.compile(r"(?i)(?:[A-Z]:\\(?:Prj|Users)\\|/(?:home|Users)/[A-Za-z0-9_.-]+/)")


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def validate_notebook(path: Path, allow_pending_code_ref: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read notebook JSON: {exc}"]

    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    cells = notebook.get("cells", [])
    if not cells:
        errors.append("notebook has no cells")
    cell_ids = [cell.get("id") for cell in cells]
    if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(set(cell_ids)):
        errors.append("cell ids must be present and unique")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    code_sources = [source_text(cell) for cell in code_cells]
    total_code_chars = sum(len(source) for source in code_sources)
    if total_code_chars > MAX_NOTEBOOK_CODE_CHARS:
        errors.append(f"notebook code totals {total_code_chars} characters")

    for index, (cell, source) in enumerate(zip(code_cells, code_sources), start=1):
        if len(source) > MAX_CODE_CELL_CHARS:
            errors.append(f"code cell {index} is {len(source)} characters")
        if len(source.splitlines()) > MAX_CODE_CELL_LINES:
            errors.append(f"code cell {index} has {len(source.splitlines())} lines")
        try:
            ast.parse(source, filename=f"code-cell-{index}")
        except SyntaxError as exc:
            errors.append(f"code cell {index} has invalid Python: {exc}")
        if cell.get("execution_count") is not None:
            errors.append(f"code cell {index} has an execution count")
        if cell.get("outputs") != []:
            errors.append(f"code cell {index} contains stored output")
        if LOCAL_PATH_PATTERN.search(source):
            errors.append(f"code cell {index} contains a developer-local path")

    all_source = "\n".join(source_text(cell) for cell in cells)
    if LOCAL_PATH_PATTERN.search(all_source):
        errors.append("notebook contains a developer-local absolute path")

    refs = []
    for source in code_sources:
        refs.extend(
            re.findall(
                r"""^CODE_REF\s*=\s*["']([^"']+)["']""",
                source,
                re.MULTILINE,
            )
        )
    if len(refs) != 1:
        errors.append(f"expected one CODE_REF assignment, found {len(refs)}")
    elif refs[0] == PENDING_CODE_REF:
        if not allow_pending_code_ref:
            errors.append("CODE_REF is still a placeholder")
    elif not CODE_REF_PATTERN.fullmatch(refs[0]):
        errors.append("CODE_REF must be a 40-character Git commit SHA")

    for key in MODEL_KEYS:
        matches = [
            source for source in code_sources
            if re.search(
                rf"""lab\.run\(["']{re.escape(key)}["']\s*,\s*repeats=REPEATS\)""",
                source,
            )
        ]
        if len(matches) != 1:
            errors.append(f"expected exactly one named run cell for {key}, found {len(matches)}")

    playground = [source for source in code_sources if "lab.run_text(PROMPT, repeats=1)" in source]
    if len(playground) != 1 or 'print(f"回答: {answer.answer}")' not in playground[0]:
        errors.append("editable playground must run PROMPT and print the model answer")

    if "2026-10-28" not in all_source:
        errors.append("historical results must carry their recording date")
    if "allocated / reserved" not in all_source:
        errors.append("VRAM output must label allocated / reserved")
    if "すべてのセルを実行" not in all_source:
        errors.append("quickstart must tell the reader how to run all cells")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the notebook structure, pin, and bounded outputs."
    )
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--allow-pending-code-ref",
        action="store_true",
        help="Allow the builder's local placeholder while the public source SHA is pending.",
    )
    args = parser.parse_args()
    errors = validate_notebook(args.path, args.allow_pending_code_ref)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
