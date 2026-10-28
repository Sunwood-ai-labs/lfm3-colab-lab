from __future__ import annotations

import json
import ast
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.build_notebook import MODEL_SECTIONS, build_notebook
from tools.check_notebook import (
    MAX_CODE_CELL_CHARS,
    MAX_CODE_CELL_LINES,
    MAX_NOTEBOOK_CODE_CHARS,
    source_text,
    validate_notebook,
)


class NotebookBuilderTests(unittest.TestCase):
    def test_install_cell_imports_checked_out_source_in_the_running_kernel(self) -> None:
        notebook = build_notebook("a1" * 20)
        install_cell = next(source_text(cell) for cell in notebook["cells"]
                            if cell["cell_type"] == "code" and '"--no-deps"' in source_text(cell))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src" / "lfm3_lab"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("MARKER = 'checked-out-code'\n", encoding="utf-8")
            probe = """import sys, types
from pathlib import Path
stale = types.ModuleType('lfm3_lab')
stale.MARKER = 'stale-kernel-module'
sys.modules['lfm3_lab'] = stale
before = ('test-torch', 'test-cuda', {})
context = {'sys': sys, 'REPO_ROOT': Path(sys.argv[2]), 'requirement': Path('unused'),
           'pip_step': lambda args: None, 'before': before, 'runtime_snapshot': lambda: before}
exec(sys.argv[1], context)
import lfm3_lab
assert lfm3_lab.MARKER == 'checked-out-code'
assert Path(lfm3_lab.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
"""
            completed = subprocess.run([sys.executable, "-c", probe, install_cell, str(root)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_builder_pins_the_requested_public_commit(self) -> None:
        commit = "a1" * 20
        notebook = build_notebook(commit)
        code = "\n".join(
            source_text(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn(f'CODE_REF = "{commit}"', code)

    def test_builder_rejects_a_non_commit_pin(self) -> None:
        with self.assertRaises(ValueError):
            build_notebook("main")

    def test_each_model_has_its_own_run_cell(self) -> None:
        notebook = build_notebook("b2" * 20)
        code_cells = [
            source_text(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for model in MODEL_SECTIONS:
            matches = [
                source for source in code_cells
                if f'lab.run("{model["key"]}", repeats=REPEATS)' in source
            ]
            self.assertEqual(len(matches), 1, model["key"])

    def test_playground_executes_and_prints_the_editable_answer(self) -> None:
        notebook = build_notebook("c3" * 20)
        code_cells = [
            source_text(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        playgrounds = [source for source in code_cells if "lab.run_text(PROMPT, repeats=1)" in source]
        self.assertEqual(len(playgrounds), 1)
        self.assertIn("PROMPT =", playgrounds[0])
        self.assertIn('print(f"回答: {answer.answer}")', playgrounds[0])

    def test_cells_are_unexecuted_and_code_stays_bounded(self) -> None:
        notebook = build_notebook("d4" * 20)
        code_cells = [
            cell for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        code_sources = [source_text(cell) for cell in code_cells]
        self.assertLessEqual(sum(map(len, code_sources)), MAX_NOTEBOOK_CODE_CHARS)
        self.assertEqual(
            len({cell["id"] for cell in notebook["cells"]}),
            len(notebook["cells"]),
        )
        for cell, source in zip(code_cells, code_sources):
            self.assertLessEqual(len(source), MAX_CODE_CELL_CHARS)
            self.assertLessEqual(len(source.splitlines()), MAX_CODE_CELL_LINES)
            ast.parse(source)
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_audio_suite_precedes_the_d1_omni_fixture_consumer(self) -> None:
        notebook = build_notebook("f6" * 20)
        code_sources = [
            source_text(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        audio_index = next(i for i, source in enumerate(code_sources) if 'lab.run("audio"' in source)
        d1_omni_index = next(i for i, source in enumerate(code_sources) if 'lab.run("d1_omni"' in source)
        self.assertLess(audio_index, d1_omni_index)

    def test_checker_rejects_the_local_placeholder_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "notebook.ipynb"
            path.write_text(
                json.dumps(build_notebook()),
                encoding="utf-8",
            )
            errors = validate_notebook(path)
        self.assertIn("CODE_REF is still a placeholder", errors)

    def test_checker_accepts_a_complete_generated_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "notebook.ipynb"
            path.write_text(
                json.dumps(build_notebook("e5" * 20)),
                encoding="utf-8",
            )
            self.assertEqual(validate_notebook(path), [])


if __name__ == "__main__":
    unittest.main()
