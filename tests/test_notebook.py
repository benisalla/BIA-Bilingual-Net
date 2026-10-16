"""The training notebook must embed the current package code and be valid Python."""

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_colab_kaggle.ipynb"


def code_cells():
    return [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"]


def test_notebook_embeds_the_current_package_code():
    """Fails when translation_engine changed but the notebook was not rebuilt
    (python notebooks/build_notebook.py)."""
    written = {}
    for source in code_cells():
        if source.startswith("%%writefile "):
            header, _, body = source.partition("\n")
            written[header.split(maxsplit=1)[1]] = body
    assert written, "no %%writefile cells"
    for path, body in written.items():
        assert body.strip() == (ROOT / path).read_text().strip(), f"{path} is out of date in the notebook"


def test_notebook_code_is_valid_python():
    for source in code_cells():
        if not source.startswith("%%"):
            ast.parse(source)


def test_notebook_defaults_are_for_real_training():
    settings = next(s for s in code_cells() if "SMOKE_TEST =" in s)
    assert "SMOKE_TEST = False" in settings
