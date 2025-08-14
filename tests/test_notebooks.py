import pathlib

import pytest

nbformat = pytest.importorskip("nbformat")
nbconvert = pytest.importorskip("nbconvert")
PythonExporter = nbconvert.PythonExporter


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = list(ROOT.glob("*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_compiles(notebook_path):
    """Ensure each notebook converts to valid Python code."""
    nb = nbformat.read(notebook_path, as_version=4)
    source, _ = PythonExporter().from_notebook_node(nb)
    # Compilation should fail if there are syntax errors
    compile(source, str(notebook_path), "exec")

