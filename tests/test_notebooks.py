import pathlib
import subprocess

import nbformat
import pytest
from nbconvert import PythonExporter


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / path
    for path in subprocess.check_output(
        ["git", "ls-files", "*.ipynb"], cwd=ROOT, text=True
    ).splitlines()
]


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_compiles(notebook_path):
    """Ensure each notebook converts to valid Python code."""
    nb = nbformat.read(notebook_path, as_version=4)
    source, _ = PythonExporter().from_notebook_node(nb)
    # Compilation should fail if there are syntax errors
    compile(source, str(notebook_path), "exec")

