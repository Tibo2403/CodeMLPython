import os
import shutil
import sys
import types
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

# Create stub for google.colab so notebooks expecting it don't fail
class _DummyDrive:
    def mount(self, path):
        os.makedirs(path, exist_ok=True)

_dummy_colab = types.SimpleNamespace(drive=_DummyDrive())
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules["google.colab"] = _dummy_colab

NOTEBOOK_TIMEOUT = 600  # seconds

# Gather all notebooks in repository root
NOTEBOOKS = [os.path.basename(nb) for nb in glob(os.path.join(os.path.dirname(__file__), "..", "*.ipynb"))]

@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_execution(notebook, tmp_path):
    # Ensure directory used in some notebooks exists with required data
    colab_dir = "/content/drive/My Drive/Colab Notebooks"
    os.makedirs(colab_dir, exist_ok=True)
    for data_file in ["diabetes.csv", "Ch3.ClevelandData.xlsx"]:
        src = os.path.join(os.path.dirname(__file__), "..", data_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(colab_dir, data_file))

    nb_path = os.path.join(os.path.dirname(__file__), "..", notebook)
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=NOTEBOOK_TIMEOUT, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nb_path) or "."}})
