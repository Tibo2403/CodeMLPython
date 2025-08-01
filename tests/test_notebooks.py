import nbformat
from nbconvert import PythonExporter


def test_breast_cancer_notebook_compiles():
    """Ensure Breast_cancer_projects.ipynb converts to valid Python code."""
    nb = nbformat.read('Breast_cancer_projects.ipynb', as_version=4)
    source, _ = PythonExporter().from_notebook_node(nb)
    # Compilation should fail if there are syntax errors
    compile(source, 'Breast_cancer_projects.ipynb', 'exec')
