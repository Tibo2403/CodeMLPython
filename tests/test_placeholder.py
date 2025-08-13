"""Simple tests for repository metadata files."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_requirements_pinned():
    """All requirements should pin exact versions."""
    path = ROOT / "requirements.txt"
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert lines, "requirements.txt is empty"
    assert all("==" in line for line in lines)
