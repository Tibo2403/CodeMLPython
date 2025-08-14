import csv
import math
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def is_valid_number(value):
    try:
        return not math.isnan(float(value))
    except ValueError:
        return False


DATASETS = [
    ("diabetes.csv", 768, 9, True),
    ("Ch3.ClevelandData.csv", 303, 14, False),
]


@pytest.mark.parametrize("filename, rows_expected, cols_expected, has_header", DATASETS)
def test_dataset_shape(filename, rows_expected, cols_expected, has_header):
    rows = load_csv(ROOT / filename)
    data = rows[1:] if has_header else rows
    if has_header:
        assert len(rows[0]) == cols_expected
    assert len(data) == rows_expected
    assert all(len(r) == cols_expected for r in data)


@pytest.mark.parametrize("filename, has_header", [(d[0], d[3]) for d in DATASETS])
def test_dataset_values(filename, has_header):
    rows = load_csv(ROOT / filename)
    data = rows[1:] if has_header else rows
    assert all(all(is_valid_number(c) and c.strip() for c in row) for row in data)
    assert len(data) == len({tuple(r) for r in data})

