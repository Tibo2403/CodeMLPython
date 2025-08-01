import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        return list(reader)

def test_diabetes_dataset_shape():
    rows = load_csv(ROOT / 'diabetes.csv')
    header, data = rows[0], rows[1:]
    assert len(header) == 9
    assert len(data) == 768  # number of data rows
    assert all(len(r) == 9 for r in data)

def test_cleveland_dataset_shape():
    rows = load_csv(ROOT / 'Ch3.ClevelandData.csv')
    assert len(rows) == 303
    assert all(len(r) == 14 for r in rows)
