"""Run the repository's lightweight unit tests without pytest.

The GitHub workflow can use pytest, but this script keeps a zero-extra-dependency
test path for machines where Python is available but pytest is not installed.
It intentionally skips notebook compilation tests, which require nbformat and
nbconvert.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILES = [
    ROOT / "tests" / "test_datasets.py",
    ROOT / "tests" / "test_placeholder.py",
    ROOT / "tests" / "test_statistics_calculator.py",
    ROOT / "tests" / "test_ai_drug_discovery.py",
]


def load_module(path: Path) -> ModuleType:
    module_name = f"_local_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def iter_test_functions(module: ModuleType):
    for name, value in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_") and not inspect.signature(value).parameters:
            yield name, value


def run_tests(test_files: list[Path]) -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    failures = 0
    total = 0

    for path in test_files:
        module = load_module(path)
        for name, test_function in iter_test_functions(module):
            total += 1
            label = f"{path.relative_to(ROOT)}::{name}"
            try:
                test_function()
            except Exception:
                failures += 1
                print(f"FAIL {label}")
                traceback.print_exc()
            else:
                print(f"PASS {label}")

    print(f"\n{total - failures} passed, {failures} failed")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight repository tests.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional test files to run. Defaults to core unit tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_files = [path.resolve() for path in args.paths] if args.paths else DEFAULT_TEST_FILES
    raise SystemExit(run_tests(test_files))


if __name__ == "__main__":
    main()
