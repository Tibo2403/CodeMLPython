"""Run lightweight tests without requiring pytest."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [ROOT / "tests" / "test_security_optimizer.py"]


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / "BayesCore" / "src"))
    sys.path.insert(0, str(ROOT / "src"))
    failures = 0
    total = 0
    for test_file in TESTS:
        spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {test_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                value()
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
            else:
                print(f"PASS {name}")
    print(f"\n{total - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
