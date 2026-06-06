"""Run local quality checks in a portable order."""

from __future__ import annotations

import subprocess
import sys

COMMANDS = [
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "scripts/run_tests.py"],
    [sys.executable, "-m", "pip_audit", "-r", "requirements.txt", "--progress-spinner", "off"],
]


def main() -> None:
    for command in COMMANDS:
        print(f"+ {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
