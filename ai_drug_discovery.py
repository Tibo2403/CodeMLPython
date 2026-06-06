"""Compatibility wrapper for the installable drug discovery CLI.

Prefer using:

    codeml-drug-discovery --training-csv examples/molecules.csv --seed-csv examples/seeds.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from codeml.drug_discovery.pipeline import *  # noqa: F403
from codeml.drug_discovery.pipeline import main

if __name__ == "__main__":
    main()
