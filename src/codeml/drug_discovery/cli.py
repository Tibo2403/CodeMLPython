"""Command-line entry point for the drug discovery pipeline."""

from __future__ import annotations

from codeml.drug_discovery.pipeline import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
