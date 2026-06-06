.PHONY: test lint audit verify

PYTHON ?= python

test:
	$(PYTHON) scripts/run_tests.py

lint:
	$(PYTHON) -m ruff check .

audit:
	$(PYTHON) -m pip_audit -r requirements.txt --progress-spinner off

verify:
	$(PYTHON) scripts/verify.py
