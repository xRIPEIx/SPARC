.PHONY: install test test-all lint format check clean

install:  ## editable install with dev extras
	pip install -e ".[dev,timm,viz]"

test:  ## fast CPU suite (what CI runs)
	pytest

test-all:  ## include GPU tests
	pytest -m "not slurm and not legacy"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

check: lint test  ## everything CI checks, locally

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
