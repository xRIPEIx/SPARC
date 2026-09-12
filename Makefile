.PHONY: install test test-all lint format check clean paper paper-check

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

AGG := experiments/results/aggregate

paper:  ## regenerate README results block, figures and paper tables from the aggregates
	sparc-report check  --aggregates $(AGG)
	sparc-report readme --aggregates $(AGG) --readme README.md
	sparc-report figure --aggregates $(AGG) --task segmentation --metric mIoU -o docs/figures/lambda_mIoU.png
	sparc-report figure --aggregates $(AGG) --task detection    --metric AP   -o docs/figures/lambda_AP.png
	sparc-report figure --aggregates $(AGG) --task segmentation --metric mIoU --format pgfplots -o docs/paper/lambda_mIoU.tex
	sparc-report figure --aggregates $(AGG) --task detection    --metric AP   --format pgfplots -o docs/paper/lambda_AP.tex
	sparc-report table  --aggregates $(AGG) --format latex > docs/paper/baselines_table.tex

paper-check:  ## fail if the README results block is stale (what CI runs)
	sparc-report check  --aggregates $(AGG)
	sparc-report readme --aggregates $(AGG) --readme README.md --check
