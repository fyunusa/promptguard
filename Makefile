.PHONY: help install dev test format lint type clean docs

help:
	@echo "PromptGuard Development Commands"
	@echo "================================="
	@echo "make install      Install dependencies"
	@echo "make dev          Install dev dependencies"
	@echo "make test         Run tests"
	@echo "make test-cov     Run tests with coverage"
	@echo "make format       Format code with black"
	@echo "make lint         Lint code with ruff"
	@echo "make type         Type check with mypy"
	@echo "make clean        Clean build artifacts"
	@echo "make docs         Build documentation"
	@echo "make all          Run format, lint, type, test"

install:
	pip install -e .

dev:
	pip install -e ".[all]"
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=promptguard --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated: htmlcov/index.html"

format:
	black promptguard tests

lint:
	ruff check promptguard tests --fix

type:
	mypy promptguard --ignore-missing-imports

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

docs:
	cd docs && make html
	@echo "Documentation built: docs/_build/html/index.html"

all: format lint type test
	@echo "All checks passed!"

.DEFAULT_GOAL := help
