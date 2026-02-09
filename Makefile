# Makefile for Ames ML Project

.PHONY: setup train test lint docker-build docker-run-all clean

# Dependency management
setup:
	uv venv
	uv pip install -e .[dev]

# Pipeline execution
train:
	python src/ames_mlproject/pipelines/train.py

# Quality checks
test:
	pytest tests/ -v --tb=short

lint:
	pre-commit run --all-files

# Deployment (Docker)
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov .log/
