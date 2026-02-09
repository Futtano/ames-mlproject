# Ames Housing Price Predictor

[![CI/CD Pipeline](https://github.com/Futtano/ames-mlproject/actions/workflows/pipeline.yml/badge.svg)](https://github.com/Futtano/ames-mlproject/actions)
[![codecov](https://codecov.io/gh/Futtano/ames-mlproject/graph/badge.svg?token=RK0TKRDCPB)](https://codecov.io/gh/Futtano/ames-mlproject)
[![Python Version](https://img.shields.io/github/python-version/Futtano/ames-mlproject)](https://github.com/Futtano/ames-mlproject/blob/main/pyproject.toml)
[![Linting](https://img.shields.io/badge/lint-Ruff-orange.svg)](https://github.com/astral-sh/ruff)

An end-to-end, production-grade Machine Learning pipeline that predicts house sale prices using the comprehensive Ames Housing dataset. This project serves as a blueprint for professional ML engineering, emphasizing **clean code**, **automated testing**, and **scalable architecture**.

---

## Project Objective

The goal is to provide a robust system that can:
1.  **Ingest & Clean**: Automatically load and validate the Ames dataset, handling outliers and logical inconsistencies.
2.  **Train & Tune**: Perform model selection and hyperparameter optimization via **Nested Cross-Validation**.
3.  **Deploy**: Serve predictions through a high-performance **FastAPI** backend with a modern web interface.
4.  **Monitor**: Maintain reliability through centralized, rotating logs and detailed exception handling.

---

## Key Features

- **Dynamic Model Engine**: Configure model algorithms and hyperparameter search spaces entirely through YAML.
- **Nested Cross-Validation**: Prevents data leakage during model selection for highly reliable $R^2$ scores.
- **Modern Stack**: Powered by `uv` (dependency management), `scikit-learn`, `XGBoost`, and `Pydantic V2`.
- **Professional Logging**: Dual-output (Terminal + File) with **Rotating Handlers** to prevent disk bloat.
- **Industrial Testing**: 92%+ code coverage using `pytest` with extensive mocking of ML artifacts.
- **Containerization**: Optimized multi-stage Docker builds for minimal production image sizes.

---

## Installation & Setup

We recommend using **[uv](https://github.com/astral-sh/uv)** for the fastest dependency resolution.

```bash
# 1. Clone the repository
git clone https://github.com/Futtano/ames-mlproject.git
cd ames-mlproject

# 2. Setup environment and install dependencies
make setup
```

---

## Usage Guide

### 1. Training the Model
The training pipeline orchestrates data ingestion, cleaning, and model selection. It saves the best model to `artifacts/model.pkl`.
```bash
# Set PROJECT_ROOT and run the pipeline
make train
```

### 2. Running the API
The API provides a Swagger documentation interface and a built-in web dashboard.
```bash
# Start locally
python src/ames_mlproject/api/main.py

# Or run via Docker Compose
docker compose up -d
```
*   **Web Interface**: [http://localhost:5000](http://localhost:5000)
*   **Swagger Docs**: [http://localhost:5000/docs](http://localhost:5000/docs)

### 3. Developer Quality Gates
Ensure code integrity before pushing:
```bash
make lint   # Run Ruff, Black, and Mypy
make test   # Run all unit and integration tests
```

---

## System Architecture

*   `src/ames_mlproject/api/`: FastAPI server and Request/Response schemas.
*   `src/ames_mlproject/data/`: Ingestion and Preprocessing logic.
*   `src/ames_mlproject/models/`: Model training engine and hyperparameter tuning.
*   `src/ames_mlproject/pipelines/`: High-level orchestrators for Train/Predict flows.
*   `src/ames_mlproject/core/`: Application-wide logging and exception handling.

---

## Author

**Daniele Loru**
- Email: [futhanos@gmail.com](mailto:futhanos@gmail.com)
