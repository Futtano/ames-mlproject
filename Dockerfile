# Use a lightweight Python base image matching the project's requirement
FROM python:3.11-slim

# Set environment variables for Python and application behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROJECT_ENV=production

# Install essential build tools (needed for some ML library native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for extremely fast and reliable dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy only the dependency definition files first to leverage Docker layer caching
COPY pyproject.toml setup.py requirements.txt ./

# Install project dependencies into the system python environment
# using the requirements.txt file to avoid 'src' layout issues during metadata generation
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application source code
# (.dockerignore handles excluding unnecessary files)
COPY . .

# Install the project itself in non-editable mode
RUN uv pip install --system --no-cache --no-deps .

# Ensure necessary directories exist for runtime operations
RUN mkdir -p artifacts .log

# Expose the port defined in config.yaml or env (default 5000)
# Note: EXPOSE is documentary, actual binding happens in CMD
EXPOSE 5000

# Metadata labels
LABEL maintainer="Daniele Loru <futhanos@gmail.com>"
LABEL version="0.1.0"
LABEL description="Ames House Price Prediction API"

# Use uvicorn to serve the FastAPI application
# We use shell form to allow variable expansion for PORT (default 5000)
CMD uvicorn ames_mlproject.api.main:app --host 0.0.0.0 --port ${PORT:-5000}
