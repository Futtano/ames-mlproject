"""
Configuration management for the Ames ML project.
Loads and validates configuration from YAML files using Pydantic.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Get project root directory
# 1. Check if PROJECT_ROOT is set in environment
# 2. Else use relative path from this file (dev mode)
# 3. Else fallback to current directory
env_root = os.getenv("PROJECT_ROOT")
if env_root:
    PROJECT_ROOT = Path(env_root)
else:
    # In dev, this is src/ames_mlproject/config.py, so parent.parent.parent is root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Fallback: if config/config.yaml doesn't exist at the detected root, try CWD
if not (PROJECT_ROOT / "config" / "config.yaml").exists():
    PROJECT_ROOT = Path.cwd()

CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.yaml"


class GeneralConfig(BaseModel):
    """General configuration settings."""

    random_state: int
    environment: str


class DataConfig(BaseModel):
    """Data-related configuration."""

    dataset_path: str
    separator: str
    test_size: float
    shuffle: bool
    target_feature: str
    feature_subset: list[str]


class ArtifactsConfig(BaseModel):
    """Artifacts storage configuration."""

    base_path: str
    train_data: str
    test_data: str
    clean_data: str
    preprocessor: str
    model: str

    @property
    def train_data_path(self) -> str:
        """Get full path to training data."""
        return str(Path(self.base_path) / self.train_data)

    @property
    def test_data_path(self) -> str:
        """Get full path to test data."""
        return str(Path(self.base_path) / self.test_data)

    @property
    def clean_data_path(self) -> str:
        """Get full path to clean data."""
        return str(Path(self.base_path) / self.clean_data)

    @property
    def preprocessor_path(self) -> str:
        """Get full path to preprocessor object."""
        return str(Path(self.base_path) / self.preprocessor)

    @property
    def model_path(self) -> str:
        """Get full path to model object."""
        return str(Path(self.base_path) / self.model)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str
    format: str
    level: str


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    imputation: dict[str, str]
    encoding: dict[str, Any]


class ModelTrainingConfig(BaseModel):
    """Model training configuration."""

    scoring: str
    cv_folds: int
    random_search_iter: int
    random_search_cv: int
    n_jobs: int
    nested_cv_outer_folds: int
    nested_cv_inner_folds: int
    models: list[dict[str, Any]]


class APIConfig(BaseModel):
    """API configuration."""

    host: str
    port: int = Field(default=5000)
    debug: bool
    cors_origins: list[str]

    def __init__(self, **data):
        super().__init__(**data)
        # Override port if PORT env var is set (for cloud deployment)
        env_port = os.getenv("PORT")
        if env_port:
            self.port = int(env_port)


class ValidationConfig(BaseModel):
    """Data validation configuration."""

    outlier_threshold: dict[str, float]


class Config(BaseModel):
    """Main configuration class containing all sub-configurations."""

    general: GeneralConfig
    data: DataConfig
    artifacts: ArtifactsConfig
    logging: LoggingConfig
    preprocessing: PreprocessingConfig
    model_training: ModelTrainingConfig
    api: APIConfig
    validation: ValidationConfig

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file. If None, uses default config.yaml.

        Returns:
            Config: Configuration object.

        Raises:
            FileNotFoundError: If configuration file is not found.
            ValidationError: If configuration data is invalid.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_FILE

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)


# Global configuration instance
_config: Config | None = None


def get_config(config_path: Path | None = None) -> Config:
    """
    Get the global configuration instance.
    If not loaded yet, loads from the specified path or default.

    Args:
        config_path: Path to configuration file.

    Returns:
        Config: Global configuration instance.
    """
    global _config
    if _config is None:
        _config = Config.from_yaml(config_path)
    return _config


def reload_config(config_path: Path | None = None) -> Config:
    """
    Reload configuration from file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Config: Reloaded configuration instance.
    """
    global _config
    _config = Config.from_yaml(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration loading
    cfg = get_config()
    print(f"Environment: {cfg.general.environment}")
    print(f"Random State: {cfg.general.random_state}")
    print(f"Dataset Path: {cfg.data.dataset_path}")
    print(f"Train Data Path: {cfg.artifacts.train_data_path}")
    print(f"Feature Subset: {cfg.data.feature_subset}")
    print(f"API Host: {cfg.api.host}:{cfg.api.port}")
