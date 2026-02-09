import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ames_mlproject.config import get_config

# Load configuration
config = get_config()

# Configure logging format
LOG_FORMAT = config.logging.format
LOG_LEVEL = getattr(logging, config.logging.level.upper())

# Ensure logs directory exists
LOGS_DIR = Path(os.getcwd()) / config.logging.log_dir
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "running_logs.log"

# Setup Handlers
# 1. Rotating File Handler (Keeps logs between runs, rotates when file is 5MB)
file_handler = RotatingFileHandler(
    filename=str(LOG_FILE_PATH), maxBytes=5 * 1024 * 1024, backupCount=5  # 5MB
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
file_handler.setLevel(LOG_LEVEL)

# 2. Console Handler (Standard output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
console_handler.setLevel(LOG_LEVEL)

# Configure Root Logger
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=[file_handler, console_handler])

# Export the root logger
logger = logging.getLogger("ames_mlproject")

if __name__ == "__main__":
    logger.info("Logging system initialized with RotatingFile and Console handlers.")
