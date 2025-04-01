"""
Night_watcher Logging Utilities
Utility functions for logging configuration.
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to store log files (default: "logs")
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/night_watcher_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("night_watcher")


def get_level_from_string(level_str: str) -> int:
    """
    Convert a string logging level to the corresponding logging constant.

    Args:
        level_str: String representation of the logging level

    Returns:
        Logging level constant
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    return level_map.get(level_str.upper(), logging.INFO)