import os
import logging
from logging.handlers import RotatingFileHandler


def setup_production_logging():
    """Setup logging for production debugging."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    main_handler = logging.FileHandler(f"{log_dir}/night_watcher.log")
    main_handler.setLevel(logging.INFO)

    error_handler = logging.FileHandler(f"{log_dir}/errors.log")
    error_handler.setLevel(logging.ERROR)

    debug_handler = RotatingFileHandler(
        f"{log_dir}/debug.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    debug_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    for handler in [main_handler, error_handler, debug_handler]:
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for handler in [main_handler, error_handler, debug_handler]:
        root_logger.addHandler(handler)
