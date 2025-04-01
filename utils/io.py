"""
Night_watcher I/O Utilities
Utility functions for file input/output operations.
"""

import os
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def save_to_file(content: Any, filepath: str) -> bool:
    """
    Save content to a file with appropriate format.

    Args:
        content: The content to save (dict, list, or string)
        filepath: Path where the file should be saved

    Returns:
        True if save was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(content) if content is not None else "No content generated")

        return True
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {str(e)}")
        return False


def load_json_file(filepath: str) -> Any:
    """
    Load JSON from a file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed JSON data, or None if file couldn't be loaded
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        return None


def load_text_file(filepath: str) -> str:
    """
    Load text from a file.

    Args:
        filepath: Path to the text file

    Returns:
        File contents as string, or empty string if file couldn't be loaded
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading text from {filepath}: {str(e)}")
        return ""


def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists.

    Args:
        directory: Directory path to ensure exists

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        return False