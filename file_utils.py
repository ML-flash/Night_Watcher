import os
import json
import logging

logger = logging.getLogger(__name__)


def safe_json_load(filepath, default=None):
    """Safely load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return default
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return default
    except PermissionError:
        logger.error(f"Permission denied reading {filepath}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {e}")
        return default


def safe_json_save(filepath, data):
    """Safely save JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_path = f"{filepath}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.rename(temp_path, filepath)
        return True
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        try:
            if os.path.exists(f"{filepath}.tmp"):
                os.remove(f"{filepath}.tmp")
        except Exception:
            pass
        return False
