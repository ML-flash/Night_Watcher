"""
Night_watcher Configuration Module
Handles configuration loading, validation, and defaults.
"""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "llm_provider": {
        "type": "lm_studio",
        "host": "http://localhost:1234",
        "model": "default"
    },
    "content_collection": {
        "article_limit": 5,
        "sources": [
            {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"},
            {"url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "type": "rss", "bias": "center-left"},
            {"url": "https://feeds.foxnews.com/foxnews/politics", "type": "rss", "bias": "right"}
        ]
    },
    "content_analysis": {
        "manipulation_threshold": 6
    },
    "memory": {
        "store_type": "simple",
        "file_path": "data/memory/night_watcher_memory.pkl",
        "embedding_provider": "simple"
    },
    "output": {
        "base_dir": "data",
        "save_collected": True,
        "save_analyses": True
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs"
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration as a dictionary
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Merge with default config to ensure all required fields
        merged_config = DEFAULT_CONFIG.copy()
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(merged_config, config)
        return merged_config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        logger.warning("Using default configuration")
        return DEFAULT_CONFIG


def create_default_config(config_path: str) -> bool:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to save the default configuration
        
    Returns:
        True if creation was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception as e:
        logger.error(f"Error saving default config to {config_path}: {str(e)}")
        return False


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check required sections
    required_sections = ["llm_provider", "content_collection", "content_analysis", "output"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")
    
    # Check LLM provider
    if "llm_provider" in config:
        if "type" not in config["llm_provider"]:
            errors.append("Missing 'type' in llm_provider configuration")
        
        provider_type = config["llm_provider"].get("type")
        if provider_type == "lm_studio" and "host" not in config["llm_provider"]:
            errors.append("Missing 'host' for LM Studio provider")
    
    # Check content collection
    if "content_collection" in config:
        if "sources" not in config["content_collection"]:
            errors.append("Missing 'sources' in content_collection configuration")
        
        sources = config["content_collection"].get("sources", [])
        for i, source in enumerate(sources):
            if "url" not in source:
                errors.append(f"Missing 'url' in source {i}")
            if "type" not in source:
                errors.append(f"Missing 'type' in source {i}")
    
    return errors
