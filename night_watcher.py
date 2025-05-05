#!/usr/bin/env python3
"""
Night_watcher - Intelligence Gathering System
A streamlined system for analyzing news and identifying authoritarian patterns.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any

# ==========================================
# Configuration Functions
# ==========================================

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
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

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
        logging.error(f"Error loading config from {config_path}: {str(e)}")
        logging.warning("Using default configuration")
        return DEFAULT_CONFIG

def create_default_config(config_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving default config to {config_path}: {str(e)}")
        return False

# ==========================================
# Execution Logic
# ==========================================

def run_workflow(config, reset_date=False):
    from providers import initialize_llm_provider
    from memory import MemorySystem, KnowledgeGraph
    from workflow import NightWatcherWorkflow
    from utils import get_analysis_date_range, save_run_date

    llm_provider = initialize_llm_provider(config)
    llm_available = llm_provider is not None

    memory_system = MemorySystem(
        store_type=config.get("memory", {}).get("store_type", "simple"),
        config=config.get("memory", {})
    )

    memory_path = config.get("memory", {}).get("file_path")
    if memory_path and os.path.exists(memory_path):
        logging.info(f"Loading memory from {memory_path}")
        memory_system.load(memory_path)

    start_date = datetime(2025, 1, 20) if reset_date else get_analysis_date_range(config["output"]["base_dir"])[0]
    end_date = datetime.now()

    workflow = NightWatcherWorkflow(
        llm_provider=llm_provider,
        memory_system=memory_system,
        output_dir=config["output"]["base_dir"]
    )

    workflow_params = {
        "article_limit": config["content_collection"]["article_limit"],
        "sources": config["content_collection"]["sources"],
        "start_date": start_date,
        "end_date": end_date,
        "llm_provider_available": llm_available
    }

    result = workflow.run(workflow_params)

    if memory_path:
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        memory_system.save(memory_path)

    save_run_date(config["output"]["base_dir"], end_date)

    # === Post-run Knowledge Graph Summary ===
    print("\n=== Knowledge Graph Summary ===")
    kg_stats = memory_system.knowledge_graph.summarize()

    print("Entities:")
    for etype, count in kg_stats["entity_counts"].items():
        print(f"  - {etype}: {count}")

    print("\nRelationships:")
    for rtype, count in kg_stats["relationship_types"].items():
        print(f"  - {rtype}: {count}")

    print("\nTop Actors:")
    for actor in kg_stats["top_actors"]:
        print(f"  - {actor['name']} ({actor['mentions']} mentions)")

    print("\nTop Institutions:")
    for inst in kg_stats["top_institutions"]:
        print(f"  - {inst['name']} (targeted by {inst['targeted_by']} actors)")

    print(f"\nRisk Level: {kg_stats['risk_level']} (Score: {kg_stats['erosion_score']}/10)")

    print(f"\n=== Processing complete ===")
    print(f"Articles collected: {result.get('articles_collected', 0)}")
    print(f"Articles analyzed: {result.get('articles_analyzed', 0) if llm_available else 'N/A (No LLM)'}")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Outputs saved to: {config['output']['base_dir']}")

    return 0

def main():
    parser = argparse.ArgumentParser(description="Night_watcher - Intelligence Gathering System")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--reset-date", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    run_workflow(config, args.reset_date)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  Night_watcher Intelligence Gathering System              ║
    ║                                                           ║
    ║  Monitoring and analyzing authoritarian patterns          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    sys.exit(main())
