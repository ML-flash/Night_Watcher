#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.logging import setup_logging
from agents.lm_studio import LMStudioProvider
from memory.advanced_system import AdvancedMemorySystem
from integration.enhanced_system import EnhancedNightWatcherWorkflow
from config import load_config, create_default_config


def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting first-time setup of Enhanced Night_watcher")

    # Create default config if it doesn't exist
    if not os.path.exists("config.json"):
        logger.info("Creating default configuration...")
        create_default_config("config.json")

    # Load configuration
    config = load_config("config.json")

    # Add memory configuration if not present
    if "memory" not in config:
        logger.info("Adding memory configuration to config...")
        config["memory"] = {
            "store_type": "simple",  # Start with simple store for first run
            "use_networkx": True,
            "file_path": "data/memory/night_watcher_memory.pkl"
        }

        # Save updated config
        import json
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Initialize memory system with defaults
    logger.info("Initializing advanced memory system...")
    memory_system = AdvancedMemorySystem(config.get("memory", {}))

    # Initialize tracked entities for knowledge graph
    logger.info("Adding default tracked entities to knowledge graph...")
    kg = memory_system.knowledge_graph

    # Add default entities to track
    default_actors = [
        "Donald Trump", "Joe Biden", "Congress", "Supreme Court",
        "Republican Party", "Democratic Party", "White House"
    ]

    # Add default organizations
    default_orgs = [
        "FBI", "Justice Department", "State Department", "Pentagon",
        "Fox News", "CNN", "MSNBC", "New York Times"
    ]

    # Add actors to knowledge graph
    for actor in default_actors:
        kg.add_actor(actor)

    # Add organizations
    for org in default_orgs:
        kg.add_organization(org)

    # Add indicators
    indicators = [
        "Institutional Undermining", "Democratic Norm Violations",
        "Media Delegitimization", "Opposition Targeting",
        "Power Concentration", "Accountability Evasion"
    ]

    for indicator in indicators:
        kg.add_indicator(indicator)

    # Save initialized memory
    logger.info("Saving initialized memory system...")
    memory_system.save("data/memory/night_watcher_memory.pkl")

    # Initialize LLM provider
    llm_host = config.get("llm_provider", {}).get("host", "http://localhost:1234")
    logger.info(f"Connecting to LLM at {llm_host}...")
    llm_provider = LMStudioProvider(host=llm_host)

    # Check LLM connection with a simple test
    try:
        response = llm_provider.complete("Test connection. Say 'Connected'.", max_tokens=10)
        if "error" in response:
            logger.error(f"LLM connection failed: {response['error']}")
            print(f"Error connecting to LLM at {llm_host}. Please ensure LM Studio is running.")
            return 1
        else:
            logger.info("LLM connection successful")
    except Exception as e:
        logger.error(f"LLM connection error: {str(e)}")
        print(f"Error connecting to LLM: {str(e)}")
        return 1

    # Run minimal first workflow to populate memory
    logger.info("Running initial workflow to populate memory...")

    # Limit articles for first run
    config["content_collection"]["article_limit"] = 3

    # Initialize workflow
    workflow = EnhancedNightWatcherWorkflow(
        llm_provider=llm_provider,
        memory_config=config.get("memory", {}),
        output_dir="data"
    )

    # Run minimal workflow
    result = workflow.run({
        "article_limit": 3,
        "manipulation_threshold": config["content_analysis"]["manipulation_threshold"],
        "sources": config["content_collection"]["sources"][:2],  # Limit sources for first run
        "generate_reports": True
    })

    print("\n=== First-time setup complete ===")
    print(f"Articles collected: {result['articles_collected']}")
    print(f"Articles analyzed: {result['articles_analyzed']}")
    print(f"Counter-narratives generated: {result['counter_narratives_generated']}")
    print(f"Memory initialized and saved to: data/memory/night_watcher_memory.pkl")
    print(f"You can now run the full workflow with: python run_enhanced.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())