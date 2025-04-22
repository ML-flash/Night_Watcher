#!/usr/bin/env python3
"""
Night_watcher - Intelligence Analysis System
Starts the Night_watcher framework focused on intelligence gathering and analysis.
"""

import sys
import os

# Add project root and agents directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "agents"))

import logging
import argparse
from datetime import datetime
from pathlib import Path

# Import Night_watcher modules
from config import load_config, create_default_config
from agents.base import LLMProvider
from agents.lm_studio import LMStudioProvider

from workflow.orchestrator import NightWatcherWorkflow
from memory.system import MemorySystem
from utils.logging import setup_logging
from utils.date_tracking import get_analysis_date_range


def run_workflow(config_path, llm_host=None, article_limit=50, output_dir=None, reset_date=False):
    """Run the Night_watcher intelligence analysis workflow with the given configuration."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Night_watcher intelligence analysis workflow")

    # Load configuration
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}")
        print("Creating default configuration...")
        create_default_config(config_path)
        print(f"Default configuration created at {config_path}")

    config = load_config(config_path)

    # Override config with command-line arguments
    if llm_host:
        config["llm_provider"]["host"] = llm_host

    # Set default article limit to 50 if not specified in config
    if "article_limit" not in config["content_collection"]:
        config["content_collection"]["article_limit"] = 50

    # Override article limit if specified
    if article_limit:
        config["content_collection"]["article_limit"] = article_limit

    # Set default output directory to current directory if not specified
    if "base_dir" not in config["output"]:
        config["output"]["base_dir"] = "."

    # Override output directory if specified
    if output_dir:
        config["output"]["base_dir"] = output_dir
        
    # Create necessary directories
    ensure_directories(config["output"]["base_dir"])

    try:
        # Initialize LLM provider
        llm_config = config["llm_provider"]
        llm_provider = LMStudioProvider(host=llm_config.get("host", "http://localhost:1234"))

        # Initialize memory system
        memory_system = MemorySystem(
            store_type=config.get("memory", {}).get("store_type", "simple"),
            config=config.get("memory", {})
        )

        # Load existing memory if available
        memory_path = config.get("memory", {}).get("file_path")
        if memory_path and os.path.exists(memory_path):
            logger.info(f"Loading memory from {memory_path}")
            memory_system.load(memory_path)

        # Initialize and run workflow
        workflow = NightWatcherWorkflow(
            llm_provider=llm_provider,
            memory_system=memory_system,
            output_dir=config["output"]["base_dir"]
        )
        
        # If reset_date is specified, we'll use Jan 20, 2025 as the start date
        if reset_date:
            start_date = datetime(2025, 1, 20)
            end_date = datetime.now()
            logger.info(f"Date range reset to start from inauguration day: {start_date.isoformat()}")
        else:
            # Get the date range from the tracking system
            start_date, end_date = get_analysis_date_range(config["output"]["base_dir"])

        result = workflow.run({
            "article_limit": config["content_collection"]["article_limit"],
            "sources": config["content_collection"]["sources"],
            "pattern_analysis_days": 30,  # Default analysis period
            "start_date": start_date,
            "end_date": end_date
        })

        # Save memory system
        if memory_path:
            memory_dir = os.path.dirname(memory_path)
            if memory_dir:
                os.makedirs(memory_dir, exist_ok=True)
            logger.info(f"Saving memory to {memory_path}")
            memory_system.save(memory_path)

        print(f"\n=== Processing complete ===")
        print(f"Articles collected: {result['articles_collected']}")
        print(f"Articles analyzed: {result['articles_analyzed']}")
        print(f"Pattern analyses generated: {result['pattern_analyses_generated']}")
        print(f"Date range: {result['date_range']['start_date']} to {result['date_range']['end_date']}")
        print(f"All outputs saved in {result['output_dir']}")

        return 0
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


def ensure_directories(base_dir="."):
    """Ensure required directories exist."""
    directories = [
        os.path.join(base_dir, "data", "collected"),
        os.path.join(base_dir, "data", "analyzed"),
        os.path.join(base_dir, "data", "memory"),
        os.path.join(base_dir, "data", "analysis"),
        os.path.join(base_dir, "logs")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    """Main entry point for Night_watcher intelligence analysis system."""
    parser = argparse.ArgumentParser(
        description="Night_watcher - Intelligence Analysis System"
    )

    parser.add_argument("--config", default="config.json",
                        help="Path to configuration file (default: config.json)")
    parser.add_argument("--llm-host", default="http://localhost:1234",
                        help="Override LLM API host URL (default: http://localhost:1234)")
    parser.add_argument("--article-limit", type=int, default=50,
                        help="Override maximum articles to collect per source (default: 50)")
    parser.add_argument("--output-dir", default=".",
                        help="Override output directory (default: current directory)")
    parser.add_argument("--reset-date", action="store_true",
                        help="Reset date tracking to start from inauguration day (Jan 20, 2025)")

    args = parser.parse_args()

    # Run workflow
    return run_workflow(
        config_path=args.config,
        llm_host=args.llm_host,
        article_limit=args.article_limit,
        output_dir=args.output_dir,
        reset_date=args.reset_date
    )


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  Night_watcher Intelligence Analysis System               ║
    ║                                                           ║
    ║  Monitoring and analyzing authoritarian patterns          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    sys.exit(main())
