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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

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
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
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
        logging.error(f"Error loading config from {config_path}: {str(e)}")
        logging.warning("Using default configuration")
        return DEFAULT_CONFIG

def create_default_config(config_path: str) -> bool:
    """Create a default configuration file."""
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception as e:
        logging.error(f"Error saving default config to {config_path}: {str(e)}")
        return False

# ==========================================
# Main Function
# ==========================================

def main():
    """Main entry point for Night_watcher intelligence gathering system."""
    parser = argparse.ArgumentParser(
        description="Night_watcher - Intelligence Gathering System"
    )

    parser.add_argument("--config", default="config.json",
                        help="Path to configuration file (default: config.json)")
    parser.add_argument("--llm-host", default=None,
                        help="Override LLM API host URL (default: http://localhost:1234)")
    parser.add_argument("--article-limit", type=int, default=None,
                        help="Override maximum articles to collect per source (default: 5)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: ./data)")
    parser.add_argument("--reset-date", action="store_true",
                        help="Reset date tracking to start from inauguration day (Jan 20, 2025)")
    parser.add_argument("--use-anthropic", action="store_true",
                        help="Force using Anthropic API instead of LM Studio")

    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logging.info("Starting Night_watcher intelligence gathering workflow")

    # Load or create configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found at {args.config}")
        print("Creating default configuration...")
        create_default_config(args.config)
        print(f"Default configuration created at {args.config}")

    config = load_config(args.config)

    # Override config with command-line arguments
    if args.llm_host:
        config["llm_provider"]["host"] = args.llm_host

    if args.article_limit:
        config["content_collection"]["article_limit"] = args.article_limit

    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
        
    # Create necessary directories
    ensure_directories(config["output"]["base_dir"])

    # Handle --use-anthropic flag
    if args.use_anthropic:
        try:
            from providers import AnthropicProvider
            
            print("\nUsing Anthropic API as requested.")
            api_key, model = get_anthropic_credentials()
            
            # Update config to use Anthropic
            config["llm_provider"]["type"] = "anthropic"
            config["llm_provider"]["api_key"] = api_key
            config["llm_provider"]["model"] = model
                
        except ImportError:
            print("\nError: Anthropic SDK not installed.")
            print("Install with: pip install anthropic")
    
    # Run the workflow
    run_workflow(config, args.reset_date)
    
    return 0

def setup_logging():
    """Set up logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/night_watcher_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def ensure_directories(base_dir="."):
    """Ensure required directories exist."""
    directories = [
        os.path.join(base_dir, "collected"),
        os.path.join(base_dir, "analyzed"),
        os.path.join(base_dir, "memory"),
        os.path.join(base_dir, "analysis"),
        os.path.join(base_dir, "logs")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_anthropic_credentials():
    """Get Anthropic API key and model from user."""
    api_key = input("\nEnter your Anthropic API key: ").strip()
    
    print("\nAvailable Claude models:")
    print("1. Claude 3 Opus (most powerful)")
    print("2. Claude 3 Sonnet (balanced)")
    print("3. Claude 3 Haiku (fastest)")
    
    model_choice = input("Select model (1-3, default=3): ").strip()
    
    models = {
        "1": "claude-3-opus-20240229",
        "2": "claude-3-sonnet-20240229",
        "3": "claude-3-haiku-20240307"
    }
    
    model = models.get(model_choice, "claude-3-haiku-20240307")
    
    return api_key, model

def check_lm_studio_connection(host: str) -> bool:
    """Check if LM Studio is running and accessible."""
    import requests
    try:
        response = requests.get(f"{host}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def use_anthropic_api() -> bool:
    """Ask user if they want to use the Anthropic API."""
    print("\nLM Studio server is not available.")
    print("Would you like to use the Anthropic API instead? (requires API key)")
    response = input("Use Anthropic API? (y/n): ").strip().lower()
    return response == 'y' or response == 'yes'

def run_workflow(config, reset_date=False):
    """Run the Night_watcher intelligence gathering workflow."""
    try:
        # Import the necessary modules
        from providers import initialize_llm_provider
        from memory import MemorySystem, KnowledgeGraph
        from workflow import NightWatcherWorkflow
        
        # Initialize LLM provider
        llm_provider = initialize_llm_provider(config)
        
        # If no LLM provider is available, run limited workflow
        if llm_provider is None:
            logging.warning("Running with limited capabilities (content collection only)")
            print("\nRunning with limited capabilities (content collection only)")
            llm_available = False
        else:
            llm_available = True

        # Initialize memory system
        memory_system = MemorySystem(
            store_type=config.get("memory", {}).get("store_type", "simple"),
            config=config.get("memory", {})
        )

        # Load existing memory if available
        memory_path = config.get("memory", {}).get("file_path")
        if memory_path and os.path.exists(memory_path):
            logging.info(f"Loading memory from {memory_path}")
            memory_system.load(memory_path)

        # Initialize knowledge graph
        kg_path = os.path.join(config["output"]["base_dir"], "memory", "knowledge_graph.pkl")
        os.makedirs(os.path.dirname(kg_path), exist_ok=True)

        knowledge_graph = KnowledgeGraph()
        
        # Load existing knowledge graph if available
        if os.path.exists(kg_path) and not reset_date:
            logging.info(f"Loading knowledge graph from {kg_path}")
            knowledge_graph.load(kg_path)
            
        # Initialize workflow
        workflow = NightWatcherWorkflow(
            llm_provider=llm_provider,
            memory_system=memory_system,
            output_dir=config["output"]["base_dir"]
        )
        
        # Determine date range
        from utils import get_analysis_date_range, save_run_date
        
        if reset_date:
            # Start from inauguration day
            start_date = datetime(2025, 1, 20)
            end_date = datetime.now()
            logging.info(f"Reset date specified - starting from inauguration day: {start_date.isoformat()}")
        else:
            # Get the date range from the tracking system
            start_date, end_date = get_analysis_date_range(config["output"]["base_dir"])
            logging.info(f"Continuing from last run - date range: {start_date.isoformat()} to {end_date.isoformat()}")

        # Set up workflow parameters
        workflow_params = {
            "article_limit": config["content_collection"]["article_limit"],
            "sources": config["content_collection"]["sources"],
            "pattern_analysis_days": 30,  # Default analysis period
            "start_date": start_date,
            "end_date": end_date,
            "llm_provider_available": llm_available
        }
            
        # Run workflow
        result = workflow.run(workflow_params)
        
        # Save memory system
        if memory_path:
            memory_dir = os.path.dirname(memory_path)
            if memory_dir:
                os.makedirs(memory_dir, exist_ok=True)
            logging.info(f"Saving memory to {memory_path}")
            memory_system.save(memory_path)
            
        # Update date tracking - save the current end date for next run
        save_run_date(config["output"]["base_dir"], end_date)

        print(f"\n=== Processing complete ===")
        print(f"Articles collected: {result.get('articles_collected', 0)}")
        print(f"Articles analyzed: {result.get('articles_analyzed', 0) if llm_available else 'N/A (No LLM available)'}")
        print(f"Pattern analyses generated: {result.get('pattern_analyses_generated', 0) if llm_available else 'N/A (No LLM available)'}")
        print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
        print(f"All outputs saved in {config['output']['base_dir']}")
        print(f"Next run will continue from: {end_date.isoformat()}")

        return 0
    except Exception as e:
        logging.error(f"Error in workflow execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1

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
