Here's the complete updated `run.py` file for the Night_watcher framework:

```python
#!/usr/bin/env python3
"""
Night_watcher - Simple Launcher
Starts the Night_watcher framework with minimal setup required.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Ensure the package is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import Night_watcher modules
try:
    from config import load_config, create_default_config
    from agents.base import LLMProvider
    from agents.lm_studio import LMStudioProvider
    from workflow.orchestrator import NightWatcherWorkflow
    from memory.system import MemorySystem
    from utils.logging import setup_logging
except ImportError as e:
    print(f"Error importing Night_watcher modules: {e}")
    print("Make sure you're running this script from the Night_watcher directory.")
    sys.exit(1)


def run_workflow(config_path, llm_host=None, article_limit=50, output_dir=None):
    """Run the Night_watcher workflow with the given configuration."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Night_watcher workflow")
    
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
        
        result = workflow.run({
            "article_limit": config["content_collection"]["article_limit"],
            "manipulation_threshold": config["content_analysis"]["manipulation_threshold"],
            "sources": config["content_collection"]["sources"]
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
        print(f"Counter-narratives generated: {result['counter_narratives_generated']}")
        print(f"All outputs saved in {result['output_dir']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


def ensure_directories(base_dir="."):
    """Ensure required directories exist."""
    directories = [
        f"{base_dir}/data/collected",
        f"{base_dir}/data/analyzed",
        f"{base_dir}/data/counter_narratives",
        f"{base_dir}/data/memory",
        f"{base_dir}/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    """Main entry point for Night_watcher simple launcher."""
    parser = argparse.ArgumentParser(
        description="Night_watcher - Simple Launcher"
    )
    
    parser.add_argument("--config", default="config.json", 
                        help="Path to configuration file (default: config.json)")
    parser.add_argument("--llm-host", default="http://localhost:1234",
                        help="Override LLM API host URL (default: http://localhost:1234)")
    parser.add_argument("--article-limit", type=int, default=50,
                        help="Override maximum articles to collect per source (default: 50)")
    parser.add_argument("--output-dir", default=".",
                        help="Override output directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    ensure_directories(args.output_dir)
    
    # Run workflow
    return run_workflow(
        config_path=args.config,
        llm_host=args.llm_host,
        article_limit=args.article_limit,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  Night_watcher Framework - Simple Launcher                ║
    ║                                                           ║
    ║  A counter-narrative tool for democratic resilience       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    sys.exit(main())
    
