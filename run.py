#!/usr/bin/env python3
"""
Night_watcher - Intelligence Gathering System
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
from datetime import datetime, timedelta
from pathlib import Path

# Import Night_watcher modules
from config import load_config, create_default_config
from agents.base import LLMProvider
from agents.lm_studio import LMStudioProvider
from workflow.orchestrator import NightWatcherWorkflow
from memory.system import MemorySystem
from memory.knowledge_graph import KnowledgeGraph
from analysis.entity_extractor import EntityExtractor
from utils.logging import setup_logging
from utils.date_tracking import get_analysis_date_range, save_run_date


def check_lm_studio_connection(host: str) -> bool:
    """Check if LM Studio is running and accessible"""
    import requests
    try:
        response = requests.get(f"{host}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def use_anthropic_api() -> bool:
    """Ask user if they want to use the Anthropic API"""
    print("\nLM Studio server is not available.")
    print("Would you like to use the Anthropic API instead? (requires API key)")
    response = input("Use Anthropic API? (y/n): ").strip().lower()
    return response == 'y' or response == 'yes'


def get_anthropic_credentials():
    """Get Anthropic API key and model from user"""
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


def initialize_llm_provider(config, logger) -> LLMProvider:
    """Initialize LLM provider (either LM Studio or Anthropic)"""
    llm_host = config["llm_provider"]["host"]
    
    # Check if LM Studio is running
    if check_lm_studio_connection(llm_host):
        logger.info(f"Connected to LM Studio at {llm_host}")
        return LMStudioProvider(host=llm_host)
    
    # If LM Studio is not available, offer to use Anthropic API
    if use_anthropic_api():
        try:
            # Import here to avoid dependency requirement if not used
            from agents.anthropic_provider import AnthropicProvider
            
            api_key, model = get_anthropic_credentials()
            logger.info(f"Using Anthropic API with model: {model}")
            return AnthropicProvider(api_key=api_key, model=model)
            
        except ImportError:
            print("\nError: Anthropic SDK not installed.")
            print("Install with: pip install anthropic")
            print("Continuing without LLM capabilities (content collection only)...")
            return None
    else:
        print("\nContinuing without LLM capabilities (content collection only)...")
        return None


def run_workflow(config_path, llm_host=None, article_limit=50, output_dir=None, 
                 reset_date=False, use_repository=False):
    """Run the Night_watcher intelligence gathering workflow with the given configuration."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Night_watcher intelligence gathering workflow")

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
        # Initialize LLM provider with fallback support
        llm_provider = initialize_llm_provider(config, logger)
        
        # If no LLM provider is available, run limited workflow
        if llm_provider is None:
            logger.warning("Running with limited capabilities (content collection only)")
            print("\nRunning with limited capabilities (content collection only)")

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

        # Initialize knowledge graph
        kg_path = os.path.join(config["output"]["base_dir"], "memory", "knowledge_graph.pkl")
        os.makedirs(os.path.dirname(kg_path), exist_ok=True)

        knowledge_graph = KnowledgeGraph(use_networkx=True)
        
        # Check if knowledge graph exists
        if os.path.exists(kg_path) and not reset_date:
            logger.info(f"Loading knowledge graph from {kg_path}")
            knowledge_graph.load(kg_path)
        else:
            logger.info("Initializing new knowledge graph")

        # Initialize entity extractor if LLM provider is available
        entity_extractor = None
        if llm_provider:
            entity_extractor = EntityExtractor(llm_provider, knowledge_graph)
            
        # Initialize and run workflow
        workflow = NightWatcherWorkflow(
            llm_provider=llm_provider,
            memory_system=memory_system,
            output_dir=config["output"]["base_dir"]
        )
        
        # Determine date range
        if reset_date:
            # Start from inauguration day
            start_date = datetime(2025, 1, 20)
            end_date = datetime.now()
            logger.info(f"Reset date specified - starting from inauguration day: {start_date.isoformat()}")
        else:
            # Get the date range from the tracking system
            start_date, end_date = get_analysis_date_range(config["output"]["base_dir"])
            logger.info(f"Continuing from last run - date range: {start_date.isoformat()} to {end_date.isoformat()}")

        # Set up workflow parameters
        workflow_params = {
            "article_limit": config["content_collection"]["article_limit"],
            "sources": config["content_collection"]["sources"],
            "pattern_analysis_days": 30,  # Default analysis period
            "start_date": start_date,
            "end_date": end_date,
            "llm_provider_available": llm_provider is not None
        }
            
        # Run workflow
        result = workflow.run(workflow_params)
        
        # Process analyses with entity extraction if LLM provider is available
        extracted_entities = 0
        extracted_relationships = 0
        
        if llm_provider is not None and entity_extractor is not None and "analyses" in result:
            analyses = result.get("analyses", [])
            logger.info(f"Processing {len(analyses)} analyses with entity extractor")
            
            # Process each analysis
            for analysis in analyses:
                try:
                    extraction_result = entity_extractor.extract_from_analysis(analysis)
                    extracted_entities += len(extraction_result.get("entities", {}))
                    extracted_relationships += len(extraction_result.get("relationships", []))
                except Exception as e:
                    logger.error(f"Error extracting from analysis: {str(e)}")
            
            # Save updated knowledge graph
            logger.info(f"Saving knowledge graph with {extracted_entities} entities and {extracted_relationships} relationships")
            knowledge_graph.save(kg_path)
        
        # Save memory system
        if memory_path:
            memory_dir = os.path.dirname(memory_path)
            if memory_dir:
                os.makedirs(memory_dir, exist_ok=True)
            logger.info(f"Saving memory to {memory_path}")
            memory_system.save(memory_path)
            
        # Update date tracking - save the current end date for next run
        save_run_date(config["output"]["base_dir"], end_date)

        print(f"\n=== Processing complete ===")
        print(f"Articles collected: {result.get('articles_collected', 0)}")
        print(f"Articles analyzed: {result.get('articles_analyzed', 0) if llm_provider else 'N/A (No LLM available)'}")
        print(f"Pattern analyses generated: {result.get('pattern_analyses_generated', 0) if llm_provider else 'N/A (No LLM available)'}")
        
        if entity_extractor is not None:
            print(f"Entities extracted: {extracted_entities}")
            print(f"Relationships extracted: {extracted_relationships}")
        
        print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
        print(f"All outputs saved in {result.get('output_dir', config['output']['base_dir'])}")
        print(f"Knowledge graph saved to {kg_path}")
        print(f"Next run will continue from: {end_date.isoformat()}")

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
    """Main entry point for Night_watcher intelligence gathering system."""
    parser = argparse.ArgumentParser(
        description="Night_watcher - Intelligence Gathering System"
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
    parser.add_argument("--use-anthropic", action="store_true",
                        help="Force using Anthropic API instead of LM Studio")
    parser.add_argument("--use-repository", action="store_true",
                        help="Use repository-based architecture for data provenance")

    args = parser.parse_args()
    
    # Handle --use-anthropic flag
    if args.use_anthropic:
        try:
            from agents.anthropic_provider import AnthropicProvider
            
            print("\nUsing Anthropic API as requested.")
            api_key, model = get_anthropic_credentials()
            
            # Create config to use Anthropic
            config = load_config(args.config)
            config["llm_provider"]["type"] = "anthropic"
            config["llm_provider"]["api_key"] = api_key
            config["llm_provider"]["model"] = model
            
            # Save updated config
            with open(args.config, "w") as f:
                import json
                json.dump(config, f, indent=2)
                
        except ImportError:
            print("\nError: Anthropic SDK not installed.")
            print("Install with: pip install anthropic")

    # Run workflow
    return run_workflow(
        config_path=args.config,
        llm_host=args.llm_host,
        article_limit=args.article_limit,
        output_dir=args.output_dir,
        reset_date=args.reset_date,
        use_repository=args.use_repository
    )


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
