#!/usr/bin/env python3
"""
Night_watcher Framework - Main Entry Point
A modular system for analyzing news, identifying divisive content, and generating counter-narratives.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

from .config import load_config, create_default_config
from .agents.base import LLMProvider
from .agents.lm_studio import LMStudioProvider
from .workflow.orchestrator import NightWatcherWorkflow
from .memory.system import MemorySystem
from .analysis.patterns import PatternRecognition
from .utils.logging import setup_logging


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser for Night_watcher"""
    parser = argparse.ArgumentParser(
        description="Night_watcher - News Analysis and Counter-Narrative System"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run workflow command
    run_parser = subparsers.add_parser("run", help="Run the analysis workflow")
    run_parser.add_argument("--config", default="config.json", help="Path to configuration file")
    run_parser.add_argument("--llm-host", help="Override LLM API host URL")
    run_parser.add_argument("--threshold", type=int, help="Override manipulation score threshold")
    run_parser.add_argument("--article-limit", type=int, help="Override maximum articles to collect per source")
    run_parser.add_argument("--output-dir", help="Override output directory")
    
    # Initialize config command
    init_parser = subparsers.add_parser("init", help="Initialize default configuration")
    init_parser.add_argument("--output", default="config.json", help="Output path for configuration file")
    init_parser.add_argument("--force", action="store_true", help="Force overwrite if file exists")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing data for patterns")
    analyze_parser.add_argument("--config", default="config.json", help="Path to configuration file")
    analyze_parser.add_argument("--memory-file", required=True, help="Path to memory system file")
    analyze_parser.add_argument("--output-dir", default="analysis_results", help="Directory for analysis results")
    analyze_parser.add_argument("--days", type=int, default=30, help="Days to look back for analysis")
    
    # Search memory command
    search_parser = subparsers.add_parser("search", help="Search the memory system")
    search_parser.add_argument("--memory-file", required=True, help="Path to memory system file")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    
    return parser


def handle_run_command(args: argparse.Namespace) -> int:
    """Handle the 'run' command to execute the workflow"""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Night_watcher workflow")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.llm_host:
        config["llm_provider"]["host"] = args.llm_host
        
    if args.threshold:
        config["content_analysis"]["manipulation_threshold"] = args.threshold
        
    if args.article_limit:
        config["content_collection"]["article_limit"] = args.article_limit
        
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
    
    try:
        # Initialize LLM provider
        llm_provider = create_llm_provider(config["llm_provider"])
        
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


def handle_init_command(args: argparse.Namespace) -> int:
    """Handle the 'init' command to create a default configuration"""
    if os.path.exists(args.output) and not args.force:
        print(f"Configuration file {args.output} already exists. Use --force to overwrite.")
        return 1
        
    success = create_default_config(args.output)
    
    if success:
        print(f"Default configuration created at {args.output}")
        return 0
    else:
        print(f"Failed to create configuration file")
        return 1


def handle_analyze_command(args: argparse.Namespace) -> int:
    """Handle the 'analyze' command to analyze existing data"""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting data analysis")
    
    try:
        # Load memory system
        memory_system = MemorySystem()
        if not memory_system.load(args.memory_file):
            print(f"Failed to load memory from {args.memory_file}")
            return 1
            
        # Initialize pattern recognition
        pattern_recognition = PatternRecognition(memory_system)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run analyses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Source bias analysis
        bias_analysis = pattern_recognition.analyze_source_bias_patterns(days=args.days)
        from .utils.io import save_to_file
        save_to_file(
            bias_analysis,
            os.path.join(args.output_dir, f"bias_analysis_{timestamp}.json")
        )
        
        # Topic analysis
        topic_analysis = pattern_recognition.identify_recurring_topics()
        save_to_file(
            topic_analysis,
            os.path.join(args.output_dir, f"topic_analysis_{timestamp}.json")
        )
        
        # Narrative effectiveness
        narrative_analysis = pattern_recognition.analyze_narrative_effectiveness()
        save_to_file(
            narrative_analysis,
            os.path.join(args.output_dir, f"narrative_analysis_{timestamp}.json")
        )
        
        # Temporal trends
        trend_analysis = pattern_recognition.analyze_temporal_trends(lookback_days=args.days)
        save_to_file(
            trend_analysis,
            os.path.join(args.output_dir, f"trend_analysis_{timestamp}.json")
        )
        
        print(f"Analysis complete. Results saved in {args.output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Error in data analysis: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


def handle_search_command(args: argparse.Namespace) -> int:
    """Handle the 'search' command to search the memory system"""
    try:
        # Load memory system
        memory_system = MemorySystem()
        if not memory_system.load(args.memory_file):
            print(f"Failed to load memory from {args.memory_file}")
            return 1
            
        # Perform search
        results = memory_system.search_all(args.query, args.limit)
        
        # Display results
        print(f"\n=== Search Results for '{args.query}' ===\n")
        
        if not any(results.values()):
            print("No results found.")
            return 0
            
        for category, items in results.items():
            if items:
                print(f"\n== {category.replace('_', ' ').title()} ==")
                for i, item in enumerate(items):
                    metadata = item.get("metadata", {})
                    print(f"\n{i+1}. {metadata.get('title', 'Untitled')}")
                    
                    if "source" in metadata:
                        print(f"   Source: {metadata['source']}")
                        
                    if "manipulation_score" in metadata:
                        print(f"   Manipulation Score: {metadata['manipulation_score']}")
                        
                    print(f"   Similarity: {item.get('similarity', 0):.2f}")
                    
                    # Display excerpt of text
                    text = item.get("text", "")
                    if text:
                        excerpt = text[:200] + "..." if len(text) > 200 else text
                        print(f"\n   Excerpt: {excerpt}")
                        
                    print(f"   ID: {item.get('id', '')}")
        
        return 0
    except Exception as e:
        print(f"Error searching memory: {str(e)}")
        return 1


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Create LLM provider based on configuration"""
    provider_type = config.get("type", "lm_studio")
    
    if provider_type == "lm_studio":
        host = config.get("host", "http://localhost:1234")
        return LMStudioProvider(host=host)
    else:
        raise ValueError(f"Unsupported LLM provider type: {provider_type}")


def main() -> int:
    """Main entry point for Night_watcher"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if args.command == "run":
        return handle_run_command(args)
    elif args.command == "init":
        return handle_init_command(args)
    elif args.command == "analyze":
        return handle_analyze_command(args)
    elif args.command == "search":
        return handle_search_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
