#!/usr/bin/env python3
"""
Night_watcher - Intelligence Gathering System (Multi-Round Prompting Version)
A streamlined system for analyzing news through multiple rounds of prompting.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

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
        "article_limit": 2,
        "sources": [
            {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"},
            {"url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "type": "rss", "bias": "center-left"},
            {"url": "https://feeds.foxnews.com/foxnews/politics", "type": "rss", "bias": "right"}
        ]
    },
    "content_analysis": {
        "manipulation_threshold": 6
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

def get_anthropic_credentials():
    """
    Prompt the user to paste in their Anthropic API key (and optional model).
    Returns (api_key, model_name), or ('','') if none provided.
    """
    api_key = input("LM Studio is unavailable. Enter your Anthropic API key (or leave blank to skip): ").strip()
    if not api_key:
        print("No Anthropic key provided; continuing without LLM.")
        return "", ""
    
    # Make sure we get a valid model name
    model = input("Enter Anthropic model (default: claude-3-haiku-20240307): ").strip()
    
    # If no model specified or just "3", use the default
    if not model or model == "3" or not model.startswith("claude-"):
        model = "claude-3-haiku-20240307"
        print(f"Using default model: {model}")
    
    return api_key, model

# ==========================================
# File I/O Utilities
# ==========================================

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
        logging.error(f"Error saving to {filepath}: {str(e)}")
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
        logging.error(f"Error loading JSON from {filepath}: {str(e)}")
        return None

# ==========================================
# Date Tracking Utilities
# ==========================================

def get_last_run_date(data_dir: str) -> datetime:
    """
    Get the last run date or return the default start date (Jan 20, 2025).
    
    Args:
        data_dir: Directory where date tracking is stored
        
    Returns:
        Last run date as datetime object
    """
    date_file = os.path.join(data_dir, "last_run_date.txt")
    
    if os.path.exists(date_file):
        try:
            with open(date_file, 'r') as f:
                date_str = f.read().strip()
                return datetime.fromisoformat(date_str)
        except (ValueError, IOError) as e:
            logging.error(f"Error reading last run date: {str(e)}")
            # Return default date if there's an error reading the file
            return datetime(2025, 1, 20)
    else:
        # Default to inauguration day if no previous run
        logging.info("No previous run date found, starting from inauguration day (Jan 20, 2025)")
        return datetime(2025, 1, 20)

def save_run_date(data_dir: str, date: Optional[datetime] = None) -> bool:
    """
    Save the current date as the last run date.
    
    Args:
        data_dir: Directory where date tracking is stored
        date: Date to save as last run date (defaults to current date)
        
    Returns:
        True if successful, False otherwise
    """
    if date is None:
        date = datetime.now()
        
    date_file = os.path.join(data_dir, "last_run_date.txt")
    
    try:
        os.makedirs(os.path.dirname(date_file), exist_ok=True)
        
        with open(date_file, 'w') as f:
            f.write(date.isoformat())
            
        logging.info(f"Saved run date: {date.isoformat()}")
        return True
    except Exception as e:
        logging.error(f"Error saving run date: {str(e)}")
        return False

# ==========================================
# Execution Logic
# ==========================================

def run_workflow(config, reset_date=False):
    """
    Run the Night_watcher workflow with multi-round prompting.
    
    Args:
        config: Configuration dictionary
        reset_date: Whether to reset the date tracking
        
    Returns:
        Exit code (0 for success)
    """
    # Import required modules
    from providers import initialize_llm_provider
    from collector import ContentCollector
    from analyzer import ContentAnalyzer
    
    # Setup directories
    output_dir = config["output"]["base_dir"]
    collected_dir = os.path.join(output_dir, "collected")
    analyzed_dir = os.path.join(output_dir, "analyzed")
    prompt_chains_dir = os.path.join(output_dir, "prompt_chains")
    
    os.makedirs(collected_dir, exist_ok=True)
    os.makedirs(analyzed_dir, exist_ok=True)
    os.makedirs(prompt_chains_dir, exist_ok=True)
    
    # Initialize LLM provider
    llm_provider = initialize_llm_provider(config)
    llm_available = llm_provider is not None
    
    # Get date range
    start_date = datetime(2025, 1, 20) if reset_date else get_last_run_date(output_dir)
    end_date = datetime.now()
    
    # Initialize components
    collector = ContentCollector(config["content_collection"]["article_limit"])
    
    # Only initialize analyzer if LLM is available
    analyzer = ContentAnalyzer(llm_provider) if llm_available else None
    
    # 1. Content Collection
    logging.info("Starting content collection...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    collection_input = {
        "limit": config["content_collection"]["article_limit"],
        "sources": config["content_collection"]["sources"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    
    collection_result = collector.process(collection_input)
    articles = collection_result.get("articles", [])
    
    # Save collected articles
    for i, article in enumerate(articles):
        filename = f"article_{i+1}_{timestamp}.json"
        save_to_file(article, os.path.join(collected_dir, filename))
    
    logging.info(f"Collected {len(articles)} articles")
    
    # 2. Content Analysis with Multi-Round Prompting (only if LLM is available)
    analysis_results = {"analyses": []}
    if llm_available and analyzer and articles:
        logging.info("Starting multi-round content analysis...")
        
        analysis_input = {"articles": articles}
        analysis_results = analyzer.process(analysis_input)
        
        # Get results
        analyses = analysis_results.get("analyses", [])
        
        # Save analysis results with prompt chains in separate files
        for i, analysis in enumerate(analyses):
            article = analysis.get("article", {})
            article_title = article.get("title", f"Article_{i+1}")
            safe_title = "".join([c if c.isalnum() or c.isspace() else "_" for c in article_title])[:50]
            
            # Save full analysis JSON
            analysis_filename = f"{safe_title}_{timestamp}.json"
            save_to_file(analysis, os.path.join(analyzed_dir, analysis_filename))
            
            # Save prompt chain in a readable format
            prompt_chain = analysis.get("prompt_chain", [])
            
            if prompt_chain:
                prompt_chain_text = []
                for round_data in prompt_chain:
                    round_num = round_data.get("round", "?")
                    round_name = round_data.get("name", "Unknown")
                    prompt = round_data.get("prompt", "")
                    response = round_data.get("response", "")
                    
                    prompt_chain_text.append(f"=== ROUND {round_num}: {round_name} ===\n")
                    prompt_chain_text.append("--- PROMPT ---\n")
                    prompt_chain_text.append(f"{prompt}\n\n")
                    prompt_chain_text.append("--- RESPONSE ---\n")
                    prompt_chain_text.append(f"{response}\n\n")
                    prompt_chain_text.append("=" * 80 + "\n\n")
                
                prompt_chain_content = "".join(prompt_chain_text)
                prompt_chain_filename = f"prompt_chain_{safe_title}_{timestamp}.txt"
                save_to_file(prompt_chain_content, os.path.join(prompt_chains_dir, prompt_chain_filename))
        
        logging.info(f"Analyzed {len(analyses)} articles with multi-round prompting")
        
        # Print analysis example for immediate feedback
        if analyses:
            print("\n=== SAMPLE MULTI-ROUND ANALYSIS ===")
            sample = analyses[0]
            article = sample.get("article", {})
            print(f"Title: {article.get('title', 'Untitled')}")
            print(f"Source: {article.get('source', 'Unknown')}")
            
            # Show the number of rounds completed
            prompt_chain = sample.get("prompt_chain", [])
            print(f"Completed {len(prompt_chain)} rounds of analysis")
            
            # Show structured facts if available
            structured_facts = sample.get("structured_facts", {})
            if structured_facts:
                fact_count = len(structured_facts.get("facts", []))
                event_count = len(structured_facts.get("events", []))
                quote_count = len(structured_facts.get("direct_quotes", []))
                relationship_count = len(structured_facts.get("relationships", []))
                
                print(f"\nExtracted data:")
                print(f"- Facts: {fact_count}")
                print(f"- Events: {event_count}")
                print(f"- Direct quotes: {quote_count}")
                print(f"- Relationships: {relationship_count}")
            
            # Provide path to prompt chain file
            article_title = article.get("title", f"Article_1")
            safe_title = "".join([c if c.isalnum() or c.isspace() else "_" for c in article_title])[:50]
            prompt_chain_filename = f"prompt_chain_{safe_title}_{timestamp}.txt"
            print(f"\nComplete prompt chain saved to: {os.path.join(prompt_chains_dir, prompt_chain_filename)}")
    else:
        if not llm_available:
            logging.warning("LLM provider not available, skipping analysis")
        elif not articles:
            logging.warning("No articles collected, skipping analysis")
        
    # Save the run date
    save_run_date(output_dir, end_date)
    
    # Display summary
    print(f"\n=== Processing complete ===")
    print(f"Articles collected: {len(articles)}")
    print(f"Articles analyzed: {len(analysis_results.get('analyses', [])) if llm_available and analyzer else 'N/A (No LLM)'}")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Prompt chains saved to: {prompt_chains_dir}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Night_watcher - Intelligence Gathering System (Multi-Round Prompting)")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--reset-date", action="store_true", help="Reset date tracking to inauguration day")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--use-anthropic", action="store_true", help="Force using Anthropic API")
    parser.add_argument("--anthropic-key", help="Anthropic API key")
    parser.add_argument("--anthropic-model", default="claude-3-haiku-20240307", 
                       help="Anthropic model (default: claude-3-haiku-20240307)")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"night_watcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    if args.use_anthropic or args.anthropic_key:
        config["llm_provider"]["type"] = "anthropic"
        
        if args.anthropic_key:
            config["llm_provider"]["api_key"] = args.anthropic_key
        
        # Make sure we use a valid model name
        if args.anthropic_model:
            # Fix model name if needed
            if args.anthropic_model == "3" or not args.anthropic_model.startswith("claude-"):
                logging.warning(f"Invalid model name: {args.anthropic_model}, using default")
                config["llm_provider"]["model"] = "claude-3-haiku-20240307"
            else:
                config["llm_provider"]["model"] = args.anthropic_model
    
    # Run workflow
    return run_workflow(config, args.reset_date)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  Night_watcher Intelligence Gathering System              ║
    ║  [MULTI-ROUND PROMPTING VERSION]                          ║
    ║                                                           ║
    ║  Monitoring and analyzing authoritarian patterns          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    sys.exit(main())