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
from typing import Dict, List, Any, Optional, Tuple

# ==========================================
# Configuration Functions
# ==========================================

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
            user_cfg = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        def deep_update(b: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and k in b and isinstance(b[k], dict):
                    deep_update(b[k], v)
                else:
                    b[k] = v
        deep_update(merged, user_cfg)
        return merged
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG


def create_default_config(config_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving default config: {e}")
        return False


def get_anthropic_credentials() -> Tuple[str, str]:
    api_key = input("LM Studio unavailable. Enter Anthropic API key (or blank to skip): ").strip()
    if not api_key:
        print("No Anthropic key provided; continuing without LLM.")
        return "", ""
    model = input("Enter Anthropic model (default: claude-3-haiku-20240307): ").strip()
    if not model or model == "3" or not model.startswith("claude-"):
        model = "claude-3-haiku-20240307"
        print(f"Using default model: {model}")
    return api_key, model

# ==========================================
# File I/O Utilities
# ==========================================

def save_to_file(content: Any, filepath: str) -> bool:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if isinstance(content, (dict, list)):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(content) if content is not None else "No content")
        return True
    except Exception as e:
        logging.error(f"Error saving to {filepath}: {e}")
        return False


def load_json_file(filepath: str) -> Any:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return None

# ==========================================
# Date Tracking Utilities
# ==========================================

def get_last_run_date(data_dir: str) -> datetime:
    date_file = os.path.join(data_dir, "last_run_date.txt")
    if os.path.exists(date_file):
        try:
            with open(date_file, 'r') as f:
                return datetime.fromisoformat(f.read().strip())
        except Exception as e:
            logging.error(f"Error reading last run date: {e}")
    logging.info("No previous run date found, starting from inauguration day (Jan 20, 2025)")
    return datetime(2025, 1, 20)


def save_run_date(data_dir: str, date: Optional[datetime] = None) -> bool:
    date = date or datetime.now()
    date_file = os.path.join(data_dir, "last_run_date.txt")
    try:
        os.makedirs(os.path.dirname(date_file), exist_ok=True)
        with open(date_file, 'w') as f:
            f.write(date.isoformat())
        logging.info(f"Saved run date: {date.isoformat()}")
        return True
    except Exception as e:
        logging.error(f"Error saving run date: {e}")
        return False

# ==========================================
# Execution Logic
# ==========================================

def run_workflow(config: Dict[str, Any], reset_date: bool=False) -> int:
    from providers import initialize_llm_provider
    from collector import ContentCollector
    from analyzer import ContentAnalyzer

    output_dir = config["output"]["base_dir"]
    collected_dir = os.path.join(output_dir, "collected")
    analyzed_dir = os.path.join(output_dir, "analyzed")
    prompt_chains_dir = os.path.join(output_dir, "prompt_chains")
    os.makedirs(collected_dir, exist_ok=True)
    os.makedirs(analyzed_dir, exist_ok=True)
    os.makedirs(prompt_chains_dir, exist_ok=True)

    llm_provider = initialize_llm_provider(config)
    llm_available = llm_provider is not None

    start_date = datetime(2025, 1, 20) if reset_date else get_last_run_date(output_dir)
    end_date = datetime.now()

    collector = ContentCollector(config)
    analyzer = ContentAnalyzer(llm_provider) if llm_available else None

    logging.info("Starting content collection...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_input = {
        "limit": config["content_collection"]["article_limit"],
        "sources": config["content_collection"]["sources"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    result = collector.process(collection_input)
    articles = result.get("articles", [])

    for idx, art in enumerate(articles, start=1):
        fname = f"article_{idx}_{timestamp}.json"
        save_to_file(art, os.path.join(collected_dir, fname))
    logging.info(f"Collected {len(articles)} articles")

    analyses = []
    if llm_available and analyzer and articles:
        logging.info("Starting multi-round content analysis...")
        analysis_input = {"articles": articles}
        analysis_res = analyzer.process(analysis_input)
        analyses = analysis_res.get("analyses", [])

        for ana in analyses:
            art = ana.get("article", {})
            title = art.get("title", "article")
            safe = "".join(c if c.isalnum() or c.isspace() else "_" for c in title)[:50]
            save_to_file(ana, os.path.join(analyzed_dir, f"analysis_{safe}_{timestamp}.json"))
            chain = ana.get("prompt_chain", [])
            if chain:
                lines = []
                for rd in chain:
                    lines.append(f"=== ROUND {rd.get('round')} ({rd.get('name')}) ===\n")
                    lines.append("-- PROMPT --\n" + rd.get('prompt', '') + "\n")
                    lines.append("-- RESPONSE --\n" + rd.get('response', '') + "\n\n")
                save_to_file(''.join(lines), os.path.join(prompt_chains_dir, f"prompt_chain_{safe}_{timestamp}.txt"))
        logging.info(f"Analyzed {len(analyses)} articles")
    else:
        if not llm_available:
            logging.warning("LLM provider not available; skipping analysis")
        elif not articles:
            logging.warning("No articles collected; skipping analysis")

    save_run_date(output_dir, end_date)

    print(f"\n=== Processing complete ===")
    print(f"Articles collected: {len(articles)}")
    print(f"Articles analyzed: {len(analyses) if analyses else 'N/A'}")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Prompt chains saved to: {prompt_chains_dir}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Night_watcher - Intelligence Gathering System (Multi-Round)")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--reset-date", action="store_true", help="Reset tracking date to inauguration day")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--use-anthropic", action="store_true", help="Use Anthropic API")
    parser.add_argument("--anthropic-key", help="Anthropic API key")
    parser.add_argument("--anthropic-model", default="claude-3-haiku-20240307",
                        help="Anthropic model to use")
    args = parser.parse_args()

    loglvl = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs(DEFAULT_CONFIG['logging']['log_dir'], exist_ok=True)
    logf = os.path.join(DEFAULT_CONFIG['logging']['log_dir'], f"night_watcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=loglvl,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(logf), logging.StreamHandler()])

    config = load_config(args.config)
    if args.use_anthropic or args.anthropic_key:
        config["llm_provider"]["type"] = "anthropic"
        if args.anthropic_key:
            config["llm_provider"]["api_key"] = args.anthropic_key
        if args.anthropic_model.startswith("claude-"):
            config["llm_provider"]["model"] = args.anthropic_model
        else:
            logging.warning(f"Invalid model {args.anthropic_model}, using default")
            config["llm_provider"]["model"] = "claude-3-haiku-20240307"

    return run_workflow(config, args.reset_date)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║           Night_watcher Intelligence            ║
    ║      Multi-Round Prompting Version               ║
    ╚═══════════════════════════════════════════════════╝
    """)
    sys.exit(main())
