#!/usr/bin/env python3
"""
Night_watcher Collector
Script to run the collector component of the Night_watcher framework.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import local modules
from collector import ContentCollector
from document_repository import DocumentRepository

# Default configuration
DEFAULT_CONFIG = {
    "content_collection": {
        "article_limit": 50,
        "sources": [
            {"url": "https://apnews.com/rss", "type": "rss", "bias": "center"},
            {"url": "https://feeds.npr.org/1001/rss.xml", "type": "rss", "bias": "center-left"},
            {"url": "https://thehill.com/rss/feed/", "type": "rss", "bias": "center"},
            {"url": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml", "type": "rss", "bias": "center"},
            {"url": "https://www.whitehouse.gov/feed/", "type": "rss", "bias": "official-government"},
            {"url": "https://www.federalregister.gov/presidential-documents.rss", "type": "rss",
             "bias": "official-government"}
        ],
        "govt_keywords": [
            "executive order", "administration", "white house", "congress", "senate",
            "house of representatives", "supreme court", "federal", "president",
            "department of", "agency", "regulation", "policy", "law", "legislation",
            "election", "democracy", "constitution", "amendment"
        ]
    },
    "output": {
        "base_dir": "data",
        "save_collected": True
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file with fallback to defaults.
    """
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)

        # Deep merge with defaults
        merged = DEFAULT_CONFIG.copy()

        def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
            for k, v in update.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_update(base[k], v)
                else:
                    base[k] = v

        deep_update(merged, user_cfg)
        return merged
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file.
    """
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving config: {e}")
        return False


def save_to_file(content: Any, filepath: str) -> bool:
    """
    Save content to file.
    """
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


def get_last_run_date(data_dir: str) -> datetime:
    """
    Get the last run date from the date tracking file.
    If no previous date, return Inauguration Day (January 20, 2025).
    """
    date_file = os.path.join(data_dir, "last_run_date.txt")

    if os.path.exists(date_file):
        try:
            with open(date_file, 'r') as f:
                date_str = f.read().strip()
                return datetime.fromisoformat(date_str)
        except Exception as e:
            logging.error(f"Error reading last run date: {e}")

    # Default to Inauguration Day if no previous date
    logging.info("No previous run date found, using Inauguration Day (Jan 20, 2025)")
    return datetime(2025, 1, 20)


def save_run_date(data_dir: str, date: datetime) -> bool:
    """
    Save the current run date for future reference.
    """
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


def run_collector(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """
    Run the collector with given configuration and arguments.
    """
    # Set up output directories
    output_dir = args.output_dir or config["output"].get("base_dir", "data")
    collected_dir = os.path.join(output_dir, "collected")
    document_repo_dir = os.path.join(output_dir, "documents")

    os.makedirs(collected_dir, exist_ok=True)

    # Set up document repository
    doc_repo = DocumentRepository(base_dir=document_repo_dir)

    # Set up collector
    collector = ContentCollector(config, doc_repo)

    # Determine date range
    start_date = None
    end_date = datetime.now()

    if args.reset_date:
        # Use Inauguration Day
        start_date = datetime(2025, 1, 20)
        logging.info("Reset date flag used - starting from Inauguration Day")
    elif args.days:
        # Use specified number of days back
        start_date = end_date - timedelta(days=args.days)
        logging.info(f"Using start date {args.days} days ago: {start_date.isoformat()}")
    else:
        # Use last run date from tracking file
        start_date = get_last_run_date(output_dir)
        logging.info(f"Using last run date: {start_date.isoformat()}")

    # Configure collection
    collection_input = {
        "limit": args.article_limit or config["content_collection"].get("article_limit", 5),
        "sources": config["content_collection"]["sources"],
        "start_date": start_date,
        "end_date": end_date,
        "document_repository": doc_repo,
        "store_documents": True
    }

    # Run collection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting collection at {timestamp}")

    result = collector.process(collection_input)

    # Get results
    articles = result.get("articles", [])
    document_ids = result.get("document_ids", [])
    status = result.get("status", {})

    # Find the most recent article date to update tracking
    latest_article_date = None

    for article in articles:
        if article.get("published"):
            try:
                article_date = datetime.fromisoformat(article["published"])
                if latest_article_date is None or article_date > latest_article_date:
                    latest_article_date = article_date
            except (ValueError, TypeError):
                pass

    # Save the latest article date as the next start date
    if latest_article_date:
        # Add a small buffer (1 second) to avoid duplicate collection
        next_start_date = latest_article_date + timedelta(seconds=1)
        save_run_date(output_dir, next_start_date)
        logging.info(f"Updated next start date to: {next_start_date.isoformat()}")
    else:
        # If no articles found, keep the current end date as next start
        save_run_date(output_dir, end_date)
        logging.info(f"No articles found, updated next start date to: {end_date.isoformat()}")

    # Save articles to individual files if requested
    if config["output"].get("save_collected", True):
        for idx, article in enumerate(articles, start=1):
            # Use document_id if available, otherwise generate a filename
            if "document_id" in article:
                fname = f"article_{article['document_id'][:8]}_{timestamp}.json"
            else:
                fname = f"article_{idx}_{timestamp}.json"

            save_to_file(article, os.path.join(collected_dir, fname))

    # Save collection summary
    summary = {
        "timestamp": timestamp,
        "articles_collected": len(articles),
        "documents_stored": len(document_ids),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat(),
        "next_start_date": latest_article_date.isoformat() if latest_article_date else end_date.isoformat(),
        "document_ids": document_ids,
        "status": status
    }

    summary_file = os.path.join(output_dir, f"collection_summary_{timestamp}.json")
    save_to_file(summary, summary_file)

    # Print summary
    print(f"\n=== Collection Summary ===")
    print(f"Timestamp: {timestamp}")
    print(f"Articles collected: {len(articles)}")
    print(f"Documents stored: {len(document_ids)}")
    print(f"Time range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Next start date: {latest_article_date.isoformat() if latest_article_date else end_date.isoformat()}")

    # Print document repository stats
    repo_stats = doc_repo.get_statistics()
    print(f"\n=== Document Repository Statistics ===")
    print(f"Total documents: {repo_stats['total_documents']}")
    print(f"Total content size: {repo_stats['content_size_bytes'] / 1024:.1f} KB")

    # Print source distribution
    print(f"\n=== Source Distribution ===")
    for source, count in repo_stats.get('sources', {}).items():
        print(f"  - {source}: {count} documents")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")

    return 0


def main() -> int:
    """
    Main entry point.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Night_watcher Collector")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--article-limit", type=int, help="Max articles per source")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--reset-date", action="store_true", help="Reset date to inauguration day")
    parser.add_argument("--days", type=int, help="Collect articles from the last N days")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Create necessary directories
    log_dir = "logs"
    data_dir = args.output_dir or "data"
    collected_dir = os.path.join(data_dir, "collected")
    document_dir = os.path.join(data_dir, "documents")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(collected_dir, exist_ok=True)
    os.makedirs(os.path.join(document_dir, "content"), exist_ok=True)
    os.makedirs(os.path.join(document_dir, "metadata"), exist_ok=True)

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = os.path.join(log_dir, f"collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Load configuration
    config = load_config(args.config)

    # Create default config if it doesn't exist
    if not os.path.exists(args.config):
        save_config(DEFAULT_CONFIG, args.config)
        logging.info(f"Created default configuration at {args.config}")

    print("""
    ╔═══════════════════════════════════════════════════╗
    ║      Night_watcher Collector Component            ║
    ╚═══════════════════════════════════════════════════╝
    """)

    # Run the collector
    return run_collector(config, args)


if __name__ == "__main__":
    sys.exit(main())