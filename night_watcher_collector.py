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
        "article_limit": 10,
        "sources": [
            {"url": "https://apnews.com/rss", "type": "rss", "bias": "center"},
            {"url": "https://feeds.npr.org/1001/rss.xml", "type": "rss", "bias": "center-left"},
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
    },
    "provenance": {
        "enabled": True,
        "dev_mode": True
    }
}

# Constants
INAUGURATION_DAY = datetime(2025, 1, 20)

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


def get_collection_date_range(data_dir: str) -> tuple[datetime, datetime, bool]:
    """
    Get the date range for collection and whether this is a first run.
    
    Returns:
        (start_date, end_date, is_first_run)
    """
    date_file = os.path.join(data_dir, "last_run_date.txt")
    end_date = datetime.now()
    
    if os.path.exists(date_file):
        # This is a subsequent run
        try:
            with open(date_file, 'r') as f:
                date_str = f.read().strip()
                start_date = datetime.fromisoformat(date_str)
                logging.info(f"Using last run date as start: {start_date.isoformat()}")
                return start_date, end_date, False
        except Exception as e:
            logging.error(f"Error reading last run date: {e}")
    
    # This is the first run - collect everything from a wide range
    # Set start_date to Jan 1, 2025 to capture all available content
    # We'll filter to Inauguration Day later in post-processing
    start_date = datetime(2025, 1, 1)  # Wide net for RSS feeds
    logging.info("First run detected - collecting from Jan 1, 2025 to capture all available content")
    logging.info("Articles will be filtered to Inauguration Day (Jan 20, 2025) and later")
    return start_date, end_date, True


def filter_articles_by_inauguration_day(articles: List[Dict[str, Any]], is_first_run: bool) -> List[Dict[str, Any]]:
    """
    Filter articles to only include those from Inauguration Day forward on first run.
    """
    if not is_first_run:
        return articles  # No filtering needed for subsequent runs
    
    filtered_articles = []
    filtered_count = 0
    
    for article in articles:
        article_date = None
        
        # Try to parse the publication date
        if article.get("published"):
            try:
                article_date = datetime.fromisoformat(article["published"])
            except (ValueError, TypeError):
                # If we can't parse the date, include the article to be safe
                filtered_articles.append(article)
                continue
        
        # If no date or date is after Inauguration Day, include it
        if not article_date or article_date >= INAUGURATION_DAY:
            filtered_articles.append(article)
        else:
            filtered_count += 1
    
    if filtered_count > 0:
        logging.info(f"Filtered out {filtered_count} articles published before Inauguration Day")
    
    return filtered_articles


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

    # Determine if provenance is enabled
    provenance_enabled = config.get("provenance", {}).get("enabled", True)
    if args.disable_provenance:
        provenance_enabled = False
    
    dev_mode = config.get("provenance", {}).get("dev_mode", True)

    # Set up document repository with provenance tracking
    if provenance_enabled:
        logging.info("Initializing document repository with provenance tracking")
        doc_repo = DocumentRepository(
            base_dir=document_repo_dir,
            dev_passphrase=args.provenance_passphrase,
            dev_mode=dev_mode
        )
    else:
        logging.info("Initializing document repository without provenance tracking")
        doc_repo = DocumentRepository(base_dir=document_repo_dir)

    # Set up collector
    collector = ContentCollector(config, doc_repo)

    # Determine date range
    if args.reset_date:
        # Use Inauguration Day as start for reset
        start_date = datetime(2025, 1, 1)  # Wide net for collection
        end_date = datetime.now()
        is_first_run = True
        logging.info("Reset date flag used - collecting from Jan 1, 2025 (will filter to Inauguration Day)")
    elif args.days:
        # Use specified number of days back
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        is_first_run = False
        logging.info(f"Using start date {args.days} days ago: {start_date.isoformat()}")
    else:
        # Use automatic date range detection
        start_date, end_date, is_first_run = get_collection_date_range(output_dir)

    # Configure collection
    collection_input = {
        "limit": args.article_limit or config["content_collection"].get("article_limit", 5),
        "sources": config["content_collection"]["sources"],
        "start_date": start_date,
        "end_date": end_date,
        "document_repository": doc_repo,
        "store_documents": False  # We'll store after filtering
    }

    # Run collection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting collection at {timestamp}")
    logging.info(f"Collection date range: {start_date.isoformat()} to {end_date.isoformat()}")

    result = collector.process(collection_input)

    # Get results and apply filtering
    raw_articles = result.get("articles", [])
    logging.info(f"Raw articles collected: {len(raw_articles)}")
    
    # Filter articles for first run (Inauguration Day and later)
    articles = filter_articles_by_inauguration_day(raw_articles, is_first_run)
    logging.info(f"Articles after filtering: {len(articles)}")

    # Now store the filtered articles in the document repository
    document_ids = []
    if articles:
        for article in articles:
            try:
                if not article.get("id"):
                    article["id"] = collector._generate_document_id(article)
                
                doc_id = doc_repo.store_document(
                    article["content"],
                    collector._create_metadata(article)
                )
                document_ids.append(doc_id)
                article["document_id"] = doc_id
            except Exception as e:
                logging.error(f"Error storing document: {e}")

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

    # Save the run date
    if latest_article_date:
        # Add a small buffer (1 second) to avoid duplicate collection
        next_start_date = latest_article_date + timedelta(seconds=1)
        save_run_date(output_dir, next_start_date)
        logging.info(f"Updated next start date to: {next_start_date.isoformat()}")
    else:
        # If no articles found, save current end date as next start
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
        "raw_articles_collected": len(raw_articles),
        "documents_stored": len(document_ids),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat(),
        "next_start_date": latest_article_date.isoformat() if latest_article_date else end_date.isoformat(),
        "document_ids": document_ids,
        "provenance_enabled": provenance_enabled,
        "is_first_run": is_first_run,
        "inauguration_day_filter_applied": is_first_run,
        "status": result.get("status", {})
    }

    summary_file = os.path.join(output_dir, f"collection_summary_{timestamp}.json")
    save_to_file(summary, summary_file)

    # Print summary
    print(f"\n=== Collection Summary ===")
    print(f"Timestamp: {timestamp}")
    print(f"Raw articles collected: {len(raw_articles)}")
    if is_first_run:
        print(f"Articles after Inauguration Day filter: {len(articles)}")
    else:
        print(f"Articles collected: {len(articles)}")
    print(f"Documents stored: {len(document_ids)}")
    print(f"Collection range: {start_date.isoformat()} to {end_date.isoformat()}")
    if is_first_run:
        print(f"Filtered to: {INAUGURATION_DAY.isoformat()} and later")
    print(f"Next start date: {latest_article_date.isoformat() if latest_article_date else end_date.isoformat()}")
    print(f"Provenance tracking: {'Enabled' if provenance_enabled else 'Disabled'}")

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
    
    # Add provenance options
    parser.add_argument("--provenance-passphrase", help="Passphrase for document provenance (dev mode)")
    parser.add_argument("--disable-provenance", action="store_true", help="Disable provenance tracking")

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

    # If provenance passphrase is not provided but environment variable exists, use it
    if not args.provenance_passphrase:
        env_passphrase = os.environ.get("NIGHT_WATCHER_PASSPHRASE")
        if env_passphrase:
            args.provenance_passphrase = env_passphrase
            logging.info("Using provenance passphrase from environment variable")

    # Run the collector
    return run_collector(config, args)


if __name__ == "__main__":
    sys.exit(main())