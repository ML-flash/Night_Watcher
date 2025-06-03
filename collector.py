#!/usr/bin/env python3
"""
Enhanced Night_watcher Historical Content Collector
Specialized for collecting historical political content with advanced date handling,
RSS pagination, archive discovery, and comprehensive fallback strategies.
"""

import time
import logging
import hashlib
import re
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin, parse_qs, urlunparse
import concurrent.futures
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests
import feedparser
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import newspaper3k and trafilatura
try:
    from newspaper import Article, Config
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.error("newspaper3k not installed. Run: pip install newspaper3k")

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logging.warning("trafilatura not installed. Install with: pip install trafilatura")

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    logging.warning("cloudscraper not installed. Cloudflare bypass not available.")

logger = logging.getLogger(__name__)

@dataclass
class HistoricalCollectionConfig:
    """Configuration for historical content collection"""
    start_date: datetime
    end_date: datetime
    target_articles_per_day: int = 10
    max_articles_total: int = 1000
    enable_archive_discovery: bool = True
    enable_pagination: bool = True
    enable_sitemap_crawling: bool = True
    max_pages_per_source: int = 50
    archive_services: List[str] = None
    
    def __post_init__(self):
        if self.archive_services is None:
            self.archive_services = [
                'web.archive.org',
                'archive.today',
                'webcitation.org'
            ]

class HistoricalContentCollector:
    """
    Enhanced collector for historical political content with advanced discovery mechanisms.
    Supports both RSS sources and direct article links with intelligent run behavior.
    """

    def __init__(self, config: Dict[str, Any], document_repository=None, base_dir: str = "data"):
        """Initialize the historical content collector."""
        self.config = config
        self.document_repository = document_repository
        self.base_dir = base_dir
        
        # Extract configuration
        cc = config.get("content_collection", {})
        self.article_limit = cc.get("article_limit", 50)
        self.sources = cc.get("sources", [])
        self.max_workers = cc.get("max_workers", 3)
        self.request_timeout = cc.get("request_timeout", 45)
        self.retry_count = cc.get("retry_count", 3)
        self.bypass_cloudflare = cc.get("bypass_cloudflare", True) and CLOUDSCRAPER_AVAILABLE
        self.delay_between_requests = cc.get("delay_between_requests", 2.0)
        self.use_llm_fallback = cc.get("use_llm_navigation_fallback", True)
        
        # Run state management
        self.last_run_file = os.path.join(base_dir, "last_run_date.txt")
        self.collection_state_file = os.path.join(base_dir, "collection_state.json")
        self.inauguration_day = datetime(2025, 1, 20)
        
        # Ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Historical collection specific settings
        hc = cc.get("historical_collection", {})
        self.enable_archive_discovery = hc.get("enable_archive_discovery", True)
        self.enable_pagination = hc.get("enable_pagination", True)
        self.enable_sitemap_crawling = hc.get("enable_sitemap_crawling", True)
        self.max_pages_per_source = hc.get("max_pages_per_source", 50)
        self.archive_services = hc.get("archive_services", [
            'web.archive.org',
            'archive.today'
        ])
        
        # Political/governmental keywords for filtering
        self.govt_keywords = set(kw.lower() for kw in cc.get("govt_keywords", [
            "executive order", "administration", "white house", "president", "presidential",
            "congress", "congressional", "senate", "senator", "house of representatives",
            "supreme court", "federal court", "federal", "government", "politics", "political",
            "election", "campaign", "democracy", "constitution", "biden", "trump", "harris"
        ]))
        
        # User agents rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        ]
        
        # Initialize logger
        self.logger = logging.getLogger("HistoricalContentCollector")
        
        # Track collection statistics
        self.collection_stats = {
            'total_articles': 0,
            'articles_by_date': {},
            'articles_by_source': {},
            'failed_urls': [],
            'archive_discoveries': 0,
            'pagination_discoveries': 0,
            'sitemap_discoveries': 0
        }
        
        # Initialize sessions
        self._init_sessions()
        
        # Initialize LLM navigator if available
        self.llm_navigator = None
        self._check_llm_availability()

    def _init_sessions(self):
        """Initialize HTTP sessions for requests."""
        self.session = requests.Session()
        
        # Enhanced retry strategy
        retry_strategy = Retry(
            total=self.retry_count,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504, 522, 524],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self._rotate_session_headers()
        
        # Initialize cloudscraper if available
        if self.bypass_cloudflare:
            try:
                self.cloudscraper_session = cloudscraper.create_scraper(
                    browser={'browser': 'firefox', 'platform': 'linux'},
                    delay=5,
                    debug=False
                )
                self.logger.info("Cloudflare bypass initialized")
            except Exception as e:
                self.logger.error(f"Error initializing Cloudflare bypass: {e}")
                self.bypass_cloudflare = False

    def _rotate_session_headers(self):
        """Rotate session headers for stealth."""
        user_agent = random.choice(self.user_agents)
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })

    def _check_llm_availability(self):
        """Check if LLM is available for navigation fallback."""
        # Simplified check - in real implementation, this would check providers
        self.logger.info("LLM navigation checking skipped for historical collector")

    def add_source(self, source_data: Dict[str, Any]) -> bool:
        """
        Add a new source to the configuration.
        
        Args:
            source_data: Source configuration dictionary with keys:
                - url: RSS feed URL or direct article URL
                - type: 'rss', 'article', or 'auto' (auto-detect)
                - bias: bias label (center, left, right, etc.)
                - name: optional display name
                - enabled: whether source is active (default: True)
                
        Returns:
            True if source was added successfully
        """
        try:
            # Validate required fields
            if not source_data.get("url"):
                self.logger.error("Source URL is required")
                return False
            
            url = source_data["url"].strip()
            source_type = source_data.get("type", "auto").lower()
            
            # Auto-detect source type if needed
            if source_type == "auto":
                source_type = self._detect_source_type(url)
            
            # Create source entry
            new_source = {
                "url": url,
                "type": source_type,
                "bias": source_data.get("bias", "unknown"),
                "name": source_data.get("name", self._extract_source_name(url)),
                "enabled": source_data.get("enabled", True),
                "added_at": datetime.now().isoformat(),
                "collection_stats": {
                    "total_articles": 0,
                    "last_successful_collection": None,
                    "consecutive_failures": 0
                }
            }
            
            # Check if source already exists
            existing_source = None
            for i, source in enumerate(self.sources):
                if source.get("url") == url:
                    existing_source = i
                    break
            
            if existing_source is not None:
                # Update existing source
                self.sources[existing_source].update(new_source)
                self.logger.info(f"Updated existing source: {url}")
            else:
                # Add new source
                self.sources.append(new_source)
                self.logger.info(f"Added new source: {url} (type: {source_type})")
            
            # Save updated configuration
            self._save_sources_config()
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return False

    def _detect_source_type(self, url: str) -> str:
        """
        Auto-detect whether URL is RSS feed or direct article.
        
        Args:
            url: URL to analyze
            
        Returns:
            'rss' or 'article'
        """
        url_lower = url.lower()
        
        # Common RSS indicators
        rss_indicators = [
            '/rss', '/feed', '.rss', '.xml', '/atom',
            'feed.xml', 'rss.xml', 'atom.xml'
        ]
        
        if any(indicator in url_lower for indicator in rss_indicators):
            return "rss"
        
        # Try to fetch and check content type
        try:
            response = self.session.head(url, timeout=10)
            content_type = response.headers.get('content-type', '').lower()
            
            if any(ct in content_type for ct in ['xml', 'rss', 'atom']):
                return "rss"
        except:
            pass
        
        # Default to article if not clearly RSS
        return "article"

    def _extract_source_name(self, url: str) -> str:
        """Extract a display name from URL."""
        try:
            domain = urlparse(url).netloc
            # Remove www. and common subdomains
            domain = re.sub(r'^(www\.|feeds\.|rss\.)', '', domain)
            # Capitalize first letter
            return domain.replace('.com', '').replace('.org', '').replace('.net', '').title()
        except:
            return "Unknown Source"

    def _save_sources_config(self):
        """Save current sources configuration."""
        try:
            # Update the main config
            self.config["content_collection"]["sources"] = self.sources
            
            # Save to file if config has a file path
            config_file = getattr(self, 'config_file', 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved sources configuration to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving sources configuration: {e}")

    def get_collection_mode(self, force_mode: Optional[str] = None) -> Tuple[str, datetime, datetime]:
        """
        Determine collection mode and date range based on run history.
        
        Args:
            force_mode: Force specific mode ('first_run', 'incremental', 'full')
            
        Returns:
            Tuple of (mode, start_date, end_date)
        """
        current_time = datetime.now()
        
        if force_mode:
            if force_mode == 'first_run':
                return self._get_first_run_dates(current_time)
            elif force_mode == 'incremental':
                return self._get_incremental_dates(current_time)
            elif force_mode == 'full':
                return self._get_full_collection_dates(current_time)
        
        # Auto-detect mode based on state
        if self._is_first_run():
            return self._get_first_run_dates(current_time)
        else:
            return self._get_incremental_dates(current_time)

    def _is_first_run(self) -> bool:
        """Check if this is the first run of the collector."""
        return not os.path.exists(self.last_run_file)

    def _get_first_run_dates(self, current_time: datetime) -> Tuple[str, datetime, datetime]:
        """
        Get date range for first run - wide net approach.
        Collects everything available, then filters to post-inauguration.
        """
        # Cast wide net for RSS feeds (they often have limited history)
        start_date = datetime(2025, 1, 1)
        end_date = current_time
        
        self.logger.info("FIRST RUN: Collecting wide date range, will filter to post-inauguration")
        return ("first_run", start_date, end_date)

    def _get_incremental_dates(self, current_time: datetime) -> Tuple[str, datetime, datetime]:
        """
        Get date range for incremental run - since last successful collection.
        """
        try:
            if os.path.exists(self.last_run_file):
                with open(self.last_run_file, 'r') as f:
                    last_run_str = f.read().strip()
                    last_run = datetime.fromisoformat(last_run_str)
                    
                    # Small overlap to avoid missing articles
                    start_date = last_run - timedelta(hours=1)
                    end_date = current_time
                    
                    self.logger.info(f"INCREMENTAL RUN: Collecting since {start_date.isoformat()}")
                    return ("incremental", start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Error reading last run date: {e}")
        
        # Fallback to last 24 hours
        start_date = current_time - timedelta(days=1)
        end_date = current_time
        
        self.logger.info("INCREMENTAL RUN (fallback): Collecting last 24 hours")
        return ("incremental", start_date, end_date)

    def _get_full_collection_dates(self, current_time: datetime) -> Tuple[str, datetime, datetime]:
        """Get date range for full collection - everything from inauguration."""
        start_date = self.inauguration_day
        end_date = current_time
        
        self.logger.info("FULL RUN: Collecting everything since inauguration")
        return ("full", start_date, end_date)

    def filter_articles_by_inauguration(self, articles: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        """
        Filter articles based on inauguration day and collection mode.
        
        Args:
            articles: List of collected articles
            mode: Collection mode ('first_run', 'incremental', 'full')
            
        Returns:
            Filtered articles list
        """
        if mode != 'first_run':
            return articles  # No filtering needed for incremental/full runs
        
        filtered_articles = []
        filtered_count = 0
        
        for article in articles:
            article_date = None
            
            # Try to parse publication date
            if article.get("published"):
                try:
                    article_date = datetime.fromisoformat(article["published"].replace('Z', '+00:00').replace('+00:00', ''))
                except (ValueError, TypeError):
                    # If we can't parse date, include the article to be safe
                    filtered_articles.append(article)
                    continue
            
            # Filter based on inauguration day for first run
            if not article_date or article_date >= self.inauguration_day:
                filtered_articles.append(article)
            else:
                filtered_count += 1
        
        if filtered_count > 0:
            self.logger.info(f"FIRST RUN: Filtered out {filtered_count} articles before inauguration day")
        
        return filtered_articles

    def update_collection_state(self, articles: List[Dict[str, Any]], mode: str):
        """
        Update collection state tracking.
        
        Args:
            articles: Successfully collected articles
            mode: Collection mode used
        """
        try:
            current_time = datetime.now()
            
            # Find most recent article date
            latest_article_date = None
            for article in articles:
                if article.get("published"):
                    try:
                        article_date = datetime.fromisoformat(article["published"].replace('Z', '+00:00').replace('+00:00', ''))
                        if latest_article_date is None or article_date > latest_article_date:
                            latest_article_date = article_date
                    except (ValueError, TypeError):
                        continue
            
            # Update last run time
            # Use latest article date + small buffer, or current time if no articles
            if latest_article_date:
                next_start_time = latest_article_date + timedelta(seconds=1)
            else:
                next_start_time = current_time
            
            with open(self.last_run_file, 'w') as f:
                f.write(next_start_time.isoformat())
            
            # Update collection state
            state = {
                "last_run": current_time.isoformat(),
                "last_mode": mode,
                "articles_collected": len(articles),
                "latest_article_date": latest_article_date.isoformat() if latest_article_date else None,
                "next_start_time": next_start_time.isoformat()
            }
            
            with open(self.collection_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Updated collection state: next run will start from {next_start_time.isoformat()}")
            
        except Exception as e:
            self.logger.error(f"Error updating collection state: {e}")

    def collect_from_direct_article(self, article_url: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Collect content from a direct article URL.
        
        Args:
            article_url: Direct URL to article
            source_config: Source configuration
            
        Returns:
            Article data or None if failed
        """
        try:
            self.logger.info(f"Collecting direct article: {article_url}")
            
            # Extract article content
            content, article_data = self._extract_article_content(article_url)
            
            if not content or not self._is_valid_content(content):
                self.logger.warning(f"Failed to extract valid content from: {article_url}")
                return None
            
            # Check if content is government-related
            title = article_data.get("title", "")
            if not self._is_government_related(title, content[:1000]):
                self.logger.info(f"Article not government-related: {article_url}")
                return None
            
            # Create article entry
            article = {
                "title": title or "Direct Article",
                "url": article_url,
                "source": source_config.get("name", self._extract_source_name(article_url)),
                "bias_label": source_config.get("bias", "unknown"),
                "published": article_data.get("published", datetime.now().isoformat()),
                "content": content,
                "authors": article_data.get("authors", []),
                "top_image": article_data.get("top_image"),
                "collection_method": "direct_article",
                "collected_at": datetime.now().isoformat()
            }
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error collecting direct article {article_url}: {e}")
            return None

    def collect_content(self, force_mode: Optional[str] = None, custom_date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Main collection method with intelligent run behavior.
        
        Args:
            force_mode: Force specific collection mode
            custom_date_range: Override date range (start_date, end_date)
            
        Returns:
            Collection results
        """
        # Determine collection strategy
        if custom_date_range:
            mode = "custom"
            start_date, end_date = custom_date_range
            self.logger.info(f"CUSTOM RUN: Collecting from {start_date} to {end_date}")
        else:
            mode, start_date, end_date = self.get_collection_mode(force_mode)
        
        self.logger.info(f"Starting collection - Mode: {mode}, Range: {start_date} to {end_date}")
        
        # Reset statistics
        self.collection_stats = {
            'total_articles': 0,
            'articles_by_date': {},
            'articles_by_source': {},
            'failed_urls': [],
            'archive_discoveries': 0,
            'pagination_discoveries': 0,
            'sitemap_discoveries': 0,
            'direct_articles': 0
        }
        
        all_articles = []
        document_ids = []
        successful_sources = 0
        failed_sources = 0
        
        for source in self.sources:
            if not source.get("enabled", True):
                self.logger.info(f"Skipping disabled source: {source.get('url')}")
                continue
                
            try:
                source_articles = []
                source_type = source.get("type", "rss").lower()
                
                if source_type == "rss":
                    # Collect from RSS feed
                    source_articles = self._collect_from_rss_with_date_range(source, start_date, end_date)
                    
                elif source_type == "article":
                    # Collect from direct article URL
                    article = self.collect_from_direct_article(source.get("url"), source)
                    if article:
                        source_articles = [article]
                        self.collection_stats['direct_articles'] += 1
                
                else:
                    self.logger.warning(f"Unknown source type: {source_type}")
                    continue
                
                if source_articles:
                    successful_sources += 1
                    self.collection_stats['articles_by_source'][source.get('name', source.get('url'))] = len(source_articles)
                    
                    # Apply inauguration day filtering for first run
                    if mode == 'first_run':
                        source_articles = self.filter_articles_by_inauguration(source_articles, mode)
                    
                    # Generate document IDs and store
                    for article in source_articles:
                        article["id"] = self._generate_document_id(article)
                        
                        # Store in document repository if available
                        if self.document_repository:
                            try:
                                doc_id = self.document_repository.store_document(
                                    article["content"],
                                    self._create_metadata(article)
                                )
                                document_ids.append(doc_id)
                                article["document_id"] = doc_id
                            except Exception as e:
                                self.logger.error(f"Error storing document: {e}")
                        
                        # Update date statistics
                        if article.get("published"):
                            try:
                                pub_date = datetime.fromisoformat(article["published"].replace('Z', '+00:00').replace('+00:00', ''))
                                date_key = pub_date.date().isoformat()
                                self.collection_stats['articles_by_date'][date_key] = \
                                    self.collection_stats['articles_by_date'].get(date_key, 0) + 1
                            except:
                                pass
                    
                    all_articles.extend(source_articles)
                    self.collection_stats['total_articles'] += len(source_articles)
                    
                    # Update source statistics
                    source["collection_stats"]["total_articles"] += len(source_articles)
                    source["collection_stats"]["last_successful_collection"] = datetime.now().isoformat()
                    source["collection_stats"]["consecutive_failures"] = 0
                
                else:
                    failed_sources += 1
                    source["collection_stats"]["consecutive_failures"] += 1
                
                # Add delay between sources
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                self.logger.error(f"Error processing source {source.get('url')}: {e}")
                failed_sources += 1
                source["collection_stats"]["consecutive_failures"] += 1
        
        # Update collection state for next run
        if mode != "custom":  # Don't update state for custom runs
            self.update_collection_state(all_articles, mode)
        
        # Save updated source configurations
        self._save_sources_config()
        
        # Log collection statistics
        self._log_collection_stats()
        
        return {
            "articles": all_articles,
            "document_ids": document_ids,
            "status": {
                "successful_sources": successful_sources,
                "failed_sources": failed_sources,
                "articles_collected": len(all_articles),
                "collection_mode": mode,
                "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
                "timestamp": datetime.now().isoformat(),
                "collection_stats": self.collection_stats
            }
        }

    def _collect_from_rss_with_date_range(self, source: Dict[str, Any], start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Collect from RSS source with specific date range.
        
        Args:
            source: Source configuration
            start_date: Start date for collection
            end_date: End date for collection
            
        Returns:
            List of collected articles
        """
        url = source.get("url", "")
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(url, agent=random.choice(self.user_agents))
            
            if not feed.get("entries"):
                self.logger.warning(f"No entries found in RSS feed: {url}")
                return []
            
            articles = []
            
            for entry in feed.entries:
                try:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "").strip()
                    
                    if not link or not title:
                        continue
                    
                    # Parse publication date
                    pub_date = None
                    if entry.get("published_parsed"):
                        try:
                            pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                            
                            # Skip if outside date range
                            if pub_date < start_date or pub_date > end_date:
                                continue
                        except Exception as e:
                            self.logger.debug(f"Error parsing date for {title}: {e}")
                            # If we can't parse date, include for first run, skip for incremental
                            if start_date != datetime(2025, 1, 1):  # Not first run
                                continue
                    
                    # Check for political content
                    rss_content = entry.get("summary", "")
                    if not self._is_government_related(title, rss_content):
                        continue
                    
                    # Extract full article content
                    content, article_data = self._extract_article_content(link, title, pub_date)
                    
                    if content and self._is_valid_content(content):
                        article = {
                            "title": title,
                            "url": link,
                            "source": source.get("name", urlparse(url).netloc),
                            "bias_label": source.get("bias", "unknown"),
                            "published": pub_date.isoformat() if pub_date else None,
                            "content": content,
                            "authors": article_data.get("authors", []),
                            "top_image": article_data.get("top_image"),
                            "collection_method": "rss",
                            "collected_at": datetime.now().isoformat()
                        }
                        
                        articles.append(article)
                        
                        if len(articles) >= self.article_limit:
                            break
                    
                    # Rate limiting
                    time.sleep(random.uniform(0.5, 1.5))
                    
                except Exception as e:
                    self.logger.debug(f"Error processing RSS entry: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error collecting from RSS {url}: {e}")
            return []
        """
        Collect historical content based on configuration.
        
        Args:
            historical_config: Configuration for historical collection
            
        Returns:
            Dictionary with collection results
        """
        self.logger.info(f"Starting historical collection from {historical_config.start_date} to {historical_config.end_date}")
        
        # Reset statistics
        self.collection_stats = {
            'total_articles': 0,
            'articles_by_date': {},
            'articles_by_source': {},
            'failed_urls': [],
            'archive_discoveries': 0,
            'pagination_discoveries': 0,
            'sitemap_discoveries': 0
        }
        
        all_articles = []
        document_ids = []
        successful_sources = 0
        failed_sources = 0
        
        # Calculate date chunks for efficient collection
        date_chunks = self._calculate_date_chunks(historical_config)
        
        for source in self.sources:
            try:
                self.logger.info(f"Processing source: {source.get('url')}")
                
                source_articles = []
                
                # Strategy 1: RSS feed with date filtering
                rss_articles = self._collect_historical_rss(source, historical_config)
                source_articles.extend(rss_articles)
                
                # Strategy 2: Archive discovery if enabled
                if historical_config.enable_archive_discovery and len(source_articles) < historical_config.target_articles_per_day * len(date_chunks):
                    archive_articles = self._discover_archived_content(source, historical_config)
                    source_articles.extend(archive_articles)
                
                # Strategy 3: Pagination discovery if enabled
                if historical_config.enable_pagination and len(source_articles) < historical_config.target_articles_per_day * len(date_chunks):
                    paginated_articles = self._discover_paginated_content(source, historical_config)
                    source_articles.extend(paginated_articles)
                
                # Strategy 4: Sitemap crawling if enabled
                if historical_config.enable_sitemap_crawling and len(source_articles) < historical_config.target_articles_per_day * len(date_chunks):
                    sitemap_articles = self._discover_sitemap_content(source, historical_config)
                    source_articles.extend(sitemap_articles)
                
                # Deduplicate by URL
                unique_articles = self._deduplicate_articles(source_articles)
                
                # Filter and limit articles
                filtered_articles = self._filter_and_limit_articles(unique_articles, historical_config)
                
                if filtered_articles:
                    successful_sources += 1
                    self.collection_stats['articles_by_source'][source.get('url', 'unknown')] = len(filtered_articles)
                    
                    # Generate document IDs and store
                    for article in filtered_articles:
                        article["id"] = self._generate_document_id(article)
                        
                        # Store in document repository if available
                        if self.document_repository:
                            try:
                                doc_id = self.document_repository.store_document(
                                    article["content"],
                                    self._create_metadata(article)
                                )
                                document_ids.append(doc_id)
                                article["document_id"] = doc_id
                            except Exception as e:
                                self.logger.error(f"Error storing document: {e}")
                    
                    all_articles.extend(filtered_articles)
                    self.collection_stats['total_articles'] += len(filtered_articles)
                else:
                    failed_sources += 1
                
                # Add delay between sources
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                self.logger.error(f"Error processing source {source.get('url')}: {e}")
                failed_sources += 1
        
        # Log collection statistics
        self._log_collection_stats()
        
        return {
            "articles": all_articles,
            "document_ids": document_ids,
            "status": {
                "successful_sources": successful_sources,
                "failed_sources": failed_sources,
                "articles_collected": len(all_articles),
                "timestamp": datetime.now().isoformat(),
                "historical_range": f"{historical_config.start_date.isoformat()} to {historical_config.end_date.isoformat()}",
                "collection_stats": self.collection_stats
            }
        }

    def _calculate_date_chunks(self, config: HistoricalCollectionConfig) -> List[Tuple[datetime, datetime]]:
        """Calculate date chunks for efficient historical collection."""
        chunks = []
        current_date = config.start_date
        chunk_size = timedelta(days=7)  # Weekly chunks
        
        while current_date < config.end_date:
            chunk_end = min(current_date + chunk_size, config.end_date)
            chunks.append((current_date, chunk_end))
            current_date = chunk_end
        
        return chunks

    def _collect_historical_rss(self, source: Dict[str, Any], config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """
        Collect historical content from RSS feeds with enhanced date handling.
        """
        url = source.get("url", "")
        bias = source.get("bias", "unknown")
        
        self.logger.info(f"Collecting historical RSS from: {url}")
        
        articles = []
        
        try:
            # Try original RSS feed
            feed = feedparser.parse(url, agent=random.choice(self.user_agents))
            
            if feed.get("entries"):
                rss_articles = self._process_rss_entries(feed.entries, source, config)
                articles.extend(rss_articles)
            
            # Try RSS feed variations for historical content
            historical_rss_variants = self._generate_historical_rss_variants(url)
            
            for variant_url in historical_rss_variants:
                try:
                    time.sleep(random.uniform(1, 3))
                    variant_feed = feedparser.parse(variant_url, agent=random.choice(self.user_agents))
                    
                    if variant_feed.get("entries"):
                        variant_articles = self._process_rss_entries(variant_feed.entries, source, config)
                        articles.extend(variant_articles)
                        
                        self.logger.info(f"Found {len(variant_articles)} articles from RSS variant: {variant_url}")
                
                except Exception as e:
                    self.logger.debug(f"RSS variant failed {variant_url}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error collecting historical RSS from {url}: {e}")
        
        return articles

    def _generate_historical_rss_variants(self, base_url: str) -> List[str]:
        """Generate potential RSS feed variants that might contain historical content."""
        variants = []
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        # Common RSS archive patterns
        archive_patterns = [
            "/archives/feed/",
            "/archive/rss/",
            "/feed/archive/",
            "/rss/archive/",
            "/feeds/archive/",
            "/archive.xml",
            "/archive/",
            "/archives/",
            "/feed/?year=2025",
            "/feed/?year=2024",
            "/rss.xml?archive=true"
        ]
        
        for pattern in archive_patterns:
            variants.append(domain + pattern)
        
        # Date-based RSS feeds
        for year in [2024, 2025]:
            for month in range(1, 13):
                variants.extend([
                    f"{domain}/feed/?year={year}&month={month:02d}",
                    f"{domain}/rss/?year={year}&month={month:02d}",
                    f"{domain}/{year}/{month:02d}/feed/",
                    f"{domain}/archives/{year}/{month:02d}/feed/"
                ])
        
        return variants[:20]  # Limit to avoid excessive requests

    def _discover_archived_content(self, source: Dict[str, Any], config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """
        Discover historical content using web archive services.
        """
        url = source.get("url", "")
        self.logger.info(f"Discovering archived content for: {url}")
        
        articles = []
        domain = urlparse(url).netloc
        
        # Try Internet Archive Wayback Machine
        try:
            wayback_articles = self._search_wayback_machine(domain, config)
            articles.extend(wayback_articles)
            self.collection_stats['archive_discoveries'] += len(wayback_articles)
        except Exception as e:
            self.logger.debug(f"Wayback Machine search failed: {e}")
        
        # Try Archive.today
        try:
            archive_today_articles = self._search_archive_today(domain, config)
            articles.extend(archive_today_articles)
            self.collection_stats['archive_discoveries'] += len(archive_today_articles)
        except Exception as e:
            self.logger.debug(f"Archive.today search failed: {e}")
        
        return articles

    def _search_wayback_machine(self, domain: str, config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Search Wayback Machine for historical content."""
        articles = []
        
        # Wayback Machine CDX API
        cdx_url = "http://web.archive.org/cdx/search/cdx"
        
        params = {
            'url': f"{domain}/*",
            'from': config.start_date.strftime('%Y%m%d'),
            'to': config.end_date.strftime('%Y%m%d'),
            'output': 'json',
            'fl': 'timestamp,original,mimetype,statuscode',
            'filter': 'statuscode:200',
            'filter': 'mimetype:text/html',
            'collapse': 'urlkey',
            'limit': 1000
        }
        
        try:
            response = self.session.get(cdx_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for row in data[1:]:  # Skip header row
                    timestamp, original_url, mimetype, statuscode = row
                    
                    # Parse timestamp
                    try:
                        archived_date = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
                        
                        # Check if URL looks like a political article
                        if self._is_political_url(original_url):
                            wayback_url = f"http://web.archive.org/web/{timestamp}/{original_url}"
                            
                            # Try to extract content
                            article_content = self._extract_archived_content(wayback_url, original_url)
                            
                            if article_content:
                                articles.append({
                                    "title": article_content.get("title", "Archived Article"),
                                    "url": original_url,
                                    "archived_url": wayback_url,
                                    "source": domain,
                                    "published": archived_date.isoformat(),
                                    "content": article_content.get("content", ""),
                                    "collection_method": "wayback_machine",
                                    "collected_at": datetime.now().isoformat()
                                })
                            
                            time.sleep(random.uniform(1, 2))  # Rate limiting
                            
                            if len(articles) >= 50:  # Limit per source
                                break
                
        except Exception as e:
            self.logger.error(f"Error searching Wayback Machine: {e}")
        
        return articles

    def _search_archive_today(self, domain: str, config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Search Archive.today for historical content."""
        # Archive.today doesn't have a public API, so this would be more limited
        # This is a placeholder for the implementation
        self.logger.info(f"Archive.today search for {domain} (placeholder)")
        return []

    def _discover_paginated_content(self, source: Dict[str, Any], config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """
        Discover historical content through pagination exploration.
        """
        url = source.get("url", "")
        domain = urlparse(url).netloc
        
        self.logger.info(f"Discovering paginated content for: {domain}")
        
        articles = []
        
        # Common pagination patterns to explore
        pagination_patterns = [
            f"https://{domain}/politics/",
            f"https://{domain}/government/",
            f"https://{domain}/news/politics/",
            f"https://{domain}/political/",
            f"https://{domain}/election/",
            f"https://{domain}/congress/",
            f"https://{domain}/white-house/"
        ]
        
        for base_url in pagination_patterns:
            try:
                paginated_articles = self._explore_pagination(base_url, config)
                articles.extend(paginated_articles)
                self.collection_stats['pagination_discoveries'] += len(paginated_articles)
                
                if len(articles) >= 100:  # Limit per source
                    break
                    
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                self.logger.debug(f"Pagination exploration failed for {base_url}: {e}")
                continue
        
        return articles

    def _explore_pagination(self, base_url: str, config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Explore pagination for a given URL."""
        articles = []
        
        # Try different pagination patterns
        pagination_formats = [
            "{base_url}?page={page}",
            "{base_url}/page/{page}/",
            "{base_url}?p={page}",
            "{base_url}/page-{page}/",
            "{base_url}?offset={offset}"
        ]
        
        for format_str in pagination_formats:
            for page in range(1, min(self.max_pages_per_source, 21)):  # Limit pages
                try:
                    if "offset" in format_str:
                        page_url = format_str.format(base_url=base_url, offset=(page-1)*10)
                    else:
                        page_url = format_str.format(base_url=base_url, page=page)
                    
                    response = self.session.get(page_url, timeout=15)
                    
                    if response.status_code == 200:
                        page_articles = self._extract_articles_from_page(response.text, page_url, config)
                        
                        if page_articles:
                            articles.extend(page_articles)
                        else:
                            break  # No more articles found
                    else:
                        break  # Invalid page
                    
                    time.sleep(random.uniform(1, 2))
                    
                except Exception as e:
                    self.logger.debug(f"Pagination page failed {page_url}: {e}")
                    break
            
            if articles:
                break  # Found working pagination format
        
        return articles

    def _discover_sitemap_content(self, source: Dict[str, Any], config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """
        Discover historical content through sitemap exploration.
        """
        url = source.get("url", "")
        domain = urlparse(url).netloc
        
        self.logger.info(f"Discovering sitemap content for: {domain}")
        
        articles = []
        
        # Common sitemap locations
        sitemap_urls = [
            f"https://{domain}/sitemap.xml",
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemaps.xml",
            f"https://{domain}/sitemap/",
            f"https://{domain}/robots.txt"  # May contain sitemap references
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                sitemap_articles = self._parse_sitemap(sitemap_url, config)
                articles.extend(sitemap_articles)
                self.collection_stats['sitemap_discoveries'] += len(sitemap_articles)
                
                if len(articles) >= 100:  # Limit per source
                    break
                    
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                self.logger.debug(f"Sitemap parsing failed for {sitemap_url}: {e}")
                continue
        
        return articles

    def _parse_sitemap(self, sitemap_url: str, config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Parse a sitemap for relevant URLs."""
        articles = []
        
        try:
            response = self.session.get(sitemap_url, timeout=15)
            
            if response.status_code == 200:
                # Handle robots.txt
                if sitemap_url.endswith('robots.txt'):
                    sitemap_refs = re.findall(r'Sitemap:\s*(.+)', response.text, re.IGNORECASE)
                    for sitemap_ref in sitemap_refs:
                        sub_articles = self._parse_sitemap(sitemap_ref.strip(), config)
                        articles.extend(sub_articles)
                    return articles
                
                # Parse XML sitemap
                try:
                    root = ET.fromstring(response.content)
                    
                    # Handle sitemap index
                    if 'sitemapindex' in root.tag:
                        for sitemap in root:
                            loc_elem = sitemap.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc_elem is not None:
                                sub_articles = self._parse_sitemap(loc_elem.text, config)
                                articles.extend(sub_articles[:10])  # Limit sub-sitemaps
                    
                    # Handle regular sitemap
                    else:
                        for url_elem in root:
                            loc_elem = url_elem.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            lastmod_elem = url_elem.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                            
                            if loc_elem is not None:
                                url = loc_elem.text
                                
                                # Check if URL is political and within date range
                                if self._is_political_url(url):
                                    # Check date if available
                                    if lastmod_elem is not None:
                                        try:
                                            lastmod = datetime.fromisoformat(lastmod_elem.text.replace('Z', '+00:00'))
                                            if not (config.start_date <= lastmod <= config.end_date):
                                                continue
                                        except:
                                            pass
                                    
                                    # Extract article content
                                    article_content = self._extract_article_from_url(url)
                                    
                                    if article_content:
                                        articles.append(article_content)
                                    
                                    time.sleep(random.uniform(0.5, 1))
                                    
                                    if len(articles) >= 50:  # Limit per sitemap
                                        break
                
                except ET.ParseError:
                    self.logger.debug(f"Failed to parse XML sitemap: {sitemap_url}")
        
        except Exception as e:
            self.logger.debug(f"Error parsing sitemap {sitemap_url}: {e}")
        
        return articles

    def _process_rss_entries(self, entries: List[Any], source: Dict[str, Any], config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Process RSS entries with historical date filtering."""
        articles = []
        
        for entry in entries:
            try:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                
                if not link or not title:
                    continue
                
                # Parse publication date
                pub_date = None
                if entry.get("published_parsed"):
                    try:
                        pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        
                        # Skip if outside date range
                        if not (config.start_date <= pub_date <= config.end_date):
                            continue
                    except Exception as e:
                        self.logger.debug(f"Error parsing date for {title}: {e}")
                        continue
                
                # Filter for political content
                if not self._is_government_related(title, entry.get("summary", "")):
                    continue
                
                # Extract content
                content, article_data = self._extract_article_content(link, title, pub_date)
                
                if content and self._is_valid_content(content):
                    article = {
                        "title": title,
                        "url": link,
                        "source": source.get("name", urlparse(source.get("url", "")).netloc),
                        "bias_label": source.get("bias", "unknown"),
                        "published": pub_date.isoformat() if pub_date else None,
                        "content": content,
                        "collection_method": "rss",
                        "collected_at": datetime.now().isoformat(),
                        **article_data
                    }
                    articles.append(article)
                    
                    # Update date statistics
                    date_key = pub_date.date().isoformat() if pub_date else "unknown"
                    self.collection_stats['articles_by_date'][date_key] = \
                        self.collection_stats['articles_by_date'].get(date_key, 0) + 1
                
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                self.logger.debug(f"Error processing RSS entry: {e}")
                continue
        
        return articles

    def _is_political_url(self, url: str) -> bool:
        """Check if a URL appears to be political content."""
        url_lower = url.lower()
        
        political_indicators = [
            '/politics/', '/political/', '/government/', '/election/', '/congress/',
            '/senate/', '/white-house/', '/president/', '/campaign/', '/policy/',
            '/federal/', '/supreme-court/', '/judiciary/', '/legislation/',
            'trump', 'biden', 'harris', 'election', 'congress', 'senate'
        ]
        
        return any(indicator in url_lower for indicator in political_indicators)

    def _is_government_related(self, title: str, content: str) -> bool:
        """Check if content is government/politics related."""
        sample_text = (title + " " + content[:500]).lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in self.govt_keywords if keyword in sample_text)
        
        return keyword_matches >= 1

    def _extract_article_content(self, url: str, title: str = None, pub_date: Optional[datetime] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract article content using multiple methods."""
        article_data = {"authors": [], "published": None, "top_image": None}
        
        try:
            # Try trafilatura first
            if TRAFILATURA_AVAILABLE:
                if self.bypass_cloudflare:
                    response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                else:
                    response = self.session.get(url, timeout=self.request_timeout)
                
                content = trafilatura.extract(response.text, favor_precision=True, target_language='en')
                
                if self._is_valid_content(content):
                    if pub_date:
                        article_data["published"] = pub_date.isoformat()
                    return content, article_data
            
            # Fall back to newspaper3k
            if NEWSPAPER_AVAILABLE:
                config = Config()
                config.browser_user_agent = random.choice(self.user_agents)
                config.request_timeout = self.request_timeout
                
                article = Article(url, config=config)
                article.download()
                article.parse()
                
                content = article.text
                
                if self._is_valid_content(content):
                    article_data.update({
                        "authors": article.authors,
                        "published": article.publish_date.isoformat() if article.publish_date else (pub_date.isoformat() if pub_date else None),
                        "top_image": article.top_image
                    })
                    return content, article_data
        
        except Exception as e:
            self.logger.debug(f"Content extraction failed for {url}: {e}")
        
        return None, article_data

    def _extract_archived_content(self, wayback_url: str, original_url: str) -> Optional[Dict[str, str]]:
        """Extract content from archived URL."""
        try:
            response = self.session.get(wayback_url, timeout=30)
            
            if response.status_code == 200:
                if TRAFILATURA_AVAILABLE:
                    content = trafilatura.extract(response.text, favor_precision=True)
                    
                    if self._is_valid_content(content):
                        # Extract title from HTML
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title_elem = soup.find('title')
                        title = title_elem.get_text().strip() if title_elem else "Archived Article"
                        
                        return {"title": title, "content": content}
        
        except Exception as e:
            self.logger.debug(f"Archived content extraction failed for {wayback_url}: {e}")
        
        return None

    def _extract_articles_from_page(self, html: str, page_url: str, config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Extract article links from a page."""
        articles = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for article links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Make URL absolute
                if href.startswith('/'):
                    full_url = urljoin(page_url, href)
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                # Check if it looks like a political article
                if self._is_political_url(full_url) and text and len(text) > 20:
                    article_content = self._extract_article_from_url(full_url)
                    
                    if article_content:
                        articles.append(article_content)
                    
                    if len(articles) >= 10:  # Limit per page
                        break
        
        except Exception as e:
            self.logger.debug(f"Error extracting articles from page: {e}")
        
        return articles

    def _extract_article_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract a single article from URL."""
        try:
            content, article_data = self._extract_article_content(url)
            
            if content and self._is_valid_content(content):
                return {
                    "title": article_data.get("title", "Article"),
                    "url": url,
                    "source": urlparse(url).netloc,
                    "published": article_data.get("published"),
                    "content": content,
                    "collection_method": "discovery",
                    "collected_at": datetime.now().isoformat(),
                    **article_data
                }
        
        except Exception as e:
            self.logger.debug(f"Article extraction failed for {url}: {e}")
        
        return None

    def _is_valid_content(self, content: Optional[str]) -> bool:
        """Check if content meets quality criteria."""
        if not content:
            return False
        
        content = re.sub(r'\s+', ' ', content.strip())
        
        return (
            len(content) >= 200 and
            len(content.split()) >= 50 and
            not re.search(r'(.)\1{10,}', content)
        )

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles by URL."""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles

    def _filter_and_limit_articles(self, articles: List[Dict[str, Any]], config: HistoricalCollectionConfig) -> List[Dict[str, Any]]:
        """Filter articles and apply limits."""
        # Sort by publication date (newest first)
        articles.sort(key=lambda x: x.get("published", ""), reverse=True)
        
        # Apply total limit
        if len(articles) > config.max_articles_total:
            articles = articles[:config.max_articles_total]
        
        return articles

    def _generate_document_id(self, article: Dict[str, Any]) -> str:
        """Generate document ID."""
        url = article.get("url", "")
        timestamp = int(time.time() * 1000)
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
        return f"{url_hash}_{timestamp}"

    def _create_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for document storage."""
        return {
            "title": article.get("title", "Untitled"),
            "url": article.get("url", ""),
            "source": article.get("source", "Unknown"),
            "bias_label": article.get("bias_label", "unknown"),
            "published": article.get("published"),
            "authors": article.get("authors", []),
            "collection_method": article.get("collection_method", "unknown"),
            "collected_at": article.get("collected_at", datetime.now().isoformat()),
            "id": article.get("id")
        }

    def _log_collection_stats(self):
        """Log collection statistics."""
        stats = self.collection_stats
        
        self.logger.info(f"=== Historical Collection Statistics ===")
        self.logger.info(f"Total articles collected: {stats['total_articles']}")
        self.logger.info(f"Archive discoveries: {stats['archive_discoveries']}")
        self.logger.info(f"Pagination discoveries: {stats['pagination_discoveries']}")
        self.logger.info(f"Sitemap discoveries: {stats['sitemap_discoveries']}")
        
        if stats['articles_by_date']:
            self.logger.info(f"Articles by date: {dict(list(stats['articles_by_date'].items())[:5])}")
        
        if stats['articles_by_source']:
            self.logger.info(f"Articles by source: {stats['articles_by_source']}")


def create_historical_collection_config(
    start_date: str,
    end_date: str,
    target_per_day: int = 10,
    max_total: int = 1000
) -> HistoricalCollectionConfig:
    """
    Helper function to create historical collection configuration.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        target_per_day: Target articles per day
        max_total: Maximum total articles
        
    Returns:
        HistoricalCollectionConfig instance
    """
    return HistoricalCollectionConfig(
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        target_articles_per_day=target_per_day,
        max_articles_total=max_total,
        enable_archive_discovery=True,
        enable_pagination=True,
        enable_sitemap_crawling=True,
        max_pages_per_source=20
    )


# Example usage function
# Example usage functions
def create_historical_collection_config(
    start_date: str,
    end_date: str,
    target_per_day: int = 10,
    max_total: int = 1000
) -> HistoricalCollectionConfig:
    """
    Helper function to create historical collection configuration.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        target_per_day: Target articles per day
        max_total: Maximum total articles
        
    Returns:
        HistoricalCollectionConfig instance
    """
    return HistoricalCollectionConfig(
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        target_articles_per_day=target_per_day,
        max_articles_total=max_total,
        enable_archive_discovery=True,
        enable_pagination=True,
        enable_sitemap_crawling=True,
        max_pages_per_source=20
    )


def run_night_watcher_collection(config_path: str = "config.json", 
                                force_mode: Optional[str] = None,
                                reset_date: bool = False) -> Dict[str, Any]:
    """
    Main entry point for Night_watcher content collection with intelligent run behavior.
    
    Args:
        config_path: Path to configuration file
        force_mode: Force specific mode ('first_run', 'incremental', 'full')
        reset_date: Reset to first run behavior
        
    Returns:
        Collection results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create collector
    collector = HistoricalContentCollector(config)
    
    # Handle reset date
    if reset_date:
        if os.path.exists(collector.last_run_file):
            os.remove(collector.last_run_file)
            collector.logger.info("Reset date tracking - will run as first run")
        force_mode = "first_run"
    
    # Run collection
    results = collector.collect_content(force_mode=force_mode)
    
    return results


def add_source_to_config(config_path: str, source_data: Dict[str, Any]) -> bool:
    """
    Add a new source to the configuration file.
    
    Args:
        config_path: Path to configuration file
        source_data: Source configuration data
        
    Returns:
        True if successful
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create collector to use add_source method
        collector = HistoricalContentCollector(config)
        collector.config_file = config_path  # Track config file for saving
        
        # Add source
        return collector.add_source(source_data)
        
    except Exception as e:
        logging.error(f"Error adding source to config: {e}")
        return False


def collect_inauguration_era_content(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Collect historical content from around Inauguration Day 2025.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Collection results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create historical collector
    collector = HistoricalContentCollector(config)
    
    # Create historical collection config for inauguration era
    historical_config = create_historical_collection_config(
        start_date="2025-01-15",  # Week before inauguration
        end_date="2025-02-15",    # Month after inauguration
        target_per_day=15,
        max_total=500
    )
    
    # Collect historical content
    results = collector.collect_historical_content(historical_config)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Night_watcher Content Collector")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--mode", choices=["auto", "first_run", "incremental", "full"], 
                       default="auto", help="Collection mode")
    parser.add_argument("--reset-date", action="store_true", help="Reset date tracking")
    parser.add_argument("--add-source", help="Add source: 'url,type,bias,name'")
    parser.add_argument("--add-article", help="Add direct article URL")
    parser.add_argument("--historical", action="store_true", help="Run historical collection")
    parser.add_argument("--start-date", help="Start date for historical (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for historical (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Handle adding sources
        if args.add_source:
            parts = args.add_source.split(',')
            if len(parts) >= 2:
                source_data = {
                    "url": parts[0].strip(),
                    "type": parts[1].strip() if len(parts) > 1 else "auto",
                    "bias": parts[2].strip() if len(parts) > 2 else "unknown",
                    "name": parts[3].strip() if len(parts) > 3 else None
                }
                
                if add_source_to_config(args.config, source_data):
                    print(f"✓ Added source: {source_data['url']}")
                else:
                    print(f"✗ Failed to add source: {source_data['url']}")
                    sys.exit(1)
            else:
                print("Error: Source format should be 'url,type,bias,name'")
                sys.exit(1)
        
        # Handle adding article
        elif args.add_article:
            source_data = {
                "url": args.add_article,
                "type": "article",
                "bias": "unknown"
            }
            
            if add_source_to_config(args.config, source_data):
                print(f"✓ Added article: {args.add_article}")
            else:
                print(f"✗ Failed to add article: {args.add_article}")
                sys.exit(1)
        
        # Handle historical collection
        elif args.historical:
            if not args.start_date or not args.end_date:
                print("Error: Historical collection requires --start-date and --end-date")
                sys.exit(1)
            
            # Load configuration
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Create collector
            collector = HistoricalContentCollector(config)
            
            # Create historical config
            historical_config = create_historical_collection_config(
                start_date=args.start_date,
                end_date=args.end_date,
                target_per_day=15,
                max_total=1000
            )
            
            # Collect content
            results = collector.collect_historical_content(historical_config)
            
            # Print results
            print(f"\n=== Historical Collection Complete ===")
            print(f"Articles collected: {len(results['articles'])}")
            print(f"Date range: {args.start_date} to {args.end_date}")
            print(f"Successful sources: {results['status']['successful_sources']}")
            print(f"Failed sources: {results['status']['failed_sources']}")
        
        # Regular collection
        else:
            force_mode = None if args.mode == "auto" else args.mode
            results = run_night_watcher_collection(
                config_path=args.config,
                force_mode=force_mode,
                reset_date=args.reset_date
            )
            
            # Print results
            print(f"\n=== Collection Complete ===")
            print(f"Mode: {results['status']['collection_mode']}")
            print(f"Articles collected: {len(results['articles'])}")
            print(f"Date range: {results['status']['date_range']}")
            print(f"Successful sources: {results['status']['successful_sources']}")
            print(f"Failed sources: {results['status']['failed_sources']}")
            
            # Show collection stats
            stats = results['status']['collection_stats']
            if stats.get('direct_articles', 0) > 0:
                print(f"Direct articles: {stats['direct_articles']}")
            
            if stats.get('articles_by_source'):
                print("\nArticles by source:")
                for source, count in stats['articles_by_source'].items():
                    print(f"  - {source}: {count}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
