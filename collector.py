#!/usr/bin/env python3
"""
Night_watcher Unified Content Collector
Combines all collection functionality into a single, efficient module.
"""

import os
import time
import logging
import hashlib
import re
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import traceback

import requests
import feedparser
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional imports
try:
    from newspaper import Article, Config
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import for document repository if needed
try:
    from document_repository import DocumentRepository
except ImportError:
    DocumentRepository = None


class ContentCollector:
    """Unified collector for all Night_watcher content gathering."""

    def __init__(self, config: Dict[str, Any], document_repository=None, base_dir: str = "data"):
        self.config = config
        self.document_repository = document_repository
        self.base_dir = base_dir
        
        # Configuration
        cc = config.get("content_collection", {})
        self.article_limit = cc.get("article_limit", 50)
        self.sources = cc.get("sources", [])
        self.request_timeout = cc.get("request_timeout", 45)
        self.delay_between_requests = cc.get("delay_between_requests", 2.0)

        self.cancelled = False
        
        # State files
        self.last_run_file = os.path.join(base_dir, "last_run_date.txt")
        self.inauguration_day = datetime(2025, 1, 20)
        
        # Keywords
        self.govt_keywords = set(kw.lower() for kw in cc.get("govt_keywords", [
            "executive order", "administration", "white house", "president",
            "congress", "senate", "supreme court", "federal", "government", 
            "politics", "election", "democracy", "biden", "trump"
        ]))
        
        # User agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        ]
        
        self.logger = logging.getLogger("ContentCollector")
        self._init_session()

    def _init_session(self):
        """Initialize HTTP session."""
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._rotate_headers()
        
        if CLOUDSCRAPER_AVAILABLE:
            try:
                self.cloudscraper = cloudscraper.create_scraper()
            except:
                self.cloudscraper = None

    def cancel(self):
        """Signal to stop collection."""
        self.cancelled = True

    def _rotate_headers(self):
        """Rotate session headers."""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1'
        })

    def collect_content(self, force_mode: Optional[str] = None, callback=None) -> Dict[str, Any]:
        """Main collection method with intelligent date handling.

        Args:
            force_mode: Optional mode override.
            callback: Optional function to receive progress events.
        """
        self.cancelled = False
        # Determine collection mode
        mode, start_date, end_date = self._get_collection_mode(force_mode)
        self.logger.info(f"Collection mode: {mode}, Range: {start_date} to {end_date}")
        
        all_articles = []
        document_ids = []
        
        for source in self.sources:
            if not source.get("enabled", True):
                continue

            try:
                if self.cancelled:
                    break

                limit = source.get("limit", self.article_limit)
                articles = self._collect_from_source(source, start_date, end_date, limit, callback)
                
                # Filter by inauguration day if first run
                if mode == "first_run":
                    articles = [a for a in articles if self._after_inauguration(a)]
                
                # Store documents
                for article in articles:
                    article["id"] = self._generate_id(article["url"])
                    
                    if self.document_repository:
                        doc_id = self.document_repository.store_document(
                            article["content"],
                            self._create_metadata(article)
                        )
                        document_ids.append(doc_id)
                        article["document_id"] = doc_id
                
                all_articles.extend(articles)
                time.sleep(random.uniform(1, 3))
                if self.cancelled:
                    break
                
            except Exception as e:
                self.logger.error(f"Error processing {source.get('url')}: {e}")
        
        # Update state
        if mode != "custom":
            self._update_last_run()

        if self.cancelled and callback:
            callback({"type": "cancelled"})
        
        return {
            "articles": all_articles,
            "document_ids": document_ids,
            "status": {
                "articles_collected": len(all_articles),
                "collection_mode": mode,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _get_collection_mode(self, force_mode: Optional[str]) -> Tuple[str, datetime, datetime]:
        """Determine collection mode and date range."""
        current_time = datetime.now()
        
        if force_mode == "full":
            return ("full", self.inauguration_day, current_time)
        
        # Check if first run
        if not os.path.exists(self.last_run_file):
            # Wide net for RSS feeds
            return ("first_run", datetime(2025, 1, 1), current_time)
        
        # Incremental
        try:
            with open(self.last_run_file, 'r') as f:
                last_run = datetime.fromisoformat(f.read().strip())
                return ("incremental", last_run - timedelta(hours=1), current_time)
        except:
            return ("incremental", current_time - timedelta(days=1), current_time)

    def _collect_from_source(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect from a single source."""
        url = source.get("url", "")

        if source.get("type") == "article":
            # Direct article
            article = self._extract_article(url)
            if article:
                domain = source.get("site_domain") or urlparse(article["url"]).netloc or urlparse(url).netloc
                archive_url = self._query_wayback(domain)
                if archive_url:
                    article["archive_url"] = archive_url
                if callback:
                    callback({"type": "article", "source": source.get("name", url), "title": article.get("title")})
            return [article] if article else []
        
        # RSS feed
        try:
            feed = feedparser.parse(url, agent=random.choice(self.user_agents))
            articles = []
            
            for entry in feed.entries[:limit]:
                if self.cancelled:
                    break

                if not entry.get("link"):
                    continue
                
                # Date filtering
                pub_date = self._parse_date(entry)
                if pub_date and (pub_date < start_date or pub_date > end_date):
                    continue
                
                # Check if political
                if not self._is_political(entry.get("title", ""), entry.get("summary", "")):
                    continue
                
                # Extract content
                article = self._extract_article(entry.link, entry.get("title"), pub_date)
                if article:
                    article["source"] = source.get("name", urlparse(url).netloc)
                    article["bias_label"] = source.get("bias", "unknown")
                    domain = source.get("site_domain") or urlparse(entry.link).netloc or urlparse(url).netloc
                    archive_url = self._query_wayback(domain)
                    if archive_url:
                        article["archive_url"] = archive_url
                    articles.append(article)
                    if callback:
                        callback({"type": "article", "source": article.get("source"), "title": article.get("title")})
                    
                time.sleep(random.uniform(0.5, 1.5))
            
            return articles
            
        except Exception as e:
            self.logger.error(f"RSS collection failed for {url}: {e}")
            if callback:
                callback({"type": "error", "source": source.get("name", url), "message": str(e)})
            return []

    def _extract_article(self, url: str, title: str = None, pub_date: datetime = None) -> Optional[Dict[str, Any]]:
        """Extract article content."""
        try:
            # Try trafilatura first
            if TRAFILATURA_AVAILABLE:
                session = self.cloudscraper if hasattr(self, 'cloudscraper') else self.session
                response = session.get(url, timeout=self.request_timeout)
                content = trafilatura.extract(response.text, favor_precision=True)
                
                if content and len(content) > 200:
                    return {
                        "title": title or self._extract_title(response.text),
                        "url": url,
                        "content": content,
                        "published": pub_date.isoformat() if pub_date else None,
                        "collected_at": datetime.now().isoformat()
                    }
            
            # Fallback to newspaper3k
            if NEWSPAPER_AVAILABLE:
                config = Config()
                config.browser_user_agent = random.choice(self.user_agents)
                
                article = Article(url, config=config)
                article.download()
                article.parse()
                
                if article.text and len(article.text) > 200:
                    return {
                        "title": title or article.title,
                        "url": url,
                        "content": article.text,
                        "published": (article.publish_date or pub_date).isoformat() if (article.publish_date or pub_date) else None,
                        "collected_at": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.debug(f"Extraction failed for {url}: {e}")
        
        return None

    def _is_political(self, title: str, content: str) -> bool:
        """Check if content is political."""
        text = f"{title} {content}".lower()
        return sum(1 for kw in self.govt_keywords if kw in text) >= 2

    def _after_inauguration(self, article: Dict[str, Any]) -> bool:
        """Check if article is after inauguration day."""
        if not article.get("published"):
            return True  # Include if no date
        
        try:
            pub_date = datetime.fromisoformat(article["published"].replace('Z', '+00:00'))
            return pub_date >= self.inauguration_day
        except:
            return True

    def _parse_date(self, entry: Any) -> Optional[datetime]:
        """Parse date from RSS entry."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime.fromtimestamp(time.mktime(entry.published_parsed))
            except:
                pass
        return None

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            return title.get_text().strip() if title else "Untitled"
        except:
            return "Untitled"

    def _generate_id(self, url: str) -> str:
        """Generate document ID."""
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
            "collected_at": article.get("collected_at"),
            "id": article.get("id")
        }

    def _update_last_run(self):
        """Update last run timestamp."""
        try:
            with open(self.last_run_file, 'w') as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            self.logger.error(f"Error updating last run: {e}")

    def add_source(self, source_data: Dict[str, Any]) -> bool:
        """Add a new source."""
        try:
            # Auto-detect type
            if source_data.get("type") == "auto":
                source_data["type"] = "article" if not any(
                    ind in source_data["url"].lower() 
                    for ind in ['/rss', '/feed', '.xml']
                ) else "rss"
            
            # Add to sources
            self.sources.append({
                "url": source_data["url"],
                "type": source_data.get("type", "rss"),
                "bias": source_data.get("bias", "unknown"),
                "name": source_data.get("name", urlparse(source_data["url"]).netloc),
                "enabled": True
            })
            
            # Save config
            self.config["content_collection"]["sources"] = self.sources
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return False

    def _query_wayback(self, domain: str) -> Optional[str]:
        """Return archive snapshot URL for the given domain if available."""
        try:
            resp = self.session.get(
                "https://archive.org/wayback/available",
                params={"url": domain},
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                closest = data.get("archived_snapshots", {}).get("closest")
                if closest and closest.get("available"):
                    return closest.get("url")
        except Exception as e:
            self.logger.debug(f"Wayback query failed for {domain}: {e}")
        return None
