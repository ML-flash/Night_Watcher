#!/usr/bin/env python3
"""
Night_watcher Unified Content Collector
Enhanced with automatic historical gap detection and collection.
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
from urllib.parse import urlparse, urljoin, quote
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
    """Enhanced collector with automatic historical gap detection."""

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
        self.max_gap_days = cc.get("max_gap_days", 30)  # Max days to look back for gaps
        self.gap_detection_enabled = cc.get("gap_detection_enabled", True)

        self.cancelled = False
        
        # State files
        self.last_run_file = os.path.join(base_dir, "last_run_date.txt")
        self.collection_history_file = os.path.join(base_dir, "collection_history.json")
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
        """Main collection method with automatic gap detection."""
        self.cancelled = False
        
        # Check for gaps if enabled and not forcing a specific mode
        if self.gap_detection_enabled and not force_mode:
            gaps = self._detect_collection_gaps()
            if gaps:
                self.logger.info(f"Detected {len(gaps)} collection gaps")
                return self._collect_with_gap_filling(gaps, callback)
        
        # Normal collection
        mode, start_date, end_date = self._get_collection_mode(force_mode)
        self.logger.info(f"Collection mode: {mode}, Range: {start_date} to {end_date}")
        
        return self._collect_date_range(start_date, end_date, mode, callback)

    def _detect_collection_gaps(self) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in collection history."""
        gaps = []
        
        # Load collection history
        history = self._load_collection_history()
        if not history:
            # No history, consider everything since inauguration as a gap
            if datetime.now() > self.inauguration_day:
                gaps.append((self.inauguration_day, datetime.now()))
            return gaps
        
        # Sort history by date
        sorted_dates = sorted(history.keys())
        
        # Check for gaps between collected dates
        for i in range(len(sorted_dates) - 1):
            current_date = datetime.fromisoformat(sorted_dates[i])
            next_date = datetime.fromisoformat(sorted_dates[i + 1])
            
            # If more than 1 day gap
            if (next_date - current_date).days > 1:
                gap_start = current_date + timedelta(days=1)
                gap_end = next_date - timedelta(days=1)
                
                # Limit gap size to max_gap_days
                if (gap_end - gap_start).days > self.max_gap_days:
                    gap_start = gap_end - timedelta(days=self.max_gap_days)
                
                gaps.append((gap_start, gap_end))
        
        # Check for gap from last collection to now
        if sorted_dates:
            last_date = datetime.fromisoformat(sorted_dates[-1])
            if (datetime.now() - last_date).days > 1:
                gap_start = last_date + timedelta(days=1)
                gap_end = datetime.now()
                
                # Limit gap size
                if (gap_end - gap_start).days > self.max_gap_days:
                    gap_start = gap_end - timedelta(days=self.max_gap_days)
                
                gaps.append((gap_start, gap_end))
        
        return gaps

    def _collect_with_gap_filling(self, gaps: List[Tuple[datetime, datetime]], callback=None) -> Dict[str, Any]:
        """Collect content focusing on filling gaps."""
        all_articles = []
        document_ids = []
        gap_stats = []
        
        for gap_start, gap_end in gaps:
            if self.cancelled:
                break
                
            self.logger.info(f"Filling gap: {gap_start.date()} to {gap_end.date()}")
            
            if callback:
                callback({"type": "gap_fill", "start": gap_start.isoformat(), "end": gap_end.isoformat()})
            
            # Use archive sources for older gaps
            is_old_gap = (datetime.now() - gap_end).days > 7
            if is_old_gap:
                # Prioritize archive and sitemap sources for older content
                result = self._collect_historical_range(gap_start, gap_end, callback)
            else:
                # Use regular collection for recent gaps
                result = self._collect_date_range(gap_start, gap_end, "gap_fill", callback)
            
            articles = result.get("articles", [])
            doc_ids = result.get("document_ids", [])
            
            all_articles.extend(articles)
            document_ids.extend(doc_ids)
            
            gap_stats.append({
                "gap": f"{gap_start.date()} to {gap_end.date()}",
                "articles_collected": len(articles),
                "is_historical": is_old_gap
            })
            
            # Small delay between gaps
            if gaps.index((gap_start, gap_end)) < len(gaps) - 1:
                time.sleep(2)
        
        # Update collection history
        self._update_collection_history(all_articles)
        
        return {
            "articles": all_articles,
            "document_ids": document_ids,
            "status": {
                "articles_collected": len(all_articles),
                "collection_mode": "gap_fill",
                "gaps_filled": gap_stats,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _collect_historical_range(self, start_date: datetime, end_date: datetime, callback=None) -> Dict[str, Any]:
        """Collect historical content using archive and sitemap sources."""
        # Get archive and sitemap sources
        historical_sources = [s for s in self.sources 
                            if s.get("type") in ["archive", "sitemap"] and s.get("enabled", True)]
        
        if not historical_sources:
            # Fallback to regular collection
            self.logger.warning("No historical sources available, using regular sources")
            return self._collect_date_range(start_date, end_date, "historical", callback)
        
        all_articles = []
        document_ids = []
        
        for source in historical_sources:
            if self.cancelled:
                break
                
            try:
                limit = source.get("limit", self.article_limit)
                articles = self._collect_from_source(source, start_date, end_date, limit, callback)
                
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
                
            except Exception as e:
                self.logger.error(f"Error with historical source {source.get('name')}: {e}")
        
        return {
            "articles": all_articles,
            "document_ids": document_ids
        }

    def _collect_date_range(self, start_date: datetime, end_date: datetime, mode: str, callback=None) -> Dict[str, Any]:
        """Collect content for a specific date range."""
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
        
        # Update collection history
        self._update_collection_history(all_articles)

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

    def _load_collection_history(self) -> Dict[str, int]:
        """Load collection history from file."""
        if not os.path.exists(self.collection_history_file):
            return {}
        
        try:
            with open(self.collection_history_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _update_collection_history(self, articles: List[Dict[str, Any]]):
        """Update collection history with newly collected articles."""
        history = self._load_collection_history()
        
        # Count articles by date
        for article in articles:
            pub_date = article.get("published")
            if pub_date:
                try:
                    date_str = datetime.fromisoformat(pub_date.replace('Z', '+00:00')).date().isoformat()
                    history[date_str] = history.get(date_str, 0) + 1
                except:
                    pass
        
        # Save updated history
        try:
            with open(self.collection_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving collection history: {e}")

    def _get_collection_mode(self, force_mode: Optional[str]) -> Tuple[str, datetime, datetime]:
        """Determine collection mode and date range."""
        current_time = datetime.now()
        
        if force_mode == "full":
            return ("full", self.inauguration_day, current_time)
        
        # Check if first run
        if not os.path.exists(self.last_run_file):
            # First run - check if we should do historical collection
            days_since_inauguration = (current_time - self.inauguration_day).days
            
            if days_since_inauguration > 7:
                # Been more than a week, do smart historical collection
                # Start with last 7 days for RSS, will use gap detection for the rest
                return ("first_run", current_time - timedelta(days=7), current_time)
            else:
                # Recent inauguration, collect everything
                return ("first_run", self.inauguration_day, current_time)
        
        # Incremental
        try:
            with open(self.last_run_file, 'r') as f:
                last_run = datetime.fromisoformat(f.read().strip())
                
                # Add small overlap to avoid missing articles
                return ("incremental", last_run - timedelta(hours=1), current_time)
        except:
            return ("incremental", current_time - timedelta(days=1), current_time)

    def _collect_from_source(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect from a single source with enhanced historical support."""
        url = source.get("url", "")
        source_type = source.get("type", "rss")

        if source_type == "article":
            # Direct article
            article = self._extract_article(url)
            if article:
                if callback:
                    callback({"type": "article", "source": source.get("name", url), "title": article.get("title")})
            return [article] if article else []
        
        elif source_type == "archive":
            # Use web archive for historical data
            return self._collect_from_archive(source, start_date, end_date, limit, callback)
        
        elif source_type == "sitemap":
            # Parse sitemap for article URLs
            return self._collect_from_sitemap(source, start_date, end_date, limit, callback)
        
        else:  # RSS feed (default)
            try:
                # First try regular RSS
                feed = feedparser.parse(url, agent=random.choice(self.user_agents))
                articles = self._process_feed_entries(feed.entries, source, url, start_date, end_date, limit, callback)

                # If we need more historical data and got fewer articles than limit
                if len(articles) < limit and (datetime.now() - start_date).days > 7:
                    # Try to get archived RSS snapshots
                    archived = self._collect_archived_feed(source, start_date, end_date, limit - len(articles), callback)
                    seen = {a["url"] for a in articles}
                    for art in archived:
                        if art["url"] not in seen:
                            articles.append(art)
                            seen.add(art["url"])
                            if len(articles) >= limit:
                                break

                return articles
                
            except Exception as e:
                self.logger.error(f"RSS collection failed for {url}: {e}")
                if callback:
                    callback({"type": "error", "source": source.get("name", url), "message": str(e)})
                return []

    def _collect_from_archive(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect articles from web archives (Wayback Machine)."""
        base_url = source.get("archive_url", source.get("url"))
        site_domain = urlparse(base_url).netloc
        
        articles = []
        
        # Query Wayback Machine CDX API
        cdx_url = "https://web.archive.org/cdx/search/cdx"
        params = {
            "url": f"{site_domain}/*",
            "output": "json",
            "from": start_date.strftime('%Y%m%d'),
            "to": end_date.strftime('%Y%m%d'),
            "filter": "statuscode:200",
            "collapse": "urlkey",
            "limit": limit * 2  # Get more URLs to filter
        }
        
        try:
            response = self.session.get(cdx_url, params=params, timeout=self.request_timeout)
            if response.status_code != 200:
                return articles
            
            data = response.json()
            if len(data) <= 1:  # First row is headers
                return articles
            
            # Process CDX results
            for row in data[1:]:  # Skip header row
                if self.cancelled or len(articles) >= limit:
                    break
                
                timestamp = row[1]
                original_url = row[2]
                
                # Filter for article-like URLs
                if not self._is_article_url(original_url):
                    continue
                
                # Build Wayback URL
                wayback_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
                
                # Extract article
                article = self._extract_article(wayback_url)
                if article and self._is_political(article.get("title", ""), article.get("content", "")):
                    article["source"] = source.get("name", site_domain)
                    article["bias_label"] = source.get("bias", "unknown")
                    article["archive_url"] = wayback_url
                    article["original_url"] = original_url
                    articles.append(article)
                    
                    if callback:
                        callback({"type": "article", "source": article["source"], "title": article.get("title")})
                
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            self.logger.error(f"Archive collection failed for {site_domain}: {e}")
            if callback:
                callback({"type": "error", "source": source.get("name"), "message": str(e)})
        
        return articles

    def _collect_from_sitemap(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect articles from XML sitemaps."""
        sitemap_url = source.get("sitemap_url", source.get("url"))
        articles = []
        
        try:
            response = self.session.get(sitemap_url, timeout=self.request_timeout)
            if response.status_code != 200:
                return articles
            
            soup = BeautifulSoup(response.content, 'xml')
            urls = soup.find_all('url')
            
            # Sort by lastmod date if available
            url_data = []
            for url in urls:
                loc = url.find('loc')
                lastmod = url.find('lastmod')
                
                if loc:
                    url_str = loc.text
                    mod_date = None
                    
                    if lastmod:
                        try:
                            mod_date = datetime.fromisoformat(lastmod.text.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    url_data.append((url_str, mod_date))
            
            # Sort by date (newest first)
            url_data.sort(key=lambda x: x[1] or datetime.min, reverse=True)
            
            # Process URLs
            for url_str, mod_date in url_data:
                if self.cancelled or len(articles) >= limit:
                    break
                
                # Check date range
                if mod_date:
                    if mod_date < start_date or mod_date > end_date:
                        continue
                
                # Check if it's an article URL
                if not self._is_article_url(url_str):
                    continue
                
                # Extract article
                article = self._extract_article(url_str)
                if article and self._is_political(article.get("title", ""), article.get("content", "")):
                    article["source"] = source.get("name", urlparse(sitemap_url).netloc)
                    article["bias_label"] = source.get("bias", "unknown")
                    
                    # Use lastmod date if no publish date found
                    if mod_date and not article.get("published"):
                        article["published"] = mod_date.isoformat()
                    
                    articles.append(article)
                    
                    if callback:
                        callback({"type": "article", "source": article["source"], "title": article.get("title")})
                
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            self.logger.error(f"Sitemap collection failed for {sitemap_url}: {e}")
            if callback:
                callback({"type": "error", "source": source.get("name"), "message": str(e)})
        
        return articles

    def _is_article_url(self, url: str) -> bool:
        """Check if URL likely points to an article."""
        # Skip common non-article patterns
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/page/', '/search/',
            '/about', '/contact', '/privacy', '/terms', '/subscribe',
            '.pdf', '.jpg', '.png', '.gif', '/feed/', '/rss'
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False
        
        # Look for article-like patterns
        article_patterns = [
            r'/\d{4}/\d{2}/',  # Date in URL
            r'/article/',
            r'/story/',
            r'/news/',
            r'/\d{4}/\d{2}/\d{2}/',  # Full date
            r'-\d{5,}',  # Article ID
        ]
        
        for pattern in article_patterns:
            if re.search(pattern, url):
                return True
        
        # Default to True if unsure
        return True

    def _process_feed_entries(self, entries: List[Any], source: Dict[str, Any], base_url: str, start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Process RSS feed entries into article records."""
        articles = []
        for entry in entries:
            if self.cancelled or len(articles) >= limit:
                break

            if not entry.get("link"):
                continue

            pub_date = self._parse_date(entry)
            if pub_date and (pub_date < start_date or pub_date > end_date):
                continue

            if not self._is_political(entry.get("title", ""), entry.get("summary", "")):
                continue

            article = self._extract_article(entry.link, entry.get("title"), pub_date)
            if article:
                article["source"] = source.get("name", urlparse(base_url).netloc)
                article["bias_label"] = source.get("bias", "unknown")
                articles.append(article)
                if callback:
                    callback({"type": "article", "source": article.get("source"), "title": article.get("title")})

            time.sleep(random.uniform(0.5, 1.5))

        return articles

    def _collect_archived_feed(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect articles from archived snapshots of an RSS feed."""
        url = source.get("url", "")
        cdx_url = "https://web.archive.org/cdx/search/cdx"
        params = {
            "url": url,
            "output": "json",
            "from": start_date.strftime('%Y%m%d'),
            "to": end_date.strftime('%Y%m%d'),
            "filter": "statuscode:200",
            "collapse": "timestamp:8",  # Daily snapshots
            "limit": 30  # Get up to 30 snapshots
        }

        articles = []
        try:
            resp = self.session.get(cdx_url, params=params, timeout=self.request_timeout)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            if len(data) <= 1:  # First row is headers
                return []
            
            # Process snapshots
            for row in data[1:]:  # Skip header
                if self.cancelled or len(articles) >= limit:
                    break
                    
                timestamp = row[1]
                snapshot_url = f"https://web.archive.org/web/{timestamp}/{url}"
                
                try:
                    r = self.session.get(snapshot_url, timeout=self.request_timeout)
                    feed = feedparser.parse(r.text)
                    
                    # Process entries from archived feed
                    processed = self._process_feed_entries(feed.entries, source, url, start_date, end_date, limit - len(articles), callback)
                    
                    # Mark as archived and add to results
                    for art in processed:
                        art["archive_snapshot"] = snapshot_url
                        art["archived_date"] = timestamp
                        articles.append(art)
                        if len(articles) >= limit:
                            break
                            
                    time.sleep(random.uniform(1, 2))
                    
                except Exception as e:
                    self.logger.debug(f"Failed to process snapshot {snapshot_url}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Archive collection failed for {url}: {e}")

        return articles

    def _extract_article(self, url: str, title: str = None, pub_date: datetime = None) -> Optional[Dict[str, Any]]:
        """Extract article content with enhanced extraction."""
        try:
            # Try trafilatura first (it's generally more reliable)
            if TRAFILATURA_AVAILABLE:
                session = self.cloudscraper if hasattr(self, 'cloudscraper') and self.cloudscraper else self.session
                response = session.get(url, timeout=self.request_timeout)
                
                # Extract with trafilatura
                content = trafilatura.extract(
                    response.text, 
                    favor_precision=True,
                    include_comments=False,
                    include_tables=True,
                    deduplicate=True
                )
                
                # Also try to get metadata
                metadata = trafilatura.extract_metadata(response.text)
                
                if content and len(content) > 200:
                    return {
                        "title": title or (metadata.title if metadata else self._extract_title(response.text)),
                        "url": url,
                        "content": content,
                        "published": (metadata.date if metadata and metadata.date else pub_date).isoformat() if (metadata and metadata.date) or pub_date else None,
                        "collected_at": datetime.now().isoformat(),
                        "author": metadata.author if metadata else None,
                        "description": metadata.description if metadata else None
                    }
            
            # Fallback to newspaper3k
            if NEWSPAPER_AVAILABLE:
                config = Config()
                config.browser_user_agent = random.choice(self.user_agents)
                config.request_timeout = self.request_timeout
                
                article = Article(url, config=config)
                article.download()
                article.parse()
                
                if article.text and len(article.text) > 200:
                    return {
                        "title": title or article.title,
                        "url": url,
                        "content": article.text,
                        "published": (article.publish_date or pub_date).isoformat() if (article.publish_date or pub_date) else None,
                        "collected_at": datetime.now().isoformat(),
                        "author": ", ".join(article.authors) if article.authors else None,
                        "top_image": article.top_image
                    }
            
            # Last resort: BeautifulSoup
            response = self.session.get(url, timeout=self.request_timeout)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find article content
            content = ""
            for selector in ['article', 'main', '.article-content', '.entry-content', '.post-content']:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(separator='\n', strip=True)
                    break
            
            if content and len(content) > 200:
                return {
                    "title": title or self._extract_title(response.text),
                    "url": url,
                    "content": content,
                    "published": pub_date.isoformat() if pub_date else None,
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
        
        # Try other date fields
        for field in ['updated_parsed', 'created_parsed']:
            if hasattr(entry, field) and getattr(entry, field):
                try:
                    return datetime.fromtimestamp(time.mktime(getattr(entry, field)))
                except:
                    pass
        
        return None

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try og:title first
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title['content'].strip()
            
            # Try regular title
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
            "id": article.get("id"),
            "author": article.get("author"),
            "archive_url": article.get("archive_url"),
            "original_url": article.get("original_url"),
            "archived_date": article.get("archived_date")
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
                url = source_data["url"].lower()
                if "sitemap" in url or url.endswith(".xml"):
                    source_data["type"] = "sitemap"
                elif not any(ind in url for ind in ['/rss', '/feed', '.xml']):
                    source_data["type"] = "article"
                else:
                    source_data["type"] = "rss"
            
            # Add to sources
            self.sources.append({
                "url": source_data["url"],
                "type": source_data.get("type", "rss"),
                "bias": source_data.get("bias", "unknown"),
                "name": source_data.get("name", urlparse(source_data["url"]).netloc),
                "enabled": True,
                "limit": source_data.get("limit", 50),
                "sitemap_url": source_data.get("sitemap_url"),
                "archive_url": source_data.get("archive_url")
            })
            
            # Save config
            self.config["content_collection"]["sources"] = self.sources
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return False
