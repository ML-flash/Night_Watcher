#!/usr/bin/env python3
"""
Night_watcher Unified Content Collector
Enhanced with comprehensive logging for debugging discovery issues.
"""

import os
import time
import logging
import hashlib
import re
import random
import json
from file_utils import safe_json_load, safe_json_save
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote
import traceback

import requests
import feedparser
import gc
import psutil
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from gov_scrapers import (
    scrape_federal_register,
    scrape_white_house_actions,
    scrape_congress_bills,
)

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

try:
    from googlenewsdecoder import gnewsdecoder
    GOOGLENEWSDECODER_AVAILABLE = True
except ImportError:
    GOOGLENEWSDECODER_AVAILABLE = False
    print("Warning: googlenewsdecoder not installed. Google News URL decoding will be limited.")
    print("Install with: pip install googlenewsdecoder")

logger = logging.getLogger(__name__)

# Import for document repository if needed
try:
    from document_repository import DocumentRepository
except ImportError:
    DocumentRepository = None


class CollectionStats:
    """Track detailed collection statistics for debugging."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sources_attempted = 0
        self.sources_succeeded = 0
        self.sources_failed = 0
        self.feeds_parsed = 0
        self.feed_entries_found = 0
        self.feed_entries_processed = 0
        self.urls_attempted = 0
        self.articles_extracted = 0
        self.articles_filtered_out = 0
        self.political_articles = 0
        self.extraction_failures = 0
        self.source_errors = {}
        self.extraction_methods = {'trafilatura': 0, 'newspaper': 0, 'beautifulsoup': 0}
        self.filter_reasons = {}
        self.google_news_decoded = 0
        self.google_news_failed = 0
        
    def log_source_error(self, source_name, error):
        if source_name not in self.source_errors:
            self.source_errors[source_name] = []
        self.source_errors[source_name].append(str(error))
        
    def log_filter_reason(self, reason):
        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
        
    def get_summary(self):
        return {
            'sources': {
                'attempted': self.sources_attempted,
                'succeeded': self.sources_succeeded,
                'failed': self.sources_failed,
                'success_rate': f"{(self.sources_succeeded/max(1,self.sources_attempted)*100):.1f}%"
            },
            'feeds': {
                'parsed': self.feeds_parsed,
                'entries_found': self.feed_entries_found,
                'entries_processed': self.feed_entries_processed
            },
            'articles': {
                'urls_attempted': self.urls_attempted,
                'extracted_successfully': self.articles_extracted,
                'extraction_failures': self.extraction_failures,
                'extraction_success_rate': f"{(self.articles_extracted/max(1,self.urls_attempted)*100):.1f}%",
                'political_articles': self.political_articles,
                'filtered_out': self.articles_filtered_out
            },
            'google_news': {
                'decoded_successfully': self.google_news_decoded,
                'decode_failures': self.google_news_failed,
                'decode_success_rate': f"{(self.google_news_decoded/max(1,self.google_news_decoded+self.google_news_failed)*100):.1f}%"
            },
            'extraction_methods': self.extraction_methods,
            'filter_reasons': self.filter_reasons,
            'source_errors': self.source_errors
        }


class ContentCollector:
    """Enhanced collector with comprehensive logging for debugging discovery issues."""

    def __init__(self, config: Dict[str, Any], document_repository=None, base_dir: str = "data"):
        self.config = config
        self.document_repository = document_repository
        self.base_dir = base_dir
        self.stats = CollectionStats()
        
        # Configuration
        cc = config.get("content_collection", {})
        self.article_limit = cc.get("article_limit", 50)
        self.sources = cc.get("sources", [])
        self.request_timeout = cc.get("request_timeout", 45)
        self.delay_between_requests = cc.get("delay_between_requests", 2.0)
        self.max_gap_days = cc.get("max_gap_days", 30)
        self.gap_detection_enabled = cc.get("gap_detection_enabled", True)
        self.use_google_news = cc.get("use_google_news", True)
        self.use_gdelt = cc.get("use_gdelt", True)
        self.use_gov_scrapers = cc.get("use_gov_scrapers", True)

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
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
        
        # Cache for decoded URLs
        self._decoded_url_cache = {}
        
        self.logger = logging.getLogger("ContentCollector")
        
        # Setup detailed logging
        self._setup_debug_logging()
        self._init_session()
        
        self.logger.info(f"Collector initialized with {len(self.sources)} sources")
        self.logger.info(f"Political keywords: {len(self.govt_keywords)} loaded")
        self.logger.info(f"Google News decoder available: {GOOGLENEWSDECODER_AVAILABLE}")

    def _setup_debug_logging(self):
        """Setup detailed logging for debugging collection issues."""
        # Create a detailed file logger
        log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Collection debug log
        debug_handler = logging.FileHandler(
            os.path.join(log_dir, f"collection_debug_{datetime.now().strftime('%Y%m%d')}.log")
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        debug_handler.setFormatter(debug_formatter)
        self.logger.addHandler(debug_handler)
        self.logger.setLevel(logging.DEBUG)

    def _init_session(self):
        """Initialize HTTP session."""
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._rotate_headers()
        
        # Sites that need special handling
        self.problematic_sites = {
            'washingtonpost.com': {'timeout': 15, 'delay': 3},
            'nytimes.com': {'timeout': 15, 'delay': 3},
            'wsj.com': {'timeout': 20, 'delay': 4},
            'cnn.com': {'timeout': 10, 'delay': 2},
            'foxnews.com': {'timeout': 10, 'delay': 2}
        }
        
        if CLOUDSCRAPER_AVAILABLE:
            try:
                self.cloudscraper = cloudscraper.create_scraper()
                self.logger.info("CloudScraper initialized successfully")
            except Exception as e:
                self.logger.warning(f"CloudScraper init failed: {e}")
                self.cloudscraper = None
        else:
            self.logger.info("CloudScraper not available")

    def cancel(self):
        """Signal to stop collection."""
        self.cancelled = True
        self.logger.info("Collection cancellation requested")

    def check_memory_usage(self, threshold_mb: int = 1024) -> bool:
        """Check memory usage and trigger cleanup if needed."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > threshold_mb:
                self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                gc.collect()
                if hasattr(self, '_decoded_url_cache'):
                    if len(self._decoded_url_cache) > 1000:
                        self._decoded_url_cache.clear()
                        self.logger.info("Cleared URL cache")
                return True
        except Exception as e:
            self.logger.debug(f"Memory check failed: {e}")
        return False

    def _rotate_headers(self):
        """Rotate session headers."""
        user_agent = random.choice(self.user_agents)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1'
        })
        self.logger.debug(f"Rotated to User-Agent: {user_agent[:50]}...")

    def _resolve_google_news_url(self, google_url: str) -> Optional[str]:
        """Resolve Google News redirect URL to actual article URL using googlenewsdecoder."""
        # Check cache first
        if google_url in self._decoded_url_cache:
            cached_url = self._decoded_url_cache[google_url]
            self.logger.debug(f"Using cached decoded URL: {cached_url}")
            return cached_url
        
        if GOOGLENEWSDECODER_AVAILABLE:
            try:
                self.logger.debug(f"Attempting to decode Google News URL with googlenewsdecoder: {google_url}")
                
                # Use googlenewsdecoder with appropriate delay
                result = gnewsdecoder(google_url, interval=2)
                
                if result.get("status") and result.get("decoded_url"):
                    decoded_url = result["decoded_url"]
                    self.logger.debug(f"Successfully decoded Google News URL: {google_url} -> {decoded_url}")
                    
                    # Cache the result
                    self._decoded_url_cache[google_url] = decoded_url
                    self.stats.google_news_decoded += 1
                    
                    return decoded_url
                else:
                    self.logger.warning(f"googlenewsdecoder failed for {google_url}: {result.get('message', 'Unknown error')}")
                    self.stats.google_news_failed += 1
                    
            except Exception as e:
                self.logger.error(f"Error using googlenewsdecoder for {google_url}: {e}")
                self.stats.google_news_failed += 1
        
        # Fallback to manual decoding methods
        self.logger.debug("Falling back to manual Google News URL resolution methods")
        
        try:
            # Try manual base64 decoding for older URLs
            if "/articles/" in google_url:
                import base64
                
                # Extract the article ID part
                article_id = google_url.split("/articles/")[1].split("?")[0]
                
                # Add padding and try to decode
                encoded_part = article_id + "==="
                
                try:
                    decoded_bytes = base64.urlsafe_b64decode(encoded_part)
                    
                    # Look for URL pattern in decoded bytes
                    # Google News uses protocol buffer encoding
                    if decoded_bytes.startswith(b'\x08\x13\x22'):
                        decoded_bytes = decoded_bytes[3:]  # Remove prefix
                        if decoded_bytes.endswith(b'\xd2\x01\x00'):
                            decoded_bytes = decoded_bytes[:-3]  # Remove suffix
                        
                        # Extract URL based on length prefix
                        length = decoded_bytes[0]
                        if length < 0x80:
                            decoded_url = decoded_bytes[1:length+1].decode('utf-8')
                            self.logger.debug(f"Manual decode successful: {decoded_url}")
                            self._decoded_url_cache[google_url] = decoded_url
                            self.stats.google_news_decoded += 1
                            return decoded_url
                            
                except Exception as e:
                    self.logger.debug(f"Manual base64 decode failed: {e}")
                    
        except Exception as e:
            self.logger.debug(f"URL parsing failed: {e}")
        
        # If all else fails, try following redirects with session
        try:
            session = self.cloudscraper if self.cloudscraper else self.session
            
            # Browser-like headers for Google
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = session.get(google_url, headers=headers, allow_redirects=True, timeout=15)
            
            if response.url and response.url != google_url and not response.url.startswith('https://news.google.com'):
                self.logger.debug(f"Redirect resolution successful: {google_url} -> {response.url}")
                self._decoded_url_cache[google_url] = response.url
                self.stats.google_news_decoded += 1
                return response.url
                
        except Exception as e:
            self.logger.debug(f"Redirect resolution failed: {e}")
        
        self.logger.warning(f"All Google News URL resolution methods failed for: {google_url}")
        self.stats.google_news_failed += 1
        return None

    def _extract_article(self, url: str, title: str = None, pub_date: datetime = None) -> Optional[Dict[str, Any]]:
        """Extract article content with enhanced logging."""
        self.logger.debug(f"Starting article extraction from: {url}")
        
        # Handle Google News redirect URLs
        if "news.google.com" in url:
            self.logger.debug("Detected Google News URL, resolving redirect...")
            actual_url = self._resolve_google_news_url(url)
            if actual_url and actual_url != url:
                self.logger.debug(f"Resolved to actual URL: {actual_url}")
                url = actual_url
            else:
                self.logger.warning(f"Failed to resolve Google News URL: {url}")
                return None
        
        # Check for problematic sites and adjust timeout/delay
        domain = urlparse(url).netloc.replace('www.', '')
        site_config = self.problematic_sites.get(domain, {})
        timeout = site_config.get('timeout', self.request_timeout)
        delay = site_config.get('delay', 0)
        
        if site_config:
            self.logger.debug(f"Using special config for {domain}: timeout={timeout}s, delay={delay}s")
            time.sleep(delay)  # Pre-request delay for problematic sites
        
        try:
            # Use cloudscraper if available for better success rate
            session = self.cloudscraper if self.cloudscraper else self.session
            session_type = "cloudscraper" if self.cloudscraper else "requests"
            
            self.logger.debug(f"Using {session_type} for extraction")
            
            # Try trafilatura first (it's generally more reliable)
            if TRAFILATURA_AVAILABLE:
                self.logger.debug("Trying trafilatura extraction")
                try:
                    timeout_config = {
                        'connect_timeout': 10,
                        'read_timeout': 30,
                        'total_timeout': 45,
                    }
                    response = session.get(
                        url,
                        timeout=(timeout_config['connect_timeout'], timeout_config['read_timeout']),
                        allow_redirects=True,
                    )
                    self.logger.debug(f"HTTP response: {response.status_code}")
                    
                    if response.status_code != 200:
                        self.logger.warning(f"HTTP {response.status_code} for {url}")
                        # Don't return yet, try other methods
                    else:
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
                            self.stats.extraction_methods['trafilatura'] += 1
                            self.logger.debug(f"Trafilatura success: {len(content)} chars extracted")
                            
                            # Check if content is political
                            if not self._is_political(title or '', content):
                                self.logger.debug("Content not political, skipping")
                                self.stats.log_filter_reason("content_not_political")
                                return None
                            
                            self.stats.political_articles += 1
                            
                            # Handle date properly - trafilatura may return string or datetime
                            published_date = None
                            if metadata and metadata.date:
                                if isinstance(metadata.date, str):
                                    published_date = metadata.date
                                elif hasattr(metadata.date, 'isoformat'):
                                    published_date = metadata.date.isoformat()
                                else:
                                    published_date = str(metadata.date)
                            elif pub_date:
                                if isinstance(pub_date, str):
                                    published_date = pub_date
                                elif hasattr(pub_date, 'isoformat'):
                                    published_date = pub_date.isoformat()
                                else:
                                    published_date = str(pub_date)
                            
                            return {
                                "title": title or (metadata.title if metadata else self._extract_title(response.text)),
                                "url": url,
                                "content": content,
                                "published": published_date,
                                "collected_at": datetime.now().isoformat(),
                                "author": metadata.author if metadata else None,
                                "description": metadata.description if metadata else None,
                                "extraction_method": "trafilatura"
                            }
                        else:
                            self.logger.debug(f"Trafilatura extracted insufficient content: {len(content) if content else 0} chars")
                
                except requests.exceptions.ConnectTimeout:
                    self.logger.warning(f"Connection timeout for {url}")
                    return None
                except requests.exceptions.ReadTimeout:
                    self.logger.warning(f"Read timeout for {url}")
                    return None
                except requests.exceptions.Timeout:
                    self.logger.warning(f"General timeout for {url}")
                    return None
                except Exception as e:
                    self.logger.debug(f"Trafilatura extraction failed: {e}")
            
            # Fallback to newspaper3k
            if NEWSPAPER_AVAILABLE:
                self.logger.debug("Trying newspaper3k extraction")
                try:
                    config = Config()
                    config.browser_user_agent = random.choice(self.user_agents)
                    config.request_timeout = self.request_timeout
                    config.fetch_images = False  # Skip images to speed up
                    config.memoize_articles = False  # Don't cache
                    
                    article = Article(url, config=config)
                    
                    # Try with existing session first (faster if it works)
                    try:
                        article.download(input_html=response.text if 'response' in locals() else None)
                    except:
                        # If that fails, let newspaper3k download it itself
                        article.download()
                    
                    article.parse()
                    
                    if article.text and len(article.text) > 200:
                        self.stats.extraction_methods['newspaper'] += 1
                        self.logger.debug(f"Newspaper3k success: {len(article.text)} chars extracted")
                        
                        # Check if content is political
                        if not self._is_political(title or article.title, article.text):
                            self.logger.debug("Content not political, skipping")
                            self.stats.log_filter_reason("content_not_political")
                            return None
                        
                        self.stats.political_articles += 1
                        
                        # Handle date properly - may be string or datetime
                        published_date = None
                        if article.publish_date:
                            if isinstance(article.publish_date, str):
                                published_date = article.publish_date
                            elif hasattr(article.publish_date, 'isoformat'):
                                published_date = article.publish_date.isoformat()
                            else:
                                published_date = str(article.publish_date)
                        elif pub_date:
                            if isinstance(pub_date, str):
                                published_date = pub_date
                            elif hasattr(pub_date, 'isoformat'):
                                published_date = pub_date.isoformat()
                            else:
                                published_date = str(pub_date)
                        
                        return {
                            "title": title or article.title,
                            "url": url,
                            "content": article.text,
                            "published": published_date,
                            "collected_at": datetime.now().isoformat(),
                            "author": ", ".join(article.authors) if article.authors else None,
                            "top_image": article.top_image,
                            "extraction_method": "newspaper3k"
                        }
                    else:
                        self.logger.debug(f"Newspaper3k extracted insufficient content: {len(article.text) if article.text else 0} chars")
                        
                except Exception as e:
                    self.logger.debug(f"Newspaper3k extraction failed: {e}")
            
            # Last resort: BeautifulSoup
            self.logger.debug("Trying BeautifulSoup extraction")
            try:
                timeout_config = {
                    'connect_timeout': 10,
                    'read_timeout': 30,
                    'total_timeout': 45,
                }
                response = session.get(
                    url,
                    timeout=(timeout_config['connect_timeout'], timeout_config['read_timeout']),
                    allow_redirects=True,
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find article content
                content = ""
                for selector in ['article', 'main', '.article-content', '.entry-content', '.post-content', 
                                '[class*="article-body"]', '[class*="story-body"]', '.content-body']:
                    elem = soup.select_one(selector)
                    if elem:
                        content = elem.get_text(separator='\n', strip=True)
                        self.logger.debug(f"BeautifulSoup found content with selector '{selector}': {len(content)} chars")
                        break
                
                if content and len(content) > 200:
                    self.stats.extraction_methods['beautifulsoup'] += 1
                    self.logger.debug(f"BeautifulSoup success: {len(content)} chars extracted")
                    
                    # Check if content is political
                    if not self._is_political(title or '', content):
                        self.logger.debug("Content not political, skipping")
                        self.stats.log_filter_reason("content_not_political")
                        return None
                    
                    self.stats.political_articles += 1
                    
                    # Handle date properly
                    published_date = None
                    if pub_date:
                        if isinstance(pub_date, str):
                            published_date = pub_date
                        elif hasattr(pub_date, 'isoformat'):
                            published_date = pub_date.isoformat()
                        else:
                            published_date = str(pub_date)
                    
                    return {
                        "title": title or self._extract_title(response.text),
                        "url": url,
                        "content": content,
                        "published": published_date,
                        "collected_at": datetime.now().isoformat(),
                        "extraction_method": "beautifulsoup"
                    }
                else:
                    self.logger.debug(f"BeautifulSoup extracted insufficient content: {len(content)} chars")
                    
            except requests.exceptions.ConnectTimeout:
                self.logger.warning(f"Connection timeout for {url}")
                return None
            except requests.exceptions.ReadTimeout:
                self.logger.warning(f"Read timeout for {url}")
                return None
            except requests.exceptions.Timeout:
                self.logger.warning(f"General timeout for {url}")
                return None
            except Exception as e:
                self.logger.debug(f"BeautifulSoup extraction failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Article extraction completely failed for {url}: {e}")
            self.logger.debug(f"Full extraction error:", exc_info=True)
        
        self.logger.warning(f"All extraction methods failed for {url}")
        return None

    def collect_content(self, force_mode: Optional[str] = None, callback=None) -> Dict[str, Any]:
        """Main collection method with enhanced logging."""
        self.cancelled = False
        self.stats.reset()
        
        self.logger.info("=== STARTING CONTENT COLLECTION ===")
        
        # Determine collection mode and date range
        mode, start_date, end_date = self._get_collection_mode(force_mode)
        self.logger.info(f"Collection mode: {mode}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Enabled sources: {len([s for s in self.sources if s.get('enabled', True)])}")
        
        all_articles = []
        document_ids = []
        
        # 1. Collect from regular sources (RSS, etc.)
        self.logger.info("=== PHASE 1: RSS/REGULAR SOURCES ===")
        if callback:
            callback({"type": "status", "message": "Collecting from RSS sources..."})
        
        regular_result = self._collect_date_range(start_date, end_date, mode, callback)
        all_articles.extend(regular_result.get("articles", []))
        document_ids.extend(regular_result.get("document_ids", []))
        
        self.logger.info(f"Phase 1 complete: {len(regular_result.get('articles', []))} articles")
        
        # 2. Collect from Google News if enabled
        if self.use_google_news:
            self.logger.info("=== PHASE 2: GOOGLE NEWS ===")
            if callback:
                callback({"type": "status", "message": "Collecting from Google News historical search..."})
            
            google_articles = self._collect_google_news_historical(start_date, end_date, callback)
            self.logger.info(f"Google News returned {len(google_articles)} articles")
            
            # Process and store Google News articles
            for article in google_articles:
                article["id"] = self._generate_id(article["url"])
                
                if self.document_repository:
                    doc_id = self._store_article(article)
                    document_ids.append(doc_id)
                    article["document_id"] = doc_id
                
                all_articles.append(article)
        
        # 3. Collect from GDELT if enabled
        if self.use_gdelt:
            self.logger.info("=== PHASE 3: GDELT ===")
            if callback:
                callback({"type": "status", "message": "Collecting from GDELT database..."})
            
            gdelt_articles = self._collect_gdelt_comprehensive(start_date, end_date, callback)
            self.logger.info(f"GDELT returned {len(gdelt_articles)} articles")
            
            # Process and store GDELT articles
            for article in gdelt_articles:
                article["id"] = self._generate_id(article["url"])
                
                if self.document_repository:
                    doc_id = self._store_article(article)
                    document_ids.append(doc_id)
                    article["document_id"] = doc_id
                
                all_articles.append(article)
        
        # 4. Collect from government websites
        self.logger.info("=== PHASE 4: GOVERNMENT SOURCES ===")
        if callback:
            callback({"type": "status", "message": "Collecting from government sources..."})

        if self.use_gov_scrapers:
            gov_articles = self._collect_government_scrapers(start_date, end_date, callback)
        else:
            gov_articles = self._collect_government_apis(start_date, end_date, callback)
        self.logger.info(f"Government sources returned {len(gov_articles)} articles")
        
        for article in gov_articles:
            article["id"] = self._generate_id(article["url"])
            
            if self.document_repository:
                doc_id = self._store_article(article)
                document_ids.append(doc_id)
                article["document_id"] = doc_id
            
            all_articles.append(article)
        
        # Update collection history
        self._update_collection_history(all_articles)
        
        # Deduplicate articles
        seen_urls = set()
        unique_articles = []
        unique_doc_ids = []
        
        duplicates_removed = 0
        for i, article in enumerate(all_articles):
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)
                if i < len(document_ids):
                    unique_doc_ids.append(document_ids[i])
            else:
                duplicates_removed += 1
        
        # Log final statistics
        stats_summary = self.stats.get_summary()
        self.logger.info("=== COLLECTION COMPLETE ===")
        self.logger.info(f"Total articles collected: {len(unique_articles)}")
        self.logger.info(f"Duplicates removed: {duplicates_removed}")
        self.logger.info(f"Source success rate: {stats_summary['sources']['success_rate']}")
        self.logger.info(f"Article extraction success rate: {stats_summary['articles']['extraction_success_rate']}")
        self.logger.info(f"Google News decode success rate: {stats_summary['google_news']['decode_success_rate']}")
        
        # Log detailed statistics
        self.logger.info("=== DETAILED STATISTICS ===")
        for category, data in stats_summary.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    self.logger.info(f"{category}.{key}: {value}")
        
        # Log any source errors
        if stats_summary['source_errors']:
            self.logger.warning("=== SOURCE ERRORS ===")
            for source, errors in stats_summary['source_errors'].items():
                for error in errors:
                    self.logger.warning(f"{source}: {error}")
        
        return {
            "articles": unique_articles,
            "document_ids": unique_doc_ids,
            "status": {
                "articles_collected": len(unique_articles),
                "collection_mode": mode,
                "sources_used": {
                    "rss": len([a for a in unique_articles if not a.get("via_google_news") and not a.get("via_gdelt")]),
                    "google_news": len([a for a in unique_articles if a.get("via_google_news")]),
                    "gdelt": len([a for a in unique_articles if a.get("via_gdelt")]),
                    "government": len([a for a in unique_articles if a.get("via_gov_api")])
                },
                "timestamp": datetime.now().isoformat(),
                "detailed_stats": stats_summary
            }
        }

    def _collect_date_range(self, start_date: datetime, end_date: datetime, mode: str, callback=None) -> Dict[str, Any]:
        """Collect content for a specific date range with detailed logging."""
        all_articles = []
        document_ids = []
        
        enabled_sources = [s for s in self.sources if s.get("enabled", True)]
        self.logger.info(f"Processing {len(enabled_sources)} enabled sources out of {len(self.sources)} total")
        
        for i, source in enumerate(enabled_sources):
            self.stats.sources_attempted += 1
            source_name = source.get("name", source.get("url", "Unknown"))
            
            self.logger.info(f"=== SOURCE {i+1}/{len(enabled_sources)}: {source_name} ===")
            self.logger.info(f"URL: {source.get('url')}")
            self.logger.info(f"Type: {source.get('type', 'rss')}")
            self.logger.info(f"Bias: {source.get('bias', 'unknown')}")
            
            # Skip archive sources that are timing out
            if source.get("type") == "archive":
                self.logger.info(f"Skipping archive source {source_name} - using alternative methods")
                self.stats.log_filter_reason("archive_type_skipped")
                continue

            try:
                if self.cancelled:
                    self.logger.info("Collection cancelled by user")
                    break

                limit = source.get("limit", self.article_limit)
                self.logger.info(f"Attempting to collect up to {limit} articles")
                
                articles = self._collect_from_source(source, start_date, end_date, limit, callback)
                
                self.logger.info(f"Source returned {len(articles)} articles")
                
                # Filter by inauguration day if first run
                if mode == "first_run":
                    before_filter = len(articles)
                    articles = [a for a in articles if self._after_inauguration(a)]
                    filtered_out = before_filter - len(articles)
                    if filtered_out > 0:
                        self.logger.info(f"Filtered out {filtered_out} articles (before inauguration)")
                        self.stats.articles_filtered_out += filtered_out
                
                # Store documents
                for article in articles:
                    article["id"] = self._generate_id(article["url"])
                    
                    if self.document_repository:
                        doc_id = self._store_article(article)
                        document_ids.append(doc_id)
                        article["document_id"] = doc_id
                
                all_articles.extend(articles)
                self.stats.sources_succeeded += 1
                
                self.logger.info(f"Successfully processed {len(articles)} articles from {source_name}")
                
                time.sleep(random.uniform(1, 3))
                if self.cancelled:
                    break
                
            except Exception as e:
                self.stats.sources_failed += 1
                self.stats.log_source_error(source_name, e)
                error_msg = f"Error processing {source_name}: {e}"
                self.logger.error(error_msg)
                self.logger.debug(f"Full traceback for {source_name}:", exc_info=True)
        
        # Update state
        if mode != "custom":
            self._update_last_run()

        if self.cancelled and callback:
            callback({"type": "cancelled"})
        
        self.logger.info(f"Date range collection complete: {len(all_articles)} total articles")
        
        return {
            "articles": all_articles,
            "document_ids": document_ids,
            "status": {
                "articles_collected": len(all_articles),
                "collection_mode": mode,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _collect_from_source(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect from a single source with detailed logging."""
        url = source.get("url", "")
        source_type = source.get("type", "rss")
        source_name = source.get("name", url)

        self.logger.debug(f"Collecting from {source_name} (type: {source_type})")

        if source_type == "article":
            self.logger.debug("Processing direct article URL")
            # Direct article
            article = self._extract_article(url)
            if article:
                if callback:
                    callback({"type": "article", "source": source_name, "title": article.get("title")})
                self.logger.debug(f"Successfully extracted direct article: {article.get('title')}")
            else:
                self.logger.warning(f"Failed to extract direct article from {url}")
            return [article] if article else []
        
        elif source_type == "sitemap":
            self.logger.debug("Processing sitemap")
            # Parse sitemap for article URLs
            return self._collect_from_sitemap(source, start_date, end_date, limit, callback)
        
        else:  # RSS feed (default)
            self.logger.debug(f"Processing RSS feed: {url}")
            try:
                # Parse RSS feed
                self.logger.debug(f"Fetching RSS feed with user agent: {self.session.headers.get('User-Agent', 'Unknown')[:50]}...")
                
                feed = feedparser.parse(url, agent=random.choice(self.user_agents))
                self.stats.feeds_parsed += 1
                
                # Check feed status
                if hasattr(feed, 'status'):
                    self.logger.debug(f"Feed HTTP status: {feed.status}")
                    if feed.status >= 400:
                        raise Exception(f"HTTP {feed.status} error fetching feed")
                
                if hasattr(feed, 'bozo') and feed.bozo:
                    self.logger.warning(f"Feed parser detected issues: {getattr(feed, 'bozo_exception', 'Unknown')}")
                
                entries_count = len(feed.entries)
                self.stats.feed_entries_found += entries_count
                self.logger.info(f"Feed parsed successfully: {entries_count} entries found")
                
                if entries_count == 0:
                    self.logger.warning(f"No entries found in RSS feed {url}")
                    return []
                
                articles = self._process_feed_entries_with_circuit_breaker(
                    feed.entries,
                    source,
                    url,
                    start_date,
                    end_date,
                    limit,
                    callback,
                )
                self.logger.info(f"Feed processing complete: {len(articles)} articles extracted from {entries_count} entries")
                return articles
                
            except Exception as e:
                error_msg = f"RSS collection failed for {url}: {e}"
                self.logger.error(error_msg)
                self.logger.debug(f"RSS error details:", exc_info=True)
                if callback:
                    callback({"type": "error", "source": source_name, "message": str(e)})
                return []

    def _process_feed_entries(self, entries: List[Any], source: Dict[str, Any], base_url: str, start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Process RSS feed entries into article records with detailed logging."""
        articles = []
        source_name = source.get("name", urlparse(base_url).netloc)
        
        self.logger.debug(f"Processing {len(entries)} feed entries")
        
        for i, entry in enumerate(entries):
            if self.cancelled or len(articles) >= limit:
                if self.cancelled:
                    self.logger.info("Processing cancelled by user")
                else:
                    self.logger.info(f"Reached article limit ({limit})")
                break

            self.stats.feed_entries_processed += 1
            
            # Log entry details
            entry_title = getattr(entry, 'title', 'No title')
            entry_link = getattr(entry, 'link', None)
            
            self.logger.debug(f"Entry {i+1}/{len(entries)}: {entry_title}")
            self.logger.debug(f"Link: {entry_link}")

            if not entry_link:
                self.logger.debug("Skipping entry: no link found")
                self.stats.log_filter_reason("no_link")
                continue

            # Check publication date
            pub_date = self._parse_date(entry)
            if pub_date:
                self.logger.debug(f"Publication date: {pub_date}")
                if pub_date < start_date or pub_date > end_date:
                    self.logger.debug(f"Skipping entry: outside date range ({start_date} to {end_date})")
                    self.stats.log_filter_reason("outside_date_range")
                    continue
            else:
                self.logger.debug("No publication date found, including anyway")

            # Check political relevance
            entry_summary = getattr(entry, 'summary', '')
            political_check = self._is_political(entry_title, entry_summary)
            self.logger.debug(f"Political relevance check: {political_check}")
            
            if not political_check:
                self.logger.debug("Skipping entry: not political")
                self.stats.log_filter_reason("not_political")
                continue

            # Extract article
            self.logger.debug(f"Attempting to extract article from: {entry_link}")
            self.stats.urls_attempted += 1
            
            article = self._extract_article(entry_link, entry_title, pub_date)
            if article:
                self.stats.articles_extracted += 1
                article["source"] = source_name
                article["bias_label"] = source.get("bias", "unknown")
                articles.append(article)
                
                self.logger.debug(f"Successfully extracted article: {article.get('title')}")
                if callback:
                    callback({"type": "article", "source": article.get("source"), "title": article.get("title")})
            else:
                self.stats.extraction_failures += 1
                self.logger.warning(f"Failed to extract article from {entry_link}")

            time.sleep(random.uniform(0.5, 1.5))

        self.logger.info(f"Feed entry processing complete: {len(articles)} articles from {len(entries)} entries")
        return articles

    def _process_single_entry(self, entry: Any, source: Dict[str, Any], start_date: datetime, end_date: datetime) -> Optional[Dict[str, Any]]:
        """Process a single RSS feed entry into an article record."""
        source_url = source.get("url", "")
        source_name = source.get("name", urlparse(source_url).netloc if source_url else "Unknown")

        self.stats.feed_entries_processed += 1

        entry_title = getattr(entry, 'title', 'No title')
        entry_link = getattr(entry, 'link', None)

        self.logger.debug(f"Processing entry: {entry_title}")
        self.logger.debug(f"Link: {entry_link}")

        if not entry_link:
            self.logger.debug("Skipping entry: no link found")
            self.stats.log_filter_reason("no_link")
            return None

        pub_date = self._parse_date(entry)
        if pub_date:
            self.logger.debug(f"Publication date: {pub_date}")
            if pub_date < start_date or pub_date > end_date:
                self.logger.debug(
                    f"Skipping entry: outside date range ({start_date} to {end_date})")
                self.stats.log_filter_reason("outside_date_range")
                return None
        else:
            self.logger.debug("No publication date found, including anyway")

        entry_summary = getattr(entry, 'summary', '')
        if not self._is_political(entry_title, entry_summary):
            self.logger.debug("Skipping entry: not political")
            self.stats.log_filter_reason("not_political")
            return None

        self.logger.debug(f"Attempting to extract article from: {entry_link}")
        self.stats.urls_attempted += 1

        article = self._extract_article(entry_link, entry_title, pub_date)
        if article:
            self.stats.articles_extracted += 1
            article["source"] = source_name
            article["bias_label"] = source.get("bias", "unknown")
            self.logger.debug(f"Successfully extracted article: {article.get('title')}")
        else:
            self.stats.extraction_failures += 1
            self.logger.warning(f"Failed to extract article from {entry_link}")
        time.sleep(random.uniform(0.5, 1.5))
        return article

    def _process_feed_entries_with_circuit_breaker(self, entries: List[Any], source: Dict[str, Any], base_url: str,
                                                   start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Process RSS feed entries with circuit breaker protection."""
        articles = []
        source_name = source.get("name", urlparse(base_url).netloc)

        max_consecutive_failures = 5
        consecutive_failures = 0

        for i, entry in enumerate(entries):
            if self.cancelled or len(articles) >= limit:
                break

            try:
                article = self._process_single_entry(entry, source, start_date, end_date)
                if article:
                    articles.append(article)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                if consecutive_failures >= max_consecutive_failures:
                    self.logger.warning(
                        f"Circuit breaker triggered for {source_name} after {consecutive_failures} consecutive failures")
                    break

            except Exception as e:
                consecutive_failures += 1
                self.logger.error(f"Entry processing failed: {e}")

                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error(f"Too many failures for {source_name}, stopping")
                    break

        return articles

    def _is_political(self, title: str, content: str) -> bool:
        """Check if content is political with logging."""
        text = f"{title} {content}".lower()
        matches = [kw for kw in self.govt_keywords if kw in text]
        is_political = len(matches) >= 2
        
        self.logger.debug(f"Political check: {len(matches)} keyword matches ({'political' if is_political else 'not political'})")
        if matches:
            self.logger.debug(f"Matched keywords: {matches[:5]}")  # Log first 5 matches
        
        return is_political

    def _collect_google_news_historical(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect historical news using Google News search with detailed logging."""
        
        self.logger.info("Starting Google News historical collection")
        
        # Key search terms for comprehensive political coverage
        search_queries = [
            # Executive branch
            "white house announcement",
            "president biden statement", 
            "president trump statement",
            "executive order signed",
            "presidential memorandum",
            
            # Legislative
            "congress passes bill",
            "senate votes",
            "house approves",
            "legislation introduced",
            "congressional hearing",
            
            # Judicial
            "supreme court decision",
            "federal court ruling",
            "judge blocks",
            "judicial nomination",
            
            # General government
            "federal agency",
            "government policy",
            "cabinet meeting",
            "state department",
            "pentagon announces",
            
            # Political events
            "political news",
            "investigation launched",
            "committee hearing",
            "government accountability",
            "federal budget"
        ]
        
        all_articles = []
        
        # Format dates for Google News
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Google News date range: {start_str} to {end_str}")
        self.logger.info(f"Will try {len(search_queries)} search queries")
        
        for i, query in enumerate(search_queries):
            if self.cancelled:
                break
                
            self.logger.debug(f"Google News query {i+1}/{len(search_queries)}: '{query}'")
            
            try:
                # Build Google News RSS URL with date filter
                encoded_query = quote(f'{query} after:{start_str} before:{end_str}')
                google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                
                self.logger.debug(f"Google News URL: {google_news_url}")
                
                # Parse the RSS feed
                feed = feedparser.parse(google_news_url)
                entries_found = len(feed.entries)
                
                self.logger.debug(f"Google News returned {entries_found} entries for query '{query}'")
                
                articles_from_query = 0
                for entry in feed.entries[:20]:  # Limit per query
                    if self.cancelled:
                        break
                        
                    # Extract the article
                    article = self._extract_article(entry.link, entry.title)
                    if article and self._is_political(article.get("title", ""), article.get("content", "")):
                        article['search_query'] = query
                        article['via_google_news'] = True
                        article['source'] = 'Google News Aggregate'
                        all_articles.append(article)
                        articles_from_query += 1
                        
                        if callback:
                            callback({
                                "type": "article",
                                "source": "Google News Search",
                                "title": article.get("title"),
                                "query": query
                            })
                    
                    # Small delay to be respectful
                    time.sleep(random.uniform(0.5, 1.5))
                
                self.logger.debug(f"Query '{query}' yielded {articles_from_query} political articles")
                
                # Delay between queries
                time.sleep(random.uniform(2, 3))
                
            except Exception as e:
                self.logger.error(f"Google News search failed for '{query}': {e}")
                if callback:
                    callback({
                        "type": "error",
                        "source": "Google News",
                        "message": f"Search failed: {query}"
                    })
        
        self.logger.info(f"Google News collection complete: {len(all_articles)} articles")
        return all_articles

    def _collect_gdelt_comprehensive(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect from GDELT's comprehensive news database with detailed logging."""
        self.logger.info("Starting GDELT collection")
        
        articles = []
        
        # GDELT queries for comprehensive coverage
        queries = [
            'government',
            'congress',
            'white house',
            'supreme court',
            'executive order',
            'legislation',
            'federal agency',
            'political'
        ]
        
        days_back = (end_date - start_date).days
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        self.logger.info(f"GDELT: {days_back} days lookback, {len(queries)} queries")
        
        for i, query in enumerate(queries):
            if self.cancelled:
                break
                
            self.logger.debug(f"GDELT query {i+1}/{len(queries)}: '{query}'")
            
            try:
                params = {
                    'query': query,
                    'mode': 'artlist',
                    'maxrecords': 100,
                    'timespan': f'{days_back}d',
                    'sort': 'hybridrel',
                    'format': 'json'
                }
                
                self.logger.debug(f"GDELT API call with params: {params}")
                
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    self.logger.warning(f"GDELT API returned status {response.status_code}")
                    continue
                
                data = response.json()
                gdelt_articles = data.get('articles', [])
                
                self.logger.debug(f"GDELT returned {len(gdelt_articles)} articles for query '{query}'")
                
                articles_from_query = 0
                for article_data in gdelt_articles:
                    # Extract the full article
                    article = self._extract_article(article_data['url'], article_data['title'])
                    if article and self._is_political(article.get("title", ""), article.get("content", "")):
                        article['via_gdelt'] = True
                        article['gdelt_query'] = query
                        article['source'] = article_data.get('domain', 'Unknown')
                        articles.append(article)
                        articles_from_query += 1
                        
                        if callback:
                            callback({
                                "type": "article",
                                "source": "GDELT",
                                "title": article.get("title")
                            })
                
                self.logger.debug(f"GDELT query '{query}' yielded {articles_from_query} political articles")
                
                time.sleep(1)  # Be nice to GDELT
                
            except Exception as e:
                self.logger.error(f"GDELT error for '{query}': {e}")
                
        self.logger.info(f"GDELT collection complete: {len(articles)} articles")
        return articles

    def _collect_government_apis(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect from government APIs with detailed logging."""
        self.logger.info("Starting Government API collection")
        
        articles = []
        
        # Federal Register API
        self.logger.debug("Trying Federal Register API")
        try:
            fr_url = "https://www.federalregister.gov/api/v1/documents"
            params = {
                "conditions[publication_date][gte]": start_date.strftime('%Y-%m-%d'),
                "conditions[publication_date][lte]": end_date.strftime('%Y-%m-%d'),
                "per_page": 100,
                "order": "newest"
            }
            
            self.logger.debug(f"Federal Register API params: {params}")
            
            response = requests.get(fr_url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.warning(f"Federal Register API returned status {response.status_code}")
            else:
                data = response.json()
                documents = data.get('results', [])
                
                self.logger.info(f"Federal Register API returned {len(documents)} documents")
                
                for doc in documents:
                    articles.append({
                        "title": doc['title'],
                        "url": doc['html_url'],
                        "content": doc.get('abstract', ''),
                        "published": doc['publication_date'],
                        "source": "Federal Register",
                        "type": doc['type'],
                        "agencies": doc.get('agencies', []),
                        "via_gov_api": True,
                        "collected_at": datetime.now().isoformat()
                    })
                    
                    if callback:
                        callback({
                            "type": "article",
                            "source": "Federal Register API",
                            "title": doc['title']
                        })
                        
        except Exception as e:
            self.logger.error(f"Federal Register API error: {e}")
        
        # Congress.gov API (if available)
        self.logger.debug("Congress.gov API not configured (requires API key)")
        
        self.logger.info(f"Government API collection complete: {len(articles)} documents")
        return articles

    def _collect_government_scrapers(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect from government websites via scraping."""
        self.logger.info("Starting government website scraping")

        articles: List[Dict[str, Any]] = []

        # Federal Register
        try:
            fr_docs = scrape_federal_register(start_date, end_date, limit=50)
            for doc in fr_docs:
                doc["via_gov_scraper"] = True
                articles.append(doc)
                if callback:
                    callback({"type": "article", "source": "Federal Register", "title": doc["title"]})
        except Exception as exc:
            self.logger.error(f"Federal Register scraping failed: {exc}")

        # White House presidential actions
        try:
            wh_docs = scrape_white_house_actions(start_date, end_date, limit=50)
            for doc in wh_docs:
                doc["via_gov_scraper"] = True
                articles.append(doc)
                if callback:
                    callback({"type": "article", "source": "White House", "title": doc["title"]})
        except Exception as exc:
            self.logger.error(f"White House scraping failed: {exc}")

        # Congress bills
        try:
            bill_docs = scrape_congress_bills(start_date, end_date, limit=50)
            for doc in bill_docs:
                doc["via_gov_scraper"] = True
                articles.append(doc)
                if callback:
                    callback({"type": "article", "source": "Congress.gov", "title": doc["title"]})
        except Exception as exc:
            self.logger.error(f"Congress bill scraping failed: {exc}")

        self.logger.info(f"Government scraping complete: {len(articles)} documents")
        return articles

    # ... (rest of the methods remain the same)

    def _collect_from_sitemap(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect articles from XML sitemaps with detailed logging."""
        sitemap_url = source.get("sitemap_url", source.get("url"))
        source_name = source.get("name", "Sitemap")
        articles = []
        
        self.logger.debug(f"Processing sitemap: {sitemap_url}")
        
        try:
            response = self.session.get(sitemap_url, timeout=self.request_timeout)
            if response.status_code != 200:
                self.logger.warning(f"Sitemap HTTP {response.status_code}: {sitemap_url}")
                return articles
            
            soup = BeautifulSoup(response.content, 'xml')
            urls = soup.find_all('url')
            
            self.logger.debug(f"Sitemap contains {len(urls)} URLs")
            
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
            
            self.logger.debug(f"Processing {len(url_data)} URLs from sitemap")
            
            # Process URLs
            processed = 0
            for url_str, mod_date in url_data:
                if self.cancelled or len(articles) >= limit:
                    break
                
                processed += 1
                
                # Check date range
                if mod_date:
                    if mod_date < start_date or mod_date > end_date:
                        self.logger.debug(f"URL {processed}: outside date range")
                        continue
                
                # Check if it's an article URL
                if not self._is_article_url(url_str):
                    self.logger.debug(f"URL {processed}: not an article URL")
                    continue
                
                self.logger.debug(f"URL {processed}: attempting extraction from {url_str}")
                
                # Extract article
                article = self._extract_article(url_str)
                if article and self._is_political(article.get("title", ""), article.get("content", "")):
                    article["source"] = source_name
                    article["bias_label"] = source.get("bias", "unknown")
                    
                    # Use lastmod date if no publish date found
                    if mod_date and not article.get("published"):
                        article["published"] = mod_date.isoformat()
                    
                    articles.append(article)
                    
                    if callback:
                        callback({"type": "article", "source": article["source"], "title": article.get("title")})
                    
                    self.logger.debug(f"URL {processed}: successfully extracted article")
                else:
                    self.logger.debug(f"URL {processed}: extraction failed or not political")
                
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            self.logger.error(f"Sitemap collection failed for {sitemap_url}: {e}")
            if callback:
                callback({"type": "error", "source": source_name, "message": str(e)})
        
        self.logger.info(f"Sitemap collection complete: {len(articles)} articles from {sitemap_url}")
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
            "via_google_news": article.get("via_google_news", False),
            "via_gdelt": article.get("via_gdelt", False),
            "via_gov_api": article.get("via_gov_api", False),
            "extraction_method": article.get("extraction_method")
        }

    def _store_article(self, article: Dict[str, Any]) -> str:
        """Store an article and return document ID, using crypto lineage if available."""
        if not self.document_repository:
            return ""

        content = article.get("content", "")
        metadata = self._create_metadata(article)

        try:
            if hasattr(self.document_repository, "store_document_with_crypto_chain"):
                doc_id = self.document_repository.store_document_with_crypto_chain(content, metadata)
                self.logger.debug(f"Stored document {doc_id} with crypto chain")
            else:
                doc_id = self.document_repository.store_document(content, metadata)
                self.logger.debug(f"Stored document {doc_id} (legacy)")
            return doc_id
        except Exception as e:
            self.logger.warning(f"Crypto storage failed: {e}")
            return self.document_repository.store_document(content, metadata)

    def _update_last_run(self):
        """Update last run timestamp."""
        try:
            with open(self.last_run_file, 'w') as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            self.logger.error(f"Error updating last run: {e}")

    def _load_collection_history(self) -> Dict[str, int]:
        """Load collection history from file."""
        if not os.path.exists(self.collection_history_file):
            return {}
        
        try:
            data = safe_json_load(self.collection_history_file, default=None)
            if data is not None:
                return data
            return {}
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
            if not safe_json_save(self.collection_history_file, history):
                raise IOError("save_failed")
        except Exception as e:
            self.logger.error(f"Error saving collection history: {e}")

    def _get_collection_mode(self, force_mode: Optional[str]) -> Tuple[str, datetime, datetime]:
        """Determine collection mode and date range."""
        current_time = datetime.now()
        
        if force_mode == "full":
            return ("full", self.inauguration_day, current_time)
        
        # Check if first run
        if not os.path.exists(self.last_run_file):
            # First run - collect everything since inauguration
            return ("first_run", self.inauguration_day, current_time)
        
        # Incremental
        try:
            with open(self.last_run_file, 'r') as f:
                last_run = datetime.fromisoformat(f.read().strip())
                
                # Add small overlap to avoid missing articles
                return ("incremental", last_run - timedelta(hours=1), current_time)
        except:
            return ("incremental", current_time - timedelta(days=1), current_time)

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
            new_source = {
                "url": source_data["url"],
                "type": source_data.get("type", "rss"),
                "bias": source_data.get("bias", "unknown"),
                "name": source_data.get("name", urlparse(source_data["url"]).netloc),
                "enabled": True,
                "limit": source_data.get("limit", 50),
                "sitemap_url": source_data.get("sitemap_url"),
                "archive_url": source_data.get("archive_url")
            }
            
            self.sources.append(new_source)
            
            # Save config
            self.config["content_collection"]["sources"] = self.sources
            
            self.logger.info(f"Added new source: {new_source['name']} ({new_source['type']})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding source: {e}")
            return False
