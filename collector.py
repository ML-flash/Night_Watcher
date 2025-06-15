#!/usr/bin/env python3
"""
Night_watcher Unified Content Collector
Enhanced with working historical collection methods.
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
    """Enhanced collector with working historical collection methods."""

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
        self.max_gap_days = cc.get("max_gap_days", 30)
        self.gap_detection_enabled = cc.get("gap_detection_enabled", True)
        self.use_google_news = cc.get("use_google_news", True)
        self.use_gdelt = cc.get("use_gdelt", True)

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
        """Main collection method with enhanced historical collection."""
        self.cancelled = False
        
        # Determine collection mode and date range
        mode, start_date, end_date = self._get_collection_mode(force_mode)
        self.logger.info(f"Collection mode: {mode}, Range: {start_date} to {end_date}")
        
        all_articles = []
        document_ids = []
        
        # 1. Collect from regular sources (RSS, etc.)
        if callback:
            callback({"type": "status", "message": "Collecting from RSS sources..."})
        
        regular_result = self._collect_date_range(start_date, end_date, mode, callback)
        all_articles.extend(regular_result.get("articles", []))
        document_ids.extend(regular_result.get("document_ids", []))
        
        # 2. Collect from Google News if enabled
        if self.use_google_news:
            if callback:
                callback({"type": "status", "message": "Collecting from Google News historical search..."})
            
            google_articles = self._collect_google_news_historical(start_date, end_date, callback)
            
            # Process and store Google News articles
            for article in google_articles:
                article["id"] = self._generate_id(article["url"])
                
                if self.document_repository:
                    doc_id = self.document_repository.store_document(
                        article["content"],
                        self._create_metadata(article)
                    )
                    document_ids.append(doc_id)
                    article["document_id"] = doc_id
                
                all_articles.append(article)
        
        # 3. Collect from GDELT if enabled
        if self.use_gdelt:
            if callback:
                callback({"type": "status", "message": "Collecting from GDELT database..."})
            
            gdelt_articles = self._collect_gdelt_comprehensive(start_date, end_date, callback)
            
            # Process and store GDELT articles
            for article in gdelt_articles:
                article["id"] = self._generate_id(article["url"])
                
                if self.document_repository:
                    doc_id = self.document_repository.store_document(
                        article.get("content", ""),
                        self._create_metadata(article)
                    )
                    document_ids.append(doc_id)
                    article["document_id"] = doc_id
                
                all_articles.append(article)
        
        # 4. Collect from Government APIs
        if callback:
            callback({"type": "status", "message": "Collecting from government APIs..."})
        
        gov_articles = self._collect_government_apis(start_date, end_date, callback)
        
        for article in gov_articles:
            article["id"] = self._generate_id(article["url"])
            
            if self.document_repository:
                doc_id = self.document_repository.store_document(
                    article.get("content", ""),
                    self._create_metadata(article)
                )
                document_ids.append(doc_id)
                article["document_id"] = doc_id
            
            all_articles.append(article)
        
        # Update collection history
        self._update_collection_history(all_articles)
        
        # Deduplicate articles
        seen_urls = set()
        unique_articles = []
        unique_doc_ids = []
        
        for i, article in enumerate(all_articles):
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)
                if i < len(document_ids):
                    unique_doc_ids.append(document_ids[i])
        
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
                "timestamp": datetime.now().isoformat()
            }
        }

    def _collect_google_news_historical(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect historical news using Google News search with date filters."""
        
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
        
        for query in search_queries:
            if self.cancelled:
                break
                
            try:
                # Build Google News RSS URL with date filter
                encoded_query = quote(f'{query} after:{start_str} before:{end_str}')
                google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                
                # Parse the RSS feed
                feed = feedparser.parse(google_news_url)
                
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
                        
                        if callback:
                            callback({
                                "type": "article",
                                "source": "Google News Search",
                                "title": article.get("title"),
                                "query": query
                            })
                    
                    # Small delay to be respectful
                    time.sleep(random.uniform(0.5, 1.5))
                
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
        
        return all_articles

    def _collect_gdelt_comprehensive(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect from GDELT's comprehensive news database."""
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
        
        for query in queries:
            if self.cancelled:
                break
                
            try:
                params = {
                    'query': query,
                    'mode': 'artlist',
                    'maxrecords': 100,
                    'timespan': f'{days_back}d',
                    'sort': 'hybridrel',
                    'format': 'json'
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                data = response.json()
                
                for article_data in data.get('articles', []):
                    # Extract the full article
                    article = self._extract_article(article_data['url'], article_data['title'])
                    if article and self._is_political(article.get("title", ""), article.get("content", "")):
                        article['via_gdelt'] = True
                        article['gdelt_query'] = query
                        article['source'] = article_data.get('domain', 'Unknown')
                        articles.append(article)
                        
                        if callback:
                            callback({
                                "type": "article",
                                "source": "GDELT",
                                "title": article.get("title")
                            })
                
                time.sleep(1)  # Be nice to GDELT
                
            except Exception as e:
                self.logger.error(f"GDELT error for '{query}': {e}")
                
        return articles

    def _collect_government_apis(self, start_date: datetime, end_date: datetime, callback=None) -> List[Dict[str, Any]]:
        """Collect from government APIs - these ALWAYS work."""
        articles = []
        
        # Federal Register API
        try:
            fr_url = "https://www.federalregister.gov/api/v1/documents"
            params = {
                "conditions[publication_date][gte]": start_date.strftime('%Y-%m-%d'),
                "conditions[publication_date][lte]": end_date.strftime('%Y-%m-%d'),
                "per_page": 100,
                "order": "newest"
            }
            
            response = requests.get(fr_url, params=params, timeout=30)
            data = response.json()
            
            for doc in data.get('results', []):
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
        try:
            # Note: Congress.gov requires API key
            # This is a placeholder for the structure
            congress_url = "https://api.congress.gov/v3/bills"
            # Would need API key in headers
            
        except Exception as e:
            self.logger.debug(f"Congress API not configured: {e}")
            
        return articles

    def _collect_date_range(self, start_date: datetime, end_date: datetime, mode: str, callback=None) -> Dict[str, Any]:
        """Collect content for a specific date range."""
        all_articles = []
        document_ids = []
        
        for source in self.sources:
            if not source.get("enabled", True):
                continue
                
            # Skip archive sources that are timing out
            if source.get("type") == "archive":
                self.logger.info(f"Skipping archive source {source.get('name')} - using alternative methods")
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

    def _collect_from_source(self, source: Dict[str, Any], start_date: datetime, end_date: datetime, limit: int, callback=None) -> List[Dict[str, Any]]:
        """Collect from a single source."""
        url = source.get("url", "")
        source_type = source.get("type", "rss")

        if source_type == "article":
            # Direct article
            article = self._extract_article(url)
            if article:
                if callback:
                    callback({"type": "article", "source": source.get("name", url), "title": article.get("title")})
            return [article] if article else []
        
        elif source_type == "sitemap":
            # Parse sitemap for article URLs
            return self._collect_from_sitemap(source, start_date, end_date, limit, callback)
        
        else:  # RSS feed (default)
            try:
                # Parse RSS feed
                feed = feedparser.parse(url, agent=random.choice(self.user_agents))
                articles = self._process_feed_entries(feed.entries, source, url, start_date, end_date, limit, callback)
                return articles
                
            except Exception as e:
                self.logger.error(f"RSS collection failed for {url}: {e}")
                if callback:
                    callback({"type": "error", "source": source.get("name", url), "message": str(e)})
                return []

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

    def _extract_article(self, url: str, title: str = None, pub_date: datetime = None) -> Optional[Dict[str, Any]]:
        """Extract article content with enhanced extraction."""
        try:
            # Use cloudscraper if available for better success rate
            session = self.cloudscraper if self.cloudscraper else self.session
            
            # Try trafilatura first (it's generally more reliable)
            if TRAFILATURA_AVAILABLE:
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
            response = session.get(url, timeout=self.request_timeout)
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
            "via_google_news": article.get("via_google_news", False),
            "via_gdelt": article.get("via_gdelt", False),
            "via_gov_api": article.get("via_gov_api", False)
        }

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