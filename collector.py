#!/usr/bin/env python3
"""
Night_watcher Content Collector
Gathers political content from RSS feeds with improved content extraction.
"""

import time
import logging
import hashlib
import re
import random
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import concurrent.futures
import traceback

import requests
import feedparser
from bs4 import BeautifulSoup

# Import newspaper3k (the standard version)
try:
    from newspaper import Article, Config
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.error("newspaper3k not installed. Run: pip install newspaper3k")

# Try to import trafilatura as a fallback
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logging.warning("trafilatura not installed. Install with: pip install trafilatura")

# Try to import cloudscraper for Cloudflare bypass
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    logging.warning("cloudscraper not installed. Cloudflare bypass not available. Install with: pip install cloudscraper")

logger = logging.getLogger(__name__)

class ContentCollector:
    """
    Collector for gathering political content from RSS feeds with efficient content extraction.
    """

    def __init__(self, config: Dict[str, Any], document_repository=None):
        """
        Initialize with configuration and optional document repository.
        """
        self.config = config
        self.document_repository = document_repository
        
        # Extract configuration
        cc = config.get("content_collection", {})
        self.article_limit = cc.get("article_limit", 5)
        self.sources = cc.get("sources", [])
        self.max_workers = cc.get("max_workers", 5)
        self.request_timeout = cc.get("request_timeout", 30)
        self.retry_count = cc.get("retry_count", 2)
        self.bypass_cloudflare = cc.get("bypass_cloudflare", True) and CLOUDSCRAPER_AVAILABLE
        self.delay_between_requests = cc.get("delay_between_requests", 2.0)
        
        # Political/governmental keywords for filtering
        self.govt_keywords = set(kw.lower() for kw in cc.get("govt_keywords", [
            # Executive branch
            "executive order", "administration", "white house", "president", "presidential",
            "cabinet", "secretary", "department of", "federal agency", "oval office",
            "executive action", "executive branch", "commander in chief", "veto",
            
            # Legislative branch
            "congress", "congressional", "senate", "senator", "house of representatives",
            "representative", "legislation", "bill", "law", "act", "resolution",
            "committee", "subcommittee", "speaker of the house", "majority leader",
            "minority leader", "filibuster", "caucus", "legislative", "lawmaker",
            
            # Judicial branch
            "supreme court", "federal court", "appeals court", "district court",
            "judge", "justice", "judicial", "ruling", "decision", "opinion",
            "constitutional", "unconstitutional", "precedent", "litigation",
            
            # Elections and democracy
            "election", "campaign", "candidate", "voter", "voting", "ballot",
            "primary", "caucus", "electoral", "democracy", "democratic", "republic",
            "poll", "polling", "constituency", "redistricting", "gerrymandering",
            
            # Policy and governance
            "policy", "regulation", "regulatory", "federal", "government", "governance",
            "politics", "political", "partisan", "bipartisan", "nonpartisan",
            "public policy", "domestic policy", "foreign policy", "national security",
            
            # Specific agencies and departments
            "state department", "defense department", "pentagon", "justice department",
            "treasury", "homeland security", "education department", "energy department",
            "fbi", "cia", "nsa", "dhs", "doj", "epa", "fda", "cdc", "fcc",
            
            # Political parties and movements
            "republican", "democrat", "democratic", "gop", "conservative", "liberal",
            "progressive", "libertarian", "independent", "tea party", "maga",
            
            # Key political figures (titles)
            "governor", "mayor", "attorney general", "chief justice", "ambassador",
            "diplomat", "lobbyist", "spokesman", "spokeswoman", "spokesperson",
            
            # Government actions
            "impeachment", "nomination", "confirmation", "appointment", "investigation",
            "hearing", "testimony", "subpoena", "executive privilege", "pardon",
            "sanction", "treaty", "trade deal", "tariff", "embargo",
            
            # Constitutional terms
            "constitution", "amendment", "bill of rights", "civil rights", "civil liberties",
            "first amendment", "second amendment", "constitutional crisis",
            
            # Budget and fiscal
            "budget", "appropriation", "spending", "deficit", "debt ceiling",
            "government shutdown", "continuing resolution", "omnibus", "reconciliation",
            
            # Other relevant terms
            "whistleblower", "classified", "declassified", "national interest",
            "state of the union", "executive session", "recess appointment",
            "confirmation hearing", "oversight", "accountability", "transparency"
        ]))
        
        # User agents for requests
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
        ]
        
        # URLs to skip (problematic or non-political)
        self.skip_url_patterns = [
            r'/entertainment/',
            r'/music/',
            r'/sports/',
            r'/lifestyle/',
            r'/style/',
            r'/food/',
            r'/travel/',
            r'/celebrity/',
            r'/culture/',
            r'/arts/',
            r'/movies/',
            r'/tv/',
            r'/gaming/',
            r'/technology/gadgets/',
            r'/health/personal/',
            r'/real-estate/',
            r'/cars/',
            r'diddy',
            r'sean-combs'
        ]
        
        # Initialize logger first
        self.logger = logging.getLogger("ContentCollector")
        
        # Initialize session objects
        self._init_sessions()
    
    def _init_sessions(self):
        """Initialize HTTP sessions for requests and cloudscraper."""
        self.session = requests.Session()
        
        # Rotate user agent for each session
        current_user_agent = random.choice(self.user_agents)
        
        self.session.headers.update({
            'User-Agent': current_user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
        })
        
        # Set up retries
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Initialize cloudscraper if available
        if self.bypass_cloudflare:
            try:
                self.cloudscraper_session = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'windows',
                        'mobile': False,
                        'desktop': True
                    },
                    delay=10,  # Add delay to appear more human-like
                    debug=False
                )
                self.logger.info("Cloudflare bypass initialized")
            except Exception as e:
                self.logger.error(f"Error initializing Cloudflare bypass: {e}")
                self.bypass_cloudflare = False

    def _should_skip_url(self, url: str) -> bool:
        """
        Check if URL should be skipped based on patterns.
        """
        url_lower = url.lower()
        for pattern in self.skip_url_patterns:
            if re.search(pattern, url_lower):
                self.logger.info(f"Skipping non-political URL pattern '{pattern}': {url}")
                return True
        return False

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: collect and process content from RSS feeds.
        
        Args:
            input_data: Dict with optional keys:
                - 'sources': override list of sources
                - 'limit': max articles per source
                - 'start_date', 'end_date': ISO strings or datetime objects
                - 'document_repository': override document repository
                - 'store_documents': boolean flag to store in repository
                
        Returns:
            Dict with collected articles and document IDs
        """
        # Extract configuration from input
        sources = input_data.get("sources", self.sources)
        limit = input_data.get("limit", self.article_limit)
        doc_repo = input_data.get("document_repository", self.document_repository)
        store_docs = input_data.get("store_documents", doc_repo is not None)
        max_workers = input_data.get("max_workers", self.max_workers)
        
        # Parse date strings if provided
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        
        if isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date)
            except ValueError:
                self.logger.warning(f"Invalid start_date format: {start_date}")
                start_date = None
                
        if isinstance(end_date, str):
            try:
                end_date = datetime.fromisoformat(end_date)
            except ValueError:
                self.logger.warning(f"Invalid end_date format: {end_date}")
                end_date = None
        
        self.logger.info(f"Starting collection from {len(sources)} sources, limit {limit}")
        
        # Initialize results
        all_articles = []
        document_ids = []
        successful_sources = 0
        failed_sources = 0
        
        # Process sources sequentially instead of in parallel to avoid rate limiting
        for source in sources:
            if source.get("type", "").lower() != "rss":
                continue
                
            try:
                articles = self._collect_from_rss(source, limit, start_date, end_date)
                
                # Skip if no articles were found
                if not articles:
                    self.logger.warning(f"No articles found from source: {source.get('url')}")
                    failed_sources += 1
                    continue
                
                # Generate document IDs
                for article in articles:
                    article["id"] = self._generate_document_id(article)
                
                # Store in document repository if available
                if store_docs and doc_repo:
                    for article in articles:
                        try:
                            doc_id = doc_repo.store_document(
                                article["content"],
                                self._create_metadata(article)
                            )
                            document_ids.append(doc_id)
                            article["document_id"] = doc_id
                        except Exception as e:
                            self.logger.error(f"Error storing document: {e}")
                
                all_articles.extend(articles)
                successful_sources += 1
                
                # Add delay between sources
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                self.logger.error(f"Error processing source {source.get('url')}: {e}")
                self.logger.error(traceback.format_exc())
                failed_sources += 1
        
        # Log results
        self.logger.info(f"Collection complete: {len(all_articles)} articles collected")
        
        return {
            "articles": all_articles,
            "document_ids": document_ids,
            "status": {
                "successful_sources": successful_sources,
                "failed_sources": failed_sources,
                "articles_collected": len(all_articles),
                "timestamp": datetime.now().isoformat()
            }
        }

    def _collect_from_rss(
        self, 
        source: Dict[str, Any],
        limit: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect articles from an RSS feed with date filtering.
        """
        url = source.get("url", "")
        bias = source.get("bias", "unknown")
        
        self.logger.info(f"Collecting from RSS: {url}")
        
        try:
            # Parse the feed with timeout
            feed = feedparser.parse(url, agent=random.choice(self.user_agents))
            
            if feed.get("bozo", 0) == 1:
                self.logger.warning(f"Feed parse error for {url}: {feed.get('bozo_exception')}")
                # Continue anyway - some feeds have minor issues but still work
            
            # Check if we got entries
            entries = feed.get("entries", [])
            if not entries:
                self.logger.warning(f"No entries found in feed: {url}")
                return []
            
            # Sort entries by publication date (newest first)
            entries.sort(key=lambda x: x.get("published_parsed", time.gmtime(0)), reverse=True)
            
            collected = []
            
            for entry in entries:
                if len(collected) >= limit:
                    break
                
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                
                if not link or not title:
                    continue
                
                # Skip non-political URLs early
                if self._should_skip_url(link):
                    continue
                
                # Publication date filtering
                pub_date = None
                if entry.get("published_parsed"):
                    try:
                        pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        
                        # Skip if outside date range
                        if start_date and pub_date < start_date:
                            continue
                        if end_date and pub_date > end_date:
                            continue
                    except Exception as e:
                        self.logger.warning(f"Error parsing date for {title}: {e}")
                
                # Filter for political/government content early using title and tags
                tags = entry.get("tags", [])
                if not self._is_government_related(title, "", tags):
                    self.logger.debug(f"Skipping non-governmental article (pre-fetch): {title}")
                    continue
                
                self.logger.info(f"Fetching article: {title}")
                
                # Extract content with timeout protection
                try:
                    content, article_data = self._extract_article_content(link, title, pub_date)
                    
                    # Add delay after each article fetch
                    time.sleep(self.delay_between_requests)
                    
                except requests.exceptions.Timeout:
                    self.logger.warning(f"Timeout fetching article: {title} - skipping")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error fetching article: {title} - {str(e)}")
                    continue
                
                # Skip if content extraction failed
                if not content or not self._is_valid_content(content):
                    self.logger.warning(f"Failed to extract valid content for: {title}")
                    continue
                
                # Re-check for political content with actual content
                if not self._is_government_related(title, content, tags):
                    self.logger.info(f"Skipping non-governmental article (post-fetch): {title}")
                    continue
                
                # Determine source name
                source_name = feed.feed.get("title") or urlparse(url).netloc.replace("www.", "")
                
                # Create article data
                article_data.update({
                    "title": title,
                    "url": link,
                    "source": source_name,
                    "bias_label": bias,
                    "published": pub_date.isoformat() if pub_date else article_data.get("published"),
                    "content": content,
                    "tags": [t.get("term") for t in tags if t.get("term")],
                    "collected_at": datetime.now().isoformat()
                })
                
                collected.append(article_data)
                self.logger.info(f"Collected: {title} ({len(content)} chars)")
            
            return collected
            
        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def _extract_article_content(self, url: str, title: str = None, pub_date: Optional[datetime] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extract article content using multiple methods with fallbacks.
        
        Returns:
            Tuple of (content, article_data_dict)
        """
        article_data = {
            "authors": [],
            "published": None,
            "top_image": None,
            "images": [],
            "movies": []
        }
        
        content = None
        
        # Check if this is a problematic domain that blocks bots
        problematic_domains = ['thehill.com', 'washingtonpost.com', 'wsj.com', 'nytimes.com']
        if any(domain in url.lower() for domain in problematic_domains):
            self.logger.warning(f"Skipping known bot-blocking domain: {url}")
            # Return minimal content from RSS feed if available
            if title:
                content = f"[Content not available due to bot protection. Title: {title}]"
                return content, article_data
        
        # Method 1: Try trafilatura first (if available)
        if TRAFILATURA_AVAILABLE:
            try:
                content = self._extract_with_trafilatura(url)
                if self._is_valid_content(content):
                    self.logger.debug(f"Successfully extracted with trafilatura: {url}")
                    # Update article_data with any missing info
                    if pub_date:
                        article_data["published"] = pub_date.isoformat()
                    return content, article_data
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    self.logger.warning(f"403 Forbidden for {url} - likely bot protection")
                    return None, article_data
            except Exception as e:
                self.logger.debug(f"Trafilatura extraction failed: {e}")
        
        # Method 2: Try newspaper3k
        if NEWSPAPER_AVAILABLE:
            try:
                content = self._extract_with_newspaper3k(url, article_data)
                if self._is_valid_content(content):
                    self.logger.debug(f"Successfully extracted with newspaper3k: {url}")
                    return content, article_data
            except Exception as e:
                if "403" in str(e):
                    self.logger.warning(f"403 Forbidden for {url} - likely bot protection")
                    return None, article_data
                self.logger.debug(f"Newspaper3k extraction failed: {e}")
        
        # Method 3: Fall back to BeautifulSoup
        try:
            self.logger.info(f"Trying BeautifulSoup fallback for {url}")
            content = self._extract_with_beautifulsoup(url)
            
            # Update article_data with any missing info
            if article_data.get("published") is None and pub_date:
                article_data["published"] = pub_date.isoformat()
                
            if not article_data.get("authors") and title:
                # Try to extract author from title as last resort
                possible_author = self._extract_author_from_title(title)
                if possible_author:
                    article_data["authors"] = [possible_author]
            
            if self._is_valid_content(content):
                self.logger.debug(f"Successfully extracted with BeautifulSoup: {url}")
                return content, article_data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                self.logger.warning(f"403 Forbidden for {url} - likely bot protection")
                return None, article_data
            self.logger.error(f"BeautifulSoup extraction failed: {e}")
        except Exception as e:
            self.logger.error(f"BeautifulSoup extraction failed: {e}")
        
        # If all methods failed
        self.logger.error(f"All content extraction methods failed for {url}")
        return None, article_data

    def _extract_with_trafilatura(self, url: str) -> Optional[str]:
        """
        Extract content using trafilatura (most reliable method).
        """
        # Fetch the page
        if self.bypass_cloudflare:
            try:
                response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                html = response.text
            except:
                response = self.session.get(url, timeout=self.request_timeout)
                html = response.text
        else:
            response = self.session.get(url, timeout=self.request_timeout)
            html = response.text
        
        # Extract with trafilatura
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
            target_language='en'
        )
        
        return content

    def _extract_with_newspaper3k(self, url: str, article_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract content using newspaper3k.
        """
        # Configure article
        config = Config()
        config.browser_user_agent = random.choice(self.user_agents)
        config.request_timeout = self.request_timeout
        config.memoize_articles = False
        config.fetch_images = False  # Disable to speed up
        config.language = 'en'
        
        # Create article instance
        article = Article(url, config=config)
        
        try:
            # First try with cloudscraper if available
            if self.bypass_cloudflare:
                try:
                    response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                    if response.status_code == 200:
                        article.download(input_html=response.text)
                    else:
                        raise Exception(f"Status code {response.status_code}")
                except:
                    # Fall back to regular download
                    article.download()
            else:
                article.download()
            
            article.parse()
            
            # Extract content and metadata
            content = article.text or ""
            
            # Update article_data with extracted metadata
            if article.publish_date:
                article_data["published"] = article.publish_date.isoformat()
            
            article_data["authors"] = article.authors if article.authors else []
            article_data["top_image"] = article.top_image
            
            return content
            
        except Exception as e:
            self.logger.debug(f"Newspaper3k download/parse error: {e}")
            raise

    def _extract_with_beautifulsoup(self, url: str) -> str:
        """
        Extract content using BeautifulSoup targeting article containers.
        """
        # Fetch the page
        try:
            if self.bypass_cloudflare:
                try:
                    response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                except:
                    response = self.session.get(url, timeout=self.request_timeout)
            else:
                response = self.session.get(url, timeout=self.request_timeout)
            
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            raise
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find JSON-LD structured data first
        json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0]
                if isinstance(data, dict):
                    # Check for NewsArticle or Article type
                    if data.get('@type') in ['NewsArticle', 'Article'] and data.get('articleBody'):
                        return data.get('articleBody')
            except:
                continue
        
        # Try common article container selectors
        selectors = [
            'article',
            '[itemprop="articleBody"]',
            '.article-content',
            '.article-body',
            '.story-body',
            '.story-content',
            '.entry-content',
            '.post-content',
            'div.content',
            'div.text',
            'main',
            '[role="main"]'
        ]
        
        content = ""
        
        for selector in selectors:
            elements = soup.select(selector)
            if not elements:
                continue
                
            for element in elements:
                # Clone to avoid modifying original
                container = element
                
                # Remove unwanted elements
                for unwanted in container.select('aside, .related, .advertisement, nav, header, footer, .comments'):
                    unwanted.decompose()
                
                # Extract paragraphs
                paragraphs = container.find_all('p')
                if paragraphs:
                    text = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                    if self._is_valid_content(text):
                        content = text
                        break
            
            if content:
                break
        
        # Final fallback: get all paragraphs
        if not content:
            all_paragraphs = soup.find_all('p')
            # Filter out short paragraphs
            paragraphs = [p.get_text().strip() for p in all_paragraphs 
                         if len(p.get_text().strip()) > 50]
            if paragraphs:
                content = "\n\n".join(paragraphs)
        
        return content
    
    def _extract_author_from_title(self, title: str) -> Optional[str]:
        """Extract possible author from title as last resort."""
        # Look for patterns like "By Author Name" or "Author Name reports"
        by_match = re.search(r'By\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', title)
        if by_match:
            return by_match.group(1)
        
        # Look for patterns with common reporting verbs
        report_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:reports|writes|says|claims)', title)
        if report_match:
            return report_match.group(1)
        
        return None

    def _is_valid_content(self, content: Optional[str]) -> bool:
        """
        Check if content meets quality criteria.
        """
        if not content:
            return False
            
        # Remove extra whitespace for accurate counting
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Check minimum length
        if len(content) < 200:
            return False
        
        # Check word count
        word_count = len(content.split())
        if word_count < 50:  # Lowered threshold
            return False
        
        # Check for too many special characters (parsing errors)
        special_char_ratio = len(re.findall(r'[^\w\s.,;:!?()\'"/-]', content)) / len(content)
        if special_char_ratio > 0.1:  # More than 10% special characters
            return False
        
        # Check for repeated characters (parsing errors)
        if re.search(r'(.)\1{10,}', content):  # Same character repeated 10+ times
            return False
            
        return True

    def _is_government_related(
        self,
        title: str,
        content: str,
        tags: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Check if article is related to government or politics.
        """
        # 1) Check for government/politics tags
        if tags:
            political_tags = {"politics", "government", "us politics", "policy", "election", "washington"}
            for tag in tags:
                tag_term = tag.get("term", "").lower()
                if tag_term in political_tags or any(kw in tag_term for kw in self.govt_keywords):
                    return True
        
        # 2) Check for keywords in title and content preview
        sample_text = (title + " " + content[:2000]).lower()
        
        # Count keyword matches
        keyword_matches = 0
        for keyword in self.govt_keywords:
            if keyword in sample_text:
                keyword_matches += 1
                if keyword_matches >= 2:  # At least 2 keywords
                    return True
        
        # 3) Check for current political figures
        political_figures = {"trump", "biden", "harris", "pence", "pelosi", "mcconnell", "schumer"}
        for figure in political_figures:
            if figure in sample_text:
                return True
        
        return False

    def _generate_document_id(self, article: Dict[str, Any]) -> str:
        """
        Generate a simple document ID based on URL and timestamp.
        """
        url = article.get("url", "")
        timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
        
        # Create URL hash
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
        
        # Combine with timestamp
        return f"{url_hash}_{timestamp}"

    def _create_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata dictionary for document storage.
        """
        return {
            "title": article.get("title", "Untitled"),
            "url": article.get("url", ""),
            "source": article.get("source", "Unknown"),
            "bias_label": article.get("bias_label", "unknown"),
            "published": article.get("published"),
            "authors": article.get("authors", []),
            "tags": article.get("tags", []),
            "top_image": article.get("top_image"),
            "images": article.get("images", []),
            "movies": article.get("movies", []),
            "collected_at": article.get("collected_at", datetime.now().isoformat()),
            "id": article.get("id")
        }