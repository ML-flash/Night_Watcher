#!/usr/bin/env python3
"""
Night_watcher Content Collector
Gathers political content from RSS feeds with improved content extraction using newspaper4k.
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

# Import newspaper4k instead of newspaper3k
try:
    import newspaper
    from newspaper import Article
    from newspaper.configuration import Configuration
    # Check if we're using newspaper4k
    if not hasattr(newspaper, "article"):
        logging.warning("Using newspaper3k instead of newspaper4k. Consider upgrading for better extraction.")
except ImportError:
    logging.error("newspaper4k not installed. Run: pip install newspaper4k")
    raise

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
        self.request_timeout = cc.get("request_timeout", 15)
        self.retry_count = cc.get("retry_count", 2)
        self.bypass_cloudflare = cc.get("bypass_cloudflare", True) and CLOUDSCRAPER_AVAILABLE
        
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
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/100.0.4896.127 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_4) AppleWebKit/605.1.15 Version/15.4 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/100.0.1185.50 Safari/537.36'
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
            r'diddy',  # Specific pattern causing issues
            r'sean-combs'
        ]
        
        # Initialize logger first
        self.logger = logging.getLogger("ContentCollector")
        
        # Initialize session objects
        self._init_sessions()
    
    def _init_sessions(self):
        """Initialize HTTP sessions for requests and cloudscraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive'
        })
        
        # Initialize cloudscraper if available
        if self.bypass_cloudflare:
            try:
                self.cloudscraper_session = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'windows',
                        'mobile': False
                    }
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
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each source
            future_to_source = {
                executor.submit(
                    self._collect_from_rss, source, limit, start_date, end_date
                ): source for source in sources if source.get("type", "").lower() == "rss"
            }
            
            # Process completed futures
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    articles = future.result()
                    
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
            # Parse the feed
            feed = feedparser.parse(url)
            if feed.get("bozo", 0) == 1:
                self.logger.warning(f"Feed parse error for {url}: {feed.get('bozo_exception')}")
            
            # Sort entries by publication date (newest first)
            entries = feed.get("entries", [])
            entries.sort(key=lambda x: x.get("published_parsed", 0), reverse=True)
            
            collected = []
            
            for entry in entries:
                if len(collected) >= limit:
                    break
                
                title = entry.get("title", "Untitled")
                link = entry.get("link")
                
                if not link:
                    continue
                
                # Skip non-political URLs early
                if self._should_skip_url(link):
                    continue
                
                # Publication date filtering
                pub_date = None
                if entry.get("published_parsed"):
                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    
                    # Skip if outside date range
                    if start_date and pub_date < start_date:
                        continue
                    if end_date and pub_date > end_date:
                        continue
                
                # Filter for political/government content early using title and tags
                tags = entry.get("tags", [])
                if not self._is_government_related(title, "", tags):
                    self.logger.info(f"Skipping non-governmental article (pre-fetch): {title}")
                    continue
                
                self.logger.info(f"Fetching article: {title}")
                
                # Extract content with timeout protection
                try:
                    content, article_data = self._extract_article_content(link, title, pub_date)
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
        Extract article content using newspaper4k with enhanced extraction and fallbacks.
        
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
        
        try:
            # Primary extraction using newspaper4k
            content = self._extract_with_newspaper4k(url, article_data)
            
            # Check if extraction was successful
            if self._is_valid_content(content):
                return content, article_data
            
            # Fall back to BeautifulSoup extraction
            self.logger.info(f"Primary extraction failed for {url}, trying fallback")
            content = self._extract_with_beautifulsoup(url)
            
            # Update article_data with any missing info (keep what we got from newspaper)
            if article_data.get("published") is None and pub_date:
                article_data["published"] = pub_date.isoformat()
                
            if not article_data.get("authors") and title:
                # Try to extract author from title as last resort
                possible_author = self._extract_author_from_title(title)
                if possible_author:
                    article_data["authors"] = [possible_author]
            
            return content, article_data
            
        except Exception as e:
            self.logger.error(f"Content extraction error for {url}: {e}")
            self.logger.error(traceback.format_exc())
            return None, article_data

    def _extract_with_newspaper4k(self, url: str, article_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract content using newspaper4k with improved configuration.
        Updates article_data with extracted metadata.
        """
        # Configure article with better settings
        config = Configuration()
        config.browser_user_agent = random.choice(self.user_agents)
        config.fetch_images = True  # We want to get images
        config.memoize_articles = False
        config.request_timeout = self.request_timeout
        
        # Add headers for better success rate
        config.headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive'
        }
        
        # Use the convenience method from newspaper4k if available
        try:
            # First try with cloudscraper if cloudflare bypass is enabled
            if self.bypass_cloudflare:
                try:
                    # Get HTML content with cloudscraper with timeout
                    response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                    if response.status_code == 200:
                        # Create article and use the pre-fetched HTML
                        article = Article(url, config=config)
                        article.download(input_html=response.text)
                        article.parse()
                    else:
                        # Fall back to regular download if cloudscraper fails
                        raise Exception(f"Cloudscraper failed with status code {response.status_code}")
                except Exception as e:
                    self.logger.warning(f"Cloudflare bypass failed, using regular download: {e}")
                    # Try using newspaper4k's article convenience function
                    article = newspaper.article(url, config=config)
            else:
                # Use newspaper4k's article convenience function if available
                article = newspaper.article(url, config=config)
        except AttributeError:
            # Fallback for newspaper3k (doesn't have the article convenience function)
            article = Article(url, config=config)
            
            # Download with retries and timeout protection
            retry_count = 0
            while retry_count < self.retry_count:
                try:
                    article.download()
                    break
                except requests.exceptions.Timeout:
                    raise  # Re-raise timeout errors, don't retry
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self.retry_count:
                        raise
                    self.logger.warning(f"Download retry {retry_count}/{self.retry_count}: {e}")
                    time.sleep(2)
            
            article.parse()
        
        # Extract content and metadata
        content = article.text or ""
        
        # Update article_data with extracted metadata
        if article.publish_date:
            article_data["published"] = article.publish_date.isoformat()
        
        article_data["authors"] = article.authors if article.authors else []
        article_data["top_image"] = article.top_image
        article_data["images"] = article.images if hasattr(article, "images") else []
        article_data["movies"] = article.movies if hasattr(article, "movies") else []
        
        # Throttle requests
        time.sleep(1.5)
        
        return content

    def _extract_with_beautifulsoup(self, url: str) -> str:
        """
        Extract content using BeautifulSoup targeting article containers.
        Enhanced with better selectors and more robust extraction.
        """
        # Use cloudscraper if available and enabled
        if self.bypass_cloudflare:
            try:
                response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                if response.status_code != 200:
                    raise Exception(f"Cloudscraper request failed with status code {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Cloudflare bypass failed, using regular session: {e}")
                headers = {
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                response = self.session.get(url, headers=headers, timeout=self.request_timeout)
        else:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            response = self.session.get(url, headers=headers, timeout=self.request_timeout)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find JSON-LD structured data first (most reliable)
        json_ld = soup.find('script', {'type': 'application/ld+json'})
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, list):
                    data = data[0]
                if isinstance(data, dict) and data.get('@type') == 'NewsArticle' and data.get('articleBody'):
                    return data.get('articleBody')
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Try common article container selectors
        selectors = [
            'article', 
            '[itemprop="articleBody"]',
            '.article-content', 
            '.content-body',
            '.story-body',
            '#article-body',
            '.story-content',
            '.article__content',
            '.article-body',
            '.entry-content',
            '.post-content',
            '#content-body',
            '.main-content',
            '[role="main"]',
            '.article'
        ]
        
        content = ""
        
        # Try each selector until we find content
        for selector in selectors:
            container = soup.select_one(selector)
            if container:
                # Remove unwanted elements
                for unwanted in container.select('aside, .related, .social, .comments, nav, .nav, header, footer, .ad, script, style, form, .subscription, .ad-container, .poll, .pullquote'):
                    unwanted.decompose()
                
                # Extract text with paragraph breaks
                paragraphs = container.find_all('p')
                if paragraphs:
                    content = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                else:
                    content = container.get_text(separator='\n\n', strip=True)
                
                if self._is_valid_content(content):
                    break
        
        # Final fallback: just get all paragraphs if nothing else worked
        if not self._is_valid_content(content):
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Filter out very short paragraphs that are likely navigation, comments, etc.
                paragraphs = [p for p in paragraphs if len(p.get_text().strip()) > 40]
                content = "\n\n".join(p.get_text().strip() for p in paragraphs)
        
        # Throttle requests
        time.sleep(1.5)
        
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
        Enhanced with better validation rules.
        """
        if not content or len(content) < 200:
            return False
        
        # Check for multiple paragraphs (at least 2)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return False
        
        # Check for sentence structure (rough approximation)
        sentences = re.findall(r'[.!?]', content)
        if len(sentences) < 5:  # Need more than a few sentences
            return False
        
        # Check for too many consecutive special characters (indicative of parsing errors)
        if re.search(r'[^\w\s.,;:!?()\'"-]{5,}', content):
            return False
            
        # Check word count (at least 100 words for a valid article)
        word_count = len(re.findall(r'\b\w+\b', content))
        if word_count < 100:
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
            political_tags = {"politics", "government", "us politics", "policy", "election"}
            for tag in tags:
                tag_term = tag.get("term", "").lower()
                if tag_term in political_tags or any(kw in tag_term for kw in self.govt_keywords):
                    return True
        
        # 2) Check for keywords in title and content preview
        sample_text = (title + " " + content[:2000]).lower()
        
        # Check each keyword
        for keyword in self.govt_keywords:
            if keyword in sample_text:
                return True
        
        # 3) Check for current president and political figures
        political_figures = {"trump", "biden", "harris", "president", "secretary", "senator", "representative", "governor"}
        for figure in political_figures:
            if figure in sample_text:
                return True
        
        return False

    def _generate_document_id(self, article: Dict[str, Any]) -> str:
        """
        Generate a simple document ID based on URL and timestamp.
        """
        url = article.get("url", "")
        timestamp = int(time.time())
        
        # Create URL hash
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:16]
        
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