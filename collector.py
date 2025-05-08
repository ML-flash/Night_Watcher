#!/usr/bin/env python3
"""
Night_watcher Content Collector
Gathers political content from RSS feeds with simplified content extraction.
"""

import time
import logging
import hashlib
import re
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import requests
import feedparser
from newspaper import Article
from bs4 import BeautifulSoup

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
        
        # Political/governmental keywords for filtering
        self.govt_keywords = set(kw.lower() for kw in cc.get("govt_keywords", [
            "executive order", "administration", "white house", "congress", "senate",
            "house of representatives", "supreme court", "federal", "president",
            "department of", "agency", "regulation", "policy", "law", "legislation",
            "election", "democracy", "constitution", "amendment"
        ]))
        
        # User agents for requests
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        self.logger = logging.getLogger("ContentCollector")

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
        
        # Process each source
        for source in sources:
            try:
                if source.get("type", "").lower() == "rss":
                    articles = self._collect_from_rss(source, limit, start_date, end_date)
                    
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
                else:
                    self.logger.warning(f"Unsupported source type: {source.get('type')}")
                    failed_sources += 1
            except Exception as e:
                self.logger.error(f"Error processing source {source.get('url')}: {e}")
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
                
                # Publication date filtering
                pub_date = None
                if entry.get("published_parsed"):
                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    
                    # Skip if outside date range
                    if start_date and pub_date < start_date:
                        continue
                    if end_date and pub_date > end_date:
                        continue
                
                self.logger.info(f"Fetching article: {title}")
                
                # Extract content
                content = self._extract_article_content(link)
                
                # Skip if content extraction failed
                if not content or not self._is_valid_content(content):
                    self.logger.warning(f"Failed to extract valid content for: {title}")
                    continue
                
                # Filter for political/government content
                tags = entry.get("tags", [])
                if not self._is_government_related(title, content, tags):
                    self.logger.info(f"Skipping non-governmental article: {title}")
                    continue
                
                # Determine source name
                source_name = feed.feed.get("title") or urlparse(url).netloc.replace("www.", "")
                
                # Create article data
                article_data = {
                    "title": title,
                    "url": link,
                    "source": source_name,
                    "bias_label": bias,
                    "published": pub_date.isoformat() if pub_date else None,
                    "content": content,
                    "tags": [t.get("term") for t in tags if t.get("term")],
                    "collected_at": datetime.now().isoformat()
                }
                
                collected.append(article_data)
                self.logger.info(f"Collected: {title} ({len(content)} chars)")
            
            return collected
            
        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {e}")
            return []

    def _extract_article_content(self, url: str) -> Optional[str]:
        """
        Extract article content with primary and fallback methods.
        """
        try:
            # Use Newspaper3k as primary extractor
            content = self._extract_with_newspaper(url)
            
            # Check if extraction was successful
            if self._is_valid_content(content):
                return content
            
            # Fall back to BeautifulSoup extraction
            self.logger.info(f"Primary extraction failed for {url}, trying fallback")
            content = self._extract_with_beautifulsoup(url)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Content extraction error for {url}: {e}")
            return None

    def _extract_with_newspaper(self, url: str) -> str:
        """
        Extract content using Newspaper3k with improved settings.
        """
        article = Article(url)
        
        # Configure with better settings
        article.config.browser_user_agent = random.choice(self.user_agents)
        article.config.fetch_images = False
        article.config.memoize_articles = False
        article.config.request_timeout = 10
        
        # Add headers for better success rate
        article.config.headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive'
        }
        
        # Download with one retry
        try:
            article.download()
        except Exception as e:
            self.logger.warning(f"Download retry needed: {e}")
            time.sleep(2)
            article.download()
            
        article.parse()
        content = article.text or ""
        
        # Throttle requests
        time.sleep(1.5)
        
        return content

    def _extract_with_beautifulsoup(self, url: str) -> str:
        """
        Extract content using BeautifulSoup targeting article containers.
        """
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
            '.article-body'
        ]
        
        content = ""
        
        # Try each selector until we find content
        for selector in selectors:
            container = soup.select_one(selector)
            if container:
                # Remove unwanted elements
                for unwanted in container.select('aside, .related, .social, .comments, nav, header, footer, .ad, script, style'):
                    unwanted.decompose()
                
                # Extract text with paragraph breaks
                paragraphs = container.find_all('p')
                if paragraphs:
                    content = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                else:
                    content = container.get_text(separator='\n\n', strip=True)
                
                if self._is_valid_content(content):
                    break
        
        # Throttle requests
        time.sleep(1.5)
        
        return content

    def _is_valid_content(self, content: Optional[str]) -> bool:
        """
        Check if content meets quality criteria.
        """
        if not content or len(content) < 200:
            return False
        
        # Check for multiple paragraphs (at least 2)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return False
        
        # Check for sentence structure (rough approximation)
        sentences = re.findall(r'[.!?]', content)
        if len(sentences) < 3:
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
                if tag_term in political_tags:
                    return True
        
        # 2) Check for keywords in title and content preview
        sample_text = (title + " " + content[:1000]).lower()
        
        # Check each keyword
        for keyword in self.govt_keywords:
            if keyword in sample_text:
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
            "tags": article.get("tags", []),
            "collected_at": article.get("collected_at", datetime.now().isoformat()),
            "id": article.get("id")
        }
