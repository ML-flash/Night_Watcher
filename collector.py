"""
Night_watcher Content Collector (Enhanced for Full Article Content)
Module for collecting articles from various sources with focus on governmental content.
"""

import time
import random
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import requests

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Content Collector
# ==========================================

class ContentCollector:
    """Collector for gathering full article content from various sources"""

    def __init__(self, limit: int = 5):
        """
        Initialize with article limit.

        Args:
            limit: Maximum number of articles to collect per source
        """
        self.article_limit = limit
        self.sources = self._load_default_sources()
        self.govt_keywords = [
            "executive order", "administration", "white house", "congress", "senate",
            "house of representatives", "supreme court", "federal", "president",
            "department of", "agency", "regulation", "policy", "law", "legislation",
            "trump", "biden", "election", "democracy", "constitution", "amendment"
        ]
        self.logger = logging.getLogger("ContentCollector")
        
        # Used to track and log article content size for debugging
        self.content_stats = {
            "total_articles": 0,
            "total_content_size": 0,
            "min_content_size": float('inf'),
            "max_content_size": 0
        }

    def _load_default_sources(self) -> List[Dict[str, str]]:
        """
        Load default news sources configuration.

        Returns:
            List of source dictionaries
        """
        return [
            # General News with Political Focus
            {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"},
            {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml", "type": "rss", "bias": "center-left"},
            {"url": "https://feeds.foxnews.com/foxnews/politics", "type": "rss", "bias": "right"},
            {"url": "https://www.huffpost.com/section/politics/feed", "type": "rss", "bias": "left"},
            {"url": "https://thehill.com/homenews/feed/", "type": "rss", "bias": "center"}
        ]

    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch the full content of an article from its URL with enhanced error handling and content verification.
    
        Args:
            url: The article URL
    
        Returns:
            The article content as string
        """
        try:
            from newspaper import Article
            
            # List of common user agents for rotation
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
            ]
    
            # Configure and download article
            article = Article(url)
            article.config.browser_user_agent = random.choice(user_agents)
            article.config.fetch_images = False  # Skip image downloading for speed
            article.config.request_timeout = 10  # Set timeout to avoid hanging
            
            # Attempt to download with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    article.download()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Download attempt {attempt+1} failed: {str(e)}. Retrying...")
                        time.sleep(2)  # Wait between retries
                    else:
                        raise
            
            # Parse the article
            article.parse()
            
            # Get the content and check its length
            content = article.text
            content_length = len(content)
            
            # Log content statistics for debugging
            self.logger.debug(f"Article content length: {content_length} characters")
            
            # If content is too short or empty, try additional methods
            if not content or content_length < 300:
                self.logger.warning(f"Article content too short ({content_length} chars). Trying alternate methods.")
                
                # Method 1: Try NLP summarization
                try:
                    article.nlp()
                    if article.summary and len(article.summary) > len(content):
                        content = article.summary
                        self.logger.debug(f"Using NLP summary instead. Length: {len(content)}")
                except Exception as nlp_error:
                    self.logger.warning(f"NLP summarization failed: {str(nlp_error)}")
                
                # Method 2: Try metadata description
                if (not content or len(content) < 200) and hasattr(article, 'meta_description'):
                    if article.meta_description:
                        content = article.meta_description
                        self.logger.debug(f"Using meta description. Length: {len(content)}")
                
                # Method 3: Try direct requests with different parser
                if not content or len(content) < 200:
                    try:
                        import bs4
                        response = requests.get(url, headers={'User-Agent': random.choice(user_agents)}, timeout=10)
                        soup = bs4.BeautifulSoup(response.text, 'html.parser')
                        
                        # Try to find article content in common containers
                        article_content = soup.find('article') or soup.find(class_=['article-content', 'content', 'story-body'])
                        if article_content:
                            text_content = article_content.get_text(separator='\n', strip=True)
                            if text_content and len(text_content) > len(content):
                                content = text_content
                                self.logger.debug(f"Using BeautifulSoup parser. Length: {len(content)}")
                    except Exception as bs_error:
                        self.logger.warning(f"BeautifulSoup parsing failed: {str(bs_error)}")
            
            # Add a small delay to prevent hammering servers
            time.sleep(random.uniform(1.0, 2.0))
            
            # Update content statistics
            if content:
                content_length = len(content)
                self.content_stats["total_articles"] += 1
                self.content_stats["total_content_size"] += content_length
                self.content_stats["min_content_size"] = min(self.content_stats["min_content_size"], content_length)
                self.content_stats["max_content_size"] = max(self.content_stats["max_content_size"], content_length)
            
            return content
        except Exception as e:
            self.logger.error(f"Error fetching article content: {str(e)}")
            # Return placeholder text on failure to indicate the error
            return f"[Failed to retrieve full article content: {str(e)}]"

    def _is_government_related(self, title: str, content: str) -> bool:
        """
        Check if article is related to government or politics.

        Args:
            title: Article title
            content: Article content

        Returns:
            True if government related, False otherwise
        """
        # Check title and first 2000 chars of content for governmental keywords
        text_to_check = (title + " " + content[:2000]).lower()

        for keyword in self.govt_keywords:
            if keyword.lower() in text_to_check:
                return True

        return False

    def _collect_from_rss(self, source: Dict[str, str], limit: int, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Collect articles from an RSS feed with enhanced content fetching.

        Args:
            source: Source configuration
            limit: Maximum number of articles to collect
            start_date: Optional start date for filtering articles
            end_date: Optional end date for filtering articles

        Returns:
            List of collected articles with full content
        """
        import feedparser

        url = source.get("url", "")
        bias_label = source.get("bias", "unknown")

        self.logger.info(f"Collecting from RSS: {url}")

        try:
            feed = feedparser.parse(url)

            if feed.get("bozo", 0) == 1:
                self.logger.warning(f"Feed error for {url}: {feed.get('bozo_exception', 'Unknown error')}")

            entries = feed.get("entries", [])
            self.logger.info(f"Found {len(entries)} entries in feed")

            # Sort by published date (newest first)
            entries.sort(key=lambda x: x.get("published_parsed", 0), reverse=True)

            articles = []
            for entry in entries:
                # Extract article URL
                article_url = entry.get("link", "")
                if not article_url:
                    continue

                # Extract source name from URL or feed title
                source_name = feed.get("feed", {}).get("title", "")
                if not source_name:
                    # Extract from URL
                    parsed_url = urlparse(url)
                    source_name = parsed_url.netloc.replace("www.", "")

                # Check if article was published within the requested date range
                pub_date = None
                if entry.get("published_parsed"):
                    from time import mktime
                    pub_datetime = datetime.fromtimestamp(mktime(entry.get("published_parsed")))
                    pub_date = pub_datetime

                # Apply date range filtering if specified
                if pub_date:
                    if start_date and pub_date < start_date:
                        self.logger.debug(f"Skipping article before start date: {entry.get('title', '')}")
                        continue
                    if end_date and pub_date > end_date:
                        self.logger.debug(f"Skipping article after end date: {entry.get('title', '')}")
                        continue

                # Log article found
                self.logger.info(f"Fetching article: {entry.get('title', '')} from {article_url}")

                # Fetch the full article content
                content = self._fetch_article_content(article_url)

                # Log content size
                self.logger.debug(f"Content size: {len(content)} characters")

                # If content is empty or too short, try to use summary from feed
                if not content or len(content) < 200:
                    self.logger.warning(f"Content too short for: {entry.get('title', '')}. Using feed summary.")
                    content = entry.get("summary", "")

                    # Clean up HTML
                    content = re.sub(r'<[^>]+>', '', content)

                # Skip if still no content
                if not content:
                    self.logger.warning(f"No content for: {article_url}")
                    continue

                # Filter for government-related content
                if not self._is_government_related(entry.get("title", ""), content):
                    self.logger.info(f"Skipping non-governmental article: {entry.get('title', '')}")
                    continue

                # Create article data
                article_data = {
                    "title": entry.get("title", "Untitled"),
                    "content": content,
                    "source": source_name,
                    "url": article_url,
                    "bias_label": bias_label,
                    "published": pub_date.isoformat() if pub_date else datetime.now().isoformat(),
                    "collected_at": datetime.now().isoformat()
                }

                articles.append(article_data)
                self.logger.info(f"Collected: {article_data['title']} ({len(content)} characters)")

                # Check if we've reached the limit
                if len(articles) >= limit:
                    break
            
            # Log content statistics
            if articles:
                avg_content_size = self.content_stats["total_content_size"] / self.content_stats["total_articles"] if self.content_stats["total_articles"] > 0 else 0
                self.logger.info(f"Content statistics: min={self.content_stats['min_content_size']}, max={self.content_stats['max_content_size']}, avg={avg_content_size:.2f}")

            return articles

        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {str(e)}")
            return []

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect articles from sources with focus on governmental content.

        Args:
            input_data: Dict with optional 'sources', 'limit', 'start_date', and 'end_date' keys

        Returns:
            Dict with 'articles' key containing collected articles with full content
        """
        # Get sources from input or use defaults
        sources = input_data.get("sources", self.sources)
        limit = input_data.get("limit", self.article_limit)
        
        # Get date range if specified
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        
        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date)
            except ValueError:
                self.logger.warning(f"Invalid start_date format: {start_date}. Ignoring date filtering.")
                start_date = None
                
        if isinstance(end_date, str):
            try:
                end_date = datetime.fromisoformat(end_date)
            except ValueError:
                self.logger.warning(f"Invalid end_date format: {end_date}. Using current date.")
                end_date = datetime.now()

        if start_date and end_date:
            self.logger.info(f"Collecting content from {start_date.isoformat()} to {end_date.isoformat()}")
        else:
            self.logger.info(f"Starting content collection from {len(sources)} sources, limit {limit} per source")

        all_articles = []

        for source in sources:
            source_type = source.get("type", "").lower()

            if source_type == "rss":
                articles = self._collect_from_rss(source, limit, start_date, end_date)
                all_articles.extend(articles)
            else:
                self.logger.warning(f"Unsupported source type: {source_type}")

        # Validate all articles
        valid_articles = [a for a in all_articles if self._validate_article_data(a)]

        # Log article content statistics
        self.logger.info(f"Collection complete. Collected {len(valid_articles)} valid articles.")
        for i, article in enumerate(valid_articles):
            content_length = len(article.get("content", ""))
            self.logger.info(f"Article {i+1}: '{article.get('title', 'Untitled')}' - {content_length} characters")

        return {"articles": valid_articles}
        
    def _validate_article_data(self, article: Dict[str, Any]) -> bool:
        """
        Validate article data has the required fields and minimum content length.

        Args:
            article: Article data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['title', 'content', 'source']
        has_required_fields = all(field in article for field in required_fields)
        
        if not has_required_fields:
            missing = [field for field in required_fields if field not in article]
            self.logger.warning(f"Article missing required fields: {missing}")
            return False
        
        # Check for minimum content length
        min_content_length = 200  # Minimum characters to consider valid
        content_length = len(article.get('content', ''))
        
        if content_length < min_content_length:
            self.logger.warning(f"Article content too short: {content_length} characters (minimum: {min_content_length})")
            return False
            
        return True
        