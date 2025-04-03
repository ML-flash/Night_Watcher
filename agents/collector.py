"""
Night_watcher Content Collector Agent
Agent for collecting articles from various sources.
"""

import time
import random
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse

from agents.base import Agent, LLMProvider
from utils.helpers import rate_limiter, validate_article_data


class ContentCollector(Agent):
    """Agent for collecting articles from various sources"""

    def __init__(self, llm_provider: LLMProvider, limit: int = 5):
        """
        Initialize with LLM provider and article limit.

        Args:
            llm_provider: LLM provider
            limit: Maximum number of articles to collect per source
        """
        super().__init__(llm_provider, name="ContentCollector")
        self.article_limit = limit
        self.sources = self._load_default_sources()

    def _load_default_sources(self) -> List[Dict[str, str]]:
        """
        Load default news sources configuration.

        Returns:
            List of source dictionaries
        """
        return [
            {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"},
            {"url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "type": "rss", "bias": "center-left"},
            {"url": "https://feeds.foxnews.com/foxnews/politics", "type": "rss", "bias": "right"},
            {"url": "https://www.huffpost.com/section/politics/feed", "type": "rss", "bias": "left"},
            {"url": "https://www.washingtontimes.com/rss/headlines/news/politics/", "type": "rss", "bias": "right"},
            {"url": "https://thehill.com/homenews/feed/", "type": "rss", "bias": "center"},
            {"url": "https://www.breitbart.com/feed/", "type": "rss", "bias": "right"},
            {"url": "https://www.motherjones.com/feed/", "type": "rss", "bias": "left"}
        ]

    @rate_limiter(max_rate=5, period=1.0)
    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch the full content of an article from its URL.

        Args:
            url: The article URL

        Returns:
            The article content as string
        """
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()

            return article.text
        except Exception as e:
            self.logger.warning(f"Error fetching article content: {str(e)}")
            return ""

    def _collect_from_rss(self, source: Dict[str, str], limit: int) -> List[Dict[str, Any]]:
        """
        Collect articles from an RSS feed.

        Args:
            source: Source configuration
            limit: Maximum number of articles to collect

        Returns:
            List of collected articles
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
            for entry in entries[:limit]:
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
                
                # Check if article was published recently (within 3 days)
                pub_date = entry.get("published_parsed")
                is_recent = True
                if pub_date:
                    from time import mktime
                    pub_datetime = datetime.fromtimestamp(mktime(pub_date))
                    is_recent = (datetime.now() - pub_datetime).days <= 3
                
                if not is_recent:
                    self.logger.debug(f"Skipping old article: {entry.get('title', '')}")
                    continue
                
                # Fetch the full article content
                content = self._fetch_article_content(article_url)
                
                # If content is empty or too short, try to use summary from feed
                if not content or len(content) < 200:
                    content = entry.get("summary", "")
                    
                    # Clean up HTML
                    content = re.sub(r'<[^>]+>', '', content)
                
                # Skip if still no content
                if not content:
                    self.logger.warning(f"No content for: {article_url}")
                    continue
                
                # Create article data
                article_data = {
                    "title": entry.get("title", "Untitled"),
                    "content": content,
                    "source": source_name,
                    "url": article_url,
                    "bias_label": bias_label,
                    "published": entry.get("published", ""),
                    "collected_at": datetime.now().isoformat()
                }
                
                articles.append(article_data)
                self.logger.info(f"Collected: {article_data['title']}")
                
                # Add a small delay to prevent hammering servers
                time.sleep(random.uniform(1.0, 3.0))
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {str(e)}")
            return []

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect articles from sources.

        Args:
            input_data: Dict with optional 'sources' and 'limit' keys

        Returns:
            Dict with 'articles' key containing collected articles
        """
        # Get sources from input or use defaults
        sources = input_data.get("sources", self.sources)
        limit = input_data.get("limit", self.article_limit)
        
        self.logger.info(f"Starting content collection from {len(sources)} sources, limit {limit} per source")
        
        all_articles = []
        
        for source in sources:
            source_type = source.get("type", "").lower()
            
            if source_type == "rss":
                articles = self._collect_from_rss(source, limit)
                all_articles.extend(articles)
            else:
                self.logger.warning(f"Unsupported source type: {source_type}")
        
        # Validate all articles
        valid_articles = [a for a in all_articles if validate_article_data(a)]
        
        self.logger.info(f"Collection complete. Collected {len(valid_articles)} valid articles.")
        
        return {"articles": valid_articles}
