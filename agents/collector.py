"""
Night_watcher Content Collector Agent
Agent for collecting articles from various sources.
"""

import time
import random
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from .base import Agent, LLMProvider
from .utils.helpers import rate_limiter, validate_article_data


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