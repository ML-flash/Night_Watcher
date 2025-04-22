"""
Night_watcher Content Collector Agent
Agent for collecting articles from various sources with focus on governmental content.
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
    """Agent for collecting articles from various sources with focus on governmental content"""

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
        self.govt_keywords = [
            "executive order", "administration", "white house", "congress", "senate",
            "house of representatives", "supreme court", "federal", "president",
            "department of", "agency", "regulation", "policy", "law", "legislation",
            "trump", "biden", "election", "democracy", "constitution", "amendment"
        ]

    def _load_default_sources(self) -> List[Dict[str, str]]:
        """
        Load default news sources configuration with enhanced focus on government sources.

        Returns:
            List of source dictionaries
        """
        return [
            # General News with Political Focus
            {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"},
            {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml", "type": "rss", "bias": "center-left"},
            {"url": "https://feeds.foxnews.com/foxnews/politics", "type": "rss", "bias": "right"},
            {"url": "https://www.huffpost.com/section/politics/feed", "type": "rss", "bias": "left"},
            {"url": "https://www.washingtontimes.com/rss/headlines/news/politics/", "type": "rss", "bias": "right"},
            {"url": "https://thehill.com/homenews/feed/", "type": "rss", "bias": "center"},

            # Government-Focused Sources
            {"url": "https://www.govexec.com/rss/", "type": "rss", "bias": "center"},
            {"url": "https://www.whitehouse.gov/feed/", "type": "rss", "bias": "government"},
            {"url": "https://www.federalregister.gov/reader-aids/government-policy-and-ofr-procedures/rss-feeds", "type": "rss", "bias": "government"},
            {"url": "https://www.congress.gov/rss/most-viewed-bills.xml", "type": "rss", "bias": "government"},
            {"url": "https://www.scotusblog.com/feed/", "type": "rss", "bias": "legal"},
            {"url": "https://constitutionallawreporter.com/feed/", "type": "rss", "bias": "legal"},

            # Policy-Focused Sources
            {"url": "https://www.brookings.edu/feed/", "type": "rss", "bias": "center-left"},
            {"url": "https://www.heritage.org/rss/", "type": "rss", "bias": "right"},
            {"url": "https://www.aei.org/feed/", "type": "rss", "bias": "right"},
            {"url": "https://www.cato.org/rss.xml", "type": "rss", "bias": "libertarian"},
            {"url": "https://www.brennancenter.org/feed", "type": "rss", "bias": "center-left"}
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

    def _is_government_related(self, title: str, content: str) -> bool:
        """
        Check if article is related to government or politics.

        Args:
            title: Article title
            content: Article content

        Returns:
            True if government related, False otherwise
        """
        # Check title and first 1000 chars of content for governmental keywords
        text_to_check = (title + " " + content[:1000]).lower()

        for keyword in self.govt_keywords:
            if keyword.lower() in text_to_check:
                return True

        return False

    def _collect_from_rss(self, source: Dict[str, str], limit: int, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Collect articles from an RSS feed with government content filtering and date range filtering.

        Args:
            source: Source configuration
            limit: Maximum number of articles to collect
            start_date: Optional start date for filtering articles
            end_date: Optional end date for filtering articles

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

                # Filter for government-related content (unless from a government source)
                is_govt_source = source.get("bias") in ["government", "legal"]
                if not is_govt_source and not self._is_government_related(entry.get("title", ""), content):
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
                self.logger.info(f"Collected: {article_data['title']}")

                # Add a small delay to prevent hammering servers
                time.sleep(random.uniform(1.0, 3.0))

                # Check if we've reached the limit
                if len(articles) >= limit:
                    break

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
            Dict with 'articles' key containing collected articles
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
        valid_articles = [a for a in all_articles if validate_article_data(a)]

        self.logger.info(f"Collection complete. Collected {len(valid_articles)} valid articles.")

        return {"articles": valid_articles}
