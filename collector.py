# collector.py
"""
Night_watcher Content Collector: gathers full article content with political filtering.
"""

import time
import random
import logging
import re
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
    Collector for gathering full article content from various RSS sources,
    filtering only for political/governmental content via category tags and keyword matching.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Configuration dict loaded from config.json
        """
        self.config = config
        cc = config.get("content_collection", {})
        self.article_limit = cc.get("article_limit", 5)
        self.sources = cc.get("sources", [])
        # Governmental keywords for fallback filtering
        self.govt_keywords: List[str] = config.get("govt_keywords", [
            "executive order", "administration", "white house", "congress", "senate",
            "house of representatives", "supreme court", "federal", "president",
            "department of", "agency", "regulation", "policy", "law", "legislation",
            "election", "democracy", "constitution", "amendment"
        ])
        self.logger = logging.getLogger("ContentCollector")

    def _is_government_related(
        self,
        title: str,
        content: str,
        tags: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Check if article is related to government or politics.
        1) Permit if any RSS <category> term matches.
        2) Otherwise, fallback to keyword scan of title + content snippet.
        """
        # 1) Tag-based allow
        if tags:
            allowed = {"politics", "government", "us politics", "policy", "election"}
            for t in tags:
                term = t.get("term", "").lower()
                if term in allowed:
                    return True

        # 2) Free-text keyword scan
        text = (title + " " + content[:2000]).lower()
        for kw in self.govt_keywords:
            if kw.lower() in text:
                return True

        return False

    def _validate_article_data(self, article: Dict[str, Any]) -> bool:
        """
        Ensure article has required fields and meets minimum content length.
        """
        required = ['title', 'content', 'source']
        for field in required:
            if not article.get(field):
                self.logger.warning(f"Article missing required field: {field}")
                return False

        min_length = 200
        if len(article['content']) < min_length:
            self.logger.warning(f"Article content too short: {len(article['content'])} chars (min {min_length})")
            return False

        return True

    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch full content via newspaper3k, with fallbacks to summary, meta_description, and BeautifulSoup parsing.
        """
        try:
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.1.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
            ]

            article = Article(url)
            article.config.browser_user_agent = random.choice(user_agents)
            article.config.fetch_images = False
            article.config.request_timeout = 10

            # Download with retry logic
            for attempt in range(3):
                try:
                    article.download()
                    break
                except Exception as e:
                    if attempt < 2:
                        self.logger.warning(f"Download attempt {attempt+1} failed: {e}, retrying...")
                        time.sleep(1)
                    else:
                        raise

            article.parse()
            content = article.text or ""

            # Fallback methods if parsed content is too short
            if not content or len(content) < 300:
                self.logger.warning(f"Parsed content too short ({len(content)} chars), applying fallbacks")

                # 1) NLP summarization
                try:
                    article.nlp()
                    if article.summary and len(article.summary) > len(content):
                        content = article.summary
                        self.logger.debug(f"Using NLP summary (len={len(content)})")
                except Exception as e:
                    self.logger.warning(f"NLP summarization failed: {e}")

                # 2) Meta description
                if (not content or len(content) < 200) and hasattr(article, 'meta_description'):
                    if article.meta_description:
                        content = article.meta_description
                        self.logger.debug(f"Using meta_description (len={len(content)})")

                # 3) BeautifulSoup parsing
                if not content or len(content) < 200:
                    resp = requests.get(url, headers={'User-Agent': random.choice(user_agents)}, timeout=10)
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    node = soup.find('article') or soup.find(class_=['article-content', 'content', 'story-body'])
                    if node:
                        text = node.get_text(separator='\n', strip=True)
                        if text and len(text) > len(content):
                            content = text
                            self.logger.debug(f"Using BeautifulSoup parse (len={len(content)})")

            # Throttle requests
            time.sleep(random.uniform(1.0, 2.0))
            return content

        except Exception as e:
            self.logger.error(f"Error fetching article content: {e}")
            return ""

    def _collect_from_rss(
        self,
        source: Dict[str, Any],
        limit: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect up to `limit` articles from an RSS feed, filtering for government/political content.
        """
        url = source.get("url", "")
        bias = source.get("bias", "unknown")
        self.logger.info(f"Collecting from RSS: {url}")

        try:
            feed = feedparser.parse(url)
            if feed.get("bozo", 0) == 1:
                self.logger.warning(f"Feed parse error for {url}: {feed.get('bozo_exception')}")

            entries = feed.get("entries", [])
            entries.sort(key=lambda x: x.get("published_parsed", 0), reverse=True)
            collected: List[Dict[str, Any]] = []

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
                    if start_date and pub_date < start_date:
                        continue
                    if end_date and pub_date > end_date:
                        continue

                self.logger.info(f"Fetching article: {title} from {link}")
                content = self._fetch_article_content(link)

                # Fallback to summary if still too short
                if not content or len(content) < 200:
                    self.logger.warning(f"Content too short for '{title}'. Using feed summary.")
                    summary = entry.get("summary", "")
                    content = re.sub(r"<[^>]+>", "", summary)

                if not content:
                    self.logger.warning(f"No content available for: {title}")
                    continue

                tags = entry.get("tags", [])
                if not self._is_government_related(title, content, tags):
                    self.logger.info(f"Skipping non-governmental article: {title}")
                    continue

                # Determine source name
                source_name = feed.feed.get("title") or urlparse(url).netloc.replace("www.", "")
                article_data = {
                    "title": title,
                    "url": link,
                    "source": source_name,
                    "bias_label": bias,
                    "published": pub_date.isoformat() if pub_date else None,
                    "content": content,
                    "tags": [t.get("term") for t in tags],
                    "collected_at": datetime.now().isoformat()
                }

                collected.append(article_data)
                self.logger.info(f"Collected: {title} ({len(content)} chars)")

            return collected

        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {e}")
            return []

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: collect and return only government/political articles.

        Args:
            input_data: Dict with optional keys:
                - 'sources': override list of sources
                - 'limit': max articles per source
                - 'start_date', 'end_date': ISO strings or datetime objects

        Returns:
            Dict with key 'articles' containing the filtered articles.
        """
        sources = input_data.get("sources", self.sources)
        limit = input_data.get("limit", self.article_limit)
        sd = input_data.get("start_date")
        ed = input_data.get("end_date")

        # Parse date strings
        if isinstance(sd, str):
            try:
                sd = datetime.fromisoformat(sd)
            except ValueError:
                self.logger.warning(f"Ignoring invalid start_date: {sd}")
                sd = None
        if isinstance(ed, str):
            try:
                ed = datetime.fromisoformat(ed)
            except ValueError:
                self.logger.warning(f"Ignoring invalid end_date: {ed}")
                ed = None

        self.logger.info(f"Starting collection: {len(sources)} sources, limit {limit}")
        all_articles: List[Dict[str, Any]] = []

        for src in sources:
            if src.get("type", "").lower() == "rss":
                arts = self._collect_from_rss(src, limit, sd, ed)
                all_articles.extend(arts)
            else:
                self.logger.warning(f"Unsupported source type: {src.get('type')}")

        # Validate and log
        valid = [a for a in all_articles if self._validate_article_data(a)]
        self.logger.info(f"Collection complete. {len(valid)} valid articles collected.")
        for i, art in enumerate(valid, start=1):
            self.logger.info(f"Article {i}: '{art['title']}' – {len(art['content'])} chars")

        return {"articles": valid}
