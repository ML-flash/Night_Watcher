#!/usr/bin/env python3
"""
Night_watcher Content Collector
Enhanced collector with LLM-guided navigation fallback for problematic sites
"""

import time
import logging
import hashlib
import re
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import concurrent.futures
import traceback
import xml.etree.ElementTree as ET

import requests
import feedparser
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


class LLMWebNavigator:
    """
    LLM-based web navigator that treats the LLM like a human user
    """
    
    def __init__(self, llm_provider, session: requests.Session):
        self.llm = llm_provider
        self.session = session
        self.logger = logging.getLogger("LLMWebNavigator")
        self.max_navigation_steps = 3
        self.current_url = None
        self.navigation_history = []
        
    def navigate_to_rss_feed(self, failed_rss_url: str, source_info: Dict[str, Any]) -> Optional[str]:
        """
        Navigate to find the actual RSS feed when the provided URL fails
        
        Args:
            failed_rss_url: The RSS URL that failed (might be returning HTML)
            source_info: Information about the source (bias, type, etc.)
            
        Returns:
            Working RSS feed URL or None if not found
        """
        self.logger.info(f"LLM navigating to find RSS feed for: {failed_rss_url}")
        self.current_url = failed_rss_url
        self.navigation_history = [failed_rss_url]
        
        for step in range(self.max_navigation_steps):
            self.logger.info(f"Navigation step {step + 1}: {self.current_url}")
            
            # Fetch the current page
            try:
                response = self.session.get(self.current_url, timeout=30)
                if response.status_code != 200:
                    self.logger.warning(f"Got {response.status_code} for {self.current_url}")
                    return None
                    
                page_content = response.text
                
            except Exception as e:
                self.logger.error(f"Failed to fetch {self.current_url}: {e}")
                return None
            
            # Check if this is actually an RSS feed now
            if self._is_rss_content(page_content):
                self.logger.info(f"Found working RSS feed at: {self.current_url}")
                return self.current_url
            
            # Ask LLM to navigate the page
            navigation_result = self._ask_llm_to_navigate(page_content, failed_rss_url, source_info)
            
            if navigation_result["action"] == "found_rss_link":
                # LLM found a direct RSS link
                rss_url = navigation_result["url"]
                full_rss_url = urljoin(self.current_url, rss_url)
                
                self.logger.info(f"LLM found RSS link: {full_rss_url}")
                
                # Test the RSS link
                if self._test_rss_url(full_rss_url):
                    return full_rss_url
                else:
                    self.logger.warning(f"LLM-suggested RSS URL failed: {full_rss_url}")
                    continue
                    
            elif navigation_result["action"] == "navigate_to_page":
                # LLM wants to navigate to another page
                next_url = navigation_result["url"]
                full_next_url = urljoin(self.current_url, next_url)
                
                if full_next_url in self.navigation_history:
                    self.logger.warning(f"Already visited {full_next_url}, avoiding loop")
                    break
                    
                self.logger.info(f"LLM navigating to: {full_next_url}")
                self.current_url = full_next_url
                self.navigation_history.append(full_next_url)
                continue
                
            elif navigation_result["action"] == "no_rss_found":
                self.logger.info("LLM could not find RSS feed on this page")
                break
                
            else:
                self.logger.warning(f"Unknown LLM action: {navigation_result['action']}")
                break
        
        self.logger.warning(f"Failed to find working RSS feed for {failed_rss_url}")
        return None
    
    def find_articles_on_page(self, page_url: str, source_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Ask LLM to find political articles on a webpage
        
        Returns:
            List of {"title": "...", "url": "..."} dictionaries
        """
        self.logger.info(f"LLM finding articles on: {page_url}")
        
        try:
            response = self.session.get(page_url, timeout=30)
            if response.status_code != 200:
                return []
                
            page_content = response.text
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {page_url}: {e}")
            return []
        
        # Ask LLM to identify articles
        articles = self._ask_llm_to_find_articles(page_content, page_url, source_info)
        
        # Convert relative URLs to absolute
        result = []
        for article in articles:
            full_url = urljoin(page_url, article["url"])
            result.append({
                "title": article["title"],
                "url": full_url
            })
        
        self.logger.info(f"LLM found {len(result)} articles on {page_url}")
        return result
    
    def _is_rss_content(self, content: str) -> bool:
        """Check if content is actually RSS/XML"""
        content_lower = content.lower().strip()
        return (
            content_lower.startswith('<?xml') or
            '<rss' in content_lower or
            '<feed' in content_lower or
            'application/rss+xml' in content_lower
        )
    
    def _test_rss_url(self, url: str) -> bool:
        """Test if a URL returns valid RSS content"""
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return self._is_rss_content(response.text)
        except:
            pass
        return False
    
    def _ask_llm_to_navigate(self, page_content: str, original_rss_url: str, source_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Ask LLM to analyze a webpage and decide how to navigate to find RSS feeds
        """
        # Simplify the page content for the LLM
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Remove scripts, styles, and other noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Extract key links that might lead to RSS feeds
        links = []
        for link in soup.find_all('a', href=True)[:50]:  # Limit to 50 links
            href = link['href']
            text = link.get_text(strip=True)
            
            # Focus on links that might be RSS-related
            if any(keyword in (href + ' ' + text).lower() for keyword in 
                   ['rss', 'feed', 'xml', 'syndication', 'subscribe', 'news']):
                links.append({
                    "text": text[:100],
                    "url": href,
                    "title": link.get('title', '')
                })
        
        # Get page title and key text
        title = soup.find('title')
        title_text = title.get_text() if title else "No title"
        
        # Get some body text for context
        body_text = soup.get_text()[:1000] if soup.get_text() else ""
        
        prompt = f"""You are a web user trying to find the RSS feed for a news website.

ORIGINAL RSS URL THAT FAILED: {original_rss_url}
CURRENT PAGE: {self.current_url}
SOURCE TYPE: {source_info.get('bias', 'unknown')} news source

PAGE TITLE: {title_text}

KEY LINKS FOUND:
{json.dumps(links, indent=2)}

PAGE PREVIEW:
{body_text[:500]}...

TASK: You need to find the actual RSS/XML feed for political news from this source.

Look for:
1. Direct RSS/XML feed links (usually end in .xml, .rss, or contain "feed")
2. Links to "RSS", "Feeds", "Syndication" pages
3. Politics/Government section links that might have their own feeds

RESPOND WITH JSON:
{{
    "action": "found_rss_link|navigate_to_page|no_rss_found",
    "url": "exact_url_from_links_above",
    "reasoning": "brief explanation of why you chose this link"
}}

If you found a direct RSS link, use "found_rss_link".
If you need to navigate to another page first, use "navigate_to_page".
If no RSS options are visible, use "no_rss_found".
"""
        
        try:
            response = self.llm.complete(prompt, temperature=0.2, max_tokens=300)
            
            # Handle response format
            if isinstance(response, dict):
                if 'choices' in response and response['choices']:
                    response_text = response['choices'][0].get('text', '')
                elif 'error' in response:
                    self.logger.error(f"LLM error: {response['error']}")
                    return {"action": "no_rss_found", "url": "", "reasoning": "LLM error"}
                else:
                    response_text = str(response)
            else:
                response_text = str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                self.logger.info(f"LLM navigation decision: {result['action']} - {result.get('reasoning', '')}")
                return result
            else:
                self.logger.warning("LLM response did not contain valid JSON")
                return {"action": "no_rss_found", "url": "", "reasoning": "Invalid response format"}
                
        except Exception as e:
            self.logger.error(f"Error asking LLM to navigate: {e}")
            return {"action": "no_rss_found", "url": "", "reasoning": f"Error: {e}"}
    
    def _ask_llm_to_find_articles(self, page_content: str, page_url: str, source_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Ask LLM to find political article links on a webpage
        """
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "sidebar"]):
            tag.decompose()
        
        # Extract potential article links
        article_links = []
        for link in soup.find_all('a', href=True)[:100]:  # Limit to 100 links
            href = link['href']
            text = link.get_text(strip=True)
            
            # Filter for substantial text that could be article titles
            if text and len(text) > 20 and len(text) < 200:
                article_links.append({
                    "text": text,
                    "url": href
                })
        
        if not article_links:
            return []
        
        prompt = f"""You are looking for POLITICAL/GOVERNMENT news articles on this webpage.

WEBPAGE: {page_url}
SOURCE TYPE: {source_info.get('bias', 'unknown')} news source

POTENTIAL ARTICLE LINKS:
{json.dumps(article_links[:30], indent=2)}

TASK: Identify which links are political/government news articles published after January 20, 2025.

Look for articles about:
- Politics, elections, campaigns
- Government actions, policies, regulations  
- Congress, Senate, House activities
- Executive branch, White House, President
- Supreme Court, federal courts
- Political figures (Trump, Biden, etc.)
- Government agencies (FBI, EPA, etc.)

IGNORE:
- Sports, entertainment, lifestyle articles
- Pure business/tech news (unless government-related)
- Opinion/editorial pieces
- Old articles (before Jan 2025)

RESPOND WITH JSON ARRAY:
[
    {{
        "title": "exact_title_text",
        "url": "exact_url_from_above",
        "reasoning": "why this is political news"
    }}
]

Maximum 10 articles. Only include articles you're confident are political/government news.
"""
        
        try:
            response = self.llm.complete(prompt, temperature=0.2, max_tokens=800)
            
            # Handle response format
            if isinstance(response, dict):
                if 'choices' in response and response['choices']:
                    response_text = response['choices'][0].get('text', '')
                else:
                    response_text = str(response)
            else:
                response_text = str(response)
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                articles = json.loads(json_match.group())
                return articles if isinstance(articles, list) else []
            else:
                self.logger.warning("LLM article response did not contain valid JSON array")
                return []
                
        except Exception as e:
            self.logger.error(f"Error asking LLM to find articles: {e}")
            return []


class ContentCollector:
    """
    Collector for gathering political content from RSS feeds with LLM-guided fallback
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
        self.max_workers = cc.get("max_workers", 2)
        self.request_timeout = cc.get("request_timeout", 45)
        self.retry_count = cc.get("retry_count", 3)
        self.bypass_cloudflare = cc.get("bypass_cloudflare", True) and CLOUDSCRAPER_AVAILABLE
        self.delay_between_requests = cc.get("delay_between_requests", 5.0)
        self.use_llm_fallback = cc.get("use_llm_navigation_fallback", True)
        
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
        
        # User agents rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
        ]
        
        # URLs to skip (problematic or non-political)
        self.skip_url_patterns = [
            r'/entertainment/', r'/music/', r'/sports/', r'/lifestyle/',
            r'/style/', r'/food/', r'/travel/', r'/celebrity/', r'/culture/',
            r'/arts/', r'/movies/', r'/tv/', r'/gaming/', r'/technology/gadgets/',
            r'/health/personal/', r'/real-estate/', r'/cars/', r'diddy', r'sean-combs'
        ]
        
        # Initialize logger
        self.logger = logging.getLogger("ContentCollector")
        
        # Initialize failure tracker
        self._failure_tracker = {}
        
        # Initialize session objects
        self._init_sessions()
        
        # Initialize LLM navigator
        self.llm_navigator = None
        self.llm_provider = None
        
        # Check for LLM availability at startup
        self._check_llm_availability()
        
    def _check_llm_availability(self):
        """Check LLM availability at startup and initialize if available"""
        if not self.use_llm_fallback:
            self.logger.info("LLM fallback disabled in configuration")
            return
            
        self.logger.info("Checking LLM availability for navigation fallback...")
        
        try:
            # Try to initialize LLM provider using the existing provider system
            llm_provider = self._initialize_llm_provider()
            
            if llm_provider:
                # Test the connection
                test_response = llm_provider.complete("Test", max_tokens=10)
                if not test_response.get("error"):
                    self.llm_provider = llm_provider
                    self.llm_navigator = LLMWebNavigator(self.llm_provider, self.session)
                    self.logger.info("✓ LLM navigation initialized successfully")
                    return
                else:
                    self.logger.warning(f"LLM provider test failed: {test_response.get('error')}")
            else:
                self.logger.info("✗ No LLM provider available")
        
        except Exception as e:
            self.logger.error(f"Error checking LLM availability: {e}")
        
        self.logger.warning("✗ LLM navigation not available - will use traditional methods only")
    
    def _initialize_llm_provider(self):
        """Initialize LLM provider using the existing provider system"""
        try:
            import providers
            
            # Get LLM configuration
            config = self.config.copy()
            llm_config = config.get("llm_provider", {})
            
            # Check if LM Studio is available by trying to connect
            lm_studio_host = llm_config.get("host", "http://localhost:1234")
            self.logger.info(f"Checking LM Studio connection at {lm_studio_host}...")
            
            lm_studio_available = False
            try:
                import requests
                response = requests.get(f"{lm_studio_host}/v1/models", timeout=5)
                lm_studio_available = response.status_code == 200
            except Exception as e:
                self.logger.debug(f"LM Studio connection check failed: {e}")
            
            if lm_studio_available:
                self.logger.info("✓ LM Studio available")
            else:
                self.logger.info("✗ LM Studio not available")
            
            # If no API key and LM Studio isn't available, prompt for Anthropic credentials
            if not llm_config.get("api_key") and not lm_studio_available:
                self.logger.info("LM Studio not available, prompting for Anthropic credentials...")
                try:
                    api_key, model = self._get_anthropic_credentials_secure()
                    if api_key:
                        # Update the config for this session
                        config["llm_provider"]["type"] = "anthropic"
                        config["llm_provider"]["api_key"] = api_key
                        config["llm_provider"]["model"] = model
                        self.logger.info("✓ Anthropic credentials provided")
                    else:
                        self.logger.info("✗ No credentials provided")
                        return None
                except Exception as e:
                    self.logger.warning(f"Error getting credentials: {e}")
                    return None
            
            # Use the existing provider initialization function
            return providers.initialize_llm_provider(config)
            
        except ImportError as e:
            self.logger.warning(f"Provider module not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error initializing LLM provider: {e}")
            return None
    
    def _get_anthropic_credentials_secure(self) -> tuple[str, str]:
        """
        Securely prompt the user for Anthropic API credentials.
        Returns a tuple of (api_key, model_name)
        """
        import getpass
        
        print("\nLM Studio server is not available.")
        print("Falling back to Anthropic API...\n")
        
        # Use getpass to hide the API key input
        try:
            api_key = getpass.getpass("Enter your Anthropic API key (input will be hidden): ").strip()
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return "", ""
        except Exception as e:
            print(f"Error reading API key: {e}")
            return "", ""
        
        if not api_key:
            print("No API key provided. Continuing without LLM capabilities.")
            return "", ""
            
        model = input("Enter Anthropic model (default: claude-3-haiku-20240307): ").strip()
        if not model or model == "3" or not model.startswith("claude-"):
            model = "claude-3-haiku-20240307"
            print(f"Using default model: {model}")
            
        return api_key, model
                
    def _init_sessions(self):
        """Initialize HTTP sessions for requests with enhanced stealth."""
        self.session = requests.Session()
        
        # Enhanced retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504, 522, 524],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rotate user agent for each session
        self._rotate_session_headers()
        
        # Initialize cloudscraper if available
        if self.bypass_cloudflare:
            try:
                self.cloudscraper_session = cloudscraper.create_scraper(
                    browser={
                        'browser': 'firefox',
                        'platform': 'linux',
                        'mobile': False,
                        'desktop': True
                    },
                    delay=10,
                    debug=False
                )
                self.logger.info("Cloudflare bypass initialized")
            except Exception as e:
                self.logger.error(f"Error initializing Cloudflare bypass: {e}")
                self.bypass_cloudflare = False
                
    def _rotate_session_headers(self):
        """Rotate session headers for stealth"""
        user_agent = random.choice(self.user_agents)
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Pragma': 'no-cache'
        })

    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped based on patterns."""
        url_lower = url.lower()
        for pattern in self.skip_url_patterns:
            if re.search(pattern, url_lower):
                self.logger.info(f"Skipping non-political URL pattern '{pattern}': {url}")
                return True
        return False
    
    def _track_collection_result(self, url: str, success: bool):
        """Track success/failure rates by domain"""
        domain = urlparse(url).netloc.lower()
        
        if domain not in self._failure_tracker:
            self._failure_tracker[domain] = {'attempts': 0, 'failures': 0}
        
        self._failure_tracker[domain]['attempts'] += 1
        if not success:
            self._failure_tracker[domain]['failures'] += 1

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: collect and process content from sources.
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
        if self.llm_navigator:
            self.logger.info("LLM navigation fallback available")
        else:
            self.logger.info("LLM navigation fallback not available")
        
        # Initialize results
        all_articles = []
        document_ids = []
        successful_sources = 0
        failed_sources = 0
        
        # Process sources sequentially with delays
        for idx, source in enumerate(sources):
            # Rotate headers periodically
            if idx % 5 == 0:
                self._rotate_session_headers()
            
            source_type = source.get("type", "rss").lower()
            
            try:
                # Try traditional collection first
                articles = []
                
                if source_type != "rss":
                    self.logger.warning(f"Unknown source type: {source_type}")
                    continue
                
                articles = self._collect_from_rss(source, limit, start_date, end_date)
                
                # Track collection result
                url = source.get("url", "")
                self._track_collection_result(url, bool(articles))
                        
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
                
                # Random delay between sources
                delay = random.uniform(3, 8)
                time.sleep(delay)
                
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
                "timestamp": datetime.now().isoformat(),
                "llm_navigation_available": self.llm_navigator is not None
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
            # Add random delay before request
            time.sleep(random.uniform(1, 3))
            
            # Parse the feed with timeout
            feed = feedparser.parse(url, agent=random.choice(self.user_agents))
            
            if feed.get("bozo", 0) == 1:
                self.logger.warning(f"Feed parse error for {url}: {feed.get('bozo_exception')}")
                # If RSS parsing completely fails, try fallback strategies
                if not feed.get("entries"):
                    self.logger.info(f"RSS completely failed for {url}, attempting fallback")
                    return self._handle_rss_failure_with_fallback(source, limit, start_date, end_date)
            
            # Check if we got entries
            entries = feed.get("entries", [])
            if not entries:
                self.logger.warning(f"No entries found in feed: {url}")
                # Try fallback strategies
                return self._handle_rss_failure_with_fallback(source, limit, start_date, end_date)
            
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
                
                # Get content from RSS entry first (often available)
                rss_content = self._extract_rss_content(entry)
                
                # Filter for political/government content early
                tags = entry.get("tags", [])
                if not self._is_government_related(title, rss_content or "", tags):
                    self.logger.debug(f"Skipping non-governmental article: {title}")
                    continue
                
                # If we have good RSS content, use it
                if rss_content and self._is_valid_content(rss_content):
                    content = rss_content
                    article_data = {
                        "authors": [entry.get("author")] if entry.get("author") else [],
                        "published": pub_date.isoformat() if pub_date else None,
                        "top_image": None,
                        "images": [],
                        "movies": []
                    }
                else:
                    # Try to fetch full content with enhanced error handling
                    self.logger.info(f"Fetching article: {title}")
                    
                    try:
                        content, article_data = self._extract_article_content(link, title, pub_date)
                        
                        # Add delay after fetch
                        time.sleep(random.uniform(2, 5))
                        
                    except requests.exceptions.Timeout:
                        self.logger.warning(f"Timeout fetching article: {title}")
                        # Use RSS content if available
                        if rss_content:
                            content = rss_content
                            article_data = {"authors": [], "published": pub_date.isoformat() if pub_date else None}
                        else:
                            continue
                    except Exception as e:
                        self.logger.warning(f"Error fetching article: {title} - {str(e)}")
                        # Use RSS content if available
                        if rss_content:
                            content = rss_content
                            article_data = {"authors": [], "published": pub_date.isoformat() if pub_date else None}
                        else:
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
            # Try fallback strategies as last resort
            return self._handle_rss_failure_with_fallback(source, limit, start_date, end_date)
    
    def _handle_rss_failure_with_fallback(
        self,
        source: Dict[str, Any],
        limit: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Handle RSS feed failures using LLM-guided navigation and simple fallbacks
        """
        url = source.get("url", "")
        self.logger.info(f"Attempting fallback strategies for failed RSS: {url}")
        
        # Strategy 1: Try LLM-guided recovery if available
        if self.llm_navigator:
            try:
                self.logger.info("Trying LLM-guided RSS recovery...")
                
                # Try to find a working RSS feed via LLM navigation
                working_rss = self.llm_navigator.navigate_to_rss_feed(url, source)
                
                if working_rss and working_rss != url:
                    self.logger.info(f"LLM found alternative RSS feed: {working_rss}")
                    alt_source = source.copy()
                    alt_source["url"] = working_rss
                    articles = self._collect_from_rss(alt_source, limit, start_date, end_date)
                    if articles:
                        return articles
                
                # If RSS navigation fails, try direct article extraction
                self.logger.info("Trying LLM-guided article extraction...")
                domain = urlparse(url).netloc
                main_url = f"https://{domain}"
                politics_urls = [f"https://{domain}/politics", main_url]
                
                for page_url in politics_urls:
                    try:
                        articles = self.llm_navigator.find_articles_on_page(page_url, source)
                        if articles:
                            collected = []
                            for article in articles[:limit]:
                                try:
                                    content, article_data = self._extract_article_content(
                                        article["url"], article["title"]
                                    )
                                    if content and self._is_valid_content(content):
                                        article_entry = {
                                            "title": article["title"],
                                            "url": article["url"],
                                            "source": domain,
                                            "bias_label": source.get("bias", "unknown"),
                                            "published": article_data.get('published'),
                                            "content": content,
                                            "tags": [],
                                            "collected_at": datetime.now().isoformat(),
                                            "extraction_method": "llm_guided"
                                        }
                                        collected.append(article_entry)
                                        
                                        if len(collected) >= limit:
                                            break
                                            
                                        time.sleep(random.uniform(2, 4))
                                
                                except Exception as e:
                                    self.logger.error(f"Error extracting LLM-suggested article: {e}")
                                    continue
                            
                            if collected:
                                self.logger.info(f"Successfully collected {len(collected)} articles via LLM guidance")
                                return collected
                    
                    except Exception as e:
                        self.logger.error(f"Error in LLM-guided article extraction from {page_url}: {e}")
                        continue
            
            except Exception as e:
                self.logger.error(f"Error in LLM-guided RSS recovery: {e}")
        else:
            self.logger.warning("LLM navigator not available for RSS failure recovery")
        
        # Strategy 2: Try simple fallback strategies
        return self._try_simple_fallback_strategies(source, limit, start_date, end_date)
    
    def _try_simple_fallback_strategies(
        self,
        source: Dict[str, Any],
        limit: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Try simple fallback strategies when LLM is not available
        """
        url = source.get("url", "")
        domain = urlparse(url).netloc
        
        self.logger.info(f"Trying simple fallback strategies for {domain}")
        
        # Strategy 1: Try common alternative RSS feed URLs
        common_rss_paths = [
            "/feed/",
            "/feeds/",
            "/rss/",
            "/rss.xml",
            "/feed.xml",
            "/atom.xml",
            "/news/feed/",
            "/politics/feed/",
            "/news/rss/",
            "/politics/rss/"
        ]
        
        base_url = f"https://{domain}"
        for path in common_rss_paths:
            try:
                # Don't append to existing RSS URLs - replace the path entirely
                if url.endswith(('.xml', '.rss')):
                    alt_url = base_url + path
                else:
                    alt_url = base_url + path
                    
                self.logger.info(f"Trying alternative RSS: {alt_url}")
                
                alt_source = source.copy()
                alt_source["url"] = alt_url
                
                # Quick test with minimal parsing
                feed = feedparser.parse(alt_url, agent=random.choice(self.user_agents))
                if feed.get("entries") and not feed.get("bozo"):
                    articles = self._collect_from_rss(alt_source, limit, start_date, end_date)
                    if articles:
                        self.logger.info(f"Successfully collected from alternative RSS: {alt_url}")
                        return articles
                
                time.sleep(1)  # Brief delay between attempts
                
            except Exception as e:
                self.logger.debug(f"Alternative RSS {alt_url} failed: {e}")
                continue
        
        # Strategy 2: Try direct scraping of common news section URLs
        politics_paths = [
            "/politics/",
            "/news/politics/",
            "/political/",
            "/government/",
            "/news/",
            "/"
        ]
        
        for path in politics_paths:
            try:
                scrape_url = base_url + path
                self.logger.info(f"Trying direct scraping: {scrape_url}")
                
                response = self.session.get(scrape_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for article links using simple heuristics
                    article_links = []
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        text = link.get_text(strip=True)
                        
                        # Make URL absolute
                        if href.startswith('/'):
                            href = base_url + href
                        elif not href.startswith('http'):
                            continue
                        
                        # Simple heuristics for political articles
                        if (text and len(text) > 15 and len(text) < 200 and
                            any(keyword in text.lower() for keyword in ['trump', 'biden', 'congress', 'senate', 'politics', 'government', 'election', 'policy'])):
                            article_links.append((href, text))
                    
                    # Try to extract content from found links
                    collected = []
                    for article_url, title in article_links[:limit]:
                        try:
                            content, article_data = self._extract_article_content(article_url, title)
                            if content and self._is_valid_content(content):
                                article_entry = {
                                    "title": title,
                                    "url": article_url,
                                    "source": domain,
                                    "bias_label": source.get("bias", "unknown"),
                                    "published": article_data.get('published'),
                                    "content": content,
                                    "tags": [],
                                    "collected_at": datetime.now().isoformat(),
                                    "extraction_method": "simple_fallback"
                                }
                                collected.append(article_entry)
                                
                                if len(collected) >= limit:
                                    break
                                    
                                time.sleep(2)  # Delay between extractions
                        
                        except Exception as e:
                            self.logger.debug(f"Failed to extract {article_url}: {e}")
                            continue
                    
                    if collected:
                        self.logger.info(f"Successfully collected {len(collected)} articles via simple scraping")
                        return collected
                
                time.sleep(2)  # Delay between scraping attempts
                
            except Exception as e:
                self.logger.debug(f"Simple scraping of {scrape_url} failed: {e}")
                continue
        
        self.logger.info(f"All fallback strategies failed for {domain}")
        return []
            
    def _extract_rss_content(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract content from RSS entry itself"""
        # Try different content fields
        content = None
        
        # Check for content:encoded (common in RSS 2.0)
        if hasattr(entry, 'content'):
            for c in entry.content:
                if c.get('value'):
                    content = c['value']
                    break
        
        # Check for summary/description
        if not content and hasattr(entry, 'summary'):
            content = entry.summary
            
        if not content and hasattr(entry, 'description'):
            content = entry.description
        
        # Clean HTML if present
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text if len(text) > 100 else None
            
        return None

    def _extract_article_content(self, url: str, title: str = None, pub_date: Optional[datetime] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extract article content using multiple methods with fallbacks.
        """
        article_data = {
            "authors": [],
            "published": None,
            "top_image": None,
            "images": [],
            "movies": []
        }
        
        content = None
        
        # Check if this is a problematic domain
        problematic_domains = [
            'wsj.com', 'nytimes.com', 'washingtonpost.com', 
            'ft.com', 'bloomberg.com', 'businessinsider.com'
        ]
        
        domain = urlparse(url).netloc.lower()
        if any(prob_domain in domain for prob_domain in problematic_domains):
            self.logger.warning(f"Known paywall/bot-blocking domain: {domain}")
            # For these, we'll only try if we have cloudscraper
            if self.bypass_cloudflare and TRAFILATURA_AVAILABLE:
                try:
                    response = self.cloudscraper_session.get(url, timeout=self.request_timeout)
                    html = response.text
                    
                    content = trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=False,
                        no_fallback=False,
                        favor_precision=True,
                        target_language='en'
                    )
                    
                    if content and self._is_valid_content(content):
                        if pub_date:
                            article_data["published"] = pub_date.isoformat()
                        return content, article_data
                except Exception as e:
                    self.logger.debug(f"Failed to extract from paywall site: {e}")
            
            return None, article_data
        
        # Method 1: Try trafilatura first (most reliable)
        if TRAFILATURA_AVAILABLE:
            try:
                content = self._extract_with_trafilatura(url)
                if self._is_valid_content(content):
                    self.logger.debug(f"Successfully extracted with trafilatura: {url}")
                    if pub_date:
                        article_data["published"] = pub_date.isoformat()
                    return content, article_data
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [403, 429]:
                    self.logger.warning(f"{e.response.status_code} error for {url}")
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
                if "403" in str(e) or "429" in str(e):
                    self.logger.warning(f"Rate limited for {url}")
                    return None, article_data
                self.logger.debug(f"Newspaper3k extraction failed: {e}")
        
        # Method 3: Fall back to BeautifulSoup
        try:
            self.logger.info(f"Trying BeautifulSoup fallback for {url}")
            content = self._extract_with_beautifulsoup(url)
            
            # Update article_data with any missing info
            if article_data.get("published") is None and pub_date:
                article_data["published"] = pub_date.isoformat()
            
            if self._is_valid_content(content):
                self.logger.debug(f"Successfully extracted with BeautifulSoup: {url}")
                return content, article_data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [403, 429]:
                self.logger.warning(f"{e.response.status_code} error for {url}")
                return None, article_data
            self.logger.error(f"BeautifulSoup extraction failed: {e}")
        except Exception as e:
            self.logger.error(f"BeautifulSoup extraction failed: {e}")
        
        # If all methods failed
        self.logger.error(f"All content extraction methods failed for {url}")
        return None, article_data

    def _extract_with_trafilatura(self, url: str) -> Optional[str]:
        """Extract content using trafilatura."""
        # Add delay before request
        time.sleep(random.uniform(1, 2))
        
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
        """Extract content using newspaper3k."""
        # Configure article
        config = Config()
        config.browser_user_agent = random.choice(self.user_agents)
        config.request_timeout = self.request_timeout
        config.memoize_articles = False
        config.fetch_images = False
        config.language = 'en'
        
        # Create article instance
        article = Article(url, config=config)
        
        # Add delay before request
        time.sleep(random.uniform(1, 2))
        
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
        """Extract content using BeautifulSoup."""
        # Add delay before request
        time.sleep(random.uniform(1, 2))
        
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
        for script in soup(["script", "style", "noscript"]):
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
        
        # Enhanced selectors for article content
        selectors = [
            'article',
            '[itemprop="articleBody"]',
            '.article-content',
            '.article-body',
            '.story-body',
            '.story-content',
            '.entry-content',
            '.post-content',
            '.content-body',
            '.article__body',
            '.article-text',
            '.story-text',
            'div.content',
            'div.text',
            'main',
            '[role="main"]',
            '.main-content',
            '#main-content'
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
                for unwanted in container.select('aside, .related, .advertisement, nav, header, footer, .comments, .social-share, .newsletter-signup'):
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

    def _is_valid_content(self, content: Optional[str]) -> bool:
        """Check if content meets quality criteria."""
        if not content:
            return False
            
        # Remove extra whitespace for accurate counting
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Check minimum length
        if len(content) < 200:
            return False
        
        # Check word count
        word_count = len(content.split())
        if word_count < 50:
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
        """Check if article is related to government or politics."""
        # 1) Check for government/politics tags
        if tags:
            political_tags = {"politics", "government", "us politics", "policy", "election", 
                             "washington", "congress", "white house", "federal", "democracy"}
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
        political_figures = {
            "trump", "biden", "harris", "pence", "pelosi", "mcconnell", 
            "schumer", "mccarthy", "johnson", "jeffries"
        }
        for figure in political_figures:
            if figure in sample_text:
                return True
        
        return False

    def _generate_document_id(self, article: Dict[str, Any]) -> str:
        """Generate a simple document ID based on URL and timestamp."""
        url = article.get("url", "")
        timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
        
        # Create URL hash
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
        
        # Combine with timestamp
        return f"{url_hash}_{timestamp}"

    def _create_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata dictionary for document storage."""
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
            "id": article.get("id"),
            "extraction_method": article.get("extraction_method", "traditional")
        }