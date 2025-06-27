"""Government web scrapers for Night_Watcher.

These helpers provide lightweight HTML scraping for various
federal resources when API access is not available.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
}


def scrape_federal_register(start_date: datetime, end_date: datetime, limit: int = 50) -> List[Dict[str, str]]:
    """Scrape documents from the Federal Register website within a date range."""
    results = []
    url = "https://www.federalregister.gov/documents/search"
    params = {
        "conditions[publication_date][gte]": start_date.strftime("%Y-%m-%d"),
        "conditions[publication_date][lte]": end_date.strftime("%Y-%m-%d"),
        "page": 1,
    }
    while len(results) < limit:
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logger.warning("Federal Register HTML request failed: %s", resp.status_code)
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            entries = soup.select(".document-row")
            if not entries:
                break
            for row in entries:
                if len(results) >= limit:
                    break
                title_tag = row.find(class_="document-title")
                link_tag = title_tag.find("a") if title_tag else None
                summary_tag = row.find(class_="abstract")
                date_tag = row.find(class_="meta")
                if link_tag:
                    results.append({
                        "title": link_tag.get_text(strip=True),
                        "url": "https://www.federalregister.gov" + link_tag.get("href", ""),
                        "content": summary_tag.get_text(strip=True) if summary_tag else "",
                        "published": date_tag.get_text(strip=True) if date_tag else "",
                        "source": "Federal Register",
                    })
            params["page"] += 1
        except Exception as exc:
            logger.error("Federal Register scrape error: %s", exc)
            break
    return results


def scrape_white_house_actions(start_date: datetime, end_date: datetime, limit: int = 50) -> List[Dict[str, str]]:
    """Scrape presidential actions from whitehouse.gov."""
    base_url = "https://www.whitehouse.gov/briefing-room/presidential-actions/"
    page = 1
    results = []
    while len(results) < limit:
        url = f"{base_url}page/{page}/" if page > 1 else base_url
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = soup.select("article")
            if not articles:
                break
            for art in articles:
                if len(results) >= limit:
                    break
                date_tag = art.find("time")
                if not date_tag:
                    continue
                pub_date = datetime.strptime(date_tag.get("datetime")[:10], "%Y-%m-%d")
                if pub_date < start_date or pub_date > end_date:
                    continue
                link = art.find("a", class_="news-item__title")
                if not link:
                    continue
                summary = art.find(class_="news-item__dek")
                results.append({
                    "title": link.get_text(strip=True),
                    "url": link.get("href"),
                    "content": summary.get_text(strip=True) if summary else "",
                    "published": pub_date.isoformat(),
                    "source": "White House",
                })
            page += 1
        except Exception as exc:
            logger.error("White House scrape error: %s", exc)
            break
    return results


def scrape_congress_bills(start_date: datetime, end_date: datetime, limit: int = 50) -> List[Dict[str, str]]:
    """Scrape recent bills from congress.gov search results."""
    results = []
    page = 1
    while len(results) < limit:
        url = f"https://www.congress.gov/search?pageSize=100&page={page}&q=%7B%22source%22%3A%22legislation%22%7D"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select("ol.basic-search-results-lists li")
            if not items:
                break
            for item in items:
                if len(results) >= limit:
                    break
                date_span = item.find(class_="result-item")
                if not date_span:
                    continue
                date_text = date_span.get_text()
                try:
                    pub_date = datetime.strptime(date_text.strip(), "%B %d, %Y")
                except Exception:
                    continue
                if pub_date < start_date or pub_date > end_date:
                    continue
                link = item.find("a")
                if not link:
                    continue
                title = link.get_text(strip=True)
                results.append({
                    "title": title,
                    "url": "https://www.congress.gov" + link.get("href"),
                    "content": "",
                    "published": pub_date.isoformat(),
                    "source": "Congress.gov",
                })
            page += 1
        except Exception as exc:
            logger.error("Congress bill scrape error: %s", exc)
            break
    return results


def fetch_federal_register_api(start_date: datetime, end_date: datetime, limit: int = 200) -> List[Dict[str, str]]:
    """Fetch Federal Register documents via the official JSON API."""
    results: List[Dict[str, str]] = []
    page = 1
    base_url = "https://www.federalregister.gov/api/v1/documents.json"
    while len(results) < limit:
        params = {
            "conditions[publication_date][gte]": start_date.strftime("%Y-%m-%d"),
            "conditions[publication_date][lte]": end_date.strftime("%Y-%m-%d"),
            "per_page": 100,
            "page": page,
        }
        try:
            resp = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logger.warning("Federal Register API error: %s", resp.status_code)
                break
            data = resp.json()
            docs = data.get("results", [])
            if not docs:
                break
            for doc in docs:
                if len(results) >= limit:
                    break
                results.append({
                    "title": doc.get("title", ""),
                    "url": doc.get("html_url", ""),
                    "content": doc.get("abstract", ""),
                    "published": doc.get("publication_date", ""),
                    "source": "Federal Register",
                })
            page += 1
        except Exception as exc:
            logger.error("Federal Register API fetch error: %s", exc)
            break
    return results


def fetch_white_house_actions_api(start_date: datetime, end_date: datetime, limit: int = 200) -> List[Dict[str, str]]:
    """Fetch presidential actions via the whitehouse.gov WordPress API."""
    results: List[Dict[str, str]] = []
    page = 1
    base_url = "https://www.whitehouse.gov/wp-json/wp/v2/posts"
    after = f"{start_date.strftime('%Y-%m-%d')}T00:00:00"
    before = f"{end_date.strftime('%Y-%m-%d')}T23:59:59"
    while len(results) < limit:
        params = {
            "per_page": 100,
            "page": page,
            "after": after,
            "before": before,
        }
        try:
            resp = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                break
            posts = resp.json()
            if not posts:
                break
            for post in posts:
                if len(results) >= limit:
                    break
                link = post.get("link", "")
                if "presidential-actions" not in link:
                    continue
                results.append({
                    "title": post.get("title", {}).get("rendered", ""),
                    "url": link,
                    "content": BeautifulSoup(post.get("excerpt", {}).get("rendered", ""), "html.parser").get_text(strip=True),
                    "published": post.get("date", ""),
                    "source": "White House",
                })
            page += 1
        except Exception as exc:
            logger.error("White House API fetch error: %s", exc)
            break
    return results


def fetch_govinfo_bills_api(start_date: datetime, end_date: datetime, api_key: Optional[str] = None, limit: int = 200) -> List[Dict[str, str]]:
    """Fetch bill metadata from the govinfo API."""
    results: List[Dict[str, str]] = []
    key = api_key or "DEMO_KEY"
    page = 1
    base_url = f"https://api.govinfo.gov/collections/BILLS/{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    while len(results) < limit:
        params = {
            "page": page,
            "pageSize": 100,
            "api_key": key,
        }
        try:
            resp = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logger.warning("govinfo API error: %s", resp.status_code)
                break
            data = resp.json()
            packages = data.get("packages", [])
            if not packages:
                break
            for pkg in packages:
                if len(results) >= limit:
                    break
                link = pkg.get("packageLink", "")
                results.append({
                    "title": pkg.get("title", ""),
                    "url": link,
                    "content": "",
                    "published": pkg.get("lastModified", ""),
                    "source": "Congress.gov",
                })
            page += 1
        except Exception as exc:
            logger.error("govinfo API fetch error: %s", exc)
            break
    return results
