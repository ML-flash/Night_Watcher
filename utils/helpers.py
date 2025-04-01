"""
Night_watcher Helper Utilities
General helper functions for Night_watcher.
"""

import requests
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def safe_request(url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None,
                 timeout: int = 30, retries: int = 3, backoff_factor: float = 0.5) -> Optional[Dict[str, Any]]:
    """
    Make a safe HTTP request with retries and error handling.

    Args:
        url: URL to request
        method: HTTP method (default: "GET")
        data: Data to send with the request (default: None)
        timeout: Request timeout in seconds (default: 30)
        retries: Number of retries (default: 3)
        backoff_factor: Backoff factor for retries (default: 0.5)

    Returns:
        Response data as dict, or None if request failed
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        if method.upper() == "GET":
            response = session.get(url, timeout=timeout)
        elif method.upper() == "POST":
            response = session.post(url, json=data, timeout=timeout)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return None


def validate_article_data(article: Dict[str, Any]) -> bool:
    """
    Validate article data has the required fields.

    Args:
        article: Article data to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['title', 'content', 'source']
    return all(field in article for field in required_fields)


def rate_limiter(max_rate: int = 5, period: float = 1.0):
    """
    Decorator to apply rate limiting to a function.

    Args:
        max_rate: Maximum number of calls per period
        period: Time period in seconds

    Returns:
        Decorated function with rate limiting
    """
    import threading
    from collections import deque
    from functools import wraps

    # Track call times
    call_times = deque()
    lock = threading.Lock()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                # Clean up old call times
                now = time.time()
                while call_times and call_times[0] < now - period:
                    call_times.popleft()

                # Check if we're over the limit
                if len(call_times) >= max_rate:
                    # Calculate sleep time
                    sleep_time = call_times[0] + period - now
                    if sleep_time > 0:
                        logger.debug(f"Rate limit hit, sleeping for {sleep_time:.2f}s")
                        time.sleep(sleep_time)

                # Record this call
                call_times.append(time.time())

            # Call the function
            return func(*args, **kwargs)

        return wrapper

    return decorator