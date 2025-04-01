"""
Night_watcher Text Utilities
Utility functions for text processing.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def create_slug(title: str, max_length: int = 40) -> str:
    """
    Create a URL-friendly slug from a title.

    Args:
        title: The title to convert to a slug
        max_length: Maximum length of the slug (default: 40)

    Returns:
        A URL-friendly slug string
    """
    if not title:
        return "untitled"

    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    # Replace whitespace with hyphens
    slug = re.sub(r'[\s]+', '-', slug)
    # Trim to max length
    slug = slug[:max_length]
    # Remove trailing hyphens
    slug = slug.rstrip('-')

    return slug


def extract_manipulation_score(analysis_text: str) -> int:
    """
    Extract manipulation score from analysis text.

    Args:
        analysis_text: The text of the analysis to extract score from

    Returns:
        An integer score from 1-10 or 0 if no score can be extracted
    """
    try:
        if "MANIPULATION SCORE" in analysis_text:
            score_text = analysis_text.split("MANIPULATION SCORE:")[1].split("\n")[0]
            # Extract numbers from text
            numbers = [int(s) for s in re.findall(r'\d+', score_text)]
            if numbers:
                return numbers[0]
        return 0
    except Exception as e:
        logger.error(f"Error extracting manipulation score: {str(e)}")
        return 0


def truncate_text(text: str, max_length: int = 5000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length (default: 5000)
        suffix: Suffix to add to truncated text (default: "...")

    Returns:
        Truncated text with suffix if needed
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def extract_topics(text: str) -> List[str]:
    """
    Extract topics from analysis text.

    Args:
        text: Text to extract topics from

    Returns:
        List of extracted topics
    """
    topics = []

    try:
        if "MAIN TOPICS" in text:
            topics_section = text.split("MAIN TOPICS:")[1].split("\n\n")[0]

            # Simple extraction - split by commas, newlines, and clean up
            for item in re.split(r'[,\n]', topics_section):
                topic = item.strip()
                if topic and len(topic) > 3 and not topic.startswith("FRAMING"):
                    # Clean up bullet points and numbering
                    topic = re.sub(r'^[\d\.\-\*]+\s*', '', topic)
                    if topic:
                        topics.append(topic)
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")

    return topics


def extract_sentences(text: str, max_sentences: int = 3) -> str:
    """
    Extract a limited number of sentences from text.

    Args:
        text: Text to extract sentences from
        max_sentences: Maximum number of sentences to extract

    Returns:
        Extracted sentences as a string
    """
    try:
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        selected = sentences[:max_sentences]
        return ' '.join(selected)
    except Exception as e:
        logger.error(f"Error extracting sentences: {str(e)}")
        return text[:200] + "..." if len(text) > 200 else text