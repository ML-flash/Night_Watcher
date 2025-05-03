"""
Night_watcher Utilities
Utility functions for the Night_watcher system.
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Date Tracking Utilities
# ==========================================

def get_last_run_date(data_dir: str) -> datetime:
    """
    Get the last run date or return the default start date (Jan 20, 2025).
    
    Args:
        data_dir: Directory where date tracking is stored
        
    Returns:
        Last run date as datetime object
    """
    date_file = os.path.join(data_dir, "last_run_date.txt")
    
    if os.path.exists(date_file):
        try:
            with open(date_file, 'r') as f:
                date_str = f.read().strip()
                return datetime.fromisoformat(date_str)
        except (ValueError, IOError) as e:
            logger.error(f"Error reading last run date: {str(e)}")
            # Return default date if there's an error reading the file
            return datetime(2025, 1, 20)
    else:
        # Default to inauguration day if no previous run
        logger.info("No previous run date found, starting from inauguration day (Jan 20, 2025)")
        return datetime(2025, 1, 20)

def save_run_date(data_dir: str, date: Optional[datetime] = None) -> bool:
    """
    Save the current date as the last run date.
    
    Args:
        data_dir: Directory where date tracking is stored
        date: Date to save as last run date (defaults to current date)
        
    Returns:
        True if successful, False otherwise
    """
    if date is None:
        date = datetime.now()
        
    date_file = os.path.join(data_dir, "last_run_date.txt")
    
    try:
        os.makedirs(os.path.dirname(date_file), exist_ok=True)
        
        with open(date_file, 'w') as f:
            f.write(date.isoformat())
            
        logger.info(f"Saved run date: {date.isoformat()}")
        return True
    except Exception as e:
        logger.error(f"Error saving run date: {str(e)}")
        return False

def get_analysis_date_range(data_dir: str, days_overlap: int = 1) -> Tuple[datetime, datetime]:
    """
    Get the date range for the current analysis run with optional overlap
    to ensure no gaps in coverage.
    
    Args:
        data_dir: Directory where date tracking is stored
        days_overlap: Number of days to overlap with previous run (default: 1)
        
    Returns:
        Tuple of (start_date, end_date) for the current run
    """
    start_date = get_last_run_date(data_dir)
    end_date = datetime.now()
    
    # Apply overlap to avoid gaps
    if days_overlap > 0:
        start_date = start_date - timedelta(days=days_overlap)
    
    logger.info(f"Analysis date range: {start_date.isoformat()} to {end_date.isoformat()}")
    return start_date, end_date

# ==========================================
# Text Processing Utilities
# ==========================================

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

# ==========================================
# File I/O Utilities
# ==========================================

def save_to_file(content: Any, filepath: str) -> bool:
    """
    Save content to a file with appropriate format.

    Args:
        content: The content to save (dict, list, or string)
        filepath: Path where the file should be saved

    Returns:
        True if save was successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(content) if content is not None else "No content generated")

        return True
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {str(e)}")
        return False

def load_json_file(filepath: str) -> Any:
    """
    Load JSON from a file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed JSON data, or None if file couldn't be loaded
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        return None

def load_text_file(filepath: str) -> str:
    """
    Load text from a file.

    Args:
        filepath: Path to the text file

    Returns:
        File contents as string, or empty string if file couldn't be loaded
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading text from {filepath}: {str(e)}")
        return ""
