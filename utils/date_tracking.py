"""
Night_watcher Date Tracking Utilities
Utilities for tracking and managing analysis date ranges.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

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
