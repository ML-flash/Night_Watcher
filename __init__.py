"""
Night_watcher Framework
A modular system for analyzing news, identifying divisive content, and generating counter-narratives.
"""

__version__ = "0.1.0"

from .agents.base import Agent, LLMProvider
from .workflow.orchestrator import NightWatcherWorkflow
