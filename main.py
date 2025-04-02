#!/usr/bin/env python3
"""
Night_watcher Framework - Main Entry Point
A modular system for analyzing news, identifying divisive content, and generating counter-narratives.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# Modify these imports to use absolute imports
import config
from agents.base import LLMProvider
from agents.lm_studio import LMStudioProvider
from workflow.orchestrator import NightWatcherWorkflow
from memory.system import MemorySystem
from analysis.patterns import PatternRecognition
from utils.logging import setup_logging


# The rest of the script remains the same as in the original file you showed

def main() -> int:
    """Main entry point for Night_watcher"""
    parser = create_cli_parser()
    args = parser.parse_args()

    if args.command == "run":
        return handle_run_command(args)
    elif args.command == "init":
        return handle_init_command(args)
    elif args.command == "analyze":
        return handle_analyze_command(args)
    elif args.command == "search":
        return handle_search_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())