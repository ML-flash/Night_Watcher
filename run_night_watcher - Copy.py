#!/usr/bin/env python3
"""
Night_watcher Quick Start Script
One command to rule them all!
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required = ['flask', 'flask_cors', 'requests', 'feedparser', 'beautifulsoup4']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ Dependencies installed!")
        else:
            print("Please install dependencies manually:")
            print(f"pip install {' '.join(missing)}")
            sys.exit(1)
    else:
        print("✅ All dependencies found!")

def check_config():
    """Check if config exists and has API key."""
    if not os.path.exists("config.json"):
        print("\n📝 No config.json found - will create on first run")
        return
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  No Anthropic API key found!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Or Night_watcher will try to use LM Studio locally")

def main():
    print("""
    🌙 Night_watcher Intelligence Framework
    =====================================
    Fighting authoritarian patterns with data
    """)
    
    # Check dependencies
    check_dependencies()
    
    # Check configuration
    check_config()
    
    # Determine host for codespace
    host = "127.0.0.1"
    if os.environ.get("CODESPACES"):
        host = "0.0.0.0"
        print("🚀 Running in GitHub Codespace")
    
    # Start the web server
    print(f"\n🌐 Starting Night_watcher web server on {host}:5000...")
    
    try:
        # Run the web server
        if os.path.exists("Night_Watcher.py"):
            subprocess.run([sys.executable, "Night_Watcher.py", "--web", "--host", host])
        else:
            print("Error: Night_Watcher.py not found!")
            print("Make sure you're running from the Night_watcher directory")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Night_watcher shutting down... Stay vigilant!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
