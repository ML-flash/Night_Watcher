#!/usr/bin/env python3
"""
Night_watcher - Installer Script
Downloads and installs the Night_watcher framework as a standalone application.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import argparse
from pathlib import Path
import urllib.request
import zipfile
import io
import time
import random

# Base directory for the Night_watcher installation
DEFAULT_INSTALL_DIR = os.path.join(os.path.expanduser("~"), ".documents", "analysis_tool")

# Required dependencies
REQUIRED_PACKAGES = [
    "requests>=2.28.1",
    "feedparser>=6.0.10",
    "newspaper3k>=0.2.8", 
    "numpy>=1.22.0"
]

# Optional enhanced capabilities
ENHANCED_PACKAGES = [
    "faiss-cpu>=1.7.3",
    "chromadb>=0.3.26",
    "sentence-transformers>=2.2.2"
]

# File structure to create
FILE_STRUCTURE = [
    "agents/__init__.py",
    "agents/analyzer.py",
    "agents/base.py",
    "agents/collector.py",
    "agents/counter_narrative.py", 
    "agents/distribution.py",
    "agents/lm_studio.py",
    "agents/strategic.py",
    "analysis/__init__.py",
    "analysis/patterns.py",
    "memory/__init__.py",
    "memory/system.py",
    "utils/__init__.py",
    "utils/helpers.py",
    "utils/io.py",
    "utils/logging.py", 
    "utils/text.py",
    "workflow/__init__.py",
    "workflow/orchestrator.py",
    "config.py",
    "main.py",
    "__init__.py",
    "data/collected/.gitkeep",
    "data/analyzed/.gitkeep",
    "data/counter_narratives/.gitkeep",
    "data/memory/.gitkeep",
    "logs/.gitkeep"
]

REPO_URLs = [
    "https://raw.githubusercontent.com/user/night_watcher/main/{file}",
    "https://raw.githubusercontent.com/user/analysis-tools/main/{file}"
]

def print_banner():
    """Print installation banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  Night_watcher Framework - Covert Installation            ║
    ║                                                           ║
    ║  A counter-narrative tool for democratic resilience       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def install_dependencies(enhanced=False):
    """Install required dependencies"""
    print("[+] Installing dependencies...")
    
    packages = REQUIRED_PACKAGES.copy()
    if enhanced:
        packages.extend(ENHANCED_PACKAGES)
        
    for package in packages:
        try:
            print(f"    Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package], 
                                 stdout=subprocess.DEVNULL)
            print(f"    ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Error installing {package}: {e}")
            return False
            
    return True

def create_structure(base_dir):
    """Create the file and directory structure"""
    print(f"[+] Creating directory structure in {base_dir}...")
    
    for file_path in FILE_STRUCTURE:
        full_path = os.path.join(base_dir, file_path)
        
        # Create directory if needed
        directory = os.path.dirname(full_path)
        if directory:
            ensure_dir(directory)
            
        # Create empty file if it's not a directory marker
        if not file_path.endswith("/.gitkeep"):
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass
                    
    print(f"    ✓ Directory structure created")
    return True

def download_file(url, destination):
    """Download a file from URL to destination"""
    try:
        # Add a random delay to avoid detection patterns
        time.sleep(random.uniform(0.5, 2.0))
        
        # Create a request with user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        print(f"    ✗ Error downloading {url}: {e}")
        return False

def create_launcher(base_dir):
    """Create launcher script"""
    launcher_path = os.path.join(base_dir, "nightwatcher.py")
    
    launcher_content = f"""#!/usr/bin/env python3
import os
import sys

# Add the installation directory to the Python path
sys.path.insert(0, "{base_dir}")

# Import and run the main function
from main import main

if __name__ == "__main__":
    sys.exit(main())
"""
    
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    # Make executable
    os.chmod(launcher_path, 0o755)
    
    print(f"    ✓ Launcher script created at {launcher_path}")
    return launcher_path

def main():
    """Main installer function"""
    parser = argparse.ArgumentParser(
        description="Night_watcher Framework Installer"
    )
    
    parser.add_argument("--dir", default=DEFAULT_INSTALL_DIR,
                      help="Installation directory (default: hidden in user's home)")
    parser.add_argument("--enhanced", action="store_true",
                      help="Install enhanced capabilities (may require additional dependencies)")
    parser.add_argument("--no-deps", action="store_true", 
                      help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Install dependencies
    if not args.no_deps:
        if not install_dependencies(args.enhanced):
            print("[!] Error installing dependencies. Please fix the issues and try again.")
            return 1
    
    # Create directory structure
    if not create_structure(args.dir):
        print("[!] Error creating directory structure.")
        return 1
    
    # Create launcher
    launcher_path = create_launcher(args.dir)
    
    # Instructions
    print("\n" + "=" * 70)
    print(f"Night_watcher Framework installed to: {args.dir}")
    print("\nTo use Night_watcher:")
    print(f"  1. Place source code files in the appropriate directories")
    print(f"  2. Run: {launcher_path} init")
    print(f"  3. Then: {launcher_path} run --config {os.path.join(args.dir, 'config.json')}")
    print("\nOptional: Create an alias for easier access:")
    print(f"  alias nightwatcher=\"{launcher_path}\"")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
