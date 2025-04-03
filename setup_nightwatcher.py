#!/usr/bin/env python3
"""
Night_watcher - Setup Script
Installs dependencies and sets up the Night_watcher framework.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import argparse
from pathlib import Path
import zipfile
import io
import base64

# Base directory for the covert Night_watcher installation
DEFAULT_INSTALL_DIR = os.path.join(os.path.expanduser("~"), ".documents", "analysis_tool")

# Required dependencies for Night_watcher
REQUIREMENTS = [
    "requests>=2.28.1",
    "feedparser>=6.0.10",
    "newspaper3k>=0.2.8",
    "numpy>=1.22.0"
]

# Optional advanced dependencies
ADVANCED_REQUIREMENTS = [
    "faiss-cpu>=1.7.3",
    "chromadb>=0.3.26",
    "sentence-transformers>=2.2.2"
]

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def install_dependencies(advanced=False):
    """Install required Python dependencies"""
    print("[+] Installing dependencies...")
    
    requirements = REQUIREMENTS.copy()
    if advanced:
        requirements.extend(ADVANCED_REQUIREMENTS)
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"    - Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"    - Error installing {package}: {e}")
            return False
    
    return True

def create_directory_structure(base_dir):
    """Create the directory structure for Night_watcher"""
    print(f"[+] Creating directory structure in {base_dir}...")
    
    directories = [
        "agents",
        "analysis",
        "memory",
        "utils",
        "workflow",
        "data/collected",
        "data/analyzed",
        "data/counter_narratives",
        "data/memory",
        "logs"
    ]
    
    for directory in directories:
        ensure_dir(os.path.join(base_dir, directory))
    
    return True

def extract_codebase(base_dir):
    """Extract the Night_watcher codebase from embedded data"""
    print("[+] Extracting codebase...")
    
    # The embedded code will be inserted here by the script generator
    embedded_code_base64 = """
    # BASE64_ENCODED_ZIP_WILL_BE_HERE
    """
    
    try:
        if embedded_code_base64.strip().startswith("#"):
            # Development mode - files will be created individually
            print("    - Development mode: Creating files from templates")
            create_code_files(base_dir)
        else:
            # Production mode - extract from embedded zip
            zip_data = base64.b64decode(embedded_code_base64.strip())
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
                zip_file.extractall(base_dir)
    except Exception as e:
        print(f"    - Error extracting codebase: {e}")
        return False
    
    return True

def create_code_files(base_dir):
    """Create individual code files (for development mode)"""
    # This function will be filled by the script generator
    # Each file's content will be created individually
    pass

def create_config(base_dir):
    """Create default configuration file"""
    print("[+] Creating default configuration...")
    
    config_path = os.path.join(base_dir, "config.json")
    
    # Import the configuration module from the installed files
    sys.path.insert(0, base_dir)
    from config import create_default_config
    
    success = create_default_config(config_path)
    if success:
        print(f"    - Created configuration at {config_path}")
    else:
        print(f"    - Failed to create configuration")
    
    return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Night_watcher Setup - Covert Counter-Narrative Framework"
    )
    
    parser.add_argument("--dir", default=DEFAULT_INSTALL_DIR,
                      help="Installation directory")
    parser.add_argument("--advanced", action="store_true",
                      help="Install advanced dependencies for enhanced capabilities")
    parser.add_argument("--no-deps", action="store_true",
                      help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Night_watcher Setup")
    print("=" * 60)
    
    # Install dependencies
    if not args.no_deps:
        if not install_dependencies(args.advanced):
            print("[!] Failed to install dependencies. Aborting.")
            return 1
    
    # Create directory structure
    if not create_directory_structure(args.dir):
        print("[!] Failed to create directory structure. Aborting.")
        return 1
    
    # Extract codebase
    if not extract_codebase(args.dir):
        print("[!] Failed to extract codebase. Aborting.")
        return 1
    
    # Create default configuration
    if not create_config(args.dir):
        print("[!] Failed to create configuration. Continuing anyway.")
    
    # Create launcher script
    launcher_path = os.path.join(args.dir, "nightwatcher.py")
    with open(launcher_path, "w") as f:
        f.write(f"""#!/usr/bin/env python3
import os
import sys

# Add this directory to Python path
sys.path.insert(0, "{args.dir}")

# Import and run the main entry point
from main import main

if __name__ == "__main__":
    sys.exit(main())
""")
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    print("\n" + "=" * 60)
    print(f"Night_watcher has been installed to: {args.dir}")
    print("\nTo run Night_watcher:")
    print(f"  {launcher_path} run --config {os.path.join(args.dir, 'config.json')}")
    print("\nYou can also create an alias for easier access:")
    print(f"  alias nightwatcher=\"{launcher_path}\"")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
