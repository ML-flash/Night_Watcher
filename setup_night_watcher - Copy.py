#!/usr/bin/env python3
"""
Night_watcher Setup and Configuration Helper
Ensures the system is properly configured before first run.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   You have: Python {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        'feedparser',
        'requests', 
        'beautifulsoup4',
        'networkx',
        'flask',
        'flask_cors',
        'numpy',
        'faiss',
        'sentence_transformers'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        response = input("\nInstall missing packages? (y/n): ")
        if response.lower() == 'y':
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                print("✅ Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install dependencies")
                return False
        return False
    
    print("✅ All required dependencies found")
    return True

def check_optional_dependencies():
    """Check optional dependencies"""
    optional = {
        'anthropic': 'Anthropic API support',
        'newspaper': 'Enhanced article extraction',
        'trafilatura': 'Alternative article extraction',
        'cloudscraper': 'Cloudflare bypass support'
    }
    
    print("\n📦 Optional dependencies:")
    for package, description in optional.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ⚪ {package}: {description} (not installed)")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data",
        "data/collected",
        "data/analyzed", 
        "data/documents",
        "data/documents/content",
        "data/documents/metadata",
        "data/documents/signatures",
        "data/documents/analysis_provenance",
        "data/knowledge_graph",
        "data/knowledge_graph/nodes",
        "data/knowledge_graph/edges",
        "data/knowledge_graph/snapshots",
        "data/knowledge_graph/provenance",
        "data/vector_store",
        "logs"
    ]
    
    print("\n📁 Creating directories:")
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")

def check_config():
    """Check and create config.json if needed"""
    if os.path.exists("config.json"):
        print("\n✅ config.json found")
        return True
    
    print("\n⚠️  config.json not found")
    response = input("Create default config.json? (y/n): ")
    
    if response.lower() == 'y':
        # The config will be created by Night_Watcher on first run
        print("✅ config.json will be created on first run")
        return True
    
    return False

def check_llm_setup():
    """Check LLM provider setup"""
    print("\n🤖 LLM Provider Check:")
    
    # Check for LM Studio
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code == 200:
            print("  ✅ LM Studio detected at localhost:1234")
            models = response.json().get("data", [])
            if models:
                print(f"     Models available: {len(models)}")
            else:
                print("     ⚠️  No models loaded in LM Studio")
        else:
            print("  ⚪ LM Studio not responding at localhost:1234")
    except:
        print("  ⚪ LM Studio not detected (start LM Studio and load a model)")
    
    # Check for Anthropic API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  ✅ Anthropic API key found in environment")
    else:
        print("  ⚪ Anthropic API key not set (optional)")
        print("     To use Anthropic: export ANTHROPIC_API_KEY='your-key-here'")

def check_analysis_templates():
    """Check for analysis templates"""
    templates = []
    for file in os.listdir("."):
        if file.endswith("_analysis.json"):
            templates.append(file)
    
    if templates:
        print(f"\n📋 Found {len(templates)} analysis template(s):")
        for template in templates:
            print(f"  ✅ {template}")
    else:
        print("\n⚠️  No analysis templates found")
        print("   standard_analysis.json will be needed for analysis")

def run_system_check():
    """Run a basic system check"""
    print("\n🔍 Running system check...")
    
    try:
        # Try importing the main module
        from Night_Watcher import NightWatcher
        
        # Try to initialize
        nw = NightWatcher()
        status = nw.status()
        
        print("\n✅ System check passed!")
        print(f"   Documents: {status['documents']['total']}")
        print(f"   Knowledge Graph Nodes: {status['knowledge_graph']['nodes']}")
        print(f"   Vector Store Items: {status['vector_store']['total_vectors']}")
        
        return True
    except Exception as e:
        print(f"\n⚠️  System check failed: {e}")
        print("   This is normal for first run")
        return False

def main():
    """Main setup function"""
    print("🌙 Night_watcher Setup Helper")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install dependencies before continuing")
        sys.exit(1)
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check config
    check_config()
    
    # Check LLM setup
    check_llm_setup()
    
    # Check templates
    check_analysis_templates()
    
    # Run system check
    run_system_check()
    
    print("\n" + "=" * 40)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. If using LM Studio: Start it and load a model")
    print("2. Run: python Night_Watcher.py --status")
    print("3. Or start web interface: python Night_Watcher.py --web")
    print("4. Or run full pipeline: python Night_Watcher.py --full")
    
    print("\nFor help, check the README.md file")

if __name__ == "__main__":
    main()
