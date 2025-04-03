#!/usr/bin/env python3
"""
Night_watcher - Bundler Script
Creates a self-contained distribution package for the Night_watcher framework.
"""

import os
import sys
import shutil
import zipfile
import base64
import argparse
from pathlib import Path
import tempfile

# Source code directory (current directory by default)
DEFAULT_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

def print_banner():
    """Print installation banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  Night_watcher Framework - Distribution Builder           ║
    ║                                                           ║
    ║  Creates a self-contained deployment package              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_zip_archive(source_dir, output_file):
    """Create a ZIP archive of the source directory"""
    print(f"[+] Creating ZIP archive from {source_dir}...")
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Skip __pycache__ directories
            if '__pycache__' in root:
                continue
            
            # Skip .git directories
            if '.git' in root:
                continue
                
            for file in files:
                # Skip .pyc files
                if file.endswith('.pyc'):
                    continue
                    
                # Skip the output file itself if it's in the source directory
                if os.path.abspath(os.path.join(root, file)) == os.path.abspath(output_file):
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
                print(f"    Added: {arcname}")
    
    print(f"    ✓ ZIP archive created at {output_file}")
    return True

def create_self_extracting_script(zip_file, output_file):
    """Create a self-extracting Python script"""
    print(f"[+] Creating self-extracting script at {output_file}...")
    
    # Read the ZIP file as binary data
    with open(zip_file, 'rb') as f:
        zip_data = f.read()
    
    # Encode as base64
    zip_base64 = base64.b64encode(zip_data).decode('utf-8')
    
    # Read the setup script template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_template_path = os.path.join(script_dir, "setup_nightwatcher.py")
    
    if not os.path.exists(setup_template_path):
        print(f"    ✗ Setup template not found at {setup_template_path}")
        # Use default template content if not found
        with open(os.path.join(script_dir, "utils", "setup_template.py"), 'r') as f:
            setup_template = f.read()
    else:
        with open(setup_template_path, 'r') as f:
            setup_template = f.read()
    
    # Replace the placeholder with the actual ZIP data
    setup_script = setup_template.replace("    # BASE64_ENCODED_ZIP_WILL_BE_HERE", zip_base64)
    
    # Write to the output file
    with open(output_file, 'w') as f:
        f.write(setup_script)
    
    # Make the script executable
    os.chmod(output_file, 0o755)
    
    print(f"    ✓ Self-extracting script created at {output_file}")
    return True

def create_standalone_installer(source_dir, output_file):
    """Create a standalone installer that includes all source code"""
    # Create a temporary ZIP file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        temp_zip_path = temp_zip.name
    
    try:
        # Create ZIP archive
        if not create_zip_archive(source_dir, temp_zip_path):
            print("[!] Failed to create ZIP archive.")
            return False
        
        # Create self-extracting script
        if not create_self_extracting_script(temp_zip_path, output_file):
            print("[!] Failed to create self-extracting script.")
            return False
        
        return True
    finally:
        # Clean up the temporary ZIP file
        if os.path.exists(temp_zip_path):
            os.unlink(temp_zip_path)

def main():
    """Main bundler function"""
    parser = argparse.ArgumentParser(
        description="Night_watcher Framework Distribution Builder"
    )
    
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR,
                      help="Source code directory")
    parser.add_argument("--output", default="install_nightwatcher.py",
                      help="Output installer script")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Create standalone installer
    if create_standalone_installer(args.source, args.output):
        print("\n" + "=" * 70)
        print(f"Distribution package created: {args.output}")
        print("\nTo deploy Night_watcher on a target system:")
        print(f"  1. Transfer {args.output} to the target system")
        print(f"  2. Run: python {args.output}")
        print(f"  3. Follow the instructions to complete setup")
        print("=" * 70)
        return 0
    else:
        print("[!] Failed to create distribution package.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
