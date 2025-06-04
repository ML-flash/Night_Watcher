#!/usr/bin/env python3
"""
Fix SSL certificate issues for Windows/Anaconda environments
"""

import ssl
import certifi
import os

# Print current certificate locations
print("Current SSL certificate configuration:")
print(f"Default cert file: {ssl.get_default_verify_paths().cafile}")
print(f"Certifi cert file: {certifi.where()}")

# Set environment variables for certificate bundle
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

print("\nEnvironment variables set:")
print(f"SSL_CERT_FILE: {os.environ.get('SSL_CERT_FILE')}")
print(f"REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE')}")

# Test HTTPS connection
import requests
try:
    response = requests.get('https://www.google.com', timeout=5)
    print("\nSSL test successful!")
except Exception as e:
    print(f"\nSSL test failed: {e}")

print("\nTo make these changes permanent, add these to your environment variables:")
print(f"SSL_CERT_FILE={certifi.where()}")
print(f"REQUESTS_CA_BUNDLE={certifi.where()}")
