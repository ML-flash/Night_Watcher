#!/usr/bin/env python3
"""Export Night_Watcher artifact with versioned provenance chain and torrent."""

from __future__ import annotations

import os
import json
import tarfile
import hashlib
import tempfile
import shutil
from datetime import datetime
from typing import List, Optional
import base64
import logging

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from document_repository import DocumentRepository
from torrent_generator import generate_torrent

logger = logging.getLogger(__name__)


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(directory: str, version: str) -> dict:
    manifest = {
        "version": version,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": {},
    }
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            rel = os.path.relpath(path, directory)
            manifest["files"][rel] = f"sha256:{file_hash(path)}"
    return manifest


def export_versioned_artifact(
    output_path: str,
    version: str,
    private_key_path: str,
    *,
    kg_dir: str = "data/knowledge_graph",
    vector_dir: str = "data/vector_store",
    documents_dir: str = "data/documents",
    previous_artifact_path: Optional[str] = None,
    bundled_files: Optional[List[str]] = None,
) -> str:
    """Export signed artifact linking to previous version."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export intelligence data
        kg = KnowledgeGraph(graph_file=os.path.join(kg_dir, "graph.json"), taxonomy_file="KG_Taxonomy.csv")
        kg.export_graph(os.path.join(tmpdir, "intelligence", "graph"))
        VectorStore(base_dir=vector_dir).export_vector_store(os.path.join(tmpdir, "intelligence", "vector_store"))
        DocumentRepository(base_dir=documents_dir, dev_mode=True).export_repository(os.path.join(tmpdir, "intelligence", "documents"))

        # Include bundled files
        bundle_dir = os.path.join(tmpdir, "bundled_files")
        os.makedirs(bundle_dir, exist_ok=True)
        bundled = []
        for f in bundled_files or []:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(bundle_dir, os.path.basename(f)))
                bundled.append(os.path.basename(f))

        # Build manifest
        manifest = build_manifest(tmpdir, version)
        with open(os.path.join(tmpdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Build provenance
        prev_hash = file_hash(previous_artifact_path) if previous_artifact_path else None
        prev_version = None
        if previous_artifact_path:
            base = os.path.basename(previous_artifact_path)
            if "_v" in base:
                prev_version = base.split("_v")[-1].split(".")[0]
        provenance = {
            "version": version,
            "previous_version": prev_version,
            "previous_hash": f"sha256:{prev_hash}" if prev_hash else None,
            "export_time": datetime.utcnow().isoformat() + "Z",
            "genesis": previous_artifact_path is None,
            "bundled_files": bundled,
        }
        with open(os.path.join(tmpdir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

        # Sign manifest + provenance
        data = json.dumps(manifest, sort_keys=True).encode() + json.dumps(provenance, sort_keys=True).encode()
        digest = hashlib.sha256(data).hexdigest()
        private_key = serialization.load_pem_private_key(open(private_key_path, "rb").read(), password=None)
        signature_bytes = private_key.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        signature = {
            "algorithm": "RSA-PSS-SHA256",
            "signature": base64.b64encode(signature_bytes).decode(),
            "signed_data_hash": f"sha256:{digest}",
        }
        with open(os.path.join(tmpdir, "signature.json"), "w", encoding="utf-8") as f:
            json.dump(signature, f, indent=2)

        # Include public key
        pub_bytes = private_key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        with open(os.path.join(tmpdir, "public_key.pem"), "wb") as f:
            f.write(pub_bytes)

        # Package
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmpdir, arcname=".")

    torrent_name = f"night_watcher_{version}.torrent"
    generate_torrent(output_path, torrent_name)
    print(f"Exported versioned artifact to {output_path} and {torrent_name}")
    return output_path


def _create_genesis_release(output_path: str, version: str, bundle_files: List[str], nw_instance) -> bool:
    """Create v001 genesis release with complete platform."""

    with tempfile.TemporaryDirectory() as temp_dir:
        intelligence_dir = os.path.join(temp_dir, "intelligence")
        _export_intelligence_data(intelligence_dir, nw_instance)

        client_dir = os.path.join(temp_dir, "client_module")
        _bundle_distribution_client(client_dir)

        platform_dir = os.path.join(temp_dir, "platform")
        _bundle_web_platform(platform_dir)

        docs_dir = os.path.join(temp_dir, "documentation")
        _bundle_documentation(docs_dir)

        if bundle_files:
            bundle_dir = os.path.join(temp_dir, "bundled_files")
            _bundle_additional_files(bundle_dir, bundle_files)

        provenance = {
            "version": version,
            "previous_version": None,
            "previous_hash": None,
            "genesis": True,
            "export_time": datetime.now().isoformat(),
            "bundled_files": bundle_files or []
        }

        return _sign_and_package(temp_dir, output_path, provenance)


def _create_incremental_release(output_path: str, version: str, previous_path: str, bundle_files: List[str], nw_instance) -> bool:
    """Create v002+ incremental release."""

    with tempfile.TemporaryDirectory() as temp_dir:
        previous_hash = _calculate_file_hash(previous_path)

        intelligence_dir = os.path.join(temp_dir, "intelligence")
        _export_intelligence_delta(intelligence_dir, nw_instance)

        if bundle_files:
            bundle_dir = os.path.join(temp_dir, "bundled_files")
            _bundle_additional_files(bundle_dir, bundle_files)

        from version_manager import VersionManager
        version_mgr = VersionManager()

        provenance = {
            "version": version,
            "previous_version": version_mgr.get_current_version(),
            "previous_hash": previous_hash,
            "genesis": False,
            "export_time": datetime.now().isoformat(),
            "bundled_files": bundle_files or []
        }

        return _sign_and_package(temp_dir, output_path, provenance)


def _bundle_distribution_client(client_dir: str):
    """Copy distribution client files into package."""
    os.makedirs(client_dir, exist_ok=True)

    client_files = [
        "distribution_client.py",
        "install_genesis.py",
        "requirements.txt"
    ]

    for file in client_files:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(client_dir, os.path.basename(file)))


def _bundle_web_platform(platform_dir: str):
    """Bundle web UI for standalone operation."""
    os.makedirs(platform_dir, exist_ok=True)

    if os.path.exists("night_watcher_dashboard.html"):
        shutil.copy2("night_watcher_dashboard.html", platform_dir)

    # Standalone API server
    if os.path.exists("standalone_api.py"):
        shutil.copy2("standalone_api.py", platform_dir)


def _bundle_documentation(docs_dir: str):
    """Bundle user documentation."""
    os.makedirs(docs_dir, exist_ok=True)

    _create_user_guide(os.path.join(docs_dir, "user_guide.md"))
    _create_technical_specs(os.path.join(docs_dir, "technical_specs.md"))
    _create_readme(os.path.join(docs_dir, "README.md"))
    _create_donation_info(os.path.join(docs_dir, "donation_info.md"))


def _export_intelligence_data(path: str, nw_instance):
    """Export full intelligence data."""
    kg = nw_instance.knowledge_graph
    vs = nw_instance.vector_store
    repo = nw_instance.document_repository

    kg.export_graph(os.path.join(path, "graph"))
    vs.export_vector_store(os.path.join(path, "vector_store"))
    repo.export_repository(os.path.join(path, "documents"))


def _export_intelligence_delta(path: str, nw_instance):
    """Export only new/changed intelligence data since last release."""
    # Placeholder implementation - exports all data like genesis
    _export_intelligence_data(path, nw_instance)


def _bundle_additional_files(dest_dir: str, files: List[str]):
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.copytree(f, os.path.join(dest_dir, os.path.basename(f)), dirs_exist_ok=True)
            else:
                shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))


def _create_user_guide(file_path: str):
    """Create user guide for complete beginners."""
    content = """# Night_watcher Intelligence Platform - User Guide

## Quick Start
1. Extract the package: `tar -xzf night_watcher_v001.tar.gz`
2. Run installer: `cd client_module && python install_genesis.py`
3. Start web interface: `python platform/standalone_api.py`
4. Open browser: http://localhost:5000

## What This Is
Night_watcher analyzes political intelligence to detect authoritarian patterns and threats to democracy.

## Basic Usage
- **Browse Intelligence**: Use web interface to explore data
- **Search Entities**: Find political actors, institutions, events
- **Analyze Patterns**: Look for authoritarian behavior indicators
- **Check for Updates**: Client will notify when new intelligence available

## Getting Updates
Updates contain new intelligence data and analysis tools. Install manually when notified.

## Support
- Technical docs: See technical_specs.md
- Donations: See donation_info.md
- Community: [Add contact methods]
"""
    with open(file_path, 'w') as f:
        f.write(content)


def _create_technical_specs(file_path: str):
    """Create technical specifications for developers."""
    content = """# Night_watcher Technical Specifications

## Architecture
- SQLite database for intelligence storage
- Flask API for web interface
- Cryptographic verification for updates
- Modular analysis tools

## Database Schema
[Document the schema used by distribution_client]

## API Endpoints
[Document the standalone_api endpoints]

## Building Custom Tools
[Provide examples and guidance]

## Update System
[Explain provenance chain and verification]
"""
    with open(file_path, 'w') as f:
        f.write(content)


def _create_donation_info(file_path: str):
    """Create donation information."""
    content = """# Support Night_watcher Development

## Bitcoin Donations
[Add bitcoin address and QR code]

## Why Donate
Supporting decentralized intelligence gathering helps protect democracy.

## How Funds Are Used
- Server costs for initial distribution
- Development of new analysis capabilities
- Documentation and user support
"""
    with open(file_path, 'w') as f:
        f.write(content)


def _create_readme(file_path: str):
    content = "# Night_watcher Distribution Release"
    with open(file_path, 'w') as f:
        f.write(content)


def _calculate_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sign_and_package(directory: str, output_path: str, provenance: dict) -> bool:
    try:
        manifest = build_manifest(directory, provenance.get("version", "v001"))
        with open(os.path.join(directory, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)

        with open(os.path.join(directory, "provenance.json"), 'w') as f:
            json.dump(provenance, f, indent=2)

        data = json.dumps(manifest, sort_keys=True).encode() + json.dumps(provenance, sort_keys=True).encode()
        digest = hashlib.sha256(data).hexdigest()

        private_key_path = os.environ.get("NW_PRIVATE_KEY")
        if private_key_path and os.path.exists(private_key_path):
            private_key = serialization.load_pem_private_key(open(private_key_path, 'rb').read(), password=None)
            signature_bytes = private_key.sign(
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            signature = {
                "algorithm": "RSA-PSS-SHA256",
                "signature": base64.b64encode(signature_bytes).decode(),
                "signed_data_hash": f"sha256:{digest}",
            }
            with open(os.path.join(directory, "signature.json"), 'w') as f:
                json.dump(signature, f, indent=2)

            pub_bytes = private_key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
            with open(os.path.join(directory, "public_key.pem"), 'wb') as f:
                f.write(pub_bytes)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(directory, arcname=".")

        torrent_name = os.path.splitext(os.path.basename(output_path))[0] + ".torrent"
        generate_torrent(output_path, torrent_name)

        return True

    except Exception as e:
        logger.error(f"Packaging failed: {e}")
        return False


def create_distribution_release(
    output_path: str,
    version: str,
    previous_release_path: str = None,
    bundle_files: List[str] = None,
    night_watcher_instance=None
) -> bool:
    """Create complete distribution release package."""

    try:
        from version_manager import VersionManager

        version_mgr = VersionManager()

        if version != "v001":
            if not previous_release_path or not os.path.exists(previous_release_path):
                raise ValueError("Previous release required for non-genesis versions")

            previous_version = version_mgr.get_current_version()
            if not version_mgr.validate_version_sequence(version, previous_version):
                raise ValueError(f"Invalid version sequence: {previous_version} -> {version}")

        if version == "v001":
            ok = _create_genesis_release(output_path, version, bundle_files, night_watcher_instance)
        else:
            ok = _create_incremental_release(
                output_path, version, previous_release_path, bundle_files, night_watcher_instance
            )

        if ok:
            version_mgr.record_release(version, output_path, _calculate_file_hash(output_path))

        return ok

    except Exception as e:
        logger.error(f"Failed to create distribution release: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Night_watcher distribution release")
    parser.add_argument("--output", required=True, help="Output archive path")
    parser.add_argument("--version", required=True, help="Release version like v001")
    parser.add_argument("--previous-release", help="Previous release archive for validation")
    parser.add_argument("--bundle-files", nargs="+", help="Additional files to include")
    args = parser.parse_args()

    create_distribution_release(
        output_path=args.output,
        version=args.version,
        previous_release_path=args.previous_release,
        bundle_files=args.bundle_files or [],
    )
