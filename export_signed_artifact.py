#!/usr/bin/env python3
"""Export signed, versioned Night_watcher artifacts."""

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
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from document_repository import DocumentRepository


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(directory: str) -> dict:
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "files": {},
    }
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            rel = os.path.relpath(path, directory)
            manifest["files"][rel] = f"sha256:{file_hash(path)}"
    return manifest


def _bencode(value) -> bytes:
    """Very small bencode implementation for torrent generation."""
    if isinstance(value, int):
        return f"i{value}e".encode()
    if isinstance(value, bytes):
        return str(len(value)).encode() + b":" + value
    if isinstance(value, str):
        b = value.encode()
        return str(len(b)).encode() + b":" + b
    if isinstance(value, list):
        return b"l" + b"".join(_bencode(x) for x in value) + b"e"
    if isinstance(value, dict):
        items = sorted(value.items())
        out = b"d"
        for k, v in items:
            out += _bencode(str(k)) + _bencode(v)
        out += b"e"
        return out
    raise TypeError(f"Unsupported type: {type(value)}")


def _generate_torrent(file_path: str, torrent_path: str):
    piece_length = 262144  # 256 KiB
    pieces = []
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(piece_length)
            if not chunk:
                break
            pieces.append(hashlib.sha1(chunk).digest())
    info = {
        "name": os.path.basename(file_path),
        "piece length": piece_length,
        "length": os.path.getsize(file_path),
        "pieces": b"".join(pieces),
    }
    torrent = {
        "announce": "https://example.com/announce",
        "creation date": int(time.time()),
        "created by": "NightWatcher",
        "info": info,
    }
    with open(torrent_path, "wb") as f:
        f.write(_bencode(torrent))


def export_signed_artifact(
    output_path: str,
    version: str,
    private_key_path: str,
    kg_dir: str = "data/knowledge_graph",
    vector_dir: str = "data/vector_store",
    documents_dir: str = "data/documents",
    previous_artifact_path: Optional[str] = None,
    bundled_files: Optional[List[str]] = None,
):
    """Export signed, versioned artifact with provenance chain."""
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
        manifest = build_manifest(tmpdir)
        with open(os.path.join(tmpdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Build provenance
        prev_hash = file_hash(previous_artifact_path) if previous_artifact_path else None
        prev_version = None
        if previous_artifact_path:
            # assume filename includes version like night_watcher_v001.tar.gz
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

        # Sign
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

        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        with open(os.path.join(tmpdir, "public_key.pem"), "wb") as f:
            f.write(pub_bytes)

        # Package
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmpdir, arcname=".")

    # Generate torrent
    torrent_name = f"night_watcher_{version}.torrent"
    _generate_torrent(output_path, torrent_name)
    print(f"Exported signed artifact to {output_path} and {torrent_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export signed Night_watcher artifact")
    parser.add_argument("output", help="Output archive path")
    parser.add_argument("--version", required=True, help="Version string")
    parser.add_argument("--private-key", required=True, help="Private key path")
    parser.add_argument("--previous-artifact", help="Previous artifact")
    parser.add_argument("--bundle-files", nargs="+", help="Extra files to bundle")
    args = parser.parse_args()

    export_signed_artifact(
        args.output,
        version=args.version,
        private_key_path=args.private_key,
        previous_artifact_path=args.previous_artifact,
        bundled_files=args.bundle_files,
    )
