#!/usr/bin/env python3
"""Export Night_Watcher artifact with graph, vector store and documents."""

import os
import json
import tarfile
import hashlib
import tempfile
import shutil
from datetime import datetime
import base64
import typing

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


def build_manifest(directory: str, version: str = "1.0") -> dict:
    manifest = {
        "version": version,
        "generated_at": datetime.now().isoformat(),
        "files": {}
    }
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            rel = os.path.relpath(path, directory)
            manifest["files"][rel] = file_hash(path)
    return manifest



def export_artifact(
    output_path: str,
    kg_dir: str = "data/knowledge_graph",
    vector_dir: str = "data/vector_store",
    documents_dir: str = "data/documents",
    *,
    private_key_path: typing.Optional[str] = None,
    version: str = "v001",
):
    """Package repository data into a signed archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = KnowledgeGraph(graph_file=os.path.join(kg_dir, "graph.json"), taxonomy_file="KG_Taxonomy.csv")
        kg.export_graph(os.path.join(tmpdir, "intelligence", "graph"))

        vs = VectorStore(base_dir=vector_dir)
        vs.export_vector_store(os.path.join(tmpdir, "intelligence", "vector_store"))

        repo = DocumentRepository(base_dir=documents_dir, dev_mode=True)

        repo.export_repository(os.path.join(tmpdir, "intelligence", "documents"))

        manifest = build_manifest(os.path.join(tmpdir, "intelligence"), version)
        with open(os.path.join(tmpdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        provenance = {
            "version": version,
            "export_time": datetime.now().isoformat(),
        }
        with open(os.path.join(tmpdir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

        if private_key_path:
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

            pub_bytes = private_key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
            with open(os.path.join(tmpdir, "public_key.pem"), "wb") as f:
                f.write(pub_bytes)
        verify_src = os.path.join(os.path.dirname(__file__), "verify.py")
        if os.path.exists(verify_src):
            shutil.copy2(verify_src, os.path.join(tmpdir, "verify.py"))

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmpdir, arcname=".")
        print(f"Exported artifact to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Night_Watcher artifact")
    parser.add_argument("--output", default="night_watcher_artifact.tar.gz", help="Output archive path")
    parser.add_argument("--kg-dir", default="data/knowledge_graph", help="Knowledge graph directory")
    parser.add_argument("--vector-dir", default="data/vector_store", help="Vector store directory")
    parser.add_argument("--documents-dir", default="data/documents", help="Document repository directory")
    parser.add_argument("--private-key", required=True, help="Private key for signing")
    args = parser.parse_args()

    export_artifact(
        args.output,
        kg_dir=args.kg_dir,
        vector_dir=args.vector_dir,
        documents_dir=args.documents_dir,
        private_key_path=args.private_key,
    )

