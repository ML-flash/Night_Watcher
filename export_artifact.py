#!/usr/bin/env python3
"""Export Night_Watcher artifact with graph, vector store and documents."""

import os
import json
import tarfile
import hashlib
import tempfile
from datetime import datetime
import shutil

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


def export_artifact(output_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = KnowledgeGraph(graph_file="data/knowledge_graph/graph.json", taxonomy_file="KG_Taxonomy.csv")
        kg.export_graph(os.path.join(tmpdir, "graph"))

        vs = VectorStore(base_dir="data/vector_store")
        vs.export_vector_store(os.path.join(tmpdir, "vector_store"))

        repo = DocumentRepository(base_dir="data/documents", dev_mode=True)
        repo.export_repository(os.path.join(tmpdir, "documents"))

        manifest = build_manifest(tmpdir)
        with open(os.path.join(tmpdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmpdir, arcname=".")
        print(f"Exported artifact to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Night_Watcher artifact")
    parser.add_argument("--output", default="night_watcher_artifact.tar.gz", help="Output archive path")
    args = parser.parse_args()

    export_artifact(args.output)
