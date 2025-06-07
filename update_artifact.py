#!/usr/bin/env python3
"""Apply an exported Night_Watcher artifact."""

import os
import json
import tarfile
import hashlib
import tempfile
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


def verify_manifest(directory: str) -> bool:
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for rel, expected in manifest.get("files", {}).items():
        path = os.path.join(directory, rel)
        if not os.path.exists(path) or file_hash(path) != expected:
            return False
    return True


def verify_graph_provenance(directory: str) -> bool:
    graph_path = os.path.join(directory, "graph", "graph.json")
    prov_path = os.path.join(directory, "graph", "graph_provenance.json")
    if not (os.path.exists(graph_path) and os.path.exists(prov_path)):
        return False
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)
    return graph_data.get("metadata", {}).get("graph_id") == prov.get("graph_id")


def apply_update(archive: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmpdir)

        if not verify_manifest(tmpdir) or not verify_graph_provenance(tmpdir):
            print("Manifest or provenance validation failed")
            return

        kg = KnowledgeGraph(graph_file="data/knowledge_graph/graph.json", taxonomy_file="KG_Taxonomy.csv")
        kg.merge_graph(os.path.join(tmpdir, "graph", "graph.json"), description="artifact import")

        repo = DocumentRepository(base_dir="data/documents", dev_mode=True)
        repo.import_repository(os.path.join(tmpdir, "documents"))

        vs_dir = os.path.join(tmpdir, "vector_store")
        if os.path.exists(vs_dir):
            dest = "data/vector_store"
            os.makedirs(dest, exist_ok=True)
            if os.path.exists(os.path.join(vs_dir, "faiss_index.bin")):
                shutil.copy2(os.path.join(vs_dir, "faiss_index.bin"), os.path.join(dest, "faiss_index.bin"))
            if os.path.exists(os.path.join(vs_dir, "metadata.json")):
                shutil.copy2(os.path.join(vs_dir, "metadata.json"), os.path.join(dest, "metadata.json"))
            # Reload vector store
            VectorStore(base_dir=dest)

        print("Update applied successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply Night_Watcher artifact")
    parser.add_argument("archive", help="Path to artifact archive")
    args = parser.parse_args()

    apply_update(args.archive)
