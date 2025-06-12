"""Bootstrap installer for Night_Watcher distribution client."""

from __future__ import annotations

import os
import json
import tarfile
import tempfile
import hashlib
import shutil
import sqlite3


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def install_genesis(artifact_path: str, install_dir: str = "nw_client"):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(artifact_path, "r:gz") as tar:
            tar.extractall(tmpdir)
        prov_path = os.path.join(tmpdir, "provenance.json")
        if not os.path.exists(prov_path):
            raise RuntimeError("provenance.json missing")
        provenance = json.load(open(prov_path))
        if not provenance.get("genesis"):
            raise RuntimeError("Artifact is not genesis")

        client_src = os.path.join(tmpdir, "client_module")
        if not os.path.exists(client_src):
            raise RuntimeError("client_module missing")
        os.makedirs(install_dir, exist_ok=True)
        for name in os.listdir(client_src):
            shutil.copy2(os.path.join(client_src, name), os.path.join(install_dir, name))

        shutil.copy2(os.path.join(tmpdir, "public_key.pem"), os.path.join(install_dir, "public_key.pem"))

        # Initialize local data dirs
        os.makedirs(os.path.join(install_dir, "bundled_files"), exist_ok=True)
        db_path = os.path.join(install_dir, "intelligence.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, data BLOB)")
        base = os.path.join(tmpdir, "intelligence")
        for root, _, files in os.walk(base):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, base)
                with open(fpath, "rb") as fh:
                    data = fh.read()
                cur.execute("REPLACE INTO files (path, data) VALUES (?, ?)", (rel, data))
        conn.commit()
        conn.close()

        state = {
            "version": provenance["version"],
            "hash": f"sha256:{file_hash(artifact_path)}",
        }
        json.dump(state, open(os.path.join(install_dir, "current.json"), "w"))
        print(f"Installed distribution client version {provenance['version']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Install genesis Night_Watcher client")
    parser.add_argument("artifact", help="Path to genesis artifact")
    parser.add_argument("--install-dir", default="nw_client", help="Installation directory")
    args = parser.parse_args()
    install_genesis(args.artifact, args.install_dir)
