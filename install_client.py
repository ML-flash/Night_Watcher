#!/usr/bin/env python3
"""Installer for Night_watcher client from genesis artifact."""

from __future__ import annotations

import os
import json
import tarfile
import tempfile
import hashlib
import shutil
from typing import Optional


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def install_genesis_client(artifact_path: str, install_dir: str = "nw_client"):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(artifact_path, "r:gz") as tar:
            tar.extractall(tmpdir)
        prov_path = os.path.join(tmpdir, "provenance.json")
        if not os.path.exists(prov_path):
            raise RuntimeError("provenance.json missing")
        provenance = json.load(open(prov_path))
        if not provenance.get("genesis"):
            raise RuntimeError("Artifact is not genesis")

        # Copy client module
        client_src = os.path.join(tmpdir, "client_module")
        if not os.path.exists(client_src):
            raise RuntimeError("client_module missing")
        os.makedirs(install_dir, exist_ok=True)
        for name in os.listdir(client_src):
            src = os.path.join(client_src, name)
            dst = os.path.join(install_dir, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    continue
                os.makedirs(dst, exist_ok=True)
                for fname in os.listdir(src):
                    shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
            else:
                shutil.copy2(src, dst)

        # Save public key
        shutil.copy2(os.path.join(tmpdir, "public_key.pem"), os.path.join(install_dir, "public_key.pem"))

        # Initialize state
        state = {
            "version": provenance["version"],
            "hash": f"sha256:{file_hash(artifact_path)}",
        }
        json.dump(state, open(os.path.join(install_dir, "current.json"), "w"))
        print(f"Installed Night_watcher client version {provenance['version']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Install genesis Night_watcher client")
    parser.add_argument("artifact", help="Path to genesis artifact")
    parser.add_argument("--install-dir", default="nw_client", help="Installation directory")
    args = parser.parse_args()
    install_genesis_client(args.artifact, args.install_dir)
