#!/usr/bin/env python3
"""Standalone distribution client for Night_Watcher signed artifacts."""

from __future__ import annotations

import os
import json
import tarfile
import tempfile
import hashlib
import base64
import sqlite3
import shutil
from typing import Tuple, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class SignedArtifactVerifier:
    """Verify signed artifacts using the embedded public key."""

    def verify_artifact(self, archive_path: str) -> Tuple[bool, str, dict]:
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmpdir)
            try:
                manifest = json.load(open(os.path.join(tmpdir, "manifest.json")))
                provenance = json.load(open(os.path.join(tmpdir, "provenance.json")))
                signature = json.load(open(os.path.join(tmpdir, "signature.json")))
                pub_key_bytes = open(os.path.join(tmpdir, "public_key.pem"), "rb").read()
            except Exception as e:
                return False, f"missing files: {e}", {}

            for rel, expected in manifest.get("files", {}).items():
                fpath = os.path.join(tmpdir, rel)
                if not os.path.exists(fpath):
                    return False, f"missing {rel}", {}
                h = hashlib.sha256()
                with open(fpath, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        h.update(chunk)
                if f"sha256:{h.hexdigest()}" != expected:
                    return False, f"hash mismatch for {rel}", {}

            data = json.dumps(manifest, sort_keys=True).encode() + json.dumps(provenance, sort_keys=True).encode()
            digest = hashlib.sha256(data).hexdigest()
            if signature.get("signed_data_hash") != f"sha256:{digest}":
                return False, "signed data hash mismatch", {}

            pubkey = serialization.load_pem_public_key(pub_key_bytes)
            try:
                pubkey.verify(
                    base64.b64decode(signature.get("signature")),
                    data,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                    hashes.SHA256(),
                )
            except Exception:
                return False, "signature verification failed", {}

            return True, "ok", provenance


class ProvenanceChain:
    """Maintain and validate the provenance chain."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.state_file = os.path.join(self.data_dir, "current.json")
        self.current_version = self._load_field("version")
        self.current_hash = self._load_field("hash")

    def _load_field(self, key: str) -> Optional[str]:
        if os.path.exists(self.state_file):
            data = json.load(open(self.state_file))
            return data.get(key)
        return None

    def _write_state(self, version: str, hash_: str):
        json.dump({"version": version, "hash": hash_}, open(self.state_file, "w"))
        self.current_version = version
        self.current_hash = hash_

    def validate_update(self, new_prov: dict) -> bool:
        if self.current_version is None:
            return new_prov.get("genesis", False)
        if new_prov.get("previous_version") != self.current_version:
            return False
        if new_prov.get("previous_hash") != self.current_hash:
            return False
        try:
            curr_num = int(self.current_version.lstrip("v"))
            new_num = int(new_prov.get("version", "v0").lstrip("v"))
        except Exception:
            return False
        return new_num == curr_num + 1


class UpdateDiscovery:
    """Discover new versions via torrent DHT search (fallback to local files)."""

    def find_next_torrent(self, current_version: str) -> Optional[str]:
        if not current_version:
            return None
        try:
            curr_num = int(current_version.lstrip("v"))
        except Exception:
            return None
        next_ver = f"v{curr_num+1:03d}"
        name = f"night_watcher_{next_ver}.torrent"
        if os.path.exists(name):
            return os.path.abspath(name)
        try:
            import libtorrent as lt
        except Exception:
            return None
        ses = lt.session()
        ses.listen_on(6881, 6891)
        ses.add_dht_router("router.bittorrent.com", 6881)
        ses.add_dht_router("dht.transmissionbt.com", 6881)
        ses.start_dht()
        result = None
        for _ in range(10):
            alerts = ses.pop_alerts()
            for a in alerts:
                if hasattr(a, "url") and a.url.endswith(name):
                    result = a.url
            if result:
                break
        ses.pause()
        return result


class DistributionClient:
    """Main client for installing and verifying artifacts."""

    def __init__(self, data_dir: str = "nw_client_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.verifier = SignedArtifactVerifier()
        self.provenance = ProvenanceChain(self.data_dir)
        self.discovery = UpdateDiscovery()
        self.db_path = os.path.join(self.data_dir, "intelligence.db")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, data BLOB)")
        conn.commit()
        conn.close()

    def _store_intelligence(self, directory: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        base = os.path.join(directory, "intelligence")
        for root, _, files in os.walk(base):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, base)
                with open(fpath, "rb") as fh:
                    data = fh.read()
                cur.execute("REPLACE INTO files (path, data) VALUES (?, ?)", (rel, data))
        conn.commit()
        conn.close()

    def _install_bundled(self, directory: str):
        bundle_dir = os.path.join(directory, "bundled_files")
        if not os.path.exists(bundle_dir):
            return
        dest = os.path.join(self.data_dir, "bundled_files")
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(bundle_dir):
            src = os.path.join(bundle_dir, name)
            dst = os.path.join(dest, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    def install_artifact(self, archive_path: str) -> bool:
        ok, reason, prov = self.verifier.verify_artifact(archive_path)
        if not ok:
            print(f"Verification failed: {reason}")
            return False
        if not self.provenance.validate_update(prov):
            print("Provenance chain validation failed")
            return False
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmpdir)
            self._store_intelligence(tmpdir)
            self._install_bundled(tmpdir)
        hash_ = f"sha256:{file_hash(archive_path)}"
        self.provenance._write_state(prov["version"], hash_)
        return True

    def check_for_updates(self) -> Optional[str]:
        return self.discovery.find_next_torrent(self.provenance.current_version)


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Night_Watcher Distribution Client")
    sub = parser.add_subparsers(dest="cmd")
    inst = sub.add_parser("install", help="Install artifact")
    inst.add_argument("archive")
    sub.add_parser("status", help="Show status")
    sub.add_parser("check-updates", help="Check for updates")
    args = parser.parse_args()

    client = DistributionClient()
    if args.cmd == "install":
        if client.install_artifact(args.archive):
            print("Installation successful")
    elif args.cmd == "status":
        print(json.dumps({"version": client.provenance.current_version}, indent=2))
    elif args.cmd == "check-updates":
        info = client.check_for_updates()
        print(info or "No updates found")
    else:
        parser.print_help()
