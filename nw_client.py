#!/usr/bin/env python3
"""Night_watcher Intelligence Client - Decentralized Version."""

from __future__ import annotations

import os
import json
import tarfile
import tempfile
import hashlib
import base64
from typing import Tuple, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class SignedArtifactVerifier:
    """Handles verification of signed Night_watcher artifacts."""

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

            # verify hashes
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
    """Manages version chain and validates updates."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.state_file = os.path.join(self.data_dir, "current.json")
        self.current_version = self._load_current_version()
        self.current_hash = self._load_current_hash()

    def _load_current_version(self) -> Optional[str]:
        if os.path.exists(self.state_file):
            data = json.load(open(self.state_file))
            return data.get("version")
        return None

    def _load_current_hash(self) -> Optional[str]:
        if os.path.exists(self.state_file):
            data = json.load(open(self.state_file))
            return data.get("hash")
        return None

    def _write_state(self, version: str, hash_: str):
        json.dump({"version": version, "hash": hash_}, open(self.state_file, "w"))
        self.current_version = version
        self.current_hash = hash_

    def validate_update(self, new_provenance: dict) -> bool:
        if self.current_version is None:
            return new_provenance.get("genesis", False)
        if new_provenance.get("previous_version") != self.current_version:
            return False
        if new_provenance.get("previous_hash") != self.current_hash:
            return False
        new_version = new_provenance.get("version")
        try:
            curr_num = int(self.current_version.lstrip("v"))
            new_num = int(new_version.lstrip("v"))
        except Exception:
            return False
        return new_num == curr_num + 1


class UpdateDiscovery:
    """Discovers new versions via torrent naming."""

    def check_for_updates(self, current_version: str) -> Optional[str]:
        if not current_version:
            return None
        try:
            curr_num = int(current_version.lstrip("v"))
        except Exception:
            return None
        next_ver = f"v{curr_num+1:03d}"
        fname = f"night_watcher_{next_ver}.torrent"
        if os.path.exists(fname):
            return os.path.abspath(fname)
        return None


class NightWatcherClient:
    """Main client for handling signed artifacts."""

    def __init__(self, data_dir: str = "nw_client_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.verifier = SignedArtifactVerifier()
        self.provenance = ProvenanceChain(self.data_dir)
        self.discovery = UpdateDiscovery()

    def install_artifact(self, archive_path: str) -> bool:
        ok, reason, provenance = self.verifier.verify_artifact(archive_path)
        if not ok:
            print(f"Verification failed: {reason}")
            return False
        if not self.provenance.validate_update(provenance):
            print("Provenance chain validation failed")
            return False
        # Extract
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.data_dir)
        hash_ = f"sha256:{file_hash(archive_path)}"
        self.provenance._write_state(provenance["version"], hash_)
        return True

    def check_for_updates(self) -> Optional[str]:
        return self.discovery.check_for_updates(self.provenance.current_version)

    def auto_update_check(self):
        info = self.check_for_updates()
        if info:
            print(f"Update available: {info}")


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Night_watcher Client")
    sub = parser.add_subparsers(dest="cmd")

    inst = sub.add_parser("install", help="Install artifact")
    inst.add_argument("archive")

    sub.add_parser("status", help="Show status")
    sub.add_parser("check-updates", help="Check for updates")

    args = parser.parse_args()
    client = NightWatcherClient()

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
