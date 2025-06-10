#!/usr/bin/env python3
"""Anonymous verification tool for Night_watcher export packages."""
import json
import hashlib
import base64
import sys
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def verify_package(path: Path) -> bool:
    base = Path(path)
    manifest_path = base / "manifest.json"
    signature_path = base / "signature.json"
    provenance_path = base / "provenance.json"
    public_key_path = base / "public_key.pem"

    if not (manifest_path.exists() and signature_path.exists() and provenance_path.exists() and public_key_path.exists()):
        print("Package missing required files")
        return False

    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    signature = json.loads(signature_path.read_text())

    # Verify file hashes
    for rel, expected in manifest.get("files", {}).items():
        fpath = base / rel
        if not fpath.exists():
            print(f"Missing file: {rel}")
            return False
        h = hashlib.sha256()
        with open(fpath, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        if h.hexdigest() != expected.split(":")[-1]:
            print(f"Hash mismatch for {rel}")
            return False

    # Verify signature
    data = json.dumps(manifest, sort_keys=True).encode() + json.dumps(provenance, sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    if "sha256:" + digest != signature.get("signed_data_hash"):
        print("Signed data hash mismatch")
        return False

    public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
    try:
        public_key.verify(
            base64.b64decode(signature.get("signature")),
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
    except Exception:
        print("Digital signature invalid")
        return False

    print("Package verification successful")
    return True

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    verify_package(target)
