"""Simple torrent file generator for Night_Watcher artifacts."""
from __future__ import annotations

import os
import hashlib
import time
from typing import Dict


def _bencode(value) -> bytes:
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


def generate_torrent(file_path: str, torrent_path: str) -> None:
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
    torrent: Dict[str, object] = {
        "announce": "https://example.com/announce",
        "creation date": int(time.time()),
        "created by": "NightWatcher",
        "info": info,
    }
    with open(torrent_path, "wb") as f:
        f.write(_bencode(torrent))
