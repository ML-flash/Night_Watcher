# Night_watcher Export Package Receiver Guide

This document outlines the basic steps for verifying and importing an anonymous
Night_watcher export package.

## 1. Verify Package Integrity

1. Extract the archive:
   ```bash
   tar -xzf night_watcher_v001.tar.gz -C ./package
   ```
2. Run the included verification script:
   ```bash
   cd package
   python3 verify.py
   ```
   The script checks SHA-256 hashes for all files and validates the digital
   signature using the provided `public_key.pem`.

## 2. Review Contents

- `manifest.json` – listing of files with their hashes.
- `provenance.json` – high level description of the export.
- `intelligence/` – knowledge graph, vector store and document repository.

Only proceed if verification reports **"Package verification successful"**.

## 3. Import Into Night_watcher

Use the `update_artifact.py` script from your Night_watcher installation:

```bash
python update_artifact.py /path/to/night_watcher_v001.tar.gz
```

This merges the knowledge graph, documents and vector store into your local
instance. Always verify the package before importing.

---
This is a skeleton document. Expand each step with more detail as needed.
