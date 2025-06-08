import os
import tarfile
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from export_artifact import export_artifact
from update_artifact import apply_update


def test_export_and_update_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        kg_dir = os.path.join(tmpdir, "kg")
        vec_dir = os.path.join(tmpdir, "vec")
        doc_dir = os.path.join(tmpdir, "docs")
        os.makedirs(kg_dir, exist_ok=True)
        os.makedirs(vec_dir, exist_ok=True)
        os.makedirs(doc_dir, exist_ok=True)

        archive = os.path.join(tmpdir, "bundle.tar.gz")
        export_artifact(archive, kg_dir=kg_dir, vector_dir=vec_dir, documents_dir=doc_dir)

        assert os.path.exists(archive)
        with tarfile.open(archive, "r:gz") as t:
            basenames = [os.path.basename(m) for m in t.getnames()]
        assert "manifest.json" in basenames

        apply_update(archive, kg_dir=kg_dir, vector_dir=vec_dir, documents_dir=doc_dir)
        assert os.path.exists(os.path.join(vec_dir, "metadata.json"))
