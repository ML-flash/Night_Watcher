import sys
import os
from types import SimpleNamespace
from datetime import datetime
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collector import ContentCollector

@pytest.fixture

def collector(tmp_path):
    return ContentCollector({"content_collection": {}}, base_dir=str(tmp_path))


def test_fr_citation_date(collector):
    entry = SimpleNamespace(fr_citation_date="January 20, 2025")
    dt = collector._parse_date(entry)
    assert dt == datetime(2025, 1, 20)


def test_dublin_core_date(collector):
    entry = SimpleNamespace(**{"dc_date": "2025-02-15"})
    dt = collector._parse_date(entry)
    assert dt == datetime(2025, 2, 15)


def test_url_date_extraction(collector):
    entry = SimpleNamespace(link="https://example.gov/documents/2025/03/05/test")
    dt = collector._parse_date(entry)
    assert dt == datetime(2025, 3, 5)


def test_partial_month_date(collector):
    entry = SimpleNamespace(publication_date="2025-04")
    dt = collector._parse_date(entry)
    assert dt == datetime(2025, 4, 1)


def test_year_only_date(collector):
    entry = SimpleNamespace(publication_date="2026")
    dt = collector._parse_date(entry)
    assert dt == datetime(2026, 1, 1)
