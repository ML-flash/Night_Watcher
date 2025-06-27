import sys
import os
import json
from datetime import datetime
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gov_scrapers import (
    fetch_federal_register_api,
    fetch_white_house_actions_api,
    fetch_govinfo_bills_api,
)


def _mock_response(json_data):
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = json_data
    return mock_resp


def test_fetch_federal_register_api():
    sample = {"results": [{"title": "Test FR", "html_url": "u", "abstract": "a", "publication_date": "2025-01-01"}]}
    with patch("requests.get", return_value=_mock_response(sample)) as pg:
        docs = fetch_federal_register_api(datetime(2025,1,1), datetime(2025,1,2), limit=1)
        assert len(docs) == 1
        assert docs[0]["title"] == "Test FR"
        pg.assert_called()


def test_fetch_white_house_actions_api():
    sample = [{"title": {"rendered": "WH"}, "link": "https://www.whitehouse.gov/briefing-room/presidential-actions/test", "excerpt": {"rendered": "<p>x</p>"}, "date": "2025-01-02T00:00:00"}]
    with patch("requests.get", return_value=_mock_response(sample)) as pg:
        docs = fetch_white_house_actions_api(datetime(2025,1,1), datetime(2025,1,3), limit=1)
        assert len(docs) == 1
        assert "whitehouse.gov" in docs[0]["url"]
        pg.assert_called()


def test_fetch_govinfo_bills_api():
    sample = {"packages": [{"title": "Bill", "packageLink": "https://x", "lastModified": "2025-01-03"}]}
    with patch("requests.get", return_value=_mock_response(sample)) as pg:
        docs = fetch_govinfo_bills_api(datetime(2025,1,1), datetime(2025,1,5), api_key="KEY", limit=1)
        assert len(docs) == 1
        assert docs[0]["source"] == "Congress.gov"
        pg.assert_called()

