"""Tests for the urlc.cn short-link client.

The live API is not called here: its result depends on the user's account and
network.  These tests pin the request shape (header, JSON body, pagination) and
the response/error handling that the batch tool depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import short_url  # noqa: E402


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def test_add_sends_bearer_token_and_parses_short(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"error": 0, "short": "http://i8c.cn/RkhQm"})

    monkeypatch.setattr(short_url.urllib.request, "urlopen", fake_urlopen)
    client = short_url.UrlcClient("secret-token")
    assert client.add("https://prts.wiki/w/阿米娅") == "http://i8c.cn/RkhQm"
    assert seen["url"].endswith("/api/url/add")
    assert seen["auth"] == "Token secret-token"
    assert seen["body"] == {"url": "https://prts.wiki/w/阿米娅"}


def test_api_error_is_raised(monkeypatch):
    def fake_urlopen(request, timeout):
        return _Response({"error": 1, "msg": "ApiKey错误"})

    monkeypatch.setattr(short_url.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(short_url.UrlcError):
        short_url.UrlcClient("bad").add("https://example.com")


def test_iter_urls_follows_maxpage(monkeypatch):
    pages = {
        1: {"error": 0, "data": {"maxpage": 2, "urls": [{"id": 1}]}},
        2: {"error": 0, "data": {"maxpage": 2, "urls": [{"id": 2}]}},
    }

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        return _Response(pages[body["page"]])

    monkeypatch.setattr(short_url.urllib.request, "urlopen", fake_urlopen)
    rows = list(short_url.UrlcClient("token").iter_urls(limit=1))
    assert [row["id"] for row in rows] == [1, 2]


def test_load_token_accepts_bom_and_env(monkeypatch, tmp_path):
    path = tmp_path / "urlc.json"
    path.write_text("\ufeff" + json.dumps({"token": "from-file"}), encoding="utf-8")
    monkeypatch.delenv("URLC_TOKEN", raising=False)
    assert short_url.load_token(path) == "from-file"
    monkeypatch.setenv("URLC_TOKEN", "from-env")
    assert short_url.load_token(path) == "from-env"
