"""Client for the URL shortener used by the 通行证 QR code.

The provider's API is a small JSON service:

* ``POST /api/url/add`` with ``Authorization: Token <key>`` creates or reuses
  a short URL for a long URL;
* ``POST /api/url/get`` lists the account's links.

The token is never stored in this module.  Callers pass it in, usually from
``URLC_TOKEN`` or the per-user secrets file.  Keeping the HTTP layer here means
the batch tool and any future panel code share one implementation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "UrlcError",
    "UrlcClient",
    "default_token_path",
    "load_token",
]

DEFAULT_BASE = "https://www.urlc.cn/api"
DEFAULT_TOKEN_PATH = Path.home() / ".zcode" / "secrets" / "urlc.json"


class UrlcError(RuntimeError):
    """An error returned by the shortener API."""


def default_token_path() -> Path:
    return DEFAULT_TOKEN_PATH


def load_token(path: Path | str | None = None) -> str:
    """Read ``URLC_TOKEN`` or a small JSON secrets file."""
    env = os.environ.get("URLC_TOKEN", "").strip()
    if env:
        return env
    target = Path(path) if path is not None else default_token_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise UrlcError(
            f"短网址密钥不存在：{target}；请写入 {{\"token\": \"...\"}} 或设置 URLC_TOKEN"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise UrlcError(f"短网址密钥文件读取失败：{target}") from exc
    token = str(payload.get("token") or "").strip()
    if not token:
        raise UrlcError(f"短网址密钥文件里没有 token：{target}")
    return token


@dataclass
class UrlcClient:
    """Minimal JSON client for urlc.cn."""

    token: str
    base: str = DEFAULT_BASE
    timeout: float = 30.0

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        request = urllib.request.Request(
            self.base.rstrip("/") + "/" + path.lstrip("/"),
            data=body,
            headers={
                "Authorization": f"Token {self.token}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise UrlcError(f"短网址接口 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise UrlcError(f"短网址接口连接失败：{exc.reason}") from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UrlcError(f"短网址接口返回的不是 JSON：{raw[:200]!r}") from exc
        if int(result.get("error", 1)) != 0:
            raise UrlcError(str(result.get("msg") or "短网址接口返回错误"))
        return result

    def add(self, long_url: str) -> str:
        """Create or reuse a short URL and return it with its scheme."""
        result = self._post("/url/add", {"url": str(long_url)})
        short = str(result.get("short") or "").strip()
        if not short:
            raise UrlcError(f"短网址接口没有返回 short：{result!r}")
        return short

    def list_urls(
        self,
        *,
        limit: int = 100,
        page: int = 1,
        order: str = "date",
    ) -> dict[str, Any]:
        """One page of the account's links."""
        return self._post(
            "/url/get",
            {"limit": max(1, int(limit)), "page": max(1, int(page)), "order": order},
        )

    def iter_urls(self, *, limit: int = 100) -> Iterable[dict[str, Any]]:
        """Every link in the account, page by page."""
        page = 1
        while True:
            result = self.list_urls(limit=limit, page=page)
            data = result.get("data") or {}
            rows = data.get("urls") or []
            for row in rows:
                if isinstance(row, dict):
                    yield row
            max_page = int(data.get("maxpage") or page)
            if page >= max_page or not rows:
                break
            page += 1
