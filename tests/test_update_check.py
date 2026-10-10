"""Tests for the in-app update check.

No test touches the network: every URL opener is a stub, the same way
``tests/test_short_url.py`` does it. The live manifest and the real installer
are exercised by the end-to-end check instead, not here.

The tests are grouped the way the module is: versions, manifest, download.
The version comparison gets the most attention because it is the one piece
that fails *silently* when it is wrong -- a bad comparator stops offering
updates and nothing looks broken.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart import update_check  # noqa: E402


class _Response:
    """Minimal stand-in for the object ``urlopen`` returns.

    ``read`` advances through the body like a real stream does. A stub that
    returns the whole payload on every call makes the download loop run
    forever -- the loop stops on an empty read, so the position has to move.
    """

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self._position = 0
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._body[self._position :]
        else:
            chunk = self._body[self._position : self._position + size]
        self._position += len(chunk)
        return chunk


def _manifest(**overrides) -> bytes:
    payload = {
        "version": "1.1.0",
        "date": "2026-11-01",
        "notes": ["新增：更新检查", "修复：某处崩溃"],
        "download": "https://cnb.cool/xikaaa/Ez_PCBart/-/releases/download/v1.1.0/EzPCBArtTool_v1.1.0_安装包.exe",
        "size": 50143110,
        "sha256": "a" * 64,
        "download_page": "https://www.xikaaa.cn/tools/pcb-art/",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------- versions


def test_version_segments_compare_numerically():
    """1.10.0 is newer than 1.9.0.

    This is the whole reason :func:`is_newer` exists instead of a string
    comparison. **Injection check**: replace the body with
    ``return remote > local`` and this test goes red, while every other
    version test in this file stays green -- which is exactly why the case is
    written down separately.
    """
    assert update_check.is_newer("1.10.0", "1.9.0") is True


@pytest.mark.parametrize(
    "remote,local,expected",
    [
        ("1.1.0", "1.0.0", True),
        ("2.0.0", "1.99.99", True),
        ("1.0.10", "1.0.9", True),
        ("1.0.0", "1.0.0", False),
        ("1.0.0", "1.1.0", False),
        ("1.0.0.1", "1.0.0", True),
        # Same length, one segment lower: the padding must not turn this true.
        ("1.0", "1.0.1", False),
        ("1.0.0", "1.0", False),
    ],
)
def test_version_comparison_matrix(remote, local, expected):
    assert update_check.is_newer(remote, local) is expected


def test_version_tolerates_prefix_and_suffix():
    assert update_check.parse_version("v1.2.3") == (1, 2, 3)
    assert update_check.parse_version("1.2.0-beta1") == (1, 2, 0)
    assert update_check.is_newer("v1.2.0", "1.1.0") is True


def test_unparsable_versions_never_claim_an_update():
    """A garbage version must fail closed.

    Claiming an update from an unparsable string would nag the user with a
    dialog every launch, so the safe answer is "no".
    """
    assert update_check.is_newer("", "1.0.0") is False
    assert update_check.is_newer("abc", "1.0.0") is False
    assert update_check.is_newer("1.1.0", "") is False


# ------------------------------------------------------------- manifest


def test_fetch_parses_a_full_manifest():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["accept"] = request.headers.get("Accept")
        return _Response(_manifest())

    info = update_check.fetch_latest("https://example.test/update.json", opener=opener)
    assert info is not None
    assert info.version == "1.1.0"
    assert info.date == "2026-11-01"
    assert info.notes == ("新增：更新检查", "修复：某处崩溃")
    assert info.size == 50143110
    assert info.sha256 == "a" * 64
    assert seen["url"] == "https://example.test/update.json"
    assert seen["accept"] == "application/json"


def test_filename_comes_from_the_download_url():
    """The saved file name is derived, not read from a second field.

    Two places stating the same name is how a downloaded file ends up not
    matching the link it came from.
    """
    info = update_check.fetch_latest("https://example.test/update.json", opener=lambda *a, **k: _Response(_manifest()))
    assert info is not None
    assert info.filename == "EzPCBArtTool_v1.1.0_安装包.exe"


@pytest.mark.parametrize(
    "payload,label",
    [
        (b"not json at all", "JSON 损坏"),
        (b"[]", "顶层不是对象"),
        (json.dumps({"version": "1.1.0"}).encode(), "缺 download"),
        (json.dumps({"download": "https://example.test/a.exe"}).encode(), "缺 version"),
        (
            json.dumps({"version": "1.1.0", "download": "ftp://example.test/a.exe"}).encode(),
            "download 不是 http(s)",
        ),
        (
            json.dumps({"version": "1.1.0", "download": "https://e.test/a.exe", "sha256": "xyz"}).encode(),
            "sha256 长度不对",
        ),
    ],
)
def test_bad_manifests_return_none(payload, label):
    """Every malformed manifest is "no update", not an exception."""
    info = update_check.fetch_latest("https://example.test/update.json", opener=lambda *a, **k: _Response(payload))
    assert info is None, label


def test_network_failure_returns_none():
    def opener(request, timeout):
        raise OSError("network is down")

    assert update_check.fetch_latest("https://example.test/update.json", opener=opener) is None


def test_oversized_manifest_is_rejected():
    """An HTML error page or a hostile payload must not be parsed as JSON."""
    big = b"x" * (update_check.MAX_MANIFEST_BYTES + 10)
    assert update_check.fetch_latest("https://example.test/update.json", opener=lambda *a, **k: _Response(big)) is None


def test_check_reports_an_update():
    status = update_check.check_for_update(
        "1.0.0", opener=lambda *a, **k: _Response(_manifest())
    )
    assert status.available is True
    assert status.release is not None
    assert status.release.version == "1.1.0"


def test_check_stays_quiet_when_already_current():
    status = update_check.check_for_update(
        "1.1.0", opener=lambda *a, **k: _Response(_manifest())
    )
    assert status.available is False
    assert status.reason == "已是最新版本"


def test_check_stays_quiet_on_a_rolled_back_manifest():
    """A manifest pointing at an older release must not prompt.

    Servers get rolled back and files get swapped; treating "older" as
    "different" would nag the user to downgrade.
    """
    status = update_check.check_for_update(
        "2.0.0", opener=lambda *a, **k: _Response(_manifest(version="1.1.0"))
    )
    assert status.available is False


def test_check_failure_is_indistinguishable_from_no_update():
    """No network means no prompt, and no error the caller has to handle."""

    def opener(request, timeout):
        raise OSError("offline")

    status = update_check.check_for_update("1.0.0", opener=opener)
    assert status.release is None
    assert status.available is False


# ------------------------------------------------------------- download


def _release(tmp_path: Path, body: bytes, *, with_digest: bool = True):
    digest = hashlib.sha256(body).hexdigest() if with_digest else ""
    return update_check.ReleaseInfo(
        version="1.1.0",
        date="2026-11-01",
        notes=(),
        download_url="https://example.test/EzPCBArtTool_v1.1.0_安装包.exe",
        size=len(body),
        sha256=digest,
    )


def test_download_writes_the_file(tmp_path):
    body = b"installer bytes" * 1000
    release = _release(tmp_path, body)
    target = tmp_path / "out" / release.filename
    progress: list[tuple[int, int]] = []

    ok = update_check.download(
        release,
        target,
        opener=lambda *a, **k: _Response(body),
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert ok is True
    assert target.read_bytes() == body
    assert progress[-1] == (len(body), len(body))


def test_download_resumes_from_a_partial_file(tmp_path):
    """A half-finished download continues instead of starting over.

    The stub answers 206 like CNB does, and asserts the range header asked
    for exactly the bytes already on disk.
    """
    body = b"A" * 500 + b"B" * 500
    release = _release(tmp_path, body)
    target = tmp_path / release.filename
    target.write_bytes(body[:500])

    seen = {}

    def opener(request, timeout):
        seen["range"] = request.headers.get("Range")
        return _Response(body[500:], status=206)

    ok = update_check.download(release, target, opener=opener)
    assert ok is True
    assert seen["range"] == "bytes=500-"
    assert target.read_bytes() == body


def test_download_discards_partial_file_when_server_ignores_the_range(tmp_path):
    """A 200 answer means the body is the whole file, so the stub must go.

    Appending a full body to a partial file produces a corrupt installer that
    then fails its digest check -- a confusing symptom for a simple cause.
    """
    body = b"COMPLETE" * 100
    release = _release(tmp_path, body)
    target = tmp_path / release.filename
    target.write_bytes(b"stale partial data")

    ok = update_check.download(
        release, target, opener=lambda *a, **k: _Response(body, status=200)
    )
    assert ok is True
    assert target.read_bytes() == body


def test_download_failure_returns_false(tmp_path):
    release = _release(tmp_path, b"x")

    def opener(request, timeout):
        raise OSError("connection reset")

    target = tmp_path / release.filename
    assert update_check.download(release, target, opener=opener) is False


# ---------------------------------------------------------- verification


def test_verify_accepts_a_matching_digest(tmp_path):
    body = b"payload"
    path = tmp_path / "file.bin"
    path.write_bytes(body)
    assert update_check.verify_download(path, hashlib.sha256(body).hexdigest()) is True


def test_verify_rejects_a_mismatched_digest(tmp_path):
    """**Injection check**: drop the digest comparison and this goes red.

    This is the guard that stops a truncated or tampered installer from being
    run, so it has to be proven to actually compare.
    """
    path = tmp_path / "file.bin"
    path.write_bytes(b"payload")
    assert update_check.verify_download(path, "b" * 64) is False


def test_verify_rejects_a_missing_digest(tmp_path):
    """No digest in the manifest means no verification is possible, so no."""
    path = tmp_path / "file.bin"
    path.write_bytes(b"payload")
    assert update_check.verify_download(path, "") is False


def test_verify_is_case_insensitive(tmp_path):
    body = b"payload"
    path = tmp_path / "file.bin"
    path.write_bytes(body)
    upper = hashlib.sha256(body).hexdigest().upper()
    assert update_check.verify_download(path, upper) is True


def test_quarantine_moves_the_bad_file_aside(tmp_path):
    path = tmp_path / "installer.exe"
    path.write_bytes(b"corrupt")
    update_check.quarantine_partial(path)
    assert not path.exists()
    assert path.with_suffix(".exe.bad").exists()


def test_quarantine_is_safe_when_there_is_nothing_to_move(tmp_path):
    update_check.quarantine_partial(tmp_path / "missing.exe")
