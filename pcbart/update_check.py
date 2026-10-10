"""In-app update check: fetch the release manifest, compare versions, download.

Pure Python, no Qt, so the whole check can be unit-tested without a window.

The manifest lives on the project's own site and the installer comes from CNB:

    https://www.xikaaa.cn/update.json     <- a few KB, always reachable in CN
    https://cnb.cool/.../EzPCBArtTool_v*.exe   <- 50 MB, does not use our uplink

The split is deliberate. CNB's **API requires a login** (401 without a token)
while its **download links are anonymous**, and the GitHub API is unreliable
from CN, so only the static site can answer "is there a newer version".

Design rule for every function here: **a failed check must look like "no
update", never like an error.** The application has to start and work with no
network at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

#: Where the manifest is served from. Overridable so tests and local runs can
#: point at a file:// URL or a temporary server without touching this module.
DEFAULT_MANIFEST_URL = "https://www.xikaaa.cn/update.json"

#: Checking must never make startup feel slow. A short timeout is the point:
#: it is better to skip one check than to hang the splash for fifteen seconds.
DEFAULT_TIMEOUT = 3.0

#: Chunk size for the downloader. 256 KiB keeps progress updates responsive
#: without turning a 50 MB download into thousands of tiny writes.
CHUNK_SIZE = 256 * 1024

#: Cap on the manifest size. It is a handful of lines; anything larger is
#: either an error page or an attack, and both should be rejected before
#: parsing.
MAX_MANIFEST_BYTES = 256 * 1024


@dataclass(frozen=True)
class ReleaseInfo:
    """One entry from ``update.json``."""

    version: str
    date: str
    notes: tuple[str, ...]
    download_url: str
    size: int
    sha256: str
    download_page: str = ""

    @property
    def filename(self) -> str:
        """The installer's file name, taken from the download URL.

        Comes from the URL rather than the manifest so there is exactly one
        place a file name is stated. A name that disagrees with the URL would
        be a bug waiting to happen -- the saved file and the link must match.
        """
        name = os.path.basename(urlparse(self.download_url).path)
        return name or f"EzPCBArtTool_v{self.version}.exe"


@dataclass(frozen=True)
class UpdateStatus:
    """The outcome of a check, including "nothing to report"."""

    #: ``None`` means the check did not produce a comparable release -- no
    #: network, a broken manifest, or a server error. Callers must treat it as
    #: "no update" rather than as a failure to report.
    release: ReleaseInfo | None
    #: True only when the remote version is strictly newer than the local one.
    available: bool
    #: Why there is no release to show. Empty when ``release`` is set.
    reason: str = ""


# --------------------------------------------------------------- versions


def parse_version(text: str) -> tuple[int, ...]:
    """Turn ``"1.10.2"`` into ``(1, 10, 2)``.

    Non-numeric suffixes are dropped (``"1.2.0-beta1"`` -> ``(1, 2, 0)``), and
    anything unparsable becomes an empty tuple so comparisons stay total
    instead of raising in the middle of a startup path.
    """
    cleaned = str(text).strip().lstrip("vV")
    if not cleaned:
        return ()
    parts: list[int] = []
    for chunk in cleaned.split("."):
        digits = ""
        for character in chunk:
            if character.isdigit():
                digits += character
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def is_newer(remote: str, local: str) -> bool:
    """Whether ``remote`` is a strictly newer version than ``local``.

    Compares numerically, segment by segment. Comparing the strings would put
    ``"1.10.0"`` *below* ``"1.9.0"``, which is the classic way an update
    checker silently stops offering updates once a version reaches ten.
    """
    remote_parts = parse_version(remote)
    local_parts = parse_version(local)
    if not remote_parts or not local_parts:
        return False
    width = max(len(remote_parts), len(local_parts))
    padded_remote = remote_parts + (0,) * (width - len(remote_parts))
    padded_local = local_parts + (0,) * (width - len(local_parts))
    return padded_remote > padded_local


# --------------------------------------------------------------- manifest


def _parse_manifest(raw: bytes) -> ReleaseInfo:
    """Validate one manifest payload. Raises ``ValueError`` when unusable."""
    try:
        data: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"update.json 不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("update.json 顶层必须是对象")

    version = str(data.get("version") or "").strip()
    download_url = str(data.get("download") or "").strip()
    if not version:
        raise ValueError("update.json 缺少 version")
    if not download_url:
        raise ValueError("update.json 缺少 download")
    if not download_url.lower().startswith(("http://", "https://")):
        raise ValueError(f"download 必须是 http(s) 地址，得到 {download_url!r}")

    notes_value = data.get("notes") or []
    if isinstance(notes_value, str):
        notes = (notes_value,)
    elif isinstance(notes_value, list):
        notes = tuple(str(item).strip() for item in notes_value if str(item).strip())
    else:
        notes = ()

    try:
        size = int(data.get("size") or 0)
    except (TypeError, ValueError):
        size = 0

    sha256 = str(data.get("sha256") or "").strip().lower()
    if sha256 and not _looks_like_sha256(sha256):
        raise ValueError(f"sha256 不是 64 位十六进制: {sha256!r}")

    return ReleaseInfo(
        version=version,
        date=str(data.get("date") or "").strip(),
        notes=notes,
        download_url=download_url,
        size=max(size, 0),
        sha256=sha256,
        download_page=str(data.get("download_page") or "").strip(),
    )


def _looks_like_sha256(text: str) -> bool:
    """Whether the text is a hex digest, in either case.

    Uppercase is accepted because digests get copied out of release pages and
    file managers that upper-case them, and rejecting that would silently
    disable verification for a manifest that is perfectly fine.
    """
    if len(text) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in text)


def fetch_latest(
    url: str = DEFAULT_MANIFEST_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Callable[..., Any] | None = None,
) -> ReleaseInfo | None:
    """Fetch and parse the manifest. **Never raises.**

    Every failure -- no network, DNS trouble, timeout, 404, an HTML error page
    where JSON should be -- comes back as ``None``. The caller cannot tell the
    difference on purpose: to the user they all mean "nothing to report".
    """
    open_url = opener or urllib.request.urlopen
    try:
        # The Request constructor validates the URL and raises ValueError for
        # anything without a scheme. That has to sit inside the guard: the
        # contract of this function is that *no* input produces an exception.
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "EzPCBArtTool-update-check",
            },
        )
        with open_url(request, timeout=timeout) as response:
            raw = response.read(MAX_MANIFEST_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if len(raw) > MAX_MANIFEST_BYTES:
        return None
    try:
        return _parse_manifest(raw)
    except ValueError:
        return None


def check_for_update(
    local_version: str,
    *,
    url: str = DEFAULT_MANIFEST_URL,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Callable[..., Any] | None = None,
) -> UpdateStatus:
    """Compare the manifest against ``local_version``.

    ``available`` is only true for a strictly newer remote version: a manifest
    left pointing at an older release (someone rolled back) must not prompt.
    """
    release = fetch_latest(url, timeout=timeout, opener=opener)
    if release is None:
        return UpdateStatus(release=None, available=False, reason="检查失败")
    if not is_newer(release.version, local_version):
        return UpdateStatus(release=release, available=False, reason="已是最新版本")
    return UpdateStatus(release=release, available=True)


# --------------------------------------------------------------- download


def sha256_of(path: Path) -> str:
    """Streamed digest, so a 50 MB installer does not sit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_download(path: Path, expected_sha256: str) -> bool:
    """Whether the downloaded file matches the manifest's digest.

    A manifest with no digest cannot be verified, so it is rejected: running
    an unverified installer is exactly the case this check exists to prevent.
    """
    if not _looks_like_sha256(expected_sha256):
        return False
    try:
        return sha256_of(path) == expected_sha256.lower()
    except OSError:
        return False


def download(
    release: ReleaseInfo,
    target: Path,
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> bool:
    """Download the installer to ``target``, resuming a partial file if there is one.

    Returns whether a file arrived. **Does not verify** -- call
    :func:`verify_download` afterwards; keeping the two steps apart makes the
    "verified" decision explicit at the call site rather than buried here.

    The range request is unconditional rather than gated on ``Accept-Ranges``:
    that header is not present in CNB's ``HEAD`` response even though
    ``Range`` requests do return ``206``, so checking the header would disable
    resuming against the one host that supports it.
    """
    open_url = opener or urllib.request.urlopen
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0

    headers = {"User-Agent": "EzPCBArtTool-update-check"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    try:
        request = urllib.request.Request(release.download_url, headers=headers)
        with open_url(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            # A server that ignores the range sends 200 and the whole body; in
            # that case the partial file on disk must be discarded, not
            # appended to, or the result is a corrupt installer that then
            # fails its digest check for a confusing reason.
            resuming = existing > 0 and status == 206
            mode = "ab" if resuming else "wb"
            if not resuming:
                existing = 0
            total = release.size or 0
            written = existing
            if on_progress is not None:
                on_progress(written, total)
            with open(target, mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if on_progress is not None:
                        on_progress(written, total)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return target.exists() and target.stat().st_size > 0


def staged_path(release: ReleaseInfo, directory: Path) -> Path:
    """Where an installer for ``release`` is downloaded.

    Under the cache directory resolved by :mod:`pcbart.paths` -- never beside
    the running program. A frozen build's ``__file__`` points into the
    unpack directory, so writing there either fails or loses the file.
    """
    return directory / release.filename


def quarantine_partial(path: Path) -> None:
    """Move aside a download that failed verification.

    Deleting it outright would hide the evidence; keeping it in place would
    let the next run try to resume onto a corrupt file.
    """
    try:
        if path.exists():
            path.replace(path.with_suffix(path.suffix + ".bad"))
    except OSError:
        pass
