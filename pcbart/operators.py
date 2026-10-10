"""The operator database the 通行证 panel searches.

One row per operator, read from the index the PRTS fetcher writes
(``01-干员代号\干员代号与编号.csv``). That file is generated, not maintained by
hand, so this module's job is small on purpose: load it, answer "which
operators match these characters", and hand back the four values a card needs
-- 代号, 编号, 职业 and 阵营 (with the logo and class-icon paths already
resolved to files in the library).

**Why the CSV and not a bundled table.** The roster changes with every game
update, and the faction and artwork columns are measurements of the library
that is actually on disk. A copy compiled into the program would go stale
silently and could name artwork files that the user's library does not have --
which reaches the canvas as a blank where a logo should be, with nothing
anywhere saying why.

**The library is optional.** On a machine that has not downloaded it, the panel
has to say so and carry on rather than fail: the card template and its text
editing do not need the database at all.
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path

from .paths import badge_asset_dir

__all__ = [
    "Operator",
    "OperatorDatabase",
    "db_path",
    "load",
    "find",
    "search",
    "database",
    "reset_cache",
    "match_score",
]

INDEX_FOLDER = "01-干员代号"
INDEX_CSV = "干员代号与编号.csv"


def db_path() -> Path:
    return badge_asset_dir() / INDEX_FOLDER / INDEX_CSV


@dataclass(frozen=True)
class Operator:
    """One operator, as the card needs them.

    Field names are the index's own column names, kept in Chinese, because they
    are what the CSV says and what the panel's labels say. Renaming them here
    would mean the one place a reader compares the two has to translate.

    ``阵营logo`` and ``职业图标_*`` are **library-relative paths**, already
    checked to exist when the index was written, or empty when the library has
    no such file.
    """

    name: str
    number: str = ""
    operator_id: str = ""
    rarity: int = 0
    profession: str = ""
    branch: str = ""
    faction: str = ""
    faction_logo: str = ""
    class_icon_black: str = ""
    class_icon_white: str = ""
    page: str = ""
    prts_url: str = ""

    @property
    def display(self) -> str:
        """What the search list shows: the name, then what tells it apart.

        The number and profession are what separate an operator from their
        alternate form -- 阿米娅 appears three times with the same 编号 and
        different classes -- so a list of bare names would offer three
        identical-looking rows.
        """
        parts = [self.name]
        if self.profession:
            parts.append(self.profession)
        if self.number:
            parts.append(self.number)
        return "　".join(parts)

    def search_key(self) -> str:
        return self.name.casefold()


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class OperatorDatabase:
    """The roster, loaded once and searched by prefix then substring.

    Prefix matches rank above substring matches, which is what makes typing
    ``阿米`` put 阿米娅 first instead of after 假日威龙陈.
    """

    def __init__(self, operators: list[Operator], source: Path | None = None) -> None:
        self.operators = operators
        self.source = source
        #: Sorted once by name so a prefix search is a scan over a stable order
        #: rather than over whatever order the file happened to be written in.
        self._ordered = sorted(operators, key=lambda item: item.name.casefold())

    def __len__(self) -> int:
        return len(self.operators)

    def __bool__(self) -> bool:
        return bool(self.operators)

    def find(self, name: str) -> Operator | None:
        """The operator whose name is exactly ``name``.

        Exact, because this is the lookup the panel performs after the user has
        already chosen a row. An approximate fallback here would silently pick
        a different operator than the one clicked.

        ⚠️ **A name is not an identity.** 暮落 and 暮落(集成战略) share the
        display name, so this method returns whichever is first and cannot
        reach the other. Anything that has to survive a round trip through the
        document must store :attr:`Operator.page` and use
        :meth:`find_page`.
        """
        target = (name or "").strip().casefold()
        if not target:
            return None
        for item in self._ordered:
            if item.name.casefold() == target:
                return item
        return None

    def find_page(self, page: str) -> Operator | None:
        """The operator whose wiki page is exactly ``page``.

        The page name is the roster's unique key -- 暮落 and
        暮落(集成战略) are different people with different QR codes -- so it is
        what the panel stores on a card and what the artwork builder resolves.
        An empty page returns ``None`` so callers fall back to a name lookup
        instead of matching some unrelated row with a blank page.
        """
        target = (page or "").strip()
        if not target:
            return None
        for item in self._ordered:
            if item.page == target:
                return item
        return None

    def search(self, query: str, limit: int = 50) -> list[Operator]:
        """Operators matching ``query``, best first.

        Empty query returns the first ``limit`` rows rather than nothing, so
        opening the panel shows that the database loaded instead of an empty
        list that looks the same as a missing one.
        """
        text = (query or "").strip().casefold()
        if not text:
            return self._ordered[:limit]
        prefix: list[Operator] = []
        contains: list[Operator] = []
        for item in self._ordered:
            key = item.search_key()
            if key.startswith(text):
                prefix.append(item)
            elif text in key or text in item.number.casefold():
                contains.append(item)
            if len(prefix) >= limit:
                break
        return (prefix + contains)[:limit]


def load(path: Path | None = None) -> OperatorDatabase:
    """Read the index. A missing or unreadable file yields an empty database.

    Empty rather than raising: no library is a normal state, and every caller
    would otherwise have to wrap this in the same try. :func:`database` keeps
    the path so the panel can still say *where* it looked.
    """
    target = path or db_path()
    if not target.exists():
        return OperatorDatabase([], source=target)
    try:
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return OperatorDatabase([], source=target)

    operators: list[Operator] = []
    for row in rows:
        name = (row.get("干员名") or "").strip()
        if not name:
            continue
        operators.append(
            Operator(
                name=name,
                number=(row.get("情报编号") or "").strip(),
                operator_id=(row.get("干员id") or "").strip(),
                rarity=_to_int(row.get("稀有度") or ""),
                profession=(row.get("职业") or "").strip(),
                branch=(row.get("分支") or "").strip(),
                faction=(row.get("阵营") or "").strip(),
                faction_logo=(row.get("阵营logo") or "").strip(),
                class_icon_black=(row.get("职业图标_黑") or "").strip(),
                class_icon_white=(row.get("职业图标_白") or "").strip(),
                page=(row.get("页面名") or name).strip(),
                prts_url=(row.get("PRTS链接") or "").strip(),
            )
        )
    return OperatorDatabase(operators, source=target)


_cache: OperatorDatabase | None = None
_cache_key: tuple[str, float] | None = None
_lock = threading.Lock()

#: Library-relative paths are resolved against this when the panel needs a
#: real file. Kept as a function rather than a constant so a test that moves
#: the library between calls gets the new root.
def asset_path(relative: str) -> Path:
    return badge_asset_dir() / relative


def database() -> OperatorDatabase:
    """The shared database, reloaded when the index file changes.

    Keyed on the file's mtime and size rather than cached forever: the fetcher
    rewrites this file in place, and a cached copy would keep the panel showing
    the previous roster until the program was restarted. Re-reading a 463-row
    CSV is cheap enough that being wrong about it is the only real cost.
    """
    global _cache, _cache_key
    target = db_path()
    try:
        stat = target.stat()
        key = (str(target), stat.st_mtime)
    except OSError:
        key = (str(target), 0.0)
    with _lock:
        if _cache is not None and _cache_key == key:
            return _cache
        _cache = load(target)
        _cache_key = key
        return _cache


def reset_cache() -> None:
    """Drop the cached database. For tests and for the fetcher's own callers."""
    global _cache, _cache_key
    with _lock:
        _cache = None
        _cache_key = None


def find(name: str) -> Operator | None:
    return database().find(name)


def search(query: str, limit: int = 50) -> list[Operator]:
    return database().search(query, limit=limit)


def match_score(name: str, query: str) -> int:
    """How well ``name`` matches ``query``; lower is better, ``-1`` is no match.

    Exposed separately from :meth:`OperatorDatabase.search` so the ordering
    rule can be tested on its own, without building a database.
    """
    text = (query or "").strip().casefold()
    key = (name or "").casefold()
    if not text:
        return 0
    if key == text:
        return 0
    if key.startswith(text):
        return 1
    if text in key:
        return 2
    return -1
