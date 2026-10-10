"""Download Arknights artwork from PRTS wiki into a local asset library.

The template work needs a stable, browsable copy of the game's artwork rather
than live wiki URLs, so this tool mirrors the categories the template will be
built from. Two facts about PRTS shape the whole design:

1. The file index page (``PRTS:文件一览``) does not render its lists in the
   wikitext -- every section is a ``<dpl>`` call. The regexes inside those calls
   *are* the naming rules, and they are what :data:`GROUPS` copies. Reading them
   from the page instead would mean scraping rendered HTML, and the site answers
   plain HTTP clients with 403.
2. ``list=allimages`` used to be the obvious enumerator, but for ``ainamespace``
   to be honoured it must be the only parameter of that kind, and PRTS rejects
   the request otherwise. Everything here enumerates by title prefix with
   ``aiprefix`` and filters with the regex locally. Titles are exact, so no file
   can be missed by a prefix scan.

Downloads are resumable: an existing file of non-zero size is left alone.
Standard library only, so it runs from the packaged interpreter without extras.

Usage::

    python tools/fetch_prts_assets.py --list           # counts, no downloads
    python tools/fetch_prts_assets.py                  # everything
    python tools/fetch_prts_assets.py --only 03-立绘-精英0 --only 15-团队logo
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(ROOT))

from pcbart.paths import asset_root  # noqa: E402

API = "https://prts.wiki/api.php"
USER_AGENT = (
    "PCBArtTool-prts-fetch/1.0 "
    "(local asset library for a personal PCB art tool; contact: project owner)"
)

#: Where the library lands by default. Deliberately *outside* the source tree:
#: the art is several gigabytes and must never be swept into a git commit.
DEFAULT_OUT = asset_root() / "明日方舟"

OPERATOR_CATEGORY = "分类:干员"
ISSUE_NUMBER_PAGE = "干员编号一览"
FILE_INDEX_PAGE = "PRTS:文件一览"
FACTION_PAGE = "干员一览"


@dataclass(frozen=True)
class Group:
    """One comma-separated asset class from the request, saved to one folder."""

    key: str
    folder: str
    prefixes: tuple[str, ...]
    pattern: re.Pattern[str]
    note: str = ""


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


#: Folder order mirrors the order the request listed the classes in.
#:
#: ⚠️ **Never write ``[^_]*`` for a name segment.** ``list=allimages`` returns
#: the database form of a title, where every space is an underscore, so a
#: segment that forbids underscores forbids *spaces* -- and multi-word names
#: are exactly where spaces occur: ``Logo Ave Mujica.png`` arrives as
#: ``Logo_Ave_Mujica.png`` and was silently dropped by a rule that looked right
#: for all 48 single-word factions. Use ``.+`` and let the **anchored suffix**
#: keep the classes apart: ``_1.png`` cannot end ``skin1.png``, and
#: ``_大图.png`` cannot end ``_大图_白.png``. Where the trailing segment really
#: is another class, exclude it with a lookahead instead of banning underscores.
GROUPS: tuple[Group, ...] = (
    Group(
        "half_body_e0",
        "02-半身像-精英0",
        ("半身像",),
        _rx(r"^半身像_(.+)_1\.png$"),
    ),
    Group(
        "half_body_e1",
        "03-半身像-精英1",
        ("半身像",),
        _rx(r"^半身像_(.+)_精英1\.png$"),
        note="只有斑点；维基把这张传成了皮肤名，见索引的 curated",
    ),
    Group(
        "half_body_e2",
        "04-半身像-精英2",
        ("半身像",),
        _rx(r"^半身像_(.+)_2\.png$"),
    ),
    Group(
        "half_body_skin",
        "05-半身像-皮肤",
        ("半身像",),
        _rx(r"^半身像_(.+)_skin[0-9]\.png$"),
        note="文件名是时装名（见下方「按时装名改名」）",
    ),
    Group(
        "full_e0",
        "06-立绘-精英0",
        ("立绘",),
        _rx(r"^立绘_(.+)_1\.png$"),
    ),
    Group(
        "full_e1",
        "07-立绘-精英1",
        ("立绘",),
        _rx(r"^立绘_(.+)_1\+\.png$"),
        note="阿米娅的 1+ 与斑点的晋升立绘；已按半身像的写法改名「精英1」",
    ),
    Group(
        "full_e2",
        "08-立绘-精英2",
        ("立绘",),
        _rx(r"^立绘_(.+)_2\.png$"),
        note="通常 2048×2048",
    ),
    Group(
        "full_skin",
        "09-立绘-皮肤",
        ("立绘",),
        _rx(r"^立绘_(.+)_skin[0-9]\.png$"),
        note="文件名是时装名（见下方「按时装名改名」）",
    ),
    Group(
        "summon_avatar",
        "10-召唤物头像",
        ("头像",),
        _rx(r"^头像_召唤物_.*\.png$"),
    ),
    Group(
        "skill_icon",
        "11-技能图标",
        ("技能",),
        _rx(r"^技能_.*\.png$"),
        note="后勤技能是 Bskill_ 前缀，不会被收进来",
    ),
    Group(
        "token_icon",
        "12-信物无框图标",
        ("道具",),
        _rx(r"^道具_(?!带框_).*信物\.png$"),
    ),
    Group(
        "elite_progress",
        "13-精英化进度图标",
        ("精英",),
        _rx(r"^精英.*\.png$"),
    ),
    Group(
        "rarity_black",
        "14-稀有度星级/黑",
        ("稀有度",),
        _rx(r"^稀有度_黑_[0-5]\.png$"),
    ),
    Group(
        "rarity_white",
        "14-稀有度星级/白",
        ("稀有度",),
        _rx(r"^稀有度_白_[0-5]\.png$"),
    ),
    Group(
        "class_big_black",
        "15-职业大图/黑",
        ("图标 职业",),
        _rx(r"^图标_职业_(.+)_大图\.png$"),
        note="文件名是「大图」，实际背景是黑色",
    ),
    Group(
        "class_big_white",
        "15-职业大图/白",
        ("图标 职业",),
        _rx(r"^图标_职业_(.+)_大图_白\.png$"),
    ),
    Group(
        "team_logo",
        "16-团队logo",
        ("团队logo",),
        _rx(r"^团队logo_(.+)\.png$"),
    ),
    Group(
        "skin_logo",
        "17-皮肤logo（大）",
        ("Skin logo",),
        _rx(r"^Skin_logo_(?!.*_小\.png$).+\.png$"),
        note="小尺寸那套是「… 小.png」，这里不收",
    ),
    Group(
        "faction_logo",
        "18-阵营logo",
        ("Logo",),
        _rx(r"^Logo_(.+)\.png$"),
    ),
    Group(
        "medal",
        "19-蚀刻章",
        ("蚀刻章",),
        _rx(r"^蚀刻章_.*\.png$"),
        note="套组预览图是「蚀刻章套组_」前缀，这里不收",
    ),
)


# --------------------------------------------------------------------------
# wiki access


def api(params: dict[str, str], *, tries: int = 5, timeout: int = 120) -> dict:
    """One API call, retried on transient failures.

    PRTS answers with a 504 often enough that a bare call fails a long run, so
    exponential backoff is not optional here.
    """
    query = dict(params)
    query.setdefault("format", "json")
    url = f"{API}?{urllib.parse.urlencode(query)}"
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable
            last = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"PRTS API failed after {tries} tries: {url}") from last


def page_wikitext(title: str) -> str:
    payload = api({"action": "parse", "page": title, "prop": "wikitext"})
    if "error" in payload:
        raise RuntimeError(f"cannot read {title}: {payload['error']['info']}")
    return payload["parse"]["wikitext"]["*"]


def prefix_titles(prefix: str) -> list[str]:
    """Every file title starting with ``prefix``, extension included."""
    titles: list[str] = []
    cont: str | None = None
    while True:
        params = {
            "action": "query",
            "list": "allimages",
            "aiprefix": prefix,
            "ailimit": "500",
            "aiprop": "size",
            "ainamespace": "6",
        }
        if cont:
            params["aicontinue"] = cont
        payload = api(params)
        titles.extend(item["name"] for item in payload["query"]["allimages"])
        if "continue" not in payload:
            return titles
        cont = payload["continue"]["aicontinue"]


def display_name(title: str) -> str:
    """The name a PRTS page shows, which is the one worth writing to disk.

    MediaWiki treats ``_`` and a space as the same character in a title, and
    which one comes back depends on the endpoint: ``list=allimages`` returns the
    raw database form (underscores) while ``prop=imageinfo`` returns the
    canonical form (spaces). Normalising on the display form keeps the two
    lookups matching *and* gives the user filenames that read like the wiki.
    """
    return title.replace("_", " ").strip()


def resolve_urls(titles: Sequence[str]) -> dict[str, dict]:
    """Title -> {url, size} for up to 50 titles at a time.

    The API answers with the *localised* namespace (``文件:``) even though the
    request uses ``File:``, so the prefix is split off rather than sliced by
    length -- slicing five characters would leave a mangled title and every
    lookup would silently miss.
    """
    resolved: dict[str, dict] = {}
    for start in range(0, len(titles), 50):
        chunk = titles[start : start + 50]
        payload = api(
            {
                "action": "query",
                "titles": "|".join(f"File:{name}" for name in chunk),
                "prop": "imageinfo",
                "iiprop": "url|size",
                "formatversion": "2",
            }
        )
        for page in payload.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            title = page.get("title", "")
            name = title.split(":", 1)[1] if ":" in title else title
            if name and info.get("url"):
                resolved[display_name(name)] = {
                    "url": info["url"],
                    "size": info.get("size", 0),
                }
    return resolved


# --------------------------------------------------------------------------
# operator index (代号 + 编号)


def operator_pages() -> list[str]:
    names: list[str] = []
    cont: str | None = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": OPERATOR_CATEGORY,
            "cmlimit": "500",
            "cmnamespace": "0",
        }
        if cont:
            params["cmcontinue"] = cont
        payload = api(params)
        names.extend(member["title"] for member in payload["query"]["categorymembers"])
        if "continue" not in payload:
            break
        cont = payload["continue"]["cmcontinue"]
    return sorted(set(names))


_FIELD = re.compile(
    r"\|(干员名|干员id|干员序号|稀有度|职业|分支|情报编号"
    r"|所属国家|所属组织|所属团队)=([^\n|]*)"
)
"""The infobox fields this tool keeps.

The three ``所属*`` fields are the wiki's names for the game's nation/group/
team slots. They cost nothing extra -- they ride along in the same 50-pages-
per-request batch read that already fetches 职业 and 情报编号.
"""


def operator_records(pages: Sequence[str]) -> list[dict[str, str]]:
    """Read the infobox of each operator page, 50 pages per request."""
    records: list[dict[str, str]] = []
    for start in range(0, len(pages), 50):
        chunk = pages[start : start + 50]
        payload = api(
            {
                "action": "query",
                "prop": "revisions",
                "rvslots": "main",
                "rvprop": "content",
                "titles": "|".join(chunk),
                "formatversion": "2",
            }
        )
        for page in payload.get("query", {}).get("pages", []):
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            text = revisions[0]["slots"]["main"]["content"]
            fields = dict(_FIELD.findall(text))
            fields["页面名"] = page["title"]
            records.append(fields)
        print(f"  算子档案 {min(start + 50, len(pages))}/{len(pages)}", flush=True)
    records.sort(key=lambda item: int(item.get("干员序号") or 9999))
    return records


_TABLE_ROW = re.compile(
    r"\|\s*([A-Z]{1,3}\d{1,4}[A-Z]?)\s*\n\|\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]"
)
"""One ``|CODE`` / ``|[[名字]]`` row of ``干员编号一览``.

The digit run is one-to-four, not two-to-four: the robot operators carry a
single digit (``RCX2``) and so does the 乌萨斯学生自治团 block (``USS0``).
Requiring two silently dropped 13 rows -- every robot and the whole student
group -- while the page still looked parsed.
"""


def number_table_rows() -> list[tuple[str, str]]:
    """``干员编号一览`` 的 (编号, 代号) 表，用于和档案字段互相校验。"""
    text = page_wikitext(ISSUE_NUMBER_PAGE)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for code, name in _TABLE_ROW.findall(text):
        if code in seen:
            continue
        seen.add(code)
        rows.append((code, name.strip()))
    return rows


# --------------------------------------------------------------------------
# 阵营（所属势力）
#
# The badge template draws one faction logo in a fixed area, so the database has
# to answer "which logo image belongs to this operator". Two sources answer it
# and they are not interchangeable:
#
# * ``干员一览`` renders one card per **playable** operator into a hidden
#   ``#filter-data`` block. ``data-logo`` is the value the wiki itself turns
#   into the logo image, so it is the only one guaranteed to name a file.
# * The operator's own infobox carries 所属国家 / 所属组织 / 所属团队, the
#   wiki's names for the game's nation/group/team slots. Those arrive free with
#   the batch read above and cover the operators the grid leaves out -- NPCs,
#   and the elite operators that only appear in game modes.
#
# The grid wins where both exist: it is the copy the logo value is consistent
# with. The infobox is only consulted for rows the grid does not have, and only
# when its value actually matches a logo in the library.


class _FilterDataParser(HTMLParser):
    """Collect the operator cards from the grid's hidden ``#filter-data`` block.

    Parsed with :mod:`html.parser` rather than a regex over the block: each card
    is a ``<div>`` whose *text content* contains nested spans and literal ``>``
    characters, and the card after it starts with the same ``<div data-zh=``
    shape. A regex gets this right only as long as no description ever contains
    a quote, which is not a property worth relying on.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        #: Nesting depth inside the container, or None while outside it. Zero
        #: means the container itself is open but no child has been seen.
        self._depth: int | None = None
        self.cards: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if self._depth is None:
            if attributes.get("id") == "filter-data":
                self._depth = 0
            return
        if tag != "div":
            return
        self._depth += 1
        if attributes.get("data-zh"):
            self.cards.append(attributes)

    def handle_endtag(self, tag: str) -> None:
        if self._depth is None or tag != "div":
            return
        self._depth -= 1
        if self._depth < 0:
            self._depth = None


def parse_filter_data(html: str) -> list[dict[str, str]]:
    """Every operator card in ``#filter-data``, as its raw attributes.

    An empty list means the block was not found -- a page answered by something
    other than the wiki, or a renamed container. The caller treats that as an
    error rather than as "no operator has a faction", because an empty column
    is indistinguishable from a wiki that dropped the field, and the panel
    would simply draw no logo.
    """
    parser = _FilterDataParser()
    parser.feed(html)
    return [card for card in parser.cards if card.get("data-zh")]


def faction_page_html() -> str:
    """Rendered HTML of ``干员一览``: API first, page view as the fallback.

    ⚠️ The fallback is not decoration. On 2026-09-26 the API answered 503 for
    hours while ordinary page views were still served from the cache, so
    without it a refresh of the index would have to wait for the API to
    recover -- and the index is the one thing that has to be refreshable
    without touching the several gigabytes of artwork.
    """
    try:
        payload = api({"action": "parse", "page": FACTION_PAGE, "prop": "text"}, tries=2)
        if "error" in payload:
            raise RuntimeError(payload["error"]["info"])
        return payload["parse"]["text"]["*"]
    except Exception as exc:  # noqa: BLE001 - any failure is worth the fallback
        print(f"   API 读不到 {FACTION_PAGE}（{exc}），改用页面 HTML", flush=True)
    url = f"https://prts.wiki/w/{urllib.parse.quote(FACTION_PAGE)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Referer": "https://prts.wiki/"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read().decode("utf-8", "replace")


def faction_records() -> dict[str, dict[str, str]]:
    """页面名 -> {logo, nation, group, team, profession, id} from the grid.

    Keyed by the name the wiki shows, which is what the operator index's
    ``页面名`` and ``干员名`` hold. Alternate forms are separate cards with
    separate professions (阿米娅/近卫/医疗), so the lookup must stay per page
    rather than collapsing onto the shared 情报编号.
    """
    cards = parse_filter_data(faction_page_html())
    if not cards:
        raise RuntimeError(
            f"{FACTION_PAGE} 里找不到 #filter-data：页面结构变了，阵营没抓到。"
            "（不要把它当成「这些干员没有阵营」）"
        )
    found: dict[str, dict[str, str]] = {}
    for card in cards:
        found[card["data-zh"].strip()] = {
            "logo": card.get("data-logo", "").strip(),
            "nation": card.get("data-nation", "").strip(),
            "group": card.get("data-group", "").strip(),
            "team": card.get("data-team", "").strip(),
            "profession": card.get("data-profession", "").strip(),
            "id": card.get("data-id", "").strip(),
        }
    print(f"  {FACTION_PAGE} -> {len(found)} 张干员卡", flush=True)
    return found


FACTION_LOGO_FOLDER = "18-阵营logo"

#: (column suffix, folder, file-name pattern) for the two class-icon tones.
#: Both are kept: the template area is dark and the *white* variant is the one
#: that reads on it, but which one the user wants is a design decision the
#: panel has to make -- the index only reports what exists.
CLASS_ICON_FOLDERS: tuple[tuple[str, str, str], ...] = (
    ("黑", "15-职业大图/黑", "图标 职业 {profession} 大图.png"),
    ("白", "15-职业大图/白", "图标 职业 {profession} 大图 白.png"),
)


def library_file(root: Path, folder: str, name: str) -> str:
    """``folder/name`` when that file is really in the library, else "".

    Never returns a plausible-looking path for a file that is not there: a
    wrong path moves the failure into the panel, where it looks like a drawing
    bug instead of a gap in the data.
    """
    return f"{folder}/{name}" if (root / folder / name).exists() else ""


def library_logo_names(root: Path) -> set[str]:
    """Faction names that have a ``18-阵营logo`` image in this library."""
    folder = root / FACTION_LOGO_FOLDER
    if not folder.exists():
        return set()
    return {
        item.name[len("Logo ") : -len(".png")]
        for item in folder.iterdir()
        if item.name.startswith("Logo ") and item.suffix == ".png"
    }


def apply_factions(
    rows: list[dict[str, str]], root: Path, factions: dict[str, dict[str, str]]
) -> dict:
    """Fill 阵营 / 所属* / 素材路径 on every row and report what did not fit.

    The paths are relative to the library root and are identical in the full
    mirror and in the badge subset -- both carry ``15-职业大图`` and
    ``18-阵营logo`` under the same names -- which is why they can be written
    into the index that the subset copies.
    """
    available = library_logo_names(root)
    report: dict = {
        "from_grid": 0,
        "from_infobox": 0,
        "missing": [],
        #: Faction-less rows that have no in-game index either. These are the
        #: event-mode rosters (卫戍协议) and NPCs such as 博士/文月/魏彦吾, which
        #: the grid legitimately does not list. Kept apart from :data:`missing`
        #: because a count that mixes "expected" with "the fetch broke" cannot
        #: tell the two apart -- and the expected group is the larger one.
        "missing_unindexed": [],
        "mismatched": [],
        "logo_without_file": [],
        "icon_missing": [],
    }

    for row in rows:
        row["阵营"] = (row.get("阵营") or "").strip()
        # Every row carries every column even when the value is empty: the JSON
        # copy is read straight by the panel, and a key that is only present on
        # some rows is a KeyError waiting for whoever iterates it.
        row.setdefault("阵营来源", "")
        grid = factions.get(row.get("页面名", "")) or factions.get(row.get("干员名", ""))
        if grid:
            # Only fill blanks here: the grid leaves group/team empty for most
            # operators, and emptying a value the infobox did document would
            # throw away the only description of it on this row.
            for column, key in (
                ("所属国家", "nation"),
                ("所属组织", "group"),
                ("所属团队", "team"),
            ):
                if grid[key] and not (row.get(column) or "").strip():
                    row[column] = grid[key]
            if grid["logo"]:
                row["阵营"] = grid["logo"]
                row["阵营来源"] = FACTION_PAGE
                report["from_grid"] += 1
            # The grid is the second independent read of 职业 and 情报编号.
            # A disagreement is printed rather than resolved: both copies look
            # authoritative, and picking one silently is how a wrong icon ends
            # up on a card that nobody can trace back to a source.
            for column, value in (("职业", grid["profession"]), ("情报编号", grid["id"])):
                if value and (row.get(column) or "").strip() and row[column] != value:
                    report["mismatched"].append(
                        f"{row.get('干员名') or row.get('页面名')}："
                        f"{column} {row[column]}（档案）≠ {value}（{FACTION_PAGE}）"
                    )

        for column in ("所属国家", "所属组织", "所属团队"):
            row[column] = (row.get(column) or "").strip()
        if not (row.get("阵营") or "").strip():
            # No grid card: this operator is an NPC or an elite operator, and
            # the infobox is all there is. Accept it only when the value names
            # a logo that exists -- otherwise leave it empty and say so.
            for column in ("所属国家", "所属组织", "所属团队"):
                if row[column] and row[column] in available:
                    row["阵营"] = row[column]
                    row["阵营来源"] = "干员档案"
                    report["from_infobox"] += 1
                    break

        faction = (row.get("阵营") or "").strip()
        row["阵营logo"] = (
            library_file(root, FACTION_LOGO_FOLDER, f"Logo {faction}.png")
            if faction and faction in available
            else ""
        )
        if not faction:
            # 干员序号 is the game's own operator index; ``-1`` marks a page with
            # no in-game entry. That is the same marker the roster merge above
            # treats as "not a real operator", so it is the honest separator
            # between "the wiki has nothing to give" and "this fetch missed".
            serial = (row.get("干员序号") or "").strip()
            name = row.get("干员名") or row.get("页面名") or "?"
            if serial.lstrip("-").isdigit() and int(serial) >= 0:
                report["missing"].append(name)
            else:
                report["missing_unindexed"].append(name)
        elif not row["阵营logo"]:
            report["logo_without_file"].append(
                f"{row.get('干员名') or row.get('页面名')} / {faction}"
            )

        profession = (row.get("职业") or "").strip()
        for tone, folder, pattern in CLASS_ICON_FOLDERS:
            column = f"职业图标_{tone}"
            if not profession:
                row[column] = ""
                continue
            row[column] = library_file(
                root, folder, pattern.format(profession=profession)
            )
            if not row[column]:
                report["icon_missing"].append(f"{row.get('干员名')} / {profession} / {tone}")

        page = (row.get("页面名") or row.get("干员名") or "").strip()
        row["PRTS链接"] = f"https://prts.wiki/w/{urllib.parse.quote(page)}" if page else ""

    return report


def print_faction_report(report: dict) -> None:
    """Say out loud what the faction pass did, and what it could not do.

    Every count printed here is a thing that would otherwise be invisible: an
    empty cell in a spreadsheet nobody opens, or a logo area that draws nothing.
    """
    print(
        f"  阵营：干员一览 {report['from_grid']} 行、档案页补 {report['from_infobox']} 行；"
        f"仍为空 {len(report['missing'])} 行"
        f"（另有 {len(report['missing_unindexed'])} 行本就没有游戏编号，属预期内）",
        flush=True,
    )
    if report["missing"]:
        print(f"    !! 有编号却没阵营（{len(report['missing'])}）：{'、'.join(report['missing'])}")
    if report["missing_unindexed"]:
        print(f"    无编号条目（卫戍协议/NPC，维基一览里就没有）：{'、'.join(report['missing_unindexed'])}")
    if report["logo_without_file"]:
        print(f"    阵营有值但库里没有 logo（{len(report['logo_without_file'])}）：")
        for item in report["logo_without_file"]:
            print(f"      {item}")
    if report["mismatched"]:
        print(f"    !! 与档案页不一致（{len(report['mismatched'])}）：")
        for item in report["mismatched"]:
            print(f"      {item}")
    if report["icon_missing"]:
        print(f"    !! 职业图标缺失（{len(report['icon_missing'])}）：")
        for item in set(report["icon_missing"]):
            print(f"      {item}")


INDEX_FOLDER = "01-干员代号"
INDEX_CSV = "干员代号与编号.csv"
INDEX_JSON = "干员代号与编号.json"

#: Everything the badge template composites needs, in one row per operator.
#: The faction and asset columns are at the end so that a reader who only wants
#: 代号 / 编号 / 职业 reads the first seven and stops.
INDEX_COLUMNS: tuple[str, ...] = (
    "干员序号",
    "干员名",
    "情报编号",
    "干员id",
    "稀有度",
    "职业",
    "分支",
    "阵营",
    "所属国家",
    "所属组织",
    "所属团队",
    "阵营来源",
    "阵营logo",
    "职业图标_黑",
    "职业图标_白",
    "PRTS链接",
    "页面名",
    "来源",
)

#: The columns :func:`apply_factions` owns. Refreshable on their own, without
#: re-reading a single operator page.
FACTION_COLUMNS: tuple[str, ...] = (
    "阵营",
    "所属国家",
    "所属组织",
    "所属团队",
    "阵营来源",
    "阵营logo",
    "职业图标_黑",
    "职业图标_白",
    "PRTS链接",
)


def read_index_rows(out_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    """The index on disk, as (rows, column order)."""
    path = out_dir / INDEX_FOLDER / INDEX_CSV
    if not path.exists():
        raise SystemExit(f"还没有干员索引，先跑一次完整抓取：{path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_index_rows(
    out_dir: Path, rows: Sequence[dict[str, str]], columns: Sequence[str]
) -> Path:
    """Write both copies of the index, with a stable column order.

    Both are written from the same list on purpose: the CSV is for a person
    opening the folder, the JSON is what the panel reads, and letting them be
    produced by two passes is how they end up describing different rosters.

    Every row is filled out to every column first. ``csv`` would paper over a
    missing key with ``restval``, but the JSON copy would simply omit it -- so
    a refresh that only recomputed the faction columns would produce a file
    whose rows disagree about which keys exist, and the panel reads the JSON.
    """
    folder = out_dir / INDEX_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    complete = [
        {column: (row.get(column) or "") for column in columns} for row in rows
    ]
    csv_path = folder / INDEX_CSV
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="")
        writer.writeheader()
        writer.writerows(complete)
    (folder / INDEX_JSON).write_text(
        json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path


def refresh_factions(
    out_dir: Path, factions: dict[str, dict[str, str]] | None = None
) -> dict:
    """Recompute every faction and asset column from the index already on disk.

    Separate from :func:`write_operator_index` because the two halves of this
    file have completely different availability: the roster needs 463 operator
    pages through the API, while the factions need one page view. On
    2026-09-26 the API was down for hours and the page views were still served
    -- a full rebuild was impossible while this refresh was not.
    """
    rows, columns = read_index_rows(out_dir)
    if factions is None:
        factions = faction_records()
    report = apply_factions(rows, out_dir, factions)
    print_faction_report(report)
    # Rewritten onto the canonical column list rather than left in whatever
    # order the file happened to have, so the two copies of this index -- the
    # mirror's and the badge subset's -- stay comparable. A refresh that
    # silently reshuffles columns makes every later diff of the two look like a
    # data change, and the data is the part that matters.
    ordered = list(INDEX_COLUMNS) + [
        column for column in columns if column not in INDEX_COLUMNS
    ]
    write_index_rows(out_dir, rows, ordered)
    return {
        "rows": len(rows),
        "factions_from_grid": report["from_grid"],
        "factions_from_infobox": report["from_infobox"],
        "factions_missing": len(report["missing"]),
        "factions_unindexed": len(report["missing_unindexed"]),
        "faction_mismatches": len(report["mismatched"]),
        "faction_logos_missing": len(report["logo_without_file"]),
        "class_icons_missing": len(set(report["icon_missing"])),
    }


def write_operator_index(out_dir: Path) -> dict:
    folder = out_dir / INDEX_FOLDER
    folder.mkdir(parents=True, exist_ok=True)

    pages = operator_pages()
    print(f"分类:干员 -> {len(pages)} 个干员页面")
    records = operator_records(pages)
    table = number_table_rows()
    factions = faction_records()

    # Two wiki quirks have to be normalised before the two sources can be
    # merged: missing values are written as the literal string "None", and
    # ``干员序号=-1`` marks an operator that has no in-game index (event-only
    # and NPC entries such as the 卫戍协议 roster).
    for row in records:
        for key, value in list(row.items()):
            if value.strip() in {"None", "-", "—", "?"}:
                row[key] = ""

    # The number table is the older of the two sources and lags behind new
    # operators, while brand new pages sometimes ship without 情报编号. Filling
    # in either direction is what makes the index complete on its own.
    #
    # Preference matters: an alternate-mode page (e.g. 暮落(集成战略)) copies the
    # base operator's 干员名, so a plain name lookup can backfill the wrong page.
    # Pages whose own title matches the table entry are the real operators.
    lookup: dict[str, dict] = {}
    for row in records:
        if row.get("干员名") and row["干员名"] == row.get("页面名"):
            lookup[row["干员名"]] = row
    for row in records:
        for key in ("干员名", "页面名"):
            if row.get(key):
                lookup.setdefault(row[key], row)

    filled = 0
    for code, name in table:
        row = lookup.get(name)
        if row is not None and not row.get("情报编号"):
            row["情报编号"] = code
            row["来源"] = f"干员档案 + {ISSUE_NUMBER_PAGE}"
            filled += 1

    known = {row.get("情报编号", "") for row in records if row.get("情报编号")}
    extra = [
        {
            "干员名": name,
            "情报编号": code,
            "页面名": name,
            "来源": f"{ISSUE_NUMBER_PAGE}（不在分类:干员）",
        }
        for code, name in table
        if code not in known
    ]

    columns = list(INDEX_COLUMNS)
    rows = []
    for row in records:
        rows.append(
            {column: row.get(column, "") for column in columns}
            | {"来源": row.get("来源") or "干员档案"}
        )
    rows.extend(
        {column: row.get(column, "") for column in columns} | {"来源": row["来源"]}
        for row in extra
    )

    report = apply_factions(rows, out_dir, factions)
    print_faction_report(report)

    def sort_key(row: dict[str, str]) -> tuple[int, int, str]:
        raw = row.get("干员序号") or ""
        index = int(raw) if raw.lstrip("-").isdigit() else -1
        return (1, 0, row.get("干员名") or row.get("页面名", "")) if index < 0 else (0, index, "")

    rows.sort(key=sort_key)
    csv_path = write_index_rows(out_dir, rows, columns)
    (folder / f"{ISSUE_NUMBER_PAGE}.wikitext").write_text(
        page_wikitext(ISSUE_NUMBER_PAGE), encoding="utf-8"
    )
    return {
        "operators": len(records),
        "numbers_from_table": len(table),
        "numbers_filled_from_table": filled,
        "faction_page": FACTION_PAGE,
        "factions_from_grid": report["from_grid"],
        "factions_from_infobox": report["from_infobox"],
        "factions_missing": len(report["missing"]),
        "factions_unindexed": len(report["missing_unindexed"]),
        "faction_mismatches": len(report["mismatched"]),
        "faction_logos_missing": len(report["logo_without_file"]),
        "class_icons_missing": len(set(report["icon_missing"])),
        "rows_written": len(rows),
        "csv": str(csv_path),
    }


# --------------------------------------------------------------------------
# downloads


def download(session_headers: dict, url: str, target: Path) -> str:
    if target.exists() and target.stat().st_size > 0:
        return "skip"
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=session_headers)
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError("empty body")
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(payload)
            temp.replace(target)
            return "ok"
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}") from last


def collect(group: Group) -> list[str]:
    """Every file in this group, as its on-disk display name.

    The regexes in :data:`GROUPS` are the wiki's own naming rules and are written
    against the underscore form, so matching happens before the name is
    normalised for the filesystem.
    """
    found: set[str] = set()
    for prefix in group.prefixes:
        found.update(name for name in prefix_titles(prefix) if group.pattern.match(name))
    return sorted(display_name(name) for name in found)


def sanitize(name: str) -> str:
    """PRTS titles are already path-safe, but never trust that for a write."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip().rstrip(".")
    return cleaned or "unnamed"


def run_groups(out_dir: Path, selected: Sequence[str], jobs: int) -> dict:
    headers = {"User-Agent": USER_AGENT, "Referer": "https://prts.wiki/"}
    manifest: dict[str, list[dict]] = {}
    unresolved: list[str] = []
    for group in GROUPS:
        if selected and group.key not in selected and group.folder not in selected:
            continue
        titles = collect(group)
        print(f"[{group.folder}] 找到 {len(titles)} 个文件", flush=True)
        if not titles:
            manifest[group.key] = []
            continue
        urls = resolve_urls(titles)
        entries: list[dict] = []
        tasks = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for title in titles:
                info = urls.get(title)
                if not info:
                    unresolved.append(f"{group.folder}/{title}")
                    continue
                target = out_dir / group.folder / sanitize(title)
                entry = {
                    "file": str(target.relative_to(out_dir)).replace("\\", "/"),
                    "title": title,
                    "bytes": 0,
                }
                entries.append(entry)
                tasks.append(
                    (pool.submit(download, headers, info["url"], target), entry, target)
                )
            done = 0
            for future, entry, target in tasks:
                future.result()
                # The size that goes in the index is the size that landed on
                # disk, not the one the API declared. The two differ whenever
                # a wiki file was replaced between the listing and the fetch,
                # and storing the declared one made every check afterwards
                # report a mismatch that no re-download could ever clear.
                entry["bytes"] = target.stat().st_size
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)}", flush=True)
        manifest[group.key] = sorted(entries, key=lambda entry: entry["file"])
    if unresolved:
        # Loud, and counted: a silent shortfall here looks exactly like a
        # category that legitimately has fewer files.
        print(f"\n!! {len(unresolved)} 个标题拿不到直链：")
        for item in unresolved[:20]:
            print(f"   {item}")
        if len(unresolved) > 20:
            print(f"   …还有 {len(unresolved) - 20} 个")
    return manifest


# --------------------------------------------------------------------------
# naming the skin files after the outfits they show
#
# Both the 半身像 and the 立绘 sets ship as ``<前缀> <干员> skinN.png``, and N is
# the same number as the ``时装N名称`` field on the operator's own page. One
# rule therefore names both classes; :data:`SKIN_CLASSES` is what says which
# prefix belongs to which group.

#: The two asset classes that carry skins, as (group key, file-name prefix).
#:
#: Both use the wiki's ``<前缀> <干员> skinN.png`` shape and the operator page's
#: own ``时装N名称`` numbering, so one rule names all of them. The prefix stays
#: part of the rule because it is the only thing that says which class a file
#: belongs to once the two libraries are read side by side.
SKIN_CLASSES: tuple[tuple[str, str], ...] = (
    ("half_body_skin", "半身像"),
    ("full_skin", "立绘"),
)

#: ``|时装N名称=报童`` inside an operator's infobox.
SKIN_FIELD = re.compile(r"\|时装(\d+)名称=([^\n|]*)")


def skin_file_pattern(prefix: str) -> re.Pattern[str]:
    """``<前缀> <干员> skinN.png``, with the prefix escaped rather than inlined."""
    return re.compile(rf"^{re.escape(prefix)} (.+) skin(\d+)\.png$")


#: Files PRTS stores under a name that does not describe them.
#:
#: Keyed by wiki title; the value is the group it actually belongs to, the name
#: it should carry, and why. This is a **curated** table: unlike the skin names
#: it cannot be derived from the wiki, because the wiki is what is wrong. It
#: lives here rather than only on disk so that re-downloading the library
#: reproduces the correction instead of quietly undoing it.
#:
#: ``半身像 斑点 skin2.png`` is 斑点's **精英1** artwork. It is not a costume:
#: the operator page documents a single 时装 (专业人士), 斑点 is a 3★ whose
#: promotion art arrives at 精英1, and the file's own pixels are the same
#: uniform as his 精英0 portrait in a different pose. Re-checked against every
#: other 3★: none of the 22 has an 精英2 portrait -- 3★ cap at 精英1 -- and
#: none carries a stray ``skinN``.
#:
#: The 立绘 side of the same pair gets the identical treatment. ``立绘 斑点
#: skin2.png`` is his promotion entry as well, and its pixels are near-identical
#: to ``立绘 斑点 1.png`` -- for a 3★ the promotion art is the same illustration
#: as 精英0, which is exactly what the bust re-crop shows.
MISFILED_ASSETS: dict[str, tuple[str, str, str]] = {
    "半身像 斑点 skin2.png": (
        "half_body_e1",
        "半身像 斑点 精英1.png",
        "维基把精英1晋升立绘传成了 skin2",
    ),
    "立绘 斑点 skin2.png": (
        "full_e1",
        "立绘 斑点 精英1.png",
        "维基把精英1晋升立绘传成了 skin2；与 立绘 斑点 1.png 近乎逐像素相同",
    ),
    "立绘 阿米娅 1+.png": (
        "full_e1",
        "立绘 阿米娅 精英1.png",
        "维基用 1+ 表示精英1晋升立绘；改成与半身像一致的「精英1」写法",
    ),
}



def operator_skin_names(pages: Sequence[str]) -> dict[str, dict[str, str]]:
    """Operator page title -> {skin index: outfit name}.

    The names live in the operator's 干员信息 block (``时装1名称``, ``时装2名称``
    ...), which is the same numbering the portrait files use. Reading them from
    the page rather than hard-coding a table means a newly released outfit is
    picked up by re-running this, with nothing to update here.
    """
    found: dict[str, dict[str, str]] = {}
    for start in range(0, len(pages), 50):
        chunk = pages[start : start + 50]
        payload = api(
            {
                "action": "query",
                "prop": "revisions",
                "rvslots": "main",
                "rvprop": "content",
                "titles": "|".join(chunk),
                "formatversion": "2",
            }
        )
        for page in payload.get("query", {}).get("pages", []):
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            text = revisions[0]["slots"]["main"]["content"]
            names = {
                index: name.strip()
                for index, name in SKIN_FIELD.findall(text)
                if name.strip()
            }
            if names:
                found[page["title"]] = names
    return found


def skin_rename_plan(out_dir: Path, manifest: dict) -> list[dict]:
    """What each skin file should be called, and why it may not change.

    Covers every class in :data:`SKIN_CLASSES`. They share both the wiki's
    ``skinN`` convention and the operator page's ``时装N`` numbering, so this is
    one rule applied twice rather than two rules that have to be kept in step.

    Returns one record per skin file with ``target`` set when the rename is
    possible and ``reason`` set when it is not. Nothing is touched here; the
    caller decides whether to apply.
    """
    names = operator_skin_names(operator_pages())

    plan: list[dict] = []
    claimed: dict[str, dict] = {}
    for group_key, prefix in SKIN_CLASSES:
        pattern = skin_file_pattern(prefix)
        for entry in manifest.get("groups", {}).get(group_key, []):
            name = Path(entry["file"]).name
            record = {
                "file": entry["file"],
                "name": name,
                "target": "",
                "reason": "",
                "group": group_key,
            }
            plan.append(record)
            match = pattern.match(name)
            if not match:
                # Already renamed, or not a skin file at all. Both are fine --
                # the second run of this command must not try to rename again.
                record["reason"] = "文件名不含 skinN（已改过名或不是皮肤图）"
                continue

            operator, index = match.group(1), match.group(2)
            outfit = (names.get(operator) or {}).get(index)
            if not outfit:
                # No operator page at all, or the page documents fewer outfits
                # than there are files -- 嵯峨 carries a stray ``skin0`` beside
                # the outfit the page actually documents. Both are reported
                # rather than guessed at: a wrong costume name is worse than
                # the wiki's own skinN.
                record["reason"] = "wiki 上没有对应时装名称"
                continue

            target = (
                f"{Path(entry['file']).parent}/{sanitize(f'{prefix} {operator} {outfit}.png')}"
            )
            target = target.replace("\\", "/")
            # Keyed by the whole target path, so a clash is only ever two files
            # in one folder claiming the same name -- the 半身像 and 立绘 sets
            # share outfit names by design and live in different folders.
            holder = claimed.get(target)
            if holder is not None:
                record["reason"] = f"与其他皮肤重名（{holder['name']}）"
                continue

            claimed[target] = record
            record["target"] = target
    return plan


def rename_skins(out_dir: Path, manifest: dict, *, apply: bool) -> dict:
    """Rename ``<前缀> <干员> skinN.png`` to ``<前缀> <干员> <时装名>.png``.

    The manifest is rewritten alongside the files. Leaving it pointing at the
    old names would make the very next ``--verify`` report the whole skin set
    as missing, which is the loudest possible way to get a self-inflicted
    false alarm.

    A file whose operator has no page keeps its ``skinN`` name. The ones that
    land there are printed with the reason, so a gap in the naming is visible
    instead of looking like a complete rename.
    """
    plan = skin_rename_plan(out_dir, manifest)
    renamable = [item for item in plan if item["target"]]
    skipped = [item for item in plan if not item["target"]]

    if not plan:
        # Nothing to look at. Checking here rather than fetching every operator
        # page first means a run over an unrelated group costs no API calls.
        print("皮肤图：本库没有这些分组，跳过")
        return {"renamed": 0, "planned": 0, "skipped": []}
    print(f"皮肤图共 {len(plan)} 个，可改名 {len(renamable)} 个")
    for item in renamable[:10]:
        print(f"  {item['name']}  ->  {Path(item['target']).name}")
    if len(renamable) > 10:
        print(f"  …还有 {len(renamable) - 10} 个")

    if skipped:
        print(f"\n保留原名 {len(skipped)} 个：")
        for item in skipped:
            print(f"  {item['name']}：{item['reason']}")

    if not apply:
        print("\n预览模式，没有改动。加 --apply 执行。")
        return {"renamed": 0, "planned": len(renamable), "skipped": skipped}

    by_old = {item["file"]: item for item in renamable}
    # Captured before the entries are rewritten: the exclusion list still holds
    # the old paths, and it is matched against them.
    moves = {item["file"]: item["target"] for item in renamable}

    renamed = 0
    for group_key, _prefix in SKIN_CLASSES:
        for entry in manifest.get("groups", {}).get(group_key, []):
            item = by_old.get(entry["file"])
            if item is None:
                continue
            source = out_dir / item["file"]
            target = out_dir / item["target"]
            if target.exists() and target != source:
                print(f"  !! 目标已存在，跳过：{item['target']}")
                continue
            if not source.exists():
                print(f"  !! 源文件不存在，跳过：{item['file']}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            entry["file"] = item["target"]
            entry["bytes"] = target.stat().st_size
            renamed += 1

    # Exclusions are paths too. One that named a renamed skin has to follow it,
    # or a later check would treat the exclusion as stale and report the file as
    # missing in the same breath.
    skip_list = excluded_files(manifest)
    if skip_list:
        followed = [moves.get(path, path) for path in skip_list]
        if followed != list(skip_list):
            print(f"排除清单里 {sum(1 for a, b in zip(skip_list, followed) if a != b)} 项跟着改名")
        manifest[EXCLUDED_KEY] = sorted(followed)

    write_manifest(out_dir, manifest)
    # A record of what moved, kept next to the index. Renaming is the one
    # operation whose result cannot be re-derived from the library alone -- the
    # old name is gone -- so the mapping is written down for auditing and for
    # undoing the rename if a name turns out to be wrong.
    if moves:
        (out_dir / "_索引").mkdir(parents=True, exist_ok=True)
        (out_dir / "_索引" / "skin_rename_map.json").write_text(
            json.dumps(
                {old: new for old, new in sorted(moves.items())},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"\n已改名 {renamed} 个，索引已同步")
    return {"renamed": renamed, "planned": len(renamable), "skipped": skipped}


def apply_curation(out_dir: Path, manifest: dict, *, apply: bool) -> dict:
    """Move and rename the files :data:`MISFILED_ASSETS` describes.

    Re-runnable: a file already at its corrected path is skipped, so this can
    run after every download without duplicating anything.

    The file leaves its original group and joins the ``curated`` list instead.
    It has to be in exactly one of them: leaving a copy in the group would make
    the check look for it at the old path, which no longer exists.
    """
    plan: list[tuple[dict, str, str, str]] = []
    for entry in all_entries(manifest):
        fix = MISFILED_ASSETS.get(Path(entry["file"]).name)
        if fix is None:
            continue
        group_key, new_name, reason = fix
        folder = next(group.folder for group in GROUPS if group.key == group_key)
        plan.append((entry, f"{folder}/{new_name}", group_key, reason))

    print(f"需要归位的文件 {len(plan)} 个")
    for entry, target, _, reason in plan:
        already = entry["file"] == target
        print(f"  {entry['file']}  ->  {target}（{reason}）{'  [已就位]' if already else ''}")

    if not apply:
        print("\n预览模式，没有改动。加 --apply 执行。")
        return {"planned": len(plan), "moved": 0}

    moved = 0
    curated: list[dict] = []
    for entry, target, group_key, reason in plan:
        source = out_dir / entry["file"]
        destination = out_dir / target
        if source != destination:
            if destination.exists():
                print(f"  !! 目标已存在，跳过：{target}")
                continue
            if not source.exists():
                print(f"  !! 源文件不存在，跳过：{entry['file']}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved += 1
        curated.append(
            {
                "file": target,
                "title": entry.get("title") or Path(entry["file"]).name,
                "bytes": destination.stat().st_size if destination.exists() else 0,
                "group": group_key,
                "reason": reason,
            }
        )

    # Drop the moved files from their original groups, keeping every other
    # curated entry from an earlier run.
    names = set(MISFILED_ASSETS)
    for key, entries in manifest.get("groups", {}).items():
        manifest["groups"][key] = [
            entry for entry in entries if Path(entry["file"]).name not in names
        ]
    keep = [
        entry
        for entry in curated_entries(manifest)
        if Path(entry["file"]).name not in {Path(item["file"]).name for item in curated}
    ]
    manifest[CURATED_KEY] = keep + curated

    write_manifest(out_dir, manifest)
    print(f"\n已归位 {moved} 个，索引已同步")
    return {"planned": len(plan), "moved": moved}


# --------------------------------------------------------------------------
# checking what is already on disk


EXCLUDED_KEY = "excluded"
"""Manifest key listing files deliberately removed from the library.

A mirror the user has pruned is not a broken mirror. Without recording those
deletions, every later check would report them as missing and either nag
forever or -- worse -- helpfully put back something that was removed on
purpose.
"""

CURATED_KEY = "curated"
"""Manifest key listing files whose placement came from :data:`MISFILED_ASSETS`.

These are real files that no wiki query can reproduce: the wiki title they were
downloaded under is wrong, so re-collecting titles from the wiki would put them
back where they do not belong. Keeping them in their own list means a rebuild
knows about them, the check still verifies them, and the subset copy still
carries them.
"""


def curated_entries(manifest: dict) -> list[dict]:
    return list(manifest.get(CURATED_KEY) or [])


def all_entries(manifest: dict) -> list[dict]:
    """Every file the library claims to hold, curated ones included.

    One place to ask "what should be on disk", so checking and copying cannot
    disagree about whether curated files count.
    """
    entries = [
        entry
        for group_entries in manifest.get("groups", {}).values()
        for entry in group_entries
    ]
    entries.extend(curated_entries(manifest))
    return entries


def read_manifest(out_dir: Path) -> dict:
    path = out_dir / "_索引" / "manifest.json"
    if not path.exists():
        raise SystemExit(f"没有索引文件，先跑一次完整下载：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def excluded_files(manifest: dict) -> set[str]:
    return set(manifest.get(EXCLUDED_KEY) or [])


def verify_library(out_dir: Path, manifest: dict) -> list[dict]:
    """Compare the manifest against what is actually on disk.

    Written after one file went missing between a completed run and the next
    look at the folder -- the run itself reported success, and nothing about
    the folder *looked* wrong. A mirror that is only checked by the program
    that wrote it is not checked at all; this is the pass that can be re-run at
    any time, including after the user has moved things around.

    Files listed under :data:`EXCLUDED_KEY` are skipped: those are the ones the
    user removed on purpose. Returns the entries that are missing or the wrong
    size.
    """
    skip = excluded_files(manifest)
    problems: list[dict] = []
    checked = 0
    for key, entries in manifest.get("groups", {}).items():
        for entry in entries:
            if entry["file"] in skip:
                continue
            target = out_dir / entry["file"]
            checked += 1
            if not target.exists():
                problems.append({**entry, "group": key, "problem": "缺失"})
                continue
            actual = target.stat().st_size
            if actual != entry.get("bytes"):
                problems.append(
                    {**entry, "group": key, "problem": f"大小不符 {actual}≠{entry['bytes']}"}
                )
    for entry in curated_entries(manifest):
        if entry["file"] in skip:
            continue
        target = out_dir / entry["file"]
        checked += 1
        if not target.exists():
            problems.append(
                {**entry, "group": entry.get("group", CURATED_KEY), "problem": "缺失"}
            )
            continue
        actual = target.stat().st_size
        if actual != entry.get("bytes"):
            problems.append(
                {
                    **entry,
                    "group": entry.get("group", CURATED_KEY),
                    "problem": f"大小不符 {actual}≠{entry['bytes']}",
                }
            )
    print(f"核对 {checked} 个文件（另有 {len(skip)} 个已按记录排除），问题 {len(problems)} 个")
    for item in problems[:30]:
        print(f"  [{item['group']}] {item['problem']}：{item['file']}")
    if len(problems) > 30:
        print(f"  …还有 {len(problems) - 30} 个")
    return problems


def repair_library(out_dir: Path, manifest: dict, problems: Sequence[dict], jobs: int) -> int:
    """Re-download the files :func:`verify_library` flagged."""
    if not problems:
        return 0
    headers = {"User-Agent": USER_AGENT, "Referer": "https://prts.wiki/"}
    by_path = {
        entry["file"]: entry
        for entries in manifest.get("groups", {}).values()
        for entry in entries
    }
    fixed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        tasks = {}
        for item in problems:
            url = resolve_urls([item["title"]]).get(item["title"])
            if not url:
                print(f"  !! 仍然拿不到直链：{item['title']}")
                continue
            target = out_dir / item["file"]
            if target.exists():
                target.unlink()
            tasks[pool.submit(download, headers, url["url"], target)] = item
        for future in as_completed(tasks):
            item = tasks[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  !! 修复失败 {item['file']}：{exc}")
                continue
            # Record what was actually written, so the next check compares
            # against reality rather than against a stale API figure.
            by_path[item["file"]]["bytes"] = (out_dir / item["file"]).stat().st_size
            fixed += 1
    print(f"已补 {fixed}/{len(problems)} 个")
    return fixed


# --------------------------------------------------------------------------


MANIFEST_NAME = "manifest.json"


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "_索引" / MANIFEST_NAME


def write_manifest(out_dir: Path, manifest: dict) -> None:
    """Persist the index, keeping its non-group keys intact.

    Written as a whole dict rather than rebuilt from the group lists, so the
    exclusion list and the operator index survive a repair that only touches
    one file.
    """
    manifest_path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(out_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_manifest(manifest: dict, index: dict) -> dict:
    """Assemble the index document, preserving anything already recorded."""
    return {
        "source": manifest.get("source")
        or f"https://prts.wiki/w/{urllib.parse.quote(FILE_INDEX_PAGE)}",
        "fetched": time.strftime("%Y-%m-%d %H:%M"),
        "operator_index": index or manifest.get("operator_index") or {},
        "groups": manifest.get("groups", {}),
        EXCLUDED_KEY: manifest.get(EXCLUDED_KEY, []),
        CURATED_KEY: manifest.get(CURATED_KEY, []),
    }


def _manifest_groups(previous: dict, fresh: dict, selected: Sequence[str]) -> dict:
    """Group lists for the index after a run.

    A run limited by ``--only`` touches some groups and knows nothing about the
    others. Older entries for the groups it did not touch are kept, because
    dropping them would make the next check compare against half a manifest and
    call the untouched categories missing.
    """
    if not selected:
        return {"groups": fresh}
    groups = dict(previous.get("groups", {}))
    groups.update(fresh)
    return {"groups": groups}


def write_readme(out_dir: Path, manifest: dict, index: dict) -> None:
    exclude_note = ""
    skip = excluded_files(manifest)
    if skip:
        exclude_note = (
            f"另有 **{len(skip)} 个文件**是用户自己删掉的（记录在索引的 "
            f"`{EXCLUDED_KEY}` 里），核对时会跳过，不会被重新下载。"
        )

    # Counted from what is actually there, not from the length of the index:
    # excluded files are gone, and curated files belong to a folder without
    # being part of that group's list. Both would make the table disagree with
    # the folder the reader is about to open.
    counts: dict[str, int] = {}
    for group in GROUPS:
        entries = manifest.get("groups", {}).get(group.key, [])
        counts[group.key] = sum(1 for entry in entries if entry["file"] not in skip)
    curated_note = ""
    curated = [
        entry for entry in curated_entries(manifest) if entry["file"] not in skip
    ]
    for entry in curated:
        key = entry.get("group")
        if key in counts:
            counts[key] += 1
    if curated:
        lines = [
            "以下文件是**人工归位**的 —— PRTS 用的名字对不上它们实际是什么，"
            "见索引的 `curated`，以及 `tools/fetch_prts_assets.py` 里的 "
            "`MISFILED_ASSETS`：",
            "",
        ]
        for entry in curated:
            lines.append(
                f"- `{entry['file']}`（维基原名 `{entry.get('title', '')}`）——"
                f"{entry.get('reason', '')}"
            )
        curated_note = "\n".join(lines) + "\n"

    lines = [
        "# 明日方舟素材库（PRTS 抓取）",
        "",
        f"来源：<https://prts.wiki/w/{urllib.parse.quote(FILE_INDEX_PAGE)}>",
        f"抓取时间：{manifest.get('fetched') or time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "按请求的分类分别存放，一个文件夹对应一类素材：",
        "",
        "| 文件夹 | 数量 | 说明 |",
        "|---|---:|---|",
    ]
    for group in GROUPS:
        lines.append(f"| `{group.folder}` | {counts[group.key]} | {group.note} |")
    lines += [
        "",
        "`01-干员代号/干员代号与编号.csv`：全部干员的代号、情报编号、"
        "干员 id、稀有度、职业与分支（合并「分类:干员」各档案页与"
        f"「{ISSUE_NUMBER_PAGE}」表，两者不一致时以档案页为准）。",
        "",
        "同一张表还带通行证模板要用的合成素材：**阵营**（`所属国家` / "
        "`所属组织` / `所属团队` 三格同表记录）"
        "以及 `阵营logo`、`职业图标_黑`、`职业图标_白` 三个**库内相对路径**。"
        f"阵营优先取「{FACTION_PAGE}」的 `logo` 值 —— 那正是维基用来选 logo 图的值；"
        "该页没覆盖的干员（卫戍协议临时干员、`预备干员-*`、NPC）"
        "退回档案页的 `所属*` 三格，**且只有值确实能在 `18-阵营logo` 里"
        "找到文件时才填** —— 填一个不存在的路径会把「数据里没有」"
        "变成面板上看起来的绘图故障。",
        "",
        "`PRTS链接` 是该干员页面的完整地址，供二维码使用。",
        "",
        "⚠️ 只刷新这张表（不碰任何素材文件）："
        "`python tools/fetch_prts_assets.py --index-only`",
        "",
        "`_索引/manifest.json`：每个文件的来源标题与字节数，供调用逻辑核对。"
        "核对命令：`python tools/fetch_prts_assets.py --verify`"
        "（加 `--repair` 则补回缺失的文件）。",
        "",
        *([curated_note, ""] if curated_note else []),
        *([exclude_note, ""] if exclude_note else []),
        "素材版权归鹰角网络所有，仅供个人非商业使用；请勿再分发。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_manifest(out_dir, build_manifest(manifest, index))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="group key or folder name; repeatable",
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--list", action="store_true", help="只统计，不下载")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="只重抓干员索引（含阵营、素材路径），一个素材文件都不碰",
    )
    parser.add_argument(
        "--factions-only",
        action="store_true",
        help="只按「干员一览」重算已有索引里的阵营与素材路径（无需 API）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="核对索引与实际文件，不下载",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="核对并重新下载缺失或大小不符的文件",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="记为「用户已删除」的库内相对路径（可重复），核对时跳过",
    )
    parser.add_argument(
        "--name-skins",
        action="store_true",
        help="把「<前缀> X skinN.png」改成「<前缀> X 时装名.png」（半身像 + 立绘）",
    )
    parser.add_argument(
        "--curate",
        action="store_true",
        help="把维基传错名字的文件归位（见 MISFILED_ASSETS）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="与 --name-skins / --curate 一起用才真正改动，否则只预览",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.curate:
        apply_curation(out_dir, read_manifest(out_dir), apply=args.apply)
        return 0

    if args.name_skins:
        rename_skins(out_dir, read_manifest(out_dir), apply=args.apply)
        return 0

    if args.exclude:
        manifest = read_manifest(out_dir)
        skip = excluded_files(manifest)
        for item in args.exclude:
            relative = item.replace("\\", "/")
            skip.add(relative)
            target = out_dir / relative
            if target.exists():
                target.unlink()
                print(f"已删除并记录：{relative}")
            else:
                print(f"已在排除清单：{relative}")
        manifest[EXCLUDED_KEY] = sorted(skip)
        write_manifest(out_dir, manifest)
        print(f"排除清单现有 {len(skip)} 项")
        # Recording an exclusion is a complete operation on its own. Falling
        # through would start a download, which would put back exactly the file
        # the user just asked to leave out.
        if not (args.verify or args.repair):
            return 0

    if args.verify or args.repair:
        manifest = read_manifest(out_dir)
        problems = verify_library(out_dir, manifest)
        if args.repair and problems:
            repair_library(out_dir, manifest, problems, max(1, args.jobs))
            write_manifest(out_dir, manifest)
            print("\n复查：")
            problems = verify_library(out_dir, manifest)
        return 0 if not problems else 1

    if args.factions_only:
        # One page view, no API, no artwork. This is the path that works when
        # the wiki's API is down, and the one to use after the roster was
        # updated and only the faction columns need recomputing.
        result = refresh_factions(out_dir)
        print(f"阵营刷新：{result}")
        # The README lists what the index holds, and the index just gained nine
        # columns. Leaving it stale is how the documentation starts describing
        # a file that no longer exists. Only the manifest-bearing libraries can
        # be rewritten; the badge subset has no manifest of its own.
        if manifest_path(out_dir).exists():
            manifest = read_manifest(out_dir)
            manifest["operator_index"] = result
            write_readme(out_dir, manifest, result)
        return 0

    if args.index_only:
        # Refreshing the index must not depend on the artwork being reachable,
        # and must not touch it: the wiki text changes far more often than the
        # 4 GB of images does.
        index = write_operator_index(out_dir)
        print(f"干员索引：{index}")
        if manifest_path(out_dir).exists():
            manifest = read_manifest(out_dir)
            manifest["operator_index"] = index
            write_manifest(out_dir, manifest)
        return 0

    if args.list:
        total = 0
        for group in GROUPS:
            if args.only and group.key not in args.only and group.folder not in args.only:
                continue
            titles = collect(group)
            total += len(titles)
            print(f"{group.folder:24s} {len(titles):5d}  {group.note}")
        print(f"{'合计':24s} {total:5d}")
        return 0

    index: dict = {}
    if not args.skip_index and (not args.only or "01-干员代号" in args.only):
        index = write_operator_index(out_dir)
        print(f"干员索引：{index}")

    manifest = run_groups(out_dir, args.only, max(1, args.jobs))
    # A fresh download of a subset must not wipe the exclusion list or the
    # operator index written by an earlier, fuller run.
    previous: dict = {}
    if manifest_path(out_dir).exists():
        try:
            previous = read_manifest(out_dir)
        except (OSError, json.JSONDecodeError):
            previous = {}
    merged = {
        **previous,
        **_manifest_groups(previous, manifest, args.only),
    }
    # Curated files are re-filed as part of the download rather than left as a
    # step someone has to remember. Their wiki titles are wrong, so a fresh
    # fetch always drops them in the wrong folder; without this, re-downloading
    # the library would silently undo the correction.
    apply_curation(out_dir, merged, apply=True)
    # Same reasoning for the skin names: a raw download carries ``skinN``, so
    # the finished-by-name library only survives a re-download if the rename
    # runs here too. It costs one pass over the operator pages, which is noise
    # beside downloading the artwork itself.
    rename_skins(out_dir, merged, apply=True)
    write_readme(out_dir, merged, index)
    print(f"\n完成，输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
