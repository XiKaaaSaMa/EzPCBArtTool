"""Repackage an ``.epro2`` project as an ``.eprj3`` folder.

Why this exists
---------------
EasyEDA Pro's CLI (``lceda-pro``) is the only supported way for this project to
produce a **colour-silkscreen** manufacturing package -- the ``Fabrication_
Colorful*`` files are written by the EDA in an encoding this project cannot
reproduce (measured: encrypted or private, entropy 7.93-8.00 bits/byte).

But that CLI refuses ``.epro2`` outright::

    lceda-pro open --path card.epro2
    → unsupported file type: card.epro2
      (expected .eprj/.eprj2/.eprj3/.elib)

The documented import API
(``SYS_FileManager.importProjectByProjectFile`` with ``fileType: "EasyEDA
Pro"``) also returned ``undefined`` for every spelling tried. So the project is
converted instead.

The conversion is *repackaging*, not translation
------------------------------------------------
Both formats store a document as a sequence of records::

    {"type":"FILL","ticket":12,"id":"e3"}||{"layerId":1,"path":[...]}|

``.epru`` and ``.epcb2`` use the same delimiter and the same record shapes; the
only differences are the container (a zip entry versus a file on disk) and the
project index (``project2.json`` versus ``<name>.eprj3``). So the PCB records are
copied **verbatim** -- nothing is re-derived from millimetres, which is the
failure mode this project keeps running into when a second implementation of
the same geometry is written.
"""

from __future__ import annotations

import json
import re
import time
import uuid
import zipfile
from pathlib import Path

__all__ = [
    "EPRJT3_EXTENSION",
    "document_sections",
    "epro_to_eprj3",
    "pcb_records",
]


EPRJT3_EXTENSION = ".eprj3"

_RECORD = re.compile(r'^\{"type":"([^"]+)"')
_DOC_TYPE = re.compile(r'"docType":"([^"]+)"')
_UUID = re.compile(r'"uuid":"([^"]+)"')
_TITLE = re.compile(r'"title":"([^"]*)"')


def _read_epru(path: Path) -> str:
    """The document text inside an ``.epro2``.

    The archive holds one ``.epru`` beside ``project2.json``; its name is the
    project title, which is why it is found by extension rather than by name.
    """
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".epru")
        ]
        if not names:
            raise ValueError(f"{path.name} 里没有 .epru 文档")
        return archive.read(names[0]).decode("utf-8")


def document_sections(text: str) -> list[tuple[str, str, list[str]]]:
    """Split a document stream into ``(doc_type, uuid, lines)`` sections.

    A ``DOCHEAD`` opens a section and the next ``DOCHEAD`` closes it. The
    trailing ``CONFIG`` and ``BLOB`` sections are returned too -- the caller
    decides which it needs.
    """
    sections: list[tuple[str, str, list[str]]] = []
    current: list[str] | None = None
    doc_type = uid = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith('{"type":"DOCHEAD"'):
            if current is not None:
                sections.append((doc_type, uid, current))
            head = line.split("||", 1)[-1]
            found_type = _DOC_TYPE.search(head)
            found_uuid = _UUID.search(head)
            doc_type = found_type.group(1) if found_type else ""
            uid = found_uuid.group(1) if found_uuid else ""
            current = [line]
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        sections.append((doc_type, uid, current))
    return sections


def pcb_records(text: str) -> tuple[str, list[str]]:
    """The PCB document's ``(uuid, lines)`` from an ``.epru`` stream.

    ⚠️ **The section is copied verbatim.** Re-encoding it -- even through a
    faithful reader and writer -- is how a second implementation of the same
    geometry gets introduced, and this project has measured what that costs:
    two paths that each traced a mask produced an 88 mm² uncovered seam, and a
    ``PAD`` written from the specification's example was silently dropped.
    """
    for doc_type, uid, lines in document_sections(text):
        if doc_type == "PCB":
            return uid, lines
    raise ValueError("这个工程里没有 PCB 文档")


#: The ``board`` uuid the PCB document points at, and its ``title``.
#:
#: ⚠️ **The index has to use these exact values, not fresh ones.** The PCB's
#: ``META`` record carries ``"board":"<uuid>"`` and the editor resolves it
#: against the project index; when the index named a board the document did not
#: point at, the PCB loaded as an empty frame -- layers and rules present, all
#: 139 ``FILL`` records gone -- and ``getGerberFile`` returned ``undefined``.
_BOARD = re.compile(r'"board":"([^"]+)"')


def document_board(text: str) -> str:
    """The board uuid a PCB document belongs to, or an empty string."""
    for doc_type, _uid, lines in document_sections(text):
        if doc_type != "PCB":
            continue
        for line in lines:
            found = _BOARD.search(line)
            if found:
                return found.group(1)
    return ""


def _manifest(
    title: str, pcb_uuid: str, board_uuid: str, *, owner_uuid: str, now: str
) -> dict:
    """The ``<name>.eprj3`` index for a single-PCB project.

    Shapes copied from the official example in
    ``easyeda/easyeda-eprj3-format``. Only the parts a one-PCB project needs are
    filled in; the rest are the empty values that example carries.

    ``pcb_uuid`` and ``board_uuid`` are the ids the *document itself* carries --
    see :data:`_BOARD` for why inventing new ones silently empties the PCB.
    """
    board_uuid = board_uuid or uuid.uuid4().hex[:16]
    version = str(int(time.time() * 1000))
    return {
        "name": title,
        "owner_uuid": owner_uuid,
        "creator_uuid": owner_uuid,
        "created_at": now,
        "updated_at": now,
        "modifier_uuid": owner_uuid,
        "content": "",
        "archive": False,
        "cbb_project": 0,
        "thumb": "",
        "ticket": 1,
        "g_ticket": 1,
        "boards": [],
        "block_symbol_attrs_groups": {},
        "default_sheet": "",
        "branch_uuid": "",
        "pcb_count": 1,
        "format": "folder",
        "profile": {
            "boards": {
                board_uuid: {"uuid": board_uuid, "title": "Board1", "zIndex": 1}
            },
            "schematics": {},
            "sheets": {},
            "pcbs": {
                pcb_uuid: {
                    "uuid": pcb_uuid,
                    "title": title,
                    "board": board_uuid,
                    "parent_uuid": "",
                    "source": "",
                    "version": version,
                    "updateTime": int(version),
                }
            },
            "panels": {},
            "blockSymbols": {},
            "owner": {"uuid": owner_uuid},
            "simSchematics": {},
        },
        "config": {"defaultSheet": "", "settings": {}},
    }


def epro_to_eprj3(
    source: str | Path,
    target_dir: str | Path,
    *,
    title: str = "",
    owner_uuid: str = "70fdbe422b754f5088c0a554a06ba72b",
) -> Path:
    """Write ``source`` (an ``.epro2``) as an ``.eprj3`` folder at ``target_dir``.

    Returns the path of the ``<name>.eprj3`` index. ``target_dir`` is the
    *project* directory; it is created if missing, and must be empty enough
    that the PCB file can be written without clobbering something else.
    """
    source = Path(source)
    target_dir = Path(target_dir)
    text = _read_epru(source)
    pcb_uuid, lines = pcb_records(text)

    name = title.strip() or _safe_name(source.stem) or "PCBartProject"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "pcb").mkdir(exist_ok=True)

    body = "\n".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    (target_dir / "pcb" / f"{name}.epcb2").write_text(body, encoding="utf-8")

    index = _manifest(
        name,
        pcb_uuid,
        document_board(text),
        owner_uuid=owner_uuid,
        now=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    manifest_path = target_dir / f"{name}{EPRJT3_EXTENSION}"
    manifest_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest_path


def _safe_name(value: str) -> str:
    """A file name the EDA will accept: keep letters, digits, dash and CJK.

    The project-name rule the API documents is ``a-zA-Z0-9-``, but project
    *friendly* names and the official examples carry CJK, so CJK is kept and
    only the characters that break a path are replaced.
    """
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    return cleaned.strip(".")
