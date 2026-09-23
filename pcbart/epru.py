"""Reader and writer for the EasyEDA Pro ``.epru`` project log.

An ``.epru`` file is a line-based append-only log. Every line is two JSON
objects joined by ``||`` and terminated by ``|``::

    {"type":"POLY","ticket":351,"id":"b5a1"}||{"layerId":11,...}|

The first object is the consistency-framework header (``type``/``id``/
``ticket``), the second is the primitive's payload. Within one document a
record is uniquely identified by ``type`` + ``id``, and when the same key
appears more than once the highest ``ticket`` wins. EasyEDA Pro relies on that
to append edits without rewriting the file, so writing a record with a fresh,
higher ticket is how we modify an existing design.

See ``docs/M0_FINDINGS.md`` for how this was verified against a real project.
"""

from __future__ import annotations

import io
import json
import secrets
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "Record",
    "EproDocument",
    "EproArchive",
    "split_record_line",
    "new_id",
    "now_ms",
]

JSON_SEPARATORS = (",", ":")
"""EasyEDA Pro writes minified JSON; match it so diffs stay clean."""


def new_id() -> str:
    """A 16-character lowercase hex id, the form EasyEDA Pro uses for primitives."""
    return secrets.token_hex(8)


def now_ms() -> int:
    return int(time.time() * 1000)


def split_record_line(line: str) -> tuple[str, str] | None:
    """Split a log line into its raw header and body JSON text.

    The split point is the first ``||`` that sits outside a JSON string and
    after the header object has closed. A naive ``str.index`` would also work
    on files EasyEDA Pro writes today, but primitive ids are themselves
    serialised JSON in a few record types, so scanning is safer.
    """
    depth = 0
    in_string = False
    escaped = False
    header_end = -1

    for i, ch in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                header_end = i
        elif ch == "|" and depth == 0 and header_end >= 0:
            if i + 1 < len(line) and line[i + 1] == "|":
                return line[:header_end + 1], line[i + 2:]

    return None


@dataclass
class Record:
    """One line of the log: a consistency header plus an optional payload."""

    header: dict[str, Any]
    body: Any = None
    raw: str | None = field(default=None, repr=False, compare=False)

    @property
    def type(self) -> str:
        return str(self.header.get("type", ""))

    @property
    def id(self) -> str | None:
        value = self.header.get("id")
        return None if value is None else str(value)

    @property
    def ticket(self) -> int:
        return int(self.header.get("ticket", 0))

    @property
    def is_deleted(self) -> bool:
        """A record is deleted by writing an empty payload at a higher ticket."""
        return self.body is None or self.body == ""

    def matches(self, record_type: str, record_id: str | None = None) -> bool:
        if self.type != record_type:
            return False
        return record_id is None or self.id == record_id

    def to_line(self) -> str:
        header_text = json.dumps(self.header, separators=JSON_SEPARATORS, ensure_ascii=False)
        if self.body is None:
            body_text = ""
        else:
            body_text = json.dumps(self.body, separators=JSON_SEPARATORS, ensure_ascii=False)
        return f"{header_text}||{body_text}|"

    def to_mm(self) -> Any:
        """Payload with all unit-typed fields converted to millimetres.

        Delegates to :mod:`pcbart.primitives`; kept here as a convenience so
        callers do not need to know which module owns the field table.
        """
        from .primitives import record_to_mm

        return record_to_mm(self)


class EproDocument:
    """An in-memory ``.epru`` log that can be parsed, edited and re-emitted."""

    def __init__(self, records: Iterable[Record] = ()) -> None:
        self.records: list[Record] = list(records)
        self.unparsed: list[str] = []

    # ---------------------------------------------------------------- parsing

    @classmethod
    def parse(cls, lines: Iterable[str]) -> "EproDocument":
        doc = cls()
        for line in lines:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            split = split_record_line(line)
            if split is None:
                doc.unparsed.append(line)
                continue
            header_text, body_text = split
            if body_text.endswith("|"):
                body_text = body_text[:-1]
            try:
                header = json.loads(header_text)
            except json.JSONDecodeError:
                doc.unparsed.append(line)
                continue
            body: Any = None
            if body_text.strip():
                try:
                    body = json.loads(body_text)
                except json.JSONDecodeError:
                    doc.unparsed.append(line)
                    continue
            doc.records.append(Record(header=header, body=body, raw=line))
        return doc

    @classmethod
    def loads(cls, text: str) -> "EproDocument":
        return cls.parse(text.splitlines())

    @classmethod
    def read(cls, path: str | Path) -> "EproDocument":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.parse(handle)

    # ---------------------------------------------------------------- emitting

    def dumps(self) -> str:
        """Serialise back to log text.

        Records that cannot be represented (an empty payload, a bare-string
        body) fall back to their original line so an unmodified document round
        trips byte for byte.
        """
        out: list[str] = []
        for record in self.records:
            if isinstance(record.body, str) or record.body is None:
                if record.raw is not None:
                    out.append(record.raw)
                    continue
            out.append(record.to_line())
        out.extend(self.unparsed)
        return "\n".join(out) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")

    # ---------------------------------------------------------------- querying

    @property
    def max_ticket(self) -> int:
        return max((r.ticket for r in self.records), default=0)

    def next_ticket(self) -> int:
        return self.max_ticket + 1

    def iter_type(self, record_type: str) -> Iterator[Record]:
        for record in self.records:
            if record.type == record_type:
                yield record

    def find(self, record_type: str, record_id: str | None = None) -> list[Record]:
        return [r for r in self.records if r.matches(record_type, record_id)]

    def first(self, record_type: str, record_id: str | None = None) -> Record | None:
        for record in self.records:
            if record.matches(record_type, record_id):
                return record
        return None

    def documents(self) -> list[dict[str, Any]]:
        """DOCHEAD payloads in file order."""
        return [r.body for r in self.iter_type("DOCHEAD") if isinstance(r.body, dict)]

    def sections(self) -> dict[str, list[Record]]:
        """Group records by the document they belong to.

        Keys are ``docType`` values (``PCB``, ``SCH_PAGE``, ``BLOB``, ...).
        Records that appear before any DOCHEAD land under ``"(none)"``.
        """
        grouped: dict[str, list[Record]] = {}
        current = "(none)"
        for record in self.records:
            if record.type == "DOCHEAD" and isinstance(record.body, dict):
                current = str(record.body.get("docType", "?"))
                grouped.setdefault(current, []).append(record)
                continue
            grouped.setdefault(current, []).append(record)
        return grouped

    def append(self, record_type: str, body: Any, record_id: str | None = None) -> Record:
        """Append a new record with the next available ticket."""
        header: dict[str, Any] = {"type": record_type, "ticket": self.next_ticket()}
        if record_id is not None:
            header["id"] = record_id
        record = Record(header=header, body=body)
        self.records.append(record)
        return record

    def append_to_section(self, record: Record, doc_type: str = "PCB") -> None:
        """Insert a record at the end of the ``doc_type`` document section.

        A record belongs to whichever ``DOCHEAD`` precedes it, so geometry has
        to sit *inside* its document's section -- not merely somewhere before
        ``BLOB``. Appending at the very end of the file would file the
        primitives under ``CONFIG``, and EasyEDA Pro would not find them.

        The record is spliced in immediately before the next ``DOCHEAD`` after
        the target section, or at the end of the log if the target is last.
        """
        start = None
        for index, existing in enumerate(self.records):
            if existing.type != "DOCHEAD" or not isinstance(existing.body, dict):
                continue
            if start is None:
                if existing.body.get("docType") == doc_type:
                    start = index
                continue
            # First DOCHEAD after the target section ends it.
            self.records.insert(index, record)
            return
        if start is None:
            raise ValueError(f"no {doc_type!r} section found in this document")
        self.records.append(record)

    def new_record(
        self,
        record_type: str,
        body: Any,
        record_id: str | None = None,
    ) -> Record:
        header: dict[str, Any] = {"type": record_type, "ticket": self.next_ticket()}
        if record_id is not None:
            header["id"] = record_id
        return Record(header=header, body=body)


DEFAULT_PROJECT_META: dict[str, Any] = {
    "cbb_project": False,
    "editorVersion": "",
    "introduction": "",
    "description": "",
    "tags": "[]",
}
"""Fields EasyEDA Pro expects in ``project2.json``.

Partial metadata is filled in from here so a caller cannot accidentally write
an archive that is missing fields the editor relies on.
"""


class EproArchive:
    """The ``.epro2`` ZIP container: ``project2.json`` + one ``.epru``."""

    def __init__(self, document: EproDocument, meta: dict[str, Any] | None = None) -> None:
        self.document = document
        merged = dict(DEFAULT_PROJECT_META)
        merged.update(meta or {})
        self.meta = merged

    @staticmethod
    def _epru_name(archive: zipfile.ZipFile) -> str:
        for name in archive.namelist():
            if name.lower().endswith(".epru"):
                return name
        raise ValueError("no .epru entry found in archive")

    @classmethod
    def read(cls, path: str | Path) -> "EproArchive":
        with zipfile.ZipFile(path, "r") as archive:
            meta: dict[str, Any] = {}
            if "project2.json" in archive.namelist():
                meta = json.loads(archive.read("project2.json").decode("utf-8"))
            epru_name = cls._epru_name(archive)
            with archive.open(epru_name) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8")
                document = EproDocument.parse(text)
        return cls(document=document, meta=meta)

    @classmethod
    def create(cls, title: str, document: EproDocument | None = None) -> "EproArchive":
        return cls(document=document or EproDocument(), meta={"title": title})

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        title = str(self.meta.get("title") or path.stem)
        meta = dict(self.meta)
        meta["title"] = title
        epru_name = f"{title}.epru"

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "project2.json",
                json.dumps(meta, indent=2, ensure_ascii=False),
            )
            archive.writestr(epru_name, self.document.dumps())
            # EasyEDA Pro keeps a thumbnail folder here; an empty entry is valid
            # and keeps the archive shaped like the ones the editor produces.
            archive.writestr("IMAGE/", "")
        return path
