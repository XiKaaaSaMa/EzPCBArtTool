"""Saving the working document to a draft file, and knowing when it changed.

A *draft* is the whole editing session: the board outline, every artwork
placement, and which layers are switched on. It is deliberately not the
``.epro2`` export -- that is the finished board for the editor, this is the
work in progress so the next launch can carry on where the last one stopped.

Written as JSON, for three reasons that all matter more than compactness here:

* it is diffable and readable when something goes wrong, which is how the
  hand-written files in ``tests/fixtures`` stay useful;
* it carries no format version to migrate -- the loader fills in whatever a
  field defaults to, so an older draft still opens;
* it never silently mangles a path with Chinese characters in it, which is the
  norm for this tool's users and a recurring source of bugs in binary formats.

**Dirty tracking is a fingerprint, not a flag.** A flag has to be set by every
edit path, and there are dozens -- dragging a picture, moving a slider, toggling
a layer, resizing the board from the panel -- so a missed one means the exit
prompt never appears and work is lost exactly when it mattered. Comparing a
fingerprint of the document against the one taken at the last save cannot miss
an edit, because it does not depend on anyone remembering to announce it.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from .artboard import (
    ArtBoard,
    BoardOutlineSpec,
    CutoutElement,
    ImagePlacement,
    PlacementGroup,
    TextElement,
)
from .design_layers import LAYER_KEYS, LayerStack

__all__ = [
    "DRAFT_SUFFIX",
    "DRAFT_FILTER",
    "DraftInfo",
    "DraftStore",
    "board_to_dict",
    "board_from_dict",
    "apply_board_dict",
    "fingerprint",
    "save_draft",
    "load_draft",
    "default_draft_path",
]


DRAFT_SUFFIX = ".pcbdraft"
DRAFT_FILTER = "PCB 艺术板草稿 (*.pcbdraft);;所有文件 (*)"

_FORMAT = "pcb-art-board-draft"
_VERSION = 1


def default_draft_path() -> Path:
    """Where an unnamed draft is saved.

    Follows :func:`pcbart.paths.drafts_dir`: ``out/drafts`` from source,
    ``%LOCALAPPDATA%`` when packaged. Somewhere writable, and somewhere an
    upgrade does not replace.
    """
    from .paths import drafts_dir

    return drafts_dir() / f"未命名草稿{DRAFT_SUFFIX}"


def board_to_dict(board: ArtBoard) -> dict:
    """The whole document as plain JSON-ready data.

    Built from the dataclass fields rather than by listing them by hand, so a
    field added to the model is saved without anyone remembering to come back
    here. The two nested objects and the layer stack are the only things that
    need naming explicitly.
    """
    return {
        "format": _FORMAT,
        "version": _VERSION,
        "outline": _dataclass_dict(board.outline),
        "placements": [_dataclass_dict(item) for item in board.placements],
        "texts": [_dataclass_dict(item) for item in board.texts],
        "cutouts": [_dataclass_dict(item) for item in board.cutouts],
        "groups": [_dataclass_dict(item) for item in board.groups],
        "operator_name": board.operator_name,
        "operator_number": board.operator_number,
        "operator_page": getattr(board, "operator_page", ""),
        # The per-element overrides for the card's artwork. A plain dict, so
        # it serialises as-is; a board that never touched them writes `{}` and
        # reads back as automatic.
        "art_choices": dict(getattr(board, "art_choices", None) or {}),
        "outline_ready": bool(board.outline_ready),
        "layers": {
            "current": board.layers.current,
            "visible": {
                key: board.layers.is_visible(key) for key in LAYER_KEYS
            },
        },
    }


def _dataclass_dict(instance) -> dict:
    """One dataclass as a dictionary, with paths kept as strings."""
    result = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        result[field.name] = value
    return result


def board_from_dict(data: dict) -> ArtBoard:
    """Rebuild a document from :func:`board_to_dict` output.

    Unknown keys are ignored and missing ones keep their defaults, so a draft
    written by an older build still opens instead of failing on a field that
    did not exist yet. A file that is not a draft at all raises, rather than
    producing an empty board that looks like the user's work vanished.
    """
    if not isinstance(data, dict) or data.get("format") != _FORMAT:
        raise ValueError("不是 PCB 艺术板草稿文件")

    outline = _build(BoardOutlineSpec, data.get("outline") or {})
    board = ArtBoard(
        outline=outline,
        outline_ready=bool(data.get("outline_ready")),
        operator_name=str(data.get("operator_name") or ""),
        operator_number=str(data.get("operator_number") or ""),
        operator_page=str(data.get("operator_page") or ""),
        art_choices=dict(data.get("art_choices") or {}),
        groups=[_build(PlacementGroup, item) for item in data.get("groups") or []],
    )
    for item in data.get("placements") or []:
        board.placements.append(_build(ImagePlacement, item))
    for item in data.get("texts") or []:
        board.texts.append(_build(TextElement, item))
    for item in data.get("cutouts") or []:
        board.cutouts.append(_build(CutoutElement, item))

    layers = data.get("layers") or {}
    current = layers.get("current")
    visible = layers.get("visible") or {}
    board.layers = LayerStack(
        current=current if current in LAYER_KEYS else LAYER_KEYS[0],
        visible={
            key: bool(visible.get(key, True)) for key in LAYER_KEYS
        },
    )
    return board


def _build(cls, values: dict):
    """Construct ``cls`` from ``values``, dropping anything it does not take."""
    if not isinstance(values, dict):
        values = {}
    known = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in known})


def _restore_list(live: list, restored: list, cls) -> None:
    """Copy ``restored`` into ``live`` in place, reusing the existing objects.

    Matched by position and overwritten field by field, for the reason the
    placements were already handled this way: a panel or the canvas may be
    holding one of these objects, and swapping in fresh ones would leave that
    reference pointing at something no longer in the document -- so the next
    edit would change an object nothing draws.

    Items past the end of the snapshot are dropped, and short snapshots append.
    """
    for index, item in enumerate(restored):
        if index < len(live):
            target = live[index]
            for field_ in fields(cls):
                setattr(target, field_.name, getattr(item, field_.name))
        else:
            live.append(item)
    del live[len(restored) :]


def apply_board_dict(board: ArtBoard, data: dict, *, include_layers: bool = True) -> None:
    """Write a :func:`board_to_dict` snapshot back into a *live* document.

    Undo needs the restored state to land in the objects everything else is
    already holding. Building a new :class:`ArtBoard` -- what
    :func:`board_from_dict` does -- would leave the canvas, the layer panel and
    the board-setup panel pointing at the document that was just replaced, so
    this mutates in place instead.

    Placements are matched by *position*, reusing the existing objects and
    overwriting their fields. Reusing them is what keeps a stale reference --
    a panel's current placement, the canvas selection -- pointing at a real
    object rather than one no longer on the board; the value it now carries is
    the restored one. Objects past the end of the snapshot are dropped.

    ``include_layers=False`` leaves the layer stack alone. Undo uses it because
    which layers are shown is view state, not work: reverting an object's
    position should not also switch a layer back on.

    The board's outline caches are left alone deliberately: they are keyed on
    the geometry they were computed from, so restored geometry misses the cache
    and is recomputed, while untouched geometry keeps its cached trace.
    """
    restored = board_from_dict(data)

    for field_ in fields(BoardOutlineSpec):
        setattr(board.outline, field_.name, getattr(restored.outline, field_.name))

    _restore_list(board.placements, restored.placements, ImagePlacement)
    _restore_list(board.texts, restored.texts, TextElement)
    _restore_list(board.cutouts, restored.cutouts, CutoutElement)

    # ⚠️ Groups are restored by **matching id**, not by position, unlike the
    # lists above. Those are matched positionally because a panel may be
    # holding one of the objects; a group holds only ids, so the thing that has
    # to stay stable is the membership, not the object identity. Restoring
    # positionally would also fight the ids themselves: ``_restore_list``
    # overwrites every field including ``element_id``, so a reordered group
    # list would re-point at different artwork.
    board.groups = [replace(group) for group in restored.groups]
    for index, group in enumerate(board.groups):
        group.members = list(restored.groups[index].members)

    board.operator_name = restored.operator_name
    board.operator_number = restored.operator_number
    board.operator_page = restored.operator_page
    # Replaced rather than merged: applying a snapshot has to put the document
    # back exactly, including the choices the user had *not* made.
    board.art_choices = dict(restored.art_choices)
    board.outline_ready = restored.outline_ready
    if include_layers:
        # Mutated rather than swapped for the same reason the placements are:
        # the image inspector holds a reference to the stack.
        board.layers.current = restored.layers.current
        for key in LAYER_KEYS:
            board.layers.visible[key] = restored.layers.is_visible(key)


def fingerprint(board: ArtBoard) -> str:
    """A stable hash of everything the draft file stores.

    Taken from the saved representation, so it answers exactly the question the
    exit prompt asks: *would saving now write a different file?* Computing it
    from anything else would let the two drift -- a field that saves but does
    not count, or counts but does not save.
    """
    import hashlib

    payload = json.dumps(
        board_to_dict(board), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_draft(board: ArtBoard, path: str | Path) -> Path:
    """Write the document to ``path``, creating the directory if needed.

    Written through a temporary file and then moved into place: a crash or a
    full disk halfway through would otherwise leave a truncated draft, and a
    truncated draft is indistinguishable from a corrupted project to the person
    who just lost their work.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        board_to_dict(board), indent=2, ensure_ascii=False, sort_keys=True
    )
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_draft(path: str | Path) -> ArtBoard:
    """Read a draft file back into a document."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"找不到草稿文件：{source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"草稿文件已损坏，无法读取：{exc}") from None
    return board_from_dict(raw)


# --------------------------------------------------------------- the library
#
# Drafts live inside the application rather than only in files the user picks.
# The point is that reopening yesterday's work is a single click in the sidebar
# instead of remembering where it was saved, which is how "continue where I
# left off" actually gets used.


class DraftInfo:
    """One saved draft, as the sidebar needs to list it."""

    def __init__(self, path: Path, name: str, saved_at: float, bytes_: int) -> None:
        self.path = path
        self.name = name
        self.saved_at = saved_at
        self.bytes = bytes_

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DraftInfo({self.name!r}, {self.path})"

    @property
    def size_text(self) -> str:
        """A short size for the list, in the units a person reads."""
        if self.bytes < 1024:
            return f"{self.bytes} B"
        if self.bytes < 1024 * 1024:
            return f"{self.bytes / 1024:.0f} KB"
        return f"{self.bytes / (1024 * 1024):.1f} MB"


class DraftStore:
    """The drafts kept inside the application, newest first.

    A plain directory of ``.pcbdraft`` files, so it stays inspectable and a
    user can copy one out or drop one in. The ordering is by modification time
    rather than by name, because "the one I was working on" is what the list is
    for -- and each save touches the file, so a draft in use rises to the top
    on its own.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        """Where the library lives, created on first use."""
        root = self._root
        if root is None:
            from .paths import drafts_dir

            root = drafts_dir()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list(self) -> list[DraftInfo]:
        """Every readable draft, most recently saved first.

        A file that cannot be read at all is skipped rather than listed: a row
        that opens nothing is worse than a row that is not there.
        """
        found: list[DraftInfo] = []
        for path in self.root.glob(f"*{DRAFT_SUFFIX}"):
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append(
                DraftInfo(
                    path=path,
                    name=path.stem,
                    saved_at=stat.st_mtime,
                    bytes_=stat.st_size,
                )
            )
        found.sort(key=lambda item: item.saved_at, reverse=True)
        return found

    def path_for(self, name: str) -> Path:
        """Where a draft called ``name`` lives.

        The name is reduced to a file name: a draft called ``../x`` must not be
        able to write outside the library, and the user's own choice of name
        should not be able to either.
        """
        cleaned = _safe_name(name)
        return self.root / f"{cleaned}{DRAFT_SUFFIX}"

    def save(self, board: ArtBoard, name: str) -> Path:
        """Save under ``name``, replacing a draft already called that."""
        return save_draft(board, self.path_for(name))

    def load(self, path: str | Path) -> ArtBoard:
        return load_draft(path)

    def delete(self, path: str | Path) -> bool:
        """Remove a draft. Returns whether it was there."""
        target = Path(path)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True

    def unique_name(self, base: str = "未命名草稿") -> str:
        """A name not already taken, by appending a number."""
        existing = {info.name for info in self.list()}
        candidate = _safe_name(base)
        if candidate not in existing:
            return candidate
        index = 2
        while f"{candidate} {index}" in existing:
            index += 1
        return f"{candidate} {index}"


#: Characters Windows forbids in a file name, plus the path separators that
#: would let a name escape the library.
_UNSAFE = frozenset('<>:"/\\|?*')


def _safe_name(name: str) -> str:
    """A draft name reduced to something safe to use as a file name."""
    cleaned = "".join(
        "_" if character in _UNSAFE or ord(character) < 32 else character
        for character in str(name)
    ).strip().rstrip(".")
    return cleaned or "未命名草稿"
