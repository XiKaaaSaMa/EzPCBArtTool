"""Colour pictures, carried into the EDA project as ``OBJ`` records.

The problem this solves
-----------------------
Exporting a card as 彩色丝印 produced **plain single-colour silkscreen**: the
artwork arrived at the fab as traced outlines, not as a picture.

``tools.build_copper`` traces every placement into ``FILL`` polygons, which is
right for line art and board paintings -- those *are* two-tone geometry -- but
it throws the colour away. A colour print needs the pixels.

What the EDA wants
------------------
EasyEDA Pro distinguishes two kinds of picture-shaped record, and the file
format specification (``easyeda/easyeda-format-skill``) is explicit:

* ``IMAGE`` -- a *complex polygon*: ``path`` is a vertex list, so it carries
  shape but no pixels. It can only ever print in one colour.
* ``OBJ`` -- a *binary embedded object*: ``path`` is a Data URL,
  ``data:image/png;base64,...``, i.e. the actual picture. This is the record
  the colour-silkscreen export takes its pixels from.

So a placement that holds a colour picture is written as an ``OBJ``, and
``PCB_ManufactureData.getGerberFile(name, colorSilkscreen=True)`` in the EDA
turns it into the ``Fabrication_Colorful*`` files.

Coordinates
-----------
``OBJ`` places the picture by its **top-left corner** (``startX`` / ``startY``,
in mil, Y up) and rotates about that corner, whereas ``ImagePlacement`` stores
its **centre**. The conversion below is the only place that difference is
handled, so the export and the canvas cannot drift apart.

``y`` up is not an assumption: the template's own ``IMAGE`` records put the
vertical wordmark at ``startY = 1966 mil`` for a card that spans
``-1969 .. 1969``, i.e. ``startY`` is the top edge and larger values are
higher.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import uuid
from pathlib import Path

from .units import EPRU_UNIT_MM

__all__ = [
    "is_color_artwork",
    "color_object_record",
    "color_objects",
]

_MIL = EPRU_UNIT_MM
"""One ``.epru`` coordinate unit. The conversion lives in :mod:`pcbart.units`."""

#: The only layers an embedded object may live on.
#:
#: ⚠️ **Measured against the editor, and it is narrower than it looks.** The
#: API's own enum ``TPCB_LayersOfObject`` lists exactly ``3`` (top silk), ``4``
#: (bottom silk) and ``13`` (document). Requesting copper is *accepted and then
#: silently moved*: ``create(1, ...)`` returned ``layer: 1`` and reading the
#: object back gave ``layer: 3``. So a picture placed on copper does not stay on
#: copper -- it ends up printed on the silkscreen.
_OBJECT_LAYERS = ("top_silk", "bottom_silk")


def is_color_artwork(placement) -> bool:
    """Whether ``placement`` should be placed as a **picture**, not as geometry.

    True for a picture the user imported and left unprocessed, for **线稿**, and
    for the card's generated elements that live on a silkscreen layer (二维码 /
    条形码 / 职业图标). Those are transparent-background pictures -- black ink
    for the codes and the icon -- and tracing them turned each into a pile of
    filled regions, which is both a different object in the editor ("填充区域"
    rather than a picture) and a lossier rendering.

    ⚠️ **Line art is a picture, and that is the user's decision.** 「按理来说图片
    应该能保留更多细节，应当保留黑白图片」. A traced silhouette is simplified
    geometry: thin strokes are dropped, curves become chords, and every shape
    arrives in the editor as a filled region the user cannot adjust the way they
    can a picture. The pixels keep the drawing the user approved on the import
    preview. So a placement carrying 线稿's own output is embedded, not traced.
    See :data:`_TRACED_KINDS` for what still is.

    ⚠️ **The 阵营 logo is deliberately NOT in this set.** It is white line art
    that belongs on **顶层铜**, and an embedded object cannot stay there: see
    :data:`_OBJECT_LAYERS`. Asking for copper is silently answered with
    silkscreen, which is the one layer the user said must not be used. The logo
    therefore keeps the traced path, which is the only way to put it on copper.

    What **also keeps** the traced path:

    * a **板画** placement, whose two tones *are* geometry by construction --
      the whole style is "white ink here, copper there", and the two halves have
      to be traced together so their edges meet (see :mod:`pcbart.boardart`);
    * a picture the editor cannot read: a document from another machine names
      files that are not here, and tracing still writes the artwork's shape.
      Reporting "no picture" instead would silently drop the content.
    """
    if not placement.path:
        return False
    if getattr(placement, "tone", "") in ("white", "black"):
        return False
    if placement.layer_key not in _OBJECT_LAYERS:
        return False
    if placement.processed_path:
        # 线稿 publishes a finished black-and-white picture; 板画's tones were
        # already excluded above. Anything else carrying a processed file is a
        # two-tone result and stays geometry.
        if placement.kind != "lineart":
            return False
    return _readable(placement.processed_path or placement.path)


def _readable(path: str) -> bool:
    """Whether the picture can be opened. Cheap: the header only."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:  # noqa: BLE001 - unreadable means "trace it instead"
        return False


def _encoded_picture(path: str, window, *, ink_only: bool = False) -> str:
    """A Data URL for the picture, cropped to ``window`` when one is set.

    Cropping here rather than in the EDA keeps the framing promise: the canvas
    shows the part inside the window, so that part is what gets printed. The
    window is the same ``(x0, y0, x1, y1)`` fraction stored on the placement,
    so the canvas and the ODB both read one definition.

    Re-encoded as PNG rather than passed through as the original bytes: the
    OBJ record embeds the picture, and a JPEG or WebP would be re-compressed by
    the EDA on every load with no control over the result. PNG is lossless, and
    a colour print has to keep the colours the user chose.

    ``ink_only`` turns a **black-on-white** drawing into black ink on
    transparent paper. 线稿 is stored that way because the canvas needs it (see
    ``design_canvas._tinted``, which derives the ink from the grey level), but
    an embedded picture prints its own pixels -- so the white paper would come
    out as a solid block of ink over the card. The conversion therefore happens
    here, at the last moment, where the bytes are decided.
    """
    from PIL import Image

    image = Image.open(path)
    if window is not None:
        width, height = image.size
        x0, y0, x1, y1 = window
        left = max(0, min(width, int(round(min(x0, x1) * width))))
        right = max(0, min(width, int(round(max(x0, x1) * width))))
        top = max(0, min(height, int(round(min(y0, y1) * height))))
        bottom = max(0, min(height, int(round(max(y0, y1) * height))))
        if right - left >= 1 and bottom - top >= 1:
            image = image.crop((left, top, right, bottom))
    if ink_only:
        import numpy as np

        from .image_prep import ink_only as to_ink_only

        # After the crop, so the conversion only ever touches pixels that are
        # actually going to print.
        converted = to_ink_only(np.asarray(image.convert("L")))
        image = Image.fromarray(converted, mode="RGBA")
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _corner_offset(
    width_mm: float, height_mm: float, rotation_deg: float
) -> tuple[float, float]:
    """Where the top-left corner sits relative to the centre, after rotation.

    ``ImagePlacement`` rotates about the centre; ``OBJ`` rotates about its
    top-left. Putting the *rotated* corner at the right place and handing the
    same angle to the record makes the two agree, which is what lets a turned
    picture stay put.
    """
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Top-left in the picture's own frame, Y up: (-w/2, +h/2).
    lx, ly = -width_mm / 2.0, height_mm / 2.0
    return (lx * cos_a - ly * sin_a, lx * sin_a + ly * cos_a)


def _blob_id(data_url: str) -> str:
    """A stable 64-hex id for a picture, the way ``OBJ`` refers to its bytes.

    ⚠️ **Not a content hash**, and it does not need to be. Measured against a
    project the editor itself wrote: the ``resources`` entry's ``hash`` is
    neither the SHA-256 of the picture nor of its Data URL. It is an id the
    editor assigns, and all that matters is that the ``OBJ``'s
    ``"path":"blob:<id>"`` matches a ``BLOB`` record with the same ``id``.

    Derived from the data here rather than drawn at random so that exporting
    the same board twice produces the same file -- a diff stays readable.
    """
    return hashlib.sha256(data_url.encode("utf-8")).hexdigest()


def color_object_record(board, placement, *, document=None) -> dict:
    """One ``OBJ`` record plus the ``BLOB`` record holding its bytes.

    ⚠️ **The picture is not embedded in the ``OBJ``.** The format reference
    describes ``path`` as accepting a Data URL, but the editor **writes and
    reads only a blob reference**: a record created through the API with a
    ``data:`` URL comes back out as ``"path":"blob:<64 hex>"``, and a file whose
    ``OBJ`` carried an inline Data URL imported as an **empty rectangle** --
    the outline appeared with its file name and no picture inside it.

    So the export writes the pair the editor itself writes: a ``BLOB`` record
    (``{"filename":..., "content":"data:image/png;base64,..."}``) and an
    ``OBJ`` whose ``path`` is ``blob:<that record's id>``.

    Returns ``{"object": {...}, "blob": {...}}``.
    """
    from .artboard import drawn_size_mm

    # The pixels come from the same file the canvas framed, which for 线稿 is
    # the processed black-and-white result rather than the source photograph.
    # Reading the source here would embed the original colours while the window
    # was measured against the line art -- and the two are not even the same
    # size, because the line-art pass works at its own resolution.
    picture = placement.processed_path or placement.path

    left, top, right, bottom = placement.source_rect(*(_pixel_size(placement)))
    used_w = max(1, right - left)
    used_h = max(1, bottom - top)
    drawn_w, drawn_h = drawn_size_mm(
        placement.width_mm, placement.height_mm, used_w / used_h
    )

    dx, dy = _corner_offset(drawn_w, drawn_h, placement.rotation_deg)
    start_x = (placement.x_mm + dx) / _MIL
    start_y = (placement.y_mm + dy) / _MIL

    # 线稿 publishes black ink on white paper; anything else that gets embedded
    # already carries the alpha it should print with (the codes and the
    # profession plate are ink-on-transparent, a colour import is opaque).
    data_url = _encoded_picture(
        picture,
        placement.source_window,
        ink_only=getattr(placement, "kind", "") == "lineart",
    )
    blob_id = _blob_id(data_url)
    name = Path(picture).name or "artwork.png"

    body = {
        "partitionId": "",
        "groupId": 0,
        "layerId": _layer_id(placement),
        "fileName": name,
        "startX": start_x,
        "startY": start_y,
        "width": drawn_w / _MIL,
        "height": drawn_h / _MIL,
        # ``OBJ`` takes the same sign convention as the property panel:
        # counter-clockwise positive.
        "angle": placement.rotation_deg,
        "mirror": bool(placement.flip_horizontal),
        "path": f"blob:{blob_id}",
        "locked": False,
        "zIndex": -1,
    }
    return {
        "object": {
            "type": "OBJ",
            "body": body,
            "id": uuid.uuid4().hex[:16],
        },
        "blob": {
            "type": "BLOB",
            "body": {"filename": name, "content": data_url},
            "id": blob_id,
        },
    }


def _pixel_size(placement) -> tuple[int, int]:
    """The framed picture's pixel size, or ``(0, 0)`` when it cannot be read.

    ⚠️ **The processed file wins, and it is not just a preference.** A framing
    window is a fraction of the picture, so it only means something against the
    file it was measured on. 线稿's output has its own resolution, and reading
    the source here instead would map the same fractions onto different pixels
    -- cropping a different part of the drawing than the canvas showed.
    """
    from .image_size import UnreadablePicture, picture_size

    try:
        return picture_size(placement.processed_path or placement.path)
    except UnreadablePicture:
        # Reported by the caller's own checks; a size is not an error here.
        return (0, 0)


def _layer_id(placement) -> int:
    from .design_layers import layer

    return layer(placement.layer_key).easyeda_id


def color_objects(board, *, document=None) -> list[dict]:
    """Every colour picture on ``board``, as ``OBJ`` records.

    Order follows ``board.placements`` so the records land in the order the
    user created them; the EDA sorts by layer for display anyway.

    ⚠️ **A picture that cannot be read is skipped, not an error.** Documents
    written elsewhere -- and tests -- carry paths that do not resolve on this
    machine, and ``build_copper`` already reports those through its own
    ``problems`` list. Raising here would turn an unreadable file into a failed
    export of the whole board, which is a worse answer than the traced path the
    board would otherwise get.
    """
    found: list[dict] = []
    for placement in getattr(board, "placements", []):
        if not is_color_artwork(placement):
            continue
        try:
            found.append(color_object_record(board, placement, document=document))
        except Exception:  # noqa: BLE001 - reported by build_copper's own pass
            continue
    return found
