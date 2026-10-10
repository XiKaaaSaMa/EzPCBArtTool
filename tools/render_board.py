#!/usr/bin/env python
"""Render PCB primitives to a PNG using raw file coordinates, with no Y flip.

The point is to see the artwork the way the file stores it. If the result looks
upside down relative to the reference art, then the file's Y axis points the
other way than assumed, and that settles the coordinate convention by direct
observation rather than inference.

Usage::

    python tools/render_board.py <file.epro2> --out out/render.png
    python tools/render_board.py <file.epro2> --out flipped.png --flip-y
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from pcbart.epru import EproArchive, EproDocument  # noqa: E402

UNIT_MM = 0.0254


def decode_path(node) -> list[list[tuple[float, float]]]:
    """Decode a path into rings of raw-file-coordinate points.

    Delegates to :func:`pcbart.geometry.densify_path`, which is the single
    decoder for ``L`` / ``ARC`` / ``C`` / ``R`` / ``CIRCLE``.  A second decoder
    here had already drifted twice -- rectangle height direction and cubic
    Bezier controls -- and both mistakes showed up as misplaced template art.
    """
    from pcbart.geometry import densify_path

    return densify_path(node)


def shape_ring(shape, cx: float, cy: float, segments: int = 64):
    """Approximate a pad or drill shape descriptor as a ring of points.

    Accepts the object form EasyEDA Pro writes (``{"padType"/"holeType",
    "width", "height"}``) and the positional array the specification documents.
    """
    if isinstance(shape, dict):
        kind = str(shape.get("padType") or shape.get("holeType") or "")
        width, height = shape.get("width"), shape.get("height")
    elif isinstance(shape, list) and len(shape) >= 3:
        kind, width, height = str(shape[0]), shape[1], shape[2]
    else:
        return []
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return []
    if width <= 0 or height <= 0:
        return []

    rx, ry = width / 2.0, height / 2.0
    if kind.upper() == "RECT":
        return [
            (cx - rx, cy - ry),
            (cx + rx, cy - ry),
            (cx + rx, cy + ry),
            (cx - rx, cy + ry),
        ]
    return [
        (
            cx + rx * math.cos(2 * math.pi * i / segments),
            cy + ry * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/render.png"))
    parser.add_argument("--flip-y", action="store_true", help="negate Y before drawing")
    parser.add_argument("--scale", type=float, default=6.0, help="pixels per mm")
    parser.add_argument("--layers", default="1,2,3,4,11,12",
                        help="comma-separated layer ids to draw")
    parser.add_argument("--section", type=int, default=0,
                        help="which PCB document to render (0 = first)")
    args = parser.parse_args()

    doc = (
        EproArchive.read(args.source).document
        if args.source.suffix.lower() == ".epro2"
        else EproDocument.read(args.source)
    )
    wanted = {int(v) for v in args.layers.split(",") if v.strip()}

    # A project can hold several PCB documents that share one coordinate space.
    # Rendering them together superimposes unrelated boards, so keep only the
    # requested section.
    if args.section > 0:
        seen = -1
        kept = []
        active = False
        for record in doc.records:
            if record.type == "DOCHEAD" and isinstance(record.body, dict):
                if record.body.get("docType") == "PCB":
                    seen += 1
                active = (record.body.get("docType") == "PCB" and seen == args.section)
                kept.append(record)
                continue
            if active or record.type == "DOCHEAD":
                kept.append(record)
        doc = EproDocument(kept)

    # Collect geometry first so the canvas can be sized to fit everything.
    shapes: list[tuple[int, list[list[tuple[float, float]]], str]] = []
    drills: list[list[tuple[float, float]]] = []
    for record in doc.records:
        body = record.body
        if not isinstance(body, dict):
            continue
        layer = body.get("layerId")
        if layer not in wanted:
            continue
        if record.type in ("POLY", "FILL", "REGION", "POUR", "IMAGE"):
            # All rings of one record stay together. A filled region's path is
            # the outline followed by its holes -- splitting them here made the
            # renderer paint each hole solid, so it could never show whether a
            # cutout worked, which is exactly what these images are for.
            rings = decode_path(body.get("path"))
            if rings:
                shapes.append((int(layer), rings, record.type))
        elif record.type in ("LINE", "ARC"):
            p1 = (float(body.get("startX", 0)), float(body.get("startY", 0)))
            p2 = (float(body.get("endX", 0)), float(body.get("endY", 0)))
            shapes.append((int(layer), [[p1, p2]], record.type))
        elif record.type == "PAD":
            # A drilled pad is how a round cutout is expressed, so draw both the
            # copper shape and the drill. Without this the renderer silently
            # showed boards whose holes were never encoded.
            cx = float(body.get("centerX", 0.0))
            cy = float(body.get("centerY", 0.0))
            pad_ring = shape_ring(body.get("defaultPad"), cx, cy)
            if pad_ring:
                shapes.append((int(layer), [pad_ring], record.type))
            drill_ring = shape_ring(body.get("hole"), cx, cy)
            if drill_ring:
                drills.append(drill_ring)

    if not shapes:
        print("nothing to draw", file=sys.stderr)
        return 1

    ys_all = [p[1] for _, rings, _ in shapes for r in rings for p in r]
    xs_all = [p[0] for _, rings, _ in shapes for r in rings for p in r]
    y_lo, y_hi = min(ys_all), max(ys_all)
    x_lo, x_hi = min(xs_all), max(xs_all)

    span_x = (x_hi - x_lo) * UNIT_MM
    span_y = (y_hi - y_lo) * UNIT_MM
    scale = args.scale
    # Padding has to account for stroke width, which is drawn in pixels and
    # grows with the scale. A fixed 20 px margin let the board outline itself
    # fall outside the canvas at low scales, cutting off part of the picture.
    drawn_widths = [
        shape[3] * UNIT_MM * scale for shape in shapes if len(shape) > 3 and shape[3]
    ]
    pad = max(20, int(max(drawn_widths + [1.0])) + 12)
    width = max(64, int(span_x * scale) + pad * 2)
    height = max(64, int(span_y * scale) + pad * 2)

    canvas = np.full((height, width, 3), 255, np.uint8)

    layer_colors = {
        1: (0, 0, 200),
        2: (200, 0, 0),
        3: (0, 160, 200),
        4: (120, 160, 0),
        11: (255, 0, 255),
        12: (128, 128, 128),
    }

    def to_px(point: tuple[float, float]) -> tuple[int, int]:
        x_mm = (point[0] - x_lo) * UNIT_MM
        y_mm = (point[1] - y_lo) * UNIT_MM
        col = int(x_mm * scale) + pad
        # Row index grows downward. Without --flip-y the raw file Y is mapped
        # straight onto rows, so what you see is the file's own ordering.
        row = int(y_mm * scale) + pad
        if args.flip_y:
            row = height - 1 - row
        return (col, row)

    for layer, rings, kind in shapes:
        color = layer_colors.get(layer, (0, 0, 0))
        if kind in ("FILL", "REGION", "POUR", "IMAGE") and len(rings) > 1:
            # A filled region's rings are one shape, not several: the first is
            # the outline and the rest are its holes. Filling each ring on its
            # own painted the holes solid, so the renderer could never show
            # whether a cutout worked -- and a hole is exactly what this
            # picture is needed to verify.
            #
            # Even-odd filling cuts the interior rings regardless of their
            # winding, which is deliberate: the point is to see the *shape*
            # that was written, leaving the winding question to the editor.
            polys = [np.array([to_px(p) for p in ring], np.int32) for ring in rings]
            polys = [p for p in polys if len(p) >= 3]
            if polys:
                cv2.fillPoly(canvas, polys, color)
            continue

        for ring in rings:
            pts = np.array([to_px(p) for p in ring], np.int32)
            if len(pts) >= 3 and kind in ("FILL", "REGION", "POUR", "IMAGE"):
                cv2.fillPoly(canvas, [pts], color)
            elif len(pts) >= 3 and kind == "PAD":
                cv2.fillPoly(canvas, [pts], color)
            else:
                cv2.polylines(canvas, [pts], len(pts) > 2, color, 1)

    # Drills are punched out after everything else, so a hole always shows as
    # background even when the copper shape covers the same area.
    for ring in drills:
        pts = np.array([to_px(p) for p in ring], np.int32)
        cv2.fillPoly(canvas, [pts], (255, 255, 255))
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # imwrite_unicode, not cv2.imwrite: the latter cannot express a non-ASCII
    # path on Windows and fails by *returning False*, so a Chinese output name
    # silently wrote nothing.
    from pcbart.image_prep import imwrite_unicode

    imwrite_unicode(args.out, canvas)

    print(f"file     : {args.source.name}")
    print(f"raw Y    : [{y_lo:.1f}, {y_hi:.1f}] units  = [{y_lo * UNIT_MM:.2f}, {y_hi * UNIT_MM:.2f}] mm")
    print(f"raw X    : [{x_lo:.1f}, {x_hi:.1f}] units  = [{x_lo * UNIT_MM:.2f}, {x_hi * UNIT_MM:.2f}] mm")
    print(f"size     : {span_x:.2f} x {span_y:.2f} mm")
    print(f"drawn    : {len(shapes)} shapes on layers {sorted(wanted)}")
    if drills:
        print(f"drills   : {len(drills)} hole(s) punched")
    print(f"flip-y   : {args.flip_y}")
    print(f"wrote    : {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
