#!/usr/bin/env python
"""Turn an illustration into a two-tone board painting and write it out.

    image -> subject -> tone split -> white contour -> .epro2 + preview

Usage::

    python tools/make_board_art.py "P:/PS抠图/由比滨透明背景.png" --outdir out

Writes three files into ``--outdir``:

``<name>.epro2``
    The board: outline plus the two filled layers.
``<name>_art.png``
    The two tones as flat black and white, at source resolution. This is what
    the board will actually look like, so it is the file to judge the style by.
``<name>_render.png``
    The written file read back and drawn by the independent renderer, which is
    the check that the geometry survived the trip through the file format.

The two are not redundant. ``_art.png`` shows the style; ``_render.png`` shows
what was really encoded, drawn by code that shares nothing with the writer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.boardart import (  # noqa: E402
    BoardArtParams,
    board_art,
    board_art_outline_mm,
    board_art_preview,
    board_art_records,
)
from pcbart.generator import generate_outline_board  # noqa: E402
from pcbart.image_prep import imread_unicode, imwrite_unicode  # noqa: E402
from pcbart.paths import default_output_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path, help="source illustration")
    parser.add_argument("--outdir", type=Path, default=default_output_dir())
    parser.add_argument("--name", default="board_art")
    parser.add_argument(
        "--size", type=float, default=76.2,
        help="longest side of the board, in mm (default 76.2 = 3 inch)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="tone split, 0..1 (default: calibrated 0.62)",
    )
    parser.add_argument(
        "--outline-px", type=int, default=0,
        help="contour width in pixels (default: derived from --outline-ratio)",
    )
    parser.add_argument(
        "--outline-mm", type=float, default=0.0,
        help="contour width in mm; converted using --size, overrides the ratio",
    )
    parser.add_argument("--min-island-px", type=int, default=None,
                        help="drop tone specks smaller than this many pixels")
    parser.add_argument(
        "--min-feature-mm", type=float, default=None,
        help="narrowest feature either tone may keep, in mm "
             "(default 0.10, a common 1 oz copper floor; 0 keeps everything)",
    )
    parser.add_argument("--cleanup-px", type=int, default=None,
                        help="morphological kernel for the silhouette edge")
    parser.add_argument("--no-black-background", action="store_true",
                        help="leave the background as bare board instead of filling it")
    parser.add_argument("--preview", action="store_true",
                        help="write the flat two-tone PNG (always written when --render)")
    parser.add_argument("--render", action="store_true",
                        help="also write the file back through tools/render_board.py")
    parser.add_argument("--render-scale", type=float, default=12.0,
                        help="pixels per mm for the re-render (default 12)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.image.exists():
        print(f"找不到图片: {args.image}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = imread_unicode(args.image, -1)  # IMREAD_UNCHANGED: keep alpha
    if raw is None:
        print(f"无法读取图片: {args.image}", file=sys.stderr)
        return 2

    height_px, width_px = raw.shape[:2]
    longest_px = max(height_px, width_px)
    width_mm = args.size * width_px / longest_px
    height_mm = args.size * height_px / longest_px
    px_per_mm = width_px / width_mm

    params = BoardArtParams()
    if args.threshold is not None:
        params.tone_threshold = args.threshold
    if args.min_island_px is not None:
        params.min_island_px = args.min_island_px
    if args.min_feature_mm is not None:
        params.min_feature_mm = args.min_feature_mm
    if args.cleanup_px is not None:
        params.cleanup_px = args.cleanup_px
    if args.no_black_background:
        params.black_background = False
    if args.outline_px:
        params.outline_px = args.outline_px
    elif args.outline_mm > 0:
        params.outline_px = max(1, round(args.outline_mm * width_px / width_mm))

    print(f"image  : {args.image}")
    print(f"  size : {width_px} x {height_px} px  ->  "
          f"{width_mm:.2f} x {height_mm:.2f} mm")
    print(f"  board: longest side {args.size:.2f} mm")

    art = board_art(raw, params=params, px_per_mm=px_per_mm)
    print(f"style  : {art.summary()}")
    print(f"  tone split   : {params.tone_threshold:.2f}")
    print(f"  min island   : {params.min_island_px} px")
    print(f"  min feature  : {params.min_feature_mm:.3f} mm "
          f"({art.min_feature_px} px at this size)")
    print(f"  background   : {'black (filled)' if params.black_background else 'bare board'}")

    art_path = args.outdir / f"{args.name}_art.png"
    imwrite_unicode(art_path, board_art_preview(art))
    print(f"  preview      : {art_path}")

    records = board_art_records(art, width_mm)
    by_layer: dict[int, int] = {}
    vertices = 0
    for layer_id, rings, _width in records:
        by_layer[layer_id] = by_layer.get(layer_id, 0) + 1
        vertices += sum(len(ring) for ring in rings)

    output = args.outdir / f"{args.name}.epro2"
    generate_outline_board(
        outer_mm=board_art_outline_mm(art, width_mm),
        output_path=output,
        title=args.name,
        fills=records,
    )

    print()
    print("wrote")
    print(f"  file     : {output.resolve()}")
    print(f"  size     : {output.stat().st_size / 1024:.1f} KB")
    print(f"  shapes   : {len(records)} filled, {vertices} vertices")
    for layer_id, count in sorted(by_layer.items()):
        what = {1: "顶层铜（黑）", 3: "顶层丝印层（白）"}.get(layer_id, f"layer {layer_id}")
        print(f"    layer {layer_id:>2}  {what}: {count} shape(s)")

    if args.render:
        import subprocess

        render_path = args.outdir / f"{args.name}_render.png"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("render_board.py")),
                str(output),
                "--out", str(render_path),
                "--layers", "11,3,1",
                "--scale", str(args.render_scale),
                "--flip-y",
            ],
            check=False,
        )
        print(f"  render   : {render_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
