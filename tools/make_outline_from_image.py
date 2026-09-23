#!/usr/bin/env python
"""Generate a board outline from an image (the M2 pipeline).

    image -> denoise -> binarize -> trace contours -> simplify
          -> scale to mm -> process compensation -> board outline

Usage::

    # A 60 mm wide board, plus a preview of the mask
    python tools/make_outline_from_image.py art.png --width 60

    # Synthesise a test image and run it, for a quick self-check
    python tools/make_outline_from_image.py --test star --width 60 --preview

The preview PNG writes the binary mask next to the output, which is the fastest
way to see whether the threshold picked up the right thing before blaming the
geometry stage.

Board cutouts are out of scope: the outline is a single closed edge. Cutouts are
placed in the editor, where the geometry can be judged as it is drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.generator import generate_from_image  # noqa: E402
from pcbart.image_prep import PrepParams, make_test_image, mask_to_preview, preprocess  # noqa: E402
from pcbart.outline import OutlineParams  # noqa: E402
from pcbart.paths import default_output_dir  # noqa: E402
from pcbart.validation import check_outline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", type=Path, help="source image")
    parser.add_argument("--test", choices=["blob", "star", "ring", "letter"],
                        help="synthesise a test image instead of reading one")
    parser.add_argument("--outdir", type=Path, default=default_output_dir())
    parser.add_argument("--name", default="m2_outline")
    parser.add_argument("--width", type=float, help="target board width in mm")
    parser.add_argument("--height", type=float, help="target board height in mm")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="process compensation in mm (positive grows the outline)")
    parser.add_argument("--simplify", type=float, default=1.5,
                        help="Douglas-Peucker tolerance in pixels")
    parser.add_argument("--line-width", type=float, default=0.125)
    parser.add_argument("--threshold", choices=["otsu", "adaptive"], default="otsu")
    parser.add_argument("--blur", type=int, default=1, help="median blur radius")
    parser.add_argument("--open", dest="morph_open", type=int, default=1)
    parser.add_argument("--close", dest="morph_close", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=64,
                        help="drop connected components smaller than this (px)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="noise level for --test images")
    parser.add_argument("--preview", action="store_true", help="write the binary mask as PNG")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.image and not args.test:
        print("需要提供图片路径，或使用 --test 合成测试图", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.test:
        image = make_test_image(shape=args.test, noise=args.noise)
        image_path: Path | None = None
        print(f"test image: {args.test} {image.shape[1]}x{image.shape[0]}"
              + (f", noise={args.noise}" if args.noise else ""))
    else:
        image = args.image
        image_path = args.image
        print(f"image: {image_path}")

    prep_params = PrepParams(
        threshold_mode=args.threshold,
        blur_radius=args.blur,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_component_area=args.min_area,
    )

    prep = preprocess(image, prep_params)
    print(f"prep -> {prep.summary()}")

    if args.preview:
        from pcbart.image_prep import imwrite_unicode

        preview_path = args.outdir / f"{args.name}_mask.png"
        # imwrite_unicode: cv2.imwrite cannot open a non-ASCII path on Windows.
        imwrite_unicode(preview_path, mask_to_preview(prep.mask))
        print(f"mask preview -> {preview_path}")

    if not args.width and not args.height:
        print("未指定 --width/--height，将按 100 mm 宽输出（可用 --width 指定）")

    outline_params = OutlineParams(
        simplify_tolerance_px=args.simplify,
        target_width_mm=args.width,
        target_height_mm=args.height,
        offset_mm=args.offset,
        center=True,
    )

    output = args.outdir / f"{args.name}.epro2"
    try:
        report = generate_from_image(
            image_path if image_path is not None else image,
            output,
            title=args.name,
            prep_params=prep_params,
            outline_params=outline_params,
            line_width_mm=args.line_width,
        )
    except ValueError as exc:
        # The message already says what to change.
        print()
        print(f"生成失败: {exc}", file=sys.stderr)
        return 1

    print()
    print("outline")
    print(f"  points       : {report['points_before']} -> {report['points_after']}")
    print(f"  size         : {report['width_mm']:.3f} x {report['height_mm']:.3f} mm")

    # Re-read the result and validate it the same way the generator does.
    from pcbart.outline import extract_outline_rings

    rings = extract_outline_rings(prep.mask, outline_params)
    validation = check_outline(rings.outer)
    print()
    print("validation")
    for line in validation.summary().splitlines():
        print(f"  {line}")

    print()
    print(f"wrote {Path(report['output']).resolve()}")
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
