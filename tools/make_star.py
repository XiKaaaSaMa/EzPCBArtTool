#!/usr/bin/env python
"""Generate a five-pointed star board outline with 30 mm edges.

Geometry
--------
A five-pointed star has ten vertices: five outer points and five inner
notches, alternating around the centre. Each of the ten edges spans 36 degrees
at the centre and connects one outer vertex to one inner vertex.

For every edge to measure exactly ``L``, the outer radius follows from the law
of cosines applied to that triangle::

    R = L / (2 * sin(pi / 5))            ->  L = 30 mm  =>  R = 25.5168 mm

and the inner radius is then fixed by requiring the two triangles meeting at
each inner notch to produce equal-length edges::

    r = R * cos(2*pi / 5) / cos(pi / 5)  ->  r = 9.8591 mm

With ``L = 30`` the overall star spans roughly 49.7 mm wide by 47.3 mm tall.

The outline is drawn with the polyline tool in free-angle mode
(``PREFERENCE.routingCorner = "L"``).

Usage::

    python tools/make_star.py --outdir out
    python tools/make_star.py --edge 30 --points 5 --rotation 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbart.generator import describe_outline, generate_polyline_board  # noqa: E402
from pcbart.paths import default_output_dir  # noqa: E402
from pcbart.shapes import (  # noqa: E402
    polygon_edge_lengths,
    regular_star_points,
    star_circumradius,
    star_inner_radius,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=5, help="number of star points")
    parser.add_argument("--edge", type=float, default=30.0, help="edge length in mm")
    parser.add_argument("--rotation", type=float, default=0.0, help="rotation in degrees")
    parser.add_argument("--width", type=float, default=0.125, help="outline line width in mm")
    parser.add_argument(
        "--outdir", type=Path, default=default_output_dir(), help="output directory"
    )
    parser.add_argument("--name", type=str, default="m2_star5_edge30", help="output base name")
    args = parser.parse_args()

    if args.points < 3:
        print("a star needs at least 3 points", file=sys.stderr)
        return 2
    if args.edge <= 0:
        print("edge length must be positive", file=sys.stderr)
        return 2

    outer_radius = star_circumradius(args.points, args.edge)
    inner_radius = star_inner_radius(args.points, outer_radius)

    points = regular_star_points(
        points=args.points,
        edge_length=args.edge,
        rotation_deg=args.rotation,
    )

    print("star geometry")
    print(f"  points        : {args.points}")
    print(f"  edge length   : {args.edge:.4f} mm (all {2 * args.points} edges)")
    print(f"  outer radius  : {outer_radius:.6f} mm")
    print(f"  inner radius  : {inner_radius:.6f} mm")
    print(f"  rotation      : {args.rotation:.3f} deg")

    lengths = polygon_edge_lengths(points, closed=True)
    print()
    print("edge check")
    for index, length in enumerate(lengths, start=1):
        print(f"  edge {index:>2}: {length:.9f} mm")
    spread = max(lengths) - min(lengths)
    print(f"  spread: {spread:.3e} mm")

    summary = describe_outline(points)
    print()
    print("outline summary")
    print(f"  vertices      : {summary['vertices']}")
    print(f"  perimeter     : {summary['perimeter_mm']:.4f} mm")
    print(f"  overall size  : {summary['width_mm']:.4f} x {summary['height_mm']:.4f} mm")

    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / f"{args.name}.epro2"

    # expected_edge_mm turns the equilateral requirement into a hard check:
    # generation fails rather than silently writing a lopsided star.
    generate_polyline_board(
        points_mm=points,
        output_path=output,
        title=args.name,
        line_width_mm=args.width,
        center=True,
        expected_edge_mm=args.edge,
        edge_tolerance_mm=1e-6,
        routing_corner="L",
    )

    print()
    print(f"wrote {output.resolve()}  ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
