"""Turn a pixel staircase into arcs the editor draws as smooth curves.

Why this exists
---------------
``findContours`` walks the *pixel boundary*, so a soft edge arrives as a
staircase: measured on a 256 px icon, the outer ring had **1018 points and only
4 genuine corners**. The rest were the 45°/90° alternation of the pixel grid.

Two ways to make that look smooth, and only one of them works:

* **Move the vertices** (Douglas-Peucker). The tolerance must stay small enough
  to keep real corners, so the staircase survives along shallow curves -- which
  is exactly the ragged edge the user reported.
* **Fit curves through them.** Keep the shape, describe each run of steps as an
  arc, and let the renderer draw the curve. That is what this module does.

The result is written in the editor's own notation -- ``["ARC", sweep, x, y]``
between absolute points, measured from the template's own ``FILL`` records.

⚠️ **Corners survive on purpose.** A vertex where the boundary really turns is
kept as a corner; an arc fitted through it would round it off, and a letter's
serif or a star's point disappearing is worse than a slightly faceted curve.

Reading the staircase correctly
-------------------------------
The trap this module was rewritten around: judging a corner by the angle
between **consecutive** points marks *every step* of a staircase as a corner,
because each step turns 45° or 90°. Measured on a real ring, that kept the
vertex list almost unchanged (``[1, 2, 4, 5, 6, 7, ...]``) and left no interior
points to fit anything to, so every segment came out straight.

A staircase has to be judged over a **distance**, not between neighbours: a
point is a corner when the ring leaves a small disc around it in a
substantially different direction going out than coming in. Everything between
two such points is one run, and a run that bows away from its chord is an arc.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "fit_ring_arcs",
    "sweep_through",
    "find_corners",
]

#: How many vertices the corner test looks ahead and behind.
#:
#: A staircase alternates direction every one or two vertices, so a window of a
#: few steps is enough for that alternation to average out -- and short enough
#: that a genuine corner is still seen as one.
_CORNER_WINDOW = 4

#: How far a run may bow away from its own chord, as a fraction of the chord
#: length, and still count as straight. A run of steps wobbles by about one
#: pixel; a curve bows by much more.
_FLAT_RATIO = 0.035

#: Fewer points than this in a run and there is nothing to fit.
_MIN_RUN = 3


def sweep_through(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    """The signed sweep, in degrees, of the arc from ``a`` to ``c`` through ``b``.

    Positive is counter-clockwise, matching the editor. ``0.0`` when the three
    points are collinear, which the writer turns into a straight segment.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    twice_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(twice_area) < 1e-12:
        return 0.0

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) < 1e-12:
        return 0.0
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / denominator
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / denominator

    start = math.atan2(ay - uy, ax - ux)
    through = math.atan2(by - uy, bx - ux)
    end = math.atan2(cy - uy, cx - ux)

    ccw_to_end = (end - start) % (2 * math.pi)
    ccw_to_through = (through - start) % (2 * math.pi)
    if ccw_to_through <= ccw_to_end:
        sweep = ccw_to_end
    else:
        sweep = ccw_to_end - 2 * math.pi
    return math.degrees(sweep)


def find_corners(points: np.ndarray, radius: int) -> list[int]:
    """Indices where the ring genuinely turns, judged over ``radius`` vertices.

    Walking forward ``radius`` **vertices** gives a direction and walking back
    gives another; the point is a corner when those differ by more than 35°.
    A staircase turns 45° or 90° between neighbours but averages out over a
    handful of steps, so it stops looking like a corner.

    ⚠️ **The window is counted in vertices, not in arc length.** A staircase
    packs many vertices into a short run while a polygon has one per edge, so a
    length-based window collapsed a rectangle's four corners into two while
    still smoothing nothing on the dense ring. Vertex counts are the same
    currency on both.

    A ring with no corner at all -- a circle -- returns a single index, which
    the caller seeds for fitting.
    """
    count = len(points)
    if count < 4:
        return list(range(count))

    # ⚠️ **The window must shrink on a sparse ring.** A rectangle has four
    # vertices; a window of four makes every point compare against itself, so
    # every corner read as straight and the shape was fitted as two arcs.
    # A quarter of the ring is the most that can be asked for while still
    # comparing distinct points, and even that is too much below eight.
    window = max(1, min(int(radius), max(1, count // 4)))
    corners: list[int] = []
    for index in range(count):
        outgoing = points[(index + window) % count] - points[index]
        incoming = points[index] - points[(index - window) % count]
        lengths = np.linalg.norm(outgoing), np.linalg.norm(incoming)
        if lengths[0] < 1e-9 or lengths[1] < 1e-9:
            continue
        cosine = float(np.dot(outgoing, incoming) / (lengths[0] * lengths[1]))
        if cosine < math.cos(math.radians(35.0)):
            corners.append(index)

    if not corners:
        return [0]
    # ⚠️ **No thinning.** A polygon's corners are *adjacent* points -- a
    # rectangle has four vertices and four corners -- so merging neighbours
    # collapsed it to two and the fit then described a straight diagonal.
    # Density is what separates the two cases, and it is already handled by the
    # window: on a dense ring a corner spans a few vertices and the window makes
    # the ones inside it read as gradual, while a polygon's corners each stand
    # alone.
    return corners


def _seed_for_smooth(points: np.ndarray) -> list[int]:
    """Two vertices to split a corner-less ring at.

    A round shape has no corner to break it into runs, so the ring is cut at a
    vertex and the one farthest from it along the ring. Those two points, plus
    the points between them, are enough for :func:`sweep_through` -- which is
    all a circle needs to come back as two arcs.
    """
    count = len(points)
    if count < 3:
        return list(range(count))

    # Start at vertex 0, then take the vertex farthest from it. Scanning every
    # pair would be quadratic; for a closed ring the farthest point from any
    # vertex is a good enough cut, and the two halves are then fitted
    # independently.
    distances = np.linalg.norm(points - points[0], axis=1)
    far = int(np.argmax(distances))
    if far == 0:
        return [0]
    return sorted((0, far))


def fit_ring_arcs(
    ring: list[tuple[float, float]], *, closed: bool = True
) -> tuple[list[tuple[float, float]], list[float]]:
    """``(points, sweeps)`` for a ring, ready for :func:`arc_ring_to_path`.

    The ring is reduced to its genuine corners, and each surviving run gets a
    sweep: ``0.0`` for a straight run, the fitted arc otherwise.
    """
    points = np.asarray(ring, dtype=float)
    if len(points) < 4:
        return list(ring), [0.0] * len(ring)

    span = points.max(axis=0) - points.min(axis=0)
    if float(np.linalg.norm(span)) < 1e-9:
        return list(ring), [0.0] * len(ring)

    # A handful of vertices, which is how many it takes a staircase's steps to
    # average out.
    kept = find_corners(points, _CORNER_WINDOW)
    if len(kept) < 3:
        # ⚠️ **A shape with no corners still needs fitting.** A circle has none,
        # so ``find_corners`` returns a single index -- and bailing out here
        # meant a 200-point circle came back as 200 points and zero arcs, i.e.
        # the staircase kept. Two points plus a sweep describe a circle, so a
        # smooth shape is seeded with its two extreme vertices and fitted like
        # any other run.
        kept = _seed_for_smooth(points)
        if len(kept) < 2:
            return list(ring), [0.0] * len(ring)

    dense = points[kept]
    sweeps = [0.0] * len(dense)
    for index in range(len(dense)):
        start = dense[index]
        end = dense[(index + 1) % len(dense)]

        first = kept[index]
        last = kept[index + 1] if index + 1 < len(kept) else len(points) + kept[0]
        interior = [
            points[position % len(points)] for position in range(first + 1, last)
        ]
        if len(interior) < _MIN_RUN:
            continue

        chord = end - start
        chord_length = float(np.linalg.norm(chord))
        if chord_length < 1e-9:
            continue
        offsets = np.asarray(interior) - start
        cross = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
        distances = cross / chord_length
        furthest = int(np.argmax(distances))
        if float(distances[furthest]) <= _FLAT_RATIO * chord_length:
            continue  # straight: the bow is the pixel grid's wobble

        middle = tuple(float(v) for v in interior[furthest])
        sweeps[index] = sweep_through(
            (float(start[0]), float(start[1])),
            middle,
            (float(end[0]), float(end[1])),
        )

    return [(float(x), float(y)) for x, y in dense], sweeps
