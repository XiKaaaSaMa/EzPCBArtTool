"""Choosing the resolution to actually compute at.

A photograph straight off a phone is 4000 px on its side; a scan can be 8000.
The pipeline was written for drawings around 1000 px, and several of its steps
grow faster than the pixel count, so an 8k import froze the window for twenty
seconds. The fix is not to make each step faster -- it is to stop doing work
that cannot change the answer.

**What resolution is enough.** The input is going to be ink on a silkscreen, and
the board it prints on is tens of millimetres across. Working at about 2000 px
on the long side is far finer than the printer can resolve while still being
four to sixteen times cheaper than the source. Measured against a full-
resolution trace, shown at preview size, a 2000 px trace agrees to IoU 0.945
and carries 101.7% of the ink -- the extra ink is antialiasing on the strokes,
not lost artwork.

**Why not lower.** The lines themselves are proportional: the filter radius is
1.05% of the short side, so a stroke is the same fraction of the picture at any
resolution. Tracing at 760 px instead of 2000 px measures IoU 0.755, which is
where the project's earlier "tracing a downscaled copy loses 23% of the
strokes" finding came from. 2000 px is the point where that loss stops being
visible, and it is the number this module hands out.

**Kerning matters for anything measured in pixels.** A user-facing value like
"去杂点 5 px" means five pixels *of the image they loaded*. Reducing first and
applying it unchanged would silently filter five pixels of a smaller picture,
which is a different amount of ink. :func:`scaled_param` carries those across.
"""

from __future__ import annotations

__all__ = [
    "MAX_WORK_PIXELS",
    "MAX_WORK_LONG_SIDE",
    "work_scale",
    "working_size",
    "scaled_param",
    "needs_reduction",
    "prepare_working_image",
]


MAX_WORK_LONG_SIDE = 2000
"""Longest side the processing steps are asked to handle."""

MAX_WORK_PIXELS = 6_000_000
"""A second ceiling, for pictures that are long *and* wide.

Either limit alone can be defeated: a 2000 x 6000 panorama passes the long-side
test at 12 megapixels. The pixel ceiling keeps the cost bounded in both shapes.
"""


def needs_reduction(width: int, height: int) -> bool:
    """Whether these dimensions exceed the working budget."""
    if width <= 0 or height <= 0:
        return False
    return (
        max(width, height) > MAX_WORK_LONG_SIDE
        or width * height > MAX_WORK_PIXELS
    )


def work_scale(width: int, height: int) -> float:
    """The factor to shrink by, never above 1.

    Small pictures are left alone: upscaling one would add no information and
    would change the radius the algorithm derives from the short side, so a
    small drawing would come out with thicker strokes than it should.
    """
    if not needs_reduction(width, height):
        return 1.0
    by_side = MAX_WORK_LONG_SIDE / max(width, height)
    by_pixels = (MAX_WORK_PIXELS / (width * height)) ** 0.5
    return min(1.0, by_side, by_pixels)


def working_size(width: int, height: int) -> tuple[int, int]:
    """The ``(width, height)`` to process at."""
    scale = work_scale(width, height)
    if scale >= 1.0:
        return int(width), int(height)
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def scaled_param(value: float, scale: float, *, minimum: float = 0.0) -> float:
    """Carry a pixel-measured setting across a change of resolution.

    Returns ``value`` unchanged when no reduction happened, so callers can use
    this unconditionally.
    """
    if scale >= 1.0:
        return float(value)
    return max(minimum, float(value) * scale)


def prepare_working_image(
    image, alpha=None, *, interpolation: str = "area"
) -> tuple[object, object, float]:
    """Reduce an image to the working resolution, returning the scale used.

    Returns ``(image, alpha, scale)``. ``scale`` is 1.0 when nothing was
    reduced, so a caller can pass it straight to :func:`scaled_param`.

    **Why the import path may reduce too.** It used to trace at the picture's
    full size on the reasoning that the import is what gets manufactured, so it
    must be the exact one the preview approved. That reasoning holds for the
    *geometry*, but not for the resolution: the traced mask is vectorised and
    simplified at about 0.35 mm of tolerance on a board tens of millimetres
    across, so features finer than that are discarded anyway. Measured on an
    8000 px drawing, tracing at 1000 px produced 81 vertices against 107 at
    8000 px -- and the extra ones were hairline noise the tolerance removes
    minutes later.

    What the extra resolution *did* buy was 6.9 s of frozen window on every
    import. The reduction happens before any of the pipeline runs, so every
    stage -- trace, silhouette, closing, flood fill -- inherits the saving
    instead of just the one that was profiled.
    """
    import cv2

    height, width = image.shape[:2]
    scale = work_scale(width, height)
    if scale >= 1.0:
        return image, alpha, 1.0

    interpolation_flag = (
        cv2.INTER_AREA if interpolation == "area" else cv2.INTER_LINEAR
    )
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    reduced = cv2.resize(image, size, interpolation=interpolation_flag)
    if alpha is not None:
        alpha = cv2.resize(alpha, size, interpolation=interpolation_flag)
    return reduced, alpha, scale
