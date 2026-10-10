"""Manufacturability and geometry checks for board outlines.

Two kinds of rule live here:

* **Geometric** -- the outline must be a single simple closed ring with no
  self-intersections and no zero-length edges. EasyEDA Pro cannot turn a
  self-intersecting path into a board.
* **Process** -- the outline must be compatible with JLCPCB's stated
  capabilities, quoted in millimetres. Numbers come from
  https://jlcpcb.com/capabilities/pcb-capabilities

Violations are returned as structured issues rather than raised, so a caller
(a GUI, a CLI, a batch job) can decide how to present or override them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "Issue",
    "OutlineReport",
    "MIN_BOARD_SIDE_MM",
    "MIN_ROUTED_SLOT_MM",
    "MIN_TRACE_WIDTH_MM",
    "MIN_TRACE_SPACING_MM",
    "MIN_NECK_WIDTH_MM",
    "check_outline",
    "check_trace_width",
    "min_neck_width_mm",
    "raise_for_errors",
    "warning_sentence",
]

# JLCPCB capabilities, converted to millimetres.
MIN_BOARD_SIDE_MM = 3.0
"""Smallest board dimension FR4 accepts, 3 x 3 mm."""

MIN_ROUTED_SLOT_MM = 0.2
"""Narrowest routable edge/slot."""

MIN_TRACE_WIDTH_MM = 0.10
"""Minimum trace width, 1-2 layer 1 oz."""

MIN_TRACE_SPACING_MM = 0.10
"""Minimum trace spacing, 1-2 layer 1 oz."""

MIN_NECK_WIDTH_MM = 0.5
"""Narrowest board neck worth keeping.

An art board can easily trace into a thin waisted neck, and one thinner than
this snaps during routing or handling. This is a practical floor for the board
body, not a fabrication limit: JLCPCB's published minimum routed slot is 0.2 mm
(:data:`MIN_ROUTED_SLOT_MM`), which applies to a *gap*, not to a piece of board
that has to survive handling.
"""

MIN_VERTEX_SPACING_MM = 1e-6
"""Below this, two vertices are treated as coincident."""


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class OutlineReport:
    width_mm: float
    height_mm: float
    vertex_count: int
    edge_count: int
    perimeter_mm: float
    min_edge_mm: float
    issues: list[Issue]
    min_neck_mm: float | None = None
    """Width of the narrowest neck, or ``None`` if not measured.

    A large value means the shape has no bottleneck; see
    :func:`min_neck_width_mm`.
    """

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"outline: {self.width_mm:.4f} x {self.height_mm:.4f} mm, "
            f"{self.vertex_count} vertices, perimeter {self.perimeter_mm:.4f} mm",
            f"shortest edge: {self.min_edge_mm:.6f} mm",
        ]
        if self.min_neck_mm is not None:
            lines.append(f"narrowest neck: {self.min_neck_mm:.4f} mm")
        if not self.issues:
            lines.append("no issues")
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)


def check_outline(
    points_mm: Sequence[tuple[float, float]],
    min_side_mm: float = MIN_BOARD_SIDE_MM,
    min_edge_mm: float = MIN_ROUTED_SLOT_MM,
    min_neck_mm: float | None = MIN_NECK_WIDTH_MM,
) -> OutlineReport:
    """Validate a closed outline given as a ring of vertices.

    The ring is treated as implicitly closed: the last vertex connects back to
    the first. A repeated final vertex is tolerated.
    """
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity

    points = [(float(x), float(y)) for x, y in points_mm]
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]

    issues: list[Issue] = []

    if len(points) < 3:
        issues.append(Issue("error", "too-few-vertices", "板框至少需要 3 个顶点"))
        return OutlineReport(0, 0, len(points), 0, 0, 0, issues)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    edges: list[float] = []
    import math

    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        edges.append(math.hypot(x2 - x1, y2 - y1))
    perimeter = sum(edges)

    for index, length in enumerate(edges):
        if length < MIN_VERTEX_SPACING_MM:
            issues.append(
                Issue(
                    "error",
                    "zero-length-edge",
                    f"第 {index} 条边长度近似为零（{length:.9f} mm）",
                )
            )
            break

    if width < min_side_mm or height < min_side_mm:
        issues.append(
            Issue(
                "error",
                "board-too-small",
                f"板框 {width:.3f} x {height:.3f} mm 小于最小可加工尺寸 "
                f"{min_side_mm:.1f} x {min_side_mm:.1f} mm",
            )
        )

    if min(edges) < min_edge_mm:
        issues.append(
            Issue(
                "warning",
                "short-edge",
                f"最短边 {min(edges):.4f} mm 低于建议的最小锣边宽度 {min_edge_mm:.2f} mm",
            )
        )

    polygon = Polygon(points)
    measured_feature: float | None = None
    if not polygon.is_valid:
        issues.append(
            Issue("error", "invalid-polygon", f"轮廓无效：{explain_validity(polygon)}")
        )
    elif not polygon.exterior.is_simple:
        issues.append(Issue("error", "self-intersecting", "轮廓自交，无法作为板框"))
    elif polygon.area <= 0:
        issues.append(Issue("error", "zero-area", "轮廓面积为 0"))
    elif min_neck_mm is not None:
        measured_feature = min_neck_width_mm(points)
        if measured_feature < min_neck_mm:
            issues.append(
                Issue(
                    "warning",
                    "thin-neck",
                    f"最窄处约 {measured_feature:.3f} mm，低于建议最小值 "
                    f"{min_neck_mm:.2f} mm；细颈在加工和搬运时容易断裂",
                )
            )

    return OutlineReport(
        width_mm=width,
        height_mm=height,
        vertex_count=len(points),
        edge_count=len(edges),
        perimeter_mm=perimeter,
        min_edge_mm=min(edges),
        issues=issues,
        min_neck_mm=measured_feature,
    )


def min_neck_width_mm(ring: Sequence[tuple[float, float]], steps: int = 28) -> float:
    """Width of the narrowest neck in a board region, in millimetres.

    Eroding a shape by ``r`` pulls every boundary inward by ``r``. The shape can
    only survive that while ``r`` stays below half of its narrowest passage, so
    the first radius at which the shape **splits into separate pieces or
    disappears** is half the bottleneck width. Bisecting on ``r`` gives ``2r``.

    For a shape with no neck the erosion never splits, and the value returned is
    the widest inscribed passage instead. That over-estimate is deliberate: the
    question this answers is "is there anywhere thin enough to break?", and
    reporting a large number for a shape with no bottleneck means "no".

    Two plausible alternatives were tried and rejected:

    * **Opening area difference.** Erosion rounds convex corners, which sheds
      area at *every* radius, so a 10 mm square appears about 1.5 mm thick.
    * **Medial axis distance.** Twice the smallest distance-to-boundary along
      the medial axis is the textbook local thickness, but the medial axis of a
      star runs branches into its five concave tips where the width legitimately
      tapers to zero. Every star would be flagged as a sliver.

    Known gap: a thin protrusion sticking out of a thick body is not detected,
    because eroding it away leaves the body intact and connected. Necks and
    thin bodies -- the failures that actually break a board -- are covered.
    """
    from shapely.geometry import Polygon

    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return 0.0

    def survives(radius: float) -> bool:
        eroded = polygon.buffer(-radius)
        if eroded.is_empty:
            return False
        if eroded.geom_type == "MultiPolygon" and len(eroded.geoms) > 1:
            return False
        return True

    low, high = 0.0, 1.0
    for _ in range(30):
        if not survives(high):
            break
        low = high
        high *= 2.0
    else:
        return high * 2.0

    for _ in range(steps):
        middle = (low + high) / 2.0
        if survives(middle):
            low = middle
        else:
            high = middle

    return 2.0 * high


def check_trace_width(width_mm: float) -> list[Issue]:
    """Check a single trace width against the process minimum."""
    if width_mm < MIN_TRACE_WIDTH_MM:
        return [
            Issue(
                "error",
                "trace-too-thin",
                f"线宽 {width_mm:.4f} mm 低于最小可加工线宽 {MIN_TRACE_WIDTH_MM:.2f} mm",
            )
        ]
    return []


def raise_for_errors(report: OutlineReport, prefix: str) -> None:
    """Stop an export whose outline cannot be manufactured.

    The counterpart to :func:`warning_sentence`: an error is a file that would
    open and be wrong, so the export refuses; a warning is a board that will be
    made and is fragile, so the export proceeds and says so.
    """
    if report.ok:
        return
    details = "；".join(str(issue) for issue in report.errors)
    raise ValueError(f"{prefix}: {details}")


def warning_sentence(report: OutlineReport) -> str:
    """The report's warnings as one user-facing sentence, or an empty string.

    ⚠️ **Warnings were being discarded, and that is a real defect rather than a
    cosmetic one.** ``check_outline`` computes them and every export path threw
    them away -- measured on a board with a 0.15 mm waist, ``min_neck_width_mm``
    produced ``[warning] thin-neck`` and neither writer's ``note`` mentioned it.
    The user got a board that snaps in the post and had no way to know from the
    tool that it would. A warning the user never sees is not a warning.

    Returns ``""`` when there is nothing to say, so a caller can concatenate it
    unconditionally.
    """
    warnings = report.warnings
    if not warnings:
        return ""
    return "⚠️ " + "；".join(issue.message for issue in warnings) + "。"
