"""Crop an illustration down to a badge region's shape, keeping the head.

The badge template reserves two rectangles for artwork, and their shapes are
very different -- the tall one is 42.526 × 99.626 mm (aspect 0.427) and the
short one 38.861 × 56.324 mm (aspect 0.690). A 2048-pixel square illustration
wasted most of its pixels once placed in either, so the library is cropped to
match before it is ever opened.

The rule from the user: **the head must be complete, the bottom may be cut.**
That is what makes the crop well-defined, and it is why the window is anchored
at the top rather than centred on the picture -- centring a standing figure in
a 0.427 window trims the bottom nicely but decapitates nothing; centring it in
a 0.690 window is fine too, but centring it in a *portrait* window while the
figure sits low in the frame would cut the head off. Anchoring at the top
cannot.

Finding the person is the one part that is not arithmetic. Two kinds of
artwork live in this library and they need different treatment:

* **Figure art** -- the illustration is a cut-out on transparency, and the
  alpha channel *is* the silhouette. Measured across the library, 87% of the
  files are this kind: the alpha bounding box is sparse inside itself
  (median fill 0.45 for 精英0) and rarely spans the whole frame.
* **Scene art** -- the illustration is an opaque painted scene (a room, a
  street, a stage) with the character somewhere inside it. Median alpha fill
  is 0.69 and the box spans 96% of the width, so the alpha tells us nothing
  about where the person is. These are found by saliency: edge and colour
  energy, weighted toward the middle of the frame, thresholded and reduced to
  its largest blob.

The choice between them is made from the same measurements that separate the
two groups, not from a hand-written list of file names -- a new skin lands in
the right branch without anyone updating anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .badge_template import LARGE_REGION, SMALL_REGION, ArtworkRegion

__all__ = [
    "FigureBox",
    "Detection",
    "CropPlan",
    "detect_figure",
    "plan_crop",
    "crop",
    "region_aspect",
    "SCENE_BOX_WIDTH",
    "SCENE_BOX_FILL",
]

SCENE_BOX_WIDTH = 0.85
"""Above this share of the frame, the alpha box is the whole frame, not a person."""

SCENE_BOX_FILL = 0.72
"""Above this fill inside its own box, the alpha shape is a painted scene.

Both thresholds were measured, not guessed. Across 1406 illustrations the two
groups separate cleanly on these numbers: figure art sits at a median alpha
fill of 0.45 (精英0) to 0.61 (精英2) and a median box width of 0.60 to 0.96,
while the 182 scene files sit above both. They are a pair of conditions rather
than one because either alone miscatches: a figure holding a wide prop fills
its box but does not span the frame, and a sparse scene can span the frame
while staying hollow.
"""

HEAD_BAND = 0.12
"""How much of the figure's height counts as "the head" when centring.

The window is centred on the *head*, not on the figure's bounding box. Props
give the box away -- 暮落 carries a case that doubles its width, 贝洛内 drags a
suitcase -- and centring on the box would push the character off to one side.
The top band of the silhouette is the head for every example checked.
"""

SIDE_MARGIN = 0.06
"""Share of the figure's width left as breathing room on each side."""

TOP_MARGIN = 0.02
"""Share of the crop's height left above the highest pixel of the figure."""

FACE_MODEL_CACHE = Path(tempfile.gettempdir()) / "pcbart_face_detector.onnx"
"""An ASCII path for the detector weights.

OpenCV cannot open a path with non-ASCII characters on Windows, and this
project sits under a Chinese directory name, so the weights are unpacked here
at run time. Passing the model as a bytes object instead segfaults OpenCV 5.0;
a real file with an ASCII name works.
"""

#: The crop is sized from the *face*, so every picture comes out looking alike.
#:
#: Measured off the reference the user picked (安洁莉娜 精英0): its window is
#: 4.36 face-widths across, the eye midpoint sits at 0.499 of the window's
#: width, and the top of the face is 0.25 of the window's height below the top
#: edge. Reproducing those three numbers on every picture is what "脸部大小
#: 基本一致" means -- the face lands the same size and in the same place
#: whatever the illustration does around it.
#:
#: The window is *derived from* the face rather than clamped to the picture, so
#: a face near an edge is still centred and the overhang is filled with
#: transparency.
FACE_WINDOW_WIDTHS = 4.36
"""Window width as a multiple of the detected face's width."""

FACE_TOP_SHARE = 0.25
"""Where the top of the face sits, as a share of the window's height."""

MIN_FACE_PIXELS = 20
"""Below this a detection is noise, not a face."""


def face_window(face: FaceBox, aspect: float, image_size: tuple[int, int]) -> CropPlan:
    """The window that shows ``face`` at the reference size and position.

    The height follows from the aspect rather than the face, because the aspect
    is fixed by the stencil the artwork has to fill; the face decides how big
    the window is, and the aspect decides its shape.
    """
    width = max(1.0, face.width * FACE_WINDOW_WIDTHS)
    height = width / aspect
    # Centred on the eyes, not the box: the box includes hair and ears, and on
    # the reference the two differ by half a percent of the window.
    left = face.eye_x - width / 2.0
    top = face.y - height * FACE_TOP_SHARE
    return CropPlan(
        left=int(round(left)),
        top=int(round(top)),
        width=max(1, int(round(width))),
        height=max(1, int(round(height))),
        aspect=aspect,
        method="face",
    )


#: Per-asset subject boxes, in fractions of the image, for pictures the
#: automatic rules get wrong.
#:
#: ``name -> (x0, y0, x1, y1)``, each 0-1 with y measured from the top. The
#: crop puts this box in the middle of the window and grows the window to the
#: target aspect around it.
#:
#: Hand-tuned on the user's instruction, and kept here rather than applied to
#: the file once, so a re-crop reproduces the same framing instead of silently
#: undoing it. Every entry should say what the subject is and why the automatic
#: rules missed it.
ASSET_SUBJECTS: dict[str, tuple[float, float, float, float]] = {
    # 猫猫坐像。The subject is the seated cat with the red bow, and neither
    # automatic anchor finds it: the silhouette's head band centres on the
    # picture (0.51) because the gold frame spans the same width, and the
    # saliency blob lands on the cat's face but is not consulted for a
    # non-scene picture. The window therefore opened on the top of the frame
    # and left the cat along the bottom edge.
    # The box is centred on the cat's face, not on its body: measured from the
    # artwork, the seated cat's eye midpoint is at 0.552 of the width, while its
    # body runs from about 0.39 to 0.715. Centring the body instead left the
    # face 4% of a window to the left of the axis.
    "立绘 Miss.Christine 1.png": (0.39, 0.22, 0.715, 0.90),
}


def subject_window(
    subject: tuple[float, float, float, float],
    aspect: float,
    image_size: tuple[int, int],
) -> CropPlan:
    """A window that centres ``subject`` and is grown to ``aspect``.

    Grown rather than cropped to: the subject is what has to be visible, so the
    extra area the aspect demands is taken from around it, never out of it.
    """
    width, height = image_size
    x0, y0, x1, y1 = subject
    left = x0 * width
    top = y0 * height
    box_w = max(1.0, (x1 - x0) * width)
    box_h = max(1.0, (y1 - y0) * height)

    if box_w / box_h < aspect:
        box_w = box_h * aspect
    else:
        box_h = box_w / aspect

    centre_x = (x0 + x1) / 2.0 * width
    centre_y = (y0 + y1) / 2.0 * height
    _ = left, top
    return CropPlan(
        left=int(round(centre_x - box_w / 2.0)),
        top=int(round(centre_y - box_h / 2.0)),
        width=max(1, int(round(box_w))),
        height=max(1, int(round(box_h))),
        aspect=aspect,
        method="manual",
    )


VEHICLE_SHARE = 0.90
"""How much of a vehicle's body the crop keeps, as a share of its height.

The user: "非人类（小车）类型的角色截图尽量把五分之四个车体放入图片，不要只留
一小部分." Four fifths was the floor; 90% is used because the window can have
it without any cost -- the aspect ratio still fills the short region, and the
extra tenth is body the user wants to see rather than blank margin.

These units are not people, so a waist crop made no sense on them at all: at
the human 45% share, Lancet-2 kept only its satellite dome, THRM-EX a single
arm and PhonoR-0 the top of its hood.
"""


#: Units that are a vehicle or a piece of furniture rather than a person.
#:
#: A named list on purpose. Every shape measurement that was tried to find them
#: automatically also matched ordinary characters -- silhouette aspect ratio,
#: fill density and "no face detected" all overlap heavily (297 files have no
#: detectable face, almost all of them people). The set is small, stable and
#: visible to a reader, which beats a heuristic that is wrong quietly.
#:
#: Found with the roster's own data: every one of these is 稀有度=0 or an
#: RCX-series unit. The other 稀有度=0 units -- 泰拉大陆调查团, U-Official,
#: 罗德岛隐秘队, 虎狼丸 -- are a person, a person in a chair, a squad and a dog,
#: so they keep the ordinary rule.
VEHICLE_UNITS: frozenset[str] = frozenset(
    {
        "Lancet-2",
        "Castle-3",
        "THRM-EX",
        "Friston-3",
        "PhonoR-0",
        "GALLUS²",
        "CONFESS-47",
        "正义骑士号",
    }
)


def is_vehicle(path: str) -> bool:
    """Whether an illustration belongs to a vehicle unit.

    Matched on the file name, because that is what the crop tool has: the
    assets are named ``立绘 <干员> <名字>.png``.
    """
    stem = Path(path).stem
    for unit in VEHICLE_UNITS:
        if stem.startswith(f"立绘 {unit} ") or stem == f"立绘 {unit}":
            return True
    return False


FACE_MIN_SCORE = 0.5
"""Below this the detection is a guess; the coarse fallback is used instead."""


WAIST_SHARE = 0.45
"""Where the waist sits, as a share of the figure's height from the top.

Measured, not assumed: this fraction was drawn onto twelve illustrations at
35/42/50/58% and 45% is the line that lands on the belt or waist for nearly all
of them. It is a *share of the figure*, not a fixed pixel count and not a share
of the frame, so it survives every source resolution and every framing -- a
bust painting and a full-length one both put their waist at the same fraction
of what is visible.
"""

MIN_FIGURE_PIXELS = 32
"""Below this the detection is meaningless and the frame is used instead."""


@dataclass(frozen=True)
class FigureBox:
    """Where the person is, in pixels, and how it was found."""

    x0: int
    y0: int
    x1: int
    y1: int
    method: str

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def centre_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


_FACE_DETECTOR = None


def face_detector():
    """The shared YuNet detector, or ``None`` when OpenCV lacks the API.

    One instance is reused: creating it per picture costs about 20 ms, which
    over 1500 pictures is half a minute for nothing.
    """
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            _FACE_DETECTOR = False
        else:
            import contextlib
            import os

            from .face_model import FACE_MODEL_B64

            if not FACE_MODEL_CACHE.exists():
                import base64

                FACE_MODEL_CACHE.write_bytes(base64.b64decode(FACE_MODEL_B64))
            with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
                _FACE_DETECTOR = cv2.FaceDetectorYN.create(
                    str(FACE_MODEL_CACHE), "", (320, 320), FACE_MIN_SCORE, 0.3, 50
                )
    return _FACE_DETECTOR or None


@dataclass(frozen=True)
class FaceBox:
    """A detected face, in pixels, with Y down like the image."""

    x: float
    y: float
    width: float
    height: float
    eye_x: float
    """Midpoint between the two eyes. The axis the crop is centred on."""
    eye_y: float

    @property
    def centre_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def bottom(self) -> float:
        return self.y + self.height


def detect_face(rgb: np.ndarray) -> FaceBox | None:
    """Where the face is, or ``None``.

    Returns the whole box, not just a point, because the crop is scaled *by*
    the face: the user asked for the face to come out the same size in every
    picture, which means the window is a fixed multiple of the face width.

    ``None`` is not an error. A good number of the operators are robots, and
    玫兰莎's face is behind her own hair; those fall back to the silhouette.
    """
    detector = face_detector()
    if detector is None:
        return None

    import cv2

    height, width = rgb.shape[:2]
    scale = 1.0
    if max(height, width) > 1600:
        scale = 1600.0 / max(height, width)
    image = (
        cv2.resize(rgb, (int(width * scale), int(height * scale)))
        if scale < 1.0
        else rgb
    )
    detector.setInputSize((image.shape[1], image.shape[0]))
    _count, faces = detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if faces is None or not len(faces):
        return None

    best = max(faces, key=lambda row: row[-1])
    if float(best[-1]) < FACE_MIN_SCORE:
        return None
    x, y, w, h = (float(v) for v in best[:4])
    landmarks = best[4:14].reshape(5, 2)
    # The eye midpoint, not the box centre. The box includes hair and ears, and
    # on the reference image it sat 5 px off while the eyes were exact.
    eye_x = float((landmarks[0][0] + landmarks[1][0]) / 2.0)
    eye_y = float((landmarks[0][1] + landmarks[1][1]) / 2.0)
    return FaceBox(
        x=x / scale,
        y=y / scale,
        width=w / scale,
        height=h / scale,
        eye_x=eye_x / scale,
        eye_y=eye_y / scale,
    )


@dataclass(frozen=True)
class Detection:
    """What was found in one illustration, and how it was found."""

    box: FigureBox
    """The person, when the silhouette is a person; otherwise the scene."""

    weights: np.ndarray
    """Per-pixel weight for the detected subject, used to find the head."""

    head_centre_x: float
    head_top_y: int

    scene_like: bool
    """Whether the alpha silhouette spans and fills the frame.

    True means the alpha cannot be read as a person, so the subject has to be
    found by saliency instead.
    """

    saliency: tuple[FigureBox, float, int] | None = None
    """``(box, head_x, head_y)`` from the saliency pass, when one was run."""

    face: FaceBox | None = None
    """The detected face, when there is one. Drives scaling and centring."""

    alpha_box: FigureBox | None = None
    """The alpha silhouette's bounds, whatever ``box`` ended up meaning.

    Needed because ``box`` is replaced by the saliency blob for a scene-like
    picture, and a vehicle that saliency misfires on needs its *silhouette*
    height -- 立绘 THRM-EX 双料四分卫 came back with a 106 x 130 blob on an
    880 px canvas, and measuring the body from that produced a sliver.
    """


@dataclass(frozen=True)
class CropPlan:
    """The window to cut out, in pixels, with Y down like the image."""

    left: int
    top: int
    width: int
    height: int
    aspect: float
    method: str
    upper_body: bool = False
    """Whether the window was cut at the waist rather than at the frame edge."""

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def region_aspect(region: ArtworkRegion) -> float:
    return region.width_mm / region.height_mm


def _alpha_mask(rgba: np.ndarray) -> np.ndarray:
    return rgba[:, :, 3] > 8


def _scene_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """The largest salient blob inside an opaque scene.

    Edge energy plus colour energy, blurred, multiplied by a centre prior and
    thresholded. The prior is what keeps a busy background (a painted sky, a
    rack of tools) from outscoring the character, which is nearly always near
    the middle of a card illustration.
    """
    import cv2

    height, width = mask.shape
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    energy = cv2.magnitude(
        cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3),
    ) + 0.6 * cv2.magnitude(
        cv2.Sobel(saturation, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(saturation, cv2.CV_32F, 0, 1, ksize=3),
    )
    energy = cv2.GaussianBlur(energy, (0, 0), max(height, width) * 0.012)

    yy, xx = np.mgrid[0:height, 0:width]
    sigma = 0.30 * max(height, width)
    prior = np.exp(
        -(((xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2) / (2 * sigma * sigma))
    )
    score = energy * prior * mask.astype(np.float32)
    inside = score[mask]
    if inside.size == 0 or float(inside.max()) <= 0.0:
        return None

    selected = (score >= np.percentile(inside, 96)).astype(np.uint8)
    size = max(3, height // 200)
    selected = cv2.morphologyEx(
        selected, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, 4]))
    blob = labels == biggest
    # Blur the selection into a weight map: the crop centres on the head, and a
    # hard-edged blob's centroid is noisier than its smoothed mass.
    return cv2.GaussianBlur(blob.astype(np.float32), (0, 0), max(2.0, height * 0.01))


def _saliency_head(
    rgb: np.ndarray, mask: np.ndarray
) -> tuple[FigureBox, float, int] | None:
    """Where the character is inside an opaque painted scene.

    The box is the character's head-and-shoulders area, not the whole person --
    measured at 10-25% of the frame across the scene files. That is exactly why
    it is used to *place* the crop and never to size it.
    """
    weights = _scene_mask(rgb, mask)
    if weights is None or float(weights.sum()) <= 0.0:
        return None
    ys, xs = np.where(weights > weights.max() * 0.5)
    if not xs.size:
        return None
    box = FigureBox(
        int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), "saliency"
    )
    return box, (box.x0 + box.x1) / 2.0, box.y0


def detect_figure(rgba: np.ndarray) -> Detection | None:
    """Find the person in an RGBA illustration.

    Returns ``None`` when the picture has no usable subject -- an empty frame,
    a fully opaque rectangle, a decorative tile -- so the caller can fall back
    to using the frame as-is instead of inventing a crop.
    """
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError("需要 RGBA 图像")

    height, width = rgba.shape[:2]
    mask = _alpha_mask(rgba)
    if int(mask.sum()) < MIN_FIGURE_PIXELS:
        return None

    ys, xs = np.where(mask)
    ax0, ax1, ay0, ay1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    box_width = (ax1 - ax0 + 1) / width
    box_fill = float(mask.sum()) / max(1, (ax1 - ax0 + 1) * (ay1 - ay0 + 1))

    scene_like = box_width >= SCENE_BOX_WIDTH and box_fill >= SCENE_BOX_FILL
    head = _saliency_head(rgba[:, :, :3], mask)
    face = detect_face(rgba[:, :, :3])
    alpha_box = FigureBox(ax0, ay0, ax1, ay1, "alpha")

    if scene_like and head is not None:
        box, head_x, head_y = head
        return Detection(
            box=box,
            weights=np.zeros_like(mask, dtype=np.float32),
            head_centre_x=head_x,
            head_top_y=head_y,
            scene_like=True,
            saliency=head,
            face=face,
            alpha_box=alpha_box,
        )

    # The silhouette itself is the subject. A little smoothing keeps the head
    # band from being dominated by a single stray pixel of hair.
    import cv2

    weights = cv2.GaussianBlur(
        mask.astype(np.float32), (0, 0), max(1.0, width * 0.004)
    )
    return replace(
        _measure(weights, ax0, ay0, ax1, ay1, "alpha"),
        scene_like=scene_like,
        saliency=head,
        face=face,
        alpha_box=alpha_box,
    )


def _measure(
    weights: np.ndarray, x0: int, y0: int, x1: int, y1: int, method: str
) -> Detection:
    """Centre on the head band, not on the bounding box."""
    box = FigureBox(x0, y0, x1, y1, method)
    band = max(2, int(round((y1 - y0 + 1) * HEAD_BAND)))
    top = weights[y0 : min(weights.shape[0], y0 + band), x0 : x1 + 1]
    column = top.sum(axis=0)
    if column.sum() > 0:
        head_x = x0 + float(np.dot(np.arange(column.size), column) / column.sum())
    else:
        head_x = box.centre_x
    return Detection(
        box=box,
        weights=weights,
        head_centre_x=head_x,
        head_top_y=y0,
        scene_like=False,
    )


def plan_crop(
    detection: Detection,
    aspect: float,
    image_size: tuple[int, int],
    *,
    side_margin: float = SIDE_MARGIN,
    top_margin: float = TOP_MARGIN,
    waist_share: float | None = None,
    asset: str = "",
) -> CropPlan:
    """The window to cut, for one target aspect ratio.

    Deliberately simple, because the first version was not and it went wrong in
    the direction that matters. That version classified each picture as figure
    or scene and anchored differently per branch -- and a scene misread as a
    figure, or the reverse, could drop the window *below* the top of the head.
    "The head must be complete" is the one rule with no recovery, so the anchor
    is now taken from whichever source says the figure starts higher:

    * the top of the alpha silhouette -- for a cut-out, the head itself; and
      for a painted scene, the top of the artwork, which is above the head; and
    * the top of the salient blob, when one was found.

    Taking the higher of the two means a wrong classification can only ever
    include *more* picture above the head. It cannot cut it. The same argument
    covers props: a case or a suitcase lifts the silhouette's top, never lowers
    it.

    The window is then the largest rectangle of the wanted aspect that fits the
    frame. It is never larger than the source -- the whole point is a smaller
    library, and an overhanging window is pure wasted pixels; one scene file
    produced a 2048 x 4798 result from a 2048-square source that way.
    """
    if aspect <= 0:
        raise ValueError("宽高比必须为正")

    # A hand-tuned subject wins over every automatic rule. These are the
    # pictures where none of the anchors found the subject, and a person looked
    # at it and said where it is.
    if asset:
        subject = ASSET_SUBJECTS.get(Path(asset).name)
        if subject is not None:
            return subject_window(subject, aspect, image_size)

    # A detected face sets the scale and the position, and nothing else is
    # consulted. This is what makes the face the same size in every picture.
    if detection.face is not None and detection.face.width >= MIN_FACE_PIXELS:
        return face_window(detection.face, aspect, image_size)

    frame_w, frame_h = image_size
    # A painted scene has no figure box to take a waist from -- its detected box
    # is only the character's head -- so the waist rule is skipped rather than
    # guessed at, and the window falls back to filling the frame.
    # A vehicle keeps most of its body; a person is cut at the waist. The
    # vehicle share only takes effect where the waist rule would have applied,
    # so a painted scene still falls back to filling the frame.
    if waist_share is not None and asset and is_vehicle(asset):
        waist_share = max(float(waist_share), VEHICLE_SHARE)

    # The scene exclusion is for painted scenes, whose detected box is only the
    # character's head. A vehicle is never that: when saliency misfires on one
    # -- 立绘 THRM-EX 双料四分卫 came back scene-like with a 106x130 box on an
    # 880px picture -- the box is far too small to be a scene, and treating it
    # as one threw the vehicle away and cropped to a slice of hull.
    waist_applies = bool(
        waist_share is not None
        and (not detection.scene_like or (asset and is_vehicle(asset)))
    )
    if waist_applies:
        if asset and is_vehicle(asset) and detection.alpha_box is not None:
            body = detection.alpha_box
        else:
            body = detection.box
        # The height comes from the waist and the width follows from the
        # aspect -- that is what makes the result an upper-body crop.
        #
        # A note for whoever changes this next, because it was changed once on
        # a bad measurement: driving the window from the figure's *width*
        # instead also fits the frame and keeps more of the silhouette, but it
        # is not an upper-body crop at all -- it framed the whole character
        # again. "Retained 16-39% of the silhouette" sounded alarming and was
        # in fact correct: a head-to-waist crop of a standing figure keeps
        # roughly a third of its pixels, because the legs are most of the rest.
        # The retention figure is a poor test of this rule; the picture is not.
        height = max(1.0, float(body.y1 - body.y0 + 1) * float(waist_share))
        width = height * aspect
        # Fitting the frame keeps the aspect, so a subject too wide for the
        # window is centred rather than sliced off to one side.
        if width > frame_w:
            width = float(frame_w)
            height = width / aspect
        if height > frame_h:
            height = float(frame_h)
            width = height * aspect
    elif aspect >= frame_w / frame_h:
        width = float(frame_w)
        height = width / aspect
    else:
        height = float(frame_h)
        width = height * aspect

    # Higher of the two tops, so the head is always inside the window.
    top_of_figure = detection.box.y0
    if detection.saliency is not None:
        top_of_figure = min(top_of_figure, detection.saliency[2])
    if detection.face is not None:
        # The face's own top edge is the most reliable mark above the eyes.
        top_of_figure = min(top_of_figure, detection.face.y)

    # Whose face position to trust. Third version of this rule, and the first
    # two are worth recording because both looked reasonable and both were
    # wrong on real pictures:
    #
    # 1. The centre of the silhouette's head band. It is a mass centroid, so
    #    hair, ears, a raised weapon and a shoulder rig all pull it sideways --
    #    on 安洁莉娜 精英0 it sat 53 px off her face, which the user spotted.
    # 2. The saliency blob. It locked onto whatever was busiest: a palm tree, a
    #    stone pillar, a prop, with 13 of 182 scenes more than 10% off centre.
    #
    # Now the face is found with YuNet and the eye midpoint is the axis. When no
    # face is found -- a robot, a character facing away -- the older anchors
    # remain, because one that is roughly right beats no crop at all.
    if detection.face is not None:
        head_x = detection.face.eye_x
    elif detection.scene_like:
        head_x = frame_w / 2.0
    else:
        head_x = detection.head_centre_x

    # The window is centred on the face, and it is allowed to hang off the
    # frame. Clamping it inside instead is what left faces at the edge of the
    # delivered picture: on 埃拉托 and 巫恋 the face sits near the left edge of
    # the source, so holding the window inside the frame held the face 30-44%
    # of a window off the axis. The overhang is filled with transparency by
    # :func:`crop`, which costs nothing on a print -- the alternative costs the
    # one thing the user asked for.
    left = head_x - width / 2.0
    top = top_of_figure - height * top_margin

    return CropPlan(
        left=int(round(left)),
        top=int(round(top)),
        width=max(1, int(round(width))),
        height=max(1, int(round(height))),
        aspect=aspect,
        method="saliency" if detection.saliency is not None else "alpha",
        upper_body=waist_applies,
    )


def crop(image, plan: CropPlan, *, background=(0, 0, 0, 0)):
    """Cut ``plan`` out of ``image``, padding where it runs off the frame.

    Padding rather than sliding the window back inside: sliding would move the
    top edge down and could clip the head, which is the one thing the crop is
    not allowed to do. The overhang is at the bottom and the sides, where a
    transparent margin costs nothing.
    """
    from PIL import Image

    source = image if image.mode == "RGBA" else image.convert("RGBA")
    out = Image.new("RGBA", (plan.width, plan.height), background)

    src_left = max(0, plan.left)
    src_top = max(0, plan.top)
    src_right = min(source.width, plan.right)
    src_bottom = min(source.height, plan.bottom)
    if src_right <= src_left or src_bottom <= src_top:
        # Nothing of the picture falls inside the window at all.
        return out

    piece = source.crop((src_left, src_top, src_right, src_bottom))
    out.paste(piece, (src_left - plan.left, src_top - plan.top))
    return out


def aspect_options() -> Iterable[tuple[str, float]]:
    """The two crop shapes, keyed the way the folders are named."""
    return (("窄", region_aspect(SMALL_REGION)), ("长", region_aspect(LARGE_REGION)))
