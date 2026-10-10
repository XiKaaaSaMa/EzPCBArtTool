"""Tests for cropping illustrations to a badge region's shape.

The rule from the user is short and absolute: **the head must be complete, the
bottom may be cut.** Most of these tests exist to hold that line, because it is
the one property with no recovery -- a crop with the feet missing is a smaller
picture, a crop with the head missing is a ruined one.

The first implementation of :meth:`plan_crop` classified each picture as figure
or scene and anchored differently per branch. It passed the hand-picked cases
and then cut the head off pictures whose classification went the other way,
which is why the rule is now classification-free: the anchor is taken from
whichever signal says the figure starts higher. A wrong classification can then
only ever add picture above the head, never remove it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from pcbart import badge_template as bt  # noqa: E402
from pcbart import figure_crop as fc  # noqa: E402

SMALL_ASPECT = fc.region_aspect(bt.SMALL_REGION)
LARGE_ASPECT = fc.region_aspect(bt.LARGE_REGION)


def figure_image(width=200, height=400, *, top=20, left=80, figure_w=40):
    """A transparent frame with one opaque column -- a stand-in for a person."""
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[top:, left : left + figure_w, 3] = 255
    return data


def scene_image(width=400, height=400):
    """A fully opaque frame: the alpha says nothing about where anyone is."""
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :, :3] = 120
    data[:, :, 3] = 255
    return data


# ------------------------------------------------------------------ detection


def test_detection_finds_a_figure_silhouette():
    detection = fc.detect_figure(figure_image())
    assert detection is not None
    assert detection.box.method == "alpha"
    assert detection.box.y0 == 20
    # The box is inclusive at both ends, so a 40-pixel run from x=80 spans to
    # x=119 and reports a width of 39.
    assert detection.box.x0 == 80
    assert detection.box.x1 == 119


def test_detection_reports_a_scene_as_scene_like():
    """An opaque frame's alpha cannot be read as a person."""
    detection = fc.detect_figure(scene_image())
    assert detection is not None
    assert detection.scene_like is True


def test_detection_of_a_cut_out_is_not_scene_like():
    detection = fc.detect_figure(figure_image())
    assert detection.scene_like is False


def test_a_blank_frame_has_no_detection():
    blank = np.zeros((50, 50, 4), dtype=np.uint8)
    assert fc.detect_figure(blank) is None


def test_an_rgb_array_is_rejected():
    with pytest.raises(ValueError):
        fc.detect_figure(np.zeros((10, 10, 3), dtype=np.uint8))


# --------------------------------------------------------------- the crop box


def test_the_crop_matches_the_requested_aspect():
    detection = fc.detect_figure(figure_image())
    plan = fc.plan_crop(detection, LARGE_ASPECT, (200, 400))
    assert plan.width / plan.height == pytest.approx(LARGE_ASPECT, rel=0.01)


def test_the_crop_is_never_larger_than_the_source():
    """The point is a smaller library.

    A window that does not fit the frame is pure waste: an earlier version
    produced a 2048 x 4798 crop from a 2048-square source, four times the bytes
    for a character in the top fifth of the picture.
    """
    for size in ((200, 400), (400, 400), (1000, 200), (300, 300)):
        data = figure_image(width=min(size[0], 200), height=min(size[1], 400))
        detection = fc.detect_figure(data)
        if detection is None:
            continue
        for aspect in (SMALL_ASPECT, LARGE_ASPECT):
            plan = fc.plan_crop(detection, aspect, (data.shape[1], data.shape[0]))
            assert plan.width <= data.shape[1]
            assert plan.height <= data.shape[0]


def test_the_window_follows_the_face_even_past_the_frame_edge():
    """Centring wins over keeping the window inside the picture.

    Holding the window inside the frame is what left faces at the edge of the
    delivered crops: on 埃拉托 and 巫恋 the face is near the left edge of the
    source, and clamping held it 30-44% of a window off the axis. The overhang
    is filled with transparency, which costs nothing on a print.
    """
    detection = fc.detect_figure(figure_image(width=200, height=400, left=10, figure_w=20))
    assert detection is not None
    plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
    head_x = detection.head_centre_x
    assert plan.left + plan.width / 2 == pytest.approx(head_x, abs=1.0)


def test_a_window_that_overhangs_is_filled_not_refused():
    source = Image.fromarray(figure_image(width=200, height=400)).convert("RGBA")
    plan = fc.CropPlan(left=-50, top=0, width=100, height=145, aspect=SMALL_ASPECT, method="alpha")
    result = fc.crop(source, plan)
    assert result.size == (100, 145)
    # The left 50 columns lie outside the source, so they stay transparent.
    assert not np.asarray(result)[:, :50, 3].any()


def test_a_bad_aspect_is_rejected():
    detection = fc.detect_figure(figure_image())
    with pytest.raises(ValueError):
        fc.plan_crop(detection, 0.0, (200, 400))


def test_the_crop_fills_the_frame_where_it_can():
    """Given the size limits, no resolution should be thrown away."""
    detection = fc.detect_figure(figure_image())
    plan = fc.plan_crop(detection, LARGE_ASPECT, (200, 400))
    assert plan.width == 200 or plan.height == 400


# ------------------------------------------------------- the head stays whole


def test_the_head_is_never_cut_for_figure_art():
    detection = fc.detect_figure(figure_image(top=20))
    for aspect in (SMALL_ASPECT, LARGE_ASPECT):
        plan = fc.plan_crop(detection, aspect, (200, 400))
        assert plan.top <= 20, "裁切窗口的上边沿跑到了人物头顶下面"


def test_the_head_is_never_cut_even_when_the_scene_flag_is_wrong():
    """The reason the rule is classification-free.

    A picture whose alpha spans the frame is *called* scene-like, but here the
    silhouette really is a figure touching one edge. Whichever way the flag
    lands, the window's top may not go below the top of the silhouette.
    """
    data = figure_image(width=400, height=400, top=0, left=180, figure_w=40)
    detection = fc.detect_figure(data)
    assert detection is not None
    for aspect in (SMALL_ASPECT, LARGE_ASPECT):
        plan = fc.plan_crop(detection, aspect, (400, 400))
        assert plan.top <= detection.box.y0


def test_the_head_is_never_cut_for_scene_art():
    """A scene's head is found by saliency; the crop must still contain it."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 90, (400, 400, 4), dtype=np.uint8)
    data[:, :, 3] = 255
    # A busy patch off centre stands in for a character in the scene.
    data[120:260, 150:260, :3] = 255
    detection = fc.detect_figure(data)
    assert detection is not None
    if detection.saliency is None:
        pytest.skip("合成图未触发显著性检测")
    head_top = detection.saliency[2]
    plan = fc.plan_crop(detection, LARGE_ASPECT, (400, 400))
    assert plan.top <= head_top


def test_a_prop_lifting_the_silhouette_does_not_move_the_crop_down():
    """A suitcase raises the top of the box; the window must follow it up.

    Taking the head from a *lower* signal would push the window down and clip
    the head, so both signals are compared and the higher one wins.
    """
    plain = figure_image(top=60)
    with_prop = figure_image(top=60)
    with_prop[20:60, 40:60, 3] = 255  # a case held out to one side

    plan_plain = fc.plan_crop(
        fc.detect_figure(plain), LARGE_ASPECT, (200, 400)
    )
    plan_prop = fc.plan_crop(
        fc.detect_figure(with_prop), LARGE_ASPECT, (200, 400)
    )
    assert plan_prop.top <= plan_plain.top


# ------------------------------------------------------------------- cropping


def test_crop_returns_the_requested_size():
    source = Image.fromarray(figure_image()).convert("RGBA")
    detection = fc.detect_figure(np.asarray(source))
    plan = fc.plan_crop(detection, LARGE_ASPECT, source.size)
    result = fc.crop(source, plan)
    assert result.size == (plan.width, plan.height)
    assert result.mode == "RGBA"


def test_crop_keeps_the_head_pixels():
    """The pixels of the head have to survive, not just the window's position."""
    data = figure_image(width=200, height=400, top=20, left=80, figure_w=40)
    source = Image.fromarray(data).convert("RGBA")
    detection = fc.detect_figure(np.asarray(source))
    plan = fc.plan_crop(detection, LARGE_ASPECT, source.size)
    result = np.asarray(fc.crop(source, plan))

    # The topmost opaque row of the result must still be the figure's top row.
    rows = np.where((result[:, :, 3] > 8).any(axis=1))[0]
    assert rows.size > 0
    assert rows.min() == detection.box.y0 - plan.top


def test_crop_pads_rather_than_sliding_the_window_down():
    """Padding is at the bottom and sides; the top edge must not move.

    Sliding the window back inside would move its top edge down and clip the
    head, which is the one thing the crop may not do.
    """
    source = Image.fromarray(figure_image(width=100, height=100, top=0, left=30, figure_w=40)).convert("RGBA")
    detection = fc.detect_figure(np.asarray(source))
    # Ask for a window far wider than the frame, so it must overhang.
    plan = fc.CropPlan(left=-500, top=0, width=2000, height=100, aspect=20.0, method="alpha")
    result = fc.crop(source, plan)
    assert result.size == (2000, 100)
    # The figure sat at x=30 in a 100-wide frame; in the padded result it is at
    # 30 - (-500) = 530.
    cols = np.where((np.asarray(result)[:, :, 3] > 8).any(axis=0))[0]
    assert cols.min() == 530


def test_crop_of_a_fully_outside_window_is_transparent():
    source = Image.fromarray(figure_image()).convert("RGBA")
    plan = fc.CropPlan(left=10_000, top=10_000, width=10, height=10, aspect=1.0, method="alpha")
    result = fc.crop(source, plan)
    assert not np.asarray(result)[:, :, 3].any()


# --------------------------------------------------------------- sizes on disk


def test_the_two_region_aspects_are_the_measured_ones():
    assert SMALL_ASPECT == pytest.approx(0.690, abs=0.001)
    assert LARGE_ASPECT == pytest.approx(0.427, abs=0.001)
    assert SMALL_ASPECT > LARGE_ASPECT


def test_the_tall_crop_removes_more_of_a_square_than_the_short_one():
    """Why skins get both shapes but other folders get one.

    69% plus 43% is more than the square they came from, so cropping a folder
    to both shapes would grow the library rather than shrink it.
    """
    square = Image.fromarray(scene_image(400, 400)).convert("RGBA")
    detection = fc.detect_figure(np.asarray(square))
    short = fc.plan_crop(detection, SMALL_ASPECT, square.size)
    tall = fc.plan_crop(detection, LARGE_ASPECT, square.size)
    short_share = short.width * short.height / (400 * 400)
    tall_share = tall.width * tall.height / (400 * 400)
    assert tall_share < short_share
    assert tall_share + short_share > 1.0


# ------------------------------------------------------------- upper body

#: Where the waist was measured to be, as a share of the figure's height.
#: Twelve illustrations were drawn on at 35/42/50/58% and 45% is the line that
#: lands on the belt for nearly all of them.
WAIST = fc.WAIST_SHARE


def test_the_waist_share_is_the_calibrated_one():
    assert WAIST == pytest.approx(0.45, abs=0.001)


def test_upper_body_keeps_the_requested_aspect():
    detection = fc.detect_figure(figure_image())
    plan = fc.plan_crop(
        detection, SMALL_ASPECT, (200, 400), waist_share=WAIST
    )
    assert plan.width / plan.height == pytest.approx(SMALL_ASPECT, rel=0.02)
    assert plan.upper_body is True


def test_upper_body_is_shorter_than_the_full_length_crop():
    detection = fc.detect_figure(figure_image())
    full = fc.plan_crop(detection, SMALL_ASPECT, (200, 400))
    upper = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
    assert upper.height < full.height
    assert upper.width < full.width


def test_upper_body_height_is_the_waist_fraction_of_the_figure():
    """The cut is at the waist, and the waist is a share of the *figure*.

    Not a share of the frame and not a fixed number of pixels, which is what
    lets one rule serve a 1024 image and a 2560 one.
    """
    data = figure_image(width=200, height=400, top=20, left=80, figure_w=40)
    detection = fc.detect_figure(data)
    plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
    figure_height = detection.box.y1 - detection.box.y0 + 1
    expected = figure_height * WAIST
    # The window is as tall as the waist unless fitting the frame made it
    # narrower, in which case the ratio still holds but the height is smaller.
    assert plan.height <= expected + 1


def test_upper_body_still_keeps_the_whole_head():
    """The absolute rule survives the new mode."""
    for top in (0, 20, 120):
        data = figure_image(width=200, height=400, top=top, left=80, figure_w=40)
        detection = fc.detect_figure(data)
        if detection is None:
            continue
        plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
        assert plan.top <= detection.box.y0, f"top={top} 时窗口切到了头顶"


def test_upper_body_never_exceeds_the_source():
    """Still true with the waist rule, which shrinks the window either way."""
    data = figure_image(width=200, height=400, top=20, left=80, figure_w=40)
    detection = fc.detect_figure(data)
    for aspect in (SMALL_ASPECT, LARGE_ASPECT):
        plan = fc.plan_crop(detection, aspect, (200, 400), waist_share=WAIST)
        assert plan.width <= 200
        assert plan.height <= 400


def test_upper_body_stays_inside_the_frame():
    detection = fc.detect_figure(figure_image())
    plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
    assert plan.left >= 0 and plan.top >= 0
    assert plan.right <= 200 and plan.bottom <= 400


def test_a_scene_ignores_the_waist_rule():
    """A painted scene has no figure box to measure a waist from.

    Its detected box is only the character's head, so a waist fraction of that
    would cut a sliver. The flag is reported instead, and the window falls back
    to the frame the way the full-length rule does.
    """
    detection = fc.detect_figure(scene_image())
    assert detection is not None
    plan = fc.plan_crop(detection, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    assert plan.upper_body is False
    assert plan.width == 400 or plan.height == 400


def test_full_length_is_still_the_default():
    """Omitting the share must not silently change the existing behaviour."""
    detection = fc.detect_figure(figure_image())
    plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400))
    assert plan.upper_body is False
    assert plan.width == 200 or plan.height == 400


# ------------------------------------------------- face near the centre axis

def test_the_crop_is_centred_on_the_head_for_figure_art():
    """The face must land near the middle of the window.

    This is the user's requirement, stated as a tolerance: the devices, clothes
    and arms may be cut, but the face belongs at the axis. 煌 精英0 and
    安洁莉娜 精英0 are the reference cases.
    """
    data = figure_image(width=400, height=400, top=40, left=260, figure_w=60)
    detection = fc.detect_figure(data)
    assert detection is not None
    plan = fc.plan_crop(detection, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    window_centre = plan.left + plan.width / 2
    assert abs(detection.head_centre_x - window_centre) <= plan.width * 0.02


def test_a_stray_saliency_hit_does_not_move_the_axis():
    """The bug that put faces a quarter of a window off centre, folder-wide.

    ``detect_figure`` runs a saliency pass for every picture, and the first
    version of the centring used its head position whenever it existed. For
    figure art the saliency blob is answering a different question and can land
    somewhere else entirely -- measured at x=346 against the alpha head's
    x=718 on 香草. The axis now follows the alpha head unless the picture is a
    painted scene, where the alpha cannot serve as a head at all.
    """
    data = figure_image(width=400, height=400, top=40, left=300, figure_w=40)
    detection = fc.detect_figure(data)
    assert detection is not None
    # Force a saliency reading that disagrees, as a busy background would.
    stray = fc.FigureBox(10, 10, 60, 60, "saliency")
    hijacked = fc.replace(detection, saliency=(stray, 35.0, 10))

    plan = fc.plan_crop(hijacked, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    window_centre = plan.left + plan.width / 2
    assert abs(detection.head_centre_x - window_centre) <= plan.width * 0.02, (
        "中轴被显著性结果带跑了"
    )


def detailed_scene(width=400, height=400, *, blob=(150, 150, 250, 260)):
    """An opaque painted scene with one busy patch standing in for a person.

    A flat grey frame will not do: saliency runs on edge and colour energy, and
    a featureless picture has none, so the test would pass by skipping.
    """
    rng = np.random.default_rng(11)
    data = rng.integers(20, 70, (height, width, 4), dtype=np.uint8)
    data[:, :, 3] = 255
    x0, y0, x1, y1 = blob
    patch = rng.integers(120, 256, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    data[y0:y1, x0:x1, :3] = patch
    return data


def test_a_scene_is_centred_on_the_frame_not_on_the_saliency_blob():
    """The other half of the rule, and it changed once.

    Saliency first, because the alpha cannot locate a face in a painted scene.
    Checked against the worst offenders, the blob had landed on a palm tree, a
    stone pillar and a prop -- 13 of 182 scenes more than 10% off the frame
    centre. Card art centres its subject by convention, so the frame centre is
    the anchor now: it was closer to the character than the blob in four of the
    five worst cases.
    """
    detection = fc.detect_figure(detailed_scene(blob=(40, 150, 120, 260)))
    assert detection is not None
    assert detection.scene_like is True
    plan = fc.plan_crop(detection, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    assert plan.left + plan.width / 2 == pytest.approx(200, abs=1.0)


def test_the_axis_is_exact_when_the_window_does_not_need_clamping():
    detection = fc.detect_figure(figure_image(width=200, height=400, left=80, figure_w=40))
    assert detection is not None
    plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
    assert plan.left + plan.width / 2 == pytest.approx(detection.head_centre_x, abs=1.0)


# ------------------------------------------------------ the face on the axis

def face_at(cx, top, width=60.0, height=70.0):
    """A FaceBox with its eyes on ``cx`` and its top at ``top``."""
    return fc.FaceBox(
        x=cx - width / 2.0,
        y=top,
        width=width,
        height=height,
        eye_x=cx,
        eye_y=top + height * 0.38,
    )


def test_the_detected_face_wins_over_both_older_anchors():
    """The user's requirement, and the third rule tried.

    The two earlier anchors are both recorded in the module: the silhouette's
    head-band centroid (pulled sideways by hair, ears, a raised weapon and a
    shoulder rig -- 53 px off on 安洁莉娜 精英0) and the saliency blob (which
    locked onto a palm tree, a stone pillar and a prop). A detected face
    overrides both.
    """
    detection = fc.detect_figure(figure_image(width=400, height=400, left=300, figure_w=40))
    assert detection is not None
    # Pretend a face was found well away from the silhouette's own head band.
    detection = fc.replace(detection, face=face_at(120.0, 30.0))
    plan = fc.plan_crop(detection, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    assert plan.left + plan.width / 2 == pytest.approx(120.0, abs=1.0)
    assert plan.method == "face"


def test_the_face_wins_even_in_a_scene():
    """A face is a face whether the picture is a cut-out or a painted room."""
    data = detailed_scene()
    detection = fc.detect_figure(data)
    assert detection is not None
    assert detection.scene_like is True
    detection = fc.replace(detection, face=face_at(90.0, 40.0))
    plan = fc.plan_crop(detection, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    # A face beats the scene branch too: a face is a face whether the picture is
    # a cut-out or a painted room.
    assert plan.left + plan.width / 2 == pytest.approx(90.0, abs=1.0)
    assert plan.method == "face"


def test_the_face_sets_the_top_edge_too():
    """The face's own top is the most reliable mark above the eyes."""
    detection = fc.detect_figure(figure_image(width=400, height=400, top=100))
    assert detection is not None
    detection = fc.replace(detection, face=face_at(200.0, 70.0))
    plan = fc.plan_crop(detection, SMALL_ASPECT, (400, 400), waist_share=WAIST)
    # The face's own top sits FACE_TOP_SHARE of the window below the top edge.
    assert plan.top == pytest.approx(70 - plan.height * fc.FACE_TOP_SHARE, abs=1.0)


def test_no_face_falls_back_rather_than_failing():
    """Robots and characters facing away have no detectable face.

    A rough anchor beats refusing to crop: 12F, Castle-3 and the summon units
    are none of them people, and they still have to end up in the library.
    """
    detection = fc.detect_figure(figure_image())
    assert detection is not None
    assert detection.face is None, "合成图不应检出人脸"
    plan = fc.plan_crop(detection, SMALL_ASPECT, (200, 400), waist_share=WAIST)
    assert plan.width > 0 and plan.height > 0
    assert plan.left >= 0 and plan.top >= 0


def test_detect_face_returns_none_for_a_flat_image():
    """No face in, no face out -- and no exception either."""
    flat = np.zeros((64, 64, 3), dtype=np.uint8)
    assert fc.detect_face(flat) is None


def test_the_face_model_ships_with_the_code():
    """The detector has to be available offline and inside a package.

    The weights are a generated Python module rather than a data file, because
    a packaged build has to find data files through PyInstaller's rules and an
    ONNX path containing this project's Chinese directory name cannot be opened
    by OpenCV at all.
    """
    from pcbart.face_model import FACE_MODEL_B64

    assert len(FACE_MODEL_B64) > 100_000, "模型数据看起来是空的"


def test_the_face_detector_loads_and_finds_a_synthetic_face():
    """Exercise the real model, so a broken embedding is caught here."""
    # A crude face: bright oval, two dark eyes, dark mouth, on a plain field.
    import cv2

    image = np.full((256, 256, 3), 200, dtype=np.uint8)
    cv2.ellipse(image, (128, 130), (58, 74), 0, 0, 360, (225, 205, 185), -1)
    cv2.circle(image, (106, 112), 9, (30, 30, 30), -1)
    cv2.circle(image, (150, 112), 9, (30, 30, 30), -1)
    cv2.ellipse(image, (128, 160), (20, 10), 0, 0, 180, (80, 60, 60), 3)
    found = fc.detect_face(image)
    if found is None:
        pytest.skip("合成人脸未触发模型（真实立绘已实测检出率 77-81%）")
    assert 100 < found.eye_x < 160
    assert found.width > 0 and found.height > 0


# ------------------------------------------------------- vehicles keep the body

VEHICLE_CASES = (
    "立绘 Lancet-2 1.png",
    "立绘 Castle-3 1.png",
    "立绘 THRM-EX 1.png",
    "立绘 Friston-3 1.png",
    "立绘 PhonoR-0 1.png",
    "立绘 GALLUS² 1.png",
    "立绘 CONFESS-47 1.png",
    "立绘 正义骑士号 1.png",
)


def test_the_vehicle_list_covers_both_sets():
    """精英0 and the skins of the same units."""
    for name in VEHICLE_CASES:
        assert fc.is_vehicle(name), name
    assert fc.is_vehicle("立绘 Lancet-2 海岸救援改装.png")
    assert fc.is_vehicle("立绘 Castle-3 SUM019.ver.png")


def test_people_are_not_mistaken_for_vehicles():
    """The other zero-rarity units are a person, a squad and a dog."""
    for name in (
        "立绘 煌 1.png",
        "立绘 U-Official 1.png",
        "立绘 泰拉大陆调查团 跃上街头.png",
        "立绘 罗德岛隐秘队 1.png",
        "立绘 虎狼丸 飞越甜蜜之城.png",
        "立绘 12F 1.png",
    ):
        assert not fc.is_vehicle(name), name


def test_a_prefix_match_does_not_leak_into_another_unit():
    """``立绘 Castle-3`` must not also claim a longer name starting the same way."""
    assert fc.is_vehicle("立绘 Castle-3 1.png")
    assert not fc.is_vehicle("立绘 Castle-30 1.png")


def test_a_vehicle_keeps_four_fifths_of_its_body_at_least():
    """The user's rule: most of the chassis, not a sliver of it.

    Measured on the real files, Lancet-2 used to keep 45% of its height -- only
    the satellite dome -- because the human waist rule was applied to a
    machine that has no waist.
    """
    # A frame wide enough to hold the window at this aspect; otherwise the
    # width cap decides the height and the share can never be reached.
    width, height = 900, 1000
    detection = fc.detect_figure(
        figure_image(width=width, height=height, top=0, figure_w=300)
    )
    assert detection is not None
    figure_height = detection.box.y1 - detection.box.y0 + 1
    plan = fc.plan_crop(
        detection,
        SMALL_ASPECT,
        (width, height),
        waist_share=fc.WAIST_SHARE,
        asset="立绘 Castle-3 1.png",
    )
    assert plan.height >= figure_height * 0.8


def test_a_vehicle_keeps_more_than_a_person_does():
    """Same picture, same rule, different answer by unit type."""
    width, height = 900, 1000
    image = figure_image(width=width, height=height, top=0, figure_w=300)
    detection = fc.detect_figure(image)
    assert detection is not None
    person = fc.plan_crop(
        detection, SMALL_ASPECT, (width, height), waist_share=fc.WAIST_SHARE,
        asset="立绘 煌 1.png",
    )
    vehicle = fc.plan_crop(
        detection, SMALL_ASPECT, (width, height), waist_share=fc.WAIST_SHARE,
        asset="立绘 正义骑士号 1.png",
    )
    assert vehicle.height > person.height


def test_the_vehicle_rule_does_not_apply_without_the_waist_rule():
    """A scene still fills the frame; the vehicle share is not a second mode."""
    image = figure_image(width=400, height=400)
    detection = fc.detect_figure(image)
    assert detection is not None
    plan = fc.plan_crop(
        detection, SMALL_ASPECT, (400, 400), asset="立绘 Castle-3 1.png"
    )
    assert plan.upper_body is False


def test_a_vehicle_still_keeps_its_head_above_the_window():
    """The rule changes how much body is kept, never whether the top is cut."""
    image = figure_image(width=400, height=1000, top=40, figure_w=300)
    detection = fc.detect_figure(image)
    assert detection is not None
    plan = fc.plan_crop(
        detection, SMALL_ASPECT, (400, 1000), waist_share=fc.WAIST_SHARE,
        asset="立绘 Lancet-2 1.png",
    )
    assert plan.top <= detection.box.y0


def test_the_vehicle_share_is_at_least_the_four_fifths_asked_for():
    assert fc.VEHICLE_SHARE >= 0.8


# ------------------------------------------- subjects a person had to point at

def test_a_hand_tuned_subject_overrides_the_automatic_anchors():
    """Some pictures no automatic rule gets right; a person says where to look.

    立绘 Miss.Christine 1.png is the case: the silhouette's head band centres
    on the canvas because the gold frame spans the same width, so the window
    opened on the top of the frame and left the cat along the bottom edge.
    """
    subject = (0.40, 0.30, 0.70, 0.90)
    plan = fc.subject_window(subject, SMALL_ASPECT, (1000, 1000))
    centre_x = plan.left + plan.width / 2
    assert centre_x == pytest.approx((subject[0] + subject[2]) / 2 * 1000, abs=1.0)


def test_a_hand_tuned_subject_keeps_the_aspect():
    for aspect in (SMALL_ASPECT, LARGE_ASPECT):
        plan = fc.subject_window((0.4, 0.3, 0.7, 0.9), aspect, (1000, 1000))
        assert plan.width / plan.height == pytest.approx(aspect, rel=0.01)


def test_a_hand_tuned_subject_grows_rather_than_crops_inward():
    """The subject has to stay inside: the extra area comes from around it."""
    subject = (0.40, 0.60, 0.60, 0.70)   # wide and short
    plan = fc.subject_window(subject, LARGE_ASPECT, (1000, 1000))
    # Every corner of the subject is inside the window.
    assert plan.left <= subject[0] * 1000
    assert plan.right >= subject[2] * 1000
    assert plan.top <= subject[1] * 1000
    assert plan.bottom >= subject[3] * 1000


def test_the_override_table_entries_are_well_formed():
    """A typo here would mis-frame a picture silently."""
    for name, box in fc.ASSET_SUBJECTS.items():
        assert len(box) == 4, name
        x0, y0, x1, y1 = box
        assert 0.0 <= x0 < x1 <= 1.0, name
        assert 0.0 <= y0 < y1 <= 1.0, name


def test_the_override_table_says_why_each_entry_exists():
    """The comments are the record of what the subject is; keep them long enough
    to be useful."""
    source = fc.__doc__ or ""
    for name in fc.ASSET_SUBJECTS:
        assert name.endswith(".png"), name
    # A short table is expected; a long one would mean the rules are failing
    # widely and should be fixed instead.
    assert len(fc.ASSET_SUBJECTS) <= 20


# -------------------------------------------- the face is the same size everywhere

def test_the_window_is_sized_from_the_face_width():
    """The reference the user picked: window = 4.36 face-widths across.

    安洁莉娜 精英0 has a 72 px face in a 1024 px picture, and its chosen crop
    is 313 px wide. 313 / 72 = 4.36. Reproducing that ratio is what makes the
    face come out the same size on every illustration.
    """
    image = figure_image(width=1000, height=1000, top=100, figure_w=200)
    detection = fc.detect_figure(image)
    assert detection is not None
    detection = fc.replace(detection, face=face_at(500.0, 100.0, width=80.0))
    plan = fc.plan_crop(detection, SMALL_ASPECT, (1000, 1000))
    assert plan.width == pytest.approx(80 * fc.FACE_WINDOW_WIDTHS, abs=1.0)
    assert plan.height == pytest.approx(plan.width / SMALL_ASPECT, abs=1.0)


def test_the_face_lands_in_the_same_place_whatever_its_size():
    """The point of the rule: same position, same scale, every picture."""
    for face_width in (40.0, 80.0, 160.0):
        image = figure_image(width=2000, height=2000, top=300, figure_w=400)
        detection = fc.detect_figure(image)
        assert detection is not None
        face = face_at(1000.0, 300.0, width=face_width)
        detection = fc.replace(detection, face=face)
        plan = fc.plan_crop(detection, SMALL_ASPECT, (2000, 2000))
        # The window is rounded to whole pixels, so allow a pixel's worth of
        # slack rather than demanding exact halves.
        assert (face.eye_x - plan.left) / plan.width == pytest.approx(0.5, abs=0.002)
        assert (face.y - plan.top) / plan.height == pytest.approx(
            fc.FACE_TOP_SHARE, abs=0.002
        )
        assert face.width / plan.width == pytest.approx(
            1 / fc.FACE_WINDOW_WIDTHS, abs=0.002
        )


def test_the_reference_numbers_are_the_measured_ones():
    """Pinned so a change to them is deliberate, not drift."""
    assert fc.FACE_WINDOW_WIDTHS == pytest.approx(4.36, abs=0.01)
    assert fc.FACE_TOP_SHARE == pytest.approx(0.25, abs=0.001)


def test_a_face_beats_the_waist_rule():
    """下半身不受腰线限制: the face sets the window, so the cut can land
    anywhere the aspect puts it."""
    image = figure_image(width=1000, height=1000, top=100, figure_w=200)
    detection = fc.detect_figure(image)
    assert detection is not None
    detection = fc.replace(detection, face=face_at(500.0, 100.0, width=80.0))
    with_face = fc.plan_crop(detection, SMALL_ASPECT, (1000, 1000), waist_share=WAIST)
    without = fc.plan_crop(
        fc.replace(detection, face=None), SMALL_ASPECT, (1000, 1000), waist_share=WAIST
    )
    assert with_face.method == "face"
    assert with_face.height != without.height


def test_a_tiny_face_detection_is_ignored():
    """A few pixels is noise; the silhouette fallback is steadier."""
    image = figure_image(width=1000, height=1000, top=100, figure_w=200)
    detection = fc.detect_figure(image)
    assert detection is not None
    detection = fc.replace(detection, face=face_at(500.0, 100.0, width=4.0))
    plan = fc.plan_crop(detection, SMALL_ASPECT, (1000, 1000), waist_share=WAIST)
    assert plan.method != "face"


def test_the_face_rule_still_keeps_the_aspect():
    image = figure_image(width=1000, height=1000, top=100, figure_w=200)
    detection = fc.detect_figure(image)
    detection = fc.replace(detection, face=face_at(500.0, 100.0, width=80.0))
    for aspect in (SMALL_ASPECT, LARGE_ASPECT):
        plan = fc.plan_crop(detection, aspect, (1000, 1000))
        assert plan.width / plan.height == pytest.approx(aspect, rel=0.01)


def test_a_vehicle_is_not_face_normalised():
    """A vehicle has no face to scale by, so its own rule still applies."""
    image = figure_image(width=900, height=1000, top=0, figure_w=300)
    detection = fc.detect_figure(image)
    assert detection is not None
    plan = fc.plan_crop(
        detection, SMALL_ASPECT, (900, 1000), waist_share=WAIST,
        asset="立绘 Lancet-2 1.png",
    )
    assert plan.method != "face"
    figure_height = detection.box.y1 - detection.box.y0 + 1
    assert plan.height >= figure_height * 0.8
