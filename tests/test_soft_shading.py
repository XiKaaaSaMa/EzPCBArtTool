"""Softly shaded pictures ask for denoising before tracing.

The line algorithm's core measure is "is this pixel darker than its
neighbourhood" -- a question about *edges*. Across a smooth gradient there are
no edges, only a slow ramp, so the threshold chops the response's tiny wobbles
into hundreds of irregular rings. That is the artefact the user described as
"到处都是不规则的圈", and the fix is to blur before tracing, because a blur
flattens the wobble before the threshold ever sees it.

The panel now ticks 降噪 for such a picture on its own. It is a suggestion, not
a lock: the checkbox stays live so the raw trace is always available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gui_mockup  # noqa: E402
from gui_mockup import build_window, show_preprocess  # noqa: E402
from pcbart.image_prep import imwrite_unicode  # noqa: E402
from pcbart.lineart import (  # noqa: E402
    SOFT_SHADING_MIDDLE_SHARE,
    line_art,
    needs_denoising,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _soft_shaded(path: Path, size: int = 1400) -> str:
    """A shape whose interior is a gradient, like a shaded illustration.

    The gradient is the point: flat ink on flat paper is what the line
    algorithm is good at, and this is the opposite.

    ⚠️ **Size matters and is not arbitrary.** Measured fragment counts on this
    synthetic shape: 0 at 500 px, 136 after a 3 px blur at 800 px, 416 with no
    blur at 1400 px. A small version produces no artefact at all, so a test
    built on one would pass no matter what the code did. 1400 px is where the
    behaviour the real drawing showed actually appears.
    """
    image = np.zeros((size, size, 3), np.uint8)
    for y in range(size):
        image[y, :] = (
            min(255, 30 + y // 3),
            min(255, 40 + y // 4),
            min(255, 60 + y // 2),
        )
    centre = size // 2
    radius = int(size * 0.34)
    cv2.circle(image, (centre, centre), radius, (255, 255, 255), -1)
    # A ring of soft shading inside the white, so the interior has mid-tones.
    for r in range(int(radius * 0.9), int(radius * 0.35), -1):
        shade = int(255 * (r - radius * 0.35) / (radius * 0.55))
        cv2.circle(image, (centre, centre), r, (shade, shade, shade), 1)
    imwrite_unicode(str(path), image)
    return str(path)


def _flat_drawing(path: Path, size: int = 500) -> str:
    """Flat black ink on flat white paper: nothing in between."""
    image = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    cv2.circle(image, (c, c), int(size * 0.3), (0, 0, 0), 4)
    cv2.line(image, (int(size * 0.3), c), (int(size * 0.7), c), (0, 0, 0), 4)
    imwrite_unicode(str(path), image)
    return str(path)


def _fragments(mask: np.ndarray) -> int:
    """How many small pieces the ink is broken into."""
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask).astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas < 200).sum())


# ----------------------------------------------------------------- detection


def test_a_shaded_picture_is_detected(tmp_path):
    assert (
        needs_denoising(cv2.imread(_soft_shaded(tmp_path / "s.png"), cv2.IMREAD_UNCHANGED))
        is True
    )


def test_a_flat_drawing_is_not(tmp_path):
    assert (
        needs_denoising(
            cv2.imread(_flat_drawing(tmp_path / "f.png"), cv2.IMREAD_UNCHANGED)
        )
        is False
    )


def test_the_threshold_sits_well_above_what_flat_ink_produces(tmp_path):
    """Flat drawings measure 0.4-3.9% mid-tones; the cut is at 12%.

    Pinned so a future tweak cannot quietly start blurring clean line art,
    which loses detail for no gain.
    """
    flat = cv2.imread(_flat_drawing(tmp_path / "f.png"), cv2.IMREAD_UNCHANGED)
    grey = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    middle = float(((grey > 0.2) & (grey < 0.8)).mean())
    assert middle < SOFT_SHADING_MIDDLE_SHARE / 2, (
        f"平坦线稿的中间调占比 {100 * middle:.1f}%，离阈值太近了"
    )


# --------------------------------------------------------------------- effect


def test_the_fill_reading_leaves_almost_no_fragments():
    """The defect this file was written for is now largely gone by construction.

    ⚠️ **The old expectation no longer holds, and it is worth being explicit
    about why.** The fragments came from tracing *edges* across a gradient: the
    response wobbled around the threshold and shed hundreds of specks (227 on
    the full-size drawing). The fill reading puts every pixel on one side of a
    global threshold instead, so there is no wobbling response to fragment --
    measured on this fixture: **9 fragments**, against 63 when it was traced,
    and 227 on the full-size original.

    What was a "does denoising rescue this?" test is therefore now a "does the
    default already handle it?" test. The bound is the measured value with room
    to move, not a tight fit.
    """
    from pcbart.image_prep import imread_unicode

    fixture = Path(__file__).resolve().parent / "fixtures" / "soft_shading.png"
    image = imread_unicode(fixture, cv2.IMREAD_UNCHANGED)

    fragments = _fragments(line_art(image) == 0)

    assert fragments < 40, (
        f"填充读法不该产生大量碎块，实际 {fragments}"
    )


def test_loading_a_shaded_picture_ticks_denoise(app, tmp_path):
    window = build_window()
    try:
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.resize(1400, 900)
        window.show()
        app.processEvents()
        panel = show_preprocess(window)

        panel.load_image(_soft_shaded(tmp_path / "s.png"))
        app.processEvents()

        assert panel.denoise_check.isChecked() is True
        chosen = panel.denoise_slider.value()
        expected = round(1400 * gui_mockup.SOFT_SHADING_KERNEL_SHARE)
        assert chosen == expected, f"应当按短边比例选核，得到 {chosen}，期望 {expected}"
        assert chosen > 5, (
            "默认的 5px 核不够：实测 5px 还剩 74 个碎块，9px 才降到 31"
        )
    finally:
        window.hide()


def test_loading_a_flat_drawing_leaves_denoise_alone(app, tmp_path):
    window = build_window()
    try:
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.resize(1400, 900)
        window.show()
        app.processEvents()
        panel = show_preprocess(window)

        panel.load_image(_flat_drawing(tmp_path / "f.png"))
        app.processEvents()

        assert panel.denoise_check.isChecked() is False, "平坦线稿不该被模糊"
    finally:
        window.hide()


def test_the_denoise_choice_stays_the_users_to_make(app, tmp_path):
    """A hint the user cannot override is worse than the artefact it fixes."""
    window = build_window()
    try:
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.resize(1400, 900)
        window.show()
        app.processEvents()
        panel = show_preprocess(window)

        panel.load_image(_soft_shaded(tmp_path / "s.png"))
        app.processEvents()
        assert panel.denoise_check.isChecked() is True

        panel.denoise_check.setChecked(False)
        app.processEvents()
        assert panel.denoise_check.isChecked() is False, "用户应当能取消自动降噪"
    finally:
        window.hide()


def test_the_panel_says_why_it_changed_a_setting(app, tmp_path):
    """A control that moves on its own needs to explain itself."""
    window = build_window()
    try:
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.resize(1400, 900)
        window.show()
        app.processEvents()
        panel = show_preprocess(window)

        panel.load_image(_soft_shaded(tmp_path / "s.png"))
        app.processEvents()

        assert "降噪" in panel.process_note.text(), panel.process_note.text()
    finally:
        window.hide()
