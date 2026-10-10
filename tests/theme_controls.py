"""The check box and slider must carry the theme colour.

These live in their own module and are re-exported into ``test_theme_switch``
so the rendering helpers can be reused without the Qt-heavy file growing past
readability.

**Checked on rendered pixels, not on the stylesheet text.** The colour rules are
strings, so reading them back would pass even while the control drew the
platform's blue -- which is exactly what was happening: the custom proxy style
was installed and then silently discarded by ``QApplication.setStyleSheet``.
Nothing short of looking at what was painted would have shown it.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QCheckBox, QSlider

from pcbart.ui_theme import _hex_to_rgb, _on_colour, scheme_from_seed

import gui_mockup as g


def render_widget(window, widget):
    """The widget as drawn inside ``window``, as an ``(h, w, 4)`` RGBA array."""
    import numpy as np

    top_left = widget.mapTo(window, widget.rect().topLeft())
    # A little slack so an indicator's rounded corners are inside the crop. The
    # left/top are clamped to the window (a widget at 0,0 would otherwise ask
    # for a negative origin, and ``copy`` then returns a strip of the wrong
    # size); the right/bottom are clamped to what remains.
    left = max(0, top_left.x() - 2)
    top = max(0, top_left.y() - 2)
    right = min(window.width(), top_left.x() + widget.width() + 2)
    bottom = min(window.height(), top_left.y() + widget.height() + 2)
    image = (
        window.grab()
        .copy(QRect(left, top, right - left, bottom - top))
        .toImage()
        .convertToFormat(QImage.Format_RGBA8888)
    )

    # ⚠️ **Copied out of the QImage's buffer, not viewed through it.**
    # ``np.frombuffer`` returns an array that holds a pointer into memory Qt
    # owns, and the ``QImage`` here is a local -- once it is collected the
    # array dangles. Reading it afterwards segfaults the interpreter rather
    # than raising, which under pytest is a bare crash with no failing test.
    # Three of the five checks died that way before this line was added.
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height)
    return np.array(
        raw.reshape(height, stride)[:, : width * 4].reshape(height, width, 4)
    )


def colour_shares(pixels) -> dict[str, float]:
    """How much of the drawn area each opaque colour takes up."""
    total = pixels.shape[0] * pixels.shape[1]
    counts: dict[str, int] = {}
    for pixel in pixels.reshape(-1, 4):
        if pixel[3] < 200:
            continue
        key = f"#{pixel[0]:02X}{pixel[1]:02X}{pixel[2]:02X}"
        counts[key] = counts.get(key, 0) + 1
    return {colour: count / total for colour, count in counts.items()}


def share_of(shares: dict[str, float], target: str) -> float:
    """The share held by ``target``, counting near-matches from antialiasing.

    A flat fill produces the exact colour in its middle; the tolerance only
    matters for a shape so small that every pixel of it is on an edge.
    """
    wanted = QColor(target)
    total = 0.0
    for colour, share in shares.items():
        got = QColor(colour)
        distance = sum(
            abs(getattr(got, channel)() - getattr(wanted, channel)())
            for channel in ("red", "green", "blue")
        )
        if distance <= 12:
            total += share
    return total


def luminance(colour: str) -> float:
    red, green, blue = _hex_to_rgb(colour)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def make_check_box(window, app, checked: bool = True) -> QCheckBox:
    """A lone ticked check box, sized and positioned so it can be photographed.

    ⚠️ **The size has to be set explicitly.** A bare child of a window takes the
    window's whole geometry, so cropping "the widget" crops the entire window and
    the indicator's colour drowns in the background -- the share measured was
    0.0004 instead of the 0.05 a real indicator covers.
    """
    box = QCheckBox("")
    """No label text: the crop then covers the indicator alone.

    With text, the box is mostly glyphs and background, and the indicator's
    share of the area drops below any threshold that would also catch a genuine
    failure -- the test would have to be tuned so loosely that it stopped
    discriminating.
    """
    box.setParent(window)
    box.resize(24, 24)
    box.show()
    box.setChecked(checked)
    app.processEvents()
    return box


def make_slider(window, app, value: int = 60) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setParent(window)
    slider.resize(160, 24)
    slider.setRange(0, 100)
    slider.setValue(value)
    slider.show()
    app.processEvents()
    return slider


def check_box_fill_share(window, app, scheme) -> float:
    """How much of a ticked box is painted in the scheme's accent."""
    box = make_check_box(window, app)
    try:
        return share_of(colour_shares(render_widget(window, box)), scheme["accent"])
    finally:
        box.deleteLater()


def slider_fill_share(window, app, scheme) -> float:
    """How much of a slider is painted in the scheme's accent."""
    slider = make_slider(window, app)
    try:
        return share_of(colour_shares(render_widget(window, slider)), scheme["accent"])
    finally:
        slider.deleteLater()


__all__ = [
    "check_box_fill_share",
    "colour_shares",
    "luminance",
    "make_check_box",
    "make_slider",
    "render_widget",
    "scheme_from_seed",
    "share_of",
    "slider_fill_share",
    "_on_colour",
]
