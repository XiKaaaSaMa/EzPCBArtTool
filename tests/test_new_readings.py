"""The two readings added to 线稿调试: XDoG (Kazuhito00) and 脊线 (Meijering).

Both come from the 线稿工坊 workspace, where they were already ported and
checked; what these tests pin is their behaviour *in this pipeline*: strictly
binary output, a slider whose direction matches the readout, scale-invariant
size parameters, and the reading-specific preview bound that keeps the ridge
filter off the slider-timing path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.lineart import (  # noqa: E402
    RIDGE_PREVIEW_TRACE_SIDE,
    LineArtParams,
    line_art,
    line_art_preview,
    xdog_response,
)

from dataclasses import fields  # noqa: E402


def _drawing(shape=(400, 400)) -> np.ndarray:
    """A grey picture with a few dark strokes on light paper."""
    image = np.full(shape, 235, dtype=np.uint8)
    cv2.line(image, (40, 60), (360, 90), 40, 5)
    cv2.line(image, (60, 200), (340, 260), 20, 9)
    cv2.circle(image, (200, 330), 45, 60, 6)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _ink(mask: np.ndarray) -> float:
    return float((mask < 128).mean() * 100.0)


# ------------------------------------------------------------------ strictness


@pytest.mark.parametrize("mode", ["xdog", "ridge"])
def test_the_new_readings_return_strictly_binary_masks(mode):
    """Silkscreen cannot print a grey pixel, so 0/255 or nothing.

    The XDoG port returns a *soft* image upstream; this is where the pipeline
    turns it into a mask, so the guarantee has to be asserted here rather than
    assumed from the port.
    """
    traced = line_art(_drawing(), None, LineArtParams(mode=mode, silhouette=False))
    assert set(np.unique(traced).tolist()) <= {0, 255}


def test_an_unknown_reading_is_refused():
    """It used to fall through to dodge, which draws a plausible wrong picture."""
    with pytest.raises(ValueError):
        line_art(_drawing(), None, LineArtParams(mode="not-a-reading"))


# --------------------------------------------------------------------- XDoG


def test_the_xdog_level_moves_the_ink_the_way_the_slider_says():
    """Higher level = more ink, monotonically.

    The label promises it and the panel's readout repeats it, so a reversed
    implementation would be believed rather than tested; the ramp is drawn at
    five levels and compared as a chain of supersets, not just by totals.
    """
    picture = _drawing()
    masks = [
        line_art(
            picture, None, LineArtParams(mode="xdog", xdog_level=level, silhouette=False)
        )
        < 128
        for level in (40, 80, 128, 180, 230)
    ]
    shares = [float(m.mean() * 100.0) for m in masks]
    assert shares == sorted(shares), shares
    assert shares[0] < shares[-1]
    for smaller, larger in zip(masks, masks[1:]):
        assert not (smaller & ~larger).any(), "上一档的墨必须全部保留在下一档里"


def test_the_xdog_soft_response_runs_dark_on_dark_features():
    """The response is the upstream's own scale, ink low, paper high."""
    grey8 = cv2.cvtColor(_drawing(), cv2.COLOR_BGR2GRAY)
    response = xdog_response(grey8, sigma=1.4, k_sigma=1.6, gamma=0.98, epsilon=0.0, phi=10.0)
    assert response.shape == grey8.shape
    assert 0.0 <= response.min() and response.max() <= 255.0
    # The stroke drawn at grey 40 sits well below paper; a flat patch is paper.
    stroke = response[60:64, 100:140].min()
    flat = response[10:20, 10:20].mean()
    assert stroke < 128 < flat


def test_a_flat_picture_comes_back_as_paper():
    """Upstream's guard: no response anywhere means every pixel is paper."""
    flat = np.full((64, 64), 180, dtype=np.uint8)
    response = xdog_response(flat, sigma=1.4, k_sigma=1.6, gamma=0.98, epsilon=0.0, phi=10.0)
    assert np.array_equal(response, np.full((64, 64), 255.0))


def test_the_xdog_sigma_is_a_fraction_of_the_picture_not_a_pixel_count():
    """A fixed pixel sigma would draw thinner lines on a bigger picture.

    The parameter means "pixels at a 1000 px long side"; the same drawing
    sampled at different resolutions must therefore give nearly the same ink
    share. The tolerance is generous because resampling changes the strokes
    themselves -- it is here to catch a sigma that is *not* scaled at all,
    which lands tens of percent away.
    """
    small = _drawing((300, 300))
    large = cv2.resize(small, (900, 900), interpolation=cv2.INTER_CUBIC)
    shares = [
        _ink(line_art(pic, None, LineArtParams(mode="xdog", silhouette=False)))
        for pic in (small, large)
    ]
    assert abs(shares[0] - shares[1]) < 3.0, shares


# -------------------------------------------------------------------- ridge


def test_the_ridge_percentile_ranks_the_responding_pixels():
    """``100 - percentile``% of the pixels *that respond*, strongest first.

    ⚠️ **Not ``100 - percentile`` of the whole picture** -- that is the
    workspace panel's rule and it is the same answer on smoothly shaded
    artwork, but on hard-edged artwork (a logo, a flat colour field) most of
    the picture shares one ridge value, a percentile lands inside it, and the
    mask comes out as a black rectangle. See the trap test below.

    The responding share is measured through the panel-facing promise rather
    than recomputed: ``percentile=0`` means "ink everything that responds", so
    that run *is* the set the other runs rank.
    """
    picture = _drawing()
    everything = _ink(
        line_art(picture, None, LineArtParams(mode="ridge", ridge_percentile=0.0, silhouette=False))
    )
    assert everything > 1.0, "这张测试图应当有响应像素"

    for percentile in (30.0, 50.0, 90.0):
        traced = line_art(
            picture,
            None,
            LineArtParams(mode="ridge", ridge_percentile=percentile, silhouette=False),
        )
        expected = everything * (100.0 - percentile) / 100.0
        assert _ink(traced) == pytest.approx(expected, abs=0.2), percentile

    # The far end is honest too: "keep the strongest 0%" leaves only the peak
    # pixels themselves (a handful out of 160 000), not an empty mask -- they
    # are the ones that satisfy ``response >= max``.
    nothing = line_art(
        picture, None, LineArtParams(mode="ridge", ridge_percentile=100.0, silhouette=False)
    )
    assert _ink(nothing) < 0.01


def test_hard_edged_artwork_does_not_flood():
    """⚠️ The trap the ranking rule exists for, measured.

    A hard-edged drawing puts 95% of its pixels on one ridge value (zero, plus
    a float-sized floor). Ranking over the whole picture -- what the workspace
    panel does -- lands a 90th-percentile threshold inside that plateau and
    inks the entire page: measured 100.0% through scikit-image's own numbers,
    and 85.4% through an earlier version of this port. Ranking the responding
    pixels instead keeps the default setting usable on exactly the artwork this
    tool is fed (logos, flat-colour art).
    """
    picture = _drawing()
    traced = line_art(picture, None, LineArtParams(mode="ridge", silhouette=False))
    ink = _ink(traced)
    assert ink < 12.0, f"硬边图在默认档位下墨量 {ink:.1f}%，说明阈值落进了平坦区平台"


def test_a_blank_picture_produces_no_ink():
    blank = np.full((120, 160, 3), 255, dtype=np.uint8)
    traced = line_art(blank, None, LineArtParams(mode="ridge", silhouette=False))
    assert _ink(traced) == 0.0
    assert np.array_equal(traced, np.full((120, 160), 255, dtype=np.uint8))


def test_black_ridges_reaches_the_filter_and_changes_the_mask():
    """The checkbox is wired through to the model, and it matters.

    The *semantics* of the flag -- dark strands vs bright ones -- are pinned on
    the filter itself in ``tests/test_ridge.py``, where a thin strand gives
    0.311 under its own polarity and 0.000 under the other. At this level the
    strand is wide relative to the tiny scaled scales, so both polarities
    respond to *something*; what must hold here is that the flag arrives and
    changes the result rather than silently doing nothing.
    """
    picture = _drawing()
    dark = line_art(picture, None, LineArtParams(mode="ridge", silhouette=False))
    bright = line_art(
        picture, None, LineArtParams(mode="ridge", ridge_black=False, silhouette=False)
    )
    assert not np.array_equal(dark, bright)
    assert _ink(dark) > 0.5 and _ink(bright) > 0.5

    # And it really is the model's own field, not a panel-side re-run.
    params = LineArtParams(mode="ridge", ridge_black=False)
    assert params.normalised().ridge_black is False


def test_the_ridge_preview_traces_at_the_bounded_resolution():
    """⚠️ The live preview must not pay the full Hessian cost per slider tick.

    Measured before the bound: 733 ms at 2000 px for the response alone against
    36-58 ms for every other reading's whole pass. The bound is deliberately
    only on ``line_art_preview`` -- the settled pass still traces at the work
    budget -- and the check is **relative**, preview against the same picture's
    full trace, so it holds on a fast machine and a slow one alike.

    The picture is the full work budget (2000 px) rather than something
    smaller, because that is where the gap is wide: the preview traces at
    1000 px, a quarter of the area. Measured ratio 0.44-0.45 across runs;
    removing the bound puts both traces at 2000 px and the ratio at ~1.0, which
    is how this test was verified to catch the regression.
    """
    picture = cv2.resize(_drawing((400, 400)), (2000, 2000), interpolation=cv2.INTER_CUBIC)
    params = LineArtParams(mode="ridge", silhouette=False)
    # Warm up first: otherwise whichever trace runs first pays the library's
    # first-call cost and the ratio stops describing the bound.
    line_art_preview(cv2.resize(picture, (200, 200)), None, params, max_side=180)

    start = time.perf_counter()
    preview = line_art_preview(picture, None, params, max_side=760)
    preview_time = time.perf_counter() - start

    start = time.perf_counter()
    line_art(picture, None, params)
    full_time = time.perf_counter() - start

    assert max(preview.shape[:2]) <= 760
    assert preview_time < full_time * 0.75, (
        f"脊线预览 {preview_time:.2f}s 相对成品 {full_time:.2f}s 太慢，"
        "说明没有走限分辨率分支"
    )


def test_the_ridge_preview_still_looks_like_the_finished_artwork():
    """A cheaper preview is only acceptable while it shows the same drawing."""
    picture = cv2.resize(_drawing((400, 400)), (1400, 1400), interpolation=cv2.INTER_CUBIC)
    params = LineArtParams(mode="ridge", silhouette=False)
    preview = line_art_preview(picture, None, params, max_side=760)
    full = line_art(picture, None, params)
    full_small = cv2.resize(full, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA)
    a = preview < 128
    b = full_small < 128
    iou = float((a & b).sum()) / float((a | b).sum())
    assert iou > 0.75, f"预览与成品 IoU 只有 {iou:.3f}"


# ------------------------------------------------------- parameter plumbing


def test_normalising_carries_every_new_field():
    """⚠️ ``normalised()`` rebuilds the dataclass field by field.

    A field left out of it silently reverts to its default -- the slider would
    look live and do nothing, which is this panel's most-reported defect. The
    test is written over the dataclass itself so a *future* field added to the
    constructor without extending ``normalised()`` fails here as well.
    """
    from pcbart.lineart import LineArtParams as Params

    params = Params(
        mode="ridge",
        xdog_sigma=7.5,
        xdog_k=2.5,
        xdog_epsilon=-0.5,
        xdog_phi=30.0,
        xdog_gamma=0.5,
        xdog_level=200,
        ridge_sigma_min=2.5,
        ridge_sigma_max=8.0,
        ridge_steps=3,
        ridge_black=False,
        ridge_percentile=77.0,
    )
    normalised = params.normalised()
    for name in (
        "xdog_sigma",
        "xdog_k",
        "xdog_epsilon",
        "xdog_phi",
        "xdog_gamma",
        "xdog_level",
        "ridge_sigma_min",
        "ridge_sigma_max",
        "ridge_steps",
        "ridge_black",
        "ridge_percentile",
    ):
        assert getattr(normalised, name) == getattr(params, name), name

    # And the dataclass-vs-normaliser coverage itself: every field must be
    # mentioned in normalised(), or be one of the deprecated no-ops.
    source = Path("pcbart/lineart.py").read_text(encoding="utf-8")
    body = source.split("def normalised", 1)[1].split("def auto_radius", 1)[0]
    for item in fields(Params):
        if item.name in ("denoise", "denoise_kernel", "despeckle_area"):
            continue  # deprecated no-ops, deliberately reset
        assert f"{item.name}=" in body, f"{item.name} 没有写进 normalised()"
