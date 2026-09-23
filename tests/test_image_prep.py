"""Preprocessing tests: polarity detection, denoising, hole handling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.image_prep import (  # noqa: E402
    PrepParams,
    auto_polarity,
    binarize,
    imread_unicode,
    imwrite_unicode,
    load_gray,
    make_test_image,
    mask_to_preview,
    mask_to_rgb_preview,
    preprocess,
)


# ------------------------------------------------------- non-ASCII file paths


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG without going through cv2.imwrite's ANSI-only path handling."""
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def test_loads_an_image_from_a_chinese_path(tmp_path: Path):
    """Regression: cv2.imread returns None for non-ASCII paths on Windows.

    The project's own folder name contains Chinese characters, so a user
    picking a picture from a similarly named folder would have hit a failure
    that looks like a corrupt file.
    """
    folder = tmp_path / "中文目录"
    folder.mkdir()
    path = folder / "测试图案.png"
    _write_png(path, make_test_image(shape="blob"))

    gray, alpha = load_gray(path)

    assert gray.shape == (400, 400)
    assert alpha is None


def test_imread_unicode_reads_what_imdecode_would(tmp_path: Path):
    path = tmp_path / "路径带中文.png"
    _write_png(path, make_test_image(shape="star"))

    image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape[0] == 400


def test_imwrite_unicode_round_trips(tmp_path: Path):
    path = tmp_path / "写出的图.png"
    source = make_test_image(shape="ring")
    imwrite_unicode(path, source)

    assert path.exists()
    reread = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(reread, source)


def test_missing_file_message_names_the_path(tmp_path: Path):
    missing = tmp_path / "不存在.png"
    with pytest.raises(ValueError) as caught:
        load_gray(missing)
    # The message has to identify the file, otherwise the user cannot tell
    # which of several images failed.
    assert "不存在.png" in str(caught.value)


def test_undecodable_file_is_rejected(tmp_path: Path):
    bogus = tmp_path / "假图片.png"
    bogus.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="无法解码图片"):
        load_gray(bogus)


def test_rgb_preview_is_three_channel_and_scales():
    mask = np.zeros((10, 20), np.uint8)
    mask[2:8, 5:15] = 1

    plain = mask_to_rgb_preview(mask)
    assert plain.shape == (10, 20, 3)

    doubled = mask_to_rgb_preview(mask, scale=2.0)
    assert doubled.shape == (20, 40, 3)



def test_test_image_is_dark_subject_on_light_background():
    image = make_test_image(shape="blob")
    assert image.shape == (400, 400)
    # Corners are background.
    assert image[0, 0] > 200
    # Centre is subject.
    assert image[200, 200] < 100


def test_auto_polarity_detects_background_border():
    """A white border means the mask is already the right way round."""
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 1
    assert auto_polarity(mask) is False


def test_auto_polarity_detects_inverted_mask():
    """A foreground border means the polarity is wrong."""
    mask = np.ones((50, 50), np.uint8)
    mask[10:40, 10:40] = 0
    assert auto_polarity(mask) is True


def test_binarize_keeps_subject_as_foreground():
    image = make_test_image(shape="blob")
    mask, inverted = binarize(image, PrepParams())

    assert inverted is False
    assert mask[200, 200] == 1
    assert mask[0, 0] == 0


def test_binarize_handles_light_subject_on_dark_background():
    """An inverted source must be flipped automatically."""
    image = 255 - make_test_image(shape="blob")
    mask, inverted = binarize(image, PrepParams())

    assert inverted is True
    assert mask[200, 200] == 1, "subject should be foreground after inversion"


def test_explicit_invert_overrides_detection():
    image = make_test_image(shape="blob")
    mask, inverted = binarize(image, PrepParams(invert=True))
    assert inverted is True
    assert mask[200, 200] == 0


def test_preprocess_returns_binary_mask():
    result = preprocess(make_test_image(shape="blob"))
    assert set(np.unique(result.mask)).issubset({0, 1})
    assert result.foreground_ratio > 0.05


def test_preprocess_fills_holes_when_asked():
    """With fill_holes the ring becomes a disc, so the centre turns solid."""
    image = make_test_image(shape="ring")

    filled = preprocess(image, PrepParams(fill_holes=True, min_component_area=1))
    kept = preprocess(image, PrepParams(fill_holes=False, min_component_area=1))

    assert filled.mask[200, 200] == 1
    assert kept.mask[200, 200] == 0
    assert filled.mask.sum() > kept.mask.sum()


def test_preprocess_drops_small_speckles():
    image = make_test_image(shape="blob")
    noisy = image.copy()
    # A single-pixel speck plus a larger blob of noise.
    noisy[5, 5] = 0
    noisy[380:385, 380:385] = 0

    result = preprocess(
        noisy,
        PrepParams(morph_open=0, morph_close=0, min_component_area=25, fill_holes=False),
    )
    # Neither speck survives the area filter.
    assert result.mask[5, 5] == 0
    assert result.mask[382, 382] == 0


def test_preprocess_records_the_steps_it_applied():
    result = preprocess(make_test_image(shape="blob"))
    assert "load" not in result.steps  # array input, not a path
    assert any(step.startswith("threshold") for step in result.steps)


def test_preprocess_rejects_a_blank_result():
    blank = np.full((50, 50), 255, np.uint8)
    with pytest.raises(ValueError, match="没有任何前景"):
        preprocess(blank, PrepParams(min_component_area=1, morph_open=0, morph_close=0))


def test_loading_from_a_file(tmp_path: Path):
    path = tmp_path / "src.png"
    cv2.imwrite(str(path), make_test_image(shape="star"))

    result = preprocess(path)
    assert result.source_size == (400, 400)
    assert result.foreground_ratio > 0.05
    assert "load" in result.steps


def test_missing_file_is_reported(tmp_path: Path):
    with pytest.raises(ValueError, match="无法读取图片"):
        preprocess(tmp_path / "nope.png")


def test_alpha_channel_is_used_as_mask(tmp_path: Path):
    """A transparent PNG's alpha should drive the mask directly."""
    image = make_test_image(shape="blob")
    rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    alpha = np.where(image < 128, 255, 0).astype(np.uint8)
    rgba[:, :, 3] = alpha

    path = tmp_path / "alpha.png"
    cv2.imwrite(str(path), rgba)

    result = preprocess(path, PrepParams(fill_holes=False))
    assert "alpha" in result.steps
    assert result.mask[200, 200] == 1


def test_mask_to_preview_renders_black_on_white():
    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 2:5] = 1
    preview = mask_to_preview(mask)

    assert preview[3, 3] == 0
    assert preview[0, 0] == 255


def test_noise_is_suppressed_by_default_pipeline():
    clean = make_test_image(shape="blob")
    noisy = make_test_image(shape="blob", noise=0.02, seed=7)

    result = preprocess(noisy, PrepParams(min_component_area=64))
    clean_result = preprocess(clean, PrepParams(min_component_area=64))

    # The cleaned mask should be close to the clean one, not wildly different.
    difference = np.mean(result.mask != clean_result.mask)
    assert difference < 0.05, f"noise changed {difference * 100:.1f}% of pixels"
