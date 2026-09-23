"""Denoising, as a step of its own.

Deliberately **not** part of :mod:`pcbart.lineart`. Blurring and line
extraction are separate operations on the image pipeline, and folding one into
the other made them impossible to judge or tune independently: changing the
denoise setting would silently change the artwork, and the line algorithm
could not be assessed on its own terms.

The pipeline is therefore explicit and ordered::

    source -> denoise_image() -> line_art() -> clean_binary() -> board

Each arrow is a separate call, so a change in one can be attributed to that
step and the output of each is inspectable on its own.

The Gaussian prefilter is a port of ``gaussian_blur.cpp`` from PCB_lightgraph
(MIT licensed), which blurs before edge detection so sensor noise and paper
grain do not turn into lines.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "ORIGINAL_MIN_KERNEL",
    "ORIGINAL_MAX_KERNEL",
    "MIN_KERNEL",
    "MAX_KERNEL",
    "MIN_SIGMA",
    "MAX_SIGMA",
    "blur_sigma_for_kernel",
    "gaussian_prefilter",
    "denoise_image",
    "clean_binary",
    "remove_specks",
    "remove_pinholes",
]


ORIGINAL_MIN_KERNEL = 3
ORIGINAL_MAX_KERNEL = 7
"""The kernel bounds PCB_lightgraph uses. Kept for reference.

The upper bound is raised to :data:`MAX_KERNEL` here because the original was
tuned against a downscaled interactive preview: fine-grain noise on a
full-resolution scan is proportionally larger and needs a wider kernel.
"""

MIN_KERNEL = 3
MAX_KERNEL = 31
"""Our bounds. The upper end is raised from the original 7 because it was
tuned against a downscaled preview; fine-grain noise on a full-resolution scan
is proportionally larger and needs a wider kernel to suppress."""

MIN_SIGMA = 0.3
MAX_SIGMA = 8.0


def blur_sigma_for_kernel(kernel_size: int) -> float:
    """A sensible sigma for a given kernel size.

    OpenCV derives sigma from the kernel when asked to, but the relation is
    opaque; this makes it explicit so a caller can reason about the pair.
    The original's default pairing (kernel 5, sigma 1.1) is close to
    ``kernel / 4.5``, which is what this reproduces.
    """
    return max(MIN_SIGMA, min(MAX_SIGMA, kernel_size / 4.5))


def denoise_image(
    image: np.ndarray,
    enabled: bool = True,
    kernel_size: int = 5,
    sigma: float | None = None,
) -> np.ndarray:
    """Step 1 of the pipeline: suppress noise in the source image.

    This is the point of the split. Whatever this returns is what
    :func:`pcbart.lineart.line_art` will trace, and it can be rendered on its
    own to see exactly what the line algorithm is being given.

    ``enabled=False`` returns the input unchanged, so the caller does not have
    to branch around it.
    """
    if not enabled:
        return image
    return gaussian_prefilter(image, kernel_size, sigma)


def clean_binary(
    binary: np.ndarray,
    min_blob_area: int = 0,
    max_hole_area: int = 0,
    foreground_is_dark: bool = True,
) -> np.ndarray:
    """Step 3 of the pipeline: tidy a binary image after line extraction.

    Both operations are off by default (area 0), because they change the
    artwork and should be a deliberate choice rather than a side effect of
    tracing.

    ``min_blob_area`` drops isolated specks; ``max_hole_area`` closes pinholes
    inside strokes.
    """
    result = binary
    if min_blob_area > 0:
        result = remove_specks(result, min_blob_area, foreground_is_dark)
    if max_hole_area > 0:
        result = remove_pinholes(result, max_hole_area, foreground_is_dark)
    return result


def gaussian_prefilter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float | None = None,
) -> np.ndarray:
    """Blur an image before edge extraction, to suppress noise.

    Works on greyscale, BGR or BGRA. ``kernel_size`` is forced odd and clamped;
    ``sigma`` defaults to a value derived from the kernel.
    """
    size = int(kernel_size)
    if size % 2 == 0:
        size -= 1
    size = int(np.clip(size, MIN_KERNEL, MAX_KERNEL))

    if sigma is None:
        sigma = blur_sigma_for_kernel(size)
    sigma = float(np.clip(sigma, MIN_SIGMA, MAX_SIGMA))

    # BORDER_REPLICATE matches the original's coordinate clamping.
    return cv2.GaussianBlur(
        image, (size, size), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )


def remove_specks(
    binary: np.ndarray,
    min_area: int = 8,
    foreground_is_dark: bool = True,
) -> np.ndarray:
    """Drop small isolated blobs from a binary image.

    ``min_area`` is in pixels. The default is deliberately small: it removes
    grain without touching thin strokes, which are long rather than small.
    """
    if min_area <= 1:
        return binary

    ink = (binary < 128) if foreground_is_dark else (binary >= 128)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(ink, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label

    if foreground_is_dark:
        return np.where(keep, 0, 255).astype(np.uint8)
    return np.where(keep, 255, 0).astype(np.uint8)


def remove_pinholes(
    binary: np.ndarray,
    max_area: int = 8,
    foreground_is_dark: bool = True,
) -> np.ndarray:
    """Fill tiny background holes inside ink.

    Thresholding a scan often leaves isolated light pixels inside a stroke.
    They cannot be printed, so they are closed up.
    """
    if max_area <= 0:
        return binary

    paper = (binary >= 128) if foreground_is_dark else (binary < 128)
    if not paper.any():
        return binary

    # Label the paper. A paper region touching the image border is the real
    # background; anything else is enclosed inside ink and therefore a pinhole.
    count, labels = cv2.connectedComponents(paper.astype(np.uint8), connectivity=8)
    border_labels = set(labels[0, :].tolist())
    border_labels |= set(labels[-1, :].tolist())
    border_labels |= set(labels[:, 0].tolist())
    border_labels |= set(labels[:, -1].tolist())
    border_labels.discard(0)

    fill = np.zeros_like(paper)
    for label in range(1, count):
        if label in border_labels:
            continue
        area = int((labels == label).sum())
        if area <= max_area:
            fill |= labels == label

    if not fill.any():
        return binary
    if foreground_is_dark:
        return np.where(fill, 0, binary).astype(np.uint8)
    return np.where(fill, 255, binary).astype(np.uint8)
