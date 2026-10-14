"""Anime4K upscaling -- the super-resolution step that runs before line extraction.

Why it is here, and where it sits
---------------------------------
The 线稿工坊 workspace offered this as an optional pre-step ("超分预处理") and
the instruction for this tool was explicit: **add it before the line-art step
only if the upscaler stays raster.** It does -- the output is a plain BGR
uint8 picture of twice (or four times) the input size, with no vector data
anywhere in the pipeline -- so it is wired in ahead of the reading, where it
buys the readings a finer grid to draw on.

Upstream: **Anime4K v0.9** by bloc97 (MIT), which states it uses no machine
learning; the Python reference implementation is TianZerL/Anime4KPython (MIT),
originally a per-pixel loop that its author notes is about 100x too slow for
real use. This is the workspace's vectorised rewrite of that reference, with
the same rules, order and uint8 truncation, carried over unchanged.

What one 2x level does
----------------------
``INTER_CUBIC`` 2x resize, then per pass: write the luma into the alpha
channel, refine line width along eight directional rules (``pushColor``),
write the Sobel gradient into alpha (``getGradient``), and sharpen along the
same eight directions (``pushGradient``). Neighbour reads come from a snapshot
taken at the start of each round (Jacobi-style), and every blend truncates to
uint8 the way the reference's C-style cast does.

⚠️ **One deliberate deviation, inherited from the workspace**: the reference's
``getGradient`` sums the Sobel response in uint8 and *wraps around* on bright
areas (its own runtime prints "overflow encountered in scalar multiply"). This
version computes the gradient in float64, which is both what the algorithm
means and what Anime4K's official GLSL / Anime4KCPP do. See the workspace's
README "移植说明与已知偏差".

⚠️ **Cost is seconds, not milliseconds.** Measured: 1.2 s for a 180x360
半身像 at x2, 4.4 s for a 512 px picture at x2, and roughly 17 s for 1024 px at
x2 -- it is the reason the panel computes it once per source instead of per
slider tick, and why the two size limits below exist.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["upscale", "SUPPORTED_FACTORS", "max_input_long_side"]

#: The factors the panel offers (the algorithm is defined as repeated 2x levels).
SUPPORTED_FACTORS = (2, 4)

#: Longest input side each factor accepts, in pixels.
#:
#: From the workspace's own limits, and they are about time rather than memory:
#: the cost grows with the *output* pixel count, so x4 on a 1024 px picture and
#: x2 on a 2048 px one both land around a minute. Past that the user is better
#: served by a smaller working resolution than by waiting.
_INPUT_LIMITS = {2: 2048, 4: 1024}


def max_input_long_side(factor: int) -> int:
    """The longest input side ``factor`` will accept."""
    return int(_INPUT_LIMITS.get(int(factor), 0))


def _pad(volume: np.ndarray) -> np.ndarray:
    """Edge-replicate by one pixel, the reference's own boundary clamping."""
    return np.pad(volume, ((1, 1), (1, 1), (0, 0)), mode="edge")


def _neighbours(padded: np.ndarray, height: int, width: int):
    """The eight-neighbourhood views, as the reference's tl/tc/tr/... names."""
    return (
        padded[0:height, 0:width],          # tl
        padded[0:height, 1:width + 1],      # tc
        padded[0:height, 2:width + 2],      # tr
        padded[1:height + 1, 0:width],      # ml
        padded[1:height + 1, 2:width + 2],  # mr
        padded[2:height + 2, 0:width],      # bl
        padded[2:height + 2, 1:width + 1],  # bc
        padded[2:height + 2, 2:width + 2],  # br
    )


def _write_luma(volume: np.ndarray) -> None:
    """``A = 0.299R + 0.587G + 0.114B`` (BGRA channel order), floored."""
    blue, green, red = volume[..., 0], volume[..., 1], volume[..., 2]
    volume[..., 3] = np.floor(0.299 * red + 0.587 * green + 0.114 * blue)


def _blend(current, first, second, third, strength, set_alpha_full=False):
    """The reference's ``mc*(1-s) + (a/3+b/3+c/3)*s``, floored to match its cast."""
    average = first / 3.0 + second / 3.0 + third / 3.0
    blended = current * (1.0 - strength) + average * strength
    if set_alpha_full:
        blended[..., 3] = 255.0
    return np.floor(blended)


def _push_color(volume: np.ndarray, strength: float) -> None:
    """Refine line width along eight directions, in the reference's order.

    The rules are applied in sequence and each one sees the *updated* centre
    pixel (``mcA`` is re-read between rules) while the neighbours come from the
    round's snapshot -- that combination is what the reference does, and it is
    why this cannot be reordered or vectorised across rules.
    """
    height, width = volume.shape[:2]
    snapshot = volume.copy()
    tl, tc, tr, ml, mr, bl, bc, br = _neighbours(_pad(snapshot), height, width)
    alpha = 3

    centre = volume[..., alpha]
    top_bottom = (
        np.minimum(np.minimum(tl[..., alpha], tc[..., alpha]), tr[..., alpha]) > centre
    ) & (centre > np.maximum(np.maximum(bl[..., alpha], bc[..., alpha]), br[..., alpha]))
    update = _blend(volume, tl, tc, tr, strength)
    volume[...] = np.where(top_bottom[..., None], update, volume)
    centre = volume[..., alpha]
    bottom_top = (~top_bottom) & (
        np.minimum(np.minimum(bl[..., alpha], bc[..., alpha]), br[..., alpha]) > centre
    ) & (centre > np.maximum(np.maximum(tl[..., alpha], tc[..., alpha]), tr[..., alpha]))
    update = _blend(volume, bl, bc, br, strength)
    volume[...] = np.where(bottom_top[..., None], update, volume)

    centre = volume[..., alpha]
    anti_diagonal = np.minimum(np.minimum(tc[..., alpha], tr[..., alpha]), mr[..., alpha]) > np.maximum(
        np.maximum(ml[..., alpha], centre), bc[..., alpha]
    )
    update = _blend(volume, tc, tr, mr, strength)
    volume[...] = np.where(anti_diagonal[..., None], update, volume)
    centre = volume[..., alpha]
    anti_diagonal_2 = (~anti_diagonal) & (
        np.minimum(np.minimum(ml[..., alpha], bl[..., alpha]), bc[..., alpha])
        > np.maximum(np.maximum(tc[..., alpha], centre), mr[..., alpha])
    )
    update = _blend(volume, ml, bl, bc, strength)
    volume[...] = np.where(anti_diagonal_2[..., None], update, volume)

    centre = volume[..., alpha]
    left_right = (
        np.minimum(np.minimum(tr[..., alpha], mr[..., alpha]), br[..., alpha]) > centre
    ) & (centre > np.maximum(np.maximum(tl[..., alpha], ml[..., alpha]), bl[..., alpha]))
    update = _blend(volume, tr, mr, br, strength)
    volume[...] = np.where(left_right[..., None], update, volume)
    centre = volume[..., alpha]
    right_left = (~left_right) & (
        np.minimum(np.minimum(tl[..., alpha], ml[..., alpha]), bl[..., alpha]) > centre
    ) & (centre > np.maximum(np.maximum(tr[..., alpha], mr[..., alpha]), br[..., alpha]))
    update = _blend(volume, tl, ml, bl, strength)
    volume[...] = np.where(right_left[..., None], update, volume)

    centre = volume[..., alpha]
    diagonal = np.minimum(np.minimum(mr[..., alpha], br[..., alpha]), bc[..., alpha]) > np.maximum(
        np.maximum(tc[..., alpha], centre), ml[..., alpha]
    )
    update = _blend(volume, mr, br, bc, strength)
    volume[...] = np.where(diagonal[..., None], update, volume)
    centre = volume[..., alpha]
    diagonal_2 = (~diagonal) & (
        np.minimum(np.minimum(ml[..., alpha], tl[..., alpha]), tc[..., alpha])
        > np.maximum(np.maximum(bc[..., alpha], centre), mr[..., alpha])
    )
    update = _blend(volume, ml, tl, tc, strength)
    volume[...] = np.where(diagonal_2[..., None], update, volume)


def _write_gradient(volume: np.ndarray, fast: bool) -> None:
    """``A = 255 - clamp(round(Sobel magnitude))``; border pixels keep their A."""
    alpha = volume[..., 3]
    padded = np.pad(alpha, 1, mode="edge")
    gradient_x = (
        padded[2:, 0:-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    )
    gradient_y = (
        padded[0:-2, 0:-2]
        + 2 * padded[1:-1, 0:-2]
        + padded[2:, 0:-2]
        - (padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
    )
    if fast:
        magnitude = (np.abs(gradient_x) + np.abs(gradient_y)) / 2.0
    else:
        magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    # ⚠️ float64 on purpose -- see the module docstring's deviation note.
    quantised = np.clip(np.floor(magnitude + 0.5), 0, 255)
    new_alpha = 255.0 - quantised
    updated = alpha.copy()
    updated[1:-1, 1:-1] = new_alpha[1:-1, 1:-1]
    volume[..., 3] = updated


def _push_gradient(volume: np.ndarray, strength: float) -> None:
    """Sharpen along the same eight directions, first match wins."""
    height, width = volume.shape[:2]
    snapshot = volume.copy()
    tl, tc, tr, ml, mr, bl, bc, br = _neighbours(_pad(snapshot), height, width)
    alpha = 3
    applied = np.zeros((height, width), dtype=bool)

    def apply_rule(condition, first, second, third) -> None:
        nonlocal applied
        matched = condition & (~applied)
        if not matched.any():
            return
        update = _blend(volume, first, second, third, strength, set_alpha_full=True)
        volume[...] = np.where(matched[..., None], update, volume)
        applied = applied | matched

    centre = volume[..., alpha]
    apply_rule(
        (np.minimum(np.minimum(tl[..., alpha], tc[..., alpha]), tr[..., alpha]) > centre)
        & (centre > np.maximum(np.maximum(bl[..., alpha], bc[..., alpha]), br[..., alpha])),
        tl, tc, tr,
    )
    centre = volume[..., alpha]
    apply_rule(
        (np.minimum(np.minimum(bl[..., alpha], bc[..., alpha]), br[..., alpha]) > centre)
        & (centre > np.maximum(np.maximum(tl[..., alpha], tc[..., alpha]), tr[..., alpha])),
        bl, bc, br,
    )
    centre = volume[..., alpha]
    apply_rule(
        np.minimum(np.minimum(tc[..., alpha], tr[..., alpha]), mr[..., alpha])
        > np.maximum(np.maximum(ml[..., alpha], centre), bc[..., alpha]),
        tc, tr, mr,
    )
    centre = volume[..., alpha]
    apply_rule(
        np.minimum(np.minimum(ml[..., alpha], bl[..., alpha]), bc[..., alpha])
        > np.maximum(np.maximum(tc[..., alpha], centre), mr[..., alpha]),
        ml, bl, bc,
    )
    centre = volume[..., alpha]
    apply_rule(
        (np.minimum(np.minimum(tr[..., alpha], mr[..., alpha]), br[..., alpha]) > centre)
        & (centre > np.maximum(np.maximum(tl[..., alpha], ml[..., alpha]), bl[..., alpha])),
        tr, mr, br,
    )
    centre = volume[..., alpha]
    apply_rule(
        (np.minimum(np.minimum(tl[..., alpha], ml[..., alpha]), bl[..., alpha]) > centre)
        & (centre > np.maximum(np.maximum(tr[..., alpha], mr[..., alpha]), br[..., alpha])),
        tl, ml, bl,
    )
    centre = volume[..., alpha]
    apply_rule(
        np.minimum(np.minimum(mr[..., alpha], br[..., alpha]), bc[..., alpha])
        > np.maximum(np.maximum(tc[..., alpha], centre), ml[..., alpha]),
        mr, br, bc,
    )
    centre = volume[..., alpha]
    apply_rule(
        np.minimum(np.minimum(ml[..., alpha], tl[..., alpha]), tc[..., alpha])
        > np.maximum(np.maximum(bc[..., alpha], centre), mr[..., alpha]),
        ml, tl, tc,
    )
    volume[..., 3] = 255.0


def _upscale_2x(
    bgr: np.ndarray, passes: int, strength_color: float, strength_gradient: float, fast: bool
) -> np.ndarray:
    """One 2x level: cubic resize plus the four-step refinement, ``passes`` times."""
    upscaled = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    volume = np.dstack(
        [upscaled.astype(np.float64), np.full(upscaled.shape[:2], 255.0, np.float64)]
    )
    for _ in range(passes):
        _write_luma(volume)
        _push_color(volume, strength_color)
        _write_gradient(volume, fast)
        _push_gradient(volume, strength_gradient)
    return np.clip(volume[..., :3], 0, 255).astype(np.uint8)


def upscale(
    image: np.ndarray,
    factor: int = 2,
    passes: int = 2,
    strength_color: float = 1.0 / 3.0,
    strength_gradient: float = 1.0,
    fast: bool = False,
) -> np.ndarray:
    """Anime4K upscale of a BGR uint8 picture. ``factor`` is 2 or 4.

    A **raster** result, and deliberately nothing more: no geometry leaves this
    function, which is what decides that the step belongs *before* line
    extraction rather than replacing it.
    """
    factor = int(factor)
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"超分倍数只支持 {SUPPORTED_FACTORS}，收到 {factor}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("超分只接受三通道 BGR 图")
    out = image
    for _ in range(factor // 2):
        out = _upscale_2x(out, passes, strength_color, strength_gradient, fast)
    return out
