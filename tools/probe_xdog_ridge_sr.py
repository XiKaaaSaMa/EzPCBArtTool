# -*- coding: utf-8 -*-
"""Probe: the two new readings and the 超分 pre-step, measured against their originals.

Three questions had to be answered with numbers rather than opinions before the
线稿工坊 workspace's readings could be wired into this tool, and this script is
where the numbers come from:

1. **Are the ports the same algorithms?** ``pcbart.ridge`` reimplements
   ``skimage.filters.meijering`` on numpy + OpenCV so the tool does not carry
   scipy/networkx/imageio/tifffile (about 150 MB unpacked) for one reading;
   ``pcbart.upscale`` is the workspace's Anime4K vectorisation. Both are
   compared against their references here -- scikit-image is imported directly,
   the workspace's modules are imported from the workspace folder when it is
   present.
2. **What do the responses look like on our own artwork**, and where should the
   XDoG binarisation level sit? Kazuhito00's port returns a *soft* image and
   this tool's masks must be strictly binary, so a level has to be picked: the
   script scans a few and reports ink coverage.
3. **How expensive is each reading at the sizes the panel actually uses** --
   the live preview (bounded trace) and the settled pass (work budget). The
   ridge filter's answer is what forced the preview bound.

Outputs land in ``out/<date>test/`` per the workspace export rule.

Run from the repository root::

    set PCBART_DRAFTS_DIR=%TEMP%\\probe-drafts
    python tools\\probe_xdog_ridge_sr.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

WORKSPACE = Path(r"P:\Zcode开发专用文件夹\线稿test")
if WORKSPACE.exists():
    sys.path.insert(0, str(WORKSPACE))

from pcbart.image_prep import flatten_alpha, imread_unicode, imwrite_unicode  # noqa: E402
from pcbart.lineart import (  # noqa: E402
    LineArtParams,
    line_art,
    line_art_preview,
    xdog_response,
)
from pcbart.paths import asset_root  # noqa: E402
from pcbart.ridge import meijering_response  # noqa: E402
from pcbart.upscale import upscale  # noqa: E402

OUT = REPO / "out" / "2026.10.13test" / "_probe_line_readings"


def load(path: Path) -> np.ndarray:
    raw = imread_unicode(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"无法读取 {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = flatten_alpha(raw[:, :, :3], raw[:, :, 3])
    elif raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return raw


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - start) * 1000.0


def scale_to_long(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) == long_side:
        return image
    scale = long_side / max(h, w)
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    art = asset_root() / "明日方舟"
    drawn = load(art / "06-立绘-精英0" / "立绘 惊蛰 1.png")
    half_dir = art / "02-半身像-精英0"
    half = load(half_dir / sorted(half_dir.glob("*.png"))[0].name)
    print(f"立绘惊蛰 {drawn.shape}  半身像 {half.shape}")
    imwrite_unicode(str(OUT / "src_drawn.png"), drawn)
    imwrite_unicode(str(OUT / "src_half.png"), half)

    # ------------------------------------------------------- XDoG fidelity
    grey8 = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
    ours = xdog_response(grey8, sigma=1.4, k_sigma=1.6, gamma=0.98, epsilon=0.0, phi=10.0)
    try:
        from algorithms import xdog_kazuhito  # type: ignore

        reference = xdog_kazuhito.process(
            drawn,
            {"sigma": 1.4, "k_sigma": 1.6, "epsilon": 0.0, "phi": 10.0, "gamma": 0.98},
            size_scale=1.0,
        )
        difference = np.abs(reference.astype(np.float64) - ours)
        print(
            f"XDoG vs 线稿工坊：最大差 {difference.max():.3f}（参考实现量化到 uint8，"
            f"本级即为舍入）  平均差 {difference.mean():.4f}"
        )
    except ImportError:
        print("XDoG：线稿工坊不在本机，跳过对照（公式见 pcbart.lineart.xdog_response）")

    for level in (64, 128, 192):
        traced, ms = timed(
            line_art,
            drawn,
            None,
            LineArtParams(mode="xdog", xdog_level=level, silhouette=False),
        )
        ink = float((traced < 128).mean() * 100)
        print(f"XDoG 墨量 @阈值 {level}: {ink:.1f}%  ({ms:.0f} ms)")

    # ------------------------------------------------------ ridge fidelity
    grey01 = grey8.astype(np.float32) / 255.0
    sigmas = [1.0, 2.0, 3.0]
    ours = meijering_response(grey01, sigmas)
    try:
        from skimage.filters import meijering

        reference = meijering(grey01, sigmas=sigmas, black_ridges=True)
        print(f"脊线 vs scikit-image：最大差 {np.abs(reference - ours).max():.2e}")
    except ImportError:
        print("脊线：本机没有 scikit-image，跳过对照")
    try:
        from algorithms import ridge_skimage  # type: ignore

        reference = ridge_skimage.process(
            drawn,
            {"sigma_min": 1.0, "sigma_max": 3.0, "steps": 5, "black_ridges": True,
             "percentile": 90.0, "invert": True},
            1.0,
        )
        mask = line_art(drawn, None, LineArtParams(mode="ridge", silhouette=False))
        agree = float(((mask < 128) == (reference < 128)).mean() * 100)
        print(f"脊线掩膜 vs 线稿工坊（p50 对它的 p90）：逐像素一致 {agree:.2f}%")
    except ImportError:
        print("脊线：线稿工坊不在本机，跳过掩膜对照")

    # -------------------------------------------------------- cost by size
    print("\n耗时（工作分辨率 2000px）")
    big = scale_to_long(drawn, 2000)
    for mode in ("fill", "outline", "dodge", "xdog", "ridge"):
        params = LineArtParams(mode=mode, silhouette=False)
        _, full_ms = timed(line_art, big, None, params)
        _, preview_ms = timed(line_art_preview, big, None, params, max_side=760)
        print(f"  {mode:8s} 全分辨率 {full_ms:6.0f} ms　预览 {preview_ms:6.0f} ms")
        if mode in ("xdog", "ridge"):
            traced, _ = timed(line_art, big, None, params)
            imwrite_unicode(str(OUT / f"reading_{mode}_2000.png"), traced)

    # --------------------------------------------------- super-resolution
    print("\n超分（位图放大，非矢量）")
    for label, picture, factor in (
        ("half", half, 2),
        ("half", half, 4),
        ("drawn512", scale_to_long(drawn, 512), 2),
    ):
        up, ms = timed(upscale, picture, factor)
        print(
            f"  {label} ×{factor}: {picture.shape[:2]} -> {up.shape[:2]}  "
            f"{ms:.0f} ms  类型={type(up).__name__} dtype={up.dtype}"
        )
        imwrite_unicode(str(OUT / f"up{factor}_{label}.png"), up)

    print("输出目录:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
