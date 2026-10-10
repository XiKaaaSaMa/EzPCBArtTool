"""One answer to "how many pixels does this picture have".

Why this module exists
----------------------
The framing code needs the picture's pixel size to turn a window into a crop.
That question used to be answered by the canvas through ``QPixmap``, and a
``QPixmap`` that fails to decode -- a path with a character Qt's loader will not
take, or a file the loader has no plugin for -- comes back **null**, so the size
read ``(0, 0)``. Every caller treated ``(0, 0)`` as "nothing to do" and carried
on, which turned a hard error into a silent behaviour change:

* ``_fit_window_to_region`` skipped ``_set_window``, so the placement kept
  ``source_window = None`` -- meaning *the whole picture*;
* the export then placed the whole picture inside the illustration opening,
  shrinking it to fit instead of cropping it.

That is the reported defect: 「放置短区是强制把原图缩小，而非裁切」. The framing
looked applied because the box did take the region's size, and the smallest
evidence -- the window -- was the one part silently skipped.

⚠️ **So the size is read here, with Pillow, and a failure raises.** Pillow
decodes what the rest of the program writes, and it does not care about the
path's encoding. A caller that cannot get a size must decide what to do; it
must not inherit a quiet zero. A picture that genuinely cannot be measured
cannot be cropped to a region either, and reporting that is the only honest
answer.

The functions take only a path, with no Qt in sight, so the model layer, the
canvas and both writers can share them.
"""

from __future__ import annotations

__all__ = ["UnreadablePicture", "picture_size"]


class UnreadablePicture(RuntimeError):
    """A picture whose pixel size cannot be read."""


def picture_size(path: str) -> tuple[int, int]:
    """``(width_px, height_px)`` of the picture at ``path``.

    Raises :class:`UnreadablePicture` when the file is missing, empty, or not a
    picture Pillow can open. Callers that can carry on without a size should
    catch it and say so in their own terms; callers that cannot should let it
    out rather than proceeding with zeros, because a zero size silently becomes
    "no crop" a few call frames later.
    """
    if not path:
        raise UnreadablePicture("没有图源路径")
    from PIL import Image

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - every failure means "no pixels"
        raise UnreadablePicture(f"读不出图片尺寸：{path}（{exc}）") from exc
    if width <= 0 or height <= 0:
        raise UnreadablePicture(f"图片尺寸无效：{path}（{width}x{height}）")
    return int(width), int(height)
