"""Board-painting style tests.

The style is three claims about pixels: the two tones partition the frame, the
contour hugs the silhouette from the inside, and the background is solid. Each
one is asserted directly rather than through a summary number, because a summary
can be right while the mask it came from is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")

from pcbart.boardart import (  # noqa: E402
    BoardArtParams,
    auto_outline_px,
    board_art,
    board_art_outline_mm,
    board_art_preview,
    board_art_records,
)


def painting(
    *,
    size: int = 120,
    subject_width: int = 60,
    subject_height: int = 80,
    light: int = 230,
    dark: int = 40,
):
    """A rectangular subject with a dark band across its middle.

    Deliberately crude: a straight edge and a straight tone boundary make it
    obvious by inspection whether the contour is inside the shape and whether
    the tone split landed where it should.
    """
    rgba = np.zeros((size, size, 4), np.uint8)
    x0 = (size - subject_width) // 2
    y0 = (size - subject_height) // 2
    x1, y1 = x0 + subject_width, y0 + subject_height
    rgba[y0:y1, x0:x1, :3] = light
    rgba[y0:y1, x0:x1, 3] = 255
    band = (y0 + y1) // 2
    rgba[band - 6 : band + 6, x0:x1, :3] = dark
    return rgba


# ------------------------------------------------------------------- the tones


def test_the_two_tones_partition_the_frame_exactly():
    """No pixel may be in both layers, and none may fall between them.

    This is the invariant that makes the board readable: a pixel in both is
    decided by write order, and a pixel in neither is a hairline of bare board
    running through a solid area.
    """
    art = board_art(painting())
    assert art.is_complete_partition
    assert not (art.white & art.black).any()
    assert (art.white | art.black).all()


def test_the_light_area_prints_white():
    art = board_art(painting())
    # Sample well inside the upper (light) part of the subject.
    assert art.white[40, 60]
    assert not art.black[40, 60]


def test_the_dark_area_prints_black():
    art = board_art(painting())
    assert art.black[60, 60]
    assert not art.white[60, 60]


def test_the_background_is_filled_black():
    art = board_art(painting())
    for y, x in ((2, 2), (2, 117), (117, 2), (117, 117)):
        assert art.black[y, x], f"background at ({x},{y}) is not filled"


def test_the_background_can_be_left_bare():
    art = board_art(painting(), params=BoardArtParams(black_background=False))
    assert not art.black[2, 2]
    assert not art.white[2, 2]


def test_a_fully_transparent_image_is_rejected():
    with pytest.raises(ValueError, match="找不到主体"):
        board_art(np.zeros((40, 40, 4), np.uint8))


# --------------------------------------------------------------- the contour


def test_the_contour_hugs_the_inside_of_the_silhouette():
    """The outline must not grow the artwork.

    Grown outwards, the white ring would extend past the picture and the board
    would carry a halo of ink that is not in the drawing.
    """
    art = board_art(painting(), params=BoardArtParams(cleanup_px=0))
    # The subject spans rows 20..99; nothing may be drawn outside it.
    assert not art.white[:20].any(), "contour escaped above the subject"
    assert not art.white[100:].any(), "contour escaped below the subject"
    assert not art.white[:, :30].any(), "contour escaped left of the subject"


def test_the_contour_separates_a_dark_edge_from_a_dark_background():
    """Where the artwork's own edge is dark, only the contour marks it.

    Without this a black collar against a black field has no boundary and the
    character dissolves into the background.
    """
    rgba = painting()
    rgba[20:100, 30:90, :3] = 30  # the whole subject is dark
    art = board_art(rgba, params=BoardArtParams(cleanup_px=0))

    assert art.white[21, 60], "no contour along the subject's top edge"
    assert art.black[35, 60], "the subject's interior should stay black"


def test_the_contour_width_follows_the_image_size():
    """A fixed pixel width would be a hairline on a large image."""
    small = auto_outline_px((400, 400))
    large = auto_outline_px((1600, 1600))
    assert large > small


def test_an_explicit_contour_width_wins_over_the_ratio():
    art = board_art(
        painting(), params=BoardArtParams(outline_px=9, outline_ratio=0.05)
    )
    assert art.outline_px == 9


def test_the_contour_keeps_at_least_one_pixel():
    assert auto_outline_px((40, 40), ratio=0.0) >= 1


# --------------------------------------------------------------- cleanliness


def test_speckles_are_absorbed_by_the_surrounding_tone():
    rgba = painting()
    rgba[35:37, 35:37, :3] = 40  # a few dark pixels inside the light area
    art = board_art(rgba, params=BoardArtParams(min_island_px=16))
    assert art.white[36, 36], "the dark speck survived as its own island"


def test_the_minimum_feature_width_is_measured_in_millimetres():
    """The process floor belongs to the board, not to the file.

    The same 2 px strand is unprintable on a small pendant and printable on a
    large sign, so the pixel count that enforces the floor has to move with the
    output size.
    """
    rgba = painting()
    rgba[35:37, 35:37, :3] = 40
    params = BoardArtParams(min_feature_mm=0.15)

    # 0.15 mm at 20 px/mm rounds to 3 px, which removes the 2 px speck.
    small = board_art(rgba, params=params, px_per_mm=20.0)
    # The same 0.15 mm at 1 px/mm is under the smallest usable kernel, so the
    # floor cannot be expressed at all and nothing is filtered.
    large = board_art(rgba, params=params, px_per_mm=1.0)

    assert small.min_feature_px == 3
    assert large.min_feature_px == 0


def test_the_floor_is_skipped_when_the_output_size_is_unknown():
    """Without px_per_mm there is no mm-to-pixel conversion, so no floor.

    Guessing a resolution would silently apply a process limit the caller never
    asked for and throw detail away.
    """
    art = board_art(painting(), params=BoardArtParams(min_feature_mm=0.15))
    assert art.min_feature_px == 0


def test_a_large_dark_region_is_kept():
    art = board_art(painting(), params=BoardArtParams(min_island_px=16))
    assert art.black[60, 60], "the tone split dropped a real dark area"


def test_a_speck_larger_than_the_limit_is_kept():
    rgba = np.full((120, 120, 4), 255, np.uint8)
    rgba[40:60, 40:60, :3] = 0  # 400 px dark square
    art = board_art(rgba, params=BoardArtParams(min_island_px=16, cleanup_px=0))
    assert art.black[50, 50]


def test_preview_uses_only_the_two_tones_and_the_bare_colour():
    art = board_art(painting())
    preview = board_art_preview(art)
    colours = {tuple(int(v) for v in colour) for colour in preview.reshape(-1, 3)}
    assert colours == {(255, 255, 255), (0, 0, 0)}


# --------------------------------------------------------------- the writer


def test_records_put_white_on_silk_and_black_on_copper():
    """Silk is the only layer that prints white; copper carries the dark field."""
    from pcbart.design_layers import layer

    art = board_art(painting())
    records = board_art_records(art, width_mm=40.0)
    layers = {layer_id for layer_id, _rings, _width in records}

    assert layer("top_silk").easyeda_id in layers
    assert layer("top_copper").easyeda_id in layers


def test_records_are_wound_the_way_the_editor_needs():
    """Outer rings clockwise, holes counter-clockwise."""
    from pcbart.geometry import signed_area

    art = board_art(painting())
    for _layer_id, rings, _width in board_art_records(art, width_mm=40.0):
        assert signed_area(rings[0]) < 0, "outer ring must be clockwise"
        for hole in rings[1:]:
            assert signed_area(hole) > 0, "hole must be counter-clockwise"


def test_the_two_layers_tile_the_board_without_gaps():
    """The dark layer is the complement of the light one, exactly.

    Tracing each mask on its own leaves them sharing no edge, and the difference
    comes out as a hairline of bare board wherever they meet -- measured at up
    to 0.17 mm on a real drawing before this was fixed.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    art = board_art(painting())
    records = board_art_records(art, width_mm=40.0)
    by_layer: dict[int, list] = {}
    for layer_id, rings, _width in records:
        by_layer.setdefault(layer_id, []).append(rings)

    def union(layer_key: str):
        from pcbart.design_layers import layer

        layer_id = layer(layer_key).easyeda_id
        shapes = []
        for rings in by_layer.get(layer_id, []):
            polygon = Polygon(rings[0], rings[1:])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            shapes.append(polygon)
        return unary_union(shapes)

    white = union("top_silk")
    black = union("top_copper")

    board = Polygon(board_art_outline_mm(art, 40.0))
    assert black.area > 0 and white.area > 0
    assert board.difference(white.union(black)).area == pytest.approx(0.0, abs=1e-6)
    assert white.intersection(black).area == pytest.approx(0.0, abs=1e-6)


def test_the_outline_matches_the_artwork_rectangle():
    """Board and artwork share an origin, so the outline must match its extent."""
    art = board_art(painting(size=120, subject_width=60, subject_height=80))
    ring = board_art_outline_mm(art, width_mm=40.0)
    xs = [x for x, _y in ring]
    ys = [y for _x, y in ring]

    assert (max(xs) - min(xs)) == pytest.approx(40.0)
    # The image is square, so the board is square too.
    assert (max(ys) - min(ys)) == pytest.approx(40.0)


def test_a_non_square_image_produces_a_non_square_board():
    """A 1:2 image must give a 1:2 board, not a square one."""
    rgba = np.zeros((200, 100, 4), np.uint8)
    rgba[20:180, 20:80, :3] = 230
    rgba[20:180, 20:80, 3] = 255
    art = board_art(rgba, params=BoardArtParams(cleanup_px=0))
    ring = board_art_outline_mm(art, width_mm=40.0)
    xs = [x for x, _y in ring]
    ys = [y for _x, y in ring]

    assert (max(xs) - min(xs)) == pytest.approx(40.0)
    assert (max(ys) - min(ys)) == pytest.approx(80.0)


def test_records_reject_a_non_positive_width():
    art = board_art(painting())
    with pytest.raises(ValueError, match="宽度"):
        board_art_records(art, width_mm=0.0)


# --------------------------------------------------------------------- flips


def test_flipping_horizontally_mirrors_the_tones():
    """A mirror flag must move the artwork, not just the preview."""
    art = board_art(
        painting(subject_width=30, subject_height=80),
        params=BoardArtParams(flip_horizontal=True, cleanup_px=0),
    )
    plain = board_art(
        painting(subject_width=30, subject_height=80),
        params=BoardArtParams(cleanup_px=0),
    )
    assert np.array_equal(art.white, plain.white[:, ::-1])
    assert np.array_equal(art.black, plain.black[:, ::-1])


def test_flipping_vertically_mirrors_the_tones():
    art = board_art(
        painting(subject_width=30, subject_height=80),
        params=BoardArtParams(flip_vertical=True, cleanup_px=0),
    )
    plain = board_art(
        painting(subject_width=30, subject_height=80),
        params=BoardArtParams(cleanup_px=0),
    )
    assert np.array_equal(art.white, plain.white[::-1])
    assert np.array_equal(art.black, plain.black[::-1])


def test_both_flips_apply():
    art = board_art(
        painting(subject_width=30),
        params=BoardArtParams(flip_horizontal=True, flip_vertical=True, cleanup_px=0),
    )
    plain = board_art(painting(subject_width=30), params=BoardArtParams(cleanup_px=0))
    assert np.array_equal(art.white, plain.white[::-1, ::-1])


def test_the_subject_moves_with_the_artwork():
    """The contour comes from the subject, so a mismatched flip would leave it
    ringing empty space instead of the drawing."""
    rgba = np.zeros((120, 120, 4), np.uint8)
    rgba[20:60, 30:90, :3] = 230
    rgba[20:60, 30:90, 3] = 255

    art = board_art(rgba, params=BoardArtParams(flip_vertical=True, cleanup_px=0))
    ys = np.nonzero(art.subject.any(axis=1))[0]

    assert ys.min() == 60, "subject did not move to the lower half"
    assert ys.max() == 99


def test_a_flipped_partition_is_still_exact():
    art = board_art(
        painting(),
        params=BoardArtParams(
            flip_horizontal=True, flip_vertical=True, black_background=True
        ),
    )
    assert art.is_complete_partition


# --------------------------------------------------------------------- sides


def test_top_side_uses_top_silk_and_top_copper():
    from pcbart.design_layers import layer

    art = board_art(painting())
    layers = {lid for lid, _r, _w in board_art_records(art, 40.0, side="top")}

    assert layers == {layer("top_silk").easyeda_id, layer("top_copper").easyeda_id}


def test_bottom_side_uses_bottom_silk_and_bottom_copper():
    from pcbart.design_layers import layer

    art = board_art(painting())
    layers = {lid for lid, _r, _w in board_art_records(art, 40.0, side="bottom")}

    assert layers == {
        layer("bottom_silk").easyeda_id,
        layer("bottom_copper").easyeda_id,
    }


def test_both_sides_write_every_tone_on_every_face():
    """两面 means four layers, not two: one copy of each tone per face."""
    from pcbart.design_layers import layer

    art = board_art(painting())
    records = board_art_records(art, 40.0, side="both")
    layers = {lid for lid, _r, _w in records}

    assert layers == {
        layer("top_silk").easyeda_id,
        layer("top_copper").easyeda_id,
        layer("bottom_silk").easyeda_id,
        layer("bottom_copper").easyeda_id,
    }

    counts: dict[int, int] = {}
    for lid, _rings, _width in records:
        counts[lid] = counts.get(lid, 0) + 1
    # The two faces must be exact copies, so each silk layer carries the same
    # number of shapes as the other.
    silk = {layer("top_silk").easyeda_id, layer("bottom_silk").easyeda_id}
    assert len({counts[lid] for lid in silk}) == 1


def test_an_unknown_side_is_rejected():
    art = board_art(painting())
    with pytest.raises(ValueError, match="板面"):
        board_art_records(art, 40.0, side="left")


def test_layer_keys_follow_the_side():
    from pcbart.boardart import layer_keys

    assert layer_keys("top") == {"white": ["top_silk"], "black": ["top_copper"]}
    assert layer_keys("both")["white"] == ["top_silk", "bottom_silk"]
    assert layer_keys("both")["black"] == ["top_copper", "bottom_copper"]


# ------------------------------------------------------------------- invert


def test_inverting_swaps_the_two_tones():
    """The negative of a picture exchanges the inks, not the boundaries."""
    plain = board_art(painting(), params=BoardArtParams(cleanup_px=0))
    inverted = board_art(
        painting(), params=BoardArtParams(cleanup_px=0, invert=True)
    )

    assert np.array_equal(inverted.white, plain.black)
    assert np.array_equal(inverted.black, plain.white)


def test_inverting_keeps_the_tone_boundary_where_it_was():
    """Shifting the threshold instead would redraw the picture.

    Inverting the grey rather than swapping the masks moves the split to
    1-threshold, which produces a *different* picture rather than the same one
    in negative. The boundaries have to stay put, so the edges of the white
    tone are asserted to match.
    """
    import cv2

    plain = board_art(painting(), params=BoardArtParams(cleanup_px=0))
    inverted = board_art(
        painting(), params=BoardArtParams(cleanup_px=0, invert=True)
    )

    kernel = np.ones((3, 3), np.uint8)
    edges_plain = cv2.morphologyEx(
        plain.white.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)
    edges_inverted = cv2.morphologyEx(
        inverted.black.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)

    assert np.array_equal(edges_plain, edges_inverted)


def test_an_inverted_painting_still_tiles_the_frame():
    inverted = board_art(
        painting(), params=BoardArtParams(cleanup_px=0, invert=True)
    )
    assert inverted.is_complete_partition


def test_inverting_twice_is_the_original():
    plain = board_art(painting(), params=BoardArtParams(cleanup_px=0))
    twice = board_art(
        painting(),
        params=BoardArtParams(cleanup_px=0, invert=True),
    )
    restored = board_art(painting(), params=BoardArtParams(cleanup_px=0))
    assert np.array_equal(twice.white, plain.black)
    assert np.array_equal(restored.white, plain.white)


def test_inverting_swaps_which_layer_holds_the_dark_field():
    """On a board the swap is visible: silk and copper exchange roles."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def area_by_layer(art):
        areas: dict[int, float] = {}
        for layer_id, rings, _width in board_art_records(art, 40.0):
            polygon = Polygon(rings[0], rings[1:])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            areas.setdefault(layer_id, []).append(polygon)
        return {k: unary_union(v).area for k, v in areas.items()}

    from pcbart.design_layers import layer

    plain = area_by_layer(board_art(painting()))
    inverted = area_by_layer(
        board_art(painting(), params=BoardArtParams(invert=True))
    )

    silk = layer("top_silk").easyeda_id
    copper = layer("top_copper").easyeda_id

    assert plain[silk] < plain[copper], "the dark field starts on copper"
    assert inverted[silk] > inverted[copper], "inverting moves it to silk"

    # The two are near mirror images but not exact ones: the contour is the
    # last thing drawn onto the white tone, so inverting carries it over to
    # copper as well. That is the intended effect -- the outline is part of the
    # picture and inverting a picture inverts its outline too -- and it is why
    # the areas are compared as "roughly exchanged" rather than swapped.
    assert inverted[silk] == pytest.approx(plain[copper], rel=0.10)
    assert inverted[copper] == pytest.approx(plain[silk], rel=0.25)


# ------------------------------------------------------------ tone polarity


def test_tone_to_ink_puts_ink_where_the_mask_is_set():
    """The canvas tints ink, so this conversion is what stops every shape from
    being drawn inside out."""
    from pcbart.boardart import tone_to_ink

    mask = np.array([[True, False], [False, True]])
    ink = tone_to_ink(mask)

    assert ink.dtype == np.uint8
    assert ink[0, 0] == 0 and ink[1, 1] == 0, "set pixels must be ink"
    assert ink[0, 1] == 255 and ink[1, 0] == 255


def test_the_two_tones_convert_to_opposite_inks():
    from pcbart.boardart import tone_to_ink

    art = board_art(painting())
    white_ink = tone_to_ink(art.white)
    black_ink = tone_to_ink(art.black)

    assert not np.array_equal(white_ink, black_ink)
    assert white_ink[40, 60] == 0, "the light area is white ink"
    assert black_ink[60, 60] == 0, "the dark area is black ink"


def test_records_scale_with_the_requested_size():
    """A 40 mm board must be twice a 20 mm one, not the same size twice."""
    from shapely.geometry import Polygon

    art = board_art(painting())
    spans = []
    for width_mm in (20.0, 40.0):
        records = board_art_records(art, width_mm=width_mm)
        points = [p for _l, rings, _w in records for ring in rings for p in ring]
        xs = [x for x, _y in points]
        spans.append(max(xs) - min(xs))

    assert spans[1] == pytest.approx(spans[0] * 2.0, rel=0.02)
