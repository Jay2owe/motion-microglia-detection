"""Cell-to-object geometry, checked against answers worked out by hand.

Nothing here compares the module with itself. Every expected number is one that
can be counted off the diagram in the fixture docstring, because a geometry
test that asserts what the code currently returns is a test that pins a bug as
firmly as it pins the behaviour.

The fixture is a square field with one wall down part of it and cells placed at
distances a person can add up: five pixels away, four pixels overlapping, wholly
inside. The contact-duration fixture is separate and deliberately has a hole in
it - a cell that is not seen for a frame must not have its run bridged.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.modules.object_geometry import measure
from analysis.registry import MeasurementContext, ObjectStack
from analysis.units import Scale

FIELD = 20
FRAMES = 4


def _stack(values: np.ndarray, name: str) -> ObjectStack:
    return ObjectStack(name=name, values=values.astype(np.int32), static=False,
                       path="synthetic", description="")


def _placed() -> MeasurementContext:
    """Three cells against one square wall, at distances that can be counted.

    Rows and columns 0-19. The wall is object 1, the four-by-four block at rows
    8-11 and columns 8-11::

        cell 1   rows 8-9,   cols 2-3    five columns clear of the wall
        cell 2   rows 10-12, cols 10-12  nine pixels, four of them on the wall
        cell 3   rows 8-9,   cols 8-9    four pixels, all four on the wall

    A second set, ``posts``, is one pixel at the top-left corner, so the two
    sets have different answers for every cell and cannot be confused.
    """
    labels = np.zeros((FRAMES, FIELD, FIELD), dtype=np.uint16)
    labels[:, 8:10, 2:4] = 1
    labels[:, 10:13, 10:13] = 2
    labels[:, 8:10, 8:10] = 3

    wall = np.zeros((FRAMES, FIELD, FIELD), dtype=np.int32)
    wall[:, 8:12, 8:12] = 1
    posts = np.zeros((FRAMES, FIELD, FIELD), dtype=np.int32)
    posts[:, 0, 0] = 1

    return MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=[1, 2, 3],
        objects={"wall": _stack(wall, "wall"), "posts": _stack(posts, "posts")},
        params={"object_geometry": {"dilation_px": 1, "radius_px": 5.0}},
    )


def _cell(table, identity: int, name: str, frame: int = 0):
    row = table[(table["identity"] == identity)
                & (table["object_set"] == name)
                & (table["frame_index"] == frame)]
    assert len(row) == 1, (identity, name, frame)
    return row.iloc[0]


# ------------------------------------------------------------------ distances

def test_a_cell_five_pixels_clear_of_a_wall_is_five_pixels_away():
    """Counted off the grid: column 3 to column 8 is five, same row."""
    cells = measure(_placed())["cell_objects"]
    row = _cell(cells, 1, "wall")
    assert row["nearest_object_edge_px"] == pytest.approx(5.0)
    assert row["nearest_object"] == 1
    assert row["object_overlap_px"] == 0
    assert row["object_overlap_share"] == 0.0
    assert bool(row["object_touching"]) is False


def test_centre_to_centre_is_not_the_same_question_as_outline_to_outline():
    """Cell 1's centre is (8.5, 2.5); the wall's is (9.5, 9.5)."""
    row = _cell(measure(_placed())["cell_objects"], 1, "wall")
    assert row["nearest_object_centre_px"] == pytest.approx(np.sqrt(50.0))
    assert row["nearest_object_centre"] == 1
    # Seven pixels apart at the centres, five at the edges. A figure drawn from
    # one of these is not a figure about the other.
    assert row["nearest_object_centre_px"] > row["nearest_object_edge_px"]


def test_a_cell_overlapping_a_wall_is_at_distance_zero_and_counted():
    """Rows 10-11 by columns 10-11 is four pixels of nine on the wall."""
    row = _cell(measure(_placed())["cell_objects"], 2, "wall")
    assert row["nearest_object_edge_px"] == 0.0
    assert row["object_overlap_px"] == 4
    assert row["object_overlap_share"] == pytest.approx(4 / 9)
    assert bool(row["object_touching"]) is True


def test_a_cell_wholly_inside_a_wall_reads_zero_and_one():
    """The pair that has to be read together: both cells above read distance 0.

    Distance alone cannot tell a cell brushing an object from a cell buried in
    one. The share is what separates 4/9 from 1.
    """
    row = _cell(measure(_placed())["cell_objects"], 3, "wall")
    assert row["nearest_object_edge_px"] == 0.0
    assert row["object_overlap_share"] == 1.0
    assert row["object_overlap_px"] == 4


def test_two_object_sets_are_measured_independently():
    """One pixel in the corner and a wall in the middle share no answers."""
    cells = measure(_placed())["cell_objects"]
    assert set(cells["object_set"]) == {"posts", "wall"}
    assert len(cells) == 3 * FRAMES * 2

    posts = _cell(cells, 1, "posts")
    # Cell 1's nearest pixel to the corner is (8, 2).
    assert posts["nearest_object_edge_px"] == pytest.approx(np.sqrt(68.0))
    assert posts["object_overlap_px"] == 0
    assert _cell(cells, 3, "posts")["object_overlap_share"] == 0.0
    # And the wall answer for the same cell-frame is a different number.
    assert posts["nearest_object_edge_px"] != _cell(cells, 1, "wall")[
        "nearest_object_edge_px"]


def test_the_threshold_and_the_count_ride_on_every_row():
    """Touching is a stated tolerance; nearest-of-four is not nearest-of-forty."""
    cells = measure(_placed())["cell_objects"]
    assert set(cells["dilation_px"]) == {1.0}
    assert set(cells["object_radius_px"]) == {5.0}
    assert set(cells.loc[cells["object_set"] == "wall", "object_count"]) == {1}


def test_the_neighbourhood_columns_use_the_declared_radius():
    """Cell 3 sits inside the wall, so its neighbourhood is largely wall."""
    cells = measure(_placed())["cell_objects"]
    inside = _cell(cells, 3, "wall")
    away = _cell(cells, 1, "wall")
    assert inside["objects_within_radius"] == 1     # the wall centre is 1.4 px off
    assert away["objects_within_radius"] == 0       # 7.07 px off, radius 5
    assert 0.0 < inside["object_cover_within_radius"] <= 1.0
    assert inside["object_cover_within_radius"] > away["object_cover_within_radius"]


def test_a_frame_with_no_object_in_it_is_blank_rather_than_far_away():
    """No shape is not a shape infinitely far off, and is certainly not zero."""
    context = _placed()
    empty = np.zeros((FRAMES, FIELD, FIELD), dtype=np.int32)
    context.objects = {"wall": _stack(empty, "wall")}
    row = _cell(measure(context)["cell_objects"], 1, "wall")
    assert np.isnan(row["nearest_object_edge_px"])
    assert np.isnan(row["nearest_object_centre_px"])
    assert row["object_count"] == 0
    assert bool(row["object_touching"]) is False


# ----------------------------------------------------------- contact duration

def _bouts() -> MeasurementContext:
    """A bar down the left, and two cells with holes in their contact.

    Six frames. Object 1 is rows 0-5, columns 0-1::

        cell 1  frames 0 1 . 3 4 .   touching (one pixel clear of the bar)
                frames . . 2 . . 5   eight columns away
        cell 2  frames 0 1 . 3 . .   touching
                frames . . 2 . 4 5   not on screen at all

    Cell 1's longest unbroken contact is two frames, not four. Cell 2's is two,
    not three: the frame it was not seen in breaks the run rather than bridging
    it, because nothing was observed to be touching during a frame nobody
    looked at.
    """
    frames, size = 6, 12
    labels = np.zeros((frames, size, size), dtype=np.uint16)
    for frame in range(frames):
        columns = slice(2, 4) if frame in (0, 1, 3, 4) else slice(8, 10)
        labels[frame, 0:2, columns] = 1
        if frame in (0, 1, 3):
            labels[frame, 4:6, 2:4] = 2
    bar = np.zeros((frames, size, size), dtype=np.int32)
    bar[:, 0:6, 0:2] = 1
    return MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=[1, 2], objects={"bar": _stack(bar, "bar")},
        params={"object_geometry": {"dilation_px": 1, "radius_px": 5.0}},
    )


def test_a_contact_that_stops_and_starts_again_is_two_bouts_not_one():
    tracks = measure(_bouts())["cell_object_tracks"]
    row = tracks[tracks["identity"] == 1].iloc[0]
    assert row["object_frames"] == 6
    assert row["object_contact_frames"] == 4
    assert row["object_contact_share"] == pytest.approx(4 / 6)
    assert row["object_longest_contact_frames"] == 2


def test_a_frame_the_cell_was_not_seen_in_does_not_bridge_a_run():
    tracks = measure(_bouts())["cell_object_tracks"]
    row = tracks[tracks["identity"] == 2].iloc[0]
    assert row["object_frames"] == 3                 # seen in three frames
    assert row["object_contact_frames"] == 3         # touching in all three
    assert row["object_contact_share"] == 1.0
    assert row["object_longest_contact_frames"] == 2


def test_contact_time_uses_the_configured_frame_interval():
    """Thirty minutes a frame here; a fifteen-minute recording is not this."""
    context = _bouts()
    half_hourly = measure(context)["cell_object_tracks"]
    context.scale = Scale(15.0)
    quarter_hourly = measure(context)["cell_object_tracks"]
    assert half_hourly.loc[0, "object_contact_hours"] == pytest.approx(2.0)
    assert quarter_hourly.loc[0, "object_contact_hours"] == pytest.approx(1.0)


def test_the_track_distances_summarise_the_frames_they_came_from():
    cells = measure(_bouts())["cell_objects"]
    tracks = measure(_bouts())["cell_object_tracks"]
    for identity in (1, 2):
        seen = cells[cells["identity"] == identity]["nearest_object_edge_px"]
        row = tracks[tracks["identity"] == identity].iloc[0]
        assert row["object_distance_median_px"] == pytest.approx(seen.median())
        assert row["object_distance_min_px"] == pytest.approx(seen.min())


# ------------------------------------------------------------ the declarations

def test_neither_table_folds_into_a_rollup():
    """Both carry ``object_set`` on top of a roll-up's grain."""
    import analysis.modules  # noqa: F401
    from analysis.registry import declared_tables
    from analysis.run import _fold_target

    declared = declared_tables()
    for name in ("cell_objects", "cell_object_tracks"):
        assert declared[name].fold is False, name
        assert _fold_target(name, declared) is None, name


def test_the_declared_grain_is_unique_on_a_real_measurement():
    """The grain is checked on the fixture, not only on the declaration."""
    tables = measure(_placed())
    assert not tables["cell_objects"].duplicated(
        subset=["identity", "frame_index", "object_set"]).any()
    assert not tables["cell_object_tracks"].duplicated(
        subset=["identity", "object_set"]).any()


def test_the_column_names_do_not_collide_with_the_cell_to_cell_ones():
    """``nearest_edge_px`` already means the nearest other *cell*.

    Reusing it and meaning a shape would make the axis label depend on which
    module happened to sort first. This collision has happened once before,
    between ``territory`` and ``contacts`` over ``overlap_px``.
    """
    from analysis.modules import neighbours, object_geometry

    theirs = {column.name for column in neighbours.PRODUCES}
    ours = {column.name for column in object_geometry.PRODUCES}
    assert not (theirs & ours)


def test_a_cell_inside_one_of_two_objects_names_the_one_it_is_inside():
    """The subtle property the whole module rests on, asserted rather than assumed.

    ``nearest_object`` is read off the distance transform's nearest-background
    index. For a pixel *inside* an object that index points at the pixel itself,
    so the answer is that object - which is right, and is not obvious from the
    call. If scipy ever returned something else, every overlapping cell would be
    labelled with a neighbouring shape at distance 0 and nothing would look odd.
    """
    labels = np.zeros((1, 12, 12), dtype=np.uint16)
    labels[0, 8:10, 8:10] = 1                 # sits inside object 9
    shapes = np.zeros((1, 12, 12), dtype=np.int32)
    shapes[0, 1:4, 1:4] = 5
    shapes[0, 7:11, 7:11] = 9
    context = MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=[1], objects={"two": _stack(shapes, "two")},
        params={"object_geometry": {"dilation_px": 1, "radius_px": 5.0}},
    )
    row = measure(context)["cell_objects"].iloc[0]
    assert row["nearest_object"] == 9
    assert row["nearest_object_edge_px"] == 0.0
    assert row["object_overlap_share"] == 1.0
    assert row["object_count"] == 2
