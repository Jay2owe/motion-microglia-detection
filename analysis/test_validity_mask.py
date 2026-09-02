"""Where measuring can be trusted, and the denominators that assumed everywhere.

This is the only part of the extra-inputs work that deliberately changes numbers
the package already reported, so it is the only one whose tests are mostly about
what must *not* change. The order below is the order of the risk:

1. **A movie with no mask must be bit-identical to before.** Almost every movie
   is one, and a mask nobody declared moving a density in the seventh decimal
   place would be a silent regression across every existing figure.
2. **Declaring a mask that excludes nothing must also change nothing**, which
   is what separates "the mask works" from "declaring a mask perturbs things".
3. Only then: a mask that excludes something changes exactly the columns that
   divide by a field, and no others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from analysis.config import MovieConfig
from analysis.io import load_valid_mask
from analysis.modules import neighbours, object_geometry, objects
from analysis.registry import MeasurementContext, ObjectStack
from analysis.summarise import build_frame_summary
from analysis.units import Scale

FRAMES, HEIGHT, WIDTH = 6, 24, 30
FIELD_PX = HEIGHT * WIDTH


def _movie(context: MeasurementContext, valid=None) -> MeasurementContext:
    context.valid = valid
    return context


def _cells(valid=None) -> MeasurementContext:
    """Four cells spread over the field, with one set of shapes to measure."""
    labels = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.uint16)
    for frame in range(FRAMES):
        labels[frame, 2:5, 2:5] = 1
        labels[frame, 2:5, 20:23] = 2
        labels[frame, 16:19, 3:6] = 3
        labels[frame, 16:19, 22:25] = 4
    shapes = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.int32)
    shapes[:, 10:14, 10:16] = 1
    return MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=[1, 2, 3, 4], valid=valid,
        objects={"wall": ObjectStack(name="wall", values=shapes, static=False,
                                     path="synthetic")},
        params={"object_geometry": {"dilation_px": 1, "radius_px": 6.0}},
    )


def _all_tables(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    tables = {}
    tables.update(neighbours.measure(context))
    tables.update(objects.measure(context))
    tables.update(object_geometry.measure(context))
    cell_frame = pd.DataFrame({
        "identity": np.repeat([1, 2, 3, 4], FRAMES),
        "frame_index": np.tile(range(FRAMES), 4),
    })
    tables["frame_summary"] = build_frame_summary(cell_frame, context, {})
    return tables


def _identical(left: dict[str, pd.DataFrame], right: dict[str, pd.DataFrame],
               ) -> list[str]:
    """Every column of every table, compared by value at zero tolerance."""
    changed = []
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        assert list(a.columns) == list(b.columns), name
        for column in a.columns:
            if pd.api.types.is_numeric_dtype(a[column]):
                if not np.allclose(a[column].astype(float), b[column].astype(float),
                                   rtol=0, atol=0, equal_nan=True):
                    changed.append(f"{name}.{column}")
            elif not a[column].astype(str).equals(b[column].astype(str)):
                changed.append(f"{name}.{column}")
    return changed


# ------------------------------------------------- what must not change at all

def test_a_movie_with_no_mask_measures_exactly_the_whole_field():
    """The compatibility contract, stated as arithmetic rather than as a hope.

    ``valid_px`` multiplies the two side lengths when no mask is declared,
    rather than summing an array of ``True``. Both give 720 here; only one of
    them is guaranteed to give the same float after a division as the code did
    before a mask existed.
    """
    context = _cells()
    assert context.valid is None
    assert context.valid_px(0) == FIELD_PX
    assert context.valid_frame(3).shape == (HEIGHT, WIDTH)
    assert context.valid_frame(3).all()


def test_declaring_a_mask_that_excludes_nothing_changes_nothing():
    """This is what separates "the mask works" from "a mask perturbs things"."""
    without = _all_tables(_cells())
    with_mask = _all_tables(_cells(valid=np.ones((HEIGHT, WIDTH), dtype=bool)))
    assert _identical(without, with_mask) == []


def test_an_all_true_stack_is_the_same_as_an_all_true_frame():
    """The per-frame path and the single-frame path must agree where they can."""
    single = _all_tables(_cells(valid=np.ones((HEIGHT, WIDTH), dtype=bool)))
    stacked = _all_tables(
        _cells(valid=np.ones((FRAMES, HEIGHT, WIDTH), dtype=bool)))
    assert _identical(single, stacked) == []


# --------------------------------------------------- what must change, and only that

def _margin(rows: int) -> np.ndarray:
    """A mask with ``rows`` rows blanked off the top, like a registration margin."""
    mask = np.ones((HEIGHT, WIDTH), dtype=bool)
    mask[:rows, :] = False
    return mask


def test_a_mask_changes_the_field_denominators_and_nothing_else():
    """The exact list of columns a mask is allowed to move.

    Anything else moving is a bug, and the test names what moved rather than
    just failing, because the interesting part of the failure is which column.
    """
    without = _all_tables(_cells())
    masked = _all_tables(_cells(valid=_margin(4)))
    allowed = {
        "neighbour_frame.neighbour_ground_px",
        "neighbour_frame.nearest_neighbour_expected_px",
        "neighbour_frame.nearest_neighbour_index",
        "neighbours.local_density",
        "neighbours.local_density_area_px",
        "objects.object_field_share",
        "cell_objects.object_cover_within_radius",
        "frame_summary.valid_px",
        "frame_summary.valid_share",
    }
    changed = set(_identical(without, masked))
    assert changed <= allowed, sorted(changed - allowed)
    # And the ones that must actually move, so the test cannot pass by the mask
    # being ignored.
    assert {"neighbour_frame.neighbour_ground_px",
            "neighbour_frame.nearest_neighbour_expected_px",
            "neighbour_frame.nearest_neighbour_index",
            "frame_summary.valid_px"} <= changed


def test_half_a_field_halves_the_ground_the_spacing_index_divides_by():
    half = np.zeros((HEIGHT, WIDTH), dtype=bool)
    half[: HEIGHT // 2, :] = True
    tables = _all_tables(_cells(valid=half))
    assert set(tables["neighbour_frame"]["neighbour_ground_px"]) == {FIELD_PX // 2}
    assert set(tables["frame_summary"]["valid_px"]) == {FIELD_PX // 2}
    assert set(tables["frame_summary"]["valid_share"]) == {0.5}


def test_a_smaller_study_area_makes_the_cells_look_more_regularly_spaced():
    """The direction, asserted, because the sign of this is easy to get backwards.

    A smaller study area means random placement would put the cells closer
    together, so the same observed spacing reads as more regular: the expected
    distance falls and the index rises.
    """
    whole = _all_tables(_cells())["neighbour_frame"]
    masked = _all_tables(_cells(valid=_margin(8)))["neighbour_frame"]
    assert (masked["nearest_neighbour_expected_px"]
            < whole["nearest_neighbour_expected_px"]).all()
    assert (masked["nearest_neighbour_index"]
            > whole["nearest_neighbour_index"]).all()


def test_a_per_frame_mask_varies_with_the_frame():
    """A stage that jolts blanks a different strip in every frame."""
    mask = np.ones((FRAMES, HEIGHT, WIDTH), dtype=bool)
    for frame in range(FRAMES):
        mask[frame, :frame, :] = False
    tables = _all_tables(_cells(valid=mask))
    expected = [FIELD_PX - frame * WIDTH for frame in range(FRAMES)]
    assert tables["frame_summary"]["valid_px"].tolist() == expected
    assert tables["neighbour_frame"]["neighbour_ground_px"].tolist() == expected


def test_the_mask_changes_denominators_and_never_drops_a_row():
    """Whether a cell in a blanked corner counts is a scientific decision."""
    inside_the_margin = _all_tables(_cells(valid=_margin(8)))
    assert len(inside_the_margin["neighbours"]) == 4 * FRAMES
    assert set(inside_the_margin["neighbours"]["identity"]) == {1, 2, 3, 4}
    assert len(inside_the_margin["cell_objects"]) == 4 * FRAMES


# ---------------------------------------------------------------- the loader

def _write(tmp_path, array, name="valid.tif"):
    path = tmp_path / name
    tifffile.imwrite(path, array)
    return path


def _spec(path) -> MovieConfig:
    return MovieConfig(stem="t", labels="l.tif", raw="r.tif", valid_mask=path)


def test_a_single_frame_mask_applies_to_every_label_frame(tmp_path):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[4:, :] = 1
    valid, record = load_valid_mask(_spec(_write(tmp_path, mask)),
                                    FRAMES, HEIGHT, WIDTH)
    assert valid.dtype == bool and valid.shape == (HEIGHT, WIDTH)
    assert record["per_frame"] is False
    assert record["valid_px_min"] == record["valid_px_max"] == (HEIGHT - 4) * WIDTH


def test_a_per_frame_mask_is_read_frame_by_frame(tmp_path):
    mask = np.ones((FRAMES, HEIGHT, WIDTH), dtype=np.uint8)
    mask[0, :5, :] = 0
    valid, record = load_valid_mask(_spec(_write(tmp_path, mask)),
                                    FRAMES, HEIGHT, WIDTH)
    assert valid.shape == (FRAMES, HEIGHT, WIDTH)
    assert record["per_frame"] is True
    assert record["valid_px_min"] == FIELD_PX - 5 * WIDTH
    assert record["valid_px_max"] == FIELD_PX


def test_the_mask_is_read_in_label_space_and_not_shifted_by_the_source_offset(tmp_path):
    """The trap ``unclaimed`` and ``provenance`` already avoid, asserted here.

    Reading this in source space would shift it by two frames on the pinned
    movie and produce a complete, confident, wrong answer with no column
    anywhere recording it.
    """
    mask = np.ones((FRAMES, HEIGHT, WIDTH), dtype=np.uint8)
    mask[0] = 0
    mask[0, 0, 0] = 1                       # not all false, so it loads
    movie = MovieConfig(stem="t", labels="l.tif", raw="r.tif",
                        valid_mask=_write(tmp_path, mask), source_frame_offset=2)
    valid, _ = load_valid_mask(movie, FRAMES, HEIGHT, WIDTH)
    assert int(valid[0].sum()) == 1         # frame 0 of the file is frame 0 here
    assert valid[1].all()


def test_a_mask_of_the_wrong_size_is_refused_naming_both_shapes(tmp_path):
    mask = np.ones((HEIGHT + 3, WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(24, 30\)"):
        load_valid_mask(_spec(_write(tmp_path, mask)), FRAMES, HEIGHT, WIDTH)


def test_a_mask_with_the_wrong_number_of_dimensions_is_refused(tmp_path):
    mask = np.ones((2, FRAMES, HEIGHT, WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError, match="single"):
        load_valid_mask(_spec(_write(tmp_path, mask)), FRAMES, HEIGHT, WIDTH)


def test_a_mask_that_marks_nothing_measurable_fails_loudly(tmp_path):
    """Refused at load, not left to divide by zero inside a square root."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError, match="no pixel as measurable"):
        load_valid_mask(_spec(_write(tmp_path, mask)), FRAMES, HEIGHT, WIDTH)


def test_a_frame_with_nothing_measurable_gives_a_blank_not_an_infinity():
    """A blank plots as absent; an infinity plots."""
    mask = np.ones((FRAMES, HEIGHT, WIDTH), dtype=bool)
    mask[2] = False
    frame = _all_tables(_cells(valid=mask))["neighbour_frame"]
    row = frame[frame["frame_index"] == 2].iloc[0]
    assert row["neighbour_ground_px"] == 0
    assert np.isnan(row["nearest_neighbour_expected_px"])
    assert np.isnan(row["nearest_neighbour_index"])
    assert np.isfinite(frame.loc[frame["frame_index"] == 3,
                                 "nearest_neighbour_index"]).all()


def test_the_frame_table_always_says_how_much_was_measurable():
    """Present and honest with no mask, rather than absent."""
    cell_frame = pd.DataFrame({"identity": [1], "frame_index": [0]})
    frames = build_frame_summary(cell_frame, _cells(), {})
    assert frames["valid_px"].tolist() == [FIELD_PX] * FRAMES
    assert frames["valid_share"].tolist() == [1.0] * FRAMES
