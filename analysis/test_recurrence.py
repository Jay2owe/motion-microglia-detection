"""The recurrence arithmetic, and the four ways it can be handed nothing usable.

The numbers this module writes used to be computed inside a figure for sixteen
cells and thrown away, so they were never checked against anything. Two of the
tests below work a recurrence matrix out on paper; the rest are the edges that a
figure never met because it only ever drew the sixteen best-observed cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.modules.recurrence import derive, recurrence_runs, state_matrix
from analysis.registry import MeasurementContext
from analysis.units import Scale

#: The metrics the module asks for by default, in its own order.
METRICS = ["area_px", "circularity", "solidity", "ramification_index",
           "aspect_ratio", "skeleton_branches", "turnover_index",
           "step_px_gapless", "punctateness"]


def _context(minutes: float = 30.0, params: dict | None = None) -> MeasurementContext:
    """The little of a context a derived module actually reads.

    A derived module never opens an image, so the label stack here exists only
    to satisfy the dataclass: what is read is the frame interval and the
    module's settings.
    """
    return MeasurementContext(
        stem="synthetic",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        raw=np.zeros((2, 2, 2), dtype=np.uint16),
        scale=Scale(minutes),
        identities=[1],
        params={"recurrence": dict(params or {})},
    )


def _cell_frame(traces: dict[int, np.ndarray], minutes: float = 30.0) -> pd.DataFrame:
    """One cell per entry, every default metric taking the same trace.

    Every metric carrying one trace is deliberate: it makes the state vector a
    line through nine-dimensional space, so a distance between two frames is a
    fixed multiple of the difference between two trace values and the recurrence
    structure can be reasoned about from the trace alone.
    """
    rows = []
    for identity, values in traces.items():
        for frame, value in enumerate(values):
            rows.append({"identity": identity, "frame_index": frame,
                         "hours": frame * minutes / 60,
                         **{metric: float(value) for metric in METRICS}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- on paper


def test_a_matrix_of_isolated_points_has_no_structure() -> None:
    """Determinism and laminarity worked out by hand on a five-frame cell.

    Six recurrent points. Four of them lie in the two diagonal runs of length
    three either side of the main diagonal, and the pair at the far corner is on
    its own, so determinism is four in six. No column holds two adjacent points,
    so laminarity is zero: this cell repeated a sequence, it never sat still.
    """
    matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=bool)
    determinism, laminarity = recurrence_runs(matrix)
    assert determinism == pytest.approx(4 / 6)
    assert laminarity == 0.0


def test_a_matrix_of_held_states_is_laminar_and_deterministic_in_equal_measure() -> None:
    """Eight recurrent points, four in vertical runs and four in diagonal ones.

    Columns 0 and 3 each hold two adjacent points, which is the cell holding a
    state while time passes; the two offset-two diagonals hold two each, which is
    the cell repeating a sequence. Both come to a half.
    """
    matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ], dtype=bool)
    determinism, laminarity = recurrence_runs(matrix)
    assert determinism == pytest.approx(0.5)
    assert laminarity == pytest.approx(0.5)


def test_a_matrix_with_no_recurrent_points_is_zero_rather_than_undefined() -> None:
    """Nothing over nothing. The division has to be guarded, not attempted."""
    assert recurrence_runs(np.zeros((4, 4), dtype=bool)) == (0.0, 0.0)


def test_the_main_diagonal_never_counts_as_a_recurrence() -> None:
    """A frame resembling itself is not evidence of anything."""
    assert recurrence_runs(np.eye(5, dtype=bool)) == (0.0, 0.0)


# --------------------------------------------------------------- the edges


def test_a_cell_that_never_changes_does_not_divide_by_zero() -> None:
    """Zero spread in every metric is the guard in ``state_matrix``.

    Standardising divides by the spread, and a cell whose measurements never
    move has none. Dividing by one instead leaves that metric contributing
    nothing, which is the honest answer for a measurement that says nothing
    about this cell - and it must not be a ``nan`` or an infinity, because every
    distance downstream would inherit it.
    """
    hours = np.arange(8) / 2
    values = np.ones((8, 3))
    state = state_matrix(hours, values)
    assert np.isfinite(state).all()
    assert np.allclose(state, 0.0)

    tables = derive(_cell_frame({1: np.ones(12)}), _context(params={"surrogates": 3}))
    quantified = tables["recurrence_quantification"]
    assert len(quantified) == 1
    # Every frame is every other frame, so all 132 off-diagonal pairs of a
    # twelve-frame cell are recurrent. Two of them are in no run of two: the far
    # corners, whose diagonal holds one point, and the single points stranded
    # beside the main diagonal in the second and eleventh columns. 130 in 132.
    assert quantified["determinism"].iloc[0] == pytest.approx(130 / 132)
    assert quantified["laminarity"].iloc[0] == pytest.approx(130 / 132)


def test_a_cell_with_too_few_frames_gets_no_row_rather_than_a_row_of_blanks() -> None:
    """The convention ``rhythms`` already uses, so a table is filled in or absent.

    Four frames is the smallest matrix that can hold a diagonal run of two.
    """
    frame = _cell_frame({1: np.arange(3, dtype=float), 2: np.linspace(0, 5, 9)})
    tables = derive(frame, _context(params={"surrogates": 3}))
    assert sorted(tables["recurrence_quantification"]["identity"]) == [2]
    assert sorted(tables["recurrence"]["identity"].unique()) == [2]


def test_a_metric_that_is_not_in_the_table_is_dropped_and_counted() -> None:
    """A settings file naming a column this movie does not have must still run.

    ``recurrence_metrics_used`` is what says how much of the requested state
    vector was actually available, so a run against a movie missing three of the
    nine is readable rather than silently narrower.
    """
    frame = _cell_frame({1: np.linspace(0, 4, 10)})
    frame = frame.drop(columns=["punctateness", "skeleton_branches"])
    tables = derive(frame, _context(params={"surrogates": 3}))
    assert tables["recurrence_quantification"]["recurrence_metrics_used"].iloc[0] == 7


def test_no_requested_metric_present_writes_nothing_at_all() -> None:
    """An empty state vector is not a distance of zero; it is no measurement."""
    frame = _cell_frame({1: np.linspace(0, 4, 10)}).drop(columns=METRICS)
    assert derive(frame, _context()) == {}


def test_a_threshold_outside_zero_to_one_is_refused_by_name() -> None:
    """It is a fraction of frame pairs, and a distance would silently work."""
    frame = _cell_frame({1: np.linspace(0, 4, 10)})
    with pytest.raises(ValueError, match="threshold_quantile"):
        derive(frame, _context(params={"threshold_quantile": 2.5}))


# --------------------------------------------------------------- the shape


def test_the_lag_axis_is_capped_by_the_setting_and_by_the_cell() -> None:
    """Whichever runs out first: the requested window, or the cell's own frames."""
    frame = _cell_frame({1: np.linspace(0, 4, 10)})
    tables = derive(frame, _context(params={"surrogates": 3, "max_lag_hours": 2.0}))
    rates = tables["recurrence"]
    # Two hours at thirty minutes a frame is four lags, and the cell has nine.
    assert sorted(rates["lag_frames"]) == [1, 2, 3, 4]
    assert rates["lag_hours"].tolist() == [0.5, 1.0, 1.5, 2.0]
    assert rates["recurrence_pairs"].tolist() == [9, 8, 7, 6]

    short = derive(_cell_frame({1: np.linspace(0, 4, 5)}),
                   _context(params={"surrogates": 3, "max_lag_hours": 12.0}))
    assert sorted(short["recurrence"]["lag_frames"]) == [1, 2, 3, 4]


def test_a_cells_noise_band_does_not_depend_on_which_other_cells_were_measured() -> None:
    """The one deliberate departure from the figure this module replaced.

    The figure drew from a single generator in cell order, so a cell's matched-
    noise band changed when a different sixteen cells were drawn. Seeding per
    cell makes the band a property of the cell, which is the only way it can be
    quoted beside the cell's own recurrence rate.
    """
    alone = derive(_cell_frame({7: np.linspace(0, 4, 12)}),
                   _context(params={"surrogates": 5}))
    crowd = derive(_cell_frame({1: np.linspace(0, 9, 12), 7: np.linspace(0, 4, 12),
                                9: np.linspace(3, 0, 12)}),
                   _context(params={"surrogates": 5}))
    one = alone["recurrence"].set_index("lag_frames")["recurrence_surrogate_mean"]
    many = crowd["recurrence"]
    many = many[many["identity"] == 7].set_index("lag_frames")["recurrence_surrogate_mean"]
    assert np.allclose(one.to_numpy(), many.to_numpy(), rtol=0, atol=0, equal_nan=True)


def test_the_frame_interval_comes_from_the_movie_not_from_a_constant() -> None:
    """Fifteen-minute imaging must halve the hours a lag stands for."""
    frame = _cell_frame({1: np.linspace(0, 4, 10)}, minutes=15.0)
    tables = derive(frame, _context(minutes=15.0,
                                    params={"surrogates": 3, "max_lag_hours": 1.0}))
    assert tables["recurrence"]["lag_hours"].tolist() == [0.25, 0.5, 0.75, 1.0]
