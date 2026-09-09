"""The robust and least-squares drift arithmetic."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from analysis.modules.trend import derive
from analysis.registry import MeasurementContext
from analysis.units import Scale

METRIC = "area_px"


def _context(params: dict | None = None, minutes: float = 30.0) -> MeasurementContext:
    """The little of a context a derived module reads: the settings, and nothing else."""
    return MeasurementContext(
        stem="synthetic",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        raw=np.zeros((2, 2, 2), dtype=np.uint16),
        scale=Scale(minutes),
        identities=[1],
        params={"trend": dict(params or {})},
    )


def _trace(values: np.ndarray, minutes: float = 30.0, identity: int = 1) -> pd.DataFrame:
    hours = np.arange(len(values)) * minutes / 60
    return pd.DataFrame({"identity": identity, "frame_index": np.arange(len(values)),
                         "hours": hours, METRIC: values.astype(float)})


def _one(frame: pd.DataFrame, params: dict | None = None) -> pd.Series:
    table = derive(frame, _context({"metrics": [METRIC], **(params or {})}))["trend"]
    assert len(table) == 1, table
    return table.iloc[0]


# ------------------------------------------------------------------ the line


def test_a_straight_line_gives_its_slope_back():
    hours = np.arange(99) / 2
    row = _one(_trace(3.0 + 0.5 * hours))
    assert row["slope_per_hour"] == pytest.approx(0.5)
    assert row["slope_least_squares_per_hour"] == pytest.approx(0.5)
    assert row["monotone_rho"] == pytest.approx(1.0)


def test_the_change_over_the_recording_is_the_slope_across_the_span():
    """The interpretable number, and the one a reader will quote.

    Kept as a column rather than left for a figure to multiply, because the
    span is per cell - a cell present for thirty hours and one present for
    forty-nine do not share a multiplier, and a figure that used one number for
    both would be wrong in a way nothing would catch.
    """
    hours = np.arange(99) / 2
    row = _one(_trace(3.0 + 0.5 * hours))
    assert row["span_hours"] == pytest.approx(49.0)
    assert row["change_over_recording"] == pytest.approx(24.5)


# ------------------------------------------------------------ the robustness


def test_one_wild_frame_moves_the_line_of_best_fit_and_not_the_robust_one():
    """Why both slopes are written and neither is chosen for the reader.

    A single bad frame - a focus slip, a cell touched by a passing object - is
    the commonest thing in a long recording. Least squares follows it; the
    median of pairwise slopes does not. A gap between the two columns is the
    signal that the trace should be looked at.
    """
    hours = np.arange(99) / 2
    clean = _trace(3.0 + 0.5 * hours)
    spiked = clean.copy()
    spiked.loc[50, METRIC] = 5000.0

    before, after = _one(clean), _one(spiked)
    assert after["slope_per_hour"] == pytest.approx(before["slope_per_hour"], rel=0.05)
    assert abs(after["slope_least_squares_per_hour"]
               - before["slope_least_squares_per_hour"]) > 0.05


def test_the_robust_slope_carries_an_interval_that_brackets_it():
    hours = np.arange(99) / 2
    generator = np.random.default_rng(7)
    row = _one(_trace(3.0 + 0.5 * hours + generator.normal(0, 2, hours.size)))
    assert row["slope_low_per_hour"] <= row["slope_per_hour"] <= row["slope_high_per_hour"]


# ------------------------------------------------------------------ the edges


@contextmanager
def _no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_a_flat_trace_has_no_correlation_and_does_not_warn():
    """Zero slope is an answer; a correlation with a zero spread is not.

    ``spearmanr`` divides by that spread, returns ``NaN`` and warns while doing
    it, and a warning in a run over ninety cells is noise nobody reads.
    """
    with _no_warnings():
        row = _one(_trace(np.full(99, 7.0)))
    assert row["slope_per_hour"] == 0.0
    assert np.isnan(row["monotone_rho"])
    assert np.isnan(row["monotone_p_value"])
    assert row["level_median"] == pytest.approx(7.0)


def test_a_cell_with_too_few_frames_gets_no_row():
    """Absent rather than fitted, because a slope through a fragment of a
    rhythmic trace says nothing about the recording it came from."""
    table = derive(_trace(np.arange(10.0)),
                   _context({"metrics": [METRIC]}))["trend"]
    assert table.empty
    assert list(table.columns)[:4] == ["identity", "metric", "observations", "span_hours"]


def test_frames_the_cell_was_missing_from_are_dropped_not_filled():
    hours = np.arange(99) / 2
    values = 3.0 + 0.5 * hours
    frame = _trace(values)
    frame.loc[10:20, METRIC] = np.nan
    row = _one(frame)
    assert row["observations"] == 99 - 11
    assert row["slope_per_hour"] == pytest.approx(0.5)


def test_it_fits_only_the_metrics_it_was_told_to():
    """Never every numeric column: the package writes hundreds, and a slope
    through a bookkeeping count is a number with no meaning at all."""
    hours = np.arange(99) / 2
    frame = _trace(3.0 + 0.5 * hours)
    frame["circularity"] = 0.4
    table = derive(frame, _context({"metrics": [METRIC]}))["trend"]
    assert set(table["metric"]) == {METRIC}


def test_every_cell_gets_its_own_slope():
    hours = np.arange(99) / 2
    frame = pd.concat([_trace(3.0 + 0.5 * hours, identity=1),
                       _trace(3.0 - 0.25 * hours, identity=2)], ignore_index=True)
    table = derive(frame, _context({"metrics": [METRIC]}))["trend"]
    slopes = table.set_index("identity")["slope_per_hour"]
    assert slopes[1] == pytest.approx(0.5)
    assert slopes[2] == pytest.approx(-0.25)
