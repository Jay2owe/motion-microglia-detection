"""The non-parametric circadian block, checked against a second implementation.

This is the stage where a wrong formula is easiest to ship and hardest to
notice: interdaily stability and intradaily variability both look entirely
plausible when coded slightly wrong, and nothing downstream would complain. So
the first test here is not a hand-worked example at all - it feeds one trace to
this module and to the lab's own ``circadian_workbench``, which implements all
five of these measures already, and requires the same answer. That turns "I
think this formula is right" into a comparison against code the lab trusts.

The rest cover what the workbench cannot check for us: the guard that stops a
flat trace inventing an onset, the honesty columns, and the fact that these
hours are hours since the recording started rather than times of day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.modules.rhythms import (DEFAULTS, _template_marker, _template_scores,
                                      derive, nonparametric)
from analysis.registry import MeasurementContext
from analysis.units import Scale

#: One sample an hour, so a bin holds exactly one value and the workbench's
#: summing resampler and this module's averaging one cannot disagree about what
#: a bin is. That is what makes the cross-check a test of the formulas rather
#: than of the binning.
HOURS_PER_SAMPLE = 1.0


def _params(**overrides) -> dict:
    return {**DEFAULTS, **overrides}


def _sine(days: float, peak_hour: float = 14.0, amplitude: float = 30.0,
          level: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """A clean 24-hour rhythm, one sample an hour, peaking at a known hour."""
    hours = np.arange(0.0, days * 24.0, HOURS_PER_SAMPLE)
    values = level + amplitude * np.cos(2 * np.pi * (hours - peak_hour) / 24.0)
    return hours, values


def _context(minutes: float = 60.0, params: dict | None = None) -> MeasurementContext:
    return MeasurementContext(
        stem="synthetic",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        raw=np.zeros((2, 2, 2), dtype=np.uint16),
        scale=Scale(minutes),
        identities=[1],
        params={"rhythms": dict(params or {})},
    )


def _cell_frame(hours: np.ndarray, values: np.ndarray, identity: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "identity": identity,
        "frame_index": np.arange(len(hours)),
        "hours": hours,
        "corrected_mean": values,
    })


# ------------------------------------------------- against the lab's own code


def test_the_five_shared_measures_agree_with_circadian_workbench() -> None:
    """The same trace through both implementations, to floating-point tolerance.

    Skipped rather than made a hard dependency: ``circadian-workbench`` pulls a
    web stack into what ships as one measurement tool, so the four functions are
    copied and this is what stops the copy drifting.
    """
    workbench = pytest.importorskip("circadian_workbench.analysis")

    hours, values = _sine(days=4, peak_hour=14.0)
    # Two gaps, because a formula that is right on a complete trace and wrong on
    # a gapped one is the failure this dataset will actually meet.
    values[30:33] = np.nan
    values[70] = np.nan
    measured = np.isfinite(values)

    ours = nonparametric(hours[measured], values[measured], _params())

    theirs = workbench.nonparametric_metrics(
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01 00:00", periods=len(hours), freq="1h")[measured],
            "analysis_activity": values[measured],
        }),
        bin_minutes=60,
    )

    assert ours["l5"] == pytest.approx(theirs["l5_mean"])
    assert ours["m10"] == pytest.approx(theirs["m10_mean"])
    assert ours["l5_onset_hour"] == pytest.approx(theirs["l5_start_hours"])
    assert ours["m10_onset_hour"] == pytest.approx(theirs["m10_start_hours"])
    assert ours["relative_amplitude"] == pytest.approx(theirs["relative_amplitude"])
    assert ours["interdaily_stability"] == pytest.approx(theirs["interdaily_stability"])
    assert ours["intradaily_variability"] == pytest.approx(theirs["intradaily_variability"])


def test_the_onset_template_agrees_with_circadian_workbench() -> None:
    """Onset and offset use ClockLab's template, scored the same way.

    The scores are compared rather than only the markers, because two
    implementations can pick the same bin from differently shaped curves and
    then disagree on the next dataset.
    """
    workbench = pytest.importorskip("circadian_workbench.analysis")
    thresholded = np.where(np.arange(24) % 24 >= 8, 1.0, -1.0)
    thresholded[20:] = -1.0
    ours = _template_scores(thresholded, off_bins=6, on_bins=6)
    theirs = workbench._template_scores(thresholded, 6, 6)
    assert np.allclose(ours[0], theirs[0])
    assert np.allclose(ours[1], theirs[1])
    assert _template_marker(ours[0], 1.0) == pytest.approx(workbench._template_marker(theirs[0], 60)[0])


# ---------------------------------------------------------- a clean rhythm


def test_a_clean_rhythm_puts_the_busiest_stretch_around_its_peak() -> None:
    """M10 straddles the peak, L5 straddles the trough, half a day apart.

    A cosine peaking at hour 14 has its ten busiest hours centred there, so the
    window starts five hours earlier; the quietest five are centred on hour 2,
    twelve hours away. Both are read off the folded average day, so the answer
    is a time within the day rather than a position in the recording.
    """
    hours, values = _sine(days=4, peak_hour=14.0)
    result = nonparametric(hours, values, _params())
    assert result["m10_onset_hour"] == pytest.approx(9.0)
    assert result["l5_onset_hour"] == pytest.approx(0.0)
    assert result["m10"] > result["l5"]
    assert 0.0 < result["relative_amplitude"] < 1.0
    # Four identical days: one day is exactly the next, and the trace is one
    # smooth rise and fall rather than fragments.
    assert result["interdaily_stability"] == pytest.approx(1.0, abs=1e-6)
    assert result["intradaily_variability"] < 0.1
    assert result["onset_found"]
    assert np.isfinite(result["active_duration_hours"])


def test_the_quietest_stretch_may_straddle_the_end_of_the_day() -> None:
    """The window is circular, and this is the case a flat implementation loses.

    A cosine peaking at hour 12 has its trough at hour 0, so the quietest five
    hours run from late in one day into the start of the next. A window that
    stopped at the edge of the folded day would find some other five hours and
    quietly report the wrong ones.
    """
    hours, values = _sine(days=4, peak_hour=12.0)
    result = nonparametric(hours, values, _params())
    assert result["l5_onset_hour"] == pytest.approx(22.0)


def test_flat_noise_has_almost_no_amplitude_and_no_onset() -> None:
    """The two answers that must not look like a rhythm.

    Relative amplitude near zero because the busiest and quietest stretches of
    the average day are the same, and intradaily variability high because the
    trace is fragments rather than one rise and fall.
    """
    generator = np.random.default_rng(20260901)
    hours = np.arange(0.0, 96.0, HOURS_PER_SAMPLE)
    values = 100.0 + generator.normal(0, 5, len(hours))
    result = nonparametric(hours, values, _params())
    assert result["relative_amplitude"] < 0.05
    assert result["intradaily_variability"] > 1.0
    assert result["interdaily_stability"] < 0.3


def test_a_trace_with_no_active_phase_returns_no_onset_rather_than_hour_zero() -> None:
    """The bug the workbench shipped, found, fixed, and this copy inherits fixed.

    Every bin of a flat trace falls on the same side of the threshold, so every
    candidate hour scores identically. ``argmax`` on a flat curve returns bin 0,
    which reads downstream as a perfectly regular onset at the moment the
    recording began - the workbench's own note calls the result that produced
    "the most convincing-looking result in the app, and entirely fabricated".
    """
    hours = np.arange(0.0, 96.0, HOURS_PER_SAMPLE)
    result = nonparametric(hours, np.full(len(hours), 42.0), _params())
    assert not result["onset_found"]
    assert np.isnan(result["onset_hour"])
    assert np.isnan(result["offset_hour"])
    assert np.isnan(result["active_duration_hours"])


# ------------------------------------------------------------ the honesty


def test_two_cells_seen_over_different_stretches_share_one_clock() -> None:
    """The bins are anchored at the recording, not at each cell's own first frame.

    Both cells follow the same 24-hour rhythm peaking at hour 14 of the movie.
    The second is not named until hour 9, so it is measured over a different
    stretch of the same recording. Anchoring the fold on each cell's own start
    would put the second cell's day nine hours out and report a busiest stretch
    nine hours earlier - silently, and only visibly wrong once somebody compared
    two cells. Every hour in this table is a position in the recording, so two
    cells of one movie are directly comparable and two cells of two movies are
    not comparable at all.
    """
    hours, values = _sine(days=4, peak_hour=14.0)
    early = nonparametric(hours, values, _params())
    late = nonparametric(hours[9:], values[9:], _params())
    assert late["m10_onset_hour"] == early["m10_onset_hour"]
    assert late["l5_onset_hour"] == early["l5_onset_hour"]
    # Onset moves by a bin and should. The template is scored against a
    # threshold taken over the bins this cell was actually measured in, so a
    # cell that missed nine quiet hours has a slightly higher bar to clear -
    # which is the workbench's behaviour and a property of the measure rather
    # than of the fold. The fold itself is now shared, which is what the two
    # assertions above check.
    assert abs(late["onset_hour"] - early["onset_hour"]) <= 1.0


def test_a_two_day_recording_says_its_stability_is_underdetermined() -> None:
    """Two days is one comparison, which is computable and weak.

    The number is written anyway, because withholding it silently is no better
    than publishing it silently; the flag is what stops it being quoted as if it
    were solid.
    """
    hours, values = _sine(days=2)
    short = nonparametric(hours, values, _params())
    assert short["days_covered"] == pytest.approx(47 / 24)
    assert short["stability_underdetermined"]
    assert np.isfinite(short["interdaily_stability"])

    hours, values = _sine(days=5)
    assert not nonparametric(hours, values, _params())["stability_underdetermined"]


def test_one_day_gives_no_stability_at_all() -> None:
    """It compares one day with the next, and there is no next."""
    hours, values = _sine(days=1)
    assert np.isnan(nonparametric(hours, values, _params())["interdaily_stability"])


def test_the_bins_a_measurement_landed_in_are_counted() -> None:
    """``hours_binned`` is what says how much of the recording is behind a row."""
    hours, values = _sine(days=3)
    keep = np.ones(len(hours), dtype=bool)
    keep[10:30] = False
    result = nonparametric(hours[keep], values[keep], _params())
    assert result["hours_binned"] == int(keep.sum())


def test_a_bin_length_that_does_not_divide_a_day_is_refused_by_name() -> None:
    """Rounding it would make interdaily stability subtly wrong rather than absent.

    Fifty minutes is the case to think about: a plausible imaging interval, and
    24 hours is not a whole number of them. Forty-five is not - it gives 32 bins
    a day - which is exactly why the check has to be arithmetic rather than a
    list of intervals somebody thought of.
    """
    hours, values = _sine(days=3)
    with pytest.raises(ValueError, match="bin_hours"):
        nonparametric(hours, values, _params(bin_hours=50 / 60))
    for divides in (0.25, 0.5, 0.75, 1.0, 2.0):
        assert np.isfinite(nonparametric(hours, values, _params(bin_hours=divides))["m10"])


# -------------------------------------------------------- inside the module


def test_the_module_writes_the_new_columns_beside_the_fits() -> None:
    """Same table, same grain: these answer the same question about the same thing."""
    hours, values = _sine(days=4)
    table = derive(_cell_frame(hours, values),
                   _context(params={"metrics": ["corrected_mean"],
                                    "null_surrogates_per_cell": 0}))["rhythms"]
    assert len(table) == 1
    row = table.iloc[0]
    for column in ("m10", "l5", "relative_amplitude", "onset_hour", "offset_hour",
                   "interdaily_stability", "intradaily_variability", "days_covered",
                   "hours_binned", "stability_underdetermined", "onset_found"):
        assert column in table.columns, column
    assert np.isfinite(row["relative_amplitude"])
    assert np.isfinite(row["cosinor_relative_amplitude"])


def test_the_two_relative_amplitudes_are_not_the_same_number() -> None:
    """If they were equal on every row, one of them would not be being computed.

    One divides the busiest ten hours of the average day by the quietest five;
    the other divides a fitted amplitude by a mean level. They answer the same
    question and are not interchangeable, which is why neither is labelled
    "relative amplitude" on its own.
    """
    hours, values = _sine(days=4, amplitude=30.0, level=100.0)
    row = derive(_cell_frame(hours, values),
                 _context(params={"metrics": ["corrected_mean"],
                                  "null_surrogates_per_cell": 0}))["rhythms"].iloc[0]
    assert row["relative_amplitude"] != pytest.approx(row["cosinor_relative_amplitude"])


def test_a_cell_below_the_observation_floor_is_skipped_for_these_too() -> None:
    """One gate, so a row is filled in or is not there - never half a row."""
    hours, values = _sine(days=4)
    frame = pd.concat([_cell_frame(hours, values, identity=1),
                       _cell_frame(hours[:10], values[:10], identity=2)])
    table = derive(frame, _context(params={"metrics": ["corrected_mean"],
                                           "null_surrogates_per_cell": 0}))["rhythms"]
    assert table["identity"].tolist() == [1]
