"""Overlap selection changes the mapped values, not just the figure wording."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from analysis.theme import load_theme
from _schema import FigureContext, load_all
from panels import territory


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


@pytest.mark.parametrize("method,expected,weight,contributors", [
    ("first", 2.0, 1, 1),
    ("last", 14.0, 1, 1),
    ("most_frequent", 8.0, 1, 1),
    ("equal_mean", 8.0, 3, 3),
    ("occupancy_weighted_mean", 7.0, 6, 3),
])
def test_overlap_selection_changes_each_pixel_and_its_export(
    axes, method, expected, weight, contributors,
):
    # First cell: 2 frames at value 2; second: 3 at 8; last: 1 at 14.
    # The second pixel sees only the first cell, with unoccupied gaps.
    movie = np.array([[[1, 1]], [[1, 0]], [[2, 0]], [[2, 0]], [[2, 0]], [[3, 1]]])
    result = territory.cell_metric_map(
        axes[1], movie, {1: 2.0, 2: 8.0, 3: 14.0}, load_theme(),
        field={"width": 2, "height": 1}, metric="measurement", label="Value",
        assignment=method, range_mode="full",
    )
    np.testing.assert_allclose(result.extra["handle"].get_array(), [[expected, 2.0]])
    pixels = result.extra["pixel_values"].set_index("column")
    assert pixels.loc[0, "value"] == pytest.approx(expected)
    assert pixels.loc[0, "total_weight"] == weight
    assert pixels.loc[0, "contributor_count"] == contributors
    assert pixels.map_assignment.eq(method).all()
    assert pixels.loc[1, "contributor_count"] == 1


@pytest.mark.parametrize("method", ["equal_mean", "occupancy_weighted_mean"])
def test_missing_occupants_and_background_do_not_dilute_averages(axes, method):
    movie = np.array([
        [[1, 3, 0, 4, 5]],
        [[2, 3, 0, 4, 5]],
        [[3, 3, 0, 4, 5]],
        [[0, 3, 0, 4, 5]],
    ])
    result = territory.cell_metric_map(
        axes[1], movie, {1: 2.0, 2: 8.0, 3: np.nan, 4: np.inf}, load_theme(),
        field={"width": 5, "height": 1}, metric="measurement", label="Value",
        assignment=method, range_mode="full",
    )
    # Cell 5 is absent from the metric table; cells 3/4 have invalid values.
    pixels = result.extra["pixel_values"]
    assert pixels["column"].tolist() == [0]
    assert pixels["value"].tolist() == [5.0]
    assert pixels.total_weight.tolist() == [2.0]
    assert pixels.contributor_count.tolist() == [2]
    image = result.extra["handle"].get_array()
    assert np.ma.getmaskarray(image).tolist() == [[False, True, True, True, True]]
    cells = result.data.set_index("identity")
    assert cells.loc[3, "contributing_pixels"] == 0
    assert cells.assigned_pixels.isna().all()


def test_each_metric_uses_its_own_valid_contributors(axes):
    result = territory.cell_metric_maps(
        axes[0], (0.1, 0.1, 0.8, 0.8), np.array([[[1]], [[2]], [[2]]]),
        {"complete": {1: 2.0, 2: 8.0}, "missing": {1: np.nan, 2: 8.0}},
        load_theme(), field={"width": 1, "height": 1},
        assignment="occupancy_weighted_mean", range_mode="full",
    )
    pixels = result.extra["pixel_values"].set_index("metric")
    assert pixels.loc["complete", "value"] == 6.0
    assert pixels.loc["complete", "total_weight"] == 3
    assert pixels.loc["missing", "value"] == 8.0
    assert pixels.loc["missing", "total_weight"] == 2


def test_average_requires_occupancy_history(axes):
    with pytest.raises(ValueError, match="label movie"):
        territory.cell_metric_map(
            axes[1], np.array([[1]]), {1: 2.0}, load_theme(),
            field={"width": 1, "height": 1}, metric="measurement", label="Value",
            assignment="occupancy_weighted_mean",
        )


def test_default_and_saved_options_reach_the_figure_and_flags_override_them(tmp_path):
    spec = load_all()["tissue-tectonics"]

    def context(argv=()):
        return FigureContext(
            spec=spec, run=tmp_path, tables=tmp_path, bundle=tmp_path,
            theme=None, summary={"minutes_per_frame": 30.0, "stem": "test"},
            field={}, stem=None, argv=list(argv),
        )

    assert context().option("map_assignment") == "occupancy_weighted_mean"
    settings = {"tissue-tectonics": {"options": {"map_assignment": "equal_mean"}}}
    (tmp_path / "figures.json").write_text(json.dumps(settings), encoding="utf-8")
    assert context().option("map_assignment") == "equal_mean"
    for method in territory.MAP_ASSIGNMENT_LABELS:
        assert context(["--map-assignment", method]).option("map_assignment") == method
        assert method in spec.usage()
    assert context(["--map-assignment", "weighted-mean"]).option(
        "map_assignment") == "occupancy_weighted_mean"
    with pytest.raises(SystemExit, match="--map-assignment"):
        context(["--map-assignment", "invalid"]).option("map_assignment")


@pytest.mark.parametrize("assignment", ["equal_mean", "occupancy_weighted_mean"])
def test_phase_averages_cross_cycle_boundary_and_export_cancelled_mixtures(axes, assignment):
    movie = np.array([[[1, 3]], [[2, 4]]])
    result = territory.cell_metric_map(
        axes[1], movie, {1: 23.0, 2: 1.0, 3: 0.0, 4: 12.0}, load_theme(),
        field={"width": 2, "height": 1}, metric="phase", label="Hours",
        assignment=assignment, period=24.0,
    )
    pixels = result.extra["pixel_values"].set_index("column")
    assert pixels.loc[0, "value"] == pytest.approx(0.0, abs=1e-12)
    assert pixels.loc[0, "phase_coherence"] == pytest.approx(np.cos(np.pi / 12))
    assert np.isnan(pixels.loc[1, "value"])
    assert pixels.loc[1, "mixed_phase"]
    assert pixels.loc[1, "total_weight"] == 2
    assert pixels.loc[1, "contributor_count"] == 2
    assert result.extra["display_minimum"] == 0
    assert result.extra["display_maximum"] == 24


@pytest.mark.parametrize("phase", [0.0, 1 / 24, 5 / 24, 0.49])
def test_occupancy_time_controls_phase_mixture_and_threshold(axes, phase):
    movie = np.array([[[1]], [[1]], [[1]], [[2]]])
    result = territory.cell_metric_map(
        axes[1], movie, {1: phase, 2: phase + 0.5}, load_theme(),
        field={"width": 1, "height": 1}, metric="phase", label="Cycles",
        assignment="occupancy_weighted_mean", period=1.0,
    )
    pixel = result.extra["pixel_values"].iloc[0]
    assert pixel.value == pytest.approx(phase, abs=1e-12)
    assert pixel.phase_coherence == pytest.approx(0.5)
    assert pixel.total_weight == 4
    assert pixel.displayed


def test_empty_supported_phase_map_still_renders_with_its_scale(axes):
    result = territory.cell_metric_map(
        axes[1], np.array([[[1]], [[2]]]), {1: np.nan, 2: np.nan}, load_theme(),
        field={"width": 1, "height": 1}, metric="phase", label="Cycles",
        assignment="occupancy_weighted_mean", period=1.0, allow_empty=True,
    )
    assert result.extra["pixel_values"].empty
    assert np.ma.getmaskarray(result.extra["handle"].get_array()).all()
    assert result.extra["display_maximum"] == 1


def rhythm_context(tmp_path, argv=(), **params):
    class Context(FigureContext):
        def module_params(self, module):
            assert module == "rhythms"
            return {"primary_rhythm_test": "lomb", "period_search_hours": [18, 32],
                    "detrend": "none", **params}
    return Context(
        spec=load_all()["tissue-tectonics"], run=tmp_path, tables=tmp_path,
        bundle=tmp_path, theme=None, summary={"minutes_per_frame": 30, "stem": "synthetic"},
        field={}, stem=None, argv=list(argv),
    )


def synthetic_rhythm_frame():
    # Same physical phases, with different trace starts and a missing interval.
    frames = []
    for identity, start in [(1, 5.0), (2, 11.0)]:
        h = np.arange(start, 149.0, 0.5)
        frame = pd.DataFrame({
            "identity": identity, "hours": h,
            "corrected_mean": 10 + 2 * np.cos(2 * np.pi * (h - 7) / 27),
            "turnover_index": .4 + .1 * np.cos(2 * np.pi * (h - 13) / 27),
        })
        frame.loc[frame.hours.between(46, 49), "corrected_mean"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_optional_fitted_maps_keep_selected_workbench_estimates(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps, FITTED_MAPS, DEFAULT_RHYTHM_MAPS
    frame = synthetic_rhythm_frame()
    values, fits, daily, params = analyse_rhythm_maps(
        rhythm_context(tmp_path), frame, [*DEFAULT_RHYTHM_MAPS, *FITTED_MAPS])
    np.testing.assert_allclose(fits.period_hours, 27.0, atol=0.05)
    assert fits.period_map_available.all()
    assert fits.family_tests.eq(4).all()
    np.testing.assert_allclose(values["reporter_period_hours"], 27, atol=.05)
    np.testing.assert_allclose(values["turnover_fitted_variation"], 100, atol=.01)
    reporter = frame.groupby("identity").corrected_mean.mean()
    np.testing.assert_allclose(values["reporter_fitted_relative_amplitude"], 2 / reporter, atol=.001)
    assert daily.empty
    assert "map_peak_phase" not in fits


def test_intensity_period_map_uses_lomb_scargle_significance_per_cell(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps

    frame = synthetic_rhythm_frame()
    values, fits, _, params = analyse_rhythm_maps(
        rhythm_context(tmp_path), frame, ["reporter_period_hours"]
    )
    reporter = fits[fits.metric.eq("corrected_mean")]
    assert len(reporter) == frame.identity.nunique()
    assert fits.method.eq("lomb").all()
    assert fits.significance_method.eq("lomb").all()
    assert fits.correction.eq("none").all()
    assert reporter.period_map_available.all()
    np.testing.assert_allclose(values["reporter_period_hours"], 27.0, atol=0.1)
    assert params["period_search_hours"] == [2, 48]

    _, corrected_fits, _, _ = analyse_rhythm_maps(
        rhythm_context(tmp_path, ["--multiple-testing", "bh"]),
        frame,
        ["reporter_period_hours"],
    )
    assert corrected_fits.correction.eq("bh").all()


def test_fit_method_and_detrending_controls_reach_workbench(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps
    frame = synthetic_rhythm_frame()
    ctx = rhythm_context(tmp_path, ["--fit-method", "fft_nlls", "--detrend", "linear"])
    values, fits, _, params = analyse_rhythm_maps(ctx, frame, ["reporter_period_hours"])
    assert fits.method.eq("fft_nlls").all()
    assert fits.significance_method.eq("lomb").all()
    assert fits.detrend.eq("linear").all()
    assert params["period_search_hours"] == [2, 48]
    # Check this adapter preserves the selected Workbench estimator exactly;
    # its multi-component fit can differ from the generating single cosine.
    from analysis import circadian as gateway
    trace = frame[frame.identity.eq(1)].dropna(subset=["corrected_mean"])
    direct = gateway.estimate_one(trace.hours, trace.corrected_mean, params, "fft_nlls")
    reporter = fits[fits.identity.eq(1) & fits.metric.eq("corrected_mean")].iloc[0]
    assert reporter.period_hours == pytest.approx(direct["period_hours"])
    assert reporter.phase_hours == pytest.approx(direct["phase_hours"])
    assert values["reporter_period_hours"].notna().all()


def test_flat_and_short_traces_leave_period_maps_blank_without_crashing(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps, FITTED_MAPS, DEFAULT_RHYTHM_MAPS
    frame = synthetic_rhythm_frame()
    frame.loc[frame.identity.eq(1), ["corrected_mean", "turnover_index"]] = 1.0
    frame = frame[frame.identity.eq(1) | frame.hours.lt(20)]
    values, fits, _, _ = analyse_rhythm_maps(rhythm_context(tmp_path), frame, [*DEFAULT_RHYTHM_MAPS, *FITTED_MAPS])
    assert all(value.isna().all() for value in values.values())
    assert not fits.period_map_available.any()
    assert fits.rhythm_status.eq("not tested").all()


def test_period_map_displays_significant_incomplete_cycles_as_exploratory(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps, FITTED_MAPS, DEFAULT_RHYTHM_MAPS
    frame = synthetic_rhythm_frame()
    frame = frame[frame.hours.lt(46)]
    values, fits, _, _ = analyse_rhythm_maps(rhythm_context(tmp_path), frame, [*DEFAULT_RHYTHM_MAPS, *FITTED_MAPS])
    assert fits.period_underdetermined.all()
    assert values["reporter_period_hours"].notna().all()
    assert fits.period_map_displayed.all()
    assert not fits.period_map_supported.any()
    assert fits.period_map_status.eq("exploratory period").all()
    assert values["reporter_fitted_relative_amplitude"].notna().all()


def test_duplicate_times_are_rejected_before_workbench(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps
    frame = synthetic_rhythm_frame()
    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one observation"):
        analyse_rhythm_maps(rhythm_context(tmp_path), frame, ["reporter_period_hours"])


def test_workbench_cosinor_reports_peak_time_without_reflecting_it():
    from analysis.circadian import cosinor
    hours = np.arange(0, 96, .5)
    result = cosinor(hours, 10 + 2 * np.cos(2 * np.pi * (hours - 6) / 24), 24)
    assert result["cosinor_peak_hour"] == pytest.approx(6)
    assert result["cosinor_phase_convention"] == "positive_peak_time"


def test_benjamini_hochberg_is_not_silently_replaced_by_sidak(tmp_path, monkeypatch):
    from analysis import circadian as gateway
    from _tissue_rhythms import analyse_rhythm_maps
    pvalues = iter([.01, .02, .1, np.nan])
    def estimate(*args, **kwargs):
        return {"status": "ok", "period_hours": 27, "phase_hours": 2,
                "amplitude": 1, "p_value": next(pvalues), "diagnostics": {}}
    monkeypatch.setattr(gateway, "estimate_one", estimate)
    _, fits, _, _ = analyse_rhythm_maps(
        rhythm_context(tmp_path, ["--multiple-testing", "bh"]),
        synthetic_rhythm_frame(),
        ["reporter_period_hours"],
    )
    np.testing.assert_allclose(fits.q_value, [.03, .03, .1, np.nan], equal_nan=True)


@pytest.mark.parametrize("name", [
    "reporter_m10_onset", "reporter_daily_contrast", "turnover_daily_repeatability",
    "reporter_turnover_phase_difference", "reporter_phase_offset",
    "turnover_active_duration", "reporter_fitted_peak_phase", "reporter_peak_phase",
    "reporter_relative_amplitude", "turnover_rhythm_strength",
])
def test_retired_daily_and_cross_period_comparisons_are_not_silent_aliases(tmp_path, name):
    from _tissue_rhythms import analyse_rhythm_maps
    with pytest.raises(ValueError, match="Microglial periods are unknown"):
        analyse_rhythm_maps(rhythm_context(tmp_path), synthetic_rhythm_frame(), [name])


def test_cumulative_occupancy_pools_identities_revisits_and_irregular_intervals():
    labels = np.array([[[1, 1, 0]], [[2, 0, 0]], [[3, 4, 0]]])
    actual = territory.cumulative_occupancy(labels, [10, 12, 16])
    np.testing.assert_array_equal(actual[0], [[0, 0, 0]])
    np.testing.assert_allclose(actual[-1], [[6, 3, 0]])
    assert actual.max() <= 6
    np.testing.assert_allclose(actual, territory.cumulative_occupancy(labels > 0, [10, 12, 16]))


def test_cumulative_occupancy_default_map_is_final_state(axes):
    labels = np.array([[[1, 0]], [[0, 2]], [[1, 2]]])
    result = territory.cumulative_occupancy_map(axes[1], labels, load_theme(),
                 hours=[1, 2, 3], field={"width": 2, "height": 1})
    np.testing.assert_allclose(result.extra["handle"].get_array(), [[1, 1.5]])
    assert result.data.hours.eq(3).all()
    assert result.data.frame_index.eq(2).all()
    assert result.extra["display_minimum"] == 0
    assert result.extra["display_maximum"] == 2


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, np.nan]])
def test_cumulative_occupancy_rejects_invalid_timing(times):
    with pytest.raises(ValueError, match="increasing"):
        territory.cumulative_occupancy(np.ones((2, 1, 1)), times)


def test_cumulative_pixel_field_does_not_average_cell_values(axes):
    result = territory.cell_metric_maps(axes[0], (.1, .1, .8, .8), np.ones((2, 1, 2)),
        {"cell": {1: 99.}, "occupied_hours": {}}, load_theme(), field={"width": 2, "height": 1},
        pixel_fields={"occupied_hours": np.array([[1., 2.]])},
        limits={"occupied_hours": (0, 2)}, assignment="occupancy_weighted_mean")
    pixels = result.extra["pixel_values"]
    assert pixels[pixels.metric.eq("occupied_hours")].value.tolist() == [1., 2.]
    assert pixels[pixels.metric.eq("cell")].value.tolist() == [99., 99.]


def test_count_map_includes_single_cell_pixels_by_default(axes):
    result = territory.owner_count_map(axes[1], [[0, 1, 2]], load_theme(), field={"width": 3, "height": 1})
    assert result.data.owner_count.tolist() == [1, 2]
    assert result.data.minimum_owner_count.eq(1).all()


def test_first_coverage_uses_elapsed_time_and_leaves_never_covered_pixels_blank():
    labels = np.array([
        [[0, 1, 0, 0]],
        [[2, 1, 0, 0]],
        [[2, 1, 3, 0]],
    ])
    actual = territory.first_coverage_time(labels, [5.0, 7.0, 12.0])
    np.testing.assert_allclose(actual[0, :2], [2.0, 0.0])
    assert actual[0, 2] == 7.0
    assert np.isnan(actual[0, 3])


def test_split_event_areas_use_last_connected_and_first_separated_footprints():
    labels = np.array([
        [[1, 2, 0, 0]],
        [[1, 0, 2, 0]],
    ])
    events = pd.DataFrame([{
        "event_id": "C0001", "event_type": "contact_separate", "identity": 1,
        "candidate_identities": "2", "gap_end_frame_index": 0,
        "post_frame_index": 1,
    }, {
        "event_id": "E0001", "event_type": "soft_gap", "identity": 1,
        "candidate_identities": "2", "gap_end_frame_index": 0,
        "post_frame_index": 1,
    }])
    counts, selected = territory.split_event_areas(labels, events)
    np.testing.assert_array_equal(counts, [[1, 1, 1, 0]])
    assert selected.event_id.tolist() == ["C0001"]
    assert selected.event_area_px.tolist() == [3]


def mixed_period_frame():
    # Non-sinusoidal pulses: neither a common day nor a common waveform phase.
    hours = np.arange(0, 192, .5)
    return pd.concat([pd.DataFrame({
        "identity": identity, "hours": hours,
        "corrected_mean": 2 + 8 * ((hours % reporter_period) < reporter_period / 3),
        "turnover_index": .2 + .6 * ((hours % turnover_period) < turnover_period / 3),
    }) for identity, reporter_period, turnover_period in [(1, 6, 30), (2, 30, 8)]], ignore_index=True)


def test_mixed_short_and_long_periods_are_estimated_per_cell_and_signal(tmp_path, monkeypatch):
    from analysis import circadian as gateway
    from _tissue_rhythms import analyse_rhythm_maps, DEFAULT_RHYTHM_MAPS
    def forbidden(*args, **kwargs):
        raise AssertionError("unknown-period map called a fixed daily summary")
    monkeypatch.setattr(gateway, "daily_profile_metrics", forbidden)
    monkeypatch.setattr(gateway, "daily_measures", forbidden)
    ctx = rhythm_context(tmp_path, ["--fit-method", "chi_square", "--significance-method", "chi_square"])
    values, fits, daily, params = analyse_rhythm_maps(ctx, mixed_period_frame(), DEFAULT_RHYTHM_MAPS)
    np.testing.assert_allclose(values["reporter_period_hours"], [6, 30], atol=.5)
    np.testing.assert_allclose(values["turnover_period_hours"], [30, 8], atol=.5)
    assert fits.method.eq("chi_square").all()
    assert fits.significance_method.eq("chi_square").all()
    assert fits.family_tests.eq(4).all()
    assert fits.period_map_available.all()
    assert params["period_search_hours"] == [2, 48]
    assert daily.empty
    assert "map_peak_phase" not in fits
    assert "profile_period_hours" not in fits


def test_cycle_requirement_uses_each_estimated_period(tmp_path):
    from _tissue_rhythms import analyse_rhythm_maps, DEFAULT_RHYTHM_MAPS
    # Identical 48-hour records contain eight 6-hour cycles but < two 30-hour cycles.
    frame = mixed_period_frame().query("hours < 48")
    values, fits, _, _ = analyse_rhythm_maps(rhythm_context(tmp_path), frame, DEFAULT_RHYTHM_MAPS)
    assert np.isfinite(values["reporter_period_hours"].loc[1])
    assert np.isfinite(values["reporter_period_hours"].loc[2])
    assert np.isfinite(values["turnover_period_hours"].loc[1])
    assert np.isfinite(values["turnover_period_hours"].loc[2])
    long = fits[(fits.identity.eq(2) & fits.metric.eq("corrected_mean")) |
                (fits.identity.eq(1) & fits.metric.eq("turnover_index"))]
    assert long.period_map_displayed.all()
    assert not long.period_map_supported.any()
    assert long.period_map_exclusion.eq("insufficient observed cycles").all()


def test_period_support_masks_separate_supported_and_exploratory_coverage():
    from importlib import import_module
    tissue = import_module("28_tissue_tectonics")
    labels = np.array([
        [[1, 2, 0]],
        [[1, 3, 0]],
    ])
    fits = pd.DataFrame({
        "identity": [1, 2, 3],
        "period_map_displayed": [True, True, False],
        "period_map_supported": [True, False, False],
    })

    supported, exploratory = tissue._period_support_masks(labels, fits)
    np.testing.assert_array_equal(supported, [[True, False, False]])
    np.testing.assert_array_equal(exploratory, [[False, True, False]])


def test_period_mean_keeps_distinct_cell_estimates_in_its_export(axes):
    # Six hours and thirty hours are separate cell estimates, not an 18-hour rhythm.
    result = territory.cell_metric_map(axes[1], np.array([[[1]], [[2]]]),
        {1: 6., 2: 30.}, load_theme(), field={"width": 1, "height": 1},
        metric="reporter_period_hours", label="Mean reporter cell period",
        assignment="occupancy_weighted_mean", range_mode="full")
    assert result.extra["pixel_values"].value.tolist() == [18.]
    assert result.extra["pixel_values"].contributor_count.tolist() == [2]
    assert sorted(result.data.metric_value) == [6., 30.]
