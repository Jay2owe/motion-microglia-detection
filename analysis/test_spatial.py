"""Regression checks for the additive spatial time-series plot family."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import spatial
from analysis.modules.rhythms import DEFAULTS


def test_radial_reduction_requires_all_bands_and_support():
    rings = pd.DataFrame(dict(identity=[1, 1, 1, 1, 2], frame_index=[0, 0, 1, 1, 0],
                              hours=[0, 0, 1, 1, 0], ring=[0, 1, 0, 1, 0],
                              scaling="cell", radius_inner=[0, 1, 0, 1, 0],
                              radius_outer=[1, 2, 1, 2, 1],
                              annulus_px=[4, 10, 4, 1, 4], occupancy=[1, .5, 1, .1, 1]))
    result = spatial.radial_occupancy(rings).set_index(["identity", "frame_index"])
    assert result.loc[(1, 0), spatial.RADIAL_METRIC] == .75
    assert np.isnan(result.loc[(1, 1), spatial.RADIAL_METRIC])
    assert np.isnan(result.loc[(2, 0), spatial.RADIAL_METRIC])


def test_radial_calibration_and_duplicate_refusal():
    rings = pd.DataFrame(dict(identity=[1, 1], frame_index=0, hours=0, ring=[0, 1],
                              scaling="cell", radius_inner=[0, 2], radius_outer=[2, 4],
                              annulus_px=[4, 10], occupancy=[1, .5]))
    assert spatial.radial_occupancy(rings, length_per_pixel=2)[spatial.RADIAL_METRIC].iloc[0] == .75
    with pytest.raises(ValueError, match="duplicate"):
        spatial.radial_occupancy(pd.concat([rings, rings]))


@pytest.mark.parametrize("display", ["raw", "residual", "standardized"])
def test_display_preserves_missing_observations(display):
    frame = pd.DataFrame(dict(identity=1, frame_index=np.arange(40), hours=np.arange(40) / 2,
                              area=np.cos(np.arange(40) / 2) + np.arange(40)))
    frame.loc[5, "area"] = np.nan
    out = spatial.display_traces(frame, "area", DEFAULTS, display)
    assert len(out) == len(frame)
    assert np.isnan(out.loc[5, "value"])
    assert out.value.notna().sum() == 39


def test_detrend_full_options_forwarded(monkeypatch):
    captured = {}
    def detrend(hours, values, params):
        captured.update(params)
        return {"values": np.zeros(len(values))}
    monkeypatch.setattr(spatial.workbench, "detrend_trace", detrend)
    frame = pd.DataFrame(dict(identity=1, frame_index=np.arange(10), hours=np.arange(10), area=np.arange(10)))
    out = spatial.display_traces(frame, "area", {"detrend": "poly6", "detrend_polynomial_degree": 6})
    assert captured["detrend"] == "poly6"
    assert captured["detrend_polynomial_degree"] == 6
    assert (out.value == 0).all()


def test_snapshot_times_are_actual_and_validated():
    np.testing.assert_array_equal(spatial.snapshot_indices([1, 1.5, 2], [1.1, 1.6]), [0, 1])
    with pytest.raises(ValueError, match="outside"):
        spatial.snapshot_indices([1, 2], [0])
    with pytest.raises(ValueError, match="increasing"):
        spatial.snapshot_indices([1, 1])


def test_snapshot_default_is_last_recorded_frame_not_first():
    np.testing.assert_array_equal(spatial.snapshot_indices([1, 1.5, 2]), [2])
    np.testing.assert_array_equal(spatial.snapshot_indices([1, 1.5, 2], []), [2])
    np.testing.assert_array_equal(spatial.snapshot_indices([1]), [0])


def test_snapshot_count_and_named_hours():
    np.testing.assert_array_equal(spatial.snapshot_indices(np.arange(9), count=3), [0, 4, 8])
    np.testing.assert_array_equal(spatial.snapshot_indices([0, 1, 2, 99, 100], count=5), np.arange(5))
    # Explicit times override the requested count; output is unique and chronological.
    np.testing.assert_array_equal(spatial.snapshot_indices([1, 2, 3], [2.9, 1.1, 1.2], count=99), [0, 2])


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "2"])
def test_invalid_snapshot_counts_are_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        spatial.snapshot_indices([1, 2, 3], count=count)


def test_snapshot_count_cannot_exceed_available_frames():
    with pytest.raises(ValueError, match="exceeds"):
        spatial.snapshot_indices([1, 2], count=3)
    with pytest.raises(ValueError, match="finite"):
        spatial.snapshot_indices([1, 2], [np.nan])


def test_phase_delay_has_sign_and_does_not_assume_24_hours():
    fits = pd.DataFrame(dict(identity=[1, 1, 2, 2], metric=["shape", "reporter"] * 2,
                             period_hours=[8., 8., 6., 24.], phase_hours=[3., 1., 1., 1.],
                             supported=True, span_hours=48.))
    out = spatial.timing_difference(fits, "shape", "reporter", reference_hour=20).set_index("identity")
    assert out.loc[1, "value"] == pytest.approx(2)
    assert out.loc[1, "phase_drift_hours"] == 0
    assert np.isnan(out.loc[2, "value"])
    assert not out.loc[2, "compatible_periods"]


def test_unsupported_period_never_gets_timing():
    fits = pd.DataFrame(dict(identity=[1, 1], metric=["a", "b"], period_hours=24.,
                             phase_hours=[1., 3.], supported=[False, True], span_hours=48.))
    assert np.isnan(spatial.timing_difference(fits, "a", "b").value.iloc[0])


def test_rhythm_support_broad_search():
    hours = np.arange(0, 72, .5)
    frame = pd.DataFrame(dict(identity=1, hours=hours, metric=np.cos(2 * np.pi * (hours - 2) / 8)))
    fits = spatial.rhythm_fits(frame, ["metric"], {**DEFAULTS, "period_search_hours": [2, 48]})
    assert fits.period_hours.iloc[0] == pytest.approx(8, abs=.1)
    assert fits.supported.iloc[0]
    assert fits.phase_hours.iloc[0] == pytest.approx(2, abs=.1)


def test_default_spatial_peak_map_shows_all_significant_estimates_and_marks_support():
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _spatial import _period_data

    values = {
        "phase_group_hours": [], "period_tolerance": .1, "spatial_cmap": None,
    }
    context = SimpleNamespace(
        spec=SimpleNamespace(options=[SimpleNamespace(name=name) for name in values]),
        option=lambda name: values[name],
    )
    fits = pd.DataFrame({
        "identity": [1, 2, 3],
        "period_hours": [8., 10., 12.],
        "phase_hours": [2., 5., 3.],
        "significant": [True, True, True],
        "period_available": [True, True, True],
        "supported": [True, True, False],
        "display_status": ["supported rhythm", "supported rhythm", "exploratory period"],
        "period_search_min_hours": [2.] * 3,
        "period_search_max_hours": [48.] * 3,
    })
    positions = pd.DataFrame({"identity": [1, 2, 3], "x": [1., 2., 3.], "y": [4., 5., 6.]})

    plotted, tiles = _period_data(context, fits, positions)
    phase = plotted[plotted.tile.eq("phase_percent")].set_index("identity")

    assert phase.loc[1, "value"] == pytest.approx(25.)
    assert phase.loc[2, "value"] == pytest.approx(50.)
    assert phase.loc[3, "value"] == pytest.approx(25.)
    assert phase.loc[3, "display_status"] == "exploratory period"
    assert tiles[1]["vmin"] == 0
    assert tiles[1]["vmax"] == 100
    assert tiles[1]["label"] == "Peak position within own cycle (%)"
    assert tiles[1]["cmap"] == "phase_cycle"
    assert "3 significant; 2 supported" in tiles[1]["title"]


def test_phase_cycle_palette_has_a_visible_cyclic_seam():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from panels import spatial as panels

    np.testing.assert_allclose(panels.PHASE_CYCLE(0.), panels.PHASE_CYCLE(1.))
    assert panels.PHASE_CYCLE(.5)[:3] != pytest.approx((.933, .933, .933))


def test_neighbour_test_preserves_pair_dependence_and_is_reproducible():
    rng = np.random.default_rng(1)
    frames = []
    for identity in range(10):
        frames.append(pd.DataFrame(dict(identity=identity, hours=np.arange(40), value=rng.normal(size=40))))
    frame = pd.concat(frames, ignore_index=True)
    pos = pd.DataFrame(dict(identity=np.arange(10), x=np.arange(10), y=0))
    a = spatial.neighbour_coordination(frame, pos, neighbours=2, permutations=49)
    b = spatial.neighbour_coordination(frame, pos, neighbours=2, permutations=49)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert len(a[0]) == 45
    assert a[2].cells.iloc[0] == 10
    assert a[2].permutations.iloc[0] == 49
    assert 0 < a[2].p_value.iloc[0] <= 1


def test_coverage_unknown_is_not_vacancy_and_observation_break_resets_age():
    occupied = np.array([[[1, 0]], [[0, 1]], [[0, 0]], [[0, 0]], [[1, 0]]], bool)
    valid = np.ones_like(occupied); valid[2, 0, 0] = False
    domain, age, table = spatial.coverage_gaps(occupied, np.arange(5), valid)
    assert domain.all()
    assert np.isnan(age[0, 0, 1])
    assert age[1, 0, 0] == 1
    assert np.isnan(age[3, 0, 0])
    assert age[4, 0, 0] == 0
    assert table.measurable_pixels.tolist() == [2, 2, 1, 2, 2]


def test_all_six_are_registered_and_tectonics_still_present():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _schema import load_all
    figures = load_all()
    slugs = ["tissue-expansion-sequence", "spatial-rhythm-maps", "spatial-rhythm-progression",
             "reporter-shape-timing", "neighbour-coordination", "tissue-coverage-gaps"]
    assert "tissue-tectonics" in figures
    for slug in slugs:
        assert slug in figures
        names = {o.name for o in figures[slug].options}
        if slug != "tissue-coverage-gaps":
            assert set(spatial.workbench.DETREND_DEFAULTS) <= names
        else:
            assert "detrend" not in names  # binary occupancy has no baseline fit


def test_spatial_renderer_is_independent_of_measurement_code():
    source = (Path(__file__).parent / "figures/panels/spatial.py").read_text()
    assert "from analysis" not in source
    assert "estimate_grouped_rhythms" not in source


def test_spatial_plot_plan_accepts_shared_detrending_and_metric_sweeps():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _schema import load_all
    from analysis.plots import parse, expand
    requests = parse([{"figure": "spatial-rhythm-maps", "for_each": {
        "spatial_metric": ["area_px", "corrected_mean"]}, "options": {
        "detrend": "polynomial", "detrend_polynomial_degree": 6,
        "fit_method": "fft_nlls", "significance_method": "lomb",
        "phase_group_hours": [6, 24], "period_min_hours": 2, "period_max_hours": 48}}], {})
    items = expand(requests, load_all())
    assert len(items) == 2


@pytest.mark.parametrize("slug", ["tissue-expansion-sequence", "tissue-coverage-gaps", "spatial-rhythm-progression"])
def test_timepoint_controls_work_in_main_analysis(slug):
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _schema import load_all
    from analysis.plots import parse, expand
    specs = load_all()
    assert specs[slug].option("snapshot_count").default == 1
    options = dict(snapshot_count=4, snapshot_hours=[1, 13, 25, 49])
    if slug != "tissue-expansion-sequence":
        assert specs[slug].option("show_history").default is False
        options["show_history"] = True
    items = expand(parse([{"figure": slug, "options": options}], {}), specs)
    assert len(items) == 1


def test_irrelevant_spatial_flags_are_not_advertised():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _schema import load_all
    specs = load_all()
    assert "spatial_arrows" not in {o.name for o in specs["spatial-rhythm-maps"].options}
    assert "phase_group_hours" not in {o.name for o in specs["tissue-expansion-sequence"].options}


def test_generic_field_map_keeps_requested_panel_width_and_common_colour_limits():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from panels import spatial as panels
    import matplotlib.pyplot as plt
    data = pd.DataFrame(dict(record="pixel", tile=[0, 1], hours=[0, 2],
                             x=[1, 1], y=[1, 1], value=[-1., 1.]))
    config = dict(kind="expansion", shape=[10, 12], scale=1., length_unit="px",
                   panel_inches=6., cmap="RdBu_r", vmin=-2., vmax=2.,
                   value_label="Value", title="Title", subtitle="Subtitle", footnote="Footnote")
    fig = panels.render(data, config)
    fig.canvas.draw()
    maps = [ax for ax in fig.axes if ax.images]
    assert len(maps) == 2
    for ax in maps:
        assert ax.get_position().width * fig.get_figwidth() == pytest.approx(6.)
        assert ax.images[0].get_clim() == (-2, 2)
    plt.close(fig)


def test_layout_refresh_keeps_original_input_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    import refresh_spatial
    slug = "neighbour-coordination"
    bundle = tmp_path / "figures" / slug
    bundle.mkdir(parents=True)
    (bundle / "plot.py").write_text("CONFIG = {}")
    (bundle / "README.md").write_text("Original analysis")
    (bundle / "src_measurement.csv").write_text("value\n1\n")
    (tmp_path / "measurement.csv").write_text("value\n999\n")
    pd.DataFrame({"value": [1]}).to_csv(bundle / "figure_data.csv", index=False)
    pd.DataFrame([{"copied_path": "src_measurement.csv", "original_path": str(tmp_path / "measurement.csv")}]).to_csv(
        bundle / "sources.csv", index=False)
    monkeypatch.setattr(refresh_spatial, "load_all", lambda: {slug: SimpleNamespace(summary="test")})
    monkeypatch.setattr(refresh_spatial.spatial, "render", lambda *a: SimpleNamespace(axes=[]))
    captured = []
    monkeypatch.setattr(refresh_spatial, "finish", lambda ctx, result: captured.append(ctx))
    refresh_spatial.refresh(tmp_path, slug)
    assert captured[0].sources["measurement.csv"] == bundle / "src_measurement.csv"
    assert captured[0].source_origins["measurement.csv"] == str(tmp_path / "measurement.csv")


def test_cached_progression_can_switch_between_final_named_and_full_history(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    import refresh_spatial
    traces = pd.DataFrame(dict(identity=[1, 1, 1, 2, 2, 2], hours=[1, 2, 3] * 2,
                              frame_index=[0, 1, 2] * 2, value=[1., 2., np.nan, 4., 5., 6.]))
    traces.to_csv(tmp_path / "der_display_traces.csv", index=False)
    cells = pd.DataFrame(dict(record="cell", identity=[1, 2], x=[1, 2], y=[1, 2], spatial_order=[1, 2]))
    cfg = dict(kind="progression", hours=[1, 2, 3], subtitle="Spatially ordered values")
    final, cfg, extra = refresh_spatial.time_view(tmp_path, cells, cfg)
    assert final[final.record.eq("trace")].hours.tolist() == [3, 3]
    assert np.isnan(final.loc[final.record.eq("trace") & final.identity.eq(1), "value"].iloc[0])
    assert extra["der_snapshots.csv"].frame_index.tolist() == [2]
    named, cfg, _ = refresh_spatial.time_view(tmp_path, final, cfg, snapshot_hours=[1, 3])
    assert set(named[named.record.eq("trace")].hours) == {1, 3}
    full, cfg, _ = refresh_spatial.time_view(tmp_path, named, cfg, show_history=True)
    assert len(full[full.record.eq("trace")]) == 6
    assert cfg["hours"] == [1, 2, 3]
    # Starting from a final-only encoding never loses the other cached frames.
    assert pd.read_csv(tmp_path / "der_display_traces.csv").shape == traces.shape


def test_empty_final_field_is_not_replaced_by_earlier_occupancy(tmp_path):
    import tifffile
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    import refresh_spatial
    from panels import spatial as panels
    import matplotlib.pyplot as plt
    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[0, 1, 1] = 1
    tifffile.imwrite(tmp_path / "src_labels.tif", labels, photometric="minisblack")
    pd.DataFrame(dict(identity=[1], frame_index=[0], hours=[1], value=[2.])).to_csv(
        tmp_path / "der_display_traces.csv", index=False)
    pd.DataFrame(dict(settings_json=["{}"])).to_csv(tmp_path / "der_settings.csv", index=False)
    prior = pd.DataFrame(dict(record=["pixel"], tile=[0], hours=[1], x=[1], y=[1], value=[2.]))
    cfg = dict(kind="expansion", hours=[1, 2], shape=[4, 4], scale=1., length_unit="px",
               panel_inches=6., cmap="RdBu_r", vmin=-2., vmax=2.,
               value_label="Value", title="Title", subtitle="Subtitle", footnote="Footnote")
    final, cfg, _ = refresh_spatial.time_view(tmp_path, prior, cfg)
    assert final.empty
    fig = panels.render(final, cfg)
    maps = [ax for ax in fig.axes if ax.images]
    assert len(maps) == 1
    assert maps[0].images[0].get_array().mask.all()
    assert maps[0].get_title() == "2 h"
    plt.close(fig)


def test_snapshot_matrix_does_not_paint_unobserved_intervals():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from panels import spatial as panels
    import matplotlib.pyplot as plt
    traces = pd.DataFrame(dict(identity=[1, 1], hours=[1, 49], value=[1., 2.]))
    fig, ax = plt.subplots()
    handle = panels.spatial_matrix(ax, traces, order=[1], hours=[1, 49], discrete=True)
    np.testing.assert_array_equal(handle.get_coordinates()[0, :, 0], [-.5, .5, 1.5])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["1", "49"]
    plt.close(fig)


@pytest.mark.parametrize("history", [False, True])
def test_coverage_history_is_opt_in(history):
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from panels import spatial as panels
    import matplotlib.pyplot as plt
    data = pd.DataFrame(dict(record=["pixel", "coverage"], tile=[0, np.nan], hours=[2, 2],
                             x=[1, np.nan], y=[1, np.nan], value=[0., .5]))
    cfg = dict(kind="gaps", snapshots=[dict(tile=0, hours=2)], show_history=history,
               hours=[1, 2], hour_ticks=[1, 2], shape=[4, 4], scale=1., length_unit="px",
               panel_inches=6., cmap="magma", vmin=0., vmax=2., line_color="black",
               value_label="Value", title="Title", subtitle="Subtitle", footnote="Footnote")
    fig = panels.render(data, cfg)
    assert len([ax for ax in fig.axes if ax.lines]) == int(history)
    assert len([ax for ax in fig.axes if ax.images]) == 1
    plt.close(fig)


def test_tracks_never_bridge_missing_frames_or_positions():
    frame = pd.DataFrame(dict(identity=1, frame_index=[0, 1, 3, 4, 5, 6],
                              hours=[0, 1, 3, 4, 5, 6],
                              soma_x=[0, 1, 3, np.nan, 5, 6], soma_y=2.))
    tracks = spatial.track_segments(frame, length_per_pixel=2)
    assert tracks.from_frame_index.tolist() == [0, 5]
    assert tracks.frame_index.tolist() == [1, 6]
    assert tracks.x.tolist() == [0, 10]
    assert tracks.x2.tolist() == [2, 12]


def _track_fixture():
    frame = pd.DataFrame(dict(identity=[1, 1, 2, 2, 3, 3], frame_index=[0, 1] * 3,
                              hours=[0, 1] * 3, soma_x=[0, 1, 2, 3, 4, 5], soma_y=0.))
    cells = pd.DataFrame(dict(identity=[1, 2, 3], record="cell", tile="period",
                              value=[8., np.nan, 30.], significant=[True, False, True],
                              display_status=["supported rhythm", "not significant", "exploratory period"]))
    return frame, cells


def test_tracks_default_to_rhythmic_cells_not_only_supported_periods():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _spatial import period_track_data
    frame, cells = _track_fixture()
    default = period_track_data(frame, cells)
    assert default.identity.tolist() == [1, 3]
    assert default.value.tolist() == [8., 30.]
    all_cells = period_track_data(frame, cells, scope="all")
    assert all_cells.identity.tolist() == [1, 2, 3]
    assert np.isnan(all_cells.loc[all_cells.identity.eq(2), "value"].iloc[0])
    with pytest.raises(ValueError, match="rhythmic or all"):
        period_track_data(frame, cells, scope="unknown")


def test_period_track_options_are_available_in_main_analysis():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _schema import load_all
    from analysis.plots import parse, expand
    specs = load_all()
    spec = specs["spatial-rhythm-maps"]
    assert spec.option("spatial_tracks").default is True
    assert spec.option("spatial_track_cells").default == "rhythmic"
    for scope in ("rhythmic", "all"):
        requests = parse([{"figure": spec.slug, "options": {
            "spatial_tracks": True, "spatial_track_cells": scope, "spatial_track_width": 2.}}], {})
        assert len(expand(requests, specs)) == 1
    assert "spatial_tracks" not in {o.name for o in specs["neighbour-coordination"].options}


def test_generic_track_map_uses_period_scale_and_grey_missing_values():
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from panels import spatial as panels
    from matplotlib.colors import to_rgba
    from _spatial import period_track_data
    import matplotlib.pyplot as plt
    tracks = period_track_data(*_track_fixture(), scope="all")
    fig, ax = plt.subplots()
    handle = panels.track_map(ax, tracks, vmin=2, vmax=48, linewidth=2.)
    assert len(handle.get_segments()) == 3
    np.testing.assert_allclose(handle.get_colors()[1], to_rgba("#999999"))
    np.testing.assert_allclose(handle.get_colors()[0], plt.get_cmap("viridis")((8 - 2) / 46))
    assert handle.get_linewidths()[0] == 2.
    assert len(panels.track_map(ax, pd.DataFrame()).get_segments()) == 0
    plt.close(fig)


def test_cached_tracks_switch_scope_without_changing_saved_cell_results(tmp_path):
    import json
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    import refresh_spatial
    frame, cells = _track_fixture()
    frame.to_csv(tmp_path / "src_cell_frame.csv", index=False)
    pd.DataFrame(dict(settings_json=[json.dumps({"spatial_centre": "soma"})])).to_csv(
        tmp_path / "der_settings.csv", index=False)
    cfg = dict(kind="period", scale=1., footnote="Original test settings.")
    first, cfg, extra = refresh_spatial.track_view(tmp_path, cells, cfg)
    assert first[first.record.eq("track")].identity.tolist() == [1, 3]
    assert cfg["spatial_track_cells"] == "rhythmic"
    pd.testing.assert_frame_equal(first[first.record.eq("cell")][cells.columns].reset_index(drop=True),
                                  cells, check_dtype=False)
    full, cfg, _ = refresh_spatial.track_view(tmp_path, first, cfg, scope="all")
    assert full[full.record.eq("track")].identity.tolist() == [1, 2, 3]
    off, cfg, extra = refresh_spatial.track_view(tmp_path, full, cfg, enabled=False)
    assert off.record.eq("cell").all()
    assert extra["der_tracks.csv"].empty
    assert cfg["footnote"] == "Original test settings."
