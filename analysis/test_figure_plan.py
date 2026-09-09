"""Regression tests for the figure-plan primitives and measurements."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from analysis.modules import contacts, coupling, regimes, sholl, territory
from analysis.registry import MeasurementContext
from analysis.theme import load_theme
from analysis.units import Scale
from panels import common
from panels import coupling as coupling_panels


def _context(labels: np.ndarray, **params) -> MeasurementContext:
    return MeasurementContext(
        stem="test", labels=labels.astype(np.uint16), raw=(labels > 0).astype(float),
        scale=Scale(30.0), identities=[int(value) for value in np.unique(labels) if value],
        params=params,
    )


def test_rose_counts_cover_exactly_one_period():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    drawn = common.rose(ax, [0, 1, 23, 24], load_theme(), bins=24)
    assert drawn.data["bin_start"].iloc[0] == 0
    assert drawn.data["bin_end"].iloc[-1] == 24
    assert drawn.data["count"].sum() == 4
    plt.close(fig)


def test_rose_labels_phase_in_cycle_time_instead_of_degrees():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    common.rose(
        ax, [0, 6, 12, 18], load_theme(), period=24,
        tick_step=6, phase_unit="h",
    )
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "0 h", "6 h", "12 h", "18 h",
    ]
    plt.close(fig)


def test_vectors_resultant_is_a_concentration():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    drawn = common.vectors(ax, [0, 6, 12, 18], [1, 1, 1, 1], load_theme())
    assert 0 <= drawn.extra["concentration"] <= 1
    assert len(drawn.data) == 4
    plt.close(fig)


def test_vectors_radial_scale_contains_the_longest_arrow():
    """Annotation arrows need an explicit limit because they do not autoscale axes."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    common.vectors(ax, [0, 6], [250, 4_000], load_theme())
    assert ax.get_ylim()[1] >= 4_000
    plt.close(fig)


def test_contact_metric_shuffle_keeps_contact_edges_and_pair_values():
    contacts = pd.DataFrame({
        "identity_a": [1, 3, 5, 7],
        "identity_b": [2, 4, 6, 8],
        "hours_in_contact": [1.0, 2.0, 3.0, 4.0],
    })
    values = pd.Series(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        index=range(1, 9),
    )
    pairs, result = coupling.contact_metric_permutation_test(
        contacts, values, shuffles=500, random_state=7,
    )
    assert list(zip(pairs["identity_a"], pairs["identity_b"])) == [
        (7, 8), (5, 6), (3, 4), (1, 2)
    ]
    assert np.allclose(pairs["pair_difference"], 0.0)
    assert result["difference_ratio"] == 0.0
    assert result["pairs"] == 4


def test_contact_metric_circular_difference_wraps_at_cycle_end():
    contacts = pd.DataFrame({
        "identity_a": [1, 3, 5],
        "identity_b": [2, 4, 6],
        "hours_in_contact": [3.0, 2.0, 1.0],
    })
    values = pd.Series([0.95, 0.05, 0.2, 0.4, 0.6, 0.9], index=range(1, 7))
    pairs, _ = coupling.contact_metric_permutation_test(
        contacts, values, circular_period=1.0, shuffles=200, random_state=11,
    )
    assert pairs.loc[pairs["identity_a"].eq(1), "pair_difference"].iloc[0] == \
        pytest.approx(0.1)


def test_pair_metric_ledger_maps_requested_pair_and_metric_order():
    frame = pd.DataFrame({
        "pair": ["1-2", "1-2", "3-4", "3-4"],
        "metric": ["area", "speed", "area", "speed"],
        "difference_ratio": [0.5, 1.5, 0.8, 1.2],
        "hours_in_contact": [2.0, 2.0, 5.0, 5.0],
    })
    fig, ax = plt.subplots()
    drawn = coupling_panels.pair_metric_ledger(
        ax, frame, load_theme(), pair_order=["3-4", "1-2"],
        metric_order=["speed", "area"],
        metric_labels={"speed": "Speed", "area": "Area"},
    )
    assert drawn.extra["matrix"].tolist() == [[1.2, 0.8], [1.5, 0.5]]
    assert drawn.data["display_row"].tolist() == [0, 0, 1, 1]
    plt.close(fig)


def test_matrix_annotations_match_cells():
    fig, ax = plt.subplots()
    common.matrix(ax, [[1.25, 2.5]], load_theme(), row_labels=["a"],
                  column_labels=["x", "y"], fmt="{:.2f}")
    assert [text.get_text() for text in ax.texts] == ["1.25", "2.50"]
    plt.close(fig)


def test_stacked_histogram_is_generic_and_returns_one_axes_per_group():
    fig = plt.figure()
    drawn = common.stacked_histogram(
        fig, (0.1, 0.1, 0.8, 0.8), [[1, 2], [2, 3]], load_theme(),
        labels=["arbitrary group a", "arbitrary group b"], bins=4,
        limits=(0, 4), overlap=0.2, labels_outside=True,
        x_label="Any numeric measurement",
    )
    assert len(drawn.axes) == 2
    assert set(drawn.data["group"]) == {"arbitrary group a", "arbitrary group b"}
    assert len(drawn.data) == 8
    assert drawn.axes[-1].get_xlabel() == "Any numeric measurement"
    plt.close(fig)


def test_ridgeline_remains_a_compatibility_name_for_stacked_histogram():
    fig = plt.figure()
    drawn = common.ridgeline(
        fig, (0.1, 0.1, 0.8, 0.8), [[1, 2]], load_theme(), labels=["a"],
    )
    assert len(drawn.axes) == 1
    assert set(drawn.data["group"]) == {"a"}
    plt.close(fig)


def test_stacked_area_returns_the_bands_it_draws():
    fig, ax = plt.subplots()
    drawn = common.stacked_area(ax, [0, 1], ["a", "b"],
                                {"a": [1, 3], "b": [1, 1]}, load_theme(), normalise=True)
    assert np.allclose(drawn.data["total"], 1)
    plt.close(fig)


def test_flow_conserves_each_stage_pair():
    fig = plt.figure()
    observations = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
    drawn = common.flow(fig, (0.1, 0.1, 0.8, 0.8), observations, load_theme(),
                        class_labels=["a", "b"])
    assert drawn.data.groupby("from_stage")["count"].sum().eq(len(observations)).all()
    plt.close(fig)


def test_paired_slopes_returns_one_row_per_subject():
    fig, ax = plt.subplots()
    drawn = common.paired_slopes(ax, [1, 2, 3], [2, 1, 4], load_theme())
    assert drawn.data["left"].tolist() == [1, 2, 3]
    assert drawn.data["right"].tolist() == [2, 1, 4]
    assert drawn.extra["segments"].shape == (3, 2, 2)
    plt.close(fig)


def test_event_average_interval_contains_the_mean():
    fig, ax = plt.subplots()
    table = common.event_average(ax, [-1, 0, 1], [[0, 1, 0], [1, 2, 1]],
                                 load_theme(), bootstrap=100).data
    assert ((table["lo"] <= table["mean"]) & (table["mean"] <= table["hi"])).all()
    assert table["n_events"].tolist() == [2, 2, 2]
    plt.close(fig)


def test_dendrogram_returns_scipys_leaf_order():
    tree = linkage(np.array([[0.0], [1.0], [5.0], [6.0]]), method="average")
    expected = scipy_dendrogram(tree, no_plot=True)["leaves"]
    fig, ax = plt.subplots()
    assert common.dendrogram(ax, tree, load_theme()).extra["leaf_order"] == expected
    plt.close(fig)


def test_territory_writes_tables_and_fixed_field_stacks():
    labels = np.zeros((3, 6, 6), dtype=np.uint16)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 2:4] = 1
    labels[2, 1:3, 3:5] = 1
    output = territory.measure(_context(labels, territory={"core_fraction": 0.8}))
    assert {"territory", "territory_frame", "revisit_count", "first_owner",
            "owner_count", "never_visited"} <= set(output)
    assert output["revisit_count"].shape == labels.shape[1:]
    assert output["territory_frame"]["cumulative_unique_px"].is_monotonic_increasing


def test_territory_coverage_order_test_preserves_active_frames_and_detects_late_discovery():
    occupancy = np.zeros((12, 1, 6), dtype=bool)
    active_frames = np.array([1, 3, 5, 7, 9, 11])
    for slot, frame_index in enumerate(active_frames):
        occupancy[frame_index, 0, 0] = True
        if slot >= 2:
            occupancy[frame_index, 0, slot] = True

    units, curves, result = territory.coverage_order_permutation_test(
        {"cell-a": occupancy}, shuffles=500, random_state=7,
    )

    observed = curves.drop_duplicates("frame_index").set_index("frame_index")
    assert observed.loc[0, "observed_coverage_fraction"] == 0.0
    assert observed.loc[2, "observed_coverage_fraction"] == observed.loc[
        1, "observed_coverage_fraction"
    ]
    assert observed.loc[2, "permutation_median"] == observed.loc[
        1, "permutation_median"
    ]
    assert units.loc[0, "active_frames"] == len(active_frames)
    assert units.loc[0, "observed_auc"] < units.loc[0, "permutation_median_auc"]
    assert 0.0 < result["p_value"] <= 1.0
    assert result["test"] == "two-sided frame-order permutation test"


def test_sholl_is_long_by_ring_and_has_reach():
    labels = np.zeros((2, 12, 12), dtype=np.uint16)
    labels[:, 4:8, 4:8] = 1
    output = sholl.measure(_context(labels, sholl={"n_rings": 3, "centre": "centroid"}))
    # Two frames, three rings, and the profile measured under both ring widths.
    assert len(output["sholl"]) == 2 * 3 * len(sholl.SCALINGS)
    assert {"occupancy", "intersections", "radius_inner", "radius_outer",
            "scaling", "ring_width"} <= set(output["sholl"])
    assert set(output["sholl"]["scaling"]) == set(sholl.SCALINGS)
    assert len(output["sholl_reach"]) == 2
    # Every summary appears once per scaling and nowhere unsuffixed, so no
    # column can be read without knowing which x axis it was measured on.
    reach = output["sholl_reach"]
    for name, _, _ in sholl._SUMMARIES:
        assert f"sholl_{name}" not in reach
        for scaling in sholl.SCALINGS:
            assert f"sholl_{name}{sholl.SUFFIX[scaling]}" in reach


def test_default_classifier_crosses_three_size_and_three_movement_levels():
    rows = []
    identity = 1
    for area in (10.0, 100.0, 1000.0):
        for movement in (0.1, 1.0, 10.0):
            rows.append({
                "identity": identity,
                "frame_index": 0,
                "hours": 0.0,
                "area_px": area,
                "step_px_gapless": movement,
            })
            identity += 1
    context = _context(np.zeros((1, 2, 2)), regimes={"min_frames_per_cell": 1})
    output = regimes.derive(pd.DataFrame(rows), context)

    profiles = output["regime_profiles"].sort_values("regime")
    assert {"regime", "regime_label", "regime_size_level",
            "regime_movement_level", "regime_observations", "area_px",
            "step_px_gapless"} == set(profiles.columns)
    assert list(profiles["regime"]) == list(range(9))
    assert set(output["regimes"]["regime"]) == set(range(9))
    assert list(profiles["regime_label"]) == [
        f"{size.title()} size, {movement} movement"
        for size in regimes.LEVELS for movement in regimes.LEVELS
    ]
    assigned = output["regimes"].set_index(["regime_size_level", "regime_movement_level"])
    for size_index, size in enumerate(regimes.LEVELS):
        for movement_index, movement in enumerate(regimes.LEVELS):
            assert assigned.loc[(size, movement), "regime"] == size_index * 3 + movement_index
    assert {"regime_distance", "regime_second", "regime_margin"} <= set(output["regimes"])


def test_a_cell_frame_without_a_preceding_movement_is_not_classified():
    frame = pd.DataFrame({
        "identity": [1, 1, 1], "frame_index": [0, 1, 2], "hours": [0.0, 0.5, 1.0],
        "area_px": [10.0, 20.0, 30.0], "step_px_gapless": [np.nan, 1.0, 2.0],
    })
    output = regimes.derive(
        frame, _context(np.zeros((1, 2, 2)), regimes={"min_frames_per_cell": 1}))
    assert list(output["regimes"]["frame_index"]) == [1, 2]


def test_coupling_never_invents_lag_pairs_across_missing_frames():
    frame = pd.DataFrame({
        "identity": [1] * 6, "frame_index": [0, 1, 2, 5, 6, 7],
        "hours": np.array([0, 1, 2, 5, 6, 7]) / 2,
        "a": [0, 1, 2, 5, 6, 7], "b": [1, 2, 3, 6, 7, 8],
        "centroid_y": 1.0, "centroid_x": 1.0,
    })
    context = _context(np.zeros((1, 2, 2)), coupling={
        "metric_pairs": [["a", "b"]], "between_metrics": [],
        "min_overlap_frames": 3, "max_lag_frames": 1, "surrogates": 2,
    })
    output = coupling.derive(frame, context)["lag_profiles"]
    zero = output[output["lag_frames"] == 0].iloc[0]
    one = output[output["lag_frames"] == 1].iloc[0]
    assert zero["pairs"] == 6
    assert one["pairs"] == 4


def test_contacts_summarise_touching_identities():
    labels = np.zeros((3, 8, 8), dtype=np.uint16)
    labels[:, 2:6, 1:4] = 1
    labels[:, 2:6, 4:7] = 2
    output = contacts.measure(_context(labels, contacts={"dilation_px": 1}))
    assert len(output["contacts_frame"]) == 3
    assert output["contacts"].iloc[0]["frames_in_contact"] == 3
    assert output["contacts_frame"]["overlap_px"].eq(0).all()



# --------------------------------------------------- the footnote clearance


def _sheet(lines: int):
    """A figure with one axis, an x-label, and a footnote of `lines` lines."""
    figure = plt.figure(figsize=(8.0, 6.0))
    ax = figure.add_axes([0.1, 0.16, 0.8, 0.74])
    ax.set_xlabel("a label with descenders: pqgy")
    footnote = figure.text(0.02, 0.008, "\n".join(f"line {i}" for i in range(lines)),
                           fontsize=10, va="bottom")
    return figure, ax, footnote


def test_a_footnote_that_fits_moves_nothing():
    from _options import Length
    from _schema import _clear_footnote

    figure, ax, footnote = _sheet(1)
    before = tuple(ax.get_position().bounds), figure.get_figheight()
    _clear_footnote(figure, footnote, Length(0.008), [])
    assert (tuple(ax.get_position().bounds), figure.get_figheight()) == before
    plt.close(figure)


def test_a_footnote_that_would_sit_under_the_x_label_gets_more_paper():
    from _options import Length
    from _schema import _clear_footnote

    figure, ax, footnote = _sheet(9)
    height = figure.get_figheight()
    inches_from_top = (1.0 - ax.get_position().y1) * height
    _clear_footnote(figure, footnote, Length(0.008), [])
    assert figure.get_figheight() > height
    # The drawing keeps its size and its distance from the top edge: the sheet
    # grew downwards, it was not rescaled around the plot.
    assert (1.0 - ax.get_position().y1) * figure.get_figheight() == pytest.approx(
        inches_from_top, abs=1e-6)
    assert ax.get_position().height * figure.get_figheight() == pytest.approx(
        0.74 * height, abs=1e-6)
    plt.close(figure)


def test_the_header_text_keeps_its_place_when_the_sheet_grows():
    from _options import Length
    from _schema import _clear_footnote

    figure, _ax, footnote = _sheet(9)
    height = figure.get_figheight()
    title = figure.text(0.02, 1.0 - 0.20 / height, "a title", va="top")
    _clear_footnote(figure, footnote, Length(0.008), [title])
    assert (1.0 - title.get_position()[1]) * figure.get_figheight() == pytest.approx(
        0.20, abs=1e-6)
    plt.close(figure)
