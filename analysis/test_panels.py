"""Tests for the reusable panels and the metric vocabulary.

Run with ``python -m pytest analysis/test_panels.py``.

The property under test: a builder can be pointed at any numeric column and any
colour the user names, and either gets a correctly labelled panel or a refusal -
never a silently wrong one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from analysis.theme import load_theme  # noqa: E402
import matrix_ordering  # noqa: E402
from _metrics import METRICS, axis_label, describe, role_for, semantic_label  # noqa: E402
from panels import Look, image_strip, resolve_look, trace  # noqa: E402
from panels import common, intensity, motility, presence, rhythms, surveillance, territory  # noqa: E402


def _axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots()


# ------------------------------------------------------------- the vocabulary

def test_a_known_column_keeps_its_written_label():
    assert describe("corrected_mean").label == "Reporter intensity"
    assert role_for("turnover_index") == "surveillance"


def test_an_unknown_column_is_still_labelled_rather_than_refused():
    """A user may trace any column; a missing entry is worse wording, not a crash."""
    metric = describe("some_new_readout_px")
    assert metric.label == "Some new readout"
    assert metric.unit == "px"


def test_a_per_frame_unit_names_the_actual_interval():
    """Hard-coding 30 min makes the label wrong on a 15 min recording."""
    assert axis_label("turnover_index", 30.0) == (
        "Footprint turnover fraction\n(fraction per 30 min)"
    )
    assert axis_label("turnover_index", 15.0) == (
        "Footprint turnover fraction\n(fraction per 15 min)"
    )


def test_a_unit_that_is_a_quantity_gets_brackets():
    assert axis_label("area_px", 30.0) == "Area\n(px)"


def test_every_entry_names_a_role_the_theme_can_resolve():
    theme = load_theme()
    for metric in METRICS.values():
        assert theme.colour(metric.role).startswith("#"), metric.column


def test_the_wording_comes_from_the_module_that_writes_the_column():
    """Not a lookup table here: ``describe`` reads what ``surveillance`` declared.

    The point of the arrangement is that changing the label means editing the
    module that computes the number, so the two cannot drift apart. If this
    ever passes by coincidence - the same words written in two places - the
    drift it guards against has already started.
    """
    import analysis.modules  # noqa: F401  - registers every module
    from analysis.registry import get_module

    declared = {column.name: column for column in get_module("surveillance").produces}
    assert describe("turnover_index").label == declared["turnover_index"].label
    assert role_for("jaccard") == declared["jaccard"].role


def test_every_planned_dynamic_label_has_a_written_meaning():
    """Database column names must never become unexplained labels on the figure."""
    plotted = {
        "area_px", "ramification_index", "corrected_mean", "turnover_index",
        "jaccard", "msd_alpha", "unclaimed_px", "unclaimed_fraction",
        "identities_named", "identities_expected", "frame_gained_share",
        "core_share", "occupancy", "intersections", "determinism", "dtw_distance",
    }
    for column in plotted:
        label = semantic_label(column)
        assert "_" not in label and label != column


def test_pixel_fates_have_distinct_colours_and_explanations():
    theme = load_theme()
    colours = [theme.colour(territory.FATE_ROLES[name]) for name in territory.FATE_ROLES]
    assert len(colours) == len(set(colours))
    assert set(territory.FATE_ROLES) == set(territory.FATE_LABELS)


def test_pixel_fates_accept_named_event_windows_and_detect_owner_transfer():
    labels = np.zeros((4, 2, 2), dtype=np.uint16)
    labels[0:2, 0, 0] = 1
    labels[2:4, 0, 0] = 2
    states, composition = territory.pixel_fate_states(
        labels, {"Baseline": [0, 1], "Treatment": [2, 3]},
        core_fraction=0.8, transient_fraction=0.2,
    )
    codes = {name: index for index, name in enumerate(territory.FATE_ROLES)}
    assert states.tolist() == [[codes["core"], codes["transferred"]]]
    treatment = composition[composition["event"].eq("Treatment")]
    assert treatment.loc[treatment["class"].eq("transferred"), "pixels"].iloc[0] == 1
    assert set(composition["event"]) == {"Baseline", "Treatment"}


def test_pixel_fate_flow_accepts_custom_classes_and_event_labels():
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 4))
    try:
        drawn = territory.fate_flow(
            figure, (0.1, 0.1, 0.8, 0.7),
            np.asarray([["quiet", "active"], ["active", "active"]]),
            load_theme(), stage_labels=["Before\n0 h", "After\n24 h"],
            class_roles={"quiet": "missing", "active": "highlight"},
            class_labels={"quiet": "Quiet pixels", "active": "Active pixels"},
        )
        assert set(drawn.data["from_class"]) == {"quiet", "active"}
        assert drawn.data["pixels"].sum() == 2
        assert drawn.extra["class_labels"]["active"] == "Active pixels"
    finally:
        plt.close(figure)


def test_generic_coverage_hues_distinguish_every_observed_source_combination():
    import matplotlib.pyplot as plt

    first = np.zeros((3, 2, 2), dtype=bool)
    second = np.zeros_like(first)
    first[0, 0, 0] = True
    first[1, 0, 1] = True
    second[1, 1, 0] = True
    second[2, 0, 0] = True
    figure = plt.figure(figsize=(8, 4))
    try:
        drawn = territory.coverage_history(
            figure, (0.1, 0.1, 0.65, 0.8),
            {"first_kind": first, "second_kind": second}, load_theme(),
            hours=[0.0, 1.0, 2.0], view="final",
            source_labels={"first_kind": "First kind", "second_kind": "Second kind"},
            class_colours={
                (): "missing",
                ("first_kind",): "morphology",
                ("second_kind",): "unclaimed",
                ("first_kind", "second_kind"): "surveillance",
            },
        )
        occupied = drawn.data.loc[drawn.data["pixels"] > 0]
        assert set(occupied["occupancy_class"]) == {
            "Never occupied", "First kind only", "Second kind only",
            "First kind and Second kind",
        }
        assert occupied["colour"].nunique() == 4
        assert len(drawn.axes) == 1
        assert drawn.extra["frame_indices"].tolist() == [2]
    finally:
        plt.close(figure)


def test_coverage_composition_sums_to_the_whole_field_at_every_frame():
    figure, ax = _axes()
    first = np.zeros((3, 2, 2), dtype=bool)
    second = np.zeros_like(first)
    first[0, 0, 0] = True
    second[1, 1, 1] = True
    try:
        drawn = territory.coverage_composition(
            ax, [0.0, 1.0, 2.0], {"first": first, "second": second},
            load_theme(), hour_ticks=1.0,
        )
        assert np.allclose(drawn.data.groupby("frame_index")["share"].sum(), 1.0)
        assert drawn.data.groupby("frame_index")["total_occupied_share"].first().tolist() == [
            0.25, 0.5, 0.5,
        ]
        assert 0.0 in ax.get_xticks()
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_coverage_view_refuses_an_ambiguous_combined_choice():
    import matplotlib.pyplot as plt

    figure = plt.figure()
    try:
        with pytest.raises(ValueError, match="final or stages"):
            territory.coverage_history(
                figure, (0.1, 0.1, 0.8, 0.8),
                {"source": np.ones((2, 2), dtype=bool)}, load_theme(),
                view="both",
            )
    finally:
        plt.close(figure)


def test_revisit_grid_labels_each_cell_and_keeps_one_crop_scale():
    import matplotlib.pyplot as plt

    first = np.zeros((10, 10), dtype=float)
    second = np.zeros((10, 10), dtype=float)
    first[2:5, 3:6] = 4
    second[4:9, 1:4] = 6
    figure = plt.figure(figsize=(8, 4))
    try:
        drawn = territory.revisit_grid(
            figure, (0.08, 0.12, 0.76, 0.72), {1: first, 2: second}, load_theme(),
            observed_frames={1: 8, 2: 12}, soma={1: (4.0, 3.0), 2: (2.0, 6.0)},
            max_columns=2,
        )
        assert [ax.get_title() for ax in drawn.axes] == ["Cell 1", "Cell 2"]
        assert drawn.data["crop_size_px"].nunique() == 1
        assert drawn.data["maximum_occupancy_fraction"].tolist() == [0.5, 0.5]
    finally:
        plt.close(figure)


def test_coverage_curve_names_the_permutation_interval_and_records_its_values():
    figure, ax = _axes()
    hours = np.arange(4, dtype=float)
    try:
        drawn = territory.coverage_curve(
            ax, hours, [[1, 2, 3, 4], [1, 1, 3, 4]], load_theme(),
            denominator=[4, 4], permutation_median=[0.25, 0.5, 0.75, 1.0],
            permutation_lo=[0.2, 0.4, 0.7, 1.0],
            permutation_hi=[0.3, 0.6, 0.8, 1.0], test_p_value=0.031,
            shuffles=1_000,
        )
        labels = [artist.get_label() for artist in [*ax.lines, *ax.collections]]
        assert "Frame-order permutation median" in labels
        assert "95% frame-order permutation interval" in labels
        assert "Frames in random order" not in labels
        assert drawn.data["permutation_lo"].notna().all()
        assert "p = 0.031" in ax.texts[0].get_text()
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_owner_assignment_can_use_first_last_or_most_frequent_occupant():
    labels = np.asarray([
        [[2, 1, 0]],
        [[1, 2, 3]],
        [[1, 0, 3]],
    ])
    assert territory.owner_assignment(labels, "first").tolist() == [[2, 1, 3]]
    assert territory.owner_assignment(labels, "last").tolist() == [[1, 2, 3]]
    # The middle pixel is a one-frame tie; the smaller identity wins stably.
    assert territory.owner_assignment(labels, "most_frequent").tolist() == [[1, 1, 3]]


def test_cell_metric_map_paints_arbitrary_values_and_records_display_limits():
    figure, ax = _axes()
    try:
        drawn = territory.cell_metric_map(
            ax, [[1, 1], [2, 0]], pd.Series({1: 2.0, 2: 5.0}),
            load_theme(), field={"width": 2, "height": 2},
            metric="measurement", label="Measurement", range_mode="full",
        )
        rows = drawn.data.set_index("identity")
        assert rows.loc[1, "assigned_pixels"] == 2
        assert rows.loc[2, "assigned_pixels"] == 1
        assert rows["metric_value"].to_dict() == {1: 2.0, 2: 5.0}
        assert rows["display_minimum"].eq(2.0).all()
        assert rows["display_maximum"].eq(5.0).all()
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_cell_metric_map_robust_limits_weight_each_cell_once():
    figure, ax = _axes()
    owners = np.ones((20, 20), dtype=int)
    owners[0, 0] = 2
    owners[0, 1] = 3
    values = pd.Series({1: 0.0, 2: 10.0, 3: 100.0})
    try:
        drawn = territory.cell_metric_map(
            ax, owners, values, load_theme(),
            field={"width": 20, "height": 20}, metric="measurement",
            label="Measurement", range_mode="robust",
        )
        # Per-pixel percentiles would collapse near zero because identity 1
        # owns 398 pixels. Per-cell percentiles retain the small territories.
        assert drawn.extra["display_maximum"] > 90.0
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_direct_pixel_handoffs_exclude_appearances_and_disappearances():
    labels = np.asarray([
        [[1, 0], [1, 2]],
        [[2, 1], [0, 2]],
        [[2, 1], [3, 1]],
    ])
    counts, events = territory.direct_pixel_handoffs(labels)
    assert counts.tolist() == [[1, 0], [0, 1]]
    assert events[["from_identity", "to_identity", "transferred_pixels"]].to_dict(
        "records"
    ) == [
        {"from_identity": 1, "to_identity": 2, "transferred_pixels": 1},
        {"from_identity": 2, "to_identity": 1, "transferred_pixels": 1},
    ]


# ------------------------------------------------------------------ the look

def test_a_colour_map_name_becomes_a_map():
    look = resolve_look(load_theme(), "viridis")
    assert look.is_map


def test_a_role_name_becomes_a_flat_colour():
    theme = load_theme()
    look = resolve_look(theme, "reporter")
    assert not look.is_map and look.colour == theme.colour("reporter")


def test_a_hex_value_is_taken_as_written():
    assert resolve_look(load_theme(), "#123456").colour == "#123456"


def test_a_typo_is_refused_rather_than_drawn_in_the_wrong_colour():
    with pytest.raises(KeyError):
        resolve_look(load_theme(), "repoter")


# ---------------------------------------------------------------- the panels

def test_a_trace_with_a_flat_colour_draws_a_line():
    theme = load_theme()
    figure, ax = _axes()
    try:
        trace(ax, [0, 1, 2], [1.0, 2.0, 3.0], theme, look=Look(colour="#c0392b"))
        assert len(ax.lines) == 1
        assert ax.lines[0].get_color() == "#c0392b"
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_a_trace_with_a_map_is_coloured_along_time():
    theme = load_theme()
    figure, ax = _axes()
    try:
        trace(ax, [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], theme,
              look=resolve_look(theme, "viridis"))
        assert not ax.lines
        assert len(ax.collections) == 1
        # one segment fewer than points, each carrying its own hour
        assert len(ax.collections[0].get_array()) == 3
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_a_strip_shares_one_contrast_across_every_tile():
    """A per-tile stretch would hide the change the strip exists to show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = load_theme()
    figure = plt.figure(figsize=(6, 2))
    try:
        images = [np.full((8, 8), 10.0), np.full((8, 8), 200.0)]
        strip = image_strip(figure, (0.05, 0.1, 0.9, 0.8), images, theme)
        assert len(strip.axes) == 2
        limits = {tuple(ax.images[0].get_clim()) for ax in strip.axes}
        assert len(limits) == 1
        assert strip.extra["vmin"] < strip.extra["vmax"]
        # One row per tile, and the contrast recorded with it.
        assert strip.data["index"].tolist() == [0, 1]
        assert strip.data["vmin"].nunique() == 1
    finally:
        plt.close(figure)


def test_an_empty_strip_draws_nothing_rather_than_failing():
    """``--images 0`` is a supported page, not an error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    try:
        strip = image_strip(figure, (0.1, 0.1, 0.8, 0.8), [], load_theme())
        assert strip.axes == []
        assert strip.data.empty
    finally:
        plt.close(figure)


# ------------------------------------------------- a summary of a known column

def test_a_per_cell_summary_keeps_its_measurement_wording():
    """A summarised turnover column retains the footprint-turnover meaning."""
    assert describe("turnover_index_median").label == describe("turnover_index").label
    assert role_for("area_px_median") == "morphology"


# --------------------------------------------------------- the rhythms panels

def test_detrending_removes_a_straight_line():
    """The raster must show what the rhythm test saw, which is the residual."""
    hours = np.arange(0.0, 48.0, 0.5)
    ramp = 3.0 + 0.25 * hours
    assert np.allclose(rhythms.detrended_z(hours, ramp), 0.0, atol=1e-6)


def test_a_detrended_trace_is_in_units_of_its_own_spread():
    hours = np.arange(0.0, 48.0, 0.5)
    wave = 500.0 + 40.0 * np.sin(2 * np.pi * hours / 24.0)
    assert abs(rhythms.detrended_z(hours, wave).std() - 1.0) < 1e-6


def test_the_phase_histogram_covers_exactly_one_cycle():
    """A peak at 23.5 h and one at 0.5 h are neighbours, so the axis must wrap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        peaks = np.array([1.0, 5.0, 12.0, 23.5])
        passed = np.array([True, False, True, False])
        drawn = rhythms.phase_histogram(
            ax, peaks, load_theme(), passed=passed, bins=12
        )
        bands = drawn.data
        assert bands["bin_start_hour"].iloc[0] == 0.0
        assert bands["bin_end_hour"].iloc[-1] == 24.0
        assert bands["cells_all_tested"].sum() == len(peaks)
        assert (bands["cells_rhythmic_by_both_tests"]
                <= bands["cells_all_tested"]).all()
    finally:
        plt.close(figure)


def test_significant_period_histogram_counts_only_the_periods_it_is_given():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        drawn = rhythms.significant_period_histogram(
            ax, [3.0, 3.5, 10.0, np.nan], load_theme(),
            bins=[2.0, 6.0, 12.0, 24.0],
        )
        bands = drawn.data
        assert bands["significant_cell_count"].tolist() == [2, 1, 0]
        assert bands["significant_cell_count"].sum() == 3
        assert np.isclose(bands["significant_cell_frequency"].sum(), 1.0)
        assert drawn.extra["significant_cells"] == 3
        assert ax.get_xlabel() == "Estimated period (h)"
        assert ax.get_ylabel() == "Significant cells (count)"
    finally:
        plt.close(figure)


def test_paired_image_strip_keeps_pairs_large_and_repeats_the_before_outline():
    """The after crop must show both outlines, with a larger gap between cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 3))
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    shifted = np.zeros((8, 8), dtype=bool)
    shifted[3:6, 4:7] = True
    try:
        drawn = common.paired_image_strip(
            figure, (0.05, 0.1, 0.9, 0.8),
            [np.ones((8, 8)), np.ones((8, 8)) * 2],
            [np.ones((8, 8)) * 3, np.ones((8, 8)) * 4],
            load_theme(), before_masks=[mask, mask], after_masks=[shifted, shifted],
            within_pair_gap=0.002, between_pair_gap=0.04,
        )
        assert len(drawn.axes) == 4
        within = drawn.axes[1].get_position().x0 - drawn.axes[0].get_position().x1
        between = drawn.axes[2].get_position().x0 - drawn.axes[1].get_position().x1
        assert between > within
        assert len(drawn.axes[1].images) == 3  # raw, after outline, repeated before outline
        assert drawn.data.loc[drawn.data["state"] == "after", "before_reference_px"].gt(0).all()
    finally:
        plt.close(figure)


def test_raster_order_accepts_any_precomputed_columns():
    summary = pd.DataFrame({
        "identity": [1, 2, 3],
        "onset_hour": [8.0, 2.0, 2.0],
        "period_hours": [12.0, 18.0, 6.0],
    })
    assert matrix_ordering.ordered_identities(
        summary, ["onset_hour", "period_hours"]
    ) == [3, 2, 1]


def test_trace_pattern_order_puts_matching_blue_red_shapes_together():
    """Period order hid coherent colour bands that the raster should reveal."""
    matrix = pd.DataFrame(
        [
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [0.8, 1.1, -0.9, -1.2],
            [-0.9, -1.1, 0.8, 1.2],
        ],
        index=[11, 22, 33, 44],
    )

    order = matrix_ordering.trace_pattern_order(matrix)
    positions = {identity: position for position, identity in enumerate(order)}

    assert abs(positions[11] - positions[33]) == 1
    assert abs(positions[22] - positions[44]) == 1


def test_trace_pattern_order_defaults_to_the_accepted_principal_component_gradient():
    matrix = pd.DataFrame(
        [
            [-1.0, -0.8, 0.2, 1.0, 0.4],
            [1.0, 0.7, -0.1, -0.8, -0.4],
            [-0.7, -0.4, 0.3, 0.9, 0.6],
        ],
        index=[31, 12, 24],
    )

    assert matrix_ordering.trace_pattern_order(
        matrix,
    ) == matrix_ordering.trace_pattern_order(
        matrix, method="principal_component_gradient",
    )


def test_trace_pattern_order_retains_the_spectral_continuum_option():
    matrix = pd.DataFrame(
        [
            [-1.0, -0.8, 0.2, 1.0, 0.4],
            [1.0, 0.7, -0.1, -0.8, -0.4],
            [-0.7, -0.4, 0.3, 0.9, 0.6],
        ],
        index=[31, 12, 24],
    )

    order = matrix_ordering.trace_pattern_order(
        matrix, method="spectral_continuum",
    )

    assert sorted(order) == [12, 24, 31]
    assert order == matrix_ordering.trace_pattern_order(matrix, method="spectral")


def test_displayed_onset_is_the_first_sustained_blue_to_red_transition():
    matrix = pd.DataFrame(
        [
            [-1.0, -0.5, 0.2, 0.3, 0.4, -0.2],
            [0.2, 0.3, 0.4, -0.2, -0.3, -0.4],
            [-1.0, 0.2, -0.1, 0.3, 0.4, 0.5],
        ],
        index=[10, 20, 30],
        columns=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    )

    onsets = matrix_ordering.displayed_onsets(
        matrix, smooth_points=1, sustain_points=3,
    )

    assert onsets.to_dict() == {10: 1.0, 20: 0.0, 30: 1.5}


def test_trace_pattern_order_keeps_unusable_rows_at_the_bottom():
    matrix = pd.DataFrame(
        [[np.nan, np.nan, np.nan], [1.0, 0.0, -1.0], [np.nan, np.nan, np.nan]],
        index=[99, 11, 88],
    )

    order = matrix_ordering.trace_pattern_order(matrix)

    assert order == [11, 99, 88]


def test_rhythm_panel_exports_the_general_matrix_ordering_engine():
    assert rhythms.ordered_identities is matrix_ordering.ordered_identities
    assert rhythms.trace_pattern_order is matrix_ordering.trace_pattern_order
    assert rhythms.trace_displayed_onsets is matrix_ordering.displayed_onsets


def test_period_aware_timing_normalises_each_cells_own_cycle():
    """A 1.5 h peak on 6 h and a 9 h peak on 36 h share quarter-cycle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        drawn = rhythms.timing_by_period(
            ax,
            timing_hours=[1.5, 9.0],
            cycle_hours=[6.0, 36.0],
            detected_period_hours=[6.0, 36.0],
            theme=load_theme(),
            bins=4,
            period_edges=[2.0, 12.0, 24.0, 48.0],
        )
        cells = drawn.data[drawn.data["cells"] > 0]
        assert len(cells) == 2
        assert set(cells["phase_start_fraction"]) == {0.25}
        assert cells["cells"].sum() == 2
    finally:
        plt.close(figure)


# -------------------------------------------------------- the motility panels

def test_a_path_is_cut_at_a_gap_rather_than_bridged():
    """A bridged gap draws a journey the movie never showed."""
    frames = [0, 1, 2, 5, 6]
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (9.0, 0.0), (10.0, 0.0)]
    pieces = motility.path_segments(frames, points)
    assert len(pieces) == 2
    assert sum(len(piece) for piece in pieces) == 3      # 2 + 1 segments, not 4


def test_a_run_too_short_to_draw_contributes_no_segment():
    assert motility.path_segments([0, 4], [(0.0, 0.0), (5.0, 5.0)]) == []


def test_trajectory_width_and_dash_stay_with_each_identity_across_gaps():
    """Every segment retains its cell-level encodings after gap cutting."""
    figure, ax = _axes()
    try:
        groups = [
            ([0, 1, 4, 5], [(0, 0), (1, 0), (4, 0), (5, 0)], 2.0),
            ([0, 1, 2], [(0, 1), (1, 1), (2, 1)], 4.0),
        ]
        drawn = motility.trajectory_map(
            ax, groups, load_theme(), field={"width": 8, "height": 6},
            identities=[17, 29], line_widths=[0.8, 2.4],
            line_dash_categories=["Six-hour rhythm", "Twenty-four-hour rhythm"],
            line_dash_styles={
                "Six-hour rhythm": (0, (3, 2)),
                "Twenty-four-hour rhythm": (0, (9, 3)),
            },
        )
        assert drawn.extra["breaks"] == 1
        segments = drawn.data[drawn.data["mark_type"] == "segment"]
        assert set(segments.loc[segments["identity"] == 17, "line_width"]) == {0.8}
        assert set(segments.loc[segments["identity"] == 29, "line_width"]) == {2.4}
        assert set(drawn.data.groupby("identity")["line_dash_category"].first()) == {
            "Six-hour rhythm", "Twenty-four-hour rhythm"
        }
        assert len(drawn.extra["collection"].get_linewidths()) == len(segments)
        assert set(drawn.data["mark_type"]) == {"segment", "start"}
        assert {"x", "y", "x_end", "y_end"} <= set(drawn.data)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_trajectory_encoding_count_must_match_the_paths():
    figure, ax = _axes()
    try:
        groups = [([0, 1], [(0, 0), (1, 0)], 1.0)]
        with pytest.raises(ValueError, match="line_widths has 2 values for 1"):
            motility.trajectory_map(
                ax, groups, load_theme(), field={"width": 2, "height": 2},
                line_widths=[1.0, 2.0],
            )
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_the_step_histogram_marks_only_what_it_measured():
    """Two lines, both read off the distribution; no tracker threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        values = np.logspace(-2, 1, 400)
        drawn = motility.step_histogram(ax, values, load_theme(), bins=20)
        assert drawn.data["count"].sum() == len(values)
        assert len([line for line in ax.lines]) == 2
    finally:
        plt.close(figure)


# -------------------------------------------------- presence and surveillance

def test_every_presence_state_has_a_colour_and_a_label():
    theme = load_theme()
    assert set(presence.STATE_CODES) == set(presence.STATE_ROLES)
    assert set(presence.STATE_CODES) == set(presence.STATE_LABELS)
    for role in presence.STATE_ROLES.values():
        assert theme.colour(role).startswith("#")
    assert theme.colour(presence.STATE_ROLES["named"]) == theme.colour("ink")
    assert presence.STATE_LABELS["named"] == "present on screen"
    assert presence.STATE_LABELS["unclaimed"] == "temporary gap"


def test_lifespan_bars_order_cells_and_cut_unnamed_gaps():
    """A missing name is drawn as a cut in its cell's own lifespan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        spans = [
            {"identity": 2, "first_hour": 2.0, "last_hour": 4.0},
            {"identity": 1, "first_hour": 0.0, "last_hour": 3.0},
        ]
        gaps = [
            {"identity": 1, "hours": 0.0, "named": True},
            {"identity": 1, "hours": 1.0, "named": False},
            {"identity": 1, "hours": 2.0, "named": True},
        ]
        drawn = presence.lifespan_bars(ax, spans, load_theme(), gaps=gaps).data
        assert drawn["identity"].tolist() == [1, 2]
        assert {"row_order", "first_hour", "last_hour", "marked"} <= set(drawn)
        assert len(ax.patches) == 3  # two lifespans and the unnamed cut
    finally:
        plt.close(figure)


def test_persistence_raster_always_draws_exactly_three_states():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        matrix = np.array([[0, 1, 2], [1, 1, 1]], dtype=float)
        presence.persistence_raster(ax, matrix, [0.0, 0.5, 1.0], load_theme())
        assert len(ax.get_legend().get_texts()) == 3
    finally:
        plt.close(figure)


def test_persistence_raster_refuses_a_fourth_state_code():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        matrix = np.array([[0, 1, 2], [3, 1, 1]], dtype=float)
        with pytest.raises(ValueError, match="unknown persistence state codes"):
            presence.persistence_raster(
                ax, matrix, [0.0, 0.5, 1.0], load_theme())
    finally:
        plt.close(figure)


def test_the_three_state_codes_never_move():
    assert presence.STATE_CODES["outside_lifespan"] == 0
    assert presence.STATE_CODES["named"] == 1
    assert presence.STATE_CODES["unclaimed"] == 2


def test_persistence_preparation_and_events_share_the_same_row_order():
    rows = pd.DataFrame([
        {"identity": 2, "frame_index": 0, "hours": 0.0, "state": "outside_lifespan",
         "first_frame_index": 1, "last_frame_index": 2},
        {"identity": 2, "frame_index": 1, "hours": 1.0, "state": "named",
         "first_frame_index": 1, "last_frame_index": 2},
        {"identity": 2, "frame_index": 2, "hours": 2.0, "state": "named",
         "first_frame_index": 1, "last_frame_index": 2},
        {"identity": 1, "frame_index": 0, "hours": 0.0, "state": "named",
         "first_frame_index": 0, "last_frame_index": 2},
        {"identity": 1, "frame_index": 1, "hours": 1.0, "state": "unclaimed",
         "first_frame_index": 0, "last_frame_index": 2},
        {"identity": 1, "frame_index": 2, "hours": 2.0, "state": "named",
         "first_frame_index": 0, "last_frame_index": 2},
    ])
    plotted, matrix, hours, identities = presence.persistence_values(rows)
    events = presence.persistence_events(
        plotted,
        pd.DataFrame([{"identity": 1, "frame_index": 1,
                       "mechanism": "same_host_merge_hiding",
                       "still_missing_in_accepted_labels": True}]),
        pd.DataFrame([{"identity": 2, "silent_nonborder_ending": True}]),
    )
    assert identities == [1, 2]
    assert matrix.tolist() == [[1.0, 2.0, 1.0], [0.0, 1.0, 1.0]]
    assert hours.tolist() == [0.0, 1.0, 2.0]
    assert events["row_position"].tolist() == [0, 1]
    assert events["event_class"].tolist() == ["temporary_absence", "silent_ending"]


def test_claimed_foreground_is_drawn_with_unclaimed_foreground_on_one_axis():
    figure, ax = _axes()
    try:
        drawn = presence.unclaimed_area(
            ax, [0.0, 1.0], [2.0, 3.0], load_theme(), claimed=[8.0, 7.0])
        assert drawn.data["foreground_px"].tolist() == [10.0, 10.0]
        assert len(ax.collections) == 2
        assert len(ax.lines) == 2
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_displacement_curves_export_the_population_summary_they_draw():
    figure, ax = _axes()
    curves = pd.DataFrame({
        "identity": [1, 1, 1, 2, 2, 2],
        "lag_hours": [0.5, 1.0, 2.0, 0.5, 1.0, 2.0],
        "msd": [1.0, 2.0, 4.0, 3.0, 6.0, 12.0],
        "pairs": [10, 9, 8, 10, 9, 8],
    })
    try:
        drawn = motility.msd_curves(ax, curves, load_theme())
        medians = drawn.data.groupby(
            "lag_hours"
        )["population_median_msd"].first().tolist()
        assert medians == [2.0, 4.0, 8.0]
        assert drawn.extra["summary"]["cells_at_lag"].tolist() == [2, 2, 2]
        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert all("10^" not in label for label in labels)
        assert labels == ["0.5", "1", "2"]
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_displacement_exponent_panel_marks_its_source_interpretation():
    figure, ax = _axes()
    try:
        drawn = motility.msd_exponent_distribution(
            ax, [0.25, 0.5, 0.75, 1.25], load_theme(), bins=4,
        )
        assert drawn.data["cells_total"].iloc[0] == 4
        assert drawn.data["median_alpha"].iloc[0] == pytest.approx(0.625)
        assert drawn.data["random_walk_reference"].iloc[0] == 1.0
        assert sorted(line.get_xdata()[0] for line in ax.lines) == [0.625, 1.0]
        assert "fitted from the upper curve" in ax.get_xlabel()
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_one_mechanism_keeps_one_colour_across_panels():
    """The strip and the lifespan cuts must agree, or a colour means two things."""
    theme = load_theme()
    names = ["same_host_merge_hiding", "unresolved", "local_raw_supported_dropout"]
    first = presence.mechanism_colours(names, theme)
    second = presence.mechanism_colours(list(reversed(names)) + [None], theme)
    assert first == second
    assert len(set(first.values())) == len(names)


def test_lifespan_bars_colour_a_cut_by_its_mechanism_and_mark_a_silent_end():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = load_theme()
    figure, ax = plt.subplots()
    try:
        spans = [{"identity": 1, "first_hour": 0.0, "last_hour": 3.0}]
        gaps = [
            {"identity": 1, "hours": 0.0, "named": True, "mechanism": None},
            {"identity": 1, "hours": 1.0, "named": False,
             "mechanism": "same_host_merge_hiding"},
            {"identity": 1, "hours": 2.0, "named": True, "mechanism": None},
        ]
        drawn = presence.lifespan_bars(ax, spans, theme, gaps=gaps, mark={1}).data
        expected = presence.mechanism_colours(["same_host_merge_hiding"], theme)
        cut = ax.patches[-1]
        assert cut.get_facecolor()[:3] != tuple(
            int(theme.colour("unclaimed").lstrip("#")[i:i + 2], 16) / 255
            for i in (0, 2, 4)
        )
        assert list(expected) == ["same_host_merge_hiding"]
        assert bool(drawn["marked"].iloc[0])
        assert len(ax.lines) == 1  # the cross on the silent ending
    finally:
        plt.close(figure)


def test_an_unexplained_gap_is_not_given_a_mechanism_colour():
    """A gap the tracker could not classify keeps the plain unclaimed colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = load_theme()
    figure, ax = plt.subplots()
    try:
        spans = [{"identity": 1, "first_hour": 0.0, "last_hour": 2.0}]
        gaps = [
            {"identity": 1, "hours": 0.0, "named": True, "mechanism": None},
            {"identity": 1, "hours": 1.0, "named": False, "mechanism": None},
        ]
        presence.lifespan_bars(ax, spans, theme, gaps=gaps)
        assert presence.mechanism_colours([None, float("nan")], theme) == {}
        assert len(ax.patches) == 2
    finally:
        plt.close(figure)


def test_semantic_legend_deduplicates_repeated_series_names():
    theme = load_theme()
    figure, ax = _axes()
    try:
        ax.plot([0, 1], [0, 1], label="Observed cells")
        ax.plot([0, 1], [1, 0], label="Observed cells")
        ax.plot([0, 1], [0.5, 0.5], label="Matched noise")
        legend = common.semantic_legend(ax, theme, location="inside")
        assert [text.get_text() for text in legend.get_texts()] == ["Observed cells", "Matched noise"]
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_every_evidence_channel_has_a_colour_the_theme_can_resolve():
    theme = load_theme()
    for role, label in surveillance.EVIDENCE_CHANNELS.values():
        assert theme.colour(role).startswith("#")
        assert label


def test_mirrored_ledger_uses_shared_value_and_starting_time_ticks():
    figure, ax = _axes()
    try:
        frame = pd.DataFrame({
            "gained_px": [20.0, 30.0],
            "lost_px": [10.0, 25.0],
            "held_px": [80.0, 90.0],
            "area_change_px": [10.0, 5.0],
        })
        surveillance.mirrored_ledger(
            ax, [0.5, 1.0], frame, load_theme(), hour_ticks=24,
        )
        assert list(ax.get_yticks()) == [-90.0, -45.0, 0.0, 45.0, 90.0]
        assert 0.0 in ax.get_xticks()
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


# ------------------------------------------------------- the shared grammar

def test_a_histogram_returns_the_table_it_drew():
    """The plotted CSV must not be rebuilt from a second set of bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        drawn = common.histogram(ax, np.arange(100.0), load_theme(), bins=10)
        assert drawn.data["count"].sum() == 100
        assert len(drawn.data) == 10
        # One row per band, left and right edge named: a CSV of eleven edges and
        # ten counts is two columns that have to be lined up by hand.
        assert list(drawn.data) == ["bin_left", "bin_right", "count"]
    finally:
        plt.close(figure)


def test_dropping_the_margins_returns_no_margin_axes():
    """``--panels scatter`` must not need a second layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(6, 6))
    try:
        ax, top, right = common.scatter_with_margins(
            figure, (0.1, 0.1, 0.8, 0.8), [1.0, 2.0], [3.0, 4.0], load_theme(),
            margins=False,
        ).axes
        assert top is None and right is None
        assert ax is not None
    finally:
        plt.close(figure)


def test_hexbin_bins_logarithmic_x_and_returns_its_binned_mean():
    """A generic density panel must bin on the scale it displays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        drawn = common.hexbin(
            ax, [0.1, 0.2, 1.0, 10.0], [1.0, 2.0, 3.0, 5.0], load_theme(),
            gridsize=3, xscale="log", mean_label="Mean response",
        )
        assert ax.get_xscale() == "log"
        assert list(drawn.data) == ["bin_left", "bin_right", "x", "mean", "count"]
        assert drawn.data["count"].sum() == 4
        assert ax.lines[0].get_label() == "Mean response"
    finally:
        plt.close(figure)


def test_centroid_path_area_panel_preserves_a_zero_path_area():
    """A stationary cell must not disappear merely because log axes cannot show zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        drawn = motility.centroid_path_area_against_footprint(
            ax, [0.0, 20.0], [10.0, 10.0], load_theme(), scale="symlog",
        )
        assert len(drawn.data) == 2
        assert drawn.data["centroid_path_area"].iloc[0] == 0.0
        assert ax.get_xscale() == "symlog"
        assert ax.get_yscale() == "symlog"
    finally:
        plt.close(figure)


# ------------------------------------------- the panels the new pages needed

def test_a_wrapped_active_period_is_drawn_as_two_bars_that_add_up():
    """The common case, not an edge case.

    Most fits in the reference dataset have an offset earlier than their onset,
    because the quiet phase straddles the fold of the day. Drawn from the
    larger number to the smaller one the bar has negative width and vanishes;
    dropped, the cell disappears from a page that claims to show every one.
    """
    _, ax = _axes()
    drawn = rhythms.active_spans(ax, [23.0], [17.0], load_theme(),
                                 groups=["area_px"], period=24.0).data

    assert len(drawn) == 2
    assert bool(drawn["wrapped"].all())
    # 23 to 24, then 0 to 17: the eighteen hours the module recorded.
    assert drawn["hours"].sum() == pytest.approx(18.0)
    assert sorted(drawn["start_hour"]) == [0.0, 23.0]


def test_an_unwrapped_active_period_is_one_bar():
    _, ax = _axes()
    drawn = rhythms.active_spans(ax, [2.0], [10.0], load_theme(),
                                 groups=["area_px"]).data
    assert len(drawn) == 1
    assert not bool(drawn["wrapped"].any())
    assert drawn["hours"].iloc[0] == pytest.approx(8.0)


def test_an_active_period_of_no_length_is_recorded_rather_than_dropped():
    """Nothing visible is drawn, and the cell still appears in the table.

    A row that leaves no mark and no trace is indistinguishable from a cell
    that was never measured, which is the one reading this panel must not
    allow.
    """
    _, ax = _axes()
    drawn = rhythms.active_spans(ax, [5.0], [5.0], load_theme(),
                                 groups=["step_px"]).data
    assert len(drawn) == 1
    assert drawn["hours"].iloc[0] == 0.0


def test_active_spans_keeps_its_blocks_in_the_order_it_was_given():
    _, ax = _axes()
    drawn = rhythms.active_spans(
        ax, [1.0, 2.0, 3.0], [5.0, 6.0, 7.0], load_theme(),
        groups=["step_px", "area_px", "step_px"]).data
    assert list(dict.fromkeys(drawn["group"])) == ["step_px", "area_px"]


def test_ranked_events_uses_the_chosen_metric_direction_and_count():
    """Top N means top N values within each cell, not one hard-coded quantity."""
    frame = pd.DataFrame({
        "identity": [1, 1, 1, 2, 2, 2],
        "frame_index": [0, 1, 2, 0, 1, 2],
        "hours": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        "chosen_metric": [1.0, 9.0, 5.0, 8.0, 2.0, 6.0],
    })
    aligned = surveillance.ranked_events(
        frame, column="chosen_metric", direction="high", top_n=2, window=0
    )
    events = aligned[["identity", "event_rank", "event_value"]].drop_duplicates()
    assert events["event_rank"].tolist() == [1, 2, 1, 2]
    assert events["event_value"].tolist() == [9.0, 5.0, 8.0, 6.0]


def test_ranked_event_deviation_records_its_robust_scale():
    """A deviation event says both its source value and its IQR-normalised score."""
    frame = pd.DataFrame({
        "identity": [1, 1, 1, 1],
        "frame_index": [0, 1, 2, 3],
        "hours": [0.0, 1.0, 2.0, 3.0],
        "signal": [0.0, 1.0, 2.0, 100.0],
    })
    event = surveillance.ranked_events(
        frame, column="signal", direction="deviation", top_n=1, window=0
    ).iloc[0]
    assert event["event_value"] == 100.0
    assert event["event_scale"] > 0
    assert event["event_score"] > 1


def test_ranked_rate_divides_by_each_cells_measurable_observations():
    """Raw event counts cannot outrank a higher event rate from a shorter record."""
    frame = pd.DataFrame({
        "identity": [1, 2],
        "events": [5, 2],
        "measurable": [100, 4],
    })
    figure, ax = _axes()
    try:
        drawn = common.ranked_rate(
            ax, frame, load_theme(), numerator_column="events",
            denominator_column="measurable",
        ).data
        assert drawn["identity"].tolist() == [2, 1]
        assert drawn["rate_percent"].tolist() == [50.0, 5.0]
        assert ax.get_ylim()[0] == 0.0
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_low_overlap_event_raster_shows_which_time_spans_were_measurable():
    frame = pd.DataFrame({
        "identity": [1, 1, 2, 2],
        "hours": [1.0, 2.0, 1.0, 3.0],
        "jaccard": [0.1, 0.8, 0.4, 0.9],
    })
    figure, ax = _axes()
    try:
        drawn = surveillance.low_overlap_event_raster(
            ax, frame, load_theme(), threshold=0.2
        ).data
        assert drawn["identity"].tolist() == [1]
        assert len(ax.lines) == 2
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_a_refused_comparison_carries_a_sentence_and_no_point():
    """The whole reason the forest exists rather than a dotplot.

    A refusal drawn as a blank row reads as a comparison nobody asked for, and
    the contrasts step writes a refusal row precisely because those two are
    different.
    """
    _, ax = _axes()
    drawn = common.forest(ax, [float("nan")], load_theme(), labels=["refused"],
                          refusals=["only 1 subject a side"]).data
    assert not bool(drawn["drawn"].any())
    assert drawn["refusal"].iloc[0] == "only 1 subject a side"
    assert not ax.collections          # no point was placed


def test_an_estimate_without_an_interval_gets_a_point_and_no_whisker():
    """A rank test without a bootstrap has no interval, which is not a zero one."""
    _, ax = _axes()
    theme = load_theme()
    common.forest(ax, [1.0], theme, labels=["a"], lo=[float("nan")], hi=[float("nan")])
    with_interval_lines = len(ax.lines)
    _, other = _axes()
    common.forest(other, [1.0], theme, labels=["a"], lo=[0.5], hi=[1.5])
    # One line either way for the no-effect rule; the interval adds a second.
    assert len(other.lines) == with_interval_lines + 1


def test_a_forest_row_takes_its_significance_rather_than_deriving_one():
    _, ax = _axes()
    drawn = common.forest(ax, [1.0, 2.0], load_theme(), labels=["a", "b"],
                          significant=[True, False]).data
    assert list(drawn["significant"]) == [True, False]


def test_two_scales_never_share_one_horizontal_axis():
    """Six pixels and thirteen hundred camera units are not one bigger.

    On a shared axis the smaller effect collapses onto the no-effect line and
    reads as no effect, which is the opposite of what it says.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 6))
    drawn = common.forest_blocks(
        figure, (0.15, 0.1, 0.7, 0.8), [6.75, 1300.0], load_theme(),
        labels=["area", "reporter"], scales=["px", "camera units"])
    assert len(drawn.axes) == 2
    assert list(drawn.data["scale"]) == ["px", "camera units"]
    assert drawn.axes[0].get_xlim() != drawn.axes[1].get_xlim()


def test_a_refusal_is_blocked_with_the_rows_it_belongs_to():
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 6))
    drawn = common.forest_blocks(
        figure, (0.15, 0.1, 0.7, 0.8), [6.75, float("nan"), 1300.0], load_theme(),
        labels=["area", "refused", "reporter"],
        scales=["px", "px", "camera units"],
        refusals=["", "too few units", ""])
    assert len(drawn.axes) == 2
    assert list(drawn.data[drawn.data["scale"] == "px"]["label"]) == ["area", "refused"]


def test_a_negative_lollipop_annotation_sits_clear_of_its_own_stem():
    """Right of the dot on a negative row is on top of the stem it labels."""
    _, ax = _axes()
    common.lollipop(ax, [-0.2, 0.3], load_theme(), labels=["down", "up"],
                    annotations=["falls", "rises"])
    placed = {t.get_text(): (t.get_position()[0], t.get_ha()) for t in ax.texts}
    assert placed["falls"][0] < -0.2 and placed["falls"][1] == "right"
    assert placed["rises"][0] > 0.3 and placed["rises"][1] == "left"


def test_a_ridgeline_given_limits_counts_what_falls_outside_them():
    """One heavy tail must not set the range for every other row."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 6))
    drawn = common.ridgeline(
        figure, (0.15, 0.1, 0.7, 0.8),
        [np.array([0.0, 0.1, 0.2, 50.0]), np.array([0.0, 0.1])],
        load_theme(), labels=["with a tail", "without"], limits=(0.0, 1.0))
    outside = drawn.data.groupby("group")["outside"].first()
    assert outside["with a tail"] == 1
    assert outside["without"] == 0


def test_series_colours_stay_distinct_past_the_role_cycle():
    theme = load_theme()
    many = common.series_colours(theme, len(theme.cycle()) + 3)
    assert len(set(many)) == len(many)
    assert common.series_colours(theme, 3) == theme.cycle(3)


def test_auto_organotypic_presentation_filter_is_display_only_and_audited():
    """The grain-removal path cannot masquerade as a measurement transform."""
    rng = np.random.default_rng(4)
    time = np.arange(16, dtype=float)[:, None, None]
    stack = 100.0 + 8.0 * np.sin(2 * np.pi * time / 8.0)
    stack = stack + rng.normal(0, 2, size=(16, 7, 7))
    shown, settings = intensity.presentation_stack(stack, time_gain=1.0)
    assert shown.shape == stack.shape
    assert np.isfinite(shown).all()
    assert shown.min() >= 0 and shown.max() <= 1
    assert settings["display_only"] is True
    assert settings["image_filter"].startswith("auto-organotypic")
    assert settings["filter_method_version"] != "not-applied"


def test_presentation_range_can_be_chosen_from_only_displayed_frames():
    stack = np.asarray([
        [[0.0, 1.0], [2.0, 3.0]],
        [[0.0, 1.0], [2.0, 4.0]],
        [[0.0, 1000.0], [1000.0, 1000.0]],
    ])
    _, all_settings = intensity.presentation_stack(
        stack, image_filter="none", black_percentile=0,
        white_percentile=99, gamma=1, time_gain=1)
    shown, displayed_settings = intensity.presentation_stack(
        stack, image_filter="none", black_percentile=0,
        white_percentile=99, gamma=1, time_gain=1, range_frames=[0, 1])
    assert displayed_settings["display_range_scope"] == "displayed frames"
    assert displayed_settings["display_range_frame_count"] == 2
    assert displayed_settings["black_value"] == 0
    assert displayed_settings["white_value"] < all_settings["white_value"]
    assert shown[0, 0, 1] > 0


def test_the_bioluminescence_lookup_table_is_the_movie_lookup_table():
    look = resolve_look(load_theme(), "dluc_purple")
    assert look.is_map
    assert look.cmap.name == "dluc_purple"


def test_imagej_red_lookup_table_remains_a_black_to_red_intensity_map():
    look = resolve_look(load_theme(), "red")
    assert look.is_map
    assert look.cmap.name == "red"
    assert look.cmap(0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert look.cmap(1.0) == pytest.approx((1.0, 0.0, 0.0, 1.0))


def test_overlaying_unlike_metrics_standardises_each_trace_first():
    _, ax = _axes()
    drawn = common.trace_overlay(
        ax, [0, 1, 2, 3],
        {"camera_units": [10, 20, 30, 40], "pixels": [1, 4, 1, 4]},
        load_theme(),
    )
    for _, group in drawn.data.groupby("series"):
        assert float(group["value"].mean()) == pytest.approx(0.0)
        assert float(group["value"].std(ddof=0)) == pytest.approx(1.0)
    assert ax.get_legend() is not None


def test_an_explicit_period_can_drive_a_workbench_descriptive_cosinor():
    _, ax = _axes()
    hours = np.arange(20, dtype=float)
    values = np.cos(2 * np.pi * hours / 6.0)
    drawn = common.harmonic_curve(
        ax, hours, values, load_theme(), period_hours=6.0)
    assert np.isfinite(drawn.data["fitted"]).all()
    assert np.corrcoef(values, drawn.data["fitted"])[0, 1] > 0.99


def test_a_raw_harmonic_curve_uses_the_selected_workbench_baseline():
    _, ax = _axes()
    hours = np.arange(96, dtype=float)
    values = 30 + 0.03 * hours ** 2 + 2 * np.cos(2 * np.pi * hours / 12.0)
    drawn = common.cosinor_curve(
        ax, hours, values, load_theme(), period_hours=12.0,
        detrend="polynomial", detrend_window_hours=18.0,
    )

    assert set(drawn.data["detrend"]) == {"polynomial"}
    assert set(drawn.data["detrend_window_hours"]) == {18.0}
    assert np.corrcoef(values, drawn.data["fitted"])[0, 1] > 0.99


def test_generic_phase_map_wraps_any_precomputed_phase_in_its_declared_cycle():
    _, ax = _axes()
    drawn = common.phase_map(
        ax, [[1, 2], [3, 4]], [25, -1], load_theme(),
        field={"width": 5, "height": 6}, period=24, phase_unit="h",
        labels=["cell 1", "cell 2"],
    )
    assert drawn.data["unit"].tolist() == ["cell 1", "cell 2"]
    assert drawn.data["phase"].tolist() == [1.0, 23.0]
    assert drawn.data["period"].unique().tolist() == [24.0]
    assert drawn.data["phase_unit"].unique().tolist() == ["h"]
    assert drawn.extra["handle"].get_clim() == (0.0, 24.0)


def test_period_status_matrix_distinguishes_results_from_untestable_traces():
    import matplotlib.pyplot as plt

    figure, ax = _axes()
    try:
        drawn = rhythms.period_status_matrix(
            ax,
            [[6.0, np.nan, np.nan], [30.0, np.nan, np.nan]],
            [["rhythmic", "not rhythmic", "not tested"],
             ["rhythmic", "not rhythmic", "not tested"]],
            load_theme(),
            row_labels=["1", "2"], column_labels=["inner", "middle", "outer"],
            underdetermined=[[False, False, False], [True, False, False]],
            period_min=2.0, period_max=48.0,
        )
        assert set(drawn.data["rhythm_status"]) == {
            "rhythmic", "not rhythmic", "not tested"
        }
        assert len(ax.patches) == 1
        assert drawn.extra["handle"].get_clim() == (2.0, 48.0)
        assert [handle.get_label() for handle in drawn.extra["legend_handles"]] == [
            "Tested; not rhythmic", "Not tested", "Fewer than required cycles"
        ]
    finally:
        plt.close(figure)


def test_colour_map_key_keeps_its_physical_shape_beside_a_small_axes():
    import matplotlib.pyplot as plt

    theme = load_theme()
    figure, ax = plt.subplots(figsize=theme.canvas(4.0, 4.0))
    handle = ax.imshow([[0.0, 1.0], [1.0, 0.0]])
    bar = common.inset_colour_bar(ax, handle, theme, label="Value")
    width = bar.ax.get_position().width * figure.get_figwidth() / theme.canvas_scale
    height = bar.ax.get_position().height * figure.get_figheight() / theme.canvas_scale
    assert width == pytest.approx(theme["colour_bar_width_inches"])
    assert height == pytest.approx(theme["colour_bar_height_inches"])
    assert bar.ax.yaxis.label.get_fontsize() == theme.size("annotation")
    plt.close(figure)
