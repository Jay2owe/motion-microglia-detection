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
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

from analysis.theme import load_theme  # noqa: E402
from _metrics import METRICS, axis_label, describe, role_for, semantic_label  # noqa: E402
from panels import Look, image_strip, resolve_look, trace  # noqa: E402
from panels import common, motility, presence, rhythms, surveillance, territory  # noqa: E402


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
    assert axis_label("turnover_index", 30.0) == "Pixels replaced\nper 30 min"
    assert axis_label("turnover_index", 15.0) == "Pixels replaced\nper 15 min"


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
    """``turnover_index_median`` is still pixels replaced, not a new quantity."""
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
        assert len(ax.patches) == 3  # two lifespans and the unnamed cut
    finally:
        plt.close(figure)


def test_a_run_without_provenance_still_draws_three_states():
    """The default is the figure that existed before provenance did."""
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


def test_the_fourth_state_appears_only_when_it_is_asked_for():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        matrix = np.array([[0, 1, 2], [3, 1, 1]], dtype=float)
        presence.persistence_raster(ax, matrix, [0.0, 0.5, 1.0], load_theme(),
                                    states=presence.PROVENANCE_STATES)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert presence.STATE_LABELS["named_inferred"] in labels
        assert len(labels) == 4
    finally:
        plt.close(figure)


def test_the_first_three_state_codes_never_move():
    """A saved matrix must not change meaning when a state is added."""
    assert presence.STATE_CODES["outside_lifespan"] == 0
    assert presence.STATE_CODES["named"] == 1
    assert presence.STATE_CODES["unclaimed"] == 2


def test_a_state_list_that_is_not_a_contiguous_run_is_refused():
    """Codes index the colour map directly, so a hole would recolour a state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="contiguous"):
            presence.persistence_raster(
                ax, np.zeros((2, 3)), [0.0, 0.5, 1.0], load_theme(),
                states=("outside_lifespan", "named", "named_inferred"))
    finally:
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
