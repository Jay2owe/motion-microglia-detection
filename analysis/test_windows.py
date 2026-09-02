"""Windows are a filter, and these tests are about the edges of the filter.

The arithmetic is the roll-up that already existed. What can go wrong is which
rows go into it: a frame counted in two windows at once, a window that quietly
catches nothing, or a paired difference computed against a baseline that already
contains the response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.config import WindowConfig, _windows
from analysis.registry import MeasurementContext
from analysis.summarise import build_cell_summary
from analysis.units import resolve_scale
from analysis.windows import frame_mask, window_extent, windowed_summaries

N_FRAMES = 20
MINUTES = 30.0


def _context(n_frames: int = N_FRAMES, offset: int = 0) -> MeasurementContext:
    """A context with the smallest label stack that still counts pixels.

    ``build_frame_summary`` reads the label images to count what was assigned in
    each frame, so a stack of zeros is not enough - one cell has to be visible
    or every frame reports nothing and the test proves less than it looks.
    """
    labels = np.zeros((n_frames, 4, 4), dtype=np.uint16)
    labels[:, 1, 1] = 1
    labels[:, 2, 2] = 2
    return MeasurementContext(
        stem="m",
        labels=labels,
        raw=np.full_like(labels, 500),
        scale=resolve_scale(MINUTES, None, []),
        identities=[1, 2],
        source_frame_offset=offset,
    )


def _cell_frame(context: MeasurementContext, identities=(1, 2)) -> pd.DataFrame:
    """One row per cell per frame, with a metric that steps up halfway through.

    ``area_px`` is 100 for the first half of the recording and 200 for the
    second, so a before/after window pair has an answer that can be written down
    rather than merely compared against itself.
    """
    frames = context.frame_table()
    rows = []
    for identity in identities:
        block = frames.copy()
        block.insert(0, "identity", identity)
        block["area_px"] = np.where(block["frame_index"] < N_FRAMES // 2, 100.0, 200.0)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def _window(name: str, **kwargs) -> WindowConfig:
    return WindowConfig.from_dict({"name": name, **kwargs})


# ----------------------------------------------------------------- the filter


def test_a_frame_on_a_shared_edge_lands_in_one_window_only() -> None:
    """Half-open, and this is the bug that would otherwise be found in six months.

    If `baseline` is 0-10 h and `treatment` is 10-20 h and both include hour 10,
    every paired difference is computed against a baseline holding the first
    frame of the response - a treatment effect that is slightly too small, which
    nothing else in the package would ever reveal.
    """
    context = _context()
    frames = context.frame_table()
    before = frame_mask(frames, _window("baseline", from_hours=0.0, to_hours=5.0))
    after = frame_mask(frames, _window("treatment", from_hours=5.0, to_hours=10.0))

    assert not (before & after).any()
    on_the_edge = frames["hours"] == 5.0
    assert on_the_edge.any(), "the fixture must actually put a frame on the edge"
    assert not before[on_the_edge.to_numpy()].any()
    assert after[on_the_edge.to_numpy()].all()


def test_two_overlapping_windows_may_both_claim_a_frame() -> None:
    """Overlap is allowed on purpose; only a *shared edge* is the trap."""
    context = _context()
    frames = context.frame_table()
    wide = frame_mask(frames, _window("whole", from_hours=0.0, to_hours=10.0))
    narrow = frame_mask(frames, _window("middle", from_hours=2.0, to_hours=6.0))

    assert (wide & narrow).sum() == narrow.sum() > 0


def test_the_same_window_in_frames_and_in_hours_selects_the_same_rows() -> None:
    context = _context(offset=2)
    frames = context.frame_table()
    # Offset 2 at half-hour frames puts frame 0 at hour 1.0, so these two are
    # the same window written two ways.
    by_frame = frame_mask(frames, _window("w", from_frame=0, to_frame=8))
    by_hour = frame_mask(frames, _window("w", from_hours=1.0, to_hours=5.0))

    assert by_frame.tolist() == by_hour.tolist()
    assert by_frame.sum() == 8


def test_a_window_past_the_end_reports_what_it_caught() -> None:
    context = _context()
    held, length = window_extent(_window("w", from_hours=0.0, to_hours=100.0), context)

    assert held == N_FRAMES            # what the recording has
    assert length == 100.0             # what was asked for


# ------------------------------------------------------------- the roll-up


def test_a_window_covering_everything_reproduces_the_whole_recording_summary() -> None:
    """The check that a windowed roll-up is the same roll-up.

    Not "close to": identical. If this ever drifts, windowing has started
    computing something of its own.
    """
    context = _context()
    cell_frame = _cell_frame(context)
    whole = build_cell_summary(cell_frame, {}, context)
    produced = windowed_summaries(
        cell_frame, context, [_window("all", from_frame=0, to_frame=N_FRAMES)])
    windowed = produced["cell_summary_windowed"]

    shared = [c for c in whole.columns if c in windowed.columns]
    assert "area_px_median" in shared and "observed_frames" in shared
    pd.testing.assert_frame_equal(
        whole[shared].reset_index(drop=True),
        windowed[shared].reset_index(drop=True),
        check_exact=True,
    )


def test_a_new_window_adds_rows_and_not_columns() -> None:
    context = _context()
    cell_frame = _cell_frame(context)
    one = windowed_summaries(cell_frame, context,
                             [_window("a", from_frame=0, to_frame=10)])
    two = windowed_summaries(
        cell_frame, context,
        [_window("a", from_frame=0, to_frame=10),
         _window("b", from_frame=10, to_frame=20)])

    assert len(two["cell_summary_windowed"]) == 2 * len(one["cell_summary_windowed"])
    assert list(two["cell_summary_windowed"].columns) == \
        list(one["cell_summary_windowed"].columns)
    assert set(two["cell_summary_windowed"]["window"]) == {"a", "b"}


def test_the_window_honesty_columns_are_written_on_every_row() -> None:
    """A cell present for two frames of a twenty-frame window still gets a summary.

    That is correct behaviour and a trap, so the count it rests on travels in
    the same row.
    """
    context = _context()
    cell_frame = _cell_frame(context)
    # Cell 2 is only ever seen in the first two frames of the second window.
    cell_frame = cell_frame[~((cell_frame["identity"] == 2)
                              & (cell_frame["frame_index"] >= 12))]

    produced = windowed_summaries(
        cell_frame, context, [_window("late", from_frame=10, to_frame=20)])
    late = produced["cell_summary_windowed"].set_index("identity")

    assert late.loc[1, "window_frames"] == 10
    assert late.loc[2, "window_frames"] == 10
    assert late.loc[1, "observed_frames"] == 10
    assert late.loc[2, "observed_frames"] == 2
    assert late.loc[2, "window_coverage"] == pytest.approx(0.2)
    assert late.loc[1, "window_hours"] == pytest.approx(5.0)


def test_a_window_containing_no_frames_contributes_no_rows_and_is_caught_earlier() -> None:
    """An empty window is a mistyped window, and it is refused before the run.

    The stage this came from asked for the empty window to be *written*, as a
    row with its counts at zero. That would mean inventing a row for a cell in a
    stretch of the recording where it was never seen, which is the one thing a
    summary table must not do - the row would carry an identity, a window and a
    column of blanks, and blanks in a summary read as "measured, and there was
    nothing there".

    So it contributes nothing, and the record lives in two places that cannot be
    mistaken for a measurement: ``window_extent`` reports zero frames, which the
    run manifest carries per window, and ``python -m analysis doctor`` counts a
    window that catches no frames as a problem *before* anything is measured.
    """
    context = _context()
    cell_frame = _cell_frame(context)
    empty = _window("empty", from_hours=500.0, to_hours=600.0)
    produced = windowed_summaries(
        cell_frame, context, [_window("real", from_frame=0, to_frame=10), empty])

    assert set(produced["cell_summary_windowed"]["window"]) == {"real"}
    assert produced["frame_summary_windowed"].query("window == 'empty'").empty
    # What the manifest records instead, and what `doctor` refuses on.
    assert window_extent(empty, context)[0] == 0


def test_declaring_no_windows_produces_nothing() -> None:
    """Declaring nothing must cost nothing."""
    context = _context()
    assert windowed_summaries(_cell_frame(context), context, []) == {}


# ----------------------------------------------------------- paired changes


def test_the_paired_change_is_the_number_the_stage_exists_for() -> None:
    context = _context()
    cell_frame = _cell_frame(context)          # 100 before, 200 after
    produced = windowed_summaries(
        cell_frame, context,
        [_window("baseline", from_frame=0, to_frame=10),
         _window("treatment", from_frame=10, to_frame=20, baseline="baseline")])

    change = produced["window_change"]
    row = change[(change["identity"] == 1) & (change["metric"] == "area_px")
                 & (change["statistic"] == "median")].iloc[0]
    assert row["window"] == "treatment"
    assert row["baseline_window"] == "baseline"
    assert row["value"] == 200.0
    assert row["baseline_value"] == 100.0
    assert row["change"] == 100.0
    assert row["ratio"] == pytest.approx(2.0)
    # Long, not wide: a metric is a row, so `cell_summary_windowed` gained no
    # paired columns at all.
    assert not [c for c in produced["cell_summary_windowed"].columns
                if c.endswith("_change") or c.endswith("_ratio")]


def test_a_window_with_no_baseline_produces_no_change_rows() -> None:
    context = _context()
    produced = windowed_summaries(
        _cell_frame(context), context,
        [_window("a", from_frame=0, to_frame=10),
         _window("b", from_frame=10, to_frame=20)])

    assert produced["window_change"].empty


def test_a_cell_missing_from_one_window_gets_no_change_row(  ) -> None:
    """A blank in a table of differences reads as "no change", so there is no row.

    The cell is still in the windowed summary for the window it was seen in -
    what it does not get is a comparison it has only one half of.
    """
    context = _context()
    cell_frame = _cell_frame(context)
    cell_frame = cell_frame[~((cell_frame["identity"] == 2)
                              & (cell_frame["frame_index"] >= 10))]

    produced = windowed_summaries(
        cell_frame, context,
        [_window("baseline", from_frame=0, to_frame=10),
         _window("treatment", from_frame=10, to_frame=20, baseline="baseline")])

    assert set(produced["window_change"]["identity"]) == {1}
    late = produced["cell_summary_windowed"].query("window == 'treatment'")
    assert set(late["identity"]) == {1}


def test_a_ratio_against_a_zero_baseline_is_blank_not_infinite() -> None:
    context = _context()
    cell_frame = _cell_frame(context)
    cell_frame.loc[cell_frame["frame_index"] < 10, "area_px"] = 0.0

    produced = windowed_summaries(
        cell_frame, context,
        [_window("baseline", from_frame=0, to_frame=10),
         _window("treatment", from_frame=10, to_frame=20, baseline="baseline")])
    change = produced["window_change"]
    row = change[(change["metric"] == "area_px")
                 & (change["statistic"] == "median")].iloc[0]

    assert row["change"] == 200.0
    assert np.isnan(row["ratio"])


# ------------------------------------------------------------- the refusals


@pytest.mark.parametrize("block, message", [
    ({"name": "Baseline", "from_hours": 0, "to_hours": 1}, "lower-case identifier"),
    ({"name": "window 1", "from_hours": 0, "to_hours": 1}, "lower-case identifier"),
    ({"name": "", "from_hours": 0, "to_hours": 1}, "lower-case identifier"),
    ({"name": "a", "from_hours": 0, "to_hours": 1, "from_frame": 0, "to_frame": 2},
     "hours and in frames at once"),
    ({"name": "a"}, "neither hours nor frames"),
    ({"name": "a", "from_hours": 0}, "both from_hours and to_hours"),
    ({"name": "a", "from_frame": 0}, "both from_frame and to_frame"),
    ({"name": "a", "from_hours": 5, "to_hours": 5}, "not after its start"),
    ({"name": "a", "from_hours": 5, "to_hours": 1}, "not after its start"),
    ({"name": "a", "from_frame": 5, "to_frame": 5}, "not after its start"),
    ({"name": "a", "from_hours": 0, "to_hours": 1, "baseline": "a"}, "names itself"),
])
def test_every_refusal_names_the_window_and_says_what_would_have_worked(
        block, message) -> None:
    with pytest.raises(ValueError, match=message):
        WindowConfig.from_dict(block)


def test_two_windows_sharing_a_name_are_refused() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        _windows([{"name": "a", "from_hours": 0, "to_hours": 1},
                  {"name": "a", "from_hours": 1, "to_hours": 2}], "windows")


def test_a_baseline_naming_a_window_that_does_not_exist_is_refused() -> None:
    """Refused at parse rather than at use.

    At use it would be a column of blanks, which reads as "this cell had no
    baseline" rather than "you spelled the window's name wrong".
    """
    with pytest.raises(ValueError, match="no window is called that"):
        _windows([{"name": "a", "from_hours": 0, "to_hours": 1},
                  {"name": "b", "from_hours": 1, "to_hours": 2, "baseline": "basline"}],
                 "windows")


def test_windows_must_be_a_list() -> None:
    with pytest.raises(TypeError, match="must be a list"):
        _windows({"name": "a"}, "windows")


def test_no_windows_parses_to_nothing() -> None:
    assert _windows(None, "windows") == []
    assert _windows([], "windows") == []
