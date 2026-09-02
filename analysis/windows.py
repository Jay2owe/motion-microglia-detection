"""Before and after, for every metric, without remeasuring anything.

Every summary this package writes covers the whole recording. If something
changes partway through - a drug goes in, the temperature shifts, a stimulus
starts - the before and the after are averaged into one number and the change
partly cancels itself out.

Nothing needs remeasuring to fix that. The per-cell-per-frame table already
holds every frame separately, so a window is a filter on it and a windowed
summary is the existing roll-up run over that filter. This module makes that a
declared, recorded thing rather than something each reader does by hand.

Three files come out, and the split between them is the house rule about long
tables rather than wide ones:

* ``cell_summary_windowed`` - the per-cell roll-up, once per window. One row per
  cell per window, so a new window adds rows and never columns.
* ``frame_summary_windowed`` - the same for the per-frame roll-up.
* ``window_change`` - one row per cell per window per metric, carrying that
  cell's value against its own value in the named baseline window, as a
  difference and as a ratio. This is the number the whole module exists for, and
  as a wide table it would have been four hundred paired columns.

Hours here are hours since the recording began. This package has no clock, so a
window is a position in the movie and not a time of day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import WindowConfig
from analysis.registry import MeasurementContext, Output
from analysis.summarise import (SUMMARY_METRICS, _STATISTICS, build_cell_summary,
                                build_frame_summary)

#: The three tables this file builds, declared the way ``summarise.ROLLUPS``
#: declares the three it builds. They are not module tables - no module runs per
#: window - so they are not in ``registry.declared_tables``; the writer still
#: needs their grain for the manifest.
WINDOWED: tuple[Output, ...] = (
    Output("cell_summary_windowed", grain=("identity", "window")),
    Output("frame_summary_windowed", grain=("frame_index", "window")),
    Output("window_change", grain=("identity", "window", "metric")),
)


def frame_mask(table: pd.DataFrame, window: WindowConfig) -> np.ndarray:
    """Which rows of a table fall inside one window.

    Half-open, ``from <= x < to``, and that is the whole game. If ``baseline``
    is 0-20 h and ``treatment`` is 20-49 h and both claim hour 20, every paired
    difference is computed against a baseline holding the first frame of the
    response - a bug that shows up as a treatment effect that is slightly too
    small, which is the kind nobody finds.

    A window declared in frames is masked on ``frame_index`` and one declared in
    hours on ``hours``, rather than converting one into the other: a conversion
    would put a rounding step between what the reader wrote and what was
    selected.
    """
    if window.in_frames:
        values = table["frame_index"].to_numpy()
        return (values >= window.from_frame) & (values < window.to_frame)
    values = table["hours"].to_numpy()
    return (values >= window.from_hours) & (values < window.to_hours)


def window_extent(window: WindowConfig, context: MeasurementContext) -> tuple[int, float]:
    """How many of the recording's frames a window holds, and how long it is.

    ``window_frames`` counts the frames the recording actually has inside the
    window, so a window declared past the end of the movie reports what it got
    rather than what it asked for. ``window_hours`` is the declared length,
    which is what somebody who wrote ``from_hours: 20`` expects to read back.
    """
    frames = context.frame_table()
    held = int(frame_mask(frames, window).sum())
    if window.in_frames:
        length = (window.to_frame - window.from_frame) * context.scale.minutes_per_frame / 60.0
    else:
        length = float(window.to_hours - window.from_hours)
    return held, float(length)


def _metric_statistics(summary: pd.DataFrame) -> list[tuple[str, str, str]]:
    """The ``{metric}_{statistic}`` columns of a roll-up, and their two parts.

    Read off the roll-up rather than assumed, because which metrics a run has
    depends on which modules were switched on - and matched against
    ``SUMMARY_METRICS`` and ``_STATISTICS`` rather than by splitting on the last
    underscore, since several metric names end in a word that is also a
    statistic.
    """
    found = []
    for metric in SUMMARY_METRICS:
        for statistic in _STATISTICS:
            column = f"{metric}_{statistic}"
            if column in summary.columns:
                found.append((column, metric, statistic))
    return found


def _changes(cell_summary: pd.DataFrame, windows: list[WindowConfig]) -> pd.DataFrame:
    """Each window's values against its baseline window's, per cell per metric.

    Long rather than wide: one row per cell per window per metric statistic,
    carrying both the difference and the ratio.

    Both, because a change of 40 camera units means nothing without knowing
    whether the baseline was 50 or 5000, and a ratio of 1.8 means nothing
    without knowing whether the difference is 4 units or 4000.

    A cell absent from one of the two windows gets no row at all rather than a
    row of blanks: it has no value to compare, and a blank in a table of
    differences reads as "no change".
    """
    paired = [w for w in windows if w.baseline is not None]
    if not paired or cell_summary.empty:
        return pd.DataFrame(columns=["identity", "window", "baseline_window", "metric",
                                     "statistic", "value", "baseline_value",
                                     "change", "ratio"])
    columns = _metric_statistics(cell_summary)
    by_window = {name: block.set_index("identity")
                 for name, block in cell_summary.groupby("window", sort=False)}

    rows = []
    for window in paired:
        here = by_window.get(window.name)
        there = by_window.get(window.baseline)
        if here is None or there is None:
            continue
        shared = here.index.intersection(there.index)
        if not len(shared):
            continue
        for column, metric, statistic in columns:
            value = here.loc[shared, column].to_numpy(dtype=float)
            base = there.loc[shared, column].to_numpy(dtype=float)
            # A ratio against a baseline of zero is not large, it is undefined,
            # and writing an infinity here would put one on an axis.
            ratio = np.divide(value, base, out=np.full_like(value, np.nan),
                              where=base != 0)
            rows.append(pd.DataFrame({
                "identity": shared,
                "window": window.name,
                "baseline_window": window.baseline,
                "metric": metric,
                "statistic": statistic,
                "value": value,
                "baseline_value": base,
                "change": value - base,
                "ratio": ratio,
            }))
    if not rows:
        return pd.DataFrame(columns=["identity", "window", "baseline_window", "metric",
                                     "statistic", "value", "baseline_value",
                                     "change", "ratio"])
    changes = pd.concat(rows, ignore_index=True)
    return changes.sort_values(["identity", "window", "metric", "statistic"]).reset_index(
        drop=True)


def windowed_summaries(
    cell_frame: pd.DataFrame,
    context: MeasurementContext,
    windows: list[WindowConfig],
) -> dict[str, pd.DataFrame]:
    """The existing roll-ups, once per window, stacked with a ``window`` column.

    The roll-ups are called with an empty table dictionary on purpose. A module
    that folded a per-cell table into ``cell_summary`` measured the whole
    recording - a lifespan, a diffusion exponent - and folding that number into
    every window's row would repeat one whole-recording answer as though it were
    a windowed one. What a windowed summary holds is what can be recomputed from
    the window, and nothing else.
    """
    if not windows:
        return {}

    cell_rows, frame_rows = [], []
    for window in windows:
        held, length = window_extent(window, context)
        subset = cell_frame[frame_mask(cell_frame, window)]
        if subset.empty:
            # A window that caught nothing contributes nothing. Writing a row
            # per cell with its counts at zero would invent a row for a cell in
            # a stretch of the recording where it was never seen, and blanks in
            # a summary read as "measured, and there was nothing there".
            #
            # The record of the empty window lives where it cannot be mistaken
            # for a measurement: the run manifest carries every declared window
            # with the frames it held, and `python -m analysis doctor` counts a
            # window that catches no frames as a problem before anything is
            # measured.
            continue

        cells = build_cell_summary(subset, {}, context)
        cells.insert(0, "window", window.name)
        cells["window_frames"] = held
        cells["window_hours"] = length
        # Coverage *of the window*, which is the number a reader wants and is
        # not `coverage`: that one stays against the whole recording, because it
        # is the same formula the whole-recording roll-up uses, and giving one
        # name two meanings across two files is worse than adding a column.
        cells["window_coverage"] = cells["observed_frames"] / held
        cell_rows.append(cells)

        # `build_frame_summary` builds its rows from the context rather than
        # from what it is handed, so it returns every frame with metrics only
        # where the subset had rows. Filtering afterwards is what makes the
        # windowed rows identical to the whole-recording ones for those frames.
        frames = build_frame_summary(subset, context, {})
        frames = frames[frame_mask(frames, window)].copy()
        frames.insert(0, "window", window.name)
        frames["window_frames"] = held
        frames["window_hours"] = length
        frame_rows.append(frames)

    cell_summary = (pd.concat(cell_rows, ignore_index=True, sort=False)
                    if cell_rows else pd.DataFrame(columns=["window", "identity"]))
    frame_summary = (pd.concat(frame_rows, ignore_index=True, sort=False)
                     if frame_rows else pd.DataFrame(columns=["window", "frame_index"]))
    return {
        "cell_summary_windowed": cell_summary,
        "frame_summary_windowed": frame_summary,
        "window_change": _changes(cell_summary, windows),
    }
