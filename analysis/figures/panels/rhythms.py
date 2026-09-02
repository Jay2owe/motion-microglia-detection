"""Panels for what the ``rhythms`` module measures.

These are here rather than in ``common`` because each of them knows something
about the rhythm test that a general chart does not. The raster shows the trace
*after* the same straight line the test subtracted, so the picture and the
p-value are looking at the same thing. The phase histogram is 24 h wide and
wraps, because a peak at 23.5 h is next to a peak at 0.5 h and a general
histogram would put them at opposite ends. The noise-floor panel refuses to draw
a rate without the surrogate rate beside it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from _metrics import axis_label, role_for, semantic_label
from . import common
from ._contract import PanelResult

__all__ = [
    "detrended_z", "detrended_traces", "trace_raster", "phase_histogram",
    "noise_floor", "phase_dial", "phase_aligned", "population_mean",
    "cycle_loops", "channel_amplitudes",
]


def detrended_z(hours: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """A trace with its straight-line drift removed, in units of its own spread.

    Two things at once, and both are what the rhythm test does before it looks
    for a cycle. Subtracting the line stops a cell that simply gets brighter all
    recording from reading as the first half of a wave; dividing by the spread
    puts a dim cell and a bright one on one colour scale, so a raster row is
    comparable to the row above it.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(hours, values, 1)
    residual = values - (slope * hours + intercept)

    # A trace that is exactly a straight line leaves a residual of rounding
    # error. Dividing that by its own standard deviation would magnify the
    # rounding to full contrast and draw a cell with no variation as the
    # noisiest row on the raster, so a negligible spread stays flat.
    spread = float(residual.std())
    scale = float(np.abs(values).max()) or 1.0
    if spread <= scale * 1e-9:
        return np.zeros_like(residual)
    return residual / spread


def detrended_traces(cell_frame: pd.DataFrame, identities: Sequence[Any],
                     column: str) -> pd.DataFrame:
    """Long table of every named cell's detrended trace: identity, hours, value."""
    pieces = []
    wanted = cell_frame[cell_frame["identity"].isin(list(identities))]
    for identity, group in wanted.groupby("identity"):
        usable = group[["hours", column]].dropna().sort_values("hours")
        if len(usable) < 2:
            continue
        pieces.append(pd.DataFrame({
            "identity": identity,
            "hours": usable["hours"].to_numpy(float),
            "detrended_z": detrended_z(usable["hours"], usable[column]),
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["identity", "hours", "detrended_z"])


def trace_raster(
    ax: Any,
    matrix: Any,
    hours: Sequence[float],
    theme: Any,
    *,
    blocks: Sequence[tuple[int, str, str]] = (),
    limit: float | None = None,
    cmap: Any = None,
    hour_ticks: float | None = None,
    y_label: str = "Cell",
    x_label: str = "Hours from start of recording",
) -> PanelResult:
    """Every cell's detrended trace as a row of colour, stacked.

    ``blocks`` is (how many rows, colour role, what to call them) from the top
    down. Each block is named beside its own rows and separated by a rule, so
    "which of these cells passed" is answered where the rows are rather than in
    a legend.

    The colour scale is symmetric about zero and clipped at the 98th percentile
    of the absolute values, so a single extreme cell cannot flatten every other
    row to grey.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows = matrix.shape[0]
    if limit is None:
        limit = float(np.nanpercentile(np.abs(matrix), 98))

    drawn = common.raster(
        ax, matrix, theme, hours=hours,
        cmap=theme["diverging_cmap"] if cmap is None else cmap,
        vmin=-limit, vmax=limit,
    )
    hours = np.asarray(hours, dtype=float)
    ax.set_xlabel(x_label)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    ax.set_ylabel(y_label)
    ax.set_yticks([0, rows - 1])
    ax.set_yticklabels(["1", str(rows)])

    start = 0
    for count, role, label in blocks:
        count = int(count)
        if count <= 0:
            continue
        centre = start + count / 2
        ax.text(
            -0.055, 1 - centre / rows, f"{label}\n{count}", transform=ax.transAxes,
            ha="right", va="center", fontsize=theme.size("subtitle"),
            color=theme.colour(role), fontweight="bold",
        )
        start += count
        if start < rows:
            ax.axhline(
                start - 0.5, color=theme.colour("ink"), linewidth=theme.stroke("line")
            )
    return PanelResult(data=drawn.data, axes=ax,
                       extra={"handle": drawn.extra["handle"]})


def phase_histogram(
    ax: Any,
    peak_hours: Sequence[float],
    theme: Any,
    *,
    passed: Sequence[bool] | None = None,
    bins: int = 12,
    period_hours: float = 24.0,
    mean_hour: float | None = None,
    hour_ticks: float | None = 6.0,
    total_label: str = "all tested",
    passed_label: str = "rhythmic by both tests",
    x_label: str = "Peak time within a 24 h cycle",
    y_label: str = "Cells",
) -> PanelResult:
    """When in the day each cell peaks, with the ones that passed stacked inside.

    Two bars per band rather than two panels: the subset is drawn inside the
    total so the reader sees the fraction and the shape at once, and cannot
    mistake a tall rhythmic bar for a common peak time when few cells were
    tested there.

    Returns (band edges, counts of everything, counts of the subset).
    """
    peak_hours = np.asarray(peak_hours, dtype=float)
    edges = np.linspace(0.0, float(period_hours), int(bins) + 1)
    width = (edges[1] - edges[0]) * 0.9
    centres = (edges[:-1] + edges[1:]) / 2

    counts_all, _ = np.histogram(peak_hours, bins=edges)
    if passed is None:
        counts_passed = np.zeros_like(counts_all)
    else:
        counts_passed, _ = np.histogram(peak_hours[np.asarray(passed, dtype=bool)], bins=edges)

    ax.bar(centres, counts_all, width=width, color=theme.colour("arrhythmic"),
           label=f"{total_label} ({len(peak_hours)})")
    if passed is not None:
        ax.bar(centres, counts_passed, width=width, color=theme.colour("rhythmic"),
               label=f"{passed_label} ({int(np.asarray(passed, dtype=bool).sum())})")
    if mean_hour is not None:
        common.reference_lines(
            ax, [common.Mark(float(mean_hour), role="highlight")], theme
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, float(period_hours))
    ax.set_xticks(theme.hour_ticks(0, float(period_hours), hour_ticks))
    return PanelResult(
        data=pd.DataFrame({"bin_start_hour": edges[:-1], "bin_end_hour": edges[1:],
                           "cells_all_tested": counts_all,
                           "cells_rhythmic_by_both_tests": counts_passed}),
        axes=ax,
        extra={"edges": edges, "counts_all": counts_all,
               "counts_passed": counts_passed},
    )


def noise_floor(
    ax: Any,
    observed: Sequence[float],
    surrogate: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str],
    excess: Sequence[float] | None = None,
    notable: float = 0.05,
    observed_label: str = "real cells",
    surrogate_label: str = "drift-matched noise",
    x_label: str = "Cells called rhythmic by both tests",
) -> PanelResult:
    """How often the test fires on the data against how often it fires on noise.

    The hollow dot is the surrogate rate and the filled dot the real one, joined
    so the gap between them is the whole reading. A rate on its own cannot be
    interpreted here: a short recording makes a smooth, drifting trace look like
    the first half of a cosine, and how often that happens differs between
    measurements - so the floor has to be drawn per row, not stated once.
    """
    observed = np.asarray(observed, dtype=float)
    surrogate = np.asarray(surrogate, dtype=float)
    gain = observed > surrogate

    annotations = annotation_colours = None
    if excess is not None:
        excess = np.asarray(excess, dtype=float)
        annotations = [f"+{v * 100:.0f} pp" if v >= 0 else f"{v * 100:.0f} pp" for v in excess]
        annotation_colours = [
            theme.colour("significant" if v >= notable else "not_significant") for v in excess
        ]

    rows = common.dumbbell(
        ax, surrogate, observed, theme,
        labels=labels,
        right_colours=[theme.colour("significant" if g else "not_significant") for g in gain],
        annotations=annotations,
        annotation_colours=annotation_colours,
        left_label=surrogate_label,
        right_label=observed_label,
    )
    ax.set_xlabel(x_label)
    drawn = rows.data.rename(columns={"left": "surrogate_rate",
                                      "right": "observed_rate"})
    if excess is not None:
        drawn["excess"] = excess
    return PanelResult(data=drawn, axes=ax)


def phase_dial(
    ax: Any,
    peak_hours: Sequence[float],
    amplitudes: Sequence[float],
    theme: Any,
    *,
    passed: Sequence[bool] | None = None,
    period_hours: float = 24.0,
    resultant: bool = True,
    hour_ticks: float = 6.0,
    radius_label: str = "Amplitude",
) -> PanelResult:
    """One cell per vector: phase is angle and amplitude is radius."""
    peaks = np.asarray(peak_hours, dtype=float)
    radii = np.asarray(amplitudes, dtype=float)
    usable = np.isfinite(peaks) & np.isfinite(radii)
    peaks, radii = peaks[usable], radii[usable]
    filled = np.ones(len(peaks), dtype=bool) if passed is None else np.asarray(passed, dtype=bool)[usable]
    summary = common.vectors(
        ax, peaks, radii, theme, period=period_hours,
        colours=[theme.colour("rhythmic" if keep else "arrhythmic") for keep in filled],
        filled=filled, resultant=resultant, role="reporter", radius_label=radius_label,
    )
    ticks = np.arange(0, period_hours, float(hour_ticks))
    ax.set_xticks(2 * np.pi * ticks / period_hours)
    ax.set_xticklabels([f"{tick:g} h" for tick in ticks])
    # ``amplitude``, not ``relative_amplitude``: this panel is handed radii and
    # does not know which amplitude they are. The caller says so through
    # ``radius_label``, and naming the column after one particular measure would
    # put that measure\'s name on every figure that ever drew a dial.
    table = pd.DataFrame({"peak_hour": peaks, "amplitude": radii, "passed": filled})
    table.attrs["mean_peak_hour"] = summary.extra["mean_period"]
    table.attrs["vector_length"] = summary.extra["concentration"]
    return PanelResult(
        data=table, axes=ax,
        extra={"mean_peak_hour": summary.extra["mean_period"],
               "vector_length": summary.extra["concentration"]},
    )


def phase_aligned(
    ax: Any,
    frame: pd.DataFrame,
    theme: Any,
    *,
    column: str,
    peak_hours: Any,
    period_hours: float = 24.0,
    aggregate: str = "mean",
    band: tuple[float, float] = (0.25, 0.75),
    bins: int = 24,
    hour_ticks: float = 6.0,
    x_label: str = "Hours from this cell's own peak",
    role: str | None = None,
    label: str = "",
) -> PanelResult:
    """Detrend each trace, shift its fitted peak to zero, then aggregate."""
    peak_map = peak_hours if isinstance(peak_hours, dict) else dict(peak_hours)
    rows = []
    for identity, group in frame.groupby("identity", sort=True):
        if identity not in peak_map:
            continue
        usable = group[["hours", column]].dropna().sort_values("hours")
        if len(usable) < 3:
            continue
        values = detrended_z(usable["hours"], usable[column])
        phase = ((usable["hours"].to_numpy(float) - float(peak_map[identity]) + period_hours / 2)
                 % period_hours) - period_hours / 2
        rows.append(pd.DataFrame({"identity": identity, "phase_hour": phase, "value_z": values}))
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["identity", "phase_hour", "value_z"]
    )
    edges = np.linspace(-period_hours / 2, period_hours / 2, int(bins) + 1)
    long["bin"] = pd.cut(long["phase_hour"], edges, labels=False, include_lowest=True)
    summaries = []
    for index, group in long.dropna(subset=["bin"]).groupby("bin"):
        values = group["value_z"].to_numpy(float)
        summaries.append({
            "phase_hour": float((edges[int(index)] + edges[int(index) + 1]) / 2),
            "mean_z": float(np.mean(values) if aggregate == "mean" else np.median(values)),
            "lo": float(np.quantile(values, band[0])),
            "hi": float(np.quantile(values, band[1])),
            "cells": int(group["identity"].nunique()),
        })
    table = pd.DataFrame(summaries)
    colour = theme.colour(role or role_for(column))
    if not table.empty:
        ax.fill_between(table["phase_hour"], table["lo"], table["hi"],
                        color=colour, alpha=0.22, linewidth=0)
        ax.plot(table["phase_hour"], table["mean_z"], color=colour,
                linewidth=theme.stroke("emphasis"),
                label=label or semantic_label(column))
    ax.axvline(0, color=theme.colour("reference"), linewidth=theme.stroke("guide"))
    ax.set_xlim(-period_hours / 2, period_hours / 2)
    ax.set_xticks(np.arange(-period_hours / 2, period_hours / 2 + 1e-9, hour_ticks))
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{semantic_label(column)}\n(standardised per cell)")
    return PanelResult(data=table, axes=ax)


def population_mean(
    ax: Any,
    frame: pd.DataFrame,
    theme: Any,
    *,
    column: str,
    hour_ticks: float | None = None,
    band: tuple[float, float] = (0.25, 0.75),
    role: str | None = None,
    label: str = "",
) -> PanelResult:
    """Population trace on wall-clock time with an interquartile band."""
    data = frame[["identity", "hours", column]].dropna().copy()
    grouped = data.groupby("hours")[column]
    table = pd.DataFrame({
        "hours": grouped.mean().index,
        "mean": grouped.mean().values,
        "lo": grouped.quantile(band[0]).values,
        "hi": grouped.quantile(band[1]).values,
        "cells": grouped.count().values,
    })
    colour = theme.colour(role or role_for(column))
    if not table.empty:
        ax.fill_between(table["hours"], table["lo"], table["hi"],
                        color=colour, alpha=0.22, linewidth=0)
        ax.plot(table["hours"], table["mean"], color=colour,
                linewidth=theme.stroke("emphasis"),
                label=label or semantic_label(column))
        ax.set_xticks(theme.hour_ticks(float(table["hours"].min()),
                                       float(table["hours"].max()), hour_ticks))
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(axis_label(column, wrap=True))
    return PanelResult(data=table, axes=ax)


def cycle_loops(
    ax: Any,
    hours: Sequence[float],
    values: Sequence[float],
    theme: Any,
    *,
    period_hours: float = 24.0,
    cycles: Sequence[int] | None = None,
    looks: dict[int, common.Look] | None = None,
    detrended: bool = True,
    hour_ticks: float = 6.0,
) -> PanelResult:
    """One polar loop per observed cycle; incomplete last cycles stay open."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    usable = np.isfinite(hours) & np.isfinite(values)
    hours, values = hours[usable], values[usable]
    if detrended and len(values) >= 2:
        slope, intercept = np.polyfit(hours, values, 1)
        plotted = values - (slope * hours + intercept)
    else:
        slope, intercept, plotted = 0.0, 0.0, values.copy()
    base = float(hours.min()) if len(hours) else 0.0
    cycle_ids = np.floor((hours - base) / period_hours).astype(int)
    selected = set(cycle_ids if cycles is None else cycles)
    rows = []
    looks = looks or {}
    roles = ("reporter", "morphology", "surveillance", "motility")
    for cycle in sorted(selected):
        keep = cycle_ids == cycle
        if not keep.any():
            continue
        phase = np.mod(hours[keep] - base, period_hours)
        theta = 2 * np.pi * phase / period_hours
        radius = plotted[keep] - float(np.nanmin(plotted)) + 1e-9
        look = looks.get(cycle) or common.Look(colour=theme.colour(roles[cycle % len(roles)]))
        ax.plot(theta, radius, color=look.colour, linewidth=theme.stroke("line"),
                label=f"cycle {cycle + 1}")
        rows.extend({"cycle": cycle + 1, "phase_hour": float(p), "value": float(v),
                     "detrended_value": float(d), "trend_slope": float(slope)}
                    for p, v, d in zip(phase, values[keep], plotted[keep]))
    ticks = np.arange(0, period_hours, hour_ticks)
    ax.set_xticks(2 * np.pi * ticks / period_hours)
    ax.set_xticklabels([f"{tick:g} h" for tick in ticks])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return PanelResult(data=pd.DataFrame(rows), axes=ax)


def channel_amplitudes(
    ax: Any,
    observed: Sequence[float],
    surrogate: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str],
    kinds: Sequence[str],
    notable: float | None = None,
    x_label: str = "Relative amplitude at the assumed period",
) -> PanelResult:
    """Observed fitted amplitude beside each channel's own surrogate floor."""
    observed = np.asarray(observed, dtype=float)
    surrogate = np.asarray(surrogate, dtype=float)
    kinds = np.asarray(kinds, dtype=object)
    colours = [theme.colour("invalid" if kind in {"tracker", "artefact"} else "reporter")
               for kind in kinds]
    positions = np.arange(len(labels))
    for y, (floor, value, colour) in enumerate(zip(surrogate, observed, colours)):
        ax.plot([floor, value], [y, y], color=colour, alpha=0.55,
                linewidth=theme.stroke("emphasis"))
        ax.scatter(floor, y, facecolor=theme.colour("page"), edgecolor=colour,
                   s=theme.point_area(1.25), linewidth=theme.stroke("line"))
        ax.scatter(value, y, color=colour, s=theme.point_area(1.25))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    return PanelResult(
        data=pd.DataFrame({
            "channel": list(labels), "kind": kinds, "amplitude": observed,
            "surrogate_amplitude": surrogate, "excess": observed - surrogate,
        }),
        axes=ax,
    )
