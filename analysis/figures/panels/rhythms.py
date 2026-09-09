"""Panels for what the ``rhythms`` module measures.

These are here rather than in ``common`` because each of them knows something
about rhythmic data that a general chart does not. The raster shows the trace
after the configured Circadian Workbench detrending. The significant-period
histogram receives only cells that passed the selected rhythm verdict, so a
grey background population can never be mistaken for supported periods. The
older timing panels remain available for analyses that genuinely establish a
shared timing reference.
"""

from __future__ import annotations

from typing import Any, Sequence

from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from analysis import circadian as workbench
import matrix_ordering as _matrix_ordering

from _metrics import axis_label, role_for, semantic_label
from . import common
from ._contract import PanelResult

__all__ = [
    "detrended_z", "detrended_traces", "ordered_identities",
    "trace_displayed_onsets", "trace_pattern_order", "trace_raster",
    "significant_period_histogram", "phase_histogram", "timing_by_period",
    "noise_floor", "phase_dial",
    "phase_aligned", "population_mean", "cycle_loops", "channel_amplitudes",
    "active_spans", "period_status_matrix",
]


def detrended_z(hours: Sequence[float], values: Sequence[float], *,
                method: str = "linear", window_hours: float = 24.0,
                detrend_options: dict | None = None) -> np.ndarray:
    """A workbench-detrended trace, in units of its own spread.

    Two things at once, and both are what the rhythm test does before it looks
    for a cycle. Subtracting the line stops a cell that simply gets brighter all
    recording from reading as the first half of a wave; dividing by the spread
    puts a dim cell and a bright one on one colour scale, so a raster row is
    comparable to the row above it.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    residual = np.asarray(workbench.detrend_trace(
        hours, values,
        {**dict(detrend_options or {}), "detrend": method, "detrend_window_hours": window_hours},
    )["values"], dtype=float)

    # A trace that is exactly a straight line leaves a residual of rounding
    # error. Dividing that by its own standard deviation would magnify the
    # rounding to full contrast and draw a cell with no variation as the
    # noisiest row on the raster, so a negligible spread stays flat.
    return workbench.scale_detrended(residual, values)


def detrended_traces(cell_frame: pd.DataFrame, identities: Sequence[Any],
                     column: str, *, method: str = "linear",
                     window_hours: float = 24.0,
                     detrend_options: dict | None = None) -> pd.DataFrame:
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
            "detrended_z": detrended_z(
                usable["hours"], usable[column], method=method,
                window_hours=window_hours, detrend_options=detrend_options),
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["identity", "hours", "detrended_z"])


def ordered_identities(
    summary: pd.DataFrame,
    order_by: Sequence[str],
    *,
    identity_column: str = "identity",
) -> list[Any]:
    """Identity values in a stable, configurable order.

    This helper is deliberately unaware of circadian tests.  ``order_by`` may
    name any columns already present in ``summary``; prefixing a name with ``-``
    sorts that value from high to low.  A caller can therefore order a rhythm
    raster by onset then period, or reuse the same function for a non-rhythm
    raster with entirely different columns.
    """
    if identity_column not in summary.columns:
        raise KeyError(f"ordering table has no {identity_column!r} column")
    keys = list(order_by) or [identity_column]
    columns: list[str] = []
    ascending: list[bool] = []
    for key in keys:
        text = str(key).strip()
        descending = text.startswith("-")
        column = text[1:] if descending else text
        if not column:
            raise ValueError("an ordering column cannot be empty")
        if column not in summary.columns:
            available = ", ".join(map(str, summary.columns))
            raise KeyError(
                f"ordering column {column!r} is absent; available columns: {available}"
            )
        columns.append(column)
        ascending.append(not descending)
    ordered = summary.sort_values(
        columns, ascending=ascending, na_position="last", kind="mergesort"
    )
    return ordered[identity_column].tolist()


def _smoothed_ordering_copy(
    matrix: pd.DataFrame, smooth_points: int = 5,
) -> pd.DataFrame:
    """Centred smoothing used only to derive a row order."""
    smooth_points = max(1, int(smooth_points))
    interpolated = matrix.interpolate(axis=1, limit_area="inside")
    smoothed = interpolated.T.rolling(
        smooth_points, center=True, min_periods=1,
    ).mean().T
    values = matrix.to_numpy(float)
    within_span = np.zeros(values.shape, dtype=bool)
    for row, source in enumerate(values):
        finite = np.flatnonzero(np.isfinite(source))
        if len(finite):
            within_span[row, finite[0]:finite[-1] + 1] = True
    return smoothed.where(within_span)


def _filled_ordering_values(matrix: pd.DataFrame) -> np.ndarray:
    interpolated = matrix.interpolate(axis=1, limit_area="inside")
    return interpolated.fillna(0.0).to_numpy(float)


def _correlation_distances(values: np.ndarray) -> np.ndarray:
    rows = len(values)
    distances = np.zeros((rows, rows), dtype=float)
    for left in range(rows):
        for right in range(left + 1, rows):
            overlap = np.isfinite(values[left]) & np.isfinite(values[right])
            if overlap.sum() < 3:
                distance = 1.0
            else:
                x = values[left, overlap]
                y = values[right, overlap]
                x = x - x.mean()
                y = y - y.mean()
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                if denominator <= np.finfo(float).eps:
                    distance = 0.0 if np.allclose(x, y) else 1.0
                else:
                    correlation = float(np.dot(x, y) / denominator)
                    distance = float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
            distances[left, right] = distances[right, left] = distance
    return distances


def _principal_component_order(matrix: pd.DataFrame) -> list[Any]:
    values = matrix.to_numpy(float)
    comparable = np.isfinite(values).sum(axis=1) >= 3
    usable = matrix.loc[comparable]
    unavailable = matrix.index[~comparable].tolist()
    if len(usable) < 2:
        return [*usable.index.tolist(), *unavailable]

    filled = _filled_ordering_values(usable)
    centred = filled - filled.mean(axis=0, keepdims=True)
    left_vectors, singular_values, loadings = np.linalg.svd(
        centred, full_matrices=False,
    )
    coordinate = left_vectors[:, 0] * singular_values[0]
    time_direction = np.linspace(-1.0, 1.0, loadings.shape[1])
    if float(np.dot(loadings[0], time_direction)) < 0:
        coordinate = -coordinate
    positions = np.argsort(coordinate, kind="mergesort")
    return [*usable.index[positions].tolist(), *unavailable]


def _positive_centres(matrix: pd.DataFrame) -> np.ndarray:
    hours = matrix.columns.to_numpy(float)
    centres = []
    for values in matrix.to_numpy(float):
        weights = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
        centres.append(
            float(np.dot(weights, hours) / weights.sum())
            if weights.sum() > np.finfo(float).eps else np.nan
        )
    return np.asarray(centres, dtype=float)


def _spectral_continuum_order(matrix: pd.DataFrame) -> list[Any]:
    values = matrix.to_numpy(float)
    comparable = np.isfinite(values).sum(axis=1) >= 3
    usable = matrix.loc[comparable]
    unavailable = matrix.index[~comparable].tolist()
    if len(usable) < 3:
        return [*usable.index.tolist(), *unavailable]

    distances = _correlation_distances(usable.to_numpy(float))
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-np.square(distances) / (2.0 * scale * scale))
    np.fill_diagonal(similarity, 0.0)
    degree = similarity.sum(axis=1)
    inverse = np.diag(1.0 / np.sqrt(np.maximum(degree, np.finfo(float).eps)))
    laplacian = np.eye(len(usable)) - inverse @ similarity @ inverse
    _, vectors = np.linalg.eigh(laplacian)
    coordinate = vectors[:, 1]

    positive_centres = _positive_centres(usable)
    finite = np.isfinite(positive_centres)
    if finite.sum() >= 2:
        correlation = np.corrcoef(
            coordinate[finite], positive_centres[finite],
        )[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            coordinate = -coordinate
    positions = np.argsort(coordinate, kind="mergesort")
    return [*usable.index[positions].tolist(), *unavailable]


def trace_displayed_onsets(
    matrix: pd.DataFrame,
    *,
    smooth_points: int = 5,
    sustain_points: int = 3,
) -> pd.Series:
    """First sustained blue-to-red transition in absolute recording hours."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("trace onset ordering needs a pandas DataFrame")
    sustain_points = max(1, int(sustain_points))
    smoothed = _smoothed_ordering_copy(matrix, smooth_points)
    hours = smoothed.columns.to_numpy(float)
    onsets = []
    for values in smoothed.to_numpy(float):
        finite = np.flatnonzero(np.isfinite(values))
        onset = np.nan
        if len(finite):
            first = int(finite[0])
            initial = values[first:first + sustain_points]
            if (
                len(initial) == sustain_points
                and np.all(np.isfinite(initial))
                and np.all(initial > 0)
            ):
                onset = float(hours[first])
            else:
                for position in finite[1:]:
                    previous = position - 1
                    run = values[position:position + sustain_points]
                    if (
                        np.isfinite(values[previous])
                        and values[previous] <= 0 < values[position]
                        and len(run) == sustain_points
                        and np.all(np.isfinite(run))
                        and np.all(run > 0)
                    ):
                        onset = float(hours[position])
                        break
        onsets.append(onset)
    return pd.Series(onsets, index=matrix.index, name="displayed_onset_hours")


def trace_pattern_order(
    matrix: pd.DataFrame,
    *,
    method: str = "principal_component_gradient",
    smooth_points: int = 5,
) -> list[Any]:
    """Order whole blue/red trace patterns without shifting displayed time.

    ``principal_component_gradient`` sorts cells along the dominant whole-trace
    colour progression. ``spectral_continuum`` instead builds a smooth path
    through pairwise trace similarities. Both use a centred smoothed copy only
    to derive the order; the raster still displays the original detrended rows.
    Rows with fewer than three finite samples retain their input order at the
    bottom because there is too little displayed shape to compare.
    """
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("trace pattern ordering needs a pandas DataFrame")
    aliases = {
        "principal_component": "principal_component_gradient",
        "pca": "principal_component_gradient",
        "spectral": "spectral_continuum",
    }
    method = aliases.get(str(method), str(method))
    smoothed = _smoothed_ordering_copy(matrix, smooth_points)
    if method == "principal_component_gradient":
        return _principal_component_order(smoothed)
    if method == "spectral_continuum":
        return _spectral_continuum_order(smoothed)
    raise ValueError(
        "trace pattern method must be principal_component_gradient or "
        "spectral_continuum"
    )


# Compatibility exports now delegate to the general matrix-ordering engine.
ordered_identities = _matrix_ordering.ordered_identities
trace_displayed_onsets = _matrix_ordering.displayed_onsets
trace_pattern_order = _matrix_ordering.trace_pattern_order


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


def period_status_matrix(
    ax: Any,
    periods: Any,
    statuses: Any,
    theme: Any,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    underdetermined: Any | None = None,
    period_min: float,
    period_max: float,
    cmap: Any = None,
    column_label_rotation: float = 0.0,
    x_label: str = "",
    y_label: str = "Cell identity",
) -> PanelResult:
    """A generic row-by-column rhythm matrix with an explicit test verdict.

    Colour is only used after a trace passes its family-corrected rhythm test;
    its value is that trace's strongest period. Neutral fills distinguish a
    completed non-significant test from a trace that could not be tested. The
    optional outline marks a period supported by fewer cycles than the caller
    considers well constrained.
    """
    values = np.asarray(periods, dtype=float)
    verdicts = np.asarray(statuses, dtype=object)
    if values.shape != verdicts.shape:
        raise ValueError("periods and statuses must have the same matrix shape")
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError("matrix shape must match its row and column labels")
    uncertain = (
        np.zeros(values.shape, dtype=bool) if underdetermined is None
        else np.asarray(underdetermined, dtype=bool)
    )
    if uncertain.shape != values.shape:
        raise ValueError("underdetermined must match the period matrix shape")

    background = np.where(verdicts == "not rhythmic", 1.0, 0.0)
    ax.imshow(
        background, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
        cmap=ListedColormap([
            theme.colour("missing"), theme.colour("arrhythmic")
        ]),
    )
    chosen_cmap = colormaps.get_cmap(
        theme["sequential_cmap"] if cmap is None else cmap).copy()
    chosen_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    coloured = np.ma.masked_where(verdicts != "rhythmic", values)
    handle = ax.imshow(
        coloured, aspect="auto", interpolation="nearest", cmap=chosen_cmap,
        vmin=float(period_min), vmax=float(period_max),
    )

    outlined = uncertain & (verdicts == "rhythmic")
    unavailable = (verdicts == "period unavailable") | ((verdicts == "rhythmic") & ~np.isfinite(values))
    for row, column in zip(*np.nonzero(unavailable)):
        ax.add_patch(Rectangle(
            (column - 0.5, row - 0.5), 1.0, 1.0,
            facecolor=theme.colour("missing"), edgecolor=theme.colour("ink"),
            linewidth=0, hatch="///"))
    for row, column in zip(*np.nonzero(outlined)):
        ax.add_patch(Rectangle(
            (column - 0.5, row - 0.5), 1.0, 1.0,
            fill=False, edgecolor=theme.colour("ink"),
            linewidth=theme.stroke("hairline"),
        ))

    ax.set_xticks(np.arange(len(column_labels)))
    rotation = float(column_label_rotation)
    ax.set_xticklabels(
        list(column_labels), rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    ax.tick_params(axis="y", labelsize=max(6.0, theme.size("caption") * 0.62))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(
        which="minor", color=theme.colour("page"),
        linewidth=theme.stroke("hairline"),
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    rows, columns = np.indices(values.shape)
    table = pd.DataFrame({
        "row": [list(row_labels)[index] for index in rows.ravel()],
        "column": [list(column_labels)[index] for index in columns.ravel()],
        "period_hours": values.ravel(),
        "rhythm_status": verdicts.ravel(),
        "period_underdetermined": uncertain.ravel(),
    })
    handles = [
        Patch(facecolor=theme.colour("arrhythmic"), edgecolor="none",
              label="Tested; not rhythmic"),
        Patch(facecolor=theme.colour("missing"), edgecolor="none",
              label="Not tested"),
        Patch(facecolor="none", edgecolor=theme.colour("ink"),
              linewidth=theme.stroke("hairline"),
              label="Fewer than required cycles"),
    ]
    if unavailable.any():
        handles.append(Patch(facecolor=theme.colour("missing"), edgecolor=theme.colour("ink"),
                             hatch="///", label="Rhythmic; period fit unavailable"))
    return PanelResult(
        data=table, axes=ax,
        extra={"handle": handle, "legend_handles": handles},
    )


def significant_period_histogram(
    ax: Any,
    period_hours: Sequence[float],
    theme: Any,
    *,
    bins: int | Sequence[float],
    x_label: str = "Estimated period (h)",
    y_label: str = "Significant cells (count)",
) -> PanelResult:
    """Frequency of estimated periods among significant cells only.

    Significance is deliberately settled by the caller because the panel must
    not invent a test or silently reinterpret a saved verdict. The returned
    table carries both the plotted count and its share of all significant cells
    with an available period.
    """
    periods = np.asarray(period_hours, dtype=float)
    periods = periods[np.isfinite(periods)]
    drawn = common.histogram(
        ax, periods, theme, bins=bins, role="rhythmic",
        x_label=x_label, y_label=y_label,
    )
    table = drawn.data.rename(columns={
        "bin_left": "period_start_hour",
        "bin_right": "period_end_hour",
        "count": "significant_cell_count",
    })
    total = int(table["significant_cell_count"].sum())
    table["significant_cell_frequency"] = (
        table["significant_cell_count"] / total if total else 0.0
    )
    for patch, count in zip(ax.patches, table["significant_cell_count"]):
        if int(count) > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(), str(int(count)),
                ha="center", va="bottom",
                fontsize=theme.size("annotation"),
                color=theme.colour("ink"),
            )
    if total:
        ax.margins(y=0.12)
    return PanelResult(
        data=table, axes=ax,
        extra={**drawn.extra, "significant_cells": total},
    )


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


def timing_by_period(
    ax: Any,
    timing_hours: Sequence[float],
    cycle_hours: Sequence[float],
    detected_period_hours: Sequence[float],
    theme: Any,
    *,
    valid: Sequence[bool] | None = None,
    bins: int = 12,
    period_edges: Sequence[float],
    timing_label: str = "Peak",
    x_label: str | None = None,
    y_label: str = "Detected period (h)",
) -> PanelResult:
    """A timing distribution for cells whose cycle lengths differ.

    Every timing is first divided by the cycle it was measured on.  The x-axis
    is therefore position within that cell's own cycle, while rows are bands of
    the independently supplied detected period.  ``timing_hours`` and
    ``cycle_hours`` can be any precomputed columns; no rhythm test is run here.
    """
    timing = np.asarray(timing_hours, dtype=float)
    cycles = np.asarray(cycle_hours, dtype=float)
    periods = np.asarray(detected_period_hours, dtype=float)
    if not (len(timing) == len(cycles) == len(periods)):
        raise ValueError("timing, cycle and detected-period arrays need equal lengths")
    keep = np.isfinite(timing) & np.isfinite(cycles) & (cycles > 0)
    keep &= np.isfinite(periods) & (periods > 0)
    if valid is not None:
        selected = np.asarray(valid, dtype=bool)
        if len(selected) != len(timing):
            raise ValueError("valid needs one value per timing")
        keep &= selected

    period_edges = np.asarray(period_edges, dtype=float)
    if (period_edges.ndim != 1 or len(period_edges) < 2
            or not np.isfinite(period_edges).all()
            or not np.all(np.diff(period_edges) > 0)):
        raise ValueError("period_edges needs at least two increasing finite values")
    bins = int(bins)
    if bins < 2:
        raise ValueError("bins must be at least 2")

    phase = np.mod(timing[keep], cycles[keep]) / cycles[keep]
    phase_edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(
        periods[keep], phase, bins=(period_edges, phase_edges)
    )
    handle = ax.imshow(
        counts, aspect="auto", interpolation="nearest",
        cmap=theme["sequential_cmap"], vmin=0,
        vmax=max(1.0, float(np.nanmax(counts)) if counts.size else 1.0),
    )

    period_labels = [
        f"{start:g}–{end:g}"
        for start, end in zip(period_edges[:-1], period_edges[1:])
    ]
    ax.set_yticks(np.arange(len(period_labels)))
    ax.set_yticklabels(period_labels)
    quarter_positions = np.linspace(-0.5, bins - 0.5, 5)
    ax.set_xticks(quarter_positions)
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel(x_label or f"{timing_label} position in its own cycle (%)")
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(bins + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(period_labels) + 1) - 0.5, minor=True)
    ax.grid(
        which="minor", color=theme.colour("page"),
        linewidth=theme.stroke("hairline")
    )
    ax.tick_params(which="minor", bottom=False, left=False)
    for row, column in zip(*np.nonzero(counts)):
        ratio = counts[row, column] / max(float(np.nanmax(counts)), 1.0)
        ax.text(
            column, row, f"{int(counts[row, column])}",
            ha="center", va="center", fontsize=theme.size("annotation"),
            color=theme.colour("ink" if ratio >= 0.55 else "page"),
        )

    rows, columns = np.indices(counts.shape)
    data = pd.DataFrame({
        "period_start_hour": period_edges[rows.ravel()],
        "period_end_hour": period_edges[rows.ravel() + 1],
        "phase_start_fraction": phase_edges[columns.ravel()],
        "phase_end_fraction": phase_edges[columns.ravel() + 1],
        "cells": counts.ravel().astype(int),
    })
    return PanelResult(
        data=data, axes=ax,
        extra={"handle": handle, "cells": int(keep.sum())},
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
    tick_values: Sequence[float] | None = None,
    tick_labels: Sequence[str] | None = None,
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
    ticks = (np.arange(0, period_hours, float(hour_ticks))
             if tick_values is None else np.asarray(tick_values, dtype=float))
    ax.set_xticks(2 * np.pi * ticks / period_hours)
    if tick_labels is None:
        labels = [f"{tick:g} h" for tick in ticks]
    else:
        labels = list(tick_labels)
        if len(labels) != len(ticks):
            raise ValueError("tick_labels must contain one label per tick_values entry")
    ax.set_xticklabels(labels)
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
    detrend: str = str(workbench.DETREND_DEFAULTS["detrend"]),
    detrend_window_hours: float = float(
        workbench.DETREND_DEFAULTS["detrend_window_hours"]),
    detrend_options: dict | None = None,
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
        values = detrended_z(
            usable["hours"], usable[column], method=detrend,
            window_hours=detrend_window_hours,
            detrend_options=detrend_options,
        )
        phase = ((usable["hours"].to_numpy(float) - float(peak_map[identity]) + period_hours / 2)
                 % period_hours) - period_hours / 2
        rows.append(pd.DataFrame({"identity": identity, "phase_hour": phase, "value_z": values}))
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["identity", "phase_hour", "value_z"]
    )
    edges = np.linspace(-period_hours / 2, period_hours / 2, int(bins) + 1)
    long["bin"] = pd.cut(long["phase_hour"], edges, labels=False, include_lowest=True)
    summaries = []
    for index, group in long.dropna(subset=["bin", "value_z"]).groupby("bin"):
        values = group["value_z"].to_numpy(float)
        summaries.append({
            "phase_hour": float((edges[int(index)] + edges[int(index) + 1]) / 2),
            "mean_z": float(np.mean(values) if aggregate == "mean" else np.median(values)),
            "lo": float(np.quantile(values, band[0])),
            "hi": float(np.quantile(values, band[1])),
            "cells": int(group["identity"].nunique()),
            "detrend": detrend,
            "detrend_window_hours": float(detrend_window_hours),
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
    detrend: str = str(workbench.DETREND_DEFAULTS["detrend"]),
    detrend_window_hours: float = float(
        workbench.DETREND_DEFAULTS["detrend_window_hours"]),
    detrend_options: dict | None = None,
    hour_ticks: float = 6.0,
) -> PanelResult:
    """One polar loop per observed cycle; incomplete last cycles stay open."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    usable = np.isfinite(hours) & np.isfinite(values)
    hours, values = hours[usable], values[usable]
    if detrended and len(values) >= 2:
        result = workbench.detrend_trace(
            hours, values,
            {**dict(detrend_options or {}), "detrend": detrend,
             "detrend_window_hours": detrend_window_hours},
        )
        plotted = np.asarray(result["values"], dtype=float)
        slope = float(
            np.polyfit(hours[np.isfinite(plotted)], values[np.isfinite(plotted)], 1)[0]
        ) if detrend == "linear" and np.isfinite(plotted).sum() >= 2 else np.nan
    else:
        slope, plotted = 0.0, values.copy()
    finite_plotted = plotted[np.isfinite(plotted)]
    if finite_plotted.size == 0:
        return PanelResult(data=pd.DataFrame(), axes=ax)
    radial_origin = float(np.min(finite_plotted))
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
        radius = plotted[keep] - radial_origin + 1e-9
        look = looks.get(cycle) or common.Look(colour=theme.colour(roles[cycle % len(roles)]))
        ax.plot(theta, radius, color=look.colour, linewidth=theme.stroke("line"),
                label=f"cycle {cycle + 1}")
        rows.extend({"cycle": cycle + 1, "phase_hour": float(p), "value": float(v),
                     "detrended_value": float(d), "trend_slope": float(slope),
                     "detrend": detrend if detrended else "none",
                     "detrend_window_hours": float(detrend_window_hours)}
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


def active_spans(
    ax: Any,
    onsets: Sequence[float],
    offsets: Sequence[float],
    theme: Any,
    *,
    groups: Sequence[str],
    colours: Sequence[str] | None = None,
    period: float = 24.0,
    role: str = "rhythmic",
    gap_rows: int = 2,
    x_label: str = "",
) -> PanelResult:
    """Each cell's active period as a bar, blocked by which signal it belongs to.

    A bar runs from the hour the active phase begins to the hour it ends. When
    the end is earlier than the start the period runs through the end of the
    day, and the bar is drawn as two: one to the day boundary and one from it.
    That case is not exotic - on a signal whose quiet phase straddles the fold,
    it is most of the rows - and the two alternatives are both wrong. A bar
    drawn from the larger number to the smaller one has negative width and
    disappears; a row quietly dropped removes a cell that was measured.

    Rows are drawn in the order given, and blocks are separated by
    ``gap_rows`` blank rows so the eye can find where one signal ends. The
    caller chooses the order, because "sorted by onset" and "sorted by identity"
    answer different questions and this panel should not pick one.

    One row per drawn *segment*, so a wrapped period is two rows of the returned
    table and the segment lengths still add up to the active duration.
    """
    onsets = np.asarray(onsets, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    groups = np.asarray(groups, dtype=object)
    if not (len(onsets) == len(offsets) == len(groups)):
        raise ValueError("onsets, offsets and groups must be the same length")

    order = list(dict.fromkeys(groups.tolist()))
    palette = (list(colours) if colours is not None
               else [theme.colour(role)] * len(order))
    colour_of = dict(zip(order, palette))

    rows = []
    y = 0.0
    block_centres = []
    for group in order:
        members = np.flatnonzero(groups == group)
        first = y
        for index in members:
            start, end = float(onsets[index]), float(offsets[index])
            if not (np.isfinite(start) and np.isfinite(end)):
                y += 1.0
                continue
            segments = ([(start, end, False)] if end >= start
                        else [(start, period, True), (0.0, end, True)])
            for low, high, wrapped in segments:
                # A zero-width segment is recorded and not drawn. It is a real
                # measurement - an active phase that begins and ends in the same
                # hour - and dropping the row instead would take the cell off
                # the table as well as off the page.
                if high > low:
                    ax.barh(y, high - low, left=low, height=0.86,
                            color=colour_of[group], linewidth=0)
                rows.append({"group": str(group), "row": int(y), "index": int(index),
                             "start_hour": low, "end_hour": high,
                             "hours": high - low, "wrapped": bool(wrapped)})
            y += 1.0
        block_centres.append((group, (first + y - 1.0) / 2.0))
        y += gap_rows

    ax.set_ylim(y - gap_rows, -1.0)
    ax.set_xlim(0.0, period)
    ax.set_yticks([centre for _, centre in block_centres])
    ax.set_yticklabels([semantic_label(name) if _known(name) else str(name)
                        for name, _ in block_centres])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(x_label)
    return PanelResult(
        data=pd.DataFrame(rows, columns=["group", "row", "index", "start_hour",
                                         "end_hour", "hours", "wrapped"]),
        axes=ax)


def _known(column: str) -> bool:
    """Whether ``semantic_label`` will answer for this name rather than raise."""
    try:
        semantic_label(str(column))
    except (ValueError, KeyError):
        return False
    return True
