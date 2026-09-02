"""Panels for temporal and pairwise coupling measurements."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult


__all__ = [
    "lag_profile", "phase_difference", "similarity_matrix", "recurrence",
    "contact_chord", "phase_map",
]


def lag_profile(
    ax: Any,
    lags: Sequence[float],
    correlation: Sequence[float],
    theme: Any,
    *,
    surrogate: Any = None,
    pairs: Sequence[float] | None = None,
    minutes_per_frame: float | None = None,
    look: common.Look | None = None,
    role: str = "morphology",
    thin_below: int | None = None,
    x_label: str = "Lag (hours)",
    y_label: str = "Within-cell correlation",
    direction_note: bool = True,
    label: str = "Observed cells",
) -> PanelResult:
    """Correlation against lag with the matched null behind it."""
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(correlation, dtype=float)
    if minutes_per_frame is not None and x_label == "Lag (hours)" and np.allclose(lags, np.round(lags)):
        x = lags * float(minutes_per_frame) / 60.0
    else:
        x = lags
    result = pd.DataFrame({"lag": x, "correlation": values})
    if pairs is not None:
        result["pairs"] = np.asarray(pairs, dtype=float)
    if surrogate is not None:
        if isinstance(surrogate, pd.DataFrame):
            lo = surrogate["surrogate_lo"].to_numpy(float)
            hi = surrogate["surrogate_hi"].to_numpy(float)
            mean = surrogate["surrogate_mean"].to_numpy(float)
        else:
            null = np.asarray(surrogate, dtype=float)
            if null.ndim == 2 and null.shape[1] >= 2:
                lo, hi = null[:, 0], null[:, 1]
                mean = np.mean(null[:, :2], axis=1)
            else:
                mean = null
                lo = hi = null
        ax.fill_between(x, lo, hi, color=theme.colour("reference"), alpha=0.24,
                        linewidth=0, label="Matched-noise interval")
        ax.plot(x, mean, color=theme.colour("reference"), linewidth=theme.stroke("guide"))
        result["surrogate_mean"] = mean
        result["surrogate_lo"] = lo
        result["surrogate_hi"] = hi
    colour = (look or common.Look(colour=theme.colour(role))).colour or theme.colour(role)
    if pairs is None or thin_below is None:
        ax.plot(x, values, color=colour, linewidth=theme.stroke("emphasis"),
                label=label or None)
    else:
        pair_values = np.asarray(pairs, dtype=float)
        for index in range(len(x) - 1):
            opacity = 0.22 if min(pair_values[index:index + 2]) < thin_below else 0.9
            ax.plot(x[index:index + 2], values[index:index + 2], color=colour,
                    linewidth=theme.stroke("emphasis"), alpha=opacity)
        if len(x):
            ax.plot([], [], color=colour, linewidth=theme.stroke("emphasis"),
                    label=label or None)
    ax.axvline(0, color=theme.colour("ink"), linewidth=theme.stroke("guide"))
    ax.axhline(0, color=theme.colour("reference"), linewidth=theme.stroke("guide"))
    suffix = " (negative: first series leads; positive: second leads)" if direction_note else ""
    ax.set_xlabel(x_label + suffix)
    ax.set_ylabel(y_label)
    return PanelResult(data=result, axes=ax)


def phase_difference(
    ax: Any,
    distance: Sequence[float],
    difference: Sequence[float],
    theme: Any,
    *,
    period: float = 24.0,
    null_level: float | None = None,
    gridsize: int = 24,
    binned_mean: bool = True,
    x_label: str = "Distance between cells",
    y_label: str = "Peak time difference",
) -> PanelResult:
    """Circular phase difference against spatial separation."""
    distance = np.asarray(distance, dtype=float)
    difference = np.asarray(difference, dtype=float)
    usable = np.isfinite(distance) & np.isfinite(difference)
    distance, difference = distance[usable], difference[usable]
    density = common.hexbin(
        ax, distance, difference, theme, gridsize=gridsize,
        binned_mean=binned_mean, x_label=x_label, y_label=y_label,
    )
    if null_level is not None:
        ax.axhline(float(null_level), color=theme.colour("reference"),
                   linewidth=theme.stroke("emphasis"), linestyle=(0, (5, 3)),
                   label="random-pair level")
    ax.set_ylim(0, period / 2)
    table = pd.DataFrame({"distance": distance, "phase_difference": difference})
    table.attrs["binned_mean"] = density.data
    return PanelResult(data=table, axes=ax,
                       extra={"binned_mean": density.data,
                              "handle": density.extra["handle"]})


def similarity_matrix(
    ax: Any,
    values: Any,
    theme: Any,
    *,
    labels: Sequence[str] | None = None,
    order: Sequence[int] | None = None,
    cmap: Any = None,
    diagonal_mask: bool = True,
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """A cell-by-cell similarity heatmap, in whatever order the caller chose."""
    data = np.asarray(values, dtype=float)
    names = list(labels or [str(index + 1) for index in range(data.shape[0])])
    if order is not None:
        indices = np.asarray(order, dtype=int)
        data = data[np.ix_(indices, indices)]
        names = [names[index] for index in indices]
    return common.matrix(
        ax, data, theme, row_labels=names, column_labels=names, cmap=cmap,
        diagonal_mask=diagonal_mask, x_label=x_label, y_label=y_label, annotate=False,
    )


def recurrence(
    ax: Any,
    distances: Any,
    hours: Sequence[float],
    theme: Any,
    *,
    threshold: float | None = None,
    cmap: Any = None,
    hour_ticks: float | None = None,
) -> PanelResult:
    """Frame-by-frame state distance and its data-derived recurrence contour."""
    data = np.asarray(distances, dtype=float)
    finite = data[np.isfinite(data)]
    used_threshold = float(np.quantile(finite, 0.1)) if threshold is None and finite.size else float(threshold or 0)
    drawn = common.surface(
        ax, data, theme, x=hours, y=hours, cmap=cmap,
        contour_at=used_threshold, x_label="Hours", y_label="Hours",
    )
    hours = np.asarray(hours, dtype=float)
    ticks = theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    return PanelResult(data=drawn.data, axes=ax,
                       extra={"handle": drawn.extra["handle"],
                              "threshold": used_threshold})


def contact_chord(
    ax: Any,
    weights: Any,
    theme: Any,
    *,
    labels: Sequence[str],
    threshold: float = 0.0,
    flagged: Any = None,
) -> PanelResult:
    """Hours in contact as arc width; flagged handoff pairs are marked red."""
    drawn = common.chord(ax, weights, theme, labels=labels, threshold=threshold,
                         role="surveillance")
    if flagged is not None:
        flagged_values = np.asarray(flagged, dtype=bool)
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        for left in range(n):
            for right in range(left + 1, n):
                if flagged_values[left, right]:
                    midpoint = (points[left] + points[right]) / 2
                    ax.scatter([midpoint[0]], [midpoint[1]], marker="x",
                               color=theme.colour("highlight"), s=theme.point_area(1.1))
    return PanelResult(data=drawn.data, axes=ax,
                       extra={"weights": np.asarray(weights, dtype=float)})


def phase_map(
    ax: Any,
    positions: Any,
    phases: Sequence[float],
    theme: Any,
    *,
    field: dict,
    period_hours: float = 24.0,
    cmap: Any = None,
    footprints: Any = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Cell positions coloured by peak time through a cyclic colour map."""
    import matplotlib

    if cmap is None:
        chosen = matplotlib.colormaps["twilight_shifted"]
    elif isinstance(cmap, str):
        if cmap not in {"twilight", "twilight_shifted", "hsv"}:
            raise ValueError(f"phase_map needs a cyclic colour map, not {cmap!r}")
        chosen = matplotlib.colormaps[cmap]
    else:
        name = getattr(cmap, "name", "")
        if name not in {"twilight", "twilight_shifted", "hsv"}:
            raise ValueError(f"phase_map needs a cyclic colour map, not {name or 'an unnamed map'}")
        chosen = cmap
    points = np.asarray(positions, dtype=float)
    phases = np.mod(np.asarray(phases, dtype=float), period_hours)
    handle = ax.scatter(points[:, 0], points[:, 1], c=phases, cmap=chosen,
                        vmin=0, vmax=period_hours, s=theme.point_area(1.8),
                        edgecolor=theme.colour("page"), linewidth=theme.stroke("hairline"))
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return PanelResult(
        data=pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "phase": phases}),
        axes=ax, extra={"handle": handle},
    )
