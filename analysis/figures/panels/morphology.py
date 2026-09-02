"""Panels whose colour and geometry carry morphology-specific meaning."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult


REGIME_ROLES: dict[int, str] = {
    0: "morphology",
    1: "reporter",
    2: "surveillance",
    3: "motility",
    4: "evidence",
    5: "highlight",
    6: "rhythmic",
    7: "stable",
}

__all__ = [
    "REGIME_ROLES", "regime_ribbon", "regime_transitions", "dwell_times",
    "occupancy_area", "sholl_kymograph", "sholl_profile",
]


def _role(regime: int) -> str:
    return REGIME_ROLES[int(regime) % len(REGIME_ROLES)]


def regime_ribbon(
    ax: Any,
    matrix: Any,
    hours: Sequence[float],
    theme: Any,
    *,
    order: Sequence[int] | None = None,
    margins: Any = None,
    hour_ticks: float | None = None,
    legend: bool = True,
    regime_labels: dict[int, str] | None = None,
    y_label: str = "Cell",
    x_label: str = "Hours from start of recording",
) -> PanelResult:
    """One row per cell, with uncertain assignments washed out."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    values = np.asarray(matrix, dtype=float)
    if order is not None:
        values = values[np.asarray(order, dtype=int)]
        if margins is not None:
            margins = np.asarray(margins, dtype=float)[np.asarray(order, dtype=int)]
    confidence = np.ones_like(values) if margins is None else np.clip(np.asarray(margins, dtype=float), 0, 1)
    rgba = np.zeros((*values.shape, 4), dtype=float)
    for regime in sorted(int(value) for value in np.unique(values[np.isfinite(values)])):
        colour = np.asarray(to_rgba(theme.colour(_role(regime))))
        selected = values == regime
        rgba[selected, :3] = colour[:3]
        rgba[selected, 3] = 0.18 + 0.82 * confidence[selected]
    missing = ~np.isfinite(values)
    rgba[missing] = to_rgba(theme.colour("missing"))
    hours = np.asarray(hours, dtype=float)
    handle = ax.imshow(
        rgba, aspect="auto", interpolation="nearest",
        extent=(float(hours.min()), float(hours.max()), values.shape[0] - 0.5, -0.5),
    )
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        regimes = sorted(int(value) for value in np.unique(values[np.isfinite(values)]))
        names = regime_labels or {}
        ax.legend(handles=[Patch(color=theme.colour(_role(regime)),
                                 label=names.get(regime, f"State {regime}"))
                           for regime in regimes], frameon=False, ncol=max(1, len(regimes)))
    rows, columns = np.indices(values.shape)
    return PanelResult(
        data=pd.DataFrame({
            "row": rows.ravel(),
            "hours": hours[np.clip(columns.ravel(), 0, len(hours) - 1)],
            "regime": values.ravel(),
            "confidence": confidence.ravel(),
        }),
        axes=ax, extra={"handle": handle},
    )


def _normalise(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "none":
        return values
    axis = 1 if mode == "row" else 0 if mode == "column" else None
    if axis is None:
        raise ValueError("normalise must be row, column or none")
    totals = values.sum(axis=axis, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)


def regime_transitions(
    ax: Any,
    counts: Any,
    theme: Any,
    *,
    labels: Sequence[str],
    column_labels: Sequence[str] | None = None,
    null: Any = None,
    normalise: str = "row",
    diagonal_mask: bool = False,
) -> PanelResult:
    """Observed transition matrix and, when supplied, its shuffled-time null."""
    observed = _normalise(np.asarray(counts, dtype=float), normalise)
    shown_columns = list(labels if column_labels is None else column_labels)
    common.matrix(ax, observed, theme, row_labels=labels, column_labels=shown_columns,
                  cmap=theme["sequential_cmap"], vmin=0,
                  vmax=float(np.nanmax(observed)) if observed.size else 1,
                  x_label="To state", y_label="From state",
                  diagonal_mask=diagonal_mask)
    null_values = None
    if null is not None:
        null_values = _normalise(np.asarray(null, dtype=float), normalise)
        inset = ax.inset_axes([1.12, 0.0, 1.0, 1.0])
        common.matrix(inset, null_values, theme, row_labels=labels, column_labels=shown_columns,
                      cmap=theme["sequential_cmap"], vmin=0,
                      vmax=float(np.nanmax(observed)) if observed.size else 1,
                      x_label="Shuffled to", y_label="Shuffled from")
    rows = []
    for source in range(observed.shape[0]):
        for target in range(observed.shape[1]):
            rows.append({
                "from_regime": source,
                "to_regime": target,
                "value": float(observed[source, target]),
                "null_value": (float(null_values[source, target]) if null_values is not None else np.nan),
            })
    return PanelResult(data=pd.DataFrame(rows), axes=ax)


def dwell_times(
    ax: Any,
    runs: Sequence[float],
    theme: Any,
    *,
    regime_ids: Sequence[int],
    minutes_per_frame: float,
    bins: int | Sequence[float] | None = None,
    x_label: str = "",
    regime_labels: dict[int, str] | None = None,
) -> PanelResult:
    """Quantised dwell distributions, one stepped outline per regime."""
    if isinstance(runs, pd.DataFrame):
        values = runs["dwell_frames"].to_numpy(float)
        states = runs["regime" if "regime" in runs else "from_regime"].to_numpy(int)
    elif isinstance(runs, dict):
        pieces = [(int(regime), np.asarray(group, dtype=float)) for regime, group in runs.items()]
        values = np.concatenate([group for _, group in pieces]) if pieces else np.array([])
        states = np.concatenate([np.full(len(group), regime) for regime, group in pieces]) if pieces else np.array([], dtype=int)
        regime_ids = [regime for regime, _ in pieces]
    else:
        array = np.asarray(runs, dtype=float)
        if array.ndim == 2 and array.shape[0] == len(regime_ids):
            values = array.ravel()
            states = np.repeat(np.asarray(regime_ids, dtype=int), array.shape[1])
        else:
            values = array.ravel()
            supplied = np.asarray(regime_ids, dtype=int)
            states = supplied if len(supplied) == len(values) else np.full(len(values), int(supplied[0]))
            regime_ids = sorted(np.unique(states))
    maximum = int(np.nanmax(values)) if len(values) and np.isfinite(values).any() else 1
    edges = np.asarray(bins, dtype=float) if bins is not None and not isinstance(bins, int) else np.arange(0.5, maximum + 1.6)
    rows = []
    for regime in regime_ids:
        selected = values[states == int(regime)]
        counts, used = np.histogram(selected[np.isfinite(selected)], bins=edges)
        centres = (used[:-1] + used[1:]) / 2
        names = regime_labels or {}
        ax.step(centres, counts, where="mid", color=theme.colour(_role(int(regime))),
                linewidth=theme.stroke("emphasis"),
                label=names.get(int(regime), f"State {int(regime)}"))
        rows.extend({"regime": int(regime), "bin_left_frames": left,
                     "bin_right_frames": right, "runs": int(count)}
                    for left, right, count in zip(used[:-1], used[1:], counts))
    ax.set_xlabel(x_label or "Dwell (frames; first bin is at least one frame)")
    ax.set_ylabel("Runs")
    secondary = ax.secondary_xaxis(
        "top",
        functions=(lambda frames: frames * minutes_per_frame / 60.0,
                   lambda hours: hours * 60.0 / minutes_per_frame),
    )
    secondary.set_xlabel("Dwell (hours)")
    return PanelResult(data=pd.DataFrame(rows), axes=ax)


def occupancy_area(
    ax: Any,
    hours: Sequence[float],
    fractions: Any,
    theme: Any,
    *,
    regime_ids: Sequence[int],
    hour_ticks: float | None = None,
    wrap_hours: float | None = None,
    regime_labels: dict[int, str] | None = None,
) -> PanelResult:
    """Population regime composition over time as a normalised stack."""
    data = np.asarray(fractions, dtype=float)
    frame = {f"regime_{regime}": data[:, index] for index, regime in enumerate(regime_ids)}
    looks = {f"regime_{regime}": common.Look(colour=theme.colour(_role(int(regime))))
             for regime in regime_ids}
    names = regime_labels or {}
    drawn = common.stacked_area(
        ax, hours, list(frame), frame, theme, looks=looks,
        labels=[names.get(int(regime), f"State {regime}") for regime in regime_ids], normalise=True,
        x_label="Hours from start of recording", y_label="Population share",
    )
    hours = np.asarray(hours, dtype=float)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    return PanelResult(data=drawn.data, axes=ax)


def sholl_kymograph(
    ax: Any,
    occupancy: Any,
    hours: Sequence[float],
    radii: Sequence[float],
    theme: Any,
    *,
    reach: Sequence[float] | None = None,
    cmap: Any = None,
    contour_at: float = 0.5,
    hour_ticks: float | None = None,
    y_label: str = "Distance from soma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> PanelResult:
    """Radial occupancy surface with the 95th-percentile reach overlaid."""
    drawn = common.surface(
        ax, occupancy, theme, x=hours, y=radii, cmap=cmap,
        contour_at=contour_at, x_label="Hours from start of recording", y_label=y_label,
        vmin=vmin, vmax=vmax,
    )
    if reach is not None:
        ax.plot(hours, reach, color=theme.colour("ink"), linewidth=theme.stroke("emphasis"),
                label="Radius containing 95% of occupied pixels")
    hours = np.asarray(hours, dtype=float)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    return PanelResult(data=drawn.data, axes=ax,
                       extra={"handle": drawn.extra["handle"]})


def sholl_profile(
    ax: Any,
    radii: Sequence[float],
    occupancy: Sequence[float],
    theme: Any,
    *,
    look: common.Look | None = None,
    marks: Sequence[common.Mark] = (),
    x_label: str = "Distance from soma",
    y_label: str = "Annulus occupancy",
    label: str = "",
) -> PanelResult:
    """Occupancy against distance from the soma, for one moment or one cell."""
    values = np.asarray(occupancy, dtype=float)
    radii = np.asarray(radii, dtype=float)
    colour = (look or common.Look(colour=theme.colour("morphology"))).colour
    ax.plot(radii, values, color=colour, linewidth=theme.stroke("emphasis"),
            label=label or None)
    if marks:
        common.reference_lines(ax, marks, theme)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return PanelResult(
        data=pd.DataFrame({"radius": radii, "occupancy": values}), axes=ax)
