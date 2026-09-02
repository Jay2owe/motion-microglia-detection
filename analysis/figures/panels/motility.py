"""Panels for what the ``motility`` module measures.

Both panels here exist because of one rule that a general chart would break: a
cell that was missing for three frames must not be drawn as though it travelled
in a straight line while nobody was looking. On the map that means a path is cut
at every gap; in the histogram it means a step measured across a gap is not a
step. Put a bridged gap on either panel and it reads as evidence of travel the
movie never showed.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult

__all__ = [
    "path_segments", "trajectory_map", "step_histogram", "wander_against_size",
    "msd_curves",
]


def path_segments(frames: Sequence[int], points: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """One cell's path as drawable pieces, cut wherever a frame is missing.

    Returns a list of (n, 2, 2) arrays - each a run of consecutive frames - so a
    two-frame absence leaves a hole rather than a long stroke across it.
    """
    frames = np.asarray(frames, dtype=int)
    points = np.asarray(points, dtype=float)
    cuts = np.flatnonzero(np.diff(frames) > 1) + 1
    pieces = []
    for piece in np.split(points, cuts):
        if len(piece) < 2:
            continue
        pieces.append(np.stack([piece[:-1], piece[1:]], axis=1).reshape(-1, 2, 2))
    return pieces


def trajectory_map(
    ax: Any,
    groups: Iterable[tuple[Sequence[int], Sequence[Sequence[float]], float]],
    theme: Any,
    *,
    field: dict,
    look: common.Look | None = None,
    vmax: float | None = None,
    show_starts: bool = True,
    x_label: str = "X position in the field (px)",
    y_label: str = "Y position in the field (px)",
) -> PanelResult:
    """Every cell's centroid path, drawn in the movie's own coordinates.

    ``groups`` is (frame numbers, centroids, the value that colours this cell).
    One row per drawn segment end, so the table holds the path as it was cut;
    ``extra`` carries the line collection - hand it to :func:`common.colour_bar`
    - and how many gaps the cutting left.

    The axes are bounded by the field rather than by the data. A map cropped to
    wherever the cells happen to be changes magnification between runs, and then
    two maps get compared at different scales without anybody being told.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    look = look or common.Look(cmap=theme["sequential_cmap"])
    blocks: list[np.ndarray] = []
    values: list[float] = []
    starts: list[tuple[float, float, float]] = []
    breaks = 0

    rows: list[pd.DataFrame] = []
    for index, (frames, points, value) in enumerate(groups):
        points = np.asarray(points, dtype=float)
        if not len(points):
            continue
        starts.append((points[0][0], points[0][1], float(value)))
        pieces = path_segments(frames, points)
        breaks += max(len(pieces) - 1, 0)
        for run, piece in enumerate(pieces):
            blocks.append(piece)
            values.append(float(value))
            rows.append(pd.DataFrame({
                "path": index, "run": run,
                "x": piece[:, 0, 0], "y": piece[:, 0, 1], "value": float(value),
            }))

    lines = np.concatenate(blocks) if blocks else np.empty((0, 2, 2))
    colours = (
        np.concatenate([np.full(len(block), value) for block, value in zip(blocks, values)])
        if blocks else np.empty(0)
    )
    if vmax is None:
        vmax = float(np.nanmax(colours)) if colours.size else 1.0

    collection = LineCollection(
        lines, array=colours,
        cmap=look.cmap if look.is_map else None,
        color=None if look.is_map else look.colour,
        linewidths=theme.stroke("line") * 0.55,
        norm=Normalize(0, vmax) if look.is_map else None,
    )
    ax.add_collection(collection)

    if show_starts and starts:
        first = np.asarray(starts, dtype=float)
        ax.scatter(
            first[:, 0], first[:, 1],
            c=first[:, 2] if look.is_map else None,
            color=None if look.is_map else look.colour,
            cmap=look.cmap if look.is_map else None,
            vmin=0 if look.is_map else None, vmax=vmax if look.is_map else None,
            s=theme.point_area(0.55), edgecolor="none", zorder=3,
        )

    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)          # image coordinates: y counts downwards
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    drawn = (pd.concat(rows, ignore_index=True) if rows
             else pd.DataFrame(columns=["path", "run", "x", "y", "value"]))
    return PanelResult(data=drawn, axes=ax,
                       extra={"collection": collection, "breaks": breaks})


def step_histogram(
    ax: Any,
    values: Sequence[float],
    theme: Any,
    *,
    bins: int = 45,
    role: str = "motility",
    look: common.Look | None = None,
    log_x: bool = True,
    unit: str = "px",
    x_label: str = "",
    y_label: str = "Cell-frames",
    marks: bool = True,
) -> PanelResult:
    """How far a cell moves in one frame interval, pooled over every cell-frame.

    Log bands by default: the distribution is one dense peak below a pixel with
    a tail three decades long, and even bands put nearly all of it in the first
    bar.

    The only lines drawn are the median and the 95th percentile, both read off
    this distribution. The tracker's own distance tolerance is deliberately not
    drawn - it lives in a configuration this package does not read, and a figure
    that quoted it would keep quoting it after somebody changed it.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)] if log_x else values[np.isfinite(values)]

    bands = common.histogram(
        ax, values, theme, bins=bins, role=role, look=look, log_x=log_x,
        x_label=x_label, y_label=y_label,
    )
    counts, edges = bands.extra["counts"], bands.extra["edges"]
    ax.set_xlim(edges[0], edges[-1])

    if marks and values.size:
        median = float(np.median(values))
        p95 = float(np.percentile(values, 95))
        common.reference_lines(
            ax,
            [
                common.Mark(median, f"median {median:.2f} {unit}", "ink", at=0.94),
                common.Mark(p95, f"95th percentile {p95:.1f} {unit}", "highlight",
                            dashed=True, at=0.80),
            ],
            theme,
        )
    return PanelResult(data=bands.data, axes=ax,
                       extra={"counts": counts, "edges": edges})


def wander_against_size(
    ax: Any,
    hull: Sequence[float],
    footprint: Sequence[float],
    theme: Any,
    *,
    look: common.Look | None = None,
    parity: bool = True,
    log: bool = True,
    labels: Sequence[str] | None = None,
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """Centroid-path hull against the cell's median footprint area."""
    hull = np.asarray(hull, dtype=float)
    footprint = np.asarray(footprint, dtype=float)
    usable = np.isfinite(hull) & np.isfinite(footprint)
    if log:
        usable &= (hull > 0) & (footprint > 0)
    hull, footprint = hull[usable], footprint[usable]
    common.scatter(ax, footprint, hull, theme, role="motility", look=look)
    if parity and len(hull):
        low = min(float(hull.min()), float(footprint.min()))
        high = max(float(hull.max()), float(footprint.max()))
        ax.plot([low, high], [low, high], color=theme.colour("reference"),
                linewidth=theme.stroke("guide"), linestyle=(0, (5, 3)))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if labels is not None:
        kept_labels = np.asarray(labels, dtype=object)[usable]
        for x, y, label in zip(footprint, hull, kept_labels):
            ax.text(x, y, str(label), fontsize=theme.size("caption"), alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return PanelResult(
        data=pd.DataFrame({"footprint": footprint, "hull": hull,
                           "hull_over_area": hull / footprint}),
        axes=ax,
    )


def msd_curves(
    ax: Any,
    curves: pd.DataFrame,
    theme: Any,
    *,
    alpha: Any = None,
    look: common.Look | None = None,
    reference: Sequence[float] = (1.0,),
    thin_below: int | None = None,
    log: bool = True,
    x_label: str = "Lag (hours)",
    y_label: str = "Mean squared displacement",
) -> PanelResult:
    """One gap-respecting mean-squared-displacement curve per identity."""
    data = curves.copy()
    colour = (look or common.Look(colour=theme.colour("motility"))).colour or theme.colour("motility")
    for cell_index, (_, group) in enumerate(data.groupby("identity", sort=True)):
        group = group.sort_values("lag_hours")
        opacity = 0.18
        ax.plot(group["lag_hours"], group["msd"], color=colour,
                linewidth=theme.stroke("line"), alpha=opacity,
                label="Individual cells" if cell_index == 0 else None)
        if thin_below is not None and "pairs" in group:
            strong = group["pairs"].to_numpy(float) >= int(thin_below)
            ax.plot(group.loc[strong, "lag_hours"], group.loc[strong, "msd"],
                    color=colour, linewidth=theme.stroke("line"), alpha=0.70)
    positive = data[(data["lag_hours"] > 0) & (data["msd"] > 0)]
    if not positive.empty:
        anchor_x = float(positive["lag_hours"].median())
        anchor_y = float(positive["msd"].median())
        x_line = np.asarray([positive["lag_hours"].min(), positive["lag_hours"].max()], dtype=float)
        for slope in reference:
            y_line = anchor_y * (x_line / anchor_x) ** float(slope)
            ax.plot(x_line, y_line, color=theme.colour("reference"),
                    linewidth=theme.stroke("guide"), linestyle=(0, (5, 3)),
                    label=("Free diffusion reference (slope 1)" if float(slope) == 1.0
                           else f"Reference slope {float(slope):g}"))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if alpha is not None:
        mapping = alpha if isinstance(alpha, dict) else None
        if mapping:
            data["msd_alpha"] = data["identity"].map(mapping)
    return PanelResult(data=data, axes=ax)
