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
    "path_segments", "trajectory_map", "step_histogram",
    "centroid_path_area_against_footprint", "wander_against_size",
    "msd_curves", "msd_exponent_distribution",
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
    identities: Sequence[Any] | None = None,
    line_widths: Sequence[float] | None = None,
    line_dash_categories: Sequence[str] | None = None,
    line_dash_styles: dict[str, Any] | None = None,
    x_label: str = "X position in the field (px)",
    y_label: str = "Y position in the field (px)",
) -> PanelResult:
    """Every cell's centroid path, drawn in the movie's own coordinates.

    ``groups`` is (frame numbers, centroids, the value that colours this cell).
    The optional identity, width and dash sequences are aligned one-for-one
    with those groups. One row per drawn line segment holds both endpoints and
    all three visual encodings, plus one start row per identity for the dot;
    ``extra`` carries the line collection - hand it to
    :func:`common.colour_bar` - and how many gaps the cutting left.

    The axes are bounded by the field rather than by the data. A map cropped to
    wherever the cells happen to be changes magnification between runs, and then
    two maps get compared at different scales without anybody being told.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    groups = list(groups)
    count = len(groups)

    def aligned(values: Sequence[Any] | None, name: str, default: Any) -> list[Any]:
        if values is None:
            return [default for _ in range(count)]
        values = list(values)
        if len(values) != count:
            raise ValueError(
                f"{name} has {len(values)} values for {count} trajectory groups"
            )
        return values

    base_width = theme.stroke("line") * 0.55
    identity_values = aligned(identities, "identities", None)
    width_values = aligned(line_widths, "line_widths", base_width)
    dash_values = [str(value) for value in aligned(
        line_dash_categories, "line_dash_categories", "All trajectories"
    )]
    styles = dict(line_dash_styles or {"All trajectories": "solid"})
    absent_styles = [value for value in dict.fromkeys(dash_values) if value not in styles]
    if absent_styles:
        raise ValueError(
            "line_dash_styles has no pattern for " + ", ".join(absent_styles)
        )
    for value in width_values:
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("line_widths must be finite positive numbers")

    look = look or common.Look(cmap=theme["sequential_cmap"])
    blocks: list[np.ndarray] = []
    values: list[float] = []
    block_widths: list[float] = []
    block_dashes: list[str] = []
    starts: list[tuple[float, float, float]] = []
    start_rows: list[dict[str, Any]] = []
    breaks = 0

    rows: list[pd.DataFrame] = []
    for index, (frames, points, value) in enumerate(groups):
        points = np.asarray(points, dtype=float)
        if not len(points):
            continue
        identity = index if identity_values[index] is None else identity_values[index]
        width = float(width_values[index])
        dash = dash_values[index]
        starts.append((points[0][0], points[0][1], float(value)))
        start_rows.append({
            "identity": identity, "path": index, "run": -1, "segment": -1,
            "mark_type": "start", "x": points[0][0], "y": points[0][1],
            "x_end": np.nan, "y_end": np.nan, "value": float(value),
            "line_width": np.nan, "line_dash_category": dash,
            "line_dash_pattern": "", "marker_area": theme.point_area(0.55),
        })
        pieces = path_segments(frames, points)
        breaks += max(len(pieces) - 1, 0)
        for run, piece in enumerate(pieces):
            blocks.append(piece)
            values.append(float(value))
            block_widths.append(width)
            block_dashes.append(dash)
            rows.append(pd.DataFrame({
                "identity": identity, "path": index, "run": run,
                "segment": np.arange(len(piece), dtype=int), "mark_type": "segment",
                "x": piece[:, 0, 0], "y": piece[:, 0, 1],
                "x_end": piece[:, 1, 0], "y_end": piece[:, 1, 1],
                "value": float(value), "line_width": width,
                "line_dash_category": dash,
                "line_dash_pattern": repr(styles[dash]),
                "marker_area": np.nan,
            }))

    lines = np.concatenate(blocks) if blocks else np.empty((0, 2, 2))
    colours = (
        np.concatenate([np.full(len(block), value) for block, value in zip(blocks, values)])
        if blocks else np.empty(0)
    )
    widths = (
        np.concatenate([np.full(len(block), value)
                        for block, value in zip(blocks, block_widths)])
        if blocks else np.empty(0)
    )
    dashes = [
        styles[dash]
        for block, dash in zip(blocks, block_dashes)
        for _ in range(len(block))
    ]
    if vmax is None:
        vmax = float(np.nanmax(colours)) if colours.size else 1.0

    collection = LineCollection(
        lines, array=colours,
        cmap=look.cmap if look.is_map else None,
        color=None if look.is_map else look.colour,
        linewidths=widths,
        linestyles=dashes,
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
    parts = [*rows, pd.DataFrame(start_rows)] if start_rows else rows
    drawn = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=[
                 "identity", "path", "run", "segment", "mark_type", "x", "y",
                 "x_end", "y_end", "value", "line_width",
                 "line_dash_category", "line_dash_pattern", "marker_area",
             ]))
    return PanelResult(data=drawn, axes=ax,
                       extra={"collection": collection, "breaks": breaks,
                              "line_dash_styles": styles})


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


def centroid_path_area_against_footprint(
    ax: Any,
    path_area: Sequence[float],
    footprint_area: Sequence[float],
    theme: Any,
    *,
    look: common.Look | None = None,
    parity: bool = True,
    parity_label: str = "Equal areas",
    scale: str = "symlog",
    linear_threshold: float = 1.0,
    labels: Sequence[str] | None = None,
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """Cell-centre path area against typical segmented footprint area.

    ``path_area`` is the area of the convex hull enclosing every observed cell
    centroid; ``footprint_area`` is the median segmented cell area across
    frames. ``symlog`` preserves a zero path area while giving large values a
    logarithmic scale. The equal-area line is a geometric reference, not a
    fitted relationship or a statistical threshold.
    """
    if scale not in ("linear", "log", "symlog"):
        raise ValueError("scale must be linear, log or symlog")
    path_area = np.asarray(path_area, dtype=float)
    footprint_area = np.asarray(footprint_area, dtype=float)
    usable = np.isfinite(path_area) & np.isfinite(footprint_area)
    usable &= (path_area >= 0) & (footprint_area > 0)
    if scale == "log":
        usable &= path_area > 0
    path_area, footprint_area = path_area[usable], footprint_area[usable]
    common.scatter(ax, footprint_area, path_area, theme, role="motility", look=look)
    if parity and len(path_area):
        low = (0.0 if scale == "symlog" else
               min(float(path_area.min()), float(footprint_area.min())))
        high = max(float(path_area.max()), float(footprint_area.max()))
        ax.plot([low, high], [low, high], color=theme.colour("reference"),
                linewidth=theme.stroke("guide"), linestyle=(0, (5, 3)),
                label=parity_label or None)
    if scale == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif scale == "symlog":
        ax.set_xscale("symlog", linthresh=float(linear_threshold))
        ax.set_yscale("symlog", linthresh=float(linear_threshold))
    if labels is not None:
        kept_labels = np.asarray(labels, dtype=object)[usable]
        for x, y, label in zip(footprint_area, path_area, kept_labels):
            ax.text(x, y, str(label), fontsize=theme.size("caption"), alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return PanelResult(
        data=pd.DataFrame({
            "footprint_area": footprint_area,
            "centroid_path_area": path_area,
            "path_area_over_footprint": path_area / footprint_area,
            "axis_scale": scale,
        }),
        axes=ax,
    )


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
    """Compatibility name for :func:`centroid_path_area_against_footprint`."""
    return centroid_path_area_against_footprint(
        ax, hull, footprint, theme, look=look, parity=parity,
        scale="log" if log else "linear", labels=labels,
        x_label=x_label, y_label=y_label,
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
    summary: bool = True,
    x_label: str = "Time between compared frames (hours)",
    y_label: str = "Average squared centroid shift",
) -> PanelResult:
    """One mean-squared-centroid-displacement curve per identity.

    Every value is the average squared distance between observed centroid
    positions exactly that many hours apart. Missing frames are not joined or
    interpolated. Multiplicative axes retain the range needed for fitting but
    show ordinary numbers instead of powers-of-ten notation.
    """
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
    aggregate = pd.DataFrame()
    if summary and not positive.empty:
        aggregate = positive.groupby("lag_hours", as_index=False).agg(
            cells_at_lag=("identity", "nunique"),
            population_median_msd=("msd", "median"),
            population_q25_msd=("msd", lambda values: values.quantile(0.25)),
            population_q75_msd=("msd", lambda values: values.quantile(0.75)),
        )
        x = aggregate["lag_hours"].to_numpy(float)
        median = aggregate["population_median_msd"].to_numpy(float)
        q25 = aggregate["population_q25_msd"].to_numpy(float)
        q75 = aggregate["population_q75_msd"].to_numpy(float)
        ax.fill_between(
            x, q25, q75, color=colour, alpha=0.18, linewidth=0,
            label="Middle 50% of cells",
        )
        ax.plot(
            x, median, color=colour, linewidth=theme.stroke("emphasis"),
            label="Median across cells",
        )
        data = data.merge(aggregate, on="lag_hours", how="left")
    if not positive.empty:
        anchor_x = float(positive["lag_hours"].median())
        anchor_y = float(positive["msd"].median())
        x_line = np.asarray([positive["lag_hours"].min(), positive["lag_hours"].max()], dtype=float)
        for slope in reference:
            y_line = anchor_y * (x_line / anchor_x) ** float(slope)
            ax.plot(x_line, y_line, color=theme.colour("reference"),
                    linewidth=theme.stroke("guide"), linestyle=(0, (5, 3)),
                    label=("Random-walk-like growth (α = 1)" if float(slope) == 1.0
                           else f"Reference growth exponent {float(slope):g}"))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        if not positive.empty:
            x_ticks = _power_ticks(
                float(positive["lag_hours"].min()),
                float(positive["lag_hours"].max()), base=2.0,
            )
            y_ticks = _power_ticks(
                float(positive["msd"].min()),
                float(positive["msd"].max()), base=10.0,
            )
            ax.set_xlim(x_ticks[0], x_ticks[-1])
            ax.set_ylim(y_ticks[0], y_ticks[-1])
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([_ordinary_number(value) for value in x_ticks])
            ax.set_yticklabels([_ordinary_number(value) for value in y_ticks])
            ax.minorticks_off()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if alpha is not None:
        mapping = alpha if isinstance(alpha, dict) else None
        if mapping:
            data["msd_alpha"] = data["identity"].map(mapping)
    common.semantic_legend(ax, theme, location="above")
    return PanelResult(data=data, axes=ax, extra={"summary": aggregate})


def msd_exponent_distribution(
    ax: Any,
    values: Sequence[float],
    theme: Any,
    *,
    bins: int | Sequence[float] = 18,
    x_label: str = (
        "Growth exponent fitted from the upper curve, α\n"
        "<1 slower than random walk | 1 random walk | >1 faster"
    ),
    y_label: str = "Cells",
) -> PanelResult:
    """Distribution of the exponent fitted from each cell's displacement curve."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    drawn = common.histogram(
        ax, finite, theme, bins=bins, role="motility",
        x_label=x_label, y_label=y_label,
    )
    median = float(np.median(finite)) if finite.size else float("nan")
    if np.isfinite(median):
        ax.axvline(
            median, color=theme.colour("ink"),
            linewidth=theme.stroke("emphasis"),
            label=f"Median α = {median:.2f}",
        )
    ax.axvline(
        1.0, color=theme.colour("reference"),
        linewidth=theme.stroke("guide"), linestyle=(0, (5, 3)),
        label="Random-walk reference (α = 1)",
    )
    theme.legend(ax, location="above", columns=2, pad=0.055)
    table = drawn.data.copy()
    table["cells_total"] = int(finite.size)
    table["median_alpha"] = median
    table["random_walk_reference"] = 1.0
    return PanelResult(
        data=table, axes=ax,
        extra={"median": median, "random_walk_reference": 1.0},
    )


def _power_ticks(low: float, high: float, *, base: float,
                 maximum: int = 7) -> np.ndarray:
    """Multiplicative ticks bracketing a positive range, including both ends."""
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low <= high):
        raise ValueError("multiplicative tick limits must be finite and positive")
    start = int(np.floor(np.log(low) / np.log(base)))
    stop = int(np.ceil(np.log(high) / np.log(base)))
    step = max(1, int(np.ceil((stop - start) / max(maximum - 1, 1))))
    exponents = list(range(start, stop + 1, step))
    if exponents[-1] != stop:
        exponents.append(stop)
    return np.asarray([base ** exponent for exponent in exponents], dtype=float)


def _ordinary_number(value: float) -> str:
    """A real-value tick label, never scientific powers-of-ten notation."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:.3g}"
