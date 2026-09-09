"""Panels for per-pixel occupancy history in image coordinates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult


FATE_ROLES: dict[str, str] = {
    "core": "stable",
    "fringe": "motility",
    "transient": "surveillance",
    "vacated": "missing",
    "transferred": "highlight",
}

FATE_LABELS: dict[str, str] = {
    "core": "Core: high occupancy",
    "fringe": "Fringe: intermediate occupancy",
    "transient": "Transient: low occupancy",
    "vacated": "Vacant in this event window",
    "transferred": "Transferred to another cell",
}

MAP_ASSIGNMENT_LABELS = {
    "occupancy_weighted_mean": "average weighted by occupancy time",
    "equal_mean": "equal average across occupants",
    "first": "first occupant",
    "last": "last occupant",
    "most_frequent": "most frequent occupant",
}


__all__ = [
    "FATE_ROLES", "FATE_LABELS", "revisit_map", "revisit_grid",
    "coverage_curve", "ownership_map",
    "MAP_ASSIGNMENT_LABELS", "metric_assignment", "owner_assignment",
    "spatial_value_map", "cell_metric_map",
    "cell_metric_maps", "owner_count_map", "direct_pixel_handoffs",
    "direct_pixel_handoff_map", "first_coverage_time", "first_coverage_time_map",
    "cumulative_occupancy", "cumulative_occupancy_map",
    "split_event_areas", "split_event_map",
    "coverage_history", "coverage_composition",
    "never_visited", "never_visited_mark", "pixel_fate_states",
    "fate_transition_table", "fate_flow",
]


def _extent(field: dict) -> tuple[float, float, float, float]:
    return (0, float(field["width"]), float(field["height"]), 0)


def revisit_map(
    ax: Any,
    counts: Any,
    theme: Any,
    *,
    field: dict,
    cmap: Any = None,
    contour_at: float | None = None,
    soma: tuple[float, float] | None = None,
    look: common.Look | None = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> PanelResult:
    """Number of occupied frames per pixel, at fixed field magnification."""
    values = np.asarray(counts, dtype=float)
    chosen = cmap or (look.cmap if look is not None and look.is_map else theme["sequential_cmap"])
    handle = ax.imshow(values, extent=_extent(field), cmap=chosen, interpolation="nearest",
                       vmin=vmin, vmax=vmax)
    if contour_at is not None:
        ax.contour(values, levels=[float(contour_at)], colors=[theme.colour("ink")],
                   linewidths=theme.stroke("guide"), origin="upper",
                   extent=_extent(field))
    if soma is not None:
        ax.scatter([soma[0]], [soma[1]], marker="+", color=theme.colour("highlight"),
                   s=theme.point_area(1.5), linewidth=theme.stroke("emphasis"))
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    rows, columns = np.indices(values.shape)
    return PanelResult(
        data=pd.DataFrame({"row": rows.ravel(), "column": columns.ravel(),
                           "frames": values.ravel()}),
        axes=ax, extra={"handle": handle},
    )


def _square_crop(values: np.ndarray, centre: tuple[float, float], side: int) -> tuple[np.ndarray, int, int]:
    """Return a padded square crop plus its source origin."""
    row0 = int(np.floor(float(centre[0]) - side / 2.0))
    column0 = int(np.floor(float(centre[1]) - side / 2.0))
    row1, column1 = row0 + side, column0 + side
    source_row0, source_column0 = max(row0, 0), max(column0, 0)
    source_row1, source_column1 = min(row1, values.shape[0]), min(column1, values.shape[1])
    cropped = np.zeros((side, side), dtype=float)
    cropped[
        source_row0 - row0:source_row1 - row0,
        source_column0 - column0:source_column1 - column0,
    ] = values[source_row0:source_row1, source_column0:source_column1]
    return cropped, row0, column0


def revisit_grid(
    figure: Any,
    rect: tuple[float, float, float, float],
    count_maps: Mapping[Any, Any],
    theme: Any,
    *,
    observed_frames: Mapping[Any, int],
    soma: Mapping[Any, tuple[float, float]] | None = None,
    contour_fraction: float = 0.5,
    max_columns: int = 8,
    padding_fraction: float = 0.15,
    gap_x: float = 0.0022,
    gap_y: float = 0.028,
    heading: str = "",
    unit_label: str = "Cell",
) -> PanelResult:
    """Small multiples of unit occupancy frequency at one shared spatial scale."""
    from matplotlib.lines import Line2D

    if not count_maps:
        raise ValueError("revisit_grid needs at least one occupancy map")
    if not 0.0 <= float(contour_fraction) <= 1.0:
        raise ValueError("contour_fraction must be between zero and one")
    if int(max_columns) < 1:
        raise ValueError("max_columns must be at least one")

    prepared = []
    largest_span = 1
    for unit, counts in count_maps.items():
        values = np.asarray(counts, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"occupancy map for {unit!r} must be two-dimensional")
        frames = int(observed_frames[unit])
        if frames <= 0:
            raise ValueError(f"observed_frames for {unit!r} must be positive")
        occupied = np.argwhere(values > 0)
        if not occupied.size:
            continue
        row_min, column_min = occupied.min(axis=0)
        row_max, column_max = occupied.max(axis=0)
        span = max(int(row_max - row_min + 1), int(column_max - column_min + 1))
        largest_span = max(largest_span, span)
        prepared.append((unit, values, frames, occupied))
    if not prepared:
        raise ValueError("revisit_grid has no occupied pixels")

    crop_side = max(3, int(np.ceil(largest_span * (1.0 + 2.0 * float(padding_fraction)))))
    columns = min(int(max_columns), len(prepared))
    rows = int(np.ceil(len(prepared) / columns))
    left, bottom, width, height = rect
    tile_width = (width - gap_x * (columns - 1)) / columns
    tile_height = (height - gap_y * (rows - 1)) / rows
    if heading:
        figure.text(
            left, bottom + height + 0.014, heading,
            ha="left", va="bottom", fontsize=theme.size("subtitle"), fontweight="bold",
        )

    axes = []
    summary_rows = []
    map_handle = None
    for position, (unit, values, frames, occupied) in enumerate(prepared):
        row = position // columns
        column = position % columns
        x = left + column * (tile_width + gap_x)
        y = bottom + (rows - row - 1) * (tile_height + gap_y)
        ax = figure.add_axes([x, y, tile_width, tile_height])
        centre = tuple(occupied.mean(axis=0))
        crop, row0, column0 = _square_crop(values / frames, centre, crop_side)
        cell_soma = None
        if soma is not None and unit in soma:
            cell_soma = (float(soma[unit][0]) - column0, float(soma[unit][1]) - row0)
        drawn = revisit_map(
            ax, crop, theme,
            field={"width": crop_side, "height": crop_side},
            contour_at=float(contour_fraction), soma=cell_soma,
            x_label="", y_label="", vmin=0.0, vmax=1.0,
        )
        map_handle = drawn.extra["handle"]
        ax.set_title(
            f"{unit_label} {unit}", fontsize=theme.size("annotation"),
            fontweight="bold", pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes.append(ax)
        summary_rows.append(
            {
                "unit": unit,
                "observed_frames": frames,
                "union_px": int(np.count_nonzero(values)),
                "crop_row_origin": row0,
                "crop_column_origin": column0,
                "crop_size_px": crop_side,
                "maximum_occupancy_fraction": float(np.nanmax(values / frames)),
                "contour_fraction": float(contour_fraction),
            }
        )

    common.inset_colour_bar(
        axes[-1], map_handle, theme,
        label="Occupancy frequency within observed frames",
        ticks=[0.0, float(contour_fraction), 1.0],
    )
    axes[-1].legend(
        handles=[
            Line2D([0], [0], color=theme.colour("ink"),
                   linewidth=theme.stroke("guide"), label=f"≥{contour_fraction:.0%} occupancy contour"),
            Line2D([0], [0], marker="+", linestyle="none",
                   color=theme.colour("highlight"), markersize=5, label="Median soma position"),
        ],
        loc="upper left", bbox_to_anchor=(1.14, 1.0), frameon=False,
        fontsize=theme.size("caption"), handlelength=1.5,
    )
    return PanelResult(
        data=pd.DataFrame(summary_rows), axes=axes,
        extra={"handle": map_handle, "crop_size_px": crop_side},
    )


def coverage_curve(
    ax: Any,
    hours: Sequence[float],
    cumulative: Any,
    theme: Any,
    *,
    denominator: Any = None,
    permutation_median: Any = None,
    permutation_lo: Any = None,
    permutation_hi: Any = None,
    test_p_value: float | None = None,
    shuffles: int | None = None,
    look: common.Look | None = None,
    hour_ticks: float | None = None,
    y_label: str = "Share of final footprint touched",
) -> PanelResult:
    """Cumulative unique pixels divided by each cell's own union footprint."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(cumulative, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(hours) and values.shape[0] == len(hours):
        values = values.T
    if denominator is None:
        denominators = np.nanmax(values, axis=1)
    else:
        denominators = np.broadcast_to(np.asarray(denominator, dtype=float), (values.shape[0],))
    fractions = np.divide(values, denominators[:, None], out=np.full_like(values, np.nan),
                          where=denominators[:, None] > 0)
    colour = (look or common.Look(colour=theme.colour("surveillance"))).colour
    null_median = None
    null_lo = None
    null_hi = None
    if permutation_median is not None:
        null_median = np.asarray(permutation_median, dtype=float)
        if null_median.shape != hours.shape:
            raise ValueError("permutation_median must have one value per time point")
        if permutation_lo is not None and permutation_hi is not None:
            null_lo = np.asarray(permutation_lo, dtype=float)
            null_hi = np.asarray(permutation_hi, dtype=float)
            if null_lo.shape != hours.shape or null_hi.shape != hours.shape:
                raise ValueError("permutation interval must have one value per time point")
            ax.fill_between(
                hours, null_lo, null_hi, color=theme.colour("reference"), alpha=0.22,
                linewidth=0, label="95% frame-order permutation interval",
            )
        ax.plot(
            hours, null_median, color=theme.colour("reference"),
            linewidth=theme.stroke("guide"), label="Frame-order permutation median",
        )
    for curve in fractions:
        ax.plot(hours, curve, color=colour, alpha=0.18, linewidth=theme.stroke("guide"))
    ax.plot(hours, np.nanmedian(fractions, axis=0), color=colour,
            linewidth=theme.stroke("emphasis"), label="Median observed cell")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(y_label)
    if test_p_value is not None:
        permutation_text = "" if shuffles is None else f"; {int(shuffles):,} permutations"
        ax.text(
            0.018, 0.965,
            f"Two-sided frame-order permutation test\np = {test_p_value:.3g}{permutation_text}",
            transform=ax.transAxes, ha="left", va="top", fontsize=theme.size("caption"),
            color=theme.colour("ink"),
        )
    data = pd.DataFrame({
        "identity_row": np.repeat(np.arange(values.shape[0]), len(hours)),
        "hours": np.tile(hours, values.shape[0]),
        "cumulative": values.ravel(),
        "coverage_fraction": fractions.ravel(),
    })
    data["permutation_median"] = np.tile(
        null_median if null_median is not None else np.full(len(hours), np.nan),
        values.shape[0],
    )
    data["permutation_lo"] = np.tile(
        null_lo if null_lo is not None else np.full(len(hours), np.nan), values.shape[0]
    )
    data["permutation_hi"] = np.tile(
        null_hi if null_hi is not None else np.full(len(hours), np.nan), values.shape[0]
    )
    return PanelResult(
        data=data,
        axes=ax,
    )


def _coverage_encoding(
    occupancy_sources: Mapping[str, Any],
    theme: Any,
    *,
    source_labels: Mapping[str, str] | None = None,
    class_colours: Mapping[tuple[str, ...], str] | None = None,
) -> dict[str, Any]:
    """Cumulative per-pixel source combinations for generic coverage panels."""
    if not isinstance(occupancy_sources, Mapping) or not occupancy_sources:
        raise ValueError("coverage needs at least one named occupancy source")
    names = [str(name) for name in occupancy_sources]
    if len(set(names)) != len(names):
        raise ValueError("coverage source names must stay unique when converted to text")
    if len(names) > 8:
        raise ValueError("coverage supports at most eight occupancy sources")

    arrays: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for name, values in zip(names, occupancy_sources.values()):
        array = np.asarray(values, dtype=bool)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(
                f"coverage source {name!r} must be a 2D mask or a frame-by-row-by-column stack"
            )
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"coverage sources must share one shape; {name!r} is {array.shape}, expected {shape}"
            )
        arrays.append(array)

    label_map = {name: name.replace("_", " ").strip().capitalize() for name in names}
    if source_labels is not None:
        unknown = set(source_labels) - set(names)
        if unknown:
            raise ValueError("source_labels names unknown coverage sources: "
                             + ", ".join(sorted(unknown)))
        label_map.update({str(name): str(label) for name, label in source_labels.items()})

    cumulative = np.logical_or.accumulate(np.stack(arrays), axis=1)
    codes = np.zeros(cumulative.shape[1:], dtype=np.uint16)
    for bit, source in enumerate(cumulative):
        codes |= source.astype(np.uint16) << bit
    observed_codes = [int(code) for code in np.unique(codes)]
    class_keys = {
        code: tuple(name for bit, name in enumerate(names) if code & (1 << bit))
        for code in observed_codes
    }
    overrides = dict(class_colours or {})
    unknown_overrides = [key for key in overrides
                         if any(name not in names for name in tuple(key))]
    if unknown_overrides:
        raise ValueError(f"class_colours names unknown coverage sources: {unknown_overrides}")
    nonempty = [code for code in observed_codes if class_keys[code]]
    fallback_colours = dict(zip(nonempty, common.series_colours(theme, len(nonempty))))
    classes = []
    for display_code, source_code in enumerate(observed_codes):
        key = class_keys[source_code]
        if not key:
            label = "Never occupied"
            colour = theme.colour(overrides.get((), "missing"))
        else:
            pieces = [label_map[name] for name in key]
            label = (pieces[0] + " only" if len(names) > 1 and len(pieces) == 1
                     else " and ".join(pieces))
            wanted = overrides.get(key)
            colour = theme.colour(wanted) if wanted is not None else fallback_colours[source_code]
        classes.append({
            "display_code": display_code,
            "source_code": source_code,
            "source_keys": "|".join(key),
            "occupancy_class": label,
            "colour": colour,
        })
    display = np.zeros_like(codes, dtype=np.uint16)
    for row in classes:
        display[codes == row["source_code"]] = row["display_code"]
    return {
        "source_names": names,
        "cumulative": cumulative,
        "source_codes": codes,
        "display_codes": display,
        "classes": classes,
    }


def _coverage_rows(
    encoding: Mapping[str, Any],
    hours: np.ndarray,
    frame_indices: Sequence[int],
    *,
    view: str,
) -> pd.DataFrame:
    rows = []
    codes = np.asarray(encoding["source_codes"])
    for frame_index in frame_indices:
        for klass in encoding["classes"]:
            pixels = int(np.count_nonzero(codes[frame_index] == klass["source_code"]))
            rows.append({
                "view": view,
                "frame_index": int(frame_index),
                "hours": float(hours[frame_index]),
                **klass,
                "pixels": pixels,
                "share": float(pixels / codes[frame_index].size),
            })
    return pd.DataFrame(rows)


def coverage_history(
    figure: Any,
    rect: tuple[float, float, float, float],
    occupancy_sources: Mapping[str, Any],
    theme: Any,
    *,
    hours: Sequence[float] | None = None,
    view: str = "stages",
    stages: int = 6,
    field: Mapping[str, float] | None = None,
    backgrounds: Any = None,
    source_labels: Mapping[str, str] | None = None,
    class_colours: Mapping[tuple[str, ...], str] | None = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
    gap: float = 0.0022,
) -> PanelResult:
    """Cumulative coverage maps for any named set of pixel-occupancy masks.

    ``view='final'`` draws one final map. ``view='stages'`` draws selected
    cumulative maps, including the final state, but never adds a second copy of
    that state elsewhere on the page.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    chosen_view = str(view).strip().lower()
    if chosen_view not in {"final", "stages"}:
        raise ValueError("coverage view must be final or stages")
    encoding = _coverage_encoding(
        occupancy_sources, theme, source_labels=source_labels,
        class_colours=class_colours,
    )
    codes = np.asarray(encoding["display_codes"])
    n_frames = codes.shape[0]
    time = np.arange(n_frames, dtype=float) if hours is None else np.asarray(hours, dtype=float)
    if time.ndim != 1 or len(time) != n_frames:
        raise ValueError(f"coverage hours has {len(time)} values for {n_frames} frames")
    if chosen_view == "final":
        frame_indices = np.asarray([n_frames - 1], dtype=int)
    else:
        if int(stages) < 2:
            raise ValueError("staged coverage needs at least two stages")
        frame_indices = np.unique(
            np.linspace(0, n_frames - 1, int(stages)).round().astype(int)
        )

    backdrop = None
    if backgrounds is not None:
        backdrop = np.asarray(backgrounds)
        if backdrop.ndim == 2:
            if len(frame_indices) != 1:
                raise ValueError("one coverage background can only accompany the final view")
            backdrop = np.broadcast_to(backdrop, (n_frames, *backdrop.shape))
        if backdrop.ndim != 3 or backdrop.shape != codes.shape:
            raise ValueError(
                f"coverage backgrounds are {backdrop.shape}; expected {codes.shape}"
            )

    colours = [row["colour"] for row in encoding["classes"]]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(-0.5, len(colours) + 0.5), len(colours))
    left, bottom, width, height = rect
    each = (width - gap * (len(frame_indices) - 1)) / len(frame_indices)
    extent = (_extent(dict(field)) if field is not None
              else (0, codes.shape[2], codes.shape[1], 0))
    axes = []
    for position, frame_index in enumerate(frame_indices):
        ax = figure.add_axes([left + position * (each + gap), bottom, each, height])
        if backdrop is not None:
            ax.imshow(backdrop[frame_index], cmap=theme["image_cmap"],
                      extent=extent, interpolation="nearest")
        ax.imshow(codes[frame_index], cmap=cmap, norm=norm, extent=extent,
                  interpolation="nearest", alpha=0.72 if backdrop is not None else 1.0)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        title = (f"Final · {time[frame_index]:g} h" if chosen_view == "final"
                 else f"{time[frame_index]:g} h")
        ax.text(
            0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=theme.size("annotation"), fontweight="bold",
            color=theme.colour("ink"),
            bbox={"facecolor": theme.colour("page"), "alpha": 0.78,
                  "edgecolor": "none", "pad": 1.5},
        )
        if chosen_view == "final":
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
        axes.append(ax)
    common.semantic_legend(
        axes[-1], theme,
        handles=[Patch(facecolor=row["colour"]) for row in encoding["classes"]],
        labels=[row["occupancy_class"] for row in encoding["classes"]],
        location="right", columns=1,
    )
    return PanelResult(
        data=_coverage_rows(encoding, time, frame_indices, view=chosen_view),
        axes=axes,
        extra={
            "frame_indices": frame_indices,
            "classes": pd.DataFrame(encoding["classes"]),
            "display_codes": codes[frame_indices],
        },
    )


def coverage_composition(
    ax: Any,
    hours: Sequence[float],
    occupancy_sources: Mapping[str, Any],
    theme: Any,
    *,
    source_labels: Mapping[str, str] | None = None,
    class_colours: Mapping[tuple[str, ...], str] | None = None,
    hour_ticks: float | None = None,
    x_label: str = "Hours from start of recording",
    y_label: str = "Share of field",
) -> PanelResult:
    """Share of the field in each cumulative occupancy-source combination."""
    from matplotlib.patches import Patch

    encoding = _coverage_encoding(
        occupancy_sources, theme, source_labels=source_labels,
        class_colours=class_colours,
    )
    codes = np.asarray(encoding["source_codes"])
    time = np.asarray(hours, dtype=float)
    if time.ndim != 1 or len(time) != codes.shape[0]:
        raise ValueError(f"coverage hours has {len(time)} values for {codes.shape[0]} frames")
    rows = _coverage_rows(encoding, time, range(len(time)), view="all_frames")
    by_code = {
        row["source_code"]: rows.loc[
            rows["source_code"] == row["source_code"], "share"
        ].to_numpy(float)
        for row in encoding["classes"]
    }
    # Occupied classes start at zero; never-occupied space sits above their
    # boundary, so the boundary itself reads as total coverage.
    ordered = [row for row in encoding["classes"] if row["source_code"] != 0]
    ordered += [row for row in encoding["classes"] if row["source_code"] == 0]
    ax.stackplot(
        time, [by_code[row["source_code"]] for row in ordered],
        colors=[row["colour"] for row in ordered], linewidth=0,
    )
    never = by_code.get(0, np.zeros(len(time), dtype=float))
    total_occupied = 1.0 - never
    ax.plot(time, total_occupied, color=theme.colour("ink"),
            linewidth=theme.stroke("emphasis"), label="Total occupied")
    ax.set_xlim(float(time.min()), float(time.max()))
    ax.set_ylim(0, 1)
    ax.set_xticks(theme.hour_ticks(float(time.min()), float(time.max()), hour_ticks))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    common.semantic_legend(
        ax, theme,
        handles=[Patch(facecolor=row["colour"]) for row in ordered],
        labels=[row["occupancy_class"] for row in ordered],
        location="right", columns=1,
    )
    rows["total_occupied_share"] = rows["frame_index"].map(
        dict(enumerate(total_occupied))
    )
    return PanelResult(data=rows, axes=ax, extra={"classes": pd.DataFrame(ordered)})


def ownership_map(
    ax: Any,
    first_owner: Any,
    owner_count: Any,
    theme: Any,
    *,
    field: dict,
    contested_role: str = "highlight",
    never_role: str = "missing",
) -> PanelResult:
    """First identity to claim each pixel, highlighting later contest."""
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import to_rgba

    owners = np.asarray(first_owner, dtype=int)
    counts = np.asarray(owner_count, dtype=int)
    identities = sorted(int(value) for value in np.unique(owners) if value)
    roles = ("morphology", "reporter", "surveillance", "motility", "evidence")
    palette = [theme.colour(never_role)] + [theme.colour(roles[index % len(roles)])
                                             for index in range(len(identities))]
    code = np.zeros_like(owners, dtype=int)
    for index, identity in enumerate(identities, start=1):
        code[owners == identity] = index
    handle = ax.imshow(code, cmap=ListedColormap(palette), vmin=0, vmax=max(len(palette) - 1, 1),
                       extent=_extent(field), interpolation="nearest")
    overlay = np.zeros((*owners.shape, 4), dtype=float)
    overlay[counts > 1] = to_rgba(theme.colour(contested_role), 0.75)
    ax.imshow(overlay, extent=_extent(field), interpolation="nearest")
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    rows, columns = np.indices(owners.shape)
    return PanelResult(
        data=pd.DataFrame({"row": rows.ravel(), "column": columns.ravel(),
                           "first_owner": owners.ravel(),
                           "owners": counts.ravel()}),
        axes=ax, extra={"handle": handle, "identities": identities},
    )


def metric_assignment(method: str) -> str:
    """Validate and canonicalise the way overlapping cell values are combined."""
    chosen = str(method).strip().lower().replace("-", "_")
    chosen = {
        "dominant": "most_frequent", "mode": "most_frequent",
        "mean": "equal_mean", "average": "equal_mean",
        "weighted_mean": "occupancy_weighted_mean",
        "weighted_average": "occupancy_weighted_mean",
    }.get(chosen, chosen)
    if chosen not in MAP_ASSIGNMENT_LABELS:
        raise ValueError("--map-assignment must be " + ", ".join(MAP_ASSIGNMENT_LABELS))
    return chosen


def owner_assignment(
    labels: Any,
    method: str = "first",
    *,
    first_owner: Any = None,
) -> np.ndarray:
    """Choose which cell carries a value at every ever-occupied field pixel.

    ``first`` reproduces the tissue-tectonics ownership map. ``last`` uses the
    final named occupant, while ``most_frequent`` uses the identity present for
    the most frames. Ties in ``most_frequent`` go to the smaller identity so a
    repeated build cannot change because dictionary order changed.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError("owner assignment needs a frame-by-row-by-column label stack")
    chosen = str(method).strip().lower().replace("-", "_")
    aliases = {"dominant": "most_frequent", "mode": "most_frequent"}
    chosen = aliases.get(chosen, chosen)
    if chosen not in {"first", "last", "most_frequent"}:
        raise ValueError("owner assignment must be first, last, or most_frequent")

    if chosen == "first" and first_owner is not None:
        owners = np.asarray(first_owner)
        if owners.shape != stack.shape[1:]:
            raise ValueError("first_owner must match the label field shape")
        return owners.astype(np.int64, copy=True)

    owners = np.zeros(stack.shape[1:], dtype=np.int64)
    if chosen == "first":
        for frame in stack:
            take = (owners == 0) & (frame > 0)
            owners[take] = frame[take]
        return owners
    if chosen == "last":
        for frame in stack:
            take = frame > 0
            owners[take] = frame[take]
        return owners

    largest = np.zeros(stack.shape[1:], dtype=np.int32)
    identities = sorted(int(value) for value in np.unique(stack) if value)
    for identity in identities:
        count = np.count_nonzero(stack == identity, axis=0).astype(np.int32)
        take = count > largest
        owners[take] = identity
        largest[take] = count[take]
    return owners


def _continuous_cmap(theme: Any, wanted: str | None, role: str) -> Any:
    """A colour map, accepting either a map name or one semantic colour."""
    from matplotlib.colors import LinearSegmentedColormap

    look = common.resolve_look(theme, wanted, fallback_role=role)
    if look.is_map:
        cmap = look.cmap
    else:
        cmap = LinearSegmentedColormap.from_list(
            f"{role}_spatial_metric", [theme.colour("page"), look.colour]
        )
    return cmap.with_extremes(bad=theme.colour("missing"))


def _value_limits(values: Any, range_mode: str) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("a spatial value map needs at least one finite value")
    mode = str(range_mode).strip().lower()
    if mode == "robust":
        low, high = map(float, np.nanpercentile(finite, [2.0, 98.0]))
    elif mode == "full":
        low, high = float(finite.min()), float(finite.max())
    else:
        raise ValueError("map range must be robust or full")
    if high <= low:
        padding = max(abs(low) * 0.05, 0.5)
        low, high = low - padding, high + padding
    return low, high


def spatial_value_map(
    ax: Any,
    values: Any,
    theme: Any,
    *,
    field: dict,
    label: str,
    cmap: str | None = None,
    role: str = "morphology",
    mask: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    range_mode: str = "full",
    ticks: Sequence[float] | None = None,
    allow_empty: bool = False,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Draw any two-dimensional numeric field through one shared map backend."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("a spatial value map needs a two-dimensional array")
    shown = array.copy()
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != shown.shape:
            raise ValueError("a spatial value-map mask must match the value field")
        shown[~selected] = np.nan
    finite = shown[np.isfinite(shown)]
    if not finite.size and not allow_empty:
        raise ValueError("a spatial value map has no finite pixels to draw")
    automatic_low, automatic_high = (_value_limits(finite, range_mode)
                                     if finite.size else (0.0, 1.0))
    low = automatic_low if vmin is None else float(vmin)
    high = automatic_high if vmax is None else float(vmax)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("spatial value-map limits must be finite and ascending")

    handle = ax.imshow(
        shown, cmap=_continuous_cmap(theme, cmap, role), vmin=low, vmax=high,
        extent=_extent(field), interpolation="nearest",
    )
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    common.inset_colour_bar(ax, handle, theme, label=label, ticks=ticks)
    rows, columns = np.where(np.isfinite(shown))
    return PanelResult(
        data=pd.DataFrame({
            "row": rows,
            "column": columns,
            "value": shown[rows, columns],
            "display_minimum": low,
            "display_maximum": high,
        }),
        axes=ax,
        extra={"handle": handle, "display_minimum": low, "display_maximum": high},
    )


def cell_metric_map(
    ax: Any,
    owners: Any,
    metric_values: Mapping[Any, Any] | pd.Series,
    theme: Any,
    *,
    field: dict,
    metric: str,
    label: str,
    cmap: str | None = None,
    role: str = "morphology",
    range_mode: str = "robust",
    assignment: str = "first",
    period: float | None = None,
    phase_origin: float = 0.0,
    min_phase_coherence: float = 0.5,
    limits: tuple[float, float] | None = None,
    allow_empty: bool = False,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Map cell values from assigned owners (2-D) or an occupancy movie (3-D).

    The averaging methods require the movie: equal_mean counts each occupant
    once at each pixel; occupancy_weighted_mean counts its occupied frames.
    Frame weighting is time weighting for these regularly sampled movies.
    Missing/nonfinite cell values contribute neither value nor weight.
    With a period, average positions on a circle and mask pixels whose mean
    vector length falls below min_phase_coherence. Export those mixed pixels
    too, with their coherence and a missing displayed value.
    """
    if period is not None and (not np.isfinite(period) or period <= 0):
        raise ValueError("a circular map needs a positive finite period")
    if not np.isfinite(phase_origin) or not 0 <= min_phase_coherence <= 1:
        raise ValueError("phase origin must be finite and minimum coherence must be between 0 and 1")
    assignment = metric_assignment(assignment)
    carrier = np.asarray(owners, dtype=int)
    averaging = assignment in {"equal_mean", "occupancy_weighted_mean"}
    if averaging and carrier.ndim != 3:
        raise ValueError("averaging cell metrics needs a frame-by-row-by-column label movie")
    if carrier.ndim not in {2, 3}:
        raise ValueError("a cell metric map needs an owner map or a label movie")
    if not averaging and carrier.ndim == 3:
        carrier = owner_assignment(carrier, assignment)
    series = pd.Series(metric_values, dtype=float)
    series.index = pd.Index([int(value) for value in series.index], name="identity")
    if not series.index.is_unique:
        raise ValueError("cell metric values need one row per identity")
    if (series.index <= 0).any():
        raise ValueError("cell metric identities must be positive; zero is background")
    series = pd.to_numeric(series, errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.empty and not allow_empty:
        raise ValueError(f"metric {metric!r} has no finite per-cell values")

    shape = carrier.shape[-2:]
    mapped = np.full(shape, np.nan, dtype=float)
    contributor_count = np.zeros(shape, dtype=int)
    total_weight = np.zeros(shape, dtype=float)
    phase_coherence = np.full(shape, np.nan, dtype=float)
    counts: dict[int, int] = {}
    if averaging:
        numerator = np.zeros(shape, dtype=complex if period is not None else float)
        for identity, value in finite.items():
            occupied_frames = np.count_nonzero(carrier == int(identity), axis=0)
            visited = occupied_frames > 0
            counts[int(identity)] = int(np.count_nonzero(visited))
            weights = visited if assignment == "equal_mean" else occupied_frames
            contribution = (np.exp(2j * np.pi * (float(value) - phase_origin) / period)
                            if period is not None else float(value))
            numerator += contribution * weights
            total_weight += weights
            contributor_count += visited
        if period is None:
            np.divide(numerator, total_weight, out=mapped, where=total_weight > 0)
        else:
            vector = np.full(shape, complex(np.nan), dtype=complex)
            np.divide(numerator, total_weight, out=vector, where=total_weight > 0)
            phase_coherence = np.clip(np.abs(vector), 0.0, 1.0)
            mapped = phase_origin + (np.angle(vector) * period / (2 * np.pi)) % period
            # Floating point cancellation near midnight must not print 24 h
            # while an otherwise identical cell prints 0 h.
            mapped[np.isclose(mapped, phase_origin + period, atol=1e-10, rtol=0)] = phase_origin
            # Equivalent sums (one vote per frame or one weighted vote per
            # cell) can straddle an exact threshold by machine precision.
            mixed = ((phase_coherence < min_phase_coherence - 1e-12)
                     | (phase_coherence < 1e-8))
            mapped[mixed] = np.nan
    else:
        identities, pixels = np.unique(carrier[carrier > 0], return_counts=True)
        counts = dict(zip(identities.astype(int), pixels.astype(int)))
        for identity, value in finite.items():
            selected = carrier == int(identity)
            mapped[selected] = float(value)
            if period is not None:
                mapped[selected] = phase_origin + (float(value) - phase_origin) % period
                phase_coherence[selected] = 1.0
            contributor_count[selected] = 1
            total_weight[selected] = 1
    # Choose robust limits from cells, not from the painted raster. Otherwise a
    # large territory votes thousands of times and a small cell votes only a
    # few hundred times for what the same per-cell measurement means.
    low, high = (_value_limits(finite.to_numpy(float), range_mode)
                 if not finite.empty else (0.0, 1.0))
    if limits is not None:
        low, high = limits
    if period is not None:
        low, high = phase_origin, phase_origin + period
    drawn = spatial_value_map(
        ax, mapped, theme, field=field, label=label, cmap=cmap, role=role,
        vmin=low, vmax=high, range_mode="full",
        x_label=x_label, y_label=y_label, allow_empty=allow_empty or period is not None,
        ticks=np.linspace(low, high, 5) if period is not None else None,
    )
    table = pd.DataFrame({
        "identity": series.index.astype(int),
        "metric": metric,
        "metric_value": series.to_numpy(float),
        "assigned_pixels": [np.nan if averaging else counts.get(int(identity), 0)
                            for identity in series.index],
        "contributing_pixels": [counts.get(int(identity), 0) if identity in finite.index else 0
                                for identity in series.index],
        "map_assignment": assignment,
        "map_range": range_mode,
        "display_minimum": drawn.extra["display_minimum"],
        "display_maximum": drawn.extra["display_maximum"],
        "circular_period": period,
    })
    rows, columns = np.where(total_weight > 0)
    pixel_values = pd.DataFrame({
        "row": rows, "column": columns, "value": mapped[rows, columns],
        "display_minimum": low, "display_maximum": high,
    })
    pixel_values["metric"] = metric
    pixel_values["map_assignment"] = assignment
    pixel_values["map_range"] = range_mode
    pixel_values["contributor_count"] = contributor_count[rows, columns]
    pixel_values["total_weight"] = total_weight[rows, columns]
    pixel_values["weight_unit"] = (
        "occupied frames" if assignment == "occupancy_weighted_mean" else "cells"
    )
    pixel_values["circular_period"] = period
    pixel_values["phase_coherence"] = phase_coherence[rows, columns]
    pixel_values["displayed"] = np.isfinite(pixel_values["value"])
    pixel_values["mixed_phase"] = ((period is not None) & ~pixel_values["displayed"])
    if not np.isfinite(mapped).any():
        message = "Mixed phases" if (total_weight > 0).any() else "No supported values"
        ax.text(0.5, 0.5, message,
                transform=ax.transAxes, ha="center", va="center",
                color=theme.colour("caption"), fontsize=theme.size("caption"))
    return PanelResult(data=table, axes=ax, extra={**drawn.extra, "pixel_values": pixel_values})


def cell_metric_maps(
    figure: Any,
    rect: tuple[float, float, float, float],
    owners: Any,
    metric_values: Mapping[str, Mapping[Any, Any] | pd.Series],
    theme: Any,
    *,
    field: dict,
    labels: Mapping[str, str] | None = None,
    colour_bar_labels: Mapping[str, str] | None = None,
    colour_maps: Sequence[str | None] | None = None,
    roles: Mapping[str, str] | None = None,
    range_mode: str = "robust",
    assignment: str = "first",
    columns: int = 3,
    periods: Mapping[str, float] | None = None,
    phase_origins: Mapping[str, float] | None = None,
    min_phase_coherence: float = 0.5,
    limits: Mapping[str, tuple[float, float]] | None = None,
    allow_empty_metrics: Sequence[str] = (),
    pixel_fields: Mapping[str, Any] | None = None,
) -> PanelResult:
    """Full-sized maps mixing cell measurements and already-computed pixel fields."""
    if not metric_values:
        raise ValueError("cell metric maps need at least one metric")
    if int(columns) < 1:
        raise ValueError("cell metric map columns must be at least one")
    names = list(metric_values)
    maps = list(colour_maps or [None] * len(names))
    if len(maps) == 1 and len(names) > 1:
        maps *= len(names)
    if len(maps) != len(names):
        raise ValueError("colour_maps must contain one entry or one per metric")

    n_columns = min(int(columns), len(names))
    n_rows = int(np.ceil(len(names) / n_columns))
    left, bottom, width, height = rect
    # A vertical colour key and its tick labels need physical room between
    # maps. The page grows for that room; the maps keep their declared size.
    gap_x = min(1.8 / figure.get_figwidth(), width * 0.16)
    gap_y = min(0.035, height * 0.065)
    item_width = (width - gap_x * (n_columns - 1)) / n_columns
    item_height = (height - gap_y * (n_rows - 1)) / n_rows
    axes = []
    tables = []
    pixel_tables = []
    handles = []
    label_map = dict(labels or {})
    colour_bar_label_map = dict(colour_bar_labels or {})
    role_map = dict(roles or {})
    for index, (metric, values, colour_map) in enumerate(
        zip(names, metric_values.values(), maps)
    ):
        row, column = divmod(index, n_columns)
        x = left + column * (item_width + gap_x)
        y = bottom + (n_rows - row - 1) * (item_height + gap_y)
        ax = figure.add_axes([x, y, item_width, item_height])
        label = label_map.get(metric, metric.replace("_", " ").capitalize())
        if metric in (pixel_fields or {}):
            array = np.asarray(pixel_fields[metric], float)
            bound = (limits or {}).get(metric, (None, None))
            drawn = spatial_value_map(
                ax, array, theme, field=field, label=colour_bar_label_map.get(metric, ""),
                cmap=colour_map, role=role_map.get(metric, "morphology"),
                vmin=bound[0], vmax=bound[1], range_mode=range_mode,
                allow_empty=metric in allow_empty_metrics,
                x_label="X position" if row == n_rows - 1 else "",
                y_label="Y position" if column == 0 else "",
            )
            drawn.data["metric"] = metric
            drawn.data["map_assignment"] = "direct pixel measurement"
            drawn.extra["pixel_values"] = drawn.data.copy()
        else:
            drawn = cell_metric_map(
                ax, owners, values, theme, field=field, metric=metric,
                label=colour_bar_label_map.get(metric, ""),
                cmap=colour_map, role=role_map.get(metric, "morphology"),
                range_mode=range_mode, assignment=assignment,
                period=(periods or {}).get(metric),
                phase_origin=(phase_origins or {}).get(metric, 0.0),
                min_phase_coherence=min_phase_coherence,
                limits=(limits or {}).get(metric), allow_empty=metric in allow_empty_metrics,
                x_label="X position" if row == n_rows - 1 else "",
                y_label="Y position" if column == 0 else "",
            )
        ax.set_title(label, loc="left", fontsize=theme.size("panel"), fontweight="bold")
        if column:
            # Same field coordinates in every panel; repeating these labels
            # crowds the preceding panel's colour key.
            ax.tick_params(axis="y", labelleft=False)
        axes.append(ax)
        tables.append(drawn.data)
        pixel_tables.append(drawn.extra["pixel_values"])
        handles.append(drawn.extra["handle"])
    return PanelResult(
        data=pd.concat(tables, ignore_index=True), axes=axes,
        extra={"handles": handles, "columns": n_columns, "rows": n_rows,
               "pixel_values": pd.concat(pixel_tables, ignore_index=True)},
    )


def owner_count_map(
    ax: Any,
    owner_count: Any,
    theme: Any,
    *,
    field: dict,
    minimum: int = 1,
    cmap: str | None = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Number of distinct cells that occupied each pixel, including one by default."""
    counts = np.asarray(owner_count, dtype=float)
    threshold = max(1, int(minimum))
    maximum = int(np.nanmax(counts))
    if maximum < threshold:
        ax.text(
            0.5, 0.5, f"No pixel was occupied by {threshold} or more cells",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        return PanelResult(
            data=pd.DataFrame(columns=[
                "row", "column", "owner_count", "display_minimum",
                "display_maximum", "minimum_owner_count",
            ]),
            axes=ax,
            extra={"handle": None, "display_minimum": threshold,
                   "display_maximum": threshold},
        )
    ticks = list(range(threshold, maximum + 1)) if maximum - threshold <= 6 else None
    drawn = spatial_value_map(
        ax, counts, theme, field=field,
        label="Distinct cells", cmap=cmap,
        role="highlight", mask=counts >= threshold,
        vmin=float(threshold), vmax=float(max(maximum, threshold) + (maximum == threshold)),
        ticks=ticks, x_label=x_label, y_label=y_label,
    )
    table = drawn.data.rename(columns={"value": "owner_count"}).copy()
    table["minimum_owner_count"] = threshold
    return PanelResult(data=table, axes=drawn.axes, extra=drawn.extra)


def first_coverage_time(labels: Any, hours: Any) -> np.ndarray:
    """Elapsed hours when each eventually occupied pixel was first covered."""
    stack = np.asarray(labels)
    times = np.asarray(hours, dtype=float)
    if stack.ndim != 3 or times.shape != (len(stack),):
        raise ValueError("first coverage needs a frame-by-row-by-column label stack and one time per frame")
    if not len(stack) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("first-coverage times must be finite and strictly increasing")
    if not np.isfinite(stack).all() or np.any(stack < 0):
        raise ValueError("labels must be finite and nonnegative; missing observations are not background")

    occupied = stack > 0
    visited = occupied.any(axis=0)
    first_indices = np.argmax(occupied, axis=0)
    elapsed = np.full(stack.shape[1:], np.nan, dtype=float)
    elapsed[visited] = times[first_indices[visited]] - times[0]
    return elapsed


def first_coverage_time_map(
    ax: Any,
    labels: Any,
    theme: Any,
    *,
    hours: Any,
    field: dict,
    cmap: str | None = "viridis",
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Eventually covered area, hued by first coverage time from recording start."""
    stack = np.asarray(labels)
    times = np.asarray(hours, dtype=float)
    elapsed = first_coverage_time(stack, times)
    visited = np.isfinite(elapsed)
    span = float(times[-1] - times[0])
    drawn = spatial_value_map(
        ax, elapsed, theme, field=field, label="First covered (h)", cmap=cmap,
        role="surveillance", mask=visited, vmin=0.0,
        vmax=max(span, np.finfo(float).eps), allow_empty=True,
        x_label=x_label, y_label=y_label,
    )
    table = drawn.data.rename(columns={"value": "first_covered_hours"}).copy()
    if not table.empty:
        occupied = stack > 0
        first_indices = np.argmax(occupied, axis=0)
        rows = table["row"].to_numpy(dtype=int)
        columns = table["column"].to_numpy(dtype=int)
        table["first_covered_frame_index"] = first_indices[rows, columns]
    else:
        table["first_covered_frame_index"] = pd.Series(dtype=int)
    return PanelResult(data=table, axes=drawn.axes, extra=drawn.extra)


def cumulative_occupancy(labels: Any, hours: Any) -> np.ndarray:
    """Cumulative occupied hours per pixel, pooling all nonzero cell identities.

    Integrate observed occupancy between successive timestamps using the mean
    of the two endpoint states. This is equivalent to assigning half each
    observed interval to each endpoint; never extrapolate beyond the recording.
    The first frame has zero accumulated elapsed time. Missing observations are
    not accepted as background, and identities cannot double-count a pixel.
    """
    stack = np.asarray(labels)
    times = np.asarray(hours, float)
    if stack.ndim != 3 or len(stack) < 2 or times.shape != (len(stack),):
        raise ValueError("cumulative occupancy needs at least two labelled frames and one time per frame")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("occupancy times must be finite and strictly increasing")
    if not np.isfinite(stack).all() or np.any(stack < 0):
        raise ValueError("labels must be finite and nonnegative; missing observations are not background")
    occupied = (stack > 0).astype(float)
    increments = (occupied[:-1] + occupied[1:]) * .5 * np.diff(times)[:, None, None]
    return np.concatenate([np.zeros_like(increments[:1]), np.cumsum(increments, axis=0)])


def cumulative_occupancy_map(ax: Any, labels: Any, theme: Any, *, hours: Any,
                             field: dict, frame_index: int | None = None,
                             cmap: str | None = "viridis") -> PanelResult:
    """Generic occupied-duration map, final state by default, in recording hours."""
    accumulated = cumulative_occupancy(labels, hours)
    index = len(accumulated) - 1 if frame_index is None else int(frame_index)
    if not 0 <= index < len(accumulated):
        raise ValueError("frame_index lies outside the recording")
    span = float(np.asarray(hours)[index] - np.asarray(hours)[0])
    result = spatial_value_map(ax, accumulated[index], theme, field=field,
                              label="Cumulative occupied time (h)", cmap=cmap,
                              vmin=0., vmax=max(span, np.finfo(float).eps),
                              mask=np.any(np.asarray(labels)[:index + 1] > 0, axis=0), allow_empty=True)
    result.data["frame_index"] = index
    result.data["hours"] = float(np.asarray(hours)[index])
    result.extra["integration"] = "half of each observed interval assigned to each endpoint"
    return result


def direct_pixel_handoffs(labels: Any) -> tuple[np.ndarray, pd.DataFrame]:
    """Count immediate nonzero identity changes and list their directed pairs."""
    stack = np.asarray(labels)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError("direct pixel handoffs need at least two labelled frames")
    counts = np.zeros(stack.shape[1:], dtype=np.uint32)
    rows: list[dict[str, int]] = []
    for frame_index, (before, after) in enumerate(zip(stack[:-1], stack[1:])):
        changed = (before > 0) & (after > 0) & (before != after)
        counts += changed.astype(np.uint32)
        if not changed.any():
            continue
        pairs = np.column_stack((before[changed], after[changed]))
        unique, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        for (source, destination), pixels in zip(unique, pair_counts):
            rows.append({
                "from_frame_index": int(frame_index),
                "to_frame_index": int(frame_index + 1),
                "from_identity": int(source),
                "to_identity": int(destination),
                "transferred_pixels": int(pixels),
            })
    events = pd.DataFrame(rows, columns=[
        "from_frame_index", "to_frame_index", "from_identity", "to_identity",
        "transferred_pixels",
    ])
    return counts, events


def direct_pixel_handoff_map(
    ax: Any,
    labels: Any,
    theme: Any,
    *,
    field: dict,
    cmap: str | None = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Where a pixel passed directly between named cells in adjacent frames."""
    counts, events = direct_pixel_handoffs(labels)
    maximum = int(counts.max())
    if maximum == 0:
        ax.text(
            0.5, 0.5, "No direct cell-to-cell pixel transfers",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        return PanelResult(
            data=pd.DataFrame(columns=[
                "row", "column", "direct_handoffs", "display_minimum",
                "display_maximum",
            ]),
            axes=ax,
            extra={"handle": None, "events": events, "handoff_pixels": 0,
                   "handoff_transfers": 0},
        )
    ticks = list(range(1, maximum + 1)) if maximum <= 6 else None
    drawn = spatial_value_map(
        ax, counts.astype(float), theme, field=field,
        label="Transfers per pixel", cmap=cmap,
        role="highlight", mask=counts > 0, vmin=1.0,
        vmax=float(maximum + (maximum == 1)), ticks=ticks,
        x_label=x_label, y_label=y_label,
    )
    table = drawn.data.rename(columns={"value": "direct_handoffs"})
    extra = {
        **drawn.extra,
        "events": events,
        "handoff_pixels": int(np.count_nonzero(counts)),
        "handoff_transfers": int(counts.sum()),
    }
    return PanelResult(data=table, axes=drawn.axes, extra=extra)


def _event_identities(event: Any) -> list[int]:
    """Positive identities named on one merge/split event row."""
    identities = [int(event.identity)]
    raw = getattr(event, "candidate_identities", "")
    if not pd.isna(raw):
        for token in str(raw).split("|"):
            token = token.strip()
            if token and token.lower() != "nan":
                identities.append(int(float(token)))
    return sorted({identity for identity in identities if identity > 0})


def split_event_areas(
    labels: Any,
    events: pd.DataFrame,
    *,
    event_type: str = "contact_separate",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Count recorded contact-separation transition footprints at each pixel.

    One event area is the union of its named cells' last connected footprint and
    first separated footprint. This localises the transition without treating a
    tracker split as biological cell division.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError("split-event areas need a frame-by-row-by-column label stack")
    required = {
        "event_id", "event_type", "identity", "candidate_identities",
        "gap_end_frame_index", "post_frame_index",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError("split-event table is missing " + ", ".join(missing))

    selected = events.loc[events["event_type"].astype(str).eq(event_type)].copy()
    counts = np.zeros(stack.shape[1:], dtype=np.uint32)
    records: list[dict[str, Any]] = []
    for event in selected.itertuples(index=False):
        connected_frame = int(event.gap_end_frame_index)
        separated_frame = int(event.post_frame_index)
        if not (0 <= connected_frame < len(stack) and 0 <= separated_frame < len(stack)):
            raise ValueError(
                f"split event {event.event_id} names frames outside the label movie"
            )
        identities = _event_identities(event)
        area = (
            np.isin(stack[connected_frame], identities)
            | np.isin(stack[separated_frame], identities)
        )
        counts += area.astype(np.uint32)
        records.append({
            "event_id": str(event.event_id),
            "event_type": str(event.event_type),
            "identity": int(event.identity),
            "candidate_identities": "|".join(
                str(identity) for identity in identities
                if identity != int(event.identity)
            ),
            "connected_frame_index": connected_frame,
            "separated_frame_index": separated_frame,
            "event_area_px": int(np.count_nonzero(area)),
        })
    columns = [
        "event_id", "event_type", "identity", "candidate_identities",
        "connected_frame_index", "separated_frame_index", "event_area_px",
    ]
    return counts, pd.DataFrame(records, columns=columns)


def split_event_map(
    ax: Any,
    labels: Any,
    events: pd.DataFrame | None,
    theme: Any,
    *,
    field: dict,
    event_type: str = "contact_separate",
    cmap: str | None = None,
    x_label: str = "X position in the field",
    y_label: str = "Y position in the field",
) -> PanelResult:
    """Areas involved when recorded touching cells subsequently separated."""
    empty_columns = [
        "row", "column", "split_events", "display_minimum", "display_maximum",
    ]
    if events is None:
        ax.text(0.5, 0.5, "Split-event record unavailable", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return PanelResult(
            data=pd.DataFrame(columns=empty_columns), axes=ax,
            extra={"handle": None, "events": pd.DataFrame(), "event_count": 0,
                   "event_pixels": 0, "availability": "record unavailable"},
        )

    counts, selected = split_event_areas(labels, events, event_type=event_type)
    maximum = int(counts.max())
    if maximum == 0:
        ax.text(0.5, 0.5, "No recorded contact-separation events", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return PanelResult(
            data=pd.DataFrame(columns=empty_columns), axes=ax,
            extra={"handle": None, "events": selected, "event_count": 0,
                   "event_pixels": 0, "availability": "available"},
        )

    ticks = list(range(1, maximum + 1)) if maximum <= 6 else None
    drawn = spatial_value_map(
        ax, counts.astype(float), theme, field=field,
        label="Split transitions per pixel", cmap=cmap, role="highlight",
        mask=counts > 0, vmin=1.0, vmax=float(maximum + (maximum == 1)),
        ticks=ticks, x_label=x_label, y_label=y_label,
    )
    table = drawn.data.rename(columns={"value": "split_events"}).copy()
    return PanelResult(
        data=table, axes=drawn.axes,
        extra={**drawn.extra, "events": selected, "event_count": len(selected),
               "event_pixels": int(np.count_nonzero(counts)),
               "availability": "available"},
    )


def never_visited_mark(over_raw: bool) -> tuple[str, str]:
    """The colour role :func:`never_visited` inks, and what that ink means.

    One place, because the panel draws the mark and the figure writes the legend
    for it, and a legend that names the other side of the mask is worse than no
    legend at all.
    """
    if over_raw:
        return "missing", "Never occupied by a named cell or unclaimed foreground"
    return "visited", "Covered at some point by a named cell or unclaimed foreground"


def never_visited(
    ax: Any,
    mask: Any,
    theme: Any,
    *,
    field: dict,
    background: Any = None,
    look: common.Look | None = None,
) -> PanelResult:
    """Complement of every named or explicitly unclaimed footprint.

    Which side carries the ink depends on what is underneath, because the
    never-visited field is most of the frame. On a blank page the covered ground
    is inked and the never-visited ground is left as page, so the footprint the
    cells actually made is the mark rather than a solid panel with a few holes
    in it. Over a raw frame it is the other way round, in the pale blank colour:
    there the mark's job is to mask off the empty ground so the cells it was
    derived from still show through. :func:`never_visited_mark` says which of
    the two happened, so a legend can be written for the right one.
    """
    from matplotlib.colors import ListedColormap

    values = np.asarray(mask, dtype=bool)
    over_raw = background is not None
    if over_raw:
        ax.imshow(background, cmap=theme["image_cmap"], extent=_extent(field), interpolation="nearest")
    role, _ = never_visited_mark(over_raw)
    drawn = values if over_raw else ~values
    colour = (look or common.Look(colour=theme.colour(role))).colour
    ax.imshow(drawn.astype(int), cmap=ListedColormap(["none", colour]), vmin=0, vmax=1,
              extent=_extent(field), interpolation="nearest",
              alpha=0.85 if over_raw else 1.0)
    ax.set_xlim(0, field["width"])
    ax.set_ylim(field["height"], 0)
    ax.set_aspect("equal")
    return PanelResult(
        data=pd.DataFrame([{"pixels": int(values.size),
                            "never_visited_px": int(values.sum()),
                            "never_visited_share": float(values.mean())}]),
        axes=ax, extra={"share": float(values.mean())},
    )


def pixel_fate_states(
    labels: Any,
    event_windows: Mapping[str, Sequence[int]],
    *,
    identities: Sequence[int] | None = None,
    core_fraction: float = 0.8,
    transient_fraction: float = 0.2,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Classify selected pixels within any named sequence of frame windows.

    Each mapping entry is an event name and the frames belonging to the window
    that starts at that event. With more than one identity, a change in modal
    owner is reported as a transfer. With one identity, the same calculation
    describes that cell without inventing transfers to cells outside the
    selection.
    """
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError("pixel fate labels must have shape (frame, row, column)")
    if not 0.0 <= float(transient_fraction) < float(core_fraction) <= 1.0:
        raise ValueError("pixel fate thresholds must satisfy 0 <= transient < core <= 1")
    if len(event_windows) < 2:
        raise ValueError("pixel fate needs at least two event windows")

    windows: list[tuple[str, np.ndarray]] = []
    for name, indices in event_windows.items():
        frame_indices = np.asarray(indices, dtype=int)
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError(f"pixel fate event {name!r} has no frames")
        if np.any(frame_indices < 0) or np.any(frame_indices >= values.shape[0]):
            raise ValueError(f"pixel fate event {name!r} contains a frame outside the stack")
        if len(np.unique(frame_indices)) != len(frame_indices):
            raise ValueError(f"pixel fate event {name!r} repeats a frame")
        windows.append((str(name), frame_indices))

    available = np.asarray([int(value) for value in np.unique(values) if value], dtype=int)
    selected = available if identities is None else np.asarray(identities, dtype=int)
    unknown = sorted(set(selected.tolist()) - set(available.tolist()))
    if unknown:
        raise ValueError(f"pixel fate identities are absent from the stack: {unknown}")
    selected_labels = np.where(np.isin(values, selected), values, 0)
    selected_mask = selected_labels > 0
    used_frames = np.unique(np.concatenate([indices for _, indices in windows]))
    union = selected_mask[used_frames].any(axis=0)
    coordinates = np.flatnonzero(union)

    class_names = list(FATE_ROLES)
    class_index = {name: index for index, name in enumerate(class_names)}
    states = np.full(
        (len(coordinates), len(windows)), class_index["vacated"], dtype=int
    )
    previous_owner = np.zeros(len(coordinates), dtype=int)
    ever_occupied = np.zeros(len(coordinates), dtype=bool)
    summaries: list[dict[str, Any]] = []

    for event_index, (event_name, frame_indices) in enumerate(windows):
        frequency = selected_mask[frame_indices].mean(axis=0).ravel()[coordinates]
        current = frequency > 0
        states[current & (frequency >= core_fraction), event_index] = class_index["core"]
        states[
            current & (frequency > transient_fraction) & (frequency < core_fraction),
            event_index,
        ] = class_index["fringe"]
        states[current & (frequency <= transient_fraction), event_index] = class_index["transient"]
        states[~current & ~ever_occupied, event_index] = class_index["vacated"]

        if len(selected) > 1 and len(coordinates):
            owners = np.zeros(len(coordinates), dtype=int)
            flat = selected_labels[frame_indices].reshape(len(frame_indices), -1)[:, coordinates]
            for pixel in range(flat.shape[1]):
                nonzero = flat[:, pixel][flat[:, pixel] > 0]
                if len(nonzero):
                    owner_values, counts = np.unique(nonzero, return_counts=True)
                    owners[pixel] = int(owner_values[np.argmax(counts)])
            transferred = (
                current & (previous_owner > 0) & (owners > 0) & (owners != previous_owner)
            )
            states[transferred, event_index] = class_index["transferred"]
            previous_owner = np.where(owners > 0, owners, previous_owner)
        ever_occupied |= current

        for class_name, code in class_index.items():
            pixels = int(np.count_nonzero(states[:, event_index] == code))
            summaries.append(
                {
                    "event_index": event_index,
                    "event": event_name,
                    "class": class_name,
                    "pixels": pixels,
                    "share": float(pixels / len(coordinates)) if len(coordinates) else np.nan,
                    "frames": int(len(frame_indices)),
                    "first_frame": int(frame_indices.min()),
                    "last_frame": int(frame_indices.max()),
                }
            )
    return states, pd.DataFrame(summaries)


def fate_transition_table(
    states: Any,
    *,
    event_labels: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Exact pixel counts connecting every class across adjacent events."""
    values = np.asarray(states)
    if values.ndim != 2:
        raise ValueError("pixel fate states must be an observation-by-event matrix")
    classes = list(class_names or FATE_ROLES)
    events = list(event_labels or [f"Event {index + 1}" for index in range(values.shape[1])])
    if len(events) != values.shape[1]:
        raise ValueError("event_labels must contain one label per event")
    rows = []
    for event_index in range(values.shape[1] - 1):
        for source, source_name in enumerate(classes):
            for target, target_name in enumerate(classes):
                rows.append(
                    {
                        "from_event_index": event_index,
                        "to_event_index": event_index + 1,
                        "from_event": events[event_index],
                        "to_event": events[event_index + 1],
                        "from_class": source_name,
                        "to_class": target_name,
                        "pixels": int(np.count_nonzero(
                            (values[:, event_index] == source)
                            & (values[:, event_index + 1] == target)
                        )),
                    }
                )
    return pd.DataFrame(rows)


def fate_flow(
    figure: Any,
    rect: tuple[float, float, float, float],
    stages: Any,
    theme: Any,
    *,
    stage_labels: Sequence[str],
    class_roles: Mapping[str, str] | None = None,
    class_labels: Mapping[str, str] | None = None,
) -> PanelResult:
    """Pixel-class flow across arbitrary user-named events or time windows."""
    roles = dict(class_roles or FATE_ROLES)
    labels = list(roles)
    display_labels = dict(class_labels or FATE_LABELS)
    missing_labels = set(labels) - set(display_labels)
    if missing_labels:
        raise ValueError("class_labels is missing " + ", ".join(sorted(missing_labels)))
    values = np.asarray(stages)
    if values.dtype.kind not in "iuf":
        mapping = {label: index for index, label in enumerate(labels)}
        unknown = sorted(set(np.unique(values).tolist()) - set(mapping))
        if unknown:
            raise ValueError("pixel fate states contain unknown classes: " + ", ".join(unknown))
        values = np.vectorize(mapping.get)(values).astype(int)
    drawn = common.flow(
        figure, rect, values, theme, class_labels=labels,
        class_roles=[roles[label] for label in labels], stage_labels=stage_labels,
    )
    return PanelResult(
        data=fate_transition_table(values, event_labels=stage_labels, class_names=labels),
        axes=drawn.axes,
        extra={
            **drawn.extra,
            "class_roles": roles,
            "class_labels": display_labels,
        },
    )
