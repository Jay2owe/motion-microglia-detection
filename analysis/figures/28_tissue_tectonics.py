"""Figure 28: six canonical maps of cell coverage and motion across the field.

    python analysis/figures/28_tissue_tectonics.py <run>
    ... --map-summary mean
    ... --map-assignment occupancy_weighted_mean
    ... --fit-method lomb --significance-method lomb
    ... --period-min-hours 2 --period-max-hours 48
    ... --multiple-testing bh
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import tifffile

from analysis.circadian import CIRCADIAN_ANALYSIS_OPTION_DEFAULTS
from _derive import _scaled_field
from _metrics import describe, role_for
from _schema import (FigureContext, FigureResult, Input, Option, Panel, Stack,
                     Table, figure, run_figure)
from _tissue_rhythms import RHYTHM_MAPS, analyse_rhythm_maps
from panels import territory as territory_panels


SPEED_METRIC = "speed"
INTENSITY_PERIOD_METRIC = "reporter_period_hours"
PANEL_ORDER = (
    "first_coverage",
    "cumulative_occupancy",
    "unique_cells",
    "speed",
    "significant_period",
    "splitting_events",
)


def _circadian_options() -> tuple[Option, ...]:
    """Complete shared contract, with raw Lomb-Scargle significance by default."""
    defaults = {
        **CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
        "fit_method": "lomb",
        "significance_method": "lomb",
        "multiple_testing": "none",
    }
    return tuple(Option(name, default=default) for name, default in defaults.items())


def _frame_hours(frame_summary: pd.DataFrame, n_frames: int) -> np.ndarray:
    """One complete, consistent elapsed-hour coordinate for the label movie."""
    required = {"frame_index", "hours"}
    missing = sorted(required - set(frame_summary.columns))
    if missing:
        raise ValueError("frame_summary.csv is missing " + ", ".join(missing))
    clock = frame_summary[["frame_index", "hours"]].copy()
    clock["frame_index"] = pd.to_numeric(clock["frame_index"], errors="coerce")
    clock["hours"] = pd.to_numeric(clock["hours"], errors="coerce")
    if clock["frame_index"].duplicated().any():
        raise ValueError("frame_summary.csv needs one time for each frame")
    clock = clock.sort_values("frame_index")
    expected = np.arange(n_frames)
    if (not np.array_equal(clock["frame_index"].to_numpy(), expected)
            or not np.isfinite(clock["hours"]).all()
            or np.any(np.diff(clock["hours"].to_numpy(float)) <= 0)):
        raise ValueError(
            "tissue tectonics needs one finite, increasing time for every label frame"
        )
    return clock["hours"].to_numpy(float)


def _speed_values(cell_frame: pd.DataFrame, summary: str) -> pd.Series:
    """One median or mean speed per cell."""
    method = str(summary).strip().lower()
    if method not in {"median", "mean"}:
        raise SystemExit("--map-summary must be median or mean for the speed map")
    required = {"identity", "frame_index", SPEED_METRIC}
    missing = sorted(required - set(cell_frame.columns))
    if missing:
        raise SystemExit("cell_frame.csv is missing " + ", ".join(missing))
    work = cell_frame[["identity", "frame_index", SPEED_METRIC]].copy()
    work[SPEED_METRIC] = pd.to_numeric(work[SPEED_METRIC], errors="coerce")
    grouped = work.sort_values(["identity", "frame_index"]).groupby("identity")[SPEED_METRIC]
    values = grouped.median() if method == "median" else grouped.mean()
    values.index = values.index.astype(int)
    values.name = SPEED_METRIC
    if not np.isfinite(values).any():
        raise SystemExit("cell_frame.csv has no finite per-cell speed values")
    return values


def _panel_luts(requested: list[str]) -> dict[str, str | None]:
    """One optional colour map per canonical panel, in canonical order."""
    if not requested:
        return dict.fromkeys(PANEL_ORDER)
    if len(requested) == 1:
        return dict.fromkeys(PANEL_ORDER, requested[0])
    if len(requested) != len(PANEL_ORDER):
        raise SystemExit(
            f"--map-luts needs one colour map or {len(PANEL_ORDER)} maps in panel order"
        )
    return dict(zip(PANEL_ORDER, requested))


def _plotted_pixels(panel: str, table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Normalise one panel's exact pixel encodings for figure_data.csv."""
    result = table.copy()
    if value_column in result and value_column != "value":
        result = result.rename(columns={value_column: "value"})
    result.insert(0, "panel", panel)
    result.insert(1, "panel_order", PANEL_ORDER.index(panel) + 1)
    result["measure"] = value_column
    return result


def _period_support_masks(labels: np.ndarray, fits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pixels ever occupied by supported and exploratory displayed periods."""
    shape = labels.shape[1:]

    def occupied(rows: pd.DataFrame) -> np.ndarray:
        identities = rows.identity.dropna().astype(int).unique()
        if not len(identities):
            return np.zeros(shape, dtype=bool)
        return np.any(np.isin(labels, identities), axis=0)

    shown = fits[fits.period_map_displayed.fillna(False).astype(bool)]
    supported = shown[shown.period_map_supported.fillna(False).astype(bool)]
    exploratory = shown[~shown.period_map_supported.fillna(False).astype(bool)]
    return occupied(supported), occupied(exploratory)


def _style_map_grid(axes: dict[str, object], positions: dict[str, tuple[int, int]],
                    theme: object) -> None:
    """Use one text ladder and show coordinates only on the grid's outer edges."""
    bottom_row = max(row for row, _ in positions.values())
    left_column = min(column for _, column in positions.values())
    for key, ax in axes.items():
        row, column = positions[key]
        ax.title.set_fontsize(theme.size("panel"))
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(theme.size("annotation"))
        ax.yaxis.label.set_fontsize(theme.size("annotation"))
        ax.tick_params(axis="both", labelsize=theme.size("caption"))
        if row != bottom_row:
            ax.tick_params(axis="x", labelbottom=False)
        if column != left_column:
            ax.tick_params(axis="y", labelleft=False)


@figure(
    number=28,
    slug="tissue-tectonics",
    summary="when cells first covered the tissue, how they occupied it, moved, cycled and separated",
    title="Cell coverage and motion across the tissue field",
    reads=(
        Stack("owner_count.tif", module="territory"),
        Input("labels", optional=False),
        Table("frame_summary.csv", module="summary"),
        Table("cell_frame.csv", module="measurement modules"),
        Table("history_merge_split_events.csv", module="history", optional=True),
    ),
    panels=(
        Panel("first_coverage", territory_panels.first_coverage_time_map,
              title="First coverage time",
              min_width_inches=5.5, min_height_inches=5.5),
        Panel("cumulative_occupancy", territory_panels.cumulative_occupancy_map,
              title="Cumulative occupancy",
              min_width_inches=5.5, min_height_inches=5.5),
        Panel("unique_cells", territory_panels.owner_count_map,
              title="Unique cells",
              min_width_inches=5.5, min_height_inches=5.5),
        Panel("speed", territory_panels.cell_metric_map,
              title="Cell speed",
              min_width_inches=5.5, min_height_inches=5.5),
        Panel("significant_period", territory_panels.cell_metric_map,
              title="All significant intensity periods\nBlack boundary: supported subset",
              min_width_inches=5.5, min_height_inches=5.5),
        Panel("splitting_events", territory_panels.split_event_map,
              title="Split-event areas",
              min_width_inches=5.5, min_height_inches=5.5),
    ),
    options=(
        Option("map_summary", default="median",
               help="how each cell's speed trace becomes one value: median or mean"),
        Option("map_assignment", default="occupancy_weighted_mean",
               cast=territory_panels.metric_assignment),
        Option("map_luts", default=[]),
        Option("map_range", default="robust"),
        *_circadian_options(),
    ),
    grammar="six-panel spatial small multiple",
)
def build(ctx: FigureContext) -> FigureResult:
    label_path = ctx.input_path("labels")
    if label_path is None:
        raise SystemExit("tissue-tectonics needs the labels input recorded in the run manifest")
    labels = tifffile.imread(label_path)
    if labels.ndim != 3:
        raise SystemExit("tissue-tectonics needs a frame-by-row-by-column label stack")
    owner_count = ctx.stack("owner_count.tif")
    if owner_count.shape != labels.shape[1:]:
        raise SystemExit("owner_count.tif does not match the label field")

    frame_summary = ctx.table("frame_summary.csv")
    cell_frame = ctx.table("cell_frame.csv")
    hours = _frame_hours(frame_summary, len(labels))
    field = _scaled_field(ctx)
    summary_method = str(ctx.option("map_summary")).strip().lower()
    speed_values = _speed_values(cell_frame, summary_method)
    assignment = ctx.option("map_assignment")
    map_range = str(ctx.option("map_range")).strip().lower()
    if map_range not in {"robust", "full"}:
        raise SystemExit("--map-range must be robust or full")
    luts = _panel_luts(list(ctx.option("map_luts")))

    panels = ctx.panels()
    placements = {
        panel.key: (index // 3, index % 3, 1, 1)
        for index, panel in enumerate(panels)
    }
    fig, axes = ctx.grid_layout(
        panels, placements, bottom_inches=2.7,
        horizontal_gap_inches=2.2, vertical_gap_inches=1.0,
    )
    positions = {panel.key: placements[panel.key][:2] for panel in panels}

    def axis_labels(key: str) -> tuple[str, str]:
        row, column = positions[key]
        bottom_row = max(position[0] for position in positions.values())
        return ("X position in the field" if row == bottom_row else "",
                "Y position in the field" if column == 0 else "")

    pixel_tables: list[pd.DataFrame] = []
    auxiliary: dict[str, pd.DataFrame] = {}
    drawn_axes: list = []

    def record(key: str, result, value_column: str, *, pixels: pd.DataFrame | None = None):
        result = ctx.drew(key, result)
        exact = result.data if pixels is None else pixels
        pixel_tables.append(_plotted_pixels(key, exact, value_column))
        auxiliary[f"{key}_pixels.csv"] = exact
        axes_value = result.axes if isinstance(result.axes, (list, tuple)) else [result.axes]
        drawn_axes.extend(axis for axis in axes_value if axis is not None)
        return result

    if "first_coverage" in axes:
        x_label, y_label = axis_labels("first_coverage")
        record("first_coverage", territory_panels.first_coverage_time_map(
            axes["first_coverage"], labels, ctx.theme, hours=hours, field=field,
            cmap=luts["first_coverage"] or "viridis",
            x_label=x_label, y_label=y_label,
        ), "first_covered_hours")

    if "cumulative_occupancy" in axes:
        x_label, y_label = axis_labels("cumulative_occupancy")
        cumulative = record("cumulative_occupancy", territory_panels.cumulative_occupancy_map(
            axes["cumulative_occupancy"], labels, ctx.theme, hours=hours, field=field,
            cmap=luts["cumulative_occupancy"] or "viridis",
        ), "value")
        axes["cumulative_occupancy"].set_xlabel(x_label)
        axes["cumulative_occupancy"].set_ylabel(y_label)
        auxiliary["cumulative_occupancy_pixels.csv"] = cumulative.data

    if "unique_cells" in axes:
        x_label, y_label = axis_labels("unique_cells")
        record("unique_cells", territory_panels.owner_count_map(
            axes["unique_cells"], owner_count, ctx.theme, field=field, minimum=1,
            cmap=luts["unique_cells"], x_label=x_label, y_label=y_label,
        ), "owner_count")

    metric_carrier = (
        labels if assignment in {"equal_mean", "occupancy_weighted_mean"}
        else territory_panels.owner_assignment(labels, assignment)
    )
    speed_drawn = None
    if "speed" in axes:
        x_label, y_label = axis_labels("speed")
        speed_unit = describe(SPEED_METRIC).unit_text(ctx.interval)
        speed_drawn = record("speed", territory_panels.cell_metric_map(
            axes["speed"], metric_carrier, speed_values, ctx.theme, field=field,
            metric=SPEED_METRIC,
            label=f"{summary_method.capitalize()} speed ({speed_unit})",
            cmap=luts["speed"], role=role_for(SPEED_METRIC), range_mode=map_range,
            assignment=assignment, x_label=x_label, y_label=y_label,
        ), "value", pixels=None)
        pixel_tables[-1] = _plotted_pixels(
            "speed", speed_drawn.extra["pixel_values"], "value"
        )
        auxiliary["speed_pixels.csv"] = speed_drawn.extra["pixel_values"]
        auxiliary["speed_cells.csv"] = speed_drawn.data.assign(summary_operation=summary_method)

    rhythm_fits = pd.DataFrame()
    rhythm_params: dict = {}
    period_drawn = None
    if "significant_period" in axes:
        try:
            rhythm_values, rhythm_fits, _, rhythm_params = analyse_rhythm_maps(
                ctx, cell_frame, [INTENSITY_PERIOD_METRIC]
            )
        except ValueError as error:
            raise SystemExit(str(error)) from None
        x_label, y_label = axis_labels("significant_period")
        method = str(rhythm_fits["method_label"].iloc[0]).replace(
            "LS Periodogram", "Lomb–Scargle periodogram"
        )
        period_drawn = record("significant_period", territory_panels.cell_metric_map(
            axes["significant_period"], metric_carrier,
            rhythm_values[INTENSITY_PERIOD_METRIC], ctx.theme, field=field,
            metric=INTENSITY_PERIOD_METRIC,
            label="Significant intensity period (h)",
            cmap=luts["significant_period"] or RHYTHM_MAPS[INTENSITY_PERIOD_METRIC].cmap,
            role="reporter", range_mode="full", assignment=assignment,
            limits=tuple(rhythm_params["period_search_hours"]), allow_empty=True,
            x_label=x_label, y_label=y_label,
        ), "value", pixels=None)
        intensity_fits = rhythm_fits[rhythm_fits.metric.eq("corrected_mean")].copy()
        supported_mask, exploratory_mask = _period_support_masks(labels, intensity_fits)
        period_pixels = period_drawn.extra["pixel_values"].copy()
        rows = period_pixels.row.to_numpy(int)
        columns = period_pixels.column.to_numpy(int)
        period_pixels["has_supported_contributor"] = supported_mask[rows, columns]
        period_pixels["has_exploratory_contributor"] = exploratory_mask[rows, columns]
        period_pixels["period_support"] = np.select(
            [period_pixels.has_supported_contributor & period_pixels.has_exploratory_contributor,
             period_pixels.has_supported_contributor,
             period_pixels.has_exploratory_contributor],
            ["mixed support", "supported period", "exploratory period"],
            default="excluded",
        )
        if supported_mask.any():
            axes["significant_period"].contour(
                supported_mask.astype(float), levels=[0.5],
                colors=[ctx.theme.colour("ink")],
                linewidths=ctx.theme.stroke("guide"), origin="upper",
                extent=(0, field["width"], field["height"], 0),
            )
        period_cells = period_drawn.data.merge(
            intensity_fits[["identity", "period_map_displayed", "period_map_supported",
                            "period_map_status", "period_map_exclusion"]],
            on="identity", how="left", validate="one_to_one",
        )
        pixel_tables[-1] = _plotted_pixels(
            "significant_period", period_pixels, "value"
        )
        auxiliary["significant_period_pixels.csv"] = period_pixels
        auxiliary["significant_period_cells.csv"] = period_cells
        auxiliary["rhythm_map_fits.csv"] = rhythm_fits
        auxiliary["statistics.csv"] = rhythm_fits.copy()
    else:
        method = "Lomb–Scargle periodogram"

    split_drawn = None
    if "splitting_events" in axes:
        split_events = ctx.optional_table("history_merge_split_events.csv")
        x_label, y_label = axis_labels("splitting_events")
        split_drawn = record("splitting_events", territory_panels.split_event_map(
            axes["splitting_events"], labels, split_events, ctx.theme, field=field,
            cmap=luts["splitting_events"], x_label=x_label, y_label=y_label,
        ), "split_events")
        auxiliary["splitting_events.csv"] = split_drawn.extra["events"]

    figure_data = pd.concat(pixel_tables, ignore_index=True, sort=False)
    period_low, period_high = rhythm_params.get(
        "period_search_hours",
        [float(ctx.option("period_min_hours")), float(ctx.option("period_max_hours"))],
    )
    intensity_fits = rhythm_fits[
        rhythm_fits.get("metric", pd.Series(index=rhythm_fits.index, dtype=object)).eq(
            "corrected_mean"
        )
    ]
    rhythmic_cells = int(
        intensity_fits.get("period_map_displayed", pd.Series(dtype=bool)).fillna(False).sum()
    )
    supported_cells = int(
        intensity_fits.get("period_map_supported", pd.Series(dtype=bool)).fillna(False).sum()
    )
    exploratory_cells = rhythmic_cells - supported_cells
    period_pixels = (
        len(period_drawn.extra["pixel_values"]) if period_drawn is not None else 0
    )
    ever_covered_pixels = int(np.any(labels > 0, axis=0).sum())
    period_coverage_percent = (
        100 * period_pixels / ever_covered_pixels if ever_covered_pixels else 0.0
    )
    split_count = int(split_drawn.extra["event_count"]) if split_drawn is not None else 0
    split_pixels = int(split_drawn.extra["event_pixels"]) if split_drawn is not None else 0
    _style_map_grid(axes, positions, ctx.theme)
    correction = (
        str(rhythm_fits["correction"].iloc[0]) if not rhythm_fits.empty
        else str(ctx.option("multiple_testing"))
    )
    correction_label = {
        "bh": "Benjamini–Hochberg corrected",
        "none": "uncorrected",
    }.get(correction, correction.capitalize() + " corrected")

    return FigureResult(
        figure=fig,
        axes=drawn_axes,
        figure_data=figure_data,
        subtitle=(
            f"{ctx.summary['stem']}. Panels follow coverage, occupancy, cell count, speed, "
            f"significant intensity period and split-event area. The period map contains "
            f"{rhythmic_cells} significant cells; {supported_cells} periods are supported."
        ),
        footnote=ctx.footnote(
            "First coverage is elapsed time from the recording start; cumulative occupancy "
            "integrates all occupied intervals and revisits.",
            f"Speed is the per-cell {summary_method}; overlapping occupants use the "
            f"{territory_panels.MAP_ASSIGNMENT_LABELS[assignment]}.",
            f"Corrected mean intensity periods use Circadian Workbench {method}, a "
            f"{period_low:g}–{period_high:g} h "
            f"search and {correction_label} significance. All significant available estimates "
            f"are painted; black boundaries mark the {supported_cells} supported-period cells "
            f"and the other {exploratory_cells} are exploratory. Grey means no significant "
            "contributor, not proven absence of a rhythm. Shared-pixel period hues summarize contributing "
            "cells; they are not a period detected at that pixel.",
            "A split area is the union of the tracker-recorded cells' last connected and first "
            "separated footprints; it is not evidence of biological cell division.",
        ),
        auxiliary=auxiliary,
        readme=f"""## Panels, in canonical order

1. Area ever covered, hued by elapsed time of first coverage.
2. Total cumulative occupied time at each pixel, pooling cells and revisits.
3. Number of unique cell identities that occupied each pixel.
4. The per-cell {summary_method} speed mapped over its covered area.
5. Covered area of every cell with a significant available corrected-intensity period from
   Circadian Workbench's {method}; black boundaries mark the supported subset.
6. Areas involved in recorded contact-separation events.

## Spatial overlap

The speed and significant-period panels use `{assignment}`. With the default
`occupancy_weighted_mean`, each cell's value is weighted by the time that cell
occupied that particular pixel. Missing values and background contribute no
weight. A shared-pixel period is a summary of contributing cell periods, not a
rhythm detected at that pixel and not evidence of a shared tissue clock.

## Rhythm evidence

The corrected mean intensity trace of each cell is analysed separately through
Circadian Workbench.
The canonical estimator and primary significance test are both Lomb–Scargle,
explicitly configured rather than inferred. This build searched {period_low:g}–{period_high:g} h,
used alpha {float(ctx.option('rhythmic_alpha')):g}, {correction_label.lower()} probabilities,
at least {int(ctx.option('min_observations'))} observations. A cell is painted when
its significance test passes and its period is positive and available. The
at-least-{float(ctx.option('min_cycles')):g}-cycles and search-boundary rules determine
support rather than visibility. `{rhythmic_cells}` cells are painted in this build:
`{supported_cells}` supported periods and `{exploratory_cells}` exploratory periods.
Together they cover `{period_pixels}` of `{ever_covered_pixels}` ever-occupied pixels
({period_coverage_percent:.1f}%).
Estimator, significance method, Circadian Workbench version, exact settings,
raw probabilities, adjusted probabilities and exclusions are retained in
`rhythm_map_fits.csv` and `statistics.csv`.
The default is raw, uncorrected significance. Use `--multiple-testing bh`,
`--multiple-testing bonferroni`, or `--multiple-testing sidak` to apply a
correction; `--multiple-testing none` selects the default raw probabilities.

## Split-event area

The event source is the optional tracker table
`history_merge_split_events.csv`. Only `contact_separate` rows are used. One
event area is the union of the named cells' last connected footprint and first
separated footprint in the accepted label movie. The map counts repeated event
areas per pixel. These are tracker contact-separation events, not biological
cell divisions. This build mapped {split_count} events over {split_pixels} pixels.

## Exact plotted data

`figure_data.csv` contains every displayed pixel value with `panel` and
`panel_order`. The per-panel pixel tables, per-cell speed values, rhythm fits,
and split-event rows are retained beside it in the bundle.
""",
        console=(
            f"six canonical panels; {rhythmic_cells} significant intensity-period cells "
            f"({supported_cells} supported, {exploratory_cells} exploratory); "
            f"{split_count} split events over {split_pixels} pixels"
        ),
    )


if __name__ == "__main__":
    run_figure("tissue-tectonics")
