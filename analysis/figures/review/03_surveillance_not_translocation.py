"""Audit figure 3: inspect whether a two-measurement pattern is broadly supported.

The page is one use of the generic ``panels.common.hexbin`` density panel::

    python analysis/figures/03_surveillance_not_translocation.py <run>
    ... --metrics soma_step_px_gapless_median,turnover_index_median
    ... --bins 16
    ... --density-lut viridis
    ... --density-summary mean

Any two numeric ``cell_summary.csv`` columns can be used. The panel chooses a
logarithmic scale only when every value is positive and the range exceeds one
order of magnitude; the choice travels in the exact figure-data table.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np

from _metrics import axis_label, describe
from _options import exactly
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import Mark, reference_lines
from panels import common

DEFAULT_RUN = "outputs/g02_95_A3_trend"


@figure(
    number=3,
    slug="surveillance-not-translocation",
    purpose="review",
    summary="density audit of two per-cell measurements",
    title="{y} against {x}: cell density",
    grammar="hexagonal density map",
    reads=(Table("cell_summary.csv", module="motility"),),
    panels=(
        Panel("density", common.hexbin,
              title="One hexagon may contain several cells",
              item_width_inches=8.5, item_height_inches=6.8),
    ),
    options=(
        Option("metrics",
               default=("soma_step_px_gapless_median", "turnover_index_median"),
               cast=exactly(2),
               help="the x and y columns of the density map, in that order"),
        Option("bins", default=14,
               help="hexagonal density resolution along the x direction"),
        Option("density_lut", default="viridis",
               help="colour map used for cells per hexagon"),
        Option("density_summary", default="none",
               help="draw the mean y value within each x bin, or no summary line"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    axes_wanted = ctx.option("metrics")
    bins = int(ctx.option("bins"))
    density_lut = str(ctx.option("density_lut"))
    density_summary = str(ctx.option("density_summary")).lower()
    if len(axes_wanted) != 2:
        raise SystemExit(
            f"--metrics needs exactly two columns, x then y; got {len(axes_wanted)}")
    if bins < 2:
        raise SystemExit("--bins must be at least 2 for a density map")
    if density_summary not in ("mean", "none"):
        raise SystemExit("--density-summary must be mean or none")
    x_column, y_column = axes_wanted

    summary = ctx.table("cell_summary.csv")
    missing = [column for column in (x_column, y_column)
               if column not in summary.columns]
    if missing:
        numeric = [column for column in summary.columns
                   if summary[column].dtype.kind in "fi"]
        raise SystemExit(
            f"cell_summary.csv has no {', '.join(missing)}. Numeric columns: "
            + ", ".join(numeric)
        )

    kept = [column for column in (
        "identity", "observed_frames", "coverage", x_column, y_column,
    ) if column in summary.columns]
    plotted = summary[kept].dropna(subset=[x_column, y_column]).copy()
    if plotted.empty:
        raise SystemExit("the requested measurements have no paired cell values")

    def needs_log(values) -> bool:
        values = np.asarray(values, dtype=float)
        return bool(
            len(values) and np.all(values > 0)
            and float(values.max() / values.min()) > 10.0
        )

    log_x = needs_log(plotted[x_column])
    log_y = needs_log(plotted[y_column])
    x_scale = "log" if log_x else "linear"
    y_scale = "log" if log_y else "linear"
    median_x = float(plotted[x_column].median())
    median_y = float(plotted[y_column].median())

    panels = ctx.panels()
    figure_, axes = ctx.layout(
        panels,
        minimum_sizes={"density": (8.5, 6.8)},
        left_inches=2.5,
        right_inches=1.8,
        top_inches=2.35,
        bottom_inches=2.55,
    )
    ax = axes["density"]
    drawn = common.hexbin(
        ax,
        plotted[x_column],
        plotted[y_column],
        ctx.theme,
        gridsize=bins,
        cmap=density_lut,
        binned_mean=density_summary == "mean",
        mean_label=f"Mean {describe(y_column).label.lower()} within x bin",
        xscale=x_scale,
        yscale=y_scale,
        x_label=axis_label(x_column, ctx.interval, wrap=False) + ", median per cell",
        y_label=axis_label(y_column, ctx.interval) + "\nmedian per cell",
    )
    ctx.drew("density", drawn)
    common.inset_colour_bar(
        ax, drawn.extra["handle"], ctx.theme, label="Cells per hexagon")
    reference_lines(ax, [Mark(median_x)], ctx.theme)
    reference_lines(ax, [Mark(median_y)], ctx.theme, orientation="horizontal")
    ax.margins(x=0.04, y=0.06)

    def friendly_log_ticks(values) -> list[float]:
        low, high = float(np.min(values)), float(np.max(values))
        candidates = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5,
                      1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000)
        return [value for value in candidates if low * 0.8 <= value <= high * 1.25]

    if log_x:
        ticks = friendly_log_ticks(plotted[x_column])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])
    if log_y:
        ticks = friendly_log_ticks(plotted[y_column])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{value:g}" for value in ticks])

    figure_data = plotted.rename(
        columns={x_column: "x_value", y_column: "y_value"}
    )
    figure_data.insert(1, "x_metric", x_column)
    figure_data.insert(2, "y_metric", y_column)
    figure_data["x_scale"] = x_scale
    figure_data["y_scale"] = y_scale
    figure_data["density_bins"] = bins
    figure_data["density_lut"] = density_lut
    figure_data["density_summary"] = density_summary

    cells = int(len(plotted))
    span_hours = float(ctx.summary["hours_covered"])
    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=figure_data,
        auxiliary=(
            {"binned_mean.csv": drawn.data}
            if not drawn.data.empty else {}
        ),
        title_fields={
            "x": describe(x_column).label.lower(),
            "y": describe(y_column).label,
        },
        subtitle=(
            f"{ctx.summary['stem']}, {cells} cells over {span_hours:.0f} h. "
            "Colour is the number of cells in each hexagon."
            + (" The black line is the mean y value within each x bin."
               if density_summary == "mean" else "")
        ),
        footnote=ctx.footnote(
            f"Grey lines mark the coordinate-wise medians: x = {median_x:.2f}, "
            f"y = {median_y:.2f}. Axes are {x_scale} and {y_scale}; scale selection "
            "is data-derived and recorded in figure_data.csv. This is a distribution "
            "audit, not a test of association."
        ),
        readme=f"""## What the figure checks

This audit asks whether an apparent relationship between `{x_column}` and
`{y_column}` is supported by many cells or by isolated points. Every observation
is one cell. Hexagon colour records the local cell count; grey lines are the
coordinate-wise medians. `--density-summary mean` optionally adds the mean y
value within each x bin.

## Generic density panel

The drawing is `panels.common.hexbin`, which accepts any paired numeric arrays,
hexagon resolution, colour map, optional binned mean, optional reference band,
and linear or logarithmic axes. This page supplies two user-selected columns
from `cell_summary.csv`; it contains no surveillance-specific drawing code.

## Interpretation

The density map is descriptive and reports no inferential statistics. Automatic
log scaling is used only when every value on an axis is positive and the range
exceeds one order of magnitude. The exact choice is stored with every plotted
row in `figure_data.csv`.
""",
        console=(
            f"cells {cells}  x {x_column} median {median_x:.3f} ({x_scale})  "
            f"y {y_column} median {median_y:.3f} ({y_scale})  bins {bins}"
        ),
    )


if __name__ == "__main__":
    run_figure("surveillance-not-translocation", DEFAULT_RUN)
