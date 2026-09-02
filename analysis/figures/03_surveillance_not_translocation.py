"""Figure 3: cells rebuild themselves without going anywhere.

One panel with two optional edges, ``panels.common.scatter_with_margins``::

    python analysis/figures/03_surveillance_not_translocation.py <run>
    ... --metrics soma_step_px_gapless_median,turnover_index_median   x then y
    ... --size area_px_median      which column sets point size; empty for one size
    ... --bins 24                  bands in the edge histograms
    ... --panels scatter           drop the edge histograms

Any pair of ``cell_summary.csv`` columns can be the two axes. Their wording and
colour come from ``_metrics``, and the axis scale is chosen from the data: a
column spanning more than about a decade is drawn on a log axis, because a
linear one would put nearly every cell into the first tenth of it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _metrics import axis_label, describe, role_for
from _options import exactly
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import Mark, reference_lines, resolve_look, scatter_with_margins

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=3,
    slug="surveillance-not-translocation",
    summary="cells rebuild themselves without going anywhere",
    title="{x} against {y}",
    grammar="scatter of two per-cell summaries with edge histograms",
    reads=(Table("cell_summary.csv", module="motility"),
           Table("cell_frame.csv", module="motility")),
    panels=(Panel("scatter", scatter_with_margins, block=True,
                  title="One point per cell"),
            Panel("margins", scatter_with_margins, block=True,
                  title="The two distributions at the edges")),
    options=(
        Option("metrics",
               default=("soma_step_px_gapless_median", "turnover_index_median"),
               cast=exactly(2),
               help="the x and y columns of the scatter, in that order"),
        Option("size", default="area_px_median"),
        Option("bins", default=20),
        Option("trace_luts", default=None, cast=str, metavar="COLOUR",
               help="colour or colour map of the points"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    axes_wanted = ctx.option("metrics")
    size_by = ctx.option("size")
    bins = ctx.option("bins")
    look_name = ctx.option("trace_luts")

    # The flag's own cast already refuses a list of the wrong length; a
    # configured list is typed JSON and never goes through a cast, so it is
    # checked here too rather than unpacking into a ValueError.
    if len(axes_wanted) != 2:
        raise SystemExit(
            f"--metrics needs exactly two columns, x then y; got {len(axes_wanted)}")
    x_column, y_column = axes_wanted

    summary = ctx.table("cell_summary.csv")
    cell_frame = ctx.table("cell_frame.csv")

    wanted = [x_column, y_column] + ([size_by] if size_by else [])
    missing = [column for column in wanted if column not in summary.columns]
    if missing:
        raise SystemExit(
            f"cell_summary.csv has no {', '.join(missing)}. Columns available: "
            + ", ".join(c for c in summary.columns if summary[c].dtype.kind in "fi")
        )

    kept = ["identity", "observed_frames", "coverage"] + wanted
    kept += [c for c in ("straightness", "msd_alpha", "territory_hull_px2")
             if c in summary.columns]
    plotted = summary[list(dict.fromkeys(kept))].dropna(
        subset=[x_column, y_column]).copy()

    # The two distributions the medians came from, at cell-frame resolution rather
    # than per cell: the summary hides how wide one cell's own spread is.
    raw_columns = [column.removesuffix("_median") for column in (x_column, y_column)]
    rows = []
    for column in raw_columns:
        if column not in cell_frame.columns:
            continue
        values = cell_frame[column].dropna()
        rows.append({
            "measure": column, "n_cell_frames": len(values),
            "p10": values.quantile(0.10), "median": values.median(),
            "p90": values.quantile(0.90),
        })

    cells = int(len(plotted))
    median_x = float(plotted[x_column].median())
    median_y = float(plotted[y_column].median())
    msd_alpha = (float(plotted["msd_alpha"].median(skipna=True))
                 if "msd_alpha" in plotted else np.nan)
    straightness = (float(plotted["straightness"].median(skipna=True))
                    if "straightness" in plotted else np.nan)
    interval = f"{ctx.interval:.0f} min"
    span_hours = float(ctx.summary["hours_covered"])

    panels = ctx.panels()
    figure_ = ctx.sheet(13.4, 10.4)

    # A log axis when the column spans more than about a decade. Chosen from the
    # numbers rather than written in, so a different pair of columns is drawn on the
    # scale that suits it instead of the scale that suited these two.
    positive = plotted[x_column][plotted[x_column] > 0]
    log_x = bool(positive.size and (positive.max() / positive.min()) > 10)

    sizes = None
    if size_by:
        reference = plotted[size_by].max() or 1.0
        sizes = (ctx.theme.point_area(0.7)
                 + ctx.theme.point_area(2.0) * (plotted[size_by] / reference))

    drawn = scatter_with_margins(
        figure_, (0.095, 0.165, 0.705, 0.66),
        plotted[x_column], plotted[y_column], ctx.theme,
        sizes=sizes,
        role=role_for(y_column),
        look=resolve_look(ctx.theme, look_name, role_for(y_column)) if look_name else None,
        bins=bins, log_x=log_x, margins="margins" in panels,
    )
    ctx.drew("scatter", drawn)
    ax, ax_top, ax_right = drawn.axes

    reference_lines(ax, [Mark(median_x)], ctx.theme)
    reference_lines(ax, [Mark(median_y)], ctx.theme, orientation="horizontal")

    ax.set_xlabel(axis_label(x_column, ctx.interval, wrap=False) + ", median per cell")
    ax.set_ylabel(axis_label(y_column, ctx.interval) + "\nmedian per cell")
    if log_x:
        friendly = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]
        low, high = float(positive.min()) * 0.8, float(plotted[x_column].max()) * 1.25
        ax.set_xlim(low, high)
        ticks = [value for value in friendly if low <= value <= high]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])
    ax.set_ylim(0, plotted[y_column].max() * 1.12)

    ax.annotate(
        f"median cell:\n{median_x:.2f} {ctx.summary['scale']['length_unit']} moved,\n"
        f"{median_y:.0%} of itself replaced",
        xy=(median_x, median_y), xycoords="data",
        xytext=(0.60, 0.72), textcoords="axes fraction",
        fontsize=ctx.theme.size("subtitle"), color=ctx.theme.colour("caption"),
        arrowprops=dict(arrowstyle="-", color=ctx.theme.colour("reference"),
                        linewidth=ctx.theme.stroke("guide")),
    )

    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=plotted,
        auxiliary={"cell_frame_distributions.csv": pd.DataFrame(rows)},
        title_fields={"x": describe(x_column).label,
                      "y": describe(y_column).label.lower()},
        subtitle=(
            f"{ctx.summary['stem']}, {cells} cells with a measurable track. Each point "
            f"is one cell over {span_hours:.0f} h"
            + (f"; point size is its median {describe(size_by).label.lower()}."
               if size_by else ".")
        ),
        note=(
            "Movement is measured\nbetween consecutive\nframes only, so an\n"
            "absence of several\nframes never inflates\na step.\n\n"
            f"Replacement compares\na cell's pixels with\nits own pixels "
            f"{interval}\nearlier, not with a\nneighbour's."
        ),
        note_at={"x": 0.835, "y": 0.66, "linespacing": 1.4},
        footnote=ctx.footnote(
            f"Median squared-displacement exponent {msd_alpha:.2f} "
            f"(1.0 is a random walk) and median path straightness "
            f"{straightness:.2f}."
        ),
        readme=f"""## What the figure shows

One point per cell, every cell with a track long enough to summarise. The x
axis is `{x_column}` and the y axis `{y_column}`, both medians over that cell's lifetime.
{'Point size is `' + size_by + '`. ' if size_by else ''}Grey lines mark the
median cell on each axis, and the edge histograms show the two distributions
the medians came from.

## What to be careful of

Steps are computed between consecutive frames only, so a cell absent for
several frames never contributes one inflated step. Replacement compares a
cell with itself, not with its neighbours, so it measures the outline changing
shape rather than the cell moving.

## Choosing the axes

`--metrics <x>,<y>` takes any two numeric columns of `cell_summary.csv`, and
`--size` a third for the point area. Whether the x axis is logarithmic is
decided from the spread of the column, not written into this figure.""",
        console=(f"cells {cells}  x {x_column} median {median_x:.3f}  "
                 f"y {y_column} median {median_y:.3f}  log-x {log_x}  "
                 f"panels {ctx.drawn}"),
    )


if __name__ == "__main__":
    run_figure("surveillance-not-translocation", DEFAULT_RUN)
