"""Figure 4: which measurements are properties of a cell and which are states.

One panel, ``panels.common.lollipop``, one row per measurement::

    python analysis/figures/04_stable_traits_versus_states.py <run>
    ... --metrics area_px,corrected_mean,solidity,turnover_index   these rows

Any measurement `cell_summary.csv` holds a median and an interquartile range for
can be a row. Row wording comes from ``_metrics``, so a measurement is called
the same thing here as everywhere else it is drawn.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _metrics import describe
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import Mark, lollipop, reference_lines

#: Where the two sides of the axis are drawn. Cut-offs for colouring the
#: points, not a judgement about what the measurement is.
TRAIT_MAX = 0.5
STATE_MIN = 0.7

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=4,
    slug="stable-traits-versus-states",
    summary="which measurements are properties of a cell and which are states",
    title="Within-cell against between-cell spread, per measurement",
    grammar="lollipop of a within-over-between spread ratio, one row per measurement",
    reads=(Table("cell_summary.csv", module="motility"),),
    panels=(Panel("ranking", lollipop, title="Within-cell over between-cell spread"),),
    options=(
        Option("metrics",
               default=["area_px", "corrected_mean", "skeleton_branches", "solidity",
                        "ramification_index", "turnover_index"],
               help="which measurements get a row, one per measurement"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    metrics = ctx.option("metrics")
    if not metrics:
        raise SystemExit("--metrics needs at least one measurement to compare")

    summary = ctx.table("cell_summary.csv")

    missing = [c for c in metrics
               if f"{c}_median" not in summary.columns
               or f"{c}_iqr" not in summary.columns]
    if missing:
        available = sorted(
            c[: -len("_median")] for c in summary.columns
            if c.endswith("_median") and f"{c[:-len('_median')]}_iqr" in summary.columns
        )
        raise SystemExit(
            f"--metrics names {', '.join(missing)}, which cell_summary.csv has no "
            f"median and interquartile range for.\nMeasurements available: "
            + ", ".join(available)
        )

    rows = []
    for column in metrics:
        medians = summary[f"{column}_median"].dropna()
        within = summary[f"{column}_iqr"].dropna()
        between = float(np.percentile(medians, 75) - np.percentile(medians, 25))
        metric = describe(column)
        rows.append(
            {
                "metric": column,
                "label": metric.label,
                "unit": metric.unit_text(ctx.interval),
                "cells": int(len(medians)),
                "between_cell_iqr_of_medians": between,
                "median_within_cell_iqr": float(within.median()),
                "within_over_between": (float(within.median() / between)
                                        if between else np.nan),
            }
        )
    table = pd.DataFrame(rows).sort_values("within_over_between").reset_index(drop=True)
    table["side"] = np.where(
        table["within_over_between"] <= TRAIT_MAX, "trait",
        np.where(table["within_over_between"] >= STATE_MIN, "state", "neither"),
    )

    cells = int(table["cells"].max())
    span_hours = float(ctx.summary["hours_covered"])

    width_in = 13.2
    header_in = 2.23
    row_in = 0.96
    foot_in = 2.60
    height_in = header_in + len(table) * row_in + foot_in

    ctx.panels()
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    ax = figure_.add_axes(
        [0.235, fraction(foot_in), 0.62, fraction(len(table) * row_in)])

    upper = max(1.5, float(table["within_over_between"].max(skipna=True)) * 1.18)
    ax.set_xlim(0, upper)

    reference_lines(ax, [Mark(1.0, role="caption", dashed=True)], ctx.theme)
    lollipop(
        ax, table["within_over_between"], ctx.theme,
        labels=table["label"],
        colours=[ctx.theme.colour("variable" if value >= 0.6 else "stable")
                 for value in table["within_over_between"]],
        annotations=[f"{value:.2f}" for value in table["within_over_between"]],
    )

    ax.set_xlabel(f"One cell's own spread over {span_hours:.0f} h\n"
                  "÷ the spread between cells")
    ax.set_xticks(np.arange(0, upper + 0.01, 0.5))

    ax.text(
        0.055, len(table) - 0.42, "a stable trait of the cell",
        fontsize=ctx.theme.size("subtitle"), color=ctx.theme.colour("stable"),
        style="italic",
    )
    ax.text(
        1.02, -0.5, "as variable within a cell\nas between cells",
        fontsize=ctx.theme.size("subtitle"), color=ctx.theme.colour("caption"),
        style="italic", va="bottom",
    )

    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=table,
        heading="Within-cell against between-cell spread",
        subtitle=(
            f"{ctx.summary['stem']}, {cells} cells over {ctx.summary['frames']} frames. "
            f"For each measurement: the median cell's own interquartile range\n"
            f"across {span_hours:.0f} h, divided by the interquartile range of the "
            f"per-cell medians."
        ),
        footnote=(
            f"Both spreads are interquartile ranges. The dashed line at 1.0 is where a "
            f"measurement varies as much within one cell over {span_hours:.0f} h as it "
            f"does between cells."
        ),
        readme=f"""## What the ratio is

For every measurement, two spreads:

- **between cells** - the interquartile range of the per-cell medians;
- **within a cell** - the median cell's own interquartile range over the
  recording.

Their ratio is the x axis. Points are coloured either side of {STATE_MIN};
that cut-off decides the colour of a mark and nothing else.

## Choosing the rows

`--metrics` takes any measurement `cell_summary.csv` holds both a `_median` and
an `_iqr` for. The page grows a row for each, so a longer list is drawn taller
rather than more tightly.""",
        console=table.round(3).to_string(index=False),
    )


if __name__ == "__main__":
    run_figure("stable-traits-versus-states", DEFAULT_RUN)
