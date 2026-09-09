"""Audit figure 16: cell-centre spatial range relative to typical cell size.

The old page repeated the same two measurements in two scatters and then added
their ratio as a third plot. This audit keeps one full-size parity scatter and
defines both inputs on the page::

    python analysis/figures/16_territory_anchoring.py <run>
    ... --min-coverage 0.8

The shared motility panel preserves a zero cell-centre path area and keeps an
equal-area reference line; it does not fit or test a relationship.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np

from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import motility as motility_panels

DEFAULT_RUN = "outputs/g02_95_A3_trend"


@figure(
    number=16,
    slug="territory-anchoring",
    purpose="review",
    summary="cell-centre path area relative to typical cell footprint area",
    title="Cell-centre spatial range relative to cell size",
    grammar="symmetric-log scatter with parity line",
    reads=(Table("cell_summary.csv", module="motility"),),
    panels=(
        Panel("anchoring", motility_panels.centroid_path_area_against_footprint,
              title="Each point is one tracked cell",
              item_width_inches=8.6, item_height_inches=6.8),
    ),
    options=(
        Option("min_coverage", default=0.8,
               help="minimum fraction of the recording in which a cell was observed"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    minimum_coverage = float(ctx.option("min_coverage"))
    if not 0 <= minimum_coverage <= 1:
        raise SystemExit("--min-coverage must be between 0 and 1")

    summary = ctx.table("cell_summary.csv")
    required = ["identity", "territory_hull_px2", "area_px_median", "observed_frames"]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise SystemExit("cell_summary.csv has no " + ", ".join(missing))

    frames = int(ctx.summary["frames"])
    data = summary[required].dropna(
        subset=["territory_hull_px2", "area_px_median", "observed_frames"]
    ).copy()
    data["recording_coverage"] = data["observed_frames"] / max(frames, 1)
    data = data[
        (data["recording_coverage"] >= minimum_coverage)
        & (data["territory_hull_px2"] >= 0)
        & (data["area_px_median"] > 0)
    ].copy()
    if data.empty:
        raise SystemExit(
            f"no cells meet --min-coverage {minimum_coverage:g} with both areas measured")

    data["median_footprint_area"] = data["area_px_median"].map(ctx.scale.area)
    data["centroid_path_area"] = data["territory_hull_px2"].map(ctx.scale.area)
    data["path_area_over_footprint"] = (
        data["centroid_path_area"] / data["median_footprint_area"]
    )
    data["equal_area_side"] = np.where(
        data["centroid_path_area"] > data["median_footprint_area"],
        "path area larger than footprint",
        np.where(
            data["centroid_path_area"] < data["median_footprint_area"],
            "path area smaller than footprint",
            "equal areas",
        ),
    )

    panels = ctx.panels()
    figure_, axes = ctx.layout(
        panels,
        minimum_sizes={"anchoring": (8.6, 6.8)},
        left_inches=2.3,
        right_inches=4.0,
        top_inches=2.35,
        bottom_inches=2.65,
    )
    ax = axes["anchoring"]
    drawn = motility_panels.centroid_path_area_against_footprint(
        ax,
        data["centroid_path_area"],
        data["median_footprint_area"],
        ctx.theme,
        scale="symlog",
        linear_threshold=1.0,
        parity_label="Equal areas",
        x_label=f"Typical cell footprint area ({ctx.area_label})",
        y_label=f"Cell-centre path area ({ctx.area_label})",
    )
    ctx.drew("anchoring", drawn)

    larger = int((data["equal_area_side"] == "path area larger than footprint").sum())
    smaller = int((data["equal_area_side"] == "path area smaller than footprint").sum())
    retained = int(len(data))
    omitted = int(len(summary) - retained)
    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=data[[
            "identity", "observed_frames", "recording_coverage",
            "median_footprint_area", "centroid_path_area",
            "path_area_over_footprint", "equal_area_side",
        ]],
        subtitle=(
            f"{retained} cells observed in at least {minimum_coverage:.0%} of the "
            f"recording. Above the diagonal: the cell centre explored more than one "
            f"typical footprint area ({larger} cells); below: less ({smaller} cells)."
        ),
        note=(
            "TYPICAL FOOTPRINT\nMedian segmented cell area\nacross observed frames.\n\n"
            "CELL-CENTRE PATH AREA\nArea of the convex hull enclosing\nall observed centre positions.\n\n"
            "DIAGONAL\nThe two areas are equal."
        ),
        note_at={"x": 0.77, "y": 0.68, "linespacing": 1.35},
        footnote=ctx.footnote(
            "Axes are linear from 0 to 1 area unit and logarithmic above 1, "
            "which preserves a zero path area without adding a pseudocount. "
            f"{omitted} cells were omitted because of coverage or missing/invalid areas. "
            "Path area grows with observation duration, so this is a tracking audit, "
            "not a measure of intrinsic motility."
        ),
        readme=f"""## The two measurements

- **Typical cell footprint area** is the median segmented cell area across its
  observed frames (`area_px_median`).
- **Cell-centre path area** is the area of the convex hull enclosing all observed
  centroid positions (`territory_hull_px2`). It is not the cell's occupied
  territory and not its path length.

The diagonal is equal area. A point above it has a centre-path hull larger than
one typical cell footprint; a point below it has a smaller centre-path hull.
The line is a geometric reference, not a fitted model or statistical threshold.

## Comparable observation time

`--min-coverage` retains cells observed for at least that fraction of the full
recording; this build used {minimum_coverage:.0%}. The exact recording coverage
and both plotted areas are in `figure_data.csv`. Path area remains sensitive to
observation duration, so the page belongs to audit/review and reports no
inferential statistics.

## Scale

Both axes use the same zero-preserving scale: linear from 0 to 1 area unit and
logarithmic above 1. No pseudocount is added, and the equal-area line remains a
valid one-to-one reference.
""",
        console=(
            f"cells {retained}  omitted {omitted}  above equal area {larger}  "
            f"below equal area {smaller}  minimum coverage {minimum_coverage:.3f}"
        ),
    )


if __name__ == "__main__":
    run_figure("territory-anchoring", DEFAULT_RUN)
