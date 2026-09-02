"""Figure 16: centroid territory against median footprint area.

The hull a cell's centroid wanders over, against the area the cell occupies at
any one moment. A cell anchored in place has a hull smaller than itself.

    python analysis/figures/16_territory_anchoring.py <run>
    ... --size observed_frames   let a column set the point size
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import motility as motility_panels


@figure(
    number=16,
    slug="territory-anchoring",
    summary="centroid territory against median footprint area",
    title="Centroid territory against median footprint area",
    reads=(Table("cell_summary.csv", module="motility"),),
    panels=(
        Panel("anchoring", motility_panels.wander_against_size,
              title="Centroid territory versus cell footprint"),
        Panel("margins", common.scatter_with_margins, block=True,
              title="Distribution and relationship"),
        Panel("ratio", common.histogram, title="Territory-to-footprint ratio"),
    ),
    options=(
        Option("bins", default=24),
        Option("size", default="", cast=str,
               help="which column sets point size on the margins panel"),
    ),
    grammar="scatter with margins and histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    summary = ctx.table("cell_summary.csv")
    required = ["identity", "territory_hull_px2", "area_px_median", "net_displacement_px",
                "equivalent_diameter_px_median", "straightness", "msd_alpha", "observed_frames"]
    available = [column for column in required if column in summary]
    data = summary[available].dropna(subset=["territory_hull_px2", "area_px_median"]).copy()
    data["hull_over_area"] = data["territory_hull_px2"] / data["area_px_median"]
    diameter = "equivalent_diameter_px_median" if "equivalent_diameter_px_median" in data else None
    data["displacement_over_radius"] = (
        data["net_displacement_px"] / (data[diameter] / 2) if diameter else np.nan
    )
    data["territory_hull"] = data["territory_hull_px2"].map(ctx.scale.area)
    data["median_footprint"] = data["area_px_median"].map(ctx.scale.area)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "anchoring" in axes:
        motility_panels.wander_against_size(
            axes["anchoring"], data["territory_hull"], data["median_footprint"],
            ctx.theme, x_label=f"Median footprint area ({ctx.area_label})",
            y_label=f"Centroid hull ({ctx.area_label})",
        )
    if "margins" in axes:
        size_column = ctx.option("size")
        sizes = None
        if size_column:
            if size_column not in data:
                raise SystemExit(f"--size {size_column} is not in cell_summary.csv")
            values = data[size_column].to_numpy(float)
            scaled = (values - np.nanmin(values)) / max(float(np.nanmax(values) - np.nanmin(values)), 1e-12)
            sizes = ctx.theme.point_area(0.5 + 1.5 * scaled)
        ax, top, right = ctx.block(fig, axes["margins"], "margins", data["median_footprint"],
                                          data["territory_hull"], ctx.theme, role="motility",
                                          sizes=sizes, bins=int(ctx.option_or("bins", 20)),
                                          log_x=True).axes
        ax.set_yscale("log")
        ax.set_xlabel(f"Median footprint area ({ctx.area_label})")
        ax.set_ylabel(f"Centroid hull ({ctx.area_label})")
        axes["margins"] = ax
    if "ratio" in axes:
        common.histogram(axes["ratio"], data["hull_over_area"], ctx.theme,
                         bins=int(ctx.option("bins")), role="motility",
                         x_label="Centroid hull / median footprint", y_label="Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=f"One point per track; {len(data)} identities with comparable areas.",
    )


if __name__ == "__main__":
    run_figure("territory-anchoring")
