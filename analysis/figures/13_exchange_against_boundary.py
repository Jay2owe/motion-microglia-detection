"""Figure 13: footprint exchange against boundary length.

A cell that only shuffles its edge exchanges area in proportion to its
perimeter. The reference panel draws what a uniform one- and two-pixel edge
shift would look like, so a point above the lines is doing something else.

    python analysis/figures/13_exchange_against_boundary.py <run>
    ... --metrics area_px        plot against area rather than perimeter
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import surveillance as surveillance_panels


@figure(
    number=13,
    slug="exchange-against-boundary",
    summary="footprint exchange against boundary length",
    title="Footprint exchange against boundary length",
    reads=(Table("cell_frame.csv", module="surveillance"),),
    panels=(
        Panel("scatter", common.scatter_with_margins, block=True,
              title="Exchange by boundary length"),
        Panel("reference", surveillance_panels.boundary_reference,
              title="One- and two-pixel boundary shifts"),
        Panel("per_boundary", common.histogram,
              title="Exchange per boundary length"),
    ),
    options=(
        Option("metrics", default="perimeter_px", cast=str, metavar="COL",
               help="the x axis: perimeter_px or area_px"),
        Option("bins", default=24),
    ),
    grammar="scatter with margins small multiples",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv")
    required = ["identity", "gained_px", "lost_px", "perimeter_px", "area_px"]
    data = data.dropna(subset=required[1:]).copy()
    data["gross_px"] = data["gained_px"] + data["lost_px"]
    by_cell = data.groupby("identity").agg(
        perimeter_px=("perimeter_px", "median"), area_px=("area_px", "median"),
        gross_px=("gross_px", "median"), observed_frames=("frame_index", "count"),
    ).reset_index()
    by_cell["exchange_per_boundary_px"] = by_cell["gross_px"] / by_cell["perimeter_px"]
    by_cell["perimeter"] = by_cell["perimeter_px"].map(ctx.scale.length)
    by_cell["area"] = by_cell["area_px"].map(ctx.scale.area)
    by_cell["gross"] = by_cell["gross_px"].map(ctx.scale.area)
    by_cell["exchange_per_boundary"] = by_cell["gross"] / by_cell["perimeter"]
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    metric = ctx.option("metrics")
    if metric not in {"perimeter_px", "area_px"}:
        raise SystemExit("--metrics must be perimeter_px or area_px")
    display_metric = "perimeter" if metric == "perimeter_px" else "area"
    if "scatter" in axes:
        ax, top, right = ctx.block(fig, axes["scatter"], "scatter", by_cell[display_metric], by_cell["gross"],
                                          ctx.theme, role="surveillance",
                                          bins=int(ctx.option("bins"))).axes
        ax.set_xlabel(f"Median {'perimeter' if metric == 'perimeter_px' else 'area'} "
                      f"({ctx.length_label if metric == 'perimeter_px' else ctx.area_label})")
        ax.set_ylabel(f"Gross exchange ({ctx.area_label})")
        axes["scatter"] = ax
    if "reference" in axes:
        common.scatter(axes["reference"], by_cell["perimeter"], by_cell["gross"],
                       ctx.theme, role="surveillance")
        surveillance_panels.boundary_reference(
            axes["reference"], by_cell["perimeter"], ctx.theme,
            pixels=(ctx.scale.length(1), ctx.scale.length(2)),
        )
        axes["reference"].set_xlabel(f"Median perimeter ({ctx.length_label})")
        axes["reference"].set_ylabel(f"Gross exchange ({ctx.area_label})")
    if "per_boundary" in axes:
        common.histogram(axes["per_boundary"], by_cell["exchange_per_boundary"],
                         ctx.theme, bins=int(ctx.option("bins")), role="surveillance",
                         x_label=f"Exchange per boundary length ({ctx.length_label})", y_label="Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=by_cell,
        subtitle=f"One point per identity; {len(by_cell)} identities measured.",
        footnote=ctx.provenance_footnote(),
        auxiliary=ctx.provenance_auxiliary(),
    )


if __name__ == "__main__":
    run_figure("exchange-against-boundary")
