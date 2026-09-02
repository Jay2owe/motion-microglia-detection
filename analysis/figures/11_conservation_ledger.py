"""Figure 11: gross footprint exchange against net size change.

A cell can replace a fifth of its own footprint between two frames and finish
the transition the size it started. Gross exchange and net change go on one
axis so that cancellation is visible rather than inferred.

    python analysis/figures/11_conservation_ledger.py <run>
    ... --panels ledger          the timeline alone
    ... --metrics area_px        which size the third panel plots against
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _metrics import axis_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import surveillance as surveillance_panels


@figure(
    number=11,
    slug="conservation-ledger",
    summary="gross footprint exchange against net size change",
    title="Gross footprint exchange against net size change",
    reads=(Table("cell_frame.csv", module="surveillance"),),
    panels=(
        Panel("ledger", surveillance_panels.exchange_ledger,
              title="Exchanged area versus net size change"),
        Panel("cancelled", surveillance_panels.cancelled_fraction,
              title="Opposing edge motion"),
        Panel("against_size", common.scatter_with_margins, block=True,
              title="Exchange by cell size"),
    ),
    options=(
        Option("metrics", default="area_px", cast=str, metavar="COL",
               help="which measured size the third panel plots exchange against"),
        Option("bins", default=24),
        Option("hour_ticks", default=24.0),
    ),
    grammar="timeline small multiples",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv")
    wanted = ["identity", "frame_index", "hours", "gained_px", "lost_px", "area_change_px"]
    data = data.dropna(subset=wanted[3:]).copy()
    data["gained_area"] = data["gained_px"].map(ctx.scale.area)
    data["lost_area"] = data["lost_px"].map(ctx.scale.area)
    data["net_area"] = data["area_change_px"].map(ctx.scale.area)
    data["gross_px"] = data["gained_area"] + data["lost_area"]
    data["net_px"] = data["net_area"]
    data["cancelled_fraction"] = np.where(
        data["gross_px"] > 0, 1 - data["net_px"].abs() / data["gross_px"], np.nan
    )
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "ledger" in axes:
        surveillance_panels.exchange_ledger(
            axes["ledger"], data["hours"], data, ctx.theme,
            gross_column="gained_area", lost_column="lost_area", area_change_column="net_area",
            y_label=f"Area exchanged ({ctx.area_label})", hour_ticks=ctx.hour_ticks,
        )
    if "cancelled" in axes:
        surveillance_panels.cancelled_fraction(axes["cancelled"], data["hours"], data, ctx.theme,
                                                hour_ticks=ctx.hour_ticks)
    if "against_size" in axes:
        metric = ctx.option("metrics")
        if metric not in data:
            raise SystemExit(f"--metrics {metric} is not in cell_frame.csv")
        data["display_size"] = (data[metric].map(ctx.scale.area) if metric.endswith("_px") and "area" in metric
                                else data[metric].map(ctx.scale.length) if metric.endswith("_px") else data[metric])
        by_cell = data.groupby("identity").agg(gross_px=("gross_px", "median"),
                                                size=("display_size", "median")).reset_index()
        ax, top, right = ctx.block(fig, axes["against_size"], "against_size", by_cell["size"],
                                          by_cell["gross_px"], ctx.theme, role="surveillance",
                                          bins=int(ctx.option("bins"))).axes
        ax.set_xlabel(f"Median size ({ctx.area_label})" if metric.endswith("_px") and "area" in metric
                      else axis_label(metric, ctx.interval, wrap=False))
        ax.set_ylabel(f"Gross exchange ({ctx.area_label})")
        axes["against_size"] = ax
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data[wanted + ["gross_px", "net_px", "cancelled_fraction"]],
        subtitle=f"{data['identity'].nunique()} identities; one row per measured transition.",
        footnote=ctx.provenance_footnote(),
        auxiliary=ctx.provenance_auxiliary(),
    )


if __name__ == "__main__":
    run_figure("conservation-ledger")
