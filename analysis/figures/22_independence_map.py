"""Figure 22: phase difference against distance between cells.

If neighbouring cells kept the same time, phase difference would rise with
separation. The random-pair level is drawn so that "no relationship" has
something to look like.

    python analysis/figures/22_independence_map.py <run>
    ... --trace-luts viridis     the colour map of the phase field
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np

from analysis import circadian as workbench
from _derive import _scaled_field
from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import coupling as coupling_panels


@figure(
    number=22,
    slug="independence-map",
    summary="phase difference against distance between cells",
    title="Phase difference against distance between cells",
    reads=(Table("cell_frame.csv", module="coupling"),
           Table("coupling.csv", module="coupling"),
           Table("rhythms.csv", module="rhythms")),
    panels=(
        Panel("distance", coupling_panels.phase_difference,
              title="Phase difference by cell separation"),
        Panel("field", common.phase_map,
              title="Peak time across the field"),
        Panel("measures", title="Phase difference by measurement"),
    ),
    options=(
        Option("metrics", default=["corrected_mean"],
               help="which fitted measurements the pair comparison covers"),
        Option("bins", default=24,
               help="hexbin resolution of the distance panel"),
        Option("trace_luts", default="twilight_shifted", cast=str, metavar="CMAP",
               help="cyclic colour map for the phase field"),
    ),
    grammar="hexbin and cyclic field map",
)
def build(ctx: FigureContext) -> FigureResult:
    coupling = ctx.table("coupling.csv")
    frame = ctx.table("cell_frame.csv")
    rhythms = ctx.table("rhythms.csv")
    metrics = ctx.option("metrics")
    data = coupling[coupling["metric"].isin(metrics)].copy()
    if data.empty:
        raise SystemExit(f"coupling.csv has no requested metric; available: {', '.join(sorted(coupling['metric'].unique()))}")
    if "phase_difference_fraction" not in data:
        reference = data.get("pair_phase_reference_period_hours", 24.0)
        data["phase_difference_fraction"] = data["phase_difference"] / reference
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    null_level = float(data["phase_difference_fraction"].median())
    if "distance" in axes:
        coupling_panels.phase_difference(axes["distance"], data["distance"], data["phase_difference_fraction"],
                                         ctx.theme, period=1.0, null_level=null_level,
                                         gridsize=int(ctx.option("bins")),
                                         x_label=f"Distance between cells ({ctx.length_label})",
                                         y_label="Peak difference (fraction of own cycle)")
        if axes["distance"].collections:
            common.inset_colour_bar(axes["distance"], axes["distance"].collections[0], ctx.theme,
                                    label="Cell pairs per hexagon")
    if "field" in axes:
        metric = metrics[0]
        phases = rhythms[rhythms["metric"] == metric].copy()
        if "best_phase_fraction" not in phases:
            period = phases.get("best_period_hours", 24.0)
            peak = phases.get("best_phase_hours", phases.get("cosinor_peak_hour"))
            phases["best_phase_fraction"] = np.mod(peak, period) / period
        phases = phases[["identity", "best_phase_fraction"]]
        positions = frame.groupby("identity")[["centroid_x", "centroid_y"]].median().reset_index()
        positions["centroid_x"] = positions["centroid_x"].map(ctx.scale.length)
        positions["centroid_y"] = positions["centroid_y"].map(ctx.scale.length)
        mapped = positions.merge(phases, on="identity", how="inner")
        phase_handle = common.phase_map(
            axes["field"], mapped[["centroid_x", "centroid_y"]],
            mapped["best_phase_fraction"], ctx.theme, field=_scaled_field(ctx),
            period=1.0, phase_unit="cycle", cmap=ctx.option("trace_luts"),
            labels=mapped["identity"],
            x_label=f"X position ({ctx.length_label})",
            y_label=f"Y position ({ctx.length_label})",
        ).extra["handle"]
        common.inset_colour_bar(axes["field"], phase_handle, ctx.theme,
                                label="Peak position within own cycle",
                                ticks=[0, 0.25, 0.5, 0.75, 1])
    if "measures" in axes:
        for metric, group in data.groupby("metric"):
            axes["measures"].scatter(group["distance"], group["phase_difference_fraction"],
                                     label=semantic_label(metric), alpha=0.5,
                                     color=ctx.theme.colour(role_for(metric)),
                                     s=ctx.theme.point_area(0.9))
        axes["measures"].set_xlabel(f"Distance between cells ({ctx.length_label})")
        axes["measures"].set_ylabel("Peak difference (fraction of own cycle)")
        common.semantic_legend(axes["measures"], ctx.theme, location="inside")
    figure_columns = [
        "identity_a", "identity_b", "distance", "metric", "phase_difference",
        "phase_difference_fraction", "period_a_hours", "period_b_hours",
        "pair_phase_reference_period_hours", "period_estimation_method", "overlap_frames",
    ]
    figure_data = data[[column for column in figure_columns if column in data]]
    estimator = (
        str(data["period_estimation_method"].dropna().mode().iloc[0])
        if "period_estimation_method" in data and data["period_estimation_method"].notna().any()
        else "lomb"
    )
    estimator_label = workbench.PERIOD_METHODS.get(
        estimator, {"label": estimator})["label"]
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=(f"One row per identity pair; {estimator_label} supplies each cell's "
                  f"period and phase. Random-pair level {null_level:.2f} cycles."),
    )


if __name__ == "__main__":
    run_figure("independence-map")
