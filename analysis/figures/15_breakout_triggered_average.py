"""Figure 15: measurements aligned on each cell's largest breakout.

Every cell contributes its own biggest excursion, so the average is of an event
rather than of a time. The control panel repeats the whole thing on each cell's
second-largest excursion, which is the comparison that says whether the shape
of the first is particular to it.

    python analysis/figures/15_breakout_triggered_average.py <run>
    ... --window 24              frames either side of the event
    ... --metrics area_px,solidity
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _derive import _aligned_curves
from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import surveillance as surveillance_panels


@figure(
    number=15,
    slug="breakout-triggered-average",
    summary="measurements aligned on each cell's largest breakout",
    title="Measurements aligned on each cell's largest breakout",
    reads=(Table("cell_frame.csv", module="surveillance"),),
    panels=(
        Panel("triggered", common.event_average,
              title="Response around the largest excursion"),
        Panel("alignment", title="When each cell's event occurred"),
        Panel("control", common.event_average,
              title="Response around the second-largest excursion"),
    ),
    options=(
        Option("metrics",
               default=["area_px", "ramification_index", "corrected_mean"]),
        Option("window", default=12),
    ),
    grammar="event average small multiples",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv")
    gross = data["gained_px"] + data["lost_px"]
    data["cancelled_fraction"] = np.where(gross > 0, 1 - data["area_change_px"].abs() / gross, np.nan)
    metrics = ctx.option("metrics")
    metrics = [metric for metric in metrics if metric in data]
    window = int(ctx.option("window"))
    primary = surveillance_panels.breakout_events(data, window=window, rank=1)
    control = surveillance_panels.breakout_events(data, window=window, rank=2)
    for aligned in (primary, control):
        for metric in metrics:
            aligned[metric] = aligned.groupby("identity")[metric].transform(
                lambda values: (values - values.mean()) / (values.std(ddof=0) or 1.0)
            )
    figure_data, matrices, offsets = _aligned_curves(primary, metrics)
    control_data, control_matrices, _ = _aligned_curves(control, metrics)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "triggered" in axes:
        for metric in metrics:
            common.event_average(axes["triggered"], offsets, matrices[metric], ctx.theme,
                                 role=role_for(metric), bootstrap=200,
                                 null_curves=control_matrices.get(metric),
                                 label=semantic_label(metric))
        axes["triggered"].set_ylabel("Within-cell change\n(standard deviations)")
    if "alignment" in axes:
        axes["alignment"].scatter(primary["event_frame_index"] * ctx.interval / 60,
                                  primary["identity"], color=ctx.theme.colour("highlight"),
                                  s=ctx.theme.point_area(0.8))
        axes["alignment"].set_xlabel("Event hour")
        axes["alignment"].set_ylabel("Identity")
    if "control" in axes:
        for metric in metrics:
            common.event_average(axes["control"], offsets, control_matrices[metric], ctx.theme,
                                 role=role_for(metric), bootstrap=200,
                                 label=semantic_label(metric))
        axes["control"].set_ylabel("Within-cell change\n(standard deviations)")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Each identity is aligned on its own largest excursion; window {window} frames.",
        auxiliary={"second_excursion_control.csv": control_data},
    )


if __name__ == "__main__":
    run_figure("breakout-triggered-average")
