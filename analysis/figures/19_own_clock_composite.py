"""Figure 19: wall-clock traces and each cell's own peak-aligned composite.

The same signal twice: once on recording time, once with every cell shifted to
its own fitted peak. Aligning on a fit guarantees the defining measurement will
sharpen, which is why it is drawn beside the measurements that were carried
along and did not have to.

    python analysis/figures/19_own_clock_composite.py <run>
    ... --metrics corrected_mean,area_px    the first defines the clock
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import rhythms as rhythm_panels


@figure(
    number=19,
    slug="own-clock-composite",
    summary="wall-clock traces and each cell's own peak-aligned composite",
    title="Wall-clock traces and each cell's own peak-aligned composite",
    reads=(Table("cell_frame.csv", module="rhythms"),
           Table("rhythms.csv", module="rhythms")),
    panels=(
        Panel("wall_clock", rhythm_panels.population_mean,
              title="Reporter signal on recording time"),
        Panel("realigned", rhythm_panels.phase_aligned,
              title="Reporter signal aligned to its own peak"),
        Panel("carried", rhythm_panels.phase_aligned,
              title="Other measurements on the reporter-defined clock"),
    ),
    options=(
        Option("metrics",
               default=["corrected_mean", "turnover_index", "ramification_index"],
               help="the first defines the clock; the rest are carried on it"),
        Option("hour_ticks", default=24.0),
    ),
    grammar="aligned trace small multiples",
)
def build(ctx: FigureContext) -> FigureResult:
    frame = ctx.table("cell_frame.csv")
    rhythms = ctx.table("rhythms.csv")
    metrics = ctx.option("metrics")
    metrics = [metric for metric in metrics if metric in frame]
    if not metrics:
        raise SystemExit("none of --metrics is in cell_frame.csv")
    defining = metrics[0]
    fits = rhythms[rhythms["metric"] == defining].dropna(subset=["cosinor_peak_hour"])
    peak_map = dict(zip(fits["identity"], fits["cosinor_peak_hour"]))
    panels = ctx.panels()
    if "carried" in panels and "realigned" not in panels:
        raise SystemExit("the carried panel requires realigned, because its phase axis is defined there")
    fig, axes = ctx.layout(panels)
    auxiliaries = {}
    rows = []
    if "wall_clock" in axes:
        wall = rhythm_panels.population_mean(axes["wall_clock"], frame, ctx.theme,
                                             column=defining, hour_ticks=ctx.hour_ticks,
                                             label=semantic_label(defining)).data
        auxiliaries["wall_clock.csv"] = wall
    if "realigned" in axes:
        aligned = rhythm_panels.phase_aligned(axes["realigned"], frame, ctx.theme,
                                              column=defining, peak_hours=peak_map,
                                              hour_ticks=ctx.hour_ticks,
                                              label=semantic_label(defining)).data
        aligned.insert(0, "metric", defining)
        aligned["aligned_on"] = defining
        rows.append(aligned)
        axes["realigned"].set_ylabel("Within-cell change\n(standard deviations)")
    if "carried" in axes:
        for metric in metrics[1:]:
            aligned = rhythm_panels.phase_aligned(axes["carried"], frame, ctx.theme,
                                                  column=metric, peak_hours=peak_map,
                                                  hour_ticks=ctx.hour_ticks,
                                                  role=role_for(metric),
                                                  label=semantic_label(metric)).data
            aligned.insert(0, "metric", metric)
            aligned["aligned_on"] = defining
            rows.append(aligned)
        axes["carried"].set_ylabel("Within-cell change\n(standard deviations)")
    figure_data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["metric", "phase_hour", "mean_z", "lo", "hi", "cells", "aligned_on"]
    )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Traces are aligned on each identity's fitted {semantic_label(defining).lower()} peak.",
        footnote="Alignment guarantees sharpening of the defining metric; carried metrics are the comparison.",
        auxiliary=auxiliaries,
    )


if __name__ == "__main__":
    run_figure("own-clock-composite")
