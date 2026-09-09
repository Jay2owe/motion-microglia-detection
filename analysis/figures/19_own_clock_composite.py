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
from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS

from panels import rhythms as rhythm_panels


@figure(
    number=19,
    slug="own-clock-composite",
    summary="recording-time traces and per-cell peak-aligned population traces",
    title="Recording-time and per-cell peak-aligned population traces",
    reads=(Table("cell_frame.csv", module="rhythms"),
           Table("rhythms.csv", module="rhythms")),
    panels=(
        Panel("wall_clock", rhythm_panels.population_mean,
              title="Reporter signal on recording time"),
        Panel("realigned", rhythm_panels.phase_aligned,
              title="Reporter signal aligned to its own peak"),
        Panel("carried", rhythm_panels.phase_aligned,
              title="Other measurements aligned to reporter peak time"),
    ),
    options=(
        Option("metrics",
               default=["corrected_mean", "turnover_index", "ramification_index"],
               help="the first defines the clock; the rest are carried on it"),
        Option("hour_ticks", default=24.0),
        Option("detrend", default=None),
        Option("detrend_window_hours", default=None),
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
    rhythm_params = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    detrending = workbench.detrend_settings(
        rhythm_params,
        method=ctx.option("detrend"),
        window_hours=ctx.option("detrend_window_hours"),
    )
    fits = rhythms[rhythms["metric"] == defining].copy()
    peak_column = (
        "best_phase_hours" if "best_phase_hours" in fits else "cosinor_peak_hour"
    )
    fits = fits.dropna(subset=[peak_column])
    peak_map = dict(zip(fits["identity"], fits[peak_column]))
    estimator = str(rhythm_params.get(
        "period_estimation_method", rhythm_params.get("primary_rhythm_test", "lomb")
    ))
    estimator_label = (
        str(fits["best_method_label"].dropna().mode().iloc[0])
        if "best_method_label" in fits and fits["best_method_label"].notna().any()
        else workbench.PERIOD_METHODS.get(estimator, {"label": estimator})["label"]
    )
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
                                              label=semantic_label(defining),
                                              detrend=detrending["detrend"],
                                              detrend_options=detrending,
                                              detrend_window_hours=detrending["detrend_window_hours"]).data
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
                                                  label=semantic_label(metric),
                                                  detrend=detrending["detrend"],
                                                  detrend_options=detrending,
                                                  detrend_window_hours=detrending["detrend_window_hours"]).data
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
        subtitle=(f"Traces are aligned on each identity's {semantic_label(defining).lower()} "
                  f"peak estimated by {estimator_label}."),
        footnote=(
            "Alignment guarantees sharpening of the defining metric; carried metrics "
            f"are the comparison. Circadian Workbench detrending: "
            f"{detrending['detrend'].replace('_', ' ')}, "
            f"{detrending['detrend_window_hours']:g} h window."
        ),
        auxiliary=auxiliaries,
    )


if __name__ == "__main__":
    run_figure("own-clock-composite")
