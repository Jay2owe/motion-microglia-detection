"""Figure 20: each cell's first cycle against its second.

A rhythm that is real repeats. Each cell is fitted twice, once per cycle, and
the two peak times are plotted against each other; a cell on the diagonal kept
time with itself.

    python analysis/figures/20_second_verse.py <run>
    ... --cells 8                how many cells the polar panel traces
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _derive import _cycle_fit
from _options import commas
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import rhythms as rhythm_panels


@figure(
    number=20,
    slug="second-verse",
    summary="each cell's first cycle against its second",
    title="Each cell's first cycle against its second",
    reads=(Table("cell_frame.csv", module="rhythms"),),
    panels=(
        Panel("loops", rhythm_panels.cycle_loops, polar=True,
              title="First- and second-cycle traces"),
        Panel("repeat", common.scatter,
              title="Peak time in the first versus second cycle"),
        Panel("agreement", common.histogram,
              title="Within-cell peak-time difference"),
    ),
    options=(
        Option("metrics", default="corrected_mean", cast=str, metavar="COL",
               help="which fitted measurement is compared cycle to cycle"),
        Option("cells", default="4", cast=str),
        Option("bins", default=12),
        Option("hour_ticks", default=24.0),
    ),
    grammar="polar loops scatter and histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    frame = ctx.table("cell_frame.csv")
    metric = ctx.option("metrics")
    if metric not in frame:
        raise SystemExit(f"--metrics {metric} is not in cell_frame.csv")
    period = float(ctx.module_params("rhythms").get("fixed_period_hours", 24.0))
    rows = []
    for identity, group in frame[["identity", "hours", metric]].dropna().groupby("identity", sort=True):
        group = group.sort_values("hours")
        relative = group["hours"].to_numpy(float) - float(group["hours"].min())
        values = group[metric].to_numpy(float)
        fits = []
        counts = []
        for cycle in (0, 1):
            keep = (relative >= cycle * period) & (relative < (cycle + 1) * period)
            fits.append(_cycle_fit(relative[keep] - cycle * period, values[keep], period))
            counts.append(int(keep.sum()))
        if not fits[0] or not fits[1]:
            continue
        difference = abs((fits[0]["cosinor_peak_hour"] - fits[1]["cosinor_peak_hour"]) % period)
        difference = min(difference, period - difference)
        rows.append({
            "identity": int(identity), "metric": metric,
            "cycle_1_peak_hour": fits[0]["cosinor_peak_hour"],
            "cycle_2_peak_hour": fits[1]["cosinor_peak_hour"],
            "circular_difference": difference,
            "cycle_1_amplitude": fits[0]["cosinor_amplitude"],
            "cycle_2_amplitude": fits[1]["cosinor_amplitude"],
            "frames_cycle_1": counts[0], "frames_cycle_2": counts[1],
        })
    data = pd.DataFrame(rows)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "loops" in axes and not data.empty:
        requested_cells = str(ctx.option("cells"))
        try:
            selected = data.sort_values("frames_cycle_1", ascending=False)["identity"].head(int(requested_cells))
        except ValueError:
            selected = pd.Series([int(value) for value in commas(requested_cells)])
        for identity in selected:
            group = frame[frame["identity"] == identity].dropna(subset=[metric]).sort_values("hours")
            rhythm_panels.cycle_loops(axes["loops"], group["hours"], group[metric], ctx.theme,
                                      period_hours=period, hour_ticks=ctx.hour_ticks)
    if "repeat" in axes and not data.empty:
        common.scatter(axes["repeat"], data["cycle_1_peak_hour"], data["cycle_2_peak_hour"],
                       ctx.theme, role="reporter")
        axes["repeat"].collections[-1].set_label("Each cell")
        axes["repeat"].plot([0, period], [0, period], color=ctx.theme.colour("reference"),
                            linestyle=(0, (5, 3)), label="Same peak time in both cycles")
        axes["repeat"].set(xlim=(0, period), ylim=(0, period),
                           xlabel="First-cycle peak (h)", ylabel="Second-cycle peak (h)")
    if "agreement" in axes and not data.empty:
        common.histogram(axes["agreement"], data["circular_difference"], ctx.theme,
                         bins=int(ctx.option("bins")), role="reporter",
                         x_label="Circular peak difference (h)", y_label="Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=f"Within-cell comparison at an assumed {period:g} h wrap; {len(data)} identities cover two fits.",
    )


if __name__ == "__main__":
    run_figure("second-verse")
