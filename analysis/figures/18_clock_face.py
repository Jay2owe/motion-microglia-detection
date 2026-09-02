"""Figure 18: peak time and relative amplitude on a 24 hour dial.

Each cell is an arrow: its direction is when the fitted signal peaked, its
length is how strong the rhythm was. The dial wraps at the period the fit
assumed, which is stated on the figure rather than implied by it.

    python analysis/figures/18_clock_face.py <run>
    ... --metrics area_px        a different fitted measurement
    ... --panels amplitude       the histogram alone
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import rhythms as rhythm_panels


@figure(
    number=18,
    slug="clock-face",
    summary="peak time and relative amplitude on a 24 hour dial",
    title="Peak time and relative amplitude on a 24 hour dial",
    reads=(Table("rhythms.csv", module="rhythms"),
           Table("rhythms_population.csv", module="rhythms")),
    panels=(
        Panel("dial", rhythm_panels.phase_dial, polar=True,
              title="Peak time and rhythm strength for each cell"),
        Panel("rose", common.rose, polar=True, title="Peak-time counts"),
        Panel("amplitude", common.histogram, title="Rhythm strength"),
    ),
    options=(
        Option("metrics", default="corrected_mean", cast=str, metavar="COL",
               help="which fitted measurement the dial draws"),
        Option("bins", default=24),
        Option("hour_ticks", default=24.0),
    ),
    grammar="polar vectors and rose histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    rhythms = ctx.table("rhythms.csv")
    population_path = ctx.table_path("rhythms_population.csv")
    metric = ctx.option("metrics")
    data = rhythms[rhythms["metric"] == metric].copy()
    if data.empty:
        raise SystemExit(f"--metrics {metric} was not fitted; available: {', '.join(sorted(rhythms['metric'].unique()))}")
    # ``cosinor_relative_amplitude`` keeps its own name here. It used to be
    # shortened to ``relative_amplitude``, which now belongs to a different
    # number on the same table - the non-parametric (M10 - L5) / (M10 + L5) -
    # and one heading over two quantities is exactly the confusion the two
    # names exist to prevent.
    figure_data = data[["identity", "metric", "cosinor_peak_hour", "cosinor_relative_amplitude",
                        "cosinor_r_squared", "rhythmic_both", "fixed_period_hours"]].rename(columns={
                            "cosinor_peak_hour": "peak_hour",
                            "cosinor_r_squared": "r_squared",
                            "fixed_period_hours": "period_hours_assumed",
                        })
    panels = ctx.panels()
    if panels.keys == ["dial", "rose", "amplitude"]:
        height = 9.8
        fig = plt.figure(figsize=ctx.theme.canvas(13.8, height))
        grid = fig.add_gridspec(
            2, 2, left=0.14, right=0.94, bottom=1.55 / height,
            top=1 - 1.65 / height, hspace=0.70, wspace=0.34,
        )
        axes = {
            "dial": fig.add_subplot(grid[0, 0], projection="polar"),
            "rose": fig.add_subplot(grid[0, 1], projection="polar"),
            "amplitude": fig.add_subplot(grid[1, :]),
        }
        for name, ax in axes.items():
            ax.set_title(ctx.spec.panel(name).heading(), loc="left", fontsize=ctx.theme.size("panel"),
                         fontweight="bold")
    else:
        fig, axes = ctx.layout(panels)
    period = float(data["fixed_period_hours"].iloc[0])
    if "dial" in axes:
        rhythm_panels.phase_dial(axes["dial"], figure_data["peak_hour"],
                                 figure_data["cosinor_relative_amplitude"], ctx.theme,
                                 passed=figure_data["rhythmic_both"], period_hours=period,
                                 hour_ticks=ctx.hour_ticks,
                                 radius_label="Fitted amplitude over the mean")
        common.semantic_legend(
            axes["dial"], ctx.theme,
            handles=[
                Line2D([], [], color=ctx.theme.colour("rhythmic"), marker=">", linestyle="none",
                       label="Passed both rhythm tests"),
                Line2D([], [], color=ctx.theme.colour("arrhythmic"), marker=">", markerfacecolor="none",
                       linestyle="none", label="Did not pass both tests"),
                Line2D([], [], color=ctx.theme.colour("highlight"), linewidth=ctx.theme.stroke("emphasis"),
                       label="Amplitude-weighted mean peak"),
            ], labels=["Passed both rhythm tests", "Did not pass both tests", "Amplitude-weighted mean peak"],
            location="below",
        )
    if "rose" in axes:
        counts = common.rose(
            axes["rose"], figure_data["peak_hour"], ctx.theme,
            bins=int(ctx.option("bins")), period=period, role="reporter",
        ).data["count"].to_numpy()
        radians = 2 * np.pi * figure_data["peak_hour"].to_numpy(float) / period
        direction = float(np.angle(np.sum(np.exp(1j * radians))) % (2 * np.pi))
        concentration = float(np.abs(np.sum(np.exp(1j * radians))) / max(len(radians), 1))
        axes["rose"].annotate(
            "Mean peak time",
            xy=(direction, max(float(np.max(counts)), 1.0) * concentration),
            xytext=(16, 12), textcoords="offset points",
            fontsize=ctx.theme.size("caption"), color=ctx.theme.colour("ink"),
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
        )
        axes["rose"]._semantic_legend_handled = True
    if "amplitude" in axes:
        common.histogram(axes["amplitude"], figure_data["cosinor_relative_amplitude"], ctx.theme,
                         bins=int(ctx.option("bins")), role="reporter",
                         x_label="Fitted amplitude over the mean", y_label="Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"{len(figure_data)} fitted identities; the dial wraps at the assumed {period:g} h period.",
    )


if __name__ == "__main__":
    run_figure("clock-face")
