"""Figure 1: reporter intensity cycles, and the cells agree on when it peaks.

Two panels, either of which can be the whole page::

    python analysis/figures/01_cd68_reporter_rhythm.py <run>
    ... --panels raster           just the per-cell raster
    ... --panels phase            just the peak-time histogram
    ... --metrics area_px         a different measurement's rhythm
    ... --bins 24                 hourly bands in the histogram
    ... --trace-luts RdBu_r       the map the raster is coloured through
    ... --hour-ticks 12

The panels are ``panels.rhythms.trace_raster`` and
``panels.rhythms.phase_histogram``; this file chooses the cells, the order and
the layout, and neither panel knows it is on a page with the other.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _bundle import provenance_note
from _metrics import describe
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import category_strip, colour_bar, resolve_look
from panels import rhythms as rhythm_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=1,
    slug="cd68-reporter-rhythm",
    summary="reporter intensity cycles, and the cells agree on when it peaks",
    title="{metric} over {span} h, one row per cell",
    grammar="per-cell trace raster beside a histogram of peak times",
    reads=(Table("cell_frame.csv", module="motility"),
           Table("rhythms.csv", module="rhythms"),
           Table("rhythms_population.csv", module="rhythms"),
           Table("rhythms_null.csv", module="rhythms")),
    panels=(Panel("raster", rhythm_panels.trace_raster,
                  title="One row per cell, detrended"),
            Panel("phase", rhythm_panels.phase_histogram,
                  title="When each cell peaks")),
    options=(
        Option("metrics", default="corrected_mean", cast=str, metavar="COL",
               help="which fitted measurement the page draws"),
        Option("bins", default=12),
        Option("trace_luts", default=None, cast=str, metavar="NAME",
               help="colour map the raster is drawn through"),
        # Unset means each axis keeps its own default: 24 h for elapsed time,
        # from the theme, and 6 h for the phase histogram, which is only 24 h
        # wide and would otherwise carry two ticks. Both are factors of a day,
        # so every tick is the same time of day either way.
        Option("hour_ticks", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    metric = ctx.option("metrics")
    bins = ctx.option("bins")
    raster_lut = ctx.option("trace_luts")
    hour_ticks = ctx.option("hour_ticks")
    # The histogram's own 6 h step stands until somebody chooses a step, whether
    # they choose it on the command line or in the run's figure options.
    phase_ticks = ctx.option_or("hour_ticks", 6.0)

    panels = ctx.panels()

    cell_frame = ctx.table("cell_frame.csv")
    rhythms = ctx.table("rhythms.csv")
    population = ctx.table("rhythms_population.csv")
    null = ctx.table("rhythms_null.csv")
    params = ctx.module_params("rhythms")

    if metric not in set(rhythms["metric"]):
        raise SystemExit(
            f"--metrics {metric} was not fitted in this run; rhythms.csv holds "
            f"{', '.join(sorted(set(rhythms['metric'])))}"
        )

    label = describe(metric).label
    fits = rhythms[rhythms["metric"] == metric].copy()
    stats_all = population[
        (population["metric"] == metric) & (population["population"] == "all_tested")
    ].iloc[0]

    null_row = null[null["metric"] == metric].iloc[0]
    search = params["period_search_hours"]

    cells_tested = int(stats_all["cells"])
    span_hours = float(ctx.summary["hours_covered"])
    cycles = float(ctx.summary["rhythms"]["cycles_covered"])
    median_period = float(fits["lombscargle_period_hours"].median(skipna=True))

    # What the outlines behind these traces rest on. Empty for a run with no
    # provenance, and then the figure says nothing about it rather than hedging.
    provenance = provenance_note(ctx.summary)

    footnote = (
        f"{span_hours:.1f} h is {cycles:.1f} cycles of a 24 h rhythm, so the "
        f"period is indicative rather than resolved; every row carries "
        f"cycles_covered and period_underdetermined.\nBoth tests fire on "
        f"{null_row['observed_rate_both']:.1%} of real traces and "
        f"{null_row['false_positive_rate_both']:.1%} of matched surrogates."
        + (f"\n{provenance}" if provenance else "")
    )

    # The plotted table is whichever panel leads the page, because that is what
    # `figure_data.csv` is for: the numbers a reader would check the figure against.
    # The other panel's table keeps its own name beside it.
    lead = panels.keys[0]

    rhythmic_ids = fits[fits["rhythmic_both"]].sort_values(
        "cosinor_peak_hour")["identity"].tolist()
    other_ids = fits[~fits["rhythmic_both"]].sort_values(
        "cosinor_peak_hour")["identity"].tolist()
    order = rhythmic_ids + other_ids

    matrix = None
    hours_axis = None
    traces_table = None
    rows = 0
    if "raster" in panels:
        # Each cell's trace put through exactly the transform the rhythm test saw.
        traces = rhythm_panels.detrended_traces(cell_frame, fits["identity"], metric)
        matrix = traces.pivot(
            index="identity", columns="hours", values="detrended_z").reindex(order)
        hours_axis = matrix.columns.to_numpy(float)

        traces_table = (
            matrix.reset_index()
            .melt(id_vars="identity", var_name="hours", value_name="detrended_z")
            .dropna(subset=["detrended_z"])
        )
        traces_table["row"] = traces_table["identity"].map(
            {v: i for i, v in enumerate(order)})
        traces_table = traces_table.merge(
            fits[["identity", "cosinor_peak_hour", "cosinor_p_value",
                  "lombscargle_period_hours", "cosinor_relative_amplitude",
                  "rhythmic_both"]],
            on="identity", how="left",
        )
        rows = len(traces_table)

    width_in = 14.5
    header_in = 1.88                      # title and subtitle
    gap_in = 2.08                         # a panel's x-axis label and the next one's legend
    # Footnote and the bottom panel's x-axis label. The 2.01 was measured against
    # a three-line footnote; a run with a provenance sidecar has more to declare,
    # and the extra lines have to come out of the page rather than out of the
    # x-axis label they would otherwise land on.
    foot_lines = footnote.count("\n") + 1
    foot_in = 2.01 + 1.45 * ctx.theme.size("note") / 72.0 * max(0, foot_lines - 3)
    heights = {"raster": 4.82, "phase": 2.61}
    note_band = 2.6                       # ten lines of note, when it has nowhere else to go
    note_in = 0.0 if "phase" in panels else note_band + gap_in

    height_in = (
        header_in + foot_in + note_in
        + sum(heights[name] for name in panels.keys)
        + gap_in * (len(panels) - 1)
    )
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    left = 0.085
    tops: dict[str, float] = {}
    cursor = height_in - header_in
    for name in panels.keys:
        tops[name] = cursor
        cursor -= heights[name] + gap_in

    axes = []
    auxiliary: dict[str, pd.DataFrame] = {}
    figure_data = None

    if "raster" in panels:
        bottom = fraction(tops["raster"] - heights["raster"])
        height = fraction(heights["raster"])
        ax_map = figure_.add_axes([left, bottom, 0.745, height])
        ax_strip = figure_.add_axes([0.845, bottom, 0.017, height])
        raster = ctx.drew("raster", rhythm_panels.trace_raster(
            ax_map, matrix.to_numpy(float), hours_axis, ctx.theme,
            blocks=[(len(rhythmic_ids), "rhythmic", "rhythmic"),
                    (len(other_ids), "arrhythmic", "not")],
            cmap=resolve_look(ctx.theme, raster_lut).cmap if raster_lut else None,
            hour_ticks=hour_ticks,
        ))
        image = raster.extra["handle"]
        category_strip(
            ax_strip,
            [(len(rhythmic_ids), "rhythmic"), (len(other_ids), "arrhythmic")],
            ctx.theme,
        )
        colour_bar(
            figure_, image,
            [0.910, bottom + height * 0.17, 0.015, height * 0.64], ctx.theme,
            label=f"Detrended {label.lower()} (SD)",
        )
        axes.append(ax_map)
        if lead == "raster":
            figure_data = traces_table
        else:
            auxiliary["detrended_traces.csv"] = traces_table

    auxiliary["cosinor_fits.csv"] = fits

    if "phase" in panels:
        bottom = fraction(tops["phase"] - heights["phase"])
        ax_hist = figure_.add_axes([left, bottom, 0.40, fraction(heights["phase"])])
        # The panel's own table, rather than the same bins counted twice.
        histogram = ctx.drew("phase", rhythm_panels.phase_histogram(
            ax_hist, fits["cosinor_peak_hour"], ctx.theme,
            passed=fits["rhythmic_both"].to_numpy(bool),
            bins=bins, mean_hour=float(stats_all["mean_peak_hour"]),
            hour_ticks=phase_ticks,
        )).data
        if lead == "phase":
            figure_data = histogram
        else:
            auxiliary["peak_hour_histogram.csv"] = histogram
        # Above the axes rather than inside it: the tallest bar reaches the top of
        # this panel, so an inside legend would sit on the data.
        ctx.theme.legend(ax_hist, location="above left",
                         fontsize=ctx.theme.size("caption"))
        axes.append(ax_hist)

    auxiliary.update(ctx.provenance_auxiliary())

    return FigureResult(
        figure=figure_,
        axes=axes,
        figure_data=figure_data,
        auxiliary=auxiliary,
        header_x=left,
        heading=f"{label} per cell",
        title_fields={"metric": label, "span": f"{span_hours:.0f}"},
        subtitle=(
            f"{ctx.summary['stem']}, {span_hours:.0f} h at "
            f"{ctx.summary['minutes_per_frame']:.0f} min per frame. {cells_tested} of "
            f"{ctx.summary['identities']} cells had the {params['min_observations']} "
            f"frames a fit needs.\nEach row is one cell, linearly detrended and "
            f"scaled by its own variability; rows are sorted by peak time within "
            f"each block."
        ),
        note=(
            f"mean peak  {stats_all['mean_peak_hour']:.1f} h\n"
            f"vector length  {stats_all['vector_length']:.2f}\n"
            f"Rayleigh p  {stats_all['rayleigh_p_value']:.3f}   (n = {cells_tested})\n\n"
            f"Median free-running period {median_period:.1f} h\n"
            f"(Lomb-Scargle, {search[0]:.0f}-{search[1]:.0f} h search).\n\n"
            f"Rhythmic here means p < 0.05\non both tests. On "
            f"{null_row['null_model']}\nsurrogates the same pair\nfires on "
            f"{null_row['false_positive_rate_both']:.1%} of traces."
        ),
        # Beside the histogram when there is one - it is only 0.40 of the width -
        # and in its own band under the raster when there is not.
        note_at={
            "x": 0.545 if "phase" in panels else left,
            "y": fraction(tops["phase"] if "phase" in panels
                          else foot_in + note_band),
            "fontsize": ctx.theme.size("subtitle"),
            "linespacing": 1.35,
        },
        footnote=footnote,
        readme=f"""## What the figure shows

**Raster.** One row per cell, colour is that cell's {label.lower()} after a
straight line has been subtracted and the result scaled by the cell's own
variability - exactly the trace the rhythm test was run on. Rows are split
into the cells called rhythmic by both tests and those that were not, and
sorted by peak time within each block.

**Phase histogram.** Where each cell's peak falls within a 24 h cycle, with the
rhythmic subset stacked inside the total, and the population mean marked.

Either panel can be the whole page: `--panels raster` or `--panels phase`.

## What to be careful of

The recording covers about two cycles, so a period cannot be resolved from
it - every row carries `cycles_covered` and `period_underdetermined` so that
caveat travels with the number. The single-cell rhythmic fraction must be
read against the surrogate floor, not on its own; the population phase test
does not depend on any individual cell passing."""
        + (f"\n\n{provenance} A trace is measured over the outline as accepted, so a "
           "frame whose outline was reconstructed contributes to the fit like any "
           "other; `data/der/provenance.csv` carries the counts."
           if provenance else ""),
        console=(f"panels {panels.keys}  metric {metric}  cells {len(fits)}  "
                 f"raster rows {rows}"),
    )


if __name__ == "__main__":
    run_figure("cd68-reporter-rhythm", DEFAULT_RUN)
