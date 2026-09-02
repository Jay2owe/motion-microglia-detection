"""Figure 2: which measurements beat drift-matched noise.

One panel, ``panels.rhythms.noise_floor``, which is a dumbbell that refuses to
draw a rate without its surrogate rate beside it::

    python analysis/figures/02_rhythmicity_above_noise.py <run>
    ... --metrics corrected_mean,area_px,turnover_index    these rows, this order

Row wording comes from ``_metrics``, so a measurement is called the same thing
here as on every other figure that draws it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _metrics import describe
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import rhythms as rhythm_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=2,
    slug="rhythmicity-above-noise",
    summary="which measurements beat drift-matched noise",
    title="Rhythm calls against a drift-matched noise floor",
    grammar="paired dumbbell of observed against surrogate rhythm-call rate",
    reads=(Table("rhythms_null.csv", module="rhythms"),
           Table("rhythms_population.csv", module="rhythms")),
    panels=(Panel("floor", rhythm_panels.noise_floor,
                  title="Rhythm calls against the noise floor"),),
    options=(
        # Empty means every measurement the null was run on, sorted by how far
        # the real rate clears the surrogate one.
        Option("metrics", default=(),
               help="which tested measurements are drawn, in reading order"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    wanted = ctx.option("metrics")
    null = ctx.table("rhythms_null.csv")
    population = ctx.table("rhythms_population.csv")
    phase = population[population["population"] == "all_tested"].set_index("metric")

    if wanted:
        unknown = [name for name in wanted if name not in set(null["metric"])]
        if unknown:
            raise SystemExit(
                f"--metrics names {', '.join(unknown)}, which was not tested against the "
                f"null in this run; rhythms_null.csv holds "
                f"{', '.join(sorted(set(null['metric'])))}"
            )
        table = null.set_index("metric").loc[wanted[::-1]].reset_index()
    else:
        table = null.sort_values("excess_over_null_both").reset_index(drop=True)

    table["label"] = table["metric"].map(lambda column: describe(column).label)
    table["phase_rayleigh_p"] = table["metric"].map(phase["rayleigh_p_value"])
    table["phase_vector_length"] = table["metric"].map(phase["vector_length"])

    cells_tested = int(table["cells_tested"].max())
    surrogates = int(table["surrogates"].max())
    # The column counts every surrogate trace, not the number built per cell, and
    # the legend has to say which of the two it means.
    per_cell = surrogates // cells_tested if cells_tested else surrogates
    null_model = str(table["null_model"].iloc[0])

    # A row per measurement, so the page grows with the list rather than squeezing
    # it: at a fixed height, ten rows would put the labels closer together than the
    # words are tall.
    width_in = 13.5
    header_in = 2.10
    row_in = 0.93
    foot_in = 2.20
    height_in = header_in + len(table) * row_in + foot_in

    ctx.panels()
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    ax = figure_.add_axes(
        [0.235, fraction(foot_in), 0.715, fraction(len(table) * row_in)])

    rhythm_panels.noise_floor(
        ax,
        table["observed_rate_both"], table["false_positive_rate_both"], ctx.theme,
        labels=table["label"],
        excess=table["excess_over_null_both"],
        surrogate_label=f"drift-matched noise ({per_cell} surrogates per cell)",
    )
    ax.set_xlim(0, 0.44)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
    ctx.theme.legend(ax, fontsize=ctx.theme.size("subtitle"))

    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=table,
        subtitle=(
            f"{ctx.summary['stem']}, {cells_tested} cells tested. A cosinor and a "
            f"Lomb-Scargle test both at p < 0.05, run on the real trace and on\n"
            f"{surrogates} surrogates with the same drift and smoothness but no rhythm "
            f"({null_model})."
        ),
        footnote=(
            "The hollow dot is how often both tests fire on the surrogates; the filled "
            "dot is how often they fire on the real cells.\nThe number beside each pair "
            "is the difference between them, in percentage points."
        ),
        readme="""## What the figure shows

Two dots per measurement. The hollow dot is how often the pair of tests fires
on surrogate traces built to have that measurement's own drift and smoothness
but no rhythm; the filled dot is how often it fires on the real cells.

## Why the surrogates are there

A cosinor fit on a recording this short fires more often than its p-value
suggests, because a slowly drifting, smooth trace resembles the first half of a
cosine. The surrogates measure how often that happens for each measurement.

## Choosing the rows

`--metrics` takes any measurements `rhythms_null.csv` holds, in the order you
want them read. Left alone the rows are sorted by how far the real rate clears
the surrogate one.""",
        console=table[["metric", "observed_rate_both", "false_positive_rate_both",
                       "excess_over_null_both"]].round(3).to_string(index=False),
    )


if __name__ == "__main__":
    run_figure("rhythmicity-above-noise", DEFAULT_RUN)
