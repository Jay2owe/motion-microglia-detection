"""Figure 9: how far a cell moves between one frame and the next.

One panel, ``panels.motility.step_histogram``::

    python analysis/figures/09_step_size_distribution.py <run>
    ... --metrics soma_step_px_gapless   the soma rather than the whole outline
    ... --bins 60                        finer bands
    ... --trace-luts "#c0392b"           the colour of the bars

Only the median and the 95th percentile are drawn as lines, and both are read
off this distribution. The tracker's own distance tolerance is not drawn: it
lives in a configuration this package does not read.

This is the reference shape for a builder. Everything the figure is - its slug,
its panel, the table it reads, the three options it takes - is in the ``@figure``
block, and everything the same for all thirty-six pages - the bundle, the
header, the README, the save - is in ``_schema._finish``. Copy this file to add
figure 37.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _metrics import axis_label, describe, role_for
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import resolve_look
from panels import motility as motility_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=9,
    slug="step-size-distribution",
    summary="how far a cell moves between one frame and the next",
    title="{metric} between consecutive frames, {interval} apart",
    grammar="histogram of frame-to-frame step size on a log axis",
    reads=(Table("cell_frame.csv", module="motility"),),
    panels=(Panel("distribution", motility_panels.step_histogram,
                  title="Step size"),),
    options=(
        # Gapless by default. A step measured across a three-frame absence
        # covers three intervals, and mixing it in would put movement in the
        # fast tail that the cell had an hour and a half to make.
        Option("metrics", default="step_px_gapless", cast=str, metavar="COL",
               help="which step column the histogram draws"),
        Option("bins", default=45),
        Option("trace_luts", default=None, cast=str, metavar="COLOUR",
               help="colour or colour map of the bars"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    metric = ctx.option("metrics")
    cell_frame = ctx.table("cell_frame.csv")
    if metric not in cell_frame.columns:
        raise SystemExit(
            f"--metrics {metric} is not in cell_frame.csv. Step columns available: "
            + ", ".join(c for c in cell_frame.columns if "step" in c)
        )

    steps = cell_frame[["identity", "frame_index", "hours", metric]].dropna(
        subset=[metric]
    ).copy()
    steps = steps[steps[metric] > 0]
    values = steps[metric].to_numpy(float)

    median = float(np.median(values))
    p95 = float(np.percentile(values, 95))
    largest = float(values.max())
    cells = int(steps["identity"].nunique())
    ungapped = metric.replace("_gapless", "")
    skipped = (int(cell_frame[ungapped].notna().sum() - len(steps))
               if metric.endswith("_gapless") and ungapped in cell_frame.columns else 0)

    bar_lut = ctx.option("trace_luts")
    # The footnote is three lines and the x axis carries a label under its tick
    # labels; both live below the axes, so the axes starts a quarter of the way
    # up. One panel with its own proportions, so its own sheet rather than the
    # shared stack-of-panels layout.
    ctx.panels()
    figure_ = ctx.sheet(14.6, 9.8)
    ax = figure_.add_axes([0.105, 0.275, 0.855, 0.530])

    bands = ctx.drew("distribution", motility_panels.step_histogram(
        ax, values, ctx.theme,
        bins=ctx.option("bins"),
        role=role_for(metric),
        look=resolve_look(ctx.theme, bar_lut, role_for(metric)) if bar_lut else None,
        unit=ctx.summary["scale"]["length_unit"],
        x_label=axis_label(metric, ctx.interval, wrap=False) + ", log scale",
    ))
    counts, edges = bands.extra["counts"], bands.extra["edges"]

    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=pd.DataFrame({"bin_left_px": edges[:-1],
                                  "bin_right_px": edges[1:],
                                  "cell_frames": counts}),
        auxiliary={"steps.csv": steps},
        title_fields={"metric": describe(metric).label,
                      "interval": f"{ctx.interval:.0f} min"},
        subtitle=(
            f"{ctx.summary['stem']}, {len(values):,} steps across {cells} identities, "
            f"on a log axis because the range spans three decades.\nHalf of all steps "
            f"are under {median:.2f} px; the 95th percentile is {p95:.1f} px and the "
            f"largest is {largest:.0f} px."
        ),
        # The exclusion sentence only when there were exclusions. A figure that
        # always claims them is worse than one that never mentions them.
        footnote=ctx.footnote(
            f"Steps measured across a gap are excluded, {skipped:,} of them: one "
            f"covering three absences\nspans three intervals, and counting it here "
            f"would put an hour and a half of movement in the tail."
            if skipped else ""
        ),
        heading="Frame-to-frame step size distribution",
        readme=f"""## What the figure shows

The distance each identity's `{metric}` moved between one frame and the next,
pooled over every cell and every frame. On a log axis it is one dense peak below
a pixel with a long tail, which is why the axis is logarithmic: a linear one
puts the whole distribution in the first bin.

The two marked lines are read off this distribution - the median and the 95th
percentile - and nothing else on the figure is a threshold.

## What is not on it

The tracker's own distance tolerances are not drawn. They live in the tracking
configuration, which this package does not read, and a figure that quoted one
would keep quoting it after somebody changed it. That refusal lives in
`panels.motility.step_histogram`, not in this file, so any page that draws the
panel inherits it.

## Which steps are counted

`{metric}`. A step measured across an absence spans as many intervals as the
absence lasted; `step_px` keeps those rows and `step_px_per_frame` divides them
out, but neither belongs in a histogram whose axis is labelled per frame
interval.""",
        console=(f"steps: {len(values)}  median: {median:.3f} px  p95: {p95:.2f} px  "
                 f"max: {largest:.1f} px  excluded across gaps: {skipped}"),
    )


if __name__ == "__main__":
    run_figure("step-size-distribution", DEFAULT_RUN)
