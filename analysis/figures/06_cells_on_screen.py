"""Figure 6: how many names are on screen in each frame, and what carries none.

Two panels, either of which can be the whole page::

    python analysis/figures/06_cells_on_screen.py <run>
    ... --panels counts         just the names-per-frame panel
    ... --panels unclaimed      just the unattributed foreground
    ... --hour-ticks 12

The panels are ``panels.presence.on_screen`` and
``panels.presence.unclaimed_area``. The unclaimed panel is only offered when the
run was given an unclaimed stack; a run without one is not wrong, it simply
cannot answer that half.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import presence as presence_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=6,
    slug="cells-on-screen",
    summary="how many names are on screen in each frame, and what carries none",
    title="Identities on screen in each frame",
    grammar="stacked time series of named identities and unattributed foreground",
    reads=(Table("presence_frame.csv", module="presence"),),
    panels=(
        Panel("counts", presence_panels.on_screen,
              title="Identities carrying a mask"),
        # The presence module writes this table with or without an unclaimed
        # stack; only the column says which run this is.
        Panel("unclaimed", presence_panels.unclaimed_area,
              title="Foreground on no name",
              needs=("presence_frame.csv:unclaimed_px",)),
    ),
    options=(Option("hour_ticks", default=None),),
)
def build(ctx: FigureContext) -> FigureResult:
    hour_ticks = ctx.option("hour_ticks")
    frames = ctx.table("presence_frame.csv")
    drawn = ctx.panels()

    columns = ["frame_index", "source_imagej_frame", "hours", "identities_named",
               "identities_expected", "identities_total", "assigned_px"]
    if "unclaimed" in drawn:
        columns += ["unclaimed_px", "foreground_px", "unclaimed_fraction"]
    figure_data = frames[columns].copy()

    hours = frames["hours"].to_numpy(float)
    named = frames["identities_named"].to_numpy(float)
    expected = frames["identities_expected"].to_numpy(float)
    total = int(frames["identities_total"].iloc[0])
    missing = int((expected - named).sum())

    # Fixed panel heights: dropping one shortens the page rather than stretching
    # what is left, so the same trace is the same height whichever panels are on.
    width_in = 13.5
    header_in = 2.01
    gap_in = 0.95
    foot_in = 2.49
    heights = {"counts": 3.07, "unclaimed": 2.07}

    height_in = (
        header_in + foot_in + sum(heights[name] for name in drawn.keys)
        + gap_in * (len(drawn) - 1)
    )
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    left, plot_width = 0.115, 0.845
    axes = {}
    cursor = height_in - header_in
    for name in drawn.keys:
        axes[name] = figure_.add_axes(
            [left, fraction(cursor - heights[name]), plot_width,
             fraction(heights[name])]
        )
        cursor -= heights[name] + gap_in

    if "counts" in drawn:
        presence_panels.on_screen(
            axes["counts"], hours, named, expected, ctx.theme,
            hour_ticks=hour_ticks, total=total,
            show_x=(drawn.keys[-1] == "counts"),
        )
        ctx.theme.legend(axes["counts"], location="above",
                         fontsize=ctx.theme.size("caption"))

    if "unclaimed" in drawn:
        presence_panels.unclaimed_area(
            axes["unclaimed"], hours, frames["unclaimed_px"], ctx.theme,
            hour_ticks=hour_ticks, show_x=(drawn.keys[-1] == "unclaimed"),
        )

    return FigureResult(
        figure=figure_,
        axes=list(axes.values()),
        figure_data=figure_data,
        heading="Identities on screen per frame",
        subtitle=(
            f"{ctx.summary['stem']}, {ctx.summary['frames']} frames over "
            f"{ctx.summary['hours_covered']:.0f} h at "
            f"{ctx.summary['minutes_per_frame']:.0f} min "
            f"per frame, {ctx.field['width']} x {ctx.field['height']} px.\n"
            f"{named.mean():.0f} names carry a mask per frame on average "
            f"(range {named.min():.0f}-{named.max():.0f}) out of {total} identities "
            f"in the movie."
        ),
        footnote=(
            "The dashed line counts the cells inside their own lifespan, first\n"
            f"appearance to last. The {missing:,} cell-frames between the two lines\n"
            "are cells that exist without a name on screen; figure 7 shows which."
            + (
                "\nUnclaimed pixels are foreground the pipeline declines to put on\n"
                "the nearest name."
                if "unclaimed" in drawn else ""
            )
        ),
        readme="""## What the figure shows

**Counts.** How many distinct identities carry a mask in each frame, against how
many are inside their own lifespan in that frame. A cell counts as inside its
lifespan from the frame it first appears to the frame it was last seen, so the
dashed line rises as cells arrive and never counts a cell before it exists.

**Unclaimed.** Foreground pixels the pipeline agrees are cell signal but declines
to attribute to any identity. They are written to their own stack rather than
pushed onto the nearest name, which is why they can be counted at all.

Either panel can be the whole page: `--panels counts` or `--panels unclaimed`.

## Where the numbers come from

`presence_frame.csv`, written by the `presence` module, which reads the label
stack directly rather than counting rows in `cell_frame.csv` - a cell missing
from the tables and a cell that never existed look identical in a row count.

The unclaimed panel is offered only when the run was given an unclaimed stack. A
run without one is not wrong, it simply cannot answer that half.""",
        console=(f"panels {drawn.keys}  frames: {len(frames)}  "
                 f"named per frame: {named.mean():.1f}  "
                 f"unnamed cell-frames: {missing}"),
    )


if __name__ == "__main__":
    run_figure("cells-on-screen", DEFAULT_RUN)
