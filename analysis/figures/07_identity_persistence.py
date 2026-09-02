"""Figure 7: every cell, every frame - when its name is on screen.

One panel, ``panels.presence.persistence_raster``, which owns the states and
their colours so that a gap means the same thing here as on figure 6::

    python analysis/figures/07_identity_persistence.py <run> --hour-ticks 12
    python analysis/figures/07_identity_persistence.py <run> --inferred-threshold 0

Two things are drawn only when the run has them. If ``cell_frame.csv`` carries
``inferred_fraction`` - the ``provenance`` module ran - a frame where most of
the outline's owner was decided by reconstruction rather than by seeing that
cell there is drawn as its own state instead of as a plain name on screen. And
if the ``history`` module copied the tracker's continuity evidence in, a strip
beside the raster says why each cell's gaps are there. Without either, this is
exactly the three-colour figure it has always been.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import presence as presence_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=7,
    slug="identity-persistence-raster",
    summary="every cell, every frame - when its name is on screen",
    title="Every cell, every frame: when its name is on screen",
    grammar="raster of presence state per identity per frame",
    reads=(
        Table("presence.csv", module="presence"),
        # Both optional. A run without them draws the figure it always drew.
        Table("cell_frame.csv", module="motility", optional=True),
        Table("history_gap_frames.csv", module="history", optional=True),
    ),
    panels=(
        Panel("raster", presence_panels.persistence_raster,
              title="When each name is on screen"),
        Panel("mechanisms", presence_panels.gap_mechanism_strip,
              title="Why the gaps are there",
              needs=("history_gap_frames.csv",)),
    ),
    options=(
        Option("hour_ticks", default=None),
        # Half the outline. A judgement call, so it is stated on the figure
        # rather than hidden here, and --inferred-threshold 0 asks the honest
        # harder question: which frames contain any reconstructed pixel at all.
        Option("inferred_threshold", default=0.5),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    hour_ticks = ctx.option("hour_ticks")
    inferred_threshold = ctx.option("inferred_threshold")

    drawn = ctx.panels()
    presence = ctx.table("presence.csv")
    measured = ctx.optional_table("cell_frame.csv")
    gap_frames = ctx.optional_table("history_gap_frames.csv")

    # Rows in the order the cells arrive, so the diagonal edge on the left is the
    # arrival curve and every gap is read against it.
    order = (
        presence.groupby("identity")[["first_frame_index", "last_frame_index"]].first()
        .sort_values(["first_frame_index", "last_frame_index"])
        .index.tolist()
    )
    row_of = {identity: row for row, identity in enumerate(order)}

    # The codes and their colours belong to the panel, not to this page: a gap has
    # to be the same orange here as on figure 6 or the reader learns it twice.
    code = presence_panels.STATE_CODES
    presence["row_position"] = presence["identity"].map(row_of)

    # A name the tracker reasoned its way to is still a name on screen. It is split
    # off, never dropped: this figure reports provenance, it does not filter on it.
    states = presence_panels.BASE_STATES
    if measured is not None and "inferred_fraction" in measured.columns:
        states = presence_panels.PROVENANCE_STATES
        columns = ["identity", "frame_index", "inferred_fraction"]
        if "added_px" in measured.columns:
            columns.append("added_px")
        presence = presence.merge(
            measured[columns], on=["identity", "frame_index"], how="left",
        )
        reconstructed = (
            (presence["state"] == "named")
            & (presence["inferred_fraction"] > inferred_threshold)
        )
        presence.loc[reconstructed, "state"] = "named_inferred"
    if "inferred_fraction" not in presence.columns:
        presence["inferred_fraction"] = float("nan")

    presence["state_code"] = presence["state"].map(code)

    # Why each cell's remaining gaps are there, as the tracker classified them.
    # Only gaps that are still gaps in the accepted labels: the continuity evidence
    # is computed partway through the accepted history, and a later repair can have
    # closed a gap it recorded.
    row_mechanism: dict[int, str] = {}
    if gap_frames is not None and "mechanisms" in drawn:
        gap_mechanisms = gap_frames
        if "still_missing_in_accepted_labels" in gap_mechanisms.columns:
            gap_mechanisms = gap_mechanisms[
                gap_mechanisms["still_missing_in_accepted_labels"].astype(bool)]
        if not gap_mechanisms.empty and "mechanism" in gap_mechanisms.columns:
            row_mechanism = {
                int(identity): str(group["mechanism"].mode().iat[0])
                for identity, group in gap_mechanisms.groupby("identity")
                if identity in row_of and not group["mechanism"].mode().empty
            }
        per_gap_frame = (
            gap_mechanisms[["identity", "frame_index", "mechanism"]]
            .drop_duplicates(subset=["identity", "frame_index"])
            .rename(columns={"mechanism": "gap_mechanism"})
            if not gap_mechanisms.empty and "mechanism" in gap_mechanisms.columns
            else pd.DataFrame(columns=["identity", "frame_index", "gap_mechanism"])
        )
        presence = presence.merge(
            per_gap_frame, on=["identity", "frame_index"], how="left")
    if "gap_mechanism" not in presence.columns:
        presence["gap_mechanism"] = None

    mechanism_by_row = [row_mechanism.get(identity) for identity in order]

    figure_data = presence[
        ["identity", "row_position", "frame_index", "hours", "state", "state_code",
         "inferred_fraction", "gap_mechanism"]
    ].sort_values(["row_position", "frame_index"])

    matrix = (
        presence.pivot(index="row_position", columns="frame_index", values="state_code")
        .sort_index()
        .to_numpy(float)
    )
    hours = presence.groupby("frame_index")["hours"].first().sort_index().to_numpy(float)

    per_cell = presence.pivot_table(
        index="identity", columns="state", values="frame_index", aggfunc="count"
    ).reindex(columns=list(code), fill_value=0).fillna(0).astype(int)

    unbroken = int((per_cell["unclaimed"] == 0).sum())
    broken = int((per_cell["unclaimed"] > 0).sum())
    gap_cell_frames = int(per_cell["unclaimed"].sum())
    reconstructed_frames = int(per_cell["named_inferred"].sum())
    shows_provenance = "named_inferred" in states
    shows_mechanism = bool(row_mechanism)

    provenance_line = ""
    if shows_provenance:
        any_inferred = int((presence["inferred_fraction"] > 0).sum())
        # The rare, sharper case: not a reconstructed name over a real outline
        # but an outline with nothing under it at all. Stated as a count on the
        # existing line rather than as a fifth colour, because on this movie it
        # is 17 cell-frames in 6,073 and a colour that never appears teaches a
        # reader nothing while costing a legend row.
        supplied = (int((presence["added_px"] > 0).sum())
                    if "added_px" in presence.columns else 0)
        provenance_line = (
            f"\nIn {reconstructed_frames:,} of those on-screen cell-frames, more than "
            f"{inferred_threshold:.0%} of the outline belongs to this cell\nbecause the "
            f"tracker reconstructed the attribution rather than saw the cell there "
            f"({any_inferred:,} contain any such pixel"
            + (f"; {supplied:,} have no detection at all)."
               if supplied else ").")
        )

    # Room for what is actually drawn. Four states need a second legend row and the
    # mechanisms need a column of their own; both come out of the raster rather than
    # off the bottom or the side of the page.
    bottom = 0.265 if shows_provenance else 0.205
    # The provenance sentence takes the subtitle from two lines to four, so the top
    # of the raster has to come down with it or the text lands on the first cells.
    height = 0.545 if shows_provenance else 0.655

    figure_ = ctx.sheet(14.0, 11.6)
    if shows_mechanism:
        ax = figure_.add_axes([0.105, bottom, 0.545, height])
        strip_ax = figure_.add_axes([0.665, bottom, 0.014, height])
    else:
        ax = figure_.add_axes([0.105, bottom, 0.865, height])
        strip_ax = None

    presence_panels.persistence_raster(
        ax, matrix, hours, ctx.theme, hour_ticks=hour_ticks, states=states)
    if strip_ax is not None:
        presence_panels.gap_mechanism_strip(strip_ax, mechanism_by_row, ctx.theme)

    console = [f"identities: {len(order)}  unbroken: {unbroken}  with gaps: {broken}  "
               f"gap cell-frames: {gap_cell_frames}"]
    if shows_provenance:
        console.append(f"reconstructed cell-frames above {inferred_threshold:g}: "
                       f"{reconstructed_frames}")
    if shows_mechanism:
        console.append(f"cells with a gap mechanism: {len(row_mechanism)} "
                       f"of {len(order)}")

    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=figure_data,
        auxiliary={
            "gap_mechanism_per_cell.csv": pd.DataFrame({
                "identity": order,
                "row_position": range(len(order)),
                "dominant_gap_mechanism": mechanism_by_row,
            }),
            "frames_per_state_per_cell.csv": per_cell.reset_index(),
        },
        heading="Identity persistence raster",
        subtitle=(
            f"{ctx.summary['stem']}, {len(order)} identities over "
            f"{ctx.summary['frames']} frames "
            f"({ctx.summary['hours_covered']:.0f} h at "
            f"{ctx.summary['minutes_per_frame']:.0f} min "
            f"per frame). Rows are ordered by first appearance.\n"
            f"{unbroken} cells hold their name in every frame between their first and "
            f"last appearance; {broken} have at least one gap, {gap_cell_frames:,} "
            f"cell-frames in total."
            + provenance_line
        ),
        footnote=(
            "A gap is a frame inside a cell's own lifespan where no mask carries its "
            "number.\nBefore a cell's first appearance and after its last is blank, "
            "and is not a gap."
            + (
                "\nThe strip on the right names the mechanism accounting for most of "
                "each cell's remaining gaps, as the tracker classified it."
                if shows_mechanism else
                "\nWhat the figure does not say is why a name is absent; the tables "
                "record the absence, not its cause."
            )
        ),
        readme="""## What the figure shows

One row per identity, one column per frame. Dark means that name carries a mask
somewhere in the field. Orange means the frame lies inside the cell's own
lifespan - first appearance to last - and the name is not on screen. Blank is
before the cell arrived or after it was last seen.

Where the run carries provenance, a fourth grey splits the dark: the name is on
screen, but for most of that outline the tracker decided the owner by
reconstruction rather than by seeing this cell there. Almost always the pixels
themselves were imaged; it is the attribution that was reasoned to. The
threshold is named in the subtitle and is a choice, not a measurement.

Rows are sorted by first appearance, so the pale wedge at the bottom left is
cells that had not arrived yet, not a run of failures.

The panel is `panels.presence.persistence_raster`, which also draws the gap
colour on figure 6 - one place decides what each state looks like.

## Why the absence is measured rather than inferred

A cell missing from `cell_frame.csv` and a cell that never existed leave the
same trace in a row count. The `presence` module reads the label stack directly
and writes a row for every cell in every frame, so the two can be told apart.

## What to be careful of

The raster records where a name is absent. What it cannot show on its own is
why, and the strip on the right only helps where the `history` module found the
tracker's continuity evidence. Where there is no strip, a merged object carrying
one number instead of two, an outline that fell below threshold and a genuine
disappearance all draw the same orange.

The mechanism shown for a cell is the one accounting for most of its gap frames,
so a cell whose gaps have several causes is summarised by the commonest. Only
gaps that are still gaps in the accepted labels are counted: the continuity
evidence is computed partway through the accepted history, and a later repair
can close a gap it recorded.

Reconstructed is a threshold on a continuous quantity. At the default, a cell
whose outline is a third reconstructed in every single frame is drawn as fully
observed; `--inferred-threshold 0` colours any frame containing a single
reconstructed pixel, and the subtitle reports both counts either way. On this
dataset the two counts come out close, because the flag is nearly all-or-nothing
per frame rather than a partly redrawn boundary.""",
        console="\n".join(console),
    )


if __name__ == "__main__":
    run_figure("identity-persistence-raster", DEFAULT_RUN)
