"""Figure 10: the tracker's own motion evidence, transition by transition.

One panel, ``panels.surveillance.evidence_channels``, which also owns the
channel colours - the ones the evidence stack itself encodes them with::

    python analysis/figures/10_motion_evidence_budget.py <run>
    ... --metrics frame_gained_px,frame_lost_px       just the two that must track
    ... --metrics frame_held_px,frame_base_px,frame_trail_px
    ... --trace-luts "#c0392b,#4878A8"                override the channel colours
    ... --hour-ticks 12
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import resolve_look
from panels import surveillance as surveillance_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=10,
    slug="motion-evidence-budget",
    summary="the tracker's own motion evidence, transition by transition",
    title="Motion-evidence pixels in each {interval} transition",
    grammar="time series of evidence-channel pixel counts per frame transition",
    reads=(Table("motion_evidence_frame.csv", module="motion_evidence"),),
    panels=(Panel("channels", surveillance_panels.evidence_channels,
                  title="Pixels per evidence channel"),),
    options=(
        Option("metrics",
               default=["frame_held_px", "frame_gained_px", "frame_lost_px"],
               help="which evidence channels the budget draws, comma separated"),
        Option("trace_luts", default=[],
               help="a colour per channel, comma separated, matched to --metrics "
                    "in order; empty keeps the stack's own colours"),
        Option("hour_ticks", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    hour_ticks = ctx.option("hour_ticks")
    channels = ctx.option("metrics")
    channel_luts = ctx.option("trace_luts")

    # Matched in order, so a list of the wrong length is a question about which
    # channel got which colour that the figure cannot answer. Empty is the one
    # length that means something else: keep the stack's own colours.
    if channel_luts and len(channel_luts) != len(channels):
        raise SystemExit(
            f"--trace-luts gives {len(channel_luts)} colour(s) for "
            f"{len(channels)} channel(s); they are matched in order, so give one "
            f"per channel or none at all"
        )

    ctx.panels()
    source = ctx.table_path("motion_evidence_frame.csv")
    frames = ctx.table("motion_evidence_frame.csv")
    params = ctx.module_params("motion_evidence")

    missing = [column for column in channels if column not in frames.columns]
    if missing:
        raise SystemExit(
            f"{source.name} has no {', '.join(missing)}: the evidence stack for this "
            f"movie carries channels {params.get('channel_names')}, so this figure "
            f"cannot be drawn from it.\nChannels available: "
            + ", ".join(c for c in frames.columns if c.startswith("frame_"))
        )

    columns = ["from_frame_index", "to_frame_index", "hours"] + list(channels)
    columns += [c for c in ("frame_base_px", "frame_trail_px", "frame_gained_share")
                if c in frames.columns and c not in columns]

    hours = frames["hours"].to_numpy(float)
    correlation = float("nan")
    if {"frame_gained_px", "frame_lost_px"} <= set(frames.columns):
        gained = frames["frame_gained_px"].to_numpy(float)
        lost = frames["frame_lost_px"].to_numpy(float)
        correlation = float(np.corrcoef(gained, lost)[0, 1])
    imbalance = (float(np.nanmax(np.abs(frames["frame_gained_share"] - 0.5))) * 2
                 if "frame_gained_share" in frames.columns else float("nan"))

    claimed = None
    if {"claimed_gained_px", "claimed_lost_px",
        "frame_gained_px", "frame_lost_px"} <= set(frames.columns):
        attributed = frames["claimed_gained_px"] + frames["claimed_lost_px"]
        total = frames["frame_gained_px"] + frames["frame_lost_px"]
        claimed = float(attributed.sum() / total.sum())

    figure_ = ctx.sheet(13.5, 10.4)
    ax = figure_.add_axes([0.115, 0.265, 0.845, 0.505])

    looks = {
        column: resolve_look(
            ctx.theme, channel_luts[index],
            surveillance_panels.EVIDENCE_CHANNELS.get(column, ("evidence",))[0])
        for index, column in enumerate(channels) if index < len(channel_luts)
    }
    surveillance_panels.evidence_channels(
        ax, hours, frames, ctx.theme, columns=channels, looks=looks,
        hour_ticks=hour_ticks,
    )
    ctx.theme.legend(ax, location="above", fontsize=ctx.theme.size("caption"))

    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=frames[columns],
        heading="Motion-evidence budget",
        title_fields={"interval": f"{ctx.interval:.0f} min"},
        subtitle=(
            f"{ctx.summary['stem']}, {len(frames)} transitions over "
            f"{ctx.summary['hours_covered']:.0f} h. Every pair of consecutive frames "
            f"is scored per pixel by the tracker and written to its own evidence "
            f"stack;\nthis counts the pixels in each channel over the whole field."
            + (f" Gained and lost correlate at r = {correlation:.2f} across the "
               f"recording."
               if np.isfinite(correlation) else "")
        ),
        footnote=(
            "Colours are the ones the evidence stack itself encodes, so they match the "
            "QC movie.\nChannels are counted over the whole field, not per cell. The "
            "per-cell split is in motion_evidence.csv"
            + (f",\nwhich covers {claimed:.0%} of the moving pixels."
               if claimed is not None else ".")
        ),
        readme=f"""## What the figure shows

Pixel counts per frame-to-frame transition in the evidence channels the tracker
writes: {', '.join(channels)}. The counts are over the whole field, so they
include evidence that fell on no named cell.

Colours are the ones the evidence stack encodes the channels with, and they live
in `panels.surveillance.EVIDENCE_CHANNELS` rather than in this file. A reader who
has watched the QC movie has already learned them, and recolouring them here
would be a second vocabulary for one thing.

## Why gained and lost are drawn together

Signal that leaves one place arrives in another, so the two should track each
other. They are plotted on one axis for that reason: a transition where one runs
away from the other is a transition where evidence is being created or destroyed
rather than moved.

## Whole field, not per cell

`motion_evidence.csv` carries the same channels attributed to each identity's
own footprint. The difference between the two is evidence that landed on no
name, which is the part a tracking failure hides in - so the two tables are kept
apart rather than one derived from the other.""",
        console=(f"transitions: {len(frames)}  channels {channels}  "
                 f"gained-lost r: {correlation:.3f}  "
                 f"worst frame imbalance: {imbalance:.2f}"
                 + (f"  attributed to a cell: {claimed:.1%}"
                    if claimed is not None else "")),
    )


if __name__ == "__main__":
    run_figure("motion-evidence-budget", DEFAULT_RUN)
