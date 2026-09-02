"""Figure 36: every identity's named and unclaimed lifespan.

One bar per identity, cut where the name went missing. The gaps are coloured by
what the tracker decided happened, when it recorded a decision; a gap it could
not classify stays plain rather than being given a mechanism it does not have.

    python analysis/figures/36_lifespan_gantt.py <run>
    ... --order duration         sort the bars by how long each name lasted
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _bundle import units_note
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import presence as presence_panels


@figure(
    number=36,
    slug="lifespan-gantt",
    summary="every identity's named and unclaimed lifespan",
    title="Every identity's named and unclaimed lifespan",
    reads=(Table("presence.csv", module="presence"),
           Table("cell_summary.csv", module="presence"),
           Table("history_lifespans.csv", module="history", optional=True),
           Table("history_gap_frames.csv", module="history", optional=True)),
    panels=(
        Panel("gantt", presence_panels.lifespan_bars,
              title="Named and unnamed intervals"),
        Panel("arrivals", common.histogram, title="First appearance times"),
        Panel("coverage", common.histogram,
              title="Fraction of lifespan with a name"),
    ),
    options=(
        Option("order", default="first_appearance", cast=str,
               help="bar order: first_appearance, duration or identity"),
        Option("bins", default=20),
        Option("hour_ticks", default=24.0),
    ),
    grammar="gap-cut lifespan bars and arrival and coverage histograms",
)
def build(ctx: FigureContext) -> FigureResult:
    presence = ctx.table("presence.csv")
    summary = ctx.table("cell_summary.csv")
    longest = {}
    for identity, group in presence[presence["state"] == "unclaimed"].groupby("identity"):
        frames = np.sort(group["frame_index"].to_numpy(int))
        if not len(frames):
            longest[int(identity)] = 0
            continue
        runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
        longest[int(identity)] = max(len(run) for run in runs)
    data = summary[["identity", "first_hour", "last_hour", "observed_frames", "gap_frames", "coverage"]].copy()
    data["span_hours"] = data["last_hour"] - data["first_hour"]
    data["longest_gap_frames"] = data["identity"].map(longest).fillna(0).astype(int)
    data["ever_touches_border"] = summary.get("ever_touches_border", False)
    data = data[["identity", "first_hour", "last_hour", "span_hours", "observed_frames",
                 "gap_frames", "coverage", "longest_gap_frames", "ever_touches_border"]]

    # What the tracker recorded about these same cells, where the run has it.
    # A gap it could not explain stays uncoloured rather than being given a
    # mechanism it does not have.
    gaps = presence.copy()
    gaps["named"] = gaps["state"] == "named"
    history_gaps = ctx.optional_table("history_gap_frames.csv")
    if history_gaps is not None and "mechanism" in history_gaps.columns:
        if "still_missing_in_accepted_labels" in history_gaps.columns:
            history_gaps = history_gaps[
                history_gaps["still_missing_in_accepted_labels"].astype(bool)]
        gaps = gaps.merge(
            history_gaps[["identity", "frame_index", "mechanism"]]
            .drop_duplicates(subset=["identity", "frame_index"]),
            on=["identity", "frame_index"], how="left")
    mechanisms = (sorted({str(m) for m in gaps.get("mechanism", pd.Series(dtype=object))
                          .dropna().unique()})
                  if "mechanism" in gaps.columns else [])
    unexplained = int((~gaps["named"] & gaps["mechanism"].isna()).sum()) if (
        "mechanism" in gaps.columns) else int((~gaps["named"]).sum())

    lifespans = ctx.optional_table("history_lifespans.csv")
    silent: set[int] = set()
    if lifespans is not None and "silent_nonborder_ending" in lifespans.columns:
        silent = {int(identity) for identity in lifespans.loc[
            lifespans["silent_nonborder_ending"].astype(bool), "identity"]}
        data = data.merge(
            lifespans[[c for c in ("identity", "silent_nonborder_ending",
                                   "last_mask_touches_border",
                                   "termination_mechanism")
                       if c in lifespans.columns]],
            on="identity", how="left")
        # A cell the tracker's census never reached is not flagged; fill before
        # casting so an absent record cannot read as a silent ending.
        data["silent_nonborder_ending"] = (
            data["silent_nonborder_ending"].astype("object").where(
                data["silent_nonborder_ending"].notna(), False).astype(bool))

    order = ctx.option("order")
    if order not in {"first_appearance", "span", "coverage"}:
        raise SystemExit("--order must be first_appearance, span, or coverage")
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "gantt" in axes:
        presence_panels.lifespan_bars(axes["gantt"], data, ctx.theme, gaps=gaps,
                                     order=order, hour_ticks=ctx.hour_ticks,
                                     y_label="Cell", legend=False, mark=silent)
        axes["gantt"]._semantic_legend_handled = True
        # The panel is only a tenth of the page and holds 83 bars, so height is
        # the one thing it cannot spare. The mechanism vocabulary goes down the
        # right-hand side instead, out of the panel's width, which there is
        # plenty of; only the original single row comes off the top.
        position = axes["gantt"].get_position()
        width = position.width * (0.66 if mechanisms else 1.0)
        axes["gantt"].set_position([
            position.x0, position.y0, width, max(position.height - 0.055, 0.05)
        ])
        handles = [Patch(facecolor=ctx.theme.colour("named"))]
        labels = ["Cell identity present"]
        if unexplained:
            handles.append(Patch(facecolor=ctx.theme.colour("unclaimed")))
            labels.append("unnamed gap, no mechanism recorded"
                          if mechanisms else "Unnamed gap within lifespan")
        if silent:
            handles.append(Line2D([], [], marker="x", linestyle="none",
                                  color=ctx.theme.colour("invalid")))
            labels.append("ended silently, away from the field edge")
        fig.legend(
            handles=handles, labels=labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.905),
            ncol=min(3, len(handles)),
            frameon=False, fontsize=ctx.theme.size("caption"),
        )
        if mechanisms:
            palette = presence_panels.mechanism_colours(mechanisms, ctx.theme)
            gantt = axes["gantt"].get_position()
            fig.legend(
                handles=[Patch(facecolor=palette[name]) for name in mechanisms],
                labels=[name.replace("_", " ") for name in mechanisms],
                loc="upper left",
                bbox_to_anchor=(gantt.x1 + 0.04, gantt.y1), ncol=1,
                frameon=False, fontsize=ctx.theme.size("caption"),
                title="Why the gap, as the tracker classified it",
                title_fontsize=ctx.theme.size("caption"),
            )
    if "arrivals" in axes:
        common.histogram(axes["arrivals"], data["first_hour"], ctx.theme,
                         bins=int(ctx.option("bins")), role="named",
                         x_label="First appearance (hours)", y_label="Cells")
    if "coverage" in axes:
        common.histogram(axes["coverage"], data["coverage"], ctx.theme,
                         bins=int(ctx.option("bins")), role="unclaimed",
                         x_label="Share of recording observed", y_label="Cells")
    subtitle = (f"{len(data)} identities ordered by {order.replace('_', ' ')}; "
                "cuts in a bar are unnamed frames inside a lifespan")
    subtitle += (f", coloured by the tracker's explanation where it recorded one."
                 if mechanisms else ".")
    if silent:
        subtitle += (f"\n{len(silent)} of {len(data)} cells end silently and away "
                     "from the field edge, so the tracker stopped finding them "
                     "rather than watching them leave.")
    auxiliary = {}
    if "mechanism" in gaps.columns:
        # Only the frames the bars actually cut. ``~named`` would also sweep in
        # every frame before a cell arrived and after it left, which are blank
        # on the figure and are not gaps.
        auxiliary["gap_mechanisms.csv"] = gaps.loc[
            gaps["state"] == "unclaimed",
            ["identity", "frame_index", "hours", "mechanism"]]
    footnote = units_note(ctx.summary)
    if mechanisms:
        footnote += (
            "\nOnly gaps that are still gaps in the accepted labels are coloured; "
            "one the tracker could not classify stays plain orange."
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=subtitle,
        footnote=footnote,
        auxiliary=auxiliary,
    )


if __name__ == "__main__":
    run_figure("lifespan-gantt")
