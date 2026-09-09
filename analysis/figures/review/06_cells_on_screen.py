"""Figure 6: field occupancy, persistence, lifespan and foreground attribution.

Each reusable panel can also be the whole page::

    python analysis/figures/06_cells_on_screen.py <run>
    ... --panels counts         just the names-per-frame panel
    ... --panels foreground     claimed and unattributed foreground
    ... --panels persistence    every identity in every frame
    ... --panels lifespan       reason-coloured lifespan timeline
    ... --panels arrivals       first-appearance distribution
    ... --panels coverage       observed share of each identity's lifespan
    ... --order coverage        order the lifespan rows by observed share
    ... --hour-ticks 12

The persistence raster has exactly three state colours. Tracker explanations
are symbols over temporary absences there and gap colours in the lifespan
timeline, never an extra grey persistence state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import common
from panels import presence as presence_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


def _lifespan_inputs(
    presence: pd.DataFrame,
    summary: pd.DataFrame,
    history_gaps: pd.DataFrame | None,
    history_lifespans: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], set[int], int]:
    """Prepare the cell spans and accepted gap explanations used by the timeline."""
    longest: dict[int, int] = {}
    for identity, group in presence[presence["state"] == "unclaimed"].groupby("identity"):
        frames = np.sort(group["frame_index"].to_numpy(int))
        runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
        longest[int(identity)] = max((len(run) for run in runs), default=0)

    required = {
        "identity", "first_hour", "last_hour", "observed_frames", "gap_frames",
        "coverage",
    }
    missing = required - set(summary)
    if missing:
        raise SystemExit(
            "the lifespan panels need cell_summary.csv columns: "
            + ", ".join(sorted(missing))
        )
    data = summary[list(required)].copy()
    data["span_hours"] = data["last_hour"] - data["first_hour"]
    data["longest_gap_frames"] = data["identity"].map(longest).fillna(0).astype(int)
    data["ever_touches_border"] = summary.get("ever_touches_border", False)
    ordered = [
        "identity", "first_hour", "last_hour", "span_hours", "observed_frames",
        "gap_frames", "coverage", "longest_gap_frames", "ever_touches_border",
    ]
    data = data[ordered]

    gaps = presence.copy()
    gaps["named"] = gaps["state"] == "named"
    if history_gaps is not None and "mechanism" in history_gaps:
        accepted = history_gaps.copy()
        if "still_missing_in_accepted_labels" in accepted:
            accepted = accepted[accepted["still_missing_in_accepted_labels"].astype(bool)]
        gaps = gaps.merge(
            accepted[["identity", "frame_index", "mechanism"]]
            .drop_duplicates(["identity", "frame_index"]),
            on=["identity", "frame_index"], how="left",
        )
    if "mechanism" not in gaps:
        gaps["mechanism"] = None
    mechanisms = sorted({
        str(value) for value in gaps["mechanism"].dropna().unique()
        if str(value) not in ("", "nan")
    })
    unexplained = int(
        ((gaps["state"] == "unclaimed") & gaps["mechanism"].isna()).sum()
    )

    silent: set[int] = set()
    if (history_lifespans is not None
            and "silent_nonborder_ending" in history_lifespans):
        silent = {
            int(identity) for identity in history_lifespans.loc[
                history_lifespans["silent_nonborder_ending"].astype(bool),
                "identity",
            ]
        }
        keep = [
            column for column in (
                "identity", "silent_nonborder_ending", "last_mask_touches_border",
                "termination_mechanism",
            ) if column in history_lifespans
        ]
        data = data.merge(history_lifespans[keep], on="identity", how="left")
        data["silent_nonborder_ending"] = (
            data["silent_nonborder_ending"].astype("object").where(
                data["silent_nonborder_ending"].notna(), False
            ).astype(bool)
        )
    return data, gaps, mechanisms, silent, unexplained


@figure(
    number=6,
    slug="cells-on-screen",
    purpose="review",
    summary="who is on screen, why identities disappear, and who owns the foreground",
    title="Cells on screen: presence, persistence, lifespan, and foreground",
    grammar="reason-coloured lifespan timeline with persistence raster and categorical summary bars",
    reads=(
        Table("presence_frame.csv", module="presence"),
        Table("presence.csv", module="presence"),
        Table("cell_summary.csv", module="presence"),
        Table("history_gap_frames.csv", module="history", optional=True),
        Table("history_lifespans.csv", module="history", optional=True),
    ),
    panels=(
        Panel("counts", presence_panels.on_screen,
              min_width_inches=12.1, min_height_inches=5.0,
              title="Cell identities carrying a mask"),
        Panel("foreground", presence_panels.unclaimed_area,
              min_width_inches=12.1, min_height_inches=5.0,
              title="Claimed and unclaimed detected foreground",
              needs=("presence_frame.csv:unclaimed_px",)),
        Panel("persistence", presence_panels.persistence_raster,
              min_width_inches=12.1, min_height_inches=6.3,
              title="Identity state in every frame"),
        Panel("lifespan", presence_panels.lifespan_bars,
              min_width_inches=12.1, min_height_inches=8.0,
              title="Named lifespan and the reason for each gap"),
        Panel("arrivals", common.histogram,
              min_width_inches=12.1, min_height_inches=5.0,
              title="First appearance times"),
        Panel("coverage", common.histogram,
              min_width_inches=12.1, min_height_inches=5.0,
              title="Fraction of each lifespan carrying a name"),
    ),
    options=(
        Option("order", default="first_appearance", cast=str,
               help="lifespan row order: first_appearance, span or coverage"),
        Option("bins", default=20),
        Option("hour_ticks", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    hour_ticks = ctx.option("hour_ticks")
    order = str(ctx.option("order"))
    if order not in {"first_appearance", "span", "coverage"}:
        raise SystemExit("--order must be first_appearance, span, or coverage")
    frames = ctx.table("presence_frame.csv")
    presence = ctx.table("presence.csv")
    history_gaps = ctx.optional_table("history_gap_frames.csv")
    lifespans = ctx.optional_table("history_lifespans.csv")
    drawn = ctx.panels()

    needs_lifespan = bool({"lifespan", "arrivals", "coverage"} & set(drawn.keys))
    lifespan_data = pd.DataFrame()
    gaps = pd.DataFrame()
    mechanisms: list[str] = []
    silent: set[int] = set()
    unexplained = 0
    if needs_lifespan:
        summary = ctx.table("cell_summary.csv")
        lifespan_data, gaps, mechanisms, silent, unexplained = _lifespan_inputs(
            presence, summary, history_gaps, lifespans
        )

    persistence_table, matrix, persistence_hours, identity_order = (
        presence_panels.persistence_values(presence)
    )
    events = presence_panels.persistence_events(
        persistence_table, history_gaps, lifespans)

    hours = frames["hours"].to_numpy(float)
    named = frames["identities_named"].to_numpy(float)
    expected = frames["identities_expected"].to_numpy(float)
    total = int(frames["identities_total"].iloc[0])
    missing = int((expected - named).sum())

    show_events = "persistence" in drawn and not events.empty
    show_lifespan_key = "lifespan" in drawn
    needs_right_key = show_events or show_lifespan_key
    minimum_sizes = {}
    if show_lifespan_key:
        minimum_sizes["lifespan"] = (
            12.1, max(8.0, len(lifespan_data) * 0.09)
        )
    figure_, axes = ctx.layout(
        drawn,
        minimum_sizes=minimum_sizes,
        top_inches=2.2,
        bottom_inches=(3.45 if {"persistence", "lifespan"} & set(drawn.keys)
                       else 1.8),
        right_inches=5.8 if needs_right_key else 0.8,
        gap_inches=1.4,
    )

    panel_tables: dict[str, pd.DataFrame] = {}
    persistence_events_export = events.copy()
    gap_mechanisms_export = pd.DataFrame()

    if "counts" in drawn:
        result = ctx.drew("counts", presence_panels.on_screen(
            axes["counts"], hours, named, expected, ctx.theme,
            hour_ticks=hour_ticks, total=total,
            show_x=(drawn.keys[-1] == "counts"),
        ))
        panel_tables["counts"] = result.data
        ctx.theme.legend(axes["counts"], location="inside",
                         fontsize=ctx.theme.size("caption"))

    if "foreground" in drawn:
        result = ctx.drew("foreground", presence_panels.unclaimed_area(
            axes["foreground"], hours, frames["unclaimed_px"], ctx.theme,
            claimed=frames["assigned_px"], hour_ticks=hour_ticks,
            show_x=(drawn.keys[-1] == "foreground"),
        ))
        panel_tables["foreground"] = result.data
        ctx.theme.legend(axes["foreground"], location="inside",
                         fontsize=ctx.theme.size("caption"))

    if "persistence" in drawn:
        result = ctx.drew("persistence", presence_panels.persistence_raster(
            axes["persistence"], matrix, persistence_hours, ctx.theme,
            hour_ticks=hour_ticks, events=events, legend=False,
        ))
        axes["persistence"]._semantic_legend_handled = True
        persistence_export = persistence_table[
            ["identity", "row_position", "frame_index", "hours", "state",
             "state_code"]
        ].copy()
        persistence_export["state_colour"] = persistence_export["state"].map({
            state: ctx.theme.colour(role)
            for state, role in presence_panels.STATE_ROLES.items()
        })
        panel_tables["persistence"] = persistence_export
        if not persistence_events_export.empty:
            persistence_events_export["marker"] = (
                persistence_events_export["event_label"]
                .map(result.extra["event_styles"])
            )
        state_order = ("named", "unclaimed", "outside_lifespan")
        position = axes["persistence"].get_position()
        figure_.legend(
            handles=[
                Patch(facecolor=ctx.theme.colour(presence_panels.STATE_ROLES[state]))
                for state in state_order
            ],
            labels=[presence_panels.STATE_LABELS[state] for state in state_order],
            loc="lower left",
            bbox_to_anchor=(
                position.x1 + 0.25 / figure_.get_figwidth(), position.y0
            ),
            ncol=1, frameon=False, fontsize=ctx.theme.size("caption"),
            title="Persistence state",
            title_fontsize=ctx.theme.size("caption"),
        )

    if "lifespan" in drawn:
        result = ctx.drew("lifespan", presence_panels.lifespan_bars(
            axes["lifespan"], lifespan_data, ctx.theme, gaps=gaps,
            order=order, hour_ticks=hour_ticks, y_label="Cell",
            legend=False, mark=silent,
        ))
        lifespan_export = result.data.copy()
        lifespan_export["present_colour"] = ctx.theme.colour("named")
        lifespan_export["end_marker"] = np.where(
            lifespan_export["marked"].astype(bool), "x", ""
        )
        panel_tables["lifespan"] = lifespan_export
        axes["lifespan"]._semantic_legend_handled = True
        palette = result.extra["mechanism_colours"]
        gap_mechanisms_export = gaps.loc[
            gaps["state"] == "unclaimed",
            ["identity", "frame_index", "hours", "mechanism"],
        ].copy()
        gap_mechanisms_export["mechanism_label"] = (
            gap_mechanisms_export["mechanism"].map(
                lambda value: (presence_panels.mechanism_label(value)
                               if pd.notna(value)
                               else "Gap with no recorded explanation")
            )
        )
        gap_mechanisms_export["gap_colour"] = (
            gap_mechanisms_export["mechanism"].map(palette)
            .fillna(ctx.theme.colour("unclaimed"))
        )
        handles = [Patch(facecolor=ctx.theme.colour("named"))]
        labels = ["Cell identity present"]
        if unexplained:
            handles.append(Patch(facecolor=ctx.theme.colour("unclaimed")))
            labels.append("Gap with no recorded explanation")
        if silent:
            handles.append(Line2D(
                [], [], marker="x", linestyle="none",
                color=ctx.theme.colour("invalid"),
            ))
            labels.append("Silent ending away from the field edge")
        handles.extend(Patch(facecolor=palette[name]) for name in mechanisms)
        labels.extend(presence_panels.mechanism_label(name) for name in mechanisms)
        position = axes["lifespan"].get_position()
        figure_.legend(
            handles=handles, labels=labels,
            loc="upper left",
            bbox_to_anchor=(position.x1 + 0.25 / figure_.get_figwidth(), position.y1),
            ncol=1, frameon=False, fontsize=ctx.theme.size("caption"),
            title="Lifespan state and tracker explanation",
            title_fontsize=ctx.theme.size("caption"),
        )

    if "arrivals" in drawn:
        result = ctx.drew("arrivals", common.histogram(
            axes["arrivals"], lifespan_data["first_hour"], ctx.theme,
            bins=int(ctx.option("bins")), role="named",
            x_label="First appearance (hours)", y_label="Cells",
        ))
        panel_tables["arrivals"] = result.data

    if "coverage" in drawn:
        result = ctx.drew("coverage", common.histogram(
            axes["coverage"], lifespan_data["coverage"], ctx.theme,
            bins=int(ctx.option("bins")), role="unclaimed",
            x_label="Share of own lifespan observed", y_label="Cells",
        ))
        panel_tables["coverage"] = result.data

    first_panel = drawn.keys[0]
    figure_data = panel_tables[first_panel]
    auxiliary = {
        f"{key}.csv": table for key, table in panel_tables.items()
        if key != first_panel
    }
    if "persistence" in drawn:
        auxiliary["persistence_events.csv"] = persistence_events_export
    if needs_lifespan:
        auxiliary["lifespan_summary.csv"] = lifespan_data
    if "lifespan" in drawn:
        auxiliary["gap_mechanisms.csv"] = gap_mechanisms_export

    lifespan_sentence = ""
    if needs_lifespan:
        lifespan_sentence = (
            f" Lifespans are ordered by {order.replace('_', ' ')}; "
            f"{len(silent)} identities end silently away from the field edge."
        )
    persistence_sentence = (
        f" The persistence raster contains {len(identity_order)} cells"
        + (f" and {len(events)} tracker events." if show_events else ".")
        if "persistence" in drawn else ""
    )

    return FigureResult(
        figure=figure_,
        axes=list(axes.values()),
        figure_data=figure_data,
        auxiliary=auxiliary,
        heading="Cells on screen, persistence, and lifespan",
        subtitle=(
            f"{ctx.summary['stem']}, {ctx.summary['frames']} frames over "
            f"{ctx.summary['hours_covered']:.0f} h at "
            f"{ctx.summary['minutes_per_frame']:.0f} min "
            f"per frame, {ctx.field['width']} x {ctx.field['height']} px.\n"
            f"{named.mean():.0f} names carry a mask per frame on average "
            f"(range {named.min():.0f}-{named.max():.0f}) out of {total} identities "
            f"in the movie."
            + persistence_sentence
            + lifespan_sentence
        ),
        footnote=ctx.footnote(
            "The dashed count is the number of cells between their first and last "
            f"appearance; the {missing:,} cell-frames below it are temporary absences.\n"
            "Raster colours are fixed: black is on screen, orange is temporarily off, "
            "and grey is before first appearance or after last. Symbols are the "
            "tracker's recorded gap causes; a cross is a silent ending away from the "
            "field edge.\nIn the lifespan timeline, gap colours are the tracker's "
            "recorded explanations; an unexplained gap remains orange.\nClaimed "
            "foreground is mask area; unclaimed foreground is detected cell signal "
            "assigned to no identity."
        ),
        readme="""## What the figure shows

**Counts.** Distinct identities carrying a mask in each frame, against identities
between their own first and last appearance.

**Foreground.** Mask pixels claimed by identities and detected cell foreground
assigned to no identity, stacked on the same pixel scale.

**Persistence.** One row per identity and one column per frame. Black is on
screen, orange is a temporary absence inside that identity's observed lifespan,
and grey is before its first appearance or after its last. One symbol marks the
midpoint of each temporary-absence interval and names the tracker's recorded
cause. A cross marks a silent ending away from the field edge.

**Lifespan.** One bar per identity from first to last appearance. Present frames
are black. A missing frame is coloured by the tracker's recorded explanation;
one with no explanation stays orange. A cross marks a silent ending away from
the field edge. `--order` accepts `first_appearance`, `span`, or `coverage`.

**Arrivals and coverage.** The first distribution counts when identities first
appear. The second counts the share of each identity's own lifespan for which a
mask is present; it is not a share of the entire recording.

Each panel can be drawn alone with `--panels counts`, `foreground`,
`persistence`, `lifespan`, `arrivals`, or `coverage`.

## Where the numbers come from

`presence_frame.csv` supplies the counts and foreground totals. `presence.csv`
supplies the three states per identity per frame. Optional
`history_gap_frames.csv` and `history_lifespans.csv` supply the same event
classifications used by the persistence and lifespan panels. `cell_summary.csv`
supplies first appearance, lifespan span, and observed coverage.

The foreground panel is offered only when the run was given an unclaimed stack.
A run without one is not wrong; it cannot answer that panel.""",
        console=(f"panels {drawn.keys}  frames: {len(frames)}  "
                 f"named per frame: {named.mean():.1f}  "
                 f"unnamed cell-frames: {missing}  mechanisms: {len(mechanisms)}"),
    )


if __name__ == "__main__":
    run_figure("cells-on-screen", DEFAULT_RUN)
