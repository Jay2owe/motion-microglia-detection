"""Figure 38: when each signal's active period runs, cell by cell.

Figure 37 says whether a signal swings daily. This one says when. Every cell is
fitted seven times, once per measured signal, and each fit reports the hour its
active phase begins and the hour it ends. If area, brightness and motility all
start within an hour or two of each other in the same cell, one clock is
driving that cell; if they scatter across the day, "the cell's rhythm" is a
phrase covering several unrelated ones.

Hours here are positions in the folded day the fit was read off, not times of
day. These are slices in a dish: a slice cannot see light, there is no schedule
to be early or late against, and hour 0 is where the recording's own day was
cut, nothing more.

    python analysis/figures/38_active_span.py <run>
    ... --order duration        bar order within each block
    ... --metrics area_px,step_px
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import rhythms as rhythm_panels

#: Bar orders this page offers, and what each is for. Named here so the refusal
#: can list them and the docstring cannot drift from what is accepted.
ORDERS = {
    "onset": "when the active phase starts",
    "duration": "how long it lasts",
    "identity": "the cell's number, so blocks line up row for row",
}


def _circular_range(hours: np.ndarray, period: float = 24.0) -> float:
    """The smallest arc, in hours, containing every one of these hours.

    The plain range is wrong on a wrapped axis: onsets at 23 h and 1 h are two
    hours apart, and subtracting gives twenty-two. This sorts them, finds the
    largest empty gap, and returns what is left of the day - which is the arc
    they actually occupy.
    """
    values = np.sort(np.asarray(hours, dtype=float) % period)
    if values.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([values, [values[0] + period]]))
    return float(period - gaps.max())


@figure(
    number=38,
    slug="active-span",
    summary="when each signal's active period starts and ends, cell by cell",
    title="Active timing of {n_metrics} measured signals across cells",
    grammar="span raster, onset-agreement histogram and rest-to-peak dumbbell",
    reads=(Table("rhythms.csv", module="rhythms"),),
    panels=(
        Panel("spans", rhythm_panels.active_spans,
              title="1. When is each cell active? (one thin row per cell)"),
        Panel("agreement", common.histogram,
              title="2. Do a cell's measured signals start together?"),
        Panel("rest_to_peak", common.dumbbell,
              title="3. When do each signal's quietest and busiest windows begin?"),
    ),
    options=(
        Option("metrics", default=[],
               help="which fitted signals to draw, comma separated; empty draws every one"),
        Option("order", default="onset",
               help="bar order within a block: onset, duration or identity"),
        Option("bins", default=24),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    rhythms = ctx.table("rhythms.csv")
    available = sorted(str(value) for value in rhythms["metric"].dropna().unique())
    wanted = [str(name) for name in ctx.option("metrics")] or available
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise SystemExit(
            f"--metrics {','.join(unknown)} was not fitted in this run. "
            f"rhythms.csv holds: {', '.join(available)}"
        )
    order = str(ctx.option("order"))
    if order not in ORDERS:
        raise SystemExit(
            f"--order {order} is not one this page has. It takes: "
            + "; ".join(f"{name} ({meaning})" for name, meaning in ORDERS.items())
        )

    selected = rhythms[rhythms["metric"].isin(wanted)]
    # onset_found is the flag, not the value. Zero is a real onset here - the
    # minimum in this dataset is exactly 0.0 - so filtering on the number would
    # throw away the cells that start at the fold.
    found = selected[selected["onset_found"].fillna(False).astype(bool)].copy()
    missing = len(selected) - len(found)

    metrics = [name for name in wanted if name in set(found["metric"])]
    labels = [semantic_label(name) for name in metrics]
    # Signal identity is already written beside each block. Giving the blocks
    # seven unexplained colours creates a second, redundant vocabulary; one
    # colour now means one thing throughout this panel: detected active hours.
    active_colours = [ctx.theme.colour("rhythmic")] * len(metrics)

    sort_key = {"onset": ["onset_hour", "identity"],
                "duration": ["active_duration_hours", "identity"],
                "identity": ["identity"]}[order]
    blocks = [found[found["metric"] == name].sort_values(sort_key) for name in metrics]
    ordered = pd.concat(blocks, ignore_index=True) if blocks else found

    panels = ctx.panels()
    # The raster carries one row per cell per signal - several hundred of them -
    # while the two panels under it carry one mark per cell and one per signal.
    # Equal thirds would give every bar half a pixel.
    fig, axes = ctx.layout(panels, weights={"spans": 4.0}, gap_inches=1.35)

    if "spans" in axes:
        ctx.drew("spans", rhythm_panels.active_spans(
            axes["spans"], ordered["onset_hour"], ordered["offset_hour"], ctx.theme,
            groups=ordered["metric"], colours=active_colours,
            x_label="Hour in folded 24 h cycle (bar = detected active period)"))

    # One value per cell: the arc its signals' onsets occupy. A cell fitted on
    # one signal only has nothing to disagree with and is left out rather than
    # entered as a perfect agreement of zero.
    spread = (found.groupby("identity")["onset_hour"]
              .apply(lambda values: _circular_range(values.to_numpy(float)))
              .rename("onset_spread_hours").reset_index())
    counted = found.groupby("identity")["metric"].nunique().rename("signals")
    spread = spread.join(counted, on="identity")
    comparable = spread[spread["signals"] > 1]

    agreement_table = pd.DataFrame()
    if "agreement" in axes:
        agreement_table = ctx.drew("agreement", common.histogram(
            axes["agreement"], comparable["onset_spread_hours"], ctx.theme,
            bins=np.linspace(0.0, 24.0, int(ctx.option("bins")) + 1), role="dark",
            x_label="Within-cell spread of signal start times (h; 0 = together)",
            y_label="Number of cells",
        )).data
        axes["agreement"].set_xlim(0.0, 24.0)
        axes["agreement"].set_xticks(ctx.theme.hour_ticks(0.0, 24.0, 4.0))

    rest_to_peak_table = pd.DataFrame()
    if "rest_to_peak" in axes:
        left = [float(np.median(by["l5_onset_hour"].dropna())) for by in blocks]
        right = [float(np.median(by["m10_onset_hour"].dropna())) for by in blocks]
        rest_to_peak_table = ctx.drew("rest_to_peak", common.dumbbell(
            axes["rest_to_peak"], left, right, ctx.theme,
            labels=labels,
            right_colours=[ctx.theme.colour("highlight")] * len(labels),
            left_marker="o", right_marker="D",
            left_label="Quiet window starts (lowest 5 h)",
            right_label="Busy window starts (highest 10 h)",
        )).data
        axes["rest_to_peak"].set_xlabel(
            "Hour in folded 24 h cycle (median across cells)"
        )
        # Seven rows across a narrow span of hours leave no free corner, so the
        # key goes above the panel rather than on top of the rows.
        common.semantic_legend(axes["rest_to_peak"], ctx.theme, location="right",
                               columns=1)

    median_spread = (float(comparable["onset_spread_hours"].median())
                     if not comparable.empty else float("nan"))
    subtitle = (
        "Top: each bar runs from detected onset to offset. Middle: 0 h means all "
        "measured signals begin together. Bottom: symbols are medians across cells."
    )
    footnote_lines = []
    if missing:
        footnote_lines.append(
            f"{missing} of {len(selected)} fits found no onset at all and are absent "
            "from every panel; the template needs a quiet stretch before a candidate "
            "hour and an active one after it, and not every cell has both."
        )
    footnote_lines.append(
        f"The middle panel is one value per cell across {len(comparable)} cells: the "
        "smallest arc on the 24 h circle that holds every one of that cell's start "
        f"hours (median {median_spread:.1f} h). A cell fitted on one signal only has "
        "nothing to compare and is left out."
    )
    footnote_lines.append(
        "In the bottom panel, the open circle starts the lowest consecutive 5 h "
        "window; the filled diamond starts the highest consecutive 10 h window."
    )
    footnote_lines.append(
        "Hour 0 is where the recording's own day was cut, not a time of day and not "
        "lights-on: these are slices in a dish and a slice cannot detect light. Two "
        "recordings' hours are comparable only where both were handled to the same "
        "protocol."
    )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        subtitle=subtitle,
        footnote=ctx.footnote(*footnote_lines),
        title_fields={"n_metrics": len(metrics)},
        auxiliary={
            "onset_agreement.csv": spread,
            **({"onset_agreement_histogram.csv": agreement_table}
               if not agreement_table.empty else {}),
            **({"rest_to_peak.csv": rest_to_peak_table}
               if not rest_to_peak_table.empty else {}),
            **ctx.provenance_auxiliary(),
        },
        console=(f"{len(metrics)} signals, {len(found)} fits with an onset, "
                 f"{missing} without; median onset arc {median_spread:.1f} h"),
    )


if __name__ == "__main__":
    run_figure("active-span")
