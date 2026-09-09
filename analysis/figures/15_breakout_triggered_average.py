"""Figure 15: responses around frames ranked by a chosen measurement.

The ranking input and plotted responses are independent choices. By default,
each cell contributes the frame with its highest footprint-turnover index, and
the response panels show standardised area, ramification and reporter signal.
The next-ranked frames are shown separately as a rank comparison, not called a
control and not described as noise.

    python analysis/figures/15_breakout_triggered_average.py <run>
    ... --event-metric step_px --event-direction high
    ... --top-events 3 --metrics area_px,solidity
    ... --window 24              frames either side of each selected frame
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _derive import _aligned_curves
from _metrics import documented, semantic_label
from _schema import (FigureContext, FigureResult, Length, Option, Panel, Table,
                     figure, run_figure)

from panels import common
from panels import surveillance as surveillance_panels


EVENT_DIRECTIONS = {
    "high": "highest values",
    "low": "lowest values",
    "deviation": "largest absolute deviations from the within-cell median",
}


def _selection_text(metric_label: str, direction: str, count: int) -> str:
    """A title fragment that states exactly what was ranked."""
    label = metric_label[:1].lower() + metric_label[1:]
    if direction == "deviation":
        if count == 1:
            return f"largest within-cell deviation in {label}"
        return f"{count} largest within-cell deviations in {label}"
    order = "highest" if direction == "high" else "lowest"
    if count == 1:
        return f"frame with {order} {label}"
    return f"{count} frames with {order} {label}"


def _input_definition(metric: str) -> str | None:
    """Define transition measurements whose short labels need a formula."""
    return {
        "turnover_index": (
            "Footprint turnover fraction = (gained pixels + lost pixels) / "
            "(current footprint pixels + previous footprint pixels); it measures "
            "outline replacement, not centroid displacement."
        ),
        "balanced_turnover_fraction": (
            "Balanced footprint turnover = 1 - |gained pixels - lost pixels| / "
            "(gained pixels + lost pixels); it is highest when gained and lost "
            "areas are equal."
        ),
        "step_px": (
            "Centroid step is the centroid displacement from the preceding "
            "tracked frame."
        ),
    }.get(metric)


def _standardise_responses(
    aligned: pd.DataFrame,
    source: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Scale from each cell's full trace, not its duplicated event windows."""
    result = aligned.copy()
    for metric in metrics:
        numeric = pd.to_numeric(source[metric], errors="coerce")
        grouped = numeric.groupby(source["identity"])
        centre = grouped.mean()
        scale = grouped.std(ddof=0).replace(0, 1).fillna(1)
        result[metric] = (
            pd.to_numeric(result[metric], errors="coerce")
            - result["identity"].map(centre)
        ) / result["identity"].map(scale)
    return result


def _event_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per selected cell-frame."""
    columns = [
        "stem", "condition", "subject", "identity", "event_rank",
        "event_frame_index", "event_hours", "event_metric", "event_direction",
        "event_value", "event_score", "event_centre", "event_scale",
    ]
    columns = [column for column in columns if column in aligned]
    return aligned[columns].drop_duplicates().reset_index(drop=True)


def _plot_table(table: pd.DataFrame, metric: str, interval_minutes: float) -> pd.DataFrame:
    """Name the generic panel offset in the physical unit drawn here."""
    result = table.rename(columns={"offset": "offset_hours"}).copy()
    result.insert(0, "metric", metric)
    result.insert(
        2, "offset_frames",
        np.rint(result["offset_hours"] * 60.0 / interval_minutes).astype(int),
    )
    return result


@figure(
    number=15,
    slug="breakout-triggered-average",
    summary="response measurements around frames ranked by a chosen input measurement",
    title="Responses around each cell's {selection}",
    reads=(Table("cell_frame.csv", module="surveillance"),),
    panels=(
        Panel("triggered", common.event_average,
              title="Responses around the selected frames"),
        Panel("alignment", surveillance_panels.event_timing,
              title="Selected frame times for each cell"),
        Panel("comparison", common.event_average,
              title="Responses around the next-ranked frames"),
    ),
    options=(
        Option("metrics",
               default=["area_px", "ramification_index", "corrected_mean"],
               help="which response measurements are drawn around the selected frames"),
        Option("event_metric", default="turnover_index"),
        Option("event_direction", default="high"),
        Option("top_events", default=1),
        Option("window", default=12),
    ),
    grammar="event-aligned response traces and event-time raster",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv").copy()
    gross = data["gained_px"] + data["lost_px"]
    # New runs write this explicitly. Derive it here as a compatibility bridge
    # for old runs whose surveillance table predates the declared column.
    if "balanced_turnover_fraction" not in data:
        data["balanced_turnover_fraction"] = np.where(
            gross > 0,
            1 - (data["gained_px"] - data["lost_px"]).abs() / gross,
            np.nan,
        )

    event_metric = str(ctx.option("event_metric"))
    direction = str(ctx.option("event_direction"))
    metrics = [str(metric) for metric in ctx.option("metrics")]
    top_events = int(ctx.option("top_events"))
    window = int(ctx.option("window"))
    if direction not in EVENT_DIRECTIONS:
        raise SystemExit(
            f"--event-direction {direction} is not supported; choose "
            + ", ".join(EVENT_DIRECTIONS)
        )
    if top_events < 1:
        raise SystemExit("--top-events must be at least 1")
    if window < 0:
        raise SystemExit("--window cannot be negative")

    available = [
        column for column in data
        if documented(column) and pd.api.types.is_numeric_dtype(data[column])
    ]
    unknown = [metric for metric in [event_metric, *metrics] if metric not in available]
    if unknown:
        raise SystemExit(
            f"The requested measured column(s) are unavailable: {', '.join(unknown)}. "
            f"This cell_frame.csv can plot: {', '.join(available)}"
        )
    if not metrics:
        raise SystemExit("--metrics must name at least one response measurement")

    primary = surveillance_panels.ranked_events(
        data, column=event_metric, direction=direction,
        rank_start=1, top_n=top_events, window=window,
    )
    comparison = surveillance_panels.ranked_events(
        data, column=event_metric, direction=direction,
        rank_start=top_events + 1, top_n=top_events, window=window,
    )
    if primary.empty:
        raise SystemExit(f"No finite {event_metric} values were available to rank")
    primary = _standardise_responses(primary, data, metrics)
    comparison = _standardise_responses(comparison, data, metrics)

    primary_summary, matrices, offsets = _aligned_curves(primary, metrics)
    comparison_summary, comparison_matrices, comparison_offsets = _aligned_curves(
        comparison, metrics
    )
    interval_minutes = float(ctx.interval)
    offset_hours = offsets.astype(float) * interval_minutes / 60.0
    comparison_offset_hours = (
        comparison_offsets.astype(float) * interval_minutes / 60.0
    )
    selected_events = _event_rows(primary)
    comparison_events = _event_rows(comparison)
    selection = _selection_text(semantic_label(event_metric), direction, top_events)

    panels = ctx.panels()
    fig, axes = ctx.layout(panels, gap_inches=1.35, top_inches=2.15)
    colours = common.series_colours(ctx.theme, len(metrics))
    primary_plots: list[pd.DataFrame] = []
    if "triggered" in axes:
        axes["triggered"].set_title(
            f"Responses around the {selection}", loc="left",
            fontsize=ctx.theme.size("panel"), fontweight="bold",
        )
        for metric, colour in zip(metrics, colours):
            if not matrices[metric].size:
                continue
            plotted = common.event_average(
                axes["triggered"], offset_hours, matrices[metric], ctx.theme,
                look=common.Look(colour=colour), bootstrap=200,
                x_label="Hours from selected frame", label=semantic_label(metric),
            ).data
            primary_plots.append(_plot_table(plotted, metric, interval_minutes))
        axes["triggered"].set_ylabel("Within-cell standardised level\n(z-score)")
    alignment_table = pd.DataFrame()
    if "alignment" in axes:
        alignment_table = ctx.drew(
            "alignment",
            surveillance_panels.event_timing(
                axes["alignment"], primary, ctx.theme,
                label=("Selected frame (one per cell)" if top_events == 1
                       else f"Selected top {top_events} frames per cell"),
            ),
        ).data
    comparison_plots: list[pd.DataFrame] = []
    if "comparison" in axes:
        next_start = top_events + 1
        next_end = top_events * 2
        comparison_title = (
            "Rank comparison: second-ranked frame by the same rule"
            if top_events == 1
            else f"Rank comparison: frames ranked {next_start}-{next_end} by the same rule"
        )
        axes["comparison"].set_title(
            comparison_title,
            loc="left", fontsize=ctx.theme.size("panel"), fontweight="bold",
        )
        for metric, colour in zip(metrics, colours):
            if not comparison_matrices[metric].size:
                continue
            plotted = common.event_average(
                axes["comparison"], comparison_offset_hours,
                comparison_matrices[metric], ctx.theme,
                look=common.Look(colour=colour), bootstrap=200,
                x_label="Hours from next-ranked frame",
                label=semantic_label(metric),
            ).data
            comparison_plots.append(_plot_table(plotted, metric, interval_minutes))
        if not comparison_plots:
            axes["comparison"].text(
                0.5, 0.5, "No next-ranked frames available",
                transform=axes["comparison"].transAxes,
                ha="center", va="center", color=ctx.theme.colour("caption"),
            )
        axes["comparison"].set_ylabel("Within-cell standardised level\n(z-score)")

    primary_plot = (
        pd.concat(primary_plots, ignore_index=True) if primary_plots else pd.DataFrame()
    )
    comparison_plot = (
        pd.concat(comparison_plots, ignore_index=True)
        if comparison_plots else pd.DataFrame()
    )
    if not primary_plot.empty:
        figure_data = primary_plot
    elif not alignment_table.empty:
        figure_data = alignment_table
    else:
        figure_data = comparison_plot

    response_names = ", ".join(semantic_label(metric) for metric in metrics)
    window_hours = window * interval_minutes / 60.0
    event_count_text = (
        "one frame per cell" if top_events == 1
        else f"top {top_events} frames per cell"
    )
    input_definition = _input_definition(event_metric)
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        title_fields={"selection": selection},
        subtitle=(
            f"Input: {semantic_label(event_metric)}, {EVENT_DIRECTIONS[direction]}; "
            f"{event_count_text}. Responses: {response_names}. Window: "
            f"±{window_hours:g} h."
        ),
        footnote=ctx.footnote(
            *([input_definition] if input_definition else []),
            "Ranking is performed separately within each cell. Response values are "
            "standardised using that cell's full trace; bands are 95% bootstrap "
            "intervals across selected cell-frames.",
            ("The bottom panel uses the second-ranked frame from each cell; it is "
             "a rank comparison, not a noise control."
             if top_events == 1 else
             f"The bottom panel uses frames ranked {top_events + 1}-"
             f"{top_events * 2} within each cell; it is a rank comparison, not a "
             "noise control."),
        ),
        subtitle_y=Length(1.05, "in"),
        auxiliary={
            "selected_event_frames.csv": selected_events,
            "selected_response_quartiles.csv": primary_summary,
            **({"next_ranked_event_frames.csv": comparison_events}
               if not comparison_events.empty else {}),
            **({"next_ranked_response_quartiles.csv": comparison_summary}
               if not comparison_summary.empty else {}),
            **({"next_ranked_response_plot.csv": comparison_plot}
               if not comparison_plot.empty else {}),
            **ctx.provenance_auxiliary(),
        },
        console=(
            f"{len(selected_events)} selected frames from "
            f"{selected_events['identity'].nunique()} cells; input {event_metric}; "
            f"responses {', '.join(metrics)}"
        ),
    )


if __name__ == "__main__":
    run_figure("breakout-triggered-average")
