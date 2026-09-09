"""Figure 29: pixel fate across user-selected events or recording stages.

Every pixel is classified within each event window and the flows between
classes are drawn. User-selected event times start the windows; without them,
the figure falls back to evenly spaced recording stages. The sensitivity panel
repeats the flow at a second pair of cut-offs.

    python analysis/figures/29_pixel_fate_flow.py <run>
    ... --events Baseline,Treatment,Recovery --event-times 0,24,36
    ... --stages 12              more time slices
    ... --thresholds 0.6,0.3     core and transient cut-offs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import tifffile

from _derive import _selected_identities
from _schema import FigureContext, FigureResult, Input, Option, Panel, figure, run_figure

from panels import common
from panels import territory as territory_panels


@figure(
    number=29,
    slug="pixel-fate-flow",
    summary="pixel fate across selected events and times",
    title="Pixel fate across selected events and times",
    reads=(Input("labels"),),
    panels=(
        Panel("flow", territory_panels.fate_flow, block=True,
              title="Pooled pixel fate"),
        Panel("shares", common.stacked_area,
              title="Pixel-fate composition through time"),
        Panel("sensitivity", territory_panels.fate_flow, block=True,
              title="Threshold-sensitivity comparison"),
    ),
    options=(
        Option("stages", default=8),
        Option("events", default=[]),
        Option("event_times", default=[]),
        Option("thresholds", default=[0.8, 0.2]),
        Option("cells", default="4", cast=str),
        Option("hour_ticks", default=24.0),
    ),
    grammar="alluvial flows and stacked shares",
)
def build(ctx: FigureContext) -> FigureResult:
    label_path = ctx.input_path("labels")
    if label_path is None:
        raise SystemExit("pixel-fate-flow needs the labels input recorded in the run manifest")
    labels = tifffile.imread(label_path)
    frame_hours = np.arange(labels.shape[0], dtype=float) * ctx.interval / 60.0
    requested_names = [str(value) for value in ctx.option("events")]
    requested_times = np.asarray(ctx.option("event_times"), dtype=float)
    if requested_names and not requested_times.size:
        raise SystemExit("--events needs matching --event-times")
    if requested_times.size:
        if requested_times.size < 2:
            raise SystemExit("--event-times needs at least two event starts")
        if not np.isfinite(requested_times).all():
            raise SystemExit("--event-times must contain finite hours")
        if np.any(np.diff(requested_times) <= 0):
            raise SystemExit("--event-times must be in strictly increasing order")
        if requested_times[0] < 0 or requested_times[-1] > frame_hours[-1]:
            raise SystemExit(
                f"--event-times must lie between 0 and {frame_hours[-1]:g} hours"
            )
        event_names = requested_names or [
            f"Event {index + 1}" for index in range(len(requested_times))
        ]
        if len(event_names) != len(requested_times):
            raise SystemExit("--events and --event-times must contain the same number of values")
        if len(set(event_names)) != len(event_names):
            raise SystemExit("--events names must be unique")
        first_frames = np.searchsorted(frame_hours, requested_times, side="left")
        if np.any(np.diff(first_frames) <= 0):
            raise SystemExit("two --event-times resolve to the same recorded frame")
        bins = [
            np.arange(first_frames[index],
                      first_frames[index + 1] if index + 1 < len(first_frames) else len(labels),
                      dtype=int)
            for index in range(len(first_frames))
        ]
        display_hours = requested_times
        event_mode = "user-selected event windows"
    else:
        n_stages = max(2, int(ctx.option("stages")))
        bins = [
            np.asarray(values, dtype=int)
            for values in np.array_split(np.arange(labels.shape[0]), n_stages)
        ]
        if any(not len(values) for values in bins):
            raise SystemExit(f"--stages cannot exceed the {len(labels)} recorded frames")
        event_names = [f"Stage {index + 1}" for index in range(n_stages)]
        display_hours = np.asarray([frame_hours[values].mean() for values in bins])
        event_mode = "evenly spaced recording stages"
    event_windows = dict(zip(event_names, bins, strict=True))
    event_labels = [
        f"{name}\n{hour:g} h" for name, hour in zip(event_names, display_hours, strict=True)
    ]
    event_table = pd.DataFrame([
        {
            "event_index": index,
            "event": name,
            "event_time_hours": float(display_hours[index]),
            "requested_start_hours": (
                float(requested_times[index]) if requested_times.size else np.nan
            ),
            "first_frame": int(indices.min()),
            "last_frame": int(indices.max()),
            "window_start_hours": float(frame_hours[indices.min()]),
            "window_end_hours": float(frame_hours[indices.max()] + ctx.interval / 60.0),
            "frames": int(len(indices)),
        }
        for index, (name, indices) in enumerate(event_windows.items())
    ])
    thresholds = ctx.option("thresholds")
    if len(thresholds) not in {2, 4}:
        raise SystemExit("--thresholds needs core,transient or primary_core,primary_transient,alternate_core,alternate_transient")
    primary = thresholds[:2]
    alternate = thresholds[2:] if len(thresholds) == 4 else [0.7, 0.3]
    for pair in (primary, alternate):
        if not (0 <= pair[1] < pair[0] <= 1):
            raise SystemExit("each --thresholds pair must satisfy 0 <= transient < core <= 1")
    support = pd.DataFrame({"identity": [int(value) for value in np.unique(labels) if value]})
    support["observed_frames"] = support["identity"].map(lambda identity: int(np.count_nonzero(np.any(labels == identity, axis=(1, 2)))))
    selected = _selected_identities(support, ctx.option("cells"))
    class_labels = list(territory_panels.FATE_ROLES)
    primary_pooled, primary_composition = territory_panels.pixel_fate_states(
        labels, event_windows,
        core_fraction=float(primary[0]), transient_fraction=float(primary[1]),
    )
    primary_matrices = {0: primary_pooled}
    composition_tables = [primary_composition.assign(identity=0)]
    for identity in selected:
        matrix, composition = territory_panels.pixel_fate_states(
            labels, event_windows, identities=[identity],
            core_fraction=float(primary[0]), transient_fraction=float(primary[1]),
        )
        primary_matrices[identity] = matrix
        composition_tables.append(composition.assign(identity=identity))
    alternate_matrix, alternate_composition = territory_panels.pixel_fate_states(
        labels, event_windows,
        core_fraction=float(alternate[0]), transient_fraction=float(alternate[1]),
    )
    panels = ctx.panels()
    fig, axes = ctx.layout(
        panels, top_inches=3.5, bottom_inches=2.2, gap_inches=3.0,
    )
    flow_tables = []
    if "flow" in axes:
        rect = tuple(axes["flow"].get_position().bounds)
        axes["flow"].remove()
        drawn = ctx.drew(
            "flow",
            territory_panels.fate_flow(
                fig, rect, primary_matrices[0], ctx.theme, stage_labels=event_labels,
            ),
        )
        axes["flow"] = drawn.axes
        axes["flow"]._semantic_legend_handled = True
        fig.text(
            rect[0], rect[1] + rect[3] + 0.037,
            f"Pooled pixel fate — core ≥{primary[0]:.0%}; transient ≤{primary[1]:.0%}",
            fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom",
        )
    for identity, matrix in primary_matrices.items():
        table = territory_panels.fate_transition_table(
            matrix, event_labels=event_names, class_names=class_labels,
        )
        table.insert(0, "identity", identity)
        event_time_lookup = dict(zip(event_names, display_hours, strict=True))
        table["from_event_time_hours"] = table["from_event"].map(event_time_lookup)
        table["to_event_time_hours"] = table["to_event"].map(event_time_lookup)
        flow_tables.append(table)
    figure_data = pd.concat(flow_tables, ignore_index=True)
    if "shares" in axes:
        shares = (primary_composition.pivot(index="event", columns="class", values="pixels")
                  .reindex(index=event_names, columns=class_labels).fillna(0))
        share_result = ctx.drew(
            "shares",
            common.stacked_area(
                axes["shares"], display_hours, class_labels, shares, ctx.theme,
                looks={name: common.Look(colour=ctx.theme.colour(
                    territory_panels.FATE_ROLES[name])) for name in class_labels},
                labels=[territory_panels.FATE_LABELS[name] for name in class_labels],
                x_label="Hours from start of recording", y_label="Pooled pixels",
            ),
        )
        share_result.data.insert(1, "event", event_names)
        if requested_times.size:
            axes["shares"].set_xticks(display_hours)
            axes["shares"].set_xticklabels(event_labels)
        else:
            axes["shares"].set_xticks(ctx.theme.hour_ticks(
                float(display_hours.min()), float(display_hours.max()), ctx.hour_ticks))
        common.semantic_legend(axes["shares"], ctx.theme, location="inside")
        legend = axes["shares"].get_legend()
        if legend is not None:
            legend.remove()
        axes["shares"]._semantic_legend_handled = True
    sensitivity_data = pd.DataFrame()
    if "sensitivity" in axes:
        rect = tuple(axes["sensitivity"].get_position().bounds)
        axes["sensitivity"].remove()
        sensitivity_result = ctx.drew(
            "sensitivity",
            territory_panels.fate_flow(
                fig, rect, alternate_matrix, ctx.theme, stage_labels=event_labels,
            ),
        )
        sensitivity_data = territory_panels.fate_transition_table(
            alternate_matrix, event_labels=event_names, class_names=class_labels,
        )
        axes["sensitivity"] = sensitivity_result.axes
        axes["sensitivity"]._semantic_legend_handled = True
        fig.text(
            rect[0], rect[1] + rect[3] + 0.037,
            f"Threshold sensitivity — core ≥{alternate[0]:.0%}; transient ≤{alternate[1]:.0%}",
            fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom",
        )
    if "flow" in axes or "sensitivity" in axes:
        fig.legend(
            handles=[Patch(facecolor=ctx.theme.colour(territory_panels.FATE_ROLES[name]))
                     for name in class_labels],
            labels=[territory_panels.FATE_LABELS[name] for name in class_labels],
            loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(class_labels),
            frameon=False, fontsize=ctx.theme.size("caption"),
        )
    window_count = len(event_names)
    pooled_cells = len(support)
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=(f"{window_count} {event_mode}; {pooled_cells} cells pooled and "
                  f"{len(selected)} cell-level tables exported. Threshold pairs "
                  f"{primary[0]:g}/{primary[1]:g} and {alternate[0]:g}/{alternate[1]:g}."),
        footnote=ctx.footnote(
            "Each selected event time starts one window; the next event closes it and the final window ends with the recording. Core, fringe and transient describe the fraction of frames occupied within that window; transferred means the modal cell identity at that pixel changed since the preceding window.",
        ),
        auxiliary={
            "event_windows.csv": event_table,
            "pixel_fate_composition.csv": pd.concat(composition_tables, ignore_index=True),
            "sensitivity_composition.csv": alternate_composition,
            "sensitivity_flow.csv": sensitivity_data,
        },
    )


if __name__ == "__main__":
    run_figure("pixel-fate-flow")
