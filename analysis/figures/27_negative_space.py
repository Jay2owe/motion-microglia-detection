"""Figure 27: pixels never visited during the recording.

The complement of everything the cells did. Unclaimed foreground counts as
visited, because a pixel the tracker could not name is still a pixel something
was on - counting it as empty would overstate the empty ground.

    python analysis/figures/27_negative_space.py <run>
    ... --overlay               draw the mask over the raw image
    ... --stages 8              how many time slices the third panel shows
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.patches import Patch
import matplotlib
import numpy as np
import pandas as pd
import tifffile

from _derive import _scaled_field
from _schema import FigureContext, FigureResult, Input, Option, Panel, figure, run_figure

from panels import common
from panels import territory as territory_panels


@figure(
    number=27,
    slug="negative-space",
    summary="pixels never visited during the recording",
    title="Pixels never visited during the recording",
    reads=(Input("labels"), Input("unclaimed"), Input("raw")),
    panels=(
        Panel("mask", territory_panels.never_visited,
              title="Final never-visited pixels"),
        Panel("shrinking", common.trace,
              title="Never-visited field through time"),
        Panel("stages", title="Never-visited field at selected times"),
    ),
    options=(
        Option("stages", default=6),
        Option("hour_ticks", default=24.0),
    ),
    switches=("overlay",),
    grammar="spatial mask timeline and stage maps",
)
def build(ctx: FigureContext) -> FigureResult:
    label_path = ctx.input_path("labels")
    if label_path is None:
        raise SystemExit("negative-space needs the labels input recorded in the run manifest")
    labels = tifffile.imread(label_path)
    visited = labels > 0
    unclaimed_path = ctx.input_path("unclaimed")
    if unclaimed_path is not None:
        visited |= tifffile.imread(unclaimed_path) > 0
    cumulative = np.logical_or.accumulate(visited, axis=0)
    cumulative_px = cumulative.sum(axis=(1, 2)).astype(int)
    newly = np.diff(np.r_[0, cumulative_px]).astype(int)
    field_px = int(labels.shape[1] * labels.shape[2])
    frame_index = np.arange(labels.shape[0], dtype=int)
    data = pd.DataFrame({
        "frame_index": frame_index,
        "hours": frame_index * ctx.interval / 60.0,
        "never_visited_px": field_px - cumulative_px,
        "never_visited_share": (field_px - cumulative_px) / field_px,
        "newly_visited_px": newly,
        "cumulative_visited_px": cumulative_px,
        "field_px": field_px,
    })
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    final_mask = ~cumulative[-1]
    if "mask" in axes:
        background = None
        if ctx.switch("overlay"):
            raw_path = ctx.input_path("raw")
            if raw_path is not None:
                background = tifffile.imread(raw_path)[-1]
        territory_panels.never_visited(axes["mask"], final_mask, ctx.theme,
                                       field=_scaled_field(ctx), background=background)
        axes["mask"].set_xlabel(ctx.length_label); axes["mask"].set_ylabel(ctx.length_label)
        mark_role, mark_label = territory_panels.never_visited_mark(background is not None)
        common.semantic_legend(
            axes["mask"], ctx.theme,
            handles=[Patch(facecolor=ctx.theme.colour(mark_role))],
            labels=[mark_label],
            location="right",
        )
    if "shrinking" in axes:
        common.trace(axes["shrinking"], data["hours"], data["never_visited_share"],
                     ctx.theme, role="unvisited", hour_ticks=ctx.hour_ticks,
                     y_label="Share of field never visited")
    if "stages" in axes:
        rect = tuple(axes["stages"].get_position().bounds); axes["stages"].remove()
        count = max(2, int(ctx.option("stages")))
        indices = np.unique(np.linspace(0, len(cumulative) - 1, count).round().astype(int))
        # The covered ground, to match the mask panel above: the never-visited
        # field is what is left white, and it is read by watching the black grow
        # into it.
        stage_axes = common.image_strip(
            fig, rect, [cumulative[index].astype(float) for index in indices], ctx.theme,
            titles=[f"{data.loc[index, 'hours']:.1f} h" for index in indices],
            look=common.Look(cmap=matplotlib.colors.ListedColormap([
                ctx.theme.colour("page"), ctx.theme.colour("visited")
            ])), vmin=0, vmax=1,
        ).axes
        axes["stages"] = stage_axes[0] if stage_axes else fig.add_axes(rect)
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=f"Unclaimed foreground counts as visited; final never-visited share {final_mask.mean():.1%}.",
    )


if __name__ == "__main__":
    run_figure("negative-space")
