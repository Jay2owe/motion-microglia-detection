"""Figure 27: field coverage by occupancy source.

Each pixel is classified by every source that occupied it: named cells,
unclaimed foreground, both at different times, or neither. The map is either a
single final view or a staged history, never both copies of the final state.

    python analysis/figures/27_negative_space.py <run>
    ... --coverage-view final   one final coverage map
    ... --coverage-view stages  selected cumulative maps (default)
    ... --stages 8              how many staged maps
    ... --overlay               draw coverage hues over raw images
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tifffile

from _derive import _scaled_field
from _schema import FigureContext, FigureResult, Input, Option, Panel, figure, run_figure

from panels import territory as territory_panels


@figure(
    number=27,
    slug="negative-space",
    summary="field coverage by occupancy source",
    title="Field coverage by occupancy source",
    reads=(Input("labels"), Input("unclaimed"), Input("raw")),
    panels=(
        Panel("coverage", territory_panels.coverage_history, block=True,
              title="How each pixel was occupied",
              min_height_inches=3.0,
              item_width_inches=3.0, item_height_inches=3.0),
        Panel("composition", territory_panels.coverage_composition,
              title="Field coverage through time"),
    ),
    options=(
        Option("coverage_view", default="stages"),
        Option("stages", default=6),
        Option("hour_ticks", default=24.0),
    ),
    switches=("overlay",),
    grammar="categorical coverage maps and stacked coverage composition",
)
def build(ctx: FigureContext) -> FigureResult:
    label_path = ctx.input_path("labels")
    if label_path is None:
        raise SystemExit("negative-space needs the labels input recorded in the run manifest")
    labels = tifffile.imread(label_path)
    occupancy = {"named": labels > 0}
    unclaimed_path = ctx.input_path("unclaimed")
    if unclaimed_path is not None:
        occupancy["unclaimed"] = tifffile.imread(unclaimed_path) > 0
    hours = np.arange(labels.shape[0], dtype=float) * ctx.interval / 60.0
    source_labels = {"named": "Named cell", "unclaimed": "Unclaimed foreground"}
    source_labels = {name: source_labels[name] for name in occupancy}
    class_colours = {
        (): "missing",
        ("named",): "morphology",
        ("unclaimed",): "unclaimed",
        ("named", "unclaimed"): "surveillance",
    }
    class_colours = {
        key: colour for key, colour in class_colours.items()
        if all(name in occupancy for name in key)
    }
    panels = ctx.panels()
    view = str(ctx.option("coverage_view")).strip().lower()
    if view not in {"final", "stages"}:
        raise SystemExit("--coverage-view must be final or stages")
    stage_count = max(2, int(ctx.option("stages"))) if view == "stages" else 1
    coverage_panel = ctx.spec.panel("coverage")
    minimum_sizes = ({"coverage": coverage_panel.grid_minimum(
        stage_count, stage_count)} if "coverage" in panels else None)
    fig, axes = ctx.layout(panels, minimum_sizes=minimum_sizes)
    backgrounds = None
    if "coverage" in axes:
        if ctx.switch("overlay"):
            raw_path = ctx.input_path("raw")
            if raw_path is not None:
                raw = tifffile.imread(raw_path)
                if raw.ndim != 3 or raw.shape[1:] != labels.shape[1:] or len(raw) < len(labels):
                    raise SystemExit(
                        f"negative-space raw stack is {raw.shape}; expected at least "
                        f"{len(labels)} frames with shape {labels.shape[1:]}"
                    )
                backgrounds = raw[-len(labels):]
        coverage_result = ctx.block(
            fig, axes.pop("coverage"), "coverage", occupancy, ctx.theme,
            hours=hours, view=view, stages=stage_count,
            field=_scaled_field(ctx), backgrounds=backgrounds,
            source_labels=source_labels, class_colours=class_colours,
            x_label=f"X position ({ctx.length_label})",
            y_label=f"Y position ({ctx.length_label})",
            gap=(coverage_panel.item_gap_inches * ctx.theme.canvas_scale /
                 fig.get_figwidth()),
        )
    else:
        coverage_result = None
    if "composition" in axes:
        composition_result = ctx.drew(
            "composition",
            territory_panels.coverage_composition(
                axes["composition"], hours, occupancy, ctx.theme,
                source_labels=source_labels, class_colours=class_colours,
                hour_ticks=ctx.hour_ticks,
            ),
        )
    else:
        composition_result = None
    figure_data = (composition_result.data if composition_result is not None
                   else coverage_result.data)
    final_coverage = float(
        np.logical_or.reduce([values for values in occupancy.values()]).any(axis=0).mean()
    )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=(f"Pixel hue records cumulative occupancy source; final field coverage "
                  f"{final_coverage:.1%}. Display: {view}."),
        footnote=ctx.footnote(
            "Named cell means foreground assigned an identity; unclaimed foreground means "
            "detected foreground assigned to none. A mixed pixel was occupied by both at "
            "different frames."
        ),
    )


if __name__ == "__main__":
    run_figure("negative-space")
