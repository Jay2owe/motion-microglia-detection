"""Figure 14: low-overlap footprint upheavals over time.

An upheaval is a transition in the lowest few per cent of observed frame-to-
frame overlap. The threshold is read off this recording rather than assumed, so
it is a description of what happened and not a rule about what should.

    python analysis/figures/14_upheaval_events.py <run>
    ... --quantile 0.02          a stricter definition of an upheaval
    ... --cells 12               how many event tiles to draw
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import tifffile

from _metrics import semantic_label
from _options import commas
from _schema import (FigureContext, FigureResult, Input, Option, Panel, Table,
                     figure, run_figure)

from panels import common
from panels import surveillance as surveillance_panels


@figure(
    number=14,
    slug="upheaval-events",
    summary="low-overlap footprint upheavals over time",
    title="Low-overlap footprint upheavals over time",
    reads=(Table("cell_frame.csv", module="surveillance"),
           Input("labels"), Input("raw")),
    panels=(
        Panel("raster", surveillance_panels.upheaval_raster,
              title="Low-overlap events through time"),
        Panel("counts", common.histogram, title="Events per cell"),
        Panel("tiles", common.image_strip, block=True,
              title="Cell footprint at each event"),
    ),
    options=(
        Option("metrics", default="jaccard", cast=str, metavar="COL",
               help="the overlap column an upheaval is defined on"),
        Option("quantile", default=0.05),
        Option("bins", default=20),
        Option("cells", default="6", cast=str),
    ),
    grammar="event raster and image grid",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv")
    metric = ctx.option("metrics")
    quantile = float(ctx.option("quantile"))
    threshold = float(data[metric].dropna().quantile(quantile))
    panels = ctx.panels()
    # Source image paths must be recorded before the bundle is made, even when
    # a requested panel later has no valid event tile.
    raw_path = ctx.input_path("raw") if "tiles" in panels else None
    label_path = ctx.input_path("labels") if "tiles" in panels else None
    fig, axes = ctx.layout(panels)
    events = surveillance_panels.upheaval_raster(
        axes["raster"] if "raster" in axes else fig.add_axes([0, 0, 0.001, 0.001]),
        data, ctx.theme, jaccard_column=metric, threshold=threshold,
        y_label="Cell rank", event_label=semantic_label(metric),
    ).data
    if "raster" not in panels:
        plt.delaxes(fig.axes[-1])
    if "counts" in axes:
        counts = events.groupby("identity").size()
        common.histogram(axes["counts"], counts, ctx.theme,
                         bins=int(ctx.option("bins")), role="highlight",
                         x_label="Upheavals per identity", y_label="Cells")
    if "tiles" in axes:
        rect = tuple(axes["tiles"].get_position().bounds)
        axes["tiles"].remove()
        if raw_path and label_path and not events.empty:
            raw = tifffile.imread(raw_path)
            labels = tifffile.imread(label_path)
            requested_cells = str(ctx.option("cells"))
            try:
                picked = events.nsmallest(min(int(requested_cells), len(events)), metric)
            except ValueError:
                identities = [int(value) for value in commas(requested_cells)]
                picked = (events[events["identity"].isin(identities)].sort_values(metric)
                          .groupby("identity", as_index=False).head(1))
            images, masks, titles = [], [], []
            for _, row in picked.iterrows():
                index = int(row["frame_index"])
                images.append(raw[index])
                masks.append(labels[index] == int(row["identity"]))
                titles.append(f"Cell {int(row['identity'])}, {row['hours']:.1f} h")
            tile_axes = common.image_strip(fig, rect, images, ctx.theme,
                                           masks=masks, titles=titles).axes
            axes["tiles"] = tile_axes[0] if tile_axes else None
        else:
            ax = fig.add_axes(rect)
            ax.text(0.5, 0.5, "No event tiles available", ha="center", va="center")
            ax.set_axis_off()
            axes["tiles"] = ax
    kept = [column for column in ("identity", "frame_index", "hours", "jaccard",
                                   "turnover_index", "step_px_gapless", "area_px",
                                   "event_threshold", "row_order") if column in events]
    return FigureResult(
        figure=fig,
        axes=[ax for ax in axes.values() if ax is not None],
        figure_data=events[kept],
        subtitle=f"Cells are ordered by first appearance; events are the lowest {quantile:.1%} "
                       f"of observed {metric}, threshold {threshold:.3g}.",
    )


if __name__ == "__main__":
    run_figure("upheaval-events")
