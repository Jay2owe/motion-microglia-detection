"""Figure 26: per-cell territory coverage and pixel revisits.

How much of the ground a cell eventually covers it has reached by a given time,
and how often it comes back to the same pixel. The denominator is each cell's
own final territory, not the field, so a small cell is not penalised.

    python analysis/figures/26_patch_ledger.py <run>
    ... --contour 0.25           the persistent-core contour
    ... --cells 24               how many territories are drawn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import tifffile

from _derive import _require_columns, _scaled_field, _selected_identities
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Input, Option, Panel, Stack, Table, figure, run_figure

from panels import common
from panels import territory as territory_panels


#: The per-cell columns ``territory`` contributes to ``cell_summary``, in the
#: order the module writes them. ``observed_frames`` is not in the list: the
#: summary computes its own and the module's copy is dropped by the fold, which
#: is the right behaviour as long as the two agree.
_TERRITORY_COLUMNS = ("union_px", "core_px", "fringe_px", "transient_px",
                      "core_share", "half_coverage_hours", "saturation_hours")


def _territory_columns(summary: pd.DataFrame) -> pd.DataFrame:
    _require_columns(summary, _TERRITORY_COLUMNS, "cell_summary.csv", "territory")
    keep = ["stem", "condition", "subject", "identity", "observed_frames",
            *_TERRITORY_COLUMNS]
    return summary[[c for c in keep if c in summary]].copy()


@figure(
    number=26,
    slug="patch-ledger",
    summary="per-cell territory coverage and pixel revisits",
    title="Per-cell territory coverage and pixel revisits",
    reads=(Table("cell_summary.csv", module="territory"),
           Table("cell_frame.csv", module="territory"),
           Stack("revisit_count.tif", module="territory"),
           Input("labels")),
    panels=(
        Panel("revisit", territory_panels.revisit_map,
              title="How often each pixel was occupied"),
        Panel("coverage", territory_panels.coverage_curve,
              title="Fraction of eventual territory reached"),
        Panel("core", common.histogram,
              title="Frequently occupied share of territory"),
    ),
    options=(
        Option("contour", default=0.5),
        Option("metrics", default="core_share", cast=str, metavar="COL",
               help="which per-territory column the histogram draws"),
        Option("bins", default=20),
        Option("cells", default="16", cast=str),
        Option("hour_ticks", default=24.0),
    ),
    grammar="revisit map coverage curves and core histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    # The per-frame coverage columns are in cell_frame; the per-cell totals
    # are still their own table until stage 04 folds them.
    frame = ctx.table("cell_frame.csv")
    # The per-cell totals are columns of cell_summary now. They are narrowed
    # back to this module's own seven so the copy that rides along beside the
    # figure holds the numbers the figure used, not every per-cell number in
    # the run.
    territory = _territory_columns(ctx.table("cell_summary.csv"))
    cells = _selected_identities(territory, ctx.option("cells"))
    figure_data = frame.merge(territory[["identity", "union_px"]], on="identity", how="left")
    # `territory_shape` writes this column now. It is still computed here when a
    # run predates that module, so an older run folder keeps rebuilding.
    if "coverage_share" not in figure_data.columns:
        figure_data["coverage_share"] = (
            figure_data["cumulative_unique_px"] / figure_data["union_px"])
    figure_data = figure_data[["identity", "frame_index", "hours", "cumulative_unique_px",
                               "union_px", "coverage_share", "new_px", "revisit_fraction"]]
    panels = ctx.panels()
    label_path = ctx.input_path("labels") if "revisit" in panels or "coverage" in panels else None
    labels = tifffile.imread(label_path) if label_path else None
    cell_frame = frame if "revisit" in panels else pd.DataFrame()
    fig, axes = ctx.layout(panels)
    if "revisit" in axes:
        rect = tuple(axes["revisit"].get_position().bounds)
        axes["revisit"].remove()
        shown = cells[:16]
        columns = min(8, max(1, len(shown)))
        rows = int(np.ceil(len(shown) / columns))
        gap = 0.012
        width = (rect[2] - gap * (columns - 1)) / columns
        height = (rect[3] - gap * (rows - 1)) / rows
        map_axes, map_handles = [], []
        contour_fraction = float(ctx.option("contour"))
        soma = cell_frame.groupby("identity")[["soma_x", "soma_y"]].median() if not cell_frame.empty else pd.DataFrame()
        aggregate = ctx.stack("revisit_count.tif")
        count_maps = {
            identity: (np.count_nonzero(labels == identity, axis=0) if labels is not None else aggregate)
            for identity in shown
        }
        shared_vmax = max(float(np.nanmax(values)) for values in count_maps.values()) if count_maps else 1.0
        for position, identity in enumerate(shown):
            row, column = divmod(position, columns)
            ax = fig.add_axes([rect[0] + column * (width + gap),
                               rect[1] + (rows - 1 - row) * (height + gap), width, height])
            counts = count_maps[identity]
            observed = int(territory.set_index("identity").loc[identity, "observed_frames"])
            centre = None
            if identity in soma.index:
                centre = (ctx.scale.length(float(soma.loc[identity, "soma_x"])),
                          ctx.scale.length(float(soma.loc[identity, "soma_y"])))
            map_handle = territory_panels.revisit_map(
                ax, counts, ctx.theme, field=_scaled_field(ctx),
                contour_at=contour_fraction * observed, soma=centre,
                x_label="", y_label="", vmin=0, vmax=shared_vmax,
            ).extra["handle"]
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.97, str(identity), transform=ax.transAxes, va="top",
                    fontsize=ctx.theme.size("caption"), color=ctx.theme.colour("ink"),
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.0})
            map_axes.append(ax); map_handles.append(map_handle)
        axes["revisit"] = map_axes[0] if map_axes else fig.add_axes(rect)
        if map_axes:
            fig.text(rect[0], rect[1] + rect[3] + 0.014, ctx.spec.panel("revisit").heading(),
                     fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom")
            common.inset_colour_bar(map_axes[-1], map_handles[-1], ctx.theme,
                                    label="Frames occupied")
            fig.legend(
                handles=[
                    Line2D([], [], color=ctx.theme.colour("ink"), label="Persistent-core contour"),
                    Line2D([], [], marker="+", color=ctx.theme.colour("highlight"),
                           linestyle="none", label="Median soma position"),
                ], loc="upper center", bbox_to_anchor=(0.55, rect[1] - 0.012),
                ncol=2, frameon=False, fontsize=ctx.theme.size("caption"),
            )
    if "coverage" in axes:
        selected = frame[frame["identity"].isin(cells)]
        pivot = selected.pivot(index="identity", columns="hours", values="cumulative_unique_px")
        union = territory.set_index("identity")["union_px"].reindex(pivot.index)
        shuffled = None
        if labels is not None and len(pivot.columns):
            generator = np.random.default_rng(20260825)
            order = generator.permutation(labels.shape[0])
            curves = []
            for identity in pivot.index:
                masks = labels == int(identity)
                curves.append(np.logical_or.accumulate(masks[order], axis=0).sum(axis=(1, 2)) /
                              max(int(masks.any(axis=0).sum()), 1))
            shuffled = np.asarray(curves, dtype=float)
        territory_panels.coverage_curve(axes["coverage"], pivot.columns,
                                         pivot.to_numpy(float), ctx.theme,
                                         denominator=union.to_numpy(float), shuffled=shuffled,
                                         hour_ticks=ctx.hour_ticks)
    if "core" in axes:
        metric = ctx.option("metrics")
        if metric not in territory:
            raise SystemExit(
                f"--metrics {metric} is not one of this module's per-cell "
                f"columns in cell_summary.csv: "
                + ", ".join(c for c in territory.columns if c != "identity"))
        common.histogram(axes["core"], territory[metric], ctx.theme,
                         bins=int(ctx.option("bins")), role="stable",
                         x_label=semantic_label(metric), y_label="Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Coverage denominator is each identity's own final union; {len(territory)} territories.",
        auxiliary={"territory_summary.csv": territory},
    )


if __name__ == "__main__":
    run_figure("patch-ledger")
