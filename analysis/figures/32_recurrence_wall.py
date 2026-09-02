"""Figure 32: frame-by-frame recurrence of each cell's state.

Every frame compared with every other frame of the same cell. A diagonal streak
is a state the cell returned to and held; the recurrence threshold is each
cell's own distance quantile, so a quiet cell is not judged against a busy one.

The numbers are read from ``recurrence.csv`` and ``recurrence_quantification.csv``
rather than worked out here. One thing is still computed on this page and only
one: the wall itself, which is a 99 x 99 distance matrix per cell and exists to
be looked at. Writing that to a table would be writing a picture into a CSV.

Everything about *how* recurrence is measured - the state vector, the threshold,
the surrogate count - is a setting of the ``recurrence`` module and is read back
out of the run, so the wall is built with exactly what the table was measured
with. It cannot draw a wall made one way beside a determinism ranking made
another.

    python analysis/figures/32_recurrence_wall.py <run>
    ... --cells 9                how many walls are drawn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
import numpy as np

from _derive import _selected_identities
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from analysis.modules.recurrence import state_matrix

from panels import common
from panels import coupling as coupling_panels


@figure(
    number=32,
    slug="recurrence-wall",
    summary="frame-by-frame recurrence of each cell's state",
    title="Frame-by-frame recurrence of each cell's state",
    reads=(Table("cell_frame.csv", module="coupling"),
           Table("recurrence.csv", module="recurrence"),
           Table("recurrence_quantification.csv", module="recurrence")),
    panels=(
        Panel("wall", coupling_panels.recurrence,
              title="State distance between every frame pair"),
        Panel("rate", common.trace, title="Similar-state pairs by time lag"),
        Panel("quantified", common.lollipop,
              title="Repeated-state sequence structure"),
    ),
    options=(
        # Both are drawing choices. The method settings that used to live here -
        # metrics, threshold, max_lag, shuffles - belong to the module now, and
        # are read back from the run rather than repeated.
        Option("cells", default="16", cast=str),
        Option("hour_ticks", default=24.0),
    ),
    grammar="recurrence wall lag rates and determinism ranking",
)
def build(ctx: FigureContext) -> FigureResult:
    rates = ctx.table("recurrence.csv")
    quantified = ctx.table("recurrence_quantification.csv")
    frame = ctx.table("cell_frame.csv")
    settings = ctx.module_params("recurrence")
    metrics = [metric for metric in settings["metrics"] if metric in frame]
    if not metrics:
        raise SystemExit("none of the recurrence state vector is in cell_frame.csv")
    quantile = float(settings["threshold_quantile"])
    complete = frame.dropna(subset=metrics)
    measured = set(quantified["identity"].astype(int))
    cells = [identity for identity in _selected_identities(complete, ctx.option("cells"))
             if identity in measured]

    # The threshold comes out of the table rather than being worked out again,
    # so the line drawn on the wall is the line the determinism beside it was
    # counted against.
    thresholds = quantified.set_index("identity")["recurrence_threshold_distance"]
    walls = {}
    for identity in cells:
        group = complete[complete["identity"] == identity][
            ["frame_index", "hours", *metrics]].sort_values("frame_index")
        hours = group["hours"].to_numpy(float)
        values = state_matrix(hours, group[metrics].to_numpy(float))
        walls[int(identity)] = (cdist(values, values), hours, float(thresholds[identity]))

    # The drawn cells only, and this figure's own columns: the run stamps every
    # table with stem, condition and subject, which belong in a run folder and
    # not in a figure's audit table beside the numbers it plotted.
    drawn = rates["identity"].isin(cells)
    figure_data = rates.loc[drawn, ["identity", "lag_frames", "lag_hours", "recurrence_rate",
                                    "recurrence_surrogate_mean", "recurrence_surrogate_lo",
                                    "recurrence_surrogate_hi", "recurrence_pairs"]].copy()
    quantification = quantified[quantified["identity"].isin(cells)].drop(
        columns=[c for c in ("stem", "condition", "subject") if c in quantified.columns])
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "wall" in axes:
        rect = tuple(axes["wall"].get_position().bounds); axes["wall"].remove()
        shown = list(walls)[:16]; columns = min(4, len(shown)); rows = int(np.ceil(len(shown) / columns))
        gap = 0.012; width = (rect[2] - gap * (columns - 1)) / columns; height = (rect[3] - gap * (rows - 1)) / rows
        made, wall_handles = [], []
        wall_max = max(float(np.nanmax(values[0])) for values in walls.values()) if walls else 1.0
        for position, identity in enumerate(shown):
            row, column = divmod(position, columns)
            ax = fig.add_axes([rect[0] + column * (width + gap), rect[1] + (rows - 1 - row) * (height + gap), width, height])
            distances, hours, cutoff = walls[identity]
            wall_handle = coupling_panels.recurrence(
                ax, distances, hours, ctx.theme,
                threshold=cutoff, hour_ticks=ctx.hour_ticks,
            ).extra["handle"]
            wall_handle.set_clim(0, wall_max)
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.97, str(identity), transform=ax.transAxes, va="top",
                    fontsize=ctx.theme.size("caption"), color="white",
                    bbox={"facecolor": ctx.theme.colour("ink"), "alpha": 0.62,
                          "edgecolor": "none", "pad": 1.0})
            made.append(ax); wall_handles.append(wall_handle)
        axes["wall"] = made[0] if made else fig.add_axes(rect)
        if made:
            fig.text(rect[0], rect[1] + rect[3] + 0.014, ctx.spec.panel("wall").heading(),
                     fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom")
            common.inset_colour_bar(
                made[-1], wall_handles[-1], ctx.theme,
                label="Distance between multimetric cell states",
            )
            fig.legend(
                handles=[Line2D([], [], color=ctx.theme.colour("ink"),
                                label=f"Closest {quantile:.0%} of frame pairs: recurrent")],
                loc="upper center", bbox_to_anchor=(0.60, rect[1] - 0.012),
                ncol=1, frameon=False, fontsize=ctx.theme.size("caption"),
            )
    if "rate" in axes and not figure_data.empty:
        for cell_index, (_, group) in enumerate(figure_data.groupby("identity")):
            axes["rate"].plot(group["lag_hours"], group["recurrence_rate"],
                              color=ctx.theme.colour("morphology"), alpha=0.18,
                              linewidth=ctx.theme.stroke("guide"),
                              label="Individual cells" if cell_index == 0 else None)
        aggregate = figure_data.groupby("lag_hours", as_index=False).agg(
            recurrence_rate=("recurrence_rate", "median"),
            surrogate_mean=("recurrence_surrogate_mean", "mean"),
            surrogate_lo=("recurrence_surrogate_lo", "mean"),
            surrogate_hi=("recurrence_surrogate_hi", "mean"))
        axes["rate"].fill_between(aggregate["lag_hours"], aggregate["surrogate_lo"], aggregate["surrogate_hi"],
                                  color=ctx.theme.colour("reference"), alpha=0.25, linewidth=0,
                                  label="Matched-noise interval")
        common.trace(axes["rate"], aggregate["lag_hours"], aggregate["recurrence_rate"], ctx.theme,
                     role="morphology", label="Median observed cell",
                     y_label="Similar frame pairs\n(fraction)", x_label="Lag (hours)")
    if "quantified" in axes and not quantification.empty:
        shown = quantification.sort_values("determinism", ascending=True)
        common.lollipop(axes["quantified"], shown["determinism"], ctx.theme,
                        labels=[""] * len(shown), role="morphology")
        axes["quantified"].set_xlabel("Recurrent points in diagonal runs of at least two frames")
        axes["quantified"].set_ylabel("Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"{len(walls)} detrended cells; recurrence threshold is each cell's distance quantile {quantile:g}.",
        auxiliary={"recurrence_quantification.csv": quantification},
    )


if __name__ == "__main__":
    run_figure("recurrence-wall")
