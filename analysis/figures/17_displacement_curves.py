"""Figure 17: mean squared displacement against lag.

One gap-respecting curve per identity on log-log axes, and the distribution of
the slopes. A slope near one is a random walk; below it the cell is confined,
above it the cell is going somewhere.

    python analysis/figures/17_displacement_curves.py <run>
    ... --max-lag 48             stop the curves at this many frames
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _derive import _require_columns
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import motility as motility_panels


@figure(
    number=17,
    slug="displacement-curves",
    summary="mean squared displacement against lag",
    title="Mean squared displacement against lag",
    reads=(Table("msd_curves.csv", module="motility"),
           Table("cell_summary.csv", module="motility")),
    panels=(
        Panel("curves", motility_panels.msd_curves,
              title="Movement across time lags"),
        Panel("alpha", common.histogram, title="Motion scaling across cells"),
    ),
    options=(
        Option("metrics", default="msd_alpha", cast=str, metavar="COL",
               help="which per-track column the second panel summarises"),
        Option("max_lag", default=None, cast=int, metavar="N",
               help="largest lag drawn, in frames"),
        Option("bins", default=24),
    ),
    grammar="log-log curves and histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    curves = ctx.table("msd_curves.csv")
    tracks = ctx.table("cell_summary.csv")
    _require_columns(tracks, ["msd_alpha"], "cell_summary.csv", "motility")
    data = curves.merge(tracks[["identity", "msd_alpha"]], on="identity", how="left")
    maximum = ctx.option("max_lag")
    if maximum is not None:
        data = data[data["lag_frames"] <= maximum]
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "curves" in axes:
        motility_panels.msd_curves(axes["curves"], data, ctx.theme,
                                   thin_below=max(3, int(data["pairs"].quantile(0.1))),
                                   y_label=f"MSD ({ctx.area_label})")
    if "alpha" in axes:
        summary_metric = ctx.option("metrics")
        if summary_metric not in tracks:
            raise SystemExit(f"--metrics {summary_metric} is not in cell_summary.csv")
        common.histogram(axes["alpha"], tracks[summary_metric], ctx.theme,
                         bins=int(ctx.option("bins")), role="motility",
                         x_label=semantic_label(summary_metric), y_label="Cells")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle=f"One gap-respecting displacement curve per identity; {data['identity'].nunique()} tracks.",
    )


if __name__ == "__main__":
    run_figure("displacement-curves")
