"""Figure 17: how each cell's centroid displacement grows with time gap.

One curve per identity and the distribution of the growth exponent fitted from
those same curves. The visible wording defines the measurement, its source and
the multiplicative axes rather than assuming the reader knows MSD or log plots.

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
    summary="how centroid displacement grows with time gap and the value fitted from each curve",
    title="How far each cell centre moves over increasing time gaps",
    reads=(Table("msd_curves.csv", module="motility"),
           Table("cell_summary.csv", module="motility")),
    panels=(
        Panel("curves", motility_panels.msd_curves,
              min_width_inches=12.1, min_height_inches=6.0,
              title="Average squared change in centroid position"),
        Panel("alpha", motility_panels.msd_exponent_distribution,
              min_width_inches=12.1, min_height_inches=5.0,
              title="Growth rate fitted from each cell's curve"),
    ),
    options=(
        Option("metrics", default="msd_alpha", cast=str, metavar="COL",
               help="which per-track column the second panel summarises"),
        Option("max_lag", default=None, cast=float, metavar="HOURS",
               help="largest time separation drawn, in hours"),
        Option("bins", default=18),
    ),
    grammar="multiplicative displacement curves with fitted-exponent histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    curves = ctx.table("msd_curves.csv")
    tracks = ctx.table("cell_summary.csv")
    _require_columns(tracks, ["msd_alpha"], "cell_summary.csv", "motility")
    data = curves.merge(tracks[["identity", "msd_alpha"]], on="identity", how="left")
    summary_metric = ctx.option("metrics")
    if summary_metric not in tracks:
        raise SystemExit(f"--metrics {summary_metric} is not in cell_summary.csv")
    maximum = ctx.option("max_lag")
    if maximum is not None:
        if float(maximum) <= 0:
            raise SystemExit("--max-lag must be greater than zero hours")
        data = data[data["lag_hours"] <= float(maximum)]
    panels = ctx.panels()
    fig, axes = ctx.layout(
        panels, top_inches=2.15, bottom_inches=3.25, gap_inches=2.3,
    )
    auxiliary = {}
    if "curves" in axes:
        curve_result = ctx.drew(
            "curves",
            motility_panels.msd_curves(
                axes["curves"], data, ctx.theme,
                y_label=f"Average squared centroid shift ({ctx.area_label})",
            ),
        )
        auxiliary["curve_summary.csv"] = curve_result.extra["summary"]
    if "alpha" in axes:
        if summary_metric == "msd_alpha":
            distribution = motility_panels.msd_exponent_distribution(
                axes["alpha"], tracks[summary_metric], ctx.theme,
                bins=int(ctx.option("bins")),
            )
        else:
            axes["alpha"].set_title(
                f"{semantic_label(summary_metric)} across cells", loc="left",
                fontsize=ctx.theme.size("panel"), fontweight="bold",
            )
            distribution = common.histogram(
                axes["alpha"], tracks[summary_metric], ctx.theme,
                bins=int(ctx.option("bins")), role="motility",
                x_label=semantic_label(summary_metric), y_label="Cells",
            )
        ctx.drew("alpha", distribution)
        auxiliary["cell_distribution.csv"] = distribution.data
        auxiliary["cell_values.csv"] = tracks[["identity", summary_metric]].copy()

    finite_alpha = tracks["msd_alpha"].dropna()
    median_alpha = float(finite_alpha.median())
    below_reference = int((finite_alpha < 1.0).sum())
    n_alpha = int(len(finite_alpha))
    if summary_metric == "msd_alpha":
        subtitle = (
            "Each orange line comes from the geometric centre of one cell's outline; "
            f"the lower value is fitted from that same line. Median α = {median_alpha:.2f}; "
            f"{below_reference} of {n_alpha} cells are below the random-walk reference."
        )
    else:
        subtitle = (
            "Each orange line comes from the geometric centre of one cell's outline; "
            f"the lower panel shows the selected per-cell measurement: "
            f"{semantic_label(summary_metric)}."
        )
    unit_note = (
        "Distances are reported in calibrated micrometres."
        if ctx.scale.calibrated else
        "Distances are reported in pixels because this run has no microscope calibration."
    )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=ctx.leading_table(),
        auxiliary=auxiliary,
        subtitle=subtitle,
        footnote="\n".join((
            "At each time gap, the squared distance between every eligible pair of observed centroid positions is averaged; missing frames are never joined or interpolated.",
            "α (alpha) is the curve's growth rate in: average squared shift = constant × time gap^α. Below 1 grows more slowly than a random walk; above 1 grows faster. It summarises a trajectory but does not prove a movement mechanism.",
            "The upper axes use multiplicative spacing so small and large shifts remain visible: successive ticks double time and multiply squared displacement tenfold; labels show the real values.",
            unit_note,
        )),
    )


if __name__ == "__main__":
    run_figure("displacement-curves")
