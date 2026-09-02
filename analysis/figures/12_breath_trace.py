"""Figure 12: extension and retraction as a mirrored footprint ledger.

Gained area above the line, lost area below, held area behind both. A cell that
is breathing in place makes a symmetric figure; one that is growing does not.

    python analysis/figures/12_breath_trace.py <run>
    ... --cells 20               how many identities the second panel traces
    ... --cells 3,7,12           or which ones
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _options import commas
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import surveillance as surveillance_panels


@figure(
    number=12,
    slug="breath-trace",
    summary="extension and retraction as a mirrored footprint ledger",
    title="Extension and retraction as a mirrored footprint ledger",
    reads=(Table("cell_frame.csv", module="surveillance"),),
    panels=(
        Panel("breath", surveillance_panels.mirrored_ledger,
              title="Gained, lost, retained and net area"),
        Panel("small_multiples", title="Net area change by cell"),
    ),
    options=(
        Option("cells", default="12", cast=str),
        Option("hour_ticks", default=24.0),
    ),
    grammar="mirrored timeline small multiples",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("cell_frame.csv")
    columns = ["identity", "frame_index", "hours", "gained_px", "lost_px", "held_px", "area_change_px"]
    data = data.dropna(subset=columns[3:]).copy()
    for source, target in (("gained_px", "gained_area"), ("lost_px", "lost_area"),
                           ("held_px", "held_area"), ("area_change_px", "net_area")):
        data[target] = data[source].map(ctx.scale.area)
    data["net_px"] = data["net_area"]
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    median = data.groupby("hours", as_index=False)[["gained_px", "lost_px", "held_px", "area_change_px"]].median()
    if "breath" in axes:
        scaled_median = data.groupby("hours", as_index=False)[["gained_area", "lost_area", "held_area", "net_area"]].median()
        surveillance_panels.mirrored_ledger(
            axes["breath"], scaled_median["hours"], scaled_median, ctx.theme,
            gained_column="gained_area", lost_column="lost_area", held_column="held_area",
            net_column="net_area", hour_ticks=ctx.hour_ticks,
            y_label=f"Gained / lost area ({ctx.area_label})",
        )
    if "small_multiples" in axes:
        ax = axes["small_multiples"]
        counts = data.groupby("identity").size().sort_values(ascending=False)
        cells_option = ctx.option("cells")
        try:
            cells = counts.index[: int(cells_option)].tolist()
        except ValueError:
            cells = [int(value) for value in commas(cells_option)]
        for identity in cells:
            group = data[data["identity"] == identity].sort_values("hours")
            ax.plot(group["hours"], group["gained_area"] - group["lost_area"],
                    color=ctx.theme.colour("surveillance"), alpha=0.28,
                    linewidth=ctx.theme.stroke("guide"))
        ax.axhline(0, color=ctx.theme.colour("reference"), linewidth=ctx.theme.stroke("guide"))
        ax.set_xlabel("Hours from start of recording")
        ax.set_ylabel(f"Gained − lost ({ctx.area_label})")
        ax.set_xticks(ctx.theme.hour_ticks(float(data["hours"].min()), float(data["hours"].max()), ctx.hour_ticks))
    figure_data = data[["identity", "hours"]].copy()
    figure_data["gained_px"] = data["gained_area"]
    figure_data["lost_px"] = data["lost_area"]
    figure_data["held_px"] = data["held_area"]
    figure_data["net_px"] = data["net_area"]
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Median trace plus the longest-observed identities from {ctx.summary['stem']}.",
        footnote=ctx.provenance_footnote(),
        auxiliary=ctx.provenance_auxiliary(),
    )


if __name__ == "__main__":
    run_figure("breath-trace")
