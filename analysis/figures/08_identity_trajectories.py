"""Figure 8: where every cell went over the whole recording.

One panel, ``panels.motility.trajectory_map``, which cuts each path at the gaps
rather than bridging them::

    python analysis/figures/08_identity_trajectories.py <run>
    ... --metrics total_path_px      colour the paths by a different track column
    ... --trace-luts magma           the map that colour runs along

The colouring column is any numeric column of ``cell_summary.csv``; its
wording comes from ``_metrics``, so it is named the same here as anywhere else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _metrics import describe
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import colour_bar, resolve_look
from panels import motility as motility_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=8,
    slug="identity-trajectories",
    summary="where every cell went over the whole recording",
    title="Centroid path of every cell over {span} h",
    grammar="map of every identity's centroid path, coloured by one track column",
    reads=(Table("cell_frame.csv", module="motility"),
           Table("cell_summary.csv", module="motility")),
    panels=(Panel("map", motility_panels.trajectory_map,
                  title="Every centroid path"),),
    options=(
        Option("metrics", default="net_displacement_px", cast=str, metavar="COL",
               help="which measured column colours each path"),
        Option("trace_luts", default=None, cast=str, metavar="NAME",
               help="the colour map each path's colour runs along"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    colour_by = ctx.option("metrics")
    path_lut = ctx.option("trace_luts")

    ctx.panels()
    cell_frame = ctx.table("cell_frame.csv")
    tracks = ctx.table("cell_summary.csv").set_index("identity")

    if colour_by not in tracks.columns:
        raise SystemExit(
            f"--metrics {colour_by} is not in cell_summary.csv. Columns available: "
            + ", ".join(c for c in tracks.columns if tracks[c].dtype.kind in "fi")
        )

    path = cell_frame[
        ["identity", "frame_index", "hours", "centroid_x", "centroid_y"]
    ].dropna(subset=["centroid_x", "centroid_y"]).sort_values(
        ["identity", "frame_index"]).copy()
    path[colour_by] = path["identity"].map(tracks[colour_by])
    if "total_path_px" in tracks.columns:
        path["total_path_px"] = path["identity"].map(tracks["total_path_px"])

    values = tracks[colour_by].dropna()
    cells = int(path["identity"].nunique())
    label = describe(colour_by)

    figure_ = ctx.sheet(14.6, 12.8)
    ax = figure_.add_axes([0.095, 0.190, 0.700, 0.640])

    groups = [
        (group["frame_index"].to_numpy(int),
         group[["centroid_x", "centroid_y"]].to_numpy(float),
         float(tracks[colour_by].get(identity, float("nan"))))
        for identity, group in path.groupby("identity")
    ]
    drawn = ctx.drew("map", motility_panels.trajectory_map(
        ax, groups, ctx.theme,
        field=ctx.field,
        look=resolve_look(ctx.theme, path_lut) if path_lut else None,
        vmax=float(values.max()),
    ))
    collection = drawn.extra["collection"]
    breaks = drawn.extra["breaks"]
    colour_bar(
        figure_, collection, [0.832, 0.255, 0.018, 0.470], ctx.theme,
        label=(f"{label.label} over the movie"
               + (f" ({label.unit})" if label.unit else "")),
    )

    return FigureResult(
        figure=figure_,
        axes=[ax],
        # The map is bounded by the field, so it keeps all four edges: a field
        # with two sides missing reads as a plot that happens to stop there.
        keep_spines=("top", "right"),
        figure_data=path,
        heading="Identity trajectories",
        title_fields={"span": f"{ctx.summary['hours_covered']:.0f}"},
        subtitle=(
            f"{ctx.summary['stem']}, {cells} identities across a "
            f"{ctx.field['width']} x {ctx.field['height']} px field. One line per "
            f"cell, from its centroid in\nevery frame it holds; the dot marks first "
            f"appearance. Colour is {label.label.lower()}: median "
            f"{values.median():.1f} px, largest {values.max():.0f} px."
        ),
        footnote=ctx.footnote(
            f"The path is cut at each of the {breaks} gaps rather than bridged across "
            f"one,\nso a long straight stroke is a movement the movie showed."
        ),
        readme=f"""## What the figure shows

Every identity's centroid path across the whole recording, drawn in the movie's
own pixel coordinates with y counting downwards, so the map is oriented the way
the frames are. Stationary cells appear as tight knots; the mobile minority draw
long strokes. Colour is `{colour_by}`, so a cell that wandered and came home
stays dark when that column is net displacement.

The axes are the field, not the data. A map bounded by wherever the cells
happen to be changes magnification between runs and invites two maps to be
compared at different scales without saying so.

## What it is a read-out of

Tracking, not a separate analysis: the centroids come from `cell_frame.csv` and
the colour from `cell_summary.csv`. Paths are cut at gaps rather than bridged
across them, because a bridged gap draws a straight stroke that reads as
evidence of travel the movie never showed. The cutting is done by
`panels.motility.path_segments`, which the step histogram's own rule matches.""",
        console=(f"cells: {cells}  breaks: {breaks}  colour by {colour_by}  "
                 f"median: {values.median():.2f}"),
    )


if __name__ == "__main__":
    run_figure("identity-trajectories", DEFAULT_RUN)
