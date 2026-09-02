"""Figure 30: radial occupancy from soma over time.

How far out from the soma a cell holds ground, frame by frame. Annuli clipped
by the edge of the field are dropped rather than counted as empty, which is why
the outer rings thin out rather than falling to zero.

    python analysis/figures/30_sholl_kymograph.py <run>
    ... --rings 12               keep only the inner rings
    ... --cells 6                how many cells get a kymograph
    ... --scaling global         rings of one fixed width instead of per-cell

``sholl.csv`` holds the profile twice, under two ring widths. ``cell`` divides
each cell's own reach, so ring 3 is "half way out" for every cell and shapes are
comparable between a small cell and a large one; ``global`` uses one width for
the whole movie, so the y axis is an absolute distance and profiles can be
pooled. Drawing both at once would average two different x axes, so the figure
takes one and says which in the subtitle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
import numpy as np

from _derive import _require_columns, _selected_identities
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import morphology as morphology_panels


@figure(
    number=30,
    slug="sholl-kymograph",
    summary="radial occupancy from soma over time",
    title="Radial occupancy from soma over time",
    reads=(Table("sholl.csv", module="sholl"),
           Table("cell_frame.csv", module="sholl")),
    panels=(
        Panel("kymograph", morphology_panels.sholl_kymograph,
              title="Radial occupancy through time"),
        Panel("reach", common.trace, title="Outermost persistent reach"),
        Panel("profile", morphology_panels.sholl_profile,
              title="Radial profiles at selected times"),
    ),
    options=(
        Option("metrics", default="occupancy", cast=str, metavar="COL",
               help="which radial column the kymograph draws"),
        Option("rings", default=None, cast=int, metavar="N"),
        Option("cells", default="12", cast=str),
        Option("scaling", default="cell", cast=str, metavar="WHICH",
               help="ring width: cell (each cell's own reach) or global (one width)"),
        Option("hour_ticks", default=24.0),
    ),
    grammar="surfaces reach traces and radial profiles",
)
def build(ctx: FigureContext) -> FigureResult:
    sholl = ctx.table("sholl.csv")
    # reach_p95 and the soma centre are columns of cell_frame: they are one
    # row per cell per frame, the same rows cell_frame already carries.
    reach = ctx.table("cell_frame.csv")
    _require_columns(reach, ["reach_p95"], "cell_frame.csv", "sholl")
    metric = ctx.option("metrics")
    if metric not in {"occupancy", "intersections"}:
        raise SystemExit("--metrics must be occupancy or intersections")
    scaling = ctx.option("scaling")
    if "scaling" in sholl.columns:
        available = sorted(sholl["scaling"].dropna().unique())
        if scaling not in available:
            raise SystemExit(
                f"--scaling must be one of {', '.join(available)}; this run has no "
                f"{scaling!r} rings")
        sholl = sholl[sholl["scaling"] == scaling].copy()
    elif scaling != "cell":
        # A run made before the profile was measured twice has one unlabelled
        # scaling. Refusing is better than silently drawing it as if it were the
        # one that was asked for.
        raise SystemExit(
            "this run's sholl.csv has no 'scaling' column, so it holds one "
            "unlabelled ring width; re-run the analysis or drop --scaling")
    cells = _selected_identities(sholl, ctx.option("cells"))
    data = sholl[sholl["identity"].isin(cells)].copy()
    requested_rings = ctx.option("rings")
    if requested_rings is not None and requested_rings > 0:
        maximum = int(data["ring"].max()) + 1
        data["display_ring"] = np.minimum((data["ring"] * requested_rings // maximum).astype(int), requested_rings - 1)
        grouped = data.groupby(["identity", "frame_index", "hours", "display_ring"], as_index=False).agg(
            radius_inner=("radius_inner", "min"), radius_outer=("radius_outer", "max"),
            annulus_px=("annulus_px", "sum"), occupied_px=("occupied_px", "sum"),
            intersections=("intersections", "sum"),
        )
        grouped["occupancy"] = grouped["occupied_px"] / grouped["annulus_px"].replace(0, np.nan)
        data = grouped.rename(columns={"display_ring": "ring"})
    # Drop rings the field edge or a neighbour ate, keeping rings that are
    # merely small. What counts as "eaten" depends on the scaling, because the
    # two disagree about whether one ring index is one area.
    #
    # Under `global` every cell shares a ring width, so ring 3 has the same
    # area for everyone and a cell whose ring 3 is half the size of everyone
    # else's lost the difference to a clip. Compare across cells.
    #
    # Under `cell` the width is the cell's own reach over n_rings, so ring 3 is
    # a different area for every cell *by construction* - that is the point of
    # the scaling. The cross-cell test would then discard every small cell at
    # every ring, which is how this figure used to fail with a one-row heatmap.
    # A cell's own ring is constant through time unless something clips it, so
    # compare each frame against that cell's own best frame at the same ring.
    if scaling == "cell":
        reference = data.groupby(["identity", "ring"])["annulus_px"].transform(
            lambda values: values.quantile(0.9))
        data = data[data["annulus_px"] >= 0.5 * reference].copy()
    else:
        keep = data.groupby(["identity", "ring"])["annulus_px"].median()
        ceiling = keep.groupby(level=1).transform(lambda values: values.quantile(0.9))
        valid = set(keep[keep >= 0.5 * ceiling].index)
        data = data[data.set_index(["identity", "ring"]).index.isin(valid)].copy()
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "kymograph" in axes:
        rect = tuple(axes["kymograph"].get_position().bounds); axes["kymograph"].remove()
        shown = cells[:12]; columns = min(6, len(shown)); rows = int(np.ceil(len(shown) / columns))
        gap = 0.014; width = (rect[2] - gap * (columns - 1)) / columns; height = (rect[3] - gap * (rows - 1)) / rows
        made, heat_handles = [], []
        colour_min = 0.0
        colour_max = 1.0 if metric == "occupancy" else max(float(data[metric].max()), 1.0)
        for position, identity in enumerate(shown):
            group = data[data["identity"] == identity]
            pivot = group.pivot(index="ring", columns="hours", values=metric)
            if pivot.empty:
                continue
            radii = group.groupby("ring")[["radius_inner", "radius_outer"]].mean().mean(axis=1).reindex(pivot.index)
            reach_cell = reach[reach["identity"] == identity].set_index("hours")["reach_p95"].reindex(pivot.columns)
            row, column = divmod(position, columns)
            ax = fig.add_axes([rect[0] + column * (width + gap), rect[1] + (rows - 1 - row) * (height + gap), width, height])
            heat_handle = morphology_panels.sholl_kymograph(
                ax, pivot.to_numpy(float), pivot.columns, radii, ctx.theme,
                reach=reach_cell.to_numpy(float), hour_ticks=ctx.hour_ticks,
                y_label=f"Distance from soma ({ctx.length_label})",
                vmin=colour_min, vmax=colour_max,
            ).extra["handle"]
            ax.set_xlabel(""); ax.set_ylabel("")
            if row != rows - 1:
                ax.set_xticks([])
            if column != 0:
                ax.set_yticks([])
            ax.text(0.03, 0.97, str(identity), transform=ax.transAxes, va="top",
                    fontsize=ctx.theme.size("caption"), color=ctx.theme.colour("ink"),
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.0})
            made.append(ax); heat_handles.append(heat_handle)
        axes["kymograph"] = made[0] if made else fig.add_axes(rect)
        if made:
            fig.text(rect[0], rect[1] + rect[3] + 0.014, ctx.spec.panel("kymograph").heading(),
                     fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom")
            common.inset_colour_bar(made[-1], heat_handles[-1], ctx.theme,
                                    label=semantic_label(metric))
            fig.legend(
                handles=[Line2D([], [], color=ctx.theme.colour("ink"),
                                label="95% reach")],
                loc="upper center", bbox_to_anchor=(0.60, rect[1] - 0.045),
                ncol=1, frameon=False, fontsize=ctx.theme.size("caption"),
            )
            made[0]._semantic_legend_handled = True
    if "profile" in axes:
        identity = cells[0]
        group = data[data["identity"] == identity]
        pivot = group.pivot(index="ring", columns="hours", values=metric)
        radii = group.groupby("ring")[["radius_inner", "radius_outer"]].mean().mean(axis=1).reindex(pivot.index)
        positions = np.linspace(0, len(pivot.columns) - 1, min(4, len(pivot.columns))).round().astype(int)
        profile_roles = ("morphology", "reporter", "surveillance", "motility")
        for profile_index, position in enumerate(positions):
            morphology_panels.sholl_profile(
                axes["profile"], radii, pivot.iloc[:, position], ctx.theme,
                look=common.Look(colour=ctx.theme.colour(profile_roles[profile_index % len(profile_roles)])),
                x_label=f"Distance from soma ({ctx.length_label})",
                y_label=semantic_label(metric),
                label=f"{float(pivot.columns[position]):g} h",
            )
        axes["profile"].set_title(f"Cell {identity}: radial profiles at selected times", loc="left")
        axes["profile"].set_ylabel("Radial-ring occupancy\n(fraction)")
    if "reach" in axes:
        for cell_index, identity in enumerate(cells):
            group = reach[reach["identity"] == identity].sort_values("hours")
            common.trace(axes["reach"], group["hours"], group["reach_p95"], ctx.theme,
                         role="morphology", linewidth=ctx.theme.stroke("guide"),
                         hour_ticks=ctx.hour_ticks,
                         label="One cell" if cell_index == 0 else "",
                         y_label=f"95% reach ({ctx.length_label})")
    required = [column for column in
                ["identity", "frame_index", "hours", "scaling", "ring",
                 "radius_inner", "radius_outer", "occupancy", "intersections",
                 "annulus_px"]
                if column in data.columns]
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data[required],
        subtitle=(f"{len(cells)} identities; "
                  + ("rings divide each cell's own 95% reach, so a radius is that cell's"
                     if scaling == "cell" else
                     "rings are one fixed width across the movie, so a radius is an absolute distance")
                  + ("; annuli below half that cell's own best support at the same ring are omitted."
                     if scaling == "cell" else
                     "; field-clipped annuli below half the ring-specific reference support are omitted.")),
        footnote=ctx.provenance_footnote(),
        auxiliary={"reach.csv": reach[reach["identity"].isin(cells)],
                                  **ctx.provenance_auxiliary()},
    )


if __name__ == "__main__":
    run_figure("sholl-kymograph")
