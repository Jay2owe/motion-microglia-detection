"""Figure 30: radial occupancy from soma over time.

Each panel follows one cell. Colour shows how much of each radial ring around
its soma belongs to the cell at each time. Annuli clipped by the field edge or
another cell are omitted rather than counted as empty.

    python analysis/figures/30_sholl_kymograph.py <run>
    ... --rings 4                keep only the four inner rings
    ... --cells 7,12,40          choose cells explicitly
    ... --scaling global         rings of one fixed width instead of per-cell

``sholl.csv`` holds the profile twice, under two ring widths. ``cell`` divides
each cell's current 95% reach, so the vertical axis is 0--100% and shapes are
comparable between a small cell and a large one. ``global`` uses one fixed
width for the whole movie, so the vertical axis is an absolute distance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _derive import _selected_identities
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import morphology as morphology_panels


@figure(
    number=30,
    slug="sholl-kymograph",
    summary="radial occupancy from soma over time",
    title="Radial occupancy from soma over time",
    reads=(Table("sholl.csv", module="sholl"),),
    panels=(
        Panel("kymograph", morphology_panels.sholl_kymograph,
              item_width_inches=4.0, item_height_inches=3.0,
              title="Each panel is one cell; colour is the fraction of each radial ring occupied"),
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
    grammar="time-by-radius kymograph small multiples",
)
def build(ctx: FigureContext) -> FigureResult:
    sholl = ctx.table("sholl.csv")
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
    full_ring_count = int(data["ring"].max()) + 1
    requested_rings = ctx.option("rings")
    if requested_rings is not None:
        if requested_rings < 1:
            raise SystemExit("--rings needs a positive number of inner rings")
        data = data[data["ring"] < min(int(requested_rings), full_ring_count)].copy()
    # Compare the available pixels with the full geometric annulus. This stays
    # valid when cell-scaled radii change between frames; comparing against a
    # cell's largest annulus would wrongly label a genuinely smaller frame as
    # clipped.
    unit_per_pixel = (float(ctx.scale.microns_per_pixel)
                      if ctx.scale.calibrated else 1.0)
    radius_inner_px = data["radius_inner"].to_numpy(float) / unit_per_pixel
    radius_outer_px = data["radius_outer"].to_numpy(float) / unit_per_pixel
    expected_annulus_px = np.pi * (
        radius_outer_px ** 2 - radius_inner_px ** 2
    )
    data["annulus_support_fraction"] = np.minimum(
        data["annulus_px"].to_numpy(float) / np.maximum(expected_annulus_px, 1e-12),
        1.0,
    )
    data = data[data["annulus_support_fraction"] >= 0.5].copy()
    panels = ctx.panels()
    shown = cells
    columns = min(4, max(1, len(shown)))
    rows = max(1, int(np.ceil(len(shown) / columns)))
    kymograph_panel = ctx.spec.panel("kymograph")
    minimum_sizes = ({"kymograph": kymograph_panel.grid_minimum(
        len(shown), columns)} if "kymograph" in panels else None)
    fig, axes = ctx.layout(
        panels,
        minimum_sizes=minimum_sizes,
        bottom_inches=3.2,
    )
    plot_axes = []
    if "kymograph" in axes:
        rect = tuple(axes["kymograph"].get_position().bounds); axes["kymograph"].remove()
        scale = ctx.theme.canvas_scale
        gap_x = kymograph_panel.item_gap_inches * scale / fig.get_figwidth()
        gap_y = kymograph_panel.item_gap_inches * scale / fig.get_figheight()
        width = (rect[2] - gap_x * (columns - 1)) / columns
        height = (rect[3] - gap_y * (rows - 1)) / rows
        made, heat_handles = [], []
        colour_min = 0.0
        colour_max = 1.0 if metric == "occupancy" else max(float(data[metric].max()), 1.0)
        for position, identity in enumerate(shown):
            group = data[data["identity"] == identity]
            pivot = group.pivot(index="ring", columns="hours", values=metric)
            if pivot.empty:
                continue
            if scaling == "cell":
                radii = (pivot.index.to_numpy(float) + 0.5) / full_ring_count * 100.0
                y_label = "Distance from soma (% of current 95% reach)"
            else:
                radii = group.groupby("ring")[["radius_inner", "radius_outer"]].mean().mean(
                    axis=1).reindex(pivot.index).to_numpy(float)
                y_label = f"Distance from soma ({ctx.length_label})"
            row, column = divmod(position, columns)
            ax = fig.add_axes([
                rect[0] + column * (width + gap_x),
                rect[1] + (rows - 1 - row) * (height + gap_y), width, height])
            heat_handle = morphology_panels.sholl_kymograph(
                ax, pivot.to_numpy(float), pivot.columns, radii, ctx.theme,
                hour_ticks=ctx.hour_ticks, y_label=y_label,
                contour_at=None, vmin=colour_min, vmax=colour_max,
            ).extra["handle"]
            ax.set_xlabel(""); ax.set_ylabel("")
            if scaling == "cell":
                maximum_percent = (
                    100.0 * (float(pivot.index.max()) + 1.0) / full_ring_count
                )
                ax.set_ylim(0, maximum_percent)
                ax.set_yticks([
                    tick for tick in (0, 25, 50, 75, 100)
                    if tick <= maximum_percent
                ])
            if row != rows - 1:
                ax.set_xticks([])
            if column != 0:
                ax.set_yticks([])
            ax.text(0.03, 0.97, f"Cell {identity}", transform=ax.transAxes, va="top",
                    fontsize=ctx.theme.size("caption"), color=ctx.theme.colour("ink"),
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.0})
            made.append(ax); heat_handles.append(heat_handle)
        axes["kymograph"] = made[0] if made else fig.add_axes(rect)
        if made:
            fig.text(rect[0], rect[1] + rect[3] + 0.014, ctx.spec.panel("kymograph").heading(),
                     fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom")
            common.inset_colour_bar(made[-1], heat_handles[-1], ctx.theme,
                                    label=semantic_label(metric))
            fig.text(
                rect[0] + rect[2] / 2,
                rect[1] - 0.62 * scale / fig.get_figheight(),
                "Hours from start of recording", ha="center", va="top",
                fontsize=ctx.theme["axis_size"], color=ctx.theme.colour("ink"),
            )
            fig.text(
                rect[0] - 0.78 * scale / fig.get_figwidth(),
                rect[1] + rect[3] / 2,
                ("Distance from soma (% of current 95% reach)" if scaling == "cell"
                 else f"Distance from soma ({ctx.length_label})"),
                ha="center", va="center", rotation=90,
                fontsize=ctx.theme["axis_size"], color=ctx.theme.colour("ink"),
            )
            made[0]._semantic_legend_handled = True
            plot_axes = made
    required = [column for column in
                ["identity", "frame_index", "hours", "scaling", "ring",
                 "radius_inner", "radius_outer", "occupancy", "intersections",
                 "annulus_px", "annulus_support_fraction"]
                if column in data.columns]
    return FigureResult(
        figure=fig,
        axes=plot_axes or list(axes.values()),
        figure_data=data[required],
        subtitle=(f"{len(cells)} cells; "
                  + ("rings divide each cell's current 95% reach, so the vertical scale is relative"
                     if scaling == "cell" else
                     "rings are one fixed width across the movie, so a radius is an absolute distance")
                  + "; annuli with less than half their full geometric area available are omitted."),
        footnote=ctx.provenance_footnote(),
        auxiliary=ctx.provenance_auxiliary(),
        readme=(
            "Each small multiple is the generic `panels.morphology.sholl_kymograph` "
            "panel, so the same one-cell view can be embedded in a cell report card. "
            "No selected-time radial profile or multi-cell reach trace is drawn: those "
            "mix different questions with the time-by-radius result."
        ),
    )


if __name__ == "__main__":
    run_figure("sholl-kymograph")
