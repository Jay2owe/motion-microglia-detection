"""Figure 5: the per-cell report card the package can emit for every identity.

Every part of the page is an argument. The panels themselves live in ``panels``
and are drawn by this file, not defined in it, so the same image strip and the
same trace can be used on a page of their own::

    python analysis/figures/05_cell_report_card.py <run> --identity 7
    ... --metrics corrected_mean,area_px,solidity,step_px_gapless
    ... --images 6                       six tiles, evenly spaced
    ... --image-hours 0,12,24,36,48      or exactly these hours
    ... --images 0                       no tiles at all
    ... --cell-lut magma                 the LUT the cell is displayed through
    ... --trace-luts reporter,viridis,#c0392b
    ... --outline white
    ... --fit corrected_mean,area_px     or `--fit all`, or `--fit=` for none

The layout follows what was asked for: the canvas grows a row per metric and
loses the strip entirely when no tiles are wanted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tifffile

from _metrics import axis_label, describe, role_for
from _options import commas
from _schema import (FigureContext, FigureResult, Input, Option, Panel, Table,
                     figure, run_figure)
from panels import Look, cosinor_curve, image_strip, resolve_look, trace

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=5,
    slug="cell-report-card",
    summary="the per-cell report card the package can emit for every identity",
    title="Cell {identity}: {names}",
    grammar="image strip over one trace per requested measurement, for one cell",
    reads=(Table("cell_frame.csv", module="motility"),
           Table("rhythms.csv", module="rhythms"),
           Input("labels"), Input("raw")),
    panels=(
        Panel("tiles", image_strip, block=True, needs=("labels", "raw"),
              title="The outline over the signal"),
        Panel("traces", trace, title="One trace per measurement"),
    ),
    options=(
        # This page is a layout for any cell, not a chosen result, so which cell
        # it draws is an argument. 44 is only the one that happens to be
        # complete.
        Option("identity", default=44),
        Option("metrics",
               default=["corrected_mean", "area_px", "turnover_index"]),
        Option("images", default=10),
        Option("image_hours", default=[]),
        Option("cell_lut", default=None),
        Option("trace_luts", default=[]),
        Option("outline", default="outline"),
        Option("fit", default=None, cast=str, metavar="COL,COL|all",
               help="which traces carry the cosinor curve: a metric list, `all`, "
                    "or empty for none; unset draws it on the first fitted metric"),
        Option("hour_ticks", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    identity = ctx.option("identity")
    hour_ticks = ctx.option("hour_ticks")
    metrics = ctx.option("metrics")
    n_tiles = ctx.option("images")
    tile_hours = ctx.option("image_hours")
    cell_lut = ctx.option("cell_lut")
    trace_luts = ctx.option("trace_luts")
    outline = ctx.option("outline")
    fit = ctx.option("fit")

    if not metrics:
        raise SystemExit(
            "--metrics needs at least one column; pass --images 0 to drop the tiles")

    panels = ctx.panels()
    cell_frame = ctx.table("cell_frame.csv")
    rhythms = ctx.table("rhythms.csv")

    unknown = [column for column in metrics if column not in cell_frame.columns]
    if unknown:
        traceable = ", ".join(
            c for c in cell_frame.columns
            if c not in ("stem", "condition", "subject", "identity")
            and cell_frame[c].dtype.kind in "fi"
        )
        raise SystemExit(
            f"--metrics names {', '.join(unknown)}, which cell_frame.csv does not "
            f"have.\nColumns available: {traceable}"
        )

    cell = cell_frame[
        cell_frame["identity"] == identity].sort_values("frame_index").copy()
    if cell.empty:
        present = sorted(cell_frame["identity"].unique())
        raise SystemExit(
            f"identity {identity} is not in this run; it holds {len(present)} "
            f"identities, {present[0]} to {present[-1]}"
        )
    fits = rhythms[rhythms["identity"] == identity].set_index("metric")

    # Which traces carry the model curve. Left alone it is the first requested
    # metric that was actually fitted, so the page shows a fit when there is one to
    # show and says nothing when there is not.
    if fit is None:
        fitted_metrics = [column for column in metrics if column in fits.index][:1]
    elif fit.strip().lower() == "all":
        fitted_metrics = [column for column in metrics if column in fits.index]
    else:
        fitted_metrics = [column for column in commas(fit) if column in fits.index]

    hours = cell["hours"].to_numpy(float)

    # Label frame k is source frame k + offset; the tables carry both, so the offset
    # is read off them rather than repeated here.
    offset = int(cell["source_imagej_frame"].iloc[0] - cell["imagej_frame"].iloc[0])
    frames = cell["frame_index"].to_numpy(int)

    tile_frames: list[int] = []
    if "tiles" in panels and (n_tiles > 0 or tile_hours):
        if tile_hours:
            # Nearest observed frame to each requested hour: an hour the cell was
            # missing for still gets a tile, and the tile says which hour it is.
            tile_frames = [int(frames[int(np.abs(hours - wanted).argmin())])
                           for wanted in tile_hours]
        else:
            picks = np.linspace(0, len(frames) - 1, min(n_tiles, len(frames)))
            tile_frames = [int(frames[i]) for i in picks.round().astype(int)]

    crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    crop_box = (0, 0, 0, 0)
    if tile_frames:
        labels = tifffile.imread(ctx.input_path("labels"))
        raw_full = tifffile.imread(ctx.input_path("raw"))
        raw = raw_full[offset: offset + labels.shape[0]]

        # One crop for the whole strip, big enough for the cell at its largest, so
        # a tile that looks bigger is a bigger cell and not a tighter crop.
        centre_y = int(round(cell["centroid_y"].median()))
        centre_x = int(round(cell["centroid_x"].median()))
        half = 0
        for frame_index in frames:
            ys, xs = np.nonzero(labels[frame_index] == identity)
            if ys.size:
                half = max(half, int(np.abs(ys - centre_y).max()),
                           int(np.abs(xs - centre_x).max()))
        half = min(half + 5, 40)
        top = max(centre_y - half, 0)
        left_edge = max(centre_x - half, 0)
        bottom = min(centre_y + half + 1, labels.shape[1])
        right = min(centre_x + half + 1, labels.shape[2])
        crop_box = (top, bottom, left_edge, right)

        crops = [raw[f][top:bottom, left_edge:right].astype(float) for f in tile_frames]
        masks = [labels[f][top:bottom, left_edge:right] == identity for f in tile_frames]

    header_in = 1.7                       # title and subtitle
    panel_in = 1.75                       # one trace
    gap_in = 0.60                         # between traces
    foot_in = 2.05                        # footnote, tick labels and the x-axis label
    tile_title_in = 0.55                  # the hour written over each tile
    width_in = 15.5
    left, plot_width = 0.105, 0.845

    # A tile is square, so the strip's height follows from how many tiles share the
    # width. Computing it rather than guessing is what keeps the gap above the
    # traces the same whether the page carries three tiles or twelve.
    tile_in = (plot_width * width_in) / len(tile_frames) if tile_frames else 0.0
    strip_in = (tile_in + tile_title_in + 0.55) if tile_frames else 0.0
    traces_in = len(metrics) * panel_in + (len(metrics) - 1) * gap_in
    height_in = header_in + strip_in + traces_in + foot_in
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    tile_axes, vmin, vmax = [], float("nan"), float("nan")
    if tile_frames:
        strip = ctx.drew("tiles", image_strip(
            figure_,
            (left, fraction(height_in - header_in - tile_title_in - tile_in),
             plot_width, fraction(tile_in)),
            crops, ctx.theme,
            masks=masks,
            titles=[f"{float(cell.loc[cell.frame_index == f, 'hours'].iloc[0]):.0f} h"
                    for f in tile_frames],
            look=resolve_look(ctx.theme, cell_lut, "morphology") if cell_lut else None,
            outline_role=outline,
        ))
        tile_axes = strip.axes
        vmin, vmax = strip.extra["vmin"], strip.extra["vmax"]

    looks = {
        column: (
            resolve_look(ctx.theme, trace_luts[index], role_for(column))
            if index < len(trace_luts)
            else Look(colour=ctx.theme.colour(role_for(column)))
        )
        for index, column in enumerate(metrics)
    }

    axes = []
    each = fraction(panel_in)
    for index, column in enumerate(metrics):
        top_down = len(metrics) - 1 - index
        ax = figure_.add_axes([
            left,
            fraction(foot_in + top_down * (panel_in + gap_in)),
            plot_width,
            each,
        ])
        trace(
            ax, hours, cell[column], ctx.theme,
            look=looks[column], hour_ticks=hour_ticks,
            show_x=(index == len(metrics) - 1),
            y_label=axis_label(column, ctx.interval),
        )
        if column in fitted_metrics:
            row = fits.loc[column]
            cell[f"{column}_cosinor"] = cosinor_curve(
                ax, hours, cell[column], ctx.theme,
                period_hours=float(row["fixed_period_hours"]),
            ).extra["fitted"]
            ax.text(
                1.0, 1.04,
                f"{float(row['fixed_period_hours']):.0f} h cosinor, amplitude "
                f"{float(row['cosinor_relative_amplitude']) * 100:.0f}% of mean, peak "
                f"{float(row['cosinor_peak_hour']):.1f} h, p = "
                f"{float(row['cosinor_p_value']):.3g}   |   best free period "
                f"{float(row['lombscargle_period_hours']):.1f} h",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=ctx.theme.size("caption"), color=ctx.theme.colour("caption"),
            )
        axes.append(ax)

    kept = ["identity", "frame_index", "imagej_frame", "source_imagej_frame", "hours"]
    kept += [c for c in cell.columns if c in metrics or c.endswith("_cosinor")]

    auxiliary = {"rhythm_fits_this_cell.csv": rhythms[rhythms["identity"] == identity]}
    if tile_frames:
        auxiliary["tile_index.csv"] = pd.DataFrame({
            "position": range(len(tile_frames)),
            "frame_index": tile_frames,
            "hours": [float(cell.loc[cell.frame_index == f, "hours"].iloc[0])
                      for f in tile_frames],
            "area_px": [float(cell.loc[cell.frame_index == f, "area_px"].iloc[0])
                        if "area_px" in cell.columns else np.nan for f in tile_frames],
            "crop_top": crop_box[0], "crop_bottom": crop_box[1],
            "crop_left": crop_box[2], "crop_right": crop_box[3],
            "display_vmin": vmin, "display_vmax": vmax,
            "lut": cell_lut or ctx.theme["image_cmap"],
        })

    names = ", ".join(describe(column).label.lower() for column in metrics)

    return FigureResult(
        figure=figure_,
        axes=axes,
        figure_data=cell[kept],
        auxiliary=auxiliary,
        heading=f"Cell report card, identity {identity}",
        title_fields={"identity": identity, "names": names},
        subtitle=(
            f"{ctx.summary['stem']}. Present in {int(cell['frame_index'].count())} of "
            f"{ctx.summary['frames']} frames."
            + (f" Outline over the registered signal at {len(tile_frames)} times, "
               f"same crop and same contrast in every tile." if tile_frames else "")
        ),
        footnote=(
            f"All {ctx.summary['identities']} identities have these traces in "
            f"cell_frame.csv; this page renders columns of a table, not a separate "
            f"analysis.\n--identity draws a different cell and --metrics different "
            f"traces;\n--images and --image-hours change the tiles, --cell-lut and "
            f"--trace-luts the colours."
        ),
        readme=f"""## What the page shows

{'The outline over the raw signal at ' + str(len(tile_frames)) + ' times across the recording, then one trace per requested measurement.' if tile_frames else 'One trace per requested measurement.'}
Every tile shares one crop and one contrast, so a tile that looks brighter is
brighter signal and not a different stretch.

Traces drawn: {', '.join(metrics)}.
{'Cosinor curve on: ' + ', '.join(fitted_metrics) + '.' if fitted_metrics else 'No cosinor curve was drawn.'}

## Which cell

Identity {identity}. This is a layout for any identity, not a selected result;
every cell in `cell_frame.csv` can be drawn the same way, with any of its
numeric columns as a trace.

## How it is built

The panels are `panels.image_strip` and `panels.trace`, which this page calls
rather than contains. Anything they can draw here they can draw on their own
page, and a metric's colour and wording come from `_metrics`, so choosing a
different column does not need a builder change.""",
        console=(f"cell {identity}  frames {len(cell)}  metrics {metrics}  "
                 f"tiles {len(tile_frames)}"
                 + (f"  fit on {fitted_metrics}" if fitted_metrics else "")),
    )


if __name__ == "__main__":
    run_figure("cell-report-card", DEFAULT_RUN)
