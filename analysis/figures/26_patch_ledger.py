"""Figure 26: per-cell territory coverage and pixel revisits.

How much of the ground a cell eventually covers it has reached by a given time,
how often it comes back to the same pixel, and whether territory is revealed
differently from repeated frame-order permutations. The denominator is each
cell's own final territory, not the field, so a small cell is not penalised.

    python analysis/figures/26_patch_ledger.py <run>
    ... --contour 0.25           the occupancy-frequency contour
    ... --cells 24               how many territories are drawn
    ... --shuffles 5000          frame-order permutation count
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import tifffile

from analysis.modules.territory import coverage_order_permutation_test

from _derive import _require_columns, _selected_identities
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Input, Option, Panel, Table, figure, run_figure

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
           Input("labels")),
    panels=(
        Panel("revisit", territory_panels.revisit_grid, block=True,
              item_width_inches=2.5, item_height_inches=2.5,
              title="Per-cell occupancy frequency maps"),
        Panel("coverage", territory_panels.coverage_curve,
              title="Cumulative territory coverage: observed versus permuted frame order"),
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
        Option("shuffles", default=1000),
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
    if ("revisit" in panels or "coverage" in panels) and label_path is None:
        raise SystemExit("Patch Ledger needs the labelled-cell stack for occupancy maps and the frame-order test")
    labels = tifffile.imread(label_path) if label_path else None
    cell_frame = frame if "revisit" in panels else pd.DataFrame()
    shown = cells
    columns = min(8, max(1, len(shown)))
    revisit_panel = ctx.spec.panel("revisit")
    minimum_sizes = ({"revisit": revisit_panel.grid_minimum(len(shown), columns)}
                     if "revisit" in panels else None)
    fig, axes = ctx.layout(panels, minimum_sizes=minimum_sizes)
    occupancy = ({identity: labels == int(identity) for identity in cells}
                 if labels is not None else {})
    test_units = pd.DataFrame()
    test_curves = pd.DataFrame()
    test_summary: dict = {}
    if "coverage" in axes:
        test_units, test_curves, test_summary = coverage_order_permutation_test(
            occupancy, shuffles=int(ctx.option("shuffles")), random_state=20260825,
        )
    if "revisit" in axes:
        rect = tuple(axes["revisit"].get_position().bounds)
        axes["revisit"].remove()
        scale = ctx.theme.canvas_scale
        gap_x = revisit_panel.item_gap_inches * scale / fig.get_figwidth()
        gap_y = revisit_panel.item_gap_inches * scale / fig.get_figheight()
        contour_fraction = float(ctx.option("contour"))
        soma = cell_frame.groupby("identity")[["soma_x", "soma_y"]].median() if not cell_frame.empty else pd.DataFrame()
        count_maps = {identity: np.count_nonzero(occupancy[identity], axis=0)
                      for identity in shown}
        observed_frames = {
            int(identity): int(value)
            for identity, value in territory.set_index("identity")["observed_frames"].items()
        }
        soma_locations = {
            int(identity): (float(soma.loc[identity, "soma_x"]),
                            float(soma.loc[identity, "soma_y"]))
            for identity in shown if identity in soma.index
        }
        revisit_result = territory_panels.revisit_grid(
            fig, rect, count_maps, ctx.theme,
            observed_frames=observed_frames, soma=soma_locations,
            contour_fraction=contour_fraction, max_columns=columns,
            gap_x=gap_x, gap_y=gap_y,
            heading=ctx.spec.panel("revisit").heading(),
        )
        ctx.drew("revisit", revisit_result)
        axes["revisit"] = revisit_result.axes[0]
    if "coverage" in axes:
        hours = (frame.groupby("frame_index")["hours"].first()
                 .reindex(np.arange(labels.shape[0])).interpolate(limit_direction="both"))
        pivot = test_curves.pivot(
            index="unit", columns="frame_index", values="observed_coverage_fraction"
        ).reindex(cells)
        population = test_curves.drop_duplicates("frame_index").sort_values("frame_index")
        ctx.drew(
            "coverage",
            territory_panels.coverage_curve(
                axes["coverage"], hours.to_numpy(float), pivot.to_numpy(float), ctx.theme,
                denominator=np.ones(len(pivot)),
                permutation_median=population["permutation_median"].to_numpy(float),
                permutation_lo=population["permutation_lo"].to_numpy(float),
                permutation_hi=population["permutation_hi"].to_numpy(float),
                test_p_value=float(test_summary["p_value"]),
                shuffles=int(test_summary["shuffles"]), hour_ticks=ctx.hour_ticks,
            ),
        )
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
    subtitle = f"{len(cells)} cells shown; coverage denominator is each cell's own final territory."
    if test_summary:
        subtitle += (
            f" Observed median coverage area {test_summary['observed_median_auc']:.3f} "
            f"versus permutation median {test_summary['permutation_median_auc']:.3f}; "
            f"p = {test_summary['p_value']:.3g}."
        )
    auxiliary = {"territory_summary.csv": territory}
    if test_summary:
        auxiliary.update({
            "coverage_order_test.csv": test_units,
            "coverage_order_curves.csv": test_curves,
            "statistics.csv": pd.DataFrame([test_summary]),
        })
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=subtitle,
        footnote=ctx.footnote(
            "Top: each tile is one cell; hue is the fraction of that cell's observed frames in which each pixel was occupied. All tiles use one spatial scale; black is the configured occupancy-frequency contour and the plus is median soma position.",
            "The null repeatedly reorders each cell's observed footprints among the same frames when it was present, preserving footprints and gaps while removing temporal order. The two-sided test compares the median per-cell area under the cumulative coverage curve; lower values mean territory was revealed later.",
        ),
        auxiliary=auxiliary,
    )


if __name__ == "__main__":
    run_figure("patch-ledger")
