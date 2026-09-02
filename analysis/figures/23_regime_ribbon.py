"""Figure 23: behavioural regime assigned to every cell-frame.

The states are data-derived clusters, not categories the package believes in.
Each is named by the measurement it is most extreme on, and a cell-frame whose
assignment was marginal is washed out rather than dropped.

    python analysis/figures/23_regime_ribbon.py <run>
    ... --order first_appearance     row order of the ribbon
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _derive import _regime_labels, _regime_rows
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import morphology as morphology_panels


@figure(
    number=23,
    slug="regime-ribbon",
    summary="behavioural regime assigned to every cell-frame",
    title="Behavioural regime assigned to every cell-frame",
    reads=(Table("cell_frame.csv", module="regimes"),
           Table("regime_profiles.csv", module="regimes")),
    panels=(
        Panel("ribbon", morphology_panels.regime_ribbon,
              title="Cell state through time"),
        Panel("occupancy", morphology_panels.occupancy_area,
              title="Share of cells in each state"),
        Panel("profiles", common.matrix,
              title="Measurements defining each state"),
    ),
    options=(
        Option("order", default="dominant", cast=str,
               help="row order of the ribbon: dominant, first_appearance or identity"),
        Option("hour_ticks", default=24.0),
    ),
    grammar="categorical ribbon stacked area and matrix",
)
def build(ctx: FigureContext) -> FigureResult:
    regimes = _regime_rows(ctx.table("cell_frame.csv"))
    profiles = ctx.table("regime_profiles.csv")
    regime_names = _regime_labels(profiles)
    panels = ctx.panels()
    order_mode = ctx.option("order")
    pivot = regimes.pivot(index="identity", columns="hours", values="regime")
    margin = regimes.pivot(index="identity", columns="hours", values="regime_margin").reindex_like(pivot)
    if order_mode == "dominant":
        order_ids = regimes.groupby("identity")["regime"].agg(lambda values: values.mode().iloc[0]).sort_values().index
    elif order_mode == "switches":
        order_ids = regimes.sort_values("frame_index").groupby("identity")["regime"].agg(
            lambda values: int(np.count_nonzero(np.diff(values)))
        ).sort_values().index
    elif order_mode == "first_appearance":
        order_ids = regimes.groupby("identity")["hours"].min().sort_values().index
    else:
        raise SystemExit("--order must be dominant, first_appearance or switches")
    pivot = pivot.reindex(order_ids)
    margin = margin.reindex(order_ids)
    row_order = {identity: index for index, identity in enumerate(order_ids)}
    figure_data = regimes.copy()
    figure_data["row_order"] = figure_data["identity"].map(row_order)
    fig, axes = ctx.layout(
        panels, bottom_inches=2.35 if "profiles" in panels else 1.55)
    if "ribbon" in axes:
        morphology_panels.regime_ribbon(
            axes["ribbon"], pivot, pivot.columns, ctx.theme,
            margins=margin.to_numpy(float), hour_ticks=ctx.hour_ticks,
            legend=False, regime_labels=regime_names,
        )
        axes["ribbon"]._semantic_legend_handled = True
    regime_ids = sorted(int(value) for value in regimes["regime"].unique())
    if "occupancy" in axes:
        fractions = (regimes.groupby(["hours", "regime"])["identity"].count()
                     .unstack(fill_value=0).reindex(columns=regime_ids, fill_value=0))
        morphology_panels.occupancy_area(axes["occupancy"], fractions.index,
                                         fractions.to_numpy(float), ctx.theme,
                                         regime_ids=regime_ids, hour_ticks=ctx.hour_ticks,
                                         regime_labels=regime_names)
        common.semantic_legend(axes["occupancy"], ctx.theme, location="inside", columns=2)
    if "profiles" in axes:
        feature_columns = [column for column in profiles.columns
                           if column not in {"stem", "condition", "subject", "regime"}
                           and pd.api.types.is_numeric_dtype(profiles[column])]
        matrix_values = profiles[feature_columns].to_numpy(float)
        matrix_values = (matrix_values - np.nanmean(matrix_values, axis=0)) / np.where(
            np.nanstd(matrix_values, axis=0) == 0, 1, np.nanstd(matrix_values, axis=0)
        )
        profile_handle = common.matrix(
            axes["profiles"], matrix_values, ctx.theme,
            row_labels=[regime_names[int(value)] for value in profiles["regime"]],
            column_labels=[semantic_label(column) for column in feature_columns],
            cmap=ctx.theme["diverging_cmap"], annotate=False,
            x_label="Measurement defining the regime",
        ).extra["handle"]
        common.inset_colour_bar(
            axes["profiles"], profile_handle, ctx.theme,
            label="Regime mean (standard deviations)",
        )
        for label in axes["profiles"].get_xticklabels():
            label.set_fontsize(ctx.theme.size("caption"))
            label.set_rotation(38)
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"{len(regime_ids)} data-derived states named by their strongest relative feature; low-margin assignments are washed out.",
        auxiliary={"regime_profiles.csv": profiles},
    )


if __name__ == "__main__":
    run_figure("regime-ribbon")
