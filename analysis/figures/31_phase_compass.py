"""Figure 31: within-cell metric coupling across time lags.

Whether one measurement leads another inside the same cell. A negative lag
means the first named series leads; the comparison is within a cell, so it is
not confounded by cells differing from each other.

    python analysis/figures/31_phase_compass.py <run>
    ... --metrics area_px:corrected_mean     a different pair
    ... --max-lag 24                         hours either side
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import coupling as coupling_panels


@figure(
    number=31,
    slug="phase-compass",
    summary="within-cell metric coupling across time lags",
    title="Within-cell metric coupling across time lags",
    reads=(Table("lag_profiles.csv", module="coupling"),),
    panels=(
        Panel("profile", coupling_panels.lag_profile,
              title="Population lead-lag relationship"),
        Panel("per_cell", common.scatter, title="Peak lead or lag for each cell"),
        Panel("pairs", coupling_panels.lag_profile,
              title="Lead-lag relationship by measurement pair"),
    ),
    options=(
        Option("metrics", default=["turnover_index:corrected_mean"],
               metavar="A:B,A:B", help="which measurement pairs are compared"),
        Option("max_lag", default=12.0, metavar="HOURS",
               help="largest lag drawn, in hours"),
    ),
    grammar="population lag profile peak scatter and pair profiles",
)
def build(ctx: FigureContext) -> FigureResult:
    data = ctx.table("lag_profiles.csv")
    requested = ctx.option("metrics")
    pairs = []
    for token in requested:
        pieces = token.split(":")
        if len(pieces) != 2 or not all(pieces):
            raise SystemExit("--metrics must contain a:b metric-pair tokens")
        pairs.append(tuple(pieces))
    wanted = pd.MultiIndex.from_tuples(pairs)
    pair_index = pd.MultiIndex.from_frame(data[["metric_a", "metric_b"]])
    data = data[pair_index.isin(wanted)].copy()
    if data.empty:
        available = sorted({f"{a}:{b}" for a, b in zip(pair_index.get_level_values(0), pair_index.get_level_values(1))})
        raise SystemExit(f"requested --metrics pair was not measured; available: {', '.join(available)}")
    maximum = float(ctx.option("max_lag"))
    data = data[data["lag_hours"].abs() <= maximum]
    grouped = data.groupby(["metric_a", "metric_b", "lag_frames", "lag_hours"], as_index=False).agg(
        mean_correlation=("correlation", "mean"),
        lo=("correlation", lambda values: float(values.quantile(0.25))),
        hi=("correlation", lambda values: float(values.quantile(0.75))),
        cells=("identity", "nunique"),
        surrogate_mean=("surrogate_mean", "mean"),
        surrogate_lo=("surrogate_lo", "mean"),
        surrogate_hi=("surrogate_hi", "mean"),
        pairs=("pairs", "sum"),
    )
    figure_data = grouped[["metric_a", "metric_b", "lag_frames", "lag_hours",
                           "mean_correlation", "lo", "hi", "cells",
                           "surrogate_mean", "surrogate_lo", "surrogate_hi"]]
    peaks = []
    for keys, group in data.groupby(["identity", "metric_a", "metric_b"], sort=True):
        valid = group.dropna(subset=["correlation"])
        if valid.empty:
            continue
        row = valid.loc[valid["correlation"].abs().idxmax()]
        peaks.append({"identity": int(keys[0]), "metric_a": keys[1], "metric_b": keys[2],
                      "peak_lag_hours": float(row["lag_hours"]),
                      "peak_correlation": float(row["correlation"]), "pairs": int(row["pairs"])})
    peak_data = pd.DataFrame(peaks)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    first_pair = grouped[(grouped["metric_a"] == pairs[0][0]) & (grouped["metric_b"] == pairs[0][1])]
    if "profile" in axes:
        coupling_panels.lag_profile(
            axes["profile"], first_pair["lag_hours"], first_pair["mean_correlation"], ctx.theme,
            surrogate=first_pair, pairs=first_pair["pairs"],
            thin_below=max(3, int(first_pair["pairs"].quantile(0.1))),
            role=role_for(pairs[0][0]), x_label="Lag (h)", y_label="Mean correlation",
            direction_note=True,
            label=f"{semantic_label(pairs[0][0])} → {semantic_label(pairs[0][1])}",
        )
        axes["profile"].fill_between(first_pair["lag_hours"], first_pair["lo"], first_pair["hi"],
                                     color=ctx.theme.colour(role_for(pairs[0][0])), alpha=0.14, linewidth=0)
    if "per_cell" in axes and not peak_data.empty:
        common.scatter(axes["per_cell"], peak_data["peak_lag_hours"], peak_data["peak_correlation"],
                       ctx.theme, role="morphology")
        axes["per_cell"].axvline(0, color=ctx.theme.colour("reference"), linewidth=ctx.theme.stroke("guide"))
        axes["per_cell"].axhline(0, color=ctx.theme.colour("reference"), linewidth=ctx.theme.stroke("guide"))
        axes["per_cell"].set(xlabel="Each cell's peak lag (h)", ylabel="Peak correlation")
    if "pairs" in axes:
        for (metric_a, metric_b), group in grouped.groupby(["metric_a", "metric_b"]):
            coupling_panels.lag_profile(
                axes["pairs"], group["lag_hours"], group["mean_correlation"], ctx.theme,
                surrogate=group, pairs=group["pairs"], role=role_for(metric_a),
                x_label="Lag (h)", y_label="Mean correlation", direction_note=True,
                label=f"{semantic_label(metric_a)} → {semantic_label(metric_b)}",
            )
        common.semantic_legend(axes["pairs"], ctx.theme, location="inside")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Negative lag means the first named series leads; {data['identity'].nunique()} identities, capped at {maximum:g} h.",
        auxiliary={"per_cell_peaks.csv": peak_data},
    )


if __name__ == "__main__":
    run_figure("phase-compass")
