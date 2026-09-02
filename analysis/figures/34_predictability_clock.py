"""Figure 34: one-step regime prediction error within the assumed day.

The predictor is the whole-recording transition matrix, so it knows nothing
about time of day. If the error still rises and falls on a daily cycle, the
states are less predictable at some hours than others.

    python analysis/figures/34_predictability_clock.py <run>
    ... --shuffles 1000          replicates behind the null band
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _derive import _regime_labels
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common


@figure(
    number=34,
    slug="predictability-clock",
    summary="one-step regime prediction error within the assumed day",
    title="One-step regime prediction error within the assumed day",
    reads=(Table("cell_frame.csv", module="regimes"),
           Table("regime_profiles.csv", module="regimes"),
           Table("regime_transitions.csv", module="regimes")),
    panels=(
        Panel("error", common.trace, title="Incorrect next-state predictions"),
        Panel("by_regime", common.ridgeline, block=True,
              title="Error timing by current state"),
        Panel("dial", common.rose, polar=True, title="Error-weighted time of day"),
    ),
    options=(
        Option("bins", default=12),
        Option("shuffles", default=200),
        Option("metrics", default=[],
               help="extra measured columns to residualise beside the states"),
        Option("hour_ticks", default=24.0),
    ),
    grammar="prediction-error trace regime ridgelines and weighted dial",
)
def build(ctx: FigureContext) -> FigureResult:
    transitions = ctx.table("regime_transitions.csv")
    regime_names = _regime_labels(ctx.table("regime_profiles.csv"))
    counts = pd.crosstab(transitions["from_regime"], transitions["to_regime"])
    predicted = counts.idxmax(axis=1).to_dict()
    transitions["predicted_regime"] = transitions["from_regime"].map(predicted)
    transitions["error"] = (transitions["predicted_regime"] != transitions["to_regime"]).astype(float)
    transitions["phase_hour"] = transitions["hours"] % 24.0
    bins = int(ctx.option("bins"))
    edges = np.linspace(0, 24, bins + 1)
    transitions["phase_bin"] = pd.cut(transitions["phase_hour"], edges, labels=False, include_lowest=True)
    per_cell = transitions.groupby(["phase_bin", "identity"], as_index=False)["error"].mean()
    grouped = per_cell.groupby("phase_bin", as_index=False).agg(
        mean_error=("error", "mean"), lo=("error", lambda values: float(values.quantile(0.25))),
        hi=("error", lambda values: float(values.quantile(0.75))), cells=("identity", "nunique"))
    observations = transitions.groupby("phase_bin").size()
    grouped["observations"] = grouped["phase_bin"].map(observations).astype(int)
    grouped["hour"] = grouped["phase_bin"].map(lambda value: (edges[int(value)] + edges[int(value) + 1]) / 2)
    shuffles = max(1, int(ctx.option("shuffles")))
    generator = np.random.default_rng(20260825)
    null = np.full((shuffles, bins), np.nan)
    phase_bins = transitions["phase_bin"].to_numpy(int)
    error_values = transitions["error"].to_numpy(float)
    for replicate in range(shuffles):
        shuffled = generator.permutation(phase_bins)
        for phase_bin in range(bins):
            selected = error_values[shuffled == phase_bin]
            null[replicate, phase_bin] = selected.mean() if len(selected) else np.nan
    grouped["surrogate_mean"] = grouped["phase_bin"].map({i: np.nanmean(null[:, i]) for i in range(bins)})
    grouped["surrogate_lo"] = grouped["phase_bin"].map({i: np.nanpercentile(null[:, i], 2.5) for i in range(bins)})
    grouped["surrogate_hi"] = grouped["phase_bin"].map({i: np.nanpercentile(null[:, i], 97.5) for i in range(bins)})
    figure_data = grouped[["hour", "mean_error", "lo", "hi", "cells", "observations",
                           "surrogate_mean", "surrogate_lo", "surrogate_hi"]]

    residuals = pd.DataFrame()
    requested_metrics = ctx.option("metrics")
    if requested_metrics:
        cell_frame = ctx.table("cell_frame.csv")
        metric_rows = []
        for metric in requested_metrics:
            if metric not in cell_frame:
                raise SystemExit(f"--metrics {metric} is not in cell_frame.csv")
            for identity, cell in cell_frame[["identity", "frame_index", "hours", metric]].dropna().groupby("identity"):
                cell = cell.sort_values("frame_index")
                values = cell[metric].to_numpy(float)
                if len(values) < 3:
                    continue
                slope, intercept = np.polyfit(values[:-1], values[1:], 1)
                errors = np.abs(values[1:] - (slope * values[:-1] + intercept))
                scale = np.nanmedian(errors) or 1.0
                metric_rows.extend({"identity": int(identity), "metric": metric, "hours": hour,
                                    "phase_hour": hour % 24, "standardised_residual": error / scale}
                                   for hour, error in zip(cell["hours"].to_numpy(float)[1:], errors))
        residuals = pd.DataFrame(metric_rows)

    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "error" in axes:
        axes["error"].fill_between(grouped["hour"], grouped["surrogate_lo"], grouped["surrogate_hi"],
                                   color=ctx.theme.colour("reference"), alpha=0.25, linewidth=0,
                                   label="Hour-shuffled interval")
        axes["error"].plot(grouped["hour"], grouped["surrogate_mean"],
                           color=ctx.theme.colour("reference"), linewidth=ctx.theme.stroke("guide"))
        axes["error"].fill_between(grouped["hour"], grouped["lo"], grouped["hi"],
                                   color=ctx.theme.colour("morphology"), alpha=0.18, linewidth=0)
        common.trace(axes["error"], grouped["hour"], grouped["mean_error"], ctx.theme,
                     role="morphology", hour_ticks=ctx.hour_ticks,
                     label="Observed cells",
                     x_label="Hour within the assumed day",
                     y_label="Incorrect predictions\n(fraction)")
    if "by_regime" in axes:
        rect = tuple(axes["by_regime"].get_position().bounds); axes["by_regime"].remove()
        regimes = sorted(int(value) for value in transitions["from_regime"].unique())
        distributions = [transitions.loc[(transitions["from_regime"] == regime) &
                                         (transitions["error"] > 0), "phase_hour"].to_numpy(float)
                         for regime in regimes]
        made = common.ridgeline(fig, rect, distributions, ctx.theme,
                                labels=[regime_names.get(value, f"State {value}") for value in regimes],
                                bins=bins, x_label="Error hour").axes
        axes["by_regime"] = made[0] if made else fig.add_axes(rect)
        fig.text(rect[0], rect[1] + rect[3] + 0.012, ctx.spec.panel("by_regime").heading(),
                 fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom")
    if "dial" in axes:
        common.rose(axes["dial"], transitions["phase_hour"], ctx.theme,
                    weights=transitions["error"], bins=bins, period=24, role="morphology",
                    label="Incorrect predictions per time bin")
        common.semantic_legend(
            axes["dial"], ctx.theme,
            handles=[
                Patch(facecolor=ctx.theme.colour("morphology")),
                Line2D([], [], color=ctx.theme.colour("ink"), linewidth=ctx.theme.stroke("emphasis")),
            ],
            labels=["Incorrect predictions per time bin", "Error-weighted mean hour"],
            location="right",
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle="The predictor is the whole-recording transition matrix; the null shuffles hour labels only.",
        auxiliary={"continuous_residuals.csv": residuals},
    )


if __name__ == "__main__":
    run_figure("predictability-clock")
