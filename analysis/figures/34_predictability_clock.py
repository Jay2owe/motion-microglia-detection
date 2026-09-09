"""Figure 34: next-state mismatch across a configurable cycle.

For each current state, the descriptive baseline expects the successor seen
most often anywhere in this recording. The figure asks whether transitions
that differ from that baseline cluster at particular times in a chosen cycle;
it is not an out-of-sample forecast test.

    python analysis/figures/34_predictability_clock.py <run>
    ... --period-hours 24 --hour-ticks 6 --shuffles 1000
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
    summary="next-state mismatch across a configurable cycle",
    title="Next-state mismatch across the configured cycle",
    reads=(Table("regime_profiles.csv", module="regimes"),
           Table("regime_transitions.csv", module="regimes")),
    panels=(
        Panel("error", common.trace, min_width_inches=11.0, min_height_inches=5.0,
              title="Transitions that differ from the most common next state"),
        Panel("by_regime", common.ridgeline, block=True,
              item_width_inches=11.0, item_height_inches=1.1,
              item_gap_inches=0.0,
              title="Mismatch timing by current state"),
        Panel("dial", common.rose, polar=True,
              min_width_inches=8.0, min_height_inches=7.0,
              title="Next-state mismatches by time within the configured cycle"),
    ),
    options=(
        Option("bins", default=12),
        Option("shuffles", default=200),
        Option("period_hours", default=24.0),
        Option("hour_ticks", default=6.0),
    ),
    grammar="next-state mismatch trace, state ridgelines, and weighted phase distribution",
)
def build(ctx: FigureContext) -> FigureResult:
    transitions = ctx.table("regime_transitions.csv").copy()
    regime_names = _regime_labels(ctx.table("regime_profiles.csv"))
    bins = int(ctx.option("bins"))
    period_hours = float(ctx.option("period_hours"))
    if bins < 4:
        raise SystemExit("--bins must be at least 4")
    if not np.isfinite(period_hours) or period_hours <= 0:
        raise SystemExit("--period-hours must be finite and positive")

    counts = pd.crosstab(transitions["from_regime"], transitions["to_regime"])
    if counts.empty:
        raise SystemExit("regime_transitions.csv has no state transitions to display")
    expected = counts.idxmax(axis=1).to_dict()
    transitions["expected_next_regime"] = transitions["from_regime"].map(expected)
    transitions["mismatch"] = (
        transitions["expected_next_regime"] != transitions["to_regime"]
    ).astype(float)
    transitions["cycle_time_hours"] = transitions["hours"] % period_hours

    edges = np.linspace(0, period_hours, bins + 1)
    transitions["phase_bin"] = pd.cut(
        transitions["cycle_time_hours"], edges, labels=False, include_lowest=True,
    )
    per_cell = transitions.groupby(
        ["phase_bin", "identity"], as_index=False, observed=True,
    )["mismatch"].mean()
    grouped = per_cell.groupby("phase_bin", as_index=False, observed=True).agg(
        mismatch_fraction=("mismatch", "mean"),
        lower_quartile=("mismatch", lambda values: float(values.quantile(0.25))),
        upper_quartile=("mismatch", lambda values: float(values.quantile(0.75))),
        cells=("identity", "nunique"),
    )
    observations = transitions.groupby("phase_bin", observed=True).size()
    grouped["observations"] = grouped["phase_bin"].map(observations).astype(int)
    grouped["cycle_time_hours"] = grouped["phase_bin"].map(
        lambda value: (edges[int(value)] + edges[int(value) + 1]) / 2
    )

    shuffles = max(1, int(ctx.option("shuffles")))
    generator = np.random.default_rng(20260825)
    null = np.full((shuffles, bins), np.nan)
    phase_bins = transitions["phase_bin"].to_numpy(int)
    mismatch_values = transitions["mismatch"].to_numpy(float)
    identities = transitions["identity"].to_numpy()
    for replicate in range(shuffles):
        shuffled = generator.permutation(phase_bins)
        shuffled_per_cell = pd.DataFrame({
            "phase_bin": shuffled,
            "identity": identities,
            "mismatch": mismatch_values,
        }).groupby(["phase_bin", "identity"], observed=True)["mismatch"].mean()
        shuffled_means = shuffled_per_cell.groupby("phase_bin", observed=True).mean()
        null[replicate, shuffled_means.index.to_numpy(int)] = shuffled_means.to_numpy(float)
    grouped["surrogate_mean"] = grouped["phase_bin"].map(
        {i: np.nanmean(null[:, i]) for i in range(bins)}
    )
    grouped["surrogate_lower_95"] = grouped["phase_bin"].map(
        {i: np.nanpercentile(null[:, i], 2.5) for i in range(bins)}
    )
    grouped["surrogate_upper_95"] = grouped["phase_bin"].map(
        {i: np.nanpercentile(null[:, i], 97.5) for i in range(bins)}
    )
    figure_data = grouped[
        ["cycle_time_hours", "mismatch_fraction", "lower_quartile", "upper_quartile",
         "cells", "observations", "surrogate_mean", "surrogate_lower_95",
         "surrogate_upper_95"]
    ]

    rule_rows = []
    for current_state, expected_state in expected.items():
        state_counts = counts.loc[current_state]
        total = int(state_counts.sum())
        matching = int(state_counts.loc[expected_state])
        rule_rows.append({
            "from_regime": int(current_state),
            "current_state_label": regime_names.get(
                int(current_state), f"State {int(current_state)}"),
            "expected_next_regime": int(expected_state),
            "expected_next_state_label": regime_names.get(
                int(expected_state), f"State {int(expected_state)}"),
            "transitions": total,
            "matching_transitions": matching,
            "expected_share": matching / total,
        })
    next_state_rule = pd.DataFrame(rule_rows)

    panels = ctx.panels()
    regimes = sorted(int(value) for value in transitions["from_regime"].unique())
    ridge_panel = ctx.spec.panel("by_regime")
    minimum_sizes = ({"by_regime": ridge_panel.grid_minimum(len(regimes), 1)}
                     if "by_regime" in panels else None)
    fig, axes = ctx.layout(
        panels, minimum_sizes=minimum_sizes, gap_inches=2.0,
    )
    cycle_axis_label = f"Time within {period_hours:g} h cycle (h)"
    if "error" in axes:
        axes["error"].fill_between(
            grouped["cycle_time_hours"], grouped["surrogate_lower_95"],
            grouped["surrogate_upper_95"],
            color=ctx.theme.colour("reference"), alpha=0.25, linewidth=0,
            label="Cycle-time-label permutation 95% interval",
        )
        axes["error"].plot(
            grouped["cycle_time_hours"], grouped["surrogate_mean"],
            color=ctx.theme.colour("reference"),
            linewidth=ctx.theme.stroke("guide"),
        )
        axes["error"].fill_between(
            grouped["cycle_time_hours"], grouped["lower_quartile"],
            grouped["upper_quartile"],
            color=ctx.theme.colour("morphology"), alpha=0.18, linewidth=0,
            label="Cell interquartile range",
        )
        common.trace(
            axes["error"], grouped["cycle_time_hours"], grouped["mismatch_fraction"],
            ctx.theme, role="morphology", hour_ticks=ctx.hour_ticks,
            label="Mean across cells", x_label=cycle_axis_label,
            y_label="Transitions not matching the\nmost common next state (fraction)",
        )
    if "by_regime" in axes:
        rect = tuple(axes["by_regime"].get_position().bounds)
        axes["by_regime"].remove()
        distributions = [
            transitions.loc[
                (transitions["from_regime"] == regime)
                & (transitions["mismatch"] > 0),
                "cycle_time_hours",
            ].to_numpy(float)
            for regime in regimes
        ]
        made = common.ridgeline(
            fig, rect, distributions, ctx.theme,
            labels=[regime_names.get(value, f"State {value}") for value in regimes],
            bins=bins, overlap=0.15, labels_outside=True,
            limits=(0.0, period_hours), x_label=cycle_axis_label,
        ).axes
        axes["by_regime"] = made[0] if made else fig.add_axes(rect)
        fig.text(
            rect[0], rect[1] + rect[3] + 0.012,
            ctx.spec.panel("by_regime").heading(),
            fontsize=ctx.theme.size("panel"), fontweight="bold", va="bottom",
        )
    if "dial" in axes:
        common.rose(
            axes["dial"], transitions["cycle_time_hours"], ctx.theme,
            weights=transitions["mismatch"], bins=bins, period=period_hours,
            tick_step=ctx.hour_ticks, phase_unit="h", role="morphology",
            label="Mismatched transitions per time bin",
        )
        common.semantic_legend(
            axes["dial"], ctx.theme,
            handles=[
                Patch(facecolor=ctx.theme.colour("morphology")),
                Line2D([], [], color=ctx.theme.colour("ink"),
                       linewidth=ctx.theme.stroke("emphasis")),
            ],
            labels=["Mismatched transitions per time bin",
                    "Mismatch-weighted mean time"],
            location="right",
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=(
            "For each current state, the expected next state is its most frequent successor "
            "in this recording; this is a descriptive baseline, not forecast validation. "
            "The grey band permutes cycle-time labels."
        ),
        footnote=(
            "Cycle time is elapsed recording time modulo the selected cycle length; "
            "radial height is the number of mismatched transitions."
        ),
        auxiliary={"next_state_rule.csv": next_state_rule},
    )


if __name__ == "__main__":
    run_figure("predictability-clock")
