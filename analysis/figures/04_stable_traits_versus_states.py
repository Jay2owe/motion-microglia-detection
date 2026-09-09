"""Figure 4: which measurements are stable traits or changing states by condition.

One full-size ``panels.common.lollipop`` panel is drawn per experimental
condition, with the same measurement order and x scale in every panel::

    python analysis/figures/04_stable_traits_versus_states.py <run>
    ... --metrics area_px,corrected_mean,solidity,turnover_index
    ... --hues teal,orange                 override condition colours

The figure reads the pooled ``cell_summary.csv`` because a condition is a
run-level property. Hues default to the condition colours recorded with the
analysis run; ``--hues`` is an explicit display override in displayed condition
order. Each panel keeps its physical width when more conditions are added.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _bundle import design_for
from _metrics import describe
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import Mark, lollipop, reference_lines

#: Descriptive display guides, not tested biological cut-offs.
TRAIT_MAX = 0.5
STATE_MIN = 0.7

DEFAULT_RUN = "outputs/g02_95_A3_trend"


@figure(
    number=4,
    slug="stable-traits-versus-states",
    summary="within-cell against between-cell variability, separately by condition",
    title="Within-cell versus between-cell variability, by condition",
    grammar="condition-faceted lollipop of within-over-between spread ratio",
    reads=(Table("cell_summary.csv", module="motility", scope="run"),),
    panels=(Panel("ranking", lollipop,
                  title="Within-cell to between-cell spread ratio"),),
    options=(
        Option("metrics",
               default=["area_px", "corrected_mean", "skeleton_branches", "solidity",
                        "ramification_index", "turnover_index"],
               help="which measurements get a row in every condition panel"),
        Option("hues", default=[],
               help="condition colours in displayed order; defaults to the run design"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    metrics = ctx.option("metrics")
    requested_hues = ctx.option("hues")
    if not metrics:
        raise SystemExit("--metrics needs at least one measurement to compare")

    summary = ctx.table("cell_summary.csv")
    missing = [
        column for column in metrics
        if f"{column}_median" not in summary.columns
        or f"{column}_iqr" not in summary.columns
    ]
    if missing:
        available = sorted(
            name[: -len("_median")] for name in summary.columns
            if name.endswith("_median")
            and f"{name[:-len('_median')]}_iqr" in summary.columns
        )
        raise SystemExit(
            f"--metrics names {', '.join(missing)}, which pooled cell_summary.csv "
            f"has no median and interquartile range for.\nMeasurements available: "
            + ", ".join(available)
        )

    if "condition" not in summary.columns:
        summary = summary.copy()
        summary["condition"] = "all cells"
    summary["condition"] = summary["condition"].fillna("unassigned").astype(str)

    design = design_for(ctx.run)
    declarations = {
        str(item["name"]): item
        for item in design.get("conditions", [])
        if item.get("name") is not None
    }
    observed = list(dict.fromkeys(summary["condition"].tolist()))
    condition_order = [name for name in declarations if name in observed]
    condition_order.extend(sorted(name for name in observed if name not in declarations))
    if not condition_order:
        raise SystemExit("pooled cell_summary.csv has no condition rows to draw")

    condition_labels = {
        name: str(declarations.get(name, {}).get("label") or name.replace("_", " ").title())
        for name in condition_order
    }
    if requested_hues and len(requested_hues) != len(condition_order):
        raise SystemExit(
            f"--hues needs one colour for each displayed condition "
            f"({len(condition_order)}: {', '.join(condition_order)}); got "
            f"{len(requested_hues)}"
        )
    try:
        colours = (
            [ctx.theme.colour(value) for value in requested_hues]
            if requested_hues
            else [ctx.theme.condition_colour(name, index)
                  for index, name in enumerate(condition_order)]
        )
    except KeyError as error:
        raise SystemExit(f"--hues: {error.args[0]}") from None
    condition_colours = dict(zip(condition_order, colours))

    metric_descriptions = {column: describe(column) for column in metrics}
    rows = []
    for condition_index, condition in enumerate(condition_order):
        group = summary[summary["condition"] == condition]
        movie_count = int(group["stem"].nunique()) if "stem" in group else 1
        for column in metrics:
            median_column = f"{column}_median"
            iqr_column = f"{column}_iqr"
            usable = group[[median_column, iqr_column]].dropna()
            medians = usable[median_column].to_numpy(float)
            within = usable[iqr_column].to_numpy(float)
            between = (
                float(np.percentile(medians, 75) - np.percentile(medians, 25))
                if len(medians) >= 2 else np.nan
            )
            median_within = float(np.median(within)) if len(within) else np.nan
            ratio = (
                float(median_within / between)
                if np.isfinite(between) and between > 0 else np.nan
            )
            metric = metric_descriptions[column]
            rows.append({
                "condition": condition,
                "condition_label": condition_labels[condition],
                "condition_hue": condition_colours[condition],
                "condition_order": condition_index,
                "movies": movie_count,
                "metric": column,
                "label": metric.label,
                "unit": metric.unit_text(ctx.interval),
                "cells": int(len(usable)),
                "between_cell_iqr_of_medians": between,
                "median_within_cell_iqr": median_within,
                "within_over_between": ratio,
            })

    table = pd.DataFrame(rows)
    finite = table[np.isfinite(table["within_over_between"])]
    if finite.empty:
        raise SystemExit(
            "none of the requested measurements has a non-zero between-cell "
            "interquartile range in any condition"
        )

    metric_order = (
        finite.groupby("metric", sort=False)["within_over_between"]
        .median().sort_values(kind="stable").index.tolist()
    )
    metric_order.extend(column for column in metrics if column not in metric_order)
    metric_positions = {column: index for index, column in enumerate(metric_order)}
    table["metric_order"] = table["metric"].map(metric_positions).astype(int)
    table["classification"] = np.where(
        table["within_over_between"] <= TRAIT_MAX, "trait-like",
        np.where(table["within_over_between"] >= STATE_MIN,
                 "state-like", "intermediate"),
    )
    table = table.sort_values(
        ["condition_order", "metric_order"], kind="stable"
    ).reset_index(drop=True)

    maximum = float(finite["within_over_between"].max())
    upper = max(1.5, np.ceil(maximum * 1.22 * 2.0) / 2.0)
    panel_label_in = 2.65
    panel_plot_in = 4.65
    panel_gap_in = 0.55
    left_in = 0.38
    right_in = 0.55
    panel_total_in = panel_label_in + panel_plot_in
    width_in = (
        left_in + len(condition_order) * panel_total_in
        + max(0, len(condition_order) - 1) * panel_gap_in + right_in
    )
    header_in = 2.65
    row_in = 0.96
    plot_height_in = len(metric_order) * row_in
    foot_in = 2.85
    height_in = header_in + plot_height_in + foot_in

    ctx.panels()
    figure_ = ctx.sheet(width_in, height_in)
    axes = []
    for index, condition in enumerate(condition_order):
        panel_left_in = left_in + index * (panel_total_in + panel_gap_in)
        ax = figure_.add_axes([
            (panel_left_in + panel_label_in) / width_in,
            foot_in / height_in,
            panel_plot_in / width_in,
            plot_height_in / height_in,
        ])
        axes.append(ax)
        colour = condition_colours[condition]
        condition_table = (
            table[table["condition"] == condition]
            .set_index("metric").reindex(metric_order).reset_index()
        )

        ax.axvspan(0, min(TRAIT_MAX, upper), color=ctx.theme.colour("stable"),
                   alpha=0.055, linewidth=0, zorder=0)
        if upper > STATE_MIN:
            ax.axvspan(STATE_MIN, upper, color=ctx.theme.colour("variable"),
                       alpha=0.035, linewidth=0, zorder=0)
        reference_lines(ax, [Mark(1.0, role="caption", dashed=True)], ctx.theme)
        drawn = lollipop(
            ax,
            condition_table["within_over_between"],
            ctx.theme,
            labels=condition_table["label"],
            colours=[colour] * len(condition_table),
            annotations=[
                f"{value:.2f}" if np.isfinite(value) else "not estimable"
                for value in condition_table["within_over_between"]
            ],
        )
        if index == 0:
            ctx.drew("ranking", drawn)
        cells = int(condition_table["cells"].max())
        movies = int(condition_table["movies"].max())
        ax.set_title(
            f"{condition_labels[condition]}\n{cells} cells · {movies} "
            f"{'movie' if movies == 1 else 'movies'}",
            loc="left", color=colour, fontweight="bold",
            fontsize=ctx.theme.size("panel"), pad=12,
        )
        ax.set_xlim(0, upper)
        ax.set_xticks(np.arange(0, upper + 0.01, 0.5))
        ax.set_xlabel("Within-cell ÷ between-cell IQR\nLower = more stable within cells")

    span_hours = (
        float(summary["last_hour"].max() - summary["first_hour"].min())
        if {"first_hour", "last_hour"}.issubset(summary.columns)
        else float(ctx.summary["hours_covered"])
    )
    movie_total = int(summary["stem"].nunique()) if "stem" in summary else 1
    cell_total = (
        int(summary[["stem", "identity"]].drop_duplicates().shape[0])
        if {"stem", "identity"}.issubset(summary.columns)
        else int(summary["identity"].nunique()) if "identity" in summary else len(summary)
    )
    synthetic = bool(design.get("synthetic", False))
    synthetic_note = (
        " SYNTHETIC TEST DESIGN: condition assignments are for layout validation only."
        if synthetic else ""
    )
    one_movie = [
        condition_labels[condition]
        for condition in condition_order
        if int(table.loc[table["condition"] == condition, "movies"].max()) < 2
    ]
    replication_note = (
        " Only one movie contributes to: " + ", ".join(one_movie) + "."
        if one_movie else ""
    )

    return FigureResult(
        figure=figure_,
        axes=axes,
        figure_data=table,
        heading="Within-cell versus between-cell variability, by condition",
        subtitle=(
            f"{len(condition_order)} conditions, {movie_total} movies and {cell_total} "
            f"cells over {span_hours:.0f} h. Every ratio is calculated independently "
            f"within its condition."
        ),
        footnote=(
            "Interquartile range (IQR) is the middle 50% of values. The dashed line "
            "at 1.0 marks equal within-cell and between-cell spread. Pale regions are "
            f"descriptive guides only: trait-like ≤ {TRAIT_MAX:.1f}, state-like ≥ "
            f"{STATE_MIN:.1f}. Hues encode condition.{synthetic_note}{replication_note}"
        ),
        readme=f"""## What is compared

For each condition and measurement, the numerator is the median cell's own
interquartile range over the recording. The denominator is the interquartile
range of cell medians within that same condition. Conditions are never pooled
to calculate either spread.

Measurements have one common row order, determined by their median ratio across
conditions, and every condition panel has the same x scale. Each condition gets
a full {panel_plot_in:.2f}-inch plotting area; adding conditions widens the page
instead of shrinking the panels.

## Hues

By default, hues come from the run's condition declarations. `--hues` accepts
one theme colour, palette name or hexadecimal value per displayed condition and
records the resolved colours in `figure_data.csv`.

## Interpretation

The {TRAIT_MAX:.1f} and {STATE_MIN:.1f} boundaries are visual guides, not tested
biological thresholds. This is a descriptive figure and reports no inferential
statistics.{synthetic_note}{replication_note}
""",
        console=table.round(3).to_string(index=False),
    )


if __name__ == "__main__":
    run_figure("stable-traits-versus-states", DEFAULT_RUN)
