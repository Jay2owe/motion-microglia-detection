"""Figure 40: what changed in each cell between one window and the next.

Every cell is summarised once inside each declared window, and this page draws
the difference: one cell against itself, across every metric the run measured.
Because a cell is its own control, nothing here depends on two cells being
comparable.

What the difference means is a separate question this page does not answer.
Two windows of one recording differ by time alone, so a shift can be biology,
photobleaching, or focus drift, and the figure distinguishes none of them.

    python analysis/figures/40_change_ledger.py <run>
    ... --metrics area_px,corrected_mean    narrow the ledger
    ... --min-coverage 0.75                 how much of a window a cell must be seen in
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _metrics import describe, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common

#: Which summary statistic of a cell the page draws the change in. The window
#: tables carry four - median, iqr, mean, sd - and they are different
#: quantities: a change in a median and a change in a spread cannot share an
#: axis or a sentence. The median is drawn and the choice is stated on the page.
STATISTIC = "median"

#: How many metrics the spread panel can show before its rows stop being
#: readable. The ledger shows every metric; this one shows the largest movers,
#: and says how many it left out.
SPREAD_ROWS = 8


@figure(
    number=40,
    slug="change-ledger",
    summary="what changed in each cell between the baseline window and the next",
    title="Per-cell change from {baseline_window} to {window}",
    grammar="ranked fractional changes, paired slopes and change distributions",
    reads=(Table("window_change.csv", module="windows"),
           Table("cell_summary_windowed.csv", module="windows")),
    panels=(
        Panel("ledger", common.lollipop,
              title="All metrics: median scaled change across cells"),
        Panel("paired", common.paired_slopes,
              title="Baseline versus comparison window for each cell"),
        Panel("spread", common.ridgeline, block=True,
              item_width_inches=11.0, item_height_inches=1.0,
              item_gap_inches=0.0,
              title="Eight largest movers: scaled change for each cell"),
    ),
    options=(
        Option("metrics", default=[],
               help="which measured columns the ledger draws; empty draws every one"),
        Option("min_coverage", default=0.5),
        Option("bins", default=24),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    changes = ctx.table("window_change.csv")
    windowed = ctx.table("cell_summary_windowed.csv")

    rows = changes[changes["statistic"] == STATISTIC].copy()
    if rows.empty:
        raise SystemExit(
            f"window_change.csv holds no {STATISTIC!r} rows; this run summarised "
            f"windows with {', '.join(sorted(changes['statistic'].unique()))}")

    # A cell seen in two frames of a fifty-frame window has a window median
    # computed from two observations, and its change is noise wearing the same
    # clothes as a result. Both windows have to clear the floor: a cell fully
    # observed in the baseline and barely present afterwards is exactly the
    # case a one-sided check lets through.
    floor = float(ctx.option("min_coverage"))
    coverage = windowed.pivot_table(index="identity", columns="window",
                                    values="window_coverage", aggfunc="min")
    clears = coverage.min(axis=1) >= floor
    kept = set(clears[clears].index)
    dropped = int(rows["identity"].nunique() - len(kept & set(rows["identity"])))
    rows = rows[rows["identity"].isin(kept)]
    rows = rows.merge(coverage.min(axis=1).rename("window_coverage"),
                      left_on="identity", right_index=True, how="left")

    wanted = [str(name) for name in ctx.option("metrics")]
    available = sorted(str(value) for value in rows["metric"].dropna().unique())
    if wanted:
        unknown = [name for name in wanted if name not in available]
        if unknown:
            raise SystemExit(
                f"--metrics {','.join(unknown)} is not in window_change.csv. "
                f"It holds: {', '.join(available)}")
        rows = rows[rows["metric"].isin(wanted)]

    # Thirty-seven metrics in pixels, camera units and bare ratios cannot share
    # a horizontal axis. Each cell's change is divided by one common number per
    # metric: the median baseline value across retained cells. This is not that
    # cell's own fractional change; using an explicit column name prevents the
    # denominator being mistaken for the cell's baseline value.
    #
    # The divisor has to be safely positive or the fraction is worse than no
    # scaling at all: dividing by a baseline of zero is infinite, and dividing
    # by one that straddles zero flips the sign of the answer. dff is the case
    # that makes this concrete - it is centred on zero by construction, so a
    # fraction of its baseline means nothing whatever the arithmetic says.
    baselines = rows.groupby("metric")["baseline_value"]
    divisor = baselines.median()
    lower = baselines.quantile(0.25)
    scalable = divisor[(divisor > 0) & (lower > 0)]
    unscalable = sorted(set(divisor.index) - set(scalable.index))
    rows["population_median_baseline"] = rows["metric"].map(divisor)
    rows["change_over_population_median_baseline"] = np.where(
        rows["metric"].isin(scalable.index),
        rows["change"] / rows["population_median_baseline"], np.nan)

    ledger = (rows[rows["metric"].isin(scalable.index)]
              .groupby("metric")
              .agg(median_scaled_change=(
                       "change_over_population_median_baseline", "median"),
                   mean_scaled_change=(
                       "change_over_population_median_baseline", "mean"),
                   median_raw_change=("change", "median"),
                   population_median_baseline=(
                       "population_median_baseline", "first"),
                   cells=("identity", "nunique")))
    ledger = ledger.reindex(
        ledger["median_scaled_change"].abs().sort_values().index)
    labels = [semantic_label(name) for name in ledger.index]

    panels = ctx.panels()
    baseline_name = str(rows["baseline_window"].iloc[0])
    comparison_name = str(rows["window"].iloc[0])
    baseline_label = baseline_name.replace("_", " ")
    comparison_label = comparison_name.replace("_", " ")
    spread_count = min(len(ledger), SPREAD_ROWS)
    ridge_panel = ctx.spec.panel("spread")
    minimum_sizes = ({"spread": ridge_panel.grid_minimum(spread_count, 1)}
                     if "spread" in panels else None)
    fig, axes = ctx.layout(
        panels, weights={"ledger": max(1.0, len(ledger) / 8.0)},
        minimum_sizes=minimum_sizes, top_inches=2.5,
        bottom_inches=2.3, gap_inches=1.5)

    if "ledger" in axes:
        annotations = [
            f"{_raw(name, value, ctx)}, {int(count)} cells"
            for name, value, count in zip(
                ledger.index, ledger["median_raw_change"], ledger["cells"])]
        ctx.drew("ledger", common.lollipop(
            axes["ledger"], ledger["median_scaled_change"].to_numpy(float), ctx.theme,
            labels=labels, annotations=annotations))
        axes["ledger"].set_xlabel("Median scaled change across cells")

    leading = str(ledger.index[-1]) if len(ledger) else ""
    if "paired" in axes and leading:
        pair = rows[rows["metric"] == leading].dropna(
            subset=["baseline_value", "value"])
        common.paired_slopes(
            axes["paired"], pair["baseline_value"], pair["value"], ctx.theme,
            left_label=baseline_label,
            right_label=comparison_label,
            role=describe(leading).role,
            highlight=list(pair["change"] > 0),
            y_label=_axis(leading, ctx))
        rose = int((pair["change"] > 0).sum())
        axes["paired"].set_title(
            f"Baseline and comparison per cell: {semantic_label(leading)} "
            f"({rose}/{len(pair)} increased)",
            loc="left", fontsize=ctx.theme.size("panel"), fontweight="bold")

    outside, left_out = 0, 0
    if "spread" in axes and len(ledger):
        shown = list(ledger.index[-SPREAD_ROWS:])
        groups = [rows.loc[
                      rows["metric"] == name,
                      "change_over_population_median_baseline"]
                  .dropna().to_numpy(float) for name in shown]
        # One axis serves every row, so a single heavy-tailed metric would set
        # the range for all of them and flatten the rest into a spike. The
        # range is the middle 98% of everything drawn, and what falls outside
        # is counted rather than quietly clipped into the end bins.
        pooled = np.concatenate([group for group in groups if group.size]) \
            if any(group.size for group in groups) else np.array([0.0, 1.0])
        limits = tuple(np.percentile(pooled, [1, 99]))
        drew = ctx.block(
            fig, axes.pop("spread"), "spread", groups, ctx.theme,
            labels=[semantic_label(name) for name in shown],
            bins=int(ctx.option("bins")), overlap=0.2, labels_outside=True,
            limits=limits,
            x_label="Scaled change for each cell")
        outside = int(drew.data.groupby("group")["outside"].first().sum())
        left_out = len(ledger) - len(shown)

    figure_data = rows[[
        "stem", "condition", "subject", "identity", "window", "baseline_window",
        "metric", "statistic", "baseline_value", "value", "change", "ratio",
        "population_median_baseline",
        "change_over_population_median_baseline", "window_coverage"]]

    subtitle = (
        f"Scaled change for each cell = (comparison-window {STATISTIC} - baseline-window "
        f"{STATISTIC}) / the metric's median baseline across retained cells.\n"
        f"Each mark is the median across {rows['identity'].nunique()} cells; its raw "
        "median change is printed in the metric's own units."
    )
    footnote_lines = [
        f"Median has three separate jobs: each cell is represented by its window "
        f"{STATISTIC}; the denominator is the median baseline across retained cells; "
        "and the ledger mark is the median scaled change across cells. Zero means no "
        "change; +0.2 means an increase equal to 20% of the group's median baseline."
        f" The window tables also carry "
        f"{', '.join(s for s in sorted(set(changes['statistic'])) if s != STATISTIC)}, "
        "which are different quantities and are not drawn."
    ]
    if dropped:
        footnote_lines.append(
            f"{dropped} cell(s) were observed in less than {floor:.0%} of a window and "
            "are absent: a window median taken from a handful of frames moves for "
            "reasons that have nothing to do with the cell. Set --min-coverage to "
            "change where that line sits.")
    if unscalable:
        footnote_lines.append(
            f"{len(unscalable)} metric(s) are in the table but not on the axes "
            f"({', '.join(unscalable)}): their baseline sits at or across zero, and a "
            "using it as the common divisor would be infinite or sign-flipped. "
            "Their raw changes are in figure_data.")
    if outside or left_out:
        parts = []
        if left_out:
            parts.append(f"the bottom panel shows the {len(shown)} largest movers of "
                         f"{len(ledger)}")
        if outside:
            parts.append(f"{outside} cell-metric value(s) fall outside its middle-98% "
                         "range and are not in its histograms")
        footnote_lines.append(f"For readability, {'; '.join(parts)}.")
    footnote_lines.append(
        "Nothing was done to the slice between these windows - they are two stretches "
        "of one recording - so a shift here is drift. Biology, photobleaching and "
        "focus all produce this picture and this page separates none of them.")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=subtitle,
        footnote=ctx.footnote(*footnote_lines),
        title_fields={"window": comparison_label,
                      "baseline_window": baseline_label},
        auxiliary={
            **ctx.provenance_auxiliary(),
            "change_normalisation.csv": ledger.reset_index(),
        },
        console=(f"{len(ledger)} metric(s) ranked, {dropped} cell(s) below "
                 f"{floor:.0%} coverage, {len(unscalable)} unscalable, "
                 f"{outside} outside the spread range; largest mover {leading}"),
    )


def _axis(column: str, ctx: FigureContext) -> str:
    """A label for one metric, with its unit resolved for this run."""
    metric = describe(column)
    unit = metric.unit_text(ctx.interval)
    return f"{metric.label} ({unit})" if unit else metric.label


def _raw(column: str, value: float, ctx: FigureContext) -> str:
    """The change in the metric's own units, for the annotation beside a row."""
    unit = describe(column).unit_text(ctx.interval)
    magnitude = abs(value)
    digits = 3 if magnitude < 10 else (1 if magnitude < 1000 else 0)
    return f"{value:+,.{digits}f}{f' {unit}' if unit else ''}"


if __name__ == "__main__":
    run_figure("change-ledger")
