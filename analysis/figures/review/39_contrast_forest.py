"""Figure 39: every comparison the configuration declared, and every refusal.

The answer sheet. One row per declared contrast per metric: the effect, the
interval around it where there is one, and whether it survived the correction
applied within its family.

Refusals are on the page too, as sentences rather than as gaps. A contrast the
package declined to run - too few units a side, a group with nobody in it -
writes a row saying why, and a figure that filtered those out would make a
comparison that was asked and refused look exactly like one nobody asked.

Nothing here is computed. Every number is read from ``statistics.csv``,
including which rows are significant, so this page cannot quietly disagree with
the file it draws.

    python analysis/figures/39_contrast_forest.py <run>
    ... --order effect          rows largest effect first
    ... --metrics area_px_median
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from _metrics import describe, documented
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common

#: Row orders this page offers. ``family`` is the default because a family is
#: the set a correction was applied within, so keeping families together is
#: what makes the corrected p-values on the page readable as a group.
ORDERS = {
    "family": "declaration order, families kept together",
    "effect": "largest effect first",
    "significance": "surviving comparisons first",
}


@figure(
    number=39,
    slug="contrast-forest",
    purpose="review",
    summary="every declared comparison, its effect, and every refusal",
    title="Declared comparisons: {n_drawn} tested, {n_refused} refused",
    grammar="forest plot with correction scatter and unit ledger",
    reads=(Table("statistics.csv", module="contrasts", scope="run", optional=True),),
    panels=(
        Panel("effects", common.forest_blocks, block=True,
              title="Effect with interval, one block per scale"),
        Panel("correction", common.scatter,
              title="Raw p-value against false-discovery-rate-adjusted q-value"),
        Panel("design", common.lollipop, title="Units behind each comparison"),
    ),
    options=(
        Option("metrics", default=[],
               help="which tested metrics to draw, comma separated; empty draws every one"),
        Option("order", default="family",
               help="row order: family, effect or significance"),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    statistics = ctx.optional_table("statistics.csv")
    if statistics is None or statistics.empty:
        return _nothing_declared(ctx, absent=statistics is None)

    order = str(ctx.option("order"))
    if order not in ORDERS:
        raise SystemExit(
            f"--order {order} is not one this page has. It takes: "
            + "; ".join(f"{name} ({meaning})" for name, meaning in ORDERS.items()))

    rows = statistics.copy()
    wanted = [str(name) for name in ctx.option("metrics")]
    if wanted:
        available = sorted(str(value) for value in rows["metric"].dropna().unique())
        unknown = [name for name in wanted if name not in available]
        if unknown:
            raise SystemExit(
                f"--metrics {','.join(unknown)} was not tested in this run. "
                f"statistics.csv holds: {', '.join(available)}")
        rows = rows[rows["metric"].isin(wanted)]

    # A refusal is a row with no p-value. The declared test is still on it -
    # ``contrasts._blank_row`` keeps the contrast, family, test, alpha and
    # correction and blanks only what would be a measurement - so keying on a
    # missing test name would classify every refusal as a result. What was not
    # computed is the p-value, and that is what says so.
    #
    # It is not a missing row and it is not a failure; it is the package
    # declining to compute something the design could not support, recorded so
    # it can be seen.
    refused = rows["p_value"].isna()
    rows["refusal"] = np.where(
        refused, rows["note"].fillna("Refused, with no reason recorded"), "")
    rows["is_refusal"] = refused
    rows["label"] = rows["contrast"].astype(str) + " - " + rows["metric"].astype(str)
    rows["significant"] = rows["significant"].fillna(False).astype(bool)

    # What an effect is measured in decides which rows may share a ruler, and
    # three things decide that: the kind of effect, the unit of the column
    # tested, and - where the grouping column is itself a measurement name -
    # the unit of that measurement. A standardised difference and a median
    # difference are not points on one scale; neither are a median difference
    # of 6.75 px and one of 1300 camera units; neither are a slope of an area
    # and a slope of a brightness, both of which read "per h". Rows are blocked
    # by all three, each block on its own axis, rather than the page picking a
    # winner and hiding the rest.
    interval = ctx.interval
    rows["scale"] = [
        _scale(kind, metric, interval, group_by=group_by, group=group)
        for kind, metric, group_by, group
        in zip(rows["effect_kind"].fillna(""), rows["metric"].fillna(""),
               rows["group_by"].fillna(""), rows["group_a"].fillna(""))]
    tested = rows[~rows["is_refusal"]]

    # A refusal has no effect kind, so it has no scale of its own. It joins the
    # other rows of its contrast, which is where a reader looks for it.
    by_contrast = dict(zip(tested["contrast"], tested["scale"]))
    fallback = next(iter(tested["scale"]), "no effect recorded")
    rows.loc[rows["is_refusal"], "scale"] = [
        by_contrast.get(name, fallback)
        for name in rows.loc[rows["is_refusal"], "contrast"]]

    sort_key = {"family": ["scale", "family", "contrast", "metric"],
                "effect": ["scale", "effect"],
                "significance": ["scale", "significant", "effect"]}[order]
    ascending = order == "family"
    drawable = rows.sort_values(sort_key, ascending=ascending, kind="stable")

    panels = ctx.panels()
    n_scales = drawable["scale"].nunique()
    fig, axes = ctx.layout(
        panels, weights={"effects": max(1.0, (len(drawable) + 1.6 * n_scales) / 3.0)})

    if "effects" in axes:
        ctx.block(fig, axes.pop("effects"), "effects",
                  list(drawable["effect"]), ctx.theme,
                  labels=list(drawable["label"]), scales=list(drawable["scale"]),
                  lo=list(drawable["effect_lo"]), hi=list(drawable["effect_hi"]),
                  significant=list(drawable["significant"]),
                  refusals=list(drawable["refusal"]),
                  x_labels={scale: scale for scale in drawable["scale"]})

    if "correction" in axes:
        axis = axes["correction"]
        common.scatter(axis, tested["p_value"], tested["p_corrected"], ctx.theme,
                       role="rhythmic", alpha=0.7)
        limit = float(np.nanmax([tested["p_corrected"].max(), 1.0]))
        axis.plot([0, limit], [0, limit], color=ctx.theme.colour("reference"),
                  linewidth=ctx.theme.stroke("guide"), zorder=1)
        axis.set_xlabel("Uncorrected p-value")
        axis.set_ylabel("Corrected p-value")

    if "design" in axes:
        units = tested.dropna(subset=["n_a"])
        counts = pd.concat([units["n_a"], units["n_b"]]).groupby(
            list(units["label"]) * 2).min() if not units.empty else pd.Series(dtype=float)
        counts = counts.reindex(list(dict.fromkeys(units["label"]))).dropna()
        if not counts.empty:
            common.lollipop(axes["design"], counts.to_numpy(float), ctx.theme,
                            labels=list(counts.index))
            axes["design"].axvline(
                MIN_UNITS, color=ctx.theme.colour("invalid"),
                linewidth=ctx.theme.stroke("guide"),
                label=f"{MIN_UNITS}, the fewest this package will test")
            axes["design"].set_xlabel(
                f"Units in the smaller group ({_units(tested)})")
            common.semantic_legend(axes["design"], ctx.theme, location="above")

    figure_data = rows[[
        "contrast", "family", "metric", "unit", "aggregate", "group_a", "group_b",
        "n_a", "n_b", "test", "statistic", "p_value", "effect", "effect_kind",
        "effect_lo", "effect_hi", "correction", "p_corrected", "alpha",
        "significant", "refusal"]]

    n_refused = int(rows["is_refusal"].sum())
    n_drawn = int(len(rows) - n_refused)
    subtitle = (
        f"One row per declared comparison per metric, ordered by "
        f"{ORDERS[order]}. Points and intervals are read from statistics.csv; "
        f"nothing on this page is recomputed."
    )
    footnote_lines = [
        f"Corrections applied within a family, not across the page: "
        f"{_corrections(rows)}. A comparison is marked as surviving using the "
        f"significant column of the file, not by re-thresholding its p-value."
    ]
    if n_refused:
        footnote_lines.append(
            f"{n_refused} comparison{'s' if n_refused != 1 else ''} on this page "
            "carries a sentence instead of a point: the package declined to run it "
            "and recorded why. A refusal is a result about the design, not a "
            "missing measurement.")
    scales = list(dict.fromkeys(drawable["scale"]))
    if len(scales) > 1:
        footnote_lines.append(
            f"The comparisons fall into {len(scales)} scales and each has its own "
            "axis. Distances are comparable only within a block: an effect of 6.75 "
            "in pixels and one of 1300 in camera units are not one bigger than the "
            "other. Colour says whether a comparison survived its correction - "
            f"filled dark where it did, grey where it did not, at alpha {_alpha(rows)}.")
    else:
        footnote_lines.append(
            "Colour says whether a comparison survived its correction: filled dark "
            f"where it did, grey where it did not, at alpha {_alpha(rows)}.")
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=subtitle,
        footnote=ctx.footnote(*footnote_lines),
        title_fields={"n_drawn": n_drawn, "n_refused": n_refused},
        console=(f"{n_drawn} comparison(s) drawn, {n_refused} refused, "
                 f"across {len(scales)} scale(s)"),
    )


#: The floor the contrasts step applies before it will test anything. Restated
#: here rather than imported so that drawing a figure does not pull the
#: measurement package in; ``test_figure_schema`` checks the two still agree.
MIN_UNITS = 3

#: How each effect kind should be labelled on an axis. A kind with no entry
#: falls back to its own name, which is worse than a sentence and better than a
#: guess at what it means.
EFFECT_LABELS = {
    "median_difference": "Difference in medians, second group minus first",
    "median_paired_difference": "Median of the per-unit differences",
    "mean_difference": "Difference in means, second group minus first",
    "hedges_g": "Standardised difference (Hedges g)",
    "median_against_zero": "Median across units, against zero",
    "hedges_g_paired": "Standardised per-unit difference (Hedges g)",
    "epsilon_squared": "Share of the variation explained (epsilon squared)",
    "eta_squared": "Share of the variation explained (eta squared)",
}


def _effect_label(kind: str) -> str:
    return EFFECT_LABELS.get(kind, kind.replace("_", " ").capitalize() or "Effect")


def _scale(kind: str, metric: str, interval_minutes: float | None = None,
           group_by: str = "", group: str = "") -> str:
    """The axis this row may be drawn on, as a sentence.

    Three things decide it. The kind says what the number is - a difference of
    medians, a standardised difference. The unit of the tested column says what
    that number is counted in. And where the grouping column is itself a
    measurement name, that measurement's unit is the other half of the count.

    The third is not a nicety. A long table puts the measurement in a column:
    `trend` and `window_change` both carry a `metric` column, so a contrast on
    one of them tests a single column - a slope, a change - across groups that
    are themselves measurements. The tested column then reads "per h" for every
    row, and a slope of an area sits on the same axis as a slope of a
    brightness five orders of magnitude away, where the smaller of them is
    drawn as a dot on the no-effect line and reads as no effect.
    """
    if not kind:
        return "no effect recorded"
    # unit_text, not unit: a unit may name the frame interval as a placeholder,
    # and an axis reading "per {interval}" is a bug the reader has to decode.
    unit = describe(str(metric)).unit_text(interval_minutes)
    measured = (describe(str(group)).unit_text(interval_minutes)
                if group_by == "metric" and documented(str(group)) else "")
    counted = chr(32).join(part for part in (measured, unit) if part)
    return f"{_effect_label(kind)}{chr(32)}({counted})" if counted else _effect_label(kind)


def _alpha(rows: pd.DataFrame) -> str:
    values = sorted(set(rows["alpha"].dropna()))
    return ", ".join(f"{value:g}" for value in values) or "unset"


def _corrections(rows: pd.DataFrame) -> str:
    pairs = rows.dropna(subset=["correction"]).groupby("correction")["family"].nunique()
    return ", ".join(f"{name} across {count} famil{'y' if count == 1 else 'ies'}"
                     for name, count in pairs.items()) or "none declared"


def _units(rows: pd.DataFrame) -> str:
    return ", ".join(sorted(set(rows["unit"].dropna()))) or "unset"


def _nothing_declared(ctx: FigureContext, *, absent: bool) -> FigureResult:
    """A page for a run whose configuration declared no comparisons.

    Deliberately a page rather than a refusal. Every other figure stops when a
    table is missing, and that is right for a table a module would have written
    if it had run - the fix is to switch the module on. Nothing was switched
    off here: a run with one movie and one condition has nothing to compare,
    which is a fact about the experiment. Stopping would also make
    ``build_all`` report a failed build on every honest single-group run.
    """
    # The claim is about a file that is not there, so the evidence is the one
    # that records the configuration which did not ask for it. Without a traced
    # source the bundle is not reproducible and ReproFig refuses to write it -
    # correctly, because a figure nothing can be checked against is a picture.
    ctx.record_source("manifest.json", ctx.run / "manifest.json")
    fig = ctx.sheet(13.8, 4.4)
    axis = fig.add_axes([0.14, 0.22, 0.8, 0.5])
    axis.set_axis_off()
    reason = ("This run has no statistics.csv: the analysis configuration "
              "declared no contrasts." if absent else
              "This run declared contrasts, but none produced a row.")
    axis.text(0.0, 0.6, reason, transform=axis.transAxes, va="center", ha="left",
              fontsize=ctx.theme.size("subtitle"), color=ctx.theme.colour("ink"))
    axis.text(0.0, 0.25,
              "Add a 'contrasts' block naming a table, its metrics, the column to "
              "group by,\nthe unit of replication and a test, then re-run.",
              transform=axis.transAxes, va="center", ha="left",
              fontsize=ctx.theme.size("note"), color=ctx.theme.colour("caption"))
    empty = pd.DataFrame(columns=[
        "contrast", "family", "metric", "unit", "aggregate", "group_a", "group_b",
        "n_a", "n_b", "test", "statistic", "p_value", "effect", "effect_kind",
        "effect_lo", "effect_hi", "correction", "p_corrected", "alpha",
        "significant", "refusal"])
    return FigureResult(
        figure=fig, axes=[axis], figure_data=empty,
        subtitle="No comparison was declared, so none was made.",
        footnote=ctx.footnote(
            "An empty page here is a statement about the configuration, not a "
            "failure. Nothing was switched off and no measurement is missing."),
        title_fields={"n_drawn": 0, "n_refused": 0},
        console="no contrasts declared for this run",
    )


if __name__ == "__main__":
    run_figure("contrast-forest")
