"""Figure 37: which measured signals carry a daily rhythm.

The rhythms module fits every metric family, not only the reporter, and this is
the page that says which of those fits was worth doing. Relative amplitude is
the swing between the most-active ten hours of the day and the least-active
five, scaled by their sum: 0 is a signal with no daily swing at all, 1 is one
whose quiet period holds nothing.

The other two measures are about shape rather than size. Interdaily stability
asks how alike one day is to the next - a signal can swing hard and still be a
different swing every day. Intradaily variability asks how broken up the daily
pattern is, hour to hour. A strong, stable, unbroken rhythm sits high on the
first and low on the last.

    python analysis/figures/37_rhythm_strength.py <run>
    ... --metrics area_px,corrected_mean     draw a subset of the fitted signals
    ... --bins 40                            bands in the stacked histogram
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _bundle import units_note
from _metrics import semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common


@figure(
    number=37,
    slug="rhythm-strength",
    summary="relative rhythm amplitude and cycle consistency across measured signals",
    title="Relative rhythm amplitude and cycle consistency across {n_metrics} signals",
    grammar="stacked amplitude histograms, stability scatter and ranked medians",
    reads=(Table("rhythms.csv", module="rhythms"),),
    panels=(
        Panel("strength", common.stacked_histogram, block=True,
              item_width_inches=11.0, item_height_inches=1.0,
              item_gap_inches=0.0,
              title="Relative amplitude across cells"),
        Panel("stability", common.scatter,
              title="Interdaily stability against intradaily variability"),
        Panel("ranking", common.lollipop,
              title="Median relative amplitude per signal"),
    ),
    options=(
        Option("metrics", default=[],
               help="which fitted signals to draw, comma separated; empty draws every one"),
        Option("bins", default=24),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    rhythms = ctx.table("rhythms.csv")
    available = sorted(str(value) for value in rhythms["metric"].dropna().unique())
    wanted = [str(name) for name in ctx.option("metrics")] or available
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise SystemExit(
            f"--metrics {','.join(unknown)} was not fitted in this run. "
            f"rhythms.csv holds: {', '.join(available)}"
        )

    # Ordered by median amplitude so the stacked histogram and lollipop agree, and
    # the signal most worth fitting is the top row of both. Ordering each panel
    # on its own would put the same seven names in two different orders on one
    # page, which reads as two different sets.
    strength = (rhythms[rhythms["metric"].isin(wanted)]
                .groupby("metric")["relative_amplitude"]
                .median().sort_values(ascending=False))
    metrics = list(strength.index)
    labels = [semantic_label(name) for name in metrics]
    colours = common.series_colours(ctx.theme, len(metrics))
    by_metric = {name: rhythms[rhythms["metric"] == name] for name in metrics}

    # Interdaily stability compares one day with the next, so it needs several
    # of them. The module's own decision is to write the number and flag it
    # rather than withhold it - "two days is one comparison, which is
    # computable and weak" - so this page draws every point and carries the
    # flag, rather than dropping flagged rows and leaving a gap that reads as
    # missing data.
    #
    # The flag is drawn as an encoding only when it varies. On a recording where
    # every fit is flagged, fading all of them conveys nothing a reader can
    # decode; the sentence under the panel is what does the work.
    selected = rhythms[rhythms["metric"].isin(metrics)]
    underdetermined = selected["stability_underdetermined"].fillna(False).astype(bool)
    held_back = int(underdetermined.sum())
    uniform_flag = held_back in (0, len(selected))

    figure_data = pd.concat(
        [frame.assign(
            drawn_in_stability=~frame["stability_underdetermined"].fillna(False).astype(bool)
            & frame["interdaily_stability"].notna()
            & frame["intradaily_variability"].notna())
         [["metric", "identity", "relative_amplitude", "interdaily_stability",
           "intradaily_variability", "stability_underdetermined", "days_covered",
           "observations", "span_hours", "drawn_in_stability"]]
         for frame in by_metric.values()],
        ignore_index=True)

    panels = ctx.panels()
    ridge_panel = ctx.spec.panel("strength")
    minimum_sizes = ({"strength": ridge_panel.grid_minimum(len(metrics), 1)}
                     if "strength" in panels else None)
    fig, axes = ctx.layout(
        panels, minimum_sizes=minimum_sizes, gap_inches=1.7,
    )

    if "strength" in axes:
        groups = [by_metric[name]["relative_amplitude"].dropna().to_numpy(float)
                  for name in metrics]
        # Little overlap, because there are seven rows rather than the three or
        # four a stacked histogram usually carries. At the drawer's default each row is
        # taller than the gap to the next and every label lands on its
        # neighbour's ridge.
        ctx.block(fig, axes.pop("strength"), "strength", groups, ctx.theme,
                  labels=labels, bins=int(ctx.option("bins")), overlap=0.2,
                  labels_outside=True,
                  x_label="Relative amplitude (busiest ten hours against quietest five)")

    if "stability" in axes:
        axis = axes["stability"]
        for name, label, colour in zip(metrics, labels, colours):
            frame = by_metric[name].dropna(
                subset=["interdaily_stability", "intradaily_variability"])
            if frame.empty:
                continue
            flagged = frame["stability_underdetermined"].fillna(False).astype(bool)
            for subset, alpha in ((frame[~flagged], 0.55),
                                  (frame[flagged], 0.55 if uniform_flag else 0.18)):
                if subset.empty:
                    continue
                common.scatter(axis, subset["interdaily_stability"],
                               subset["intradaily_variability"], ctx.theme,
                               colours=colour, label=label, alpha=alpha)
        axis.set_xlabel("Interdaily stability (how alike one day is to the next)")
        axis.set_ylabel("Intradaily variability\n(how broken up the day is)")
        handles = [Line2D([], [], marker="o", linestyle="none", color=colour)
                   for colour in colours]
        keys = list(labels)
        if not uniform_flag:
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  color=ctx.theme.colour("caption"), alpha=0.18))
            keys.append("Too few days for stability")
        # Beside the panel rather than on it. Seven series over a cloud this
        # dense leave no empty corner, so an inside key covers the points it is
        # there to explain.
        common.semantic_legend(axis, ctx.theme, handles=handles, labels=keys,
                               location="right", columns=1)

    if "ranking" in axes:
        # Median on the stem, mean in the annotation. Neither is the obvious
        # right answer for a bounded, skewed quantity across ninety-odd cells,
        # so the page shows both rather than making the choice silently.
        annotations = []
        for name in metrics:
            frame = by_metric[name]["relative_amplitude"].dropna()
            annotations.append(f"mean {frame.mean():.3f}, {len(frame)} cells")
        common.lollipop(axes["ranking"], strength.to_numpy(float), ctx.theme,
                        labels=labels, colours=colours, annotations=annotations)
        axes["ranking"].set_xlabel("Median relative amplitude across cells")

    cells = int(rhythms.loc[rhythms["metric"].isin(metrics), "identity"].nunique())
    subtitle = (
        f"One point per cell per signal, {cells} cells across {len(metrics)} fitted "
        f"{'signals' if len(metrics) != 1 else 'signal'}. Rows are ordered by median "
        "relative amplitude, so the same order holds on every panel."
    )
    footnote_lines = []
    if held_back:
        days = float(selected["days_covered"].max())
        where = ("Every fit on this page" if held_back == len(selected)
                 else f"{held_back} of {len(selected)} fits")
        shown = ("they are drawn anyway, because the number exists and its ordering "
                 "is still informative" if uniform_flag else "they are the faded points")
        footnote_lines.append(
            f"{where} is flagged as having too little recording behind its interdaily "
            f"stability: the recording covers {days:.2f} days and that measure compares "
            f"one day with the next. The value is computable and weak, and {shown}. "
            "Read the middle panel's horizontal axis as a ranking rather than as a "
            "measurement."
        )
    footnote_lines.append(
        "Relative amplitude, interdaily stability and intradaily variability are all "
        "ratios and carry no unit; the amplitude and stability of two signals are "
        "comparable, their underlying activity levels are not."
    )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=subtitle,
        footnote=ctx.footnote(*footnote_lines),
        title_fields={"n_metrics": len(metrics)},
        auxiliary=ctx.provenance_auxiliary(),
        console=(f"{len(metrics)} signals, {cells} cells, "
                 f"{held_back} of {len(selected)} stability fits underdetermined"),
    )


if __name__ == "__main__":
    run_figure("rhythm-strength")
