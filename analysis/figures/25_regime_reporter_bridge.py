"""Figure 25: reporter signal inside and outside one shape regime.

Every cell is its own control: the same identity's signal while it is in the
selected state, against its signal while it is not. The regime chosen by
default is the largest-area one, which is stated on the figure.

    python analysis/figures/25_regime_reporter_bridge.py <run>
    ... --regime 2               contrast a different state
    ... --window 12              frames either side of entry
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _derive import _regime_labels, _transition_curves
from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure

from panels import common
from panels import intensity as intensity_panels


@figure(
    number=25,
    slug="regime-reporter-bridge",
    summary="reporter signal inside and outside one shape regime",
    title="Reporter signal inside and outside one shape regime",
    reads=(Table("cell_frame.csv", module="intensity"),
           Table("regime_profiles.csv", module="regimes")),
    panels=(
        Panel("paired", common.paired_slopes,
              title="Signal inside versus outside the selected state"),
        Panel("triggered", common.event_average,
              title="Signal around entry into the selected state"),
        Panel("plane", intensity_panels.texture_plane, block=True,
              title="Relationship between reporter measurements"),
    ),
    options=(
        Option("metrics", default=["punctateness", "corrected_mean"]),
        Option("regime", default=None, metavar="N",
               help="which state to contrast; unset takes the largest-area one"),
        Option("window", default=6),
        Option("bins", default=24),
    ),
    grammar="paired slopes event average and texture plane",
)
def build(ctx: FigureContext) -> FigureResult:
    frame = ctx.table("cell_frame.csv")
    profiles = ctx.table("regime_profiles.csv")
    regime_names = _regime_labels(profiles)
    # The state and the reporter signal are columns of one table now, so
    # this figure no longer joins them; it drops the cell-frames the regime
    # fit had nothing to say about, which is what the join used to do.
    joined = frame.dropna(subset=["regime"]).copy()
    joined["regime"] = joined["regime"].astype(int)
    metrics = ctx.option("metrics")
    metrics = [metric for metric in metrics if metric in joined]
    if not metrics:
        raise SystemExit("none of --metrics is in cell_frame.csv")
    default_regime = int(profiles.sort_values("area_px")["regime"].iloc[-1]) if "area_px" in profiles else int(joined["regime"].max())
    selected_regime = int(ctx.option_or("regime", default_regime))
    rows = []
    for metric in metrics:
        for identity, group in joined.groupby("identity", sort=True):
            inside = group.loc[group["regime"] == selected_regime, metric].dropna()
            outside = group.loc[group["regime"] != selected_regime, metric].dropna()
            if inside.empty or outside.empty:
                continue
            rows.append({"identity": int(identity), "metric": metric,
                         "in_regime_mean": float(inside.mean()), "out_regime_mean": float(outside.mean()),
                         "difference": float(inside.mean() - outside.mean()),
                         "frames_in": int(len(inside)), "frames_out": int(len(outside))})
    figure_data = pd.DataFrame(rows)
    window = int(ctx.option("window"))
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "paired" in axes and not figure_data.empty:
        first = figure_data[figure_data["metric"] == metrics[0]]
        common.paired_slopes(axes["paired"], first["out_regime_mean"], first["in_regime_mean"],
                             ctx.theme,
                             left_label="Other states",
                             right_label=regime_names.get(selected_regime, f"State {selected_regime}"),
                             role="reporter", y_label=semantic_label(metrics[0]))
    triggered_rows = []
    if "triggered" in axes:
        for metric in metrics:
            offsets, curves = _transition_curves(joined, metric, window)
            _, null_curves = _transition_curves(joined, metric, window, random=True, seed=20260825)
            if curves.size:
                result = common.event_average(axes["triggered"], offsets, curves, ctx.theme,
                                              role=role_for(metric), bootstrap=200,
                                              null_curves=null_curves if null_curves.size else None,
                                              null_label="Random-event 95% range",
                                              label=semantic_label(metric)).data
                result.insert(0, "metric", metric)
                triggered_rows.append(result)
    if "plane" in axes:
        rect = tuple(axes["plane"].get_position().bounds)
        axes["plane"].remove()
        drawn = intensity_panels.texture_plane(
            fig, rect, joined[metrics[1] if len(metrics) > 1 else metrics[0]], joined[metrics[0]],
            ctx.theme, bins=int(ctx.option("bins")),
            level_label=semantic_label(metrics[1] if len(metrics) > 1 else metrics[0]),
            texture_label=semantic_label(metrics[0]),
        )
        ax, top, right = drawn.axes
        plane = drawn.data
        axes["plane"] = ax
    triggered = pd.concat(triggered_rows, ignore_index=True) if triggered_rows else pd.DataFrame()
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        subtitle=f"Within-cell contrast of {regime_names.get(selected_regime, f'State {selected_regime}').lower()}; transition window {window} frames.",
        auxiliary={"transition_triggered.csv": triggered},
    )


if __name__ == "__main__":
    run_figure("regime-reporter-bridge")
