"""Figure 21: tracker uncertainty channels through the rhythm test.

Unclaimed foreground, missing names and gained pixels are put through the same
cosinor and the same AR(1) surrogate as the measured signal. A tracker artefact
with a 24 h period would show up here, and the comparison is only fair because
both sides take the identical path.

    python analysis/figures/21_null_channel_phase_test.py <run>
    ... --metrics unclaimed_px,identities_named
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _derive import _test_channel
from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from analysis.modules.rhythms import cosinor

from panels import common
from panels import rhythms as rhythm_panels


@figure(
    number=21,
    slug="null-channel-phase-test",
    summary="tracker uncertainty channels through the rhythm test",
    title="Tracker uncertainty channels through the rhythm test",
    reads=(Table("cell_frame.csv", module="rhythms"),
           Table("motion_evidence_frame.csv", module="motion_evidence"),
           Table("presence_frame.csv", module="presence")),
    panels=(
        Panel("channels", common.trace, title="Signals through time"),
        Panel("against_floor", rhythm_panels.noise_floor,
              title="Observed rhythm versus matched noise"),
        Panel("amplitudes", rhythm_panels.channel_amplitudes,
              title="Rhythm strength by signal"),
    ),
    options=(
        Option("metrics",
               default=["unclaimed_px", "unclaimed_fraction", "identities_named",
                        "identities_expected", "frame_gained_share"],
               help="which tracker channels are tested beside the measurement"),
        Option("hour_ticks", default=24.0),
    ),
    grammar="trace stack and paired intervals",
)
def build(ctx: FigureContext) -> FigureResult:
    presence = ctx.table("presence_frame.csv")
    evidence = ctx.table("motion_evidence_frame.csv")
    frame = ctx.table("cell_frame.csv")
    combined = presence.merge(evidence, left_on="frame_index", right_on="from_frame_index",
                              how="outer", suffixes=("", "_evidence"))
    if "hours" not in combined and "hours_evidence" in combined:
        combined["hours"] = combined["hours_evidence"]
    metrics = ctx.option("metrics")
    metrics = [metric for metric in metrics if metric in combined]
    if "corrected_mean" in frame:
        measured = frame.groupby("hours", as_index=False)["corrected_mean"].median()
        combined = combined.merge(measured, on="hours", how="left")
        metrics.append("corrected_mean")
    period = 24.0
    generator = np.random.default_rng(20260825)
    rows = []
    for channel in metrics:
        tested = _test_channel(combined["hours"].to_numpy(float), combined[channel].to_numpy(float),
                               period, generator)
        if not tested:
            continue
        rows.append({
            "channel": channel,
            "kind": "measurement" if channel == "corrected_mean" else "tracker",
            "amplitude": tested["cosinor_amplitude"],
            "cosinor_relative_amplitude": tested["cosinor_relative_amplitude"],
            "peak_hour": tested["cosinor_peak_hour"],
            "p_value": tested["cosinor_p_value"],
            "surrogate_amplitude": tested["surrogate_amplitude"],
            "excess": tested["cosinor_relative_amplitude"] - tested["surrogate_amplitude"],
            "observations": int(combined[channel].notna().sum()),
        })
    data = pd.DataFrame(rows)
    panels = ctx.panels()
    fig, axes = ctx.layout(panels)
    if "channels" in axes:
        line_styles = ("solid", (0, (6, 3)), (0, (2, 2)), (0, (8, 2, 2, 2)))
        for channel_index, channel in enumerate(metrics):
            values = combined[channel].to_numpy(float)
            spread = float(np.nanstd(values)) or 1.0
            standardised = (values - np.nanmean(values)) / spread
            common.trace(axes["channels"], combined["hours"], standardised, ctx.theme,
                         role=role_for(channel), label=semantic_label(channel),
                         hour_ticks=ctx.hour_ticks,
                         y_label="Within-signal change\n(standard deviations)")
            axes["channels"].lines[-1].set_linestyle(line_styles[channel_index % len(line_styles)])
        handles, labels = axes["channels"].get_legend_handles_labels()
        axes["channels"]._semantic_legend_handled = True
        position = axes["channels"].get_position()
        axes["channels"].set_position([
            position.x0, position.y0, position.width, max(position.height - 0.035, 0.05)
        ])
        fig.legend(
            handles=handles, labels=labels, loc="upper center",
            bbox_to_anchor=(0.60, 0.905), ncol=3,
            frameon=False, fontsize=ctx.theme.size("caption"),
        )
        axes["channels"].set_title("", loc="left")
    if "against_floor" in axes and not data.empty:
        rhythm_panels.noise_floor(
            axes["against_floor"], data["cosinor_relative_amplitude"], data["surrogate_amplitude"],
            ctx.theme, labels=[semantic_label(value) for value in data["channel"]],
            x_label="Rhythm amplitude divided by mean signal",
        )
        common.semantic_legend(axes["against_floor"], ctx.theme, location="inside")
    if "amplitudes" in axes and not data.empty:
        rhythm_panels.channel_amplitudes(
            axes["amplitudes"], data["cosinor_relative_amplitude"], data["surrogate_amplitude"],
            ctx.theme, labels=[semantic_label(value) for value in data["channel"]], kinds=data["kind"],
        )
        common.semantic_legend(
            axes["amplitudes"], ctx.theme,
            handles=[
                Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=ctx.theme.colour("invalid"),
                       linestyle="none"),
                Line2D([], [], marker="o", color=ctx.theme.colour("invalid"), linestyle="none"),
            ], labels=["Matched noise", "Observed signal"], location="inside",
        )
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=data,
        subtitle="Tracker channels and the measured comparison use the same 24 h cosinor and AR(1) surrogate path.",
    )


if __name__ == "__main__":
    run_figure("null-channel-phase-test")
