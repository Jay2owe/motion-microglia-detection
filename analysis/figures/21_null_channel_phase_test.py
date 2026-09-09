"""Figure 21: tracker uncertainty channels through the selected rhythm analysis.

Unclaimed foreground, missing names and gained pixels are put through the same
Circadian Workbench period estimator and the same AR(1) surrogate as the measured
signal. A tracker artefact with an apparent rhythm would show up here; the
comparison is fair because both sides take the identical path.

    python analysis/figures/21_null_channel_phase_test.py <run>
    ... --metrics unclaimed_px,identities_named
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _metrics import role_for, semantic_label
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS, _surrogate

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
        Option("fit_method", default=None),
        Option("significance_method", default=None),
        Option("detrend", default=None),
        Option("detrend_window_hours", default=None),
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
    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    rhythm_params = resolved["params"]
    detrending = {
        name: resolved[name] for name in workbench.DETREND_DEFAULTS
    }
    metrics = [metric for metric in metrics if metric in combined]
    if "corrected_mean" in frame:
        measured = frame.groupby("hours", as_index=False)["corrected_mean"].median()
        combined = combined.merge(measured, on="hours", how="left")
        if "corrected_mean" not in metrics:
            metrics.append("corrected_mean")

    traces = (
        combined[["hours", *metrics]]
        .melt(id_vars="hours", var_name="channel", value_name="value")
        .dropna(subset=["hours", "value"])
    )
    fits = workbench.estimate_grouped_rhythms(
        traces,
        group_columns=["channel"],
        value_column="value",
        params=rhythm_params,
        method=resolved["method"],
        significance_method=resolved["significance_method"],
        detrend=resolved["detrend"],
        detrend_window_hours=resolved["detrend_window_hours"],
        min_observations=resolved["min_observations"],
        correction=resolved["multiple_testing"],
        min_cycles=resolved["min_cycles"],
    )
    generator = np.random.default_rng(20260825)
    rows = []
    for _, fitted in fits.iterrows():
        channel = str(fitted["channel"])
        trace = traces[traces["channel"].eq(channel)].sort_values("hours")
        hours = trace["hours"].to_numpy(float)
        values = trace["value"].to_numpy(float)
        scale = float(np.mean(np.abs(values)))
        amplitude = pd.to_numeric(pd.Series([fitted.get("amplitude")]), errors="coerce").iloc[0]
        relative_amplitude = (
            float(amplitude) / scale
            if np.isfinite(amplitude) and np.isfinite(scale) and scale > 0 else np.nan
        )
        detrended = workbench.detrend_trace(
            hours, values, rhythm_params, method=resolved["detrend"]
        )
        detrended_values = np.asarray(detrended["values"], dtype=float)
        finite = np.isfinite(detrended_values)
        null_amplitudes = []
        for _ in range(50):
            fake = _surrogate(detrended_values[finite], "ar1", generator)
            try:
                null_fit = workbench.estimate_one(
                    hours[finite], fake, rhythm_params, resolved["method"], detrend="none"
                )
            except ValueError:
                continue
            null_amplitude = null_fit.get("amplitude")
            if null_amplitude is not None and np.isfinite(null_amplitude) and scale > 0:
                null_amplitudes.append(float(null_amplitude) / scale)
        surrogate_amplitude = (
            float(np.mean(null_amplitudes)) if null_amplitudes else np.nan
        )
        row = fitted.to_dict()
        row.update({
            "kind": "measurement" if channel == "corrected_mean" else "tracker",
            "relative_fit_amplitude": relative_amplitude,
            "surrogate_amplitude": surrogate_amplitude,
            "excess": relative_amplitude - surrogate_amplitude,
            "surrogate_count": len(null_amplitudes),
        })
        rows.append(row)
    data = pd.DataFrame(rows)
    panels = ctx.panels()
    fig, axes = ctx.layout(
        panels, top_inches=2.35 if "channels" in panels else 1.65)
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
        fig.legend(
            handles=handles, labels=labels, loc="upper center",
            bbox_to_anchor=(position.x0 + position.width / 2,
                            position.y1 + 0.65 / fig.get_figheight()), ncol=3,
            frameon=False, fontsize=ctx.theme.size("caption"),
        )
        axes["channels"].set_title("", loc="left")
    if "against_floor" in axes and not data.empty:
        rhythm_panels.noise_floor(
            axes["against_floor"], data["relative_fit_amplitude"], data["surrogate_amplitude"],
            ctx.theme, labels=[semantic_label(value) for value in data["channel"]],
            x_label="Rhythm amplitude divided by mean signal",
        )
        common.semantic_legend(axes["against_floor"], ctx.theme, location="inside")
    if "amplitudes" in axes and not data.empty:
        rhythm_panels.channel_amplitudes(
            axes["amplitudes"], data["relative_fit_amplitude"], data["surrogate_amplitude"],
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
        subtitle=(
            "Tracker channels and the measured comparison use the same "
            f"{workbench.PERIOD_METHODS[resolved['method']]['label']} estimator and "
            f"{workbench.PERIOD_METHODS[resolved['significance_method']]['label']} test."
        ),
        footnote=(
            f"Circadian Workbench searched {resolved['period_min_hours']:g}-"
            f"{resolved['period_max_hours']:g} h after {resolved['detrend'].replace('_', ' ')} "
            f"detrending. {resolved['multiple_testing']} correction at q < "
            f"{resolved['rhythmic_alpha']:g}; amplitude is divided by each channel's "
            "mean absolute signal. Matched noise uses 50 AR(1) surrogates per channel."
        ),
    )


if __name__ == "__main__":
    run_figure("null-channel-phase-test")
