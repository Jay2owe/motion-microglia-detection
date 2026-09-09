"""Figure 18: peak position and fitted oscillation amplitude.

Each arrow is one cell called rhythmic by the configured primary test. Its
angle is the fitted peak within that cell's own detected period and its length
is the fitted midline-to-peak change in the measurement's original units.

    python analysis/figures/18_clock_face.py <run>
    ... --metrics area_px        a different fitted measurement
    ... --panels amplitude       the amplitude histogram alone
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from _metrics import describe
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import common
from panels import rhythms as rhythm_panels


CYCLE_TICKS = (0.0, 0.25, 0.5, 0.75)
CYCLE_TICK_LABELS = ("0 (start)", "1/4", "1/2", "3/4")

METHOD_LABELS = {
    "lomb": "Lomb-Scargle periodogram",
    "chi_square": "Enright-Sokolove periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}


def _fallback_rows(fits: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Recover the primary Lomb-Scargle result from runs made before method rows."""
    source = fits.copy()
    period = source.get("best_period_hours", source.get("lombscargle_period_hours"))
    phase = source.get("best_phase_hours", source.get("free_cosinor_peak_hour"))
    p_value = source.get("best_p_value", source.get("lombscargle_false_alarm"))
    rhythmic = source.get("rhythmic", source.get("rhythmic_lombscargle", False))
    amplitude = source.get("free_cosinor_amplitude", source.get("cosinor_amplitude"))
    if period is None or phase is None or amplitude is None:
        raise SystemExit(
            "rhythms.csv has no detected-period phase and amplitude for the clock face"
        )

    period = pd.to_numeric(period, errors="coerce")
    phase = pd.to_numeric(phase, errors="coerce")
    rhythmic = pd.Series(rhythmic, index=source.index).fillna(False).astype(bool)
    estimator = str(params.get(
        "period_estimation_method", params.get("primary_rhythm_test", "lomb")))
    significance = str(params.get("primary_rhythm_test", "lomb"))
    default_label = METHOD_LABELS.get(estimator, estimator)
    method_label = source.get("best_method_label", pd.Series(default_label, index=source.index))
    alpha = source.get("best_alpha", pd.Series(float(params.get("alpha", 0.05)), index=source.index))
    status = source.get(
        "rhythm_status",
        pd.Series(np.where(rhythmic, "rhythmic", "arrhythmic"), index=source.index),
    )
    return pd.DataFrame({
        "identity": source["identity"],
        "metric": source["metric"],
        "method": estimator,
        "method_label": method_label,
        "significance_method": significance,
        "significance_method_label": METHOD_LABELS.get(significance, significance),
        "rhythm_status": status,
        "rhythmic": rhythmic,
        "period_hours": period,
        "phase_hours": phase,
        "phase_fraction": np.mod(phase, period) / period,
        "amplitude": pd.to_numeric(amplitude, errors="coerce"),
        "p_value": pd.to_numeric(p_value, errors="coerce") if p_value is not None else np.nan,
        "alpha": pd.to_numeric(alpha, errors="coerce"),
    })


def _clock_rows(
    fits: pd.DataFrame,
    methods: pd.DataFrame | None,
    metric: str,
    params: dict,
) -> pd.DataFrame:
    """One explicit primary-test row per cell, with phase in hours and cycles."""
    if methods is None:
        return _fallback_rows(fits, params)
    estimator_flag = methods.get(
        "period_estimator", pd.Series(False, index=methods.index)).fillna(False)
    estimated = methods[(methods["metric"] == metric) & estimator_flag].copy()
    primary = methods[(methods["metric"] == metric) & methods["primary"].fillna(False)].copy()
    if estimated.empty or primary.empty:
        return _fallback_rows(fits, params)
    verdict = primary[[
        "identity", "metric", "method", "method_label", "method_rhythm_status",
        "method_rhythmic", "p_value", "alpha",
    ]].rename(columns={
        "method": "significance_method",
        "method_label": "significance_method_label",
        "method_rhythm_status": "rhythm_status",
        "method_rhythmic": "rhythmic",
        "p_value": "significance_p_value",
        "alpha": "significance_alpha",
    })
    selected = estimated[[
        "identity", "metric", "method", "method_label", "period_hours",
        "phase_hours", "amplitude",
    ]].merge(verdict, on=["identity", "metric"], how="inner", validate="one_to_one")
    period = pd.to_numeric(selected["period_hours"], errors="coerce")
    phase = pd.to_numeric(selected["phase_hours"], errors="coerce")
    selected["phase_fraction"] = np.mod(phase, period) / period
    selected["p_value"] = selected.pop("significance_p_value")
    selected["alpha"] = selected.pop("significance_alpha")
    return selected


def _label_cycle_axis(ax) -> None:
    ax.set_xticks(2 * np.pi * np.asarray(CYCLE_TICKS))
    ax.set_xticklabels(CYCLE_TICK_LABELS)


@figure(
    number=18,
    slug="clock-face",
    summary="peak position within each detected cycle and fitted oscillation amplitude",
    title="{metric}: peak position and fitted oscillation amplitude",
    reads=(Table("rhythms.csv", module="rhythms"),
           Table("rhythm_methods.csv", module="rhythms", optional=True)),
    panels=(
        Panel("dial", rhythm_panels.phase_dial, polar=True,
              min_width_inches=6.5, min_height_inches=6.5,
              title="Each arrow is one rhythmic cell\nAngle = peak position; length = fitted amplitude"),
        Panel("rose", common.rose, polar=True,
              min_width_inches=6.5, min_height_inches=6.5,
              title="When rhythmic cells peak\nBar length = cells per phase bin"),
        Panel("amplitude", common.histogram,
              min_width_inches=12.5, min_height_inches=4.8,
              title="How large the fitted oscillations are"),
    ),
    options=(
        Option("metrics", default="corrected_mean", cast=str, metavar="COL",
               help="which fitted measurement the clock draws"),
        Option("bins", default=12),
    ),
    grammar="unit-labelled phase vectors, circular counts, and amplitude histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    metric = ctx.option("metrics")
    rhythms = ctx.table("rhythms.csv")
    methods = ctx.optional_table("rhythm_methods.csv")
    params = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    fits = rhythms[rhythms["metric"] == metric].copy()
    if fits.empty:
        available = ", ".join(sorted(set(rhythms["metric"])))
        raise SystemExit(f"--metrics {metric} was not fitted; available: {available}")

    figure_data = _clock_rows(fits, methods, metric, params)
    for column in ("period_hours", "phase_hours", "phase_fraction", "amplitude",
                   "p_value", "alpha"):
        figure_data[column] = pd.to_numeric(figure_data[column], errors="coerce")
    figure_data["rhythmic"] = figure_data["rhythmic"].fillna(False).astype(bool)
    figure_data["included_in_plot"] = (
        figure_data["rhythmic"]
        & np.isfinite(figure_data["phase_fraction"])
        & np.isfinite(figure_data["amplitude"])
    )
    plotted = figure_data[figure_data["included_in_plot"]].copy()

    panels = ctx.panels()
    if panels.keys == ["dial", "rose", "amplitude"]:
        fig, axes = ctx.grid_layout(
            panels,
            {
                "dial": (0, 0, 1, 1),
                "rose": (0, 1, 1, 1),
                "amplitude": (1, 0, 1, 2),
            },
            top_inches=2.8,
            bottom_inches=1.85,
            horizontal_gap_inches=2.0,
            vertical_gap_inches=1.35,
        )
    else:
        fig, axes = ctx.layout(panels, gap_inches=1.25)

    metric_info = describe(metric)
    unit = metric_info.unit_text(ctx.interval)
    amplitude_label = "Fitted oscillation amplitude" + (f" ({unit})" if unit else "")

    if "dial" in axes:
        rhythm_panels.phase_dial(
            axes["dial"], plotted["phase_fraction"], plotted["amplitude"], ctx.theme,
            passed=np.ones(len(plotted), dtype=bool), period_hours=1.0,
            resultant=False, radius_label=amplitude_label,
            tick_values=CYCLE_TICKS, tick_labels=CYCLE_TICK_LABELS,
        )
        axes["dial"].set_rlabel_position(135)
        axes["dial"].yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axes["dial"].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes["dial"]._semantic_legend_handled = True

    if "rose" in axes:
        common.rose(
            axes["rose"], plotted["phase_fraction"], ctx.theme,
            bins=int(ctx.option("bins")), period=1.0, role="reporter", mean_vector=False,
        )
        _label_cycle_axis(axes["rose"])
        axes["rose"].set_rlabel_position(135)
        axes["rose"].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
        axes["rose"].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes["rose"]._semantic_legend_handled = True

    if "amplitude" in axes:
        common.histogram(
            axes["amplitude"], plotted["amplitude"], ctx.theme,
            bins=int(ctx.option("bins")), role="reporter",
            x_label=amplitude_label, y_label="Rhythmic cells",
        )

    tested = len(figure_data)
    rhythmic = int(figure_data["rhythmic"].sum())
    method = (str(figure_data["method"].dropna().mode().iloc[0])
              if figure_data["method"].notna().any() else "")
    method_label = METHOD_LABELS.get(
        method,
        str(figure_data["method_label"].dropna().mode().iloc[0])
        if figure_data["method_label"].notna().any() else "the selected period estimator",
    )
    significance_label = (
        str(figure_data["significance_method_label"].dropna().mode().iloc[0])
        if "significance_method_label" in figure_data
        and figure_data["significance_method_label"].notna().any()
        else "the configured rhythm test"
    )
    alpha = figure_data["alpha"].dropna()
    threshold = f"p <= {float(alpha.iloc[0]):g}" if not alpha.empty else "its configured threshold"
    if plotted.empty:
        subtitle = f"0 of {tested} cells were rhythmic by {significance_label} ({threshold})."
        period_sentence = "No rhythmic-cell period was available to draw."
    else:
        low = float(plotted["period_hours"].min())
        high = float(plotted["period_hours"].max())
        subtitle = (
            f"{rhythmic} of {tested} cells were rhythmic by {significance_label} ({threshold}); "
            f"{method_label} supplied each cell's {low:g}-{high:g} h period and peak."
        )
        period_sentence = "Exact detected periods and phases in hours are in figure_data.csv."

    statistics = figure_data[[
        "identity", "metric", "method", "method_label", "significance_method",
        "significance_method_label", "rhythm_status", "rhythmic",
        "period_hours", "phase_hours", "p_value", "alpha",
    ]].copy()
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=figure_data,
        auxiliary={"statistics.csv": statistics},
        subtitle=subtitle,
        footnote=(
            "The dial starts at the top and runs clockwise through one detected cycle. "
            f"Amplitude is the fitted midline-to-peak change (half peak-to-trough)"
            f"{f' in {unit}' if unit else ''}. {period_sentence}"
        ),
        title_fields={"metric": metric_info.label},
    )


if __name__ == "__main__":
    run_figure("clock-face")
