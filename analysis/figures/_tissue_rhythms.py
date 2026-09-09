"""Circadian Workbench measurements prepared for the existing tissue maps."""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS


@dataclass(frozen=True)
class RhythmMap:
    label: str
    unit: str
    signals: tuple[str, ...]
    role: str = "reporter"
    period: float | None = None
    origin: float = 0.0
    limits: tuple[float, float] | None = None
    cmap: str | None = None


RHYTHM_MAPS = {
    "reporter_period_hours": RhythmMap(
        "Reporter cell period", "Estimated period (h)", ("corrected_mean",)),
    "turnover_period_hours": RhythmMap(
        "Turnover cell period", "Estimated period (h)", ("turnover_index",),
        role="surveillance"),
    "reporter_fitted_relative_amplitude": RhythmMap(
        "Fitted reporter relative amplitude", "Fitted amplitude / mean", ("corrected_mean",)),
    "turnover_fitted_variation": RhythmMap(
        "Turnover fitted variation", "% model-explained variation", ("turnover_index",),
        role="surveillance", limits=(0.0, 100.0)),
}
# A saved selection must never silently change scientific meaning. Daily
# summaries and cross-period phase comparisons need a separately justified
# question; they are not general maps of unknown microglial rhythms.
RETIRED_MAPS = {
    "reporter_m10_onset", "reporter_daily_contrast", "turnover_daily_repeatability",
    "reporter_turnover_phase_difference", "reporter_phase_offset",
    "turnover_active_duration", "reporter_fitted_peak_phase", "reporter_peak_phase",
    "reporter_relative_amplitude", "turnover_rhythm_strength",
}
FITTED_MAPS = {"reporter_fitted_relative_amplitude", "turnover_fitted_variation"}
PERIOD_MAPS = {"reporter_period_hours", "turnover_period_hours"}
DEFAULT_RHYTHM_MAPS = ["reporter_period_hours", "turnover_period_hours"]


def validate_rhythm_maps(requested):
    retired = sorted(set(requested) & RETIRED_MAPS)
    if retired:
        raise ValueError(
            "Retired tissue metrics: " + ", ".join(retired) + ". Microglial periods "
            "are unknown and may differ; these daily or phase comparisons and "
            "ambiguous aliases are no longer available. Select reporter_period_hours, "
            "turnover_period_hours, or an explicitly named fitted diagnostic.")


def analyse_rhythm_maps(ctx, frame: pd.DataFrame, requested: list[str]):
    """Estimate each trace's period through Workbench, without a shared clock.

    Both available signals remain in the correction family when maps change.
    Optional model diagnostics retain their names and are not rhythm strength.
    """
    validate_rhythm_maps(requested)
    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    method = resolved["method"]
    significance_method = resolved["significance_method"]
    min_cycles = resolved["min_cycles"]
    params = resolved["params"]
    required = {signal for name in requested for signal in RHYTHM_MAPS[name].signals}
    # Reporter and turnover maps keep their established joint correction family.
    standard = {"corrected_mean", "turnover_index"}
    candidates = standard if required <= standard else required
    signals = [name for name in dict.fromkeys(
        ("corrected_mean", "turnover_index", *sorted(required)))
        if name in candidates and name in frame]
    missing = required - set(signals)
    if missing:
        raise ValueError("cell_frame.csv needs " + ", ".join(sorted(missing)))
    long = frame[["identity", "hours", *signals]].melt(
        id_vars=["identity", "hours"], value_vars=signals,
        var_name="signal", value_name="value")
    # Do not let infinity or duplicate times alter the model or the time origin.
    long["value"] = pd.to_numeric(long.value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    long["hours"] = pd.to_numeric(long.hours, errors="coerce")
    long = long[np.isfinite(long.hours)].copy()
    if long.empty:
        raise ValueError("cell_frame.csv has no finite recording times")
    if long.duplicated(["identity", "signal", "hours"]).any():
        raise ValueError("rhythm maps require one movie and one observation per cell, signal and time")
    fits = workbench.estimate_grouped_rhythms(
        long, group_columns=["identity", "signal"], value_column="value",
        params=params, method=method, significance_method=significance_method,
        min_observations=resolved["min_observations"],
        correction=resolved["multiple_testing"], min_cycles=min_cycles,
    )
    if fits.empty:
        raise ValueError("no cell traces are available for rhythm maps")
    fits = fits.drop(columns="metric").rename(columns={"signal": "metric"})
    fits["analysis_kind"] = "period_estimation"
    fits["method_label"] = workbench.PERIOD_METHODS[method]["label"]
    fits["analysis_parameters_json"] = json.dumps(params, sort_keys=True)
    fits["minimum_cycles_for_period_map"] = min_cycles
    displayed = fits.significant & fits.period_available & fits.period_hours.gt(0)
    supported = displayed & ~fits.period_underdetermined & ~fits.period_at_search_edge
    fits["period_map_displayed"] = displayed
    fits["period_map_supported"] = supported
    # Retain the established support alias for saved tables and downstream
    # readers; the inclusive display rule is recorded separately above.
    fits["period_map_available"] = supported
    fits["period_map_status"] = np.select(
        [supported, displayed], ["supported period", "exploratory period"],
        default="excluded",
    )
    fits["period_map_exclusion"] = np.select(
        [~fits.period_available, ~fits.significant, fits.period_underdetermined,
         fits.period_at_search_edge, ~fits.period_hours.gt(0)],
        ["no period estimate", "rhythm test not supported", "insufficient observed cycles",
         "period at search boundary", "invalid period"], default="")
    by_signal = {name: group.set_index("identity") for name, group in fits.groupby("metric")}

    values = {}
    for name in requested:
        group = by_signal[RHYTHM_MAPS[name].signals[0]]
        if name in PERIOD_MAPS:
            result = group.period_hours.where(group.period_map_displayed)
        elif name == "reporter_fitted_relative_amplitude":
            amplitude = pd.to_numeric(group.get("amplitude", pd.Series(np.nan, index=group.index)), errors="coerce")
            mean = pd.to_numeric(group.input_mean, errors="coerce")
            result = (amplitude / mean).where(group.period_available & mean.gt(0) & amplitude.ge(0))
        elif name == "turnover_fitted_variation":
            result = pd.to_numeric(group.get("goodness_of_fit", pd.Series(np.nan, index=group.index)), errors="coerce")
            # Only a method explicitly reporting explained variance may use %.
            diagnostics = group.get("estimate_diagnostics", pd.Series(None, index=group.index, dtype=object))
            is_r2 = diagnostics.map(
                lambda d: "r-squared" in str(d.get("goodness_of_fit_is", "")).lower()
                if isinstance(d, dict) else False)
            result = 100.0 * result.where(group.period_available & is_r2 & result.between(0, 1))
        result = result.where(np.isfinite(result))
        result.name = name
        values[name] = result
    return values, fits, pd.DataFrame(), params
