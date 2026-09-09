"""Does each cell cycle, which tests detect it, and at what period?

Circadian Workbench runs every requested period method independently.  The
default Lomb-Scargle, chi-square and F periodograms all search the configured
period range and each writes its own period, p-value, threshold and
rhythmic/arrhythmic/unknown verdict to ``rhythm_methods.csv``. Lomb-Scargle is
the default overall verdict because it tolerates missing frames; the user can
choose another significance-bearing method. Cosinor remains available for
amplitude and phase description, but it no longer decides which cells are
rhythmic.

An explicitly enabled **daily profile block** adds Circadian Workbench's M10,
L5, onset, offset, interdaily stability and intradaily variability measures.
It is off by default because these measures fold the trace into a 24-hour day;
they answer a daily question rather than estimating an unknown period.

None of this is specific to brightness. The module fits whatever ``metrics``
names, so cell size, branching, footprint turnover and step size get the same
readouts as the reporter does.

Honest limit for this dataset: 99 frames at 30 minutes is 49.5 hours, which is
just over two cycles of a 24-hour rhythm. Two cycles can show that something
oscillates; they cannot pin the period tightly. Every row therefore carries
``cycles_covered``, and a fit with fewer than three cycles is flagged so it can
never be quoted as a precise period without the caveat travelling with it. The
same limit bites harder on interdaily stability, which compares one day with
the next and has one comparison to work with here: ``days_covered`` and
``stability_underdetermined`` are what say so.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from analysis import circadian as workbench
from analysis.registry import Column, MeasurementContext, Output, register_derived

DEFAULTS = {
    "metrics": [
        "area_px",
        "corrected_mean",
        "solidity",
        "ramification_index",
        "turnover_index",
        "step_px",
    ],
    # One shared contract spans ultradian, circadian-like and infradian
    # candidates. Period estimation remains independent of the rhythmicity
    # test, so fit-only estimators such as FFT-NLLS and MESA are selectable.
    **workbench.PERIOD_ANALYSIS_DEFAULTS,
    **workbench.DETREND_DEFAULTS,
    # Display vocabulary for a continuous period result.  These boundaries do
    # not affect the test or its p-value; they only name the returned candidate.
    "circadian_band_hours": [20.0, 28.0],
    # Surrogate control. Slowly drifting noise passes a cosinor fit far more
    # often than a p-value suggests, so the package measures its own false
    # positive rate rather than assuming it. "ar1" builds surrogates with the
    # same variance and the same frame-to-frame correlation as the real trace
    # but no rhythm; "shuffle" destroys that correlation and is therefore too
    # easy to beat. Set to 0 to skip.
    "null_surrogates_per_cell": 5,
    "null_model": "ar1",
    "null_seed": 20260824,
    # Legacy fixed-period cosinor columns are available only when explicitly
    # requested. They never supply the selected period or rhythm verdict.
    "descriptive_cosinor": False,
    # --- optional fixed-day profile block -------------------------------
    # This is a separate scientific question from free period estimation.
    "daily_profile_measures": False,
    # Window lengths are the actigraphy conventions and are settings rather
    # than constants: a 30-minute frame interval makes M10 twenty frames, and
    # a dataset imaged every 5 minutes would make it 120.
    "bin_hours": 1.0,               # the folded day's resolution
    "high_window_hours": 10.0,      # M10
    "low_window_hours": 5.0,        # L5
    # ClockLab's onset template: how much quiet has to sit before a candidate
    # hour and how much activity after it, and where the line between the two
    # is drawn. 80 means the threshold sits at the 20th percentile.
    "onset_off_hours": 6.0,
    "onset_on_hours": 6.0,
    "onset_threshold_percent": 80.0,
    "min_days_for_stability": 3.0,
}


def _detrend(
    hours: np.ndarray,
    values: np.ndarray,
    method: str,
    window_hours: float = float(workbench.DETREND_DEFAULTS["detrend_window_hours"]),
    detrend_options: dict | None = None,
) -> np.ndarray:
    """Compatibility helper; Circadian Workbench owns every implementation."""
    result = workbench.detrend_trace(
        hours,
        values,
        {**workbench.DETREND_DEFAULTS, **dict(detrend_options or {})},
        method=method,
        window_hours=window_hours,
    )
    return np.asarray(result["values"], dtype=float)


def _surrogate(values: np.ndarray, model: str, generator: np.random.Generator) -> np.ndarray:
    """A trace with the same statistics as this cell but no rhythm."""
    if model == "shuffle":
        return generator.permutation(values)
    if model == "ar1":
        centred = values - values.mean()
        if len(centred) < 3 or np.allclose(centred, 0):
            return generator.permutation(values)
        denominator = float(np.sum(centred[:-1] ** 2))
        phi = float(np.sum(centred[1:] * centred[:-1]) / denominator) if denominator > 0 else 0.0
        phi = float(np.clip(phi, -0.98, 0.98))
        sigma = float(np.std(centred) * np.sqrt(max(1.0 - phi ** 2, 1e-6)))
        out = np.empty(len(centred))
        out[0] = generator.normal(0.0, np.std(centred))
        noise = generator.normal(0.0, sigma, len(centred) - 1)
        for i in range(1, len(centred)):
            out[i] = phi * out[i - 1] + noise[i - 1]
        return out + values.mean()
    raise ValueError(f"unknown null model {model!r}")


# Public compatibility names retained for existing figure builders. Every
# calculation crosses the Circadian Workbench gateway.
cosinor = workbench.cosinor
nonparametric = workbench.nonparametric
rayleigh = workbench.phase_summary

DAILY_PROFILE_DEFAULTS = {
    "m10": np.nan,
    "m10_onset_hour": np.nan,
    "l5": np.nan,
    "l5_onset_hour": np.nan,
    "relative_amplitude": np.nan,
    "onset_hour": np.nan,
    "offset_hour": np.nan,
    "active_duration_hours": np.nan,
    "interdaily_stability": np.nan,
    "intradaily_variability": np.nan,
    "days_covered": np.nan,
    "hours_binned": np.nan,
    "stability_underdetermined": None,
    "onset_found": None,
}


#: Three tables. The wide one is long by metric, not wide by metric - one row
#: per cell per measurement - so these column names are fixed rather than
#: data-dependent, and every one of them can be declared.
#:
#: Two prefixes carry the whole argument of the module and the labels have to
#: keep them apart. ``cosinor_`` is the fit at the period the analysis was told
#: to assume, usually 24 h; ``free_cosinor_`` is the fit at the period the
#: selected estimator found. Quoting one while meaning the other is the
#: easiest mistake to make with this table, which is why no label here says
#: "amplitude" without saying which fit it came from.
#:
#: **Three amplitudes, and they are three different numbers.**
#:
#: ===============================  ==================================================
#: ``relative_amplitude``           ``(M10 - L5) / (M10 + L5)``: the busiest ten hours
#:                                  of the average day against the quietest five. What
#:                                  the field means by the term, and no fit involved.
#: ``cosinor_relative_amplitude``   fitted amplitude over the mean, at the period the
#:                                  fit was told to assume.
#: ``free_cosinor_relative_amplitude``  the same ratio at the period the selected
#:                                     estimator found.
#: ===============================  ==================================================
#:
#: The labels have to keep them apart on an axis as well as in this schema, so
#: none of the three is labelled "relative amplitude": each says what it divides.
PRODUCES = (
    # rhythms - one row per cell per measurement
    Column("metric", "Which measurement this row is about", "column name", "reference"),
    Column("observations", "Frames the fit used", "frames", "reference"),
    Column("span_hours", "Time from first to last frame used", "h", "reference"),
    Column("cycles_covered", "Cycles the recording covers at the assumed period", "count", "reference"),
    Column("mean_level", "Mean of the trace before detrending", "", "reference"),
    Column("detrend", "How the trend was removed", "name", "reference"),
    Column("detrend_window_hours", "Baseline window used for detrending", "h", "reference"),
    Column("detrended_z", "Detrended trace scaled by its own standard deviation", "SD", "rhythmic"),
    Column("workbench_version", "Circadian Workbench version", "version", "reference"),
    Column("primary_rhythm_test", "Test used for the overall rhythm verdict", "method", "reference"),
    Column("period_estimation_method", "Method used to estimate period and phase", "method", "reference"),
    Column("rhythm_status", "Overall rhythmic, arrhythmic or unknown verdict", "status", "rhythmic"),
    Column("rhythmic", "Called rhythmic by the primary test", "0 or 1", "rhythmic"),
    Column("best_method_label", "Display name of the selected period estimator", "name", "reference"),
    Column("best_period_hours", "Best period from the selected estimator", "h", "rhythmic"),
    Column("best_period_error_hours", "Uncertainty on the selected period estimate", "h", "reference"),
    Column("best_p_value", "Significance from the primary rhythm test", "p", "significant"),
    Column("best_alpha", "Significance threshold of the primary rhythm test", "p", "reference"),
    Column("best_goodness_of_fit", "Fit strength reported by the selected estimator", "", "fit"),
    Column("best_phase_hours", "Peak phase from the selected period estimator", "h from start", "rhythmic"),
    Column("best_phase_fraction", "Peak position within the detected period", "0-1 cycle", "rhythmic"),
    Column("best_period_class", "Detected period band: ultradian, circadian-like or infradian", "class", "reference"),
    Column("best_period_at_search_edge", "Detected period touches a configured search boundary", "0 or 1", "invalid"),
    Column("cycles_covered_at_best_period", "Cycles covered at the selected period", "count", "reference"),
    Column("daily_profile_measures_enabled", "Whether fixed 24-hour profile measures were requested", "0 or 1", "reference"),
    Column("fixed_period_hours", "Period the fit was told to assume", "h", "reference"),
    Column("cosinor_mesor", "Fitted mean at the assumed period", "", "rhythmic"),
    Column("cosinor_amplitude", "Fitted amplitude at the assumed period", "", "rhythmic"),
    Column("cosinor_relative_amplitude", "Fitted amplitude over the mean, assumed period", "ratio", "rhythmic"),
    Column("cosinor_peak_hour", "Time of day the fit peaks, assumed period", "h", "rhythmic"),
    Column("cosinor_phase_convention", "Convention used to encode the fitted peak time", "convention", "reference"),
    Column("cosinor_r_squared", "Variation the assumed-period fit explains", "fraction", "fit"),
    Column("cosinor_p_value", "Chance of the assumed-period fit from noise", "p", "significant"),
    Column("cosinor_relative_amplitude_error", "Uncertainty on the relative amplitude", "ratio", "reference"),
    Column("lombscargle_period_hours", "Period carrying the most power", "h", "rhythmic"),
    Column("lombscargle_power", "Power at that period", "", "rhythmic"),
    Column("lombscargle_false_alarm", "Chance of that power from noise", "p", "significant"),
    Column("free_cosinor_mesor", "Fitted mean at the found period", "", "rhythmic"),
    Column("free_cosinor_amplitude", "Fitted amplitude at the found period", "", "rhythmic"),
    Column("free_cosinor_relative_amplitude", "Fitted amplitude over the mean, found period", "ratio", "rhythmic"),
    Column("free_cosinor_peak_hour", "Time of day the fit peaks, found period", "h", "rhythmic"),
    Column("free_cosinor_phase_convention", "Convention used to encode the found-period peak time", "convention", "reference"),
    Column("free_cosinor_r_squared", "Variation the found-period fit explains", "fraction", "fit"),
    Column("free_cosinor_p_value", "Chance of the found-period fit from noise", "p", "significant"),
    Column("free_cosinor_relative_amplitude_error", "Uncertainty on the found-period relative amplitude", "ratio", "reference"),
    Column("cycles_covered_at_free_period", "Cycles the recording covers at the found period", "count", "reference"),
    Column("rhythmic_cosinor", "Called rhythmic by the cosinor fit", "0 or 1", "rhythmic"),
    Column("rhythmic_lombscargle", "Called rhythmic by Lomb-Scargle", "0 or 1", "rhythmic"),
    Column("rhythmic_both", "Called rhythmic by both tests", "0 or 1", "rhythmic"),
    Column("period_underdetermined", "Too few cycles to pin the period", "0 or 1", "invalid"),
    # rhythm_methods - one row per cell, measurement and configured method.
    Column("method", "Circadian Workbench period or rhythm method", "method", "reference"),
    Column("method_label", "Published method name", "name", "reference"),
    Column("is_significance_test", "Whether this method can make a rhythm verdict", "0 or 1", "reference"),
    Column("method_rhythm_status", "This method's rhythmic, arrhythmic or unknown verdict", "status", "rhythmic"),
    Column("method_rhythmic", "Called rhythmic by this method", "0 or 1", "rhythmic"),
    Column("primary", "Whether this method supplies the overall verdict", "0 or 1", "reference"),
    Column("period_estimator", "Whether this method supplies period and phase", "0 or 1", "reference"),
    Column("alpha", "Significance threshold used by this method", "p", "reference"),
    Column("period_hours", "Period estimated by this method", "h", "rhythmic"),
    Column("period_error_hours", "Uncertainty on this method's period", "h", "reference"),
    Column("phase_hours", "Peak phase estimated by this method", "h from start", "rhythmic"),
    Column("phase_error_hours", "Uncertainty on this method's phase", "h", "reference"),
    Column("amplitude", "Amplitude estimated by this method", "", "rhythmic"),
    Column("amplitude_error", "Uncertainty on this method's amplitude", "", "reference"),
    Column("rae", "Relative amplitude error from this method", "ratio", "reference"),
    Column("goodness_of_fit", "Fit or periodogram strength reported by this method", "", "fit"),
    Column("p_value", "Significance reported by this method", "p", "significant"),
    Column("significant", "Circadian Workbench significance verdict", "0 or 1", "rhythmic"),
    Column("status", "Whether this method returned an estimate", "status", "reference"),
    Column("phase_reference", "Origin used for phase", "name", "reference"),
    Column("phase_units", "Units used for phase", "name", "reference"),
    Column("workbench_run_record_json", "Completed Workbench inputs, source identity and replay conditions", "JSON", "reference"),
    # The non-parametric block, on the same row as the fits it sits beside.
    #
    # Every hour here is hours *since the recording started*, not a time of day,
    # which is why the unit says so rather than saying "h". This package has no
    # clock: `cell_frame` carries hours since the first frame and the
    # configuration holds no recording start time. The arithmetic is unaffected
    # and the meaning is: 7.5 is "seven and a half hours in", and two movies
    # begun at different times of day cannot have their onsets compared at all.
    Column("m10", "Busiest ten hours", "", "rhythmic"),
    Column("m10_onset_hour", "Start of the busiest stretch", "h from start", "rhythmic"),
    Column("l5", "Quietest five hours", "", "rhythmic"),
    Column("l5_onset_hour", "Start of the quietest stretch", "h from start", "rhythmic"),
    Column("relative_amplitude", "Busiest against quietest", "ratio", "rhythmic"),
    Column("onset_hour", "Active phase starts", "h from start", "rhythmic"),
    Column("offset_hour", "Active phase ends", "h from start", "rhythmic"),
    Column("active_duration_hours", "Active phase length", "h", "rhythmic"),
    Column("interdaily_stability", "How alike one day is to the next", "ratio", "rhythmic"),
    Column("intradaily_variability", "How broken up the daily pattern is", "ratio", "rhythmic"),
    Column("days_covered", "Days of recording behind these", "days", "reference"),
    Column("hours_binned", "Hourly bins with a measurement in them", "bins", "reference"),
    Column("stability_underdetermined",
           "Too few days for the stability measure", "0 or 1", "reference"),
    Column("onset_found", "A template match was found", "0 or 1", "reference"),
    # rhythms_population - one row per measurement per group of cells
    Column("population", "Which group of cells", "name", "reference"),
    Column("cells", "Cells in the group", "count", "reference"),
    Column("vector_length", "How tightly the peaks agree", "0-1", "rhythmic"),
    Column("mean_peak_hour", "Mean peak fraction expressed at the group's median selected period", "h", "rhythmic"),
    Column("mean_peak_fraction", "Circular mean peak position within each selected cycle", "0-1 cycle", "rhythmic"),
    Column("phase_reference_period_hours", "Median selected period used to express the mean peak in hours", "h", "reference"),
    Column("rayleigh_p_value", "Chance of that agreement from scattered peaks", "p", "significant"),
    # rhythms_null - one row per measurement
    Column("surrogates", "Noise traces tested per cell", "count", "reference"),
    Column("null_model", "How the noise traces were made", "name", "reference"),
    Column("cells_tested", "Cells the null was run on", "count", "reference"),
    Column("false_positive_rate_cosinor", "Noise traces the cosinor test called rhythmic", "fraction", "arrhythmic"),
    Column("false_positive_rate_lombscargle", "Noise traces Lomb-Scargle called rhythmic", "fraction", "arrhythmic"),
    Column("false_positive_rate_both", "Noise traces both tests called rhythmic", "fraction", "arrhythmic"),
    Column("observed_rate_cosinor", "Real cells the cosinor test called rhythmic", "fraction", "rhythmic"),
    Column("observed_rate_both", "Real cells both tests called rhythmic", "fraction", "rhythmic"),
    Column("excess_over_null_both", "How far the real rate beats matched noise", "fraction", "significant"),
    Column("primary_method", "Rhythm test used for the null comparison", "method", "reference"),
    Column("false_positive_rate_primary", "Surrogates called rhythmic by the primary test", "fraction", "arrhythmic"),
    Column("observed_rate_primary", "Real cells called rhythmic by the primary test", "fraction", "rhythmic"),
    Column("excess_over_null_primary", "How far the primary-test rate beats matched noise", "fraction", "significant"),
)


#: The two population tables are optional because they only exist when there
#: was something to summarise: no cell fitted, no population row.
WRITES = (
    Output("rhythms",            grain=("identity", "metric")),
    Output("rhythm_methods",     grain=("identity", "metric", "method")),
    Output("rhythm_traces",      grain=("identity", "metric", "hours")),
    Output("rhythms_population", grain=("metric", "population"), optional=True),
    Output("rhythms_null",       grain=("metric",),              optional=True),
)


@register_derived(
    name="rhythms",
    description="Circadian Workbench rhythm tests and period estimates per cell and measurement",
    needs_columns=("identity", "frame_index", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("rhythms")}
    metrics = [m for m in params["metrics"] if m in cell_frame.columns]
    fixed_period = float(params["fixed_period_hours"])
    min_observations = int(params["min_observations"])
    primary_method = str(params["primary_rhythm_test"])
    estimator_method = str(params.get("period_estimation_method", primary_method))
    configured_methods = list(params["period_methods"])
    for required_method in (estimator_method, primary_method):
        if required_method not in configured_methods:
            configured_methods.append(required_method)
    params["period_methods"] = configured_methods
    minimum_cycles = float(params["min_cycles_for_confident_period"])
    search_low, search_high = map(float, params["period_search_hours"])
    circadian_low, circadian_high = map(float, params["circadian_band_hours"])
    if not search_low < search_high:
        raise ValueError("period_search_hours needs a lower value than a higher value")
    if not circadian_low < circadian_high:
        raise ValueError("circadian_band_hours needs a lower value than a higher value")

    generator = np.random.default_rng(int(params["null_seed"]))
    surrogates_per_cell = int(params["null_surrogates_per_cell"])
    alpha_for_null = float(params["rhythmic_alpha"])
    descriptive_cosinor = bool(params.get("descriptive_cosinor", False))
    daily_profile_measures = bool(params.get("daily_profile_measures", False))

    rows: list[dict] = []
    method_rows: list[dict] = []
    trace_rows: list[dict] = []
    null_rows: list[dict] = []
    for metric in metrics:
        for identity, group in cell_frame.groupby("identity", sort=True):
            usable = group[["hours", metric]].dropna().sort_values("hours")
            if len(usable) < min_observations:
                continue
            hours = usable["hours"].to_numpy(float)
            values = usable[metric].to_numpy(float)
            if np.allclose(values, values[0]):
                continue
            span = float(hours.max() - hours.min())

            analysed = workbench.estimate_trace(hours, values, params)
            evidence = analysed["rows"]
            by_method = {entry["method"]: entry for entry in evidence}
            primary = by_method[primary_method]
            estimate = by_method[estimator_method]
            best_period = estimate.get("period_hours")
            best_period = (float(best_period) if best_period is not None and
                           np.isfinite(best_period) else np.nan)
            best_p = primary.get("p_value")
            best_p = (float(best_p) if best_p is not None and
                      np.isfinite(best_p) else np.nan)
            best_error = estimate.get("period_error_hours")
            best_error = (float(best_error) if best_error is not None and
                          np.isfinite(best_error) else np.nan)
            best_phase = estimate.get("phase_hours")
            best_phase = (float(best_phase) if best_phase is not None and
                          np.isfinite(best_phase) else np.nan)
            best_phase_fraction = (
                float((best_phase % best_period) / best_period)
                if np.isfinite(best_phase) and np.isfinite(best_period)
                and best_period > 0 else np.nan
            )
            best_period_class = (
                "unknown" if not np.isfinite(best_period)
                else "ultradian" if best_period < circadian_low
                else "circadian-like" if best_period <= circadian_high
                else "infradian"
            )
            edge_tolerance = max(0.1, 0.005 * (search_high - search_low))
            best_period_at_search_edge = bool(
                np.isfinite(best_period)
                and (best_period <= search_low + edge_tolerance
                     or best_period >= search_high - edge_tolerance)
            )
            cycles_at_best = span / best_period if np.isfinite(best_period) else np.nan
            best_goodness = estimate.get("goodness_of_fit")
            best_goodness = (
                float(best_goodness) if best_goodness is not None and
                np.isfinite(best_goodness) else np.nan)

            for entry in evidence:
                method_entry = dict(entry)
                method_rhythmic = method_entry.pop("rhythmic")
                method_rhythm_status = method_entry.pop("rhythm_status")
                method_period = entry.get("period_hours")
                method_period = (
                    float(method_period) if method_period is not None and
                    np.isfinite(method_period) else np.nan)
                method_rows.append({
                    "identity": int(identity),
                    "metric": metric,
                    "observations": int(len(usable)),
                    "span_hours": span,
                    "cycles_covered": (span / method_period
                                       if np.isfinite(method_period) else np.nan),
                    "period_underdetermined": (
                        not np.isfinite(method_period) or
                        span / method_period < minimum_cycles),
                    "method_rhythmic": method_rhythmic,
                    "method_rhythm_status": method_rhythm_status,
                    "period_estimator": entry["method"] == estimator_method,
                    "workbench_run_record_json": json.dumps(analysed["run_record"], allow_nan=False),
                    **method_entry,
                })

            detrend_result = workbench.detrend_trace(hours, values, params)
            detrended = np.asarray(detrend_result["values"], dtype=float)
            detrended_z = workbench.scale_detrended(detrended, values)
            finite = np.isfinite(detrended)
            fitted_hours = hours[finite]
            fitted_values = detrended[finite]
            detrend_name = str(detrend_result["method"])
            trace_rows.extend(
                {
                    "identity": int(identity),
                    "metric": metric,
                    "hours": float(hour),
                    "detrended_z": float(scaled),
                    "detrend": detrend_name,
                    "detrend_window_hours": float(params["detrend_window_hours"]),
                    "workbench_version": workbench.WORKBENCH_VERSION,
                }
                for hour, scaled in zip(hours, detrended_z)
                if np.isfinite(scaled)
            )

            for replicate in range(surrogates_per_cell):
                if len(fitted_values) < min_observations:
                    break
                fake = _surrogate(fitted_values, params["null_model"], generator)
                primary_null = workbench.estimate_one(
                    fitted_hours, fake, params, primary_method, detrend="none")
                lomb_null = (
                    primary_null if primary_method == "lomb"
                    else workbench.estimate_one(
                        fitted_hours, fake, params, "lomb", detrend="none"
                    ) if "lomb" in configured_methods else {}
                )
                null_cosinor = (
                    cosinor(fitted_hours, fake, fixed_period)
                    if descriptive_cosinor else {}
                )
                if primary_null.get("status") != "ok":
                    continue
                null_rows.append(
                    {
                        "metric": metric,
                        "identity": int(identity),
                        "replicate": replicate,
                        "null_model": params["null_model"],
                        "primary_method": primary_method,
                        "primary_p_value": primary_null.get("p_value", np.nan),
                        "rhythmic_primary": primary_null.get("rhythmic"),
                        "cosinor_p_value": null_cosinor.get("cosinor_p_value", np.nan),
                        "lombscargle_false_alarm": lomb_null.get("p_value", np.nan),
                        "lombscargle_period_hours": lomb_null.get("period_hours", np.nan),
                    }
                )

            row = {
                "identity": int(identity),
                "metric": metric,
                "observations": int(len(usable)),
                "span_hours": span,
                "cycles_covered": (
                    span / fixed_period
                    if descriptive_cosinor or daily_profile_measures else np.nan
                ),
                "mean_level": float(values.mean()),
                "detrend": detrend_name,
                "detrend_window_hours": float(params["detrend_window_hours"]),
                "workbench_version": workbench.WORKBENCH_VERSION,
                "primary_rhythm_test": primary_method,
                "period_estimation_method": estimator_method,
                "rhythm_status": primary["rhythm_status"],
                "rhythmic": primary["rhythmic"],
                "best_method_label": estimate["method_label"],
                "best_period_hours": best_period,
                "best_period_error_hours": best_error,
                "best_p_value": best_p,
                "best_alpha": float(primary["alpha"]),
                "best_goodness_of_fit": best_goodness,
                "best_phase_hours": best_phase,
                "best_phase_fraction": best_phase_fraction,
                "best_period_class": best_period_class,
                "best_period_at_search_edge": best_period_at_search_edge,
                "cycles_covered_at_best_period": cycles_at_best,
                "daily_profile_measures_enabled": daily_profile_measures,
                "fixed_period_hours": (
                    fixed_period
                    if descriptive_cosinor or daily_profile_measures else np.nan
                ),
                **DAILY_PROFILE_DEFAULTS,
            }
            reference = float(np.mean(np.abs(values)))
            if daily_profile_measures:
                # On the measured trace because L5 and M10 are level summaries.
                row.update(nonparametric(hours, values, params))
            if descriptive_cosinor:
                row.update(cosinor(
                    fitted_hours, fitted_values, fixed_period, reference
                ))
            lomb = by_method.get("lomb")
            if lomb is not None:
                row.update({
                    "lombscargle_period_hours": lomb.get("period_hours", np.nan),
                    "lombscargle_power": lomb.get("goodness_of_fit", np.nan),
                    "lombscargle_false_alarm": lomb.get("p_value", np.nan),
                })
            if descriptive_cosinor and np.isfinite(best_period):
                row.update(
                    {
                        f"free_{key}": value
                        for key, value in cosinor(
                            fitted_hours, fitted_values, best_period, reference
                        ).items()
                    }
                )
                row["cycles_covered_at_free_period"] = cycles_at_best
            cosinor_p = row.get("cosinor_p_value", np.nan)
            row["rhythmic_cosinor"] = (
                bool(cosinor_p < alpha_for_null) if np.isfinite(cosinor_p) else None
            )
            row["rhythmic_lombscargle"] = (
                bool(lomb.get("rhythmic"))
                if lomb is not None and lomb.get("rhythmic") is not None else None
            )
            # Compatibility only. No current result uses this pair as the
            # overall verdict; ``rhythmic`` is the selected workbench test.
            row["rhythmic_both"] = (
                bool(row["rhythmic_cosinor"] and row["rhythmic_lombscargle"])
                if row["rhythmic_cosinor"] is not None
                and row["rhythmic_lombscargle"] is not None else None
            )
            row["period_underdetermined"] = (
                not np.isfinite(cycles_at_best) or cycles_at_best < minimum_cycles)
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "rhythms": table,
            "rhythm_methods": pd.DataFrame(method_rows),
            "rhythm_traces": pd.DataFrame(trace_rows),
        }

    population_rows: list[dict] = []
    for metric, group in table.groupby("metric"):
        rhythmic_group = group[group["rhythmic"].fillna(False).astype(bool)]
        for label, subset in (
            ("all_tested", group),
            ("rhythmic_by_primary_test", rhythmic_group),
            ("rhythmic_by_both_tests", group[
                group["rhythmic_both"].map(
                    lambda value: bool(value) if pd.notna(value) else False
                )
            ]),
        ):
            phase_fraction = pd.to_numeric(
                subset["best_phase_fraction"], errors="coerce"
            ).to_numpy(float)
            result = rayleigh(phase_fraction, 1.0)
            if result:
                mean_fraction = result.pop("mean_peak_hour")
                reference_period = float(pd.to_numeric(
                    subset["best_period_hours"], errors="coerce"
                ).median())
                population_rows.append({
                    "metric": metric,
                    "population": label,
                    **result,
                    "mean_peak_fraction": mean_fraction,
                    "phase_reference_period_hours": reference_period,
                    "mean_peak_hour": mean_fraction * reference_period,
                    "period_estimation_method": estimator_method,
                })

    output = {
        "rhythms": table,
        "rhythm_methods": pd.DataFrame(method_rows),
        "rhythm_traces": pd.DataFrame(trace_rows),
    }
    if population_rows:
        output["rhythms_population"] = pd.DataFrame(population_rows)
    if null_rows:
        null = pd.DataFrame(null_rows)
        null["rhythmic_cosinor"] = np.where(
            np.isfinite(null["cosinor_p_value"]),
            null["cosinor_p_value"] < alpha_for_null, np.nan,
        )
        null["rhythmic_lombscargle"] = np.where(
            np.isfinite(null["lombscargle_false_alarm"]),
            null["lombscargle_false_alarm"] < alpha_for_null, np.nan,
        )
        valid_pair = (
            null["rhythmic_cosinor"].notna()
            & null["rhythmic_lombscargle"].notna()
        )
        cosinor_passed = null["rhythmic_cosinor"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        lomb_passed = null["rhythmic_lombscargle"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        null["rhythmic_both"] = np.where(
            valid_pair, cosinor_passed & lomb_passed, np.nan,
        )
        summary = (
            null.groupby("metric")
            .agg(
                surrogates=("replicate", "size"),
                false_positive_rate_cosinor=("rhythmic_cosinor", "mean"),
                false_positive_rate_lombscargle=("rhythmic_lombscargle", "mean"),
                false_positive_rate_both=("rhythmic_both", "mean"),
                false_positive_rate_primary=("rhythmic_primary", "mean"),
            )
            .reset_index()
        )
        summary["null_model"] = params["null_model"]
        summary["primary_method"] = primary_method
        observed = (
            table.groupby("metric")
            .agg(
                cells_tested=("identity", "size"),
                observed_rate_cosinor=("rhythmic_cosinor", "mean"),
                observed_rate_both=("rhythmic_both", "mean"),
                observed_rate_primary=("rhythmic", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(observed, on="metric", how="outer")
        # The number that matters: how much more often the real traces are
        # called rhythmic than matched noise with the same drift and smoothness.
        summary["excess_over_null_both"] = (
            summary["observed_rate_both"] - summary["false_positive_rate_both"]
        )
        summary["excess_over_null_primary"] = (
            summary["observed_rate_primary"]
            - summary["false_positive_rate_primary"]
        )
        output["rhythms_null"] = summary
    return output
