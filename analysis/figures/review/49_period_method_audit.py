"""Figure 49: reusable filtering, detrending, and period-method audit.

The default page selects the longest-observed cell and its corrected reporter
intensity. Every choice is replaceable::

    python analysis/figures/49_period_method_audit.py <run> --identity 44
    ... --cells 44,52,70                  compare several named cells
    ... --cells 5                         or the five longest-observed cells
    ... --metrics corrected_mean,area_px  compare several measurements
    ... --median-window-points 3
    ... --detrend-methods linear,robust_linear,first_difference,poly3,poly6,baseline,amp_baseline
    ... --period-methods lomb,ejtk,fft_nlls,mesa

The canvas grows for additional cell/measurement traces. Recording duration is
read from each trace and independently controls its time axis, observed-cycle
count, and the longest period that can meet the configured cycle minimum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent.parent))

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from _derive import _selected_identities
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from _text import figure_text
from panels import rhythm_audit


DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"
DEFAULT_DETRENDS = (
    "linear", "robust_linear", "first_difference", "poly3", "poly6",
    "baseline", "amp_baseline",
)
DEFAULT_METHODS = ("lomb", "ejtk", "fft_nlls", "mesa")
DETREND_LABELS = {
    "none": "No detrending",
    "linear": "Linear",
    "robust_linear": "Robust linear",
    "first_difference": "First difference",
    "poly3": "Third-degree polynomial",
    "cubic": "Third-degree polynomial",
    "bicubic": "Third-degree polynomial",
    "poly6": "Sixth-degree polynomial",
    "degree6": "Sixth-degree polynomial",
    "baseline": "Kernel baseline",
    "kernel": "Kernel baseline",
    "amp_baseline": "Amplitude and baseline",
    "amp&baseline": "Amplitude and baseline",
    "running_mean": "Running mean",
    "polynomial": "Polynomial",
    "frequency": "Frequency filter",
}
CLAIM = (
    "For each selected cell trace, the main period estimate and any candidate "
    "secondary components can be judged against input filtering, detrending "
    "choice, significance, and recording-length sufficiency."
)


def _shared_options() -> tuple[Option, ...]:
    return tuple(
        Option(name, default=workbench.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS[name])
        for name in workbench.CIRCADIAN_ANALYSIS_OPTIONS
    )


@figure(
    number=49,
    slug="period-method-audit",
    purpose="review",
    summary="filter, detrending, period-estimator, and candidate-component audit",
    title="Period-method audit: {n_cells} cells and {n_metrics} measurements",
    grammar="small multiples",
    reads=(Table("cell_frame.csv", module="motility"),),
    panels=(
        Panel(
            "method_grid", rhythm_audit.draw, block=True,
            min_width_inches=27.0, min_height_inches=20.0,
            title="Raw trace, detrended data, fitted models, and spectral diagnostics",
        ),
    ),
    options=(
        Option("identity", default=None),
        Option("cells", default="1", cast=str),
        Option("metrics", default=["corrected_mean"]),
        Option("hour_ticks", default=None),
        Option("median_window_points", default=3),
        Option("detrend_methods", default=list(DEFAULT_DETRENDS)),
        Option("period_methods", default=list(DEFAULT_METHODS)),
        *_shared_options(),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    cell_frame = ctx.table("cell_frame.csv")
    metrics = [str(value) for value in ctx.option("metrics")]
    if not metrics:
        raise SystemExit("--metrics needs at least one measured column")
    missing_metrics = [name for name in metrics if name not in cell_frame]
    if missing_metrics:
        raise SystemExit(
            "--metrics names columns not in cell_frame.csv: "
            + ", ".join(missing_metrics)
        )
    nonnumeric = [name for name in metrics if cell_frame[name].dtype.kind not in "fi"]
    if nonnumeric:
        raise SystemExit("--metrics must be numeric: " + ", ".join(nonnumeric))

    identity = ctx.option("identity")
    if identity is None:
        identities = _selected_identities(cell_frame, ctx.option("cells"))
    else:
        identities = [int(identity)]
        available = set(int(value) for value in cell_frame["identity"].unique())
        if identities[0] not in available:
            raise SystemExit(f"identity {identities[0]} is not in cell_frame.csv")

    empty_traces = [
        f"cell {identity}, {metric}"
        for identity in identities
        for metric in metrics
        if cell_frame.loc[
            cell_frame["identity"].eq(identity), ["hours", metric]
        ].dropna().empty
    ]
    if empty_traces:
        raise SystemExit(
            "selected cell/measurement traces contain no finite observations: "
            + "; ".join(empty_traces)
        )

    median_points = int(ctx.option("median_window_points"))
    if median_points < 3 or median_points % 2 == 0:
        raise SystemExit("--median-window-points must be an odd integer of at least 3")

    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    try:
        resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    params = dict(resolved["params"])

    detrends = [str(value) for value in ctx.option("detrend_methods")]
    primary_detrend = str(resolved["detrend"])
    detrends = list(dict.fromkeys([primary_detrend, *detrends]))
    unknown_detrends = [name for name in detrends if name not in workbench.DETREND_METHODS]
    if unknown_detrends:
        raise SystemExit(
            "--detrend-methods includes unknown choices: "
            + ", ".join(unknown_detrends)
            + ". Available: " + ", ".join(workbench.DETREND_METHODS)
        )

    estimators = [str(value) for value in ctx.option("period_methods")]
    primary_estimator = str(resolved["method"])
    estimators = list(dict.fromkeys([primary_estimator, *estimators]))
    unknown_estimators = [name for name in estimators if name not in workbench.PERIOD_METHODS]
    if unknown_estimators:
        raise SystemExit(
            "--period-methods includes unknown choices: "
            + ", ".join(unknown_estimators)
            + ". Available: " + ", ".join(workbench.PERIOD_METHODS)
        )

    trace_data, results, components, spectra, evidence = analyse_traces(
        cell_frame,
        identities=identities,
        metrics=metrics,
        detrends=detrends,
        estimators=estimators,
        median_window_points=median_points,
        params=params,
        fallback_significance_method=str(resolved["significance_method"]),
        correction=str(resolved["multiple_testing"]),
        alpha=float(resolved["rhythmic_alpha"]),
        min_observations=int(resolved["min_observations"]),
        min_cycles=float(resolved["min_cycles"]),
    )
    statistics = statistics_table(results)

    trace_keys = [
        {
            "trace_id": f"{identity}|{metric}",
            "identity": int(identity),
            "metric": metric,
        }
        for identity in identities for metric in metrics
    ]
    n_columns = 1 + len(estimators) + int("fft_nlls" in estimators) + int("mesa" in estimators)
    nominal_width = max(13.8, 2.0 + 3.75 * n_columns)
    # Each fit row has two completed Workbench comparison tables beneath it.
    comparison_inches = max(2.0, 0.4 * (len(estimators) + 2))
    nominal_height = 4.2 + len(trace_keys) * (2.0 + (2.45 + comparison_inches) * len(detrends))
    figure_ = ctx.sheet(nominal_width, nominal_height)
    actual_width, actual_height = map(float, figure_.get_size_inches())

    title = (
        f"Period-method audit: {len(identities)} "
        f"cell{'s' if len(identities) != 1 else ''} and {len(metrics)} "
        f"measurement{'s' if len(metrics) != 1 else ''}"
    )
    subtitle = (
        f"Raw and {median_points}-point median-filtered input · "
        f"{len(detrends)} detrenders · {len(estimators)} estimators · "
        f"{resolved['period_min_hours']:g}–{resolved['period_max_hours']:g} h search"
    )
    footnote = ctx.footnote(
        f"Circadian Workbench {workbench.WORKBENCH_VERSION}. Significance is "
        f"corrected by {resolved['multiple_testing']} at alpha "
        f"{resolved['rhythmic_alpha']:g} across unique trace tests; † means fewer "
        f"than {resolved['min_cycles']:g} observed cycles. Significance-bearing "
        f"estimators test themselves; fit-only estimators use "
        f"{workbench.PERIOD_METHODS[resolved['significance_method']]['label']}. "
        "Returned nonlinear components and maximum-entropy peaks are not "
        "individually significance-tested."
    )
    text = figure_text(
        ctx.run, ctx.spec.slug, argv=ctx.argv, item=ctx.item,
        title=title, subtitle=subtitle, footnote=footnote, note="",
    )
    canvas_text = text.on_canvas()
    config = renderer_config(
        ctx, figure_, trace_keys, detrends, estimators, resolved,
        median_points, canvas_text.title, canvas_text.subtitle,
        canvas_text.footnote,
    )
    rect = tuple(config["rect"])
    drawn = ctx.drew(
        "method_grid",
        rhythm_audit.draw(
            figure_, rect, trace_data, results, components, spectra, config,
        ),
    )

    master = ctx.name.replace("/", "-").replace("\\", "-") + ".svg"
    producer = standalone_producer(config, master)
    settings = pd.DataFrame([
        {
            "settings_json": json.dumps(_json_safe({
                "identities": identities,
                "metrics": metrics,
                "median_window_points": median_points,
                "detrend_methods": detrends,
                "period_methods": estimators,
                "fallback_significance_method": resolved["significance_method"],
            })),
            "analysis_params_json": json.dumps(_json_safe(params)),
        }
    ])
    readme = f"""# {text.title}

{text.subtitle}

## What this audit does

For every selected cell and measurement, the page shows the untouched trace,
then raw-input and {median_points}-point median-filtered versions under every
requested detrending and period-estimation combination. Significance, period
sufficiency, returned nonlinear components, and maximum-entropy spectra remain
separate outputs.

## Reuse

Run one cell with:

`python analysis/figures/49_period_method_audit.py <run> --stem <recording> --identity <cell>`

Use `--cells 44,52,70` for named cells or `--cells 5` for the five
longest-observed cells. The sheet grows instead of shrinking its panels. Use
`--metrics`, `--detrend-methods`, `--period-methods`, and
`--median-window-points` to change the comparison. All shared Circadian
Workbench controls are also exposed, including search limits, method settings,
detrending controls, alpha, correction, minimum observations, and minimum
observed cycles. Choose `--stem` for another recording; recording length is
read from each selected trace rather than assumed.

`python plot.py --output reproduced.svg` redraws the page from the bundled
tables without the source run. Re-analysis uses this registered project builder
and Circadian Workbench through `analysis/circadian.py`.

## Interpretation

{text.footnote}
"""
    return FigureResult(
        figure=figure_,
        axes=list(drawn.axes),
        figure_data=trace_data,
        subtitle=text.subtitle,
        footnote=text.footnote,
        heading=text.claim or CLAIM,
        readme=readme,
        auxiliary={
            "results.csv": results,
            "components.csv": components,
            "spectra.csv": spectra,
            "evidence_tests.csv": evidence,
            "statistics.csv": statistics,
            "settings.csv": settings,
        },
        standalone_producer=producer,
        producer_sources={
            "renderer.py": Path(rhythm_audit.__file__),
            "builder.py": Path(__file__),
            "circadian.py": Path(workbench.__file__),
        },
    )


def _safe_estimate(
    hours: np.ndarray,
    values: np.ndarray,
    params: dict[str, Any],
    method: str,
    detrend: str,
) -> dict[str, Any]:
    try:
        return workbench.estimate_one(
            hours, values, params, method, detrend=detrend,
        )
    except (ValueError, RuntimeError) as error:
        return {
            "method": method,
            "method_label": workbench.PERIOD_METHODS[method]["label"],
            "status": "failed",
            "message": str(error),
            "diagnostics": {},
            "components": [],
            "period_hours": np.nan,
            "p_value": np.nan,
            "workbench_version": workbench.WORKBENCH_VERSION,
        }


def _finite(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _array(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.full(length, np.nan, dtype=float)
    result = np.asarray(value, dtype=float)
    return result if result.shape == (length,) else np.full(length, np.nan, dtype=float)


def analyse_traces(
    cell_frame: pd.DataFrame,
    *,
    identities: list[int],
    metrics: list[str],
    detrends: list[str],
    estimators: list[str],
    median_window_points: int,
    params: dict[str, Any],
    fallback_significance_method: str,
    correction: str,
    alpha: float,
    min_observations: int,
    min_cycles: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete requested grid and return every auditable table."""
    trace_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    evidence_cache: dict[str, dict[str, Any]] = {}

    for identity in identities:
        selected = cell_frame.loc[cell_frame["identity"].eq(identity)].sort_values(
            "hours", kind="mergesort"
        )
        for metric in metrics:
            trace = selected[["hours", metric]].dropna().drop_duplicates("hours")
            hours = trace["hours"].to_numpy(float)
            raw = trace[metric].to_numpy(float)
            trace_id = f"{identity}|{metric}"
            if not len(hours):
                continue
            for hour, value in zip(hours, raw):
                trace_rows.append({
                    "trace_id": trace_id, "identity": identity, "metric": metric,
                    "display_panel": "raw_reference", "hours": hour,
                    "raw_value": value, "analysis_input_value": value,
                    "baseline_value": np.nan, "plotted_value": value,
                    "fitted_value": np.nan, "preprocessor": "raw_reference",
                    "median_window_points": median_window_points,
                    "detrend": "raw_reference", "estimator": "raw_reference",
                    "fit_kind": "none",
                })
            filtered_inputs = {
                "raw": raw.copy(),
                "median": np.asarray(
                    median_filter(
                        raw, size=median_window_points, mode="nearest"
                    ),
                    dtype=float,
                ),
            }
            span = float(hours.max() - hours.min()) if len(hours) > 1 else 0.0

            for detrend in detrends:
                for preprocessor, analysis_input in filtered_inputs.items():
                    try:
                        detrended = workbench.detrend_trace(
                            hours, analysis_input, params, method=detrend,
                        ) if len(hours) >= 2 else {}
                        residual = _array(detrended.get("values"), len(hours))
                        baseline = _array(detrended.get("baseline"), len(hours))
                        detrend_message = ""
                    except (ValueError, RuntimeError) as error:
                        residual = np.full(len(hours), np.nan)
                        baseline = np.full(len(hours), np.nan)
                        detrend_message = str(error)
                    for hour, raw_value, input_value, base, value in zip(
                        hours, raw, analysis_input, baseline, residual
                    ):
                        trace_rows.append({
                            "trace_id": trace_id, "identity": identity,
                            "metric": metric, "display_panel": "detrended",
                            "hours": hour, "raw_value": raw_value,
                            "analysis_input_value": input_value,
                            "baseline_value": base, "plotted_value": value,
                            "fitted_value": np.nan, "preprocessor": preprocessor,
                            "median_window_points": median_window_points,
                            "detrend": detrend, "estimator": "none",
                            "fit_kind": "none",
                        })

                    estimate_cache: dict[str, dict[str, Any]] = {}
                    comparison_definition = ""
                    comparison_record = ""
                    can_fit = len(hours) >= min_observations and not detrend_message
                    if can_fit:
                        try:
                            compared = workbench.estimate_trace(
                                hours, analysis_input, params, methods=estimators,
                                detrend=detrend,
                            )
                            details = {entry["method"]: entry
                                       for entry in compared["comparison"]["estimates"]}
                            comparison_record = json.dumps(compared["run_record"], allow_nan=False)
                            estimate_cache = {
                                row["method"]: {
                                    **row,
                                    "diagnostics": details[row["method"]].get("diagnostics", {}),
                                    "components": details[row["method"]].get("components", []),
                                    "workbench_run_record_json": comparison_record,
                                } for row in compared["rows"]
                            }
                            # Actual completed Result, never a reconstructed or
                            # refitted display result. The renderer uses this
                            # detached definition and never imports Workbench.
                            comparison_definition = json.dumps(
                                compared["result"].plot().definition.as_dict(), allow_nan=False)
                        except (ValueError, RuntimeError):
                            # Retain the existing per-method refusal behaviour
                            # if the multi-method call cannot complete.
                            estimate_cache = {}
                    for estimator in estimators:
                        estimate = estimate_cache.get(estimator) or (_safe_estimate(
                            hours, analysis_input, params, estimator, detrend,
                        ) if can_fit else {
                            "method": estimator,
                            "method_label": workbench.PERIOD_METHODS[estimator]["label"],
                            "status": "not_tested" if len(hours) < min_observations else "failed",
                            "message": (
                                "too_few_observations" if len(hours) < min_observations
                                else detrend_message
                            ),
                            "diagnostics": {}, "components": [],
                            "period_hours": np.nan, "p_value": np.nan,
                            "workbench_version": workbench.WORKBENCH_VERSION,
                        })
                        estimate_cache[estimator] = estimate
                        significance_method = (
                            estimator
                            if bool(workbench.PERIOD_METHODS[estimator]["gives_significance"])
                            else fallback_significance_method
                        )
                        evidence_id = json.dumps(
                            [identity, metric, preprocessor, detrend, significance_method],
                            separators=(",", ":"),
                        )
                        if evidence_id not in evidence_cache:
                            evidence = (
                                estimate if significance_method == estimator or not can_fit else
                                estimate_cache.get(significance_method) or _safe_estimate(
                                    hours, analysis_input, params,
                                    significance_method, detrend,
                                )
                            )
                            evidence_cache[evidence_id] = {
                                "evidence_id": evidence_id,
                                "trace_id": trace_id,
                                "identity": identity,
                                "metric": metric,
                                "preprocessor": preprocessor,
                                "detrend": detrend,
                                "significance_method": significance_method,
                                "significance_method_label": workbench.PERIOD_METHODS[
                                    significance_method
                                ]["label"],
                                "status": str(evidence.get("status") or "failed"),
                                "period_hours": _finite(evidence.get("period_hours")),
                                "p_value": _finite(evidence.get("p_value")),
                                "message": str(evidence.get("message") or ""),
                                "workbench_run_record_json": evidence.get("workbench_run_record_json", ""),
                            }

                        period = _finite(estimate.get("period_hours"))
                        if estimator == "fft_nlls":
                            fitted = workbench.fft_nlls_fitted_values(hours, estimate)
                            fit_kind = "returned FFT-NLLS multi-component model"
                        else:
                            fitted = workbench.descriptive_cosinor_fitted_values(
                                hours, residual, period,
                            )
                            fit_kind = "descriptive Workbench cosine at estimated period"
                        for hour, raw_value, input_value, base, value, fit in zip(
                            hours, raw, analysis_input, baseline, residual, fitted
                        ):
                            trace_rows.append({
                                "trace_id": trace_id, "identity": identity,
                                "metric": metric, "display_panel": "fit",
                                "hours": hour, "raw_value": raw_value,
                                "analysis_input_value": input_value,
                                "baseline_value": base, "plotted_value": value,
                                "fitted_value": fit, "preprocessor": preprocessor,
                                "median_window_points": median_window_points,
                                "detrend": detrend, "estimator": estimator,
                                "fit_kind": fit_kind,
                            })

                        diagnostics = dict(estimate.get("diagnostics") or {})
                        returned_components = list(estimate.get("components") or [])
                        for component_index, component in enumerate(returned_components, start=1):
                            component_period = _finite(component.get("period_hours"))
                            selected_component = bool(component.get("selected", False))
                            if (not selected_component and np.isfinite(period)
                                    and np.isfinite(component_period)):
                                selected_component = bool(np.isclose(
                                    component_period, period, rtol=1e-6, atol=1e-6,
                                ))
                            component_rows.append({
                                "trace_id": trace_id, "identity": identity,
                                "metric": metric, "preprocessor": preprocessor,
                                "detrend": detrend, "estimator": estimator,
                                "component_index": component_index,
                                "period_hours": component_period,
                                "period_error_hours": _finite(component.get("period_error_hours")),
                                "amplitude": _finite(component.get("amplitude")),
                                "amplitude_error": _finite(component.get("amplitude_error")),
                                "phase_hours": _finite(component.get("phase_hours")),
                                "phase_error_hours": _finite(component.get("phase_error_hours")),
                                "relative_amplitude_error": _finite(component.get("rae")),
                                "selected": selected_component,
                                "component_significance": "not tested",
                            })
                        diagnostic_periods = np.asarray(
                            diagnostics.get("periods_hours")
                            if diagnostics.get("periods_hours") is not None else [],
                            dtype=float,
                        )
                        diagnostic_power = np.asarray(
                            diagnostics.get("power")
                            if diagnostics.get("power") is not None else [],
                            dtype=float,
                        )
                        if diagnostic_periods.size == diagnostic_power.size and diagnostic_periods.size:
                            maximum = float(np.nanmax(diagnostic_power))
                            normalised = (
                                diagnostic_power / maximum
                                if np.isfinite(maximum) and maximum > 0
                                else np.zeros_like(diagnostic_power)
                            )
                            for spectrum_period, power, normalised_power in zip(
                                diagnostic_periods, diagnostic_power, normalised
                            ):
                                spectrum_rows.append({
                                    "trace_id": trace_id, "identity": identity,
                                    "metric": metric, "preprocessor": preprocessor,
                                    "detrend": detrend, "estimator": estimator,
                                    "period_hours": spectrum_period,
                                    "power": power,
                                    "normalised_power": normalised_power,
                                })

                        cycles = span / period if np.isfinite(period) and period > 0 else np.nan
                        low, high = map(float, params["period_search_hours"])
                        edge_tolerance = max(0.1, 0.005 * (high - low))
                        at_edge = bool(
                            np.isfinite(period)
                            and (period <= low + edge_tolerance or period >= high - edge_tolerance)
                        )
                        filter_delta = analysis_input - raw
                        result_rows.append({
                            "evidence_id": evidence_id,
                            "trace_id": trace_id, "identity": identity,
                            "metric": metric, "observations": len(hours),
                            "span_hours": span, "preprocessor": preprocessor,
                            "median_window_points": median_window_points,
                            "filter_changed_points": int(np.count_nonzero(
                                ~np.isclose(analysis_input, raw, equal_nan=True)
                            )),
                            "filter_max_abs_change": float(np.nanmax(np.abs(filter_delta))),
                            "detrend": detrend,
                            "applied_detrend": str(estimate.get("detrend") or detrend),
                            "detrend_window_hours": float(params["detrend_window_hours"]),
                            "estimator": estimator,
                            "estimator_label": workbench.PERIOD_METHODS[estimator]["label"],
                            "estimate_status": str(estimate.get("status") or "failed"),
                            "estimated_period_hours": period,
                            "period_error_hours": _finite(estimate.get("period_error_hours")),
                            "estimator_p_value": _finite(estimate.get("p_value")),
                            "estimator_goodness_of_fit": _finite(estimate.get("goodness_of_fit")),
                            "significance_method": significance_method,
                            "significance_method_label": workbench.PERIOD_METHODS[
                                significance_method
                            ]["label"],
                            "cycles_observed": cycles,
                            "min_cycles": min_cycles,
                            "period_underdetermined": bool(
                                not np.isfinite(cycles) or cycles < min_cycles
                            ),
                            "period_at_search_edge": at_edge,
                            "period_search_min_hours": low,
                            "period_search_max_hours": high,
                            "fit_kind": fit_kind,
                            "component_count": len(returned_components),
                            "component_significance": "not tested",
                            "mesa_peaks_in_search_band": diagnostics.get("peaks_in_search_band"),
                            "diagnostic_warnings_json": json.dumps(
                                list(diagnostics.get("warnings") or [])
                            ),
                            "diagnostic_stop_reason": str(
                                diagnostics.get("stopped_because") or ""
                            ),
                            "workbench_version": workbench.WORKBENCH_VERSION,
                            "workbench_run_record_json": estimate.get("workbench_run_record_json", ""),
                            "workbench_comparison_json": comparison_definition,
                            "message": str(estimate.get("message") or ""),
                        })

    evidence = pd.DataFrame(evidence_cache.values())
    if not evidence.empty:
        evidence["q_value"] = workbench.adjust_pvalues(
            evidence["p_value"].to_numpy(float), correction,
        )
        tested = evidence["status"].eq("ok") & evidence["q_value"].notna()
        evidence["significant"] = tested & evidence["q_value"].lt(alpha)
        evidence["rhythm_status"] = np.where(
            tested, np.where(evidence["significant"], "significant", "not significant"),
            "not tested",
        )
        evidence["correction_method"] = correction
        evidence["alpha"] = alpha
        evidence_by_id = evidence.set_index("evidence_id")
        for row in result_rows:
            test = evidence_by_id.loc[row["evidence_id"]]
            row.update({
                "significance_status": test["status"],
                "significance_period_hours": test["period_hours"],
                "p_value": test["p_value"],
                "q_value": test["q_value"],
                "correction_method": correction,
                "alpha": alpha,
                "significant": bool(test["significant"]),
                "rhythm_status": test["rhythm_status"],
            })

    trace_columns = [
        "trace_id", "identity", "metric", "display_panel", "hours", "raw_value",
        "analysis_input_value", "baseline_value", "plotted_value", "fitted_value",
        "preprocessor", "median_window_points", "detrend", "estimator", "fit_kind",
    ]
    component_columns = [
        "trace_id", "identity", "metric", "preprocessor", "detrend", "estimator",
        "component_index", "period_hours", "period_error_hours", "amplitude",
        "amplitude_error", "phase_hours", "phase_error_hours",
        "relative_amplitude_error", "selected", "component_significance",
    ]
    spectrum_columns = [
        "trace_id", "identity", "metric", "preprocessor", "detrend", "estimator",
        "period_hours", "power", "normalised_power",
    ]
    return (
        pd.DataFrame(trace_rows, columns=trace_columns),
        pd.DataFrame(result_rows),
        pd.DataFrame(component_rows, columns=component_columns),
        pd.DataFrame(spectrum_rows, columns=spectrum_columns),
        evidence,
    )


def statistics_table(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize every displayed trace-level test for the figure bundle."""
    statistics = results.copy()
    statistics["test_name"] = statistics["significance_method_label"]
    statistics["estimate_name"] = "estimated period"
    statistics["estimate"] = statistics["estimated_period_hours"]
    statistics["estimate_units"] = "h"
    statistics["effect_size_name"] = "estimator goodness of fit"
    statistics["effect_size"] = statistics["estimator_goodness_of_fit"]
    statistics["algorithm_id"] = (
        "circadian-workbench:" + statistics["significance_method"].astype(str)
    )
    statistics["inputs_json"] = statistics.apply(
        lambda row: json.dumps({
            "trace_id": row["trace_id"],
            "identity": int(row["identity"]),
            "metric": row["metric"],
            "observations": int(row["observations"]),
            "span_hours": float(row["span_hours"]),
            "preprocessor": row["preprocessor"],
        }), axis=1,
    )
    statistics["parameters_json"] = statistics.apply(
        lambda row: json.dumps({
            "detrend": row["detrend"],
            "detrend_window_hours": float(row["detrend_window_hours"]),
            "estimator": row["estimator"],
            "significance_method": row["significance_method"],
            "period_search_hours": [
                float(row["period_search_min_hours"]),
                float(row["period_search_max_hours"]),
            ],
            "alpha": float(row["alpha"]),
            "correction": row["correction_method"],
            "minimum_cycles": float(row["min_cycles"]),
            "median_window_points": int(row["median_window_points"]),
        }), axis=1,
    )
    statistics["expected_json"] = statistics.apply(
        lambda row: json.dumps({
            "p_value": _nullable(row["p_value"]),
            "q_value": _nullable(row["q_value"]),
            "significant": bool(row["significant"]),
            "period_hours": _nullable(row["estimate"]),
        }), axis=1,
    )
    statistics["display_json"] = statistics.apply(
        lambda row: json.dumps({
            "trace_id": row["trace_id"],
            "preprocessor": row["preprocessor"],
            "detrend": row["detrend"],
            "estimator": row["estimator"],
        }), axis=1,
    )
    statistics["tolerances_json"] = json.dumps({"rtol": 1e-9, "atol": 1e-12})
    return statistics


def _nullable(value: Any) -> float | None:
    numeric = _finite(value)
    return float(numeric) if np.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def renderer_config(
    ctx: FigureContext,
    figure_: Any,
    trace_keys: list[dict[str, Any]],
    detrends: list[str],
    estimators: list[str],
    resolved: dict[str, Any],
    median_points: int,
    title: str,
    subtitle: str,
    footnote: str,
) -> dict[str, Any]:
    width, height = map(float, figure_.get_size_inches())
    top_inches = 2.0 * ctx.theme.canvas_scale
    bottom_inches = 1.75 * ctx.theme.canvas_scale
    rect = [
        0.06,
        bottom_inches / height,
        0.925,
        1.0 - (top_inches + bottom_inches) / height,
    ]
    rc_names = (
        "font.family", "font.sans-serif", "font.size", "axes.labelsize",
        "axes.titlesize", "axes.linewidth", "xtick.labelsize", "ytick.labelsize",
        "xtick.major.width", "ytick.major.width", "svg.fonttype",
        "figure.facecolor", "savefig.facecolor",
    )
    return _json_safe({
        "figsize": [width, height],
        "rect": rect,
        "trace_keys": trace_keys,
        "detrends": detrends,
        "detrend_labels": {
            name: DETREND_LABELS.get(name, name.replace("_", " ").capitalize())
            for name in detrends
        },
        "estimators": estimators,
        "shared_comparisons": True,
        "comparison_row_height": max(0.82, 0.16 * (len(estimators) + 2)),
        "estimator_labels": {
            name: workbench.PERIOD_METHODS[name]["label"] for name in estimators
        },
        "preprocessors": ["raw", "median"],
        "preprocessor_labels": {
            "raw": "Raw input", "median": f"{median_points}-point median",
        },
        "period_min_hours": float(resolved["period_min_hours"]),
        "period_max_hours": float(resolved["period_max_hours"]),
        "min_cycles": float(resolved["min_cycles"]),
        "hour_ticks": ctx.option("hour_ticks"),
        "title": title,
        "subtitle": subtitle,
        "footnote": footnote,
        "header_x": 0.06,
        "title_y": 1.0 - 0.24 * ctx.theme.canvas_scale / height,
        "subtitle_y": 1.0 - 0.78 * ctx.theme.canvas_scale / height,
        "legend_y": 1.0 - 0.75 * ctx.theme.canvas_scale / height,
        "footnote_y": 0.18 * ctx.theme.canvas_scale / height,
        "colours": {
            "raw": ctx.theme.colour("ink"),
            "median": ctx.theme.colour("morphology"),
            "reference": ctx.theme.colour("reference"),
            "warning": ctx.theme.colour("motility"),
            "ink": ctx.theme.colour("ink"),
            "caption": ctx.theme.colour("caption"),
            "page": ctx.theme.colour("page"),
        },
        "font": {
            "title": ctx.theme.size("title"),
            "subtitle": ctx.theme.size("subtitle"),
            "legend": ctx.theme.size("caption") * 0.72,
            "column": ctx.theme.size("caption") * 0.66,
            "row": ctx.theme.size("caption") * 0.64,
            "annotation": ctx.theme.size("note") * 0.43,
            "small": ctx.theme.size("note") * 0.40,
            "tick": float(ctx.theme["tick_size"]) * 0.38,
            "axis": float(ctx.theme["axis_size"]) * 0.42,
            "footnote": ctx.theme.size("note") * 0.58,
        },
        "stroke": {
            "axis": ctx.theme.stroke("hairline"),
            "tick_length": float(ctx.theme["tick_mark_length"]) * 0.45,
            "guide": ctx.theme.stroke("hairline") * 0.75,
            "data": ctx.theme.stroke("line") * 0.30,
            "line": ctx.theme.stroke("line") * 0.48,
            "fit": ctx.theme.stroke("line") * 0.76,
        },
        "point_area": ctx.theme.point_area(0.32),
        "component_marker": float(ctx.theme["marker_size"]) * 0.48,
        "rc": {name: mpl.rcParams[name] for name in rc_names},
    })


def standalone_producer(config: dict[str, Any], master: str) -> str:
    return f'''"""Exact period-method audit reproduction from bundled tables."""
from pathlib import Path
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import reprofig
import src_renderer

CONFIG = {config!r}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    target = args.output or bundle / "reproduced.svg"
    mpl.rcParams.update(CONFIG["rc"])
    figure = plt.figure(figsize=CONFIG["figsize"])
    traces = pd.read_csv(bundle / "figure_data.csv")
    results = pd.read_csv(bundle / "der_results.csv")
    components = pd.read_csv(bundle / "der_components.csv")
    spectra = pd.read_csv(bundle / "der_spectra.csv")
    src_renderer.draw(figure, tuple(CONFIG["rect"]), traces, results,
                      components, spectra, CONFIG)
    record = reprofig.extract_record(bundle / {master!r})
    reprofig.save_figure(figure, target, record=record,
                         savefig_kwargs={{"bbox_inches": "tight", "transparent": True}})
    plt.close(figure)

if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    run_figure("period-method-audit", DEFAULT_RUN)
