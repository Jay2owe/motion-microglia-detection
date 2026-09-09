"""One gateway from cell traces into Circadian Workbench.

Circadian Workbench owns the scientific implementations.  This module only
translates the analysis package's ``hours, values`` arrays into the timestamped
frame its core accepts and translates the returned rows back into plain dicts.
Keeping that seam here means future circadian analyses can be added without
copying another implementation into ``analysis.modules``.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

import circadian_workbench as cw

WORKBENCH_VERSION = cw.__version__

# New uniform entrance: exactly Workbench's arguments, defaults, units, caller
# and completed-result/figure contract. No Motion preset or history lookup.
trace = cw.trace
describe = cw.describe


def argument_group(action: str = "compare_periods") -> dict[str, dict[str, Any]]:
    """Detached, authoritative scientific settings accepted by an action."""
    description = cw.describe(action)
    if not description["ok"]:
        raise ValueError(description["error"])
    references = {key for item in description["params"]
                  for key in item.get("references", [])}
    return {row["key"]: deepcopy(row)
            for row in description.get("config_arguments", [])
            if row["key"] in references}


# Circadian Workbench is the authority for baseline removal. Discover its
# choices through the public configuration contract; omit only alternate
# spellings of an algorithm already present under its canonical name.
_WORKBENCH_CONFIG_SCHEMA = argument_group()
_DETREND_SPEC = _WORKBENCH_CONFIG_SCHEMA["period_detrend"]
_DETREND_ALLOWED = [*_DETREND_SPEC["allowed"], *_DETREND_SPEC["aliases"]]
_PREFERRED_DETREND_ORDER = (
    "none", "running_mean", "moving_average", "moving_median",
    "running_median", "median", "linear", "robust_linear", "robust",
    "huber", "first_difference", "first-difference", "difference", "diff",
    "lowess", "loess", "savitzky_golay", "savitzky-golay", "savgol",
    "cubic", "bicubic", "poly6", "polynomial", "poly3", "degree6",
    "kernel", "baseline", "amp_baseline", "amp&baseline",
    "asymmetric_least_squares", "asymmetric-least-squares", "asls", "als",
    "frequency",
)
DETREND_METHODS: tuple[str, ...] = tuple(
    [name for name in _PREFERRED_DETREND_ORDER if name in _DETREND_ALLOWED]
    + sorted(set(_DETREND_ALLOWED) - set(_PREFERRED_DETREND_ORDER))
)

# Explicit compatibility values for existing measurement workflows, not a
# second default registry for the uniform trace entrance above.
DETREND_DEFAULTS: dict[str, Any] = {
    "detrend": "linear",
    "detrend_window_hours": 24.0,
    "detrend_polynomial_degree": 3,
    "detrend_min_valid_fraction": 0.5,
    "detrend_bandwidth_hours": None,
    "detrend_low_cut_hours": 45.0,
    "detrend_high_cut_hours": 4.0,
    "detrend_filter_order": 2,
    "detrend_lowess_fraction": None,
    "detrend_lowess_iterations": 3,
    "detrend_asls_smoothness": 1_000_000.0,
    "detrend_asls_asymmetry": 0.01,
    "detrend_asls_iterations": 10,
}

# Shared by the main rhythms module and any other module that estimates a
# period. Keeping this beside the adapter prevents a coupling analysis or a new
# figure from silently reverting to a different search or estimator.
PERIOD_ANALYSIS_DEFAULTS: dict[str, Any] = {
    "period_search_hours": [2.0, 48.0],
    "period_methods": ["lomb", "chi_square", "f"],
    "period_estimation_method": "lomb",
    "primary_rhythm_test": "lomb",
    "fixed_period_hours": 24.0,
    "workbench_config": {},
    "jtk_periods": [4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0,
                    28.0, 32.0, 36.0, 48.0],
    "ejtk_permutations": 1000,
    "jtk_correction": "none",
    "jtk_seed": 20260901,
    "min_observations": 24,
    "min_cycles_for_confident_period": 3.0,
    "rhythmic_alpha": 0.05,
}

#: One option contract for every figure that performs a fresh period or rhythm
#: analysis. The broad search and significance defaults are intentionally
#: visible on every such figure; method and detrending choices inherit the
#: run's ``rhythms`` module setting when unset.
CIRCADIAN_ANALYSIS_OPTION_DEFAULTS: dict[str, Any] = {
    "fit_method": None,
    "significance_method": None,
    "period_config": {},
    "period_min_hours": 2.0,
    "period_max_hours": 48.0,
    "rhythmic_alpha": 0.05,
    "multiple_testing": "bh",
    "min_observations": 24,
    "min_cycles": 3.0,
    **{name: None for name in DETREND_DEFAULTS},
}
CIRCADIAN_ANALYSIS_OPTIONS: tuple[str, ...] = tuple(
    CIRCADIAN_ANALYSIS_OPTION_DEFAULTS
)


def resolve_analysis_options(params: dict, option: Any) -> dict[str, Any]:
    """Resolve the shared Circadian Workbench controls for one figure.

    ``option`` is normally ``FigureContext.option``. Keeping this resolution
    beside the gateway prevents individual plots from assigning different
    meanings or inheritance rules to the same controls.
    """
    inherited = dict(params)

    def choose(name: str, fallback: Any) -> Any:
        value = option(name)
        return fallback if value is None else value

    method = str(choose(
        "fit_method",
        inherited.get("period_estimation_method",
                      inherited.get("primary_rhythm_test", "lomb")),
    ))
    significance_method = resolve_significance_method(
        method,
        choose("significance_method", inherited.get("primary_rhythm_test", "lomb")),
    )
    low = float(choose("period_min_hours", inherited["period_search_hours"][0]))
    high = float(choose("period_max_hours", inherited["period_search_hours"][1]))
    alpha = float(choose("rhythmic_alpha", inherited.get("rhythmic_alpha", 0.05)))
    min_observations = int(choose(
        "min_observations", inherited.get("min_observations", 24)))
    min_cycles = float(choose(
        "min_cycles", inherited.get("min_cycles_for_confident_period", 3.0)))
    correction = str(choose(
        "multiple_testing", inherited.get("multiple_testing", "bh")))
    if not 0 < low < high:
        raise ValueError("period limits must satisfy 0 < minimum < maximum")
    if not 0 < alpha < 1:
        raise ValueError("rhythmic_alpha must be between 0 and 1")
    if min_observations < 6:
        raise ValueError("min_observations must be at least 6")
    if not np.isfinite(min_cycles) or min_cycles <= 0:
        raise ValueError("min_cycles must be finite and positive")
    if correction not in {"bh", "bonferroni", "sidak", "none"}:
        raise ValueError(
            "multiple_testing must be bh, bonferroni, sidak or none")

    detrend_args = {
        "method": option("detrend"),
        "window_hours": option("detrend_window_hours"),
        "polynomial_degree": option("detrend_polynomial_degree"),
        "min_valid_fraction": option("detrend_min_valid_fraction"),
        "bandwidth_hours": option("detrend_bandwidth_hours"),
        "low_cut_hours": option("detrend_low_cut_hours"),
        "high_cut_hours": option("detrend_high_cut_hours"),
        "filter_order": option("detrend_filter_order"),
        "lowess_fraction": option("detrend_lowess_fraction"),
        "lowess_iterations": option("detrend_lowess_iterations"),
        "asls_smoothness": option("detrend_asls_smoothness"),
        "asls_asymmetry": option("detrend_asls_asymmetry"),
        "asls_iterations": option("detrend_asls_iterations"),
    }
    detrending = detrend_settings(inherited, **detrend_args)
    extra = option("period_config")
    if not isinstance(extra, dict):
        raise ValueError("period_config must be a JSON object")
    methods = list(inherited.get("period_methods", []))
    for required in (method, significance_method):
        if required not in methods:
            methods.append(required)
    resolved_params = {
        **inherited,
        **detrending,
        "period_search_hours": [low, high],
        "period_estimation_method": method,
        "primary_rhythm_test": significance_method,
        "period_methods": methods,
        "rhythmic_alpha": alpha,
        "multiple_testing": correction,
        "min_observations": min_observations,
        "min_cycles_for_confident_period": min_cycles,
        "workbench_config": {
            **dict(inherited.get("workbench_config") or {}), **extra},
    }
    return {
        "params": resolved_params,
        "method": method,
        "significance_method": significance_method,
        "period_min_hours": low,
        "period_max_hours": high,
        "rhythmic_alpha": alpha,
        "multiple_testing": correction,
        "min_observations": min_observations,
        "min_cycles": min_cycles,
        **detrending,
    }

PERIOD_METHODS: dict[str, dict[str, Any]] = {
    str(entry["key"]): dict(entry)
    for entry in cw.call("period_methods").data["methods"]
}
SIGNIFICANCE_METHODS: tuple[str, ...] = tuple(
    key for key, entry in PERIOD_METHODS.items() if entry["gives_significance"]
)
NORMALIZATION_METHODS: dict[str, dict[str, Any]] = {
    str(entry["key"]): dict(entry)
    for entry in cw.normalization_methods().data["methods"]
}


def scientific_options() -> dict[str, dict[str, Any]]:
    """Workbench meanings with Motion spelling/role bindings, not new defaults.

    Family correction and data/cycle sufficiency are Motion workflow controls;
    they are deliberately not presented as Workbench estimator arguments.
    """
    config = argument_group()
    mapping = {
        "fit_method": "period_method", "period_methods": "period_methods",
        "significance_method": "period_method",
        "secondary_significance_method": "period_method",
        "period_min_hours": "period_min_hours", "period_max_hours": "period_max_hours",
        "rhythmic_alpha": "periodogram_alpha",
        **{name: "period_" + name for name in DETREND_DEFAULTS},
        "detrend_methods": "period_detrend",
    }
    entries = {name: {**deepcopy(config[key]), "reference": "config." + key}
               for name, key in mapping.items()}
    for name in ("significance_method", "secondary_significance_method"):
        entries[name]["allowed"] = list(SIGNIFICANCE_METHODS)
        entries[name]["role"] = "Significance test, selected separately from period estimation."
    entries["rhythmic_alpha"]["references"] = ["config.periodogram_alpha", "config.jtk_alpha"]
    entries["rhythmic_alpha"]["role"] = "Explicit threshold also supplied to rank tests and family correction."
    entries["detrend_methods"]["type"] = "list"
    entries["period_config"] = deepcopy(next(
        item for item in cw.describe("compare_periods")["params"] if item["name"] == "config"))
    return entries


def sample_interval_minutes(hours: Sequence[float]) -> float:
    """Median positive sampling interval of one trace, in minutes."""
    ordered = np.sort(np.unique(np.asarray(hours, dtype=float)))
    differences = np.diff(ordered)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        raise ValueError("a rhythm trace needs at least two distinct times")
    return float(np.median(positive) * 60.0)


def selected_frame(hours: Sequence[float], values: Sequence[float]) -> cw.TraceData:
    """A cell trace in Circadian Workbench's public input shape."""
    return cw.TraceData(hours, values, name="cell trace")


def available_normalization_methods() -> list[dict[str, Any]]:
    """The installed workbench's normalisation keys, formulas and aliases."""

    return [dict(entry) for entry in NORMALIZATION_METHODS.values()]


def normalize_trace(
    hours: Sequence[float],
    values: Sequence[float],
    method: str = "minmax",
    *,
    target_min: float = -1.0,
    target_max: float = 1.0,
    reference_value: float | None = None,
    reference_start_hours: float | None = None,
    reference_end_hours: float | None = None,
    reference_statistic: str = "mean",
    standard_deviation_ddof: int = 0,
    detrended: bool = False,
    envelope_floor_fraction: float = 0.1,
) -> dict[str, Any]:
    """Normalise one motion trace without duplicating the shared arithmetic.

    ``method`` is a key or alias from :func:`available_normalization_methods`.
    ``target_min`` and ``target_max`` define min–max output bounds. Reference
    methods take either ``reference_value`` or both inclusive reference-window
    bounds; ``reference_statistic`` selects their mean or median.
    ``standard_deviation_ddof`` is 0 for a population z-score and 1 for a
    sample z-score. ``detrended`` guards invalid mean-ratio scaling, while
    ``envelope_floor_fraction`` controls when a decayed fitted envelope becomes
    too small to divide by safely.
    """

    result = cw.trace(hours, values, name="motion trace").normalize(
        method=method,
        target_min=target_min,
        target_max=target_max,
        reference_value=reference_value,
        reference_start_hours=reference_start_hours,
        reference_end_hours=reference_end_hours,
        reference_statistic=reference_statistic,
        standard_deviation_ddof=standard_deviation_ddof,
        detrended=detrended,
        envelope_floor_fraction=envelope_floor_fraction,
    )
    return {**dict(result.data), "source": "circadian_workbench.analysis.normalize_profile"}


def detrend_settings(
    params: dict | None = None,
    *,
    method: str | None = None,
    window_hours: float | None = None,
    polynomial_degree: int | None = None,
    min_valid_fraction: float | None = None,
    bandwidth_hours: float | None = None,
    low_cut_hours: float | None = None,
    high_cut_hours: float | None = None,
    filter_order: int | None = None,
    lowess_fraction: float | None = None,
    lowess_iterations: int | None = None,
    asls_smoothness: float | None = None,
    asls_asymmetry: float | None = None,
    asls_iterations: int | None = None,
) -> dict[str, Any]:
    """Resolve and validate the shared Circadian Workbench detrending controls."""
    supplied = dict(params or {})
    requested = str(
        method if method is not None
        else supplied.get("detrend", DETREND_DEFAULTS["detrend"])
    )
    if requested not in DETREND_METHODS:
        raise ValueError(
            f"unknown detrend method {requested!r}; available: "
            + ", ".join(DETREND_METHODS)
        )
    window = float(
        window_hours if window_hours is not None
        else supplied.get(
            "detrend_window_hours", DETREND_DEFAULTS["detrend_window_hours"])
    )
    if not np.isfinite(window) or window <= 0:
        raise ValueError("detrend_window_hours must be a finite positive number")
    degree = int(
        polynomial_degree if polynomial_degree is not None
        else supplied.get(
            "detrend_polynomial_degree",
            DETREND_DEFAULTS["detrend_polynomial_degree"],
        )
    )
    if degree < 0:
        raise ValueError("detrend_polynomial_degree must be at least zero")
    valid_fraction = float(
        min_valid_fraction if min_valid_fraction is not None
        else supplied.get(
            "detrend_min_valid_fraction",
            DETREND_DEFAULTS["detrend_min_valid_fraction"],
        )
    )
    if not np.isfinite(valid_fraction) or not 0 <= valid_fraction <= 1:
        raise ValueError("detrend_min_valid_fraction must be between zero and one")
    bandwidth = (
        bandwidth_hours if bandwidth_hours is not None
        else supplied.get("detrend_bandwidth_hours")
    )
    bandwidth = None if bandwidth in (None, "") else float(bandwidth)
    if bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("detrend_bandwidth_hours must be a finite positive number")
    low_cut = float(
        low_cut_hours if low_cut_hours is not None
        else supplied.get(
            "detrend_low_cut_hours", DETREND_DEFAULTS["detrend_low_cut_hours"])
    )
    high_cut = float(
        high_cut_hours if high_cut_hours is not None
        else supplied.get(
            "detrend_high_cut_hours", DETREND_DEFAULTS["detrend_high_cut_hours"])
    )
    if not np.isfinite(low_cut) or low_cut <= 0:
        raise ValueError("detrend_low_cut_hours must be a finite positive number")
    if not np.isfinite(high_cut) or high_cut <= 0:
        raise ValueError("detrend_high_cut_hours must be a finite positive number")
    order = int(
        filter_order if filter_order is not None
        else supplied.get(
            "detrend_filter_order", DETREND_DEFAULTS["detrend_filter_order"])
    )
    if order < 1:
        raise ValueError("detrend_filter_order must be at least one")
    lowess_share = (
        lowess_fraction if lowess_fraction is not None
        else supplied.get("detrend_lowess_fraction")
    )
    lowess_share = None if lowess_share in (None, "") else float(lowess_share)
    if (lowess_share is not None
            and (not np.isfinite(lowess_share) or not 0 < lowess_share <= 1)):
        raise ValueError("detrend_lowess_fraction must be between zero and one")
    lowess_passes = int(
        lowess_iterations if lowess_iterations is not None
        else supplied.get(
            "detrend_lowess_iterations",
            DETREND_DEFAULTS["detrend_lowess_iterations"],
        )
    )
    if not 0 <= lowess_passes <= 20:
        raise ValueError("detrend_lowess_iterations must be between zero and 20")
    smoothness = float(
        asls_smoothness if asls_smoothness is not None
        else supplied.get(
            "detrend_asls_smoothness",
            DETREND_DEFAULTS["detrend_asls_smoothness"],
        )
    )
    if not np.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("detrend_asls_smoothness must be finite and positive")
    asymmetry = float(
        asls_asymmetry if asls_asymmetry is not None
        else supplied.get(
            "detrend_asls_asymmetry",
            DETREND_DEFAULTS["detrend_asls_asymmetry"],
        )
    )
    if not np.isfinite(asymmetry) or not 0 < asymmetry < 0.5:
        raise ValueError("detrend_asls_asymmetry must be between zero and 0.5")
    asls_passes = int(
        asls_iterations if asls_iterations is not None
        else supplied.get(
            "detrend_asls_iterations",
            DETREND_DEFAULTS["detrend_asls_iterations"],
        )
    )
    if not 1 <= asls_passes <= 100:
        raise ValueError("detrend_asls_iterations must be between one and 100")
    return {
        "detrend": requested,
        "detrend_window_hours": window,
        "detrend_polynomial_degree": degree,
        "detrend_min_valid_fraction": valid_fraction,
        "detrend_bandwidth_hours": bandwidth,
        "detrend_low_cut_hours": low_cut,
        "detrend_high_cut_hours": high_cut,
        "detrend_filter_order": order,
        "detrend_lowess_fraction": lowess_share,
        "detrend_lowess_iterations": lowess_passes,
        "detrend_asls_smoothness": smoothness,
        "detrend_asls_asymmetry": asymmetry,
        "detrend_asls_iterations": asls_passes,
    }


def workbench_config(params: dict, hours: Sequence[float], *,
                     detrend: str | None = None,
                     detrend_window_hours: float | None = None,
                     methods: Sequence[str] | None = None) -> dict[str, Any]:
    """Map the rhythm-module settings onto Circadian Workbench's config."""
    low, high = params["period_search_hours"]
    chosen = list(methods if methods is not None else params["period_methods"])
    primary = str(params["primary_rhythm_test"])
    estimator = str(params.get("period_estimation_method", primary))
    unknown = [name for name in chosen if name not in PERIOD_METHODS]
    if unknown:
        raise ValueError(
            f"unknown Circadian Workbench period method {unknown[0]!r}; available: "
            + ", ".join(PERIOD_METHODS)
        )
    if methods is None:
        if primary not in chosen:
            raise ValueError(
                f"primary_rhythm_test {primary!r} is not in period_methods")
        if primary not in SIGNIFICANCE_METHODS:
            raise ValueError(
                f"primary_rhythm_test {primary!r} cannot call a trace rhythmic; "
                "choose one of " + ", ".join(SIGNIFICANCE_METHODS)
            )
        if estimator not in chosen:
            raise ValueError(
                f"period_estimation_method {estimator!r} is not in period_methods")

    detrending = detrend_settings(
        params, method=detrend, window_hours=detrend_window_hours)

    overrides = dict(params.get("workbench_config") or {})
    overrides.update({
        "period_hours": float(params["fixed_period_hours"]),
        "period_min_hours": float(low),
        "period_max_hours": float(high),
        "bin_minutes": max(1, int(round(sample_interval_minutes(hours)))),
        "periodogram_alpha": float(params["rhythmic_alpha"]),
        "jtk_alpha": float(params["rhythmic_alpha"]),
        "period_detrend": detrending["detrend"],
        "period_detrend_window_hours": detrending["detrend_window_hours"],
        "period_detrend_polynomial_degree": detrending["detrend_polynomial_degree"],
        "period_detrend_min_valid_fraction": detrending["detrend_min_valid_fraction"],
        "period_detrend_bandwidth_hours": detrending["detrend_bandwidth_hours"],
        "period_detrend_low_cut_hours": detrending["detrend_low_cut_hours"],
        "period_detrend_high_cut_hours": detrending["detrend_high_cut_hours"],
        "period_detrend_filter_order": detrending["detrend_filter_order"],
        "period_detrend_lowess_fraction": detrending["detrend_lowess_fraction"],
        "period_detrend_lowess_iterations": detrending["detrend_lowess_iterations"],
        "period_detrend_asls_smoothness": detrending["detrend_asls_smoothness"],
        "period_detrend_asls_asymmetry": detrending["detrend_asls_asymmetry"],
        "period_detrend_asls_iterations": detrending["detrend_asls_iterations"],
        "period_method": estimator if estimator in chosen else chosen[0],
        "period_methods": chosen,
        "jtk_periods": list(params["jtk_periods"]),
        "ejtk_permutations": int(params["ejtk_permutations"]),
        "jtk_correction": str(params["jtk_correction"]),
        "jtk_seed": int(params["jtk_seed"]),
        "onset_off_hours": float(params.get("onset_off_hours", 6.0)),
        "onset_on_hours": float(params.get("onset_on_hours", 6.0)),
        "onset_threshold_percent": float(
            params.get("onset_threshold_percent", 80.0)),
    })
    return dict(cw.call("normalize_config", config=overrides).data)


def detrend_trace(hours: Sequence[float], values: Sequence[float], params: dict,
                  *, method: str | None = None,
                  window_hours: float | None = None) -> dict[str, Any]:
    """Detrend through Circadian Workbench and keep its audit metadata."""
    detrending = detrend_settings(
        params, method=method, window_hours=window_hours)
    completed = cw.trace(hours, values).detrend(
        method=detrending["detrend"],
        window_hours=detrending["detrend_window_hours"],
        polynomial_degree=detrending["detrend_polynomial_degree"],
        min_valid_fraction=detrending["detrend_min_valid_fraction"],
        bandwidth_hours=detrending["detrend_bandwidth_hours"],
        low_cut_hours=detrending["detrend_low_cut_hours"],
        high_cut_hours=detrending["detrend_high_cut_hours"],
        filter_order=detrending["detrend_filter_order"],
        lowess_fraction=detrending["detrend_lowess_fraction"],
        lowess_iterations=detrending["detrend_lowess_iterations"],
        asls_smoothness=detrending["detrend_asls_smoothness"],
        asls_asymmetry=detrending["detrend_asls_asymmetry"],
        asls_iterations=detrending["detrend_asls_iterations"],
    )
    data = completed.data
    action_only = {
        "recording", "hours", "raw", "detrended", "smooth_window_hours",
        "exclude", "fit",
    }
    return {**{key: value for key, value in data.items() if key not in action_only},
            "workbench_run_record": completed.run_record}


def scale_detrended(detrended: Sequence[float], raw: Sequence[float]) -> np.ndarray:
    """Detrended values in units of their own spread, with flat traces kept flat."""
    residual = np.asarray(detrended, dtype=float)
    raw_values = np.asarray(raw, dtype=float)
    spread = float(np.nanstd(residual))
    scale = float(np.nanmax(np.abs(raw_values))) or 1.0
    if not np.isfinite(spread) or spread <= scale * 1e-9:
        return np.zeros_like(residual)
    return residual / spread


def daily_measures(hours: Sequence[float], values: Sequence[float], params: dict) -> dict:
    """Daily onset, offset and active duration from Workbench's public action."""
    config = workbench_config(params, hours)
    return dict(cw.trace(hours, values).run("daily_measures", config=config).data)


def daily_profile_metrics(hours: Sequence[float], values: Sequence[float], params: dict) -> dict:
    """Workbench's M10/L5 contrast, timing and interdaily stability only.

    The public nonparametric action works directly on the measured trace. It
    does not fit a cosine, search a period, or invoke the full analysis action.
    These are fixed 24-hour summaries, independent of period-fit settings.
    """
    bin_hours = float(params.get("bin_hours", 1.0))
    if (not np.isfinite(bin_hours) or bin_hours <= 0
            or not np.isclose(24 / bin_hours, round(24 / bin_hours))):
        raise ValueError("daily-profile bin_hours must divide a 24-hour day")
    config = workbench_config(params, hours)
    config["bin_minutes"] = int(round(bin_hours * 60))
    return dict(cw.trace(hours, values).run("nonparametric", config=config).data)


def cosinor(
    hours: Sequence[float],
    values: Sequence[float],
    period_hours: float,
    reference_level: float | None = None,
) -> dict[str, Any]:
    """Fit one fixed-period cosine through Circadian Workbench's public API."""
    if len(hours) < 4:
        return {}
    result = dict(
        cw.trace(hours, values, name="cell trace")
        .cosinor(period_hours=float(period_hours)).data
    )
    mesor = result.get("mesor")
    amplitude = result.get("amplitude")
    if amplitude is None:
        return {}
    denominator = reference_level if reference_level is not None else mesor
    relative = (
        float(amplitude) / abs(float(denominator))
        if denominator is not None and np.isfinite(denominator) and denominator != 0
        else np.nan
    )
    return {
        "cosinor_mesor": _number_or_nan(mesor),
        "cosinor_amplitude": _number_or_nan(amplitude),
        "cosinor_relative_amplitude": relative,
        "cosinor_peak_hour": _number_or_nan(result.get("acrophase_hours")),
        "cosinor_phase_convention": "positive_peak_time",
        "cosinor_r_squared": _number_or_nan(result.get("r_squared")),
        "cosinor_p_value": _number_or_nan(result.get("p_value")),
        # Circadian Workbench reserves relative-amplitude error for FFT-NLLS.
        # Keep the legacy table column explicit and empty for a fixed cosinor.
        "cosinor_relative_amplitude_error": np.nan,
        "workbench_version": WORKBENCH_VERSION,
    }


def descriptive_cosinor_fitted_values(
    hours: Sequence[float], values: Sequence[float], period_hours: float
) -> np.ndarray:
    """Evaluate an explicitly requested fixed-period Workbench cosinor fit."""
    time = np.asarray(hours, dtype=float)
    measured = np.asarray(values, dtype=float)
    fitted = np.full(time.shape, np.nan, dtype=float)
    usable = np.isfinite(time) & np.isfinite(measured)
    if usable.sum() < 4 or not np.isfinite(period_hours) or period_hours <= 0:
        return fitted
    result = dict(
        cw.trace(time[usable], measured[usable], name="display trace")
        .cosinor(period_hours=float(period_hours)).data
    )
    mesor = _number_or_nan(result.get("mesor"))
    amplitude = _number_or_nan(result.get("amplitude"))
    acrophase = _number_or_nan(result.get("acrophase_hours"))
    if not np.isfinite([mesor, amplitude, acrophase]).all():
        return fitted
    fitted[:] = mesor + amplitude * np.cos(
        2 * np.pi * (time - acrophase) / float(period_hours)
    )
    return fitted


def phase_summary(peak_hours: Sequence[float], period_hours: float) -> dict[str, Any]:
    """Summarize phase concentration through Workbench's public phase caller."""
    finite = np.asarray(peak_hours, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return {}
    result = dict(
        cw.phases(finite.tolist(), period_hours=float(period_hours))
        .summary(label="cells").data
    )
    if result.get("status") != "ok":
        return {}
    return {
        "cells": int(result["n"]),
        "vector_length": _number_or_nan(result.get("resultant_length")),
        "mean_peak_hour": _number_or_nan(result.get("mean_hours")),
        "rayleigh_p_value": _number_or_nan(result.get("p_value")),
        "workbench_version": WORKBENCH_VERSION,
    }


def _number_or_nan(value: Any) -> float:
    """One optional Workbench scalar in the project's numeric table shape."""
    return float(value) if value is not None else np.nan


def _daily_summary(days: Sequence[dict[str, Any]], key: str, *, mean: bool = False) -> float:
    """Summarize a Workbench daily field while preserving missing results."""
    values = np.asarray([
        row.get(key) if row.get(key) is not None else np.nan for row in days
    ], dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan
    return float(np.mean(finite) if mean else np.median(finite))


def nonparametric(
    hours: Sequence[float], values: Sequence[float], params: dict
) -> dict[str, Any]:
    """Return the nonparametric and active-phase block from Workbench actions."""
    bin_hours = float(params["bin_hours"])
    bins_per_day = 24.0 / bin_hours if bin_hours > 0 else 0.0
    if bin_hours <= 0 or abs(bins_per_day - round(bins_per_day)) > 1e-9:
        raise ValueError(
            f"rhythms bin_hours must divide a 24-hour day, and {bin_hours!r} "
            "does not"
        )
    if (float(params.get("low_window_hours", 5.0)) != 5.0
            or float(params.get("high_window_hours", 10.0)) != 10.0):
        raise ValueError(
            "Circadian Workbench defines the nonparametric windows as L5 and "
            "M10; low_window_hours must be 5 and high_window_hours must be 10"
        )

    trace = cw.trace(hours, values, name="cell trace")
    config = workbench_config(params, hours)
    config["bin_minutes"] = int(round(bin_hours * 60.0))
    metrics = dict(trace.run("nonparametric", config=config).data)
    daily = dict(trace.run("daily_measures", config=config).data)
    days = list(daily.get("days", []))
    hourly_config = {**config, "bin_minutes": 60}
    hourly = dict(trace.run("time_series", config=hourly_config).data)
    hourly_values = hourly.get("activity", [])
    hours_binned = sum(value is not None for value in hourly_values)

    elapsed = np.asarray(hours, dtype=float)
    span = float(np.nanmax(elapsed) - np.nanmin(elapsed)) if len(elapsed) else np.nan
    days_covered = span / 24.0
    onset = _daily_summary(days, "onset_hours")
    offset = _daily_summary(days, "offset_hours")
    duration = _daily_summary(days, "alpha_hours", mean=True)
    return {
        "m10": _number_or_nan(metrics.get("m10_mean")),
        "m10_onset_hour": _number_or_nan(metrics.get("m10_start_hours")),
        "l5": _number_or_nan(metrics.get("l5_mean")),
        "l5_onset_hour": _number_or_nan(metrics.get("l5_start_hours")),
        "relative_amplitude": _number_or_nan(metrics.get("relative_amplitude")),
        "onset_hour": onset,
        "offset_hour": offset,
        "active_duration_hours": duration,
        "interdaily_stability": _number_or_nan(
            metrics.get("interdaily_stability")),
        "intradaily_variability": _number_or_nan(
            metrics.get("intradaily_variability")),
        "days_covered": days_covered,
        "hours_binned": int(hours_binned),
        "stability_underdetermined": (
            days_covered < float(params["min_days_for_stability"])),
        "onset_found": bool(np.isfinite(onset)),
        "workbench_version": WORKBENCH_VERSION,
    }


def estimate_trace(hours: Sequence[float], values: Sequence[float], params: dict,
                   *, methods: Sequence[str] | None = None,
                   detrend: str | None = None,
                   detrend_window_hours: float | None = None) -> dict[str, Any]:
    """Run the requested Circadian Workbench period methods on one cell trace."""
    trace = cw.trace(hours, values, name="cell trace")
    config = workbench_config(
        params, hours, detrend=detrend,
        detrend_window_hours=detrend_window_hours, methods=methods)
    completed = trace.compare_periods(
        list(methods) if methods else None,
        settings=config,
    )
    comparison = completed.data
    comparison = {
        key: value for key, value in comparison.items()
        if key not in {"recording", "methods_paragraph"}
    }
    rows = []
    primary = str(params["primary_rhythm_test"])
    for returned in comparison["table"]:
        row = dict(returned)
        method_name = str(row["method"])
        row.pop("label", None)
        tests_significance = bool(
            PERIOD_METHODS[method_name]["gives_significance"])
        significant = row.get("significant") if tests_significance else None
        status = str(row.get("status") or "failed")
        if status != "ok" or significant is None:
            rhythm_status = "unknown"
            rhythmic = None
        else:
            rhythmic = bool(significant)
            rhythm_status = "rhythmic" if rhythmic else "arrhythmic"
        row.update({
            "method_label": PERIOD_METHODS[method_name]["label"],
            "is_significance_test": tests_significance,
            "alpha": (float(config["jtk_alpha"])
                      if method_name in ("jtk", "ejtk")
                      else float(config["periodogram_alpha"]))
                     if tests_significance else None,
            "rhythmic": rhythmic,
            "rhythm_status": rhythm_status,
            "primary": method_name == primary,
            "workbench_version": WORKBENCH_VERSION,
        })
        rows.append(row)
    return {"config": config, "rows": rows, "comparison": comparison,
            "result": completed, "run_record": completed.run_record}


def estimate_one(hours: Sequence[float], values: Sequence[float], params: dict,
                 method: str, *, detrend: str | None = None,
                 detrend_window_hours: float | None = None) -> dict[str, Any]:
    """One method through the same public workbench estimator used above."""
    trace = cw.trace(hours, values, name="cell trace")
    config = workbench_config(
        params, hours, detrend=detrend,
        detrend_window_hours=detrend_window_hours, methods=[method])
    completed = trace.period(method, settings=config)
    payload = completed.data
    returned = dict(payload["row"])
    returned.pop("label", None)
    tests_significance = bool(PERIOD_METHODS[method]["gives_significance"])
    significant = returned.get("significant") if tests_significance else None
    status = str(returned.get("status") or "failed")
    rhythmic = bool(significant) if status == "ok" and significant is not None else None
    return {
        **returned,
        "components": payload["estimate"].get("components", []),
        "diagnostics": payload["estimate"].get("diagnostics", {}),
        "method_label": PERIOD_METHODS[method]["label"],
        "is_significance_test": tests_significance,
        "alpha": ((float(config["jtk_alpha"])
                   if method in ("jtk", "ejtk")
                   else float(config["periodogram_alpha"]))
                  if tests_significance else None),
        "rhythmic": rhythmic,
        "rhythm_status": ("unknown" if rhythmic is None else
                          "rhythmic" if rhythmic else "arrhythmic"),
        "primary": method == str(params["primary_rhythm_test"]),
        "workbench_version": WORKBENCH_VERSION,
        "workbench_run_record_json": json.dumps(completed.run_record, allow_nan=False),
    }


def lomb_periodogram(
    hours: Sequence[float],
    values: Sequence[float],
    params: dict,
    *,
    detrend: str | None = None,
    detrend_window_hours: float | None = None,
) -> dict[str, Any]:
    """One Lomb-Scargle result through Workbench's public single-method route."""
    time = np.asarray(hours, dtype=float)
    observed = np.asarray(values, dtype=float)
    keep = np.isfinite(time) & np.isfinite(observed)
    time, observed = time[keep], observed[keep]
    order = np.argsort(time, kind="mergesort")
    time, observed = time[order], observed[order]
    if len(time) < 6 or np.allclose(observed, observed[0]):
        return {
            "method": "lomb", "status": "failed", "p_value": np.nan,
            "period_hours": np.nan, "significant": None,
            "rhythm_status": "unknown", "workbench_version": WORKBENCH_VERSION,
        }

    estimate = estimate_one(
        time, observed, params, "lomb", detrend=detrend,
        detrend_window_hours=detrend_window_hours)
    return {
        **estimate,
        "peak_power": estimate.get("goodness_of_fit", np.nan),
        "threshold": np.nan,
    }


def estimate_grouped_rhythms(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    params: dict,
    time_column: str = "hours",
    method: str = "lomb",
    significance_method: str | None = None,
    detrend: str | None = None,
    detrend_window_hours: float | None = None,
    min_observations: int = 24,
    correction: str = "bh",
    min_cycles: float = 3.0,
) -> pd.DataFrame:
    """Broad-period rhythm results for any collection of grouped traces.

    This is deliberately independent of radial occupancy.  A caller supplies
    the grouping columns and measurement column, so the same tested matrix can
    represent cells by radial band, genes by condition, or any other family of
    traces.  Period selection and raw significance come from Circadian
    Workbench; its batch correction supplies the final per-family verdict.
    """
    missing = [name for name in [*group_columns, time_column, value_column]
               if name not in frame.columns]
    if missing:
        raise KeyError("grouped rhythm input is missing " + ", ".join(missing))
    significance_method = resolve_significance_method(method, significance_method)
    params = {**params, "primary_rhythm_test": significance_method}
    if int(min_observations) < 6:
        raise ValueError("min_observations must be at least 6")
    if float(min_cycles) <= 0:
        raise ValueError("min_cycles must be positive")
    if correction not in {"bh", "bonferroni", "sidak", "none"}:
        raise ValueError(
            "multiple-testing correction must be bh, bonferroni, sidak or none"
        )

    detrending = detrend_settings(
        params, method=detrend, window_hours=detrend_window_hours)
    low, high = map(float, params["period_search_hours"])
    edge_tolerance = max(0.1, 0.005 * (high - low))
    rows: list[dict[str, Any]] = []
    grouping = list(group_columns)
    grouper: Any = grouping[0] if len(grouping) == 1 else grouping
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(grouping) == 1 else tuple(key)
        identity = dict(zip(grouping, keys))
        usable = (
            group[[time_column, value_column]].dropna()
            .sort_values(time_column, kind="mergesort")
        )
        time = usable[time_column].to_numpy(float)
        values = usable[value_column].to_numpy(float)
        span = float(time.max() - time.min()) if len(time) else 0.0
        base = {
            **identity,
            "metric": value_column,
            "observations": int(len(usable)),
            "span_hours": span,
            "input_mean": float(np.mean(values)) if len(values) else np.nan,
            "input_sd": float(np.std(values)) if len(values) else np.nan,
            "method": method,
            "significance_method": significance_method,
            "estimate_status": "not_tested",
            "estimate_reason": "",
            "cycles_observed": np.nan,
            "period_underdetermined": True,
            "period_at_search_edge": False,
            "estimator_p_value": np.nan,
            "correction": correction,
            "alpha": float(params["rhythmic_alpha"]),
            "period_search_min_hours": low,
            "period_search_max_hours": high,
            **detrending,
            "workbench_version": WORKBENCH_VERSION,
        }
        if len(usable) < int(min_observations):
            rows.append({
                **base, "test_status": "not_tested", "reason": "too_few_observations",
                "period_hours": np.nan, "p_value": np.nan,
            })
            continue
        if np.allclose(values, values[0]):
            rows.append({
                **base, "test_status": "not_tested", "reason": "no_occupancy_variation",
                "period_hours": np.nan, "p_value": np.nan,
            })
            continue

        try:
            estimate = estimate_one(
                time, values, params, method,
                detrend=detrending["detrend"],
                detrend_window_hours=detrending["detrend_window_hours"])
        except cw.WorkbenchError as error:
            estimate = {
                "status": "failed",
                "diagnostics": {"reason": str(error)},
            }
        period = estimate.get("period_hours")
        period = float(period) if period is not None and np.isfinite(period) else np.nan
        if method == significance_method:
            evidence = estimate
        else:
            try:
                evidence = estimate_one(
                    time, values, params, significance_method,
                    detrend=detrending["detrend"],
                    detrend_window_hours=detrending["detrend_window_hours"])
            except cw.WorkbenchError as error:
                evidence = {
                    "status": "failed",
                    "diagnostics": {"reason": str(error)},
                }
        p_value = evidence.get("p_value")
        p_value = float(p_value) if p_value is not None and np.isfinite(p_value) else np.nan
        status = str(evidence.get("status") or "failed")
        evidence_reason = str(
            evidence.get("diagnostics", {}).get("reason") or "workbench_failed"
        )
        cycles = span / period if np.isfinite(period) and period > 0 else np.nan
        rows.append({
            **base,
            "test_status": "ok" if status == "ok" and np.isfinite(p_value) else "not_tested",
            "reason": "" if status == "ok" and np.isfinite(p_value) else evidence_reason,
            "period_hours": period,
            "estimate_status": str(estimate.get("status") or "failed"),
            "estimate_reason": estimate.get("diagnostics", {}).get("reason", ""),
            "components": estimate.get("components", []),
            "estimate_diagnostics": estimate.get("diagnostics", {}),
            "workbench_run_record_json": estimate.get("workbench_run_record_json", ""),
            "significance_run_record_json": evidence.get("workbench_run_record_json", ""),
            "estimator_p_value": estimate.get("p_value"),
            "significance_period_hours": evidence.get("period_hours"),
            "period_difference_hours": (
                period - float(evidence["period_hours"])
                if evidence.get("period_hours") is not None else np.nan),
            **{key: estimate.get(key) for key in (
                "period_error_hours", "phase_hours", "phase_error_hours",
                "amplitude", "amplitude_error", "rae", "goodness_of_fit")},
            "p_value": p_value,
            "periodogram_significant": evidence.get("significant"),
            "peak_power": evidence.get("peak_power", evidence.get("goodness_of_fit", np.nan)),
            "threshold": evidence.get("threshold", np.nan),
            "detrend": estimate.get("detrend", detrending["detrend"]),
            "cycles_observed": cycles,
            "period_underdetermined": bool(
                not np.isfinite(cycles) or cycles < float(min_cycles)),
            "period_at_search_edge": bool(
                np.isfinite(period)
                and (period <= low + edge_tolerance or period >= high - edge_tolerance)
            ),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    raw = pd.to_numeric(result["p_value"], errors="coerce").to_numpy(float)
    adjusted = adjust_pvalues(raw, correction)
    result["q_value"] = adjusted
    tested = result["test_status"].eq("ok") & np.isfinite(adjusted)
    significant = tested & (adjusted < float(params["rhythmic_alpha"]))
    result["significant"] = significant
    result["rhythm_status"] = np.select(
        [~tested, significant], ["not tested", "rhythmic"],
        default="not rhythmic",
    )
    result["family_tests"] = int(tested.sum())
    result["period_available"] = (
        result["estimate_status"].eq("ok") & np.isfinite(result["period_hours"]))
    return result


def adjust_pvalues(values: Sequence[float], correction: str) -> np.ndarray:
    """Correct one p-value family through Workbench's public statistics API."""
    raw = np.asarray(values, dtype=float)
    adjusted = cw.statistics.adjust_pvalues(raw, correction)
    return np.asarray(adjusted, dtype=float)


def resolve_significance_method(method: str, significance_method: str | None = None) -> str:
    """Name the statistical test separately when the period fitter has none."""
    if method not in PERIOD_METHODS:
        raise ValueError(f"unknown period method {method!r}")
    selected = significance_method or (method if method in SIGNIFICANCE_METHODS else "lomb")
    if selected not in SIGNIFICANCE_METHODS:
        raise ValueError(
            f"{selected!r} cannot test rhythmicity; choose " + ", ".join(SIGNIFICANCE_METHODS))
    return selected


def fft_nlls_fitted_values(hours: Sequence[float], estimate: dict) -> np.ndarray:
    """Evaluate the returned multi-component model; never refit a substitute cosine."""
    diagnostics = estimate.get("diagnostics") or {}
    components = estimate.get("components") or []
    if (estimate.get("method") != "fft_nlls" or diagnostics.get("status") != "ok"
            or not components or not diagnostics.get("phase_zero_timestamp")):
        return np.full(len(hours), np.nan)
    # TraceData's documented arbitrary epoch is 2000-01-01. Component phases
    # stay in physical hours relative to the model's recorded midnight origin.
    origin = (pd.Timestamp(diagnostics["phase_zero_timestamp"]) -
              pd.Timestamp("2000-01-01")).total_seconds() / 3600
    local = np.asarray(hours, float) - origin
    fitted = np.full(local.shape, float(diagnostics["mesor"]))
    for component in components:
        fitted += float(component["amplitude"]) * np.cos(
            2 * np.pi * (local - float(component["phase_hours"])) /
            float(component["period_hours"]))
    return fitted
