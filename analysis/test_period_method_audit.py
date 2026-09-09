"""Regression checks for the reusable period-method audit figure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import circadian as workbench


FIGURES = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(FIGURES))

from _schema import load_all  # noqa: E402


def _builder_globals() -> dict:
    return load_all()["period-method-audit"].build.__globals__


def _params() -> dict:
    return {
        **workbench.PERIOD_ANALYSIS_DEFAULTS,
        **workbench.DETREND_DEFAULTS,
        "period_search_hours": [2.0, 48.0],
        "workbench_config": {},
    }


def _two_length_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for identity, stop in ((7, 72.0), (9, 36.0)):
        hours = np.arange(0.0, stop, 1.0)
        values = np.sin(2.0 * np.pi * hours / 12.0)
        for hour, value in zip(hours, values):
            rows.append({
                "identity": identity,
                "hours": float(hour),
                "corrected_mean": float(value),
            })
    return pd.DataFrame(rows)


def test_audit_exposes_the_complete_shared_circadian_contract():
    spec = load_all()["period-method-audit"]
    declared = {option.name for option in spec.options}
    assert spec.purpose == "review"
    assert set(workbench.CIRCADIAN_ANALYSIS_OPTIONS) <= declared
    assert {"identity", "cells", "metrics", "detrend_methods",
            "period_methods", "median_window_points"} <= declared
    detrend_default = next(
        option.default for option in spec.options if option.name == "detrend_methods"
    )
    assert "robust_linear" in detrend_default


def test_audit_keeps_recording_lengths_and_cycle_sufficiency_trace_specific():
    analyse = _builder_globals()["analyse_traces"]
    traces, results, components, spectra, evidence = analyse(
        _two_length_frame(),
        identities=[7, 9],
        metrics=["corrected_mean"],
        detrends=["linear"],
        estimators=["lomb", "fft_nlls"],
        median_window_points=3,
        params=_params(),
        fallback_significance_method="lomb",
        correction="none",
        alpha=0.05,
        min_observations=24,
        min_cycles=3.0,
    )

    assert set(traces["trace_id"]) == {"7|corrected_mean", "9|corrected_mean"}
    spans = results.groupby("identity")["span_hours"].unique().map(lambda x: x[0])
    assert spans.to_dict() == {7: 71.0, 9: 35.0}
    assert len(results) == 2 * 2 * 1 * 2
    assert len(evidence) == 2 * 2 * 1
    assert set(results["workbench_version"]) == {workbench.WORKBENCH_VERSION}
    assert set(results["min_cycles"]) == {3.0}
    assert (components["component_significance"] == "not tested").all()
    assert set(spectra.columns) == {
        "trace_id", "identity", "metric", "preprocessor", "detrend",
        "estimator", "period_hours", "power", "normalised_power",
    }


def test_median_prefilter_is_a_visible_sensitivity_input_not_raw_replacement():
    analyse = _builder_globals()["analyse_traces"]
    frame = _two_length_frame().query("identity == 7").copy()
    frame.loc[frame.index[10], "corrected_mean"] = 100.0
    traces, *_ = analyse(
        frame,
        identities=[7],
        metrics=["corrected_mean"],
        detrends=["linear"],
        estimators=["lomb"],
        median_window_points=3,
        params=_params(),
        fallback_significance_method="lomb",
        correction="none",
        alpha=0.05,
        min_observations=24,
        min_cycles=3.0,
    )

    detrended = traces.loc[traces["display_panel"].eq("detrended")]
    raw = detrended.loc[detrended["preprocessor"].eq("raw")]
    median = detrended.loc[detrended["preprocessor"].eq("median")]
    assert raw["raw_value"].max() == 100.0
    assert raw["analysis_input_value"].max() == 100.0
    assert median["raw_value"].max() == 100.0
    assert median["analysis_input_value"].max() < 2.0
