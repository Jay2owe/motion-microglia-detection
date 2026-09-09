"""Regression checks for the reusable period-method audit figure."""

from __future__ import annotations

import json
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_common_comparison_is_the_completed_workbench_builder_not_a_refit(monkeypatch):
    import matplotlib.pyplot as plt
    from panels import rhythm_audit

    captured = []
    original = workbench.estimate_trace
    def compare(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(workbench, "estimate_trace", compare)
    params = _params()
    params.update(ejtk_permutations=25, jtk_periods=[8, 12, 16], jtk_seed=19,
                  workbench_config={"nlls_max_components": 2, "mesa_model_length": 12,
                                    "jtk_max_points": 48})
    frame = _two_length_frame().query("identity == 7")
    _, results, _, _, _ = _builder_globals()["analyse_traces"](
        frame, identities=[7], metrics=["corrected_mean"], detrends=["linear"],
        estimators=["lomb", "mesa", "fft_nlls", "ejtk"], median_window_points=3,
        params=params, fallback_significance_method="lomb", correction="none",
        alpha=.05, min_observations=24, min_cycles=3,
    )
    assert len(captured) == 2
    raw = results[results["preprocessor"].eq("raw")]
    completed = captured[0]["result"]
    definition = json.loads(raw.iloc[0]["workbench_comparison_json"])
    assert definition == completed.plot().definition.as_dict()
    assert json.loads(raw.iloc[0]["workbench_run_record_json"]) == completed.run_record
    hours, values = frame["hours"].to_numpy(), frame["corrected_mean"].to_numpy()
    for row in raw.to_dict("records"):
        old = workbench.estimate_one(hours, values, params, row["estimator"], detrend="linear")
        assert row["estimated_period_hours"] == pytest.approx(old["period_hours"], rel=1e-12)
        assert row["estimator_p_value"] == pytest.approx(old["p_value"]) if old["p_value"] is not None else pd.isna(row["estimator_p_value"])

    def no_fit(*args, **kwargs):
        pytest.fail("Drawing a completed comparison must not run science")
    monkeypatch.setattr(workbench, "estimate_trace", no_fit)
    monkeypatch.setattr(workbench, "estimate_one", no_fit)
    monkeypatch.setattr(workbench.cw, "trace", no_fit)
    figure, axis = plt.subplots(figsize=(16, 4))
    try:
        rhythm_audit.draw_workbench_comparison(axis, definition)
        assert len(axis.tables) == 1
        style = definition["options"]["display"]["theme_values"]
        cell = axis.tables[0].get_celld()[(1, 1)]
        assert cell.get_text().get_text() == "{:.2f}".format(raw.iloc[0]["estimated_period_hours"])
        assert cell.get_text().get_fontsize() == style["tick_size"] * .75
        assert cell.get_text().get_color() == style["tick_fill"]
        rendered = io.BytesIO()
        figure.savefig(rendered, format="svg")
        assert b"<svg" in rendered.getvalue()
    finally:
        plt.close(figure)


def test_audit_does_not_run_fallback_significance_below_observation_minimum(monkeypatch):
    def no_fit(*args, **kwargs):
        pytest.fail("An undersampled trace must not run an estimator or fallback test")
    monkeypatch.setattr(workbench, "estimate_trace", no_fit)
    monkeypatch.setattr(workbench, "estimate_one", no_fit)
    _, results, _, _, evidence = _builder_globals()["analyse_traces"](
        _two_length_frame().query("identity == 7").head(8),
        identities=[7], metrics=["corrected_mean"], detrends=["linear"],
        estimators=["mesa"], median_window_points=3, params=_params(),
        fallback_significance_method="lomb", correction="none", alpha=.05,
        min_observations=24, min_cycles=3,
    )
    assert set(results["estimate_status"]) == {"not_tested"}
    assert set(evidence["status"]) == {"not_tested"}
    assert set(results["workbench_comparison_json"]) == {""}
