"""Circadian Workbench is the rhythm engine, not a source copied by hand."""

from __future__ import annotations

import json
from copy import deepcopy
import circadian_workbench as cw
import numpy as np
import pandas as pd
import pytest

from analysis import circadian
from analysis.modules.rhythms import DEFAULTS, derive
from analysis.registry import MeasurementContext
from analysis.units import Scale


def _wave(period: float = 27.0, days: float = 8.0,
          interval: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    hours = np.arange(0.0, days * 24.0, interval)
    values = 100.0 + 12.0 * np.sin(2.0 * np.pi * hours / period) + 0.02 * hours
    return hours, values


def _params(**overrides) -> dict:
    return {**DEFAULTS, "metrics": ["corrected_mean"],
            "null_surrogates_per_cell": 0, **overrides}


def _context(params: dict) -> MeasurementContext:
    return MeasurementContext(
        stem="synthetic",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        raw=np.zeros((2, 2, 2), dtype=np.uint16),
        scale=Scale(60.0), identities=[1], params={"rhythms": params},
    )


def test_every_declared_detrend_is_run_by_circadian_workbench() -> None:
    required = {
        "none", "running_mean", "linear", "polynomial", "kernel",
        "amp_baseline", "frequency", "cubic", "bicubic", "poly3",
        "poly6", "degree6", "baseline", "amp&baseline",
        "lowess", "loess", "first_difference", "first-difference",
        "difference", "diff", "moving_median", "running_median", "median",
        "savitzky_golay", "savitzky-golay", "savgol", "robust_linear",
        "robust", "huber", "asymmetric_least_squares",
        "asymmetric-least-squares", "asls", "als",
    }
    assert required <= set(circadian.DETREND_METHODS)
    hours, values = _wave()
    for method in circadian.DETREND_METHODS:
        result = circadian.detrend_trace(
            hours, values, _params(detrend=method))
        assert len(result["values"]) == len(values)
        assert result["method"] in circadian.DETREND_METHODS


def test_one_window_control_reaches_every_workbench_detrend() -> None:
    hours, values = _wave()
    settings = circadian.detrend_settings(
        _params(), method="running_mean", window_hours=7.5)
    result = circadian.detrend_trace(hours, values, settings)

    assert settings["detrend"] == "running_mean"
    assert settings["detrend_window_hours"] == 7.5
    assert result["window_hours"] == 7.5

    with pytest.raises(ValueError, match="finite positive"):
        circadian.detrend_settings(_params(), window_hours=0)


def test_poly6_bicubic_and_method_specific_controls_reach_the_workbench() -> None:
    hours, values = _wave()
    poly6 = circadian.detrend_trace(
        hours, values, _params(detrend="poly6"))
    explicit = circadian.detrend_trace(
        hours, values,
        _params(detrend="polynomial", detrend_polynomial_degree=6),
    )
    bicubic = circadian.detrend_trace(
        hours, values, _params(detrend="bicubic"))
    cubic = circadian.detrend_trace(
        hours, values, _params(detrend="cubic"))
    np.testing.assert_allclose(poly6["values"], explicit["values"])
    np.testing.assert_allclose(bicubic["values"], cubic["values"])
    assert poly6["polynomial_degree"] == 6
    assert bicubic["polynomial_degree"] == 3

    settings = circadian.detrend_settings(_params(
        detrend="frequency",
        detrend_min_valid_fraction=0.7,
        detrend_bandwidth_hours=5.0,
        detrend_low_cut_hours=36.0,
        detrend_high_cut_hours=6.0,
        detrend_filter_order=3,
        detrend_lowess_fraction=0.25,
        detrend_lowess_iterations=1,
        detrend_asls_smoothness=50_000.0,
        detrend_asls_asymmetry=0.05,
        detrend_asls_iterations=4,
    ))
    assert settings["detrend_min_valid_fraction"] == 0.7
    assert settings["detrend_bandwidth_hours"] == 5.0
    assert settings["detrend_low_cut_hours"] == 36.0
    assert settings["detrend_high_cut_hours"] == 6.0
    assert settings["detrend_filter_order"] == 3
    assert settings["detrend_lowess_fraction"] == 0.25
    assert settings["detrend_lowess_iterations"] == 1
    assert settings["detrend_asls_smoothness"] == 50_000.0
    assert settings["detrend_asls_asymmetry"] == 0.05
    assert settings["detrend_asls_iterations"] == 4

    lowess = circadian.detrend_trace(
        hours, values, {**settings, "detrend": "lowess"})
    assert lowess["lowess_fraction"] == 0.25
    assert lowess["lowess_iterations"] == 1

    asls = circadian.detrend_trace(
        hours, values,
        {**settings, "detrend": "asymmetric_least_squares"},
    )
    assert asls["asls_smoothness"] == 50_000.0
    assert asls["asls_asymmetry"] == 0.05
    assert asls["asls_iterations"] == 4


def test_an_unknown_detrend_is_refused_with_the_available_choices() -> None:
    hours, values = _wave()
    with pytest.raises(ValueError, match="available: none, running_mean"):
        circadian.detrend_trace(
            hours, values, _params(detrend="automatic"))


def test_default_periodograms_each_return_their_own_verdict_and_period() -> None:
    hours, values = _wave()
    result = circadian.estimate_trace(hours, values, _params())
    rows = {row["method"]: row for row in result["rows"]}

    assert set(rows) == {"lomb", "chi_square", "f"}
    for row in rows.values():
        assert row["is_significance_test"]
        assert row["rhythm_status"] == "rhythmic"
        assert row["rhythmic"] is True
        assert row["p_value"] < row["alpha"]
    # Lomb-Scargle is the primary method because it tolerates missing frames;
    # fold-and-compare methods can prefer a harmonic on a broad search.
    assert rows["lomb"]["period_hours"] == pytest.approx(27.0, abs=0.6)


@pytest.mark.parametrize("method", [entry["key"] for entry in cw.call("period_methods").data["methods"]])
def test_every_workbench_period_method_is_discovered_and_runnable(method) -> None:
    expected = {entry["key"] for entry in cw.call("period_methods").data["methods"]}
    assert set(circadian.PERIOD_METHODS) == expected
    hours, values = _wave(period=24.0, days=7.0, interval=0.5)
    result = circadian.estimate_one(
        hours, values,
        _params(
            jtk_periods=[20.0, 24.0, 28.0], ejtk_permutations=25,
            workbench_config={
                "nlls_max_components": 2, "mesa_model_length": 20,
                "mfourfit_harmonics": 2, "mfourfit_step_hours": 0.2,
                "sr_iterations": 25, "sr_grid_points": 64, "sr_seed": 19,
                "jtk_max_points": 48,
            },
        ),
        method, detrend="none",
    )

    assert result["method"] == method
    assert result["status"] == "ok"
    assert result["period_hours"] == pytest.approx(24.0, abs=0.7)


def test_uniform_trace_uses_the_exact_workbench_caller_and_defaults():
    assert circadian.trace is cw.trace
    assert circadian.describe is cw.describe
    hours, values = _wave()
    result = circadian.trace(hours, values).compare_periods()
    direct = cw.trace(hours, values).compare_periods()
    assert result.data == direct.data
    assert result.run_record["inputs"]["config"] == direct.run_record["inputs"]["config"]
    assert result.plot().definition.as_dict() == direct.plot().definition.as_dict()
    # The explicitly configured measurement workflow is a separate contract.
    assert DEFAULTS["period_search_hours"] == [2.0, 48.0]
    assert DEFAULTS["rhythmic_alpha"] == 0.05


def test_argument_groups_and_options_reference_live_workbench_definitions():
    description = cw.describe("compare_periods")
    references = next(row["references"] for row in description["params"] if row["name"] == "config")
    group = circadian.argument_group()
    assert set(group) == set(references)
    assert group == {row["key"]: row for row in description["config_arguments"] if row["key"] in references}
    options = circadian.scientific_options()
    for name, spec in options.items():
        if spec["reference"].startswith("config."):
            parent = group[spec["reference"].removeprefix("config.")]
            for key in ("description", "units", "default", "minimum", "maximum", "aliases"):
                assert spec[key] == parent[key], (name, key)
    group["period_methods"]["default"].append("not-a-method")
    assert "not-a-method" not in circadian.argument_group()["period_methods"]["default"]


def test_uniform_explicit_settings_and_completed_figures_ignore_prior_calls():
    hours, values = _wave(days=3)
    requested = {"period_min_hours": 2, "period_max_hours": 48,
                 "period_detrend_lowess_iterations": 0, "sr_trim_high_frequency": False}
    before = deepcopy(requested)
    result = circadian.trace(hours, values, value_label="Cell area", value_unit="um2",
                             settings=requested).detrend(method="linear").compare_periods()
    figure = result.plot(theme="classic")
    snapshot = figure.definition.as_dict()
    record = result.run_record
    requested["period_min_hours"] = 40
    np.random.seed(829)
    np.random.random(100)
    circadian.trace(hours, values).period().plot(theme="pyflash")
    result.data["table"].clear()
    result.provenance.clear()
    repeated = circadian.trace(hours, values, value_label="Cell area", value_unit="um2",
                               settings=before).detrend(method="linear").compare_periods()
    assert repeated.run_record["result_sha256"] == record["result_sha256"]
    assert repeated.plot(theme="classic").definition.as_dict() == snapshot
    assert result.plot(theme="classic").definition.as_dict() == snapshot
    assert record["inputs"]["config"]["period_detrend_lowess_iterations"] == 0
    assert record["inputs"]["config"]["sr_trim_high_frequency"] is False


def test_legacy_result_records_keep_exact_effective_settings_and_diagnostics():
    hours, values = _wave(period=12, days=3)
    params = _params(period_methods=["lomb", "mesa"], workbench_config={"mesa_model_length": 12})
    compared = circadian.estimate_trace(hours, values, params)
    config = compared["config"]
    direct = cw.trace(hours, values, name="cell trace").compare_periods(settings=config)
    assert compared["comparison"]["table"] == direct.data["table"]
    assert compared["run_record"]["inputs"]["config"] == config
    single = circadian.estimate_one(hours, values, params, "mesa")
    record = json.loads(single["workbench_run_record_json"])
    direct_single = cw.trace(hours, values, name="cell trace").period("mesa", settings=record["inputs"]["config"])
    assert single["diagnostics"] == direct_single.data["estimate"]["diagnostics"]
    assert single["components"] == direct_single.data["estimate"]["components"]
    assert record["inputs"]["config"]["period_min_hours"] == 2
    assert record["inputs"]["config"]["period_max_hours"] == 48


def test_rhythm_output_rows_retain_the_completed_workbench_record():
    hours, values = _wave(period=12, days=3)
    frame = pd.DataFrame({"identity": 1, "hours": hours, "corrected_mean": values})
    tables = derive(frame, _context(_params()))
    for encoded in tables["rhythm_methods"]["workbench_run_record_json"]:
        record = json.loads(encoded)
        assert record["action"] == "compare_periods"
        assert record["recordings"]
        assert record["environment"]["versions"]["circadian-workbench"] == cw.__version__


def test_matrix_lomb_route_matches_the_public_workbench_estimator() -> None:
    """The generic matrix route must be the public single-method estimator."""
    hours, values = _wave(period=13.0, days=3.0)
    params = _params(period_methods=["lomb"], primary_rhythm_test="lomb")
    public = circadian.estimate_one(hours, values, params, "lomb")
    matrix = circadian.lomb_periodogram(hours, values, params)

    assert matrix["period_hours"] == pytest.approx(public["period_hours"])
    assert matrix["p_value"] == pytest.approx(public["p_value"])
    assert matrix["significant"] is public["significant"]


def test_grouped_rhythm_table_corrects_one_family_and_keeps_untestable_rows() -> None:
    hours, rhythmic = _wave(period=10.0, days=4.0)
    flat = np.full_like(rhythmic, rhythmic.mean())
    frame = pd.concat([
        pd.DataFrame({"identity": 1, "ring": 0, "hours": hours,
                      "occupancy": rhythmic}),
        pd.DataFrame({"identity": 1, "ring": 1, "hours": hours,
                      "occupancy": flat}),
    ], ignore_index=True)
    result = circadian.estimate_grouped_rhythms(
        frame, group_columns=["identity", "ring"], value_column="occupancy",
        params=_params(period_methods=["lomb"], primary_rhythm_test="lomb"),
        correction="bh",
    ).set_index("ring")

    assert result.loc[0, "period_hours"] == pytest.approx(10.0, abs=0.3)
    assert bool(result.loc[0, "significant"])
    assert result.loc[0, "family_tests"] == 1
    assert result.loc[1, "rhythm_status"] == "not tested"
    assert result.loc[1, "reason"] == "no_occupancy_variation"


def test_grouped_rhythm_table_keeps_other_cells_when_workbench_rejects_one(
    monkeypatch,
) -> None:
    from circadian_workbench.contracts import WorkbenchInputError

    hours, values = _wave(period=10.0, days=4.0)
    frame = pd.concat([
        pd.DataFrame({"identity": 1, "hours": hours, "value": values}),
        pd.DataFrame({"identity": 2, "hours": hours, "value": values + 50.0}),
    ], ignore_index=True)

    def one_fit(time, observed, params, method, **kwargs):
        if float(np.mean(observed)) < 140.0:
            raise WorkbenchInputError("selected window has no finite samples")
        return {
            "status": "ok", "period_hours": 10.0, "p_value": 0.001,
            "significant": True, "diagnostics": {},
        }

    monkeypatch.setattr(circadian, "estimate_one", one_fit)
    result = circadian.estimate_grouped_rhythms(
        frame, group_columns=["identity"], value_column="value",
        params=_params(period_methods=["lomb"], primary_rhythm_test="lomb"),
        correction="none",
    ).set_index("identity")

    assert result.loc[1, "rhythm_status"] == "not tested"
    assert "no finite samples" in result.loc[1, "reason"]
    assert result.loc[2, "rhythm_status"] == "rhythmic"


@pytest.mark.parametrize("period", [6.0, 36.0])
def test_default_search_finds_ultradian_and_infradian_periods(period: float) -> None:
    hours, values = _wave(period=period)
    result = circadian.estimate_trace(hours, values, _params())
    primary = next(row for row in result["rows"] if row["primary"])

    assert DEFAULTS["period_search_hours"] == [2.0, 48.0]
    assert primary["rhythmic"] is True
    assert primary["period_hours"] == pytest.approx(period, abs=0.6)


def test_the_primary_workbench_test_not_cosinor_classifies_the_cell() -> None:
    hours, values = _wave()
    frame = pd.DataFrame({
        "identity": 1,
        "frame_index": np.arange(len(hours)),
        "hours": hours,
        "corrected_mean": values,
    })
    outputs = derive(frame, _context(_params()))
    summary = outputs["rhythms"].iloc[0]
    methods = outputs["rhythm_methods"]
    traces = outputs["rhythm_traces"]

    lomb = methods[methods["method"] == "lomb"].iloc[0]
    assert summary["primary_rhythm_test"] == "lomb"
    assert bool(summary["rhythmic"]) == bool(lomb["method_rhythmic"])
    assert summary["best_period_hours"] == pytest.approx(lomb["period_hours"])
    assert summary["best_p_value"] == pytest.approx(lomb["p_value"])
    assert summary["best_phase_fraction"] == pytest.approx(
        summary["best_phase_hours"] / summary["best_period_hours"]
    )
    assert summary["best_period_class"] == "circadian-like"
    assert len(methods) == 3
    assert len(traces) == len(frame)
    assert traces["detrended_z"].std(ddof=0) == pytest.approx(1.0)


def test_a_period_estimator_without_significance_never_invents_a_verdict() -> None:
    hours, values = _wave(days=5.0)
    result = circadian.estimate_trace(
        hours, values, _params(), methods=["mesa"])
    row = result["rows"][0]
    assert not row["is_significance_test"]
    assert row["rhythmic"] is None
    assert row["rhythm_status"] == "unknown"


def test_the_primary_method_must_be_a_configured_significance_test() -> None:
    hours, values = _wave()
    with pytest.raises(ValueError, match="cannot call a trace rhythmic"):
        circadian.estimate_trace(
            hours, values,
            _params(period_methods=["mesa"], primary_rhythm_test="mesa"))


def test_the_user_can_choose_which_significance_test_supplies_the_verdict() -> None:
    hours, values = _wave()
    result = circadian.estimate_trace(
        hours, values,
        _params(period_methods=["lomb", "chi_square"],
                primary_rhythm_test="chi_square"))
    primary = [row for row in result["rows"] if row["primary"]]
    assert len(primary) == 1
    assert primary[0]["method"] == "chi_square"


def test_period_estimator_is_independent_of_the_rhythm_test() -> None:
    hours, values = _wave(period=27.0)
    frame = pd.DataFrame({
        "identity": 1,
        "frame_index": np.arange(len(hours)),
        "hours": hours,
        "corrected_mean": values,
    })
    params = _params(
        period_methods=["fft_nlls", "lomb"],
        period_estimation_method="fft_nlls",
        primary_rhythm_test="lomb",
        workbench_config={"nlls_max_components": 2},
    )
    outputs = derive(frame, _context(params))
    summary = outputs["rhythms"].iloc[0]
    methods = outputs["rhythm_methods"]

    assert summary["period_estimation_method"] == "fft_nlls"
    assert summary["primary_rhythm_test"] == "lomb"
    assert summary["best_period_hours"] == pytest.approx(27.0, abs=0.1)
    assert methods.loc[methods["period_estimator"], "method"].tolist() == ["fft_nlls"]
    assert methods.loc[methods["primary"], "method"].tolist() == ["lomb"]
    assert bool(summary["rhythmic"]) is True


def test_unselected_lomb_and_legacy_cosinor_are_not_run_implicitly() -> None:
    hours, values = _wave(period=18.0, days=5.0)
    frame = pd.DataFrame({
        "identity": 1,
        "frame_index": np.arange(len(hours)),
        "hours": hours,
        "corrected_mean": values,
    })
    outputs = derive(frame, _context(_params(
        period_methods=["f"],
        period_estimation_method="f",
        primary_rhythm_test="f",
        descriptive_cosinor=False,
    )))

    summary = outputs["rhythms"].iloc[0]
    assert outputs["rhythm_methods"]["method"].tolist() == ["f"]
    assert summary["period_estimation_method"] == "f"
    assert "cosinor_peak_hour" not in outputs["rhythms"]
    assert summary["rhythmic_cosinor"] is None
