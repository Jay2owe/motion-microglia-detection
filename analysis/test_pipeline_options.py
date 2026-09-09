"""Pipeline requests preserve scientific choices before any analysis runs."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.config import AnalysisConfig
from analysis.pipelines import parse
from analysis.pipelines.contracts import (CellKey, Settings, StepResult, content_id)
from analysis.pipelines.rhythm_discovery import resolve_request


def request(**options):
    return parse([{"pipeline": "rhythm-discovery",
                   "test_measurements": ["corrected_mean"], **options}])[0]


@pytest.fixture
def tables():
    # Missing hours are gaps, not an instruction to compress recording time.
    return {
        "cell_frame": pd.DataFrame({
            "stem": ["a"] * 3 + ["b"] * 3,
            "subject": ["a"] * 3 + ["b"] * 3,
            "identity": [7] * 6,
            "frame_index": [0, 1, 4] * 2,
            "hours": [0., .5, 2.] * 2,
            "corrected_mean": [2., np.nan, 4., 3., 2., 5.],
            "area_px": [4., 5., 6., 7., 9., 8.],
        }),
        # Cell 9 is known to exist but has no observations in the trace table.
        "cell_summary": pd.DataFrame({
            "stem": ["a", "a", "b"], "subject": ["a", "a", "b"],
            "identity": [7, 9, 7], "area_px_median": [5., 0., 8.],
            "review_label": ["ok", "short", "ok"],
        }),
    }


def resolve(tables, req=None, **kwargs):
    return resolve_request(req or request(), source_run="run-one", tables=tables,
                           input_hashes={name: content_id(name) for name in tables},
                           **kwargs)


def config_file(tmp_path, **blocks):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "frame_interval_min": 30,
        "movies": [{"stem": "a", "labels": "labels.tif", "raw": "raw.tif"}],
        **blocks,
    }), encoding="utf-8")
    return path


def test_config_keeps_dependent_requests_separate_from_static_plots(tmp_path):
    raw = {"pipeline": "rhythm-discovery", "test_measurements": ["corrected_mean"]}
    config = AnalysisConfig.load(config_file(tmp_path, pipelines=[raw],
                                            plots=[{"figure": "rhythm-strength"}]))
    assert len(config.pipelines) == len(config.plots) == 1
    assert config.pipelines[0].declaration.as_dict() == raw
    assert config.plots[0].figure == "rhythm-strength"
    empty = AnalysisConfig.load(config_file(tmp_path))
    assert empty.pipelines == [] and empty.plots == []


def test_configuration_parsing_imports_no_engine_or_renderers(tmp_path):
    path = config_file(tmp_path, pipelines=[{
        "pipeline": "rhythm-discovery", "test_measurements": ["corrected_mean"]}])
    code = """
import sys
from analysis.config import AnalysisConfig
config = AnalysisConfig.load(sys.argv[1])
assert config.pipelines
assert 'analysis.circadian' not in sys.modules
assert not any(name.startswith(('circadian_workbench', 'matplotlib', 'analysis.figures'))
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(path)], check=True,
                   cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)


def test_groups_expand_in_order_and_lists_can_overlap_independently(tables):
    from analysis.metric_groups import build

    groups = build({"measurements": ["area_px", "corrected_mean"]})
    raw = {"pipeline": "rhythm-discovery",
           "test_measurements": ["corrected_mean", "@measurements"],
           "comparison_measurements": [
               "area_px_median", {"column": "corrected_mean", "summary": "median"}]}
    req = parse([raw], groups)[0]
    result = resolve(tables, req)
    assert [m.column for m in result.test_measurements] == ["corrected_mean", "area_px"]
    assert [m.column for m in result.comparison_measurements] == ["area_px_median", "corrected_mean"]
    assert result.comparison_measurements[1].summary == "median"
    assert req.declaration["test_measurements"] == raw["test_measurements"]


def test_new_declared_and_actual_measurement_is_selectable_without_a_shortlist(tables, monkeypatch):
    from analysis import registry
    from analysis.metric_groups import build

    extra = registry.DerivedModule("pipeline_test_measurement", "test declaration", (),
                                   lambda frame, context: {},
                                   produces=(registry.Column("new_signal", "New signal", "units"),))
    monkeypatch.setitem(registry._DERIVED, extra.name, extra)
    tables["cell_frame"]["new_signal"] = [1., 2., 3., 4., 5., 6.]
    groups = build({"new_measurements": {"module": extra.name}})
    req = parse([{"pipeline": "rhythm-discovery",
                  "test_measurements": ["@new_measurements"]}], groups)[0]
    metric = resolve(tables, req).test_measurements[0]
    assert (metric.column, metric.label, metric.unit, metric.declared) == (
        "new_signal", "New signal", "units", True)


def test_actual_external_numeric_column_can_be_chosen_and_metadata_is_honest(tables):
    tables["cell_frame"]["external_signal"] = 1.
    metric = resolve(tables, request(test_measurements=["external_signal"])).test_measurements[0]
    assert metric.column == "external_signal" and not metric.declared
    assert metric.unit == ""


def test_unavailable_and_unsuitable_measurements_have_different_explanations(tables):
    for column, message in [("solidity", "declared but unavailable"),
                            ("not_written", "unavailable in the actual"),
                            ("area_px_median", "wrong grain")]:
        with pytest.raises(ValueError, match=message):
            resolve(tables, request(test_measurements=[column]))
    with pytest.raises(ValueError, match="not a real numeric"):
        resolve(tables, request(comparison_measurements=["review_label"]))


def test_trace_comparison_requires_explicit_summary_and_scalar_is_used_directly(tables):
    with pytest.raises(ValueError, match="explicit summary"):
        resolve(tables, request(comparison_measurements=["area_px"]))
    result = resolve(tables, request(comparison_measurements=["area_px_median"]))
    assert result.comparison_measurements[0].summary is None
    with pytest.raises(ValueError, match="already summaries"):
        resolve(tables, request(comparison_measurements=[
            {"column": "area_px_median", "summary": "mean"}]))


def test_ambiguous_table_requires_qualification_and_respects_declared_grain(tables):
    tables["other_trace"] = tables["cell_frame"].copy()
    grains = {"other_trace": ("identity", "frame_index")}
    with pytest.raises(ValueError, match="ambiguous"):
        resolve(tables, table_grains=grains)
    req = request(test_measurements=[{"column": "corrected_mean", "table": "other_trace"}])
    assert resolve(tables, req, table_grains=grains).test_measurements[0].table == "other_trace"
    with pytest.raises(ValueError, match="conflicts with its existing declaration"):
        resolve(tables, table_grains={"cell_frame": ("identity",)})


@pytest.mark.parametrize("fault, message", [
    ("duplicate", "duplicate rows"), ("missing_time", "missing identity/time"),
    ("boolean", "not a real numeric"), ("object", "not a real numeric"),
])
def test_bad_table_shapes_do_not_get_silently_reduced(tables, fault, message):
    frame = tables["cell_frame"]
    if fault == "duplicate":
        tables["cell_frame"] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif fault == "missing_time":
        tables["cell_frame"] = frame.drop(columns="hours")
    elif fault == "boolean":
        frame["corrected_mean"] = True
    else:
        frame["corrected_mean"] = [[1., 2.]] * len(frame)
    with pytest.raises(ValueError, match=message):
        resolve(tables)


def test_pair_modes_keep_first_direction_and_deduplicate_detection_pairs():
    metrics = ["a", "b", "c"]
    req = request(test_measurements=metrics)
    assert [(p.reference, p.target) for p in req.pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    req = request(test_measurements=metrics, pairs={"mode": "explicit", "pairs": [["b", "a"], ["a", "b"], ["c", "b"]]})
    assert [(p.reference, p.target) for p in req.pairs] == [("b", "a"), ("c", "b")]
    assert req.pairs[0].detection_key == ("a", "b")
    req = request(test_measurements=metrics, pairs={"mode": "reference", "reference": "c"})
    assert [(p.reference, p.target) for p in req.pairs] == [("c", "a"), ("c", "b")]


@pytest.mark.parametrize("pairs, message", [
    ({"mode": "explicit", "pairs": [["a", "a"]]}, "self-pairs"),
    ({"mode": "explicit", "pairs": [["a", "other"]]}, "selected test measurements"),
    ({"mode": "reference", "reference": "other"}, "test measurement"),
    ({"mode": "all", "reference": "a"}, "only mode"),
])
def test_invalid_pair_requests_are_refused(pairs, message):
    with pytest.raises(ValueError, match=message):
        request(test_measurements=["a", "b"], pairs=pairs)


def test_population_preserves_missing_traces_movie_keys_and_unconfirmed_samples(tables):
    before = {name: frame.copy(deep=True) for name, frame in tables.items()}
    result = resolve(tables, request(test_measurements=["corrected_mean", "area_px"]))
    assert [(c.movie, c.identity) for c in result.inputs.cells] == [("a", 7), ("a", 9), ("b", 7)]
    assert len(result.expected_pairs) == 6
    assert all(s.sample is None and not s.confirmed for s in result.inputs.samples)
    assert [s.observed_subject for s in result.inputs.samples] == ["a", "b"]
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, before[name])
    assert CellKey("run-one", "a", 7) != CellKey("another-run", "a", 7)


def test_explicit_cells_and_biological_samples_are_preserved(tables):
    req = request(cells=[{"movie": "b", "identity": 7}, {"movie": "a", "identity": 7}],
                  biological_samples={"a": "animal-one", "b": "animal-one"})
    result = resolve(tables, req)
    assert [c.movie for c in result.inputs.cells] == ["b", "a"]
    assert all(s.confirmed and s.sample == "animal-one" for s in result.inputs.samples)
    assert resolve(tables, request(cells=[])).expected_pairs == ()
    with pytest.raises(ValueError, match="requested cell is unavailable"):
        resolve(tables, request(cells=[{"movie": "a", "identity": 999}]))


def test_all_missing_numeric_measurement_is_retained_for_later_untestable_results(tables):
    tables["cell_frame"]["corrected_mean"] = np.nan
    result = resolve(tables)
    assert len(result.expected_pairs) == 3


def test_header_only_trace_table_retains_known_cells_for_untestable_results(tables):
    tables["cell_frame"] = pd.DataFrame(columns=tables["cell_frame"].columns)
    result = resolve(tables)
    assert len(result.expected_pairs) == 3


def test_complex_time_coordinates_are_refused(tables):
    tables["cell_frame"]["hours"] = 1j
    with pytest.raises(ValueError, match="real numeric coordinates"):
        resolve(tables)


def test_engine_choices_inherit_and_a_significance_bearing_estimator_does_not_replace_test(tables):
    params = {"period_estimation_method": "chi_square", "primary_rhythm_test": "f",
              "period_search_hours": [3., 36.], "rhythmic_alpha": .02,
              "detrend": "moving_median", "detrend_window_hours": 8.,
              "min_observations": 30, "min_cycles_for_confident_period": 4.,
              "multiple_testing": "bonferroni", "workbench_config": {"nlls_max_components": 2}}
    result = resolve(tables, rhythm_params=params)
    assert result.analysis_options["fit_method"] == "chi_square"
    assert result.analysis_options["significance_method"] == "f"
    assert result.analysis_options["period_min_hours"] == 3.
    assert result.analysis_options["period_max_hours"] == 36.
    assert result.analysis_options["min_cycles"] == 4.
    for name in ("rhythmic_alpha", "detrend", "detrend_window_hours", "min_observations", "multiple_testing"):
        assert result.analysis_options[name] == params[name]
    assert result.analysis_options["period_config"] == params["workbench_config"]


def test_complete_shared_option_contract_survives_overrides(tables, monkeypatch):
    from analysis import circadian

    def must_not_fit(*args, **kwargs):
        raise AssertionError("option resolution must not fit or detrend data")

    monkeypatch.setattr(circadian, "estimate_one", must_not_fit)
    monkeypatch.setattr(circadian, "detrend_trace", must_not_fit)
    options = {**circadian.CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,
               **circadian.DETREND_DEFAULTS,
               "fit_method": "fft_nlls", "significance_method": "lomb",
               "period_config": {"nlls_max_components": 1},
               "period_min_hours": 4., "period_max_hours": 40.,
               "detrend": "none", "detrend_polynomial_degree": 4,
               "detrend_bandwidth_hours": 6., "detrend_lowess_fraction": .3,
               "detrend_asls_asymmetry": .02}
    result = resolve(tables, request(analysis_options=options),
                     rhythm_params={"workbench_config": {"jtk_seed": 42}})
    assert set(result.analysis_options) == set(circadian.CIRCADIAN_ANALYSIS_OPTIONS)
    for name, value in options.items():
        if name != "period_config":
            assert result.analysis_options[name] == value
    assert result.analysis_options["period_config"] == {"jtk_seed": 42, "nlls_max_components": 1}
    assert result.rhythm_params["workbench_config"] == result.analysis_options["period_config"]
    assert result.workbench_version == circadian.WORKBENCH_VERSION
    assert not result.rhythm_params["descriptive_cosinor"]
    assert not result.rhythm_params["daily_profile_measures"]


def test_every_live_period_estimator_remains_selectable(tables):
    from analysis import circadian

    for method in circadian.PERIOD_METHODS:
        result = resolve(tables, request(analysis_options={
            "fit_method": method, "significance_method": "lomb"}))
        assert result.analysis_options["fit_method"] == method
        assert result.analysis_options["significance_method"] == "lomb"
        assert set(result.period_methods) == set(circadian.PERIOD_METHODS)
    with pytest.raises(ValueError, match="cannot test rhythmicity"):
        resolve(tables, request(analysis_options={"significance_method": "fft_nlls"}))
    with pytest.raises(ValueError, match="unknown period method"):
        resolve(tables, request(analysis_options={"fit_method": "not_a_method"}))
    with pytest.raises(ValueError, match="unknown setting"):
        resolve(tables, request(analysis_options={"detrend_windw_hours": 8}))


def test_fingerprints_settings_and_sample_mapping_contribute_to_identity(tables):
    base = resolve(tables)
    new_settings = resolve(tables, request(analysis_options={"rhythmic_alpha": .01}))
    new_sample = resolve(tables, request(biological_samples={"a": "animal-one"}))
    assert len({base.record_id, new_settings.record_id, new_sample.record_id}) == 3
    changed = replace(base.inputs, table_hashes=Settings({"cell_frame": "b" * 64}))
    assert replace(base, inputs=changed).record_id != base.record_id
    assert json.loads(json.dumps(base.as_dict())) == base.as_dict()
    with pytest.raises(ValueError, match="verified SHA-256"):
        resolve_request(request(), source_run="run-one", tables=tables, input_hashes={})


def test_nested_settings_are_detached_and_immutable():
    raw = {"period_config": {"nlls_max_components": 2}, "metrics": ["a"]}
    saved = Settings(raw)
    identity = content_id(saved)
    raw["period_config"]["nlls_max_components"] = 3
    saved["metrics"].append("b")
    assert saved["period_config"]["nlls_max_components"] == 2
    assert saved["metrics"] == ["a"]
    assert content_id(saved) == identity
    assert content_id({"b": 2, "a": 1}) == content_id({"a": 1, "b": 2})


@pytest.mark.parametrize("entries", [{}, False, "rhythm-discovery", [None]])
def test_malformed_pipeline_blocks_are_not_treated_as_absent(entries):
    with pytest.raises(ValueError):
        parse(entries)


def test_unknown_duplicate_and_conflicting_requests_are_refused():
    with pytest.raises(ValueError, match="unknown pipeline"):
        parse([{"pipeline": "something_else"}])
    raw = {"pipeline": "rhythm-discovery", "test_measurements": ["a"]}
    with pytest.raises(ValueError, match="duplicate pipeline request"):
        parse([raw, raw])
    with pytest.raises(ValueError, match="conflicting table/summary"):
        request(test_measurements=["a", {"column": "a", "table": "another"}])
    with pytest.raises(ValueError, match="expected a list"):
        request(test_measurements="corrected_mean")


def test_step_outcomes_require_a_reason_and_distinguish_empty_from_unavailable():
    empty = StepResult("reports", "science-one", "skipped-empty", "no selected cells")
    unavailable = StepResult("reports", "science-one", "unavailable", "producer unavailable")
    assert empty.record_id != unavailable.record_id
    with pytest.raises(ValueError, match="outcome reason"):
        StepResult("reports", "science-one", "failed", "")
