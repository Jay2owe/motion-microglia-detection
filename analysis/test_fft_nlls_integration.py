"""Fit-only estimators must not masquerade as significance tests in matrices."""
import numpy as np
import pandas as pd
import pytest

from analysis import circadian
from analysis.modules.rhythms import DEFAULTS


@pytest.mark.parametrize("period", [6.0, 36.0])
def test_grouped_matrix_supports_fft_nlls_with_attributed_evidence(period):
    hours = np.arange(0, 144, 0.5)
    frame = pd.DataFrame({"identity": 1, "hours": hours,
                          "value": np.cos(2 * np.pi * hours / period)})
    result = circadian.estimate_grouped_rhythms(
        frame, group_columns=["identity"], value_column="value", method="fft_nlls",
        params={**DEFAULTS, "period_search_hours": [2, 48], "detrend": "none",
                "workbench_config": {"nlls_max_components": 2}})
    row = result.iloc[0]
    assert row.period_hours == pytest.approx(period, abs=0.1)
    assert row.significance_method == "lomb"
    assert row.rhythm_status == "rhythmic"
    assert row.period_available
    assert row.estimator_p_value is None
    assert row.p_value < 0.05
    assert row.rae is not None
    assert row.components


def test_failed_estimator_preserves_positive_test_and_failure_reason():
    hours = np.arange(0, 144, 0.5)
    result = circadian.estimate_grouped_rhythms(
        pd.DataFrame({"identity": 1, "hours": hours,
                      "value": np.cos(2 * np.pi * hours / 6)}),
        group_columns=["identity"], value_column="value", method="fft_nlls",
        params={**DEFAULTS, "period_search_hours": [2, 48], "detrend": "none",
                "workbench_config": {"nlls_max_components": 1,
                    "nlls_circadian_min": 15, "nlls_circadian_max": 35}})
    row = result.iloc[0]
    assert row.rhythm_status == "rhythmic"
    assert not row.period_available
    assert row.estimate_status == "failed"
    assert "selection band" in row.estimate_reason


def test_all_constant_groups_still_have_complete_status_columns():
    result = circadian.estimate_grouped_rhythms(
        pd.DataFrame({"identity": 1, "hours": np.arange(40), "value": 1}),
        group_columns=["identity"], value_column="value", method="fft_nlls",
        params=DEFAULTS)
    assert result.iloc[0].rhythm_status == "not tested"
    assert not result.iloc[0].period_available
    assert result.iloc[0].period_underdetermined


def test_returned_model_is_not_refitted_as_a_single_cosine():
    hours = np.arange(36, 180, 0.5)
    values = 3 + np.cos(2 * np.pi * hours / 6) + 0.4 * np.cos(2 * np.pi * hours / 36)
    result = circadian.estimate_one(hours, values,
        {**DEFAULTS, "period_search_hours": [2, 48], "detrend": "none",
         "workbench_config": {"nlls_max_components": 2}}, "fft_nlls")
    np.testing.assert_allclose(circadian.fft_nlls_fitted_values(hours, result), values, atol=0.03)


def test_generic_matrix_builds_with_fft_and_keeps_constant_traces_untested(monkeypatch):
    import sys
    from pathlib import Path
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from analysis.theme import load_theme

    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from _schema import FigureContext, load_all, get_figure
    load_all()
    spec = get_figure("metric-rhythm-matrix")
    hours = np.arange(0, 144, 0.5)
    frame = pd.concat([pd.DataFrame({
        "identity": identity, "hours": hours,
        "area_px": 10 + np.cos(2 * np.pi * hours / period),
        "corrected_mean": (3 + np.cos(2 * np.pi * hours / 36) if identity == 1
                           else np.ones_like(hours)),
    }) for identity, period in [(1, 6), (2, 36)]], ignore_index=True)
    ctx = FigureContext(
        spec=spec, run=Path("."), tables=Path("."), theme=load_theme(), bundle=Path("."),
        summary={"minutes_per_frame": 30, "hours_covered": 143.5, "stem": "synthetic"},
        field={}, stem=None, argv=["--metrics", "area_px,corrected_mean",
            "--fit-method", "fft_nlls", "--detrend", "none",
            "--period-config", '{"nlls_max_components": 2}'])
    monkeypatch.setattr(ctx, "table", lambda name: frame)
    monkeypatch.setattr(ctx, "module_params", lambda name: dict(DEFAULTS))
    monkeypatch.setattr(ctx, "provenance_auxiliary", lambda: {})
    result = spec.build(ctx)
    FigureCanvasAgg(result.figure).draw()  # Exercise rendering without exporting a new figure.
    assert len(result.figure_data) == 4
    assert result.figure_data.significance_method.eq("lomb").all()
    assert (result.figure_data.rhythm_status == "not tested").sum() == 1
    assert "not a significance test of a separate fitted period" in result.footnote


def test_generic_detrend_helper_honours_polynomial_degree():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "figures"))
    from panels.rhythms import detrended_z
    hours = np.arange(0, 96, 0.5)
    values = 3 + (hours / 96) ** 6 + np.cos(2 * np.pi * hours / 6)
    actual = detrended_z(hours, values, method="polynomial",
                        detrend_options={"detrend_polynomial_degree": 6})
    expected = detrended_z(hours, values, method="poly6")
    np.testing.assert_allclose(actual, expected)
