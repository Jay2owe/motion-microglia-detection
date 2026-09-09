"""The main rhythm detrending choice reaches derived analyses and manifests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import circadian
import analysis.modules  # noqa: F401 - populate the derived-module registry
from analysis.modules.coupling import derive
from analysis.registry import MeasurementContext, get_derived
from analysis.units import Scale


def _context(coupling: dict | None = None, rhythms: dict | None = None) -> MeasurementContext:
    return MeasurementContext(
        stem="synthetic",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        raw=np.zeros((2, 2, 2), dtype=np.uint16),
        scale=Scale(60.0),
        identities=[1, 2],
        params={
            "rhythms": {"detrend": "none", "detrend_window_hours": 7.5,
                        **dict(rhythms or {})},
            "coupling": {
                "metric_pairs": [], "between_metrics": ["corrected_mean"],
                "min_overlap_frames": 4, "surrogates": 1,
                **dict(coupling or {}),
            },
        },
    )


def _frame() -> pd.DataFrame:
    rows = []
    for identity, shift in ((1, 0.0), (2, 0.4)):
        for frame in range(12):
            rows.append({
                "identity": identity,
                "frame_index": frame,
                "hours": float(frame),
                "corrected_mean": float(np.sin(2 * np.pi * frame / 8 + shift)),
                "area_px": float(100 + frame + np.cos(2 * np.pi * frame / 8 + shift)),
                "centroid_y": float(identity),
                "centroid_x": float(identity * 2),
            })
    return pd.DataFrame(rows)


def test_coupling_inherits_and_records_the_main_detrending_choice() -> None:
    context = _context()
    table = derive(_frame(), context)["coupling"]

    assert set(table["detrend"]) == {"none"}
    assert set(table["detrend_window_hours"]) == {7.5}
    assert get_derived("coupling").parameters(context)["detrend"] == "none"
    assert get_derived("coupling").parameters(context)["detrend_window_hours"] == 7.5


def test_local_analysis_override_wins_over_the_main_choice() -> None:
    context = _context({"detrend": "kernel", "detrend_window_hours": 10.0})
    parameters = get_derived("coupling").parameters(context)

    assert parameters["detrend"] == "kernel"
    assert parameters["detrend_window_hours"] == 10.0


def test_within_cell_lag_profiles_use_the_same_detrending_choice() -> None:
    context = _context({
        "metric_pairs": [["corrected_mean", "area_px"]],
        "between_metrics": [],
        "max_lag_frames": 2,
    })
    table = derive(_frame(), context)["lag_profiles"]

    assert not table.empty
    assert set(table["detrend"]) == {"none"}
    assert set(table["detrend_window_hours"]) == {7.5}


def test_between_cell_phase_inherits_the_main_period_estimator() -> None:
    context = _context(rhythms={
        "period_estimation_method": "f",
        "primary_rhythm_test": "f",
        "period_methods": ["f"],
        "period_search_hours": [2.0, 10.0],
    })
    table = derive(_frame(), context)["coupling"]

    assert not table.empty
    assert set(table["period_estimation_method"]) == {"f"}
    assert table["phase_difference_fraction"].between(0.0, 0.5).all()
    assert set(table["workbench_version"]) == {circadian.WORKBENCH_VERSION}
