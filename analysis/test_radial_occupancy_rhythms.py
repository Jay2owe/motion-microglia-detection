from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

FIGURES = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(FIGURES))

from _schema import load_all  # noqa: E402


def _figure_module():
    load_all()
    return sys.modules["41_radial_occupancy_rhythms"]


def test_relative_occupancy_support_is_unchanged_by_length_calibration():
    module = _figure_module()
    pixels = pd.DataFrame({
        "identity": [1, 1],
        "frame_index": [0, 0],
        "hours": [1.0, 1.0],
        "ring": [0, 1],
        "radius_inner": [0.0, 1.0],
        "radius_outer": [1.0, 2.0],
        "annulus_px": [4, 10],
        "occupancy": [0.25, 0.75],
    })
    micrometres = pixels.copy()
    micrometres[["radius_inner", "radius_outer"]] *= 2.0

    pixel_trace, _ = module._relative_occupancy_trace(
        pixels, support_threshold=0.5, length_per_pixel=1.0,
    )
    calibrated_trace, _ = module._relative_occupancy_trace(
        micrometres, support_threshold=0.5, length_per_pixel=2.0,
    )

    assert pixel_trace.loc[0, module.VALUE_COLUMN] == pytest.approx(0.5)
    assert calibrated_trace.loc[0, module.VALUE_COLUMN] == pytest.approx(0.5)
    assert calibrated_trace.loc[0, "valid_rings"] == 2
