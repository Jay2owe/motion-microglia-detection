"""What the provenance module is allowed to claim about each cell.

The measured columns are trusted downstream to add up: a reader who sees
``inferred_px`` on a figure footnote and ``added_px`` in the bundle's data
folder is entitled to assume the second is a part of the first, not a second
opinion about it. These tests hold that apart from the prose that describes it.
"""
from __future__ import annotations

import numpy as np

from analysis.modules.provenance import measure
from analysis.registry import MeasurementContext
from analysis.units import Scale


def _context(added: np.ndarray | None = None) -> MeasurementContext:
    """One cell over one frame, four pixels, each with a different story."""
    labels = np.zeros((1, 1, 4), np.uint16)
    labels[0, 0, :3] = 7
    inferred = np.zeros((1, 1, 4), bool)
    inferred[0, 0, [1, 2]] = True
    unresolved = np.zeros((1, 1, 4), bool)
    unresolved[0, 0, 3] = True
    supplied = np.zeros((1, 1, 4), bool)
    supplied[0, 0, 2] = True
    return MeasurementContext(
        stem="test",
        labels=labels,
        raw=np.zeros((1, 1, 4), np.uint16),
        scale=Scale(minutes_per_frame=30.0),
        identities=[7],
        inferred=inferred,
        unresolved=unresolved,
        added=supplied if added is None else added,
    )


def _row(context: MeasurementContext) -> dict:
    return measure(context)["provenance"].iloc[0].to_dict()


def test_the_supplied_pixels_are_a_part_of_the_reconstructed_ones():
    row = _row(_context())
    assert row["inferred_px"] == 2
    assert row["renamed_px"] == 1
    assert row["added_px"] == 1
    assert row["renamed_px"] + row["added_px"] == row["inferred_px"]


def test_observed_px_still_means_everything_the_flag_did_not_touch():
    row = _row(_context())
    assert row["observed_px"] == 1
    assert row["inferred_px"] + row["observed_px"] == 3


def test_a_sidecar_without_the_third_bit_reports_no_supplied_pixels():
    """An older sidecar must not be made to look like it answered a question it
    was never asked. Everything it did flag stays counted as reconstructed."""
    row = _row(_context(added=np.zeros((1, 1, 4), bool)))
    assert row["added_px"] == 0
    assert row["renamed_px"] == row["inferred_px"] == 2


def test_a_movie_with_no_third_bit_array_at_all_is_still_measured():
    context = _context()
    context.added = None
    row = _row(context)
    assert row["added_px"] == 0
    assert row["renamed_px"] == 2


def test_the_field_count_and_the_per_cell_counts_agree():
    """Bit 2 only ever sits inside an outline, so nothing supplied may go
    missing between the per-cell table and the per-frame one."""
    tables = measure(_context())
    assert (int(tables["provenance_frame"]["added_px_field"].iloc[0])
            == int(tables["provenance"]["added_px"].sum()) == 1)
