"""A derived module may compute with a user's column, but only where named.

Side tables are now joined *before* the derived modules run rather than after,
so a dosing schedule can be one half of a lag profile. The protection that
ordering used to provide is replaced by an explicit one, and these tests are
about that replacement: a module sees the side columns the configuration named
for it and no others, and the run records which it was offered and which it
reached for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis.modules  # noqa: F401  - registers coupling
from analysis.registry import MeasurementContext
from analysis.run import _granted_side_columns, _side_columns_used
from analysis.summarise import side_column_names
from analysis.units import Scale

N_FRAMES = 8
IDENTITIES = (1, 2)


def _context(module_params: dict | None = None, **side: pd.DataFrame) -> MeasurementContext:
    labels = np.zeros((N_FRAMES, 8, 8), dtype=np.uint16)
    for index, identity in enumerate(IDENTITIES, start=1):
        labels[:, index, :3] = identity
    return MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=list(IDENTITIES), side=dict(side),
        params=dict(module_params or {}),
    )


def _schedule() -> pd.DataFrame:
    """A dosing log, already prefixed and keyed the way the loader leaves one."""
    return pd.DataFrame({
        "frame_index": range(N_FRAMES),
        "schedule_dose": np.linspace(0.0, 7.0, N_FRAMES),
    })


def _calls() -> pd.DataFrame:
    return pd.DataFrame({"identity": list(IDENTITIES), "genotype_call": ["wt", "ko"]})


def _cell_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "identity": np.repeat(IDENTITIES, N_FRAMES),
        "frame_index": np.tile(range(N_FRAMES), len(IDENTITIES)),
        "hours": np.tile(np.arange(N_FRAMES) * 0.5, len(IDENTITIES)),
        "turnover_index": np.linspace(0.0, 1.0, N_FRAMES * len(IDENTITIES)),
    })


# ------------------------------------------------------- what counts as a side column


def test_the_user_columns_are_named_and_the_keys_are_not() -> None:
    """``identity`` and ``frame_index`` in a side table are this package's own.

    The loader renamed the user's key column to the key it joins on, so treating
    those two as user columns would hide the join keys from every module.
    """
    context = _context(schedule=_schedule(), genotype=_calls())
    assert side_column_names(context) == {"schedule_dose", "genotype_call"}


def test_a_movie_with_no_side_tables_supplies_nothing() -> None:
    assert side_column_names(_context()) == set()


# ------------------------------------------------------------------ the grant


def test_a_module_that_names_nothing_is_granted_nothing() -> None:
    """Declaring nothing must cost nothing, and must grant nothing.

    This is the case every existing run is in, and the reason attaching a
    spreadsheet stayed free when the join moved.
    """
    context = _context(schedule=_schedule())
    assert _granted_side_columns("coupling", context, {"schedule_dose"}) == set()


def test_a_module_is_granted_exactly_what_it_names() -> None:
    context = _context({"coupling": {"side_columns": ["schedule_dose"]}},
                       schedule=_schedule(), genotype=_calls())
    supplied = side_column_names(context)
    assert _granted_side_columns("coupling", context, supplied) == {"schedule_dose"}
    # And another module named nothing, so it is granted nothing - the grant is
    # per module, not per run.
    assert _granted_side_columns("rhythms", context, supplied) == set()


def test_a_grant_naming_a_column_no_side_table_supplied_stops_the_run() -> None:
    """A typo in a column name is the most likely way to get here.

    Ignoring it would produce the result the reader was trying to avoid: a run
    that finishes, with the computation they asked for quietly missing.
    """
    context = _context({"coupling": {"side_columns": ["dose"]}}, schedule=_schedule())
    with pytest.raises(ValueError) as raised:
        _granted_side_columns("coupling", context, side_column_names(context))

    message = str(raised.value)
    assert "coupling" in message                    # the module
    assert "'dose'" in message                      # the column
    assert "schedule" in message                    # the side table it looked in
    assert "schedule_dose" in message               # what it should have said


def test_a_grant_that_is_not_a_list_is_refused() -> None:
    context = _context({"coupling": {"side_columns": "schedule_dose"}},
                       schedule=_schedule())
    with pytest.raises(TypeError, match="must be a list"):
        _granted_side_columns("coupling", context, side_column_names(context))


# --------------------------------------------------------------- what was used


def test_a_column_used_as_a_value_counts_as_used() -> None:
    """``coupling`` names its two series in ``metric_a`` and ``metric_b``.

    The column name is data there rather than a heading, so a check that looked
    only at headings would report every coupling grant as unused.
    """
    produced = {"lag_profiles": pd.DataFrame({
        "metric_a": ["schedule_dose", "area_px"],
        "metric_b": ["turnover_index", "turnover_index"],
        "correlation": [0.5, 0.2]})}
    assert _side_columns_used({"schedule_dose"}, produced) == {"schedule_dose"}


def test_a_column_used_as_a_heading_counts_as_used() -> None:
    produced = {"whatever": pd.DataFrame({"identity": [1], "schedule_dose": [2.0]})}
    assert _side_columns_used({"schedule_dose"}, produced) == {"schedule_dose"}


def test_a_grant_nothing_reached_for_is_reported_as_unused() -> None:
    """The misconfiguration the manifest has to expose rather than hide."""
    produced = {"lag_profiles": pd.DataFrame({
        "metric_a": ["area_px"], "metric_b": ["turnover_index"],
        "correlation": [0.1]})}
    assert _side_columns_used({"schedule_dose"}, produced) == set()


def test_nothing_granted_means_nothing_scanned() -> None:
    """The scan is skipped outright when there is no grant, which is every run
    that declares no side tables - so it costs those runs nothing."""
    assert _side_columns_used(set(), {"t": pd.DataFrame({"a": ["x"]})}) == set()


# ------------------------------------------------------------- coupling itself


def test_coupling_computes_a_lag_profile_against_a_granted_side_column() -> None:
    """The first real user of this, and the reason the stage exists."""
    from analysis.modules.coupling import derive

    context = _context({"coupling": {"side_columns": ["schedule_dose"],
                                     "metric_pairs": [["schedule_dose", "turnover_index"]],
                                     "min_overlap_frames": 4, "max_lag_frames": 2,
                                     "surrogates": 3, "between_metrics": []}},
                       schedule=_schedule())
    cell_frame = _cell_frame()
    cell_frame = cell_frame.merge(_schedule(), on="frame_index", how="left")

    produced = derive(cell_frame, context)
    profiles = produced["lag_profiles"]

    assert not profiles.empty
    assert set(profiles["metric_a"]) == {"schedule_dose"}
    assert set(profiles["metric_b"]) == {"turnover_index"}
    # The prefix survives into the output, so a reader can see the number came
    # from a spreadsheet rather than from a measurement.
    assert all(name.startswith("schedule_") for name in profiles["metric_a"])


def test_coupling_refuses_a_side_column_it_was_not_granted() -> None:
    """Proved rather than trusted to convention.

    The run hides an ungranted column from the module, so the module sees a pair
    naming a column it does not have. Skipping it - which is right for a
    measured column from a switched-off module - would be exactly the silent
    miss this feature exists to remove.
    """
    from analysis.modules.coupling import derive

    context = _context({"coupling": {"metric_pairs": [["schedule_dose", "turnover_index"]],
                                     "between_metrics": []}},
                       schedule=_schedule())

    with pytest.raises(ValueError) as raised:
        derive(_cell_frame(), context)          # joined, but the column was hidden

    message = str(raised.value)
    assert "schedule_dose" in message
    assert "not granted" in message
    assert "side_columns" in message             # and how to fix it


def test_coupling_refuses_a_side_column_that_is_not_a_number() -> None:
    """A genotype call is a string, and a lag profile is a correlation.

    Left to numpy this either fails obscurely or coerces into something
    meaningless, which is worse.
    """
    from analysis.modules.coupling import derive

    context = _context({"coupling": {"side_columns": ["genotype_call"],
                                     "metric_pairs": [["genotype_call", "turnover_index"]],
                                     "between_metrics": []}},
                       genotype=_calls())
    cell_frame = _cell_frame().merge(_calls(), on="identity", how="left")

    with pytest.raises(ValueError, match="is object"):
        derive(cell_frame, context)


def test_coupling_still_skips_a_measured_column_a_switched_off_module_never_wrote() -> None:
    """The skip that must survive: fewer modules means fewer lag profiles.

    Turning *this* into a refusal would stop every run that switches a module
    off, which is a supported thing to do.
    """
    from analysis.modules.coupling import derive

    context = _context({"coupling": {"metric_pairs": [["sholl_auc", "turnover_index"]],
                                     "between_metrics": []}})

    produced = derive(_cell_frame(), context)
    assert produced["lag_profiles"].empty


def test_coupling_refuses_a_pair_that_is_not_a_pair() -> None:
    from analysis.modules.coupling import derive

    context = _context({"coupling": {"metric_pairs": [["turnover_index"]],
                                     "between_metrics": []}})

    with pytest.raises(ValueError, match="has 1 members"):
        derive(_cell_frame(), context)
