"""Pooling is a concatenation, and these tests are about what it must not lose.

The arithmetic is nothing. What can go wrong is bookkeeping: a movie's rows
losing the label that says which movie they are, a column one movie never had
being filled with blanks that read as measured zeros, or a table only one movie
produced quietly implying the others had nothing to report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.pool import (POOLED_FOLDER, STAMP_COLUMNS, _ordered_union,
                           movie_folders, pool_run)
from analysis.run import MEASURED_FOLDER, TRACKER_FOLDER, _write_table


def _movie(run_dir, stem: str, tables: dict[str, pd.DataFrame],
           tracker: dict[str, pd.DataFrame] | None = None,
           condition: str = "control", subject: str | None = None) -> None:
    """One movie's folder, written the way ``run.analyse_movie`` writes one.

    Deliberately goes through ``_write_table`` rather than ``to_csv``: pooling
    reads back what the run wrote, so a fixture that wrote its numbers to a
    different precision would be testing a file this package never makes.
    """
    for folder, group in ((MEASURED_FOLDER, tables), (TRACKER_FOLDER, tracker or {})):
        for name, table in group.items():
            stamped = table.copy()
            for column, value in (("subject", subject or stem),
                                  ("condition", condition),
                                  ("stem", stem)):
                stamped.insert(0, column, value)
            _write_table(stamped, run_dir / stem / folder / f"{name}.csv")


def _cells(n: int, **extra) -> pd.DataFrame:
    table = pd.DataFrame({"identity": range(1, n + 1),
                          "area_px": np.linspace(100.0, 100.0 + n - 1, n)})
    for name, value in extra.items():
        table[name] = value
    return table


def test_two_movies_with_the_same_columns_stack(tmp_path) -> None:
    _movie(tmp_path, "m_a", {"cell_summary": _cells(3)}, condition="control")
    _movie(tmp_path, "m_b", {"cell_summary": _cells(4)}, condition="treated")

    fragment = pool_run(tmp_path)

    pooled = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "cell_summary.csv")
    assert len(pooled) == 7
    assert list(pooled["stem"]) == ["m_a"] * 3 + ["m_b"] * 4
    assert set(pooled["condition"]) == {"control", "treated"}
    record = fragment["tables"]["cell_summary"]
    assert record["rows_per_movie"] == {"m_a": 3, "m_b": 4}
    assert record["columns_missing"] == {}
    assert record["movies_absent"] == []


def test_a_column_one_movie_lacks_is_blank_and_the_manifest_says_which(tmp_path) -> None:
    """The failure this whole module exists to prevent.

    A movie that declared no object set has no object columns. Filled with NaN
    and left unrecorded, its rows are indistinguishable from a movie whose cells
    touched no objects - one is "not measured" and the other is "measured, and
    the answer was none".
    """
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2, object_overlap_share=0.5)})
    _movie(tmp_path, "m_b", {"cell_summary": _cells(2)})

    fragment = pool_run(tmp_path)

    pooled = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "cell_summary.csv")
    assert "object_overlap_share" in pooled.columns
    assert pooled.loc[pooled["stem"] == "m_b", "object_overlap_share"].isna().all()
    assert fragment["tables"]["cell_summary"]["columns_missing"] == {
        "m_b": ["object_overlap_share"]}


def test_a_table_only_one_movie_produced_is_still_pooled_and_the_absence_recorded(
        tmp_path) -> None:
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2), "contacts": _cells(5)})
    _movie(tmp_path, "m_b", {"cell_summary": _cells(2)})

    fragment = pool_run(tmp_path)

    contacts = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "contacts.csv")
    assert len(contacts) == 5
    assert fragment["tables"]["contacts"]["movies"] == ["m_a"]
    assert fragment["tables"]["contacts"]["movies_absent"] == ["m_b"]


def test_a_single_movie_run_still_produces_a_pooled_folder(tmp_path) -> None:
    """So that nothing downstream ever needs a one-movie special case.

    A run with one movie and a run with six must produce a ``pooled/`` folder of
    the same shape, or every later stage grows a branch for the case where there
    is nothing to compare - and that branch is where a two-movie study quietly
    starts behaving differently from a one-movie one.
    """
    _movie(tmp_path, "only", {"cell_summary": _cells(3)})

    fragment = pool_run(tmp_path)

    pooled = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "cell_summary.csv")
    assert len(pooled) == 3
    assert fragment["movies"] == ["only"]
    assert (tmp_path / POOLED_FOLDER / "manifest.json").exists()


def test_tracker_tables_stay_in_their_own_folder(tmp_path) -> None:
    """The ``origin`` contract survives the last step.

    ``tracker/`` holds tables this package copied rather than measured. Pooling
    them into one heap with the measured ones would make a run folder stop
    saying which numbers are its own.
    """
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2)}, tracker={"history_seat_table": _cells(3)})
    _movie(tmp_path, "m_b", {"cell_summary": _cells(2)}, tracker={"history_seat_table": _cells(4)})

    fragment = pool_run(tmp_path)

    assert (tmp_path / POOLED_FOLDER / TRACKER_FOLDER / "history_seat_table.csv").exists()
    assert not (tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "history_seat_table.csv").exists()
    assert fragment["tables"]["history_seat_table"]["folder"] == TRACKER_FOLDER
    assert fragment["tables"]["cell_summary"]["folder"] == MEASURED_FOLDER


def test_every_pooled_row_carries_stem_condition_and_subject(tmp_path) -> None:
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2)}, condition="control", subject="animal_1")
    _movie(tmp_path, "m_b", {"cell_summary": _cells(2)}, condition="treated", subject="animal_1")

    pool_run(tmp_path)

    pooled = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "cell_summary.csv")
    for column in STAMP_COLUMNS:
        assert column in pooled.columns
        assert pooled[column].notna().all()
        assert (pooled[column].astype(str).str.len() > 0).all()
    # Two movies from one animal are one subject, and pooling must not invent a
    # second one - this is the column the unit of replication is read from.
    assert set(pooled["subject"]) == {"animal_1"}


def test_pooling_moves_no_number(tmp_path) -> None:
    """A pooled value is the per-movie value, to the last digit written.

    Pooling reads a file written to nine significant figures and writes it again
    to nine significant figures, so there is no rounding step to hide in. If
    this ever fails, pooling has started computing.
    """
    awkward = pd.DataFrame({
        "identity": [1, 2],
        "mean_of_16_bit_pixels": [12148.314285714287, 0.1 + 0.2],
        "a_p_value_near_zero": [1.4551764212882648e-15, 6.02e23],
    })
    _movie(tmp_path, "m_a", {"cell_summary": awkward})

    pool_run(tmp_path)

    before = pd.read_csv(tmp_path / "m_a" / MEASURED_FOLDER / "cell_summary.csv")
    after = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "cell_summary.csv")
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_pooling_refuses_to_overwrite(tmp_path) -> None:
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2)})
    pool_run(tmp_path)

    with pytest.raises(FileExistsError, match="immutable"):
        pool_run(tmp_path)


def test_a_table_without_the_stamp_is_refused_by_name(tmp_path) -> None:
    """An unstamped table cannot be pooled into anything usable.

    Its rows would be indistinguishable from another movie's the moment they
    were stacked, so this stops rather than producing a table whose rows nobody
    can attribute.
    """
    path = tmp_path / "m_a" / MEASURED_FOLDER / "cell_summary.csv"
    path.parent.mkdir(parents=True)
    _cells(2).to_csv(path, index=False)

    with pytest.raises(ValueError, match="cell_summary.csv is missing"):
        pool_run(tmp_path)


def test_pooling_never_pools_the_pooled_folder_into_itself(tmp_path) -> None:
    """pooled/ holds a tables folder too, once it exists.

    Left in the list it would be pooled into itself the next time round: every
    row twice, and the result indistinguishable from a run with one extra movie
    called . The refusal in pool_run catches the ordinary case; this
    catches the one where somebody deleted the manifest and left the tables.
    """
    _movie(tmp_path, "m_a", {"cell_summary": _cells(3)})
    pool_run(tmp_path)
    (tmp_path / POOLED_FOLDER / "manifest.json").unlink()

    assert [p.name for p in movie_folders(tmp_path)] == ["m_a"]
    with pytest.raises(FileExistsError):
        pool_run(tmp_path)


def test_a_run_with_no_movie_folders_says_so(tmp_path) -> None:
    (tmp_path / "figures").mkdir()

    with pytest.raises(ValueError, match="no movie folders"):
        pool_run(tmp_path)


def test_the_manifest_is_readable_on_its_own(tmp_path) -> None:
    """A pooled folder has to explain itself without the run manifest beside it."""
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2, extra=1.0)})
    _movie(tmp_path, "m_b", {"cell_summary": _cells(3)})

    pool_run(tmp_path)

    fragment = json.loads(
        (tmp_path / POOLED_FOLDER / "manifest.json").read_text(encoding="utf-8"))
    assert fragment["movies"] == ["m_a", "m_b"]
    record = fragment["tables"]["cell_summary"]
    assert record["rows"] == 5
    assert record["sha256"]
    assert record["columns_missing"] == {"m_b": ["extra"]}


def test_column_order_is_first_seen_and_deterministic() -> None:
    """Two runs of the same data must not order a pooled file differently."""
    assert _ordered_union([["a", "b"], ["b", "c"], ["d", "a"]]) == ["a", "b", "c", "d"]
    assert _ordered_union([[], ["x"]]) == ["x"]
    assert _ordered_union([]) == []


def test_a_pooled_column_order_puts_the_stamp_first(tmp_path) -> None:
    _movie(tmp_path, "m_a", {"cell_summary": _cells(2)})
    _movie(tmp_path, "m_b", {"cell_summary": _cells(2)})

    pool_run(tmp_path)

    pooled = pd.read_csv(tmp_path / POOLED_FOLDER / MEASURED_FOLDER / "cell_summary.csv")
    assert list(pooled.columns[:3]) == list(STAMP_COLUMNS)
