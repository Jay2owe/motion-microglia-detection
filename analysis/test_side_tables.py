"""Attaching a user's own spreadsheet to the right table, checked row by row.

A side table is the one input this package deliberately does not understand.
Its columns are the user's words and its numbers are never derived from, so
almost nothing here checks a measurement - what it checks is that the right row
of the user's file ended up beside the right row of ours.

Three ways that goes wrong, and each has its own test below:

* **the key is numbered differently.** Label frame 0 is source ImageJ frame 3
  on the pinned movie. A log written in the microscope's numbering and read as
  though it were ours is off by two frames, complete, and plausible.
* **the key repeats.** A merge against a repeated key multiplies rows instead
  of failing, so a hundred-row table quietly becomes four hundred.
* **the column is already ours.** A user column called ``area_px`` landing on
  top of the measured one is the failure the whole package is built to refuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis.modules  # noqa: F401  - registers every module, so the
                         # collision check has the full vocabulary to check
                         # against rather than an empty one
from analysis.config import MovieConfig, SideTableConfig, _side_tables
from analysis.io import load_side_table
from analysis.registry import MeasurementContext
from analysis.summarise import join_side
from analysis.units import Scale

N_FRAMES = 6
IDENTITIES = [1, 2, 3]


def _movie(offset: int = 0) -> MovieConfig:
    return MovieConfig(stem="t", labels="labels.tif", raw="raw.tif",
                       source_frame_offset=offset)


def _csv(tmp_path, name: str, frame: pd.DataFrame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _load(spec: SideTableConfig, movie: MovieConfig | None = None, **kwargs):
    return load_side_table(spec, movie or _movie(), N_FRAMES, IDENTITIES, **kwargs)


def _context(**side: pd.DataFrame) -> MeasurementContext:
    """The smallest context ``join_side`` needs: a movie with three cells."""
    labels = np.zeros((N_FRAMES, 8, 8), dtype=np.uint16)
    for index, identity in enumerate(IDENTITIES, start=1):
        labels[:, index, :3] = identity
    return MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=list(IDENTITIES), side=dict(side),
    )


# --------------------------------------------------------------- the two keys

def test_a_table_keyed_on_a_timepoint_reaches_every_frame(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES),
        "focus": [0.9, 0.8, 0.1, 0.7, 0.7, 0.6],
        "suspect": [0, 0, 1, 0, 0, 0],
    }))
    table, record = _load(SideTableConfig(name="acquisition", path=path))

    assert list(table.columns) == ["frame_index", "acquisition_focus",
                                   "acquisition_suspect"]
    assert record["keyed_on"] == "frame_index"
    assert record["keys_matched"] == N_FRAMES
    assert record["movie_rows_covered"] == N_FRAMES

    frames = pd.DataFrame({"frame_index": range(N_FRAMES)})
    joined = join_side(frames, _context(acquisition=table), "frame_index")
    assert joined["acquisition_focus"].tolist() == [0.9, 0.8, 0.1, 0.7, 0.7, 0.6]
    assert joined["acquisition_suspect"].sum() == 1


def test_a_table_keyed_on_a_cell_reaches_every_row_of_that_cell(tmp_path):
    """The per-cell fact repeats down the track, which is the settled default."""
    path = _csv(tmp_path, "calls.csv", pd.DataFrame({
        "identity": IDENTITIES, "call": ["wt", "ko", "wt"],
    }))
    table, record = _load(SideTableConfig(name="genotype", path=path,
                                          keyed_on="identity"))
    assert record["keyed_on"] == "identity"

    cell_frame = pd.DataFrame({
        "identity": np.repeat(IDENTITIES, N_FRAMES),
        "frame_index": np.tile(range(N_FRAMES), len(IDENTITIES)),
    })
    joined = join_side(cell_frame, _context(genotype=table), "identity")
    assert len(joined) == len(cell_frame)
    for identity, call in zip(IDENTITIES, ["wt", "ko", "wt"]):
        rows = joined.loc[joined["identity"] == identity, "genotype_call"]
        assert set(rows) == {call}


# ------------------------------------------------------------ the frame-number trap

def test_a_key_in_the_microscopes_own_numbering_is_converted(tmp_path):
    """Off by two on the pinned movie, and complete and plausible if unfixed.

    The offset is deliberately not zero: at zero the two numberings differ only
    by ImageJ's one-based count, and a test that passes under either reading
    proves nothing about the setting it is meant to be checking.
    """
    offset = 2
    # Source ImageJ frames 3..8 are label frames 0..5.
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame": range(1, 9),
        "focus": [9.0, 9.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }))
    table, record = _load(
        SideTableConfig(name="acquisition", path=path, key_column="frame",
                        key_space="source"),
        _movie(offset=offset),
    )
    assert record["key_space"] == "source"
    # The two rows describing discarded source frames are read, keyed to -2 and
    # -1, and simply match nothing.
    assert record["keys_unmatched"] == 2
    assert record["keys_matched"] == N_FRAMES

    frames = pd.DataFrame({"frame_index": range(N_FRAMES)})
    joined = join_side(frames, _context(acquisition=table), "frame_index")
    assert joined["acquisition_focus"].tolist() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_a_label_space_key_is_taken_as_written(tmp_path):
    """The default, and the same file read the other way, so the two differ."""
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "focus": [0, 1, 2, 3, 4, 5],
    }))
    table, _ = _load(SideTableConfig(name="acquisition", path=path),
                     _movie(offset=2))
    assert table["frame_index"].tolist() == list(range(N_FRAMES))


# ----------------------------------------------------------------- the refusals

def test_a_repeated_key_is_refused_before_it_multiplies_rows(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": [0, 1, 1, 2], "focus": [1.0, 2.0, 3.0, 4.0],
    }))
    with pytest.raises(ValueError, match="repeats 1 frame_index"):
        _load(SideTableConfig(name="acquisition", path=path))


def test_a_missing_key_column_names_what_the_file_actually_has(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "timepoint": range(N_FRAMES), "focus": range(N_FRAMES),
    }))
    with pytest.raises(ValueError, match="timepoint, focus"):
        _load(SideTableConfig(name="acquisition", path=path))


def test_a_column_this_package_already_writes_is_refused(tmp_path):
    """Prefixing makes it unlikely; this makes it impossible.

    The prefix is skipped when the user already wrote it, so a column called
    ``acquisition_area_px`` in a table called ``acquisition`` lands on
    ``area_px``'s namespace after all - which is exactly the case a check
    against the prefixed name catches and a check against the raw one does not.
    """
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "px": range(N_FRAMES),
    }))
    with pytest.raises(ValueError, match="area_px"):
        _load(SideTableConfig(name="area", path=path))


def test_two_side_tables_cannot_both_contribute_one_column(tmp_path):
    first = _csv(tmp_path, "a.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "log_focus": range(N_FRAMES)}))
    second = _csv(tmp_path, "b.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "focus": range(N_FRAMES)}))

    _, record = _load(SideTableConfig(name="stage", path=first))
    assert record["columns"] == ["stage_log_focus"]
    with pytest.raises(ValueError, match="stage_log_focus"):
        _load(SideTableConfig(name="stage_log", path=second),
              taken=set(record["columns"]))


def test_a_missing_file_says_which_table_it_belongs_to(tmp_path):
    with pytest.raises(FileNotFoundError, match="acquisition"):
        _load(SideTableConfig(name="acquisition", path=tmp_path / "absent.csv"))


def test_a_column_the_allow_list_names_but_the_file_lacks_is_refused(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "focus": range(N_FRAMES)}))
    with pytest.raises(ValueError, match="drift"):
        _load(SideTableConfig(name="acquisition", path=path,
                              columns=("focus", "drift")))


# ------------------------------------------------------------ coverage and selection

def test_a_table_that_covers_part_of_the_movie_joins_as_blanks(tmp_path):
    """A cell nobody wrote a row for keeps its measurements and gets a blank.

    An inner join here would delete measured cells because somebody's notes
    were incomplete, which is the package deciding what counts as data.
    """
    path = _csv(tmp_path, "calls.csv", pd.DataFrame({
        "identity": [1, 3, 99], "call": ["wt", "ko", "wt"],
    }))
    table, record = _load(SideTableConfig(name="genotype", path=path,
                                          keyed_on="identity"))
    assert record["keys_matched"] == 2
    assert record["keys_unmatched"] == 1        # cell 99 is not in this movie
    assert record["movie_rows"] == 3
    assert record["movie_rows_covered"] == 2

    summary = pd.DataFrame({"identity": IDENTITIES})
    joined = join_side(summary, _context(genotype=table), "identity")
    assert len(joined) == 3
    assert joined["genotype_call"].tolist()[:2] == ["wt", None] or (
        joined["genotype_call"].isna().sum() == 1)
    assert joined.loc[joined["identity"] == 2, "genotype_call"].isna().all()


def test_the_allow_list_takes_two_columns_out_of_a_wide_log(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES),
        **{f"c{i}": range(N_FRAMES) for i in range(20)},
    }))
    table, record = _load(SideTableConfig(name="acquisition", path=path,
                                          columns=("c3", "c7")))
    assert list(table.columns) == ["frame_index", "acquisition_c3", "acquisition_c7"]
    assert record["columns"] == ["acquisition_c3", "acquisition_c7"]


def test_a_key_that_is_not_a_number_is_counted_rather_than_guessed(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": ["0", "1", "", "3"], "focus": [1.0, 2.0, 3.0, 4.0],
    }))
    table, record = _load(SideTableConfig(name="acquisition", path=path))
    assert record["rows_read"] == 4
    assert record["rows_unusable_key"] == 1
    assert table["frame_index"].tolist() == [0, 1, 3]


def test_a_column_the_user_already_prefixed_is_not_prefixed_twice(tmp_path):
    path = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "acquisition_focus": range(N_FRAMES),
    }))
    table, _ = _load(SideTableConfig(name="acquisition", path=path))
    assert list(table.columns) == ["frame_index", "acquisition_focus"]


# ----------------------------------------------------------- nothing declared

def test_a_movie_with_no_side_tables_is_untouched():
    """The whole compatibility contract, in one assertion.

    Almost every movie declares nothing here. The join has to be a no-op for
    those, not a no-op that reorders columns or changes a dtype on the way
    through.
    """
    before = pd.DataFrame({"identity": IDENTITIES, "area_px": [10, 20, 30]})
    after = join_side(before, _context(), "identity")
    pd.testing.assert_frame_equal(before, after)


def test_the_columns_a_side_table_contributes_are_deliberately_undeclared():
    """They are the user's words, so this package must not name them.

    ``test_column_declarations`` checks that every column a *module* writes is
    described somewhere. A side table is not a module and writes no table of
    its own: its columns are attached by the join in ``run.py``, carried
    through verbatim under the user's own name. Declaring them would be this
    package inventing a meaning for a column it copied - the same reason the
    tracker's own tables are exempt there.
    """
    from analysis.registry import declared_columns, list_derived, list_modules

    declared = set(declared_columns())
    assert not any(name.startswith("genotype_") for name in declared)
    produced = {name for module in (*list_modules(), *list_derived())
                for name in (column.name for column in module.produces)}
    assert declared == produced


# ------------------------------------------------------------- the declaration

def _resolve(value):
    return None if value in (None, "") else value


@pytest.mark.parametrize("name", ["Acquisition", "side table", "2nd", "", "raw"])
def test_a_side_table_name_that_cannot_be_a_column_prefix_is_refused(name):
    with pytest.raises(ValueError):
        SideTableConfig.from_dict({"name": name, "path": "x.csv"}, _resolve)


def test_a_key_this_package_cannot_join_on_is_refused():
    with pytest.raises(ValueError, match="frame_index or identity"):
        SideTableConfig.from_dict(
            {"name": "dose", "path": "x.csv", "keyed_on": "subject"}, _resolve)


def test_a_frame_numbering_this_package_does_not_know_is_refused():
    with pytest.raises(ValueError, match="label or source"):
        SideTableConfig.from_dict(
            {"name": "log", "path": "x.csv", "key_space": "imagej"}, _resolve)


def test_key_space_on_a_cell_key_is_refused_rather_than_ignored():
    """A setting quietly dropped reads exactly like one that was honoured."""
    with pytest.raises(ValueError, match="key_space"):
        SideTableConfig.from_dict(
            {"name": "genotype", "path": "x.csv", "keyed_on": "identity",
             "key_space": "source"}, _resolve)


def test_two_side_tables_with_one_name_are_refused():
    entries = [{"name": "log", "path": "a.csv"}, {"name": "log", "path": "b.csv"}]
    with pytest.raises(ValueError, match="declared twice"):
        _side_tables(entries, _resolve, "stem")


def test_the_defaults_are_a_frame_key_in_this_packages_own_numbering():
    spec = SideTableConfig.from_dict({"name": "log", "path": "x.csv"}, _resolve)
    assert spec.keyed_on == "frame_index"
    assert spec.key_space == "label"
    assert spec.key == "frame_index"
    assert spec.columns is None


def test_the_key_column_defaults_to_the_key_it_joins_on():
    spec = SideTableConfig.from_dict(
        {"name": "log", "path": "x.csv", "key_column": "t"}, _resolve)
    assert spec.key == "t"


# --------------------------------------------------------- through the roll-ups

def test_both_keys_reach_the_three_tables_a_reader_actually_opens(tmp_path):
    """The end of the journey: the real roll-up builders, not the join helper.

    ``join_side`` is called in three places and each one could be omitted
    without any other test noticing. This runs the frame log and the cell calls
    through ``build_cell_summary`` and ``build_frame_summary`` themselves and
    checks the values arrived where a reader would look for them.
    """
    from analysis.summarise import build_cell_summary, build_frame_summary

    log = _csv(tmp_path, "log.csv", pd.DataFrame({
        "frame_index": range(N_FRAMES), "focus": [0.9, 0.8, 0.1, 0.7, 0.7, 0.6]}))
    calls = _csv(tmp_path, "calls.csv", pd.DataFrame({
        "identity": IDENTITIES, "call": ["wt", "ko", "wt"]}))
    frame_side, _ = _load(SideTableConfig(name="acquisition", path=log))
    cell_side, _ = _load(SideTableConfig(name="genotype", path=calls,
                                         keyed_on="identity"))
    context = _context(acquisition=frame_side, genotype=cell_side)

    cell_frame = pd.DataFrame({
        "identity": np.repeat(IDENTITIES, N_FRAMES),
        "frame_index": np.tile(range(N_FRAMES), len(IDENTITIES)),
        "area_px": np.arange(N_FRAMES * len(IDENTITIES), dtype=float),
    })
    cell_frame = join_side(cell_frame, context, "frame_index")
    cell_frame = join_side(cell_frame, context, "identity")

    cell_summary = build_cell_summary(cell_frame, {}, context)
    frame_summary = build_frame_summary(cell_frame, context, {})

    # One fact per cell, on the per-cell table and on every row of that cell.
    assert cell_summary.set_index("identity")["genotype_call"].tolist() == [
        "wt", "ko", "wt"]
    assert set(cell_frame.loc[cell_frame["identity"] == 2, "genotype_call"]) == {"ko"}
    # One fact per frame, on the per-frame table and on every cell in it.
    assert frame_summary["acquisition_focus"].tolist() == [
        0.9, 0.8, 0.1, 0.7, 0.7, 0.6]
    assert set(cell_frame.loc[cell_frame["frame_index"] == 2,
                              "acquisition_focus"]) == {0.1}
    # The measured columns are untouched, and no row was added or lost.
    assert len(cell_frame) == N_FRAMES * len(IDENTITIES)
    assert cell_summary["area_px_median"].notna().all()
