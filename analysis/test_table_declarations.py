"""A module must declare the tables it writes, and what one row of each is.

``test_column_declarations`` guards the wording of a number. This guards the
shape of a table. The failure it catches is the one the run writer used to make
on every module's behalf: whether a table becomes a file of its own or columns
of a shared table at the same grain was decided by a hard-coded set of six
names in ``run.py``, so the answer to "what will my new module write?" was in a
file the module's author had no reason to open.

The checks run every module over the same small synthetic movie the column
declarations use, and compare what came back against what was declared. A table
that arrives without a grain, or with a grain that is not actually unique, fails
here rather than three stages downstream when something tries to join on it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import analysis.modules  # noqa: F401  - registers every module
from analysis.registry import (SHARED_COLUMNS, Output, declared_tables,
                               get_derived, get_module, list_derived,
                               list_modules)

# The same synthetic movie and the same measured tables the column
# declarations are checked against, so a grain is checked against the rows the
# columns were checked in rather than against a second, differently shaped
# fixture that could drift from it.
from analysis.test_column_declarations import ALL_TABLES

#: Every registered module, measurement and derived alike. No allow-list: a new
#: module is checked the moment it is registered.
CHECKED = {module.name for module in (*list_modules(), *list_derived())}


def _returned() -> dict[str, dict[str, pd.DataFrame]]:
    return {name: tables for name, tables in ALL_TABLES.items() if name in CHECKED}


@pytest.mark.parametrize("name", sorted(CHECKED))
def test_every_table_a_module_writes_is_declared(name: str) -> None:
    """No module writes a file nothing has described.

    An undeclared table is one the writer has to guess about: it cannot know
    whether the table is a file or a fold, whether this package measured it or
    copied it, or what a row of it is.
    """
    declared = declared_tables()
    undeclared = sorted(set(_returned()[name]) - set(declared))
    assert not undeclared, (
        f"{name} returns {', '.join(undeclared)} without declaring it. Add an "
        f"Output(...) to that module's WRITES, or the run writer decides the "
        f"shape of the file on the module's behalf."
    )


@pytest.mark.parametrize("name", sorted(CHECKED))
def test_every_declared_grain_column_is_in_the_table(name: str) -> None:
    """A grain is a promise about columns that are actually there."""
    declared = declared_tables()
    for table, frame in _returned()[name].items():
        if table not in declared or frame.empty:
            continue
        missing = sorted(set(declared[table].grain) - set(frame.columns))
        assert not missing, (
            f"{name} declares {table} at a grain of {', '.join(missing)}, which "
            f"the table does not have. Columns present: {sorted(frame.columns)}"
        )


@pytest.mark.parametrize("name", sorted(CHECKED))
def test_every_declared_grain_is_actually_unique(name: str) -> None:
    """One row per what, checked rather than asserted.

    A grain that repeats is the expensive mistake, because everything
    downstream joins on it: a merge against a non-unique key silently multiplies
    rows instead of failing.
    """
    declared = declared_tables()
    for table, frame in _returned()[name].items():
        if table not in declared or frame.empty:
            continue
        grain = list(declared[table].grain)
        if not grain or set(grain) - set(frame.columns):
            continue
        repeated = int(frame.duplicated(subset=grain).sum())
        assert repeated == 0, (
            f"{name} declares {table} at one row per {', '.join(grain)}, but "
            f"{repeated} row(s) repeat that key. Either the grain is wrong or "
            f"the table is."
        )


def test_a_grain_is_made_of_columns_something_produces() -> None:
    """Otherwise the table promises a key nothing can join on."""
    from analysis.registry import declared_columns

    known = set(declared_columns()) | SHARED_COLUMNS
    for name, output in declared_tables().items():
        unknown = sorted(set(output.grain) - known)
        assert not unknown, f"{name} is keyed on undeclared column(s) {unknown}"


def test_the_package_declares_every_table_it_writes() -> None:
    """A count, so a module cannot quietly stop declaring one.

    Twenty-eight from the fourteen measurement modules, nineteen from regimes,
    rhythms, coupling, territory_shape, walk, recurrence, sequence_distance and
    trend, twenty-one from history - sixteen of which are the tracker's own
    tables copied through. The derived total includes ``rhythm_traces``: the
    Circadian Workbench-detrended trace that downstream plots can reuse without
    silently choosing another baseline. If this number changes it should
    change because a table was deliberately added or removed, and the completion
    note should say which.
    """
    declared = declared_tables()
    assert len(declared) == 68, sorted(declared)
    assert len({m.name for m in list_modules() for o in m.writes}) == 14
    assert sum(1 for o in declared.values() if o.origin == "tracker") == 16


def test_two_modules_writing_one_table_name_is_refused(monkeypatch) -> None:
    """The second one silently overwrites the first in the results dictionary."""
    from analysis import registry

    monkeypatch.setattr(
        get_module("intensity"), "writes",
        (Output("morphology", grain=("identity", "frame_index")),),
    )
    with pytest.raises(ValueError, match="morphology"):
        registry.declared_tables()


def test_a_measured_table_without_a_grain_is_refused(monkeypatch) -> None:
    """Only a copied table may decline to say what one row is."""
    from analysis import registry

    monkeypatch.setattr(get_module("sholl"), "writes", (Output("sholl", grain=()),))
    with pytest.raises(ValueError, match="without a grain"):
        registry.declared_tables()


def test_a_grain_naming_a_column_nothing_writes_is_refused(monkeypatch) -> None:
    from analysis import registry

    monkeypatch.setattr(
        get_module("sholl"), "writes", (Output("sholl", grain=("furlongs",)),))
    with pytest.raises(ValueError, match="furlongs"):
        registry.declared_tables()


def test_presence_is_at_cell_frame_grain_and_is_not_folded() -> None:
    """The one place grain and fold have to disagree, asserted so it stays that way.

    ``presence`` records the frames a cell was *absent* as well as the frames it
    was seen in, so it has more rows than ``cell_frame`` does. Deriving ``fold``
    from ``grain`` would fold it, and every absence in the package would vanish
    without a single test failing.
    """
    declared = declared_tables()
    assert declared["presence"].grain == ("identity", "frame_index")
    assert declared["presence"].fold is False
    assert declared["morphology"].grain == declared["presence"].grain
    assert declared["morphology"].fold is True


def test_a_copied_table_may_decline_to_say_what_one_row_is() -> None:
    """The sixteen tracker tables are reproduced without comment, grain included.

    Naming their grain would be this package asserting what one row of somebody
    else's CSV means. The empty grain is allowed for exactly that reason and
    only for them, which the refusal above enforces from the other side.
    """
    declared = declared_tables()
    copied = {name: o for name, o in declared.items() if o.origin == "tracker"}
    assert copied, "no copied tables declared"
    for name, output in copied.items():
        assert output.grain == (), name
        assert output.optional is True, name
        assert not output.fold, name
    for name, output in declared.items():
        if output.origin == "measured":
            assert output.grain, f"{name} is measured here and must say what a row is"


def test_a_movie_with_no_tracking_history_still_runs() -> None:
    """Twenty-one declared tables that were not written is a skip, not an error.

    ``history`` returns nothing at all when the decision folder is absent, which
    is the normal case for a movie whose chain did not produce one.
    """
    from analysis.test_column_declarations import _movie

    context = _movie()
    assert context.module_params("history") == {}
    assert get_derived("history").derive(pd.DataFrame(
        {"identity": [1], "frame_index": [0]}), context) == {}
    for name, output in declared_tables().items():
        if name.startswith("history_"):
            assert output.optional, name


def test_every_folded_table_has_a_rollup_at_its_grain() -> None:
    """A fold with nowhere to go is a table that silently stops being written.

    ``_fold_target`` returns ``None`` both for a table that is a file and for a
    table that declared ``fold=True`` at a grain no roll-up shares. The first is
    correct; the second would write the file anyway and leave the declaration
    quietly meaning nothing.
    """
    from analysis.run import _ROLLUP_BY_GRAIN, _fold_target

    declared = declared_tables()
    for name, output in declared.items():
        if not output.fold:
            continue
        assert output.grain in _ROLLUP_BY_GRAIN, (
            f"{name} says it folds, but no roll-up is at one row per "
            f"{', '.join(output.grain) or 'nothing'}. Roll-ups are declared in "
            f"analysis/summarise.py:ROLLUPS."
        )
        assert _fold_target(name, declared) is not None


def test_the_join_and_the_writer_ask_the_same_question() -> None:
    """Six names in run.py used to answer this, and two places read them.

    ``presence`` is the case that matters: same grain as ``cell_frame``, and it
    must still be its own file.
    """
    from analysis.run import _fold_target

    declared = declared_tables()
    folded = {n for n in declared if _fold_target(n, declared) == "cell_frame"}
    assert "presence" not in folded
    assert {"morphology", "intensity", "motility", "surveillance",
            "motion_evidence", "provenance"} <= folded
    # The two that stage 03 of the output-schema plan added, and the derived one.
    assert {"sholl_reach", "territory_frame", "regimes"} <= folded
    # Stages 04, 05 and 06 of the metric-expansion plan.
    assert {"neighbours", "territory_shape_frame", "walk"} <= folded

    per_cell = {n for n in declared if _fold_target(n, declared) == "cell_summary"}
    assert per_cell == {"motility_tracks", "territory", "territory_shape", "walk_tracks"}

    per_frame = {n for n in declared if _fold_target(n, declared) == "frame_summary"}
    assert per_frame == {"neighbour_frame", "walk_frame"}

    # Nothing the channels module writes folds anywhere, and that is the design
    # rather than an oversight: every one of its tables is additionally keyed on
    # which channel the row is about, so folding one into a roll-up would need a
    # column per channel and the schema would then depend on the configuration.
    assert all(not declared[name].fold
               for name in ("channels", "channel_frame", "channel_tracks"))
    # The object tables for exactly the same reason, one word changed: they are
    # additionally keyed on which set of shapes the row is about.
    assert all(not declared[name].fold
               for name in ("objects", "cell_objects", "cell_object_tracks"))


def test_the_channel_tables_are_keyed_on_which_channel_a_row_is_about() -> None:
    """Long, not wide: a second channel adds rows and never adds columns.

    This is the assertion that stops the tempting rewrite. Folding these into
    ``cell_frame`` as ``green_mean``, ``dapi_mean`` and so on reads better for
    one dataset and makes the column list a function of the configuration, so
    every test, figure and roll-up that names a column breaks the first time
    somebody images a third dye.
    """
    declared = declared_tables()
    assert declared["channels"].grain == ("identity", "frame_index", "channel")
    assert declared["channel_frame"].grain == ("frame_index", "channel")
    assert declared["channel_tracks"].grain == ("identity", "channel")
    for name in ("channels", "channel_frame", "channel_tracks"):
        assert "channel" in declared[name].grain, name


def test_the_object_tables_are_keyed_on_which_set_a_row_is_about() -> None:
    """Long, not wide: a second set of shapes adds rows and never adds columns.

    The same assertion as the channels one above, and it stops the same
    tempting rewrite. Folding these in as ``vessel_distance_px`` and
    ``plaque_distance_px`` reads better for one dataset and makes the column
    list a function of the settings file, so every test, figure and roll-up
    that names a column breaks the first time somebody declares a third set.
    """
    declared = declared_tables()
    assert declared["objects"].grain == ("object_set", "object", "frame_index")
    assert declared["cell_objects"].grain == ("identity", "frame_index", "object_set")
    assert declared["cell_object_tracks"].grain == ("identity", "object_set")
    for name in ("objects", "cell_objects", "cell_object_tracks"):
        assert "object_set" in declared[name].grain, name


def test_a_movie_with_no_extra_channels_skips_the_module_rather_than_failing() -> None:
    """An empty ``channels`` is absent, not present-and-empty.

    The field is a dictionary so that no extra channels is the default rather
    than a special case, which means the availability check cannot be ``is
    None`` for it the way it is for every other optional stack.
    """
    from analysis.test_column_declarations import _movie

    context = _movie()
    assert get_module("channels").available(context)[0]
    context.channels = {}
    available, reason = get_module("channels").available(context)
    assert not available and "channels" in reason


def test_a_copied_table_is_written_beside_the_measured_ones_not_among_them() -> None:
    """``origin`` picks the folder, so a run folder says which half is whose."""
    from analysis.run import MEASURED_FOLDER, TRACKER_FOLDER, _folder_for

    declared = declared_tables()
    for name, output in declared.items():
        expected = TRACKER_FOLDER if output.origin == "tracker" else MEASURED_FOLDER
        assert _folder_for(output) == expected, name
    copied = [n for n, o in declared.items() if _folder_for(o) == TRACKER_FOLDER]
    assert len(copied) == 16
    assert all(name.startswith("history_") for name in copied)
    # The five history_ tables this package computes stay with the measured
    # ones despite the prefix. That is the prefix and the folder disagreeing,
    # and it is why renaming them is on the known-limits list.
    computed = [n for n, o in declared.items()
                if n.startswith("history_") and o.origin == "measured"]
    assert sorted(computed) == ["history_gap_frames", "history_join_audit",
                                "history_lifespans", "history_residency",
                                "history_sources"]


def test_a_figure_finds_a_copied_table_in_either_layout(tmp_path) -> None:
    """Runs written before the split must keep rebuilding.

    ``tables/`` is tried first, so in a flat run the second candidate is never
    reached and the old layout costs nothing - no version flag, no detection.
    """
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "figures"))
    from _bundle import require_table

    for layout in ("tables", "tracker"):
        movie = tmp_path / layout
        (movie / layout).mkdir(parents=True)
        (movie / layout / "history_gap_runs.csv").write_text("identity\n1\n", encoding="utf-8")
        found = require_table(movie / "tables", "history_gap_runs.csv", "history")
        assert found.parent.name == layout

    with pytest.raises(SystemExit, match="history"):
        require_table(tmp_path / "empty" / "tables", "history_gap_runs.csv", "history")


def test_a_table_is_written_to_nine_significant_figures(tmp_path) -> None:
    """A quarter off the size of a run, and nothing a figure can amplify.

    Nine significant figures can move a value by at most 5e-9 of itself. That is
    the tolerance to compare two runs at. Six was tried first and was enough for
    every measured column, but a figure that re-derives a z-score or a surrogate
    band from the file subtracts nearly-equal numbers and turned 5e-6 into a
    visible wobble; nine leaves nothing to amplify.
    """
    import numpy as np

    from analysis.run import _write_table

    values = pd.DataFrame({
        "mean_of_16_bit_pixels": [12148.314285714287],
        "a_p_value_near_zero": [1.4551764212882648e-15],
        "an_exact_integer": [42],
    })
    path = tmp_path / "precision.csv"
    _write_table(values, path)
    text = path.read_text(encoding="utf-8")
    assert "12148.3143" in text
    assert "1.45517642e-15" in text       # the exponent survives
    assert "42" in text

    back = pd.read_csv(path)
    for column in values.columns:
        assert np.allclose(values[column].astype(float), back[column].astype(float),
                           rtol=5e-9, atol=0, equal_nan=True), column
