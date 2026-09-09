"""A module must declare the columns it writes.

The failure this guards against is silent. ``describe()`` will happily turn
``punctate_fraction`` into "Punctate (fraction)" in the morphology colour rather
than "Punctate fraction" in the reporter colour, put that on every figure that
draws it, and raise nothing. Nothing else in the package notices, because
nothing else has any way to know what a column was supposed to be called.

So these tests run every registered module over a small synthetic movie and
compare what came back against what was declared. A new measurement that
arrives without wording fails here, at the point it is added, rather than on an
axis six months later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

import analysis.modules  # noqa: F401  - registers every module
from analysis.registry import (SHARED_COLUMNS, ChannelStack, Column,
                               MeasurementContext, ObjectStack, declared_columns,
                               get_derived, get_module, list_derived,
                               list_modules)
from analysis.run import _fold_derived, _join_cell_frame
from analysis.theme import load_theme
from analysis.units import Scale

from _metrics import METRICS, RESIDUAL  # noqa: E402


def _movie() -> MeasurementContext:
    """Three cells on a small field, long enough for a rhythm to be fitted.

    Every optional stack is supplied, because a module skipped for want of
    ``evidence`` or ``inferred`` is a module whose columns go unchecked - which
    is exactly the module most likely to have drifted. Forty-eight frames at
    30 minutes is a day, which is the shortest recording the rhythm fits will
    return a full row for.
    """
    n_frames, height, width = 48, 40, 40
    labels = np.zeros((n_frames, height, width), dtype=np.uint16)
    hours = np.arange(n_frames) / 2
    for frame in range(n_frames):
        drift = int(2 * np.sin(2 * np.pi * hours[frame] / 24))
        labels[frame, 8 + frame % 3:16 + frame % 3, 8 + drift:16 + drift] = 1
        labels[frame, 24:32, 22 + frame % 4:30 + frame % 4] = 2
        labels[frame, 4:9, 30:35] = 3

    noise = np.random.default_rng(0).random((n_frames, height, width))
    signal = 100 + 30 * np.sin(2 * np.pi * hours / 24)
    raw = (labels > 0) * signal[:, None, None] + noise * 5
    # Five channels, because that is what the tracker writes and the fifth is
    # ``trail``: a four-channel stack silently skips the two trail columns.
    #
    # Graded rather than boolean, and the trail channel is a genuine ramp. The
    # evidence stack a tracker writes carries magnitude, and a boolean fixture
    # would leave the totals and every age column measuring a flag - which is
    # the exact mistake the graded columns were added to stop.
    full = np.iinfo(np.uint16).max
    evidence = np.zeros((n_frames, 5, height, width), dtype=np.uint16)
    evidence[:, :, 9:15, 9:15] = full
    ramp = (np.arange(36).reshape(6, 6) % 10) + 1          # ten steps, all present
    evidence[:, 4, 9:15, 9:15] = np.rint(ramp * full / 10).astype(np.uint16)
    flagged = np.zeros((n_frames, height, width), dtype=bool)
    flagged[:, 8:10, 8:10] = True

    return MeasurementContext(
        stem="declarations", labels=labels, raw=raw, scale=Scale(30.0),
        identities=[1, 2, 3], unclaimed=(labels == 0) & (noise > 0.95),
        evidence=evidence, inferred=flagged, unresolved=flagged, added=flagged,
        channels={"extra": _channel(labels, noise, n_frames)},
        objects={"scenery": _objects(n_frames, height, width)},
        side=_side(n_frames),
    )


def _objects(n_frames: int, height: int, width: int) -> ObjectStack:
    """Two reference shapes, one of which moves and one of which does not.

    A stationary shape and a drifting one, because the two exercise different
    things: the stationary one gives every cell a fixed distance to check
    against, and the drifting one is what would go unnoticed if a module read
    frame 0's shapes for every frame. One of them touches the field edge on
    purpose, which is a state a shape measurement has to report rather than
    quietly average over.
    """
    values = np.zeros((n_frames, height, width), dtype=np.int32)
    for frame in range(n_frames):
        values[frame, 18:26, 2:8] = 1                       # fixed, mid-field
        drift = frame % 5
        values[frame, 0:4, 30 + drift:36 + drift] = 2       # moves, touches the top
    return ObjectStack(name="scenery", values=values, static=False,
                       path="synthetic",
                       description="two shapes, for the declarations")


def _side(n_frames: int) -> dict[str, pd.DataFrame]:
    """One side table of each key, in the shape the loader hands them over.

    Nothing here is checked against a declaration and nothing should be: a side
    table's columns are the user's words, carried through under the table's own
    name as a prefix, and this package naming them would be it inventing a
    meaning for a column it copied. They are in the fixture so that a module
    that starts reading ``context.side`` is exercised the moment it is written,
    rather than first meeting real data in a real run.
    """
    return {
        "acquisition": pd.DataFrame({
            "frame_index": np.arange(n_frames),
            "acquisition_focus": np.linspace(1.0, 0.5, n_frames),
            "acquisition_suspect": (np.arange(n_frames) == 11).astype(int),
        }),
        "genotype": pd.DataFrame({
            "identity": [1, 2, 3],
            "genotype_call": ["wt", "ko", "wt"],
        }),
    }


def _channel(labels: np.ndarray, noise: np.ndarray, n_frames: int) -> ChannelStack:
    """A second imaging channel, built to exercise what a real one does.

    Three things are deliberate. It fades over the recording, because a channel
    that does not would let a per-cell trend pass whether or not the field-wide
    trend is subtracted. It differs between cells, because otherwise the
    per-cell tables would be a constant. And it carries both a clipped corner
    and a strip of ``NaN``, because those are the two states a measurement can
    silently report as a brightness: a pixel at the ceiling was not measured,
    and a pixel the alignment could not reach is not there at all.
    """
    fade = np.linspace(1.0, 0.55, n_frames)[:, None, None]
    values = (300.0 + 90.0 * (labels == 1) + 180.0 * (labels == 2) + 25.0 * noise) * fade
    values = values.astype(np.float32)
    values[:, 0:3, 0:3] = float(np.iinfo(np.uint16).max)     # clipped
    values[:, :, -2:] = np.nan                               # off the source
    return ChannelStack(
        name="extra", values=values, source_dtype="uint16",
        saturation_value=float(np.iinfo(np.uint16).max),
        path="synthetic", description="a second channel, for the declarations",
    )


def _measured(context: MeasurementContext) -> dict[str, dict[str, pd.DataFrame]]:
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for module in list_modules():
        available, why = module.available(context)
        assert available, why
        tables[module.name] = {
            name: frame for name, frame in module.measure(context).items()
            # territory also returns per-pixel stacks, which are images and
            # have no columns to declare.
            if isinstance(frame, pd.DataFrame)
        }
    return tables


def _everything() -> dict[str, dict[str, pd.DataFrame]]:
    """Every table every registered module writes, module by module.

    The derived modules are handed a cell-frame that grows as they run, which is
    what ``run.analyse_movie`` does and for the same reason: ``sequence_distance``
    reads the state number ``regimes`` folds in, so a fixture that folded only at
    the end would exercise the module against a table no run ever produces.
    """
    context = _movie()
    written = _measured(context)
    tables = {name: frame for group in written.values() for name, frame in group.items()}
    cell_frame = _join_cell_frame(tables, context)
    for module in list_derived():
        cell_frame = _fold_derived(cell_frame, tables, set(tables))
        missing = [c for c in module.needs_columns if c not in cell_frame.columns]
        assert not missing, f"{module.name} needs {missing}, which no earlier module wrote"
        produced = {name: frame for name, frame in module.derive(cell_frame, context).items()
                    if isinstance(frame, pd.DataFrame)}
        written[module.name] = produced
        tables.update(produced)
    return written


#: Tables whose column names are not this package's to declare, and why.
#:
#: ``regime_profiles`` is one column per entry of the ``feature_columns``
#: setting, so its shape is configuration: it carries other modules'
#: measurements through unaltered, already labelled by whoever measured them.
#: The ``history_*`` copies are the tracker's own CSVs, reproduced verbatim on
#: purpose - naming their columns here would be this package paraphrasing an
#: account it deliberately does not paraphrase. ``history_sources`` and
#: ``history_join_audit`` are bookkeeping about the copy itself.
NOT_OURS_TO_NAME = {
    "regime_profiles": "one column per configured feature, declared by the module that measures each",
    "history_sources": "bookkeeping: which decision tables were found",
    "history_join_audit": "bookkeeping: how the copied tables joined",
}


def _exempt(table: str) -> bool:
    return table in NOT_OURS_TO_NAME or table.startswith("history_")


ALL_TABLES = _everything()


@pytest.mark.parametrize("name", sorted(ALL_TABLES))
def test_every_column_a_module_writes_is_declared(name: str) -> None:
    """No module writes a column that nothing has words for.

    Checked against every module's declarations rather than only the one under
    test, because a column legitimately has more than one producer: ``regimes``
    passes ``area_px`` through from ``morphology``, ``coupling`` reports a
    ``lag_frames`` that means what ``motility`` means by it. What must never
    happen is a column arriving that nobody has described - and two modules
    describing one column differently is refused separately, by
    ``registry.declared_columns``.
    """
    known = set(declared_columns()) | SHARED_COLUMNS
    for table, frame in ALL_TABLES[name].items():
        if _exempt(table):
            continue
        undeclared = sorted(set(frame.columns) - known)
        assert not undeclared, (
            f"{name} writes {', '.join(undeclared)} into {table}.csv without declaring "
            f"it. Add a Column(...) to that module's PRODUCES, or these end up on "
            f"figures with a label invented from the column name."
        )


def test_the_check_covers_every_registered_module() -> None:
    """No allow-list: a new module is checked the moment it is registered."""
    registered = {module.name for module in (*list_modules(), *list_derived())}
    assert set(ALL_TABLES) == registered


def test_the_only_unchecked_tables_are_ones_with_a_written_reason() -> None:
    """An exemption has to be a sentence, not a name quietly added to a set."""
    exempt = {table for tables in ALL_TABLES.values() for table in tables if _exempt(table)}
    for table in exempt:
        assert table.startswith("history_") or NOT_OURS_TO_NAME[table]


def test_no_module_declares_a_column_that_says_which_row_this_is() -> None:
    """``identity`` and ``hours`` are the index, not a measurement.

    They are on every table, so declaring them would put the same entry in
    fourteen modules and invite fourteen different opinions about what ``hours``
    means.
    """
    for module in (*list_modules(), *list_derived()):
        overlap = sorted({c.name for c in module.produces} & SHARED_COLUMNS)
        assert not overlap, f"{module.name} declares the index column(s) {overlap}"


def test_a_column_two_modules_write_has_one_meaning() -> None:
    """Both centroid columns and ``gap_frames`` have two producers by design."""
    columns = declared_columns()
    assert columns["centroid_x"].unit == "px"
    assert columns["gap_frames"].label == "Frames missing"


def test_two_modules_disagreeing_about_a_column_is_refused(monkeypatch) -> None:
    """The join keeps one copy, so the label would depend on module sort order."""
    from analysis import registry

    morphology = registry.get_module("morphology")
    monkeypatch.setattr(
        morphology, "produces",
        (Column("centroid_x", "Somewhere else entirely", "furlongs", "reporter"),),
    )
    with pytest.raises(ValueError, match="centroid_x"):
        registry.declared_columns()


def test_every_declared_column_names_a_role_the_theme_can_resolve() -> None:
    theme = load_theme()
    for name, column in declared_columns().items():
        assert theme.colour(column.role).startswith("#"), name


def test_history_declares_only_what_it_computes() -> None:
    """It copies the tracker's tables through; those columns are not ours.

    Worth asserting rather than leaving to the exemption above, because the
    tempting change is to declare the copied columns too - and then this
    package would be asserting what a column of somebody else's CSV means.
    """
    declared = {column.name for column in get_derived("history").produces}
    assert declared == {"mechanism", "still_missing_in_accepted_labels",
                        "identity_in_accepted_labels", "silent_nonborder_ending",
                        # The keys of its own two bookkeeping tables, which this
                        # module invents rather than copies. Nothing from a
                        # copied table appears here, which is the point.
                        "table", "check"}


def test_the_vocabulary_is_the_declarations_plus_what_no_module_writes() -> None:
    """``RESIDUAL`` is for producerless columns only.

    An entry that is both declared and residual is the drift this stage exists
    to remove: two spellings of one column, and whichever loaded last wins.
    """
    declared = declared_columns()
    both = sorted(set(RESIDUAL) & set(declared))
    assert not both, f"declared by a module and still listed in RESIDUAL: {both}"
    assert set(METRICS) == set(RESIDUAL) | set(declared)


def test_the_vocabulary_costs_nothing_until_something_asks() -> None:
    """``_metrics`` is imported by every figure; the modules are not cheap.

    Building the vocabulary imports scikit-image and scipy through
    ``analysis.modules``. Doing that at import time would make every figure pay
    for it, including the ones that never label a column.
    """
    import _metrics

    fresh = _metrics._Vocabulary(dict(RESIDUAL))
    assert "not yet built" in repr(fresh)
    assert fresh["turnover_index"].label == "Footprint turnover fraction"
    assert "not yet built" not in repr(fresh)
