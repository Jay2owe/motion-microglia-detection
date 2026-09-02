"""The time-warped sequence comparison, and what a gap costs.

Three of these can be worked out on paper. The rest fix the two things a reader
is most likely to get wrong about this table: that a frame the cell was not on
screen for is a symbol like any other, and which way round the offset points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.modules.sequence_distance import derive, dtw_symbols, symbol_sequences
from analysis.registry import MeasurementContext
from analysis.units import Scale


def _context(minutes: float = 30.0, params: dict | None = None) -> MeasurementContext:
    return MeasurementContext(
        stem="synthetic",
        labels=np.zeros((2, 2, 2), dtype=np.uint16),
        raw=np.zeros((2, 2, 2), dtype=np.uint16),
        scale=Scale(minutes),
        identities=[1],
        params={"sequence_distance": dict(params or {})},
    )


def _cell_frame(sequences: dict[int, list], minutes: float = 30.0) -> pd.DataFrame:
    """One row per cell per frame, with ``None`` for a frame the cell was absent.

    ``regimes`` leaves a cell-frame it could not classify blank rather than
    guessing at it, so a blank here is the state this module actually meets.
    """
    rows = []
    for identity, states in sequences.items():
        for frame, state in enumerate(states):
            rows.append({"identity": identity, "frame_index": frame,
                         "hours": frame * minutes / 60,
                         "regime": np.nan if state is None else float(state)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- on paper


def test_two_identical_sequences_are_at_no_distance_and_no_offset() -> None:
    """The only pair whose answer needs no arithmetic at all."""
    sequence = np.array([0, 1, 2, 3, 2, 1])
    distance, offset, overlap = dtw_symbols(sequence, sequence, window=2)
    assert distance == 0.0
    assert offset == 0.0
    assert overlap == len(sequence)


def test_two_sequences_that_share_no_symbol_are_at_a_distance_of_one() -> None:
    """Four frames each, every comparison a mismatch, so the mismatch rate is one.

    The cheapest path down a band of width one is the main diagonal: four steps,
    each costing one, over a path of four. The distance is a rate rather than a
    total, which is what stops a long pair of cells looking further apart than a
    short pair that agrees no better.
    """
    distance, offset, overlap = dtw_symbols(np.zeros(4, int), np.ones(4, int), window=1)
    assert distance == pytest.approx(1.0)
    assert offset == 0.0
    assert overlap == 4


def test_a_frame_the_cell_was_absent_for_costs_distance_like_a_mismatch() -> None:
    """``-1`` matches nothing, and that is deliberate rather than an oversight.

    Two cells that were rarely on screen at the same time genuinely did not do
    the same things at the same time. The gap is excluded from the offset and
    from the overlap count - there is no alignment to speak of - but not from the
    distance, and ``alignment_overlap_steps`` is what says how much of a pair's
    distance rests on real frames.
    """
    distance, offset, overlap = dtw_symbols(np.array([0, 1, 0, 1]),
                                            np.full(4, -1), window=1)
    assert distance == pytest.approx(1.0)
    assert np.isnan(offset)
    assert overlap == 0


# --------------------------------------------------------------- the offset


def test_the_offset_says_which_cell_is_running_late() -> None:
    """Positive means the second cell reached the same states later.

    ``regimes`` gives the two cells the same six-state programme, and the second
    starts it two frames after the first, with the frames before its own start
    marked absent. The alignment that costs least is the one that slides the
    second cell back onto the first, so the mean of ``b - a`` over the aligned
    pairs is positive. Swapping the two must flip the sign and nothing else.
    """
    early = np.array([0, 1, 2, 3, 4, 5, 0, 1])
    late = np.array([-1, -1, 0, 1, 2, 3, 4, 5])
    forward = dtw_symbols(early, late, window=3)
    backward = dtw_symbols(late, early, window=3)
    assert forward[1] > 0
    assert backward[1] < 0
    assert forward[0] == pytest.approx(backward[0])
    assert forward[2] == backward[2]


def test_the_offset_is_reported_in_hours_off_the_movies_own_interval() -> None:
    """A frame offset means nothing until it is told how long a frame is.

    It is a mean over the aligned pairs, so it is not a whole number of frames
    and should not be read as one: it is how far the second cell runs behind on
    average, which is the number a reader can compare with a treatment time.
    """
    programme = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    sequences = {1: programme, 2: [None, None, *programme[:-2]]}
    frames = dtw_symbols(*symbol_sequences(_cell_frame(sequences)).values(), window=8)[1]
    for minutes in (15.0, 30.0):
        table = derive(_cell_frame(sequences, minutes=minutes),
                       _context(minutes=minutes))["sequence_distance"]
        row = table.iloc[0]
        assert row["identity_a"] == 1 and row["identity_b"] == 2
        assert row["alignment_offset_hours"] > 0
        assert row["alignment_offset_hours"] == pytest.approx(frames * minutes / 60)


# --------------------------------------------------------------- the table


def test_every_pair_appears_once_with_the_lower_identity_first() -> None:
    """Long, not wide, and each pair counted once rather than in both directions."""
    frame = _cell_frame({5: [0, 1, 0, 1, 0, 1], 2: [1, 1, 0, 0, 1, 1],
                         9: [0, 0, 0, 1, 1, 1]})
    table = derive(frame, _context())["sequence_distance"]
    assert list(zip(table["identity_a"], table["identity_b"])) == [(2, 5), (2, 9), (5, 9)]
    assert (table["identity_a"] < table["identity_b"]).all()


def test_a_frame_no_cell_was_classified_in_is_still_on_the_axis() -> None:
    """Every cell is laid on the same frame axis before any pair is compared.

    Warping compares positions, so two sequences of different lengths would be
    comparing frame 3 of one cell with frame 5 of another and calling the
    difference a delay.
    """
    sequences = symbol_sequences(_cell_frame({1: [0, 1, 2, None, None],
                                              2: [None, 0, 1, 2, 3]}))
    assert sequences[1].tolist() == [0, 1, 2, -1, -1]
    assert sequences[2].tolist() == [-1, 0, 1, 2, 3]


def test_one_cell_writes_no_table_rather_than_an_empty_one() -> None:
    """A pairwise table needs a pair; a movie with one classified cell has none."""
    assert derive(_cell_frame({1: [0, 1, 2, 3]}), _context()) == {}


def test_two_cells_never_on_screen_together_can_still_overlap() -> None:
    """Because that is what warping is for, and it is easy to read the wrong way.

    Cell 1 is classified in the first three frames and cell 2 in the last three,
    so they were never on screen at the same moment. The alignment slides one
    onto the other anyway - three frames apart is well inside a four-hour window
    - and reports three aligned steps with both cells named. The overlap column
    counts steps of the alignment, not simultaneous frames - and a step is not a
    frame either, because warping may hold one cell still while the other moves.
    """
    frame = _cell_frame({1: [0, 1, 2, None, None, None],
                         2: [None, None, None, 0, 1, 2]})
    row = derive(frame, _context())["sequence_distance"].iloc[0]
    assert row["alignment_overlap_steps"] == 3
    assert row["alignment_offset_hours"] == pytest.approx(1.5)


def test_a_pair_the_warping_window_cannot_reach_is_kept_and_says_so() -> None:
    """No overlap is a finding, not a gap, so the row is written by default.

    Twenty frames apart is further than the four-hour window can slide, so no
    step of the alignment has both cells named. The two cells are maximally
    unlike and the distance says so; ``alignment_overlap_steps`` of zero is what
    tells a reader the distance rests on no shared frames, and the offset is
    blank because there is no alignment to average.

    Dropping such pairs by default would quietly remove cells from anything that
    clusters this table, so the floor exists and is off.
    """
    blank = [None] * 17
    frame = _cell_frame({1: [0, 1, 2, *blank], 2: [*blank, 0, 1, 2]})
    kept = derive(frame, _context())["sequence_distance"]
    assert kept["alignment_overlap_steps"].iloc[0] == 0
    assert np.isnan(kept["alignment_offset_hours"].iloc[0])
    assert np.isfinite(kept["dtw_distance"].iloc[0])
    assert derive(frame, _context(params={"min_overlap_steps": 1})) == {}
