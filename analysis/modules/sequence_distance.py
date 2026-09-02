"""How alike two cells' behaviour is, allowing for one running late.

``regimes`` gives every cell-frame a state number. This turns each cell into a
sequence of those numbers and compares every pair of sequences, letting one be
stretched or slid in time so that two cells doing the same thing at different
hours score as similar rather than as different.

The everyday version is two people following the same recipe at different
speeds. Compared step by step against the clock they look nothing alike; lined
up by what they are doing, they are doing the same thing.

The offset is signed and says which of the pair is running late, which is often
more interesting than the distance itself: a programme every cell runs,
staggered, is a different finding from a programme only some cells run.

Three things a reader has to know before quoting a number from this table.

**A frame where a cell was not on screen is a symbol too.** Missing frames are
filled with ``-1``, which matches nothing, so a gap costs distance exactly like
a mismatch does. It is excluded from the offset and from the overlap count but
not from the distance, because two cells that were rarely visible at the same
time genuinely did not do the same things at the same time. That is the figure's
long-standing behaviour and it is kept; ``alignment_overlap_steps`` is what
says how much of a pair's distance rests on real frames.

It counts **steps of the alignment**, not frames, and the two are not the same
number: warping may hold one cell still while the other advances, so on the
pinned 99-frame movie the count runs as high as 193. It is also not a count of
simultaneous frames - the alignment may line up a frame of one cell with a
different frame of the other, so two cells that were never on screen together
still overlap as long as the gap is inside the warping window. Beyond that
window the alignment cannot reach and the overlap is zero.

**A pair with no overlap is still a pair.** ``min_overlap_steps`` is zero by
default, so the table holds every pair of classified cells and
``alignment_overlap_steps`` is what says which distances rest on shared frames.
On the pinned movie 95 of 3,828 pairs have no overlap at all.

**Regime numbers are cluster labels.** They mean nothing outside the run that
made them, so this table cannot be compared across runs unless both were
clustered with the same seed. ``regimes`` is seeded (``random_state``), so two
runs of the same movie agree; two different movies do not share a state 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register_derived

DEFAULTS = {
    # Copied from 33_regime_programmes.py so the module reproduces the figure.
    # Four hours of warping: enough that two cells doing the same thing half a
    # night apart are called alike, little enough that "alike" still means
    # something about timing.
    "window_hours": 4.0,
    # Off by default, so every pair of classified cells gets a row. A pair whose
    # alignment never had both cells on screen still has a real distance - two
    # sequences that never coexisted are maximally unlike, which is a finding
    # rather than a gap - and dropping it here would quietly remove cells from
    # anything that clusters this table. Raise it to filter, which the package
    # prefers to be an explicit choice a reader can see.
    "min_overlap_steps": 0,
}


def dtw_symbols(left: np.ndarray, right: np.ndarray, window: int) -> tuple[float, float, int]:
    """Banded dynamic-time-warping distance and mean signed alignment offset.

    The distance is the cheapest alignment of the two sequences divided by its
    own length, so it is a mismatch rate rather than a total and two long cells
    are not automatically further apart than two short ones. The offset is the
    mean of ``b - a`` over the aligned pairs where both cells were on screen:
    positive means the second cell is reached later, which is what "running
    late" means here.

    Moved verbatim from ``analysis.figures._derive``, which now imports it from
    here, so the figure and the table cannot drift apart.
    """
    left = np.asarray(left, dtype=int); right = np.asarray(right, dtype=int)
    n, m = len(left), len(right)
    width = max(int(window), abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf); cost[0, 0] = 0.0
    parent = np.full((n + 1, m + 1, 2), -1, dtype=int)
    for i in range(1, n + 1):
        for j in range(max(1, i - width), min(m, i + width) + 1):
            choices = (cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            choice = int(np.argmin(choices))
            previous = ((i - 1, j), (i, j - 1), (i - 1, j - 1))[choice]
            cost[i, j] = choices[choice] + float(left[i - 1] != right[j - 1])
            parent[i, j] = previous
    if not np.isfinite(cost[n, m]):
        return np.nan, np.nan, 0
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = parent[i, j]
        if i < 0 or j < 0:
            break
    overlapping = [(a, b) for a, b in path if left[a] >= 0 and right[b] >= 0]
    return (float(cost[n, m] / max(len(path), 1)),
            float(np.mean([b - a for a, b in overlapping])) if overlapping else np.nan,
            len(overlapping))


def symbol_sequences(cell_frame: pd.DataFrame) -> dict[int, np.ndarray]:
    """One equal-length sequence of state numbers per cell, gaps marked ``-1``.

    Equal length because dynamic time warping compares positions, so every cell
    has to be laid on the same frame axis before any pair is compared. ``-1``
    is a symbol that matches nothing, which is the honest cost for a frame one
    cell was named in and the other was not.
    """
    rows = cell_frame.dropna(subset=["regime"])
    if rows.empty:
        return {}
    table = rows.pivot_table(index="identity", columns="frame_index",
                             values="regime", aggfunc="first")
    frames = np.arange(int(rows["frame_index"].max()) + 1)
    table = table.reindex(columns=frames)
    return {int(identity): row.fillna(-1).to_numpy(int)
            for identity, row in table.iterrows()}


PRODUCES = (
    # The first two repeat ``contacts``' wording exactly rather than inventing a
    # second opinion about what a pair is; a column has one meaning or the axis
    # label depends on which module sorted first.
    Column("identity_a", "First cell of the pair", "", "reference"),
    Column("identity_b", "Second cell of the pair", "", "reference"),
    Column("dtw_distance", "Sequence mismatch after time alignment", "fraction", "morphology"),
    Column("alignment_offset_hours", "How far the second cell runs late", "h", "morphology"),
    Column("alignment_overlap_steps", "Aligned steps with both cells named", "steps", "morphology"),
)

#: One table, and it does not fold: a roll-up is keyed on a cell, a frame or
#: both, and this is keyed on a *pair*. ``contacts`` is keyed the same way and
#: is also a file of its own.
WRITES = (
    Output("sequence_distance", grain=("identity_a", "identity_b")),
)


@register_derived(
    name="sequence_distance",
    description="How alike two cells' state sequences are, allowing one to run late",
    needs_columns=("identity", "frame_index", "regime"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("sequence_distance")}
    sequences = symbol_sequences(cell_frame)
    identities = sorted(sequences)
    if len(identities) < 2:
        return {}
    minutes = float(context.scale.minutes_per_frame)
    window_frames = max(1, int(round(float(params["window_hours"]) * 60 / minutes)))
    min_overlap = int(params["min_overlap_steps"])

    rows: list[dict] = []
    for left_index, identity_a in enumerate(identities):
        for right_index in range(left_index + 1, len(identities)):
            identity_b = identities[right_index]
            distance, offset, overlap = dtw_symbols(
                sequences[identity_a], sequences[identity_b], window_frames)
            if overlap < min_overlap:
                continue
            rows.append({
                "identity_a": identity_a,
                "identity_b": identity_b,
                "dtw_distance": distance,
                "alignment_offset_hours": offset * minutes / 60,
                "alignment_overlap_steps": overlap,
            })
    return {"sequence_distance": pd.DataFrame(rows)} if rows else {}
