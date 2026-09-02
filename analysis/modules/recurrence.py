"""Whether a cell returns to states it has been in before.

Every frame of a cell is described by the same handful of measurements, and then
compared with every other frame of that cell. Two frames close together in that
description are the cell in "the same state". The everyday version is a diary:
recurrence rate asks how often a day resembles an earlier day, determinism asks
whether those resemblances come in whole repeated stretches rather than odd
days, and laminarity asks whether the cell simply sat still.

The threshold for "the same state" is each cell's own distance quantile, so a
cell that barely changes is not judged against one that never stops. That also
means the threshold is not comparable between cells as an absolute distance -
``recurrence_threshold_distance`` is written down for exactly that reason.

Recurrence looks impressive in slowly drifting noise, so every number here is
written beside its value in matched noise: an AR(1) surrogate with the same
variance and the same frame-to-frame correlation as the real trace and no
rhythm. A recurrence rate above the band is evidence; one inside it is drift.

This arithmetic used to live inside ``figures/32_recurrence_wall.py``, where it
ran for the sixteen cells that fitted on a page and was thrown away when the
figure closed. Three deliberate differences from that version, and only three:

1. **Every cell, not a drawn sample.** The figure's ``--cells`` is a drawing
   choice. ``min_observations`` is the only filter here, and a cell that fails
   it gets no row rather than a row of blanks, which is the convention
   ``rhythms`` already uses.
2. **Each cell draws its own noise.** The figure used one generator for the
   whole page, so a cell's noise band depended on how many cells had been drawn
   before it. Measuring every cell makes that dependency meaningless, so the
   generator is seeded per cell from ``random_state`` and the identity. A cell's
   band is now a property of the cell.
3. **The frame interval comes off the measurement context**, not off a figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from analysis.modules.rhythms import _detrend, _surrogate
from analysis.registry import Column, MeasurementContext, Output, register_derived

DEFAULTS = {
    # Copied from 32_recurrence_wall.py so the module reproduces the figure.
    # The state vector is the same nine measurements ``regimes`` clusters on:
    # shape, branching, footprint turnover and step size, which between them
    # describe what a microglial cell is doing without naming any one family.
    "metrics": [
        "area_px",
        "circularity",
        "solidity",
        "ramification_index",
        "aspect_ratio",
        "skeleton_branches",
        "turnover_index",
        "step_px_gapless",
        "punctateness",
    ],
    # The closest tenth of frame pairs counts as "the same state". A quantile
    # rather than a distance, so a quiet cell is not judged against a busy one.
    "threshold_quantile": 0.1,
    "max_lag_hours": 12.0,
    "surrogates": 100,
    "null_model": "ar1",
    "random_state": 20260825,
    # Four frames is the smallest matrix with a diagonal run of two in it.
    "min_observations": 4,
}


def recurrence_runs(matrix: np.ndarray) -> tuple[float, float]:
    """Fraction of recurrent points in diagonal and vertical runs of at least two.

    A diagonal run is the cell repeating a *sequence* - it came back to a state
    and then carried on doing what it did last time. A vertical run is the cell
    holding still while time passes on the other axis. The two answer different
    questions and a matrix can score high on one and low on the other.

    Moved verbatim from ``analysis.figures._derive``, which now imports it from
    here, so the figure and the table cannot drift apart.
    """
    values = np.asarray(matrix, dtype=bool).copy()
    np.fill_diagonal(values, False)
    total = int(values.sum())
    if not total:
        return 0.0, 0.0
    diagonal = np.zeros_like(values)
    vertical = np.zeros_like(values)
    for offset in range(-len(values) + 1, len(values)):
        line = np.diagonal(values, offset=offset)
        padded = np.r_[False, line, False]
        starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
        ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
        for start, end in zip(starts, ends):
            if end - start >= 2:
                indices = np.arange(start, end)
                if offset >= 0:
                    diagonal[indices, indices + offset] = True
                else:
                    diagonal[indices - offset, indices] = True
    for column in range(values.shape[1]):
        padded = np.r_[False, values[:, column], False]
        starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
        ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
        for start, end in zip(starts, ends):
            if end - start >= 2:
                vertical[start:end, column] = True
    return float((diagonal & values).sum() / total), float((vertical & values).sum() / total)


#: How small a detrended spread has to be, against the metric's own magnitude,
#: before it is treated as no spread at all.
#:
#: A metric that never moves does not detrend to exactly zero: fitting a line
#: through a constant leaves residues of order 1e-16, and standardising divides
#: by their spread, which turns rounding error into a state vector of order one.
#: Testing ``spread == 0`` never fires, so the guard it looks like it provides
#: is not there. This is many orders below the variation of any real
#: measurement and many orders above float noise, so it separates the two
#: without touching a single real trace.
_NEGLIGIBLE_SPREAD = 1e-12


def state_matrix(hours: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Detrended, standardised state vectors: one row per frame.

    Detrended because a cell that is steadily growing would otherwise resemble
    only its immediate neighbours in time, and standardised because a distance
    that mixed pixel areas with a 0-1 solidity would be a distance in pixel
    areas. A metric that never moves is divided by one rather than by its own
    rounding error, which leaves it contributing nothing - the honest answer for
    a measurement that carries no information about this cell.
    """
    detrended = np.column_stack(
        [_detrend(hours, values[:, column], "linear") for column in range(values.shape[1])]
    )
    spread = detrended.std(axis=0)
    magnitude = np.maximum(np.abs(values).max(axis=0), 1.0)
    informative = spread > _NEGLIGIBLE_SPREAD * magnitude
    return np.where(informative,
                    (detrended - detrended.mean(axis=0)) / np.where(informative, spread, 1.0),
                    0.0)


PRODUCES = (
    # ``lag_frames``, ``lag_hours``, ``identity_a`` and friends are declared in
    # more than one module on purpose: the join keeps one copy, and a module
    # that writes a column says so. What is refused is two modules describing
    # one column differently, so these three repeat ``coupling``'s wording
    # exactly rather than inventing a second opinion about what a lag is.
    Column("lag_frames", "Lag", "frames", "motility"),
    Column("lag_hours", "Lag", "h", "motility"),
    Column("recurrence_rate", "Frame pairs in a similar state", "fraction", "morphology"),
    Column("recurrence_surrogate_mean", "Similar-state pairs in matched noise", "fraction", "reference"),
    Column("recurrence_surrogate_lo", "Matched-noise band, lower", "fraction", "reference"),
    Column("recurrence_surrogate_hi", "Matched-noise band, upper", "fraction", "reference"),
    Column("recurrence_pairs", "Frame pairs compared", "pairs", "morphology"),
    Column("determinism", "Recurrent points in diagonal runs", "fraction", "morphology"),
    Column("laminarity", "Recurrent points in vertical runs", "fraction", "morphology"),
    Column("determinism_surrogate", "Determinism in matched noise", "fraction", "reference"),
    Column("laminarity_surrogate", "Laminarity in matched noise", "fraction", "reference"),
    Column("recurrence_threshold_distance", "Distance counted as the same state", "sd", "morphology"),
    Column("recurrence_observations", "Frames the cell was measured in", "frames", "morphology"),
    Column("recurrence_metrics_used", "Measurements in the state vector", "count", "morphology"),
)

#: Neither folds. ``recurrence`` is keyed on a cell *and a lag*, which no roll-up
#: shares; ``recurrence_quantification`` is per cell and could fold, but folding
#: it would put a matched-noise column beside a measured one in cell_summary
#: with nothing saying which is which. Kept as a file.
WRITES = (
    Output("recurrence", grain=("identity", "lag_frames")),
    Output("recurrence_quantification", grain=("identity",)),
)


@register_derived(
    name="recurrence",
    description="Whether each cell returns to states it has been in before",
    needs_columns=("identity", "frame_index", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("recurrence")}
    metrics = [metric for metric in params["metrics"] if metric in cell_frame.columns]
    if not metrics:
        return {}
    quantile = float(params["threshold_quantile"])
    if not 0 < quantile < 1:
        raise ValueError(
            f"recurrence threshold_quantile must be between 0 and 1, not {quantile!r}: "
            "it is the fraction of frame pairs counted as the same state, not a distance"
        )
    complete = cell_frame.dropna(subset=metrics)
    minutes = float(context.scale.minutes_per_frame)
    max_lag_frames = max(1, int(round(float(params["max_lag_hours"]) * 60 / minutes)))
    surrogates = max(1, int(params["surrogates"]))
    min_observations = int(params["min_observations"])
    seed = int(params["random_state"])

    rate_rows: list[dict] = []
    quantified_rows: list[dict] = []
    for identity, group in complete.groupby("identity", sort=True):
        group = group.sort_values("frame_index")
        if len(group) < min_observations:
            continue
        hours = group["hours"].to_numpy(float)
        values = state_matrix(hours, group[metrics].to_numpy(float))
        distances = cdist(values, values)
        upper = distances[np.triu_indices_from(distances, 1)]
        cutoff = float(np.quantile(upper, quantile))
        recurrence = distances <= cutoff
        np.fill_diagonal(recurrence, False)
        maximum = min(max_lag_frames, len(group) - 1)
        observed = np.asarray(
            [np.mean(np.diagonal(recurrence, offset=lag)) for lag in range(1, maximum + 1)]
        )

        # Seeded from the identity as well as the run, so a cell's noise band is
        # the same number whether it is measured alone or among eighty others.
        generator = np.random.default_rng([seed, int(identity)])
        null_rates = np.empty((surrogates, maximum), dtype=float)
        null_quantified = np.empty((surrogates, 2), dtype=float)
        for replicate in range(surrogates):
            surrogate = np.column_stack(
                [_surrogate(values[:, column], params["null_model"], generator)
                 for column in range(values.shape[1])]
            )
            null_matrix = cdist(surrogate, surrogate) <= cutoff
            np.fill_diagonal(null_matrix, False)
            null_rates[replicate] = [np.mean(np.diagonal(null_matrix, offset=lag))
                                     for lag in range(1, maximum + 1)]
            null_quantified[replicate] = recurrence_runs(null_matrix)
        low, high = np.percentile(null_rates, [2.5, 97.5], axis=0)

        for position, lag in enumerate(range(1, maximum + 1)):
            rate_rows.append({
                "identity": int(identity),
                "lag_frames": lag,
                "lag_hours": lag * minutes / 60,
                "recurrence_rate": float(observed[position]),
                "recurrence_surrogate_mean": float(null_rates[:, position].mean()),
                "recurrence_surrogate_lo": float(low[position]),
                "recurrence_surrogate_hi": float(high[position]),
                "recurrence_pairs": int(len(group) - lag),
            })

        determinism, laminarity = recurrence_runs(recurrence)
        quantified_rows.append({
            "identity": int(identity),
            "recurrence_threshold_distance": cutoff,
            "determinism": determinism,
            "laminarity": laminarity,
            "determinism_surrogate": float(null_quantified[:, 0].mean()),
            "laminarity_surrogate": float(null_quantified[:, 1].mean()),
            "recurrence_observations": int(len(group)),
            "recurrence_metrics_used": int(len(metrics)),
        })

    output: dict[str, pd.DataFrame] = {}
    if rate_rows:
        output["recurrence"] = pd.DataFrame(rate_rows)
    if quantified_rows:
        output["recurrence_quantification"] = pd.DataFrame(quantified_rows)
    return output
