"""The shape of the walk: which way each step pointed, and for how long.

``motility`` measures how far a cell went. It computes the step vector and then
keeps only its length, so the direction is thrown away. Two cells can have
identical ``straightness`` and identical ``msd_alpha`` and have walked
completely differently - one drifting steadily, one alternating long runs with
sharp reversals. This is the half of the step that was discarded.

Everyday version: ``motility`` is the odometer, this is the compass.

Three things make a turn meaningless, and each is handled rather than hidden:

* **A step across a gap.** A cell missing for twenty frames did not take one
  long step. Only consecutive-frame steps get a direction, which is the same
  rule ``step_px_gapless`` already applies to length.
* **A step below the noise floor.** The step is a difference of two centres and
  the turn is a difference of two steps, so a 0.3 px wobble becomes a 180
  degree "reversal". Below ``min_step_px`` the direction is NaN. On the pinned
  movie 32% of steps are under half a pixel and 60% under one, so this
  parameter, not the arithmetic, decides what the column means.
* **Averaging an angle.** The mean of 359 and 1 degrees is 0, not 180. Every
  summary here goes through a resultant vector or through the cosine, never
  through ``np.mean`` of an angle.

The soma centre is used rather than the outline centroid: the two step series
correlate at r = 0.984 on the pinned movie, so emitting both would double the
columns to say the same thing twice, and ``motility``'s own docstring argues
the soma is the better estimate of whether the body relocated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register_derived


DEFAULTS = {
    # "soma" is the intensity-weighted centre; "centroid" is the middle of the
    # whole outline, which one extending process drags several pixels.
    "centre": "soma",
    # Steps shorter than this have no usable direction. It also defines
    # "stationary", because a step the direction cannot be read from is a step
    # the cell did not really take.
    "min_step_px": 0.5,
    # What ends a run. Fixed rather than data-driven on purpose: a threshold
    # fitted per movie makes two runs incomparable for a reason the reader
    # cannot see on the axis.
    "run_turn_threshold_deg": 90.0,
    "max_autocorrelation_lag_frames": 24,
    "min_turns_for_summary": 5,
}


def _wrap(angles: np.ndarray) -> np.ndarray:
    """Fold a difference of two angles back into (-pi, pi]."""
    return np.arctan2(np.sin(angles), np.cos(angles))


def _resultant(angles: np.ndarray) -> float:
    """Length of the mean unit vector: 1 is one direction, 0 is every direction."""
    values = np.asarray(angles, dtype=float)
    usable = values[np.isfinite(values)]
    if usable.size == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * usable))))


def _dispersion(angles: np.ndarray) -> float:
    """Circular standard deviation of a set of turns, in radians.

    The plan asked for an inter-quartile range and it cannot be one. A cell
    that reverses on almost every step has turns packed against +pi and -pi,
    which are the same direction; the wrapped IQR calls that the widest
    possible spread when it is one of the narrowest. ``sqrt(-2 ln R)``, where
    ``R`` is the length of the mean unit vector, gives 0 for turns that all
    agree and grows without bound as they scatter, whichever direction they
    agree on.
    """
    resultant = _resultant(angles)
    if not np.isfinite(resultant) or resultant <= 0:
        # Turns spread perfectly evenly round the circle have no finite spread.
        # A blank says that; an infinity would be plotted.
        return float("nan")
    return float(np.sqrt(-2.0 * np.log(min(resultant, 1.0))))


def _halflife(frames: np.ndarray, angles: np.ndarray, max_lag: int) -> float:
    """Frames until the cell has forgotten half of which way it was going.

    ``C(lag)`` is the mean cosine between a step and the step ``lag`` frames
    later. It starts at 1 by definition; the half-life is where it first falls
    through 0.5, interpolated between the two lags that straddle it rather than
    rounded to whichever lag happened to be below. NaN when it never falls that
    far within ``max_lag``, which is an honest "longer than we looked".
    """
    lookup = {int(frame): angle for frame, angle in zip(frames, angles)
              if np.isfinite(angle)}
    if len(lookup) < 3:
        return float("nan")
    previous_lag, previous_value = 0, 1.0
    for lag in range(1, max_lag + 1):
        cosines = [np.cos(lookup[frame + lag] - lookup[frame])
                   for frame in lookup if frame + lag in lookup]
        if len(cosines) < 3:
            continue
        value = float(np.mean(cosines))
        if value < 0.5:
            if value == previous_value:
                return float(lag)
            fraction = (previous_value - 0.5) / (previous_value - value)
            return float(previous_lag + fraction * (lag - previous_lag))
        previous_lag, previous_value = lag, value
    return float("nan")


def _runs(turn: np.ndarray, threshold: float) -> list[int]:
    """Lengths, in steps, of the unbroken stretches between large turns.

    A stretch of ``n`` small turns is ``n + 1`` steps travelling in roughly one
    direction, which is what gets counted. A run ends where the turn exceeds
    ``threshold``, and is abandoned entirely where the turn is not measurable -
    a gap, or a step too short to have a direction - because a run that spans an
    absence is not one run.
    """
    lengths: list[int] = []
    turns, started = 0, False
    for value in turn:
        if not np.isfinite(value):
            if started:
                lengths.append(turns + 1)
            turns, started = 0, False
        elif abs(value) > threshold:
            if started:
                lengths.append(turns + 1)
            turns, started = 0, True
        else:
            turns += 1
            started = True
    if started:
        lengths.append(turns + 1)
    return lengths


def _asymmetry(positions: np.ndarray) -> float:
    """Eigenvalue ratio of the position covariance: exploring one axis or two."""
    if positions.shape[0] < 3:
        return float("nan")
    covariance = np.cov(positions.T)
    if not np.all(np.isfinite(covariance)):
        return float("nan")
    values = np.linalg.eigvalsh(covariance)
    smallest, largest = float(values[0]), float(values[-1])
    if smallest <= 0:
        return float("nan")
    return largest / smallest


#: The turn columns take the ``motility`` colour because they describe the same
#: event as the step does; the run and persistence columns take ``stable``
#: because what they measure is how long a cell keeps doing the same thing.
PRODUCES = (
    # walk - one row per cell per frame
    Column("step_angle_rad", "Direction of this step", "rad", "motility"),
    Column("turn_angle_rad", "Turn from the previous step", "rad", "motility"),
    Column("turn_cos", "Turn from the previous step, as a cosine", "-1 to 1", "motility"),
    # walk_tracks - one row per cell
    Column("mean_turn_cos", "Directional persistence", "-1 to 1", "stable"),
    Column("turn_angle_dispersion", "Spread of turns", "rad", "motility"),
    Column("turns_measured", "Turns the summaries were built from", "count", "reference"),
    Column("directional_autocorrelation_halflife",
           "Frames before the direction is forgotten", "frames", "stable"),
    Column("run_length_frames_mean", "Unbroken run before a large turn, mean", "frames", "stable"),
    Column("run_length_frames_median",
           "Unbroken run before a large turn, median", "frames", "stable"),
    Column("runs_measured", "Runs the run lengths were built from", "count", "reference"),
    Column("stationary_fraction", "Steps too short to be a move", "fraction", "stable"),
    Column("net_displacement_angle_rad", "Overall direction of travel", "rad", "motility"),
    Column("path_asymmetry", "Exploration along one axis against the other", "ratio", "motility"),
    # walk_frame - one row per frame
    Column("mean_step_angle_resultant", "Agreement between the cells' directions",
           "0-1", "reference"),
    Column("step_angle_cells", "Cells with a usable direction this frame", "count", "reference"),
)


#: ``walk_frame`` folds into ``frame_summary``. It is one column and a count,
#: and it is a check on the data rather than a measurement of the cells: if
#: every cell in a frame points the same way the field moved, not the cells.
#: House rule 1 forbids touching the tracking, not describing it.
WRITES = (
    Output("walk",        grain=("identity", "frame_index"), fold=True),
    Output("walk_tracks", grain=("identity",),               fold=True),
    Output("walk_frame",  grain=("frame_index",),            fold=True),
)


@register_derived(
    name="walk",
    description="Step direction, turning angles, directional persistence and run lengths",
    needs_columns=("identity", "frame_index", "soma_y", "soma_x"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("walk")}
    centre = str(params["centre"])
    columns = ("soma_y", "soma_x") if centre == "soma" else ("centroid_y", "centroid_x")
    missing = [column for column in columns if column not in cell_frame.columns]
    if missing:
        raise ValueError(
            f"walk needs {', '.join(missing)} for centre={centre!r}; "
            "enable motility, or set centre to the other one"
        )
    floor = float(params["min_step_px"])
    threshold = float(np.radians(float(params["run_turn_threshold_deg"])))
    max_lag = int(params["max_autocorrelation_lag_frames"])
    min_turns = int(params["min_turns_for_summary"])

    step_tables: list[pd.DataFrame] = []
    track_rows: list[dict] = []

    for identity, group in cell_frame.groupby("identity", sort=True):
        group = group.sort_values("frame_index")
        frames = group["frame_index"].to_numpy(int)
        positions = group[list(columns)].to_numpy(float)
        count = len(frames)

        step_angle = np.full(count, np.nan)
        step_length = np.full(count, np.nan)
        if count > 1:
            deltas = np.diff(positions, axis=0)
            consecutive = np.diff(frames) == 1
            lengths = np.linalg.norm(deltas, axis=1)
            step_length[1:] = np.where(consecutive, lengths, np.nan)
            usable = consecutive & (lengths >= floor)
            angles = np.arctan2(deltas[:, 0], deltas[:, 1])
            step_angle[1:] = np.where(usable, angles, np.nan)

        turn_angle = np.full(count, np.nan)
        if count > 2:
            turn_angle[2:] = _wrap(step_angle[2:] - step_angle[1:-1])
        turn_cos = np.cos(turn_angle)

        step_tables.append(
            pd.DataFrame(
                {
                    "identity": identity,
                    "frame_index": frames,
                    "step_angle_rad": step_angle,
                    "turn_angle_rad": turn_angle,
                    "turn_cos": turn_cos,
                }
            )
        )

        measured = turn_angle[np.isfinite(turn_angle)]
        consecutive_steps = step_length[np.isfinite(step_length)]
        run_lengths = _runs(turn_angle, threshold)
        track: dict = {
            "identity": int(identity),
            "turns_measured": int(measured.size),
            "runs_measured": int(len(run_lengths)),
            "stationary_fraction": (
                float(np.mean(consecutive_steps < floor))
                if consecutive_steps.size else np.nan
            ),
            "net_displacement_angle_rad": (
                float(np.arctan2(positions[-1, 0] - positions[0, 0],
                                 positions[-1, 1] - positions[0, 1]))
                if count > 1 else np.nan
            ),
            "path_asymmetry": _asymmetry(positions),
        }
        if measured.size >= min_turns:
            track.update(
                {
                    "mean_turn_cos": float(np.mean(np.cos(measured))),
                    "turn_angle_dispersion": _dispersion(measured),
                    "directional_autocorrelation_halflife": _halflife(
                        frames, step_angle, max_lag
                    ),
                    "run_length_frames_mean": float(np.mean(run_lengths)) if run_lengths else np.nan,
                    "run_length_frames_median": (
                        float(np.median(run_lengths)) if run_lengths else np.nan
                    ),
                }
            )
        track_rows.append(track)

    walk = (
        pd.concat(step_tables, ignore_index=True)
        if step_tables else
        pd.DataFrame(columns=["identity", "frame_index", "step_angle_rad",
                              "turn_angle_rad", "turn_cos"])
    )

    per_frame = (
        walk.groupby("frame_index")["step_angle_rad"]
        .agg(mean_step_angle_resultant=_resultant,
             step_angle_cells=lambda values: int(np.isfinite(values).sum()))
        .reset_index()
        if not walk.empty else
        pd.DataFrame(columns=["frame_index", "mean_step_angle_resultant", "step_angle_cells"])
    )

    return {
        "walk": walk.sort_values(["identity", "frame_index"]).reset_index(drop=True),
        "walk_tracks": pd.DataFrame(track_rows),
        "walk_frame": per_frame,
    }
