"""Nine size-and-movement subgroups for each classified cell-frame.

Size is read from the current frame. Movement is the centroid displacement
from the immediately preceding frame into the current one, so a first frame or
a frame after a gap has no movement and is not classified. Each measurement is
ranked across the eligible cell-frames and split into low, medium and high; the
two three-level axes cross to make nine explicit, ordered subgroups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register_derived


DEFAULTS = {
    "size_column": "area_px",
    "movement_column": "step_px_gapless",
    "category_quantiles": [1 / 3, 2 / 3],
    "min_frames_per_cell": 12,
}


LEVELS = ("low", "medium", "high")


def _percentile_positions(values: np.ndarray) -> np.ndarray:
    """Empirical positions from zero to one, with tied values kept together."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=float)
    ranks = pd.Series(values).rank(method="average").to_numpy(float) - 1.0
    return ranks / float(len(values) - 1)


def _quantile_boundaries(value: object) -> tuple[float, float]:
    """Validate the two cuts that define low, medium and high."""
    try:
        boundaries = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError("regimes.category_quantiles must contain two numbers") from None
    if len(boundaries) != 2 or not (0 < boundaries[0] < boundaries[1] < 1):
        raise ValueError(
            "regimes.category_quantiles must be two increasing fractions between 0 and 1"
        )
    return boundaries


#: What this module adds. It cannot declare all of what it writes:
#: ``regime_profiles`` carries one column per entry of ``feature_columns``, which
#: is configuration - change it and the table changes shape. Those columns are
#: other modules' measurements passed through unaltered, so they arrive already
#: labelled by whoever measured them, which is why nothing is lost by leaving
#: them out here.
#:
#: A regime number encodes the crossed levels: ``size * 3 + movement``. It is a
#: label rather than a continuous measurement, so the ``reference`` role keeps
#: it out of sequential colour maps that would imply equal numeric distances.
PRODUCES = (
    # regimes - one row per cell per frame
    Column("regime", "Size and movement subgroup", "", "reference"),
    Column("regime_label", "Size and movement subgroup label", "", "reference"),
    Column("regime_size_level", "Size category", "", "morphology"),
    Column("regime_movement_level", "Movement category", "", "motility"),
    Column("regime_size_percentile", "Size percentile", "fraction", "morphology"),
    Column("regime_movement_percentile", "Movement percentile", "fraction", "motility"),
    Column("regime_distance", "Distance to its subgroup centre", "quantile fraction", "reference"),
    Column("regime_second", "Next closest subgroup", "", "reference"),
    Column("regime_margin", "How much closer than the next subgroup", "fraction", "reference"),
    # regime_transitions - one row per pair of consecutive observed frames,
    # not one per change. The rows where the regime did not change are the
    # point: figure 24 normalises the transition matrix including its diagonal,
    # and figure 34 reads "most likely next state" off it, which is usually
    # "the same one". `regime_steps` would be the honest name; renaming it is
    # recorded as a known limit rather than done here.
    Column("from_frame_index", "Frame the change started from", "frame", "reference"),
    Column("to_frame_index", "Frame the change ended at", "frame", "reference"),
    Column("from_regime", "Regime before the change", "", "reference"),
    Column("to_regime", "Regime after the change", "", "reference"),
    Column("dwell_frames", "Frames spent in the regime before changing", "frames", "stable"),
    Column("dwell_id", "Which stay in a regime this was", "identifier", "reference"),
)


#: ``regimes`` is at cell-frame grain and its four columns are a description
#: of each cell-frame rather than a table in their own right, so they are
#: folded into ``cell_frame``. This module is derived, so the fold happens
#: after the join rather than as part of it - see ``run._fold_derived``.
WRITES = (
    Output("regimes",            grain=("identity", "frame_index"), fold=True),
    Output("regime_transitions", grain=("identity", "from_frame_index")),
    Output("regime_profiles",    grain=("regime",)),
)


@register_derived(
    name="regimes",
    description="Nine per-frame subgroups from low, medium and high size crossed with movement",
    needs_columns=("identity", "frame_index", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("regimes")}
    legacy = sorted(
        {"feature_columns", "n_regimes", "standardise", "random_state"}
        & set(context.module_params("regimes"))
    )
    if legacy:
        raise ValueError(
            "regimes no longer accepts " + ", ".join(legacy)
            + "; use size_column, movement_column and category_quantiles"
        )
    requested = [str(params["size_column"]), str(params["movement_column"])]
    if requested[0] == requested[1]:
        raise ValueError("regimes size_column and movement_column must be different")
    boundaries = _quantile_boundaries(params["category_quantiles"])
    missing = [column for column in requested if column not in cell_frame.columns]
    if missing:
        raise ValueError(
            "regimes needs feature column(s) " + ", ".join(missing)
            + "; enable the modules that write them or change feature_columns"
        )

    counts = cell_frame.groupby("identity")["frame_index"].count()
    eligible = counts[counts >= int(params["min_frames_per_cell"])].index
    keys = [column for column in ("stem", "condition", "subject") if column in cell_frame]
    working = cell_frame[
        cell_frame["identity"].isin(eligible)
    ][keys + ["identity", "frame_index", "hours", *requested]].dropna(subset=requested).copy()
    if working.empty:
        empty = pd.DataFrame(columns=keys + [
            "identity", "frame_index", "hours", "regime", "regime_label",
            "regime_size_level", "regime_movement_level",
            "regime_size_percentile", "regime_movement_percentile",
            "regime_distance", "regime_second", "regime_margin",
        ])
        return {
            "regimes": empty,
            "regime_transitions": pd.DataFrame(),
            "regime_profiles": pd.DataFrame(columns=[
                "regime", "regime_label", "regime_size_level",
                "regime_movement_level", "regime_observations", *requested,
            ]),
        }

    raw = working[requested].to_numpy(float)
    percentiles = np.column_stack([
        _percentile_positions(raw[:, index]) for index in range(raw.shape[1])
    ])
    level_codes = np.column_stack([
        np.digitize(percentiles[:, index], boundaries).astype(int)
        for index in range(percentiles.shape[1])
    ])
    labels = level_codes[:, 0] * 3 + level_codes[:, 1]

    category_centres = np.asarray([
        boundaries[0] / 2,
        (boundaries[0] + boundaries[1]) / 2,
        (boundaries[1] + 1) / 2,
    ])
    centres = np.asarray([
        (category_centres[size], category_centres[movement])
        for size in range(3) for movement in range(3)
    ])
    distances = np.linalg.norm(percentiles[:, None, :] - centres[None, :, :], axis=2)
    ranked = np.argsort(distances, axis=1)
    own = distances[np.arange(len(distances)), labels]
    second = np.asarray([
        next(subgroup for subgroup in row if subgroup != labels[index])
        for index, row in enumerate(ranked)
    ], dtype=int)
    runner_up = distances[np.arange(len(distances)), second]

    regimes = working[keys + ["identity", "frame_index", "hours"]].copy()
    regimes["regime"] = labels.astype(int)
    regimes["regime_size_level"] = [LEVELS[value] for value in level_codes[:, 0]]
    regimes["regime_movement_level"] = [LEVELS[value] for value in level_codes[:, 1]]
    regimes["regime_label"] = [
        f"{LEVELS[size].title()} size, {LEVELS[movement]} movement"
        for size, movement in level_codes
    ]
    regimes["regime_size_percentile"] = percentiles[:, 0]
    regimes["regime_movement_percentile"] = percentiles[:, 1]
    regimes["regime_distance"] = own
    regimes["regime_second"] = second.astype(int)
    regimes["regime_margin"] = (runner_up - own) / np.maximum(runner_up, 1e-12)
    regimes = regimes.sort_values(["identity", "frame_index"]).reset_index(drop=True)

    transition_rows: list[dict] = []
    for identity, group in regimes.groupby("identity", sort=True):
        group = group.sort_values("frame_index").reset_index(drop=True)
        frames = group["frame_index"].to_numpy(int)
        states = group["regime"].to_numpy(int)
        # One run id and its complete duration per observed state assignment.
        run_id = np.zeros(len(group), dtype=int)
        for index in range(1, len(group)):
            run_id[index] = run_id[index - 1] + int(
                frames[index] != frames[index - 1] + 1 or states[index] != states[index - 1]
            )
        run_lengths = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy(int)
        for index in range(len(group) - 1):
            if frames[index + 1] != frames[index] + 1:
                continue
            row = {
                "identity": int(identity),
                "from_frame_index": int(frames[index]),
                "to_frame_index": int(frames[index + 1]),
                "hours": float(group.loc[index + 1, "hours"]),
                "from_regime": int(states[index]),
                "to_regime": int(states[index + 1]),
                "dwell_frames": int(run_lengths[index]),
                "dwell_id": f"{int(identity)}:{int(run_id[index])}",
            }
            for column in keys:
                row[column] = group.loc[index, column]
            transition_rows.append(row)

    profile_rows = []
    for size in range(3):
        for movement in range(3):
            regime = size * 3 + movement
            members = raw[labels == regime]
            profile_rows.append({
                "regime": regime,
                "regime_label": f"{LEVELS[size].title()} size, {LEVELS[movement]} movement",
                "regime_size_level": LEVELS[size],
                "regime_movement_level": LEVELS[movement],
                "regime_observations": int(len(members)),
                requested[0]: (float(np.median(members[:, 0])) if len(members)
                               else float(np.quantile(raw[:, 0], category_centres[size]))),
                requested[1]: (float(np.median(members[:, 1])) if len(members)
                               else float(np.quantile(raw[:, 1], category_centres[movement]))),
            })
    profiles = pd.DataFrame(profile_rows)
    transitions = pd.DataFrame(transition_rows)
    return {
        "regimes": regimes,
        "regime_transitions": transitions,
        "regime_profiles": profiles,
    }

