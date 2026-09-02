"""Behavioural regimes from the joined per-cell, per-frame measurements.

This is a derived measurement: it combines shape, turnover, movement and
reporter texture rather than reading another image stack.  Cluster numbers are
made stable by sorting the fitted centroids by area after fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register_derived


DEFAULTS = {
    "feature_columns": [
        "area_px", "circularity", "solidity", "ramification_index",
        "aspect_ratio", "skeleton_branches", "turnover_index",
        "step_px_gapless", "punctateness",
    ],
    "n_regimes": 4,
    "standardise": True,
    "random_state": 20260825,
    "min_frames_per_cell": 12,
}


def _kmeans(values: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means with k-means++ initialisation."""
    values = np.asarray(values, dtype=float)
    if len(values) < k:
        raise ValueError(f"n_regimes={k} needs at least {k} complete cell-frames")
    rng = np.random.default_rng(seed)
    centres = [values[int(rng.integers(len(values)))]]
    for _ in range(1, k):
        distance2 = np.min(
            np.stack([np.sum((values - centre) ** 2, axis=1) for centre in centres]),
            axis=0,
        )
        total = float(distance2.sum())
        pick = int(rng.choice(len(values), p=distance2 / total)) if total else len(centres)
        centres.append(values[pick].copy())
    centres_array = np.asarray(centres, dtype=float)

    labels = np.zeros(len(values), dtype=int)
    for _ in range(200):
        distances = np.linalg.norm(values[:, None, :] - centres_array[None, :, :], axis=2)
        updated_labels = np.argmin(distances, axis=1)
        updated = centres_array.copy()
        for cluster in range(k):
            members = values[updated_labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
            else:
                # Re-seed an empty cluster with the least well represented row.
                farthest = int(np.argmax(np.min(distances, axis=1)))
                updated[cluster] = values[farthest]
                updated_labels[farthest] = cluster
        if np.array_equal(labels, updated_labels) and np.allclose(centres_array, updated):
            labels, centres_array = updated_labels, updated
            break
        labels, centres_array = updated_labels, updated
    return labels, centres_array


#: What this module adds. It cannot declare all of what it writes:
#: ``regime_profiles`` carries one column per entry of ``feature_columns``, which
#: is configuration - change it and the table changes shape. Those columns are
#: other modules' measurements passed through unaltered, so they arrive already
#: labelled by whoever measured them, which is why nothing is lost by leaving
#: them out here.
#:
#: A regime number is a name, not a quantity: regime 3 is not more of anything
#: than regime 1, and the ``reference`` role keeps it out of the sequential
#: colour maps that would imply otherwise.
PRODUCES = (
    # regimes - one row per cell per frame
    Column("regime", "Behavioural regime", "", "reference"),
    Column("regime_distance", "Distance to its own regime centre", "s.d.", "reference"),
    Column("regime_second", "Next closest regime", "", "reference"),
    Column("regime_margin", "How much closer than the next regime", "fraction", "reference"),
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
    description="Per-frame behavioural state from a standardised morphology and surveillance vector",
    needs_columns=("identity", "frame_index", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("regimes")}
    requested = list(params["feature_columns"])
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
            "identity", "frame_index", "hours", "regime", "regime_distance",
            "regime_second", "regime_margin",
        ])
        return {
            "regimes": empty,
            "regime_transitions": pd.DataFrame(),
            "regime_profiles": pd.DataFrame(columns=["regime", *requested]),
        }

    raw = working[requested].to_numpy(float)
    means = np.nanmean(raw, axis=0)
    scales = np.nanstd(raw, axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    fitted = (raw - means) / scales if bool(params["standardise"]) else raw.copy()
    labels, centres = _kmeans(fitted, int(params["n_regimes"]), int(params["random_state"]))

    original_centres = centres * scales + means if bool(params["standardise"]) else centres.copy()
    area_index = requested.index("area_px") if "area_px" in requested else 0
    stable_order = np.argsort(original_centres[:, area_index], kind="stable")
    remap = np.empty(len(stable_order), dtype=int)
    remap[stable_order] = np.arange(len(stable_order))
    labels = remap[labels]
    original_centres = original_centres[stable_order]
    centres = centres[stable_order]

    distances = np.linalg.norm(fitted[:, None, :] - centres[None, :, :], axis=2)
    ranked = np.argsort(distances, axis=1)
    own = distances[np.arange(len(distances)), labels]
    second = ranked[:, 0].copy()
    for index in range(len(second)):
        second[index] = next(cluster for cluster in ranked[index] if cluster != labels[index])
    runner_up = distances[np.arange(len(distances)), second]

    regimes = working[keys + ["identity", "frame_index", "hours"]].copy()
    regimes["regime"] = labels.astype(int)
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

    profiles = pd.DataFrame(original_centres, columns=requested)
    profiles.insert(0, "regime", np.arange(len(profiles), dtype=int))
    transitions = pd.DataFrame(transition_rows)
    return {
        "regimes": regimes,
        "regime_transitions": transitions,
        "regime_profiles": profiles,
    }

