"""Join identities created independently on opposite sides of the midpoint."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from tracking import frame_observations, motion_support


DEFAULTS = {
    "minimum_identity_frames": 3,
    "minimum_identity_area_px": 30,
    "position_scale_px": 20.0,
    "maximum_distance_px": 120.0,
    "minimum_area_ratio": 0.25,
    "maximum_area_ratio": 4.0,
    "maximum_mean_log2_change": 2.0,
    "position_weight": 1.0,
    "area_weight": 0.8,
    "mean_weight": 0.3,
    "shape_weight": 0.5,
    "delayed_motion_weight": 1.5,
    "territory_weight": 0.5,
    "gap_weight": 0.005,
    "maximum_cost": 2.5,
    "minimum_assignment_margin": 0.0,
    "minimum_source_loss_coverage": 0.0,
    "minimum_destination_gain_coverage": 0.0,
    "territory_dilation_px": 2,
}


def _identity_frames(labels: np.ndarray) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for t, frame in enumerate(labels):
        for identity in set(map(int, np.unique(frame))) - {0}:
            result[identity].append(t)
    return dict(result)


def _shape(mask: np.ndarray) -> np.ndarray:
    """Small translation-independent description of the soma outline."""
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return np.ones(3, np.float64)
    height = int(yy.max() - yy.min() + 1)
    width = int(xx.max() - xx.min() + 1)
    fill = len(yy) / max(height * width, 1)
    centred = np.column_stack((yy - yy.mean(), xx - xx.mean()))
    covariance = centred.T @ centred / max(len(yy), 1)
    eigen = np.sort(np.linalg.eigvalsh(covariance))[::-1] + 1e-3
    aspect = float(np.sqrt(eigen[0] / eigen[1]))
    edge = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))
    boundary = float(edge.sum() / max(np.sqrt(len(yy)), 1.0))
    return np.array([fill, aspect, boundary], np.float64)


def _endpoint(cache: list[dict[int, dict]], identity: int, t: int,
              frames: list[int], reference_frames: int = 5) -> dict:
    observation = cache[t][identity]
    nearby = sorted(frames, key=lambda frame: abs(frame - t))[:reference_frames]
    rows = [cache[frame][identity] for frame in nearby]
    return {
        "position": observation["position"],
        "mask": observation["mask"],
        "area": float(np.median([row["area"] for row in rows])),
        "mean": float(np.median([row["mean"] for row in rows])),
        "shape": _shape(observation["mask"]),
    }


def _territory_support(labels: np.ndarray, old: dict, new: dict,
                       old_t: int, new_t: int, dilation_px: int) -> tuple[float, float]:
    absorbed = 0.0
    if old_t + 1 < len(labels):
        occupied = ndi.binary_dilation(
            labels[old_t + 1] > 0, iterations=dilation_px)
        absorbed = float(np.mean(occupied[old["mask"]]))
    released = 0.0
    if new_t > 0:
        occupied = ndi.binary_dilation(
            labels[new_t - 1] > 0, iterations=dilation_px)
        released = float(np.mean(occupied[new["mask"]]))
    return absorbed, released


def _delayed_motion_support(raw: np.ndarray, old: dict, new: dict,
                            old_t: int, new_t: int) -> dict:
    """Measure a delayed blue-to-red handoff only in the two bookends' local box."""
    union = old["mask"] | new["mask"]
    yy, xx = np.nonzero(union)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    delayed = np.log2(
        (raw[new_t, y0:y1, x0:x1].astype(np.float32) + 1.0)
        / (raw[old_t, y0:y1, x0:x1].astype(np.float32) + 1.0))
    return motion_support(old["mask"][y0:y1, x0:x1],
                          new["mask"][y0:y1, x0:x1], delayed)


def midpoint_bookend_candidates(
        labels: np.ndarray, raw: np.ndarray, anchor_t: int,
        params: dict | None = None) -> pd.DataFrame:
    """Score before-only and after-only names as two sightings of one cell."""
    settings = dict(DEFAULTS)
    settings.update(params or {})
    frames = _identity_frames(labels)
    cache = [frame_observations(labels[t], raw[t]) for t in range(len(labels))]
    pre = {identity: ts for identity, ts in frames.items()
           if max(ts) < anchor_t}
    post = {identity: ts for identity, ts in frames.items()
            if min(ts) > anchor_t}
    rows: list[dict] = []
    pre_states = {
        identity: _endpoint(cache, identity, max(ts), ts)
        for identity, ts in pre.items()
        if len(ts) >= int(settings["minimum_identity_frames"])}
    post_states = {
        identity: _endpoint(cache, identity, min(ts), ts)
        for identity, ts in post.items()
        if len(ts) >= int(settings["minimum_identity_frames"])}
    for old_identity, old in sorted(pre_states.items()):
        old_t = max(pre[old_identity])
        if old["area"] < float(settings["minimum_identity_area_px"]):
            continue
        for new_identity, new in sorted(post_states.items()):
            new_t = min(post[new_identity])
            if new["area"] < float(settings["minimum_identity_area_px"]):
                continue
            gap = new_t - old_t
            distance = float(np.linalg.norm(old["position"] - new["position"]))
            maximum_distance = min(
                float(settings["maximum_distance_px"]),
                float(settings["position_scale_px"]) * np.sqrt(max(gap, 1)))
            area_ratio = float(new["area"] / max(old["area"], 1.0))
            area_change = abs(float(np.log2(max(area_ratio, 1e-12))))
            mean_change = abs(float(np.log2(
                (new["mean"] + 1.0) / (old["mean"] + 1.0))))
            shape_change = float(np.mean(np.abs(np.log2(
                (new["shape"] + 1e-3) / (old["shape"] + 1e-3)))))
            motion = _delayed_motion_support(raw, old, new, old_t, new_t)
            absorbed, released = _territory_support(
                labels, old, new, old_t, new_t,
                int(settings["territory_dilation_px"]))
            allowed = bool(
                distance <= maximum_distance
                and float(settings["minimum_area_ratio"]) <= area_ratio
                <= float(settings["maximum_area_ratio"])
                and mean_change <= float(settings["maximum_mean_log2_change"])
                and float(motion["source_loss_coverage"])
                >= float(settings["minimum_source_loss_coverage"])
                and float(motion["destination_gain_coverage"])
                >= float(settings["minimum_destination_gain_coverage"]))
            cost = (
                float(settings["position_weight"]) * distance
                / max(maximum_distance, 1e-6)
                + float(settings["area_weight"]) * area_change
                + float(settings["mean_weight"]) * mean_change
                + float(settings["shape_weight"]) * shape_change
                - float(settings["delayed_motion_weight"])
                * float(motion["motion_support"])
                - float(settings["territory_weight"])
                * (absorbed + released) / 2.0
                + float(settings["gap_weight"]) * gap)
            rows.append({
                "pre_identity": int(old_identity),
                "post_identity": int(new_identity),
                "pre_end_t": int(old_t),
                "pre_end_imagej_frame": int(old_t + 1),
                "post_start_t": int(new_t),
                "post_start_imagej_frame": int(new_t + 1),
                "gap_steps": int(gap),
                "pre_y": float(old["position"][0]),
                "pre_x": float(old["position"][1]),
                "post_y": float(new["position"][0]),
                "post_x": float(new["position"][1]),
                "distance_px": distance,
                "maximum_distance_px": maximum_distance,
                "area_ratio": area_ratio,
                "area_log2_change": area_change,
                "mean_log2_change": mean_change,
                "shape_log2_change": shape_change,
                "delayed_motion_support": float(motion["motion_support"]),
                "source_loss_coverage": float(
                    motion["source_loss_coverage"]),
                "destination_gain_coverage": float(
                    motion["destination_gain_coverage"]),
                "absorbed_territory_fraction": absorbed,
                "released_territory_fraction": released,
                "allowed": allowed,
                "cost": float(cost),
            })
    return pd.DataFrame(rows)


def reconcile_midpoint_bookends(
        labels: np.ndarray, raw: np.ndarray, anchor_t: int,
        params: dict | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Merge globally matched post-midpoint names into their pre-midpoint identities."""
    settings = dict(DEFAULTS)
    settings.update(params or {})
    identity_frames = _identity_frames(labels)
    protected_ids = {int(identity) for identity in settings.get(
        "protected_identity_ids", [])}
    candidates = midpoint_bookend_candidates(labels, raw, anchor_t, settings)
    if candidates.empty:
        return labels.copy(), pd.DataFrame(), candidates
    pre_ids = sorted(set(candidates.pre_identity.astype(int)))
    post_ids = sorted(set(candidates.post_identity.astype(int)))
    pre_index = {identity: index for index, identity in enumerate(pre_ids)}
    post_index = {identity: index for index, identity in enumerate(post_ids)}
    matrix = np.full((len(pre_ids), len(post_ids)), 1e9, float)
    for row in candidates.itertuples():
        if bool(row.allowed):
            matrix[pre_index[int(row.pre_identity)],
                   post_index[int(row.post_identity)]] = float(row.cost)
    assigned_pre, assigned_post = linear_sum_assignment(matrix)
    decisions: list[dict] = []
    fixed = labels.copy()
    for pre_row, post_col in zip(assigned_pre, assigned_post):
        cost = float(matrix[pre_row, post_col])
        if cost >= 1e9 or cost > float(settings["maximum_cost"]):
            continue
        row_values = np.delete(matrix[pre_row], post_col)
        col_values = np.delete(matrix[:, post_col], pre_row)
        alternatives = np.concatenate((row_values[row_values < 1e9],
                                       col_values[col_values < 1e9]))
        margin = (float(np.min(alternatives) - cost)
                  if len(alternatives) else float("inf"))
        if margin < float(settings["minimum_assignment_margin"]):
            continue
        pre_identity = pre_ids[pre_row]
        post_identity = post_ids[post_col]
        evidence = candidates[
            (candidates.pre_identity == pre_identity)
            & (candidates.post_identity == post_identity)].iloc[0].to_dict()
        if post_identity in protected_ids and pre_identity not in protected_ids:
            persistent_identity = post_identity
            replaced_identity = pre_identity
            canonical_side = "post_established"
        elif pre_identity in protected_ids:
            persistent_identity = pre_identity
            replaced_identity = post_identity
            canonical_side = "pre_established"
        elif (settings.get("unprotected_canonical_policy") == "longer_fragment"
              and len(identity_frames.get(post_identity, []))
              > len(identity_frames.get(pre_identity, []))):
            persistent_identity = post_identity
            replaced_identity = pre_identity
            canonical_side = "post_longer_fragment"
        else:
            persistent_identity = pre_identity
            replaced_identity = post_identity
            canonical_side = "pre_default"
        renamed = int(np.count_nonzero(fixed == replaced_identity))
        fixed[fixed == replaced_identity] = persistent_identity
        decisions.append({
            "decision_id": f"MB{len(decisions) + 1:04d}",
            **evidence,
            "persistent_identity": int(persistent_identity),
            "replaced_identity": int(replaced_identity),
            "canonical_side": canonical_side,
            "assignment_margin": margin,
            "renamed_pixels": renamed,
        })
    if not np.array_equal(fixed > 0, labels > 0):
        raise AssertionError("midpoint reconciliation changed foreground support")
    return fixed, pd.DataFrame(decisions), candidates
