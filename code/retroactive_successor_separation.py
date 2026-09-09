"""Backfill a movie-novel successor when the old owner obtains another seat.

Discovery is field-wide. Identities, owners, tracks, frames, coordinates,
events, regions, and review cases are measured outputs and cannot select a
proposal. Only owner values on existing foreground may change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import distinct_history_transition_candidates as separation
import owner_consensus
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
ALLOWED_PARAM_KEYS = {
    "mode", "targeting_mode", "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path", "output_stem",
    "minimum_predecessor_support_movie_fraction",
    "minimum_successor_support_movie_fraction",
    "minimum_predecessor_identity_movie_fraction",
    "minimum_resident_support_movie_fraction",
    "maximum_transition_gap_movie_fraction",
    "minimum_successor_owner_purity", "minimum_resident_owner_purity",
    "minimum_track_strong_fraction", "minimum_resident_strong_fraction",
    "maximum_motion_step_quantile", "minimum_resident_separation_sum_radii",
    "minimum_component_area_ratio", "maximum_component_area_ratio",
    "minimum_chain_link_support_movie_fraction",
    "minimum_chain_resident_separation_sum_radii",
    "maximum_chain_valley_ratio", "minimum_seed_distance_px",
    "watershed_sigma_px", "minimum_basin_pixels",
    "minimum_expected_area_ratio", "maximum_expected_area_ratio",
}
REQUIRED_PATH_KEYS = {
    "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path",
}
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "predecessor_owner", "successor_owner",
    "resident_track", "transition_frame", "predecessor_first_frame",
    "predecessor_last_frame", "predecessor_support", "successor_support",
    "resident_support", "successor_purity", "resident_purity",
    "track_strong_fraction", "resident_strong_fraction",
    "transition_gap_frames", "transition_step_radii",
    "field_step_radii", "resident_separation_sum_radii",
    "component_area_ratio", "chain_links", "chain_changed_pixels",
    "chain_changed_frames", "changed_pixels", "changed_frames",
    "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "track_id", "predecessor_owner",
    "successor_owner", "component", "pixels",
]


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def selector_counts(params: dict) -> dict[str, int]:
    unexpected = [str(key) for key in params
                  if str(key) not in ALLOWED_PARAM_KEYS]
    roles = {
        "identities": ("identity", "identities", "cell", "cells"),
        "owners": ("owner", "owners"),
        "tracks": ("track", "tracks"),
        "frames": ("frame", "frames", "interval"),
        "coordinates": ("coordinate", "coordinates", "point", "points"),
        "events": ("event", "events"),
        "regions": ("region", "regions", "roi"),
        "review_cases": ("case", "cases", "review"),
    }
    return {role: sum(any(token in key.lower() for token in tokens)
                      for key in unexpected)
            for role, tokens in roles.items()}


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("successor separation must be field-wide")
    supplied = sorted(str(key) for key in params
                      if str(key) not in ALLOWED_PARAM_KEYS)
    if supplied:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(supplied))
    missing = sorted(key for key in REQUIRED_PATH_KEYS
                     if params.get(key) in (None, ""))
    if missing:
        raise ValueError("missing evidence paths: " + ", ".join(missing))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _presence(labels: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for frame in range(len(labels)):
        for owner in map(int, np.unique(labels[frame])):
            if owner > 0:
                result.setdefault(owner, set()).add(frame)
    return result


def _normalised_step(left: object, right: object) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    scale = max(0.5 * (float(left.radius_px) + float(right.radius_px)), 1.0)
    return distance / scale


def _field_step(points: pd.DataFrame, quantile: float) -> float:
    steps = []
    for _, group in points.groupby("track_id", sort=False):
        rows = list(group.sort_values("frame").itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            if int(right.frame) == int(left.frame) + 1:
                steps.append(_normalised_step(left, right))
    return float(np.quantile(steps, quantile)) if steps else 0.0


def _owner_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.candidate_owner)
        if (runs and runs[-1]["owner"] == owner
                and int(row.frame) == runs[-1]["last"] + 1):
            runs[-1]["last"] = int(row.frame)
            runs[-1]["points"].append(row)
        else:
            runs.append({"owner": owner, "first": int(row.frame),
                         "last": int(row.frame), "points": [row]})
    return runs


def _component_at(frame: np.ndarray, owner: int,
                  point: object) -> tuple[np.ndarray, int]:
    components, _ = ndi.label(frame == int(owner), STRUCTURE)
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    component = int(components[y, x])
    if component <= 0:
        yy, xx = np.nonzero(components > 0)
        if not len(xx):
            return np.zeros(frame.shape, dtype=bool), 0
        distances = (xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > max(float(point.radius_px), 2.0) ** 2:
            return np.zeros(frame.shape, dtype=bool), 0
        component = int(components[yy[nearest], xx[nearest]])
    return components == component, component


def _profile(group: pd.DataFrame, owner: int, first_frame: int) -> dict:
    future = group[group.frame.astype(int).ge(first_frame)]
    positive = future[future.candidate_owner.astype(int).gt(0)]
    owned = positive[positive.candidate_owner.astype(int).eq(int(owner))]
    return {
        "support": int(len(owned)),
        "purity": float(len(owned) / max(len(positive), 1)),
        "strong_fraction": float(owned.strong.map(_truth).mean())
        if len(owned) else 0.0,
    }


def _expected_component_area(labels: np.ndarray, group: pd.DataFrame,
                             owner: int, first_frame: int) -> float:
    areas = []
    owned = group[
        group.frame.astype(int).ge(first_frame)
        & group.candidate_owner.astype(int).eq(int(owner))]
    for point in owned.itertuples(index=False):
        mask, component = _component_at(labels[int(point.frame)], owner, point)
        if component > 0:
            areas.append(int(mask.sum()))
    return float(np.median(areas)) if areas else 0.0


def _displaced_resident_link(
        labels: np.ndarray, raw: np.ndarray | None, visible: pd.DataFrame,
        by_track: dict[int, pd.DataFrame], resident_track: int,
        successor_owner: int, root_transition: int, minimum_chain: int,
        minimum_resident: int, resident_purity_min: float,
        resident_strong_min: float, maximum_gap: int,
        minimum_separation: float, maximum_valley: float, params: dict,
        ) -> tuple[list[dict], list[str]]:
    """Recover one atomically displaced seat behind a novel-successor root."""
    group = by_track[int(resident_track)]
    runs = _owner_runs(group)
    current_indices = [
        index for index, run in enumerate(runs)
        if int(run["owner"]) == int(successor_owner)
        and int(run["first"]) <= root_transition + 1
        and int(run["last"]) >= root_transition - 1]
    if len(current_indices) != 1:
        return [], ["resident_successor_run_not_unique"]
    current_index = current_indices[0]
    preceding_indices = [index for index in range(current_index - 1, -1, -1)
                         if int(runs[index]["owner"]) > 0]
    if not preceding_indices:
        return [], []
    before = runs[preceding_indices[0]]
    displaced_owner = int(before["owner"])
    if displaced_owner == int(successor_owner):
        return [], []
    transition = int(runs[current_index]["first"])
    gap = transition - int(before["last"]) - 1
    reasons: list[str] = []
    before_points = pd.DataFrame([point._asdict() for point in before["points"]])
    if gap < 0 or gap > maximum_gap:
        reasons.append("chain_transition_gap_exceeds_movie_fraction")
    if len(before_points) < minimum_chain:
        reasons.append("chain_predecessor_support_too_short")

    boundary_tracks = visible[
        visible.frame.astype(int).between(transition - 1, transition + 1)
        & visible.candidate_owner.astype(int).eq(displaced_owner)
        & visible.track_id.astype(int).ne(resident_track)
    ].track_id.astype(int).drop_duplicates().tolist()
    candidates = []
    for track in boundary_tracks:
        resident_group = by_track[int(track)]
        profile = _profile(resident_group, displaced_owner, transition - 1)
        boundary = resident_group[
            resident_group.frame.astype(int).between(
                transition - 1, transition + 1)
            & resident_group.candidate_owner.astype(int).eq(displaced_owner)]
        if (profile["support"] >= minimum_resident
                and profile["purity"] >= resident_purity_min
                and profile["strong_fraction"] >= resident_strong_min
                and len(boundary)):
            candidates.append((int(track), resident_group,
                               profile, boundary.iloc[0]))
    if len(candidates) != 1:
        reasons.append("chain_displaced_resident_not_unique")
        return [], reasons
    displaced_track, displaced_group, _, displaced_point = candidates[0]
    successor_point = runs[current_index]["points"][0]
    distance = float(np.hypot(
        float(successor_point.x) - float(displaced_point.x),
        float(successor_point.y) - float(displaced_point.y)))
    separation_radii = distance / max(
        float(successor_point.radius_px) + float(displaced_point.radius_px),
        1.0)
    if separation_radii < minimum_separation:
        reasons.append("chain_displaced_resident_not_spatially_distinct")
    expected_successor = _expected_component_area(
        labels, group, successor_owner, transition)
    expected_displaced = _expected_component_area(
        labels, displaced_group, displaced_owner, transition)
    shared_areas = []
    for point in before_points.itertuples(index=False):
        frame = int(point.frame)
        mask, component = _component_at(labels[frame], displaced_owner, point)
        exact_resident = displaced_group[
            displaced_group.frame.astype(int).eq(frame)
            & displaced_group.candidate_owner.astype(int).eq(displaced_owner)]
        if component > 0 and len(exact_resident) == 1:
            other = exact_resident.iloc[0]
            y = int(np.clip(round(float(other.y)), 0, labels.shape[1] - 1))
            x = int(np.clip(round(float(other.x)), 0, labels.shape[2] - 1))
            if mask[y, x]:
                shared_areas.append(int(mask.sum()))
    # A resident can contract immediately after separation. The robust
    # pre-separation remainder is a better expected area for the foreground
    # that must stay with it than that transient post-separation mask alone.
    if shared_areas and expected_successor > 0:
        expected_displaced = max(
            expected_displaced,
            float(np.median(shared_areas)) - expected_successor)
    if expected_successor <= 0 or expected_displaced <= 0:
        reasons.append("chain_expected_area_unavailable")

    items = []
    for point in before_points.itertuples(index=False):
        frame = int(point.frame)
        mask, component = _component_at(labels[frame], displaced_owner, point)
        exact_resident = displaced_group[
            displaced_group.frame.astype(int).eq(frame)
            & displaced_group.candidate_owner.astype(int).eq(displaced_owner)]
        if component <= 0 or len(exact_resident) != 1:
            reasons.append("chain_frame_evidence_incomplete")
            continue
        other = exact_resident.iloc[0]
        y = int(np.clip(round(float(other.y)), 0, labels.shape[1] - 1))
        x = int(np.clip(round(float(other.x)), 0, labels.shape[2] - 1))
        if mask[y, x]:
            if raw is None:
                reasons.append("chain_raw_stack_required_for_partition")
                continue
            valley = separation._valley(raw[frame], point, other)
            if valley > maximum_valley:
                reasons.append("chain_shared_mask_valley_too_high")
                continue
            partition = separation._partition(
                mask, raw[frame], point, other,
                expected_successor, expected_displaced, params)
            if partition is None:
                reasons.append("chain_connected_partition_unavailable")
                continue
            assign_mask = partition[0]
        else:
            assign_mask = mask
        items.append({
            "frame": frame, "mask": assign_mask, "component": component,
            "pixels": int(assign_mask.sum()), "track_id": resident_track,
            "predecessor_owner": displaced_owner,
            "successor_owner": int(successor_owner),
        })
    if len(items) != len(before_points):
        reasons.append("chain_component_coverage_incomplete")
    return items, sorted(set(reasons))


def discover_and_apply(labels: np.ndarray, points: pd.DataFrame,
                       raw: np.ndarray | None,
                       params: dict) -> tuple[np.ndarray, pd.DataFrame,
                                              pd.DataFrame]:
    frame_count = len(labels)
    attached = attach_owners(points, labels)
    visible = attached[attached.physically_visible.map(_truth)].copy()
    presence = _presence(labels)
    by_track = {int(track): group.sort_values("frame").copy()
                for track, group in visible.groupby("track_id", sort=False)}
    minimum_predecessor = math.ceil(frame_count * _fraction(
        params, "minimum_predecessor_support_movie_fraction", 0.10))
    minimum_successor = math.ceil(frame_count * _fraction(
        params, "minimum_successor_support_movie_fraction", 0.15))
    minimum_identity = _fraction(
        params, "minimum_predecessor_identity_movie_fraction", 0.50)
    minimum_resident = math.ceil(frame_count * _fraction(
        params, "minimum_resident_support_movie_fraction", 0.25))
    maximum_gap = math.ceil(frame_count * _fraction(
        params, "maximum_transition_gap_movie_fraction", 0.02))
    successor_purity_min = _fraction(
        params, "minimum_successor_owner_purity", 0.95)
    resident_purity_min = _fraction(
        params, "minimum_resident_owner_purity", 0.95)
    track_strong_min = _fraction(
        params, "minimum_track_strong_fraction", 0.90)
    resident_strong_min = _fraction(
        params, "minimum_resident_strong_fraction", 0.90)
    quantile = _fraction(params, "maximum_motion_step_quantile", 0.90)
    field_step = _field_step(visible, quantile)
    minimum_separation = float(params.get(
        "minimum_resident_separation_sum_radii", 4.0))
    minimum_area_ratio = float(params.get("minimum_component_area_ratio", 0.25))
    maximum_area_ratio = float(params.get("maximum_component_area_ratio", 4.0))
    minimum_chain = math.ceil(frame_count * _fraction(
        params, "minimum_chain_link_support_movie_fraction", 0.05))
    minimum_chain_separation = float(params.get(
        "minimum_chain_resident_separation_sum_radii", 1.50))
    maximum_chain_valley = float(params.get("maximum_chain_valley_ratio", 0.65))
    audits: list[dict] = []
    plans: list[dict] = []

    for track_id, group in by_track.items():
        runs = _owner_runs(group)
        positive_indices = [index for index, run in enumerate(runs)
                            if int(run["owner"]) > 0]
        for left_index, right_index in zip(positive_indices,
                                           positive_indices[1:]):
            before, after = runs[left_index], runs[right_index]
            predecessor = int(before["owner"])
            successor = int(after["owner"])
            if predecessor == successor:
                continue
            transition = int(after["first"])
            gap = transition - int(before["last"]) - 1
            last_foreign = max(
                [int(row.frame) for row in group.itertuples(index=False)
                 if int(row.frame) < transition
                 and int(row.candidate_owner) > 0
                 and int(row.candidate_owner) != predecessor],
                default=-1)
            predecessor_points = group[
                group.frame.astype(int).gt(last_foreign)
                & group.frame.astype(int).lt(transition)
                & group.candidate_owner.astype(int).eq(predecessor)]
            next_foreign = min(
                [int(row.frame) for row in group.itertuples(index=False)
                 if int(row.frame) > transition
                 and int(row.candidate_owner) > 0
                 and int(row.candidate_owner) != successor],
                default=frame_count)
            successor_points = group[
                group.frame.astype(int).ge(transition)
                & group.frame.astype(int).lt(next_foreign)
                & group.candidate_owner.astype(int).eq(successor)]
            successor_positive = group[
                group.frame.astype(int).ge(transition)
                & group.frame.astype(int).lt(next_foreign)
                & group.candidate_owner.astype(int).gt(0)]
            successor_purity = len(successor_points) / max(
                len(successor_positive), 1)
            combined = pd.concat([predecessor_points, successor_points])
            strong_fraction = float(combined.strong.map(_truth).mean()) \
                if len(combined) else 0.0
            step = _normalised_step(before["points"][-1], after["points"][0])
            reasons: list[str] = []
            if gap < 0 or gap > maximum_gap:
                reasons.append("transition_gap_exceeds_movie_fraction")
            if len(predecessor_points) < minimum_predecessor:
                reasons.append("predecessor_track_support_too_short")
            if len(successor_points) < minimum_successor:
                reasons.append("successor_track_support_too_short")
            if len(presence.get(predecessor, set())) / frame_count < minimum_identity:
                reasons.append("predecessor_identity_not_movie_durable")
            if min(presence.get(successor, {frame_count})) < transition:
                reasons.append("successor_not_movie_novel_at_transition")
            if successor_purity < successor_purity_min:
                reasons.append("successor_track_owner_impure")
            if strong_fraction < track_strong_min:
                reasons.append("transition_track_raw_support_weak")
            if step > field_step:
                reasons.append("transition_motion_is_field_outlier")

            resident_rows = []
            # A valid resident must be present at the transition boundary.
            # Restricting the profile calculation to those boundary tracks is
            # mathematically identical to scanning every movie track, but
            # avoids thousands of redundant DataFrame filters per proposal.
            boundary_tracks = visible[
                visible.frame.astype(int).between(
                    transition - 1, transition + 1)
                & visible.candidate_owner.astype(int).eq(predecessor)
                & visible.track_id.astype(int).ne(track_id)
            ].track_id.astype(int).drop_duplicates().tolist()
            for resident_track in boundary_tracks:
                resident_group = by_track[int(resident_track)]
                profile = _profile(resident_group, predecessor, transition - 1)
                boundary = resident_group[
                    resident_group.frame.astype(int).between(
                        transition - 1, transition + 1)
                    & resident_group.candidate_owner.astype(int).eq(predecessor)]
                if (profile["support"] >= minimum_resident
                        and profile["purity"] >= resident_purity_min
                        and profile["strong_fraction"] >= resident_strong_min
                        and len(boundary)):
                    resident_rows.append((resident_track, resident_group,
                                          profile, boundary.iloc[0]))
            if len(resident_rows) != 1:
                reasons.append("durable_predecessor_seat_not_unique")
                resident_track = 0
                resident_profile = {"support": 0, "purity": 0.0,
                                    "strong_fraction": 0.0}
                separation = 0.0
            else:
                resident_track, _, resident_profile, resident_point = resident_rows[0]
                subject_point = after["points"][0]
                distance = float(np.hypot(
                    float(subject_point.x) - float(resident_point.x),
                    float(subject_point.y) - float(resident_point.y)))
                separation = distance / max(
                    float(subject_point.radius_px) + float(resident_point.radius_px),
                    1.0)
                if separation < minimum_separation:
                    reasons.append("predecessor_seat_not_spatially_distinct")

            frame_items = []
            predecessor_areas = []
            for point in predecessor_points.itertuples(index=False):
                frame = int(point.frame)
                mask, component = _component_at(
                    labels[frame], predecessor, point)
                if component <= 0:
                    reasons.append("predecessor_component_missing")
                    continue
                competing = False
                local = visible[
                    visible.frame.astype(int).eq(frame)
                    & visible.track_id.astype(int).ne(track_id)
                    & visible.candidate_owner.astype(int).eq(predecessor)]
                for other in local.itertuples(index=False):
                    y = int(np.clip(round(float(other.y)), 0,
                                    labels.shape[1] - 1))
                    x = int(np.clip(round(float(other.x)), 0,
                                    labels.shape[2] - 1))
                    competing |= bool(mask[y, x])
                if competing:
                    reasons.append("component_contains_other_predecessor_track")
                pixels = int(mask.sum())
                predecessor_areas.append(pixels)
                frame_items.append({"frame": frame, "mask": mask,
                                    "component": component, "pixels": pixels,
                                    "track_id": track_id,
                                    "predecessor_owner": predecessor,
                                    "successor_owner": successor})
            successor_areas = []
            for point in successor_points.itertuples(index=False):
                mask, component = _component_at(
                    labels[int(point.frame)], successor, point)
                if component > 0:
                    successor_areas.append(int(mask.sum()))
            area_ratio = (float(np.median(predecessor_areas))
                          / max(float(np.median(successor_areas)), 1.0)) \
                if predecessor_areas and successor_areas else 0.0
            if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
                reasons.append("component_area_not_successor_compatible")
            if len(frame_items) != len(predecessor_points):
                reasons.append("predecessor_component_coverage_incomplete")
            chain_items: list[dict] = []
            chain_reasons: list[str] = []
            if not reasons and resident_track:
                chain_items, chain_reasons = _displaced_resident_link(
                    labels, raw, visible, by_track, int(resident_track),
                    predecessor, transition, minimum_chain,
                    minimum_resident, resident_purity_min,
                    resident_strong_min, maximum_gap,
                    minimum_chain_separation, maximum_chain_valley, params)
                reasons.extend(chain_reasons)
            all_items = frame_items + chain_items
            reasons = sorted(set(reasons))
            proposal_id = f"NS{len(audits) + 1:04d}"
            audit = {
                "proposal_id": proposal_id, "track_id": track_id,
                "predecessor_owner": predecessor,
                "successor_owner": successor,
                "resident_track": resident_track,
                "transition_frame": transition,
                "predecessor_first_frame": int(predecessor_points.frame.min())
                if len(predecessor_points) else -1,
                "predecessor_last_frame": int(predecessor_points.frame.max())
                if len(predecessor_points) else -1,
                "predecessor_support": int(len(predecessor_points)),
                "successor_support": int(len(successor_points)),
                "resident_support": int(resident_profile["support"]),
                "successor_purity": successor_purity,
                "resident_purity": float(resident_profile["purity"]),
                "track_strong_fraction": strong_fraction,
                "resident_strong_fraction": float(
                    resident_profile["strong_fraction"]),
                "transition_gap_frames": gap,
                "transition_step_radii": step,
                "field_step_radii": field_step,
                "resident_separation_sum_radii": separation,
                "component_area_ratio": area_ratio,
                "chain_links": int(bool(chain_items)),
                "chain_changed_pixels": int(sum(
                    item["pixels"] for item in chain_items))
                if not reasons else 0,
                "chain_changed_frames": len(chain_items) if not reasons else 0,
                "changed_pixels": int(sum(item["pixels"] for item in all_items))
                if not reasons else 0,
                "changed_frames": len(set(
                    int(item["frame"]) for item in all_items))
                if not reasons else 0,
                "eligible": not reasons, "applied": False,
                "reason": "eligible" if not reasons else "|".join(reasons),
            }
            audits.append(audit)
            if not reasons:
                plans.append({"audit": audit, "frames": all_items})

    conflicts: set[str] = set()
    claims: list[tuple[str, int, np.ndarray]] = []
    for plan in plans:
        proposal_id = str(plan["audit"]["proposal_id"])
        for item in plan["frames"]:
            for other_id, other_frame, other_mask in claims:
                if other_frame == int(item["frame"]) and np.any(
                        other_mask & item["mask"]):
                    conflicts.update((proposal_id, other_id))
            claims.append((proposal_id, int(item["frame"]), item["mask"]))

    candidate = labels.copy()
    frame_rows = []
    for plan in plans:
        audit = plan["audit"]
        if str(audit["proposal_id"]) in conflicts:
            audit.update({"eligible": False, "changed_pixels": 0,
                          "changed_frames": 0,
                          "reason": "global_atomic_component_conflict"})
            continue
        for item in plan["frames"]:
            frame = int(item["frame"])
            candidate[frame][item["mask"]] = int(item["successor_owner"])
            frame_rows.append({
                "proposal_id": audit["proposal_id"], "frame": frame,
                "track_id": item["track_id"],
                "predecessor_owner": item["predecessor_owner"],
                "successor_owner": item["successor_owner"],
                "component": item["component"], "pixels": item["pixels"],
            })
        audit.update({"applied": True,
                      "reason": "applied_novel_successor_backfill"})
    return (candidate, pd.DataFrame(audits, columns=AUDIT_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    if not labels.shape == unclaimed.shape == raw.shape:
        raise ValueError("labels, unclaimed, and raw stacks must align")
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        audit = pd.DataFrame(columns=AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
    else:
        candidate, audit, frames = discover_and_apply(
            labels, points, raw, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("successor separation changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("successor separation changed the identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("novel-successor backfill created duplicate components")
    output_dir = Path(out.out)
    stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(
            labels_path, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    audit_path = output_dir / "retroactive_successor_audit.csv"
    frames_path = output_dir / "retroactive_successor_frames.csv"
    duplicate_path = output_dir / "new_duplicate_lineage_proof.csv"
    audit.to_csv(audit_path, index=False)
    frames.to_csv(frames_path, index=False)
    pd.DataFrame(columns=["proposal_id", "frame", "successor_owner",
                          "new_excess", "proof"]).to_csv(
        duplicate_path, index=False)
    changed = candidate != labels
    changed_frames = np.flatnonzero(
        changed.reshape(len(labels), -1).any(axis=1)).astype(int).tolist()
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": selector_counts(params),
        "physical_tracks_audited": int(points.track_id.nunique()),
        "transitions_audited": int(len(audit)),
        "eligible_transitions": int(audit.eligible.astype(bool).sum())
        if len(audit) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": len(changed_frames),
        "changed_frame_indices": changed_frames,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_duplicate_components": int(duplicates),
        "lineage_proven_new_duplicate_components": 0,
        "all_new_duplicates_projection_proven": duplicates == 0,
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "frames": frames_path,
        "duplicate_proof": duplicate_path, "metrics": metrics_path,
    }, "summary": metrics}


