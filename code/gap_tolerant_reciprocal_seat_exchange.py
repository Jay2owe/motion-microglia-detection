"""Repair bilaterally bracketed reciprocal owner exchanges field-wide.

Only movie-relative or dimensionless thresholds are accepted.  Reviewed
identities, tracks, frames, coordinates, events, and regions are deliberately
not part of the production interface.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import owner_consensus
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_pair", "forced_interval", "failure_target",
)
AUDIT_COLUMNS = [
    "proposal_id", "frame", "track_a", "track_b", "owner_a", "owner_b",
    "prior_frame", "return_frame", "prior_support_a", "prior_support_b",
    "return_support_a", "return_support_b", "return_observations_a",
    "return_observations_b", "return_owner_fraction_a",
    "return_owner_fraction_b", "minimum_owner_support",
    "minimum_return_support", "minimum_return_owner_fraction",
    "prior_separation_sum_radii", "exchange_separation_sum_radii",
    "return_separation_sum_radii", "same_track_motion_cost",
    "crossed_motion_cost", "motion_cost_ratio", "third_component_cores",
    "eligible", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("gap-tolerant reciprocal exchange must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "gap-tolerant reciprocal exchange received forbidden targets: "
            + ", ".join(supplied))


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _valid(point, minimum_purity: float, minimum_coverage: float) -> bool:
    return bool(
        point is not None and bool(point.physically_visible)
        and int(point.candidate_owner) > 0
        and float(point.candidate_owner_purity) >= minimum_purity
        and float(point.candidate_coverage_fraction) >= minimum_coverage)


def _distance(left, right) -> float:
    return float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))


def _scaled_distance(left, right) -> float:
    radii = float(left.radius_px) + float(right.radius_px)
    return _distance(left, right) / max(radii, 1.0)


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    if count <= 0:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    value = int(components[y, x])
    if value <= 0:
        radius = max(2.0, 0.75 * float(point.radius_px))
        yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
        disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
                <= radius ** 2)
        values = components[disk & (components > 0)]
        if not len(values):
            return None
        ids, counts = np.unique(values, return_counts=True)
        value = int(ids[int(np.argmax(counts))])
    return components == value


def _owner_support(index: pd.DataFrame, track: int, owner: int,
                   frames: range, minimum_purity: float,
                   minimum_coverage: float) -> int:
    return sum(
        _valid(point := _point(index, track, frame),
               minimum_purity, minimum_coverage)
        and int(point.candidate_owner) == int(owner)
        for frame in frames)


def _bracket(index: pd.DataFrame, track_a: int, track_b: int,
             owner_a: int, owner_b: int, frames: range,
             minimum_purity: float, minimum_coverage: float):
    for frame in frames:
        point_a = _point(index, track_a, frame)
        point_b = _point(index, track_b, frame)
        if (not _valid(point_a, minimum_purity, minimum_coverage)
                or not _valid(point_b, minimum_purity, minimum_coverage)):
            continue
        if (int(point_a.candidate_owner) == int(owner_a)
                and int(point_b.candidate_owner) == int(owner_b)):
            return int(frame), point_a, point_b
    return None


def _third_component_cores(attached: pd.DataFrame, frame: int,
                           excluded: set[int], component_a: np.ndarray,
                           component_b: np.ndarray) -> int:
    count = 0
    rows = attached[
        (attached.frame.astype(int) == int(frame))
        & attached.physically_visible.astype(bool)]
    for point in rows.itertuples(index=False):
        if int(point.track_id) in excluded:
            continue
        y = int(np.clip(round(float(point.y)), 0, component_a.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, component_a.shape[1] - 1))
        count += int(bool(component_a[y, x] or component_b[y, x]))
    return count


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every co-visible pair for a bracketed reciprocal exchange."""
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)
    index = attached.set_index(["track_id", "frame"], drop=False)
    minimum_purity = float(params.get("minimum_owner_purity", 0.8))
    minimum_coverage = float(params.get("minimum_owner_coverage", 0.5))
    bracket_gap = max(1, int(math.ceil(float(params.get(
        "maximum_bracket_gap_movie_fraction", 0.03)) * len(labels))))
    support_window = max(bracket_gap, int(math.ceil(float(params.get(
        "owner_support_window_movie_fraction", 0.08)) * len(labels))))
    return_window = max(bracket_gap, int(math.ceil(float(params.get(
        "return_support_window_movie_fraction", 0.04)) * len(labels))))
    minimum_support = max(2, int(math.ceil(float(params.get(
        "minimum_owner_support_movie_fraction", 0.04)) * len(labels))))
    minimum_return_support = max(2, int(math.ceil(float(params.get(
        "minimum_return_support_movie_fraction", 0.02)) * len(labels))))
    minimum_return_fraction = float(params.get(
        "minimum_return_owner_fraction", 0.75))
    maximum_separation = float(params.get(
        "maximum_pair_separation_sum_radii", 3.0))
    minimum_motion_ratio = float(params.get(
        "minimum_crossed_to_same_motion_cost_ratio", 1.5))
    rows: list[dict] = []
    by_frame = attached[
        attached.physically_visible.astype(bool)
        & (attached.candidate_owner.astype(int) > 0)].groupby("frame")
    for frame, group in by_frame:
        frame = int(frame)
        if frame <= 0 or frame >= len(labels) - 1:
            continue
        current = list(group.itertuples(index=False))
        for left_index, point_a in enumerate(current):
            if not _valid(point_a, minimum_purity, minimum_coverage):
                continue
            for point_b in current[left_index + 1:]:
                if not _valid(point_b, minimum_purity, minimum_coverage):
                    continue
                track_a, track_b = int(point_a.track_id), int(point_b.track_id)
                if track_a == track_b:
                    continue
                if track_a > track_b:
                    track_a, track_b = track_b, track_a
                    point_a, point_b = point_b, point_a
                exchanged_a = int(point_a.candidate_owner)
                exchanged_b = int(point_b.candidate_owner)
                if exchanged_a == exchanged_b:
                    continue
                owner_a, owner_b = exchanged_b, exchanged_a
                prior = _bracket(
                    index, track_a, track_b, owner_a, owner_b,
                    range(frame - 1, max(-1, frame - bracket_gap - 1), -1),
                    minimum_purity, minimum_coverage)
                returned = _bracket(
                    index, track_a, track_b, owner_a, owner_b,
                    range(frame + 1, min(len(labels), frame + bracket_gap + 1)),
                    minimum_purity, minimum_coverage)
                if prior is None or returned is None:
                    continue
                prior_frame, prior_a, prior_b = prior
                return_frame, return_a, return_b = returned
                reasons: list[str] = []
                prior_support_a = _owner_support(
                    index, track_a, owner_a,
                    range(max(0, frame - support_window), frame),
                    minimum_purity, minimum_coverage)
                prior_support_b = _owner_support(
                    index, track_b, owner_b,
                    range(max(0, frame - support_window), frame),
                    minimum_purity, minimum_coverage)
                return_frames = range(
                    frame + 1, min(len(labels), frame + 1 + return_window))
                return_support_a = _owner_support(
                    index, track_a, owner_a, return_frames,
                    minimum_purity, minimum_coverage)
                return_frames = range(
                    frame + 1, min(len(labels), frame + 1 + return_window))
                return_support_b = _owner_support(
                    index, track_b, owner_b, return_frames,
                    minimum_purity, minimum_coverage)
                return_frames = range(
                    frame + 1, min(len(labels), frame + 1 + return_window))
                return_observations_a = sum(
                    _valid(_point(index, track_a, item),
                           minimum_purity, minimum_coverage)
                    for item in return_frames)
                return_frames = range(
                    frame + 1, min(len(labels), frame + 1 + return_window))
                return_observations_b = sum(
                    _valid(_point(index, track_b, item),
                           minimum_purity, minimum_coverage)
                    for item in return_frames)
                return_fraction_a = return_support_a / max(
                    return_observations_a, 1)
                return_fraction_b = return_support_b / max(
                    return_observations_b, 1)
                if min(prior_support_a, prior_support_b) < minimum_support:
                    reasons.append("insufficient_prior_bilateral_owner_support")
                if min(return_support_a,
                       return_support_b) < minimum_return_support:
                    reasons.append("insufficient_return_bilateral_owner_support")
                if min(return_fraction_a,
                       return_fraction_b) < minimum_return_fraction:
                    reasons.append("impure_return_owner_support")
                separations = (
                    _scaled_distance(prior_a, prior_b),
                    _scaled_distance(point_a, point_b),
                    _scaled_distance(return_a, return_b),
                )
                if max(separations) > maximum_separation:
                    reasons.append("pair_not_local")
                same_cost = (
                    _distance(prior_a, point_a) + _distance(prior_b, point_b)
                    + _distance(point_a, return_a) + _distance(point_b, return_b))
                crossed_cost = (
                    _distance(prior_a, point_b) + _distance(prior_b, point_a)
                    + _distance(point_a, return_b) + _distance(point_b, return_a))
                motion_ratio = crossed_cost / max(same_cost, 1e-9)
                if motion_ratio < minimum_motion_ratio:
                    reasons.append("physical_motion_does_not_favour_same_tracks")
                component_a = _component_at(labels[frame], exchanged_a, point_a)
                component_b = _component_at(labels[frame], exchanged_b, point_b)
                if component_a is None or component_b is None:
                    reasons.append("exchange_component_missing")
                    third = 0
                else:
                    if np.any(component_a & component_b):
                        reasons.append("exchange_components_not_distinct")
                    third = _third_component_cores(
                        attached, frame, {track_a, track_b},
                        component_a, component_b)
                    if third:
                        reasons.append("third_physical_core_in_exchange_components")
                rows.append({
                    "proposal_id": "", "frame": frame,
                    "track_a": track_a, "track_b": track_b,
                    "owner_a": owner_a, "owner_b": owner_b,
                    "prior_frame": prior_frame, "return_frame": return_frame,
                    "prior_support_a": prior_support_a,
                    "prior_support_b": prior_support_b,
                    "return_support_a": return_support_a,
                    "return_support_b": return_support_b,
                    "return_observations_a": return_observations_a,
                    "return_observations_b": return_observations_b,
                    "return_owner_fraction_a": return_fraction_a,
                    "return_owner_fraction_b": return_fraction_b,
                    "minimum_owner_support": minimum_support,
                    "minimum_return_support": minimum_return_support,
                    "minimum_return_owner_fraction": minimum_return_fraction,
                    "prior_separation_sum_radii": separations[0],
                    "exchange_separation_sum_radii": separations[1],
                    "return_separation_sum_radii": separations[2],
                    "same_track_motion_cost": same_cost,
                    "crossed_motion_cost": crossed_cost,
                    "motion_cost_ratio": motion_ratio,
                    "third_component_cores": int(third),
                    "eligible": not reasons,
                    "reason": ("eligible_bilaterally_bracketed_reciprocal_exchange"
                               if not reasons else "|".join(reasons)),
                    "_component_a": component_a,
                    "_component_b": component_b,
                })
    eligible_number = 0
    for row in rows:
        if row["eligible"]:
            eligible_number += 1
            row["proposal_id"] = f"GX{eligible_number:04d}"
    public = pd.DataFrame(
        [{key: row[key] for key in AUDIT_COLUMNS} for row in rows],
        columns=AUDIT_COLUMNS)
    return public, pd.DataFrame(rows)


def apply(labels: np.ndarray, internal: pd.DataFrame,
          ) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply complete reciprocal swaps one proposal at a time with safety."""
    candidate = labels.copy()
    applications: list[dict] = []
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for _, proposal in eligible.iterrows():
        frame = int(proposal["frame"])
        trial = candidate.copy()
        component_a = proposal["_component_a"]
        component_b = proposal["_component_b"]
        trial[frame][component_a] = int(proposal["owner_a"])
        trial[frame][component_b] = int(proposal["owner_b"])
        reasons: list[str] = []
        if not np.array_equal(trial > 0, candidate > 0):
            reasons.append("foreground_changed")
        if ((set(map(int, np.unique(trial))) - {0})
                != (set(map(int, np.unique(labels))) - {0})):
            reasons.append("identity_set_changed")
        duplicates = owner_consensus.count_new_duplicate_components(labels, trial)
        if duplicates:
            reasons.append("new_duplicate_components")
        changed = int(np.count_nonzero(trial != candidate))
        if not changed:
            reasons.append("already_consistent")
        applied = not reasons
        if applied:
            candidate = trial
        applications.append({
            "proposal_id": str(proposal["proposal_id"]), "frame": frame,
            "track_a": int(proposal["track_a"]),
            "track_b": int(proposal["track_b"]),
            "owner_a": int(proposal["owner_a"]),
            "owner_b": int(proposal["owner_b"]),
            "applied": applied,
            "changed_pixels": changed if applied else 0,
            "reason": ("atomic_reciprocal_component_restore" if applied
                       else "|".join(reasons)),
        })
    return candidate, pd.DataFrame(applications)


def produce(labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
            params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                       dict]:
    """Discover and apply field-wide exchanges while preserving both ledgers."""
    assert_target_free(params)
    audit, internal = discover(labels, points, params)
    candidate, applications = apply(labels, internal)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("gap-tolerant reciprocal exchange changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("gap-tolerant reciprocal exchange changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("gap-tolerant reciprocal exchange created duplicates")
    changed = candidate != labels
    applied = (applications[applications.applied.astype(bool)]
               if len(applications) else applications)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "pairs_audited": int(len(audit)),
        "eligible_exchanges": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_exchanges": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_duplicate_components": int(duplicates),
    }
    return candidate, unclaimed.copy(), audit, applications, summary
