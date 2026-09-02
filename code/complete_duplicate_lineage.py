"""Allocate durable identities to complete physical duplicate branches.

Candidates are discovered only from the field-wide duplicate application
audit. A branch must be long relative to the movie and participate in a
sustained duplicate episode. Short branches remain unclaimed instead of
receiving identity fragments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import identity_fragmentation_confidence as fragmentation
import isolated_unowned_lifetime as isolated
import separable_merge_recovery as physical
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}


def assert_target_free(params: dict) -> None:
    forbidden = (
        "identity_target", "track_target", "frame_target", "coordinate",
        "region", "event_target", "review_case")
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("complete duplicate lineages must be field-wide")
    for key in params:
        if any(value in str(key).lower() for value in forbidden):
            raise ValueError(f"target-like producer parameter is forbidden: {key}")


def _track_values(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(item) for item in str(value).split("|") if item]


def resolve_data_relative_params(points: pd.DataFrame, params: dict) -> dict:
    """Resolve spatial windows from the field's physical soma scale."""
    resolved = dict(params)
    if "core_window_radius_scale" not in resolved:
        return resolved
    visible = points[points.state.isin(VISIBLE_STATES)]
    if visible.empty:
        raise ValueError(
            "complete duplicate-lineage recovery requires visible points")
    median_radius = float(np.median(
        visible.radius_px.astype(float).to_numpy()))
    resolved["core_window_radius_px"] = max(1, int(round(
        median_radius * float(resolved["core_window_radius_scale"]))))
    return resolved


def discover(labels: np.ndarray, points: pd.DataFrame,
             duplicate_applications: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find long physical lineages repeatedly released as duplicate losers."""
    params = resolve_data_relative_params(points, params)
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    frame_count = len(labels)
    minimum_span = int(np.ceil(frame_count * float(
        params.get("minimum_lineage_span_fraction_of_movie", 0.25))))
    minimum_duplicate_frames = int(np.ceil(frame_count * float(
        params.get("minimum_duplicate_support_fraction_of_movie", 0.10))))
    minimum_visible = float(params.get("minimum_visible_fraction", 0.90))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.10))
    maximum_foreign = float(params.get(
        "maximum_nonduplicate_owner_fraction", 0.10))
    duplicate_frames: dict[int, set[int]] = {}
    duplicate_owners: dict[int, set[int]] = {}
    for row in duplicate_applications.itertuples(index=False):
        for track in _track_values(row.loser_tracks):
            duplicate_frames.setdefault(track, set()).add(int(row.frame))
            duplicate_owners.setdefault(track, set()).add(int(row.owner))
    rows = []
    for track in sorted(duplicate_frames):
        group = scored[scored.track_id.astype(int) == track].sort_values("frame")
        first, last = int(group.frame.min()), int(group.frame.max())
        span = last - first + 1
        visible = group[group.state.isin(VISIBLE_STATES)]
        owners = visible.accepted_owner.astype(int)
        foreign = owners[(owners > 0) & ~owners.isin(duplicate_owners[track])]
        reasons = []
        if not np.array_equal(group.frame.astype(int).to_numpy(),
                              np.arange(first, last + 1)):
            reasons.append("physical_track_not_continuous")
        if span < minimum_span:
            reasons.append("lineage_too_short")
        if len(duplicate_frames[track]) < minimum_duplicate_frames:
            reasons.append("duplicate_episode_too_short")
        if len(visible) / max(span, 1) < minimum_visible:
            reasons.append("insufficient_visible_fraction")
        encounter_fraction = float(
            visible.in_encounter.astype(bool).mean()) if len(visible) else 1.0
        if encounter_fraction > maximum_encounter:
            reasons.append("too_many_physical_encounters")
        foreign_fraction = len(foreign) / max(len(visible), 1)
        if foreign_fraction > maximum_foreign:
            reasons.append("established_alternative_owner_present")
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "physical_track": track, "first_frame": first,
            "last_frame": last, "span_frames": span,
            "visible_frames": int(len(visible)),
            "duplicate_support_frames": len(duplicate_frames[track]),
            "duplicate_owner_count": len(duplicate_owners[track]),
            "foreign_owner_frames": int(len(foreign)),
            "foreign_owner_fraction": float(foreign_fraction),
            "encounter_fraction": encounter_fraction,
            "minimum_span_frames": minimum_span,
            "minimum_duplicate_frames": minimum_duplicate_frames,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows), scored


def _component_near(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    parts, _ = ndi.label(frame == owner, STRUCTURE)
    component = physical._component_at(
        parts, float(point.x), float(point.y),
        max(2, int(np.ceil(float(point.radius_px)))))
    return parts == component if component > 0 else None


def _point_inside(mask: np.ndarray, point) -> bool:
    y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _accepted_partition(labels_frame: np.ndarray, raw_frame: np.ndarray,
                        point, frame_points: pd.DataFrame) -> tuple[np.ndarray, str]:
    owner = physical._disk_owner(
        labels_frame, float(point.x), float(point.y), float(point.radius_px))
    if owner <= 0:
        return np.zeros_like(labels_frame, bool), "no_accepted_component"
    component = _component_near(labels_frame, owner, point)
    if component is None:
        return np.zeros_like(labels_frame, bool), "accepted_component_missing"
    companions = []
    for other in frame_points.itertuples(index=False):
        if int(other.track_id) == int(point.track_id):
            continue
        if int(other.accepted_owner) != owner:
            continue
        if _point_inside(component, other):
            companions.append(other)
    if not companions:
        return component, "relabel_separate_accepted_component"
    markers = np.zeros(component.shape, np.int16)
    target = physical._marker_pixel(component, float(point.x), float(point.y))
    if target is None:
        return np.zeros_like(component), "target_marker_missing"
    markers[target] = 1
    index = 2
    for other in companions:
        marker = physical._marker_pixel(
            component, float(other.x), float(other.y))
        if marker is None or markers[marker] > 0:
            continue
        markers[marker] = index
        index += 1
    if index == 2:
        return np.zeros_like(component), "companion_markers_missing"
    elevation = -ndi.gaussian_filter(raw_frame.astype(np.float32), 1.0)
    partition = watershed(
        elevation, markers=markers, mask=component,
        connectivity=STRUCTURE) == 1
    return partition, "raw_seeded_shared_component_partition"


def _unassigned_partition(raw_frame: np.ndarray, ledger_frame: np.ndarray,
                          point, weak_threshold: float, params: dict,
                          frame_points: pd.DataFrame) -> tuple[np.ndarray, str]:
    core, region, details = isolated._connected_core(
        raw_frame, point, weak_threshold, params)
    if not details["core_area_px"]:
        return np.zeros_like(ledger_frame, bool), "raw_core_missing"
    expected = np.pi * max(float(point.radius_px), 1.0) ** 2
    if details["core_area_px"] / max(expected, 1.0) > float(
            params.get("maximum_core_area_radius_ratio", 3.0)):
        return np.zeros_like(ledger_frame, bool), "raw_core_too_large"
    seed = np.zeros_like(ledger_frame, bool)
    seed[region] = core
    ledger = isolated._components_overlapping(ledger_frame, seed)
    proposed = seed | ledger
    for other in frame_points.itertuples(index=False):
        if int(other.track_id) != int(point.track_id) and _point_inside(
                proposed, other):
            return np.zeros_like(proposed), "raw_core_contains_other_body"
    return proposed, "raw_supported_unassigned_component"


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame,
            duplicate_applications: pd.DataFrame, params: dict):
    params = resolve_data_relative_params(points, params)
    proposals, scored = discover(
        labels, points, duplicate_applications, params)
    threshold_by_frame = thresholds.set_index("frame")
    candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
    next_identity = int(labels.max()) + 1
    applications = []
    frame_rows = []
    for proposal in proposals[
            proposals.discovery_status == "eligible"].itertuples(index=False):
        group = scored[
            (scored.track_id.astype(int) == int(proposal.physical_track))
            & scored.state.isin(VISIBLE_STATES)].sort_values("frame")
        trial, trial_unclaimed = candidate.copy(), candidate_unclaimed.copy()
        local_rows = []
        successful = 0
        for point in group.itertuples(index=False):
            frame = int(point.frame)
            frame_points = scored[
                (scored.frame.astype(int) == frame)
                & scored.physically_visible.astype(bool)]
            owner = physical._disk_owner(
                trial[frame], float(point.x), float(point.y),
                float(point.radius_px))
            if owner > 0:
                mask, method = _accepted_partition(
                    trial[frame], raw[frame], point, frame_points)
            else:
                mask, method = _unassigned_partition(
                    raw[frame], trial_unclaimed[frame], point,
                    float(threshold_by_frame.loc[frame, "weak_threshold"]),
                    params, frame_points)
            if not np.any(mask):
                local_rows.append({
                    "proposal_id": proposal.proposal_id, "frame": frame,
                    "operation": method, "changed_pixels": 0,
                    "raw_supported_additions": 0})
                continue
            before = trial[frame].copy()
            old_union = (before > 0) | (trial_unclaimed[frame] > 0)
            trial[frame][mask] = next_identity
            trial_unclaimed[frame][mask] = 0
            successful += 1
            local_rows.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "operation": method,
                "changed_pixels": int(np.count_nonzero(trial[frame] != before)),
                "raw_supported_additions": int(np.count_nonzero(
                    mask & ~old_union))})
        # Carry the new identity through short dormant gaps only when the
        # translated masks on both sides independently predict the same seat.
        by_frame = group.set_index("frame", drop=False)
        full_group = scored[
            scored.track_id.astype(int) == int(proposal.physical_track)
            ].sort_values("frame")
        maximum_bridge = max(1, int(np.ceil(len(labels) * float(
            params.get("maximum_bracketed_dormant_gap_fraction", 0.02)))))
        visible_frames = sorted(map(int, group.frame))
        for row in full_group[
                ~full_group.state.isin(VISIBLE_STATES)].itertuples(index=False):
            frame = int(row.frame)
            left = [value for value in visible_frames if value < frame]
            right = [value for value in visible_frames if value > frame]
            if not left or not right:
                continue
            left_frame, right_frame = left[-1], right[0]
            if (frame - left_frame > maximum_bridge
                    or right_frame - frame > maximum_bridge):
                continue
            left_point, right_point = by_frame.loc[left_frame], by_frame.loc[right_frame]
            if (not np.any(trial[left_frame] == next_identity)
                    or not np.any(trial[right_frame] == next_identity)):
                continue
            left_mask = _component_near(
                trial[left_frame], next_identity, left_point)
            right_mask = _component_near(
                trial[right_frame], next_identity, right_point)
            if left_mask is None or right_mask is None:
                continue
            left_shift = ndi.shift(
                left_mask.astype(np.uint8),
                shift=(float(row.y) - float(left_point.y),
                       float(row.x) - float(left_point.x)),
                order=0, mode="constant", cval=0) > 0
            right_shift = ndi.shift(
                right_mask.astype(np.uint8),
                shift=(float(row.y) - float(right_point.y),
                       float(row.x) - float(right_point.x)),
                order=0, mode="constant", cval=0) > 0
            consensus = left_shift & right_shift & (trial[frame] == 0)
            reference_area = min(
                int(np.count_nonzero(left_mask)),
                int(np.count_nonzero(right_mask)))
            minimum_consensus = max(3, int(np.ceil(reference_area * float(
                params.get("minimum_bracketed_mask_overlap_fraction", 0.20)))))
            if int(np.count_nonzero(consensus)) < minimum_consensus:
                continue
            if int(np.count_nonzero(consensus & (raw[frame] > 0))) < int(
                    params.get("minimum_bracketed_raw_positive_pixels", 1)):
                continue
            frame_points = scored[
                (scored.frame.astype(int) == frame)
                & scored.physically_visible.astype(bool)]
            if any(_point_inside(consensus, other)
                   for other in frame_points.itertuples(index=False)):
                continue
            old_union = (trial[frame] > 0) | (trial_unclaimed[frame] > 0)
            trial[frame][consensus] = next_identity
            trial_unclaimed[frame][consensus] = 0
            successful += 1
            local_rows.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "operation": "bracketed_dormant_mask_consensus",
                "changed_pixels": int(np.count_nonzero(consensus)),
                "raw_supported_additions": int(np.count_nonzero(
                    consensus & ~old_union))})
        required = int(np.ceil(len(group) * float(
            params.get("minimum_lifetime_application_fraction", 0.90))))
        if successful < required:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "physical_track": int(proposal.physical_track),
                "assigned_identity": 0, "outcome": "rejected_application",
                "reason": "too_few_supported_lifetime_frames",
                "successful_frames": successful, "required_frames": required,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        table, gate = fragmentation.score_new_identity_fragments(
            labels, trial,
            minimum_unanchored_lifespan_fraction=float(params.get(
                "minimum_fragmentation_lifespan_fraction", 0.25)))
        row = table[table.identity.astype(int) == next_identity]
        if row.empty or str(row.iloc[0].status) != "pass":
            applications.append({
                "proposal_id": proposal.proposal_id,
                "physical_track": int(proposal.physical_track),
                "assigned_identity": 0, "outcome": "rejected_application",
                "reason": "fragmentation_gate_failed",
                "successful_frames": successful, "required_frames": required,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        changed = trial != candidate
        candidate, candidate_unclaimed = trial, trial_unclaimed
        applications.append({
            "proposal_id": proposal.proposal_id,
            "physical_track": int(proposal.physical_track),
            "assigned_identity": next_identity, "outcome": "applied",
            "reason": "complete_duplicate_lineage_allocated",
            "successful_frames": successful, "required_frames": required,
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(
                changed, axis=(1, 2))))})
        frame_rows.extend(local_rows)
        next_identity += 1
    return (candidate, candidate_unclaimed, proposals,
            pd.DataFrame(applications), pd.DataFrame(frame_rows), scored)


def substantial_duplicate_rows(labels: np.ndarray, minimum_area: int = 8,
                               ) -> int:
    return component_accounting.duplicate_owner_frames(
        labels, minimum_area=minimum_area)
