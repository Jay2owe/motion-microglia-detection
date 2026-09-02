"""Isolate complete dim lineages that repeatedly borrow distant owners.

Discovery is field-wide. Identity values, physical-track numbers, frames,
coordinates, events, and review cases are never accepted as parameters.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import concurrent_duplicate_invasion as invasion
import ownerless_body_recovery as ownerless
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)


def assert_target_free(params: dict) -> None:
    invasion.assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("conflicted dim-lineage isolation must be field-wide")


def _raw_core(raw_frame: np.ndarray, point, threshold: float, params: dict,
              evidence: np.ndarray | None = None):
    if evidence is None:
        core, region, details = ownerless._connected_core(
            raw_frame, point, threshold, params)
    else:
        y = int(np.clip(round(float(point.y)), 0, raw_frame.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, raw_frame.shape[1] - 1))
        half = int(params.get("core_window_radius_px", 7))
        seed_y, seed_x = y, x
        peak = float(evidence[seed_y, seed_x])
        minimum_level = (float(params.get("weak_threshold_fraction", 0.5))
                         * threshold)
        if peak < minimum_level or raw_frame[seed_y, seed_x] <= 0:
            search_distance = float(params.get(
                "maximum_core_recenter_distance_radii", 2.25)) \
                * max(float(point.radius_px), 1.0)
            search = int(np.ceil(search_distance))
            sy0, sy1 = max(0, y - search), min(raw_frame.shape[0], y + search + 1)
            sx0, sx1 = max(0, x - search), min(raw_frame.shape[1], x + search + 1)
            yy, xx = np.ogrid[sy0:sy1, sx0:sx1]
            allowed = (((xx - float(point.x)) ** 2
                        + (yy - float(point.y)) ** 2) <= search_distance ** 2)
            local_scores = np.where(
                allowed & (raw_frame[sy0:sy1, sx0:sx1] > 0)
                & (evidence[sy0:sy1, sx0:sx1] >= minimum_level),
                evidence[sy0:sy1, sx0:sx1], -np.inf)
            if np.any(np.isfinite(local_scores)):
                local_y, local_x = np.unravel_index(
                    int(np.argmax(local_scores)), local_scores.shape)
                seed_y, seed_x = sy0 + int(local_y), sx0 + int(local_x)
                peak = float(evidence[seed_y, seed_x])
        y0, y1 = (max(0, seed_y - half),
                  min(raw_frame.shape[0], seed_y + half + 1))
        x0, x1 = (max(0, seed_x - half),
                  min(raw_frame.shape[1], seed_x + half + 1))
        level = max(float(params.get("core_peak_fraction", 0.35)) * peak,
                    minimum_level)
        local = ((evidence[y0:y1, x0:x1] >= level)
                 & (raw_frame[y0:y1, x0:x1] > 0))
        components, _ = ndi.label(local)
        component_id = int(components[seed_y - y0, seed_x - x0])
        core = (components == component_id if component_id
                else np.zeros_like(local, bool))
        region = (slice(y0, y1), slice(x0, x1))
        details = {"peak_evidence": peak, "core_threshold": level,
                   "core_area_px": int(core.sum()),
                   "recenter_distance_radii": float(np.hypot(
                       seed_x - float(point.x), seed_y - float(point.y))
                       / max(float(point.radius_px), 1.0))}
    expected = np.pi * max(float(point.radius_px), 1.0) ** 2
    ratio = float(details["core_area_px"] / max(expected, 1.0))
    available = bool(details["core_area_px"] > 0 and ratio <= float(
        params.get("maximum_core_area_radius_ratio", 2.5)))
    seed = np.zeros(raw_frame.shape, bool)
    seed[region] = core
    return seed, available, ratio, details


def _runs(rows: list) -> list[list]:
    result: list[list] = []
    start = 0
    while start < len(rows):
        if not bool(rows[start].raw_core_available):
            start += 1
            continue
        end = start
        while (end + 1 < len(rows)
               and bool(rows[end + 1].raw_core_available)
               and int(rows[end + 1].frame) == int(rows[end].frame) + 1):
            end += 1
        result.append(rows[start:end + 1])
        start = end + 1
    return result


def _nonzero_transactions(owners: Iterable[int]) -> int:
    values = [int(value) for value in owners if int(value) > 0]
    return int(sum(left != right for left, right in zip(values, values[1:])))


def _anchor(point, scored: pd.DataFrame, owner: int):
    peers = scored[
        scored.frame.astype(int).eq(int(point.frame))
        & ~scored.track_id.astype(int).eq(int(point.track_id))
        & scored.physically_visible.astype(bool)
        & scored.accepted_owner.astype(int).eq(int(owner))]
    if len(peers) == 0:
        return None, np.inf
    distances = np.hypot(
        peers.x.to_numpy(float) - float(point.x),
        peers.y.to_numpy(float) - float(point.y))
    index = int(np.argmin(distances))
    peer = peers.iloc[index]
    scale = max(float(point.radius_px) + float(peer.radius_px), 1.0)
    return peer, float(distances[index] / scale)


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             thresholds: pd.DataFrame, params: dict):
    """Audit every maximal, frame-complete raw-core interval."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    threshold_by_frame = thresholds.set_index("frame")
    evidence_by_frame = [ownerless.evidence_image(frame, params) for frame in raw]
    core_available: list[bool] = []
    core_ratios: list[float] = []
    for point in scored.itertuples(index=False):
        _, available, ratio, _ = _raw_core(
            raw[int(point.frame)], point,
            float(threshold_by_frame.loc[int(point.frame), "weak_threshold"]),
            params, evidence_by_frame[int(point.frame)])
        core_available.append(available)
        core_ratios.append(ratio)
    scored["raw_core_available"] = core_available
    scored["raw_core_area_radius_ratio"] = core_ratios

    frame_count = int(len(labels))
    minimum_interval = int(params.get("minimum_complete_interval_frames", 8))
    maximum_interval = int(np.floor(frame_count * float(
        params.get("maximum_complete_interval_fraction_of_movie", 0.15))))
    maximum_track_span = int(np.floor(frame_count * float(
        params.get("maximum_physical_track_span_fraction_of_movie", 0.20))))
    minimum_named = float(params.get("minimum_named_fraction", 0.60))
    minimum_transactions = int(params.get("minimum_owner_transactions", 2))
    minimum_anchor = float(params.get("minimum_concurrent_anchor_fraction", 0.75))
    minimum_anchor_separation = float(params.get(
        "minimum_anchor_separation_sum_radii", 1.5))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.0))
    minimum_isolation = float(params.get("minimum_median_separation_radii", 4.0))
    rows: list[dict] = []
    proposal = 0
    for track, group in scored.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        span = int(group.frame.max() - group.frame.min() + 1)
        for interval in _runs(list(group.itertuples(index=False))):
            proposal += 1
            first, last = int(interval[0].frame), int(interval[-1].frame)
            owners = [int(point.accepted_owner) for point in interval]
            positive = [owner for owner in owners if owner > 0]
            distinct = sorted(set(positive))
            anchors = []
            anchor_separations = []
            for point, owner in zip(interval, owners):
                if owner <= 0:
                    continue
                peer, separation = _anchor(point, scored, owner)
                anchors.append(peer is not None)
                if peer is not None:
                    anchor_separations.append(separation)
            named_fraction = float(len(positive) / max(len(interval), 1))
            anchor_fraction = float(np.mean(anchors)) if anchors else 0.0
            minimum_seen_anchor_separation = (float(min(anchor_separations))
                                               if anchor_separations else 0.0)
            encounter_fraction = float(np.mean([
                bool(point.in_encounter) for point in interval]))
            separation = np.asarray([
                float(point.nearest_track_distance_px)
                / max(float(point.radius_px), 1.0) for point in interval])
            median_separation = float(np.median(separation))
            transactions = _nonzero_transactions(owners)
            reasons: list[str] = []
            if len(interval) < minimum_interval or len(interval) > maximum_interval:
                reasons.append("complete_interval_outside_duration_range")
            if span > maximum_track_span:
                reasons.append("physical_track_too_long")
            if first <= 0 or last >= frame_count - 1:
                reasons.append("interval_censored_by_movie_boundary")
            if len(distinct) != 2:
                reasons.append("not_exactly_two_borrowed_owners")
            if named_fraction < minimum_named:
                reasons.append("insufficient_named_fraction")
            if transactions < minimum_transactions:
                reasons.append("insufficient_owner_transactions")
            if anchor_fraction < minimum_anchor:
                reasons.append("insufficient_concurrent_owner_anchors")
            if minimum_seen_anchor_separation < minimum_anchor_separation:
                reasons.append("owner_anchor_not_spatially_distinct")
            if encounter_fraction > maximum_encounter:
                reasons.append("physical_track_has_encounter")
            if median_separation < minimum_isolation:
                reasons.append("physical_track_not_isolated")
            rows.append({
                "proposal_id": f"P{proposal:04d}",
                "physical_track": int(track), "first_frame": first,
                "last_frame": last, "interval_frames": int(len(interval)),
                "physical_track_span_frames": span,
                "borrowed_owner_count": int(len(distinct)),
                "borrowed_owners": "|".join(map(str, distinct)),
                "named_frames": int(len(positive)),
                "named_fraction": named_fraction,
                "owner_transactions": transactions,
                "concurrent_anchor_fraction": anchor_fraction,
                "minimum_anchor_separation_sum_radii":
                    minimum_seen_anchor_separation,
                "encounter_fraction": encounter_fraction,
                "median_separation_radii": median_separation,
                "maximum_core_area_radius_ratio": float(max(
                    point.raw_core_area_radius_ratio for point in interval)),
                "median_x": float(np.median([point.x for point in interval])),
                "median_y": float(np.median([point.y for point in interval])),
                "discovery_status": "eligible" if not reasons else "rejected",
                "discovery_reason": "eligible" if not reasons else "|".join(reasons),
            })
    return pd.DataFrame(rows), scored


def _component_near(frame: np.ndarray, owner: int, point):
    return invasion._component_near(frame, int(owner), point)


def _owner_partition(frame: np.ndarray, raw_frame: np.ndarray,
                     scored: pd.DataFrame, point, owner: int,
                     params: dict):
    component = _component_near(frame, owner, point)
    if component is None:
        return (np.zeros_like(frame, bool), np.zeros_like(frame, bool), None,
                "owner_component_missing")
    peers = scored[
        scored.frame.astype(int).eq(int(point.frame))
        & ~scored.track_id.astype(int).eq(int(point.track_id))
        & scored.physically_visible.astype(bool)
        & scored.accepted_owner.astype(int).eq(owner)]
    donor = None
    donor_marker = None
    if len(peers):
        distances = np.hypot(
            peers.x.to_numpy(float) - float(point.x),
            peers.y.to_numpy(float) - float(point.y))
        for index in np.argsort(distances):
            candidate = peers.iloc[int(index)]
            marker = physical._marker_pixel(
                component, float(candidate.x), float(candidate.y))
            if marker is not None:
                donor, donor_marker = candidate, marker
                break
    target_marker = physical._marker_pixel(
        component, float(point.x), float(point.y))
    if target_marker is None:
        return (np.zeros_like(frame, bool), component, None,
                "target_marker_missing")
    if donor is None:
        smooth = ndi.gaussian_filter(
            raw_frame.astype(np.float32),
            float(params.get("watershed_sigma_px", 1.0)))
        yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
        scale = max(float(point.radius_px), 1.0)
        allowed = component & (
            (xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
            >= (float(params.get("minimum_residual_distance_radii", 1.5))
                * scale) ** 2)
        peaks = allowed & (smooth == ndi.maximum_filter(smooth, size=3)) \
            & (raw_frame > 0)
        if np.any(peaks):
            scores = np.where(peaks, smooth, -np.inf)
            residual = np.unravel_index(int(np.argmax(scores)), scores.shape)
            component_peak = max(float(np.max(smooth[component])), 1.0)
            residual_fraction = float(smooth[residual] / component_peak)
            target_fraction = float(smooth[target_marker] / component_peak)
            if (residual_fraction >= float(params.get(
                    "minimum_residual_peak_fraction", 0.25))
                    and target_fraction >= float(params.get(
                        "minimum_target_peak_fraction", 0.5))):
                donor_marker = (int(residual[0]), int(residual[1]))
        if donor_marker is None:
            return (component, component, None,
                    "relabel_isolated_owner_component")
    if donor_marker == target_marker:
        return component, component, None, "relabel_isolated_owner_component"
    markers = np.zeros(frame.shape, np.int16)
    markers[target_marker] = 1
    markers[donor_marker] = 2
    elevation = -ndi.gaussian_filter(
        raw_frame.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    basins = watershed(
        elevation, markers=markers, mask=component, connectivity=STRUCTURE)
    return (basins == 1, component, donor_marker,
            ("raw_seeded_shared_owner_partition" if donor is not None
             else "raw_seeded_residual_owner_partition"))


def _ledger_components(frame: np.ndarray, seed: np.ndarray) -> np.ndarray:
    selected = np.zeros_like(frame, bool)
    for value in sorted(set(map(int, np.unique(frame[seed]))) - {0}):
        components, _ = ndi.label(frame == value, STRUCTURE)
        for component_id in set(map(int, np.unique(components[seed]))) - {0}:
            selected |= components == component_id
    return selected


def _duplicate_components(stack: np.ndarray) -> int:
    total = 0
    for identity in set(map(int, np.unique(stack))) - {0}:
        for frame in stack:
            total += max(0, int(ndi.label(frame == identity, STRUCTURE)[1]) - 1)
    return int(total)


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict):
    proposals, scored = discover(labels, raw, points, thresholds, params)
    threshold_by_frame = thresholds.set_index("frame")
    evidence_by_frame = [ownerless.evidence_image(frame, params) for frame in raw]
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    next_identity = int(max(labels.max(), unclaimed.max())) + 1
    applications: list[dict] = []
    frames: list[dict] = []
    occupied = np.zeros_like(labels, bool)
    before_duplicates = _duplicate_components(labels)
    for proposal in proposals[proposals.discovery_status.eq("eligible")].sort_values(
            ["first_frame", "physical_track"]).itertuples(index=False):
        trial = candidate.copy()
        trial_unclaimed = candidate_unclaimed.copy()
        local: list[dict] = []
        reason = ""
        group = groups[int(proposal.physical_track)]
        for frame_index in range(int(proposal.first_frame),
                                 int(proposal.last_frame) + 1):
            rows = group[group.frame.astype(int).eq(frame_index)]
            if len(rows) != 1:
                reason = "physical_point_missing"
                break
            point = rows.iloc[0]
            seed, available, area_ratio, details = _raw_core(
                raw[frame_index], point,
                float(threshold_by_frame.loc[frame_index, "weak_threshold"]),
                params, evidence_by_frame[frame_index])
            if not available:
                reason = "raw_core_interval_not_complete"
                break
            owner = int(point.accepted_owner)
            if owner > 0:
                partition, owner_component, donor_marker, operation = _owner_partition(
                    trial[frame_index], raw[frame_index], scored,
                    point, owner, params)
                if not np.any(partition):
                    reason = operation
                    break
            else:
                partition = np.zeros_like(seed)
                owner_component = np.zeros_like(seed)
                donor_marker = None
                operation = "claim_ownerless_raw_core"
            ledger = _ledger_components(trial_unclaimed[frame_index], seed)
            # Every pixel in this connected, recentered raw core belongs to
            # the isolated target. Include its current borrowed-owner pixels,
            # then absorb any orphaned residual shards from the same original
            # owner component while preserving the donor-marked residual.
            clean_seed = seed & ((trial[frame_index] == 0)
                                 | (trial[frame_index] == owner))
            proposed = partition | clean_seed | ledger
            additions = proposed & (trial[frame_index] == 0)
            if np.any(additions & (raw[frame_index] <= 0)):
                reason = "zero_signal_addition"
                break
            before = trial[frame_index].copy()
            trial[frame_index][proposed] = next_identity
            trial_unclaimed[frame_index][proposed] = 0
            donor_shards = np.zeros_like(seed)
            if owner > 0 and donor_marker is not None:
                residual_labels, _ = ndi.label(
                    owner_component & (trial[frame_index] == owner), STRUCTURE)
                donor_part = int(residual_labels[donor_marker])
                if donor_part <= 0:
                    reason = "donor_partition_missing"
                    break
                donor_shards = ((residual_labels > 0)
                                & (residual_labels != donor_part))
                trial[frame_index][donor_shards] = next_identity
            if ndi.label(trial[frame_index] == next_identity, STRUCTURE)[1] != 1:
                reason = "fresh_identity_not_one_component"
                break
            local.append({
                "proposal_id": str(proposal.proposal_id), "frame": frame_index,
                "physical_track": int(proposal.physical_track),
                "assigned_identity": next_identity, "borrowed_owner": owner,
                "operation": operation,
                "donor_shards_reassigned": int(np.count_nonzero(donor_shards)),
                "raw_core_area_px": int(details["core_area_px"]),
                "raw_core_area_radius_ratio": area_ratio,
                "changed_pixels": int(np.count_nonzero(
                    trial[frame_index] != before)),
            })
        changed = trial != candidate
        if not reason and np.any(changed & occupied):
            reason = "proposal_overlaps_prior_application"
        if not reason and _duplicate_components(trial) > before_duplicates:
            reason = "new_same_frame_identity_component"
        if reason:
            applications.append({
                "proposal_id": str(proposal.proposal_id),
                "physical_track": int(proposal.physical_track),
                "assigned_identity": 0, "outcome": "rejected_application",
                "reason": reason, "changed_pixels": 0, "changed_frames": 0})
            continue
        candidate, candidate_unclaimed = trial, trial_unclaimed
        occupied |= changed
        frames.extend(local)
        applications.append({
            "proposal_id": str(proposal.proposal_id),
            "physical_track": int(proposal.physical_track),
            "assigned_identity": next_identity, "outcome": "applied",
            "reason": "complete_conflicted_raw_lineage_isolated",
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2))))})
        next_identity += 1
    columns = ["proposal_id", "physical_track", "assigned_identity", "outcome",
               "reason", "changed_pixels", "changed_frames"]
    return (candidate, candidate_unclaimed, proposals,
            pd.DataFrame(applications, columns=columns), pd.DataFrame(frames), scored)


def summarize(labels: np.ndarray, unclaimed: np.ndarray,
              candidate: np.ndarray, candidate_unclaimed: np.ndarray,
              raw: np.ndarray, proposals: pd.DataFrame,
              applications: pd.DataFrame) -> dict:
    changed = candidate != labels
    ledger_changed = candidate_unclaimed != unclaimed
    old_ids = set(map(int, np.unique(labels))) - {0}
    new_ids = set(map(int, np.unique(candidate))) - {0}
    additions = (candidate > 0) & (labels == 0)
    losses = 0
    for identity in old_ids:
        before = np.any(labels == identity, axis=(1, 2))
        after = np.any(candidate == identity, axis=(1, 2))
        losses += int(np.count_nonzero(before & ~after))
    applied = applications[applications.outcome.eq("applied")] \
        if len(applications) else applications
    return {
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "raw_intervals_audited": int(len(proposals)),
        "eligible_conflicted_lineages": int(
            proposals.discovery_status.eq("eligible").sum()),
        "applied_conflicted_lineages": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "raw_supported_additions": int(np.count_nonzero(additions)),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw <= 0))),
        "foreground_removed_pixels": int(np.count_nonzero(
            (labels > 0) & (candidate == 0))),
        "unclaimed_ledger_changed_pixels": int(np.count_nonzero(ledger_changed)),
        "old_identity_set_preserved": bool(old_ids <= new_ids),
        "new_identity_count": int(len(new_ids - old_ids)),
        "donor_named_frame_losses": losses,
        "new_same_frame_identity_components": max(
            0, _duplicate_components(candidate) - _duplicate_components(labels)),
    }

