"""Recover stable early identities hidden by a duplicated prior owner.

Discovery is field-wide. Identity values are treated only as opaque labels after
an early stable birth is discovered; no identity, track, frame, coordinate, event,
or review-case target is accepted by this producer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "identity_ids", "track_id", "track_ids", "frame_id",
    "frame_ids", "event_id", "event_ids", "coordinate", "coordinates",
    "review_case", "review_cases", "target_identity", "target_track",
)


def assert_target_free(params: dict) -> None:
    """Refuse parameters capable of carrying a selected problem case."""
    forbidden = sorted(
        key for key in params
        if any(token in key.lower() for token in FORBIDDEN_TARGET_TOKENS))
    if forbidden:
        raise ValueError(
            f"field-wide producer received forbidden targets: {forbidden}")


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    return float(xx.mean()), float(yy.mean())


def _radius(area: float) -> float:
    return float(np.sqrt(max(float(area), 1.0) / np.pi))


def _components(mask: np.ndarray) -> list[dict]:
    component_labels, count = ndi.label(mask, STRUCTURE)
    result = []
    for component_id in range(1, count + 1):
        component = component_labels == component_id
        result.append({
            "id": component_id,
            "mask": component,
            "area": int(np.count_nonzero(component)),
            "center": _centroid(component),
        })
    return result


def _distance(left: tuple[float, float],
              right: tuple[float, float]) -> float:
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))


def _median_area(labels: np.ndarray, identity: int, start: int) -> float:
    areas = np.count_nonzero(labels[start:] == identity, axis=(1, 2))
    positive = areas[areas > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def _nearest_foreground_owner(frame: np.ndarray,
                              center: tuple[float, float],
                              radius: float) -> int:
    yy, xx = np.nonzero(frame > 0)
    if not len(xx):
        return 0
    distance2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    best = int(np.argmin(distance2))
    if float(distance2[best]) > radius ** 2:
        return 0
    return int(frame[yy[best], xx[best]])


def _best_distinct_pair(parts: list[dict], released: tuple[float, float],
                        resident: tuple[float, float],
                        scale: float) -> tuple[dict, dict, float] | None:
    best = None
    for released_part in parts:
        for resident_part in parts:
            if released_part["id"] == resident_part["id"]:
                continue
            cost = (_distance(released_part["center"], released)
                    + _distance(resident_part["center"], resident)) / max(
                        scale, 1.0)
            if best is None or cost < best[2]:
                best = released_part, resident_part, float(cost)
    return best


def _strong_marker(raw: np.ndarray, shared: np.ndarray,
                   center: tuple[float, float], search_radius: float,
                   sigma: float) -> tuple[int, int] | None:
    yy, xx = np.nonzero(shared)
    if not len(xx):
        return None
    within = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2
              <= search_radius ** 2)
    yy, xx = yy[within], xx[within]
    if not len(xx):
        return None
    smooth = ndi.gaussian_filter(raw.astype(np.float32), sigma)
    best = int(np.argmax(smooth[yy, xx]))
    return int(yy[best]), int(xx[best])


def _valley_ratio(frame: np.ndarray, left: tuple[int, int],
                  right: tuple[int, int]) -> float:
    ly, lx = left
    ry, rx = right
    distance = float(np.hypot(lx - rx, ly - ry))
    samples = max(3, int(np.ceil(distance)) + 1)
    values = []
    for weight in np.linspace(0.0, 1.0, samples):
        x = int(round((1.0 - weight) * lx + weight * rx))
        y = int(round((1.0 - weight) * ly + weight * ry))
        values.append(float(frame[y, x]))
    endpoint = max(min(values[0], values[-1]), 1.0)
    return float(np.clip(min(values[1:-1]) / endpoint, 0.0, 2.0))


def recover_duplicate_owner_release(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Discover and atomically backfill every qualifying early release."""
    assert_target_free(params)
    if labels.shape != raw.shape:
        raise ValueError("labels and raw must have identical TYX shape")
    candidate = labels.copy()
    frames = len(labels)
    start_limit = max(1, int(np.ceil(
        frames * float(params.get("maximum_birth_frame_fraction", 0.08)))))
    stabilization_limit = max(start_limit, int(np.ceil(
        frames * float(params.get(
            "maximum_stabilization_frame_fraction", 0.12)))))
    minimum_persistence = float(
        params.get("minimum_post_birth_presence_fraction", 0.5))
    maximum_owner_search = float(
        params.get("maximum_owner_search_radii", 1.5))
    maximum_birth_match = float(
        params.get("maximum_birth_match_radii", 1.5))
    maximum_pair_cost = float(
        params.get("maximum_pair_assignment_cost", 2.0))
    marker_fraction = float(
        params.get("marker_search_separation_fraction", 0.35))
    minimum_marker_radius = float(
        params.get("minimum_marker_search_radius_px", 2.0))
    maximum_valley = float(params.get("maximum_raw_valley_ratio", 0.8))
    minimum_area_ratio = float(
        params.get("minimum_partition_area_ratio", 0.25))
    maximum_area_ratio = float(
        params.get("maximum_partition_area_ratio", 3.0))
    minimum_combined = float(
        params.get("minimum_shared_to_combined_area_ratio", 0.5))
    sigma = float(params.get("raw_peak_sigma_px", 1.0))

    active = sorted(set(map(int, np.unique(labels))) - {0})
    proposals: list[dict] = []
    audit: list[dict] = []
    for newcomer in active:
        presence = np.any(labels == newcomer, axis=(1, 2))
        first = int(np.flatnonzero(presence)[0])
        base = {
            "newcomer_identity": newcomer,
            "birth_frame": first,
            "incumbent_identity": 0,
            "eligible": False,
            "applied": False,
            "changed_pixels": 0,
            "changed_frames": 0,
            "separate_component_frames": 0,
            "shared_component_frames": 0,
            "bracketed_duplicate_frames": 0,
            "maximum_raw_valley_ratio": np.nan,
            "reason": "",
        }
        if first == 0 or first > start_limit:
            audit.append({**base, "reason": "outside_start_relative_birth_band"})
            continue
        post_fraction = float(np.count_nonzero(presence[first:]) /
                              max(1, frames - first))
        if post_fraction < minimum_persistence:
            audit.append({
                **base, "post_birth_presence_fraction": post_fraction,
                "reason": "insufficient_later_residence"})
            continue
        newborn_parts = _components(labels[first] == newcomer)
        if len(newborn_parts) != 1:
            audit.append({
                **base, "post_birth_presence_fraction": post_fraction,
                "reason": "newcomer_birth_not_single_component"})
            continue
        newborn = newborn_parts[0]
        newcomer_area = _median_area(labels, newcomer, first)
        newcomer_radius = _radius(newcomer_area)
        incumbent = _nearest_foreground_owner(
            labels[first - 1], newborn["center"],
            maximum_owner_search * newcomer_radius)
        detail = {**base, "incumbent_identity": incumbent,
                  "post_birth_presence_fraction": post_fraction}
        if incumbent <= 0 or incumbent == newcomer:
            audit.append({**detail, "reason": "no_prior_owner_near_birth"})
            continue
        birth_residents = _components(labels[first] == incumbent)
        prior_parts = _components(labels[first - 1] == incumbent)
        if len(birth_residents) != 1:
            audit.append({
                **detail, "reason": "incumbent_birth_not_single_component"})
            continue
        if len(prior_parts) != 2:
            audit.append({
                **detail, "reason": "prior_owner_not_exactly_two_components"})
            continue
        resident = birth_residents[0]
        incumbent_area = _median_area(labels, incumbent, first)
        scale = newcomer_radius + _radius(incumbent_area)
        pair = _best_distinct_pair(
            prior_parts, newborn["center"], resident["center"], scale)
        if pair is None:
            audit.append({**detail, "reason": "two_seat_assignment_unavailable"})
            continue
        released_part, resident_part, pair_cost = pair
        birth_match = _distance(
            released_part["center"], newborn["center"]) / max(scale, 1.0)
        if birth_match > maximum_birth_match or pair_cost > maximum_pair_cost:
            audit.append({
                **detail, "birth_match_radii": birth_match,
                "pair_assignment_cost": pair_cost,
                "reason": "birth_seats_not_continuous"})
            continue

        released_center = newborn["center"]
        resident_center = resident["center"]
        changes: list[tuple[int, np.ndarray, float, str]] = []
        refusal = ""
        for frame in range(first - 1, -1, -1):
            parts = _components(labels[frame] == incumbent)
            if len(parts) == 2:
                pair = _best_distinct_pair(
                    parts, released_center, resident_center, scale)
                if pair is None or pair[2] > maximum_pair_cost:
                    refusal = "separate_seats_not_continuous"
                    break
                released_part, resident_part, _ = pair
                change = released_part["mask"]
                released_center = released_part["center"]
                resident_center = resident_part["center"]
                changes.append((frame, change, np.nan, "separate"))
                continue
            if len(parts) != 1:
                refusal = "prior_owner_component_count_out_of_range"
                break
            if not any(item[3] == "separate" for item in changes):
                refusal = "shared_component_precedes_no_separate_seats"
                break
            shared = parts[0]["mask"]
            separation = _distance(released_center, resident_center)
            search_radius = max(minimum_marker_radius,
                                marker_fraction * separation)
            released_marker = _strong_marker(
                raw[frame], shared, released_center, search_radius, sigma)
            resident_marker = _strong_marker(
                raw[frame], shared, resident_center, search_radius, sigma)
            if (released_marker is None or resident_marker is None
                    or released_marker == resident_marker):
                refusal = "distinct_raw_peak_markers_unavailable"
                break
            valley = _valley_ratio(raw[frame], released_marker, resident_marker)
            if valley > maximum_valley:
                refusal = "raw_peaks_not_separable"
                break
            markers = np.zeros(shared.shape, np.int16)
            markers[released_marker] = 1
            markers[resident_marker] = 2
            elevation = -ndi.gaussian_filter(
                raw[frame].astype(np.float32), sigma)
            partition = watershed(
                elevation, markers=markers, mask=shared,
                connectivity=STRUCTURE)
            change = partition == 1
            remaining = partition == 2
            if (ndi.label(change, STRUCTURE)[1] != 1
                    or ndi.label(remaining, STRUCTURE)[1] != 1):
                refusal = "raw_partition_not_two_connected_seats"
                break
            new_ratio = float(
                np.count_nonzero(change) / max(newcomer_area, 1.0))
            old_ratio = float(
                np.count_nonzero(remaining) / max(incumbent_area, 1.0))
            combined_ratio = float(
                np.count_nonzero(shared)
                / max(newcomer_area + incumbent_area, 1.0))
            if not (minimum_area_ratio <= new_ratio <= maximum_area_ratio
                    and minimum_area_ratio <= old_ratio <= maximum_area_ratio
                    and combined_ratio >= minimum_combined):
                refusal = "raw_partition_area_out_of_range"
                break
            released_center = _centroid(change)
            resident_center = _centroid(remaining)
            changes.append((frame, change, valley, "shared"))

        separate_frames = sum(item[3] == "separate" for item in changes)
        shared_frames = sum(item[3] == "shared" for item in changes)
        if (refusal or len(changes) != first or not separate_frames
                or not shared_frames):
            audit.append({
                **detail, "eligible": True,
                "birth_match_radii": birth_match,
                "pair_assignment_cost": pair_cost,
                "separate_component_frames": separate_frames,
                "shared_component_frames": shared_frames,
                "reason": refusal or "incomplete_boundary_interaction",
            })
            continue

        # A duplicate-owner failure may recur shortly after the stable birth.
        # Include only complete gaps bracketed by the newcomer's own components.
        upper = min(frames - 1, stabilization_limit)
        for frame in range(first + 1, upper):
            if presence[frame]:
                continue
            before_frames = np.flatnonzero(presence[:frame])
            after_frames = np.flatnonzero(presence[frame + 1:upper + 1])
            if not len(before_frames) or not len(after_frames):
                continue
            before_frame = int(before_frames[-1])
            after_frame = int(frame + 1 + after_frames[0])
            before_new = _components(labels[before_frame] == newcomer)
            after_new = _components(labels[after_frame] == newcomer)
            before_old = _components(labels[before_frame] == incumbent)
            after_old = _components(labels[after_frame] == incumbent)
            if not (len(before_new) == len(after_new) == 1
                    and len(before_old) == len(after_old) == 1):
                continue
            weight = ((frame - before_frame)
                      / max(after_frame - before_frame, 1))
            expected_new = tuple(
                (1.0 - weight) * before_new[0]["center"][axis]
                + weight * after_new[0]["center"][axis]
                for axis in (0, 1))
            expected_old = tuple(
                (1.0 - weight) * before_old[0]["center"][axis]
                + weight * after_old[0]["center"][axis]
                for axis in (0, 1))
            owner = _nearest_foreground_owner(
                labels[frame], expected_new,
                maximum_owner_search * newcomer_radius)
            if owner != incumbent:
                continue
            parts = _components(labels[frame] == incumbent)
            if len(parts) != 2:
                refusal = "bracketed_duplicate_not_exactly_two_components"
                break
            pair = _best_distinct_pair(
                parts, expected_new, expected_old, scale)
            if pair is None or pair[2] > maximum_pair_cost:
                refusal = "bracketed_duplicate_seats_not_continuous"
                break
            released_part, _, _ = pair
            changes.append((frame, released_part["mask"], np.nan,
                            "bracketed_duplicate"))
        if refusal:
            audit.append({
                **detail, "eligible": True,
                "birth_match_radii": birth_match,
                "pair_assignment_cost": pair_cost,
                "separate_component_frames": separate_frames,
                "shared_component_frames": shared_frames,
                "bracketed_duplicate_frames": sum(
                    item[3] == "bracketed_duplicate" for item in changes),
                "reason": refusal,
            })
            continue
        proposals.append({
            "newcomer": newcomer,
            "incumbent": incumbent,
            "changes": changes,
            "detail": detail,
            "birth_match": birth_match,
            "pair_cost": pair_cost,
        })

    occupied = np.zeros_like(labels, dtype=bool)
    for proposal in proposals:
        overlap = any(np.any(occupied[frame] & change)
                      for frame, change, _, _ in proposal["changes"])
        if overlap:
            audit.append({
                **proposal["detail"], "eligible": True,
                "reason": "proposal_overlaps_another_complete_cohort",
            })
            continue
        valleys = [value for _, _, value, kind in proposal["changes"]
                   if kind == "shared"]
        for frame, change, _, _ in proposal["changes"]:
            candidate[frame][change] = proposal["newcomer"]
            occupied[frame] |= change
        audit.append({
            **proposal["detail"],
            "eligible": True,
            "applied": True,
            "birth_match_radii": proposal["birth_match"],
            "pair_assignment_cost": proposal["pair_cost"],
            "changed_pixels": int(sum(
                np.count_nonzero(item[1]) for item in proposal["changes"])),
            "changed_frames": len(proposal["changes"]),
            "separate_component_frames": sum(
                item[3] == "separate" for item in proposal["changes"]),
            "shared_component_frames": sum(
                item[3] == "shared" for item in proposal["changes"]),
            "bracketed_duplicate_frames": sum(
                item[3] == "bracketed_duplicate"
                for item in proposal["changes"]),
            "maximum_raw_valley_ratio": max(valleys),
            "reason": "applied",
        })
    return candidate, pd.DataFrame(audit).sort_values(
        ["birth_frame", "newcomer_identity"]).reset_index(drop=True)


def excess_components(labels: np.ndarray) -> int:
    """Return disconnected components beyond one per identity and frame."""
    total = 0
    for frame in labels:
        for identity in set(map(int, np.unique(frame))) - {0}:
            total += max(0, ndi.label(frame == identity, STRUCTURE)[1] - 1)
    return int(total)


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray,
              audit: pd.DataFrame, params: dict) -> dict:
    """Measure production guardrails and target-free discovery counts."""
    changed = candidate != labels
    old_ids = set(map(int, np.unique(labels))) - {0}
    new_ids = set(map(int, np.unique(candidate))) - {0}
    eligible = audit[audit["eligible"].astype(bool)] if len(audit) else audit
    applied = eligible[eligible["applied"].astype(bool)] if len(eligible) else eligible
    return {
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "review_case_targets_received": False,
        "audited_identities": len(old_ids),
        "eligible_cohorts": int(len(eligible)),
        "applied_cohorts": int(len(applied)),
        "rejected_application_cohorts": int(len(eligible) - len(applied)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "unclaimed_overlap_pixels": int(np.count_nonzero(
            changed & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(
            changed & (labels == 0) & (raw <= 0))),
        "old_identity_set_preserved": old_ids == new_ids,
        "active_identities": len(new_ids),
        "new_same_frame_identity_components": max(
            0, excess_components(candidate) - excess_components(labels)),
        "parameters": params,
    }
