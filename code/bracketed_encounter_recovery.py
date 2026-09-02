"""Recover orphaned identity seats during bracketed physical encounters.

Discovery is field-wide. The producer accepts mathematical thresholds and
complete physical-track evidence only; review identities, tracks, frames,
events, and regions are forbidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import separable_merge_recovery as accepted_merge
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}


def _inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _component_excess(frame: np.ndarray, identity: int) -> int:
    return component_accounting.component_excess(frame, identity)


def _new_duplicate_components(baseline: np.ndarray,
                              candidate: np.ndarray) -> int:
    return component_accounting.new_duplicate_components(
        baseline, candidate)


def _recent_encounter(group: pd.DataFrame, start: int, lookback: int) -> bool:
    recent = group[
        group.frame.between(start - lookback, start - 1)
        & group.in_encounter.astype(bool)]
    return bool(len(recent))


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Discover and repair every safe bracketed owner excursion in the field."""
    supplied = sorted(name for name in FORBIDDEN_TARGETS
                      if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide bracketed recovery received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("bracketed recovery must use field-wide discovery")
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")

    scored = accepted_merge.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    canonical = accepted_merge.canonical_owners(
        scored,
        int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_canonical_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, bool)
    audits: list[dict] = []
    minimum_pre = int(params.get("minimum_pre_owner_run_frames", 5))
    maximum_middle = max(1, int(np.ceil(
        len(labels) * float(params.get("maximum_middle_run_fraction", 0.11)))))
    lookback = int(params.get("encounter_lookback_frames", 7))
    maximum_distance = float(params.get("maximum_companion_distance_px", 30.0))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.65))
    maximum_orphan_ratio = float(
        params.get("maximum_orphan_to_recovered_area_ratio", 1.0))
    sigma = float(params.get("watershed_sigma_px", 1.0))
    minimum_changed = int(params.get("minimum_changed_pixels", 5))
    orphan_action = str(params.get("orphan_action", "transfer_to_host"))
    if orphan_action not in {"transfer_to_host", "demote_to_unclaimed"}:
        raise ValueError(f"unknown orphan action: {orphan_action}")
    partition_mode = str(params.get("partition_mode", "intensity"))
    if partition_mode not in {"intensity", "geodesic"}:
        raise ValueError(f"unknown partition mode: {partition_mode}")
    next_unclaimed = int(np.max(unclaimed)) + 1

    for track_id, group in visible.groupby("track_id"):
        track_id = int(track_id)
        runs = accepted_merge._owner_runs(group)
        for run_index in range(1, len(runs) - 1):
            before, middle, after = runs[run_index - 1:run_index + 2]
            base = {
                "track_id": track_id,
                "owner_before": int(before["owner"]),
                "owner_middle": int(middle["owner"]),
                "owner_after": int(after["owner"]),
                "first_frame": int(middle["start"]),
                "last_frame": int(middle["end"]),
                "middle_frames": int(len(middle["frames"])),
                "applied": False,
                "changed_pixels": 0,
                "reason": "",
            }
            if (before["owner"] <= 0 or before["owner"] != after["owner"]
                    or before["owner"] == middle["owner"]):
                audits.append({**base, "reason": "not_positive_bracketed_excursion"})
                continue
            if (len(before["frames"]) < minimum_pre
                    or len(middle["frames"]) > maximum_middle):
                audits.append({**base, "reason": "owner_run_duration_out_of_range"})
                continue
            if not _recent_encounter(group, int(middle["start"]), lookback):
                audits.append({**base, "reason": "no_recent_physical_encounter"})
                continue

            old_owner = int(before["owner"])
            host_owner = int(middle["owner"])
            prepared: list[tuple[int, np.ndarray, np.ndarray, int, float, int]] = []
            refusal = ""
            for frame in middle["frames"]:
                frame = int(frame)
                point_rows = group[group.frame == frame]
                if not len(point_rows):
                    refusal = "missing_excursion_track_point"
                    break
                resident = point_rows.iloc[0]
                old_mask = candidate[frame] == old_owner
                if not np.any(old_mask):
                    refusal = "old_owner_absent_not_orphaned"
                    break
                old_components, _ = ndi.label(old_mask, STRUCTURE)
                stable_old_core = False
                for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                    stable = canonical.get(int(other.track_id))
                    if stable is None or int(stable["owner"]) != old_owner:
                        continue
                    if accepted_merge._component_at(
                            old_components, float(other.x), float(other.y)) > 0:
                        stable_old_core = True
                        break
                if stable_old_core:
                    refusal = "old_owner_component_has_stable_core"
                    break

                host_components, _ = ndi.label(
                    candidate[frame] == host_owner, STRUCTURE)
                shared_id = accepted_merge._component_at(
                    host_components, float(resident.x), float(resident.y))
                if shared_id <= 0:
                    refusal = "resident_outside_middle_owner_component"
                    break
                shared = host_components == shared_id
                companions = []
                for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                    other_track = int(other.track_id)
                    if other_track == track_id:
                        continue
                    stable = canonical.get(other_track)
                    if stable is None or int(stable["owner"]) != host_owner:
                        continue
                    if not _inside(shared, other):
                        continue
                    distance = float(np.hypot(
                        float(resident.x) - float(other.x),
                        float(resident.y) - float(other.y)))
                    if 4.0 <= distance <= maximum_distance:
                        companions.append((distance, other))
                if not companions:
                    refusal = "no_stable_host_core_in_shared_component"
                    break
                _, companion = min(
                    companions, key=lambda item: (item[0], int(item[1].track_id)))
                valley = accepted_merge._valley_ratio(
                    raw[frame],
                    (float(resident.x), float(resident.y)),
                    (float(companion.x), float(companion.y)))
                if valley > maximum_valley:
                    refusal = "raw_cores_not_separable"
                    break
                partition_mask = (
                    shared | old_mask if partition_mode == "geodesic" else shared)
                resident_marker = accepted_merge._marker_pixel(
                    partition_mask, float(resident.x), float(resident.y))
                host_marker = accepted_merge._marker_pixel(
                    partition_mask, float(companion.x), float(companion.y))
                if (resident_marker is None or host_marker is None
                        or resident_marker == host_marker):
                    refusal = "watershed_markers_unavailable"
                    break
                markers = np.zeros(labels.shape[1:], np.int16)
                markers[resident_marker] = 1
                markers[host_marker] = 2
                elevation = (
                    np.zeros_like(raw[frame], dtype=np.float32)
                    if partition_mode == "geodesic"
                    else -ndi.gaussian_filter(raw[frame].astype(np.float32), sigma))
                partition = watershed(
                    elevation, markers=markers, mask=partition_mask)
                recovered = partition == 1
                recovered_area = int(np.count_nonzero(recovered))
                orphan_area = int(np.count_nonzero(old_mask))
                if orphan_area > maximum_orphan_ratio * max(recovered_area, 1):
                    refusal = "orphan_component_too_large"
                    break
                trial = candidate[frame].copy()
                unclaimed_trial = candidate_unclaimed[frame].copy()
                if orphan_action == "transfer_to_host":
                    trial[old_mask] = host_owner
                else:
                    trial[old_mask] = 0
                    unclaimed_trial[old_mask] = next_unclaimed
                trial[recovered] = old_owner
                host_after, host_after_count = ndi.label(
                    trial == host_owner, STRUCTURE)
                retained_host_components: set[int] = set()
                for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                    stable = canonical.get(int(other.track_id))
                    if stable is None or int(stable["owner"]) != host_owner:
                        continue
                    component_id = accepted_merge._component_at(
                        host_after, float(other.x), float(other.y))
                    if component_id > 0:
                        retained_host_components.add(component_id)
                coreless_host = np.zeros_like(recovered)
                for component_id in range(1, host_after_count + 1):
                    if component_id not in retained_host_components:
                        coreless_host |= host_after == component_id
                if np.any(coreless_host):
                    trial[coreless_host] = old_owner
                if not np.any(trial == old_owner) or not np.any(trial == host_owner):
                    refusal = "identity_extinguished"
                    break
                if (_component_excess(trial, old_owner)
                        > _component_excess(labels[frame], old_owner)):
                    refusal = "new_old_owner_duplicate"
                    break
                if (_component_excess(trial, host_owner)
                        > _component_excess(labels[frame], host_owner)):
                    refusal = "new_host_owner_duplicate"
                    break
                if not np.array_equal(
                        (trial > 0) | (unclaimed_trial > 0),
                        (labels[frame] > 0) | (unclaimed[frame] > 0)):
                    refusal = "foreground_ledger_union_changed"
                    break
                change = trial != candidate[frame]
                if np.any(occupied[frame] & change):
                    refusal = "overlapping_proposal"
                    break
                changed = int(np.count_nonzero(change))
                if changed < minimum_changed:
                    refusal = "transfer_too_small"
                    break
                prepared.append(
                    (frame, trial, unclaimed_trial, changed, valley, orphan_area))

            if refusal:
                audits.append({**base, "reason": refusal})
                continue
            for frame, trial, unclaimed_trial, _, _, _ in prepared:
                changed = trial != candidate[frame]
                candidate[frame] = trial
                candidate_unclaimed[frame] = unclaimed_trial
                occupied[frame] |= changed
            audits.append({
                **base,
                "applied": True,
                "changed_pixels": int(sum(item[3] for item in prepared)),
                "changed_frames": len(prepared),
                "maximum_valley_ratio": float(max(item[4] for item in prepared)),
                "maximum_orphan_pixels": int(max(item[5] for item in prepared)),
                "orphan_action": orphan_action,
                "partition_mode": partition_mode,
                "assigned_unclaimed_id": (
                    next_unclaimed if orphan_action == "demote_to_unclaimed" else 0),
                "reason": "applied",
            })
            if orphan_action == "demote_to_unclaimed":
                next_unclaimed += 1

    if not np.array_equal(
            (candidate > 0) | (candidate_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("bracketed recovery changed foreground-ledger union")
    if np.any((unclaimed > 0) & (candidate_unclaimed != unclaimed)):
        raise AssertionError("bracketed recovery changed existing unclaimed pixels")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("bracketed recovery changed active identities")
    duplicates = _new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError(
            f"bracketed recovery created {duplicates} duplicate components")
    return candidate, candidate_unclaimed, pd.DataFrame(audits)
