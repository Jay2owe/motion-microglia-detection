"""Complete short internal gaps in field-discovered ownerless cohorts.

The accepted rule consumes only current-run cohort discovery, physical-track,
raw-image and threshold evidence.  It has no identity, interval, event or
coordinate selector; identities and intervals are discovered from the input.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import component_accounting
from ownerless_body_recovery import evidence_image


VISIBLE_LATENT_STATE = "latent_visible"
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate_target", "event_target", "region_target",
    "review_case_target", "forced_identity", "forced_interval",
    "include_track", "exclude_track",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "ownerless-cohort gap completion received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("ownerless-cohort gap completion must be field-wide")


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    if not len(values):
        return []
    result: list[tuple[int, int]] = []
    first = previous = int(values[0])
    for value in map(int, values[1:]):
        if value != previous + 1:
            result.append((first, previous))
            first = value
        previous = value
    result.append((first, previous))
    return result


def _nearest_raw_core(frame: np.ndarray, evidence: np.ndarray, point,
                      weak_threshold: float, params: dict,
                      ) -> tuple[np.ndarray, tuple[slice, slice], dict] | None:
    radius = max(float(point.radius_px), 1.0)
    maximum_offset = float(params.get("maximum_core_offset_radii", 4.5))
    half_window = max(
        int(params.get("core_window_radius_px", 7)),
        int(math.ceil(maximum_offset * radius)))
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    y0, y1 = max(0, y - half_window), min(frame.shape[0], y + half_window + 1)
    x0, x1 = max(0, x - half_window), min(frame.shape[1], x + half_window + 1)
    raw_local = frame[y0:y1, x0:x1]
    evidence_local = evidence[y0:y1, x0:x1]
    raw_positive = raw_local > 0
    if not np.any(raw_positive):
        return None
    local_peak = float(evidence_local[raw_positive].max())
    level = max(
        float(params.get("core_peak_fraction", 0.35)) * local_peak,
        float(params.get("weak_threshold_fraction", 0.5))
        * float(weak_threshold))
    components, count = ndi.label(
        (evidence_local >= level) & raw_positive,
        structure=np.ones((3, 3), np.uint8))
    maximum_area_ratio = float(params.get("maximum_core_area_radius_ratio", 2.0))
    choices: list[tuple[float, int, int, float]] = []
    for component_id in range(1, int(count) + 1):
        yy, xx = np.nonzero(components == component_id)
        if not len(xx):
            continue
        distance = float(np.sqrt(np.min(
            (yy + y0 - float(point.y)) ** 2
            + (xx + x0 - float(point.x)) ** 2)))
        area = int(len(xx))
        area_ratio = area / max(math.pi * radius ** 2, 1.0)
        if (distance <= maximum_offset * radius
                and area_ratio <= maximum_area_ratio):
            choices.append((distance, -area, component_id, area_ratio))
    if not choices:
        return None
    distance, negative_area, component_id, area_ratio = min(choices)
    core = components == int(component_id)
    return core, (slice(y0, y1), slice(x0, x1)), {
        "core_area_px": int(-negative_area),
        "core_area_radius_ratio": float(area_ratio),
        "core_offset_radii": float(distance / radius),
        "core_threshold": float(level),
        "local_peak_evidence": local_peak,
    }


def complete(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
             points: pd.DataFrame, cohorts: pd.DataFrame,
             thresholds: pd.DataFrame, params: dict,
             ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Return labels with fully supported short cohort gaps filled atomically."""
    assert_target_free(params)
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")
    required = {"track_id", "assigned_identity", "discovery_status"}
    if not required.issubset(cohorts.columns):
        raise ValueError("ownerless-cohort audit lacks discovery mapping")
    threshold_by_frame = thresholds.set_index("frame")
    maximum_gap = max(1, int(math.ceil(
        len(labels) * float(params.get("maximum_gap_movie_fraction", 0.04)))))
    minimum_separation = float(params.get("minimum_separation_radii", 4.0))
    eligible = cohorts[
        cohorts.assigned_identity.astype(int).gt(0)
        & cohorts.discovery_status.astype(str).eq("eligible")].copy()
    candidate = labels.copy()
    evidence_cache: dict[int, np.ndarray] = {}
    audit_rows: list[dict] = []
    for identity, mapping in eligible.groupby("assigned_identity", sort=True):
        identity = int(identity)
        member_tracks = set(mapping.track_id.astype(int))
        presence = np.any(labels == identity, axis=(1, 2))
        present_frames = np.flatnonzero(presence)
        if len(present_frames) < 2:
            continue
        first, last = int(present_frames[0]), int(present_frames[-1])
        missing = np.flatnonzero(
            ~presence & (np.arange(len(labels)) > first)
            & (np.arange(len(labels)) < last))
        for gap_first, gap_last in _runs(missing):
            gap_frames = gap_last - gap_first + 1
            proposal = {
                "proposal_id": f"P{len(audit_rows) + 1:04d}",
                "discovered_identity": identity,
                "member_tracks": "|".join(map(str, sorted(member_tracks))),
                "first_frame": gap_first, "last_frame": gap_last,
                "gap_frames": gap_frames, "applied": False,
                "changed_pixels": 0, "changed_frames": 0,
                "minimum_visibility": float("nan"),
                "minimum_separation_radii": float("nan"),
                "maximum_core_offset_radii": float("nan"),
                "maximum_core_area_radius_ratio": float("nan"),
                "reason": "",
            }
            if gap_frames > maximum_gap:
                proposal["reason"] = "gap_too_long"
                audit_rows.append(proposal)
                continue
            pending: list[tuple[int, tuple[slice, slice], np.ndarray, dict]] = []
            visibilities: list[float] = []
            separations: list[float] = []
            failure = ""
            for frame_index in range(gap_first, gap_last + 1):
                rows = points[
                    points.track_id.astype(int).isin(member_tracks)
                    & points.frame.astype(int).eq(frame_index)
                    & points.state.astype(str).eq(VISIBLE_LATENT_STATE)]
                if len(rows) != 1:
                    failure = "not_exactly_one_latent_visible_member"
                    break
                point = rows.iloc[0]
                separation = (float(point.nearest_track_distance_px)
                              / max(float(point.radius_px), 1.0))
                if bool(point.in_encounter) or separation < minimum_separation:
                    failure = "latent_member_not_isolated"
                    break
                if frame_index not in evidence_cache:
                    evidence_cache[frame_index] = evidence_image(
                        raw[frame_index], params)
                found = _nearest_raw_core(
                    raw[frame_index], evidence_cache[frame_index], point,
                    float(threshold_by_frame.loc[frame_index, "weak_threshold"]),
                    params)
                if found is None:
                    failure = "no_qualified_raw_core"
                    break
                core, region, details = found
                occupied = ((candidate[frame_index][region] > 0)
                            | (unclaimed[frame_index][region] > 0))
                if np.any(core & occupied):
                    failure = "qualified_core_overlaps_existing_ledger"
                    break
                pending.append((frame_index, region, core, details))
                visibilities.append(float(point.visibility))
                separations.append(separation)
            if failure or len(pending) != gap_frames:
                proposal["reason"] = "atomic_refusal:" + (
                    failure or "incomplete_gap_support")
                audit_rows.append(proposal)
                continue
            for frame_index, region, core, _ in pending:
                candidate[frame_index][region][core] = identity
            proposal.update({
                "applied": True,
                "changed_pixels": int(sum(item[2].sum() for item in pending)),
                "changed_frames": len(pending),
                "minimum_visibility": float(min(visibilities)),
                "minimum_separation_radii": float(min(separations)),
                "maximum_core_offset_radii": float(max(
                    item[3]["core_offset_radii"] for item in pending)),
                "maximum_core_area_radius_ratio": float(max(
                    item[3]["core_area_radius_ratio"] for item in pending)),
                "reason": "atomic_isolated_latent_cohort_gap_completion",
            })
            audit_rows.append(proposal)
    audit = pd.DataFrame(audit_rows)
    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    existing_changes = int(np.count_nonzero((labels > 0) & changed))
    overlap = int(np.count_nonzero(additions & (unclaimed > 0)))
    zero_signal = int(np.count_nonzero(additions & (raw == 0)))
    duplicates = component_accounting.new_duplicate_components(labels, candidate)
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if existing_changes:
        raise AssertionError("cohort gap completion changed assigned pixels")
    if overlap:
        raise AssertionError("cohort gap completion overlapped unclaimed pixels")
    if zero_signal:
        raise AssertionError("cohort gap completion added zero-signal pixels")
    if candidate_identities != baseline_identities:
        raise AssertionError("cohort gap completion changed identity set")
    if duplicates:
        raise AssertionError("cohort gap completion created duplicate components")
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "maximum_gap_frames": maximum_gap,
        "cohort_identities_discovered": int(eligible.assigned_identity.nunique()),
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
        "changed_identities": int(np.unique(candidate[changed]).size),
        "preexisting_assigned_changed_pixels": existing_changes,
        "unclaimed_overlap_pixels": overlap,
        "zero_signal_additions": zero_signal,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(duplicates),
    }
    return candidate, audit, summary
