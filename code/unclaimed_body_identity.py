"""Discover persistent unclaimed bodies and allocate fresh identities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


@dataclass
class UnclaimedBodyResult:
    labels: np.ndarray
    unclaimed: np.ndarray
    selected_domains: np.ndarray
    source_audit: pd.DataFrame
    frame_audit: pd.DataFrame
    actions: pd.DataFrame


def _track_single_body(domain: np.ndarray, minimum_area_px: int,
                       minimum_largest_fraction: float
                       ) -> tuple[np.ndarray, dict]:
    selected = np.zeros_like(domain, dtype=bool)
    previous_centroid = None
    selected_areas: list[int] = []
    centroid_steps: list[float] = []
    candidate_counts: list[int] = []
    structure = ndi.generate_binary_structure(2, 1)
    for frame in np.flatnonzero(np.any(domain, axis=(1, 2))):
        components, count = ndi.label(domain[frame], structure=structure)
        rows = []
        for component in range(1, count + 1):
            mask = components == component
            area = int(mask.sum())
            y, x = ndi.center_of_mass(mask)
            rows.append((component, area, float(y), float(x)))
        largest = max(row[1] for row in rows)
        threshold = max(int(minimum_area_px), int(np.ceil(
            float(minimum_largest_fraction) * largest)))
        substantial = [row for row in rows if row[1] >= threshold]
        if not substantial:
            raise AssertionError(f"no substantial unclaimed body at frame {frame}")
        if previous_centroid is None:
            chosen = max(substantial, key=lambda row: row[1])
        else:
            chosen = min(substantial, key=lambda row: (
                (row[2] - previous_centroid[0]) ** 2
                + (row[3] - previous_centroid[1]) ** 2,
                -row[1], row[0]))
            centroid_steps.append(float(np.hypot(
                chosen[2] - previous_centroid[0],
                chosen[3] - previous_centroid[1])))
        selected[frame] = components == chosen[0]
        selected_areas.append(chosen[1])
        candidate_counts.append(len(substantial))
        previous_centroid = (chosen[2], chosen[3])
    return selected, {
        "selected_frames": len(selected_areas),
        "selected_area_min": min(selected_areas) if selected_areas else 0,
        "selected_area_max": max(selected_areas) if selected_areas else 0,
        "maximum_centroid_step_px": max(centroid_steps) if centroid_steps else 0.0,
        "frames_with_multiple_substantial_candidates": int(sum(
            count > 1 for count in candidate_counts)),
    }


def discover_unclaimed_bodies(labels: np.ndarray, unclaimed: np.ndarray,
                              raw: np.ndarray, params: dict
                              ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed, and raw stacks must match")
    minimum_frames = max(2, int(np.ceil(
        float(params["minimum_duration_fraction"]) * len(labels))))
    events: list[dict] = []
    source_rows: list[dict] = []
    frame_rows: list[dict] = []
    for source_identity in sorted(set(map(int, np.unique(unclaimed))) - {0}):
        full_domain = unclaimed == source_identity
        frames = np.flatnonzero(np.any(full_domain, axis=(1, 2)))
        try:
            selected, track = _track_single_body(
                full_domain, int(params["minimum_area_px"]),
                float(params["minimum_largest_fraction"]))
        except AssertionError:
            source_rows.append({
                "source_identity": source_identity,
                "frames_present": int(len(frames)),
                "first_review_frame": int(frames[0] + 1),
                "last_review_frame": int(frames[-1] + 1),
                "temporal_gaps": int(np.count_nonzero(np.diff(frames) != 1)),
                "selected_pixels": 0, "minimum_selected_area_px": 0,
                "minimum_eroded_core_px": 0, "median_raw": 0.0,
                "assigned_overlap_pixels": 0,
                "maximum_centroid_step_px": 0.0,
                "frames_with_multiple_substantial_candidates": 0,
                "decision": "refused_no_substantial_component_in_every_frame",
            })
            continue
        cores: list[int] = []
        raw_medians: list[float] = []
        overlaps: list[int] = []
        for frame in frames:
            mask = selected[frame]
            core = int(ndi.binary_erosion(mask).sum())
            raw_median = float(np.median(raw[frame][mask]))
            overlap = int(np.count_nonzero(labels[frame][mask]))
            cores.append(core)
            raw_medians.append(raw_median)
            overlaps.append(overlap)
            frame_rows.append({
                "source_identity": source_identity,
                "review_frame": int(frame + 1),
                "source_area_px": int(full_domain[frame].sum()),
                "selected_area_px": int(mask.sum()),
                "eroded_core_px": core, "raw_median": raw_median,
                "assigned_overlap_px": overlap,
                "remnant_pixels": int(np.count_nonzero(
                    full_domain[frame] & ~mask)),
            })
        continuous = bool(len(frames) and np.all(np.diff(frames) == 1))
        if len(frames) < minimum_frames:
            decision = "refused_insufficient_duration"
        elif not continuous:
            decision = "refused_temporal_gaps"
        elif min(cores, default=0) < int(params["minimum_eroded_core_px"]):
            decision = "refused_insufficient_substantial_core"
        elif float(np.median(raw_medians)) <= float(params["minimum_median_raw"]):
            decision = "refused_insufficient_raw_support"
        elif sum(overlaps):
            decision = "refused_assigned_overlap"
        else:
            decision = "accepted_persistent_unclaimed_body"
            events.append({
                "source_identity": source_identity, "domain": selected,
                "first_review_frame": int(frames[0] + 1),
                "last_review_frame": int(frames[-1] + 1),
            })
        source_rows.append({
            "source_identity": source_identity,
            "frames_present": int(len(frames)),
            "first_review_frame": int(frames[0] + 1),
            "last_review_frame": int(frames[-1] + 1),
            "temporal_gaps": int(np.count_nonzero(np.diff(frames) != 1)),
            "selected_pixels": int(selected.sum()),
            "minimum_selected_area_px": track["selected_area_min"],
            "minimum_eroded_core_px": min(cores, default=0),
            "median_raw": float(np.median(raw_medians)),
            "assigned_overlap_pixels": int(sum(overlaps)),
            "maximum_centroid_step_px": track["maximum_centroid_step_px"],
            "frames_with_multiple_substantial_candidates": track[
                "frames_with_multiple_substantial_candidates"],
            "decision": decision,
        })
    return events, pd.DataFrame(source_rows), pd.DataFrame(frame_rows)


def assign_persistent_unclaimed_bodies(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        params: dict) -> UnclaimedBodyResult:
    """Assign fresh deterministic names to all eligible unclaimed body tracks."""
    events, source_audit, frame_audit = discover_unclaimed_bodies(
        labels, unclaimed, raw, params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    domains = np.zeros(labels.shape, np.uint16)
    actions: list[dict] = []
    next_identity = max(int(np.max(labels)), int(np.max(unclaimed))) + 1
    ordered = sorted(events, key=lambda event: (
        event["first_review_frame"], event["source_identity"]))
    for event_number, event in enumerate(ordered, start=1):
        domain = event["domain"]
        if np.any(candidate[domain]) or np.any(candidate_unclaimed[domain] == 0):
            raise AssertionError("selected unclaimed domain is not available")
        allocated = next_identity
        next_identity += 1
        candidate[domain] = allocated
        candidate_unclaimed[domain] = 0
        domains[domain] = event_number
        actions.append({
            "event_id": f"UB-{event_number:04d}",
            "source_identity": event["source_identity"],
            "allocated_identity": allocated,
            "first_review_frame": event["first_review_frame"],
            "last_review_frame": event["last_review_frame"],
            "assigned_pixels": int(domain.sum()),
            "assigned_frames": int(np.count_nonzero(
                np.any(domain, axis=(1, 2)))),
            "status": "assigned_persistent_unclaimed_body",
        })
    if not np.array_equal(
            (candidate > 0) | (candidate_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("unclaimed-body assignment changed foreground")
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    return UnclaimedBodyResult(
        labels=candidate, unclaimed=candidate_unclaimed,
        selected_domains=domains, source_audit=source_audit,
        frame_audit=frame_audit,
        actions=pd.DataFrame(actions, columns=[
            "event_id", "source_identity", "allocated_identity",
            "first_review_frame", "last_review_frame", "assigned_pixels",
            "assigned_frames", "status"]))
