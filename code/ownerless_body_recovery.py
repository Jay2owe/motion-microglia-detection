"""Recover raw-supported cores for persistent physical tracks with no owner.

Discovery is field-wide. The producer accepts arrays and mathematical
thresholds only; identities, tracks, frames, coordinates, events and review
cases are forbidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from raw_physical_hypotheses import evidence_image


FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals",
}


def _disk_coverage(frame: np.ndarray, x: float, y: float,
                   radius: float) -> tuple[int, float]:
    height, width = frame.shape
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(width, int(np.ceil(x + radius + 1)))
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(height, int(np.ceil(y + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    values = frame[y0:y1, x0:x1][disk]
    positive = values[values > 0]
    if not len(positive):
        return 0, 0.0
    owners, counts = np.unique(positive, return_counts=True)
    return int(owners[int(np.argmax(counts))]), float(len(positive) / len(values))


def attach_coverage(points: pd.DataFrame, labels: np.ndarray,
                    unclaimed: np.ndarray) -> pd.DataFrame:
    attached = points.copy()
    assigned_owner: list[int] = []
    assigned_coverage: list[float] = []
    unclaimed_owner: list[int] = []
    unclaimed_coverage: list[float] = []
    for row in attached.itertuples(index=False):
        frame = int(row.frame)
        assigned = _disk_coverage(
            labels[frame], float(row.x), float(row.y), float(row.radius_px))
        ledger = _disk_coverage(
            unclaimed[frame], float(row.x), float(row.y), float(row.radius_px))
        assigned_owner.append(assigned[0])
        assigned_coverage.append(assigned[1])
        unclaimed_owner.append(ledger[0])
        unclaimed_coverage.append(ledger[1])
    attached["assigned_owner"] = assigned_owner
    attached["assigned_coverage_fraction"] = assigned_coverage
    attached["unclaimed_owner"] = unclaimed_owner
    attached["unclaimed_coverage_fraction"] = unclaimed_coverage
    attached["physically_visible"] = attached["state"].isin(
        ["observed", "latent_visible"])
    return attached


def discover_tracks(attached: pd.DataFrame, frame_count: int,
                    params: dict) -> pd.DataFrame:
    minimum_observed_fraction = float(
        params.get("minimum_observed_fraction_of_movie", 0.60))
    minimum_visible_fraction = float(
        params.get("minimum_visible_fraction_of_movie", 0.65))
    minimum_strong = int(params.get("minimum_strong_observations", 2))
    minimum_separation = float(
        params.get("minimum_median_separation_radii", 4.0))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.0))
    maximum_coverage = float(
        params.get("maximum_existing_coverage_fraction", 0.0))
    rows: list[dict] = []
    for track_id, group in attached.groupby("track_id", sort=True):
        visible = group[group["physically_visible"].astype(bool)].copy()
        observed = visible[visible["state"] == "observed"]
        observed_fraction = len(observed) / max(frame_count, 1)
        visible_fraction = len(visible) / max(frame_count, 1)
        strong_observations = int(observed["strong"].astype(bool).sum())
        radii = np.maximum(visible["radius_px"].to_numpy(float), 1.0)
        separation_radii = (
            visible["nearest_track_distance_px"].to_numpy(float) / radii)
        median_separation = float(np.median(separation_radii)) \
            if len(separation_radii) else 0.0
        encounter_fraction = float(visible["in_encounter"].astype(bool).mean()) \
            if len(visible) else 1.0
        assigned_max = float(visible["assigned_coverage_fraction"].max()) \
            if len(visible) else 1.0
        unclaimed_max = float(visible["unclaimed_coverage_fraction"].max()) \
            if len(visible) else 1.0
        reasons: list[str] = []
        if observed_fraction < minimum_observed_fraction:
            reasons.append("too_few_direct_observations")
        if visible_fraction < minimum_visible_fraction:
            reasons.append("visible_lifespan_too_short")
        if strong_observations < minimum_strong:
            reasons.append("too_few_strong_observations")
        if median_separation < minimum_separation:
            reasons.append("insufficient_isolation")
        if encounter_fraction > maximum_encounter:
            reasons.append("encountered_other_body")
        if assigned_max > maximum_coverage:
            reasons.append("has_assigned_mask_coverage")
        if unclaimed_max > maximum_coverage:
            reasons.append("has_unclaimed_mask_coverage")
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "track_id": int(track_id),
            "first_visible_frame": int(visible["frame"].min()) if len(visible) else -1,
            "last_visible_frame": int(visible["frame"].max()) if len(visible) else -1,
            "observed_frames": int(len(observed)),
            "visible_frames": int(len(visible)),
            "observed_fraction_of_movie": float(observed_fraction),
            "visible_fraction_of_movie": float(visible_fraction),
            "strong_observations": strong_observations,
            "median_separation_radii": median_separation,
            "encounter_fraction": encounter_fraction,
            "maximum_assigned_coverage": assigned_max,
            "maximum_unclaimed_coverage": unclaimed_max,
            "median_x": float(visible["x"].median()) if len(visible) else np.nan,
            "median_y": float(visible["y"].median()) if len(visible) else np.nan,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows)


def _connected_core(frame: np.ndarray, point, weak_threshold: float,
                    params: dict) -> tuple[np.ndarray, tuple[slice, slice], dict]:
    evidence = evidence_image(frame, params)
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    half_window = int(params.get("core_window_radius_px", 7))
    y0, y1 = max(0, y - half_window), min(frame.shape[0], y + half_window + 1)
    x0, x1 = max(0, x - half_window), min(frame.shape[1], x + half_window + 1)
    peak = float(evidence[y, x])
    level = max(
        float(params.get("core_peak_fraction", 0.35)) * peak,
        float(params.get("weak_threshold_fraction", 0.5)) * weak_threshold)
    local = ((evidence[y0:y1, x0:x1] >= level)
             & (frame[y0:y1, x0:x1] > 0))
    components, _ = ndi.label(local)
    component_id = int(components[y - y0, x - x0])
    core = components == component_id if component_id else np.zeros_like(local)
    details = {"peak_evidence": peak, "core_threshold": level,
               "core_area_px": int(core.sum())}
    return core, (slice(y0, y1), slice(x0, x1)), details


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame,
            params: dict) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    supplied = sorted(
        name for name in FORBIDDEN_TARGETS
        if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide ownerless-body recovery received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("ownerless-body recovery must use field-wide discovery")
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")

    attached = attach_coverage(points, labels, unclaimed)
    proposals = discover_tracks(attached, len(labels), params)
    candidate = labels.copy()
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.0))
    next_identity = int(candidate.max()) + 1
    audit_rows: list[dict] = [{
        **proposal._asdict(), "assigned_identity": 0,
        "outcome": "rejected_discovery", "changed_pixels": 0,
        "changed_frames": 0, "rejected_overlap_frames": 0,
        "rejected_core_frames": 0, "zero_signal_additions": 0,
    } for proposal in proposals[
        proposals["discovery_status"] != "eligible"].itertuples(index=False)]
    eligible = proposals[proposals["discovery_status"] == "eligible"].sort_values(
        ["first_visible_frame", "median_y", "median_x", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        identity = next_identity
        next_identity += 1
        changed_pixels = 0
        changed_frames = 0
        rejected_overlap_frames = 0
        rejected_core_frames = 0
        zero_signal_additions = 0
        group = attached[(attached["track_id"] == int(proposal.track_id))
                         & (attached["state"] == "observed")].sort_values("frame")
        for point in group.itertuples(index=False):
            frame_index = int(point.frame)
            core, region, details = _connected_core(
                raw[frame_index], point,
                float(threshold_by_frame.loc[frame_index, "weak_threshold"]),
                params)
            expected_area = np.pi * max(float(point.radius_px), 1.0) ** 2
            area_ratio = details["core_area_px"] / max(expected_area, 1.0)
            if not details["core_area_px"] or area_ratio > maximum_ratio:
                rejected_core_frames += 1
                continue
            occupied = ((candidate[frame_index][region] > 0)
                        | (unclaimed[frame_index][region] > 0))
            if np.any(core & occupied):
                rejected_overlap_frames += 1
                continue
            candidate[frame_index][region][core] = identity
            additions = int(core.sum())
            changed_pixels += additions
            changed_frames += 1
            zero_signal_additions += int(np.count_nonzero(
                core & (raw[frame_index][region] == 0)))
        audit_rows.append({
            **proposal._asdict(), "assigned_identity": identity,
            "outcome": "applied" if changed_frames else "rejected_application",
            "changed_pixels": changed_pixels, "changed_frames": changed_frames,
            "rejected_overlap_frames": rejected_overlap_frames,
            "rejected_core_frames": rejected_core_frames,
            "zero_signal_additions": zero_signal_additions,
        })
    audit = pd.DataFrame(audit_rows)
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError("ownerless-body recovery changed an existing label pixel")
    if np.any((unclaimed > 0) & (candidate > 0)):
        raise AssertionError("ownerless-body recovery overlapped the unclaimed ledger")
    if np.any((candidate > 0) & (labels == 0) & (raw == 0)):
        raise AssertionError("ownerless-body recovery added a zero-signal pixel")
    return candidate, audit, attached
