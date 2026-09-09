"""Calibrate recurrent same-owner branch swarms as non-disruptive.

The rule audits every event from current-run tables. It accepts no identity,
owner, track, frame, event, coordinate, region or review-case selector.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import separable_merge_recovery as physical


COMPATIBLE_SOURCE_FAMILIES = {
    "separable_cells_merged", "single_cell_multi_core",
}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate_target", "event_target", "region_target",
    "review_case_target", "forced_identity", "forced_interval",
    "include_track", "exclude_track",
)
STRUCTURE = np.ones((3, 3), np.uint8)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "branch-swarm calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("branch-swarm calibration must be field-wide")


def _numbers(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def event_summary(events: pd.DataFrame) -> dict:
    owner = events[events.family.eq("identity_swap_or_takeover")]
    return {
        "events": int(len(events)),
        "owner_change_events": int(len(owner)),
        "owner_change_disruption_burden": float(
            owner.disruption_score.astype(float).sum()),
        "high_disruption_events": int(
            (events.disruption_score.astype(float) >= 50).sum()),
        "total_disruption_burden": float(
            events.disruption_score.astype(float).sum()),
        "events_by_family": {str(key): int(value) for key, value in
                             events.groupby("family").size().items()},
    }


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              labels: np.ndarray, points: pd.DataFrame, params: dict,
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Remove only completely evidenced recurrent one-owner branch swarms."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    frame_count = int(len(labels))
    minimum_duration = int(np.ceil(frame_count * float(
        params.get("minimum_event_movie_fraction", 0.25))))
    minimum_tracks = int(params.get("minimum_physical_tracks", 4))
    minimum_constituents = int(params.get("minimum_constituent_events", 4))
    minimum_owner_movie = float(params.get(
        "minimum_owner_movie_fraction", 0.90))
    minimum_largest_component = float(params.get(
        "minimum_largest_component_fraction", 0.95))
    minimum_compatible_owner = float(params.get(
        "minimum_compatible_track_owner_fraction", 1.0))
    maximum_distance = float(params.get(
        "maximum_track_distance_owner_radii", 2.0))
    minimum_near = float(params.get("minimum_near_owner_fraction", 0.95))
    owner_presence = {
        int(owner): np.any(labels == int(owner), axis=(1, 2))
        for owner in set(map(int, np.unique(labels))) - {0}}
    audit_rows: list[dict] = []
    removed: set[str] = set()
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        tracks = _numbers(event.physical_tracks)
        before, after = (_numbers(event.accepted_before),
                         _numbers(event.accepted_after))
        owners = before | after
        owner = next(iter(owners)) if len(owners) == 1 else 0
        first, last = int(event.first_frame), int(event.last_frame)
        duration = last - first + 1
        source_families = set(event_members.source_family.astype(str))
        owner_movie_fraction = float(owner_presence.get(
            owner, np.zeros(frame_count, bool)).mean()) if owner else 0.0
        largest_component_fractions: list[float] = []
        if owner:
            for frame in range(first, last + 1):
                mask = labels[frame] == owner
                components, count = ndi.label(mask, STRUCTURE)
                sizes = np.bincount(components.ravel())
                largest_component_fractions.append(
                    float(sizes[1:].max() / max(int(mask.sum()), 1))
                    if int(count) else 0.0)
        minimum_component_fraction = (
            float(min(largest_component_fractions))
            if largest_component_fractions else 0.0)
        event_points = scored[
            scored.track_id.astype(int).isin(tracks)
            & scored.frame.astype(int).between(first, last)]
        compatible_owner_fraction = float(
            event_points.accepted_owner.astype(int).isin({0, owner}).mean()) \
            if len(event_points) and owner else 0.0
        near: list[bool] = []
        distance_cache: dict[int, np.ndarray] = {}
        if owner:
            for point in event_points.itertuples(index=False):
                frame = int(point.frame)
                if frame not in distance_cache:
                    distance_cache[frame] = ndi.distance_transform_edt(
                        labels[frame] != owner)
                y = int(np.clip(round(float(point.y)), 0,
                                labels.shape[1] - 1))
                x = int(np.clip(round(float(point.x)), 0,
                                labels.shape[2] - 1))
                distance = float(distance_cache[frame][y, x]) / max(
                    float(point.radius_px), 1.0)
                near.append(distance <= maximum_distance)
        near_fraction = float(np.mean(near)) if near else 0.0
        reasons: list[str] = []
        if str(event.family) != "separable_cells_merged":
            reasons.append("not_merge_family")
        if duration < minimum_duration:
            reasons.append("event_too_short")
        if len(tracks) < minimum_tracks:
            reasons.append("too_few_physical_tracks")
        if int(event.constituent_events) < minimum_constituents:
            reasons.append("too_few_constituent_events")
        if not owner or before != after:
            reasons.append("accepted_owner_not_conserved")
        if (not source_families
                or not source_families <= COMPATIBLE_SOURCE_FAMILIES):
            reasons.append("incompatible_source_family")
        if owner_movie_fraction < minimum_owner_movie:
            reasons.append("owner_not_movie_long")
        if minimum_component_fraction < minimum_largest_component:
            reasons.append("owner_not_one_dominant_component")
        if compatible_owner_fraction < minimum_compatible_owner:
            reasons.append("foreign_owner_on_event_tracks")
        if near_fraction < minimum_near:
            reasons.append("event_tracks_not_owner_proximal")
        eligible = not reasons
        if eligible:
            removed.add(event_id)
        audit_rows.append({
            "event_id": event_id, "family": str(event.family),
            "first_frame": first, "last_frame": last,
            "duration_frames": duration,
            "physical_track_count": len(tracks),
            "constituent_events": int(event.constituent_events),
            "measured_owner": owner,
            "source_families": "|".join(sorted(source_families)),
            "owner_movie_fraction": owner_movie_fraction,
            "minimum_largest_component_fraction": minimum_component_fraction,
            "compatible_track_owner_fraction": compatible_owner_fraction,
            "near_owner_fraction": near_fraction,
            "eligible": eligible,
            "reason": ("recurrent_same_owner_branch_swarm" if eligible
                       else "|".join(reasons)),
        })
    calibrated = events[
        ~events.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    calibrated = calibrated.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated["impact_rank"] = np.arange(1, len(calibrated) + 1)
    calibrated_members = members[
        ~members.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(len(events)),
        "events_calibrated_nonbiological": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events),
        "after": event_summary(calibrated),
    }
    return calibrated, calibrated_members, pd.DataFrame(audit_rows), summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis adapter using explicit current-run paths."""
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    points = pd.read_csv(params["points_path"])
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, labels, points, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "recurrent_branch_swarm_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["events"], index=False)
    calibrated_members.to_csv(outputs["members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    summary["label_bytes_unchanged"] = (
        outputs["labels"].read_bytes() == labels_path.read_bytes())
    summary["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
