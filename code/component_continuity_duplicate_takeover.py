"""Restore a resident when another established owner invades its continuous body.

Discovery is field-wide. The producer accepts only evidence paths and
dimensionless thresholds; named identities, tracks, frames, coordinates,
events, regions, and review cases are forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile

import component_accounting
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TOKENS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "forced_interval", "allowed_pair",
)
AUDIT_COLUMNS = [
    "proposal_id", "resident_track", "resident_owner", "invading_owner",
    "resident_first", "resident_last", "invasion_first", "invasion_last",
    "resident_run_frames", "prior_resident_support", "invasion_frames",
    "resident_strong_fraction", "invasion_strong_fraction",
    "maximum_endpoint_step_radii", "minimum_basin_overlap_fraction",
    "minimum_donor_tracks", "minimum_donor_separation_sum_radii",
    "shared_component_frames", "resident_present_during_invasion",
    "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "frame", "resident_owner", "invading_owner",
    "operation", "donor_tracks", "changed_pixels", "applied", "reason",
]


@dataclass(frozen=True)
class OwnerRun:
    owner: int
    first: int
    last: int
    rows: tuple

    @property
    def frames(self) -> int:
        return len(self.rows)


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("component-continuity takeover must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN_TOKENS))
    if supplied:
        raise ValueError(
            "component-continuity takeover received forbidden targets: "
            + ", ".join(supplied))


def _runs(group: pd.DataFrame) -> list[OwnerRun]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.accepted_owner)
        frame = int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame == runs[-1]["last"] + 1):
            runs[-1]["last"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "rows": [row]})
    return [OwnerRun(int(run["owner"]), int(run["first"]),
                     int(run["last"]), tuple(run["rows"]))
            for run in runs]


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _normalised_step(left, right) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    scale = max(0.5 * (float(left.radius_px) + float(right.radius_px)), 1.0)
    return distance / scale


def _component(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    parts, _ = ndi.label(frame == int(owner), STRUCTURE)
    value = physical._component_at(
        parts, float(point.x), float(point.y),
        max(2, int(math.ceil(float(point.radius_px)))))
    return parts == value if value > 0 else None


def _marker(mask: np.ndarray, point) -> tuple[int, int] | None:
    marker = physical._marker_pixel(mask, float(point.x), float(point.y))
    if marker is None:
        return None
    distance = float(np.hypot(marker[1] - float(point.x),
                              marker[0] - float(point.y)))
    return marker if distance <= max(float(point.radius_px), 2.0) else None


def _overlap_fraction(previous: np.ndarray, current: np.ndarray,
                      dilation: int) -> float:
    if not previous.any() or not current.any():
        return 0.0
    expanded = ndi.binary_dilation(previous, STRUCTURE, iterations=dilation)
    return float(np.count_nonzero(expanded & current)
                 / max(min(int(previous.sum()), int(current.sum())), 1))


def _partition(frame: np.ndarray, raw: np.ndarray, point,
               donor_points: list, params: dict
               ) -> tuple[np.ndarray | None, str]:
    source = int(point.accepted_owner)
    component = _component(frame, source, point)
    if component is None:
        return None, "invading_component_missing"
    resident_marker = _marker(component, point)
    if resident_marker is None:
        return None, "resident_marker_missing"
    inside = []
    outside = []
    for donor in donor_points:
        marker = _marker(component, donor)
        if marker is None:
            outside.append(donor)
        elif marker != resident_marker:
            inside.append((donor, marker))
    if not inside:
        if not outside:
            return None, "donor_home_missing"
        return component, "relabel_separate_invaded_component"
    markers = np.zeros(component.shape, np.int16)
    markers[resident_marker] = 1
    for _, marker in inside:
        markers[marker] = 2
    if not np.any(markers == 2):
        return None, "distinct_donor_marker_missing"
    sigma = float(params.get("watershed_sigma_radius_fraction", 0.25)) \
        * max(float(point.radius_px), 1.0)
    elevation = -ndi.gaussian_filter(raw.astype(np.float32), sigma)
    labels = watershed(elevation, markers=markers, mask=component,
                       connectivity=STRUCTURE)
    basin = labels == 1
    donor_basin = labels == 2
    if not basin.any() or not donor_basin.any():
        return None, "shared_component_partition_empty"
    if any(not bool(donor_basin[_marker(component, donor)])
           for donor, _ in inside):
        return None, "donor_core_not_preserved"
    return basin, "raw_seeded_shared_component_partition"


def _frame_plan(labels: np.ndarray, raw: np.ndarray, scored: pd.DataFrame,
                groups: dict[int, pd.DataFrame], canonical: dict[int, dict],
                track: int, frame: int, previous_basin: np.ndarray,
                resident_owner: int, invading_owner: int, params: dict
                ) -> tuple[dict | None, str]:
    point = _point(groups[track], frame)
    if point is None or int(point.accepted_owner) != int(invading_owner):
        return None, "invading_track_point_missing"
    donor_points = []
    minimum_donor_separation = float(
        params.get("minimum_donor_separation_sum_radii", 1.0))
    for donor_track, profile in canonical.items():
        if int(donor_track) == int(track) \
                or int(profile["owner"]) != int(invading_owner):
            continue
        donor = _point(groups[int(donor_track)], frame)
        if donor is None or not bool(donor.physically_visible):
            continue
        separation = float(np.hypot(float(donor.x) - float(point.x),
                                    float(donor.y) - float(point.y))) / max(
            float(donor.radius_px) + float(point.radius_px), 1.0)
        if separation >= minimum_donor_separation:
            donor_points.append(donor)
    minimum_donors = int(params.get("minimum_donor_lineage_tracks", 2))
    if len(donor_points) < minimum_donors:
        return None, "insufficient_donor_lineage_tracks"
    basin, operation = _partition(
        labels[frame], raw[frame], point, donor_points, params)
    if basin is None:
        return None, operation
    minimum_overlap = float(params.get("minimum_basin_overlap_fraction", 0.35))
    overlap = _overlap_fraction(
        previous_basin, basin,
        int(params.get("continuity_dilation_radius_fraction", 1.0)
            * max(float(point.radius_px), 1.0)))
    if overlap < minimum_overlap:
        return None, "invaded_basin_not_continuous"
    expected = math.pi * max(float(point.radius_px), 1.0) ** 2
    area_ratio = int(basin.sum()) / max(expected, 1.0)
    if not (float(params.get("minimum_basin_expected_area_fraction", 0.20))
            <= area_ratio
            <= float(params.get("maximum_basin_expected_area_fraction", 4.0))):
        return None, "invaded_basin_area_outside_radius_scale"
    frame_visible = scored[
        scored.frame.astype(int).eq(frame)
        & scored.physically_visible.astype(bool)]
    basin_tracks = []
    for other in frame_visible.itertuples(index=False):
        marker = _marker(basin, other)
        if marker is not None:
            basin_tracks.append(int(other.track_id))
    if set(basin_tracks) != {int(track)}:
        return None, "invaded_basin_core_set_not_exclusive"
    separation_values = [
        float(np.hypot(float(donor.x) - float(point.x),
                       float(donor.y) - float(point.y))) / max(
            float(donor.radius_px) + float(point.radius_px), 1.0)
        for donor in donor_points]
    return {
        "frame": frame, "basin": basin, "operation": operation,
        "donor_tracks": tuple(sorted({int(row.track_id)
                                       for row in donor_points})),
        "overlap": overlap,
        "minimum_donor_separation": min(separation_values),
        "shared": operation == "raw_seeded_shared_component_partition",
    }, "eligible"


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every direct positive owner transition in the complete field."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    minimum_donor_support = int(math.ceil(len(labels) * float(
        params.get("minimum_donor_support_movie_fraction", 0.12))))
    canonical = physical.canonical_owners(
        scored, minimum_donor_support,
        float(params.get("minimum_donor_owner_purity", 0.80)), False)
    minimum_resident = int(math.ceil(len(labels) * float(
        params.get("minimum_resident_run_movie_fraction", 0.02))))
    minimum_prior = int(math.ceil(len(labels) * float(
        params.get("minimum_prior_resident_support_movie_fraction", 0.04))))
    minimum_invasion = int(math.ceil(len(labels) * float(
        params.get("minimum_invasion_movie_fraction", 0.02))))
    maximum_invasion = int(math.ceil(len(labels) * float(
        params.get("maximum_invasion_movie_fraction", 0.04))))
    minimum_resident_strong = float(
        params.get("minimum_resident_strong_fraction", 0.50))
    minimum_invasion_strong = float(
        params.get("minimum_invasion_strong_fraction", 1.0))
    maximum_step = float(params.get("maximum_endpoint_step_radii", 0.75))
    public_rows: list[dict] = []
    internal_rows: list[dict] = []
    for track, group in groups.items():
        runs = _runs(group)
        for resident, invasion in zip(runs[:-1], runs[1:]):
            if (resident.owner <= 0 or invasion.owner <= 0
                    or resident.owner == invasion.owner
                    or invasion.first != resident.last + 1):
                continue
            reasons: list[str] = []
            prior_support = int(group[
                group.frame.astype(int).le(resident.last)
                & group.accepted_owner.astype(int).eq(resident.owner)
                ].shape[0])
            resident_strong = float(np.mean([bool(row.strong)
                                             for row in resident.rows]))
            invasion_strong = float(np.mean([bool(row.strong)
                                             for row in invasion.rows]))
            step = _normalised_step(resident.rows[-1], invasion.rows[0])
            if resident.frames < minimum_resident:
                reasons.append("resident_run_too_short")
            if prior_support < minimum_prior:
                reasons.append("insufficient_prior_resident_support")
            if not minimum_invasion <= invasion.frames <= maximum_invasion:
                reasons.append("invasion_duration_outside_range")
            if resident_strong < minimum_resident_strong:
                reasons.append("resident_raw_support_too_weak")
            if invasion_strong < minimum_invasion_strong:
                reasons.append("invasion_raw_support_too_weak")
            if step > maximum_step:
                reasons.append("physical_endpoint_discontinuity")
            resident_present = sum(
                bool(np.any(labels[frame] == resident.owner))
                for frame in range(invasion.first, invasion.last + 1))
            if resident_present:
                reasons.append("resident_owner_present_during_invasion")
            previous = _component(
                labels[resident.last], resident.owner, resident.rows[-1])
            plans = []
            failure = "resident_component_missing" if previous is None else ""
            if not reasons and previous is not None:
                for frame in range(invasion.first, invasion.last + 1):
                    plan, frame_reason = _frame_plan(
                        labels, raw, scored, groups, canonical, track, frame,
                        previous, resident.owner, invasion.owner, params)
                    if plan is None:
                        failure = frame_reason
                        break
                    plans.append(plan)
                    previous = plan["basin"]
            if failure and not reasons:
                reasons.append(failure)
            public = {
                "proposal_id": "", "resident_track": track,
                "resident_owner": resident.owner,
                "invading_owner": invasion.owner,
                "resident_first": resident.first,
                "resident_last": resident.last,
                "invasion_first": invasion.first,
                "invasion_last": invasion.last,
                "resident_run_frames": resident.frames,
                "prior_resident_support": prior_support,
                "invasion_frames": invasion.frames,
                "resident_strong_fraction": resident_strong,
                "invasion_strong_fraction": invasion_strong,
                "maximum_endpoint_step_radii": step,
                "minimum_basin_overlap_fraction": min(
                    (plan["overlap"] for plan in plans), default=0.0),
                "minimum_donor_tracks": min(
                    (len(plan["donor_tracks"]) for plan in plans), default=0),
                "minimum_donor_separation_sum_radii": min(
                    (plan["minimum_donor_separation"] for plan in plans),
                    default=float("nan")),
                "shared_component_frames": sum(plan["shared"] for plan in plans),
                "resident_present_during_invasion": resident_present,
                "eligible": not reasons and len(plans) == invasion.frames,
                "reason": ("eligible_component_continuity_duplicate_takeover"
                           if not reasons and len(plans) == invasion.frames
                           else "|".join(reasons)),
            }
            public_rows.append(public)
            internal_rows.append({**public, "frame_plans": plans})
    number = 0
    for public, internal in zip(public_rows, internal_rows):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"CT{number:04d}"
    return (pd.DataFrame(public_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internal_rows))


def apply(labels: np.ndarray, internal: pd.DataFrame
          ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    applications = []
    occupied = np.zeros_like(labels, bool)
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        masks = [(int(plan["frame"]), plan["basin"])
                 for plan in proposal.frame_plans]
        if any(np.any(occupied[frame] & mask) for frame, mask in masks):
            continue
        trial = candidate.copy()
        for frame, mask in masks:
            if not np.all(trial[frame][mask] == int(proposal.invading_owner)):
                raise AssertionError("invaded basin provenance changed")
            trial[frame][mask] = int(proposal.resident_owner)
        if component_accounting.new_duplicate_components(labels, trial):
            continue
        for plan in proposal.frame_plans:
            frame = int(plan["frame"])
            mask = plan["basin"]
            changed = int(np.count_nonzero(candidate[frame][mask]
                                           != int(proposal.resident_owner)))
            candidate[frame][mask] = int(proposal.resident_owner)
            occupied[frame] |= mask
            applications.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "resident_owner": int(proposal.resident_owner),
                "invading_owner": int(proposal.invading_owner),
                "operation": plan["operation"],
                "donor_tracks": "|".join(map(str, plan["donor_tracks"])),
                "changed_pixels": changed, "applied": True,
                "reason": "atomic_component_continuity_takeover_recovery",
            })
    return candidate, pd.DataFrame(applications, columns=APPLICATION_COLUMNS)




def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                       dict]:
    """Discover and repair only fully evidenced direct owner takeovers."""
    assert_target_free(params)
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")
    audit, internal = discover(labels, raw, points, params)
    candidate, applications = apply(labels, internal)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "component-continuity takeover changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "component-continuity takeover overlapped unclaimed ledger")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError(
            "component-continuity takeover changed identity set")
    duplicates = component_accounting.new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "component-continuity takeover created duplicates")
    changed = candidate != labels
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "transitions_audited": int(len(audit)),
        "eligible_takeovers": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_takeovers": int(applications.proposal_id.nunique())
            if len(applications) else 0,
        "application_rows": int(len(applications)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(labels), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(duplicates),
    }
    return candidate, unclaimed.copy(), audit, applications, summary
