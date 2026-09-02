"""Restore a bracketed resident and allocate its separate claimant atomically.

Discovery is field-wide.  The producer accepts arrays and mathematical thresholds
only; identities, tracks, frames, coordinates, events and review cases are forbidden.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import ownerless_body_recovery as core_recovery
from resident_takeover import attach_owners, _runs
from recurrent_exclusive_owner_relay import (
    _new_duplicate_components,
)


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}


def _longest_owner_run(group: pd.DataFrame, owner: int) -> dict | None:
    matches = [run for run in _runs(group) if int(run["owner"]) == int(owner)]
    return max(matches, key=lambda run: (int(run["frames"]),
                                         -int(run["first_frame"])),
               default=None)


def _continuous(group: pd.DataFrame, first: int, last: int) -> bool:
    frames = set(map(int, group.frame))
    return all(frame in frames for frame in range(first, last + 1))


def _separation(resident: pd.DataFrame, claimant: pd.DataFrame,
                frames: set[int]) -> tuple[float, float, int]:
    left = resident[resident.frame.isin(frames)][
        ["frame", "x", "y", "radius_px"]]
    right = claimant[claimant.frame.isin(frames)][
        ["frame", "x", "y", "radius_px"]]
    paired = left.merge(right, on="frame", suffixes=("_resident", "_claimant"))
    if not len(paired):
        return 0.0, 0.0, 0
    distance = np.hypot(
        paired.x_resident.to_numpy(float) - paired.x_claimant.to_numpy(float),
        paired.y_resident.to_numpy(float) - paired.y_claimant.to_numpy(float))
    radii = np.maximum(
        paired.radius_px_resident.to_numpy(float)
        + paired.radius_px_claimant.to_numpy(float), 1.0)
    scaled = distance / radii
    return float(np.median(scaled)), float(np.min(scaled)), int(len(scaled))


def discover_cohorts(attached: pd.DataFrame, frame_count: int,
                     params: dict) -> pd.DataFrame:
    """Find complete resident/claimant pairs without using named targets."""
    visible = attached[attached.physically_visible.astype(bool)].copy()
    minimum_resident_visible = int(np.ceil(
        frame_count * float(params.get("minimum_resident_visible_fraction", 0.9))))
    minimum_pre = int(params.get("minimum_resident_pre_frames", 20))
    minimum_post = int(params.get("minimum_resident_post_frames", 5))
    minimum_gap = int(params.get("minimum_resident_gap_frames", 40))
    minimum_zero_fraction = float(
        params.get("minimum_resident_ownerless_fraction", 0.8))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.0))
    minimum_claimant_ownerless = int(
        params.get("minimum_claimant_ownerless_prefix_frames", 5))
    minimum_claimant_takeover = int(
        params.get("minimum_claimant_takeover_frames", 30))
    minimum_median_separation = float(
        params.get("minimum_median_separation_sum_radii", 2.5))
    minimum_point_separation = float(
        params.get("minimum_point_separation_sum_radii", 2.0))
    rows: list[dict] = []

    for resident_id, resident in visible.groupby("track_id", sort=True):
        resident = resident.sort_values("frame")
        if len(resident) < minimum_resident_visible:
            continue
        if float(resident.in_encounter.astype(bool).mean()) > maximum_encounter:
            continue
        runs = _runs(resident)
        for left_index, left in enumerate(runs):
            if int(left["owner"]) <= 0 or int(left["frames"]) < minimum_pre:
                continue
            for right in runs[left_index + 2:]:
                if (int(right["owner"]) != int(left["owner"])
                        or int(right["frames"]) < minimum_post):
                    continue
                gap_first = int(left["last_frame"]) + 1
                gap_last = int(right["first_frame"]) - 1
                gap_frames = gap_last - gap_first + 1
                if gap_frames < minimum_gap:
                    continue
                interval = resident[
                    (resident.frame >= gap_first) & (resident.frame <= gap_last)]
                if not _continuous(interval, gap_first, gap_last):
                    continue
                owners = interval.candidate_owner.astype(int)
                foreign = set(owners) - {0, int(left["owner"])}
                zero_fraction = float((owners == 0).mean())
                if foreign or zero_fraction < minimum_zero_fraction:
                    continue

                claimant_candidates: list[dict] = []
                for claimant_id, claimant in visible.groupby("track_id", sort=True):
                    if int(claimant_id) == int(resident_id):
                        continue
                    claimant = claimant.sort_values("frame")
                    if float(claimant.in_encounter.astype(bool).mean()) > maximum_encounter:
                        continue
                    other_owners = set(map(int, claimant.candidate_owner)) \
                        - {0, int(left["owner"])}
                    if other_owners:
                        continue
                    takeover = _longest_owner_run(claimant, int(left["owner"]))
                    if takeover is None or int(takeover["frames"]) < minimum_claimant_takeover:
                        continue
                    takeover_first = int(takeover["first_frame"])
                    takeover_last = int(takeover["last_frame"])
                    if takeover_first < gap_first or takeover_last > gap_last:
                        continue
                    prior = claimant[claimant.frame < takeover_first]
                    if (len(prior) < minimum_claimant_ownerless
                            or not prior.candidate_owner.astype(int).eq(0).all()):
                        continue
                    claimant_first = int(claimant.frame.min())
                    claimant_last = int(claimant.frame.max())
                    if claimant_first < gap_first - 1 or claimant_last > gap_last:
                        continue
                    if not _continuous(claimant, claimant_first, claimant_last):
                        continue
                    owned_frames = set(range(takeover_first, takeover_last + 1))
                    median_sep, minimum_sep, paired = _separation(
                        resident, claimant, owned_frames)
                    if (paired < minimum_claimant_takeover
                            or median_sep < minimum_median_separation
                            or minimum_sep < minimum_point_separation):
                        continue
                    claimant_candidates.append({
                        "claimant_track": int(claimant_id),
                        "claimant_first_frame": claimant_first,
                        "claimant_last_frame": claimant_last,
                        "claimant_ownerless_prefix_frames": int(len(prior)),
                        "claimant_takeover_first": takeover_first,
                        "claimant_takeover_last": takeover_last,
                        "claimant_takeover_frames": int(takeover["frames"]),
                        "median_separation_sum_radii": median_sep,
                        "minimum_separation_sum_radii": minimum_sep,
                    })

                status = "eligible" if len(claimant_candidates) == 1 else "rejected"
                reason = ("eligible" if len(claimant_candidates) == 1 else
                          "no_complete_claimant" if not claimant_candidates else
                          "ambiguous_multiple_claimants")
                claimant = claimant_candidates[0] if len(claimant_candidates) == 1 else {
                    "claimant_track": 0, "claimant_first_frame": -1,
                    "claimant_last_frame": -1,
                    "claimant_ownerless_prefix_frames": 0,
                    "claimant_takeover_first": -1, "claimant_takeover_last": -1,
                    "claimant_takeover_frames": 0,
                    "median_separation_sum_radii": 0.0,
                    "minimum_separation_sum_radii": 0.0,
                }
                rows.append({
                    "cohort_id": f"C{len(rows) + 1:04d}",
                    "resident_track": int(resident_id),
                    "resident_owner": int(left["owner"]),
                    "resident_pre_first": int(left["first_frame"]),
                    "resident_pre_last": int(left["last_frame"]),
                    "resident_pre_frames": int(left["frames"]),
                    "resident_gap_first": gap_first,
                    "resident_gap_last": gap_last,
                    "resident_gap_frames": gap_frames,
                    "resident_ownerless_fraction": zero_fraction,
                    "resident_post_first": int(right["first_frame"]),
                    "resident_post_last": int(right["last_frame"]),
                    "resident_post_frames": int(right["frames"]),
                    **claimant,
                    "discovery_status": status,
                    "discovery_reason": reason,
                })
    columns = [
        "cohort_id", "resident_track", "resident_owner",
        "resident_pre_first", "resident_pre_last", "resident_pre_frames",
        "resident_gap_first", "resident_gap_last", "resident_gap_frames",
        "resident_ownerless_fraction", "resident_post_first",
        "resident_post_last", "resident_post_frames", "claimant_track",
        "claimant_first_frame", "claimant_last_frame",
        "claimant_ownerless_prefix_frames", "claimant_takeover_first",
        "claimant_takeover_last", "claimant_takeover_frames",
        "median_separation_sum_radii", "minimum_separation_sum_radii",
        "discovery_status", "discovery_reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def _component_near(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(frame == owner, STRUCTURE)
    if not count:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    value = int(components[y, x])
    if value:
        return components == value
    radius = float(np.clip(float(point.radius_px), 2.0, 7.0))
    ys, xs = np.nonzero(frame == owner)
    if not len(xs):
        return None
    distance = np.hypot(xs - x, ys - y)
    best = int(np.argmin(distance))
    if float(distance[best]) > radius:
        return None
    return components == int(components[ys[best], xs[best]])


def _raw_core(raw_frame: np.ndarray, point, weak_threshold: float,
              params: dict) -> tuple[np.ndarray, tuple[slice, slice], dict]:
    core, region, details = core_recovery._connected_core(
        raw_frame, point, weak_threshold, params)
    if details["core_area_px"]:
        details["seed_mode"] = "track_point"
        return core, region, details
    # Latent points can lie between the few positive pixels of a very dim body.
    # Search only within the detector-derived radius, then apply the same core
    # thresholding from the strongest nearby positive evidence pixel.
    evidence = core_recovery.evidence_image(raw_frame, params)
    y = int(np.clip(round(float(point.y)), 0, raw_frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, raw_frame.shape[1] - 1))
    radius = int(np.ceil(np.clip(float(point.radius_px), 2.0, 6.0)))
    y0, y1 = max(0, y - radius), min(raw_frame.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(raw_frame.shape[1], x + radius + 1)
    positive = raw_frame[y0:y1, x0:x1] > 0
    if not np.any(positive):
        details["seed_mode"] = "no_nearby_positive_signal"
        return core, region, details
    values = np.where(positive, evidence[y0:y1, x0:x1], -np.inf)
    local_y, local_x = np.unravel_index(int(np.argmax(values)), values.shape)
    proxy = SimpleNamespace(
        x=float(x0 + local_x), y=float(y0 + local_y),
        radius_px=float(point.radius_px))
    core, region, details = core_recovery._connected_core(
        raw_frame, proxy, weak_threshold, params)
    details["seed_mode"] = "nearest_radius_evidence_peak"
    details["seed_distance_px"] = float(np.hypot(
        float(proxy.x) - float(point.x), float(proxy.y) - float(point.y)))
    return core, region, details


def _apply_cohort(candidate: np.ndarray, unclaimed: np.ndarray,
                  raw: np.ndarray, attached: pd.DataFrame,
                  thresholds: pd.DataFrame, cohort, new_identity: int,
                  params: dict) -> tuple[np.ndarray | None, list[dict], str]:
    work = candidate.copy()
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.5))
    audit: list[dict] = []
    resident = attached[
        (attached.track_id == int(cohort.resident_track))
        & attached.physically_visible.astype(bool)].sort_values("frame")
    claimant = attached[
        (attached.track_id == int(cohort.claimant_track))
        & attached.physically_visible.astype(bool)].sort_values("frame")

    for point in claimant.itertuples(index=False):
        frame = int(point.frame)
        owner = int(point.candidate_owner)
        if owner == int(cohort.resident_owner):
            component = _component_near(
                work[frame], int(cohort.resident_owner), point)
            if component is None:
                return None, audit, "claimant_owner_component_missing"
            changed = int(np.count_nonzero(
                component & (work[frame] != new_identity)))
            work[frame][component] = new_identity
            audit.append({"cohort_id": cohort.cohort_id, "frame": frame,
                          "seat": "claimant", "operation": "relabel_component",
                          "changed_pixels": changed, "new_identity": new_identity})
        elif owner == 0:
            core, region, details = _raw_core(
                raw[frame], point,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            expected = np.pi * max(float(point.radius_px), 1.0) ** 2
            ratio = details["core_area_px"] / max(expected, 1.0)
            occupied = ((work[frame][region] > 0)
                        | (unclaimed[frame][region] > 0))
            if (not details["core_area_px"] or ratio > maximum_ratio
                    or np.any(core & occupied)
                    or np.any(core & (raw[frame][region] == 0))):
                return None, audit, "claimant_raw_core_incomplete"
            work[frame][region][core] = new_identity
            audit.append({"cohort_id": cohort.cohort_id, "frame": frame,
                          "seat": "claimant", "operation": "add_raw_core",
                          "changed_pixels": int(core.sum()),
                          "new_identity": new_identity})
        else:
            return None, audit, "claimant_has_foreign_owner"

    resident_interval = resident[
        (resident.frame >= int(cohort.resident_gap_first))
        & (resident.frame <= int(cohort.resident_gap_last))]
    for point in resident_interval.itertuples(index=False):
        frame = int(point.frame)
        owner = int(point.candidate_owner)
        if owner == int(cohort.resident_owner):
            audit.append({"cohort_id": cohort.cohort_id, "frame": frame,
                          "seat": "resident", "operation": "already_owned",
                          "changed_pixels": 0, "new_identity": new_identity})
            continue
        if owner != 0:
            return None, audit, "resident_has_foreign_owner"
        core, region, details = _raw_core(
            raw[frame], point,
            float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
        expected = np.pi * max(float(point.radius_px), 1.0) ** 2
        ratio = details["core_area_px"] / max(expected, 1.0)
        occupied = ((work[frame][region] > 0)
                    | (unclaimed[frame][region] > 0))
        if (not details["core_area_px"] or ratio > maximum_ratio
                or np.any(core & occupied)
                or np.any(core & (raw[frame][region] == 0))):
            return None, audit, "resident_raw_core_incomplete"
        work[frame][region][core] = int(cohort.resident_owner)
        audit.append({"cohort_id": cohort.cohort_id, "frame": frame,
                      "seat": "resident", "operation": "add_raw_core",
                      "changed_pixels": int(core.sum()),
                      "new_identity": new_identity})

    check = attach_owners(
        pd.concat([resident_interval, claimant], ignore_index=True), work)
    resident_check = check[check.track_id == int(cohort.resident_track)]
    claimant_check = check[check.track_id == int(cohort.claimant_track)]
    if not resident_check.candidate_owner.astype(int).eq(
            int(cohort.resident_owner)).all():
        return None, audit, "resident_verification_failed"
    if not claimant_check.candidate_owner.astype(int).eq(new_identity).all():
        return None, audit, "claimant_verification_failed"
    return work, audit, "applied"


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame,
            params: dict) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame,
                                   pd.DataFrame]:
    supplied = sorted(name for name in FORBIDDEN_TARGETS
                      if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError("field-wide two-seat recovery received forbidden targets: "
                         + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("two-seat recovery must use field-wide discovery")
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed and raw stacks must align")
    attached = attach_owners(points, labels)
    cohorts = discover_cohorts(attached, len(labels), params)
    candidate = labels.copy()
    next_identity = int(candidate.max()) + 1
    applications: list[dict] = []
    frame_rows: list[dict] = []
    eligible = cohorts[cohorts.discovery_status == "eligible"].sort_values(
        ["claimant_first_frame", "claimant_track", "cohort_id"])
    for cohort in eligible.itertuples(index=False):
        proposed, audit, reason = _apply_cohort(
            candidate, unclaimed, raw, attached, thresholds, cohort,
            next_identity, params)
        if proposed is None:
            applications.append({
                "cohort_id": cohort.cohort_id, "resident_track": cohort.resident_track,
                "claimant_track": cohort.claimant_track, "resident_owner": cohort.resident_owner,
                "assigned_claimant_identity": 0, "outcome": "rejected_application",
                "reason": reason, "changed_pixels": 0, "changed_frames": 0})
            continue
        changed = proposed != candidate
        candidate = proposed
        applications.append({
            "cohort_id": cohort.cohort_id, "resident_track": cohort.resident_track,
            "claimant_track": cohort.claimant_track, "resident_owner": cohort.resident_owner,
            "assigned_claimant_identity": next_identity, "outcome": "applied",
            "reason": reason, "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2))))})
        frame_rows.extend(audit)
        next_identity += 1
    return (candidate, cohorts, pd.DataFrame(applications),
            pd.DataFrame(frame_rows, columns=[
                "cohort_id", "frame", "seat", "operation",
                "changed_pixels", "new_identity"]))


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray,
              points: pd.DataFrame, cohorts: pd.DataFrame,
              applications: pd.DataFrame, params: dict,
              mode: str = "candidate") -> dict:
    """Return whole-field guardrails and audited producer counts."""
    baseline_ids = set(map(int, np.unique(labels))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    relabelled = changed & (labels > 0) & (candidate > 0)
    removed = changed & (labels > 0) & (candidate == 0)
    metrics = {
        "mode": mode,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "tracks_audited": int(points.track_id.nunique()),
        "cohorts_audited": int(len(cohorts)),
        "eligible_cohorts": int(cohorts.discovery_status.eq("eligible").sum()),
        "applied_cohorts": int(applications.outcome.eq("applied").sum())
            if len(applications) else 0,
        "rejected_application_cohorts": int(
            applications.outcome.eq("rejected_application").sum())
            if len(applications) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "raw_supported_additions": int(additions.sum()),
        "relabelled_foreground_pixels": int(relabelled.sum()),
        "removed_foreground_pixels": int(removed.sum()),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "unclaimed_overlap_additions": int(np.count_nonzero(
            additions & (unclaimed > 0))),
        "old_identity_set_preserved": baseline_ids.issubset(candidate_ids),
        "active_identities_before": len(baseline_ids),
        "active_identities_after": len(candidate_ids),
        "new_identities": sorted(candidate_ids - baseline_ids),
        "new_same_frame_identity_components": int(
            _new_duplicate_components(labels, candidate)),
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
    if metrics["removed_foreground_pixels"]:
        raise AssertionError("two-seat recovery removed foreground")
    if metrics["zero_signal_additions"]:
        raise AssertionError("two-seat recovery added zero-signal pixels")
    if metrics["unclaimed_overlap_additions"]:
        raise AssertionError("two-seat recovery overlapped the unclaimed ledger")
    if not metrics["old_identity_set_preserved"]:
        raise AssertionError("two-seat recovery removed an old identity")
    return metrics
