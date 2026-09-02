"""Restore a resident when an already-occupied owner invades its separate body.

Discovery is field-wide. Biological identities, tracks, frames, coordinates,
events, and review cases are inferred from the complete stack and cannot be
supplied as producer targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from atomic_two_seat_recovery import _raw_core
from late_owner_backfill import _dim_raw_core
import separable_merge_recovery as physical
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "track_id", "frame_id", "event_id", "coordinate",
    "review_case", "case_id", "target_identity", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
    "review_region",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and (name in FORBIDDEN_TARGETS or any(
            token in name.lower() for token in FORBIDDEN_TARGET_TOKENS)))
    if supplied:
        raise ValueError(
            "concurrent-owner invasion received forbidden targets: "
            + ", ".join(supplied))


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    resident_track: int
    donor_track: int
    resident_owner: int
    invading_owner: int
    resident_first: int
    resident_last: int
    blank_first: int
    blank_last: int
    invasion_first: int
    invasion_last: int
    resident_run_frames: int
    blank_frames: int
    invasion_frames: int
    donor_canonical_support: int
    donor_owned_invasion_frames: int
    maximum_endpoint_step_radii: float
    minimum_separation_sum_radii: float
    maximum_valley_ratio: float
    discovery_status: str
    discovery_reason: str


def _runs(group: pd.DataFrame) -> list[dict]:
    result: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        frame = int(row.frame)
        owner = int(row.accepted_owner)
        if (result and result[-1]["owner"] == owner
                and frame == result[-1]["last"] + 1):
            result[-1]["last"] = frame
            result[-1]["rows"].append(row)
        else:
            result.append({"owner": owner, "first": frame, "last": frame,
                           "rows": [row]})
    for run in result:
        run["frames"] = len(run["rows"])
    return result


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame == int(frame)]
    return None if len(rows) != 1 else rows.iloc[0]


def _step_ratio(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(
        0.5 * (float(left.radius_px) + float(right.radius_px)), 1.0)


def _separation_ratio(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def _component_near(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, _ = ndi.label(frame == int(owner), STRUCTURE)
    component_id = physical._component_at(
        components, float(point.x), float(point.y), search_radius=6)
    return None if component_id <= 0 else components == component_id


def discover(scored: pd.DataFrame, labels: np.ndarray, raw: np.ndarray,
             params: dict) -> pd.DataFrame:
    """Audit every owner transition for a separate, concurrent owner donor."""
    # Accepted ownership can remain informative on a detector-dormant frame.
    # Keep complete physical tracks for tenure and transition discovery;
    # application separately requires visible evidence for every changed pixel.
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    canonical = physical.canonical_owners(
        scored,
        int(params.get("minimum_donor_canonical_support_frames", 12)),
        float(params.get("minimum_donor_canonical_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    minimum_resident = int(params.get("minimum_resident_run_frames", 12))
    minimum_invasion = int(params.get("minimum_invasion_run_frames", 2))
    maximum_invasion = max(1, int(np.ceil(
        len(labels) * float(params.get("maximum_invasion_fraction", 0.08)))))
    maximum_blank = int(params.get("maximum_intervening_blank_frames", 2))
    minimum_blank = int(params.get("minimum_intervening_blank_frames", 1))
    maximum_step = float(params.get("maximum_endpoint_step_radii", 2.0))
    minimum_separation = float(
        params.get("minimum_donor_separation_sum_radii", 1.5))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.65))
    rows: list[dict] = []

    for resident_track, group in groups.items():
        runs = _runs(group)
        positive_indices = [index for index, run in enumerate(runs)
                            if int(run["owner"]) > 0]
        for left_index, right_index in zip(
                positive_indices[:-1], positive_indices[1:]):
            resident = runs[left_index]
            invasion = runs[right_index]
            if int(resident["owner"]) == int(invasion["owner"]):
                continue
            blank_frames = (int(invasion["first"])
                            - int(resident["last"]) - 1)
            if blank_frames < 0 or blank_frames > maximum_blank:
                continue
            reasons: list[str] = []
            if blank_frames < minimum_blank:
                reasons.append("no_detection_dropout_before_invasion")
            if int(resident["frames"]) < minimum_resident:
                reasons.append("resident_run_too_short")
            if int(invasion["frames"]) < minimum_invasion:
                reasons.append("invasion_run_too_short")
            if int(invasion["frames"]) > maximum_invasion:
                reasons.append("invasion_run_too_long")
            if not all(bool(row.physically_visible)
                       for row in invasion["rows"]):
                reasons.append("invasion_not_physically_visible")
            transition = group[group.frame.between(
                int(resident["last"]), int(invasion["first"]))]
            transition_rows = list(transition.itertuples(index=False))
            endpoint_step = max(
                (_step_ratio(left, right)
                 for left, right in zip(transition_rows[:-1],
                                        transition_rows[1:])),
                default=float("inf"))
            if endpoint_step > maximum_step:
                reasons.append("physical_endpoint_discontinuity")
            complete = group[group.frame.between(
                int(resident["last"]), int(invasion["last"]))]
            expected = np.arange(
                int(resident["last"]), int(invasion["last"]) + 1)
            if not np.array_equal(complete.frame.to_numpy(int), expected):
                reasons.append("physical_track_not_continuous")
            resident_owner = int(resident["owner"])
            invading_owner = int(invasion["owner"])
            interval = range(int(resident["last"]) + 1,
                             int(invasion["last"]) + 1)
            if any(np.any(labels[frame] == resident_owner)
                   for frame in interval):
                reasons.append("resident_owner_present_during_loss")

            donor_options: list[tuple[int, int, float, float]] = []
            for donor_track, detail in canonical.items():
                if (int(donor_track) == resident_track
                        or int(detail["owner"]) != invading_owner):
                    continue
                donor = groups.get(int(donor_track))
                if donor is None:
                    continue
                donor_interval = donor[donor.frame.between(
                    int(invasion["first"]), int(invasion["last"]))]
                if len(donor_interval) != int(invasion["frames"]):
                    continue
                if not donor_interval.accepted_owner.astype(int).eq(
                        invading_owner).all():
                    continue
                separations: list[float] = []
                valleys: list[float] = []
                valid = True
                for target_row in invasion["rows"]:
                    donor_row = _point(donor, int(target_row.frame))
                    if donor_row is None:
                        valid = False
                        break
                    if bool(target_row.in_encounter) or bool(donor_row.in_encounter):
                        valid = False
                        break
                    separation = _separation_ratio(target_row, donor_row)
                    if separation < minimum_separation:
                        valid = False
                        break
                    target_component = _component_near(
                        labels[int(target_row.frame)], invading_owner, target_row)
                    donor_component = _component_near(
                        labels[int(target_row.frame)], invading_owner, donor_row)
                    if target_component is None or donor_component is None:
                        valid = False
                        break
                    valley = physical._valley_ratio(
                        raw[int(target_row.frame)],
                        (float(target_row.x), float(target_row.y)),
                        (float(donor_row.x), float(donor_row.y)))
                    if valley > maximum_valley:
                        valid = False
                        break
                    separations.append(separation)
                    valleys.append(valley)
                if valid:
                    donor_options.append((
                        int(donor_track), int(detail["support"]),
                        min(separations), max(valleys)))
            donor_options.sort(key=lambda item: (-item[1], item[0]))
            if len(donor_options) != 1:
                reasons.append("concurrent_donor_not_unique")
            donor_track = donor_options[0][0] if len(donor_options) == 1 else -1
            donor_support = donor_options[0][1] if len(donor_options) == 1 else 0
            separation = (donor_options[0][2]
                          if len(donor_options) == 1 else float("nan"))
            valley = (donor_options[0][3]
                      if len(donor_options) == 1 else float("nan"))
            rows.append(asdict(Proposal(
                proposal_id=f"P{len(rows) + 1:04d}",
                resident_track=resident_track,
                donor_track=donor_track,
                resident_owner=resident_owner,
                invading_owner=invading_owner,
                resident_first=int(resident["first"]),
                resident_last=int(resident["last"]),
                blank_first=(int(resident["last"]) + 1
                             if blank_frames else -1),
                blank_last=(int(invasion["first"]) - 1
                            if blank_frames else -1),
                invasion_first=int(invasion["first"]),
                invasion_last=int(invasion["last"]),
                resident_run_frames=int(resident["frames"]),
                blank_frames=blank_frames,
                invasion_frames=int(invasion["frames"]),
                donor_canonical_support=donor_support,
                donor_owned_invasion_frames=(
                    int(invasion["frames"]) if donor_track >= 0 else 0),
                maximum_endpoint_step_radii=endpoint_step,
                minimum_separation_sum_radii=separation,
                maximum_valley_ratio=valley,
                discovery_status="eligible" if not reasons else "rejected",
                discovery_reason=("eligible" if not reasons
                                  else "|".join(reasons)),
            )))
    return pd.DataFrame(rows)


def _component_excess(frame: np.ndarray, identity: int) -> int:
    return max(0, int(ndi.label(frame == int(identity), STRUCTURE)[1]) - 1)


def _apply(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
           thresholds: pd.DataFrame, scored: pd.DataFrame, proposal,
           params: dict):
    candidate = labels.copy()
    target_group = scored[
        scored.track_id == int(proposal.resident_track)].sort_values("frame")
    donor_group = scored[
        scored.track_id == int(proposal.donor_track)].sort_values("frame")
    threshold_by_frame = thresholds.set_index("frame")
    resident = int(proposal.resident_owner)
    invader = int(proposal.invading_owner)
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.5))
    audit: list[dict] = []

    for frame in range(int(proposal.resident_last) + 1,
                       int(proposal.invasion_last) + 1):
        target = _point(target_group, frame)
        if target is None or not bool(target.physically_visible):
            return None, pd.DataFrame(audit), "target_point_missing"
        before = candidate[frame].copy()
        if frame < int(proposal.invasion_first):
            if int(target.accepted_owner) != 0:
                return None, pd.DataFrame(audit), "blank_interval_not_ownerless"
            core, region, details = _raw_core(
                raw[frame], target,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            if not details["core_area_px"]:
                core, region, details = _dim_raw_core(raw[frame], target, params)
            expected = np.pi * max(float(target.radius_px), 1.0) ** 2
            ratio = int(details["core_area_px"]) / max(expected, 1.0)
            occupied = ((candidate[frame][region] > 0)
                        | (unclaimed[frame][region] > 0))
            if (not details["core_area_px"] or ratio > maximum_ratio
                    or np.any(core & occupied)
                    or np.any(core & (raw[frame][region] == 0))):
                return None, pd.DataFrame(audit), "blank_raw_core_incomplete"
            candidate[frame][region][core] = resident
            method = "add_raw_supported_blank_core"
        else:
            donor = _point(donor_group, frame)
            if donor is None:
                return None, pd.DataFrame(audit), "donor_point_missing"
            target_component = _component_near(
                candidate[frame], invader, target)
            donor_component = _component_near(
                candidate[frame], invader, donor)
            if target_component is None or donor_component is None:
                return None, pd.DataFrame(audit), "invading_component_missing"
            component_tracks = set()
            frame_points = scored[
                (scored.frame == frame)
                & scored.physically_visible.astype(bool)]
            for other in frame_points.itertuples(index=False):
                y = int(np.clip(round(float(other.y)), 0,
                                target_component.shape[0] - 1))
                x = int(np.clip(round(float(other.x)), 0,
                                target_component.shape[1] - 1))
                if target_component[y, x]:
                    component_tracks.add(int(other.track_id))
            shared = np.any(target_component & donor_component)
            expected_tracks = ({int(proposal.resident_track),
                                int(proposal.donor_track)} if shared else
                               {int(proposal.resident_track)})
            if component_tracks != expected_tracks:
                return None, pd.DataFrame(audit), "invaded_component_has_other_core"
            if not shared:
                candidate[frame][target_component] = resident
                method = "relabel_separate_invaded_component"
            else:
                target_marker = physical._marker_pixel(
                    target_component, float(target.x), float(target.y))
                donor_marker = physical._marker_pixel(
                    target_component, float(donor.x), float(donor.y))
                if (target_marker is None or donor_marker is None
                        or target_marker == donor_marker):
                    return None, pd.DataFrame(audit), "watershed_markers_unavailable"
                markers = np.zeros(target_component.shape, np.int16)
                markers[target_marker] = 1
                markers[donor_marker] = 2
                sigma = float(params.get("watershed_sigma_px", 1.0))
                elevation = -ndi.gaussian_filter(
                    raw[frame].astype(np.float32), sigma)
                target_partition = watershed(
                    elevation, markers=markers, mask=target_component) == 1
                if not np.any(target_partition):
                    return None, pd.DataFrame(audit), "empty_target_partition"
                candidate[frame][target_partition] = resident
                residual_parts, _ = ndi.label(
                    target_component & (candidate[frame] == invader), STRUCTURE)
                dy = int(np.clip(round(float(donor.y)), 0,
                                 candidate.shape[1] - 1))
                dx = int(np.clip(round(float(donor.x)), 0,
                                 candidate.shape[2] - 1))
                donor_part = int(residual_parts[dy, dx])
                if donor_part <= 0:
                    return None, pd.DataFrame(audit), "donor_core_not_preserved"
                shards = ((residual_parts > 0)
                          & (residual_parts != donor_part))
                candidate[frame][shards] = resident
                method = "raw_seeded_shared_component_partition"

        if _component_excess(candidate[frame], resident) > max(
                0, _component_excess(before, resident)):
            return None, pd.DataFrame(audit), "new_resident_duplicate"
        if _component_excess(candidate[frame], invader) > _component_excess(
                before, invader):
            return None, pd.DataFrame(audit), "new_invader_duplicate"
        if frame >= int(proposal.invasion_first):
            donor = _point(donor_group, frame)
            if donor is None or int(physical._disk_owner(
                    candidate[frame], float(donor.x), float(donor.y),
                    float(donor.radius_px))) != invader:
                return None, pd.DataFrame(audit), "donor_owner_not_preserved"
        audit.append({
            "proposal_id": proposal.proposal_id,
            "frame": frame,
            "operation": method,
            "resident_owner": resident,
            "invading_owner": invader,
            "changed_pixels": int(np.count_nonzero(candidate[frame] != before)),
        })
    return candidate, pd.DataFrame(audit), "applied"


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("concurrent-owner invasion must be field-wide")
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed, and raw stacks must align")
    scored = physical.attach_owners(points, labels)
    proposals = discover(scored, labels, raw, params)
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    applications: list[dict] = []
    frame_audits: list[pd.DataFrame] = []
    eligible = proposals[proposals.discovery_status == "eligible"].sort_values(
        ["invasion_first", "resident_track", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        proposed, audit, reason = _apply(
            candidate, unclaimed, raw, thresholds, scored, proposal, params)
        if proposed is None:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "resident_track": int(proposal.resident_track),
                "outcome": "rejected_application", "reason": reason,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        changed = proposed != candidate
        if np.any(changed & occupied):
            applications.append({
                "proposal_id": proposal.proposal_id,
                "resident_track": int(proposal.resident_track),
                "outcome": "rejected_application", "reason": "overlap",
                "changed_pixels": 0, "changed_frames": 0})
            continue
        candidate = proposed
        occupied |= changed
        applications.append({
            "proposal_id": proposal.proposal_id,
            "resident_track": int(proposal.resident_track),
            "outcome": "applied", "reason": reason,
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(
                np.any(changed, axis=(1, 2))))})
        frame_audits.append(audit)
    return (candidate, proposals, pd.DataFrame(applications),
            pd.concat(frame_audits, ignore_index=True)
            if frame_audits else pd.DataFrame(), scored)


def _excess_components(stack: np.ndarray) -> int:
    return component_accounting.total_component_excess(stack)


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              mode: str) -> dict:
    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    active = set(map(int, np.unique(labels))) - {0}
    after_active = set(map(int, np.unique(candidate))) - {0}
    named_losses = 0
    for identity in active:
        before = np.any(labels == identity, axis=(1, 2))
        after = np.any(candidate == identity, axis=(1, 2))
        named_losses += int(np.count_nonzero(before & ~after))
    return {
        "mode": mode,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "review_case_targets_received": False,
        "tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(proposals)),
        "eligible_proposals": int(proposals.discovery_status.eq(
            "eligible").sum()) if len(proposals) else 0,
        "applied_proposals": int(applications.outcome.eq("applied").sum())
            if len(applications) else 0,
        "rejected_application_proposals": int(applications.outcome.eq(
            "rejected_application").sum()) if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "raw_supported_additions": int(np.count_nonzero(additions)),
        "relabelled_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate > 0))),
        "removed_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "unclaimed_overlap_additions": int(np.count_nonzero(
            additions & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "old_identity_set_preserved": active == after_active,
        "donor_named_frame_losses": named_losses,
        "new_same_frame_identity_components": max(
            0, _excess_components(candidate) - _excess_components(labels)),
    }
