"""Recover an owner bracket split across a physical-track handoff.

Discovery is field-wide. Identity values are opaque labels inferred from the
physical-track owner history; this producer accepts no identity, track, frame,
coordinate, event, or review-case target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from atomic_two_seat_recovery import _raw_core
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
PROPOSAL_COLUMNS = [
    "proposal_id", "successor_track", "predecessor_track", "donor_track",
    "target_owner", "foreign_owner", "first_frame", "last_frame",
    "foreign_frames", "predecessor_target_frames", "successor_target_frames",
    "handoff_step_sum_radii", "donor_owned_middle_frames",
    "discovery_status", "discovery_reason",
]


def assert_target_free(params: dict) -> None:
    """Refuse selected biological cases under singular or plural key names."""
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and (name in FORBIDDEN_TARGETS or any(
            token in name.lower() for token in FORBIDDEN_TARGET_TOKENS)))
    if supplied:
        raise ValueError(
            "field-wide handoff recovery received forbidden targets: "
            + ", ".join(supplied))


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        if not bool(row.physically_visible):
            continue
        frame = int(row.frame)
        owner = int(row.accepted_owner)
        if owner <= 0:
            continue
        if (runs and int(runs[-1]["owner"]) == owner
                and frame == int(runs[-1]["last_frame"]) + 1):
            runs[-1]["last_frame"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "first_frame": frame,
                         "last_frame": frame, "frames": [frame]})
    return runs


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame == int(frame)]
    return None if not len(rows) else rows.iloc[0]


def _distance_ratio(left, right) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    return distance / max(float(left.radius_px) + float(right.radius_px), 1.0)


def _component_near(frame: np.ndarray, owner: int, row) -> np.ndarray | None:
    parts, _ = ndi.label(frame == int(owner), STRUCTURE)
    component_id = physical._component_at(
        parts, float(row.x), float(row.y), search_radius=6)
    return None if component_id <= 0 else parts == component_id


def discover(scored: pd.DataFrame, labels: np.ndarray,
             params: dict) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=PROPOSAL_COLUMNS)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id")}
    canonical = physical.canonical_owners(
        scored,
        int(params.get("minimum_donor_canonical_support_frames", 8)),
        float(params.get("minimum_donor_canonical_purity", 0.7)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    minimum_before = int(params.get("minimum_predecessor_owner_run_frames", 4))
    minimum_after = int(params.get("minimum_successor_owner_run_frames", 4))
    maximum_middle = max(1, int(np.ceil(
        len(labels) * float(params.get("maximum_foreign_run_fraction", 0.08)))))
    maximum_gap = int(params.get("maximum_handoff_gap_frames", 0))
    maximum_step = float(params.get("maximum_handoff_step_sum_radii", 1.5))
    minimum_donor_middle = int(params.get("minimum_donor_owned_middle_frames", 3))
    rows: list[dict] = []
    for successor_id, successor in sorted(groups.items()):
        runs = _positive_runs(successor)
        if len(runs) < 2:
            continue
        foreign_run, target_run = runs[-2:]
        foreign = int(foreign_run["owner"])
        target = int(target_run["owner"])
        first_visible = int(successor.frame.min())
        reasons: list[str] = []
        if int(foreign_run["first_frame"]) != first_visible:
            reasons.append("foreign_run_not_successor_start")
        if len(foreign_run["frames"]) > maximum_middle:
            reasons.append("foreign_run_too_long")
        if len(target_run["frames"]) < minimum_after:
            reasons.append("terminal_target_run_too_short")
        if int(target_run["last_frame"]) != int(successor.frame.max()):
            reasons.append("target_run_not_terminal")
        middle_frames = list(map(int, foreign_run["frames"]))
        if any(np.any(labels[frame] == target) for frame in middle_frames):
            reasons.append("target_owner_present_during_handoff_gap")

        first_row = _point(successor, int(foreign_run["first_frame"]))
        predecessor_options: list[tuple[float, int, dict, object]] = []
        for predecessor_id, predecessor in groups.items():
            if predecessor_id == successor_id:
                continue
            gap = (int(foreign_run["first_frame"])
                   - int(predecessor.frame.max()) - 1)
            if gap < 0 or gap > maximum_gap:
                continue
            predecessor_runs = _positive_runs(predecessor)
            if not predecessor_runs:
                continue
            terminal = predecessor_runs[-1]
            if (int(terminal["owner"]) != target
                    or len(terminal["frames"]) < minimum_before
                    or int(terminal["last_frame"]) != int(predecessor.frame.max())):
                continue
            last_row = _point(predecessor, int(predecessor.frame.max()))
            ratio = _distance_ratio(last_row, first_row)
            if ratio <= maximum_step:
                predecessor_options.append(
                    (ratio, predecessor_id, terminal, last_row))
        predecessor_options.sort(key=lambda item: item[0])
        if not predecessor_options:
            reasons.append("no_continuous_predecessor_target_track")
            predecessor_id = -1
            handoff_ratio = np.nan
        else:
            predecessor_id = int(predecessor_options[0][1])
            handoff_ratio = float(predecessor_options[0][0])
            if (len(predecessor_options) > 1
                    and np.isclose(predecessor_options[0][0],
                                   predecessor_options[1][0], atol=1e-6)):
                reasons.append("predecessor_handoff_not_unique")
        if first_row is not None and bool(first_row.in_encounter):
            reasons.append("successor_handoff_starts_in_encounter")
        if predecessor_options and bool(predecessor_options[0][3].in_encounter):
            reasons.append("predecessor_handoff_ends_in_encounter")

        donor_options = []
        for donor_track, detail in canonical.items():
            if int(detail["owner"]) != foreign or donor_track == successor_id:
                continue
            donor_group = groups.get(int(donor_track))
            if donor_group is None:
                continue
            donor_middle = donor_group[donor_group.frame.isin(middle_frames)]
            if (len(donor_middle) != len(middle_frames)
                    or not donor_middle.physically_visible.astype(bool).all()):
                continue
            owned = int((donor_middle.accepted_owner.astype(int) == foreign).sum())
            if owned >= minimum_donor_middle:
                donor_options.append((
                    -int(detail["support"]), -float(detail["purity"]),
                    int(donor_track), owned))
        donor_options.sort()
        if not donor_options:
            reasons.append("no_visible_canonical_foreign_owner_track")
            donor_track = -1
            donor_owned = 0
        else:
            donor_track = int(donor_options[0][2])
            donor_owned = int(donor_options[0][3])
            if (len(donor_options) > 1
                    and donor_options[0][:2] == donor_options[1][:2]):
                reasons.append("canonical_foreign_owner_track_not_unique")
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "successor_track": successor_id,
            "predecessor_track": predecessor_id,
            "donor_track": donor_track,
            "target_owner": target,
            "foreign_owner": foreign,
            "first_frame": int(foreign_run["first_frame"]),
            "last_frame": int(foreign_run["last_frame"]),
            "foreign_frames": len(middle_frames),
            "predecessor_target_frames": (
                len(predecessor_options[0][2]["frames"])
                if predecessor_options else 0),
            "successor_target_frames": len(target_run["frames"]),
            "handoff_step_sum_radii": handoff_ratio,
            "donor_owned_middle_frames": donor_owned,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)


def apply_proposal(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
                   scored: pd.DataFrame, thresholds: pd.DataFrame,
                   proposal, params: dict,
                   ) -> tuple[np.ndarray | None, pd.DataFrame, str]:
    candidate = labels.copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id")}
    successor = groups[int(proposal.successor_track)]
    donor = groups[int(proposal.donor_track)]
    target = int(proposal.target_owner)
    foreign = int(proposal.foreign_owner)
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.5))
    frame_audit: list[dict] = []
    for frame in range(int(proposal.first_frame), int(proposal.last_frame) + 1):
        successor_row = _point(successor, frame)
        donor_row = _point(donor, frame)
        if successor_row is None or donor_row is None:
            return None, pd.DataFrame(frame_audit), "missing_physical_track_point"
        before_frame = candidate[frame].copy()
        local = _component_near(candidate[frame], foreign, successor_row)
        if local is None:
            return None, pd.DataFrame(frame_audit), "foreign_component_missing_at_successor"
        if bool(local[int(round(float(donor_row.y))),
                      int(round(float(donor_row.x)))]):
            return None, pd.DataFrame(frame_audit), "donor_and_successor_share_component"
        candidate[frame][local] = target
        operation = "relabel_successor_component"
        added = 0
        donor_component = _component_near(candidate[frame], foreign, donor_row)
        if donor_component is None:
            core, region, details = _raw_core(
                raw[frame], donor_row,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            expected = np.pi * max(float(donor_row.radius_px), 1.0) ** 2
            ratio = int(details["core_area_px"]) / max(expected, 1.0)
            if not details["core_area_px"] or ratio > maximum_ratio:
                return None, pd.DataFrame(frame_audit), "donor_raw_core_incomplete"
            if np.any(core & (candidate[frame][region] > 0)):
                return None, pd.DataFrame(frame_audit), "donor_raw_core_overlaps_labels"
            if np.any(core & (unclaimed[frame][region] > 0)):
                return None, pd.DataFrame(frame_audit), "donor_raw_core_overlaps_unclaimed"
            if np.any(core & (raw[frame][region] == 0)):
                return None, pd.DataFrame(frame_audit), "donor_raw_core_contains_zero_signal"
            candidate[frame][region][core] = foreign
            added = int(np.count_nonzero(core))
            operation += "|recover_displaced_donor_core"
        if ndi.label(candidate[frame] == target, STRUCTURE)[1] != 1:
            return None, pd.DataFrame(frame_audit), "target_topology_not_single_component"
        if ndi.label(candidate[frame] == foreign, STRUCTURE)[1] != 1:
            return None, pd.DataFrame(frame_audit), "foreign_topology_not_single_component"
        frame_audit.append({
            "proposal_id": proposal.proposal_id,
            "frame": frame,
            "operation": operation,
            "target_owner": target,
            "foreign_owner": foreign,
            "relabelled_pixels": int(np.count_nonzero(local)),
            "raw_supported_additions": added,
            "changed_pixels": int(np.count_nonzero(
                candidate[frame] != before_frame)),
        })

    verified = physical.attach_owners(
        pd.concat([successor, donor], ignore_index=True), candidate)
    successor_verified = verified[
        verified.track_id == int(proposal.successor_track)]
    successor_verified = successor_verified[successor_verified.frame.between(
        int(proposal.first_frame), int(proposal.last_frame))]
    donor_verified = verified[verified.track_id == int(proposal.donor_track)]
    donor_verified = donor_verified[donor_verified.frame.between(
        int(proposal.first_frame), int(proposal.last_frame))]
    if not successor_verified.accepted_owner.astype(int).eq(target).all():
        return None, pd.DataFrame(frame_audit), "successor_owner_verification_failed"
    if not donor_verified.accepted_owner.astype(int).eq(foreign).all():
        return None, pd.DataFrame(frame_audit), "donor_owner_verification_failed"
    return candidate, pd.DataFrame(frame_audit), "applied"


def _excess_components(labels: np.ndarray) -> int:
    return component_accounting.total_component_excess(labels)


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("handoff recovery must use field-wide discovery")
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed and raw stacks must align")
    scored = physical.attach_owners(points, labels)
    proposals = discover(scored, labels, params)
    candidate = labels.copy()
    applications: list[dict] = []
    audits: list[pd.DataFrame] = []
    eligible = proposals[proposals.discovery_status == "eligible"]
    for proposal in eligible.itertuples(index=False):
        proposed, frame_audit, reason = apply_proposal(
            candidate, unclaimed, raw, scored, thresholds, proposal, params)
        if proposed is None:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "outcome": "rejected_application", "reason": reason,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        changed = proposed != candidate
        candidate = proposed
        applications.append({
            "proposal_id": proposal.proposal_id,
            "outcome": "applied", "reason": reason,
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(
                np.any(changed, axis=(1, 2))))})
        audits.append(frame_audit)
    return (candidate, proposals, pd.DataFrame(applications),
            pd.concat(audits, ignore_index=True) if audits else pd.DataFrame())


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              mode: str) -> dict:
    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    active = set(map(int, np.unique(labels))) - {0}
    candidate_active = set(map(int, np.unique(candidate))) - {0}
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
        "eligible_handoffs": int(proposals.discovery_status.eq("eligible").sum()),
        "applied_handoffs": int(applications.outcome.eq("applied").sum())
            if len(applications) else 0,
        "rejected_application_handoffs": int(
            applications.outcome.eq("rejected_application").sum())
            if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "raw_supported_additions": int(np.count_nonzero(additions)),
        "relabelled_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate > 0))),
        "removed_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "unclaimed_overlap_additions": int(np.count_nonzero(
            additions & (unclaimed > 0))),
        "old_identity_set_preserved": active == candidate_active,
        "donor_named_frame_losses": named_losses,
        "new_same_frame_identity_components": max(
            0, _excess_components(candidate) - _excess_components(labels)),
    }
