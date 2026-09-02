"""Recover isolated owner blips censored by a recording boundary.

Discovery audits all physical tracks and accepts mathematical thresholds only.
Biological identities, tracks, frames, coordinates, events, regions, and review
cases are forbidden as producer targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import isolated_owner_blip_recovery as accepted_blip
import recurrent_exclusive_owner_relay as seat_transfer
import separable_merge_recovery as accepted_merge


FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "track_id", "frame_id", "event_id", "coordinate",
    "review_case", "case_id", "target_identity", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
    "review_region",
)
APPLICATION_COLUMNS = [
    "proposal_id", "physical_track", "stable_owner", "transient_owner",
    "outcome", "reason", "changed_pixels", "changed_frames",
    "changed_unclaimed_pixels",
]
FRAME_COLUMNS = [
    "proposal_id", "physical_track", "stable_owner", "transient_owner",
    "frame", "changed_pixels", "orphan_pixels", "recovered_pixels",
    "reason",
]


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and any(token in name.lower() for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "field-wide boundary owner-blip recovery received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "boundary owner-blip recovery must use field-wide discovery")


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    physical_track: int
    stable_owner: int
    transient_owner: int
    first_frame: int
    last_frame: int
    transient_frames: int
    pre_owner_frames: int
    post_owner_frames: int
    visible_frames: int
    first_visible_frame: int
    last_visible_frame: int
    encounter_fraction: float
    history_owner_support: int
    history_owner_purity: float
    censored_side: str
    discovery_status: str
    discovery_reason: str


def discover(scored: pd.DataFrame, frame_count: int,
             params: dict) -> pd.DataFrame:
    assert_target_free(params)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    minimum_visible = max(
        int(params.get("minimum_visible_frames", 20)),
        int(np.ceil(frame_count * float(
            params.get("minimum_visible_fraction", 0.2)))))
    standard_side = int(params.get("minimum_standard_side_frames", 5))
    boundary_side = int(params.get("minimum_boundary_side_frames", 3))
    maximum_middle = int(params.get("maximum_blip_frames", 2))
    maximum_encounter = float(params.get(
        "maximum_history_encounter_fraction", 0.0))
    minimum_purity = float(params.get("minimum_history_owner_purity", 0.8))
    canonical = accepted_merge.canonical_owners(
        scored, int(params.get("minimum_canonical_support_frames", 8)),
        minimum_purity,
        bool(params.get("canonical_exclude_encounter_frames", False)))
    rows: list[dict] = []
    for track, group in visible.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        runs = accepted_blip._positive_runs(group)
        first_visible = int(group.frame.min())
        last_visible = int(group.frame.max())
        encounter_fraction = float(group.in_encounter.astype(bool).mean())
        for run_index in range(1, len(runs) - 1):
            before, middle, after = runs[run_index - 1:run_index + 2]
            stable_owner = int(before["owner"])
            transient_owner = int(middle["owner"])
            stable = canonical.get(int(track))
            left_censored = bool(
                first_visible == 0
                and boundary_side <= len(before["frames"]) < standard_side
                and len(after["frames"]) >= standard_side)
            right_censored = bool(
                last_visible == frame_count - 1
                and boundary_side <= len(after["frames"]) < standard_side
                and len(before["frames"]) >= standard_side)
            reasons: list[str] = []
            if (stable_owner <= 0 or stable_owner != int(after["owner"])
                    or stable_owner == transient_owner):
                reasons.append("not_positive_bracketed_blip")
            if (int(middle["start"]) != int(before["end"]) + 1
                    or int(after["start"]) != int(middle["end"]) + 1):
                reasons.append("physical_owner_history_not_contiguous")
            if len(group) < minimum_visible:
                reasons.append("insufficient_visible_history")
            if len(middle["frames"]) > maximum_middle:
                reasons.append("middle_run_too_long")
            if not (left_censored or right_censored):
                reasons.append("stable_side_not_boundary_censored")
            if encounter_fraction > maximum_encounter:
                reasons.append("physical_history_contains_encounter")
            if stable is None or int(stable["owner"]) != stable_owner:
                reasons.append("bracketing_owner_not_history_dominant")
            rows.append(asdict(Proposal(
                proposal_id=f"P{len(rows) + 1:04d}",
                physical_track=int(track), stable_owner=stable_owner,
                transient_owner=transient_owner,
                first_frame=int(middle["start"]),
                last_frame=int(middle["end"]),
                transient_frames=int(len(middle["frames"])),
                pre_owner_frames=int(len(before["frames"])),
                post_owner_frames=int(len(after["frames"])),
                visible_frames=int(len(group)),
                first_visible_frame=first_visible,
                last_visible_frame=last_visible,
                encounter_fraction=encounter_fraction,
                history_owner_support=(int(stable["support"])
                                       if stable is not None else 0),
                history_owner_purity=(float(stable["purity"])
                                      if stable is not None else 0.0),
                censored_side=("start" if left_censored else
                               "end" if right_censored else "none"),
                discovery_status="eligible" if not reasons else "rejected",
                discovery_reason="eligible" if not reasons else "|".join(reasons),
            )))
    return pd.DataFrame(rows)


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                       pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed, and raw stacks must align")
    scored = accepted_merge.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    canonical = accepted_merge.canonical_owners(
        scored, int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_history_owner_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    proposals = discover(scored, len(labels), params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, bool)
    next_unclaimed = int(np.max(unclaimed)) + 1
    applications: list[dict] = []
    frame_rows: list[dict] = []
    eligible = proposals[proposals.discovery_status == "eligible"].sort_values(
        ["first_frame", "physical_track", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        rows = visible[
            (visible.track_id.astype(int) == int(proposal.physical_track))
            & visible.frame.astype(int).between(
                int(proposal.first_frame), int(proposal.last_frame))
        ].sort_values("frame")
        prepared: list[tuple] = []
        refusal = ""
        for row in rows.itertuples(index=False):
            trial, unclaimed_trial, detail = \
                accepted_blip._prepare_transient_frame(
                    candidate, candidate_unclaimed, labels, unclaimed,
                    by_frame, canonical, row, int(proposal.stable_owner),
                    next_unclaimed, params)
            if trial is None or unclaimed_trial is None:
                refusal = str(detail["reason"])
                break
            frame = int(row.frame)
            change = trial != candidate[frame]
            if np.any(occupied[frame] & change):
                refusal = "overlapping_proposal"
                break
            prepared.append((row, trial, unclaimed_trial, detail))
        if refusal:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "physical_track": int(proposal.physical_track),
                "stable_owner": int(proposal.stable_owner),
                "transient_owner": int(proposal.transient_owner),
                "outcome": "rejected_application", "reason": refusal,
                "changed_pixels": 0, "changed_frames": 0,
                "changed_unclaimed_pixels": 0,
            })
            continue
        changed_pixels = 0
        changed_unclaimed = 0
        for row, trial, unclaimed_trial, detail in prepared:
            frame = int(row.frame)
            change = trial != candidate[frame]
            ledger_change = unclaimed_trial != candidate_unclaimed[frame]
            candidate[frame] = trial
            candidate_unclaimed[frame] = unclaimed_trial
            occupied[frame] |= change
            changed_pixels += int(np.count_nonzero(change))
            changed_unclaimed += int(np.count_nonzero(ledger_change))
            frame_rows.append({
                "proposal_id": proposal.proposal_id,
                "physical_track": int(proposal.physical_track),
                "stable_owner": int(proposal.stable_owner),
                "transient_owner": int(proposal.transient_owner),
                "frame": frame, "changed_pixels": int(np.count_nonzero(change)),
                "orphan_pixels": int(detail["orphan_pixels"]),
                "recovered_pixels": int(detail["recovered_pixels"]),
                "reason": "applied",
            })
        applications.append({
            "proposal_id": proposal.proposal_id,
            "physical_track": int(proposal.physical_track),
            "stable_owner": int(proposal.stable_owner),
            "transient_owner": int(proposal.transient_owner),
            "outcome": "applied", "reason": "boundary_blip_restored",
            "changed_pixels": changed_pixels,
            "changed_frames": int(len(prepared)),
            "changed_unclaimed_pixels": changed_unclaimed,
        })
        if changed_unclaimed:
            next_unclaimed += 1
    applications_frame = pd.DataFrame(
        applications, columns=APPLICATION_COLUMNS)
    frames_frame = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    if not np.array_equal(
            (candidate > 0) | (candidate_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("boundary blip changed foreground-ledger union")
    if np.any((unclaimed > 0) & (candidate_unclaimed != unclaimed)):
        raise AssertionError("boundary blip changed existing unclaimed pixels")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("boundary blip changed active identities")
    duplicates = seat_transfer._new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError(
            f"boundary blip created {duplicates} duplicate components")
    return (candidate, candidate_unclaimed, proposals,
            applications_frame, frames_frame)


def summarize(labels: np.ndarray, unclaimed: np.ndarray,
              candidate: np.ndarray, candidate_unclaimed: np.ndarray,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              mode: str) -> dict:
    changed = candidate != labels
    ledger_changed = candidate_unclaimed != unclaimed
    active = set(map(int, np.unique(labels))) - {0}
    after = set(map(int, np.unique(candidate))) - {0}
    named_frame_losses = 0
    for identity in active:
        before = np.any(labels == identity, axis=(1, 2))
        after_presence = np.any(candidate == identity, axis=(1, 2))
        named_frame_losses += int(np.count_nonzero(before & ~after_presence))
    return {
        "mode": mode, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "proposals_audited": int(len(proposals)),
        "eligible_proposals": int(proposals.discovery_status.eq(
            "eligible").sum()) if len(proposals) else 0,
        "applied_proposals": int(applications.outcome.eq(
            "applied").sum()) if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
        "changed_unclaimed_pixels": int(np.count_nonzero(ledger_changed)),
        "foreground_changed_pixels": int(np.count_nonzero(
            ((candidate > 0) != (labels > 0)))),
        "foreground_ledger_union_changed_pixels": int(np.count_nonzero(
            (((candidate > 0) | (candidate_unclaimed > 0))
             != ((labels > 0) | (unclaimed > 0))))),
        "old_identity_set_preserved": active == after,
        "donor_named_frame_losses": named_frame_losses,
        "new_same_frame_identity_components": int(
            seat_transfer._new_duplicate_components(labels, candidate)),
    }
