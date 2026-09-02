"""Recover stable owners through isolated one- or two-frame owner blips.

Discovery is field-wide.  The producer accepts mathematical thresholds and complete
physical-track evidence only; review identities, tracks, frames, events and regions
are forbidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import separable_merge_recovery as accepted_merge
import recurrent_exclusive_owner_relay as seat_transfer


FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        if not bool(row.physically_visible) or int(row.accepted_owner) <= 0:
            continue
        frame = int(row.frame)
        owner = int(row.accepted_owner)
        if (runs and runs[-1]["owner"] == owner
                and frame == runs[-1]["end"] + 1):
            runs[-1]["end"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "start": frame, "end": frame,
                         "frames": [frame]})
    return runs


def discover_blips(scored: pd.DataFrame, frame_count: int,
                   params: dict) -> tuple[list[dict], pd.DataFrame]:
    """Discover temporally isolated A-B-A runs without using object numbers."""
    visible = scored[scored.physically_visible.astype(bool)].copy()
    minimum_visible = max(
        int(params.get("minimum_visible_frames", 20)),
        int(np.ceil(frame_count * float(
            params.get("minimum_visible_fraction", 0.2)))))
    minimum_side = int(params.get("minimum_stable_side_frames", 4))
    maximum_middle = int(params.get("maximum_blip_frames", 2))
    minimum_purity = float(params.get("minimum_history_owner_purity", 0.8))
    canonical = accepted_merge.canonical_owners(
        scored,
        int(params.get("minimum_canonical_support_frames", 8)),
        minimum_purity,
        bool(params.get("canonical_exclude_encounter_frames", False)))
    hypotheses: list[dict] = []
    audits: list[dict] = []

    for track_id, group in visible.groupby("track_id"):
        track_id = int(track_id)
        runs = _positive_runs(group)
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
                "pre_frames": int(len(before["frames"])),
                "post_frames": int(len(after["frames"])),
                "visible_frames": int(len(group)),
                "eligible": False,
                "applied": False,
                "reason": "",
            }
            if (before["owner"] <= 0 or before["owner"] != after["owner"]
                    or before["owner"] == middle["owner"]):
                audits.append({**base, "reason": "not_positive_bracketed_blip"})
                continue
            if (int(middle["start"]) != int(before["end"]) + 1
                    or int(after["start"]) != int(middle["end"]) + 1):
                audits.append({**base, "reason": "physical_owner_history_not_contiguous"})
                continue
            if len(group) < minimum_visible:
                audits.append({**base, "reason": "insufficient_visible_history"})
                continue
            if len(middle["frames"]) > maximum_middle:
                audits.append({**base, "reason": "middle_run_too_long"})
                continue
            if (len(before["frames"]) < minimum_side
                    or len(after["frames"]) < minimum_side):
                audits.append({**base, "reason": "stable_side_too_short"})
                continue
            stable = canonical.get(track_id)
            if stable is None or int(stable["owner"]) != int(before["owner"]):
                audits.append({**base, "reason": "bracketing_owner_not_history_dominant"})
                continue
            hypothesis = {
                **base,
                "history_owner_support": int(stable["support"]),
                "history_owner_purity": float(stable["purity"]),
                "eligible": True,
                "reason": "eligible",
            }
            hypotheses.append(hypothesis)
            audits.append(hypothesis.copy())
    return hypotheses, pd.DataFrame(audits)


def _prepare_transient_frame(
        candidate: np.ndarray, candidate_unclaimed: np.ndarray,
        baseline_labels: np.ndarray, baseline_unclaimed: np.ndarray,
        by_frame: dict[int, pd.DataFrame], canonical: dict[int, dict],
        row, stable_owner: int, unclaimed_id: int, params: dict,
        ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Restore A only when short-lived B owns no other physical raw core."""
    frame = int(row.frame)
    track_id = int(row.track_id)
    transient_owner = int(row.accepted_owner)
    detail = {
        "frame": frame, "transient_owner": transient_owner,
        "changed_pixels": 0, "orphan_pixels": 0,
        "recovered_pixels": 0, "reason": "",
    }
    host_components, _ = ndi.label(
        candidate[frame] == transient_owner, seat_transfer.STRUCTURE)
    component_id = accepted_merge._component_at(
        host_components, float(row.x), float(row.y))
    if component_id <= 0:
        return None, None, {**detail, "reason": "resident_outside_transient_owner"}
    shared = host_components == component_id
    for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
        other_track = int(other.track_id)
        if other_track == track_id:
            continue
        if seat_transfer._inside(shared, other):
            return None, None, {
                **detail, "reason": "transient_component_has_second_raw_core"}
        stable = canonical.get(other_track)
        if stable is None or int(stable["owner"]) != transient_owner:
            continue
        if accepted_merge._component_at(
                host_components, float(other.x), float(other.y)) > 0:
            return None, None, {
                **detail, "reason": "transient_owner_has_retained_raw_core"}

    target_mask = candidate[frame] == stable_owner
    orphan_target = np.zeros_like(shared)
    target_components, target_count = ndi.label(
        target_mask, seat_transfer.STRUCTURE)
    resident_y = int(np.clip(round(float(row.y)), 0, shared.shape[0] - 1))
    resident_x = int(np.clip(round(float(row.x)), 0, shared.shape[1] - 1))
    for target_id in range(1, target_count + 1):
        component = target_components == target_id
        for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
            if int(other.track_id) != track_id and seat_transfer._inside(component, other):
                return None, None, {
                    **detail, "reason": "stable_owner_seat_has_other_raw_core"}
        joins_resident = bool(component[resident_y, resident_x])
        touches_recovered = bool(np.any(
            ndi.binary_dilation(component, structure=seat_transfer.STRUCTURE)
            & shared))
        if not joins_resident and not touches_recovered:
            orphan_target |= component

    trial = candidate[frame].copy()
    unclaimed_trial = candidate_unclaimed[frame].copy()
    trial[orphan_target] = 0
    unclaimed_trial[orphan_target] = unclaimed_id
    trial[shared] = stable_owner
    if not np.any(trial == stable_owner):
        return None, None, {**detail, "reason": "stable_owner_not_restored"}
    if (seat_transfer._component_excess(trial, stable_owner)
            > seat_transfer._component_excess(
                baseline_labels[frame], stable_owner)):
        return None, None, {**detail, "reason": "new_stable_owner_duplicate"}
    if (seat_transfer._component_excess(trial, transient_owner)
            > seat_transfer._component_excess(
                baseline_labels[frame], transient_owner)):
        return None, None, {**detail, "reason": "new_transient_owner_duplicate"}
    if not np.array_equal(
            (trial > 0) | (unclaimed_trial > 0),
            (baseline_labels[frame] > 0) | (baseline_unclaimed[frame] > 0)):
        return None, None, {**detail, "reason": "foreground_ledger_union_changed"}
    changed = int(np.count_nonzero(trial != candidate[frame]))
    if changed < int(params.get("minimum_changed_pixels", 5)):
        return None, None, {**detail, "reason": "restoration_too_small"}
    return trial, unclaimed_trial, {
        **detail, "changed_pixels": changed,
        "orphan_pixels": int(np.count_nonzero(orphan_target)),
        "recovered_pixels": int(np.count_nonzero(shared)),
        "reason": "prepared",
    }


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    supplied = sorted(name for name in FORBIDDEN_TARGETS
                      if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide isolated owner-blip recovery received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("owner-blip recovery must use field-wide discovery")
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")

    scored = accepted_merge.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    canonical = accepted_merge.canonical_owners(
        scored,
        int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_history_owner_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    hypotheses, event_audit = discover_blips(scored, len(labels), params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, bool)
    next_unclaimed = int(np.max(unclaimed)) + 1
    frame_audits: list[dict] = []

    for hypothesis in hypotheses:
        track_id = int(hypothesis["track_id"])
        owner = int(hypothesis["owner_before"])
        first = int(hypothesis["first_frame"])
        last = int(hypothesis["last_frame"])
        rows = visible[
            (visible.track_id.astype(int) == track_id)
            & visible.frame.astype(int).between(first, last)
        ].sort_values("frame")
        prepared = []
        refusal = ""
        for row in rows.itertuples(index=False):
            trial, unclaimed_trial, detail = _prepare_transient_frame(
                candidate, candidate_unclaimed, labels, unclaimed,
                by_frame, canonical, row, owner, next_unclaimed, params)
            detail.update({"track_id": track_id, "stable_owner": owner,
                           "first_frame": first, "last_frame": last,
                           "applied": False})
            if trial is None or unclaimed_trial is None:
                refusal = str(detail["reason"])
                frame_audits.append(detail)
                break
            frame = int(row.frame)
            change = trial != candidate[frame]
            if np.any(occupied[frame] & change):
                refusal = "overlapping_proposal"
                frame_audits.append({**detail, "reason": refusal})
                break
            prepared.append((row, trial, unclaimed_trial, detail))
        index = event_audit.index[
            (event_audit.track_id.astype(int) == track_id)
            & (event_audit.first_frame.astype(int) == first)
            & (event_audit.last_frame.astype(int) == last)]
        if refusal:
            for _, _, _, detail in prepared:
                frame_audits.append({**detail, "reason": "event_atomic_refusal"})
            if len(index):
                event_audit.loc[index, "reason"] = refusal
            continue
        changed = 0
        for row, trial, unclaimed_trial, detail in prepared:
            frame = int(row.frame)
            change = trial != candidate[frame]
            candidate[frame] = trial
            candidate_unclaimed[frame] = unclaimed_trial
            occupied[frame] |= change
            changed += int(np.count_nonzero(change))
            frame_audits.append({**detail, "applied": True,
                                 "reason": "applied"})
        if prepared and len(index):
            event_audit.loc[index, "applied"] = True
            event_audit.loc[index, "changed_pixels"] = changed
            event_audit.loc[index, "assigned_unclaimed_id"] = next_unclaimed
            event_audit.loc[index, "reason"] = "applied"
            next_unclaimed += 1

    if not np.array_equal(
            (candidate > 0) | (candidate_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("owner-blip recovery changed foreground-ledger union")
    if np.any((unclaimed > 0) & (candidate_unclaimed != unclaimed)):
        raise AssertionError("owner-blip recovery changed existing unclaimed pixels")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("owner-blip recovery changed active identities")
    duplicates = seat_transfer._new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError(
            f"owner-blip recovery created {duplicates} duplicate components")
    return candidate, candidate_unclaimed, event_audit, pd.DataFrame(frame_audits)

