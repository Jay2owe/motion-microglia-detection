"""Remove a boundary identity-loss alert after its lost seat is restored.

The calibration is field-wide and consumes only the complete producer ledger.
It removes an aggregate merge/split alert when that alert's terminal identity-
loss constituent is exactly the right-censored transition repaired by an
applied two-seat partition. Labels and unclaimed foreground are copied exactly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

import terminal_boundary_vanished_seat as terminal


ALLOWED_SOURCES = {
    "merge_split_identity_loss",
    "separable_cells_merged",
    "single_cell_multi_core",
}
AUDIT_COLUMNS = [
    "event_id", "family", "first_frame", "last_frame", "event_owners",
    "event_tracks", "matched_proposal_id", "victim_owner",
    "resident_owner", "victim_track", "resident_track", "last_owned_frame",
    "repair_first", "repair_last", "continuous_two_owner_presence",
    "terminal_identity_loss_sources", "eligible", "reason",
]


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _numbers(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return set(map(int, re.findall(r"\d+", str(value))))


def event_summary(events: pd.DataFrame) -> dict:
    owner = events[events.family.astype(str).eq("identity_swap_or_takeover")]
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


def calibrate(labels: np.ndarray, events: pd.DataFrame,
              members: pd.DataFrame, applications: pd.DataFrame,
              frames: pd.DataFrame, params: dict):
    terminal.assert_target_free(params)
    movie = int(len(labels))
    maximum_suffix = max(1, int(math.ceil(movie * float(params.get(
        "maximum_terminal_suffix_movie_fraction", 0.02)))))
    applied = applications[applications.applied.map(_truth)].copy()
    rows = []
    removed = set()
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        first, last = int(event.first_frame), int(event.last_frame)
        owners = _numbers(event.accepted_before) | _numbers(
            event.accepted_after)
        tracks = _numbers(event.physical_tracks)
        event_members = members[
            members.event_id.astype(str).eq(event_id)].copy()
        source_families = set(event_members.source_family.astype(str))
        matches = []
        for proposal in applied.itertuples(index=False):
            if (owners == {int(proposal.victim_owner),
                           int(proposal.resident_owner)}
                    and {int(proposal.victim_track),
                         int(proposal.resident_track)} <= tracks
                    and int(proposal.last_owned_frame) == last):
                matches.append(proposal)
        reasons = []
        if str(event.family) != "merge_split_identity_loss":
            reasons.append("not_merge_split_identity_loss")
        if not source_families or not source_families <= ALLOWED_SOURCES:
            reasons.append("unresolved_source_family_present")
        if len(matches) != 1:
            reasons.append("applied_boundary_partition_not_unique")
            match = None
        else:
            match = matches[0]

        continuous = False
        terminal_sources = pd.DataFrame()
        if match is not None:
            victim = int(match.victim_owner)
            resident = int(match.resident_owner)
            repair_first = int(match.repair_first)
            repair_last = int(match.repair_last)
            if repair_first != last + 1 or repair_last != movie - 1:
                reasons.append("repair_does_not_complete_censored_boundary")
            if repair_last - repair_first + 1 > maximum_suffix:
                reasons.append("repair_suffix_too_long")
            proposal_frames = frames[
                frames.proposal_id.astype(str).eq(str(match.proposal_id))]
            if set(proposal_frames.frame.astype(int)) != set(
                    range(repair_first, repair_last + 1)):
                reasons.append("repair_frame_ledger_incomplete")
            interval = range(first, repair_last + 1)
            continuous = bool(all(
                np.any(labels[frame] == victim)
                and np.any(labels[frame] == resident)
                for frame in interval))
            if not continuous:
                reasons.append("two_owner_presence_not_continuous")
            terminal_sources = event_members[
                event_members.source_family.astype(str).eq(
                    "merge_split_identity_loss")
                & event_members.source_last_frame.astype(int).eq(last)
                & event_members.source_accepted_after.map(
                    lambda value: victim not in _numbers(value)
                    and resident in _numbers(value))]
            if terminal_sources.empty:
                reasons.append("terminal_identity_loss_source_not_matched")

        eligible = not reasons
        if eligible:
            removed.add(event_id)
        rows.append({
            "event_id": event_id, "family": str(event.family),
            "first_frame": first, "last_frame": last,
            "event_owners": "|".join(map(str, sorted(owners))),
            "event_tracks": "|".join(map(str, sorted(tracks))),
            "matched_proposal_id": str(match.proposal_id) if match else "",
            "victim_owner": int(match.victim_owner) if match else 0,
            "resident_owner": int(match.resident_owner) if match else 0,
            "victim_track": int(match.victim_track) if match else 0,
            "resident_track": int(match.resident_track) if match else 0,
            "last_owned_frame": int(match.last_owned_frame) if match else -1,
            "repair_first": int(match.repair_first) if match else -1,
            "repair_last": int(match.repair_last) if match else -1,
            "continuous_two_owner_presence": continuous,
            "terminal_identity_loss_sources": int(
                terminal_sources.source_id.nunique())
                if len(terminal_sources) else 0,
            "eligible": eligible,
            "reason": ("producer_proved_boundary_identity_loss_resolved"
                       if eligible else "|".join(dict.fromkeys(reasons))),
        })
    candidate_events = events[
        ~events.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    candidate_events["impact_rank"] = np.arange(1, len(candidate_events) + 1)
    candidate_members = members[
        ~members.event_id.astype(str).isin(removed)].copy()
    return (candidate_events, candidate_members,
            pd.DataFrame(rows, columns=AUDIT_COLUMNS), sorted(removed))


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    terminal.assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    applications = pd.read_csv(params["applications_path"])
    frames = pd.read_csv(params["application_frames_path"])
    labels = __import__("tifffile").imread(labels_path)
    candidate_events, candidate_members, audit, removed = calibrate(
        labels, events, members, applications, frames, params)
    outputs = {
        "labels": out.out / labels_path.name,
        "unclaimed": out.out / unclaimed_path.name,
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "terminal_boundary_event_calibration_audit.csv",
        "metrics": out.out / "metrics.json",
    }
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    candidate_events.to_csv(outputs["events"], index=False)
    candidate_members.to_csv(outputs["members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(events)),
        "events_removed": int(len(removed)),
        "removed_event_ids": removed,
        "before": event_summary(events),
        "after": event_summary(candidate_events),
        "label_bytes_unchanged": outputs["labels"].read_bytes()
            == labels_path.read_bytes(),
        "unclaimed_bytes_unchanged": outputs["unclaimed"].read_bytes()
            == unclaimed_path.read_bytes(),
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

