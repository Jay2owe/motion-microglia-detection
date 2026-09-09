"""Remove multi-core alerts already proved to be one owner's projection.

This is a score-only calibration.  It can remove an event only when a
target-free producer audit proves the exact target/anchor physical-track pair,
all proposal frames were applied atomically, no unexplained component was
created, and the detector describes the complete encounter as one unchanged
owner.  Biological labels and foreground ledgers are never edited here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region_target", "review_case",
    "case_id", "forced_identity", "forced_track", "forced_frame",
    "include_track", "exclude_track", "include_identity", "exclude_identity",
)
AUDIT_COLUMNS = [
    "event_id", "family", "event_tracks", "event_owner",
    "matched_proposal_id", "proposal_track", "anchor_track",
    "proposal_first", "proposal_last", "all_frames_applied",
    "unexplained_components", "eligible", "reason",
]


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "explained projection calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("explained projection calibration must be field-wide")


def _owners(value) -> list[int]:
    result = []
    for token in str(value).replace("+", "|").split("|"):
        token = token.strip()
        if token and token.lower() != "nan":
            result.append(int(float(token)))
    return sorted(set(result))


def _tracks(value) -> set[int]:
    return set(_owners(value))


def _summary(events: pd.DataFrame, transactions: dict | None = None) -> dict:
    owner = events[events.family.astype(str).eq("identity_swap_or_takeover")]
    result = {
        "events": int(len(events)),
        "owner_change_events": int(len(owner)),
        "owner_change_disruption_burden": float(
            owner.disruption_score.astype(float).sum()),
        "high_disruption_events": int(
            (events.disruption_score.astype(float) >= 50).sum()),
        "total_disruption_burden": float(
            events.disruption_score.astype(float).sum()),
        "events_by_family": {
            str(key): int(value) for key, value in
            events.family.astype(str).value_counts().sort_index().items()},
    }
    if transactions:
        result.update(transactions)
    return result


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              applications: pd.DataFrame):
    proposals = []
    for proposal_id, group in applications.groupby("proposal_id", sort=True):
        applied = group.applied.astype(str).str.lower().eq("true")
        proposal = {
            "proposal_id": str(proposal_id),
            "track": int(group.track_id.astype(int).iloc[0]),
            "anchor": int(group.anchor_track.astype(int).iloc[0]),
            "owner": int(group.bracket_owner.astype(int).iloc[0]),
            "first": int(group.frame.astype(int).min()),
            "last": int(group.frame.astype(int).max()),
            "all_applied": bool(applied.all()),
            "unexplained": int(
                group.unexplained_duplicate_components.astype(int).sum()),
        }
        proposals.append(proposal)

    audit_rows = []
    removed = set()
    for event in events.itertuples(index=False):
        event_tracks = _tracks(event.physical_tracks)
        before = _owners(event.accepted_before)
        after = _owners(event.accepted_after)
        matches = []
        for proposal in proposals:
            if event_tracks != {proposal["track"], proposal["anchor"]}:
                continue
            if not (int(event.first_frame) <= proposal["first"]
                    and int(event.last_frame) >= proposal["last"]):
                continue
            matches.append(proposal)
        reasons = []
        if str(event.family) != "single_cell_multi_core":
            reasons.append("not_single_cell_multi_core")
        if len(matches) != 1:
            reasons.append("producer_proposal_not_unique")
            match = None
        else:
            match = matches[0]
            if before != [match["owner"]] or after != [match["owner"]]:
                reasons.append("event_owner_changes_across_encounter")
            if not match["all_applied"]:
                reasons.append("producer_proposal_not_fully_applied")
            if match["unexplained"]:
                reasons.append("producer_reported_unexplained_components")
        eligible = not reasons
        if eligible:
            removed.add(str(event.event_id))
        audit_rows.append({
            "event_id": str(event.event_id), "family": str(event.family),
            "event_tracks": "|".join(map(str, sorted(event_tracks))),
            "event_owner": "|".join(map(str, before)),
            "matched_proposal_id": match["proposal_id"] if match else "",
            "proposal_track": match["track"] if match else 0,
            "anchor_track": match["anchor"] if match else 0,
            "proposal_first": match["first"] if match else -1,
            "proposal_last": match["last"] if match else -1,
            "all_frames_applied": match["all_applied"] if match else False,
            "unexplained_components": match["unexplained"] if match else 0,
            "eligible": eligible,
            "reason": ("explained_projection_event"
                       if eligible else "|".join(reasons)),
        })
    candidate_events = events[
        ~events.event_id.astype(str).isin(removed)].copy()
    candidate_members = members[
        ~members.event_id.astype(str).isin(removed)].copy()
    return (candidate_events, candidate_members,
            pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS), sorted(removed))


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    applications = pd.read_csv(params["applications_path"])
    candidate_events, candidate_members, audit, removed = calibrate(
        events, members, applications)
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "explained_projection_event_audit.csv",
        "metrics": out.out / "metrics.json",
    }
    candidate_events.to_csv(outputs["events"], index=False)
    candidate_members.to_csv(outputs["members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    before = _summary(events)
    after = _summary(candidate_events)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(events)),
        "producer_proposals_audited": int(
            applications.proposal_id.astype(str).nunique()),
        "events_removed_as_explained_projections": int(len(removed)),
        "removed_event_ids": removed,
        "labels_changed": False, "unclaimed_changed": False,
        "before": before, "after": after,
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

