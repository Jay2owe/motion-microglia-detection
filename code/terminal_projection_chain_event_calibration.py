"""Exclude exact producer-proved projection morphology from active evidence.

The disruption detector may describe an intentionally detached cell projection
as a separate raw core. This score-only calibration removes a constituent only
when its physical-track pair, owner, frame interval, and component creation are
all proved by a complete target-free producer application ledger. It never
edits labels or unclaimed foreground.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

import bracketed_mixed_owner_flash as flash


MORPHOLOGY_FAMILIES = {"separable_cells_merged", "single_cell_multi_core"}
AUDIT_COLUMNS = [
    "set_name", "event_id", "source_id", "source_family", "source_tracks",
    "source_first", "source_last", "source_owners", "matched_proposal_id",
    "proposal_child", "proposal_parent", "proposal_owner",
    "all_proposal_frames_applied", "explained_components", "eligible",
    "reason",
]


def _numbers(value) -> set[int]:
    if pd.isna(value):
        return set()
    return set(map(int, re.findall(r"\d+", str(value))))


def _source_id(value) -> str:
    """Normalise CSV source identifiers that pandas may infer as floats."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


def _track_refs(group: pd.DataFrame) -> set[int]:
    result = set()
    for value in group.source_ref.astype(str):
        match = re.fullmatch(r"T(\d+)", value)
        if match:
            result.add(int(match.group(1)))
    return result


def _proposals(applications: pd.DataFrame) -> list[dict]:
    proposals = []
    for proposal_id, group in applications.groupby("proposal_id", sort=True):
        applied = group.applied.astype(str).str.lower().eq("true")
        proposals.append({
            "proposal_id": str(proposal_id),
            "child": int(group.child_track.astype(int).iloc[0]),
            "parent": int(group.parent_track.astype(int).iloc[0]),
            "owner": int(group.anchor_owner.astype(int).iloc[0]),
            "frames": set(group.frame.astype(int).tolist()),
            "all_applied": bool(applied.all()),
            "explained": int(group.explained_projection_components.astype(
                int).sum()),
        })
    return proposals


def constituent_summary(members: pd.DataFrame, high_impact: float = 50.0) -> dict:
    sources = members.drop_duplicates(["event_id", "source_id"]).copy()
    by_family = {}
    for family, group in sources.groupby("source_family", sort=True):
        by_family[str(family)] = {
            "constituents": int(len(group)),
            "impact_burden": float(group.source_impact.astype(float).sum()),
        }
    return {
        "constituents": int(len(sources)),
        "high_impact_constituents": int(
            (sources.source_impact.astype(float) >= high_impact).sum()),
        "constituent_impact_burden": float(
            sources.source_impact.astype(float).sum()),
        "by_family": by_family,
    }


def event_summary(events: pd.DataFrame) -> dict:
    """Return the aggregate contract consumed by the accepted scorer."""
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
        "events_by_family": {
            str(key): int(value) for key, value in
            events.family.astype(str).value_counts().sort_index().items()},
    }


def _refresh_event_metadata(events: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    refreshed = []
    for event in events.itertuples(index=False):
        group = members[members.event_id.astype(str).eq(str(event.event_id))]
        if group.empty:
            continue
        row = event._asdict()
        refs = sorted(set(group.source_ref.astype(str)))
        tracks = sorted(_track_refs(group))
        sources = group.drop_duplicates("source_id")
        row["member_refs"] = "|".join(refs)
        row["physical_tracks"] = "|".join(map(str, tracks))
        row["constituent_events"] = int(sources.source_id.nunique())
        row["evidence"] = " ; ".join(dict.fromkeys(
            sources.source_evidence.astype(str).tolist()))
        refreshed.append(row)
    result = pd.DataFrame(refreshed, columns=events.columns)
    return result.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              proposals: list[dict], set_name: str):
    removed_keys = set()
    audit_rows = []
    total_sources = 0
    for (event_id, source_id), group in members.groupby(
            ["event_id", "source_id"], sort=True):
        total_sources += 1
        family = str(group.source_family.iloc[0])
        tracks = _track_refs(group)
        first = int(group.source_first_frame.astype(int).min())
        last = int(group.source_last_frame.astype(int).max())
        owners = set()
        for field in ("source_accepted_before", "source_accepted_after"):
            for value in group[field]:
                owners |= _numbers(value)
        matches = [proposal for proposal in proposals
                   if tracks == {proposal["child"], proposal["parent"]}]
        reasons = []
        if family not in MORPHOLOGY_FAMILIES:
            reasons.append("not_projection_morphology")
        if len(matches) != 1:
            reasons.append("producer_proposal_not_unique")
            match = None
        else:
            match = matches[0]
            source_frames = set(range(first, last + 1))
            if not source_frames <= match["frames"]:
                reasons.append("source_frames_not_fully_applied")
            if owners != {match["owner"]}:
                reasons.append("source_owner_not_conserved")
            if not match["all_applied"]:
                reasons.append("producer_proposal_not_fully_applied")
            if match["explained"] <= 0:
                reasons.append("producer_did_not_explain_component")
        eligible = not reasons
        if eligible:
            removed_keys.add((str(event_id), _source_id(source_id)))
        if match is not None or tracks & {
                value for proposal in proposals
                for value in (proposal["child"], proposal["parent"])}:
            audit_rows.append({
                "set_name": set_name, "event_id": str(event_id),
                "source_id": str(source_id), "source_family": family,
                "source_tracks": "|".join(map(str, sorted(tracks))),
                "source_first": first, "source_last": last,
                "source_owners": "|".join(map(str, sorted(owners))),
                "matched_proposal_id": match["proposal_id"] if match else "",
                "proposal_child": match["child"] if match else 0,
                "proposal_parent": match["parent"] if match else 0,
                "proposal_owner": match["owner"] if match else 0,
                "all_proposal_frames_applied": match["all_applied"]
                    if match else False,
                "explained_components": match["explained"] if match else 0,
                "eligible": eligible,
                "reason": "producer_proved_projection_morphology" if eligible
                    else "|".join(dict.fromkeys(reasons)),
            })
    keys = pd.Series(zip(
        members.event_id.astype(str), members.source_id.map(_source_id)),
        index=members.index)
    calibrated_members = members[~keys.isin(removed_keys)].copy()
    calibrated_events = _refresh_event_metadata(events, calibrated_members)
    return (calibrated_events, calibrated_members,
            pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
            sorted(removed_keys), total_sources)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    flash.assert_target_free(params)
    applications = pd.read_csv(params["applications_path"])
    proposals = _proposals(applications)
    baseline_events = pd.read_csv(params["baseline_events_path"])
    baseline_members = pd.read_csv(params["baseline_members_path"])
    candidate_events = pd.read_csv(params["candidate_events_path"])
    candidate_members = pd.read_csv(params["candidate_members_path"])
    baseline_before = constituent_summary(baseline_members)
    candidate_before = constituent_summary(candidate_members)
    be, bm, ba, br, baseline_sources = calibrate(
        baseline_events, baseline_members, proposals, "baseline")
    ce, cm, ca, cr, candidate_sources = calibrate(
        candidate_events, candidate_members, proposals, "candidate")
    baseline_after = constituent_summary(bm)
    candidate_after = constituent_summary(cm)
    audit = pd.concat([ba, ca], ignore_index=True)
    outputs = {
        "baseline_events": out.out / "baseline_disruptive_events.csv",
        "baseline_members": out.out / "baseline_event_members.csv",
        "candidate_events": out.out / "candidate_disruptive_events.csv",
        "candidate_members": out.out / "candidate_event_members.csv",
        "audit": out.out / "terminal_projection_event_calibration_audit.csv",
        "metrics": out.out / "metrics.json",
    }
    be.to_csv(outputs["baseline_events"], index=False)
    bm.to_csv(outputs["baseline_members"], index=False)
    ce.to_csv(outputs["candidate_events"], index=False)
    cm.to_csv(outputs["candidate_members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "producer_proposals_audited": int(len(proposals)),
        "baseline_constituents_audited": int(baseline_sources),
        "candidate_constituents_audited": int(candidate_sources),
        "baseline_removed_projection_sources": int(len(br)),
        "candidate_removed_projection_sources": int(len(cr)),
        "baseline_removed_keys": [list(key) for key in br],
        "candidate_removed_keys": [list(key) for key in cr],
        "baseline_before_calibration": baseline_before,
        "candidate_before_calibration": candidate_before,
        "baseline": baseline_after,
        "candidate": candidate_after,
        "before": event_summary(candidate_events),
        "after": event_summary(ce),
        "labels_changed": False,
        "unclaimed_changed": False,
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
