"""Remove false merge alerts proved to be terminal companion projections.

The calibration consumes complete-field producer audits. It accepts no event,
track, identity, frame, coordinate, region, or review-case selector.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd


FORBIDDEN = (
    "event_id", "track_id", "identity_id", "owner_id", "frame_id",
    "coordinate", "region", "review_case", "case_id", "target_",
    "forced_", "include_", "exclude_",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "terminal projection calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal projection calibration must be field-wide")


def _tracks(value: object) -> frozenset[int]:
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def _owners(value: object) -> frozenset[int]:
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def _event_summary(events: pd.DataFrame) -> dict:
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


def _read_optional_audit(path: str | Path) -> pd.DataFrame:
    """Read a producer audit, treating a zero-byte no-op audit as empty."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              params: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, dict]:
    """Filter only merge events exactly explained by an applied proposal."""
    assert_target_free(params)
    applied_ids = set(applications.loc[
        applications.applied.astype(str).str.lower().isin({"1", "true", "yes"}),
        "proposal_id"].astype(str)) if len(applications) else set()
    eligible = proposals[
        proposals.eligible.astype(str).str.lower().isin({"1", "true", "yes"})
        & proposals.proposal_id.astype(str).isin(applied_ids)] \
        if len(proposals) else proposals
    audit_rows = []
    removed: set[str] = set()
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_tracks = _tracks(event.physical_tracks)
        event_owners = _owners(event.accepted_before) | _owners(
            event.accepted_after)
        matches = []
        if str(event.family) == "separable_cells_merged":
            for proposal in eligible.itertuples(index=False):
                pair = frozenset({int(proposal.newborn_track),
                                  int(proposal.resident_track)})
                if (event_tracks == pair
                        and int(event.first_frame) >= int(proposal.boundary_frame)
                        and int(event.last_frame) <= max(
                            int(proposal.newborn_last_frame),
                            int(proposal.resident_last_frame))
                        and event_owners == {int(proposal.resident_owner)}
                        and int(proposal.qualified_two_core_frames) >= int(
                            params.get("minimum_qualified_two_core_frames", 2))):
                    matches.append(str(proposal.proposal_id))
        reason = ""
        if str(event.family) != "separable_cells_merged":
            reason = "not_separable_merge"
        elif not matches:
            reason = "no_exact_applied_projection_topology"
        elif len(matches) > 1:
            reason = "ambiguous_projection_topology"
        else:
            removed.add(event_id)
            reason = "calibrated_terminal_companion_projection"
        audit_rows.append({
            "event_id": event_id, "family": str(event.family),
            "physical_tracks": str(event.physical_tracks),
            "matched_proposals": "|".join(matches),
            "calibrated_as_non_disruptive": len(matches) == 1,
            "reason": reason,
        })
    calibrated = events[
        ~events.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    calibrated = calibrated.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated["impact_rank"] = np.arange(1, len(calibrated) + 1)
    calibrated_members = members[
        ~members.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(len(events)),
        "applied_projection_proposals": int(len(eligible)),
        "events_calibrated_non_disruptive": int(len(removed)),
        "before": _event_summary(events),
        "after": _event_summary(calibrated),
    }
    return calibrated, calibrated_members, audit, summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    proposals = _read_optional_audit(params["projection_proposals_path"])
    applications = _read_optional_audit(params["projection_applications_path"])
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, proposals, applications, params)
    outputs = {
        "candidate_events": out.out / "candidate_disruptive_events.csv",
        "candidate_event_members": out.out / "candidate_event_members.csv",
        "audit": out.out / "terminal_projection_calibration_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{params['stem']}.tif",
        "unclaimed": out.out / f"{params['stem']}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["candidate_events"], index=False)
    calibrated_members.to_csv(outputs["candidate_event_members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    shutil.copyfile(params["labels_path"], outputs["labels"])
    shutil.copyfile(params["unclaimed_path"], outputs["unclaimed"])
    summary["label_bytes_unchanged"] = (
        outputs["labels"].read_bytes() == Path(params["labels_path"]).read_bytes())
    summary["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes()
        == Path(params["unclaimed_path"]).read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
