"""Calibrate resolved owner changes created by dormant track bifurcations.

The producer's field-wide audit and the current final owner history are the only
inputs. Identity and track values are discovered opaque labels, never targets.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd


def _ids(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return {int(item) for item in re.findall(r"\d+", str(value))}


def _tracks(value: object) -> set[int]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return set(map(int, value))
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return set(map(int, parsed))
    except (SyntaxError, ValueError):
        pass
    return _ids(value)


def resolved_dormant_bridge_bifurcations(
        events: pd.DataFrame, transactions: pd.DataFrame,
        candidate_points: pd.DataFrame, application_audit: pd.DataFrame,
        minimum_tail_owner_fraction: float = 0.95,
        minimum_tail_owner_frames: int = 20,
        minimum_dormant_bridge_frames: int = 2,
        minimum_owner_tail_presence_fraction: float = 0.80,
        minimum_owner_tail_distance_radii: float = 3.0,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove only the bookkeeping transition of an applied resolved fork."""
    required = {
        "track_id", "transient_owner", "assigned_identity",
        "owner_run_first", "owner_run_last", "first_tail_visible_frame",
        "dormant_bridge_frames", "owner_tail_presence_fraction",
        "owner_tail_minimum_distance_radii", "application_status",
    }
    missing = required - set(application_audit.columns)
    if missing:
        raise ValueError(
            "dormant-bifurcation audit is missing: " + ", ".join(sorted(missing)))
    points = candidate_points.copy()
    points["track_id"] = points.track_id.astype(int)
    points["frame"] = points.frame.astype(int)
    points["candidate_owner"] = points.candidate_owner.astype(int)
    visible = points[points.physically_visible.astype(bool)] \
        if pd.api.types.is_bool_dtype(points.physically_visible) else points[
            points.physically_visible.astype(str).str.lower().isin(
                {"true", "1", "yes"})]
    owner_tracks = {
        int(owner): set(group.track_id.astype(int))
        for owner, group in visible[visible.candidate_owner > 0].groupby(
            "candidate_owner")}
    suppressed_events: set[str] = set()
    suppressed_transactions: set[int] = set()
    rows: list[dict] = []

    applied = application_audit[
        application_audit.application_status == "applied"]
    for proposal in applied.itertuples(index=False):
        track = int(proposal.track_id)
        prefix = int(proposal.transient_owner)
        branch = int(proposal.assigned_identity)
        base = {
            "proposal_id": str(proposal.proposal_id),
            "physical_track": track,
            "prefix_owner": prefix,
            "branch_owner": branch,
            "tail_owner_frames": 0,
            "tail_owner_fraction": 0.0,
            "suppressed_event_id": "",
            "suppressed_transactions": 0,
            "status": "rejected",
            "reason": "",
        }
        group = visible[visible.track_id == track].sort_values("frame")
        tail = group[group.frame >= int(proposal.first_tail_visible_frame)]
        branch_rows = tail[tail.candidate_owner == branch]
        tail_fraction = float(len(branch_rows) / max(len(tail), 1))
        base["tail_owner_frames"] = int(len(branch_rows))
        base["tail_owner_fraction"] = tail_fraction
        reasons: list[str] = []
        prefix_rows = group[
            group.frame.between(
                int(proposal.owner_run_first), int(proposal.owner_run_last))]
        if (len(prefix_rows) != int(proposal.owner_run_last)
                - int(proposal.owner_run_first) + 1
                or not prefix_rows.candidate_owner.eq(prefix).all()):
            reasons.append("owned_prefix_not_preserved")
        if int(proposal.dormant_bridge_frames) < minimum_dormant_bridge_frames:
            reasons.append("dormant_bridge_too_short")
        if len(branch_rows) < minimum_tail_owner_frames:
            reasons.append("branch_owner_run_too_short")
        if tail_fraction < minimum_tail_owner_fraction:
            reasons.append("branch_owner_tail_incomplete")
        if owner_tracks.get(branch, set()) != {track}:
            reasons.append("branch_owner_not_track_exclusive")
        if float(proposal.owner_tail_presence_fraction) \
                < minimum_owner_tail_presence_fraction:
            reasons.append("prefix_owner_does_not_continue_elsewhere")
        if float(proposal.owner_tail_minimum_distance_radii) \
                < minimum_owner_tail_distance_radii:
            reasons.append("prefix_owner_tail_not_spatially_distinct")
        candidates = events[
            (events.family == "identity_swap_or_takeover")
            & events.physical_tracks.map(_tracks).map(lambda value: value == {track})
            & events.accepted_before.map(_ids).map(lambda value: value == {prefix})
            & events.accepted_after.map(_ids).map(lambda value: value == {branch})]
        candidates = candidates[
            (candidates.first_frame.astype(int)
             <= int(proposal.owner_run_last))
            & (candidates.last_frame.astype(int)
               >= int(proposal.first_tail_visible_frame))]
        if len(candidates) != 1:
            reasons.append("resolved_transition_not_unique")
        transaction_matches = transactions[
            transactions.physical_tracks.map(_tracks).map(
                lambda value: value == {track})
            & transactions.accepted_before.map(_ids).map(
                lambda value: value == {prefix})
            & transactions.accepted_after.map(_ids).map(
                lambda value: value == {branch})]
        if len(transaction_matches) != 1:
            reasons.append("resolved_transaction_not_unique")
        if reasons:
            rows.append({**base, "reason": "|".join(reasons)})
            continue
        event = candidates.iloc[0]
        transaction_index = int(transaction_matches.index[0])
        suppressed_events.add(str(event.event_id))
        suppressed_transactions.add(transaction_index)
        rows.append({
            **base,
            "suppressed_event_id": str(event.event_id),
            "suppressed_transactions": 1,
            "status": "suppressed",
            "reason": "resolved_dormant_bridge_bifurcation",
        })
    filtered_events = events[
        ~events.event_id.astype(str).isin(suppressed_events)].copy()
    filtered_transactions = transactions[
        ~transactions.index.isin(suppressed_transactions)].copy()
    return (filtered_events.reset_index(drop=True),
            filtered_transactions.reset_index(drop=True),
            pd.DataFrame(rows))
