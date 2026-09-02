"""Calibrate terminal convergence of a newly recovered companion.

When a genuinely new identity is continuous on an independently visible raw
branch, then returns to the established owner only during the last few frames
of that branch at a complete isolated convergence, the terminal owner change
is not evidence of a biological swap.  Discovery proposals are field-wide and
target-free. Freshness is derived from the current owner/physical-track history:
the name must begin with, and remain exclusive to, that raw track. No historical
label stack or pre-acceptance identity set is consulted.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _ids(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return {int(item) for item in re.findall(r"\d+", str(value))}


def _runs(group: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        if not bool(row.physically_visible):
            continue
        frame, owner = int(row.frame), int(row.candidate_owner)
        if owner <= 0:
            continue
        if (rows and rows[-1]["owner"] == owner
                and frame == rows[-1]["last_frame"] + 1):
            rows[-1]["last_frame"] = frame
            rows[-1]["frames"].append(frame)
        else:
            rows.append({
                "owner": owner, "first_frame": frame,
                "last_frame": frame, "frames": [frame]})
    return rows


def _truth(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def discover_current_terminal_convergences(
        points: pd.DataFrame, encounter_frames: pd.DataFrame,
        reconnections: pd.DataFrame | None = None,
        minimum_fresh_owner_frames: int = 8,
        maximum_terminal_owner_frames: int = 3,
        minimum_separable_frames: int = 3,
        minimum_companion_observations: int = 8,
        minimum_companion_strong_observations: int = 5,
        minimum_survivor_tail_frames: int = 20,
        minimum_owner_anchor_frames: int = 8,
        require_exclusive_pair: bool = True,
        require_no_companion_reconnection: bool = True,
        ) -> pd.DataFrame:
    """Find resolved convergences from the current final owner history alone."""
    visible = points[_truth(points.physically_visible)].copy()
    if visible.empty or encounter_frames.empty:
        return pd.DataFrame()
    visible["track_id"] = visible.track_id.astype(int)
    visible["frame"] = visible.frame.astype(int)
    visible["candidate_owner"] = visible.candidate_owner.astype(int)
    encounters = encounter_frames.copy()
    encounters["track_a"] = encounters.track_a.astype(int)
    encounters["track_b"] = encounters.track_b.astype(int)
    encounters["frame"] = encounters.frame.astype(int)
    encounters["left"] = np.minimum(encounters.track_a, encounters.track_b)
    encounters["right"] = np.maximum(encounters.track_a, encounters.track_b)
    if "separable" in encounters:
        encounters["separable_bool"] = _truth(encounters.separable)
    else:
        encounters["separable_bool"] = False

    partners: dict[int, set[int]] = {}
    for row in encounters[["left", "right"]].drop_duplicates().itertuples(
            index=False):
        partners.setdefault(int(row.left), set()).add(int(row.right))
        partners.setdefault(int(row.right), set()).add(int(row.left))
    reconnected: set[int] = set()
    if (reconnections is not None and len(reconnections)
            and "track_before" in reconnections):
        reconnected = set(map(
            int, reconnections.track_before.dropna().astype(int)))

    owner_frames = {
        int(owner): set(group.frame.astype(int))
        for owner, group in visible[visible.candidate_owner > 0].groupby(
            "candidate_owner")}
    owner_tracks = {
        int(owner): set(group.track_id.astype(int))
        for owner, group in visible[visible.candidate_owner > 0].groupby(
            "candidate_owner")}
    rows: list[dict] = []
    for companion, group in visible.groupby("track_id", sort=True):
        companion = int(companion)
        group = group.sort_values("frame")
        runs = _runs(group)
        if len(runs) != 2:
            continue
        fresh, terminal = runs
        fresh_owner = int(fresh["owner"])
        established = int(terminal["owner"])
        first_visible = int(group.frame.min())
        last_visible = int(group.frame.max())
        if (fresh_owner == established
                or int(fresh["first_frame"]) != first_visible
                or int(fresh["last_frame"]) + 1
                != int(terminal["first_frame"])
                or int(terminal["last_frame"]) != last_visible
                or len(fresh["frames"]) < minimum_fresh_owner_frames
                or not 1 <= len(terminal["frames"])
                <= maximum_terminal_owner_frames):
            continue

        # A currently accepted name is "fresh" when its complete physical-owner
        # history starts with this raw track and never belongs to another one.
        # This remains true after the name itself is part of the accepted base.
        if owner_tracks.get(fresh_owner, set()) != {companion}:
            continue
        if min(owner_frames.get(fresh_owner, {-1})) != first_visible:
            continue
        fresh_rows = group[
            group.frame.astype(int).isin(fresh["frames"])]
        observed = fresh_rows[fresh_rows.state.astype(str) == "observed"]
        strong = (_truth(observed.strong).sum()
                  if "strong" in observed else len(observed))
        if (len(observed) < minimum_companion_observations
                or int(strong) < minimum_companion_strong_observations):
            continue

        terminal_frames = set(map(int, terminal["frames"]))
        terminal_encounters = encounters[
            encounters.frame.isin(terminal_frames)
            & ((encounters.track_a == companion)
               | (encounters.track_b == companion))]
        terminal_partners = set(
            np.where(terminal_encounters.track_a == companion,
                     terminal_encounters.track_b,
                     terminal_encounters.track_a).astype(int))
        if len(terminal_partners) != 1:
            continue
        survivor = int(next(iter(terminal_partners)))
        pair = encounters[
            (encounters.left == min(companion, survivor))
            & (encounters.right == max(companion, survivor))]
        if (set(terminal_encounters.frame.astype(int)) != terminal_frames
                or bool(terminal_encounters.separable_bool.any())
                or int(pair.separable_bool.sum()) < minimum_separable_frames):
            continue
        survivor_rows = visible[visible.track_id == survivor]
        terminal_survivor = survivor_rows[
            survivor_rows.frame.isin(terminal_frames)]
        if (set(terminal_survivor.frame.astype(int)) != terminal_frames
                or not terminal_survivor.candidate_owner.astype(int).eq(
                    established).all()
                or int(survivor_rows.frame.max()) - last_visible
                < minimum_survivor_tail_frames):
            continue
        if (require_exclusive_pair
                and (partners.get(companion, set()) != {survivor}
                     or partners.get(survivor, set()) != {companion})):
            continue
        if require_no_companion_reconnection and companion in reconnected:
            continue
        established_frames = owner_frames.get(established, set())
        pre_anchor = len({frame for frame in established_frames
                          if frame < first_visible})
        post_anchor = len({frame for frame in established_frames
                           if frame > last_visible})
        if (pre_anchor < minimum_owner_anchor_frames
                or post_anchor < minimum_owner_anchor_frames):
            continue
        rows.append({
            "proposal_id": f"C{len(rows) + 1:04d}",
            "companion_track": companion,
            "survivor_track": survivor,
            "shared_owner": established,
            "fresh_owner": fresh_owner,
            "fresh_owner_frames": len(fresh["frames"]),
            "terminal_owner_frames": len(terminal["frames"]),
            "discovery_status": "eligible",
            "discovery_basis": "current_final_owner_history",
        })
    return pd.DataFrame(rows)


def resolved_terminal_convergences(
        events: pd.DataFrame, transactions: pd.DataFrame,
        candidate_points: pd.DataFrame, encounter_frames: pd.DataFrame,
        reconnections: pd.DataFrame | None = None,
        minimum_fresh_owner_frames: int = 8,
        maximum_terminal_owner_frames: int = 3,
        minimum_separable_frames: int = 3,
        minimum_companion_observations: int = 8,
        minimum_companion_strong_observations: int = 5,
        minimum_survivor_tail_frames: int = 20,
        minimum_owner_anchor_frames: int = 8,
        require_exclusive_pair: bool = True,
        require_no_companion_reconnection: bool = True,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Suppress complete convergences recognizable from current output."""
    proposals = discover_current_terminal_convergences(
        candidate_points, encounter_frames, reconnections,
        minimum_fresh_owner_frames=minimum_fresh_owner_frames,
        maximum_terminal_owner_frames=maximum_terminal_owner_frames,
        minimum_separable_frames=minimum_separable_frames,
        minimum_companion_observations=minimum_companion_observations,
        minimum_companion_strong_observations=
            minimum_companion_strong_observations,
        minimum_survivor_tail_frames=minimum_survivor_tail_frames,
        minimum_owner_anchor_frames=minimum_owner_anchor_frames,
        require_exclusive_pair=require_exclusive_pair,
        require_no_companion_reconnection=
            require_no_companion_reconnection)
    if proposals.empty:
        return events.copy(), transactions.copy(), pd.DataFrame()
    eligible = proposals[proposals.discovery_status == "eligible"]
    audits: list[dict] = []
    event_ids: set[str] = set()
    transaction_indices: set[int] = set()
    for proposal in eligible.itertuples(index=False):
        companion = int(proposal.companion_track)
        survivor = int(proposal.survivor_track)
        established = int(proposal.shared_owner)
        group = candidate_points[
            candidate_points.track_id == companion].sort_values("frame")
        runs = _runs(group)
        base = {
            "proposal_id": str(proposal.proposal_id),
            "companion_track": companion, "survivor_track": survivor,
            "established_owner": established, "fresh_owner": 0,
            "fresh_owner_frames": 0, "terminal_owner_frames": 0,
            "suppressed_event_id": "", "suppressed_transactions": 0,
            "status": "rejected", "reason": "",
        }
        if len(runs) < 2:
            audits.append({**base, "reason": "no_terminal_owner_transition"})
            continue
        fresh, terminal = runs[-2], runs[-1]
        fresh_owner = int(proposal.fresh_owner)
        base.update({
            "fresh_owner": fresh_owner,
            "fresh_owner_frames": len(fresh["frames"]),
            "terminal_owner_frames": len(terminal["frames"]),
        })
        if int(terminal["owner"]) != established:
            audits.append({**base, "reason": "terminal_owner_not_established"})
            continue
        if len(fresh["frames"]) < minimum_fresh_owner_frames:
            audits.append({**base, "reason": "fresh_owner_run_too_short"})
            continue
        if not 1 <= len(terminal["frames"]) <= maximum_terminal_owner_frames:
            audits.append({**base, "reason": "terminal_owner_run_out_of_range"})
            continue
        visible_last = int(group[group.physically_visible].frame.max())
        if int(terminal["last_frame"]) != visible_last:
            audits.append({**base, "reason": "transition_not_terminal_on_track"})
            continue
        if int(terminal["first_frame"]) != int(fresh["last_frame"]) + 1:
            audits.append({**base, "reason": "owner_runs_not_contiguous"})
            continue

        matches = []
        for event in events.itertuples(index=False):
            tracks = _ids(event.physical_tracks)
            before = _ids(event.accepted_before)
            after = _ids(event.accepted_after)
            if ({companion, survivor}.issubset(tracks)
                    and fresh_owner in before and established in after
                    and int(event.first_frame) <= int(fresh["last_frame"])
                    and int(event.last_frame) >= int(terminal["first_frame"])):
                matches.append(event)
        if len(matches) != 1:
            audits.append({
                **base,
                "reason": ("no_matching_event" if not matches
                           else "multiple_matching_events")})
            continue
        event = matches[0]
        event_ids.add(str(event.event_id))
        transaction_count = 0
        for index, transaction in transactions.iterrows():
            tracks = _ids(transaction.get("physical_tracks", ""))
            before = _ids(transaction.get("accepted_before", ""))
            after = _ids(transaction.get("accepted_after", ""))
            if (companion in tracks and fresh_owner in before
                    and established in after
                    and int(transaction.get("first_frame", -1))
                    == int(fresh["last_frame"])):
                transaction_indices.add(int(index))
                transaction_count += 1
        if transaction_count != 1:
            event_ids.discard(str(event.event_id))
            transaction_indices -= {
                index for index in transaction_indices
                if companion in _ids(
                    transactions.loc[index].get("physical_tracks", ""))}
            audits.append({**base, "reason": "terminal_transaction_not_unique"})
            continue
        audits.append({
            **base, "suppressed_event_id": str(event.event_id),
            "suppressed_transactions": transaction_count,
            "status": "suppressed", "reason": "resolved_terminal_convergence"})

    filtered_events = events[
        ~events.event_id.astype(str).isin(event_ids)].copy()
    filtered_transactions = transactions.drop(
        index=sorted(transaction_indices)).copy()
    return filtered_events, filtered_transactions, pd.DataFrame(audits)
