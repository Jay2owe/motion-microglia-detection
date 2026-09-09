"""Remove false identity-loss events with continuously conserved owner anchors.

The calibration is score-only, complete-field, and target-free.  A compound
merge/split alert is removed only when every participating owner has exactly
one durable physical anchor, all anchors remain correctly owned throughout
the event, the owner set is unchanged on both temporal sides, and no event
reference carries an outside owner.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import tifffile

import bracketed_mixed_owner_flash as flash
from resident_takeover import attach_owners


AUDIT_COLUMNS = [
    "event_id", "family", "first_frame", "last_frame", "event_owners",
    "event_tracks", "stable_anchor_tracks", "anchor_owner_map",
    "minimum_anchor_pre_support", "minimum_anchor_post_support",
    "minimum_anchor_event_presence", "minimum_label_owner_presence",
    "reference_positive_owners", "eligible", "reason",
]


def _numbers(value) -> set[int]:
    if pd.isna(value):
        return set()
    return set(map(int, re.findall(r"\d+", str(value))))


def _stable_anchors(points: pd.DataFrame, movie: int,
                    params: dict) -> dict[int, list[int]]:
    minimum_support = int(math.ceil(movie * float(params.get(
        "minimum_anchor_support_movie_fraction", 0.75))))
    minimum_purity = float(params.get("minimum_anchor_owner_purity", 0.98))
    result: dict[int, list[int]] = {}
    visible = points[points.physically_visible.astype(bool)]
    for track, group in visible.groupby("track_id", sort=True):
        positive = group[group.candidate_owner.astype(int).gt(0)]
        if positive.empty:
            continue
        counts = positive.candidate_owner.astype(int).value_counts()
        owner, support = int(counts.index[0]), int(counts.iloc[0])
        purity = float(support / len(positive))
        if support >= minimum_support and purity >= minimum_purity:
            result.setdefault(owner, []).append(int(track))
    return result


def calibrate(labels: np.ndarray, events: pd.DataFrame,
              members: pd.DataFrame, physical_points: pd.DataFrame,
              params: dict):
    """Return event tables with only mathematically proved false alerts removed."""
    flash.assert_target_free(params)
    points = attach_owners(physical_points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in points.groupby("track_id", sort=True)}
    movie = int(len(labels))
    anchors = _stable_anchors(points, movie, params)
    maximum_duration = int(math.ceil(movie * float(params.get(
        "maximum_event_movie_fraction", 0.15))))
    minimum_owners = int(params.get("minimum_conserved_owners", 2))
    minimum_side_support = int(math.ceil(movie * float(params.get(
        "minimum_anchor_side_support_movie_fraction", 0.05))))
    rows = []
    removed = set()
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        first, last = int(event.first_frame), int(event.last_frame)
        before = _numbers(event.accepted_before)
        after = _numbers(event.accepted_after)
        owners = before | after
        tracks = _numbers(event.physical_tracks)
        reasons = []
        if str(event.family) != "merge_split_identity_loss":
            reasons.append("not_merge_split_identity_loss")
        if before != after or len(owners) < minimum_owners:
            reasons.append("owner_set_not_conserved")
        if last - first + 1 > maximum_duration:
            reasons.append("event_too_long")
        anchor_map = {}
        for owner in sorted(owners):
            candidates = anchors.get(int(owner), [])
            if len(candidates) == 1:
                anchor_map[int(owner)] = int(candidates[0])
        if set(anchor_map) != owners:
            reasons.append("unique_durable_anchor_missing")
        if not set(anchor_map.values()) <= tracks:
            reasons.append("durable_anchor_not_in_event")

        pre_supports = []
        post_supports = []
        anchor_event_presence = []
        label_presence = []
        for owner, track in anchor_map.items():
            group = groups[track]
            visible_owned = group[
                group.physically_visible.astype(bool)
                & group.candidate_owner.astype(int).eq(owner)]
            pre_supports.append(int((visible_owned.frame.astype(int) < first).sum()))
            post_supports.append(int((visible_owned.frame.astype(int) > last).sum()))
            event_frames = set(range(first, last + 1))
            owned_frames = set(visible_owned.frame.astype(int).tolist())
            anchor_event_presence.append(float(
                len(event_frames & owned_frames) / len(event_frames)))
            label_presence.append(float(np.mean([
                np.any(labels[frame] == owner) for frame in event_frames])))
        if pre_supports and min(pre_supports) < minimum_side_support:
            reasons.append("insufficient_pre_event_anchor_support")
        if post_supports and min(post_supports) < minimum_side_support:
            reasons.append("insufficient_post_event_anchor_support")
        if anchor_event_presence and min(anchor_event_presence) < 1.0:
            reasons.append("anchor_not_continuously_owned_in_event")
        if label_presence and min(label_presence) < 1.0:
            reasons.append("owner_label_absent_in_event")

        reference_owners = set()
        for track in tracks:
            group = groups.get(int(track), pd.DataFrame())
            if group.empty:
                continue
            interval = group[
                group.frame.astype(int).between(first, last)
                & group.physically_visible.astype(bool)]
            reference_owners |= set(interval[
                interval.candidate_owner.astype(int).gt(0)
            ].candidate_owner.astype(int))
        if not reference_owners <= owners:
            reasons.append("reference_carries_external_owner")

        event_members = members[members.event_id.astype(str).eq(event_id)]
        source_families = set(event_members.source_family.astype(str))
        if source_families & {
                "identity_swap_or_takeover", "temporary_owner_excursion",
                "unclaimed_visible_body"}:
            reasons.append("event_contains_unresolved_identity_source")
        eligible = not reasons
        if eligible:
            removed.add(event_id)
        rows.append({
            "event_id": event_id, "family": str(event.family),
            "first_frame": first, "last_frame": last,
            "event_owners": "|".join(map(str, sorted(owners))),
            "event_tracks": "|".join(map(str, sorted(tracks))),
            "stable_anchor_tracks": "|".join(map(str, sorted(
                anchor_map.values()))),
            "anchor_owner_map": "|".join(
                f"{owner}:{track}" for owner, track in sorted(anchor_map.items())),
            "minimum_anchor_pre_support": min(pre_supports, default=0),
            "minimum_anchor_post_support": min(post_supports, default=0),
            "minimum_anchor_event_presence": min(
                anchor_event_presence, default=0.0),
            "minimum_label_owner_presence": min(label_presence, default=0.0),
            "reference_positive_owners": "|".join(map(
                str, sorted(reference_owners))),
            "eligible": eligible,
            "reason": "conserved_multi_anchor_projection_encounter"
                if eligible else "|".join(dict.fromkeys(reasons)),
        })
    candidate_events = events[
        ~events.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    candidate_events["impact_rank"] = np.arange(1, len(candidate_events) + 1)
    candidate_members = members[
        ~members.event_id.astype(str).isin(removed)].copy()
    return (candidate_events, candidate_members,
            pd.DataFrame(rows, columns=AUDIT_COLUMNS), sorted(removed))


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


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Tuner-stage entry point used by fresh per-well scoring."""
    flash.assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    physical_points = pd.read_csv(params["physical_track_points_path"])
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    candidate_events, candidate_members, audit, removed = calibrate(
        labels, events, members, physical_points, params)
    outputs = {
        "labels": out.out / labels_path.name,
        "unclaimed": out.out / unclaimed_path.name,
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "conserved_multi_anchor_encounter_audit.csv",
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
        "before": event_summary(events), "after": event_summary(candidate_events),
        "label_bytes_unchanged": outputs["labels"].read_bytes()
            == labels_path.read_bytes(),
        "unclaimed_bytes_unchanged": outputs["unclaimed"].read_bytes()
            == unclaimed_path.read_bytes(),
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
