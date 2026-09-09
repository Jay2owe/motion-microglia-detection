"""Join score fragments proved to belong to one durable physical encounter."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd


FORBIDDEN = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "well", "review_case", "case_id",
    "identity_target", "owner_target", "track_target", "frame_target",
    "event_target", "forced_", "include_", "exclude_", "allowed_",
)
PAIR_FAMILIES = frozenset({"separable_cells_merged",
                           "reference_fragmentation"})
FAMILY_DISRUPTION = {"reference_fragmentation": 10.0}


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "fragmented encounter-lineage calibration received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "fragmented encounter-lineage calibration must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _integers(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return set(map(int, re.findall(r"\d+", str(value))))


def _owners(event) -> set[int]:
    return _integers(event.accepted_before) | _integers(event.accepted_after)


def _tracks(event) -> set[int]:
    return _integers(event.physical_tracks)


def _positive_owner_set(group: pd.DataFrame) -> set[int]:
    return set(group.loc[group.candidate_owner.astype(int).gt(0),
                         "candidate_owner"].astype(int))


def _track_profiles(points: pd.DataFrame) -> dict[int, dict]:
    profiles: dict[int, dict] = {}
    for track, group in points.groupby("track_id", sort=True):
        visible = group[group.physically_visible.map(_truth)] \
            if "physically_visible" in group else group[
                group.state.astype(str).isin({"observed", "latent_visible"})]
        profiles[int(track)] = {
            "first": int(group.frame.min()),
            "last": int(group.frame.max()),
            "span": int(group.frame.max()) - int(group.frame.min()) + 1,
            "visible": visible,
            "owners": _positive_owner_set(visible),
            "median_radius": float(visible.radius_px.astype(float).median())
                if len(visible) else 0.0,
        }
    return profiles


def _event_summary(events: pd.DataFrame) -> dict:
    owner = events[events.family.astype(str).eq("identity_swap_or_takeover")]
    return {
        "events": int(len(events)),
        "owner_change_events": int(len(owner)),
        "owner_change_disruption_burden": float(
            owner.disruption_score.astype(float).sum()),
        "high_disruption_events": int(
            events.disruption_score.astype(float).ge(50.0).sum()),
        "total_disruption_burden": float(
            events.disruption_score.astype(float).sum()),
        "events_by_family": {
            str(key): int(value)
            for key, value in events.groupby("family").size().items()},
    }


def discover(events: pd.DataFrame, points: pd.DataFrame, movie_frames: int,
             params: dict) -> pd.DataFrame:
    """Audit every temporally adjacent compatible event pair."""
    assert_target_free(params)
    maximum_gap = max(1, int(math.ceil(movie_frames * float(params.get(
        "maximum_inter_event_gap_movie_fraction", 0.12)))))
    minimum_durable_span = max(2, int(math.ceil(movie_frames * float(
        params.get("minimum_shared_track_span_movie_fraction", 0.50)))))
    maximum_fragment_span = max(1, int(math.ceil(movie_frames * float(
        params.get("maximum_fragment_track_span_movie_fraction", 0.20)))))
    minimum_bridge_owner = float(params.get(
        "minimum_bridge_owner_fraction", 0.80))
    minimum_short_fraction = float(params.get(
        "minimum_short_nonanchor_track_fraction", 0.75))
    maximum_distance_radii = float(params.get(
        "maximum_event_centre_distance_typical_radii", 4.0))
    visible = points[points.physically_visible.map(_truth)] \
        if "physically_visible" in points else points[
            points.state.astype(str).isin({"observed", "latent_visible"})]
    nonfloor = visible[visible.radius_px.astype(float).gt(
        float(visible.radius_px.astype(float).min()) + 1e-9)]
    typical_radius = float(nonfloor.radius_px.astype(float).median()) \
        if len(nonfloor) else float(visible.radius_px.astype(float).median())
    profiles = _track_profiles(points)
    ordered = events.sort_values(
        ["first_frame", "last_frame", "event_id"]).reset_index(drop=True)
    rows: list[dict] = []
    for left_index, left in enumerate(ordered.itertuples(index=False)):
        for right in ordered.iloc[left_index + 1:].itertuples(index=False):
            gap = int(right.first_frame) - int(left.last_frame) - 1
            if gap < 0:
                continue
            if gap > maximum_gap:
                break
            families = {str(left.family), str(right.family)}
            owner_left, owner_right = _owners(left), _owners(right)
            tracks_left, tracks_right = _tracks(left), _tracks(right)
            shared = tracks_left & tracks_right
            centre_distance = float(np.hypot(
                float(left.x) - float(right.x),
                float(left.y) - float(right.y)))
            reasons: list[str] = []
            if families != PAIR_FAMILIES:
                reasons.append("families_not_merge_plus_fragment")
            if owner_left != owner_right or len(owner_left) < 2:
                reasons.append("multi_owner_signature_not_equal")
            durable = [track for track in shared
                       if profiles.get(track, {}).get("span", 0)
                       >= minimum_durable_span]
            if len(durable) != 1:
                reasons.append("unique_durable_shared_track_missing")
            anchor = durable[0] if len(durable) == 1 else 0
            anchor_owner = 0
            bridge_fraction = 0.0
            if anchor:
                profile = profiles[anchor]
                if len(profile["owners"]) != 1:
                    reasons.append("shared_track_owner_not_stable")
                else:
                    anchor_owner = next(iter(profile["owners"]))
                    if anchor_owner not in owner_left:
                        reasons.append("shared_track_owner_outside_signature")
                bridge = profile["visible"]
                bridge = bridge[bridge.frame.astype(int).between(
                    int(left.last_frame) + 1, int(right.first_frame) - 1)]
                bridge_fraction = float(
                    bridge.candidate_owner.astype(int).eq(anchor_owner).mean()) \
                    if len(bridge) and anchor_owner else 0.0
                if bridge_fraction < minimum_bridge_owner:
                    reasons.append("shared_track_not_owner_stable_in_bridge")
            union_tracks = (tracks_left | tracks_right) - ({anchor} if anchor else set())
            short_fraction = float(np.mean([
                profiles.get(track, {}).get("span", movie_frames + 1)
                <= maximum_fragment_span for track in union_tracks])) \
                if union_tracks else 0.0
            if short_fraction < minimum_short_fraction:
                reasons.append("nonanchor_references_not_fragmentary")
            distance_radii = centre_distance / max(typical_radius, 1.0)
            if distance_radii > maximum_distance_radii:
                reasons.append("event_centres_too_distant")
            rows.append({
                "left_event_id": str(left.event_id),
                "right_event_id": str(right.event_id),
                "left_family": str(left.family),
                "right_family": str(right.family),
                "inter_event_gap_frames": gap,
                "maximum_inter_event_gap_frames": maximum_gap,
                "owner_signature": "|".join(map(str, sorted(owner_left))),
                "shared_tracks": "|".join(map(str, sorted(shared))),
                "durable_shared_track": anchor,
                "durable_shared_track_span": profiles.get(anchor, {}).get(
                    "span", 0),
                "durable_shared_owner": anchor_owner,
                "bridge_owner_fraction": bridge_fraction,
                "nonanchor_track_count": len(union_tracks),
                "short_nonanchor_fraction": short_fraction,
                "typical_visible_radius_px": typical_radius,
                "event_centre_distance_px": centre_distance,
                "event_centre_distance_typical_radii": distance_radii,
                "eligible": not reasons,
                "reason": ("durable_shared_encounter_lineage"
                           if not reasons else "|".join(reasons)),
            })
    return pd.DataFrame(rows)


def _merge_member_groups(members: pd.DataFrame, keep: str,
                         absorbed: str) -> pd.DataFrame:
    result = members.copy()
    mask = result.event_id.astype(str).isin({keep, absorbed})
    merged = result[mask].copy()
    result = result[~mask].copy()
    merged["event_id"] = keep
    source_keys = list(dict.fromkeys(
        zip(merged.source_family.astype(str),
            merged.source_first_frame.astype(str),
            merged.source_last_frame.astype(str),
            merged.source_x.astype(str), merged.source_y.astype(str))))
    source_map = {key: index + 1 for index, key in enumerate(source_keys)}
    merged["source_id"] = [source_map[key] for key in zip(
        merged.source_family.astype(str),
        merged.source_first_frame.astype(str),
        merged.source_last_frame.astype(str),
        merged.source_x.astype(str), merged.source_y.astype(str))]
    return pd.concat([result, merged], ignore_index=True)


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, movie_frames: int, params: dict):
    """Join eligible pairs conservatively and keep one active event."""
    audit = discover(events, points, movie_frames, params)
    candidate = events.copy()
    candidate_members = members.copy()
    eligible = audit[audit.eligible.astype(bool)] if len(audit) else audit
    used: set[str] = set()
    applied_pairs: list[dict] = []
    for row in eligible.sort_values(
            ["inter_event_gap_frames", "left_event_id",
             "right_event_id"]).itertuples(index=False):
        left_id, right_id = str(row.left_event_id), str(row.right_event_id)
        if left_id in used or right_id in used:
            applied_pairs.append({
                "left_event_id": left_id, "right_event_id": right_id,
                "applied": False, "reason": "overlapping_event_pair"})
            continue
        left = candidate[candidate.event_id.astype(str).eq(left_id)]
        right = candidate[candidate.event_id.astype(str).eq(right_id)]
        if len(left) != 1 or len(right) != 1:
            applied_pairs.append({
                "left_event_id": left_id, "right_event_id": right_id,
                "applied": False, "reason": "event_not_unique"})
            continue
        left, right = left.iloc[0], right.iloc[0]
        keep = left_id
        absorbed = right_id
        source = candidate_members[
            candidate_members.event_id.astype(str).isin({keep, absorbed})]
        merged_refs = sorted(set(source.source_ref.astype(str)))
        merged_tracks = sorted(set(map(int, re.findall(
            r"T(\d+)", "|".join(merged_refs)))))
        impact = float(max(float(left.impact), float(right.impact)))
        score = 0.65 * FAMILY_DISRUPTION["reference_fragmentation"] \
            + 0.35 * impact
        replacement = left.copy()
        replacement["family"] = "reference_fragmentation"
        replacement["first_frame"] = min(
            int(left.first_frame), int(right.first_frame))
        replacement["last_frame"] = max(
            int(left.last_frame), int(right.last_frame))
        weights = np.asarray([
            max(1, int(left.constituent_events)),
            max(1, int(right.constituent_events))], dtype=float)
        replacement["x"] = float(np.average(
            [float(left.x), float(right.x)], weights=weights))
        replacement["y"] = float(np.average(
            [float(left.y), float(right.y)], weights=weights))
        replacement["impact"] = impact
        replacement["disruption_score"] = score
        replacement["fidelity"] = 100.0 - score
        replacement["member_refs"] = "|".join(merged_refs)
        replacement["physical_tracks"] = "|".join(map(str, merged_tracks))
        replacement["constituent_events"] = int(
            left.constituent_events) + int(right.constituent_events)
        replacement["evidence"] = (
            f"durable shared-track encounter lineage; {left.evidence} ; "
            f"{right.evidence}")
        candidate = candidate[
            ~candidate.event_id.astype(str).isin({keep, absorbed})]
        candidate = pd.concat(
            [candidate, replacement.to_frame().T], ignore_index=True)
        candidate_members = _merge_member_groups(
            candidate_members, keep, absorbed)
        used.update({keep, absorbed})
        applied_pairs.append({
            "left_event_id": left_id, "right_event_id": right_id,
            "applied": True,
            "reason": "joined_as_reference_fragmentation"})
    candidate = candidate.sort_values(
        ["first_frame", "last_frame", "event_id"]).reset_index(drop=True)
    candidate["impact_rank"] = np.arange(1, len(candidate) + 1)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "event_pairs_audited": int(len(audit)),
        "eligible_event_pairs": int(len(eligible)),
        "applied_event_joins": int(sum(
            bool(row["applied"]) for row in applied_pairs)),
        "before": _event_summary(events),
        "after": _event_summary(candidate),
    }
    return (candidate, candidate_members, audit,
            pd.DataFrame(applied_pairs), summary)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    points = pd.read_csv(params["points_path"])
    candidate, candidate_members, audit, applications, summary = calibrate(
        events, members, points, int(params["movie_frames"]), params)
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "fragmented_encounter_lineage_audit.csv",
        "applications": out.out / "fragmented_encounter_lineage_applications.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / "95_A3.tif",
        "unclaimed": out.out / "95_A3_unclaimed_original_ids.tif",
    }
    candidate.to_csv(outputs["events"], index=False)
    candidate_members.to_csv(outputs["members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    applications.to_csv(outputs["applications"], index=False)
    shutil.copyfile(params["labels_path"], outputs["labels"])
    shutil.copyfile(params["unclaimed_path"], outputs["unclaimed"])
    summary["label_bytes_unchanged_by_calibration"] = (
        outputs["labels"].read_bytes() == Path(params["labels_path"]).read_bytes())
    summary["unclaimed_bytes_unchanged_by_calibration"] = (
        outputs["unclaimed"].read_bytes()
        == Path(params["unclaimed_path"]).read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

