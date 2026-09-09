"""Re-score compound alerts caused partly by detached projection relays.

Discovery is complete-field and target-free. The rule changes only event
tables; accepted biological masks are never modified.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN = (
    "identity_target", "identity_id", "target_identity", "owner_target",
    "owner_id", "target_owner", "track_target", "track_id", "target_track",
    "frame_target", "frame_id", "target_frame", "coordinate", "event_target",
    "event_id", "target_event", "region", "review_case", "case_id",
    "forced_identity", "forced_interval", "include_track", "exclude_track",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError("projection-relay calibration received forbidden "
                         "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("projection-relay calibration must be field-wide")


def _integers(value: object) -> set[int]:
    return {int(item) for item in re.findall(r"\d+", str(value))}


def _tracks(rows: pd.DataFrame) -> set[int]:
    return _integers("|".join(rows.source_ref.astype(str)))


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _component_sizes(frame: np.ndarray, identity: int,
                     point) -> tuple[int, int, int, float]:
    components, count = ndi.label(frame == int(identity), STRUCTURE)
    if not count:
        return 0, 0, 0, float("inf")
    sizes = np.bincount(components.ravel())
    yy, xx = np.nonzero(components)
    distances = (xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
    nearest = int(np.argmin(distances))
    local_label = int(components[yy[nearest], xx[nearest]])
    local_size = int(sizes[local_label])
    remote_size = int(max(
        (sizes[label] for label in range(1, len(sizes))
         if label != local_label), default=0))
    return int(count), local_size, remote_size, float(np.sqrt(distances[nearest]))


def _typical_soma_component(labels: np.ndarray) -> float:
    areas: list[int] = []
    for frame in labels:
        for identity in np.unique(frame):
            if int(identity) <= 0:
                continue
            components, count = ndi.label(frame == int(identity), STRUCTURE)
            if count:
                areas.append(int(np.bincount(components.ravel())[1:].max()))
    return float(np.median(areas)) if areas else 0.0


def event_summary(events: pd.DataFrame) -> dict:
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


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              labels: np.ndarray, points: pd.DataFrame, params: dict
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Re-score only fully evidenced detached projection relay compounds."""
    assert_target_free(params)
    attached = physical.attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group for track, group in
              attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    maximum_source_frames = max(1, int(math.ceil(movie * float(
        params.get("maximum_takeover_span_movie_fraction", 0.05)))))
    maximum_companion_gap = max(1, int(math.ceil(movie * float(
        params.get("maximum_companion_gap_movie_fraction", 0.05)))))
    minimum_strong = float(params.get(
        "minimum_transition_strong_fraction", 0.75))
    maximum_step = float(params.get(
        "maximum_endpoint_step_sum_radii", 1.0))
    maximum_marker_distance_scale = float(params.get(
        "maximum_marker_distance_radius_scale", 0.75))
    maximum_local_fraction = float(params.get(
        "maximum_local_component_typical_soma_fraction", 0.50))
    minimum_remote_fraction = float(params.get(
        "minimum_remote_component_typical_soma_fraction", 0.75))
    minimum_blank = int(params.get("minimum_blank_transition_frames", 1))
    typical_soma = _typical_soma_component(labels)
    calibrated_events = events.copy()
    calibrated_members = members.copy()
    audit_rows: list[dict] = []
    eligible_pairs: list[tuple[str, int, int]] = []

    current_ids = set(events.event_id.astype(str))
    for event_id, event_members in members[
            members.event_id.astype(str).isin(current_ids)].groupby(
                "event_id", sort=False):
        event_id = str(event_id)
        sources = event_members.drop_duplicates("source_id")
        takeover = sources[
            sources.source_family.eq("identity_swap_or_takeover")]
        temporary = sources[
            sources.source_family.eq("temporary_owner_excursion")]
        reasons: list[str] = []
        if len(sources) != 2 or len(takeover) != 1 or len(temporary) != 1:
            reasons.append("not_one_takeover_plus_one_temporary_excursion")
        if reasons:
            audit_rows.append({
                "event_id": event_id, "takeover_source_id": 0,
                "temporary_source_id": 0, "physical_track": 0,
                "old_owner": 0, "new_owner": 0,
                "takeover_span_frames": 0, "blank_transition_frames": 0,
                "endpoint_step_sum_radii": float("nan"),
                "transition_strong_fraction": float("nan"),
                "typical_soma_component_px": typical_soma,
                "old_local_component_px": 0, "old_remote_component_px": 0,
                "new_local_component_px": 0, "new_remote_component_px": 0,
                "owners_present_fraction": 0.0,
                "eligible": False, "reason": "|".join(reasons)})
            continue

        takeover_row = takeover.iloc[0]
        temporary_row = temporary.iloc[0]
        takeover_id = int(takeover_row.source_id)
        temporary_id = int(temporary_row.source_id)
        takeover_rows = event_members[
            event_members.source_id.astype(int).eq(takeover_id)]
        temporary_rows = event_members[
            event_members.source_id.astype(int).eq(temporary_id)]
        takeover_tracks = _tracks(takeover_rows)
        temporary_tracks = _tracks(temporary_rows)
        old_owners = _integers(takeover_row.source_accepted_before)
        new_owners = _integers(takeover_row.source_accepted_after)
        temporary_before = _integers(temporary_row.source_accepted_before)
        temporary_after = _integers(temporary_row.source_accepted_after)
        if len(takeover_tracks) != 1 or takeover_tracks != temporary_tracks:
            reasons.append("sources_not_on_one_shared_reference")
        if len(old_owners) != 1 or len(new_owners) != 1:
            reasons.append("takeover_owners_not_unique")
        old_owner = next(iter(old_owners), 0)
        new_owner = next(iter(new_owners), 0)
        if not old_owner or not new_owner or old_owner == new_owner:
            reasons.append("takeover_owner_transition_invalid")
        if temporary_before != {new_owner} or temporary_after != {new_owner}:
            reasons.append("temporary_excursion_not_bracketed_by_new_owner")
        first = int(takeover_row.source_first_frame)
        last = int(takeover_row.source_last_frame)
        span = last - first + 1
        temporary_first = int(temporary_row.source_first_frame)
        if span > maximum_source_frames:
            reasons.append("takeover_transition_too_long")
        companion_gap = temporary_first - last
        if not 0 <= companion_gap <= maximum_companion_gap:
            reasons.append("temporary_excursion_not_immediate_companion")
        track = next(iter(takeover_tracks), 0)
        group = groups.get(track, pd.DataFrame())
        start_point = _point(group, first) if len(group) else None
        end_point = _point(group, last) if len(group) else None
        interval = group[group.frame.astype(int).between(first, last)] \
            if len(group) else group
        if (start_point is None or end_point is None or len(interval) != span
                or not interval.physically_visible.astype(bool).all()):
            reasons.append("transition_not_physically_continuous")
        blank_frames = int(interval.accepted_owner.astype(int).eq(0).sum()) \
            if len(interval) else 0
        strong_fraction = float(interval.strong.astype(bool).mean()) \
            if len(interval) else 0.0
        if blank_frames < minimum_blank:
            reasons.append("no_ownerless_transition_frame")
        if strong_fraction < minimum_strong:
            reasons.append("transition_too_dim")
        step = float("inf")
        old_component = (0, 0, 0, float("inf"))
        new_component = (0, 0, 0, float("inf"))
        if start_point is not None and end_point is not None:
            step = float(np.hypot(
                float(start_point.x) - float(end_point.x),
                float(start_point.y) - float(end_point.y)) / max(
                    float(start_point.radius_px) + float(end_point.radius_px),
                    1.0))
            old_component = _component_sizes(
                labels[first], old_owner, start_point)
            new_component = _component_sizes(
                labels[last], new_owner, end_point)
        if step > maximum_step:
            reasons.append("transition_endpoints_too_far")
        old_limit = max(2.0, maximum_marker_distance_scale * float(
            start_point.radius_px)) if start_point is not None else 0.0
        new_limit = max(2.0, maximum_marker_distance_scale * float(
            end_point.radius_px)) if end_point is not None else 0.0
        if old_component[3] > old_limit or new_component[3] > new_limit:
            reasons.append("endpoint_local_component_missing")
        if (old_component[1] > maximum_local_fraction * typical_soma
                or new_component[1] > maximum_local_fraction * typical_soma):
            reasons.append("endpoint_component_not_projection_sized")
        if (old_component[2] < minimum_remote_fraction * typical_soma
                or new_component[2] < minimum_remote_fraction * typical_soma):
            reasons.append("independent_remote_soma_missing")
        transition_frames = range(first, last + 1)
        owners_present = [
            bool(np.any(labels[frame] == old_owner)
                 and np.any(labels[frame] == new_owner))
            for frame in transition_frames]
        owner_presence_fraction = float(np.mean(owners_present)) \
            if owners_present else 0.0
        if owner_presence_fraction < 1.0:
            reasons.append("owners_not_both_present_through_transition")
        eligible = not reasons
        if eligible:
            eligible_pairs.append((event_id, takeover_id, temporary_id))
        audit_rows.append({
            "event_id": event_id, "takeover_source_id": takeover_id,
            "temporary_source_id": temporary_id, "physical_track": track,
            "old_owner": old_owner, "new_owner": new_owner,
            "takeover_span_frames": span,
            "blank_transition_frames": blank_frames,
            "endpoint_step_sum_radii": step,
            "transition_strong_fraction": strong_fraction,
            "typical_soma_component_px": typical_soma,
            "old_local_component_px": old_component[1],
            "old_remote_component_px": old_component[2],
            "new_local_component_px": new_component[1],
            "new_remote_component_px": new_component[2],
            "owners_present_fraction": owner_presence_fraction,
            "eligible": eligible,
            "reason": ("eligible_detached_projection_owner_relay"
                       if eligible else "|".join(reasons)),
        })

    for event_id, takeover_id, temporary_id in eligible_pairs:
        remove = (calibrated_members.event_id.astype(str).eq(event_id)
                  & calibrated_members.source_id.astype(int).eq(takeover_id))
        calibrated_members = calibrated_members[~remove].copy()
        remaining = calibrated_members[
            calibrated_members.event_id.astype(str).eq(event_id)
            & calibrated_members.source_id.astype(int).eq(temporary_id)]
        source = remaining.iloc[0]
        event_mask = calibrated_events.event_id.astype(str).eq(event_id)
        impact = float(source.source_impact)
        disruption = 0.65 * 20.0 + 0.35 * impact
        updates = {
            "family": "temporary_owner_excursion",
            "first_frame": int(source.source_first_frame),
            "last_frame": int(source.source_last_frame),
            "x": float(source.source_x), "y": float(source.source_y),
            "accepted_before": str(source.source_accepted_before),
            "accepted_after": str(source.source_accepted_after),
            "impact": impact, "disruption_score": disruption,
            "fidelity": 100.0 - disruption,
            "member_refs": str(source.source_ref),
            "physical_tracks": "|".join(map(str, sorted(_tracks(remaining)))),
            "constituent_events": 1,
            "evidence": (str(source.source_evidence)
                         + " ; detached projection owner relay excluded"),
        }
        for column, value in updates.items():
            calibrated_events.loc[event_mask, column] = value

    calibrated_events = calibrated_events.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated_events["impact_rank"] = np.arange(1, len(calibrated_events) + 1)
    calibrated_events = calibrated_events[list(events.columns)]
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(events.event_id.nunique()),
        "compound_candidates_audited": int(sum(
            row["takeover_source_id"] > 0 for row in audit_rows)),
        "events_reclassified": int(len(eligible_pairs)),
        "sources_excluded_as_projection_relays": int(len(eligible_pairs)),
        "reclassified_event_ids": sorted(row[0] for row in eligible_pairs),
        "before": event_summary(events),
        "after": event_summary(calibrated_events),
    }
    return (calibrated_events, calibrated_members,
            pd.DataFrame(audit_rows), summary)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis-tuner adapter for fresh per-well detector runs."""
    import json
    import tifffile

    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    points = pd.read_csv(params["points_path"])
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, labels, points, params)
    stem = str(params["stem"])
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "detached_projection_owner_relay_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["events"], index=False)
    calibrated_members.to_csv(outputs["members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    summary["label_bytes_unchanged"] = (
        outputs["labels"].read_bytes() == labels_path.read_bytes())
    summary["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
