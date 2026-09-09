"""Calibrate short single-owner relays across several raw references.

Discovery is complete-field and target-free. Measured identifiers are emitted
only in the audit; none can be supplied as selectors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate_target", "event_target", "region_target",
    "review_case_target",
    "forced_identity", "forced_interval", "include_track", "exclude_track",
)
ALLOWED_SOURCE_FAMILIES = {
    "ordinary_gap", "separable_cells_merged", "single_cell_multi_core",
}


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "single-owner multireference calibration received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "single-owner multireference calibration must be field-wide")


def _values(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    return frozenset(map(int, re.findall(r"\d+", str(value))))


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


def _maximum_reported_separability(sources: pd.DataFrame) -> float:
    values: list[float] = []
    for evidence in sources.source_evidence.astype(str):
        values.extend(float(value) for value in re.findall(
            r"(?:^|[,; ])S=([0-9]*\.?[0-9]+)", evidence))
    return max(values, default=float("inf"))


def _owner_anchor_frames(group: pd.DataFrame, owner: int,
                         first: int, last: int) -> tuple[int, int]:
    assigned = group[group.candidate_owner.astype(int).eq(int(owner))]
    return (int((assigned.frame.astype(int) < int(first)).sum()),
            int((assigned.frame.astype(int) > int(last)).sum()))


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, labels: np.ndarray, params: dict):
    assert_target_free(params)
    required = {"candidate_owner", "frame", "track_id"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(
            "single-owner multireference calibration requires scored point "
            "fields: " + ", ".join(missing))
    movie = int(len(labels))
    maximum_duration = max(1, int(math.ceil(movie * float(
        params.get("maximum_event_movie_fraction", 0.10)))))
    minimum_sources = int(params.get("minimum_sources", 3))
    minimum_tracks = int(params.get("minimum_tracks", 3))
    minimum_anchor = int(params.get("minimum_owner_anchor_frames", 3))
    birth_slack = int(params.get("reference_birth_slack_frames", 1))
    end_slack = int(params.get("reference_end_slack_frames", 1))
    minimum_component = float(params.get(
        "minimum_largest_component_fraction", 0.90))
    groups = {int(track): group.sort_values("frame")
              for track, group in points.groupby("track_id", sort=True)}
    calibrated_events = events.copy()
    calibrated_members = members.copy()
    audit_rows: list[dict] = []
    reclassified: set[str] = set()

    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        sources = event_members.drop_duplicates("source_id")
        families = set(sources.source_family.astype(str))
        tracks = _values(event.physical_tracks)
        before, after = (_values(event.accepted_before),
                         _values(event.accepted_after))
        owner = next(iter(before)) if len(before) == 1 and before == after else 0
        first, last = int(event.first_frame), int(event.last_frame)
        duration = last - first + 1
        reasons: list[str] = []

        if str(event.family) != "separable_cells_merged":
            reasons.append("not_merge_family")
        if duration > maximum_duration:
            reasons.append("event_too_long")
        if len(sources) < minimum_sources:
            reasons.append("too_few_sources")
        if len(tracks) < minimum_tracks:
            reasons.append("too_few_references")
        if not families <= ALLOWED_SOURCE_FAMILIES:
            reasons.append("unsupported_source_family")
        if "separable_cells_merged" not in families:
            reasons.append("no_merge_source")
        if "single_cell_multi_core" not in families:
            reasons.append("no_independently_calibrated_morphology_source")
        if not owner:
            reasons.append("accepted_owner_not_conserved")

        source_owners: set[int] = set()
        for row in sources.itertuples(index=False):
            source_owners |= set(_values(row.source_accepted_before))
            source_owners |= set(_values(row.source_accepted_after))
        if owner and not source_owners <= {owner}:
            reasons.append("source_contains_foreign_owner")
        merge_sources = sources[sources.source_family.astype(str).eq(
            "separable_cells_merged")]
        maximum_separability = _maximum_reported_separability(merge_sources)
        if maximum_separability > float(params.get(
                "maximum_merge_separability", 0.0)):
            reasons.append("positive_merge_separability")

        positive_point_owners: set[int] = set()
        pre_anchors: set[int] = set()
        post_anchors: set[int] = set()
        births: set[int] = set()
        ends: set[int] = set()
        for track in tracks:
            group = groups.get(int(track), pd.DataFrame())
            if group.empty:
                continue
            positive_point_owners |= set(group[
                group.candidate_owner.astype(int).gt(0)
            ].candidate_owner.astype(int))
            pre, post = _owner_anchor_frames(group, owner, first, last) \
                if owner else (0, 0)
            if pre >= minimum_anchor:
                pre_anchors.add(int(track))
            if post >= minimum_anchor:
                post_anchors.add(int(track))
            track_first = int(group.frame.min())
            track_last = int(group.frame.max())
            if first - birth_slack <= track_first <= last:
                births.add(int(track))
            if first <= track_last <= last + end_slack:
                ends.add(int(track))
        if owner and not positive_point_owners <= {owner}:
            reasons.append("reference_contains_foreign_owner")
        if not pre_anchors:
            reasons.append("no_pre_event_owner_anchor")
        if not post_anchors:
            reasons.append("no_post_event_owner_anchor")
        if not births or not ends:
            reasons.append("no_reference_birth_death_relay")
        if pre_anchors == post_anchors:
            reasons.append("owner_not_relayed_between_references")

        component_fractions: list[float] = []
        if owner:
            for frame in range(first, last + 1):
                mask = labels[frame] == owner
                components, count = ndi.label(mask, STRUCTURE)
                sizes = np.bincount(components.ravel())
                component_fractions.append(
                    float(sizes[1:].max() / max(int(mask.sum()), 1))
                    if int(count) else 0.0)
        minimum_component_fraction = min(component_fractions, default=0.0)
        if minimum_component_fraction < minimum_component:
            reasons.append("owner_not_one_dominant_component")

        eligible = not reasons
        if eligible:
            reclassified.add(event_id)
            mask = calibrated_members.event_id.astype(str).eq(event_id)
            calibrated_members.loc[mask, "source_family"] = \
                "reference_fragmentation"
            calibrated_members.loc[mask, "source_impact"] = 10.0
            event_mask = calibrated_events.event_id.astype(str).eq(event_id)
            disruption = 0.65 * 5.0 + 0.35 * 10.0
            calibrated_events.loc[event_mask, "family"] = \
                "reference_fragmentation"
            calibrated_events.loc[event_mask, "impact"] = 10.0
            calibrated_events.loc[event_mask, "disruption_score"] = disruption
            calibrated_events.loc[event_mask, "fidelity"] = 100.0 - disruption
            calibrated_events.loc[event_mask, "evidence"] = (
                calibrated_events.loc[event_mask, "evidence"].astype(str)
                + " ; single-owner multireference relay")
        audit_rows.append({
            "event_id": event_id, "family": str(event.family),
            "measured_owner": owner, "duration_frames": duration,
            "source_count": int(len(sources)), "track_count": int(len(tracks)),
            "source_families": "|".join(sorted(families)),
            "positive_point_owners": "|".join(map(str, sorted(
                positive_point_owners))),
            "maximum_merge_separability": maximum_separability,
            "pre_anchor_tracks": "|".join(map(str, sorted(pre_anchors))),
            "post_anchor_tracks": "|".join(map(str, sorted(post_anchors))),
            "birth_tracks": "|".join(map(str, sorted(births))),
            "end_tracks": "|".join(map(str, sorted(ends))),
            "minimum_largest_component_fraction": minimum_component_fraction,
            "eligible": eligible,
            "reason": ("single_owner_multireference_relay" if eligible
                       else "|".join(reasons)),
        })

    calibrated_events = calibrated_events.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated_events["impact_rank"] = np.arange(1, len(calibrated_events) + 1)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(len(events)),
        "events_reclassified": int(len(reclassified)),
        "reclassified_event_ids": sorted(reclassified),
        "before": event_summary(events),
        "after": event_summary(calibrated_events),
    }
    return calibrated_events, calibrated_members, pd.DataFrame(audit_rows), summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    points = pd.read_csv(params["points_path"])
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, points, labels, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "single_owner_multireference_relay_audit.csv",
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
