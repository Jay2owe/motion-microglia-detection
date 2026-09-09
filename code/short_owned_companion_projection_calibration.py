"""Calibrate exact short same-owner companions as non-disruptive projections.

The rule audits all two-track merge events and accepts no identity, track,
frame, event, coordinate, region or review-case selector.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

import separable_merge_recovery as physical


VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate_target", "event_target", "region_target",
    "review_case_target",
    "forced_identity", "forced_interval", "include_track", "exclude_track",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "short-companion calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("short-companion calibration must be field-wide")


def _numbers(value: object) -> frozenset[int]:
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


def _track_metrics(group: pd.DataFrame, owner: int,
                   frame_count: int) -> dict:
    group = group.sort_values("frame")
    visible = group[group.state.isin(VISIBLE_STATES)]
    positive = visible[visible.accepted_owner.astype(int) > 0]
    support = int(positive.accepted_owner.astype(int).eq(owner).sum())
    first, last = int(group.frame.min()), int(group.frame.max())
    span = last - first + 1
    return {
        "first": first, "last": last, "span": span,
        "visible": int(len(visible)),
        "visible_fraction": len(visible) / max(span, 1),
        "owner_support": support,
        "owner_movie_fraction": support / max(frame_count, 1),
        "owner_purity": support / max(len(positive), 1),
        "median_radius": float(visible.radius_px.astype(float).median())
                         if len(visible) else float("nan"),
    }


def _point(group: pd.DataFrame, frame: int):
    row = group[group.frame.astype(int).eq(frame)]
    return None if row.empty else row.iloc[0]


def _endpoint_distance(companion: pd.DataFrame, resident: pd.DataFrame,
                       frame: int) -> float:
    left, right = _point(companion, frame), _point(resident, frame)
    if left is None or right is None:
        return float("inf")
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return distance / scale


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              labels: np.ndarray, points: pd.DataFrame, params: dict,
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Remove only uniquely oriented short same-owner companion events."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    groups = {int(track): group.copy()
              for track, group in scored.groupby("track_id", sort=True)}
    frame_count = int(len(labels))
    minimum_resident_support = float(params.get(
        "minimum_resident_owner_movie_fraction", 0.75))
    minimum_resident_purity = float(params.get(
        "minimum_resident_owner_purity", 0.95))
    minimum_resident_visible = float(params.get(
        "minimum_resident_visible_fraction", 0.90))
    maximum_companion_span = int(np.ceil(frame_count * float(params.get(
        "maximum_companion_span_movie_fraction", 0.15))))
    minimum_companion_visible = float(params.get(
        "minimum_companion_visible_fraction", 0.90))
    minimum_companion_purity = float(params.get(
        "minimum_companion_owner_purity", 0.95))
    minimum_companion_coverage = float(params.get(
        "minimum_companion_owner_span_fraction", 0.90))
    maximum_radius_ratio = float(params.get(
        "maximum_companion_resident_radius_ratio", 0.80))
    maximum_endpoint_distance = float(params.get(
        "maximum_endpoint_distance_sum_radii", 2.0))
    removed: set[str] = set()
    audit_rows: list[dict] = []
    for event in events.itertuples(index=False):
        tracks = sorted(_numbers(event.physical_tracks))
        before, after = (_numbers(event.accepted_before),
                         _numbers(event.accepted_after))
        owners = before | after
        matches: list[dict] = []
        base_reasons: list[str] = []
        if str(event.family) != "separable_cells_merged":
            base_reasons.append("not_separable_merge")
        if len(tracks) != 2:
            base_reasons.append("not_exactly_two_tracks")
        if before != after or len(owners) != 1:
            base_reasons.append("owner_not_conserved")
        if not base_reasons:
            owner = next(iter(owners))
            for resident_track, companion_track in (
                    (tracks[0], tracks[1]), (tracks[1], tracks[0])):
                resident = groups.get(resident_track)
                companion = groups.get(companion_track)
                if resident is None or companion is None:
                    continue
                resident_metrics = _track_metrics(
                    resident, owner, frame_count)
                companion_metrics = _track_metrics(
                    companion, owner, frame_count)
                first = companion_metrics["first"]
                last = companion_metrics["last"]
                radius_ratio = (companion_metrics["median_radius"]
                                / max(resident_metrics["median_radius"], 1e-9))
                first_distance = _endpoint_distance(
                    companion, resident, first)
                last_distance = _endpoint_distance(
                    companion, resident, last)
                reasons: list[str] = []
                if resident_metrics["owner_movie_fraction"] < \
                        minimum_resident_support:
                    reasons.append("resident_support_too_short")
                if resident_metrics["owner_purity"] < minimum_resident_purity:
                    reasons.append("resident_owner_not_pure")
                if resident_metrics["visible_fraction"] < minimum_resident_visible:
                    reasons.append("resident_not_visible_enough")
                if companion_metrics["span"] > maximum_companion_span:
                    reasons.append("companion_too_long")
                if companion_metrics["visible_fraction"] < \
                        minimum_companion_visible:
                    reasons.append("companion_not_visible_enough")
                if companion_metrics["owner_purity"] < minimum_companion_purity:
                    reasons.append("companion_has_foreign_owner")
                if (companion_metrics["owner_support"]
                        / max(companion_metrics["span"], 1)
                        < minimum_companion_coverage):
                    reasons.append("companion_owner_coverage_incomplete")
                if first <= 0 or last >= frame_count - 1:
                    reasons.append("companion_censored_by_movie_boundary")
                if radius_ratio > maximum_radius_ratio:
                    reasons.append("companion_not_smaller")
                if first_distance > maximum_endpoint_distance:
                    reasons.append("first_endpoint_too_far")
                if last_distance > maximum_endpoint_distance:
                    reasons.append("last_endpoint_too_far")
                if not reasons:
                    matches.append({
                        "resident_track": resident_track,
                        "companion_track": companion_track,
                        "owner": owner,
                        "resident_owner_movie_fraction":
                            resident_metrics["owner_movie_fraction"],
                        "resident_owner_purity": resident_metrics["owner_purity"],
                        "companion_span_frames": companion_metrics["span"],
                        "companion_owner_purity": companion_metrics["owner_purity"],
                        "companion_resident_radius_ratio": radius_ratio,
                        "first_endpoint_distance_sum_radii": first_distance,
                        "last_endpoint_distance_sum_radii": last_distance,
                    })
        eligible = len(matches) == 1
        if eligible:
            removed.add(str(event.event_id))
        match = matches[0] if eligible else {}
        audit_rows.append({
            "event_id": str(event.event_id), "family": str(event.family),
            "physical_tracks": str(event.physical_tracks),
            "eligible": eligible,
            "resident_track": match.get("resident_track", 0),
            "companion_track": match.get("companion_track", 0),
            "measured_owner": match.get("owner", 0),
            "resident_owner_movie_fraction": match.get(
                "resident_owner_movie_fraction", float("nan")),
            "resident_owner_purity": match.get(
                "resident_owner_purity", float("nan")),
            "companion_span_frames": match.get("companion_span_frames", 0),
            "companion_owner_purity": match.get(
                "companion_owner_purity", float("nan")),
            "companion_resident_radius_ratio": match.get(
                "companion_resident_radius_ratio", float("nan")),
            "first_endpoint_distance_sum_radii": match.get(
                "first_endpoint_distance_sum_radii", float("nan")),
            "last_endpoint_distance_sum_radii": match.get(
                "last_endpoint_distance_sum_radii", float("nan")),
            "reason": ("calibrated_short_owned_companion_projection"
                       if eligible else ("ambiguous_orientation"
                       if len(matches) > 1 else "|".join(base_reasons)
                       or "no_qualifying_orientation")),
        })
    calibrated = events[
        ~events.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    calibrated = calibrated.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated["impact_rank"] = np.arange(1, len(calibrated) + 1)
    calibrated_members = members[
        ~members.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(len(events)),
        "events_calibrated_non_disruptive": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events), "after": event_summary(calibrated),
    }
    return (calibrated, calibrated_members,
            pd.DataFrame(audit_rows), summary)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis-tuner adapter using explicit current-run file inputs."""
    import json
    import shutil
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
        "candidate_events": out.out / "candidate_disruptive_events.csv",
        "candidate_members": out.out / "candidate_event_members.csv",
        "audit": out.out / "short_owned_companion_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["candidate_events"], index=False)
    calibrated_members.to_csv(outputs["candidate_members"], index=False)
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
