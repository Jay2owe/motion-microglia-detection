"""Exclude detached fading projections from unclaimed-cell disruption totals.

Discovery is complete-field and target-free. The rule changes event tables
only; accepted labels and the unclaimed-pixel ledger remain unchanged. All
measurements are relative to the same physical reference's earlier,
owner-supported state or to its instantaneous radius.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


FORBIDDEN = (
    "identity_target", "identity_id", "target_identity", "owner_target",
    "owner_id", "target_owner", "track_target", "track_id", "target_track",
    "frame_target", "frame_id", "target_frame", "coordinate", "event_target",
    "event_id", "target_event", "region", "well_target", "well_name",
    "review_case", "case_id", "forced_identity", "forced_interval",
    "include_track", "exclude_track", "include_identity", "exclude_identity",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "detached-fading-projection calibration received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "detached-fading-projection calibration must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _integers(value: object) -> set[int]:
    return {int(item) for item in re.findall(r"\d+", str(value))}


def _tracks(rows: pd.DataFrame) -> set[int]:
    return {int(item) for item in re.findall(
        r"T(\d+)", "|".join(rows.source_ref.astype(str)))}


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
        "events_by_family": {
            str(key): int(value)
            for key, value in events.groupby("family").size().items()},
    }


def _nearest_owner_distance(frame: np.ndarray, owner: int, row) -> float:
    owner_mask = frame == int(owner)
    if not owner_mask.any():
        return float("inf")
    distances = ndi.distance_transform_edt(~owner_mask)
    y = int(np.clip(round(float(row.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, frame.shape[1] - 1))
    return float(distances[y, x])


def _scaled_maximum_step(rows: pd.DataFrame) -> float:
    if rows.empty or "frame" not in rows.columns:
        return float("inf")
    ordered = list(rows.sort_values("frame").itertuples(index=False))
    return max((
        float(np.hypot(float(left.x) - float(right.x),
                       float(left.y) - float(right.y)))
        / max(float(left.radius_px) + float(right.radius_px), 1.0)
        for left, right in zip(ordered[:-1], ordered[1:])), default=0.0)


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              labels: np.ndarray, points: pd.DataFrame, params: dict):
    """Remove only fully evidenced detached fading projection alerts."""
    assert_target_free(params)
    movie = int(len(labels))
    minimum_prior_support = max(1, int(math.ceil(movie * float(
        params.get("minimum_prior_owner_support_movie_fraction", 0.35)))))
    minimum_prior_purity = float(params.get("minimum_prior_owner_purity", 0.95))
    minimum_event_visible = float(params.get("minimum_event_visible_fraction", 0.90))
    minimum_event_unowned = float(params.get("minimum_event_unowned_fraction", 1.0))
    minimum_owner_presence = float(params.get(
        "minimum_prior_owner_event_presence_fraction", 1.0))
    minimum_distance = float(params.get(
        "minimum_event_owner_distance_radii", 3.0))
    maximum_step = float(params.get("maximum_event_step_sum_radii", 1.0))
    maximum_radius_ratio = float(params.get(
        "maximum_event_to_owned_radius_ratio", 0.70))
    maximum_evidence_ratio = float(params.get(
        "maximum_event_to_owned_evidence_ratio", 0.10))
    maximum_strong = float(params.get("maximum_event_strong_fraction", 0.40))

    point_groups = {
        int(track): group.sort_values("frame").copy()
        for track, group in points.groupby("track_id", sort=True)}
    audit_rows: list[dict] = []
    removed: list[str] = []

    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[
            members.event_id.astype(str).eq(event_id)].drop_duplicates(
                "source_id")
        reasons: list[str] = []
        is_simple_unclaimed = (
            str(event.family) == "unclaimed_visible_body"
            and len(event_members) == 1
            and str(event_members.iloc[0].source_family)
            == "unclaimed_visible_body")
        if not is_simple_unclaimed:
            reasons.append("not_simple_unclaimed_body")

        tracks = _tracks(event_members) if len(event_members) else set()
        if len(tracks) != 1:
            reasons.append("not_one_physical_reference")
        track = next(iter(tracks), 0)
        group = point_groups.get(track, pd.DataFrame())
        first, last = int(event.first_frame), int(event.last_frame)
        span = last - first + 1
        interval = group[group.frame.astype(int).between(first, last)] \
            if len(group) else group
        before = group[group.frame.astype(int).lt(first)] if len(group) else group
        owners = _integers(event.accepted_before)
        if len(owners) != 1:
            reasons.append("prior_owner_not_unique")
        owner = next(iter(owners), 0)
        owned = before[
            before.candidate_owner.astype(int).eq(owner)
            & before.physically_visible.map(_truth)] if owner and len(before) \
            else pd.DataFrame()
        positive_before = before[
            before.candidate_owner.astype(int).gt(0)
            & before.physically_visible.map(_truth)] if len(before) \
            else pd.DataFrame()
        support = int(len(owned))
        purity = float(support / len(positive_before)) \
            if len(positive_before) else 0.0
        if support < minimum_prior_support:
            reasons.append("prior_owner_support_too_short")
        if purity < minimum_prior_purity:
            reasons.append("prior_owner_not_pure")
        last_owned = int(owned.frame.astype(int).max()) if len(owned) else -1
        if last_owned >= first:
            reasons.append("prior_owner_not_before_event")

        coverage = float(len(interval) / span) if span > 0 else 0.0
        visible_fraction = float(interval.physically_visible.map(_truth).mean()) \
            if len(interval) else 0.0
        unowned_fraction = float(
            interval.candidate_owner.astype(int).eq(0).mean()) \
            if len(interval) else 0.0
        if coverage < 1.0:
            reasons.append("event_reference_not_contiguous")
        if visible_fraction < minimum_event_visible:
            reasons.append("event_not_visibly_continuous")
        if unowned_fraction < minimum_event_unowned:
            reasons.append("event_not_fully_unowned")

        owner_presence = float(np.mean([
            bool(np.any(labels[frame] == owner))
            for frame in range(first, last + 1)])) if owner and span > 0 else 0.0
        if owner_presence < minimum_owner_presence:
            reasons.append("prior_owner_absent_during_event")
        distance_ratios = [
            _nearest_owner_distance(labels[int(row.frame)], owner, row)
            / max(float(row.radius_px), 1.0)
            for row in interval.itertuples(index=False)] if owner else []
        distance_p10 = float(np.quantile(distance_ratios, 0.10)) \
            if distance_ratios else 0.0
        if distance_p10 < minimum_distance:
            reasons.append("reference_not_detached_from_owner")

        event_step = _scaled_maximum_step(interval)
        if event_step > maximum_step:
            reasons.append("event_reference_not_stationary")
        owned_radius = float(owned.radius_px.astype(float).median()) \
            if len(owned) else 0.0
        event_radius = float(interval.radius_px.astype(float).median()) \
            if len(interval) else float("inf")
        radius_ratio = event_radius / max(owned_radius, 1e-9)
        if radius_ratio > maximum_radius_ratio:
            reasons.append("event_not_sufficiently_smaller")
        owned_evidence = float(np.median(np.maximum(
            owned.evidence.astype(float).to_numpy(), 0.0))) \
            if len(owned) else 0.0
        event_evidence = float(np.median(np.maximum(
            interval.evidence.astype(float).to_numpy(), 0.0))) \
            if len(interval) else float("inf")
        evidence_ratio = event_evidence / max(owned_evidence, 1e-9)
        if evidence_ratio > maximum_evidence_ratio:
            reasons.append("event_not_sufficiently_dimmer")
        strong_fraction = float(interval.strong.map(_truth).mean()) \
            if len(interval) else 1.0
        if strong_fraction > maximum_strong:
            reasons.append("event_too_often_strong")

        eligible = not reasons
        if eligible:
            removed.append(event_id)
        audit_rows.append({
            "event_id": event_id, "family": str(event.family),
            "physical_track": track, "prior_owner": owner,
            "event_first": first, "event_last": last,
            "event_span_frames": span,
            "prior_owner_support_frames": support,
            "prior_owner_purity": purity,
            "last_owned_frame": last_owned,
            "event_reference_coverage": coverage,
            "event_visible_fraction": visible_fraction,
            "event_unowned_fraction": unowned_fraction,
            "prior_owner_event_presence_fraction": owner_presence,
            "event_owner_distance_radii_p10": distance_p10,
            "event_maximum_step_sum_radii": event_step,
            "owned_median_radius_px": owned_radius,
            "event_median_radius_px": event_radius,
            "event_to_owned_radius_ratio": radius_ratio,
            "owned_median_positive_evidence": owned_evidence,
            "event_median_positive_evidence": event_evidence,
            "event_to_owned_evidence_ratio": evidence_ratio,
            "event_strong_fraction": strong_fraction,
            "eligible": eligible,
            "reason": ("eligible_detached_fading_projection"
                       if eligible else "|".join(dict.fromkeys(reasons))),
        })

    calibrated = events[~events.event_id.astype(str).isin(removed)].copy()
    calibrated_members = members[
        ~members.event_id.astype(str).isin(removed)].copy()
    calibrated = calibrated.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated["impact_rank"] = np.arange(1, len(calibrated) + 1)
    calibrated = calibrated[list(events.columns)]
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(events)),
        "simple_unclaimed_events_audited": int(sum(
            row["family"] == "unclaimed_visible_body" for row in audit_rows)),
        "events_removed_as_nonbiological": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events),
        "after": event_summary(calibrated),
    }
    return calibrated, calibrated_members, pd.DataFrame(audit_rows), summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Tuner-stage entry point used by fresh per-well scoring."""
    assert_target_free(params)
    events = pd.read_csv(params["events_path"], keep_default_na=False)
    members = pd.read_csv(params["members_path"], keep_default_na=False)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    points = pd.read_csv(params["points_path"])
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, labels, points, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "detached_fading_projection_audit.csv",
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
