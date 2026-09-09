"""Calibrate short ownerless alerts continued from fragile ownerless references.

Discovery is complete-field and target-free.  Identifiers are measured outputs
for audit and review only; callers cannot supply identity, track, event, frame,
coordinate, region, well, or review-case selectors.
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

import isolated_unowned_lifetime as isolated


VISIBLE_STATES = frozenset({"observed", "latent_visible"})
FORBIDDEN = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "well", "review_case", "case_id",
    "identity_target", "owner_target", "track_target", "frame_target",
    "event_target", "forced_", "include_", "exclude_", "allowed_",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "ownerless reference-handoff calibration received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "ownerless reference-handoff calibration must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _tracks(rows: pd.DataFrame) -> set[int]:
    return set(map(int, re.findall(
        r"T(\d+)", "|".join(rows.source_ref.astype(str)))))


def event_summary(events: pd.DataFrame) -> dict:
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


def _profile(group: pd.DataFrame) -> dict:
    ordered = group.sort_values("frame")
    visible = ordered[ordered.state.astype(str).isin(VISIBLE_STATES)]
    observed = ordered[ordered.state.astype(str).eq("observed")]
    return {
        "rows": ordered,
        "visible": visible,
        "first": int(ordered.frame.min()),
        "last": int(ordered.frame.max()),
        "span": int(ordered.frame.max()) - int(ordered.frame.min()) + 1,
        "unowned_fraction": float(
            visible.candidate_owner.astype(int).eq(0).mean())
            if len(visible) else 0.0,
        "observed_fraction": float(len(observed) / max(len(ordered), 1)),
        "strong_fraction": float(
            observed.strong.map(_truth).mean()) if len(observed) else 0.0,
    }


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, raw: np.ndarray,
              thresholds: pd.DataFrame, params: dict):
    """Return a calibrated catalogue plus a complete event audit."""
    assert_target_free(params)
    required = {"candidate_owner", "frame", "radius_px", "state", "strong",
                "track_id", "x", "y"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError("scored point fields missing: " + ", ".join(missing))
    if not {"frame", "weak_threshold"}.issubset(thresholds.columns):
        raise ValueError("frame thresholds require frame and weak_threshold")

    movie = int(len(raw))
    maximum_event_span = max(1, int(math.ceil(movie * float(params.get(
        "maximum_event_span_movie_fraction", 0.10)))))
    maximum_handoff_delta = max(1, int(math.ceil(movie * float(params.get(
        "maximum_handoff_delta_movie_fraction", 0.05)))))
    maximum_fragile_span = max(1, int(math.ceil(movie * float(params.get(
        "maximum_fragile_predecessor_span_movie_fraction", 0.15)))))
    minimum_event_coverage = float(params.get(
        "minimum_event_visible_coverage", 0.75))
    minimum_unowned = float(params.get("minimum_unowned_fraction", 1.0))
    minimum_target_floor = float(params.get(
        "minimum_target_radius_floor_fraction", 0.60))
    minimum_predecessor_floor = float(params.get(
        "minimum_predecessor_radius_floor_fraction", 0.95))
    maximum_core = float(params.get(
        "maximum_core_area_fraction_of_typical_soma", 0.65))
    maximum_speed = float(params.get(
        "maximum_handoff_speed_typical_radii_per_frame", 3.0))
    maximum_distance = float(params.get(
        "maximum_handoff_distance_typical_radii", 8.0))
    maximum_predecessor_observed = float(params.get(
        "maximum_fragile_predecessor_observed_fraction", 0.75))
    maximum_predecessor_strong = float(params.get(
        "maximum_fragile_predecessor_strong_fraction", 0.50))

    visible_all = points[points.state.astype(str).isin(VISIBLE_STATES)]
    radius_floor = float(visible_all.radius_px.astype(float).min())
    strong = points[
        points.state.astype(str).eq("observed")
        & points.strong.map(_truth)
        & points.radius_px.astype(float).gt(radius_floor + 1e-9)]
    if strong.empty:
        strong = points[points.state.astype(str).eq("observed")
                        & points.strong.map(_truth)]
    typical_radius = float(strong.radius_px.astype(float).median())
    typical_area = float(np.median(
        np.pi * strong.radius_px.astype(float).to_numpy() ** 2))
    profiles = {
        int(track): _profile(group)
        for track, group in points.groupby("track_id", sort=True)}
    threshold_by_frame = thresholds.set_index("frame")
    core_params = {
        "core_window_radius_scale": float(params.get(
            "core_window_radius_scale", 2.0)),
        "core_peak_fraction": float(params.get("core_peak_fraction", 0.35)),
        "weak_threshold_fraction": float(params.get(
            "weak_threshold_fraction", 0.50)),
    }

    removed: set[str] = set()
    audit_rows: list[dict] = []
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        sources = event_members.drop_duplicates("source_id")
        tracks = _tracks(event_members)
        first, last = int(event.first_frame), int(event.last_frame)
        event_span = last - first + 1
        reasons: list[str] = []
        if str(event.family) != "unclaimed_visible_body":
            reasons.append("not_unclaimed_body_family")
        if (sources.empty or not sources.source_family.astype(str)
                .eq("unclaimed_visible_body").all()):
            reasons.append("sources_not_all_unclaimed_body")
        if len(tracks) != 1:
            reasons.append("not_one_reference")
        target_track = next(iter(tracks)) if len(tracks) == 1 else 0
        target = profiles.get(target_track)
        event_rows = pd.DataFrame()
        event_coverage = 0.0
        target_unowned = 0.0
        target_global_unowned = 0.0
        target_floor_fraction = 0.0
        if target is not None:
            event_rows = target["visible"][
                target["visible"].frame.astype(int).between(first, last)]
            event_coverage = len(event_rows) / max(event_span, 1)
            target_unowned = float(
                event_rows.candidate_owner.astype(int).eq(0).mean()) \
                if len(event_rows) else 0.0
            target_global_unowned = float(
                target["visible"].candidate_owner.astype(int).eq(0).mean()) \
                if len(target["visible"]) else 0.0
            target_floor_fraction = float(np.isclose(
                event_rows.radius_px.astype(float).to_numpy(), radius_floor,
                rtol=0.0, atol=1e-9).mean()) if len(event_rows) else 0.0
        if event_span > maximum_event_span:
            reasons.append("event_not_short")
        if event_coverage < minimum_event_coverage:
            reasons.append("insufficient_event_coverage")
        if target_unowned < minimum_unowned:
            reasons.append("event_not_fully_unowned")
        if target_global_unowned < minimum_unowned:
            reasons.append("reference_owned_elsewhere")
        if target_floor_fraction < minimum_target_floor:
            reasons.append("target_radius_not_near_field_floor")

        core_areas: list[int] = []
        for point in event_rows.itertuples(index=False):
            frame = int(point.frame)
            _, _, details = isolated._connected_core(
                raw[frame], point,
                float(threshold_by_frame.loc[frame, "weak_threshold"]),
                core_params)
            core_areas.append(int(details["core_area_px"]))
        core_p95 = float(np.percentile(core_areas, 95)) \
            if core_areas else float("nan")
        core_fraction = core_p95 / max(typical_area, 1.0) \
            if core_areas else float("inf")
        if not core_areas:
            reasons.append("raw_core_absent")
        if core_fraction > maximum_core:
            reasons.append("raw_core_not_sub_soma")

        predecessors: list[dict] = []
        if target is not None:
            target_first = target["rows"].iloc[0]
            for track, profile in profiles.items():
                if track == target_track or profile["last"] >= first:
                    continue
                delta = first - profile["last"]
                if delta < 1 or delta > maximum_handoff_delta:
                    continue
                predecessor_last = profile["rows"].iloc[-1]
                distance = float(np.hypot(
                    float(predecessor_last.x) - float(target_first.x),
                    float(predecessor_last.y) - float(target_first.y)))
                speed = distance / max(
                    delta * max(typical_radius, 1.0), 1.0)
                distance_radii = distance / max(typical_radius, 1.0)
                predecessor_floor = float(np.isclose(
                    profile["visible"].radius_px.astype(float).to_numpy(),
                    radius_floor, rtol=0.0, atol=1e-9).mean()) \
                    if len(profile["visible"]) else 0.0
                fragile = bool(
                    profile["span"] <= maximum_fragile_span
                    or profile["observed_fraction"] <= maximum_predecessor_observed
                    or profile["strong_fraction"] <= maximum_predecessor_strong)
                if (profile["unowned_fraction"] >= minimum_unowned
                        and predecessor_floor >= minimum_predecessor_floor
                        and speed <= maximum_speed
                        and distance_radii <= maximum_distance and fragile):
                    predecessors.append({
                        "track": track, "delta": delta,
                        "distance": distance, "distance_radii": distance_radii,
                        "speed": speed,
                        "span": profile["span"],
                        "observed_fraction": profile["observed_fraction"],
                        "strong_fraction": profile["strong_fraction"],
                        "floor_fraction": predecessor_floor,
                    })
        predecessors.sort(key=lambda row: (
            row["speed"], row["delta"], row["track"]))
        chosen = predecessors[0] if predecessors else None
        if chosen is None:
            reasons.append("compatible_fragile_ownerless_predecessor_missing")

        eligible = not reasons
        if eligible:
            removed.add(event_id)
        audit_rows.append({
            "event_id": event_id,
            "family": str(event.family),
            "measured_target_track": target_track,
            "event_first": first,
            "event_last": last,
            "event_span_frames": event_span,
            "maximum_event_span_frames": maximum_event_span,
            "event_visible_coverage": event_coverage,
            "target_unowned_fraction": target_unowned,
            "target_global_unowned_fraction": target_global_unowned,
            "radius_floor_px": radius_floor,
            "target_radius_floor_fraction": target_floor_fraction,
            "typical_strong_soma_radius_px": typical_radius,
            "typical_strong_soma_area_px": typical_area,
            "raw_core_observations": len(core_areas),
            "raw_core_area_p95_px": core_p95,
            "raw_core_fraction_of_typical_soma": core_fraction,
            "compatible_predecessor_count": len(predecessors),
            "measured_predecessor_track": (
                int(chosen["track"]) if chosen else 0),
            "handoff_delta_frames": int(chosen["delta"]) if chosen else 0,
            "handoff_distance_px": (
                float(chosen["distance"]) if chosen else float("nan")),
            "handoff_distance_typical_radii": (
                float(chosen["distance_radii"]) if chosen else float("nan")),
            "handoff_speed_typical_radii_per_frame": (
                float(chosen["speed"]) if chosen else float("nan")),
            "predecessor_span_frames": (
                int(chosen["span"]) if chosen else 0),
            "predecessor_observed_fraction": (
                float(chosen["observed_fraction"])
                if chosen else float("nan")),
            "predecessor_strong_fraction": (
                float(chosen["strong_fraction"])
                if chosen else float("nan")),
            "eligible": eligible,
            "reason": ("fragile_ownerless_reference_handoff"
                       if eligible else "|".join(dict.fromkeys(reasons))),
        })

    calibrated = events[
        ~events.event_id.astype(str).isin(removed)].copy()
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
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(events)),
        "events_calibrated_nonbiological": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events),
        "after": event_summary(calibrated),
    }
    return calibrated, calibrated_members, pd.DataFrame(audit_rows), summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    candidate, members, audit, summary = calibrate(
        pd.read_csv(params["events_path"]),
        pd.read_csv(params["members_path"]),
        pd.read_csv(params["points_path"]),
        tifffile.imread(params["raw_path"]),
        pd.read_csv(params["thresholds_path"]), params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "ownerless_reference_handoff_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    candidate.to_csv(outputs["events"], index=False)
    members.to_csv(outputs["members"], index=False)
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
