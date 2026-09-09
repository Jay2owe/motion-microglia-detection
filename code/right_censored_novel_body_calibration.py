"""Exclude unscoreable right-censored new bodies from disruption totals.

The field-wide rule changes only event tables. It never creates an identity and
never excludes an internal gap or a physical reference that has carried an
accepted owner.
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
import separable_merge_recovery as physical


VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "forced_identity", "forced_interval", "include_track", "exclude_track",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError("right-censored calibration received targets: "
                         + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("right-censored calibration must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _tracks(rows: pd.DataFrame) -> set[int]:
    return {int(value) for value in re.findall(
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
        "events_by_family": {str(key): int(value) for key, value in
                             events.groupby("family").size().items()},
    }


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              labels: np.ndarray, points: pd.DataFrame, raw: np.ndarray,
              thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    attached = physical.attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    maximum_span = max(1, int(math.ceil(movie * float(
        params.get("maximum_reference_span_movie_fraction", 0.05)))))
    emergence_start = int(math.floor((movie - 1) * (1.0 - float(
        params.get("maximum_emergence_boundary_movie_fraction", 0.05)))))
    terminal_start = int(math.ceil((movie - 1) * (1.0 - float(
        params.get("maximum_terminal_censor_movie_fraction", 0.02)))))
    minimum_visible = float(params.get("minimum_visible_fraction", 1.0))
    minimum_strong = float(params.get("minimum_strong_fraction", 1.0))
    minimum_core_observations = int(params.get(
        "minimum_raw_core_observations", 3))
    minimum_core = float(params.get(
        "minimum_core_area_fraction_of_typical_soma", 0.50))
    maximum_core = float(params.get(
        "maximum_core_area_fraction_of_typical_soma", 2.0))
    visible_all = attached[
        attached.state.astype(str).isin(VISIBLE_STATES)]
    radius_floor = float(visible_all.radius_px.astype(float).min())
    strong_all = attached[
        attached.state.astype(str).eq("observed")
        & attached.strong.map(_truth)
        & attached.radius_px.astype(float).gt(radius_floor + 1e-9)]
    if strong_all.empty:
        strong_all = attached[
            attached.state.astype(str).eq("observed")
            & attached.strong.map(_truth)]
    typical_area = float(np.median(
        np.pi * strong_all.radius_px.astype(float).to_numpy() ** 2))
    threshold_by_frame = thresholds.set_index("frame")
    core_params = {
        "core_window_radius_scale": float(params.get(
            "core_window_radius_scale", 2.0)),
        "core_peak_fraction": float(params.get("core_peak_fraction", 0.35)),
        "weak_threshold_fraction": float(params.get(
            "weak_threshold_fraction", 0.50)),
    }
    removed: set[str] = set()
    audits: list[dict] = []
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        sources = event_members.drop_duplicates("source_id")
        tracks = _tracks(event_members)
        reasons: list[str] = []
        if str(event.family) != "unclaimed_visible_body":
            reasons.append("not_unclaimed_body_family")
        if sources.empty or not sources.source_family.astype(str).eq(
                "unclaimed_visible_body").all():
            reasons.append("sources_not_all_unclaimed_body")
        if len(tracks) != 1:
            reasons.append("not_one_reference")
        track = next(iter(tracks), 0)
        group = groups.get(track, pd.DataFrame())
        first = int(group.frame.min()) if len(group) else -1
        last = int(group.frame.max()) if len(group) else -1
        span = last - first + 1 if len(group) else 0
        frames = group.frame.astype(int).to_numpy() if len(group) else np.array([])
        continuous = bool(len(frames) and np.array_equal(
            frames, np.arange(first, last + 1)))
        visible_fraction = float(group.physically_visible.astype(bool).mean()) \
            if len(group) else 0.0
        strong_fraction = float(group.strong.map(_truth).mean()) \
            if len(group) else 0.0
        never_owned = bool(len(group) and
                           group.accepted_owner.astype(int).eq(0).all())
        event_covers_reference = bool(
            len(group) and int(event.first_frame) == first
            and int(event.last_frame) == last)
        if span > maximum_span:
            reasons.append("reference_not_short")
        if first < emergence_start:
            reasons.append("body_did_not_emerge_near_recording_end")
        if last < terminal_start:
            reasons.append("reference_not_right_censored")
        if not continuous:
            reasons.append("reference_frames_not_continuous")
        if visible_fraction < minimum_visible:
            reasons.append("reference_not_fully_visible")
        if strong_fraction < minimum_strong:
            reasons.append("reference_not_fully_strong")
        if not never_owned:
            reasons.append("reference_has_accepted_owner_history")
        if not event_covers_reference:
            reasons.append("event_does_not_cover_complete_reference")
        core_areas: list[int] = []
        if len(group):
            for point in group.itertuples(index=False):
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
        if len(core_areas) < minimum_core_observations:
            reasons.append("insufficient_raw_core_observations")
        if not minimum_core <= core_fraction <= maximum_core:
            reasons.append("raw_core_not_cell_sized")
        eligible = not reasons
        if eligible:
            removed.add(event_id)
        audits.append({
            "event_id": event_id, "family": str(event.family),
            "measured_reference": int(track),
            "reference_first_frame": first, "reference_last_frame": last,
            "reference_span_frames": span,
            "maximum_reference_span_frames": maximum_span,
            "visible_fraction": visible_fraction,
            "strong_fraction": strong_fraction,
            "never_owned": never_owned,
            "event_covers_complete_reference": event_covers_reference,
            "typical_strong_soma_area_px": typical_area,
            "raw_core_observations": int(len(core_areas)),
            "raw_core_area_p95_px": core_p95,
            "raw_core_fraction_of_typical_soma": core_fraction,
            "eligible": eligible,
            "reason": ("right_censored_novel_body" if eligible
                       else "|".join(reasons)),
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
        "right_censored_events_calibrated": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events),
        "after": event_summary(calibrated),
    }
    return calibrated, calibrated_members, pd.DataFrame(audits), summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    points = pd.read_csv(params["points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    labels = tifffile.imread(labels_path)
    raw = tifffile.imread(params["raw_path"])
    calibrated, next_members, audit, summary = calibrate(
        events, members, labels, points, raw, thresholds, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "right_censored_novel_body_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["events"], index=False)
    next_members.to_csv(outputs["members"], index=False)
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
