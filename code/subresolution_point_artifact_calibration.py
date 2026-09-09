"""Remove disruption events caused entirely by sub-resolution point artifacts.

Discovery is complete-field and target-free. Identity, track, event, frame and
coordinate values are measured outputs and cannot be supplied as selectors.
The calibration changes only event tables; biological label ledgers are inputs
used to determine whether each physical track is already owned.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

import isolated_unowned_lifetime as isolated
import separable_merge_recovery as physical


VISIBLE_STATES = {"observed", "latent_visible"}
COMPATIBLE_FAMILIES = {"unclaimed_visible_body", "separable_cells_merged"}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
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
            "sub-resolution calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("sub-resolution calibration must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _source_tracks(rows: pd.DataFrame) -> frozenset[int]:
    text = "|".join(rows.source_ref.astype(str))
    return frozenset(map(int, re.findall(r"T(\d+)", text)))


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


def discover_artifacts(labels: np.ndarray, raw: np.ndarray,
                       points: pd.DataFrame, thresholds: pd.DataFrame,
                       params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all physical tracks and return sub-resolution raw-support rows."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    frame_count = int(len(labels))
    minimum_span = int(np.ceil(frame_count * float(
        params.get("minimum_track_span_movie_fraction", 0.75))))
    minimum_visible = float(params.get("minimum_visible_fraction", 0.90))
    minimum_unowned = float(params.get("minimum_unowned_fraction", 0.95))
    minimum_floor = float(params.get(
        "minimum_radius_floor_fraction", 0.95))
    maximum_core_fraction = float(params.get(
        "maximum_core_area_fraction_of_typical_soma", 0.075))
    visible_all = scored[scored.state.isin(VISIBLE_STATES)]
    radius_floor = float(visible_all.radius_px.astype(float).min())
    strong_soma = scored[
        scored.state.eq("observed")
        & scored.strong.map(_truth)
        & (scored.radius_px.astype(float) > radius_floor + 1e-9)]
    if strong_soma.empty:
        strong_soma = scored[
            scored.state.eq("observed") & scored.strong.map(_truth)]
    typical_soma_area = float(np.median(
        np.pi * strong_soma.radius_px.astype(float).to_numpy() ** 2))
    threshold_by_frame = thresholds.set_index("frame")
    rows: list[dict] = []
    core_params = {
        "core_window_radius_scale": float(params.get(
            "core_window_radius_scale", 2.0)),
        "core_peak_fraction": float(params.get("core_peak_fraction", 0.35)),
        "weak_threshold_fraction": float(params.get(
            "weak_threshold_fraction", 0.50)),
    }
    for track, group in scored.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        first, last = int(group.frame.min()), int(group.frame.max())
        span = last - first + 1
        visible = group[group.state.isin(VISIBLE_STATES)]
        reasons: list[str] = []
        continuous = np.array_equal(
            group.frame.astype(int).to_numpy(), np.arange(first, last + 1))
        visible_fraction = len(visible) / max(span, 1)
        unowned_fraction = float(
            visible.accepted_owner.astype(int).eq(0).mean()) \
            if len(visible) else 0.0
        floor_fraction = float(np.isclose(
            visible.radius_px.astype(float).to_numpy(), radius_floor,
            rtol=0.0, atol=1e-9).mean()) if len(visible) else 0.0
        if not continuous:
            reasons.append("physical_track_not_continuous")
        if span < minimum_span:
            reasons.append("track_span_too_short")
        if visible_fraction < minimum_visible:
            reasons.append("insufficient_visible_fraction")
        if unowned_fraction < minimum_unowned:
            reasons.append("accepted_owner_present")
        if floor_fraction < minimum_floor:
            reasons.append("radius_not_pinned_to_tracker_floor")
        core_areas: list[int] = []
        if not reasons:
            for point in visible.itertuples(index=False):
                frame = int(point.frame)
                _, _, details = isolated._connected_core(
                    raw[frame], point,
                    float(threshold_by_frame.loc[frame, "weak_threshold"]),
                    core_params)
                core_areas.append(int(details["core_area_px"]))
        core_p95 = float(np.percentile(core_areas, 95)) \
            if core_areas else float("nan")
        core_fraction = core_p95 / max(typical_soma_area, 1.0) \
            if core_areas else float("nan")
        if not reasons and (
                not core_areas or core_fraction > maximum_core_fraction):
            reasons.append("raw_core_not_subresolution")
        eligible = not reasons
        rows.append({
            "physical_track": int(track), "first_frame": first,
            "last_frame": last, "span_frames": span,
            "visible_frames": int(len(visible)),
            "visible_fraction": float(visible_fraction),
            "unowned_fraction": unowned_fraction,
            "radius_floor_px": radius_floor,
            "radius_floor_fraction": floor_fraction,
            "typical_strong_soma_area_px": typical_soma_area,
            "raw_core_observations": len(core_areas),
            "raw_core_area_p95_px": core_p95,
            "raw_core_area_fraction_of_typical_soma": core_fraction,
            "status": "eligible_artifact" if eligible else "rejected",
            "reason": "subresolution_point_artifact" if eligible
                      else "|".join(reasons),
        })
    return pd.DataFrame(rows), scored


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              artifact_audit: pd.DataFrame, params: dict,
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Remove an event only when every constituent has artifact evidence."""
    assert_target_free(params)
    artifacts = set(artifact_audit.loc[
        artifact_audit.status.eq("eligible_artifact"),
        "physical_track"].astype(int))
    removed: set[str] = set()
    audit_rows: list[dict] = []
    for event_id, event_members in members.groupby("event_id", sort=False):
        source_results = []
        for source_id, rows in event_members.groupby("source_id", sort=True):
            family = str(rows.source_family.iloc[0])
            tracks = _source_tracks(rows)
            explaining = sorted(tracks & artifacts)
            compatible = family in COMPATIBLE_FAMILIES
            source_results.append(bool(compatible and explaining))
            audit_rows.append({
                "event_id": str(event_id), "source_id": int(source_id),
                "source_family": family,
                "physical_tracks": "|".join(map(str, sorted(tracks))),
                "artifact_tracks": "|".join(map(str, explaining)),
                "source_explained": bool(compatible and explaining),
                "reason": ("subresolution_artifact_source"
                           if compatible and explaining
                           else "not_explained_by_artifact"),
            })
        if source_results and all(source_results):
            removed.add(str(event_id))
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
        "tracks_audited": int(len(artifact_audit)),
        "artifact_tracks_discovered": int(len(artifacts)),
        "events_audited": int(events.event_id.nunique()),
        "events_calibrated_nonbiological": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events),
        "after": event_summary(calibrated),
    }
    return (calibrated, calibrated_members,
            pd.DataFrame(audit_rows), summary)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis-tuner adapter using only explicit current-run file inputs."""
    import json
    import shutil
    import tifffile

    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    artifact_audit, _ = discover_artifacts(
        labels, raw, points, thresholds, params)
    calibrated, calibrated_members, event_audit, summary = calibrate(
        events, members, artifact_audit, params)
    stem = str(params["stem"])
    outputs = {
        "candidate_events": out.out / "candidate_disruptive_events.csv",
        "candidate_members": out.out / "candidate_event_members.csv",
        "artifact_audit": out.out / "subresolution_track_audit.csv",
        "event_audit": out.out / "subresolution_event_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["candidate_events"], index=False)
    calibrated_members.to_csv(outputs["candidate_members"], index=False)
    artifact_audit.to_csv(outputs["artifact_audit"], index=False)
    event_audit.to_csv(outputs["event_audit"], index=False)
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    summary["label_bytes_unchanged"] = (
        outputs["labels"].read_bytes() == labels_path.read_bytes())
    summary["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
