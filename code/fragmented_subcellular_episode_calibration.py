"""Remove short ownerless alerts proved to be subcellular reference episodes."""
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


VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "well", "review_case", "case_id",
    "identity_target", "owner_target", "track_target", "frame_target",
    "event_target", "forced_", "include_", "exclude_",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(str(key) for key, value in params.items()
                      if value not in (None, "", [], {})
                      and any(token in str(key).lower()
                              for token in FORBIDDEN))
    if supplied:
        raise ValueError("fragmented subcellular calibration received "
                         "forbidden targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("fragmented subcellular calibration must be field-wide")


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
            (events.disruption_score.astype(float) >= 50).sum()),
        "total_disruption_burden": float(
            events.disruption_score.astype(float).sum()),
        "events_by_family": {str(key): int(value) for key, value in
                             events.groupby("family").size().items()},
    }


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, raw: np.ndarray,
              thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    required = {"candidate_owner", "frame", "radius_px", "state", "strong",
                "track_id", "x", "y"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError("scored point fields missing: " + ", ".join(missing))
    movie = int(len(raw))
    maximum_event = max(1, int(math.ceil(movie * float(params.get(
        "maximum_event_span_movie_fraction", 0.10)))))
    minimum_track = max(1, int(math.ceil(movie * float(params.get(
        "minimum_context_track_span_movie_fraction", 0.25)))))
    minimum_coverage = float(params.get(
        "minimum_event_visible_coverage", 0.80))
    minimum_unowned = float(params.get(
        "minimum_event_unowned_fraction", 1.0))
    minimum_floor = float(params.get("minimum_radius_floor_fraction", 0.95))
    minimum_cores = int(params.get("minimum_raw_core_observations", 3))
    maximum_core = float(params.get(
        "maximum_core_area_fraction_of_typical_soma", 0.25))
    visible_all = points[points.state.astype(str).isin(VISIBLE_STATES)]
    radius_floor = float(visible_all.radius_px.astype(float).min())
    strong = points[
        points.state.astype(str).eq("observed")
        & points.strong.map(_truth)
        & points.radius_px.astype(float).gt(radius_floor + 1e-9)]
    if strong.empty:
        strong = points[points.state.astype(str).eq("observed")
                        & points.strong.map(_truth)]
    typical_area = float(np.median(
        np.pi * strong.radius_px.astype(float).to_numpy() ** 2))
    threshold_by_frame = thresholds.set_index("frame")
    groups = {int(track): group.sort_values("frame")
              for track, group in points.groupby("track_id", sort=True)}
    core_params = {
        "core_window_radius_scale": float(params.get(
            "core_window_radius_scale", 2.0)),
        "core_peak_fraction": float(params.get("core_peak_fraction", 0.35)),
        "weak_threshold_fraction": float(params.get(
            "weak_threshold_fraction", 0.50)),
    }
    removed = set()
    rows = []
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        sources = event_members.drop_duplicates("source_id")
        tracks = _tracks(event_members)
        first, last = int(event.first_frame), int(event.last_frame)
        event_span = last - first + 1
        reasons = []
        if str(event.family) != "unclaimed_visible_body":
            reasons.append("not_unclaimed_body_family")
        if (sources.empty or not sources.source_family.astype(str)
                .eq("unclaimed_visible_body").all()):
            reasons.append("sources_not_all_unclaimed_body")
        if len(tracks) != 1:
            reasons.append("not_one_reference")
        track = next(iter(tracks)) if len(tracks) == 1 else 0
        group = groups.get(track, pd.DataFrame())
        track_span = (int(group.frame.max()) - int(group.frame.min()) + 1
                      if len(group) else 0)
        owner_linked = bool(len(group)
                            and group.candidate_owner.astype(int).gt(0).any())
        visible = group[group.state.astype(str).isin(VISIBLE_STATES)] \
            if len(group) else group
        event_rows = visible[visible.frame.astype(int).between(first, last)] \
            if len(visible) else visible
        event_coverage = len(event_rows) / max(event_span, 1)
        event_unowned = float(
            event_rows.candidate_owner.astype(int).eq(0).mean()) \
            if len(event_rows) else 0.0
        floor_fraction = float(np.isclose(
            event_rows.radius_px.astype(float).to_numpy(), radius_floor,
            rtol=0.0, atol=1e-9).mean()) if len(event_rows) else 0.0
        if event_span > maximum_event:
            reasons.append("event_not_short")
        if track_span < minimum_track and not owner_linked:
            reasons.append("no_long_or_owner_linked_context")
        if event_coverage < minimum_coverage:
            reasons.append("insufficient_event_coverage")
        if event_unowned < minimum_unowned:
            reasons.append("event_not_fully_unowned")
        if floor_fraction < minimum_floor:
            reasons.append("radius_not_pinned_to_field_floor")

        core_areas = []
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
        if len(core_areas) < minimum_cores:
            reasons.append("insufficient_raw_core_observations")
        if core_fraction > maximum_core:
            reasons.append("raw_core_is_cell_sized")
        eligible = not reasons
        if eligible:
            removed.add(event_id)
        rows.append({
            "event_id": event_id, "family": str(event.family),
            "measured_track": track, "event_first": first,
            "event_last": last, "event_span_frames": event_span,
            "maximum_event_span_frames": maximum_event,
            "track_span_frames": track_span,
            "minimum_context_track_span_frames": minimum_track,
            "owner_linked_elsewhere": owner_linked,
            "event_visible_coverage": event_coverage,
            "event_unowned_fraction": event_unowned,
            "radius_floor_px": radius_floor,
            "event_radius_floor_fraction": floor_fraction,
            "typical_strong_soma_area_px": typical_area,
            "raw_core_observations": len(core_areas),
            "raw_core_area_p95_px": core_p95,
            "raw_core_fraction_of_typical_soma": core_fraction,
            "eligible": eligible,
            "reason": ("fragmented_subcellular_reference_episode"
                       if eligible else "|".join(dict.fromkeys(reasons))),
        })
    candidate = events[
        ~events.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    candidate["impact_rank"] = np.arange(1, len(candidate) + 1)
    candidate_members = members[
        ~members.event_id.astype(str).isin(removed)].copy().reset_index(drop=True)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(events)),
        "events_calibrated_nonbiological": int(len(removed)),
        "removed_event_ids": sorted(removed),
        "before": event_summary(events), "after": event_summary(candidate),
    }
    return candidate, candidate_members, pd.DataFrame(rows), summary


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
        "audit": out.out / "fragmented_subcellular_episode_audit.csv",
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
