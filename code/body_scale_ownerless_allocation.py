"""Allocate durable ownerless bodies only with scale-relative body support."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

import owner_consensus
import ownerless_body_recovery as ownerless


FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_", "forced_", "include_", "exclude_",
)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "run_first_frame", "run_last_frame",
    "run_frames", "minimum_run_frames", "second_run_frames",
    "longest_to_second_run_ratio", "observed_fraction", "strong_fraction",
    "median_separation_radii", "encounter_fraction",
    "maximum_assigned_coverage", "maximum_unclaimed_coverage",
    "median_core_area_radius_ratio", "body_supported_fraction",
    "minimum_body_supported_fraction", "failed_core_frames", "median_x",
    "median_y", "eligible", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("body-scale ownerless allocation must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "body-scale ownerless allocation received forbidden targets: "
            + ", ".join(supplied))


def _visible_runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    visible = group[group.physically_visible.astype(bool)].sort_values("frame")
    if visible.empty:
        return []
    split = visible.frame.astype(int).diff().fillna(1).ne(1).cumsum()
    return [run.copy() for _, run in visible.groupby(split)]


def _connected_core_with_relative_seed_snap(
        frame: np.ndarray, point, weak_threshold: float, params: dict):
    core, region, details = ownerless._connected_core(
        frame, point, weak_threshold, params)
    details = {**details, "seed_snapped": False,
               "seed_snap_distance_px": 0.0}
    if details["core_area_px"]:
        return core, region, details
    evidence = ownerless.evidence_image(frame, params)
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    radius = max(float(point.radius_px), 1.0)
    search_radius = max(1, int(math.ceil(float(params.get(
        "maximum_seed_snap_radius_fraction", 0.5)) * radius)))
    y0, y1 = max(0, y - search_radius), min(
        frame.shape[0], y + search_radius + 1)
    x0, x1 = max(0, x - search_radius), min(
        frame.shape[1], x + search_radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    allowed = (xx - x) ** 2 + (yy - y) ** 2 <= search_radius ** 2
    scores = np.where(allowed & (frame[y0:y1, x0:x1] > 0),
                      evidence[y0:y1, x0:x1], -np.inf)
    flat = int(np.argmax(scores))
    if not np.isfinite(float(scores.flat[flat])):
        return core, region, details
    sy, sx = np.unravel_index(flat, scores.shape)
    best_y, best_x = int(y0 + sy), int(x0 + sx)
    snapped = SimpleNamespace(
        x=float(best_x), y=float(best_y), radius_px=float(point.radius_px))
    core, region, snapped_details = ownerless._connected_core(
        frame, snapped, weak_threshold, params)
    return core, region, {
        **snapped_details,
        "seed_snapped": bool(snapped_details["core_area_px"]),
        "seed_snap_distance_px": float(math.hypot(best_x - x, best_y - y)),
    }


def _core_profile(raw: np.ndarray, thresholds: pd.DataFrame,
                  run: pd.DataFrame, params: dict):
    threshold_by_frame = thresholds.set_index("frame")
    minimum_ratio = float(params.get("minimum_core_area_radius_ratio", 0.25))
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.0))
    rows = []
    for point in run.itertuples(index=False):
        frame = int(point.frame)
        core, region, details = _connected_core_with_relative_seed_snap(
            raw[frame], point,
            float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
        expected = math.pi * max(float(point.radius_px), 1.0) ** 2
        ratio = float(details["core_area_px"] / max(expected, 1.0))
        rows.append({
            "frame": frame, "core": core, "region": region,
            "core_area_px": int(details["core_area_px"]),
            "area_radius_ratio": ratio,
            "body_supported": bool(minimum_ratio <= ratio <= maximum_ratio),
            "core_safe": bool(0 < ratio <= maximum_ratio),
            "seed_snapped": bool(details["seed_snapped"]),
        })
    return rows


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")
    attached = ownerless.attach_coverage(points, labels, unclaimed)
    minimum_run = max(2, int(math.ceil(float(params.get(
        "minimum_continuous_run_movie_fraction", 0.35)) * len(labels))))
    minimum_observed = float(params.get(
        "minimum_observed_fraction_of_run", 0.80))
    minimum_strong = float(params.get(
        "minimum_strong_fraction_of_observed", 0.70))
    minimum_separation = float(params.get(
        "minimum_median_separation_radii", 6.0))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.03))
    maximum_coverage = float(params.get(
        "maximum_existing_coverage_fraction", 0.0))
    minimum_uniqueness = float(params.get(
        "minimum_longest_to_second_run_ratio", 1.5))
    minimum_body_fraction = float(params.get(
        "minimum_body_supported_fraction", 0.5))
    candidate = labels.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    next_identity = int(labels.max()) + 1
    audits = []
    applications = []
    internal = []

    for track_id, group in attached.groupby("track_id", sort=True):
        runs = sorted(_visible_runs(group), key=len, reverse=True)
        if not runs:
            continue
        run = runs[0]
        second = len(runs[1]) if len(runs) > 1 else 0
        observed = run[run.state.astype(str).eq("observed")]
        radii = np.maximum(run.radius_px.to_numpy(float), 1.0)
        ratios = run.nearest_track_distance_px.to_numpy(float) / radii
        all_visible = group[group.physically_visible.astype(bool)]
        strong_fraction = float(observed.strong.astype(bool).mean()) \
            if len(observed) else 0.0
        assigned_max = float(all_visible.assigned_coverage_fraction.max())
        unclaimed_max = float(all_visible.unclaimed_coverage_fraction.max())
        reasons = []
        if len(run) < minimum_run:
            reasons.append("continuous_run_too_short")
        if len(run) / max(second, 1) < minimum_uniqueness:
            reasons.append("multiple_comparable_visible_runs")
        if len(observed) / max(len(run), 1) < minimum_observed:
            reasons.append("insufficient_direct_observations")
        if strong_fraction < minimum_strong:
            reasons.append("insufficient_strong_observations")
        if float(np.median(ratios)) < minimum_separation:
            reasons.append("insufficient_isolation")
        if float(run.in_encounter.astype(bool).mean()) > maximum_encounter:
            reasons.append("too_many_encounter_frames")
        if assigned_max > maximum_coverage:
            reasons.append("track_has_assigned_coverage")
        if unclaimed_max > maximum_coverage:
            reasons.append("track_has_unclaimed_coverage")

        # Raw-core extraction evaluates a full evidence image per frame. Apply
        # it only after the cheap complete-table gates have passed.
        profile = []
        median_core_ratio = float("nan")
        body_supported_fraction = float("nan")
        failed_core_frames = 0
        if not reasons:
            profile = _core_profile(raw, thresholds, run, params)
            core_ratios = np.asarray(
                [item["area_radius_ratio"] for item in profile], dtype=float)
            supported = np.asarray(
                [item["body_supported"] for item in profile], dtype=bool)
            safe = np.asarray(
                [item["core_safe"] for item in profile], dtype=bool)
            median_core_ratio = float(np.median(core_ratios))
            body_supported_fraction = float(np.mean(supported))
            failed_core_frames = int((~safe).sum())
            if not bool(np.all(safe)):
                reasons.append("unsafe_or_missing_raw_core")
            if body_supported_fraction < minimum_body_fraction:
                reasons.append("insufficient_body_scale_support")
        row = {
            "proposal_id": "", "track_id": int(track_id),
            "run_first_frame": int(run.frame.min()),
            "run_last_frame": int(run.frame.max()),
            "run_frames": int(len(run)), "minimum_run_frames": minimum_run,
            "second_run_frames": int(second),
            "longest_to_second_run_ratio": float(len(run) / max(second, 1)),
            "observed_fraction": float(len(observed) / max(len(run), 1)),
            "strong_fraction": strong_fraction,
            "median_separation_radii": float(np.median(ratios)),
            "encounter_fraction": float(run.in_encounter.astype(bool).mean()),
            "maximum_assigned_coverage": assigned_max,
            "maximum_unclaimed_coverage": unclaimed_max,
            "median_core_area_radius_ratio": median_core_ratio,
            "body_supported_fraction": body_supported_fraction,
            "minimum_body_supported_fraction": minimum_body_fraction,
            "failed_core_frames": failed_core_frames,
            "median_x": float(run.x.median()), "median_y": float(run.y.median()),
            "eligible": not reasons,
            "reason": ("eligible_body_scale_continuous_ownerless_body"
                       if not reasons else "|".join(reasons)),
        }
        audits.append(row)
        internal.append((row, profile))

    proposal_number = 0
    for row, profile in internal:
        if not row["eligible"]:
            continue
        proposal_number += 1
        row["proposal_id"] = f"CB{proposal_number:04d}"
        trial = candidate.copy()
        overlap = False
        for item in profile:
            frame, region, core = item["frame"], item["region"], item["core"]
            local_occupied = ((trial[frame][region] > 0)
                              | (unclaimed[frame][region] > 0)
                              | occupied[frame][region])
            if np.any(core & local_occupied):
                overlap = True
                break
            trial[frame][region][core] = next_identity
        if overlap:
            row["eligible"] = False
            row["reason"] = "overlapping_proposal_or_existing_ledger"
            continue
        changed = trial != candidate
        for item in profile:
            occupied[item["frame"]][item["region"]] |= item["core"]
        candidate = trial
        applications.append({
            "proposal_id": row["proposal_id"], "track_id": row["track_id"],
            "assigned_identity": next_identity, "applied": True,
            "changed_pixels": int(changed.sum()),
            "changed_frames": int(np.count_nonzero(
                changed.reshape(len(changed), -1).any(axis=1))),
            "snapped_seed_frames": int(sum(
                item["seed_snapped"] for item in profile)),
            "median_core_area_radius_ratio": row[
                "median_core_area_radius_ratio"],
            "body_supported_fraction": row["body_supported_fraction"],
            "reason": "complete_body_scale_continuous_raw_core_allocation",
        })
        next_identity += 1

    audit = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    applications = pd.DataFrame(applications)
    existing_changes = int(np.count_nonzero(
        (labels > 0) & (candidate != labels)))
    ledger_overlap = int(np.count_nonzero((unclaimed > 0) & (candidate > 0)))
    zero_signal = int(np.count_nonzero(
        (candidate > 0) & (labels == 0) & (raw == 0)))
    duplicates = int(owner_consensus.count_new_duplicate_components(
        labels, candidate))
    if existing_changes or ledger_overlap or zero_signal or duplicates:
        raise AssertionError("body-scale ownerless allocation violated safety")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    changed = candidate != labels
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "tracks_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum()),
        "applied_proposals": int(len(applications)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "existing_label_changes": existing_changes,
        "unclaimed_overlap_pixels": ledger_overlap,
        "zero_signal_additions": zero_signal,
        "new_duplicate_components": duplicates,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identities": sorted(after_ids - before_ids),
        "new_identity_count": len(after_ids - before_ids),
    }
    return candidate, audit, applications, summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    candidate, audit, applications, summary = produce(
        labels, unclaimed, raw, points, thresholds, params)
    output_stem = str(params.get("output_stem", "95_A3"))
    labels_out = out.out / f"{output_stem}.tif"
    unclaimed_out = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copyfile(labels_path, labels_out)
    else:
        tifffile.imwrite(
            labels_out, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(unclaimed_path, unclaimed_out)
    audit_out = out.out / "body_scale_ownerless_audit.csv"
    applications_out = out.out / "body_scale_ownerless_applications.csv"
    metrics_out = out.out / "metrics.json"
    audit.to_csv(audit_out, index=False)
    applications.to_csv(applications_out, index=False)
    metrics_out.write_text(json.dumps(summary, indent=2) + "\n",
                           encoding="utf-8")
    return {"outputs": {
        "labels": labels_out, "unclaimed": unclaimed_out,
        "audit": audit_out, "applications": applications_out,
        "metrics": metrics_out}, "summary": summary}

