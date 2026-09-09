"""Recover fragmented ownerless lineages with one smooth raw core per frame."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile

import body_scale_ownerless_allocation as body_scale
import owner_consensus
import ownerless_body_recovery as ownerless


FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "well_target", "allowed_", "forced_", "include_", "exclude_",
)
AUDIT_COLUMNS = [
    "event_id", "family", "first_frame", "last_frame", "span_frames",
    "minimum_span_frames", "physical_track_count", "visible_frames",
    "visible_coverage", "observed_frame_fraction", "strong_frame_fraction",
    "maximum_internal_gap_frames", "allowed_internal_gap_frames",
    "maximum_assigned_coverage", "maximum_unclaimed_coverage",
    "allowed_unclaimed_coverage",
    "interpolated_frames", "candidate_core_frames", "selected_core_frames",
    "body_supported_fraction", "minimum_body_supported_fraction",
    "maximum_selected_step_radii", "allowed_selected_step_radii",
    "median_selected_core_area", "eligible", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("temporal core-path lineage must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "temporal core-path lineage received forbidden targets: "
            + ", ".join(supplied))


def _blank(value) -> bool:
    return pd.isna(value) or not str(value).strip()


def _track_ids(value) -> set[int]:
    if _blank(value):
        return set()
    return {int(float(item)) for item in str(value).split("|") if item.strip()}


def _boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _maximum_internal_gap(frames: list[int]) -> int:
    if len(frames) < 2:
        return 0
    return max(b - a - 1 for a, b in zip(frames[:-1], frames[1:]))


def _representative(group: pd.DataFrame) -> dict:
    return {
        "x": float(group.x.median()), "y": float(group.y.median()),
        "radius_px": float(group.radius_px.median()),
    }


def _complete_seed_table(group: pd.DataFrame, first: int, last: int):
    """Add linear seeds only for internal frames missing physical references."""
    groups = {int(frame): frame_group.copy()
              for frame, frame_group in group.groupby("frame", sort=True)}
    observed_frames = sorted(groups)
    rows = []
    interpolated = set()
    for frame in range(first, last + 1):
        if frame in groups:
            for row in groups[frame].to_dict("records"):
                row["_interpolated_seed"] = False
                rows.append(row)
            continue
        before = [value for value in observed_frames if value < frame]
        after = [value for value in observed_frames if value > frame]
        if not before or not after:
            continue
        left, right = before[-1], after[0]
        left_point, right_point = (
            _representative(groups[left]), _representative(groups[right]))
        fraction = (frame - left) / max(right - left, 1)
        rows.append({
            "frame": frame,
            "x": left_point["x"] + fraction * (
                right_point["x"] - left_point["x"]),
            "y": left_point["y"] + fraction * (
                right_point["y"] - left_point["y"]),
            "radius_px": left_point["radius_px"] + fraction * (
                right_point["radius_px"] - left_point["radius_px"]),
            "_interpolated_seed": True,
        })
        interpolated.add(frame)
    return pd.DataFrame(rows), interpolated


def _component_candidates(raw: np.ndarray, thresholds: pd.DataFrame,
                          seeds: pd.DataFrame, params: dict):
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.5))
    minimum_ratio = float(params.get("minimum_core_area_radius_ratio", 0.25))
    by_frame = {}
    frame_rows = []
    for frame, group in seeds.groupby("frame", sort=True):
        frame = int(frame)
        candidates = []
        seen = set()
        for point in group.itertuples(index=False):
            core, region, _ = body_scale._connected_core_with_relative_seed_snap(
                raw[frame], point,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            mask = np.zeros(raw.shape[1:], dtype=bool)
            mask[region] = core
            area = int(mask.sum())
            if not area:
                continue
            fingerprint = np.packbits(mask).tobytes()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            yy, xx = np.nonzero(mask)
            typical_radius = max(float(point.radius_px), 1.0)
            ratio = area / max(math.pi * typical_radius ** 2, 1.0)
            supported = minimum_ratio <= ratio <= maximum_ratio
            if area and ratio <= maximum_ratio:
                candidates.append({
                    "mask": mask, "area": area,
                    "x": float(xx.mean()), "y": float(yy.mean()),
                    "radius": typical_radius, "area_radius_ratio": ratio,
                    "body_supported": supported,
                })
        by_frame[frame] = candidates
        frame_rows.append({
            "frame": frame,
            "input_seed_count": int(len(group)),
            "interpolated_seed": bool(group["_interpolated_seed"].any()),
            "raw_component_count": int(len(seen)),
            "safe_candidate_count": int(len(candidates)),
        })
    return by_frame, frame_rows


def _smooth_path(candidates: dict[int, list[dict]], first: int, last: int,
                 maximum_step_radii: float):
    frames = list(range(first, last + 1))
    if any(not candidates.get(frame) for frame in frames):
        return None, float("inf"), "missing_candidate_core_frame"
    costs = []
    parents = []
    first_nodes = candidates[frames[0]]
    costs.append(np.asarray([-0.02 * node["area"]
                             for node in first_nodes], dtype=float))
    parents.append(np.full(len(first_nodes), -1, dtype=int))
    for index in range(1, len(frames)):
        previous = candidates[frames[index - 1]]
        current = candidates[frames[index]]
        current_cost = np.full(len(current), np.inf, dtype=float)
        current_parent = np.full(len(current), -1, dtype=int)
        for j, node in enumerate(current):
            for i, prior in enumerate(previous):
                scale = max(0.5 * (node["radius"] + prior["radius"]), 1.0)
                step = math.hypot(node["x"] - prior["x"],
                                  node["y"] - prior["y"]) / scale
                if step > maximum_step_radii:
                    continue
                cost = costs[-1][i] + step - 0.02 * node["area"]
                if cost < current_cost[j]:
                    current_cost[j] = cost
                    current_parent[j] = i
        if not np.isfinite(current_cost).any():
            return None, float("inf"), "no_smooth_core_path"
        costs.append(current_cost)
        parents.append(current_parent)
    selected_indices = [int(np.argmin(costs[-1]))]
    for index in range(len(frames) - 1, 0, -1):
        selected_indices.append(int(parents[index][selected_indices[-1]]))
    selected_indices.reverse()
    selected = {frame: candidates[frame][choice]
                for frame, choice in zip(frames, selected_indices)}
    maximum_step = 0.0
    for left, right in zip(frames[:-1], frames[1:]):
        a, b = selected[left], selected[right]
        scale = max(0.5 * (a["radius"] + b["radius"]), 1.0)
        maximum_step = max(maximum_step, math.hypot(
            b["x"] - a["x"], b["y"] - a["y"]) / scale)
    return selected, maximum_step, "smooth_core_path"


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            events: pd.DataFrame, points: pd.DataFrame,
            thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")
    attached = ownerless.attach_coverage(points, labels, unclaimed)
    attached = attached.assign(
        _visible=_boolean(attached["physically_visible"]),
        _strong=_boolean(attached["strong"]),
        _observed=attached["state"].astype(str).eq("observed"))
    minimum_span = max(2, int(math.ceil(float(params.get(
        "minimum_lineage_span_movie_fraction", 0.35)) * len(labels))))
    minimum_coverage = float(params.get(
        "minimum_visible_coverage_of_span", 0.80))
    minimum_observed = float(params.get(
        "minimum_observed_frame_fraction", 0.70))
    minimum_strong = float(params.get(
        "minimum_strong_frame_fraction", 0.65))
    allowed_gap = int(math.floor(float(params.get(
        "maximum_internal_gap_movie_fraction", 0.05)) * len(labels)))
    maximum_assigned = float(params.get(
        "maximum_existing_assigned_coverage_fraction", 0.0))
    maximum_unclaimed = float(params.get(
        "maximum_existing_unclaimed_coverage_fraction", 1.0))
    minimum_body = float(params.get("minimum_body_supported_fraction", 0.90))
    maximum_step = float(params.get("maximum_selected_step_radii", 5.0))
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    next_identity = int(labels.max()) + 1
    audits = []
    applications = []
    frame_audits = []

    ordered = events.sort_values(
        ["first_frame", "last_frame", "event_id"], kind="stable")
    for event in ordered.itertuples(index=False):
        first, last = int(event.first_frame), int(event.last_frame)
        span = last - first + 1
        tracks = _track_ids(event.physical_tracks)
        reasons = []
        if str(event.family) != "reference_fragmentation":
            reasons.append("not_reference_fragmentation")
        if first != 0:
            reasons.append("not_left_censored")
        if not _blank(event.accepted_before) or not _blank(
                event.accepted_after):
            reasons.append("accepted_owner_context_present")
        if span < minimum_span:
            reasons.append("lineage_span_too_short")
        if not tracks:
            reasons.append("no_physical_tracks")
        group = attached[
            attached.track_id.astype(int).isin(tracks)
            & attached.frame.astype(int).between(first, last)
            & attached["_visible"]].copy()
        frames = sorted(set(group.frame.astype(int)))
        coverage = len(frames) / max(span, 1)
        observed = float(group.groupby("frame")._observed.any().mean()) \
            if len(group) else 0.0
        strong = float(group.groupby("frame")._strong.any().mean()) \
            if len(group) else 0.0
        internal_gap = _maximum_internal_gap(frames)
        assigned_max = float(group.assigned_coverage_fraction.max()) \
            if len(group) else 0.0
        unclaimed_max = float(group.unclaimed_coverage_fraction.max()) \
            if len(group) else 0.0
        if coverage < minimum_coverage:
            reasons.append("insufficient_temporal_coverage")
        if observed < minimum_observed:
            reasons.append("insufficient_direct_observations")
        if strong < minimum_strong:
            reasons.append("insufficient_strong_signal")
        if internal_gap > allowed_gap:
            reasons.append("internal_gap_too_long")
        if assigned_max > maximum_assigned:
            reasons.append("assigned_coverage_present")
        if unclaimed_max > maximum_unclaimed:
            reasons.append("unclaimed_representation_present")

        interpolated = set()
        candidates = {}
        selected = None
        selected_step = float("nan")
        path_reason = "cheap_gate_failed"
        component_rows = []
        if not reasons:
            seeds, interpolated = _complete_seed_table(group, first, last)
            candidates, component_rows = _component_candidates(
                raw, thresholds, seeds, params)
            selected, selected_step, path_reason = _smooth_path(
                candidates, first, last, maximum_step)
            if selected is None:
                reasons.append(path_reason)
            elif np.mean([node["body_supported"]
                          for node in selected.values()]) < minimum_body:
                reasons.append("insufficient_selected_body_scale_support")
        selected_body = (float(np.mean([node["body_supported"]
                                        for node in selected.values()]))
                         if selected else float("nan"))
        selected_areas = ([node["area"] for node in selected.values()]
                          if selected else [])
        row = {
            "event_id": str(event.event_id), "family": str(event.family),
            "first_frame": first, "last_frame": last, "span_frames": span,
            "minimum_span_frames": minimum_span,
            "physical_track_count": len(tracks), "visible_frames": len(frames),
            "visible_coverage": coverage,
            "observed_frame_fraction": observed,
            "strong_frame_fraction": strong,
            "maximum_internal_gap_frames": internal_gap,
            "allowed_internal_gap_frames": allowed_gap,
            "maximum_assigned_coverage": assigned_max,
            "maximum_unclaimed_coverage": unclaimed_max,
            "allowed_unclaimed_coverage": maximum_unclaimed,
            "interpolated_frames": len(interpolated),
            "candidate_core_frames": len(candidates),
            "selected_core_frames": len(selected) if selected else 0,
            "body_supported_fraction": selected_body,
            "minimum_body_supported_fraction": minimum_body,
            "maximum_selected_step_radii": selected_step,
            "allowed_selected_step_radii": maximum_step,
            "median_selected_core_area": (float(np.median(selected_areas))
                                           if selected_areas else float("nan")),
            "eligible": not reasons,
            "reason": ("eligible_temporally_smooth_ownerless_core_path"
                       if not reasons else "|".join(reasons)),
        }
        audits.append(row)
        for item in component_rows:
            frame = int(item["frame"])
            node = selected.get(frame) if selected else None
            frame_audits.append({
                "event_id": str(event.event_id), **item,
                "selected": node is not None,
                "selected_area": node["area"] if node else 0,
                "selected_x": node["x"] if node else float("nan"),
                "selected_y": node["y"] if node else float("nan"),
                "selected_area_radius_ratio": (
                    node["area_radius_ratio"] if node else float("nan")),
            })
        if reasons:
            continue
        trial = candidate.copy()
        trial_unclaimed = candidate_unclaimed.copy()
        overlap = False
        for frame, node in selected.items():
            mask = node["mask"]
            if np.any(mask & ((trial[frame] > 0) | occupied[frame])):
                overlap = True
                break
            trial[frame][mask] = next_identity
            trial_unclaimed[frame][mask] = 0
        if overlap:
            row["eligible"] = False
            row["reason"] = "overlapping_proposal_or_existing_label"
            continue
        if owner_consensus.count_new_duplicate_components(candidate, trial):
            row["eligible"] = False
            row["reason"] = "new_duplicate_identity_component"
            continue
        changed = trial != candidate
        claimed_unclaimed = int(np.count_nonzero(
            (candidate_unclaimed > 0) & (trial_unclaimed == 0)))
        for frame, node in selected.items():
            occupied[frame] |= node["mask"]
        candidate = trial
        candidate_unclaimed = trial_unclaimed
        applications.append({
            "proposal_id": f"FL{len(applications) + 1:04d}",
            "source_event_id": str(event.event_id),
            "assigned_identity": next_identity, "applied": True,
            "first_frame": first, "last_frame": last,
            "changed_pixels": int(changed.sum()),
            "changed_frames": int(np.count_nonzero(
                changed.reshape(len(changed), -1).any(axis=1))),
            "interpolated_frames": len(interpolated),
            "unclaimed_pixels_transferred": claimed_unclaimed,
            "body_supported_fraction": selected_body,
            "reason": "complete_temporally_smooth_ownerless_core_path",
        })
        next_identity += 1

    audit = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    applications = pd.DataFrame(applications)
    frame_audit = pd.DataFrame(frame_audits)
    existing_changes = int(np.count_nonzero(
        (labels > 0) & (candidate != labels)))
    ledger_overlap = int(np.count_nonzero(
        (candidate_unclaimed > 0) & (candidate > 0)))
    unclaimed_additions = int(np.count_nonzero(
        (candidate_unclaimed > 0) & (unclaimed == 0)))
    zero_signal = int(np.count_nonzero(
        (candidate > 0) & (labels == 0) & (raw == 0)))
    duplicates = int(owner_consensus.count_new_duplicate_components(
        labels, candidate))
    if (existing_changes or ledger_overlap or unclaimed_additions
            or zero_signal or duplicates):
        raise AssertionError("temporal core-path lineage violated safety")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    changed = candidate != labels
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum()),
        "applied_proposals": int(len(applications)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "existing_label_changes": existing_changes,
        "unclaimed_overlap_pixels": ledger_overlap,
        "unclaimed_additions": unclaimed_additions,
        "unclaimed_pixels_transferred": int(np.count_nonzero(
            (unclaimed > 0) & (candidate_unclaimed == 0))),
        "zero_signal_additions": zero_signal,
        "new_duplicate_components": duplicates,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identities": sorted(after_ids - before_ids),
        "new_identity_count": len(after_ids - before_ids),
    }
    return (candidate, candidate_unclaimed, audit, applications,
            frame_audit, summary)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    events = pd.read_csv(params["events_path"])
    points = pd.read_csv(params["points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    (candidate, candidate_unclaimed, audit, applications,
     frame_audit, summary) = produce(
        labels, unclaimed, raw, events, points, thresholds, params)
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
    if np.array_equal(candidate_unclaimed, unclaimed):
        shutil.copyfile(unclaimed_path, unclaimed_out)
    else:
        tifffile.imwrite(
            unclaimed_out, candidate_unclaimed, imagej=True,
            compression="zlib", metadata={"axes": "TYX",
                                            "unit": "pixel"})
    audit_out = out.out / "temporal_core_path_lineage_audit.csv"
    applications_out = out.out / "temporal_core_path_lineage_applications.csv"
    frames_out = out.out / "temporal_core_path_frame_audit.csv"
    metrics_out = out.out / "producer_metrics.json"
    audit.to_csv(audit_out, index=False)
    applications.to_csv(applications_out, index=False)
    frame_audit.to_csv(frames_out, index=False)
    metrics_out.write_text(json.dumps(summary, indent=2) + "\n",
                           encoding="utf-8")
    return {"outputs": {
        "labels": labels_out, "unclaimed": unclaimed_out,
        "audit": audit_out, "applications": applications_out,
        "frame_audit": frames_out, "metrics": metrics_out},
        "summary": summary}
