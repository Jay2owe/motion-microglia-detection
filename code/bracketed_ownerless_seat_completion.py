"""Complete established, bracketed identity gaps from ownerless raw cores.

Discovery is complete-field. Biological targets and local selectors are not
part of the interface. The producer can only add the already established owner
to nonzero-signal pixels where both accepted ledgers are empty.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import component_accounting
from ownerless_body_recovery import evidence_image
from ownerless_cohort_latent_gap_completion import _nearest_raw_core
from resident_takeover import attach_owners


VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "well", "review_case", "case_id",
    "target_identity", "target_owner", "target_track", "target_frame",
    "target_event", "identity_target", "owner_target", "track_target",
    "frame_target", "coordinate_target", "event_target", "region_target",
    "review_case_target", "forced_identity", "forced_interval",
    "include_track", "exclude_track",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "bracketed ownerless-seat completion received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "bracketed ownerless-seat completion must be field-wide")


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    if not len(values):
        return []
    result: list[tuple[int, int]] = []
    first = previous = int(values[0])
    for value in map(int, values[1:]):
        if value != previous + 1:
            result.append((first, previous))
            first = value
        previous = value
    result.append((first, previous))
    return result


def _centroid(frame: np.ndarray, identity: int) -> tuple[float, float] | None:
    y, x = np.nonzero(frame == int(identity))
    if not len(x):
        return None
    return float(x.mean()), float(y.mean())


def _point_local_raw_core(frame: np.ndarray, evidence: np.ndarray, point,
                          weak_threshold: float, params: dict):
    """Find a core using only the tracked seat to set the local peak.

    A bright neighbouring cell may occupy the wider search window.  It must not
    raise the threshold for a dim, independently tracked seat.  Components are
    still searched in the wider window and the accepted-ledger overlap check is
    applied by the caller.
    """
    radius = max(float(point.radius_px), 1.0)
    maximum_offset = float(params.get("maximum_core_offset_radii", 4.5))
    half_window = max(
        int(params.get("core_window_radius_px", 7)),
        int(math.ceil(maximum_offset * radius)))
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    y0, y1 = max(0, y - half_window), min(frame.shape[0], y + half_window + 1)
    x0, x1 = max(0, x - half_window), min(frame.shape[1], x + half_window + 1)
    raw_local = frame[y0:y1, x0:x1]
    evidence_local = evidence[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    seed_radius = max(1.0, radius * float(params.get(
        "core_peak_reference_radius_scale", 1.0)))
    seed = ((yy - float(point.y)) ** 2 + (xx - float(point.x)) ** 2
            <= seed_radius ** 2)
    positive_seed = seed & (raw_local > 0)
    if not np.any(positive_seed):
        return None
    local_peak = float(evidence_local[positive_seed].max())
    level = max(
        float(params.get("core_peak_fraction", 0.35)) * local_peak,
        float(params.get("weak_threshold_fraction", 0.5))
        * float(weak_threshold))
    components, count = ndi.label(
        (evidence_local >= level) & (raw_local > 0),
        structure=np.ones((3, 3), np.uint8))
    maximum_area_ratio = float(params.get(
        "maximum_core_area_radius_ratio", 2.0))
    choices: list[tuple[float, int, int, float]] = []
    for component_id in range(1, int(count) + 1):
        cy, cx = np.nonzero(components == component_id)
        if not len(cx):
            continue
        distance = float(np.sqrt(np.min(
            (cy + y0 - float(point.y)) ** 2
            + (cx + x0 - float(point.x)) ** 2)))
        area = int(len(cx))
        area_ratio = area / max(math.pi * radius ** 2, 1.0)
        if (distance <= maximum_offset * radius
                and area_ratio <= maximum_area_ratio):
            choices.append((distance, -area, component_id, area_ratio))
    if not choices:
        return None
    distance, negative_area, component_id, area_ratio = min(choices)
    core = components == int(component_id)
    return core, (slice(y0, y1), slice(x0, x1)), {
        "core_area_px": int(-negative_area),
        "core_area_radius_ratio": float(area_ratio),
        "core_offset_radii": float(distance / radius),
        "core_threshold": float(level),
        "local_peak_evidence": local_peak,
    }


def _track_owner_table(attached: pd.DataFrame, minimum_support: int,
                       minimum_purity: float) -> pd.DataFrame:
    rows: list[dict] = []
    visible = attached[attached.state.astype(str).isin(VISIBLE_STATES)]
    for track, group in visible.groupby("track_id", sort=True):
        owned = group[group.candidate_owner.astype(int).gt(0)]
        counts = owned.candidate_owner.astype(int).value_counts()
        if not len(counts):
            continue
        top = int(counts.iloc[0])
        tied = int((counts == top).sum()) > 1
        owner = int(counts.index[0])
        purity = float(top / max(int(counts.sum()), 1))
        rows.append({
            "track_id": int(track), "inferred_owner": owner,
            "owner_support_frames": top, "owner_purity": purity,
            "eligible_owner_history": bool(
                not tied and top >= minimum_support
                and purity >= minimum_purity),
        })
    return pd.DataFrame(rows, columns=[
        "track_id", "inferred_owner", "owner_support_frames",
        "owner_purity", "eligible_owner_history"])


def complete(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
             points: pd.DataFrame, thresholds: pd.DataFrame, params: dict
             ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    """Return candidate labels plus complete gap and frame audit tables."""
    assert_target_free(params)
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")
    frames = len(labels)
    attached = attach_owners(points, labels)
    minimum_track_support = max(3, int(math.ceil(
        frames * float(params["minimum_track_owner_support_movie_fraction"]))))
    track_owners = _track_owner_table(
        attached, minimum_track_support,
        float(params["minimum_track_owner_purity"]))
    eligible_tracks = track_owners[
        track_owners.eligible_owner_history.astype(bool)]
    owner_for_track = dict(zip(
        eligible_tracks.track_id.astype(int),
        eligible_tracks.inferred_owner.astype(int)))
    attached["inferred_owner"] = attached.track_id.astype(int).map(
        owner_for_track).fillna(0).astype(int)

    assigned_radii = attached[
        attached.state.astype(str).isin(VISIBLE_STATES)
        & attached.strong.astype(bool)
        & attached.candidate_owner.astype(int).gt(0)].radius_px.astype(float)
    field_radius = float(assigned_radii.median()) if len(assigned_radii) else 1.0
    typical_core_area = math.pi * max(field_radius, 1.0) ** 2
    minimum_core_area = max(1, int(math.ceil(
        typical_core_area * float(params["minimum_core_area_soma_fraction"]))))
    minimum_identity_presence = max(2, int(math.ceil(
        frames * float(params["minimum_identity_presence_movie_fraction"]))))
    maximum_gap = max(1, int(math.ceil(
        frames * float(params["maximum_gap_movie_fraction"]))))
    flank_window = max(2, int(math.ceil(
        frames * float(params["flank_window_movie_fraction"]))))
    minimum_flank_fraction = float(params["minimum_flank_presence_fraction"])
    minimum_visible_fraction = float(params["minimum_visible_gap_fraction"])
    minimum_strong_fraction = float(params["minimum_strong_gap_fraction"])
    maximum_path_offset = float(params["maximum_path_offset_field_radii"])
    maximum_step = float(params["maximum_step_field_radii"])
    threshold_by_frame = thresholds.set_index("frame")
    evidence_cache: dict[int, np.ndarray] = {}
    candidate = labels.copy()
    gap_rows: list[dict] = []
    frame_rows: list[dict] = []

    identities = sorted(int(value) for value in np.unique(labels) if value > 0)
    for identity in identities:
        presence = np.any(labels == identity, axis=(1, 2))
        present = np.flatnonzero(presence)
        if len(present) < minimum_identity_presence:
            continue
        internal = np.flatnonzero(
            ~presence & (np.arange(frames) > int(present[0]))
            & (np.arange(frames) < int(present[-1])))
        for gap_first, gap_last in _runs(internal):
            proposal_id = f"BOS{len(gap_rows) + 1:04d}"
            gap_frames = gap_last - gap_first + 1
            left = np.arange(max(0, gap_first - flank_window), gap_first)
            right = np.arange(
                gap_last + 1, min(frames, gap_last + flank_window + 1))
            left_fraction = float(presence[left].mean()) if len(left) else 0.0
            right_fraction = float(presence[right].mean()) if len(right) else 0.0
            before = _centroid(labels[gap_first - 1], identity)
            after = _centroid(labels[gap_last + 1], identity)
            reason = "eligible_bracketed_ownerless_seat"
            if gap_frames > maximum_gap:
                reason = "gap_too_long"
            elif (left_fraction < minimum_flank_fraction
                  or right_fraction < minimum_flank_fraction):
                reason = "insufficient_bilateral_owner_support"
            elif before is None or after is None:
                reason = "immediate_owner_bookend_missing"

            selected: list[tuple[int, pd.Series, float, float]] = []
            visible_selected = 0
            if reason == "eligible_bracketed_ownerless_seat":
                previous = before
                for frame in range(gap_first, gap_last + 1):
                    fraction = (frame - gap_first + 1) / (gap_frames + 1)
                    predicted = (
                        (1.0 - fraction) * before[0] + fraction * after[0],
                        (1.0 - fraction) * before[1] + fraction * after[1])
                    rows = attached[
                        attached.frame.astype(int).eq(frame)
                        & attached.state.astype(str).isin(VISIBLE_STATES)
                        & attached.physically_visible.astype(bool)
                        & attached.candidate_owner.astype(int).eq(0)
                        & attached.inferred_owner.astype(int).eq(identity)].copy()
                    used_dormant_support = False
                    if (not len(rows)
                            and bool(params.get(
                                "allow_dormant_track_raw_support", False))):
                        rows = attached[
                            attached.frame.astype(int).eq(frame)
                            & attached.candidate_owner.astype(int).eq(0)
                            & attached.inferred_owner.astype(int).eq(identity)
                        ].copy()
                        used_dormant_support = len(rows) > 0
                    if not len(rows):
                        continue
                    rows["path_offset"] = np.hypot(
                        rows.x.astype(float) - predicted[0],
                        rows.y.astype(float) - predicted[1]) / max(field_radius, 1.0)
                    rows["step"] = np.hypot(
                        rows.x.astype(float) - previous[0],
                        rows.y.astype(float) - previous[1]) / max(field_radius, 1.0)
                    rows = rows[
                        rows.path_offset.astype(float).le(maximum_path_offset)
                        & rows.step.astype(float).le(maximum_step)]
                    if not len(rows):
                        continue
                    point = rows.sort_values(
                        ["evidence", "path_offset", "track_id"],
                        ascending=[False, True, True]).iloc[0]
                    selected.append((
                        frame, point, float(point.path_offset),
                        float(point.step)))
                    if not used_dormant_support:
                        visible_selected += 1
                    previous = (float(point.x), float(point.y))

                visible_fraction = visible_selected / gap_frames
                strong_fraction = (
                    sum(bool(item[1].strong) for item in selected
                        if str(item[1].state) in VISIBLE_STATES)
                    / max(visible_selected, 1))
                if visible_fraction < minimum_visible_fraction:
                    reason = "insufficient_ownerless_visible_chain"
                elif strong_fraction < minimum_strong_fraction:
                    reason = "ownerless_chain_too_weak"
            else:
                visible_fraction = 0.0
                strong_fraction = 0.0

            pending: list[tuple[int, tuple[slice, slice], np.ndarray, dict,
                                pd.Series, float, float]] = []
            failure = ""
            if reason == "eligible_bracketed_ownerless_seat":
                for frame, point, path_offset, step in selected:
                    if frame not in evidence_cache:
                        evidence_cache[frame] = evidence_image(raw[frame], params)
                    core_finder = (_point_local_raw_core
                                   if bool(params.get(
                                       "point_local_core_threshold", False))
                                   else _nearest_raw_core)
                    found = core_finder(
                        raw[frame], evidence_cache[frame], point,
                        float(threshold_by_frame.loc[frame, "weak_threshold"]),
                        params)
                    if found is None:
                        frame_rows.append({
                            "proposal_id": proposal_id, "frame": frame,
                            "track_id": int(point.track_id),
                            "inferred_owner": identity, "selected": True,
                            "applied": False, "reason": "qualified_core_missing",
                        })
                        continue
                    core, region, detail = found
                    occupied = ((labels[frame][region] > 0)
                                | (unclaimed[frame][region] > 0))
                    if np.any(core & occupied):
                        frame_rows.append({
                            "proposal_id": proposal_id, "frame": frame,
                            "track_id": int(point.track_id),
                            "inferred_owner": identity, "selected": True,
                            "applied": False,
                            "reason": "qualified_core_overlaps_accepted_ledger",
                        })
                        continue
                    if int(core.sum()) < minimum_core_area:
                        frame_rows.append({
                            "proposal_id": proposal_id, "frame": frame,
                            "track_id": int(point.track_id),
                            "inferred_owner": identity, "selected": True,
                            "applied": False, "reason": "core_below_field_scale",
                        })
                        continue
                    pending.append((
                        frame, region, core, detail, point, path_offset, step))
                raw_fraction = len(pending) / gap_frames
                if raw_fraction < float(params["minimum_raw_core_gap_fraction"]):
                    failure = "insufficient_qualified_raw_core_coverage"
            else:
                raw_fraction = 0.0

            pending_by_frame = {item[0]: item for item in pending}
            edge_frames: set[int] = set()
            cursor = gap_first
            while cursor in pending_by_frame:
                edge_frames.add(cursor)
                cursor += 1
            cursor = gap_last
            while cursor in pending_by_frame:
                edge_frames.add(cursor)
                cursor -= 1
            unsupported_frames = gap_frames - len(pending_by_frame)
            maximum_unsupported = max(0, int(math.floor(
                frames * float(params.get(
                    "maximum_raw_unsupported_movie_fraction", 0.0)))))
            completion_frames = edge_frames
            if (bool(params.get("allow_supported_interior_completion", False))
                    and unsupported_frames <= maximum_unsupported):
                completion_frames = set(pending_by_frame)
            edge_pending = [item for item in pending if item[0] in edge_frames]
            changed_pixels = 0
            changed_frames = 0
            applied = reason == "eligible_bracketed_ownerless_seat" and not failure
            if applied:
                for frame, region, core, detail, point, path_offset, step in pending:
                    if frame not in completion_frames:
                        frame_rows.append({
                            "proposal_id": proposal_id, "frame": frame,
                            "track_id": int(point.track_id),
                            "inferred_owner": identity, "selected": True,
                            "applied": False,
                            "reason": "interior_island_not_bookend_connected",
                            "core_area_px": int(core.sum()),
                        })
                        continue
                    local = candidate[frame][region]
                    local[core] = identity
                    changed_pixels += int(core.sum())
                    changed_frames += 1
                    frame_rows.append({
                        "proposal_id": proposal_id, "frame": frame,
                        "track_id": int(point.track_id),
                        "inferred_owner": identity, "selected": True,
                        "applied": True, "reason": "raw_core_seat_completed",
                        "core_area_px": int(core.sum()),
                        "core_offset_radii": detail["core_offset_radii"],
                        "core_area_radius_ratio": detail[
                            "core_area_radius_ratio"],
                        "path_offset_field_radii": path_offset,
                        "step_field_radii": step,
                    })
            gap_rows.append({
                "proposal_id": proposal_id,
                "discovered_owner": identity,
                "gap_first_frame": gap_first,
                "gap_last_frame": gap_last,
                "gap_frames": gap_frames,
                "left_owner_fraction": left_fraction,
                "right_owner_fraction": right_fraction,
                "ownerless_visible_frames": visible_selected,
                "ownerless_visible_fraction": visible_fraction,
                "ownerless_strong_fraction": strong_fraction,
                "qualified_raw_core_frames": len(pending),
                "qualified_raw_core_fraction": raw_fraction,
                "bookend_connected_core_frames": len(edge_pending),
                "unsupported_raw_frames": unsupported_frames,
                "maximum_unsupported_raw_frames": maximum_unsupported,
                "completion_strategy": (
                    "all_supported_with_bounded_internal_absence"
                    if completion_frames == set(pending_by_frame)
                    and bool(params.get(
                        "allow_supported_interior_completion", False))
                    else "bookend_connected_only"),
                "changed_pixels": changed_pixels,
                "changed_frames": changed_frames,
                "applied": applied,
                "reason": failure or reason,
            })

    changed = labels != candidate
    additions = changed & (labels == 0) & (candidate > 0)
    if np.any(changed & (labels > 0)):
        raise AssertionError("bracketed completion changed assigned pixels")
    if np.any(additions & (unclaimed > 0)):
        raise AssertionError("bracketed completion overlapped unclaimed pixels")
    if np.any(additions & (raw == 0)):
        raise AssertionError("bracketed completion added zero-signal pixels")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("bracketed completion changed the identity set")
    duplicates = component_accounting.new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError(
            "bracketed completion created duplicate identity components")
    def persistence_counts(stack: np.ndarray) -> tuple[int, int]:
        gap_frames = gap_runs = 0
        for value in sorted(int(item) for item in np.unique(stack) if item > 0):
            present = np.any(stack == value, axis=(1, 2))
            occupied = np.flatnonzero(present)
            if len(occupied) < 2:
                continue
            missing = np.flatnonzero(
                ~present & (np.arange(len(stack)) > int(occupied[0]))
                & (np.arange(len(stack)) < int(occupied[-1])))
            gap_frames += int(len(missing))
            gap_runs += len(_runs(missing))
        return gap_frames, gap_runs

    gap_frames_before, gap_runs_before = persistence_counts(labels)
    gap_frames_after, gap_runs_after = persistence_counts(candidate)
    if gap_runs_after > gap_runs_before:
        raise AssertionError("bracketed completion increased gap runs")
    audit = pd.DataFrame(gap_rows)
    frame_audit = pd.DataFrame(frame_rows)
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "identity_gaps_audited": int(len(audit)),
        "eligible_gaps": int(audit.applied.astype(bool).sum())
            if len(audit) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(frames, -1).any(axis=1))),
        "preexisting_assigned_changed_pixels": int(np.count_nonzero(
            changed & (labels > 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "new_identity_count": 0,
        "new_duplicate_components": int(duplicates),
        "gap_frames_before": gap_frames_before,
        "gap_frames_after": gap_frames_after,
        "gap_runs_before": gap_runs_before,
        "gap_runs_after": gap_runs_after,
        "field_radius_px": field_radius,
        "minimum_core_area_px": minimum_core_area,
    }
    return candidate, audit, frame_audit, metrics


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw_all = tifffile.imread(params["raw_path"])
    offset = int(params.get("raw_frame_offset", 0))
    raw = raw_all[offset:offset + len(labels)]
    points = pd.read_csv(params["points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    candidate, audit, frames, metrics = complete(
        labels, unclaimed, raw, points, thresholds, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
        "audit": out.out / "bracketed_ownerless_seat_audit.csv",
        "frames": out.out / "bracketed_ownerless_seat_frames.csv",
        "metrics": out.out / "producer_metrics.json",
    }
    tifffile.imwrite(outputs["labels"], candidate, compression="zlib")
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    audit.to_csv(outputs["audit"], index=False)
    frames.to_csv(outputs["frames"], index=False)
    metrics["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes())
    outputs["metrics"].write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": metrics}
