"""Repair short A-B-A owner excursions on stable physical seats.

Discovery is field-wide and consumes only complete image-derived track evidence
and mathematical thresholds.  Named identities, tracks, frames, coordinates,
events, regions, and review cases are forbidden producer inputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile

import owner_consensus
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_pair", "forced_interval", "failure_target",
)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "owner_a", "owner_b", "middle_start",
    "middle_end", "middle_frames", "pre_frames", "post_frames",
    "stable_owner_support", "positive_owner_observations",
    "stable_owner_purity", "visible_span_frames", "strong_span_frames",
    "maximum_step_sum_radii", "middle_encounter_frames", "companion_track",
    "companion_owner_support", "companion_middle_overlap",
    "companion_first_visible_frame", "companion_onset_lead_frames",
    "other_owner_a_cores", "terminal_return", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "track_id", "companion_track", "frame", "owner_a",
    "owner_b", "partition_method", "changed_pixels", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("bounded owner excursion must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "bounded owner excursion received forbidden targets: "
            + ", ".join(supplied))


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for point in group.sort_values("frame").itertuples(index=False):
        owner = int(point.candidate_owner)
        if not bool(point.physically_visible) or owner <= 0:
            continue
        frame = int(point.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame == runs[-1]["end"] + 1):
            runs[-1]["end"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "start": frame, "end": frame,
                         "frames": [frame]})
    return runs


def _scaled_step(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(float(left.radius_px) + float(right.radius_px), 1.0)


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    if count <= 0:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    value = int(components[y, x])
    if value <= 0:
        radius = max(2.0, 0.75 * float(point.radius_px))
        yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
        disk = ((xx - float(point.x)) ** 2
                + (yy - float(point.y)) ** 2 <= radius ** 2)
        values = components[disk & (components > 0)]
        if not len(values):
            return None
        ids, counts = np.unique(values, return_counts=True)
        value = int(ids[int(np.argmax(counts))])
    return components == value


def _marker(mask: np.ndarray, point) -> tuple[int, int] | None:
    y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
    if mask[y, x]:
        return y, x
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return None
    radius = max(2.0, float(point.radius_px))
    distance = np.hypot(yy - float(point.y), xx - float(point.x))
    closest = int(np.argmin(distance))
    if float(distance[closest]) > radius:
        return None
    return int(yy[closest]), int(xx[closest])


def _seed_component(mask: np.ndarray,
                    seed: tuple[int, int]) -> np.ndarray:
    components, _ = ndi.label(mask, STRUCTURE)
    value = int(components[seed])
    return components == value if value else np.zeros_like(mask)


def _connected_partition(target: np.ndarray, resident_part: np.ndarray,
                         host_part: np.ndarray,
                         resident_marker: tuple[int, int],
                         host_markers: list[tuple[int, int]],
                         minimum_pixels: int
                         ) -> tuple[np.ndarray, np.ndarray] | None:
    """Make a two-owner partition exhaustive and connected for both owners."""
    options: list[tuple[int, np.ndarray, np.ndarray]] = []
    resident_core = _seed_component(resident_part, resident_marker)
    host_complement = target & ~resident_core
    if (all(bool(host_complement[marker]) for marker in host_markers)
            and int(ndi.label(host_complement, STRUCTURE)[1]) == 1
            and int(resident_core.sum()) >= minimum_pixels
            and int(host_complement.sum()) >= minimum_pixels):
        options.append((abs(int(resident_core.sum())
                            - int(resident_part.sum())),
                        resident_core, host_complement))
    # The combined host watershed can contain several source basins but must
    # form one connected owner-B remainder.  Its seed component is safe only
    # when it includes every independently observed host marker.
    host_core = _seed_component(host_part, host_markers[0])
    resident_complement = target & ~host_core
    if (all(bool(host_core[marker]) for marker in host_markers)
            and bool(resident_complement[resident_marker])
            and int(ndi.label(resident_complement, STRUCTURE)[1]) == 1
            and int(resident_complement.sum()) >= minimum_pixels
            and int(host_core.sum()) >= minimum_pixels):
        options.append((abs(int(resident_complement.sum())
                            - int(resident_part.sum())),
                        resident_complement, host_core))
    if not options:
        return None
    _, resident, host = min(options, key=lambda option: option[0])
    return resident, host


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _companion(attached: pd.DataFrame, resident_track: int, owner_b: int,
               middle_frames: list[int], minimum_support: int,
               minimum_overlap: int, maximum_distance: float
               ) -> tuple[int, int, int, int]:
    resident = attached[
        attached.track_id.astype(int) == int(resident_track)] \
        .set_index("frame", drop=False)
    candidates: list[tuple[int, int, int, int]] = []
    for track, group in attached[
            (attached.track_id.astype(int) != int(resident_track))
            & attached.physically_visible.astype(bool)].groupby("track_id"):
        owner_rows = group[group.candidate_owner.astype(int) == int(owner_b)]
        support = int(len(owner_rows))
        if support < minimum_support:
            continue
        overlap = 0
        for point in owner_rows[
                owner_rows.frame.astype(int).isin(middle_frames)] \
                .itertuples(index=False):
            frame = int(point.frame)
            if frame not in resident.index:
                continue
            seat = resident.loc[frame]
            if isinstance(seat, pd.DataFrame):
                seat = seat.iloc[0]
            if _scaled_step(seat, point) <= maximum_distance:
                overlap += 1
        if overlap >= minimum_overlap:
            candidates.append((overlap, support, int(track),
                               int(group.frame.astype(int).min())))
    if not candidates:
        return 0, 0, 0, -1
    overlap, support, track, first = max(candidates)
    return track, support, overlap, first


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every A-B-A run triple and retain only general-rule matches."""
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    frame_count = int(len(labels))
    minimum_visible = int(math.ceil(
        float(params.get("minimum_visible_movie_fraction", 0.33))
        * frame_count))
    minimum_stable = int(math.ceil(
        float(params.get("minimum_stable_support_movie_fraction", 0.20))
        * frame_count))
    minimum_bookend = int(math.ceil(
        float(params.get("minimum_bookend_movie_fraction", 0.08))
        * frame_count))
    maximum_middle = int(math.ceil(
        float(params.get("maximum_middle_movie_fraction", 0.06))
        * frame_count))
    minimum_companion = int(math.ceil(
        float(params.get("minimum_companion_support_movie_fraction", 0.05))
        * frame_count))
    minimum_purity = float(params.get("minimum_stable_owner_purity", 0.8))
    maximum_step = float(params.get("maximum_step_sum_radii", 1.0))
    maximum_companion_distance = float(params.get(
        "maximum_companion_distance_sum_radii", 3.0))
    maximum_companion_onset_lead = int(math.ceil(
        float(params.get("maximum_companion_onset_lead_movie_fraction", 0.02))
        * frame_count))
    rows: list[dict] = []
    internals: list[dict] = []
    for track, group in attached.groupby("track_id"):
        group = group.sort_values("frame")
        positive = group[
            group.physically_visible.astype(bool)
            & (group.candidate_owner.astype(int) > 0)]
        if positive.empty:
            continue
        owner_counts = positive.candidate_owner.astype(int).value_counts()
        stable_owner = int(owner_counts.index[0])
        stable_support = int(owner_counts.iloc[0])
        purity = float(stable_support / len(positive))
        visible = group[group.physically_visible.astype(bool)]
        strong = visible[visible.strong.astype(bool)]
        frames = visible.frame.astype(int).tolist()
        continuous = bool(frames and frames == list(range(frames[0], frames[-1] + 1)))
        maximum_seen_step = 0.0
        visible_rows = list(visible.itertuples(index=False))
        if len(visible_rows) > 1:
            maximum_seen_step = max(
                _scaled_step(left, right)
                for left, right in zip(visible_rows, visible_rows[1:]))
        runs = _positive_runs(group)
        for position in range(1, len(runs) - 1):
            pre, middle, post = runs[position - 1:position + 2]
            if pre["owner"] != post["owner"] or \
                    pre["owner"] == middle["owner"]:
                continue
            owner_a, owner_b = int(pre["owner"]), int(middle["owner"])
            middle_frames = list(map(int, middle["frames"]))
            terminal_return = bool(int(post["end"]) == frame_count - 1)
            minimum_overlap = int(math.ceil(len(middle_frames) / 2.0))
            companion, companion_support, companion_overlap, companion_first = _companion(
                attached, int(track), owner_b, middle_frames,
                minimum_companion, minimum_overlap,
                maximum_companion_distance)
            companion_lead = (int(middle["start"]) - companion_first
                              if companion_first >= 0 else -1)
            middle_group = group[group.frame.astype(int).isin(middle_frames)]
            other_a = attached[
                attached.frame.astype(int).isin(middle_frames)
                & (attached.track_id.astype(int) != int(track))
                & attached.physically_visible.astype(bool)
                & (attached.candidate_owner.astype(int) == owner_a)]
            reasons: list[str] = []
            if len(middle_frames) > maximum_middle:
                reasons.append("middle_too_long")
            if owner_a != stable_owner or stable_support < minimum_stable:
                reasons.append("insufficient_stable_owner_support")
            if purity < minimum_purity:
                reasons.append("insufficient_stable_owner_purity")
            bilateral = (len(pre["frames"]) >= minimum_bookend
                         and len(post["frames"]) >= minimum_bookend)
            boundary = (terminal_return and len(post["frames"]) >= 1
                        and len(pre["frames"]) >= minimum_stable)
            if not (bilateral or boundary):
                reasons.append("insufficient_bilateral_or_terminal_anchor")
            if len(visible) < minimum_visible or not continuous:
                reasons.append("seat_not_long_continuous")
            if len(strong) != len(visible):
                reasons.append("seat_not_continuously_strong")
            if maximum_seen_step > maximum_step:
                reasons.append("seat_not_stationary")
            if (len(middle_group) != len(middle_frames)
                    or not middle_group.physically_visible.astype(bool).all()
                    or not middle_group.strong.astype(bool).all()):
                reasons.append("middle_not_strong_visible")
            encounter_frames = int(
                middle_group.in_encounter.astype(bool).sum())
            if encounter_frames:
                reasons.append("middle_contains_encounter")
            if companion <= 0:
                reasons.append("retained_owner_companion_missing")
            elif not (0 <= companion_lead <= maximum_companion_onset_lead):
                reasons.append("companion_not_event_local_lineage")
            if len(other_a):
                reasons.append("other_stable_owner_core_present")
            public = {
                "proposal_id": "", "track_id": int(track),
                "owner_a": owner_a, "owner_b": owner_b,
                "middle_start": int(middle["start"]),
                "middle_end": int(middle["end"]),
                "middle_frames": len(middle_frames),
                "pre_frames": len(pre["frames"]),
                "post_frames": len(post["frames"]),
                "stable_owner_support": stable_support,
                "positive_owner_observations": int(len(positive)),
                "stable_owner_purity": purity,
                "visible_span_frames": int(len(visible)),
                "strong_span_frames": int(len(strong)),
                "maximum_step_sum_radii": maximum_seen_step,
                "middle_encounter_frames": encounter_frames,
                "companion_track": companion,
                "companion_owner_support": companion_support,
                "companion_middle_overlap": companion_overlap,
                "companion_first_visible_frame": companion_first,
                "companion_onset_lead_frames": companion_lead,
                "other_owner_a_cores": int(len(other_a)),
                "terminal_return": terminal_return,
                "eligible": not reasons,
                "reason": ("eligible_bounded_duplicate_owner_excursion"
                           if not reasons else "|".join(reasons)),
            }
            rows.append(public)
            internals.append({**public, "frame_values": middle_frames})
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"BX{number:04d}"
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internals))


def _prepare_frame(frame: np.ndarray, raw: np.ndarray, attached_index,
                   proposal, frame_index: int, params: dict
                   ) -> tuple[np.ndarray | None, str, int]:
    resident = _point(attached_index, int(proposal.track_id), frame_index)
    if resident is None or not bool(resident.physically_visible):
        return None, "resident_point_missing", 0
    target = _component_at(frame, int(proposal.owner_b), resident)
    if target is None:
        return None, "resident_component_missing", 0
    resident_marker = _marker(target, resident)
    if resident_marker is None:
        return None, "resident_marker_missing", 0
    companion_points = []
    frame_rows = attached_index.reset_index(drop=True)
    frame_rows = frame_rows[
        (frame_rows.frame.astype(int) == frame_index)
        & (frame_rows.track_id.astype(int) != int(proposal.track_id))
        & frame_rows.physically_visible.astype(bool)
        & (frame_rows.candidate_owner.astype(int) == int(proposal.owner_b))]
    for point in frame_rows.itertuples(index=False):
        marker = _marker(target, point)
        if marker is not None and marker != resident_marker:
            companion_points.append((point, marker))
    if not companion_points:
        basin = target
        method = "isolated_component_relabel"
    else:
        markers = np.zeros(frame.shape, np.int32)
        markers[resident_marker] = 1
        marker_id = 2
        used = {resident_marker}
        for _, marker in companion_points:
            if marker in used:
                continue
            markers[marker] = marker_id
            marker_id += 1
            used.add(marker)
        if marker_id == 2:
            return None, "distinct_companion_marker_missing", 0
        elevation = -ndi.gaussian_filter(
            raw.astype(np.float32),
            float(params.get("watershed_sigma_px", 1.0)))
        partition = watershed(elevation, markers=markers, mask=target)
        raw_resident = partition == 1
        raw_host = target & ~raw_resident
        minimum_pixels = int(params.get("minimum_changed_pixels", 5))
        host_markers = [marker for _, marker in companion_points]
        connected = _connected_partition(
            target, raw_resident, raw_host, resident_marker, host_markers,
            minimum_pixels)
        if connected is None:
            yy, xx = np.nonzero(target)
            resident_distance = np.hypot(
                yy - float(resident.y), xx - float(resident.x)) \
                / max(float(resident.radius_px), 1.0)
            host_distance = np.full(len(yy), np.inf, dtype=float)
            for point, _ in companion_points:
                host_distance = np.minimum(
                    host_distance,
                    np.hypot(yy - float(point.y), xx - float(point.x))
                    / max(float(point.radius_px), 1.0))
            geometry = np.zeros_like(target)
            select = resident_distance <= host_distance
            geometry[yy[select], xx[select]] = True
            connected = _connected_partition(
                target, geometry, target & ~geometry, resident_marker,
                host_markers, minimum_pixels)
            if connected is None:
                return None, "connected_partition_unavailable", 0
            method = "radius_normalised_geometry_connected"
        else:
            method = "multisource_raw_watershed_connected"
        basin, host_basin = connected
        if not np.array_equal(basin | host_basin, target):
            return None, "partition_not_exhaustive", 0
    minimum_pixels = int(params.get("minimum_changed_pixels", 5))
    if int(basin.sum()) < minimum_pixels:
        return None, "resident_basin_too_small", 0
    trial = frame.copy()
    # Existing owner-A projection fragments are intentionally preserved.
    trial[basin] = int(proposal.owner_a)
    changed = int(np.count_nonzero(trial != frame))
    if changed < minimum_pixels:
        return None, "insufficient_changed_pixels", 0
    return trial, method, changed


def apply(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
          internal: pd.DataFrame, params: dict
          ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    attached = attach_owners(points, labels).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)
    index = attached.set_index(["track_id", "frame"], drop=False)
    applications: list[dict] = []
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending: list[tuple[int, np.ndarray, str, int]] = []
        failed = ""
        for frame_index in list(proposal.frame_values):
            trial, method, changed = _prepare_frame(
                candidate[frame_index], raw[frame_index], index,
                proposal, int(frame_index), params)
            if trial is None:
                failed = method
                break
            pending.append((int(frame_index), trial, method, changed))
        if not failed:
            trial_stack = candidate.copy()
            for frame_index, trial, _, _ in pending:
                trial_stack[frame_index] = trial
            if owner_consensus.count_new_duplicate_components(
                    labels, trial_stack):
                failed = "new_duplicate_component"
        if failed:
            for frame_index in list(proposal.frame_values):
                applications.append({
                    "proposal_id": proposal.proposal_id,
                    "track_id": int(proposal.track_id),
                    "companion_track": int(proposal.companion_track),
                    "frame": int(frame_index), "owner_a": int(proposal.owner_a),
                    "owner_b": int(proposal.owner_b), "partition_method": "",
                    "changed_pixels": 0, "applied": False,
                    "reason": "atomic_refusal:" + failed,
                })
            continue
        for frame_index, trial, method, changed in pending:
            candidate[frame_index] = trial
            applications.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "companion_track": int(proposal.companion_track),
                "frame": frame_index, "owner_a": int(proposal.owner_a),
                "owner_b": int(proposal.owner_b), "partition_method": method,
                "changed_pixels": changed, "applied": True,
                "reason": "atomic_bounded_excursion_recovery",
            })
    return candidate, pd.DataFrame(applications, columns=APPLICATION_COLUMNS)


def _copy_audits(upstream_dir: Path | None, params: dict, out) -> dict:
    outputs = {}
    for name in ("application_audit.csv", "conflicted_lineage_audit.csv"):
        source = upstream_dir / name if upstream_dir is not None else None
        configured = params.get(name.removesuffix(".csv") + "_path")
        if source is None or not source.is_file():
            source = Path(configured) if configured else None
        target = out.out / name
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text("\n", encoding="utf-8")
        outputs[name.removesuffix(".csv")] = target
    return outputs


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    if len(raw) != len(labels):
        offset = int(params.get("raw_frame_offset", 2))
        raw = raw[offset:offset + len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate, applications = labels.copy(), pd.DataFrame(
            columns=APPLICATION_COLUMNS)
    else:
        candidate, applications = apply(labels, raw, points, internal, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("bounded owner excursion changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("bounded owner excursion changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("bounded owner excursion created duplicate components")

    output_stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = out.out / f"{output_stem}.tif"
    unclaimed_path = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    audit_path = out.out / "bounded_owner_excursion_audit.csv"
    application_path = out.out / "bounded_owner_excursion_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "run_triples_audited": int(len(audit)),
        "eligible_excursions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_excursions": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": 0, "removed_identity_count": 0,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    outputs = {"labels": labels_path, "unclaimed": unclaimed_path,
               "audit": audit_path, "applications": application_path,
               "metrics": metrics_path}
    outputs.update(_copy_audits(upstream_dir, params, out))
    return {"outputs": outputs, "summary": metrics}

