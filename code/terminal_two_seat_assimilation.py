"""Preserve two distinct physical owners through a terminal shared-owner tail.

Discovery is complete-field and target-free. Biological identities, physical
tracks, frames, coordinates, events, regions, and review cases are forbidden
producer inputs.
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

import bounded_owner_excursion as bounded
import component_accounting
import owner_consensus


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region_target", "review_case",
    "case_id", "forced_identity", "forced_track", "forced_frame",
    "include_track", "exclude_track", "include_identity", "exclude_identity",
)
AUDIT_COLUMNS = [
    "proposal_id", "target_track", "companion_track", "continued_owner",
    "host_owner", "anchor_frame", "tail_first", "tail_last", "tail_frames",
    "transient_prefix_frames", "host_tail_frames", "source_history_frames",
    "source_history_observations", "source_history_tracks",
    "prelude_track", "prelude_frames", "prelude_foreign_runs",
    "maximum_history_speed_sum_radii", "maximum_pair_distance_sum_radii",
    "maximum_pair_valley_ratio", "minimum_pair_endpoint_balance",
    "strong_pair_fraction", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "target_track", "companion_track", "frame",
    "continued_owner", "host_owner", "source_owner", "partition_method",
    "changed_pixels", "explained_projection_components", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "terminal two-seat assimilation received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal two-seat assimilation must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def _raw_pair(raw: np.ndarray, left, right, sigma: float) -> dict:
    smooth = ndi.gaussian_filter(raw.astype(np.float32), sigma)
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    samples = max(5, 2 * int(math.ceil(distance)) + 1)
    xs = np.linspace(float(left.x), float(right.x), samples)
    ys = np.linspace(float(left.y), float(right.y), samples)
    values = ndi.map_coordinates(smooth, [ys, xs], order=1, mode="nearest")
    low, high = float(min(values[0], values[-1])), \
        float(max(values[0], values[-1]))
    return {
        "distance_sum_radii": _scaled_distance(left, right),
        "valley_ratio": float(np.min(values[1:-1]) / max(low, 1.0)),
        "endpoint_balance": low / max(high, 1.0),
        "both_strong": _truth(left.strong) and _truth(right.strong),
    }


def _backward_owner_history(attached: pd.DataFrame, owner: int, endpoint,
                            params: dict) -> list:
    """Follow the nearest plausible prior observation of one opaque owner."""
    maximum_gap = int(params.get("maximum_source_history_gap_frames", 3))
    maximum_speed = float(params.get(
        "maximum_source_history_speed_sum_radii", 1.0))
    candidates = attached[
        attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).eq(int(owner))].copy()
    chain = [endpoint]
    current = endpoint
    while True:
        frame = int(current.frame)
        prior = candidates[
            candidates.frame.astype(int).between(frame - maximum_gap,
                                                  frame - 1)]
        options = []
        for row in prior.itertuples(index=False):
            gap = frame - int(row.frame)
            speed = _scaled_distance(row, current) / max(gap, 1)
            if speed <= maximum_speed:
                options.append((-int(row.frame), speed, int(row.track_id), row))
        if not options:
            break
        current = min(options, key=lambda item: item[:3])[3]
        chain.append(current)
    return list(reversed(chain))


def _positive_tail(group: pd.DataFrame, anchor_index: int):
    visible = group[group.physically_visible.astype(bool)].sort_values("frame")
    rows = list(visible.itertuples(index=False))
    return rows[anchor_index + 1:]


def _owner_runs(rows: list) -> list[dict]:
    runs: list[dict] = []
    for row in rows:
        owner, frame = int(row.candidate_owner), int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and runs[-1]["last"] + 1 == frame):
            runs[-1]["last"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "rows": [row]})
    return runs


def _recurrent_preludes(groups: dict[int, pd.DataFrame], history: list,
                        owner: int, labels: np.ndarray,
                        attached: pd.DataFrame, params: dict) -> list[dict]:
    """Find recurrent ownership with evidence-backed shared-mask intervals."""
    minimum_runs = int(params.get("minimum_prelude_owner_runs", 3))
    minimum_foreign = int(params.get("minimum_prelude_foreign_runs", 2))
    maximum_step = float(params.get("maximum_prelude_step_sum_radii", 1.0))
    results = []
    for track in sorted({int(row.track_id) for row in history}):
        group = groups[track].sort_values("frame")
        visible = list(group[group.physically_visible.astype(bool)]
                       .itertuples(index=False))
        owner_positions = [index for index, row in enumerate(visible)
                           if int(row.candidate_owner) == int(owner)]
        if len(owner_positions) < minimum_runs:
            continue
        first, last = owner_positions[0], owner_positions[-1]
        span = visible[first:last + 1]
        if [int(row.frame) for row in span] != list(range(
                int(span[0].frame), int(span[-1].frame) + 1)):
            continue
        if any(int(row.candidate_owner) <= 0 for row in span):
            continue
        runs = _owner_runs(span)
        owner_runs = [run for run in runs if run["owner"] == int(owner)]
        foreign_runs = [run for run in runs if run["owner"] != int(owner)]
        steps = [_scaled_distance(left, right)
                 for left, right in zip(span[:-1], span[1:])]
        if (len(owner_runs) < minimum_runs
                or len(foreign_runs) < minimum_foreign
                or max(steps, default=0.0) > maximum_step):
            continue
        shared_frames = []
        for run in foreign_runs:
            for row in run["rows"]:
                frame_index = int(row.frame)
                component = bounded._component_at(
                    labels[frame_index], int(row.candidate_owner), row)
                if component is None:
                    continue
                frame_points = attached[
                    attached.frame.astype(int).eq(frame_index)
                    & attached.physically_visible.astype(bool)]
                target_marker = bounded._marker(component, row)
                other_markers = {
                    bounded._marker(component, other)
                    for other in frame_points.itertuples(index=False)
                    if int(other.track_id) != int(row.track_id)
                }
                other_markers.discard(None)
                other_markers.discard(target_marker)
                if other_markers:
                    shared_frames.append(frame_index)
        if not shared_frames:
            continue
        results.append({
            "track": track,
            "frames": shared_frames,
            "foreign_runs": len(foreign_runs),
            "maximum_step": max(steps, default=0.0),
        })
    return results


def _shared_component(frame: np.ndarray, owner: int, target, companion):
    left = bounded._component_at(frame, int(owner), target)
    right = bounded._component_at(frame, int(owner), companion)
    if left is None or right is None or not np.any(left & right):
        return None
    left_marker = bounded._marker(left, target)
    right_marker = bounded._marker(left, companion)
    if left_marker is None or right_marker is None or left_marker == right_marker:
        return None
    return left


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit every terminal physical lineage for a two-seat assimilation."""
    assert_target_free(params)
    attached = owner_consensus.attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    movie = int(len(labels))
    terminal_slack = max(1, int(math.ceil(movie * float(
        params.get("maximum_terminal_slack_movie_fraction", 0.02)))))
    maximum_tail = max(1, int(math.ceil(movie * float(
        params.get("maximum_shared_tail_movie_fraction", 0.06)))))
    maximum_prefix = max(0, int(math.ceil(movie * float(
        params.get("maximum_transient_prefix_movie_fraction", 0.02)))))
    minimum_host = max(2, int(math.ceil(movie * float(
        params.get("minimum_host_tail_movie_fraction", 0.03)))))
    minimum_history = int(params.get("minimum_source_history_observations", 4))
    minimum_history_tracks = int(params.get("minimum_source_history_tracks", 2))
    minimum_history_span = max(1, int(math.ceil(movie * float(
        params.get("minimum_source_history_movie_fraction", 0.06)))))
    maximum_pair_distance = float(params.get(
        "maximum_pair_distance_sum_radii", 2.50))
    maximum_valley = float(params.get("maximum_pair_valley_ratio", 0.75))
    minimum_balance = float(params.get("minimum_pair_endpoint_balance", 0.14))
    minimum_strong_fraction = float(params.get("minimum_strong_pair_fraction", 0.75))
    sigma = float(params.get("pair_profile_sigma_px", 1.0))
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    rows: list[dict] = []
    internals: list[dict] = []

    for track, group in groups.items():
        visible = group[group.physically_visible.astype(bool)].sort_values("frame")
        visible_rows = list(visible.itertuples(index=False))
        if len(visible_rows) < 2 or int(visible_rows[-1].frame) < \
                movie - 1 - terminal_slack:
            continue
        for anchor_index, anchor in enumerate(visible_rows[:-1]):
            continued = int(anchor.candidate_owner)
            tail = visible_rows[anchor_index + 1:]
            if continued <= 0 or not tail or len(tail) > maximum_tail:
                continue
            expected = list(range(int(anchor.frame) + 1,
                                  int(visible_rows[-1].frame) + 1))
            if [int(row.frame) for row in tail] != expected:
                continue
            tail_owners = [int(row.candidate_owner) for row in tail]
            if any(owner <= 0 or owner == continued for owner in tail_owners):
                continue
            host = int(tail_owners[-1])
            host_tail = 0
            for owner in reversed(tail_owners):
                if owner != host:
                    break
                host_tail += 1
            prefix = len(tail_owners) - host_tail
            if host == continued or host_tail < minimum_host or \
                    prefix > maximum_prefix:
                continue
            history = _backward_owner_history(
                attached, continued, anchor, params)
            history_span = int(anchor.frame) - int(history[0].frame)
            history_tracks = len({int(row.track_id) for row in history})
            history_speeds = [
                _scaled_distance(left, right)
                / max(int(right.frame) - int(left.frame), 1)
                for left, right in zip(history[:-1], history[1:])]
            reasons: list[str] = []
            if len(history) < minimum_history:
                reasons.append("source_history_too_short")
            if history_span < minimum_history_span:
                reasons.append("source_history_span_too_short")
            if history_tracks < minimum_history_tracks:
                reasons.append("source_history_not_fragmented")
            if any(np.any(labels[int(row.frame) + 1:] == continued)
                   for row in [anchor]):
                reasons.append("continued_owner_present_after_anchor")
            preludes = _recurrent_preludes(
                groups, history, continued, labels, attached, params)
            if len(preludes) != 1:
                reasons.append("recurrent_source_prelude_not_unique")
            prelude = preludes[0] if len(preludes) == 1 else None

            companions = []
            anchor_frame = int(anchor.frame)
            for companion_track, companion_group in groups.items():
                if companion_track == track:
                    continue
                companion_anchor = _point(companion_group, anchor_frame)
                if (companion_anchor is None
                        or not _truth(companion_anchor.physically_visible)
                        or int(companion_anchor.candidate_owner) != host):
                    continue
                pair_rows = []
                shared = True
                for target_row in tail:
                    frame = int(target_row.frame)
                    companion_row = _point(companion_group, frame)
                    if (companion_row is None
                            or not _truth(companion_row.physically_visible)
                            or int(companion_row.candidate_owner)
                            != int(target_row.candidate_owner)
                            or _shared_component(
                                labels[frame], int(target_row.candidate_owner),
                                target_row, companion_row) is None):
                        shared = False
                        break
                    pair_rows.append(_raw_pair(
                        raw[frame], target_row, companion_row, sigma))
                if not shared or len(pair_rows) != len(tail):
                    continue
                qualified = [item for item in pair_rows
                             if item["distance_sum_radii"] <= maximum_pair_distance
                             and item["valley_ratio"] <= maximum_valley
                             and item["endpoint_balance"] >= minimum_balance]
                strong_fraction = float(np.mean(
                    [item["both_strong"] for item in pair_rows]))
                if len(qualified) != len(pair_rows) or \
                        strong_fraction < minimum_strong_fraction:
                    continue
                companions.append({
                    "track": companion_track, "pairs": pair_rows,
                    "strong_fraction": strong_fraction})
            if len(companions) != 1:
                reasons.append("host_companion_not_unique")
            companion = companions[0] if len(companions) == 1 else None
            pairs = companion["pairs"] if companion else []
            public = {
                "proposal_id": "", "target_track": track,
                "companion_track": int(companion["track"]) if companion else 0,
                "continued_owner": continued, "host_owner": host,
                "anchor_frame": int(anchor.frame),
                "tail_first": int(tail[0].frame),
                "tail_last": int(tail[-1].frame), "tail_frames": len(tail),
                "transient_prefix_frames": prefix,
                "host_tail_frames": host_tail,
                "source_history_frames": history_span + 1,
                "source_history_observations": len(history),
                "source_history_tracks": history_tracks,
                "prelude_track": int(prelude["track"]) if prelude else 0,
                "prelude_frames": len(prelude["frames"]) if prelude else 0,
                "prelude_foreign_runs": int(prelude["foreign_runs"])
                    if prelude else 0,
                "maximum_history_speed_sum_radii": max(
                    history_speeds, default=0.0),
                "maximum_pair_distance_sum_radii": max(
                    (item["distance_sum_radii"] for item in pairs),
                    default=float("nan")),
                "maximum_pair_valley_ratio": max(
                    (item["valley_ratio"] for item in pairs),
                    default=float("nan")),
                "minimum_pair_endpoint_balance": min(
                    (item["endpoint_balance"] for item in pairs),
                    default=float("nan")),
                "strong_pair_fraction": float(
                    companion["strong_fraction"]) if companion else 0.0,
                "eligible": not reasons,
                "reason": ("eligible_terminal_two_seat_assimilation"
                           if not reasons else "|".join(reasons)),
            }
            rows.append(public)
            internals.append({
                **public,
                "tail_frame_values": [int(row.frame) for row in tail],
                "tail_owner_values": tail_owners,
                "prelude_frame_values": list(prelude["frames"])
                    if prelude else [],
            })
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"TSA{number:04d}"
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internals), attached)


def _partition_frame(frame: np.ndarray, raw: np.ndarray, target, companion,
                     continued_owner: int, host_owner: int, params: dict):
    source_owner = int(target.candidate_owner)
    component = _shared_component(frame, source_owner, target, companion)
    if component is None:
        return None, "shared_component_missing", 0
    target_marker = bounded._marker(component, target)
    host_marker = bounded._marker(component, companion)
    if target_marker is None or host_marker is None or target_marker == host_marker:
        return None, "distinct_markers_missing", 0
    markers = np.zeros(component.shape, np.int16)
    markers[target_marker] = 1
    markers[host_marker] = 2
    elevation = -ndi.gaussian_filter(
        raw.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    basins = watershed(elevation, markers=markers, mask=component,
                       connectivity=STRUCTURE)
    minimum_pixels = int(params.get("minimum_partition_pixels", 5))
    connected = bounded._connected_partition(
        component, basins == 1, basins == 2, target_marker, [host_marker],
        minimum_pixels)
    method = "two_core_raw_watershed_connected"
    if connected is None:
        yy, xx = np.nonzero(component)
        target_distance = np.hypot(
            yy - float(target.y), xx - float(target.x)) \
            / max(float(target.radius_px), 1.0)
        host_distance = np.hypot(
            yy - float(companion.y), xx - float(companion.x)) \
            / max(float(companion.radius_px), 1.0)
        target_part = np.zeros_like(component)
        select = target_distance <= host_distance
        target_part[yy[select], xx[select]] = True
        connected = bounded._connected_partition(
            component, target_part, component & ~target_part,
            target_marker, [host_marker], minimum_pixels)
        method = "radius_normalised_geometry_connected"
    if connected is None:
        return None, "connected_two_seat_partition_unavailable", 0
    target_part, host_part = connected
    trial = frame.copy()
    trial[target_part] = int(continued_owner)
    trial[host_part] = int(host_owner)
    target_disk = owner_consensus._disk_owner(
        trial, float(target.x), float(target.y), float(target.radius_px))[0]
    host_disk = owner_consensus._disk_owner(
        trial, float(companion.x), float(companion.y),
        float(companion.radius_px))[0]
    if target_disk != continued_owner or host_disk != host_owner:
        return None, "partition_owner_dominance_failed", 0
    return trial, method, int(np.count_nonzero(trial != frame))


def _partition_prelude_frame(frame: np.ndarray, raw: np.ndarray,
                             attached_frame: pd.DataFrame, target,
                             continued_owner: int, params: dict):
    """Restore one recurrent owner while preserving every other raw core."""
    source_owner = int(target.candidate_owner)
    component = bounded._component_at(frame, source_owner, target)
    if component is None:
        return None, "prelude_component_missing", 0
    target_marker = bounded._marker(component, target)
    if target_marker is None:
        return None, "prelude_target_marker_missing", 0
    companion_markers = []
    for row in attached_frame[
            attached_frame.physically_visible.astype(bool)].itertuples(index=False):
        if int(row.track_id) == int(target.track_id):
            continue
        marker = bounded._marker(component, row)
        if marker is not None and marker != target_marker:
            companion_markers.append(marker)
    if not companion_markers:
        return None, "prelude_shared_mask_evidence_missing", 0
    before = frame.copy()
    markers = np.zeros(component.shape, np.int16)
    markers[target_marker] = 1
    for value, marker in enumerate(sorted(set(companion_markers)), start=2):
        markers[marker] = value
    elevation = -ndi.gaussian_filter(
        raw.astype(np.float32),
        float(params.get("watershed_sigma_px", 1.0)))
    partition = watershed(elevation, markers=markers, mask=component,
                          connectivity=STRUCTURE)
    target_part = partition == 1
    host_part = component & ~target_part
    connected = bounded._connected_partition(
        component, target_part, host_part, target_marker,
        sorted(set(companion_markers)),
        int(params.get("minimum_partition_pixels", 5)))
    if connected is None:
        return None, "prelude_connected_partition_unavailable", 0
    target_part, _ = connected
    trial = frame.copy()
    trial[target_part] = int(continued_owner)
    method = "multicore_recurrent_raw_watershed_connected"
    owner = owner_consensus._disk_owner(
        trial, float(target.x), float(target.y), float(target.radius_px))[0]
    changed = int(np.count_nonzero(trial != before))
    if owner != int(continued_owner) or changed <= 0:
        return None, "prelude_owner_dominance_failed", 0
    return trial, method, changed


def apply(labels: np.ndarray, raw: np.ndarray, attached: pd.DataFrame,
          internal: pd.DataFrame, params: dict):
    candidate = labels.copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    applications: list[dict] = []
    explained = applied = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending = []
        failure = ""
        for frame_index in list(proposal.prelude_frame_values):
            target = _point(groups[int(proposal.prelude_track)], frame_index)
            if target is None:
                failure = "prelude_physical_point_missing"
                break
            attached_frame = attached[
                attached.frame.astype(int).eq(int(frame_index))]
            trial, method, changed = _partition_prelude_frame(
                candidate[frame_index], raw[frame_index], attached_frame,
                target, int(proposal.continued_owner), params)
            if trial is None:
                failure = method
                break
            continued_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.continued_owner))
                - component_accounting.component_excess(
                    candidate[frame_index], int(proposal.continued_owner)))
            if continued_delta > 0:
                failure = "prelude_continued_owner_duplicate_created"
                break
            pending.append((frame_index, trial, method, changed, 0,
                            int(target.candidate_owner)))
        for frame_index in list(proposal.tail_frame_values):
            if failure:
                break
            target = _point(groups[int(proposal.target_track)], frame_index)
            companion = _point(
                groups[int(proposal.companion_track)], frame_index)
            if target is None or companion is None:
                failure = "physical_point_missing"
                break
            trial, method, changed = _partition_frame(
                candidate[frame_index], raw[frame_index], target, companion,
                int(proposal.continued_owner), int(proposal.host_owner), params)
            if trial is None:
                failure = method
                break
            before = candidate[frame_index]
            continued_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.continued_owner))
                - component_accounting.component_excess(
                    before, int(proposal.continued_owner)))
            host_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.host_owner))
                - component_accounting.component_excess(
                    before, int(proposal.host_owner)))
            if continued_delta > 0:
                failure = "continued_owner_duplicate_created"
                break
            if host_delta > 1:
                failure = "multiple_host_projection_components_created"
                break
            pending.append((frame_index, trial, method, changed,
                            max(0, int(host_delta)), int(target.candidate_owner)))
        if failure:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "target_track": int(proposal.target_track),
                "companion_track": int(proposal.companion_track),
                "frame": int(proposal.tail_first),
                "continued_owner": int(proposal.continued_owner),
                "host_owner": int(proposal.host_owner), "source_owner": 0,
                "partition_method": "atomic", "changed_pixels": 0,
                "explained_projection_components": 0, "applied": False,
                "reason": "atomic_refusal:" + failure})
            continue
        for frame_index, trial, method, changed, projection, source_owner in pending:
            candidate[frame_index] = trial
            explained += projection
            applications.append({
                "proposal_id": proposal.proposal_id,
                "target_track": int(proposal.target_track),
                "companion_track": int(proposal.companion_track),
                "frame": frame_index,
                "continued_owner": int(proposal.continued_owner),
                "host_owner": int(proposal.host_owner),
                "source_owner": source_owner, "partition_method": method,
                "changed_pixels": changed,
                "explained_projection_components": projection,
                "applied": True,
                "reason": "atomic_terminal_two_seat_assimilation_recovery"})
        applied += 1
    changed = candidate != labels
    details = {
        "proposals_audited": int(len(internal)),
        "eligible_proposals": int(internal.eligible.astype(bool).sum())
            if len(internal) else 0,
        "applied_proposals": int(applied),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_duplicate_components": 0,
        "new_explained_projection_components": int(explained),
    }
    return candidate, pd.DataFrame(applications, columns=APPLICATION_COLUMNS), details


def _save(source: Path, target: Path, before: np.ndarray,
          after: np.ndarray) -> None:
    if np.array_equal(before, after):
        shutil.copyfile(source, target)
    else:
        tifffile.imwrite(target, after, imagej=True, compression="zlib",
                         photometric="minisblack", metadata={
                             "axes": "TYX", "finterval": 1800.0,
                             "tunit": "sec", "unit": "pixel"})


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal, attached = discover(labels, raw, points, params)
    candidate, applications, details = apply(
        labels, raw, attached, internal, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("terminal two-seat assimilation changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("terminal two-seat assimilation changed identity set")

    stem = str(params.get("output_stem", labels_path.stem))
    labels_out = out.out / f"{stem}.tif"
    unclaimed_out = out.out / f"{stem}_unclaimed_original_ids.tif"
    _save(labels_path, labels_out, labels, candidate)
    shutil.copyfile(unclaimed_path, unclaimed_out)
    audit_path = out.out / "terminal_two_seat_assimilation_audit.csv"
    application_path = out.out / "terminal_two_seat_assimilation_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    outputs = {"labels": labels_out, "unclaimed": unclaimed_out,
               "audit": audit_path, "applications": application_path}
    for name in ("application_audit.csv", "conflicted_lineage_audit.csv"):
        source = upstream_dir / name if upstream_dir is not None else None
        configured = params.get(name.removesuffix(".csv") + "_path")
        if source is None or not source.is_file():
            source = Path(configured) if configured else None
        target = out.out / name
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text(
                "proposal_id,physical_track,assigned_identity,outcome,reason,"
                "changed_pixels,changed_frames\n", encoding="utf-8")
        outputs[name.removesuffix(".csv")] = target
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        **details, "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    outputs["metrics"] = metrics_path
    return {"outputs": outputs, "summary": summary}
