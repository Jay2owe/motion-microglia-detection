"""Partition a separable two-seat identity merge at a recording boundary.

Discovery is complete-field and target-free. The rule uses recording-relative
time windows, physical encounter evidence, forward owner inheritance and a
separate durable owner anchor. Measured identifiers appear only in the audit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import component_accounting
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "identity_target", "owner_target", "track_target", "frame_target",
    "event_target", "coordinate", "region_target", "review_case",
    "case_id", "target_identity", "target_owner", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
    "include_track", "exclude_track", "include_identity", "exclude_identity",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "recording-start two-seat inheritance received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "recording-start two-seat inheritance must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, _ = ndi.label(frame == int(owner), STRUCTURE)
    marker = physical._marker_pixel(
        frame == int(owner), float(point.x), float(point.y))
    if marker is None:
        return None
    component = int(components[marker])
    return components == component if component else None


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def _owner_run(group: pd.DataFrame, first: int, owner: int) -> int:
    lookup = {int(row.frame): int(row.accepted_owner)
              for row in group.itertuples(index=False)}
    length = 0
    while lookup.get(int(first) + length, 0) == int(owner):
        length += 1
    return length


def _first_distinct_run(group: pd.DataFrame, first: int, shared_owner: int,
                        stop: int, minimum: int) -> tuple[int, int, int]:
    lookup = {int(row.frame): int(row.accepted_owner)
              for row in group.itertuples(index=False)}
    for start in range(int(first), int(stop) + 1):
        owner = lookup.get(start, 0)
        if owner <= 0 or owner == int(shared_owner):
            continue
        length = 0
        while lookup.get(start + length, 0) == owner:
            length += 1
        if length >= int(minimum):
            return int(owner), int(start), int(length)
    return 0, 0, 0


def _component_tracks(mask: np.ndarray, frame_points: pd.DataFrame) -> set[int]:
    tracks: set[int] = set()
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    for point in frame_points.itertuples(index=False):
        if str(point.state) not in VISIBLE_STATES:
            continue
        radius = max(2.0, 0.5 * float(point.radius_px))
        disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
                <= radius ** 2)
        if np.any(mask & disk):
            tracks.add(int(point.track_id))
    return tracks


def _find_anchor(attached: pd.DataFrame, point, owner: int, frame: int,
                 excluded: set[int], support_stop: int,
                 minimum_support: int, maximum_distance: float):
    candidates = attached[
        attached.frame.astype(int).eq(int(frame))
        & attached.accepted_owner.astype(int).eq(int(owner))
        & attached.state.astype(str).isin(VISIBLE_STATES)
        & ~attached.track_id.astype(int).isin(excluded)]
    ranked = []
    for anchor in candidates.itertuples(index=False):
        history = attached[
            attached.track_id.astype(int).eq(int(anchor.track_id))
            & attached.frame.astype(int).between(int(frame), int(support_stop))
            & attached.accepted_owner.astype(int).eq(int(owner))]
        if len(history) < int(minimum_support):
            continue
        distance = _scaled_distance(point, anchor)
        if distance <= float(maximum_distance):
            ranked.append((distance, -len(history), int(anchor.track_id)))
    if not ranked:
        return None
    distance, negative_support, track = min(ranked)
    return track, float(distance), int(-negative_support)


def _validate_inputs(labels: np.ndarray, raw: np.ndarray,
                     points: pd.DataFrame, encounters: pd.DataFrame) -> None:
    if labels.shape != raw.shape:
        raise ValueError("labels and raw stacks must have equal shape")
    point_fields = {
        "track_id", "frame", "x", "y", "radius_px", "state",
    }
    missing_points = sorted(point_fields - set(points.columns))
    if missing_points:
        raise ValueError(
            "recording-start two-seat inheritance requires point fields: "
            + ", ".join(missing_points))
    encounter_fields = {
        "encounter_id", "frame", "separable", "track_a", "track_b",
        "valley_ratio",
    }
    missing_encounters = sorted(encounter_fields - set(encounters.columns))
    if missing_encounters:
        raise ValueError(
            "recording-start two-seat inheritance requires encounter fields: "
            + ", ".join(missing_encounters))


def discover(labels: np.ndarray, points: pd.DataFrame,
             encounters: pd.DataFrame, params: dict):
    """Audit all recording-boundary separable same-owner encounters."""
    assert_target_free(params)
    attached = physical.attach_owners(points, labels)
    index = attached.set_index(["track_id", "frame"], drop=False)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    frame_points = {int(frame): group
                    for frame, group in attached.groupby("frame", sort=True)}
    movie = int(len(labels))
    boundary_stop = int(math.floor((movie - 1) * float(
        params.get("maximum_boundary_movie_fraction", 0.01))))
    lookahead = max(3, int(math.ceil(movie * float(
        params.get("forward_lookahead_movie_fraction", 0.06)))))
    immediate = max(1, int(math.ceil(movie * float(
        params.get("maximum_distinct_owner_delay_movie_fraction", 0.02)))))
    minimum_host = max(2, int(math.ceil(movie * float(
        params.get("minimum_host_run_movie_fraction", 0.04)))))
    minimum_distinct = max(2, int(math.ceil(movie * float(
        params.get("minimum_distinct_run_movie_fraction", 0.02)))))
    minimum_anchor = max(2, int(math.ceil(movie * float(
        params.get("minimum_anchor_support_movie_fraction", 0.08)))))
    maximum_valley = float(params.get("maximum_valley_ratio", 0.50))
    maximum_pair_distance = float(params.get(
        "maximum_pair_distance_sum_radii", 1.75))
    maximum_anchor_distance = float(params.get(
        "maximum_anchor_distance_sum_radii", 3.0))
    maximum_reference_span = max(2, int(math.ceil(movie * float(
        params.get("maximum_resumed_reference_movie_fraction", 0.12)))))

    boundary = encounters[
        encounters.frame.astype(int).le(boundary_stop)
        & encounters.separable.map(_truth)].copy()
    audit_rows: list[dict] = []
    internal: list[dict] = []
    for encounter in boundary.sort_values(
            ["frame", "encounter_id", "track_a", "track_b"]).itertuples(
                index=False):
        frame = int(encounter.frame)
        left = _point(index, int(encounter.track_a), frame)
        right = _point(index, int(encounter.track_b), frame)
        if left is None or right is None:
            continue
        left_owner = int(left.accepted_owner)
        right_owner = int(right.accepted_owner)
        if left_owner <= 0 or left_owner != right_owner:
            continue
        shared_owner = left_owner
        pair = {int(encounter.track_a), int(encounter.track_b)}
        reasons: list[str] = []
        if str(left.state) not in VISIBLE_STATES or \
                str(right.state) not in VISIBLE_STATES:
            reasons.append("boundary_core_not_visible")
        scaled_pair = _scaled_distance(left, right)
        if scaled_pair > maximum_pair_distance:
            reasons.append("physical_cores_too_distant")
        if float(encounter.valley_ratio) > maximum_valley:
            reasons.append("raw_valley_not_deep_enough")
        left_component = _component_at(labels[frame], shared_owner, left)
        right_component = _component_at(labels[frame], shared_owner, right)
        if (left_component is None or right_component is None
                or not np.array_equal(left_component, right_component)):
            reasons.append("cores_do_not_share_one_owner_component")
            component = None
        else:
            component = left_component
            occupants = _component_tracks(
                component, frame_points.get(frame, attached.iloc[0:0]))
            if occupants != pair:
                reasons.append("shared_component_not_exactly_two_cores")

        alternatives = []
        for host, resumed in ((left, right), (right, left)):
            host_run = _owner_run(
                groups[int(host.track_id)], frame + 1, shared_owner)
            resumed_owner, resumed_first, resumed_run = _first_distinct_run(
                groups[int(resumed.track_id)], frame + 1, shared_owner,
                min(movie - 1, frame + lookahead), minimum_distinct)
            anchor = None
            if resumed_owner > 0 and resumed_first <= frame + immediate:
                anchor = _find_anchor(
                    attached, resumed, resumed_owner, frame, pair,
                    min(movie - 1, frame + lookahead + minimum_anchor),
                    minimum_anchor, maximum_anchor_distance)
            if (host_run >= minimum_host and resumed_owner > 0
                    and resumed_first <= frame + immediate
                    and resumed_run >= minimum_distinct and anchor is not None):
                alternatives.append({
                    "host_track": int(host.track_id),
                    "resumed_track": int(resumed.track_id),
                    "resumed_owner": int(resumed_owner),
                    "resumed_first": int(resumed_first),
                    "host_run": int(host_run),
                    "resumed_run": int(resumed_run),
                    "anchor_track": int(anchor[0]),
                    "anchor_distance": float(anchor[1]),
                    "anchor_support": int(anchor[2]),
                })
        if len(alternatives) != 1:
            reasons.append("forward_inheritance_not_unique")
            chosen = {
                "host_track": 0, "resumed_track": 0, "resumed_owner": 0,
                "resumed_first": 0, "host_run": 0, "resumed_run": 0,
                "anchor_track": 0, "anchor_distance": float("nan"),
                "anchor_support": 0,
            }
        else:
            chosen = alternatives[0]
            resumed_group = groups[int(chosen["resumed_track"])]
            resumed_span = int(resumed_group.frame.max()
                               - resumed_group.frame.min() + 1)
            chosen["resumed_reference_span"] = resumed_span
            if resumed_span > maximum_reference_span:
                reasons.append("resumed_reference_not_short")
        chosen.setdefault("resumed_reference_span", 0)
        eligible = not reasons
        public = {
            "proposal_id": "", "encounter_id": int(encounter.encounter_id),
            "frame": frame, "track_a": int(encounter.track_a),
            "track_b": int(encounter.track_b),
            "shared_owner": shared_owner, **chosen,
            "pair_distance_sum_radii": scaled_pair,
            "valley_ratio": float(encounter.valley_ratio),
            "eligible": eligible,
            "reason": ("eligible_recording_start_two_seat_inheritance"
                       if eligible else "|".join(reasons)),
        }
        audit_rows.append(public)
        internal.append({**public, "component": component})
    number = 0
    for public, private in zip(audit_rows, internal):
        if public["eligible"]:
            number += 1
            proposal_id = f"RST{number:04d}"
            public["proposal_id"] = private["proposal_id"] = proposal_id
    return pd.DataFrame(audit_rows), internal, attached


def _partition(component: np.ndarray, raw: np.ndarray, host, resumed,
               params: dict):
    host_marker = physical._marker_pixel(
        component, float(host.x), float(host.y))
    resumed_marker = physical._marker_pixel(
        component, float(resumed.x), float(resumed.y))
    if (host_marker is None or resumed_marker is None
            or host_marker == resumed_marker):
        return None, "watershed_markers_unavailable"
    markers = np.zeros(component.shape, np.int16)
    markers[host_marker] = 1
    markers[resumed_marker] = 2
    smooth = ndi.gaussian_filter(
        raw.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    basins = watershed(-smooth, markers=markers, mask=component,
                       connectivity=STRUCTURE.astype(bool))
    host_basin, resumed_basin = basins == 1, basins == 2
    minimum = int(params.get("minimum_partition_pixels", 5))
    minimum_ratio = float(params.get("minimum_basin_area_radius_ratio", 0.20))
    maximum_ratio = float(params.get("maximum_basin_area_radius_ratio", 4.0))
    for basin, point in ((host_basin, host), (resumed_basin, resumed)):
        expected = max(math.pi * float(point.radius_px) ** 2, 1.0)
        ratio = float(basin.sum()) / expected
        if (int(basin.sum()) < minimum
                or not minimum_ratio <= ratio <= maximum_ratio
                or int(ndi.label(basin, STRUCTURE)[1]) != 1):
            return None, "raw_watershed_basin_invalid"
    if not np.array_equal(host_basin | resumed_basin, component):
        return None, "raw_watershed_not_conservative"
    return resumed_basin, "two_core_raw_watershed_connected"


def apply(labels: np.ndarray, raw: np.ndarray, attached: pd.DataFrame,
          proposals: list[dict], params: dict):
    candidate = labels.copy()
    index = attached.set_index(["track_id", "frame"], drop=False)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    frame_points = {int(frame): group
                    for frame, group in attached.groupby("frame", sort=True)}
    applications: list[dict] = []
    explained = applied = 0
    occupied = np.zeros_like(labels, bool)
    minimum_changed_frames = max(1, int(math.ceil(len(labels) * float(
        params.get("minimum_corrected_movie_fraction", 0.03)))))
    for proposal in proposals:
        if not proposal["eligible"]:
            continue
        resumed_track = int(proposal["resumed_track"])
        resumed_owner = int(proposal["resumed_owner"])
        pending: list[dict] = []
        failure = ""
        for resumed in groups[resumed_track].itertuples(index=False):
            frame = int(resumed.frame)
            current_owner = int(resumed.accepted_owner)
            if current_owner <= 0 or current_owner == resumed_owner:
                continue
            component = _component_at(candidate[frame], current_owner, resumed)
            if component is None:
                continue
            occupants = _component_tracks(
                component, frame_points.get(frame, attached.iloc[0:0]))
            if resumed_track not in occupants:
                continue
            companions = sorted(occupants - {resumed_track})
            if not companions:
                change = component
                method = "isolated_component_relabel"
            elif len(companions) == 1:
                host = _point(index, companions[0], frame)
                change, method = _partition(
                    component, raw[frame], host, resumed, params)
                if change is None:
                    continue
            else:
                continue
            if np.any(occupied[frame] & change):
                failure = "overlapping_atomic_proposal"
                break
            trial = candidate[frame].copy()
            trial[change] = resumed_owner
            if not np.array_equal(trial > 0, candidate[frame] > 0):
                failure = "foreground_changed"
                break
            host_delta = (component_accounting.component_excess(
                trial, current_owner)
                - component_accounting.component_excess(
                    candidate[frame], current_owner))
            resumed_delta = (component_accounting.component_excess(
                trial, resumed_owner)
                - component_accounting.component_excess(
                    candidate[frame], resumed_owner))
            if host_delta > 0:
                failure = "source_owner_fragmented"
                break
            if resumed_delta > 1:
                failure = "more_than_one_projection_component_created"
                break
            pending.append({
                "frame": frame, "trial": trial, "change": change,
                "method": method, "current_owner": current_owner,
                "explained": max(0, int(resumed_delta)),
            })
        if not failure and len(pending) < minimum_changed_frames:
            failure = "insufficient_unambiguous_reference_corrections"
        if failure:
            applications.append({
                "proposal_id": proposal["proposal_id"],
                "frame": int(proposal["frame"]),
                "host_track": int(proposal["host_track"]),
                "resumed_track": resumed_track,
                "shared_owner": int(proposal["shared_owner"]),
                "resumed_owner": resumed_owner,
                "partition_method": "atomic", "changed_pixels": 0,
                "explained_projection_components": 0,
                "applied": False, "reason": failure})
            continue
        for item in pending:
            frame = int(item["frame"])
            candidate[frame] = item["trial"]
            occupied[frame] |= item["change"]
            explained += int(item["explained"])
            applications.append({
                "proposal_id": proposal["proposal_id"], "frame": frame,
                "host_track": int(proposal["host_track"]),
                "resumed_track": resumed_track,
                "shared_owner": int(item["current_owner"]),
                "resumed_owner": resumed_owner,
                "partition_method": item["method"],
                "changed_pixels": int(item["change"].sum()),
                "explained_projection_components": int(item["explained"]),
                "applied": True,
                "reason": "recording_start_projection_owner_inheritance"})
        applied += 1
    changed = candidate != labels
    details = {
        "applied_proposals": int(applied),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "explained_projection_components_added": int(explained),
        "unexplained_duplicate_components_added": 0,
    }
    return candidate, pd.DataFrame(applications), details


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, encounters: pd.DataFrame, params: dict):
    """Discover and atomically apply all eligible field-wide proposals."""
    assert_target_free(params)
    _validate_inputs(labels, raw, points, encounters)
    if labels.shape != unclaimed.shape:
        raise ValueError("labels and unclaimed stacks must have equal shape")
    audit, internal, attached = discover(labels, points, encounters, params)
    candidate, applications, details = apply(
        labels, raw, attached, internal, params)
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "boundary_encounters_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.map(_truth).sum())
            if len(audit) else 0,
        **details,
        "foreground_ledger_exact": bool(np.array_equal(
            candidate > 0, labels > 0)),
        "preexisting_unclaimed_exact": True,
        "old_identity_set_preserved": before_ids == after_ids,
        "new_identity_count": len(after_ids - before_ids),
        "removed_identity_count": len(before_ids - after_ids),
    }
    return candidate, unclaimed.copy(), audit, applications, summary
