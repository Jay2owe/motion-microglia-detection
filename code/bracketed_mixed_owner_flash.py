"""Repair short mixed-owner flashes bracketed by one physical owner.

Discovery is complete-field and target-free. A proposal needs a continuous
physical seat, the same owner on both sides of a short foreign-owner interval,
and at least one interval frame where the foreign mask demonstrably covers two
independent raw cores. The full interval is changed atomically. A proposal is
refused when restoring the bracket owner would create another disconnected
component, because that would be an unexplained duplicate identity.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

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
    "proposal_id", "track_id", "bracket_owner", "middle_owners",
    "middle_first", "middle_last", "middle_frames", "pre_frames",
    "post_frames", "bracket_support", "positive_observations",
    "bracket_purity", "distinct_middle_owners", "episode_strong_fraction",
    "maximum_episode_step_sum_radii", "shared_mask_frames",
    "shared_mask_fraction", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "track_id", "frame", "bracket_owner", "source_owner",
    "source_companion_cores", "partition_method", "changed_pixels",
    "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "bracketed mixed-owner flash received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("bracketed mixed-owner flash must be field-wide")


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _owner_runs(group: pd.DataFrame) -> list[dict]:
    rows = group[
        group.physically_visible.astype(bool)
        & (group.candidate_owner.astype(int) > 0)].sort_values("frame")
    runs: list[dict] = []
    for row in rows.itertuples(index=False):
        owner, frame = int(row.candidate_owner), int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and runs[-1]["last"] + 1 == frame):
            runs[-1]["last"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "frames": [frame]})
    return runs


def _scaled_step(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def _component_markers(component: np.ndarray, attached_frame: pd.DataFrame,
                       target_track: int, target_marker,
                       source_owner: int | None = None
                       ) -> list[tuple[int, int]]:
    markers = set()
    for row in attached_frame[
            attached_frame.physically_visible.astype(bool)].itertuples(index=False):
        if int(row.track_id) == int(target_track):
            continue
        if (source_owner is not None
                and int(row.candidate_owner) != int(source_owner)):
            continue
        marker = bounded._marker(component, row)
        if marker is not None and marker != target_marker:
            markers.add(marker)
    return sorted(markers)


def _valley_ratio(frame: np.ndarray, left: tuple[float, float],
                  right: tuple[float, float]) -> float:
    """Return the deepest line-sampled valley relative to the dimmer core."""
    distance = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    samples = max(3, int(np.ceil(distance)) + 1)
    values = []
    for weight in np.linspace(0.0, 1.0, samples):
        x = int(round((1.0 - weight) * left[0] + weight * right[0]))
        y = int(round((1.0 - weight) * left[1] + weight * right[1]))
        y = int(np.clip(y, 0, frame.shape[0] - 1))
        x = int(np.clip(x, 0, frame.shape[1] - 1))
        values.append(float(frame[y, x]))
    endpoint = max(min(values[0], values[-1]), 1.0)
    return float(np.clip(min(values[1:-1]) / endpoint, 0.0, 2.0))


def _geometry_partition(component: np.ndarray,
                        target_marker: tuple[int, int],
                        companion_markers: list[tuple[int, int]],
                        minimum_pixels: int,
                        ) -> tuple[np.ndarray, np.ndarray] | None:
    """Partition one mask by nearest core, then enforce connected seats."""
    yy, xx = np.indices(component.shape)
    target_distance = np.hypot(
        yy - int(target_marker[0]), xx - int(target_marker[1]))
    companion_distance = np.minimum.reduce([
        np.hypot(yy - int(marker[0]), xx - int(marker[1]))
        for marker in companion_markers])
    target_part = component & (target_distance <= companion_distance)
    source_part = component & ~target_part
    return bounded._connected_partition(
        component, target_part, source_part, target_marker,
        companion_markers, minimum_pixels)


def _candidate_intervals(runs: list[dict], maximum_middle: int):
    for left in range(len(runs) - 2):
        for right in range(left + 2, len(runs)):
            if runs[right]["owner"] != runs[left]["owner"]:
                continue
            middle = runs[left + 1:right]
            frames = [frame for run in middle for frame in run["frames"]]
            if not frames or len(frames) > maximum_middle:
                break
            yield runs[left], middle, runs[right], frames
            break


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit every short A-[foreign owners]-A interval in the field."""
    assert_target_free(params)
    attached = owner_consensus.attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    maximum_middle = max(1, int(math.ceil(movie * float(
        params.get("maximum_middle_movie_fraction", 0.02)))))
    minimum_middle = max(1, int(math.ceil(movie * float(
        params.get("minimum_middle_movie_fraction", 0.0)))))
    directly_owned_companions = bool(
        params.get("require_directly_owned_companions", False))
    minimum_bookend = max(2, int(math.ceil(movie * float(
        params.get("minimum_total_bookend_movie_fraction", 0.06)))))
    minimum_long_flank = max(1, int(math.ceil(movie * float(
        params.get("minimum_long_flank_movie_fraction", 0.03)))))
    minimum_purity = float(params.get("minimum_bracket_owner_purity", 0.65))
    minimum_strong = float(params.get("minimum_episode_strong_fraction", 0.50))
    maximum_step = float(params.get(
        "maximum_episode_step_sum_radii", 1.0))
    minimum_shared_fraction = float(params.get(
        "minimum_shared_mask_fraction", 0.50))
    maximum_shared_fraction = float(params.get(
        "maximum_shared_mask_fraction", 1.0))
    allow_geometry_fallback_above_maximum = bool(params.get(
        "allow_geometry_fallback_above_maximum_shared_fraction", False))
    minimum_distinct_middle = int(params.get(
        "minimum_distinct_middle_owners", 2))
    public_rows: list[dict] = []
    internal_rows: list[dict] = []
    for track, group in groups.items():
        positive = group[
            group.physically_visible.astype(bool)
            & (group.candidate_owner.astype(int) > 0)]
        if positive.empty:
            continue
        counts = positive.candidate_owner.astype(int).value_counts()
        runs = _owner_runs(group)
        for pre, middle, post, middle_frames in _candidate_intervals(
                runs, maximum_middle):
            bracket = int(pre["owner"])
            episode_frames = list(range(int(pre["first"]),
                                        int(post["last"]) + 1))
            episode = group[group.frame.astype(int).isin(episode_frames)]
            ordered = list(episode.sort_values("frame").itertuples(index=False))
            bracket_support = int(counts.get(bracket, 0))
            purity = float(bracket_support / len(positive))
            strong_fraction = float(episode.strong.astype(bool).mean()) \
                if len(episode) else 0.0
            maximum_seen_step = max(
                (_scaled_step(left, right)
                 for left, right in zip(ordered[:-1], ordered[1:])),
                default=float("inf"))
            distinct_middle = len({int(run["owner"]) for run in middle})
            shared_frames = []
            companion_counts: dict[int, int] = {}
            if distinct_middle >= minimum_distinct_middle:
                for frame_index in middle_frames:
                    target = _point(group, frame_index)
                    if target is None:
                        continue
                    component = bounded._component_at(
                        labels[frame_index], int(target.candidate_owner), target)
                    target_marker = (bounded._marker(component, target)
                                     if component is not None else None)
                    markers = (_component_markers(
                        component,
                        attached[attached.frame.astype(int).eq(frame_index)],
                        track, target_marker,
                        (int(target.candidate_owner)
                         if directly_owned_companions else None))
                        if component is not None and target_marker is not None
                        else [])
                    companion_counts[frame_index] = len(markers)
                    if markers:
                        shared_frames.append(frame_index)
            shared_fraction = float(len(shared_frames) / len(middle_frames))
            reasons: list[str] = []
            if len(middle_frames) < minimum_middle:
                reasons.append("middle_too_short_for_persistent_flash")
            if distinct_middle < minimum_distinct_middle:
                reasons.append("insufficient_distinct_middle_owners")
            if len(pre["frames"]) + len(post["frames"]) < minimum_bookend:
                reasons.append("insufficient_total_bookend_support")
            if max(len(pre["frames"]), len(post["frames"])) < minimum_long_flank:
                reasons.append("insufficient_long_flank_support")
            if purity < minimum_purity:
                reasons.append("insufficient_bracket_owner_purity")
            if (len(episode) != len(episode_frames)
                    or not episode.physically_visible.astype(bool).all()
                    or not (episode.candidate_owner.astype(int) > 0).all()):
                reasons.append("episode_not_positive_physically_continuous")
            if strong_fraction < minimum_strong:
                reasons.append("episode_too_dim")
            if maximum_seen_step > maximum_step:
                reasons.append("episode_physical_discontinuity")
            if shared_fraction < minimum_shared_fraction:
                reasons.append("insufficient_shared_mask_evidence")
            if (shared_fraction > maximum_shared_fraction
                    and not allow_geometry_fallback_above_maximum):
                reasons.append("persistent_shared_mask_is_not_owner_flash")
            public = {
                "proposal_id": "", "track_id": track,
                "bracket_owner": bracket,
                "middle_owners": "|".join(
                    str(int(run["owner"])) for run in middle),
                "middle_first": min(middle_frames),
                "middle_last": max(middle_frames),
                "middle_frames": len(middle_frames),
                "pre_frames": len(pre["frames"]),
                "post_frames": len(post["frames"]),
                "bracket_support": bracket_support,
                "positive_observations": int(len(positive)),
                "bracket_purity": purity,
                "distinct_middle_owners": distinct_middle,
                "episode_strong_fraction": strong_fraction,
                "maximum_episode_step_sum_radii": maximum_seen_step,
                "shared_mask_frames": len(shared_frames),
                "shared_mask_fraction": shared_fraction,
                "eligible": not reasons,
                "reason": ("eligible_bracketed_mixed_owner_flash"
                           if not reasons else "|".join(reasons)),
            }
            public_rows.append(public)
            internal_rows.append({
                **public, "frame_values": middle_frames,
                "source_companion_counts": companion_counts,
            })
    number = 0
    for public, internal in zip(public_rows, internal_rows):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"MOF{number:04d}"
    return (pd.DataFrame(public_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internal_rows), attached)


def _prepare_frame(frame: np.ndarray, raw: np.ndarray,
                   attached_frame: pd.DataFrame, target,
                   bracket_owner: int, params: dict):
    source_owner = int(target.candidate_owner)
    component = bounded._component_at(frame, source_owner, target)
    if component is None:
        return None, "source_component_missing", 0, 0
    target_marker = bounded._marker(component, target)
    if target_marker is None:
        return None, "target_marker_missing", 0, 0
    companions = _component_markers(
        component, attached_frame, int(target.track_id), target_marker,
        (source_owner if bool(params.get(
            "require_directly_owned_companions", False)) else None))
    before = frame.copy()
    if not companions:
        trial = frame.copy()
        trial[component] = int(bracket_owner)
        method = "isolated_interval_component_relabel"
    else:
        markers = np.zeros(component.shape, np.int16)
        markers[target_marker] = 1
        for marker_value, marker in enumerate(companions, start=2):
            markers[marker] = marker_value
        elevation = -ndi.gaussian_filter(
            raw.astype(np.float32),
            float(params.get("watershed_sigma_px", 1.0)))
        basins = watershed(elevation, markers=markers, mask=component,
                           connectivity=STRUCTURE)
        target_part = basins == 1
        source_part = component & ~target_part
        connected = bounded._connected_partition(
            component, target_part, source_part, target_marker, companions,
            int(params.get("minimum_partition_pixels", 5)))
        if connected is None:
            companion_rows = []
            for row in attached_frame[
                    attached_frame.physically_visible.astype(bool)
                    & attached_frame.candidate_owner.astype(int).eq(
                        source_owner)].itertuples(index=False):
                if int(row.track_id) == int(target.track_id):
                    continue
                marker = bounded._marker(component, row)
                if marker in companions:
                    companion_rows.append(row)
            separations = [
                float(np.hypot(float(target.x) - float(row.x),
                               float(target.y) - float(row.y)))
                / max(float(target.radius_px) + float(row.radius_px), 1.0)
                for row in companion_rows]
            valleys = [
                _valley_ratio(raw, (float(target.x), float(target.y)),
                              (float(row.x), float(row.y)))
                for row in companion_rows]
            minimum_separation = float(params.get(
                "minimum_geometry_marker_separation_sum_radii", 0.8))
            maximum_valley = float(params.get(
                "maximum_geometry_fallback_valley_ratio", 0.85))
            if (not separations or min(separations) < minimum_separation
                    or not valleys or max(valleys) > maximum_valley):
                return (None, "connected_partition_unavailable", 0,
                        len(companions))
            connected = _geometry_partition(
                component, target_marker, companions,
                int(params.get("minimum_partition_pixels", 5)))
            if connected is None:
                return (None, "geometry_partition_unavailable", 0,
                        len(companions))
            method = "raw_valley_geometry_connected"
        else:
            method = "multicore_raw_watershed_connected"
        target_part, source_part = connected
        trial = frame.copy()
        trial[target_part] = int(bracket_owner)
        trial[source_part] = source_owner
    target_disk = owner_consensus._disk_owner(
        trial, float(target.x), float(target.y), float(target.radius_px))[0]
    if int(target_disk) != int(bracket_owner):
        return None, "bracket_owner_dominance_failed", 0, len(companions)
    return trial, method, int(np.count_nonzero(trial != before)), len(companions)


def apply(labels: np.ndarray, raw: np.ndarray, attached: pd.DataFrame,
          internal: pd.DataFrame, params: dict):
    """Apply each fully proved interval atomically."""
    candidate = labels.copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    applications: list[dict] = []
    applied = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending = []
        failure = ""
        for frame_index in list(proposal.frame_values):
            target = _point(groups[int(proposal.track_id)], frame_index)
            if target is None:
                failure = "physical_point_missing"
                break
            trial, method, changed, companions = _prepare_frame(
                candidate[frame_index], raw[frame_index],
                attached[attached.frame.astype(int).eq(frame_index)], target,
                int(proposal.bracket_owner), params)
            if trial is None:
                failure = method
                break
            duplicate_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.bracket_owner))
                - component_accounting.component_excess(
                    candidate[frame_index], int(proposal.bracket_owner)))
            if duplicate_delta > 0:
                failure = "bracket_owner_duplicate_would_be_created"
                break
            pending.append((frame_index, trial, method, changed, companions,
                            int(target.candidate_owner)))
        if (not failure
                and float(proposal.shared_mask_fraction)
                > float(params.get("maximum_shared_mask_fraction", 1.0))
                and bool(params.get(
                    "allow_geometry_fallback_above_maximum_shared_fraction",
                    False))
                and not any(row[2] == "raw_valley_geometry_connected"
                            for row in pending)):
            failure = "high_shared_fraction_requires_geometry_fallback"
        if failure:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "frame": int(proposal.middle_first),
                "bracket_owner": int(proposal.bracket_owner),
                "source_owner": 0, "source_companion_cores": 0,
                "partition_method": "atomic", "changed_pixels": 0,
                "applied": False, "reason": "atomic_refusal:" + failure,
            })
            continue
        for frame_index, trial, method, changed, companions, source in pending:
            candidate[frame_index] = trial
            applications.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id), "frame": frame_index,
                "bracket_owner": int(proposal.bracket_owner),
                "source_owner": source,
                "source_companion_cores": companions,
                "partition_method": method, "changed_pixels": changed,
                "applied": True,
                "reason": "atomic_bracketed_mixed_owner_flash_recovery",
            })
        applied += 1
    changed = candidate != labels
    details = {
        "applied_proposals": applied,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(
            owner_consensus.count_new_duplicate_components(labels, candidate)),
    }
    return (candidate,
            pd.DataFrame(applications, columns=APPLICATION_COLUMNS), details)
