"""Extend dedicated-owner relay discovery across raw-reference handoffs.

This experimental producer is field-wide. Identity values, track numbers,
frames, events, coordinates, regions, and review cases are forbidden inputs.
"""
from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import concurrent_duplicate_invasion as invasion
import ownerless_body_recovery as ownerless
import recurrent_exclusive_owner_relay as relay
import separable_merge_recovery as physical


FORBIDDEN_TARGETS = {
    "identity", "identities", "identity_ids", "owner_ids",
    "track", "tracks", "track_ids", "physical_tracks",
    "frame", "frames", "frame_ids", "frame_range",
    "event", "events", "event_ids", "coordinates", "coordinate",
    "region", "regions", "roi", "review_cases", "review_case_ids",
}


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        key for key, value in params.items()
        if key.lower() in FORBIDDEN_TARGETS and value not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "cross-reference owner relay received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("cross-reference owner relay must be field-wide")


def _positive_runs(rows: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in rows.sort_values("frame").itertuples(index=False):
        owner = int(row.accepted_owner)
        if owner <= 0 or not bool(row.physically_visible):
            continue
        frame = int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame == runs[-1]["last_frame"] + 1):
            runs[-1]["last_frame"] = frame
            runs[-1]["frames"] += 1
        else:
            runs.append({"owner": owner, "first_frame": frame,
                         "last_frame": frame, "frames": 1})
    return runs


def _linearized(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                  int, int]:
    """Use the earlier reference through its end, then its longer successor."""
    left_key = (int(left.frame.min()), int(left.frame.max()))
    right_key = (int(right.frame.min()), int(right.frame.max()))
    if left_key <= right_key:
        predecessor, successor = left, right
    else:
        predecessor, successor = right, left
    cut = int(predecessor.frame.max())
    combined = pd.concat([
        predecessor,
        successor[successor.frame.astype(int) > cut],
    ], ignore_index=True).sort_values("frame")
    return combined, int(predecessor.track_id.iloc[0]), \
        int(successor.track_id.iloc[0])


def _parse_pair(row) -> tuple[int, int]:
    value = row.track_pair
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return int(parsed[0]), int(parsed[1])
    return int(value[0]), int(value[1])


def _track_arrays(group: pd.DataFrame) -> dict[str, np.ndarray]:
    """Freeze one sorted reference track into contiguous numeric arrays."""
    return {
        "frame": group.frame.to_numpy(dtype=np.int64, copy=True),
        "owner": group.accepted_owner.to_numpy(dtype=np.int64, copy=True),
        "x": group.x.to_numpy(dtype=float, copy=True),
        "y": group.y.to_numpy(dtype=float, copy=True),
        "radius": group.radius_px.to_numpy(dtype=float, copy=True),
    }


def _virtual_owner_history(
        left_track: int, right_track: int,
        arrays: dict[int, dict[str, np.ndarray]],
        ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return the same predecessor/successor history as `_linearized`."""
    left, right = arrays[left_track], arrays[right_track]
    left_key = (int(left["frame"][0]), int(left["frame"][-1]))
    right_key = (int(right["frame"][0]), int(right["frame"][-1]))
    if left_key <= right_key:
        predecessor_track, successor_track = left_track, right_track
    else:
        predecessor_track, successor_track = right_track, left_track
    predecessor = arrays[predecessor_track]
    successor = arrays[successor_track]
    cut = int(predecessor["frame"][-1])
    tail = successor["frame"] > cut
    frames = np.concatenate((predecessor["frame"], successor["frame"][tail]))
    owners = np.concatenate((predecessor["owner"], successor["owner"][tail]))
    return frames, owners, predecessor_track, successor_track


def _owner_run_summary(frames: np.ndarray, owners: np.ndarray,
                       ) -> tuple[int, Counter, dict[int, list[int]]]:
    """Summarize `_positive_runs` without constructing per-pair DataFrames."""
    positive = owners > 0
    positive_frames = frames[positive]
    positive_owners = owners[positive]
    if not len(positive_frames):
        return 0, Counter(), {}
    starts = np.r_[
        0,
        np.flatnonzero(
            (positive_owners[1:] != positive_owners[:-1])
            | (positive_frames[1:] != positive_frames[:-1] + 1)
        ) + 1,
    ]
    run_owners = positive_owners[starts]
    run_first_frames = positive_frames[starts]
    owner_runs: dict[int, list[int]] = {}
    for owner, first_frame in zip(run_owners, run_first_frames):
        owner_runs.setdefault(int(owner), []).append(int(first_frame))
    return int(len(starts)), Counter(map(int, positive_owners)), owner_runs


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every overlapping reference pair and its virtual owner history."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id", sort=True)}
    arrays = {track: _track_arrays(group) for track, group in groups.items()}
    owner_tracks = {
        int(owner): set(map(int, group.track_id))
        for owner, group in visible[visible.accepted_owner.astype(int) > 0]
        .groupby("accepted_owner", sort=True)
    }

    minimum_overlap = int(params.get("minimum_overlap_frames", 4))
    minimum_same_owner = float(params.get(
        "minimum_overlap_same_owner_fraction", 0.80))
    maximum_median_separation = float(params.get(
        "maximum_median_separation_sum_radii", 1.75))
    maximum_separation = float(params.get(
        "maximum_separation_sum_radii", 2.20))
    minimum_successor_extension = int(params.get(
        "minimum_successor_extension_frames", 8))
    minimum_support = int(params.get("minimum_dedicated_support_frames", 8))
    minimum_support_fraction = float(params.get(
        "minimum_dedicated_support_fraction", 0.15))
    minimum_runs = int(params.get("minimum_dedicated_runs", 2))
    minimum_transitions = int(params.get("minimum_owner_transitions", 4))
    minimum_competing = int(params.get("minimum_competing_owner_count", 1))
    minimum_competing_tracks = int(params.get(
        "minimum_competing_owner_tracks", 2))

    audits: list[dict] = []
    proposal_number = 0
    track_ids = sorted(groups)
    for left_index, left_track in enumerate(track_ids):
        left = arrays[left_track]
        for right_track in track_ids[left_index + 1:]:
            right = arrays[right_track]
            common, left_indices, right_indices = np.intersect1d(
                left["frame"], right["frame"], assume_unique=True,
                return_indices=True)
            if len(common) < minimum_overlap:
                continue
            same_owner: list[bool] = []
            separations: list[float] = []
            for left_row, right_row in zip(left_indices, right_indices):
                owner_a = int(left["owner"][left_row])
                owner_b = int(right["owner"][right_row])
                same_owner.append(owner_a > 0 and owner_a == owner_b)
                scale = max(
                    float(left["radius"][left_row])
                    + float(right["radius"][right_row]), 1.0)
                separations.append(float(np.hypot(
                    float(left["x"][left_row])
                    - float(right["x"][right_row]),
                    float(left["y"][left_row])
                    - float(right["y"][right_row])) / scale))
            same_fraction = float(np.mean(same_owner))
            median_separation = float(np.median(separations))
            max_separation = float(np.max(separations))
            virtual_frames, virtual_owners, predecessor, successor = \
                _virtual_owner_history(
                    left_track, right_track, arrays)
            extension = int(arrays[successor]["frame"][-1]
                            - arrays[predecessor]["frame"][-1])
            proposal_number += 1
            reasons: list[str] = []
            if same_fraction < minimum_same_owner:
                reasons.append("overlap_owner_not_conserved")
            if median_separation > maximum_median_separation:
                reasons.append("median_reference_separation_too_large")
            if max_separation > maximum_separation:
                reasons.append("maximum_reference_separation_too_large")
            if extension < minimum_successor_extension:
                reasons.append("successor_extension_too_short")

            run_count, owner_counts, owner_run_frames = _owner_run_summary(
                virtual_frames, virtual_owners)
            transitions = max(0, run_count - 1)
            candidates: list[dict] = []
            pair = {left_track, right_track}
            for owner, support in sorted(owner_counts.items()):
                owner_runs = owner_run_frames[owner]
                competing = [
                    int(other) for other in owner_counts
                    if other != owner
                    and len(owner_tracks.get(int(other), set()))
                    >= minimum_competing_tracks]
                if not owner_tracks.get(owner, set()) <= pair:
                    continue
                if support < minimum_support:
                    continue
                if support / max(len(virtual_frames), 1) < minimum_support_fraction:
                    continue
                if len(owner_runs) < minimum_runs:
                    continue
                if len(competing) < minimum_competing:
                    continue
                candidates.append({
                    "dedicated_owner": owner,
                    "dedicated_support_frames": support,
                    "dedicated_support_fraction": support / len(virtual_frames),
                    "dedicated_runs": len(owner_runs),
                    "first_dedicated_frame": owner_runs[0],
                    "competing_owners": competing,
                })
            if transitions < minimum_transitions:
                reasons.append("insufficient_owner_transitions")
            if len(candidates) != 1:
                reasons.append("dedicated_owner_not_unique")
            selected = candidates[0] if len(candidates) == 1 else {}
            eligible = not reasons
            audits.append({
                "proposal_id": f"P{proposal_number:04d}",
                "track_pair": str((left_track, right_track)),
                "predecessor_track": predecessor,
                "successor_track": successor,
                "overlap_first_frame": int(common[0]),
                "overlap_last_frame": int(common[-1]),
                "overlap_frames": len(common),
                "overlap_same_owner_fraction": same_fraction,
                "median_separation_sum_radii": median_separation,
                "maximum_separation_sum_radii": max_separation,
                "successor_extension_frames": extension,
                "virtual_first_frame": int(virtual_frames[0]),
                "virtual_last_frame": int(virtual_frames[-1]),
                "virtual_frames": int(len(virtual_frames)),
                "owner_runs": run_count,
                "owner_transitions": transitions,
                "dedicated_owner": selected.get("dedicated_owner", 0),
                "dedicated_support_frames": selected.get(
                    "dedicated_support_frames", 0),
                "dedicated_support_fraction": selected.get(
                    "dedicated_support_fraction", 0.0),
                "dedicated_runs": selected.get("dedicated_runs", 0),
                "first_dedicated_frame": selected.get(
                    "first_dedicated_frame", -1),
                "competing_owners": str(selected.get("competing_owners", [])),
                "eligible": eligible,
                "reason": "eligible" if eligible else "|".join(reasons),
            })
    return pd.DataFrame(audits), scored


def _virtual_rows(scored: pd.DataFrame, pair: tuple[int, int]) -> pd.DataFrame:
    left = scored[(scored.track_id.astype(int) == pair[0])
                  & scored.physically_visible.astype(bool)]
    right = scored[(scored.track_id.astype(int) == pair[1])
                   & scored.physically_visible.astype(bool)]
    return _linearized(left, right)[0]


def _bounded_anchor_core(
        component: np.ndarray, marker: tuple[int, int],
        expected_area: float, protected: np.ndarray, params: dict,
        ) -> np.ndarray:
    """Return one connected, anchor-sized subset of an existing host mask."""
    core_radius = (float(params.get("stationary_anchor_core_area_scale", 1.2))
                   * np.sqrt(max(expected_area, 1.0) / np.pi))
    yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
    core = component & (
        (xx - marker[1]) ** 2 + (yy - marker[0]) ** 2 <= core_radius ** 2)
    core &= ~protected
    components, _ = ndi.label(core, np.ones((3, 3), np.uint8))
    marker_component = int(components[marker])
    return components == marker_component if marker_component else np.zeros_like(core)


def _topology_safe_anchor_core(
        component: np.ndarray, core: np.ndarray, markers: np.ndarray,
        ) -> np.ndarray:
    """Keep one connected host remainder and absorb only detached fragments."""
    remainder, count = ndi.label(
        component & ~core, np.ones((3, 3), np.uint8))
    if count <= 1:
        return core
    companion_components = remainder[markers == 2]
    companion_components = companion_components[companion_components > 0]
    if len(companion_components):
        values, counts = np.unique(companion_components, return_counts=True)
        keep = int(values[np.argmax(counts)])
    else:
        values, counts = np.unique(remainder[remainder > 0], return_counts=True)
        keep = int(values[np.argmax(counts)])
    safe = component & (remainder != keep)
    return safe if ndi.label(safe, np.ones((3, 3), np.uint8))[1] == 1 else core


def _prepare_shared_component_frame(
        candidate: np.ndarray, candidate_unclaimed: np.ndarray,
        baseline_labels: np.ndarray, baseline_unclaimed: np.ndarray,
        raw: np.ndarray, by_frame: dict[int, pd.DataFrame], row,
        pair: tuple[int, int], dedicated_owner: int, unclaimed_id: int,
        params: dict, anchor: dict | None = None,
        ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Reserve one raw-reference basin and preserve every companion basin.

    Unlike the earlier relay helper, the host is allowed to end on this frame.
    That is safe only because the complete cascade is atomic, the dedicated
    owner's old seat is released to the ledger, and every other physical point
    inside a shared host component is retained behind an independent marker.
    """
    frame = int(row.frame)
    host_owner = int(row.accepted_owner)
    detail = {
        "frame": frame, "host_owner": host_owner,
        "changed_pixels": 0, "orphan_pixels": 0,
        "recovered_pixels": 0, "companion_markers": 0,
        "reason": "",
    }
    component = invasion._component_near(
        candidate[frame], host_owner, row)
    if component is None:
        return None, None, {**detail, "reason": "host_component_missing"}
    anchor_mode = anchor is not None
    target_x = float(anchor["anchor_x"] if anchor_mode else row.x)
    target_y = float(anchor["anchor_y"] if anchor_mode else row.y)
    anchor_marker = physical._marker_pixel(component, target_x, target_y)
    if anchor_mode:
        anchor_radius = max(float(anchor["anchor_radius_px"]), 1.0)
        yy, xx = np.nonzero(component)
        seed_distance = np.hypot(xx - target_x, yy - target_y)
        seed_domain = seed_distance <= float(params.get(
            "maximum_anchor_seed_distance_radius_scale", 2.5)) * anchor_radius
        if np.any(seed_domain):
            smooth = ndi.gaussian_filter(
                raw[frame].astype(np.float32),
                float(params.get("watershed_sigma_px", 1.0)))
            eligible = np.flatnonzero(seed_domain)
            best = int(eligible[np.argmax(smooth[yy[eligible], xx[eligible]])])
            target_marker = int(yy[best]), int(xx[best])
        else:
            target_marker = None
    else:
        target_marker = physical._marker_pixel(component, target_x, target_y)
    if target_marker is None:
        return None, None, {**detail, "reason": "target_marker_missing"}

    markers = np.zeros(candidate.shape[1:], np.int16)
    markers[target_marker] = 1
    protected_companion_core = np.zeros_like(component)
    companions: list[int] = []
    maximum_marker_offset_radii = float(params.get(
        "maximum_companion_marker_offset_radii", 1.0))
    for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
        track_id = int(other.track_id)
        if (track_id in pair
                or int(other.accepted_owner) != host_owner):
            continue
        marker = physical._marker_pixel(
            component, float(other.x), float(other.y))
        if marker is None:
            continue
        marker_distance = float(np.hypot(
            marker[1] - float(other.x), marker[0] - float(other.y)))
        maximum_offset = maximum_marker_offset_radii * max(
            float(other.radius_px), 1.0)
        if marker_distance <= maximum_offset and marker != target_marker:
            markers[marker] = 2
            companions.append(track_id)
            radius = float(np.clip(
                float(params.get("companion_protected_radius_scale", 0.75))
                * max(float(other.radius_px), 1.0), 2.0, 6.0))
            yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
            protected_companion_core |= component & (
                (xx - float(other.x)) ** 2
                + (yy - float(other.y)) ** 2 <= radius ** 2)
    component_area = int(np.count_nonzero(component))
    expected_anchor_area = float(
        anchor.get("anchor_component_area_px", 0.0)) if anchor_mode else 0.0
    component_area_ratio = (component_area / max(expected_anchor_area, 1.0)
                            if anchor_mode else np.nan)
    recovery_mode = ""
    if companions:
        elevation = -ndi.gaussian_filter(
            raw[frame].astype(np.float32),
            float(params.get("watershed_sigma_px", 1.0)))
        recovered = watershed(
            elevation, markers=markers, mask=component,
            connectivity=np.ones((3, 3), np.uint8)) == 1
        recovered &= ~protected_companion_core
        recovery_mode = "raw_seeded_shared_component_partition"
    elif (anchor_mode and component_area_ratio > float(params.get(
            "maximum_isolated_anchor_component_area_ratio", 6.0))):
        # A large single-reference host is a two-owner merge, not evidence that
        # its complete area belongs to the stationary anchor.  Seed the known
        # stationary seat and let the rest retain its current owner.  An outer
        # marker ring bounds the anchor basin while raw intensity places the
        # internal boundary; no moving-owner identity or coordinate is needed.
        recovered = _bounded_anchor_core(
            component, target_marker, expected_anchor_area,
            protected_companion_core, params)
        recovery_mode = "raw_seeded_bounded_stationary_anchor_core"
    else:
        recovered = component.copy()
        recovery_mode = "isolated_host_component_relabel"
    if (anchor_mode and component_area_ratio > float(params.get(
            "maximum_isolated_anchor_component_area_ratio", 6.0))):
        bounded_marker = anchor_marker or target_marker
        recovered = _bounded_anchor_core(
            component, bounded_marker, expected_anchor_area,
            protected_companion_core, params)
        recovered = _topology_safe_anchor_core(
            component, recovered, markers)
        recovery_mode = ("companion_safe_anchor_centred_stationary_core"
                         if companions else
                         "anchor_centred_bounded_stationary_core")
    expected_repair_area = getattr(row, "expected_repair_area_px", np.nan)
    if (not anchor_mode
            and bool(params.get(
                "enable_expected_area_bridge_fallback", False))
            and pd.notna(expected_repair_area)
            and float(expected_repair_area) > 0
            and np.count_nonzero(recovered) / float(expected_repair_area)
            < float(params.get(
                "minimum_bridge_recovered_area_ratio", 0.5))):
        bridge_params = dict(params)
        bridge_params["stationary_anchor_core_area_scale"] = float(
            params.get("moving_bridge_core_area_scale", 1.0))
        expected_resident_area = getattr(
            row, "expected_resident_area_px", np.nan)
        companion_pixels = np.argwhere(markers == 2)
        if (bool(params.get(
                "enable_expected_resident_core_fallback", False))
                and pd.notna(expected_resident_area)
                and float(expected_resident_area) > 0
                and len(companion_pixels) == 1):
            resident_marker = tuple(map(int, companion_pixels[0]))
            yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
            target_protected = component & (
                (xx - target_marker[1]) ** 2
                + (yy - target_marker[0]) ** 2
                <= float(params.get(
                    "moving_bridge_target_protected_radius_scale", 0.75))
                * max(float(row.radius_px), 1.0) ** 2)
            resident_params = dict(params)
            resident_params["stationary_anchor_core_area_scale"] = float(
                params.get("moving_bridge_resident_core_area_scale", 1.0))
            resident_core = _bounded_anchor_core(
                component, resident_marker, float(expected_resident_area),
                target_protected, resident_params)
            reverse_markers = np.zeros_like(markers)
            reverse_markers[resident_marker] = 1
            reverse_markers[target_marker] = 2
            resident_core = _topology_safe_anchor_core(
                component, resident_core, reverse_markers)
            recovered = component & ~resident_core
            recovery_mode = "topology_safe_expected_resident_remainder"
        else:
            recovered = _bounded_anchor_core(
                component, target_marker, float(expected_repair_area),
                protected_companion_core, bridge_params)
            recovered = _topology_safe_anchor_core(
                component, recovered, markers)
            recovery_mode = "topology_safe_expected_area_bridge_core"
    if (not anchor_mode and pd.notna(expected_repair_area)
            and float(expected_repair_area) > 0):
        bridge_area_ratio = (
            np.count_nonzero(recovered) / float(expected_repair_area))
        if not (float(params.get(
                "minimum_bridge_output_area_ratio", 0.2))
                <= bridge_area_ratio
                <= float(params.get(
                    "maximum_bridge_output_area_ratio", 2.5))):
            return None, None, {
                **detail, "component_area_px": component_area,
                "bridge_recovered_area_ratio": bridge_area_ratio,
                "reason": "moving_bridge_basin_area_out_of_range"}
    if anchor_mode:
        recovered_ratio = float(
            np.count_nonzero(recovered) / max(expected_anchor_area, 1.0))
        minimum_anchor_ratio = float(params.get(
            "minimum_stationary_anchor_area_ratio", 0.20))
        if not (minimum_anchor_ratio
                <= recovered_ratio
                <= float(params.get("maximum_stationary_anchor_area_ratio", 2.5))):
            return None, None, {
                **detail, "component_area_px": component_area,
                "component_area_ratio": component_area_ratio,
                "recovered_area_ratio": recovered_ratio,
                "anchor_recovery_mode": recovery_mode,
                "reason": "stationary_anchor_basin_area_out_of_range"}
    if int(np.count_nonzero(recovered)) < int(
            params.get("minimum_changed_pixels", 5)):
        return None, None, {**detail, "reason": "recovered_core_too_small"}

    trial = candidate[frame].copy()
    unclaimed_trial = candidate_unclaimed[frame].copy()
    old_dedicated = trial == int(dedicated_owner)
    trial[old_dedicated] = 0
    unclaimed_trial[old_dedicated] = int(unclaimed_id)
    trial[recovered] = int(dedicated_owner)
    unclaimed_trial[recovered] = 0
    if not np.array_equal(
            (trial > 0) | (unclaimed_trial > 0),
            (baseline_labels[frame] > 0) | (baseline_unclaimed[frame] > 0)):
        return None, None, {**detail, "reason": "foreground_ledger_union_changed"}
    if np.any((baseline_unclaimed[frame] > 0)
              & (unclaimed_trial != baseline_unclaimed[frame])):
        return None, None, {**detail, "reason": "existing_unclaimed_changed"}
    if relay._component_excess(trial, dedicated_owner) \
            > relay._component_excess(baseline_labels[frame], dedicated_owner):
        return None, None, {**detail, "reason": "new_dedicated_owner_duplicate"}
    if relay._component_excess(trial, host_owner) \
            > relay._component_excess(baseline_labels[frame], host_owner):
        return None, None, {**detail, "reason": "new_host_owner_duplicate"}
    return trial, unclaimed_trial, {
        **detail,
        "changed_pixels": int(np.count_nonzero(trial != candidate[frame])),
        "orphan_pixels": int(np.count_nonzero(old_dedicated)),
        "recovered_pixels": int(np.count_nonzero(recovered)),
        "companion_markers": len(set(companions)),
        "component_area_px": component_area,
        "component_area_ratio": component_area_ratio,
        "recovered_area_ratio": (
            float(np.count_nonzero(recovered) / max(expected_anchor_area, 1.0))
            if anchor_mode else np.nan),
        "bridge_recovered_area_ratio": (
            float(np.count_nonzero(recovered) / float(expected_repair_area))
            if pd.notna(expected_repair_area)
            and float(expected_repair_area) > 0 else np.nan),
        "anchor_recovery_mode": recovery_mode,
        "reason": recovery_mode,
    }


def _contiguous_owner_runs(rows: pd.DataFrame) -> list[dict]:
    """Return positive-owner runs from an already frame-sorted table."""
    runs: list[dict] = []
    for row in rows.sort_values("frame").itertuples(index=False):
        owner = int(row.accepted_owner)
        frame = int(row.frame)
        if owner <= 0:
            continue
        if (runs and runs[-1]["owner"] == owner
                and runs[-1]["last_frame"] + 1 == frame):
            runs[-1]["last_frame"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first_frame": frame,
                         "last_frame": frame, "rows": [row]})
    return runs


def _moving_owner_merge_bridge_rows(
        labels: np.ndarray, raw: np.ndarray, scored: pd.DataFrame,
        moving_owner: int, excluded_tracks: tuple[int, ...], params: dict,
        ) -> tuple[pd.DataFrame, dict]:
    """Bridge one split-reference excursion through an established resident.

    A moving lineage can leave one physical reference under a foreign owner
    and reappear on a new reference with its old owner. The interval is safe
    only when the same foreign owner has an independent, continuously visible
    resident reference inside the shared component, the old moving owner is
    absent, the host is enlarged, and raw signal resolves two cores on most
    interval frames. Identity and reference values are inferred field-wide.
    """
    detail = {
        "moving_merge_bridge_candidates": 0,
        "moving_merge_bridge_frames": 0,
        "moving_merge_bridge_reason": "disabled",
    }
    if not bool(params.get("enable_moving_owner_merge_bridge", False)):
        return pd.DataFrame(), detail

    visible = scored[scored.physically_visible.astype(bool)].copy()
    excluded_track_set = set(map(int, excluded_tracks))
    groups = {
        int(track): group.sort_values("frame")
        for track, group in visible.groupby("track_id", sort=True)
        if int(track) not in excluded_track_set
    }
    minimum_pre = int(params.get(
        "minimum_moving_bridge_pre_owner_frames", 3))
    minimum_post = int(params.get(
        "minimum_moving_bridge_post_owner_frames", 8))
    maximum_reference_gap = int(params.get(
        "maximum_moving_bridge_reference_gap_frames", 2))
    maximum_interval = int(params.get(
        "maximum_moving_bridge_interval_frames", 8))
    minimum_resident_support = int(params.get(
        "minimum_moving_bridge_resident_support_frames", 8))
    minimum_component_ratio = float(params.get(
        "minimum_moving_bridge_component_area_ratio", 1.5))
    minimum_separation = float(params.get(
        "minimum_moving_bridge_separation_sum_radii", 0.75))
    maximum_valley_ratio = float(params.get(
        "maximum_moving_bridge_two_core_valley_ratio", 0.8))
    minimum_two_core_fraction = float(params.get(
        "minimum_moving_bridge_two_core_fraction", 0.75))

    departures: list[dict] = []
    arrivals: list[dict] = []
    for track, group in groups.items():
        runs = _contiguous_owner_runs(group)
        for before, after in zip(runs, runs[1:]):
            if after["first_frame"] != before["last_frame"] + 1:
                continue
            if (before["owner"] == moving_owner
                    and after["owner"] not in {moving_owner, 0}
                    and len(before["rows"]) >= minimum_pre):
                departures.append({
                    "track": track, "moving": before,
                    "foreign": after})
            if (before["owner"] not in {moving_owner, 0}
                    and after["owner"] == moving_owner
                    and len(after["rows"]) >= minimum_post):
                arrivals.append({
                    "track": track, "foreign": before,
                    "moving": after})

    candidates: list[dict] = []
    structure = np.ones((3, 3), np.uint8)
    for departure in departures:
        for arrival in arrivals:
            if departure["track"] == arrival["track"]:
                continue
            foreign_owner = int(departure["foreign"]["owner"])
            if int(arrival["foreign"]["owner"]) != foreign_owner:
                continue
            left_end = int(departure["foreign"]["last_frame"])
            right_start = int(arrival["foreign"]["first_frame"])
            if not (0 < right_start - left_end <= maximum_reference_gap):
                continue
            first_frame = int(departure["foreign"]["first_frame"])
            last_frame = int(arrival["foreign"]["last_frame"])
            interval = list(range(first_frame, last_frame + 1))
            if not interval or len(interval) > maximum_interval:
                continue
            if any(np.any(labels[frame] == moving_owner)
                   for frame in interval):
                continue

            known = {
                int(row.frame): row
                for run in (departure["foreign"], arrival["foreign"])
                for row in run["rows"]
                if first_frame <= int(row.frame) <= last_frame
            }
            known_frames = sorted(known)
            path: dict[int, dict] = {}
            for frame in interval:
                if frame in known:
                    row = known[frame]
                    path[frame] = {
                        "x": float(row.x), "y": float(row.y),
                        "radius_px": float(row.radius_px)}
                    continue
                before = max((value for value in known_frames
                              if value < frame), default=None)
                after = min((value for value in known_frames
                             if value > frame), default=None)
                if before is None or after is None:
                    path = {}
                    break
                weight = (frame - before) / float(after - before)
                left, right = known[before], known[after]
                path[frame] = {
                    key: ((1.0 - weight) * float(getattr(left, key))
                          + weight * float(getattr(right, key)))
                    for key in ("x", "y", "radius_px")}
            if len(path) != len(interval):
                continue

            resident_candidates: list[tuple[int, pd.DataFrame]] = []
            excluded = {int(departure["track"]), int(arrival["track"])}
            for resident_track, resident in groups.items():
                if resident_track in excluded:
                    continue
                rows = resident[
                    resident.frame.astype(int).isin(interval)
                    & (resident.accepted_owner.astype(int) == foreign_owner)]
                if set(map(int, rows.frame)) == set(interval):
                    resident_candidates.append((resident_track, rows))
            if len(resident_candidates) != 1:
                continue
            resident_track, resident_rows = resident_candidates[0]
            resident_index = {
                int(row.frame): row
                for row in resident_rows.itertuples(index=False)}

            ordinary_areas: list[int] = []
            resident_history = groups[resident_track]
            resident_history = resident_history[
                resident_history.accepted_owner.astype(int) == foreign_owner]
            for row in resident_history.itertuples(index=False):
                components, _ = ndi.label(
                    labels[int(row.frame)] == foreign_owner, structure)
                component_id = physical._component_at(
                    components, float(row.x), float(row.y), search_radius=6)
                if component_id > 0:
                    ordinary_areas.append(
                        int(np.count_nonzero(components == component_id)))
            if len(ordinary_areas) < minimum_resident_support:
                continue
            ordinary_area = float(np.median(ordinary_areas))

            two_core_frames = 0
            valid = True
            for frame in interval:
                target = path[frame]
                resident = resident_index[frame]
                components, _ = ndi.label(
                    labels[frame] == foreign_owner, structure)
                target_component = physical._component_at(
                    components, target["x"], target["y"], search_radius=6)
                resident_component = physical._component_at(
                    components, float(resident.x), float(resident.y),
                    search_radius=6)
                if (target_component <= 0
                        or target_component != resident_component):
                    valid = False
                    break
                component = components == target_component
                area_ratio = (np.count_nonzero(component)
                              / max(ordinary_area, 1.0))
                if area_ratio < minimum_component_ratio:
                    valid = False
                    break
                separation = float(np.hypot(
                    target["x"] - float(resident.x),
                    target["y"] - float(resident.y))) / max(
                        target["radius_px"] + float(resident.radius_px), 1.0)
                if separation < minimum_separation:
                    valid = False
                    break
                valley = physical._valley_ratio(
                    raw[frame], (target["x"], target["y"]),
                    (float(resident.x), float(resident.y)))
                two_core_frames += int(valley <= maximum_valley_ratio)
            if (not valid or two_core_frames / len(interval)
                    < minimum_two_core_fraction):
                continue

            moving_areas: list[int] = []
            for run in (departure["moving"], arrival["moving"]):
                for row in run["rows"]:
                    components, _ = ndi.label(
                        labels[int(row.frame)] == moving_owner, structure)
                    component_id = physical._component_at(
                        components, float(row.x), float(row.y),
                        search_radius=6)
                    if component_id > 0:
                        moving_areas.append(
                            int(np.count_nonzero(components == component_id)))
            if len(moving_areas) < minimum_pre + minimum_post:
                continue
            expected_moving_area = float(np.median(moving_areas))

            rows = [{
                "track_id": -3, "frame": frame,
                "x": path[frame]["x"], "y": path[frame]["y"],
                "radius_px": path[frame]["radius_px"],
                "accepted_owner": foreign_owner,
                "physically_visible": True,
                "state": "moving_owner_merge_bridge",
                "repair_owner": moving_owner,
                "bridge_source_track": int(departure["track"]),
                "bridge_successor_track": int(arrival["track"]),
                "bridge_resident_track": int(resident_track),
                "expected_repair_area_px": expected_moving_area,
                "expected_resident_area_px": ordinary_area,
            } for frame in interval]
            candidates.append({
                "rows": rows, "foreign_owner": foreign_owner,
                "source_track": int(departure["track"]),
                "successor_track": int(arrival["track"]),
                "resident_track": int(resident_track),
                "two_core_frames": int(two_core_frames),
                "expected_moving_area_px": expected_moving_area,
                "expected_resident_area_px": ordinary_area,
            })

    detail["moving_merge_bridge_candidates"] = len(candidates)
    if len(candidates) != 1:
        detail["moving_merge_bridge_reason"] = (
            "not_found" if not candidates else "ambiguous")
        return pd.DataFrame(), detail
    selected = candidates[0]
    detail.update({
        "moving_merge_bridge_frames": len(selected["rows"]),
        "moving_merge_bridge_reason": "eligible",
        "moving_merge_bridge_foreign_owner": selected["foreign_owner"],
        "moving_merge_bridge_source_track": selected["source_track"],
        "moving_merge_bridge_successor_track": selected["successor_track"],
        "moving_merge_bridge_resident_track": selected["resident_track"],
        "moving_merge_bridge_two_core_frames": selected["two_core_frames"],
        "moving_merge_bridge_expected_area_px":
            selected["expected_moving_area_px"],
        "moving_merge_bridge_expected_resident_area_px":
            selected["expected_resident_area_px"],
    })
    return pd.DataFrame(selected["rows"]), detail


def _anchor_seat_repair_rows(
        labels: np.ndarray, raw: np.ndarray, scored: pd.DataFrame,
        pair: tuple[int, int],
        dedicated_owner: int,
        params: dict) -> tuple[pd.DataFrame | None, dict]:
    """Follow the spatial seat that held a recurrent dedicated owner.

    Two references may briefly carry the same owner while sampling separate
    cells. The branch nearest the earlier dedicated owner's physical anchor is
    the only branch eligible to inherit that owner; the diverging branch keeps
    its current owner. This prevents a shared numeric owner from being treated
    as proof that the two physical cells are one lineage.
    """
    visible = scored[scored.physically_visible.astype(bool)].copy()
    groups = {
        track: visible[visible.track_id.astype(int) == track].sort_values(
            "frame") for track in pair
    }
    dedicated_tracks = [
        track for track, group in groups.items()
        if int((group.accepted_owner.astype(int) == dedicated_owner).sum())
        >= int(params.get("minimum_anchor_owner_frames", 8))]
    detail = {"anchor_source_track": 0, "anchor_successor_track": 0,
              "near_anchor_distance_radii": np.nan,
              "diverging_distance_radii": np.nan,
              "anchor_first_frame": -1, "anchor_last_frame": -1,
              "anchor_x": np.nan, "anchor_y": np.nan,
              "anchor_radius_px": np.nan,
              "anchor_component_area_px": np.nan,
              "reason": ""}
    if len(dedicated_tracks) != 1:
        return None, {**detail, "reason": "anchor_owner_track_not_unique"}
    source_track = int(dedicated_tracks[0])
    successor_track = int(pair[0] if pair[1] == source_track else pair[1])
    source = groups[source_track]
    successor = groups[successor_track]
    dedicated = source[
        source.accepted_owner.astype(int) == dedicated_owner].sort_values(
            "frame")
    first_anchor = int(dedicated.frame.min())
    last_anchor = int(dedicated.frame.max())
    detail.update({"anchor_source_track": source_track,
                   "anchor_successor_track": successor_track,
                   "anchor_first_frame": first_anchor,
                   "anchor_last_frame": last_anchor})
    if int(successor.frame.min()) <= last_anchor:
        return None, {**detail, "reason": "successor_precedes_anchor_release"}
    maximum_delay = max(1, int(round(len(scored.frame.unique()) * float(
        params.get("maximum_anchor_successor_delay_movie_fraction", 0.25)))))
    if int(successor.frame.min()) - last_anchor > maximum_delay:
        return None, {**detail, "reason": "successor_too_late_for_anchor"}

    anchor_x = float(dedicated.x.median())
    anchor_y = float(dedicated.y.median())
    anchor_radius = max(float(dedicated.radius_px.median()), 1.0)
    anchor_areas = []
    for row in dedicated.itertuples(index=False):
        component = invasion._component_near(
            labels[int(row.frame)], dedicated_owner, row)
        if component is not None:
            anchor_areas.append(int(np.count_nonzero(component)))
    if not anchor_areas:
        return None, {**detail, "reason": "anchor_component_area_missing"}
    detail.update({"anchor_x": anchor_x, "anchor_y": anchor_y,
                   "anchor_radius_px": anchor_radius,
                   "anchor_component_area_px": float(np.median(anchor_areas))})
    common = sorted(set(map(int, source.frame))
                    & set(map(int, successor.frame)))
    if len(common) < int(params.get("minimum_anchor_overlap_frames", 4)):
        return None, {**detail, "reason": "insufficient_anchor_overlap"}
    source_overlap = source[source.frame.astype(int).isin(common)]
    successor_overlap = successor[successor.frame.astype(int).isin(common)]
    moving_owners = sorted(set(map(int, source_overlap.loc[
        (source_overlap.accepted_owner.astype(int) > 0)
        & (source_overlap.accepted_owner.astype(int) != dedicated_owner),
        "accepted_owner"])))
    if len(moving_owners) != 1:
        return None, {**detail, "reason": "moving_owner_not_unique"}
    moving_owner = int(moving_owners[0])
    source_distance = float(np.median(np.hypot(
        source_overlap.x.astype(float) - anchor_x,
        source_overlap.y.astype(float) - anchor_y) / anchor_radius))
    successor_distance = float(np.median(np.hypot(
        successor_overlap.x.astype(float) - anchor_x,
        successor_overlap.y.astype(float) - anchor_y) / anchor_radius))
    detail.update({"near_anchor_distance_radii": successor_distance,
                   "diverging_distance_radii": source_distance,
                   "moving_owner": moving_owner})
    if successor_distance > float(params.get(
            "maximum_anchor_successor_distance_radii", 1.0)):
        return None, {**detail, "reason": "successor_not_on_anchor_seat"}
    if source_distance < float(params.get(
            "minimum_diverging_branch_distance_radii", 2.5)):
        return None, {**detail, "reason": "source_did_not_leave_anchor_seat"}
    if source_distance - successor_distance < float(params.get(
            "minimum_anchor_branch_distance_margin_radii", 1.5)):
        return None, {**detail, "reason": "anchor_branch_not_unambiguous"}
    successor_distance_all = float(np.median(np.hypot(
        successor.x.astype(float) - anchor_x,
        successor.y.astype(float) - anchor_y) / anchor_radius))
    if successor_distance_all > float(params.get(
            "maximum_successor_lifetime_anchor_distance_radii", 1.0)):
        return None, {**detail, "reason": "successor_does_not_hold_anchor_seat"}

    bridge_rows: list[dict] = []
    first_successor = int(successor.frame.min())
    maximum_bridge_distance = float(params.get(
        "maximum_anchor_bridge_distance_radii", 1.0)) * anchor_radius
    minimum_bridge_area_ratio = float(params.get(
        "minimum_anchor_bridge_component_area_ratio", 3.0))
    yy, xx = np.indices(labels.shape[1:])
    anchor_distance = np.hypot(xx - anchor_x, yy - anchor_y)
    for frame in range(last_anchor + 1, first_successor):
        plane = labels[frame]
        foreground = plane > 0
        if not np.any(foreground):
            return None, {**detail, "reason": "anchor_bridge_foreground_missing"}
        nearest_index = int(np.argmin(np.where(
            foreground, anchor_distance, np.inf)))
        nearest_y, nearest_x = np.unravel_index(nearest_index, plane.shape)
        distance = float(anchor_distance[nearest_y, nearest_x])
        owner = int(plane[nearest_y, nearest_x])
        if distance > maximum_bridge_distance or owner != moving_owner:
            return None, {**detail, "reason": "anchor_bridge_owner_not_continuous"}
        components, _ = ndi.label(
            plane == owner, np.ones((3, 3), np.uint8))
        component_id = int(components[nearest_y, nearest_x])
        component_area = int(np.count_nonzero(components == component_id))
        if component_area / max(float(np.median(anchor_areas)), 1.0) \
                < minimum_bridge_area_ratio:
            return None, {**detail, "reason": "anchor_bridge_host_not_merged"}
        bridge_rows.append({
            "track_id": -1, "frame": frame,
            "x": anchor_x, "y": anchor_y, "radius_px": anchor_radius,
            "accepted_owner": owner, "physically_visible": True,
            "state": "field_derived_anchor_bridge",
        })
    detail["anchor_bridge_frames"] = len(bridge_rows)

    ownerless_successor_rows: list[dict] = []
    maximum_ownerless_distance = float(params.get(
        "maximum_ownerless_anchor_distance_radii", 2.5)) * anchor_radius
    for row in successor[
            successor.accepted_owner.astype(int) <= 0].itertuples(index=False):
        frame = int(row.frame)
        plane = labels[frame]
        foreground = plane > 0
        if not np.any(foreground):
            continue
        nearest_index = int(np.argmin(np.where(
            foreground, anchor_distance, np.inf)))
        nearest_y, nearest_x = np.unravel_index(nearest_index, plane.shape)
        distance = float(anchor_distance[nearest_y, nearest_x])
        owner = int(plane[nearest_y, nearest_x])
        if distance > maximum_ownerless_distance or owner != moving_owner:
            continue
        components, _ = ndi.label(
            plane == owner, np.ones((3, 3), np.uint8))
        component_id = int(components[nearest_y, nearest_x])
        component_area = int(np.count_nonzero(components == component_id))
        if component_area / max(float(np.median(anchor_areas)), 1.0) \
                < minimum_bridge_area_ratio:
            continue
        ownerless_successor_rows.append({
            "track_id": -2, "frame": frame,
            "x": anchor_x, "y": anchor_y, "radius_px": anchor_radius,
            "accepted_owner": owner, "physically_visible": True,
            "state": "field_derived_ownerless_anchor_bridge",
        })
    detail["ownerless_anchor_bridge_frames"] = len(
        ownerless_successor_rows)

    moving_bridge, moving_bridge_detail = _moving_owner_merge_bridge_rows(
        labels, raw, scored, moving_owner, pair, params)
    detail.update(moving_bridge_detail)

    bracket = source[
        (source.frame.astype(int) >= first_anchor)
        & (source.frame.astype(int) <= last_anchor)
        & (source.accepted_owner.astype(int) > 0)
        & (source.accepted_owner.astype(int) != dedicated_owner)]
    successor_changes = successor[
        (successor.accepted_owner.astype(int) > 0)
        & (successor.accepted_owner.astype(int) != dedicated_owner)]
    repairs = pd.concat([
        bracket, pd.DataFrame(bridge_rows),
        pd.DataFrame(ownerless_successor_rows), successor_changes,
        moving_bridge,
    ], ignore_index=True) \
        .sort_values(["frame", "track_id"])
    if not len(repairs):
        return None, {**detail, "reason": "no_anchor_owner_changes"}
    return repairs, {**detail, "reason": "eligible_anchor_seat_branch"}


def _future_ownerless_recipient(
        scored: pd.DataFrame, pair: tuple[int, int], owner: int,
        first_frame: int, last_frame: int, params: dict):
    """Find one ownerless reference that soon, independently, gains owner.

    This is an evidence rule, not an identity lookup: every reference outside
    the repaired pair is tested and the result is usable only when unique.
    """
    minimum_coverage = float(params.get(
        "minimum_future_recipient_visible_fraction", 0.70))
    minimum_ownerless = float(params.get(
        "minimum_future_recipient_ownerless_fraction", 0.70))
    future_window = int(params.get("future_recipient_window_frames", 5))
    minimum_future_support = int(params.get(
        "minimum_future_owner_support_frames", 2))
    duration = last_frame - first_frame + 1
    matches: list[tuple[int, pd.DataFrame, int]] = []
    for track_id, group in scored.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id in pair:
            continue
        visible = group[group.physically_visible.astype(bool)].sort_values(
            "frame")
        during = visible[(visible.frame.astype(int) >= first_frame)
                         & (visible.frame.astype(int) <= last_frame)]
        coverage = len(during) / max(duration, 1)
        ownerless_fraction = float(
            (during.accepted_owner.astype(int) == 0).mean()) \
            if len(during) else 0.0
        future = visible[
            (visible.frame.astype(int) > last_frame)
            & (visible.frame.astype(int) <= last_frame + future_window)
            & (visible.accepted_owner.astype(int) == owner)]
        if (coverage >= minimum_coverage
                and ownerless_fraction >= minimum_ownerless
                and len(future) >= minimum_future_support):
            matches.append((track_id, during, int(len(future))))
    return matches[0] if len(matches) == 1 else None


def _recent_displaced_track(
        scored: pd.DataFrame, pair: tuple[int, int], owner: int,
        first_frame: int, last_frame: int, params: dict):
    """Find one track that held owner up to the displacement boundary."""
    minimum_prior = int(params.get("minimum_recent_owner_frames", 5))
    maximum_tail = int(params.get("maximum_recent_displaced_tail_frames", 3))
    matches: list[tuple[int, pd.DataFrame]] = []
    for track_id, group in scored.groupby("track_id", sort=True):
        track_id = int(track_id)
        if track_id in pair:
            continue
        visible = group[group.physically_visible.astype(bool)].sort_values(
            "frame")
        before = visible[visible.frame.astype(int) < first_frame]
        if not len(before) or int(before.frame.max()) != first_frame - 1:
            continue
        prior = 0
        expected = first_frame - 1
        for row in reversed(list(before.itertuples(index=False))):
            if (int(row.frame) != expected
                    or int(row.accepted_owner) != owner):
                break
            prior += 1
            expected -= 1
        if prior < minimum_prior:
            continue
        after = visible[(visible.frame.astype(int) >= first_frame)
                        & (visible.frame.astype(int) <= last_frame)]
        tail_rows = []
        expected = first_frame
        tail_owner = 0
        for row in after.itertuples(index=False):
            if int(row.frame) != expected:
                break
            row_owner = int(row.accepted_owner)
            if row_owner <= 0 or row_owner == owner:
                break
            if not tail_rows:
                tail_owner = row_owner
            if row_owner != tail_owner or len(tail_rows) >= maximum_tail:
                break
            tail_rows.append(row)
            expected += 1
        if tail_rows:
            matches.append((track_id, pd.DataFrame(
                [row._asdict() for row in tail_rows])))
    return matches[0] if len(matches) == 1 else None


def _restore_displaced_seats(
        trial: np.ndarray, trial_unclaimed: np.ndarray,
        baseline_unclaimed: np.ndarray, raw: np.ndarray,
        scored: pd.DataFrame, pair: tuple[int, int],
        repair_rows: pd.DataFrame, thresholds: pd.DataFrame,
        params: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Restore displaced owners only where independent evidence is unique."""
    restored = trial.copy()
    restored_unclaimed = trial_unclaimed.copy()
    audit: list[dict] = []
    threshold_by_frame = thresholds.set_index("frame")
    maximum_core_ratio = float(params.get(
        "maximum_restored_core_area_radius_ratio", 2.0))

    for run in _contiguous_owner_runs(repair_rows):
        owner = int(run["owner"])
        first_frame = int(run["first_frame"])
        last_frame = int(run["last_frame"])

        # The ledger is direct provenance evidence. Reclaim it only while the
        # displaced identity has no assigned seat anywhere in that frame.
        ledger_pixels = 0
        ledger_frames = 0
        for frame in range(len(restored)):
            if np.any(restored[frame] == owner):
                continue
            eligible_ledger = ((restored[frame] == 0)
                               & (restored_unclaimed[frame] == owner)
                               & (baseline_unclaimed[frame] == owner)
                               & (raw[frame] > 0))
            components, component_count = ndi.label(
                eligible_ledger, np.ones((3, 3), np.uint8))
            reclaim = np.zeros_like(eligible_ledger)
            if component_count:
                sizes = np.bincount(components.ravel())[1:]
                reclaim = components == int(np.argmax(sizes) + 1)
            count = int(np.count_nonzero(reclaim))
            if count:
                restored[frame][reclaim] = owner
                restored_unclaimed[frame][reclaim] = 0
                ledger_pixels += count
                ledger_frames += 1
        audit.append({
            "restoration_type": "same_original_owner_ledger",
            "owner": owner, "first_frame": first_frame,
            "last_frame": last_frame, "physical_track": 0,
            "changed_pixels": ledger_pixels,
            "changed_frames": ledger_frames,
            "outcome": "applied" if ledger_frames else "not_found",
        })

        recent = _recent_displaced_track(
            scored, pair, owner, first_frame, last_frame, params)
        recent_pixels = 0
        recent_frames = 0
        recent_track = 0
        if recent is not None:
            recent_track, rows = recent
            for row in rows.itertuples(index=False):
                frame = int(row.frame)
                if np.any(restored[frame] == owner):
                    continue
                current_owner = int(row.accepted_owner)
                component = invasion._component_near(
                    restored[frame], current_owner, row)
                if component is None or np.any(restored_unclaimed[frame][component] > 0):
                    continue
                count = int(np.count_nonzero(component))
                if count:
                    restored[frame][component] = owner
                    recent_pixels += count
                    recent_frames += 1
        audit.append({
            "restoration_type": "unique_recent_displaced_track",
            "owner": owner, "first_frame": first_frame,
            "last_frame": last_frame, "physical_track": recent_track,
            "changed_pixels": recent_pixels,
            "changed_frames": recent_frames,
            "outcome": "applied" if recent_frames else "not_found",
        })

        future = None
        if bool(params.get("enable_future_ownerless_recovery", True)):
            future = _future_ownerless_recipient(
                scored, pair, owner, first_frame, last_frame, params)
        future_pixels = 0
        future_frames = 0
        future_track = 0
        if future is not None:
            future_track, rows, _ = future
            for row in rows.itertuples(index=False):
                frame = int(row.frame)
                if np.any(restored[frame] == owner):
                    continue
                core, region, detail = ownerless._connected_core(
                    raw[frame], row,
                    float(threshold_by_frame.loc[frame, "weak_threshold"]),
                    params)
                expected_area = np.pi * max(float(row.radius_px), 1.0) ** 2
                ratio = detail["core_area_px"] / max(expected_area, 1.0)
                if not detail["core_area_px"] or ratio > maximum_core_ratio:
                    continue
                local_labels = restored[frame][region]
                local_unclaimed = restored_unclaimed[frame][region]
                current_owner = int(row.accepted_owner)
                # A later owner match supports filling an ownerless seat.  It
                # does not, by itself, justify cutting pixels from another
                # assigned identity on an earlier frame.
                if current_owner != 0:
                    continue
                permitted = core & (local_unclaimed == 0) & (local_labels == 0)
                count = int(np.count_nonzero(permitted))
                if count:
                    local_labels[permitted] = owner
                    future_pixels += count
                    future_frames += 1
        audit.append({
            "restoration_type": "unique_future_ownerless_recipient",
            "owner": owner, "first_frame": first_frame,
            "last_frame": last_frame, "physical_track": future_track,
            "changed_pixels": future_pixels,
            "changed_frames": future_frames,
            "outcome": "applied" if future_frames else "not_found",
        })
    return restored, restored_unclaimed, audit


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            thresholds: pd.DataFrame | None = None,
            precomputed: tuple[pd.DataFrame, pd.DataFrame] | None = None,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                       pd.DataFrame]:
    """Apply only complete, atomic virtual-lineage repairs."""
    if precomputed is None:
        proposals, scored = discover(labels, points, params)
    else:
        assert_target_free(params)
        proposals, scored = precomputed
    visible = scored[scored.physically_visible.astype(bool)].copy()
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    canonical = physical.canonical_owners(
        scored,
        int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_canonical_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    # Assigned identities and original-owner ledger IDs share one namespace.
    # Allocate above both so a new orphan ledger ID cannot collide with an
    # active identity or depend on which identities happen to be unclaimed.
    next_unclaimed = max(int(np.max(labels)), int(np.max(unclaimed))) + 1
    occupied = np.zeros_like(labels, bool)
    applications: list[dict] = []
    frame_audit: list[dict] = []

    for proposal in proposals[proposals.eligible.astype(bool)].itertuples(
            index=False):
        pair = _parse_pair(proposal)
        application_rule = str(params.get(
            "application_rule", "retained_canonical_host"))
        branch_detail: dict = {}
        if application_rule in {
                "anchor_seat_branch", "stationary_anchor_merge_residency"}:
            repair_rows, branch_detail = _anchor_seat_repair_rows(
                labels, raw, scored, pair,
                int(proposal.dedicated_owner), params)
            if repair_rows is None:
                applications.append({
                    "proposal_id": proposal.proposal_id,
                    "track_pair": proposal.track_pair,
                    "dedicated_owner": int(proposal.dedicated_owner),
                    "outcome": "rejected_application",
                    "reason": branch_detail["reason"],
                    "changed_pixels": 0, "changed_frames": 0,
                    **branch_detail,
                })
                continue
        else:
            virtual = _virtual_rows(scored, pair)
            repair_rows = virtual[
                (virtual.frame.astype(int)
                 >= int(proposal.first_dedicated_frame))
                & (virtual.accepted_owner.astype(int) > 0)
                & (virtual.accepted_owner.astype(int)
                   != int(proposal.dedicated_owner))
            ].sort_values("frame")
        trial = candidate.copy()
        trial_unclaimed = candidate_unclaimed.copy()
        prepared: list[tuple[int, np.ndarray, np.ndarray, dict]] = []
        refusal = ""
        for row in repair_rows.itertuples(index=False):
            frame = int(row.frame)
            if application_rule == "retained_canonical_host":
                frame_labels, frame_unclaimed, detail = relay._prepare_frame(
                    trial, trial_unclaimed, labels, unclaimed, raw, by_frame,
                    canonical, row, int(proposal.dedicated_owner),
                    next_unclaimed, params)
            elif application_rule in {
                    "all_raw_companion_markers", "anchor_seat_branch",
                    "stationary_anchor_merge_residency"}:
                state = str(getattr(row, "state", ""))
                moving_bridge = state == "moving_owner_merge_bridge"
                if moving_bridge:
                    companion_exclusions = tuple(sorted(set(pair) | {
                        int(row.bridge_source_track),
                        int(row.bridge_successor_track)}))
                else:
                    companion_exclusions = pair \
                        if (application_rule == "all_raw_companion_markers"
                            or (application_rule ==
                                "stationary_anchor_merge_residency"
                                and int(row.track_id) < 0)) \
                        else (int(row.track_id),)
                row_repair_owner = getattr(row, "repair_owner", np.nan)
                repair_owner = (int(row_repair_owner)
                                if pd.notna(row_repair_owner)
                                and int(row_repair_owner) > 0
                                else int(proposal.dedicated_owner))
                frame_labels, frame_unclaimed, detail = \
                    _prepare_shared_component_frame(
                        trial, trial_unclaimed, labels, unclaimed, raw,
                        by_frame, row, companion_exclusions,
                        repair_owner,
                        next_unclaimed, params,
                        (branch_detail if application_rule ==
                         "stationary_anchor_merge_residency"
                         and not moving_bridge else None))
            elif application_rule == "ledger_and_future_seat_restoration":
                frame_labels, frame_unclaimed, detail = \
                    _prepare_shared_component_frame(
                        trial, trial_unclaimed, labels, unclaimed, raw,
                        by_frame, row, pair, int(proposal.dedicated_owner),
                        next_unclaimed, params)
            else:
                raise ValueError(f"unknown application rule: {application_rule}")
            detail.update({
                "proposal_id": proposal.proposal_id,
                "track_pair": proposal.track_pair,
                "physical_track": int(row.track_id),
                "dedicated_owner": int(proposal.dedicated_owner),
                "repair_owner": int(
                    getattr(row, "repair_owner", proposal.dedicated_owner))
                    if pd.notna(getattr(
                        row, "repair_owner", proposal.dedicated_owner))
                    else int(proposal.dedicated_owner),
                "applied": False,
            })
            if frame_labels is None or frame_unclaimed is None:
                refusal = str(detail["reason"])
                frame_audit.append(detail)
                break
            change = frame_labels != trial[frame]
            if np.any(occupied[frame] & change):
                refusal = "overlapping_proposal"
                frame_audit.append({**detail, "reason": refusal})
                break
            trial[frame] = frame_labels
            trial_unclaimed[frame] = frame_unclaimed
            prepared.append((frame, frame_labels, frame_unclaimed, detail))

        if (not refusal
                and application_rule == "ledger_and_future_seat_restoration"):
            if thresholds is None:
                refusal = "frame_thresholds_required"
            else:
                trial, trial_unclaimed, restoration_audit = \
                    _restore_displaced_seats(
                        trial, trial_unclaimed, unclaimed, raw, scored, pair,
                        repair_rows, thresholds, params)
                for detail in restoration_audit:
                    frame_audit.append({
                        **detail, "proposal_id": proposal.proposal_id,
                        "track_pair": proposal.track_pair,
                        "dedicated_owner": int(proposal.dedicated_owner),
                        "applied": detail["outcome"] == "applied",
                        "reason": detail["outcome"],
                    })

        if not refusal:
            before_union = (labels > 0) | (unclaimed > 0)
            after_union = (trial > 0) | (trial_unclaimed > 0)
            removed = before_union & ~after_union
            additions = ~before_union & after_union
            changed_existing_ledger = ((unclaimed > 0)
                                       & (trial_unclaimed != unclaimed))
            valid_reclaim = (changed_existing_ledger
                             & (trial_unclaimed == 0)
                             & (trial == unclaimed))
            if np.any(removed):
                refusal = "foreground_or_ledger_removed"
            elif np.any(additions & (raw <= 0)):
                refusal = "zero_signal_foreground_added"
            elif np.any(changed_existing_ledger & ~valid_reclaim):
                refusal = "existing_unclaimed_changed_without_reclaim"
            elif set(map(int, np.unique(trial))) != set(map(int, np.unique(labels))):
                refusal = "active_identity_set_changed"
            elif relay._new_duplicate_components(labels, trial):
                refusal = "new_duplicate_components"

        if refusal:
            for _, _, _, detail in prepared:
                frame_audit.append({**detail, "reason": "cascade_atomic_refusal"})
            applications.append({
                "proposal_id": proposal.proposal_id,
                "track_pair": proposal.track_pair,
                "dedicated_owner": int(proposal.dedicated_owner),
                "outcome": "rejected_application",
                "reason": refusal,
                "changed_pixels": 0,
                "changed_frames": 0,
            })
            continue

        change = trial != candidate
        candidate = trial
        candidate_unclaimed = trial_unclaimed
        occupied |= change
        for _, _, _, detail in prepared:
            frame_audit.append({**detail, "applied": True, "reason": "applied"})
        applications.append({
            "proposal_id": proposal.proposal_id,
            "track_pair": proposal.track_pair,
            "dedicated_owner": int(proposal.dedicated_owner),
            "outcome": "applied",
            "reason": "complete_cross_reference_relay",
            "changed_pixels": int(np.count_nonzero(change)),
            "changed_frames": int(np.count_nonzero(np.any(
                change, axis=(1, 2)))),
            **branch_detail,
        })
        next_unclaimed += 1

    return (candidate, candidate_unclaimed, proposals,
            pd.DataFrame(applications), pd.DataFrame(frame_audit))


def summarize(labels: np.ndarray, unclaimed: np.ndarray,
              candidate: np.ndarray, candidate_unclaimed: np.ndarray,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              raw: np.ndarray | None = None) -> dict:
    changed = candidate != labels
    before_union = (labels > 0) | (unclaimed > 0)
    after_union = (candidate > 0) | (candidate_unclaimed > 0)
    additions = ~before_union & after_union
    raw_supported = int(np.count_nonzero(additions)) if raw is None else \
        int(np.count_nonzero(additions & (raw > 0)))
    zero_signal = 0 if raw is None else int(np.count_nonzero(
        additions & (raw <= 0)))
    return {
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "region_target_count": 0,
        "review_case_target_count": 0,
        "audited_reference_pairs": int(len(proposals)),
        "eligible_reference_relays": int(proposals.eligible.sum())
        if len(proposals) else 0,
        "applied_reference_relays": int(applications.outcome.eq("applied").sum())
        if len(applications) else 0,
        "changed_label_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "foreground_ledger_changed_pixels": int(np.count_nonzero(
            after_union != before_union)),
        "foreground_removed_pixels": int(np.count_nonzero(
            before_union & ~after_union)),
        "raw_supported_foreground_additions": raw_supported,
        "zero_signal_additions": zero_signal,
        "existing_unclaimed_changed_pixels": int(np.count_nonzero(
            (unclaimed > 0) & (candidate_unclaimed != unclaimed))),
        "active_identity_set_preserved": set(map(int, np.unique(candidate)))
        == set(map(int, np.unique(labels))),
        "new_duplicate_components": relay._new_duplicate_components(
            labels, candidate),
    }
