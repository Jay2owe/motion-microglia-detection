"""Repair terminal three-seat identity diversion without named targets.

The detector looks for a recording-end topology in which an established
identity jumps to a newly born physical reference even though its original
track remains visible, and then invades a neighbouring established track. The
original owner is restored while the nearby reference is retained as a
projection of the established resident; no new biological identity is made.
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
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN = (
    "identity_target", "track_target", "frame_target", "coordinate",
    "event_target", "review_case", "case_id", "region_target",
    "forced_identity", "forced_track", "forced_frame", "include_track",
    "exclude_track", "include_identity", "exclude_identity",
)
PROPOSAL_COLUMNS = (
    "proposal_id", "source_track", "newborn_track", "resident_track",
    "boundary_frame", "source_owner", "source_replacement_owner",
    "resident_owner", "resident_takeover_start", "source_anchor_frames",
    "resident_anchor_frames", "false_transfer_jump_sum_radii",
    "newborn_resident_distance_sum_radii", "pair_frames_observed",
    "qualified_two_core_frames", "median_pair_distance_sum_radii",
    "source_last_frame", "newborn_last_frame", "resident_last_frame",
    "eligible", "reason",
)
APPLICATION_COLUMNS = (
    "proposal_id", "role", "frame", "source_owner", "restored_owner",
    "component_pixels", "applied", "reason",
)


def assert_target_free(params: dict) -> None:
    supplied = [
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN)
    ]
    if supplied:
        raise ValueError(
            "terminal diversion recovery received forbidden targets: "
            + ", ".join(sorted(supplied)))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal diversion recovery must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    rows = group[
        group.physically_visible.astype(bool)
        & group.candidate_owner.astype(int).gt(0)
    ].sort_values("frame")
    runs: list[dict] = []
    for row in rows.itertuples(index=False):
        owner, frame = int(row.candidate_owner), int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame == int(runs[-1]["last"]) + 1):
            runs[-1]["last"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "rows": [row]})
    for run in runs:
        run["frames"] = len(run["rows"])
    return runs


def _owner_run_ending(group: pd.DataFrame, owner: int, frame: int) -> dict | None:
    candidates = [
        run for run in _positive_runs(group)
        if int(run["owner"]) == int(owner) and int(run["last"]) == int(frame)
    ]
    return candidates[-1] if candidates else None


def _owner_run_starting(group: pd.DataFrame, owner: int,
                        first: int, last: int) -> dict | None:
    candidates = [
        run for run in _positive_runs(group)
        if int(run["owner"]) == int(owner)
        and int(first) <= int(run["first"]) <= int(last)
    ]
    return candidates[0] if candidates else None


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return distance / scale


def _pair_raw_support(raw: np.ndarray, left: pd.DataFrame,
                      right: pd.DataFrame, first: int, params: dict) -> dict:
    left_index = left.set_index("frame", drop=False)
    right_index = right.set_index("frame", drop=False)
    maximum = float(params.get("maximum_pair_distance_sum_radii", 2.5))
    maximum_valley = float(params.get("maximum_pair_valley_ratio", 0.70))
    minimum_balance = float(params.get("minimum_pair_endpoint_balance", 0.10))
    sigma = float(params.get("pair_profile_sigma_px", 1.0))
    qualified = 0
    observed = 0
    ratios: list[float] = []
    for frame in sorted(set(left_index.index) & set(right_index.index)):
        frame = int(frame)
        if frame < int(first):
            continue
        a, b = left_index.loc[frame], right_index.loc[frame]
        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            continue
        if not (_truth(a.physically_visible) and _truth(b.physically_visible)):
            continue
        distance = _scaled_distance(a, b)
        if distance > maximum:
            continue
        smooth = ndi.gaussian_filter(raw[frame].astype(np.float32), sigma)
        samples = max(5, 2 * int(math.ceil(float(np.hypot(
            float(a.x) - float(b.x), float(a.y) - float(b.y))))) + 1)
        xs = np.linspace(float(a.x), float(b.x), samples)
        ys = np.linspace(float(a.y), float(b.y), samples)
        values = ndi.map_coordinates(smooth, [ys, xs], order=1,
                                     mode="nearest")
        if len(values) < 3:
            continue
        low, high = float(min(values[0], values[-1])), \
            float(max(values[0], values[-1]))
        valley = float(np.min(values[1:-1]) / max(low, 1.0))
        balance = low / max(high, 1.0)
        observed += 1
        ratios.append(distance)
        if (valley <= maximum_valley and balance >= minimum_balance
                and _truth(a.strong) and _truth(b.strong)):
            qualified += 1
    return {
        "pair_frames_observed": observed,
        "qualified_two_core_frames": qualified,
        "median_pair_distance_sum_radii": (
            float(np.median(ratios)) if ratios else float("nan")),
    }


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> pd.DataFrame:
    """Enumerate every field-wide terminal diversion triple."""
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    attached["physically_visible"] = attached.state.astype(str).isin(
        VISIBLE_STATES)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    frame_count = len(labels)
    boundary_first = int(math.floor(frame_count * (
        1.0 - float(params.get("terminal_boundary_movie_fraction", 0.12)))))
    minimum_anchor = max(2, int(math.ceil(
        frame_count * float(params.get("minimum_anchor_movie_fraction", 0.08)))))
    maximum_newborn_age = max(0, int(math.ceil(
        frame_count * float(params.get("maximum_newborn_age_movie_fraction", 0.02)))))
    maximum_takeover_delay = max(1, int(math.ceil(
        frame_count * float(params.get("maximum_takeover_delay_movie_fraction", 0.03)))))
    terminal_slack = max(1, int(math.ceil(
        frame_count * float(params.get("maximum_terminal_slack_movie_fraction", 0.04)))))
    minimum_jump = float(params.get("minimum_false_transfer_jump_sum_radii", 3.0))
    maximum_neighbour = float(params.get("maximum_neighbour_distance_sum_radii", 2.0))
    minimum_two_core = int(params.get("minimum_qualified_two_core_frames", 2))
    rows: list[dict] = []

    for source_track, source_group in groups.items():
        source_runs = _positive_runs(source_group)
        for source_run in source_runs:
            owner_a = int(source_run["owner"])
            boundary = int(source_run["last"]) + 1
            if boundary < boundary_first or int(source_run["frames"]) < minimum_anchor:
                continue
            source_now = _point(
                source_group.set_index(["track_id", "frame"], drop=False),
                source_track, boundary)
            if source_now is None or not _truth(source_now.physically_visible):
                continue
            if int(source_now.candidate_owner) == owner_a:
                continue
            source_before = source_run["rows"][-1]

            receiver_options = []
            for newborn_track, newborn_group in groups.items():
                if newborn_track == source_track:
                    continue
                visible_newborn = newborn_group[
                    newborn_group.physically_visible.astype(bool)]
                if visible_newborn.empty:
                    continue
                born = int(visible_newborn.frame.min())
                if born < boundary - maximum_newborn_age or born > boundary:
                    continue
                newborn_now_rows = newborn_group[
                    newborn_group.frame.astype(int).eq(boundary)]
                if len(newborn_now_rows) != 1:
                    continue
                newborn_now = newborn_now_rows.iloc[0]
                if (not _truth(newborn_now.physically_visible)
                        or int(newborn_now.candidate_owner) != owner_a):
                    continue
                jump = _scaled_distance(source_before, newborn_now)
                if jump >= minimum_jump:
                    receiver_options.append((newborn_track, newborn_group,
                                             newborn_now, jump))

            for newborn_track, newborn_group, newborn_now, jump in receiver_options:
                resident_options = []
                for resident_track, resident_group in groups.items():
                    if resident_track in {source_track, newborn_track}:
                        continue
                    current = resident_group[
                        resident_group.frame.astype(int).eq(boundary)]
                    if len(current) != 1:
                        continue
                    resident_now = current.iloc[0]
                    owner_b = int(resident_now.candidate_owner)
                    if (owner_b <= 0 or owner_b == owner_a
                            or not _truth(resident_now.physically_visible)):
                        continue
                    anchor = _owner_run_ending(
                        resident_group, owner_b, boundary)
                    if anchor is None or int(anchor["frames"]) < minimum_anchor:
                        continue
                    neighbour = _scaled_distance(newborn_now, resident_now)
                    if neighbour > maximum_neighbour:
                        continue
                    invaded = _owner_run_starting(
                        resident_group, owner_a, boundary + 1,
                        boundary + maximum_takeover_delay)
                    if invaded is None:
                        continue
                    support = _pair_raw_support(
                        raw, newborn_group, resident_group, boundary, params)
                    resident_options.append((
                        resident_track, resident_group, resident_now, owner_b,
                        anchor, invaded, neighbour, support))

                for (resident_track, resident_group, _resident_now, owner_b,
                     anchor, invaded, neighbour, support) in resident_options:
                    reasons = []
                    track_ends = [
                        int(group.frame.max()) for group in (
                            source_group, newborn_group, resident_group)]
                    if min(track_ends) < frame_count - 1 - terminal_slack:
                        reasons.append("one_seat_not_terminal")
                    if int(support["qualified_two_core_frames"]) < minimum_two_core:
                        reasons.append("insufficient_two_core_raw_support")
                    if len(receiver_options) != 1:
                        reasons.append("ambiguous_newborn_receiver")
                    if len(resident_options) != 1:
                        reasons.append("ambiguous_invaded_resident")
                    rows.append({
                        "source_track": source_track,
                        "newborn_track": newborn_track,
                        "resident_track": resident_track,
                        "boundary_frame": boundary,
                        "source_owner": owner_a,
                        "source_replacement_owner": int(source_now.candidate_owner),
                        "resident_owner": owner_b,
                        "resident_takeover_start": int(invaded["first"]),
                        "source_anchor_frames": int(source_run["frames"]),
                        "resident_anchor_frames": int(anchor["frames"]),
                        "false_transfer_jump_sum_radii": float(jump),
                        "newborn_resident_distance_sum_radii": float(neighbour),
                        **support,
                        "source_last_frame": track_ends[0],
                        "newborn_last_frame": track_ends[1],
                        "resident_last_frame": track_ends[2],
                        "eligible": not reasons,
                        "reason": "eligible" if not reasons else "|".join(reasons),
                    })
    result = pd.DataFrame(rows)
    if len(result):
        result.insert(0, "proposal_id", [
            f"TDT{index:04d}" for index in range(1, len(result) + 1)])
    return result.reindex(columns=PROPOSAL_COLUMNS)


def _component_at_foreground(labels: np.ndarray, point) -> np.ndarray | None:
    owner = int(point.candidate_owner)
    if owner <= 0:
        return None
    return bounded._component_at(labels, owner, point)


def _core_tracks(mask: np.ndarray, frame_points: pd.DataFrame) -> list[int]:
    result = []
    for point in frame_points.itertuples(index=False):
        if not _truth(point.physically_visible):
            continue
        radius = max(2.0, float(point.radius_px) * 0.5)
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
                <= radius ** 2)
        if np.any(mask & disk):
            result.append(int(point.track_id))
    return sorted(set(result))


def _partition(mask: np.ndarray, raw: np.ndarray, left, right,
               params: dict) -> tuple[np.ndarray, np.ndarray] | None:
    markers = np.zeros(mask.shape, np.uint8)
    seeds = []
    for point in (left, right):
        y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
        if not mask[y, x]:
            yy, xx = np.nonzero(mask)
            if not len(xx):
                return None
            nearest = int(np.argmin((xx - float(point.x)) ** 2
                                    + (yy - float(point.y)) ** 2))
            y, x = int(yy[nearest]), int(xx[nearest])
        seeds.append((y, x))
    if seeds[0] == seeds[1]:
        return None
    markers[seeds[0]], markers[seeds[1]] = 1, 2
    smooth = ndi.gaussian_filter(
        raw.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    split = watershed(-smooth, markers=markers, mask=mask,
                      connectivity=STRUCTURE.astype(bool))
    left_mask, right_mask = split == 1, split == 2
    minimum = int(params.get("minimum_partition_pixels", 5))
    if int(left_mask.sum()) < minimum or int(right_mask.sum()) < minimum:
        # A flat seeded watershed gives connected geodesic basins when one
        # raw-intensity peak is too dim to win a meaningful catchment.
        split = watershed(
            np.zeros(mask.shape, np.float32), markers=markers, mask=mask,
            connectivity=STRUCTURE.astype(bool))
        left_mask, right_mask = split == 1, split == 2
    if (int(left_mask.sum()) < minimum or int(right_mask.sum()) < minimum
            or not np.array_equal(left_mask | right_mask, mask)):
        return None
    return left_mask, right_mask


def _track_basin(mask: np.ndarray, raw: np.ndarray, target,
                 frame_points: pd.DataFrame, params: dict) -> np.ndarray | None:
    """Return the target core's raw watershed basin inside a shared owner."""
    occupants = []
    for point in frame_points.itertuples(index=False):
        if not _truth(point.physically_visible):
            continue
        radius = max(2.0, 0.5 * float(point.radius_px))
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
                <= radius ** 2)
        if np.any(mask & disk):
            occupants.append(point)
    if not occupants:
        return None
    occupants.sort(key=lambda point: (
        int(point.track_id) != int(target.track_id), int(point.track_id)))
    markers = np.zeros(mask.shape, np.int32)
    used = set()
    target_marker = 0
    yy_mask, xx_mask = np.nonzero(mask)
    for marker, point in enumerate(occupants, start=1):
        y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
        if not mask[y, x] or (y, x) in used:
            order = np.argsort(
                (xx_mask - float(point.x)) ** 2
                + (yy_mask - float(point.y)) ** 2)
            chosen = next(((int(yy_mask[i]), int(xx_mask[i])) for i in order
                           if (int(yy_mask[i]), int(xx_mask[i])) not in used),
                          None)
            if chosen is None:
                return None
            y, x = chosen
        used.add((y, x))
        markers[y, x] = marker
        if int(point.track_id) == int(target.track_id):
            target_marker = marker
    if target_marker <= 0:
        return None
    if len(occupants) == 1:
        return mask
    smooth = ndi.gaussian_filter(
        raw.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    split = watershed(-smooth, markers=markers, mask=mask,
                      connectivity=STRUCTURE.astype(bool))
    basin = split == target_marker
    expected = max(math.pi * float(target.radius_px) ** 2, 1.0)
    minimum_ratio = float(params.get("minimum_track_basin_area_ratio", 0.20))
    maximum_ratio = float(params.get("maximum_track_basin_area_ratio", 3.25))
    minimum_pixels = int(params.get("minimum_partition_pixels", 5))
    ratio = float(basin.sum()) / expected
    if int(basin.sum()) < minimum_pixels or ratio < minimum_ratio:
        # A dim target may receive an unrealistically tiny intensity basin.
        # Use a radius-normalised Voronoi partition of the same accepted
        # component; this changes no foreground and remains field-derived.
        yy, xx = np.nonzero(mask)
        scores = np.vstack([
            ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2)
            / max(float(point.radius_px), 1.0) ** 2
            for point in occupants])
        winner = np.argmin(scores, axis=0)
        target_index = next(
            index for index, point in enumerate(occupants)
            if int(point.track_id) == int(target.track_id))
        basin = np.zeros(mask.shape, dtype=bool)
        basin[yy[winner == target_index], xx[winner == target_index]] = True
        ratio = float(basin.sum()) / expected
    if (int(basin.sum()) < minimum_pixels
            or not minimum_ratio <= ratio <= maximum_ratio):
        return None
    return basin


def _new_duplicate_components(before: np.ndarray, after: np.ndarray) -> int:
    total = 0
    for frame in np.flatnonzero(np.any(before != after, axis=(1, 2))):
        owners = set(map(int, np.unique(before[int(frame)]))) | \
            set(map(int, np.unique(after[int(frame)])))
        for owner in owners - {0}:
            total += max(0, component_accounting.component_excess(
                after[int(frame)], owner) - component_accounting.component_excess(
                    before[int(frame)], owner))
    return int(total)


def _duplicate_classification(before: np.ndarray, after: np.ndarray,
                              attached: pd.DataFrame) -> tuple[int, int]:
    """Separate independent-core duplicates from coreless projections."""
    unexplained = explained = 0
    frames = np.flatnonzero(np.any(before != after, axis=(1, 2)))
    for frame in frames:
        frame = int(frame)
        points = attached[attached.frame.astype(int).eq(frame)]
        owners = set(map(int, np.unique(before[frame]))) | \
            set(map(int, np.unique(after[frame])))
        for owner in owners - {0}:
            prior = component_accounting.component_excess(before[frame], owner)
            current = component_accounting.component_excess(after[frame], owner)
            added = max(0, current - prior)
            if not added:
                continue
            components, count = ndi.label(after[frame] == owner, STRUCTURE)
            cored = 0
            for component in range(1, count + 1):
                if _core_tracks(components == component, points):
                    cored += 1
            if cored > 1:
                unexplained += min(added, cored - 1)
                explained += max(0, added - (cored - 1))
            else:
                explained += added
    return int(unexplained), int(explained)


def _projection_allowance(before: np.ndarray, after: np.ndarray,
                          attached: pd.DataFrame, proposal,
                          projection_frames: set[int], params: dict) -> int:
    """Count nearby proposal-only duplicate lobes as explained projections."""
    allowed = 0
    owner = int(proposal.resident_owner)
    pair = {int(proposal.newborn_track), int(proposal.resident_track)}
    maximum = float(params.get(
        "maximum_projection_distance_sum_radii", 3.0))
    index = attached.set_index(["track_id", "frame"], drop=False)
    for frame in sorted(projection_frames):
        prior = component_accounting.component_excess(before[frame], owner)
        current = component_accounting.component_excess(after[frame], owner)
        added = max(0, current - prior)
        if not added:
            continue
        left = _point(index, int(proposal.newborn_track), frame)
        right = _point(index, int(proposal.resident_track), frame)
        if (left is None or right is None
                or _scaled_distance(left, right) > maximum):
            continue
        components, count = ndi.label(after[frame] == owner, STRUCTURE)
        cored: list[set[int]] = []
        points = attached[attached.frame.astype(int).eq(frame)]
        for component in range(1, count + 1):
            tracks = set(_core_tracks(components == component, points))
            if tracks:
                cored.append(tracks)
        if cored and all(tracks <= pair for tracks in cored) \
                and set().union(*cored) <= pair:
            allowed += added
    return int(allowed)


def discover_and_apply(labels: np.ndarray, raw: np.ndarray,
                       points: pd.DataFrame, params: dict
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    audit = discover(labels, raw, points, params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    attached["physically_visible"] = attached.state.astype(str).isin(
        VISIBLE_STATES)
    index = attached.set_index(["track_id", "frame"], drop=False)
    frame_points = {int(frame): group for frame, group in attached.groupby("frame")}
    candidate = labels.copy()
    applications: list[dict] = []
    occupied = np.zeros_like(labels, dtype=bool)
    applied = 0
    allowed_projection_duplicates = 0

    for proposal in audit.itertuples(index=False):
        if not bool(proposal.eligible):
            continue
        trial = candidate.copy()
        prepared: list[dict] = []
        failure = ""

        # Restore the established source seat on every remaining visible mask.
        for frame in range(int(proposal.boundary_frame),
                           int(proposal.source_last_frame) + 1):
            point = _point(index, int(proposal.source_track), frame)
            if point is None or not _truth(point.physically_visible):
                continue
            mask = _component_at_foreground(labels[frame], point)
            if mask is None:
                continue
            cores = _core_tracks(mask, frame_points.get(frame, attached.iloc[0:0]))
            if cores != [int(proposal.source_track)]:
                mask = _track_basin(
                    mask, raw[frame], point,
                    frame_points.get(frame, attached.iloc[0:0]), params)
                if mask is None:
                    failure = "source_component_not_partitionable"
                    break
                cores = _core_tracks(
                    mask, frame_points.get(frame, attached.iloc[0:0]))
                if cores != [int(proposal.source_track)]:
                    failure = "source_basin_not_exclusive"
                    break
            prepared.append({"role": "source", "frame": frame, "mask": mask,
                             "owner": int(proposal.source_owner), "cores": cores})

        # Keep the nearby newborn reference as a projection of the resident.
        projection_frames: set[int] = set()
        if not failure:
            last = max(int(proposal.newborn_last_frame),
                       int(proposal.resident_last_frame))
            for frame in range(int(proposal.boundary_frame), last + 1):
                newborn = _point(index, int(proposal.newborn_track), frame)
                resident = _point(index, int(proposal.resident_track), frame)
                newborn_visible = newborn is not None and _truth(
                    newborn.physically_visible)
                resident_visible = resident is not None and _truth(
                    resident.physically_visible)
                if not newborn_visible and not resident_visible:
                    continue
                newborn_mask = (_component_at_foreground(labels[frame], newborn)
                                if newborn_visible else None)
                resident_mask = (_component_at_foreground(labels[frame], resident)
                                 if resident_visible else None)
                unique_masks: list[np.ndarray] = []
                for mask in (newborn_mask, resident_mask):
                    if mask is not None and not any(
                            np.array_equal(mask, prior) for prior in unique_masks):
                        unique_masks.append(mask)
                for mask in unique_masks:
                    cores = _core_tracks(
                        mask, frame_points.get(frame, attached.iloc[0:0]))
                    if any(track not in {int(proposal.newborn_track),
                                         int(proposal.resident_track)}
                           for track in cores):
                        failure = "projection_component_has_third_core"
                        break
                    prepared.append({"role": "resident_projection",
                                     "frame": frame, "mask": mask,
                                     "owner": int(proposal.resident_owner),
                                     "cores": cores})
                if failure:
                    break
                if newborn_mask is not None:
                    projection_frames.add(frame)

        newborn_frames = sorted(projection_frames)
        minimum_newborn = int(params.get("minimum_newborn_assigned_frames", 4))
        if not failure and len(newborn_frames) < minimum_newborn:
            failure = "insufficient_newborn_component_support"
        if not failure and any(np.any(occupied[item["frame"]] & item["mask"])
                               for item in prepared):
            failure = "overlapping_atomic_proposal"
        if not failure:
            for item in prepared:
                trial[item["frame"]][item["mask"]] = item["owner"]
            unexplained, _ = _duplicate_classification(
                candidate, trial, attached)
            allowance = _projection_allowance(
                candidate, trial, attached, proposal, projection_frames,
                params)
            if unexplained > allowance:
                failure = "new_duplicate_component"

        if failure:
            applications.append({
                "proposal_id": proposal.proposal_id, "role": "atomic",
                "frame": int(proposal.boundary_frame), "source_owner": 0,
                "restored_owner": 0, "component_pixels": 0,
                "applied": False, "reason": failure})
            continue

        for item in prepared:
            trial_frame = int(item["frame"])
            occupied[trial_frame] |= item["mask"]
            applications.append({
                "proposal_id": proposal.proposal_id, "role": item["role"],
                "frame": trial_frame,
                "source_owner": int(np.bincount(
                    candidate[trial_frame][item["mask"]]).argmax()),
                "restored_owner": int(item["owner"]),
                "component_pixels": int(item["mask"].sum()),
                "applied": True, "reason": "applied_atomic_diversion"})
        candidate = trial
        allowed_projection_duplicates += allowance
        applied += 1

    application = pd.DataFrame(
        applications, columns=APPLICATION_COLUMNS)
    changed = candidate != labels
    unexplained, explained = _duplicate_classification(
        labels, candidate, attached)
    details = {
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_proposals": int(applied),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0,
        "new_duplicate_components": max(
            0, unexplained - allowed_projection_duplicates),
        "new_explained_projection_components": (
            explained + min(unexplained, allowed_projection_duplicates)),
    }
    return candidate, audit, application, details


def _save(source: Path, target: Path, before: np.ndarray,
          after: np.ndarray) -> None:
    if np.array_equal(before, after):
        shutil.copyfile(source, target)
    else:
        tifffile.imwrite(target, after, imagej=True, compression="zlib",
                         photometric="minisblack", metadata={
                             "axes": "TYX", "finterval": 1800.0,
                             "tunit": "sec", "unit": "pixel"})


def run(_upstream: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, audit, applications, details = discover_and_apply(
        labels, raw, points, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("terminal diversion recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    if details["new_duplicate_components"]:
        raise AssertionError("terminal diversion recovery created duplicates")

    stem = str(params.get("output_stem", labels_path.stem))
    labels_out = out.out / f"{stem}.tif"
    unclaimed_out = out.out / f"{stem}_unclaimed_original_ids.tif"
    _save(labels_path, labels_out, labels, candidate)
    shutil.copyfile(unclaimed_path, unclaimed_out)
    sidecars = {}
    for name, parameter, header in (
        ("application_audit.csv", "application_audit_path",
         "proposal_id,physical_track,assigned_identity,outcome,reason,"
         "changed_pixels,changed_frames\n"),
        ("conflicted_lineage_audit.csv", "conflicted_lineage_audit_path",
         "proposal_id,physical_track,assigned_identity,outcome,reason,"
         "changed_pixels,changed_frames\n"),
    ):
        target = out.out / name
        source = Path(params[parameter]) if params.get(parameter) else None
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text(header, encoding="utf-8")
        sidecars[name.removesuffix(".csv")] = target
    audit_path = out.out / "terminal_diversion_triples.csv"
    application_path = out.out / "terminal_diversion_applications.csv"
    metrics_path = out.out / "metrics.json"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        **details,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    outputs = {
        "labels": labels_out, "unclaimed": unclaimed_out,
        "audit": audit_path, "applications": application_path,
        "metrics": metrics_path, **sidecars}
    return {"outputs": outputs, "summary": summary}
