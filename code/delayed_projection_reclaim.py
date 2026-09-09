"""Recover a delayed projection owner after an asymmetric two-core overlap.

The rule is field-wide and target-free.  It handles the topology where a
durable physical lineage ends during an encounter, a newly visible physical
lineage continues, both temporarily carry the ending lineage's owner, and the
continuing lineage shortly reclaims an already-established nearby owner.  The
shared foreground is partitioned between the two physical cores; no foreground
or biological identity is created.
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

import component_accounting
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region_target", "review_case",
    "case_id", "forced_identity", "forced_track", "forced_frame",
    "include_track", "exclude_track", "include_identity",
    "exclude_identity",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "delayed projection reclaim received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("delayed projection reclaim must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _pair(track_a: int, track_b: int) -> tuple[int, int]:
    return tuple(sorted((int(track_a), int(track_b))))


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return distance / scale


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    rows = group[
        group.physically_visible.astype(bool)
        & group.accepted_owner.astype(int).gt(0)
    ].sort_values("frame")
    runs: list[dict] = []
    for row in rows.itertuples(index=False):
        owner, frame = int(row.accepted_owner), int(row.frame)
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


def _profile(raw: np.ndarray, left, right, sigma: float) -> dict:
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


def _nearby_owner_anchor(
        scored: pd.DataFrame, index: pd.DataFrame, point, owner: int,
        frame: int, excluded: set[int], first: int, minimum_support: int,
        maximum_distance: float) -> tuple[int, float, int] | None:
    candidates = scored[
        scored.frame.astype(int).eq(int(frame))
        & scored.physically_visible.astype(bool)
        & scored.accepted_owner.astype(int).eq(int(owner))
        & ~scored.track_id.astype(int).isin(excluded)]
    ranked: list[tuple[float, int, int]] = []
    for anchor in candidates.itertuples(index=False):
        track = int(anchor.track_id)
        history = scored[
            scored.track_id.astype(int).eq(track)
            & scored.frame.astype(int).lt(int(first))
            & scored.physically_visible.astype(bool)
            & scored.accepted_owner.astype(int).eq(int(owner))]
        support = int(len(history))
        if support < minimum_support:
            continue
        distance = _scaled_distance(point, anchor)
        if distance <= maximum_distance:
            ranked.append((distance, track, support))
    if not ranked:
        return None
    distance, track, support = min(ranked)
    return int(track), float(distance), int(support)


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             encounters: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate every asymmetric overlap/reclaim topology in the field."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    scored["physically_visible"] = scored.state.astype(str).isin(VISIBLE_STATES)
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    index = scored.set_index(["track_id", "frame"], drop=False)
    frames = encounters.copy()
    if frames.empty:
        return pd.DataFrame(), scored
    frames["pair"] = [_pair(row.track_a, row.track_b)
                      for row in frames.itertuples(index=False)]
    movie = len(labels)
    minimum_anchor = max(2, int(math.ceil(movie * float(
        params.get("minimum_ending_anchor_movie_fraction", 0.08)))))
    minimum_successor = max(2, int(math.ceil(movie * float(
        params.get("minimum_successor_run_movie_fraction", 0.07)))))
    minimum_projection_anchor = max(2, int(math.ceil(movie * float(
        params.get("minimum_projection_anchor_movie_fraction", 0.06)))))
    birth_slack = max(1, int(math.ceil(movie * float(
        params.get("maximum_birth_slack_movie_fraction", 0.02)))))
    end_slack = max(1, int(math.ceil(movie * float(
        params.get("maximum_end_slack_movie_fraction", 0.02)))))
    reclaim_delay = max(1, int(math.ceil(movie * float(
        params.get("maximum_reclaim_delay_movie_fraction", 0.04)))))
    minimum_shared_fraction = float(params.get(
        "minimum_shared_ending_owner_fraction", 0.75))
    minimum_balance = float(params.get("minimum_endpoint_balance", 0.50))
    maximum_pair_distance = float(params.get(
        "maximum_pair_distance_sum_radii", 1.60))
    minimum_strong = int(params.get("minimum_both_strong_frames", 2))
    maximum_projection_distance = float(params.get(
        "maximum_projection_anchor_distance_sum_radii", 3.25))
    sigma = float(params.get("pair_profile_sigma_px", 1.0))
    rows: list[dict] = []

    for pair, pair_rows in frames.groupby("pair", sort=True):
        track_left, track_right = pair
        first, last = int(pair_rows.frame.min()), int(pair_rows.frame.max())
        for ending_track, continuing_track in (
                (track_left, track_right), (track_right, track_left)):
            ending = groups.get(int(ending_track))
            continuing = groups.get(int(continuing_track))
            if ending is None or continuing is None:
                continue
            ending_visible = ending[ending.physically_visible.astype(bool)]
            continuing_visible = continuing[
                continuing.physically_visible.astype(bool)]
            reasons: list[str] = []
            if ending_visible.empty or continuing_visible.empty:
                continue
            ending_last = int(ending_visible.frame.max())
            continuing_first = int(continuing_visible.frame.min())
            if not last <= ending_last <= last + end_slack:
                reasons.append("ending_lineage_does_not_end_at_overlap")
            if not first - birth_slack <= continuing_first <= first + birth_slack:
                reasons.append("continuing_lineage_not_born_at_overlap")
            # The remaining measurements are intentionally limited to the
            # asymmetric end/birth topology.  This keeps a field-wide scan
            # linear in the small candidate set instead of repeatedly
            # filtering every physical point for every unrelated encounter.
            if reasons:
                continue

            ending_runs = [run for run in _positive_runs(ending)
                           if int(run["last"]) == ending_last]
            if not ending_runs:
                reasons.append("no_ending_owner_anchor")
                ending_owner = 0
                ending_support = 0
            else:
                ending_run = ending_runs[-1]
                ending_owner = int(ending_run["owner"])
                ending_support = int(ending_run["frames"])
                if ending_support < minimum_anchor:
                    reasons.append("ending_owner_anchor_too_short")
            if reasons:
                continue

            overlap_profiles: list[dict] = []
            shared = observed = 0
            for frame in sorted(set(pair_rows.frame.astype(int))):
                left = _point(index, ending_track, frame)
                right = _point(index, continuing_track, frame)
                if left is None or right is None or not (
                        _truth(left.physically_visible)
                        and _truth(right.physically_visible)):
                    continue
                observed += 1
                shared += int(int(left.accepted_owner) == ending_owner
                              and int(right.accepted_owner) == ending_owner)
                overlap_profiles.append(_profile(raw[frame], left, right, sigma))
            shared_fraction = shared / max(observed, 1)
            if observed < 2:
                reasons.append("insufficient_overlap_observations")
            if shared_fraction < minimum_shared_fraction:
                reasons.append("overlap_not_shared_by_ending_owner")
            if (not overlap_profiles
                    or max(item["distance_sum_radii"]
                           for item in overlap_profiles) > maximum_pair_distance):
                reasons.append("physical_cores_too_distant")
            if (not overlap_profiles
                    or min(item["endpoint_balance"]
                           for item in overlap_profiles) < minimum_balance):
                reasons.append("second_core_not_brightness_supported")
            strong_frames = sum(item["both_strong"]
                                for item in overlap_profiles)
            if strong_frames < minimum_strong:
                reasons.append("insufficient_strong_two_core_frames")

            successor_runs = [
                run for run in _positive_runs(continuing)
                if int(run["owner"]) != ending_owner
                and last < int(run["first"]) <= last + reclaim_delay]
            if not successor_runs:
                reasons.append("no_delayed_distinct_owner_reclaim")
                successor_owner = successor_first = successor_support = 0
                anchor = None
            else:
                successor_run = successor_runs[0]
                successor_owner = int(successor_run["owner"])
                successor_first = int(successor_run["first"])
                successor_support = int(successor_run["frames"])
                if successor_support < minimum_successor:
                    reasons.append("successor_owner_run_too_short")
                successor_point = _point(
                    index, continuing_track, successor_first)
                anchor = _nearby_owner_anchor(
                    scored, index, successor_point, successor_owner,
                    successor_first, {ending_track, continuing_track}, first,
                    minimum_projection_anchor, maximum_projection_distance)
                if anchor is None:
                    reasons.append("no_nearby_established_projection_owner")

            rows.append({
                "ending_track": int(ending_track),
                "continuing_track": int(continuing_track),
                "first_overlap_frame": first, "last_overlap_frame": last,
                "ending_owner": int(ending_owner),
                "successor_owner": int(successor_owner),
                "successor_first_frame": int(successor_first),
                "ending_anchor_frames": int(ending_support),
                "successor_run_frames": int(successor_support),
                "overlap_observations": int(observed),
                "shared_ending_owner_fraction": float(shared_fraction),
                "both_strong_frames": int(strong_frames),
                "minimum_endpoint_balance": float(min(
                    (item["endpoint_balance"] for item in overlap_profiles),
                    default=float("nan"))),
                "maximum_pair_distance_sum_radii": float(max(
                    (item["distance_sum_radii"] for item in overlap_profiles),
                    default=float("nan"))),
                "median_valley_ratio": float(np.median([
                    item["valley_ratio"] for item in overlap_profiles]))
                    if overlap_profiles else float("nan"),
                "projection_anchor_track": int(anchor[0]) if anchor else 0,
                "projection_anchor_distance_sum_radii": float(anchor[1])
                    if anchor else float("nan"),
                "projection_anchor_frames": int(anchor[2]) if anchor else 0,
                "eligible": not reasons,
                "reason": "eligible_asymmetric_projection_reclaim"
                if not reasons else "|".join(reasons),
            })
    audit = pd.DataFrame(rows)
    if len(audit):
        audit.insert(0, "proposal_id", "")
        for number, row_index in enumerate(
                audit.index[audit.eligible.astype(bool)], 1):
            audit.loc[row_index, "proposal_id"] = f"DPR{number:04d}"
    return audit, scored


def _marker(mask: np.ndarray, point) -> tuple[int, int] | None:
    y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
    if mask[y, x]:
        return y, x
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    nearest = int(np.argmin(
        (xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2))
    return int(yy[nearest]), int(xx[nearest])


def _core_tracks(mask: np.ndarray, frame_points: pd.DataFrame) -> set[int]:
    tracks: set[int] = set()
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    for point in frame_points.itertuples(index=False):
        if not _truth(point.physically_visible):
            continue
        radius = max(2.0, 0.5 * float(point.radius_px))
        disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
                <= radius ** 2)
        if np.any(mask & disk):
            tracks.add(int(point.track_id))
    return tracks


def _connected_seed_partition(
        mask: np.ndarray, raw: np.ndarray, ending, continuing,
        params: dict, continuing_reference_area: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray, str] | None:
    left = _marker(mask, ending)
    right = _marker(mask, continuing)
    if left is None or right is None or left == right:
        return None
    markers = np.zeros(mask.shape, np.int16)
    markers[left], markers[right] = 1, 2
    smooth = ndi.gaussian_filter(
        raw.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    candidates: list[tuple[np.ndarray, np.ndarray, str]] = []
    split = watershed(-smooth, markers=markers, mask=mask,
                      connectivity=STRUCTURE.astype(bool))
    candidates.append((split == 1, split == 2, "raw_watershed"))
    yy, xx = np.nonzero(mask)
    ending_distance = ((xx - float(ending.x)) ** 2
                       + (yy - float(ending.y)) ** 2) / max(
                           float(ending.radius_px), 1.0) ** 2
    continuing_distance = ((xx - float(continuing.x)) ** 2
                           + (yy - float(continuing.y)) ** 2) / max(
                               float(continuing.radius_px), 1.0) ** 2
    geometric = np.zeros(mask.shape, np.uint8)
    geometric[yy[ending_distance <= continuing_distance],
              xx[ending_distance <= continuing_distance]] = 1
    geometric[(mask) & (geometric == 0)] = 2
    candidates.append((geometric == 1, geometric == 2,
                       "radius_normalised_geometry"))
    minimum = int(params.get("minimum_partition_pixels", 5))
    minimum_ratio = float(params.get("minimum_basin_area_ratio", 0.20))
    maximum_ratio = float(params.get("maximum_basin_area_ratio", 3.25))
    continuing_expected = max(
        float(continuing_reference_area)
        if continuing_reference_area is not None
        else math.pi * float(continuing.radius_px) ** 2, 1.0)
    for ending_part, continuing_part, method in candidates:
        continuing_ratio = float(continuing_part.sum()) / continuing_expected
        if (int(ending_part.sum()) < minimum
                or int(continuing_part.sum()) < minimum
                or not minimum_ratio <= continuing_ratio <= maximum_ratio
                or not np.array_equal(ending_part | continuing_part, mask)
                or int(ndi.label(ending_part, STRUCTURE)[1]) != 1
                or int(ndi.label(continuing_part, STRUCTURE)[1]) != 1
                or not ending_part[left] or not continuing_part[right]):
            continue
        return ending_part, continuing_part, method
    return None


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    parts, _ = ndi.label(frame == int(owner), STRUCTURE)
    marker = _marker(frame == int(owner), point)
    if marker is None:
        return None
    value = int(parts[marker])
    return parts == value if value else None


def _component_excess(frame: np.ndarray, owner: int) -> int:
    return component_accounting.component_excess(frame, int(owner))


def _track_basin_area(mask: np.ndarray, target,
                      frame_points: pd.DataFrame) -> int:
    """Estimate one projection's area inside a multi-core owner component."""
    occupants = []
    yy_grid, xx_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
    for point in frame_points.itertuples(index=False):
        if not _truth(point.physically_visible):
            continue
        radius = max(2.0, 0.5 * float(point.radius_px))
        disk = ((xx_grid - float(point.x)) ** 2
                + (yy_grid - float(point.y)) ** 2 <= radius ** 2)
        if np.any(mask & disk):
            occupants.append(point)
    if not occupants:
        return 0
    target_index = next((index for index, point in enumerate(occupants)
                         if int(point.track_id) == int(target.track_id)), None)
    if target_index is None:
        return 0
    yy, xx = np.nonzero(mask)
    scores = np.vstack([
        ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2)
        / max(float(point.radius_px), 1.0) ** 2
        for point in occupants])
    return int(np.count_nonzero(np.argmin(scores, axis=0) == target_index))


def apply(labels: np.ndarray, raw: np.ndarray, audit: pd.DataFrame,
          scored: pd.DataFrame, params: dict
          ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    candidate = labels.copy()
    index = scored.set_index(["track_id", "frame"], drop=False)
    frame_points = {int(frame): group for frame, group in scored.groupby("frame")}
    applications: list[dict] = []
    occupied = np.zeros_like(labels, dtype=bool)
    applied = explained = 0
    eligible = audit[audit.eligible.astype(bool)] if len(audit) else audit
    for proposal in eligible.itertuples(index=False):
        trial = candidate.copy()
        prepared: list[dict] = []
        failure = ""
        successor_areas: list[int] = []
        successor_last = (int(proposal.successor_first_frame)
                          + int(proposal.successor_run_frames) - 1)
        for successor_frame in range(
                int(proposal.successor_first_frame), successor_last + 1):
            successor_point = _point(
                index, int(proposal.continuing_track), successor_frame)
            if successor_point is None:
                continue
            successor_mask = _component_at(
                labels[successor_frame], int(proposal.successor_owner),
                successor_point)
            if successor_mask is not None:
                successor_areas.append(_track_basin_area(
                    successor_mask, successor_point,
                    frame_points.get(successor_frame, scored.iloc[0:0])))
        reference_area = (float(np.median(successor_areas))
                          if successor_areas else None)
        if reference_area is None:
            failure = "successor_component_area_unavailable"
        repair_stop = (int(proposal.successor_first_frame)
                       if bool(params.get("repair_delay_after_overlap", True))
                       else int(proposal.last_overlap_frame) + 1)
        for frame in range(int(proposal.first_overlap_frame), repair_stop):
            if failure:
                break
            ending = _point(index, int(proposal.ending_track), frame)
            continuing = _point(index, int(proposal.continuing_track), frame)
            if continuing is None or not _truth(continuing.physically_visible):
                failure = "continuing_core_missing_before_reclaim"
                break
            current_owner = int(continuing.accepted_owner)
            if current_owner != int(proposal.ending_owner):
                failure = "unexpected_owner_before_reclaim"
                break
            host = _component_at(
                labels[frame], int(proposal.ending_owner), continuing)
            if host is None:
                failure = "continuing_core_outside_ending_owner"
                break
            cores = _core_tracks(host, frame_points.get(
                frame, scored.iloc[0:0]))
            allowed = {int(proposal.ending_track),
                       int(proposal.continuing_track)}
            if not cores <= allowed:
                failure = "shared_component_contains_third_core"
                break
            method = "separate_component_relabel"
            change = host
            if ending is not None and _truth(ending.physically_visible):
                ending_host = _component_at(
                    labels[frame], int(proposal.ending_owner), ending)
                if ending_host is not None and np.array_equal(ending_host, host):
                    partition = _connected_seed_partition(
                        host, raw[frame], ending, continuing, params,
                        reference_area)
                    if partition is None:
                        failure = "shared_component_not_partitionable"
                        break
                    _, change, method = partition
            if np.any(occupied[frame] & change):
                failure = "overlapping_atomic_proposal"
                break
            anchor = _nearby_owner_anchor(
                scored, index, continuing, int(proposal.successor_owner),
                frame, allowed, int(proposal.first_overlap_frame),
                int(proposal.projection_anchor_frames),
                float(params.get(
                    "maximum_projection_anchor_distance_sum_radii", 3.25)))
            if anchor is None:
                failure = "projection_anchor_missing_during_application"
                break
            prepared.append({"frame": frame, "mask": change,
                             "method": method, "anchor_track": anchor[0]})

        if not failure:
            for item in prepared:
                frame = int(item["frame"])
                trial[frame][item["mask"]] = int(proposal.successor_owner)
                delta = (_component_excess(
                    trial[frame], int(proposal.successor_owner))
                    - _component_excess(
                        candidate[frame], int(proposal.successor_owner)))
                if delta > 1:
                    failure = "more_than_one_projection_component_created"
                    break
                if delta > 0:
                    explained += int(delta)
        if failure:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "frame": int(proposal.first_overlap_frame),
                "method": "atomic", "changed_pixels": 0,
                "applied": False, "reason": failure})
            continue
        for item in prepared:
            frame = int(item["frame"])
            occupied[frame] |= item["mask"]
            applications.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "method": item["method"],
                "changed_pixels": int(item["mask"].sum()),
                "applied": True,
                "reason": "applied_delayed_projection_reclaim"})
        candidate = trial
        applied += 1
    changed = candidate != labels
    details = {
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_proposals": int(applied),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0,
        "new_duplicate_components": 0,
        "new_explained_projection_components": int(explained),
    }
    return candidate, pd.DataFrame(applications), details


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
    encounters = pd.read_csv(params["encounter_frames_path"])
    audit, scored = discover(labels, raw, points, encounters, params)
    candidate, applications, details = apply(
        labels, raw, audit, scored, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("delayed projection reclaim changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")

    stem = str(params.get("output_stem", labels_path.stem))
    labels_out = out.out / f"{stem}.tif"
    unclaimed_out = out.out / f"{stem}_unclaimed_original_ids.tif"
    _save(labels_path, labels_out, labels, candidate)
    shutil.copyfile(unclaimed_path, unclaimed_out)
    audit_path = out.out / "delayed_projection_reclaim_audit.csv"
    applications_path = out.out / "delayed_projection_reclaim_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(applications_path, index=False)
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
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        **details,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": labels_out, "unclaimed": unclaimed_out,
        "audit": audit_path, "applications": applications_path,
        "metrics": metrics_path, **sidecars}, "summary": summary}


