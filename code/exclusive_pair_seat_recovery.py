"""Recover an isolated pair whose owners collapse onto one physical seat.

The producer is field-wide and identity-blind.  It derives a one-to-one owner
assignment for repeated, raw-separable physical-track pairs from clean frames
on both sides of the encounter.  It then partitions collapsed masks and repairs
short reciprocal component swaps.  Crowded pairs, projections without a raw
valley, and pairs without bilateral owner evidence remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile


import owner_consensus
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_pair", "forced_interval",
)


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("exclusive pair-seat recovery must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "exclusive pair-seat recovery received forbidden targets: "
            + ", ".join(supplied))


def _pair(track_a: int, track_b: int) -> tuple[int, int]:
    return tuple(sorted((int(track_a), int(track_b))))


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _owner_support(group: pd.DataFrame) -> dict[int, int]:
    group = group[
        group["physically_visible"].astype(bool)
        & ~group["in_encounter"].astype(bool)
        & group["accepted_owner"].astype(int).gt(0)]
    if group.empty:
        return {}
    return {
        int(owner): int(count)
        for owner, count in group.groupby(
            group["accepted_owner"].astype(int)).size().items()}


def _best_distinct_assignment(
        track_a: int, track_b: int,
        support_a: dict[int, int], support_b: dict[int, int],
        ) -> tuple[dict | None, int]:
    owners = sorted(set(support_a) | set(support_b))
    choices: list[dict] = []
    for owner_a in owners:
        for owner_b in owners:
            if owner_a == owner_b:
                continue
            a_support = int(support_a.get(owner_a, 0))
            b_support = int(support_b.get(owner_b, 0))
            choices.append({
                "track_a": int(track_a), "track_b": int(track_b),
                "owner_a": int(owner_a), "owner_b": int(owner_b),
                "support_a": a_support, "support_b": b_support,
                "score": int(a_support + b_support),
            })
    if not choices:
        return None, 0
    choices.sort(
        key=lambda row: (
            row["score"], min(row["support_a"], row["support_b"]),
            -row["owner_a"], -row["owner_b"]),
        reverse=True)
    return choices[0], int(choices[1]["score"] if len(choices) > 1 else 0)


def _side_support(
        group: pd.DataFrame, owner: int, boundary: int, before: bool,
        ) -> int:
    keep = group[
        group["physically_visible"].astype(bool)
        & ~group["in_encounter"].astype(bool)
        & group["accepted_owner"].astype(int).eq(int(owner))]
    keep = keep[keep.frame.astype(int).lt(boundary)] if before else \
        keep[keep.frame.astype(int).gt(boundary)]
    return int(len(keep))


def _minimum_third_track_clearance(
        pair_rows: pd.DataFrame, scored: pd.DataFrame,
        track_a: int, track_b: int,
        ) -> float:
    index = scored.set_index(["track_id", "frame"], drop=False)
    clearances: list[float] = []
    for event in pair_rows[
            pair_rows.separable.astype(bool)].itertuples(index=False):
        frame = int(event.frame)
        left = _point(index, track_a, frame)
        right = _point(index, track_b, frame)
        if left is None or right is None:
            return 0.0
        left_xy = np.array([float(left.x), float(left.y)])
        right_xy = np.array([float(right.x), float(right.y)])
        separation = float(np.linalg.norm(left_xy - right_xy))
        if separation <= 0:
            return 0.0
        midpoint = (left_xy + right_xy) / 2.0
        others = scored[
            scored.frame.astype(int).eq(frame)
            & ~scored.track_id.astype(int).isin((track_a, track_b))
            & scored.strong.astype(bool)]
        if others.empty:
            clearances.append(np.inf)
            continue
        distance = np.linalg.norm(
            others[["x", "y"]].to_numpy(float) - midpoint, axis=1)
        clearances.append(float(np.min(distance) / separation))
    return float(min(clearances)) if clearances else 0.0


def _collapsed_frames(
        labels: np.ndarray, pair_rows: pd.DataFrame, scored: pd.DataFrame,
        assignment: dict,
        ) -> int:
    index = scored.set_index(["track_id", "frame"], drop=False)
    count = 0
    for event in pair_rows[
            pair_rows.separable.astype(bool)].itertuples(index=False):
        frame = int(event.frame)
        left = _point(index, assignment["track_a"], frame)
        right = _point(index, assignment["track_b"], frame)
        if left is None or right is None:
            continue
        owner_left = physical._disk_owner(
            labels[frame], left.x, left.y, left.radius_px)
        owner_right = physical._disk_owner(
            labels[frame], right.x, right.y, right.radius_px)
        if owner_left != owner_right:
            continue
        if owner_left == assignment["owner_a"]:
            lost = assignment["owner_b"]
        elif owner_left == assignment["owner_b"]:
            lost = assignment["owner_a"]
        else:
            continue
        if not np.any(labels[frame] == int(lost)):
            count += 1
    return count


def discover(
        labels: np.ndarray, points: pd.DataFrame, encounters: pd.DataFrame,
        params: dict,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every repeated pair and retain only exclusive bilateral seats."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    frames = encounters.copy()
    if frames.empty:
        return pd.DataFrame(), scored
    frames["pair"] = [
        _pair(row.track_a, row.track_b)
        for row in frames.itertuples(index=False)]
    point_groups = {
        int(track): group.copy()
        for track, group in scored.groupby("track_id", sort=True)}
    rows: list[dict] = []
    for (track_a, track_b), pair_rows in frames.groupby("pair", sort=True):
        pair_rows = pair_rows.sort_values(["frame", "encounter_id"])
        separable_frames = int(pair_rows.separable.astype(bool).sum())
        episodes = int(pair_rows.encounter_id.nunique())
        first = int(pair_rows.frame.min())
        last = int(pair_rows.frame.max())
        support_a = _owner_support(point_groups.get(
            int(track_a), pd.DataFrame(columns=scored.columns)))
        support_b = _owner_support(point_groups.get(
            int(track_b), pd.DataFrame(columns=scored.columns)))
        assignment, second_score = _best_distinct_assignment(
            track_a, track_b, support_a, support_b)
        base = {
            "track_a": int(track_a), "track_b": int(track_b),
            "first_encounter_frame": first, "last_encounter_frame": last,
            "encounter_episodes": episodes,
            "encounter_frames": int(len(pair_rows)),
            "separable_frames": separable_frames,
            "owner_a": 0, "owner_b": 0,
            "owner_a_support": 0, "owner_b_support": 0,
            "assignment_score": 0, "runner_up_score": second_score,
            "assignment_score_ratio": 0.0,
            "owner_a_before_support": 0, "owner_a_after_support": 0,
            "owner_b_before_support": 0, "owner_b_after_support": 0,
            "minimum_third_track_clearance_pair_distances": 0.0,
            "collapsed_separable_frames": 0,
            "eligible": False, "reason": "",
        }
        reasons: list[str] = []
        if separable_frames < int(params.get("minimum_separable_frames", 3)):
            reasons.append("insufficient_raw_separable_frames")
        if episodes < int(params.get("minimum_encounter_episodes", 2)):
            reasons.append("not_recurrent_pair")
        if assignment is None:
            reasons.append("no_distinct_owner_assignment")
        else:
            score = int(assignment["score"])
            ratio = float(score / max(second_score, 1))
            a_group = point_groups[int(track_a)]
            b_group = point_groups[int(track_b)]
            a_before = _side_support(
                a_group, assignment["owner_a"], first, True)
            a_after = _side_support(
                a_group, assignment["owner_a"], last, False)
            b_before = _side_support(
                b_group, assignment["owner_b"], first, True)
            b_after = _side_support(
                b_group, assignment["owner_b"], last, False)
            clearance = _minimum_third_track_clearance(
                pair_rows, scored, track_a, track_b)
            collapsed = _collapsed_frames(
                labels, pair_rows, scored, assignment)
            base.update({
                "owner_a": assignment["owner_a"],
                "owner_b": assignment["owner_b"],
                "owner_a_support": assignment["support_a"],
                "owner_b_support": assignment["support_b"],
                "assignment_score": score,
                "assignment_score_ratio": ratio,
                "owner_a_before_support": a_before,
                "owner_a_after_support": a_after,
                "owner_b_before_support": b_before,
                "owner_b_after_support": b_after,
                "minimum_third_track_clearance_pair_distances": clearance,
                "collapsed_separable_frames": collapsed,
            })
            minimum_owner_support = int(params.get(
                "minimum_owner_support_frames", 6))
            if min(assignment["support_a"], assignment["support_b"]) < \
                    minimum_owner_support:
                reasons.append("insufficient_owner_support")
            if ratio < float(params.get(
                    "minimum_assignment_score_ratio", 3.0)):
                reasons.append("ambiguous_owner_assignment")
            minimum_side = int(params.get(
                "minimum_bookend_support_frames_per_side", 2))
            if min(a_before, a_after, b_before, b_after) < minimum_side:
                reasons.append("owner_seats_not_bilaterally_bookended")
            if clearance < float(params.get(
                    "minimum_third_track_clearance_pair_distances", 2.5)):
                reasons.append("crowded_three_body_context")
            if collapsed < int(params.get(
                    "minimum_collapsed_separable_frames", 2)):
                reasons.append("no_repeated_owner_collapse")
        rows.append({
            **base, "eligible": not reasons,
            "reason": "eligible_exclusive_bookended_pair" if not reasons
            else "|".join(reasons),
        })
    audit = pd.DataFrame(rows)
    audit.insert(0, "proposal_id", "")
    eligible = list(audit.index[audit.eligible.astype(bool)])
    for number, index in enumerate(eligible, 1):
        audit.loc[index, "proposal_id"] = f"ES{number:04d}"
    return audit, scored


def _new_excess_components(
        before: np.ndarray, after: np.ndarray, owner: int) -> int:
    old = int(ndi.label(before == int(owner), STRUCTURE)[1])
    new = int(ndi.label(after == int(owner), STRUCTURE)[1])
    return max(0, max(0, new - 1) - max(0, old - 1))


def _seed_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    components, _ = ndi.label(mask, STRUCTURE)
    value = int(components[seed])
    return components == value if value else np.zeros_like(mask)


def _connected_two_part(
        target: np.ndarray, lost_part: np.ndarray, host_part: np.ndarray,
        lost_marker: tuple[int, int], host_marker: tuple[int, int],
        minimum_pixels: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Keep a two-marker partition exhaustive while removing basin slivers."""
    options: list[tuple[int, np.ndarray, np.ndarray]] = []
    lost_core = _seed_component(lost_part, lost_marker)
    host_complement = target & ~lost_core
    if (host_complement[host_marker]
            and int(ndi.label(host_complement, STRUCTURE)[1]) == 1
            and int(lost_core.sum()) >= minimum_pixels
            and int(host_complement.sum()) >= minimum_pixels):
        options.append((abs(int(lost_core.sum()) - int(lost_part.sum())),
                        lost_core, host_complement))
    host_core = _seed_component(host_part, host_marker)
    lost_complement = target & ~host_core
    if (lost_complement[lost_marker]
            and int(ndi.label(lost_complement, STRUCTURE)[1]) == 1
            and int(lost_complement.sum()) >= minimum_pixels
            and int(host_core.sum()) >= minimum_pixels):
        options.append((abs(int(lost_complement.sum()) - int(lost_part.sum())),
                        lost_complement, host_core))
    if not options:
        return None
    _, lost, host = min(options, key=lambda row: row[0])
    return lost, host


def _split_collapsed(
        frame: np.ndarray, raw: np.ndarray, lost, host,
        lost_owner: int, host_owner: int, labels: np.ndarray, params: dict,
        ) -> tuple[np.ndarray | None, str, int]:
    host_components, _ = ndi.label(frame == int(host_owner), STRUCTURE)
    lost_component = physical._component_at(
        host_components, float(lost.x), float(lost.y))
    host_component = physical._component_at(
        host_components, float(host.x), float(host.y))
    if lost_component <= 0 or host_component <= 0:
        return None, "core_outside_host_mask", 0
    target = host_components == lost_component
    target_area = int(target.sum())
    median_area = physical._median_identity_area(labels, lost_owner)
    ratio = target_area / max(median_area, 1.0)
    if not (float(params.get("minimum_area_ratio", 0.25)) <= ratio <=
            float(params.get("maximum_area_ratio", 4.0))):
        return None, "component_area_out_of_range", 0
    if lost_component != host_component:
        change = target
        method = "separate_component_relabel"
    else:
        lost_marker = physical._marker_pixel(
            target, float(lost.x), float(lost.y))
        host_marker = physical._marker_pixel(
            target, float(host.x), float(host.y))
        if lost_marker is None or host_marker is None or \
                lost_marker == host_marker:
            return None, "watershed_markers_unavailable", 0
        markers = np.zeros(frame.shape, np.int16)
        markers[lost_marker] = 1
        markers[host_marker] = 2
        elevation = -ndi.gaussian_filter(
            raw.astype(np.float32),
            float(params.get("watershed_sigma_px", 1.0)))
        partition = watershed(elevation, markers=markers, mask=target)
        raw_lost, raw_host = partition == 1, partition == 2
        minimum_pixels = int(params.get("minimum_changed_pixels", 5))
        connected = _connected_two_part(
            target, raw_lost, raw_host, lost_marker, host_marker,
            minimum_pixels)
        if connected is None:
            yy, xx = np.nonzero(target)
            lost_distance = np.hypot(
                yy - float(lost.y), xx - float(lost.x)) / max(
                    float(lost.radius_px), 1.0)
            host_distance = np.hypot(
                yy - float(host.y), xx - float(host.x)) / max(
                    float(host.radius_px), 1.0)
            geometry_lost = np.zeros_like(target)
            geometry_lost[yy[lost_distance <= host_distance],
                          xx[lost_distance <= host_distance]] = True
            connected = _connected_two_part(
                target, geometry_lost, target & ~geometry_lost,
                lost_marker, host_marker, minimum_pixels)
            if connected is None:
                return None, "connected_partition_unavailable", 0
            method = "radius_normalised_geometry_connected"
        else:
            method = "raw_watershed_connected"
        change, host_change = connected
        if not np.array_equal(change | host_change, target):
            return None, "partition_not_exhaustive", 0
    changed = int(change.sum())
    if changed < int(params.get("minimum_changed_pixels", 5)):
        return None, "recovered_region_too_small", 0
    trial = frame.copy()
    trial[change] = int(lost_owner)
    if (_new_excess_components(frame, trial, lost_owner)
            or _new_excess_components(frame, trial, host_owner)):
        return None, "new_duplicate_component", 0
    return trial, method, changed


def _swap_components(
        frame: np.ndarray, point_a, point_b,
        owner_a: int, owner_b: int,
        ) -> tuple[np.ndarray | None, int]:
    parts_b, _ = ndi.label(frame == int(owner_b), STRUCTURE)
    parts_a, _ = ndi.label(frame == int(owner_a), STRUCTURE)
    component_at_a = physical._component_at(
        parts_b, float(point_a.x), float(point_a.y))
    component_at_b = physical._component_at(
        parts_a, float(point_b.x), float(point_b.y))
    if component_at_a <= 0 or component_at_b <= 0:
        return None, 0
    mask_a = parts_b == component_at_a
    mask_b = parts_a == component_at_b
    trial = frame.copy()
    trial[mask_a] = int(owner_a)
    trial[mask_b] = int(owner_b)
    if (_new_excess_components(frame, trial, owner_a)
            or _new_excess_components(frame, trial, owner_b)):
        return None, 0
    return trial, int(mask_a.sum() + mask_b.sum())


def apply(
        labels: np.ndarray, raw: np.ndarray, pair_audit: pd.DataFrame,
        scored: pd.DataFrame, encounters: pd.DataFrame, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    applications: list[dict] = []
    index = scored.set_index(["track_id", "frame"], drop=False)
    encounter_frames = encounters.copy()
    if len(encounter_frames):
        encounter_frames["pair"] = [
            _pair(row.track_a, row.track_b)
            for row in encounter_frames.itertuples(index=False)]
    eligible = pair_audit[pair_audit.eligible.astype(bool)] \
        if len(pair_audit) else pair_audit
    for proposal in eligible.itertuples(index=False):
        track_a, track_b = int(proposal.track_a), int(proposal.track_b)
        owner_a, owner_b = int(proposal.owner_a), int(proposal.owner_b)
        pair_rows = encounter_frames[
            encounter_frames["pair"].map(
                lambda value: value == (track_a, track_b))]
        distance = int(params.get(
            "maximum_reciprocal_frame_distance_from_encounter", 3))
        first = int(proposal.first_encounter_frame)
        last = min(len(candidate) - 1,
                   int(proposal.last_encounter_frame) + distance)
        for frame_index in range(first, last + 1):
            point_a = _point(index, track_a, frame_index)
            point_b = _point(index, track_b, frame_index)
            base = {
                "proposal_id": proposal.proposal_id,
                "frame": frame_index, "track_a": track_a,
                "track_b": track_b, "owner_a": owner_a,
                "owner_b": owner_b, "application_type": "collapsed_pair",
                "applied": False, "changed_pixels": 0,
            }
            if point_a is None or point_b is None:
                applications.append({**base, "reason": "missing_pair_point"})
                continue
            if not (bool(point_a.physically_visible)
                    and bool(point_b.physically_visible)):
                applications.append({
                    **base, "reason": "pair_not_physically_visible"})
                continue
            current_a = physical._disk_owner(
                candidate[frame_index], point_a.x, point_a.y,
                point_a.radius_px)
            current_b = physical._disk_owner(
                candidate[frame_index], point_b.x, point_b.y,
                point_b.radius_px)
            if current_a == current_b == owner_a and not np.any(
                    candidate[frame_index] == owner_b):
                lost, host = point_b, point_a
                lost_owner, host_owner = owner_b, owner_a
            elif current_a == current_b == owner_b and not np.any(
                    candidate[frame_index] == owner_a):
                lost, host = point_a, point_b
                lost_owner, host_owner = owner_a, owner_b
            else:
                applications.append({
                    **base, "reason": "owners_not_exclusively_collapsed"})
                continue
            trial, method, changed = _split_collapsed(
                candidate[frame_index], raw[frame_index], lost, host,
                lost_owner, host_owner, labels, params)
            if trial is None:
                applications.append({**base, "reason": method})
                continue
            candidate[frame_index] = trial
            applications.append({
                **base, "applied": True, "changed_pixels": changed,
                "reason": method})

        reciprocal_first = max(
            0, int(proposal.first_encounter_frame) - distance)
        for frame_index in range(reciprocal_first, last + 1):
            point_a = _point(index, track_a, frame_index)
            point_b = _point(index, track_b, frame_index)
            if point_a is None or point_b is None or not (
                    bool(point_a.strong) and bool(point_b.strong)):
                continue
            current_a = physical._disk_owner(
                candidate[frame_index], point_a.x, point_a.y,
                point_a.radius_px)
            current_b = physical._disk_owner(
                candidate[frame_index], point_b.x, point_b.y,
                point_b.radius_px)
            if not (current_a == owner_b and current_b == owner_a):
                continue
            trial, changed = _swap_components(
                candidate[frame_index], point_a, point_b, owner_a, owner_b)
            base = {
                "proposal_id": proposal.proposal_id,
                "frame": frame_index, "track_a": track_a,
                "track_b": track_b, "owner_a": owner_a,
                "owner_b": owner_b, "application_type": "reciprocal_swap",
                "applied": trial is not None,
                "changed_pixels": changed,
                "reason": "atomic_component_swap" if trial is not None
                else "unsafe_component_swap",
            }
            applications.append(base)
            if trial is not None:
                candidate[frame_index] = trial
    return candidate, pd.DataFrame(applications)


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
    points = pd.read_csv(params["physical_track_points_path"])
    encounters = pd.read_csv(params["encounter_frames_path"])
    pair_audit, scored = discover(labels, points, encounters, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame()
    else:
        candidate, applications = apply(
            labels, raw, pair_audit, scored, encounters, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("exclusive pair-seat recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("exclusive pair-seat recovery changed identity set")
    duplicate_components = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicate_components:
        raise AssertionError(
            "exclusive pair-seat recovery created duplicate components")

    output_stem = str(params.get(
        "output_stem", Path(params["labels_path"]).stem))
    labels_path = out.out / f"{output_stem}.tif"
    unclaimed_path = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    pair_path = out.out / "exclusive_pair_seat_audit.csv"
    application_path = out.out / "exclusive_pair_seat_applications.csv"
    pair_audit.to_csv(pair_path, index=False)
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
        "pairs_audited": int(len(pair_audit)),
        "eligible_pairs": int(pair_audit.eligible.astype(bool).sum())
            if len(pair_audit) else 0,
        "applied_pairs": int(applied[
            ["track_a", "track_b"]].drop_duplicates().shape[0])
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicate_components),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs = {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "pair_audit": pair_path, "applications": application_path,
        "metrics": metrics_path,
    }
    outputs.update(_copy_audits(upstream_dir, params, out))
    return {"outputs": outputs, "summary": metrics}
