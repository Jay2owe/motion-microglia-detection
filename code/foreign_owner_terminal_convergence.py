"""Preserve two owner seats through a short foreign-owner convergence."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import concurrent_duplicate_invasion as invasion
import separable_merge_recovery as physical
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)


def assert_target_free(params: dict) -> None:
    invasion.assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal convergence recovery must be field-wide")


def _runs(group: pd.DataFrame) -> list[dict]:
    result: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        if not bool(row.physically_visible) or int(row.accepted_owner) <= 0:
            continue
        frame, owner = int(row.frame), int(row.accepted_owner)
        if (result and result[-1]["owner"] == owner
                and frame == result[-1]["last"] + 1):
            result[-1]["last"] = frame
            result[-1]["frames"].append(frame)
        else:
            result.append({"owner": owner, "first": frame, "last": frame,
                           "frames": [frame]})
    return result


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _residual_marker(frame: np.ndarray, component: np.ndarray,
                     targets: list, params: dict):
    sigma = float(params.get("watershed_sigma_px", 1.0))
    smooth = ndi.gaussian_filter(frame.astype(np.float32), sigma)
    allowed = component.copy()
    yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
    minimum_distance = float(
        params.get("minimum_residual_distance_sum_radii", 1.0))
    for row in targets:
        distance = minimum_distance * max(float(row.radius_px), 1.0)
        allowed &= ((xx - float(row.x)) ** 2
                    + (yy - float(row.y)) ** 2 > distance ** 2)
    if not np.any(allowed):
        return None, 0.0
    local_max = smooth == ndi.maximum_filter(smooth, size=3, mode="nearest")
    peaks = allowed & local_max & (frame > 0)
    if not np.any(peaks):
        peaks = allowed & (frame > 0)
    if not np.any(peaks):
        return None, 0.0
    scores = np.where(peaks, smooth, -np.inf)
    marker = np.unravel_index(int(np.argmax(scores)), scores.shape)
    component_peak = float(np.max(smooth[component]))
    ratio = float(smooth[marker] / max(component_peak, 1.0))
    if ratio < float(params.get("minimum_residual_peak_fraction", 0.25)):
        return None, ratio
    return (int(marker[0]), int(marker[1])), ratio


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    minimum_stable = int(params.get("minimum_preconvergence_owner_frames", 10))
    maximum_foreign = int(params.get("maximum_foreign_owner_frames", 6))
    maximum_return_gap = int(params.get("maximum_owner_return_gap_frames", 2))
    maximum_start_delta = int(
        params.get("maximum_foreign_start_delta_frames", 2))
    minimum_pre_coexist = int(
        params.get("minimum_preconvergence_coexistence", 10))
    maximum_pair_distance = float(
        params.get("maximum_pair_distance_sum_radii", 2.5))
    minimum_residual_frames = int(params.get("minimum_residual_core_frames", 3))
    by_track = {int(track): group.sort_values("frame")
                for track, group in scored.groupby("track_id", sort=True)}
    candidates: list[dict] = []
    for track, group in by_track.items():
        runs = _runs(group)
        for index in range(1, len(runs)):
            stable, foreign = runs[index - 1], runs[index]
            if stable["owner"] == foreign["owner"]:
                continue
            next_run = runs[index + 1] if index + 1 < len(runs) else None
            track_last = int(group.frame.max())
            terminal = int(foreign["last"]) >= track_last - 1
            returns = bool(
                next_run is not None
                and int(next_run["owner"]) == int(stable["owner"])
                and int(next_run["first"]) - int(foreign["last"])
                <= maximum_return_gap)
            if (len(stable["frames"]) >= minimum_stable
                    and len(foreign["frames"]) <= maximum_foreign
                    and (terminal or returns)):
                candidates.append({
                    "track": track, "stable_owner": int(stable["owner"]),
                    "foreign_owner": int(foreign["owner"]),
                    "stable_first": int(stable["first"]),
                    "stable_last": int(stable["last"]),
                    "stable_frames": len(stable["frames"]),
                    "foreign_first": int(foreign["first"]),
                    "foreign_last": int(foreign["last"]),
                    "foreign_frames": len(foreign["frames"]),
                    "terminal": terminal, "returns": returns})
    rows: list[dict] = []
    proposal = 0
    for left, right in combinations(candidates, 2):
        if (left["foreign_owner"] != right["foreign_owner"]
                or left["stable_owner"] == right["stable_owner"]
                or abs(left["foreign_first"] - right["foreign_first"])
                > maximum_start_delta):
            continue
        group_left, group_right = by_track[left["track"]], by_track[right["track"]]
        pre_last = min(left["foreign_first"], right["foreign_first"]) - 1
        pre_first = max(left["stable_first"], right["stable_first"],
                        pre_last - minimum_pre_coexist + 1)
        coexist = 0
        for frame in range(pre_first, pre_last + 1):
            lrow, rrow = _point(group_left, frame), _point(group_right, frame)
            if (lrow is not None and rrow is not None
                    and bool(lrow.physically_visible)
                    and bool(rrow.physically_visible)
                    and int(lrow.accepted_owner) == left["stable_owner"]
                    and int(rrow.accepted_owner) == right["stable_owner"]):
                coexist += 1
        first = min(left["foreign_first"], right["foreign_first"])
        last = max(left["foreign_last"], right["foreign_last"])
        close = residual = host_frames = 0
        for frame in range(first, last + 1):
            targets = []
            for item, group in ((left, group_left), (right, group_right)):
                row = _point(group, frame)
                if (row is not None and bool(row.physically_visible)
                        and int(row.accepted_owner) == left["foreign_owner"]):
                    targets.append(row)
            if not targets:
                continue
            component = invasion._component_near(
                labels[frame], left["foreign_owner"], targets[0])
            target_components = [
                invasion._component_near(
                    labels[frame], left["foreign_owner"], row)
                for row in targets[1:]]
            if component is None or any(
                    other is None or not np.any(component & other)
                    for other in target_components):
                continue
            host_frames += 1
            if len(targets) == 2:
                distance = float(np.hypot(
                    float(targets[0].x) - float(targets[1].x),
                    float(targets[0].y) - float(targets[1].y)))
                scale = max(float(targets[0].radius_px)
                            + float(targets[1].radius_px), 1.0)
                close += int(distance / scale <= maximum_pair_distance)
            marker, _ = _residual_marker(raw[frame], component, targets, params)
            residual += int(marker is not None)
        reasons: list[str] = []
        if coexist < minimum_pre_coexist:
            reasons.append("insufficient_distinct_preconvergence_tenure")
        if close < 1:
            reasons.append("tracks_do_not_converge")
        if residual < minimum_residual_frames or residual != host_frames:
            reasons.append("foreign_owner_residual_core_not_complete")
        proposal += 1
        rows.append({
            "proposal_id": f"P{proposal:04d}",
            "left_track": int(left["track"]),
            "right_track": int(right["track"]),
            "left_stable_owner": int(left["stable_owner"]),
            "right_stable_owner": int(right["stable_owner"]),
            "foreign_owner": int(left["foreign_owner"]),
            "first_frame": first, "last_frame": last,
            "preconvergence_coexistence_frames": coexist,
            "host_frames": host_frames, "close_pair_frames": close,
            "residual_core_frames": residual,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    columns = ["proposal_id", "left_track", "right_track",
               "left_stable_owner", "right_stable_owner", "foreign_owner",
               "first_frame", "last_frame", "preconvergence_coexistence_frames",
               "host_frames", "close_pair_frames", "residual_core_frames",
               "discovery_status", "discovery_reason"]
    return pd.DataFrame(rows, columns=columns), scored


def _excess_components(stack: np.ndarray) -> int:
    return component_accounting.total_component_excess(stack)


def _adjacent_owner(frame: np.ndarray, component: np.ndarray,
                    shard: np.ndarray, allowed: set[int],
                    fallback: int) -> int:
    """Choose an owner that joins the shard to an adjacent retained region."""
    ring = ndi.binary_dilation(shard, structure=STRUCTURE) & component & ~shard
    values = frame[ring]
    values = values[np.isin(values, list(allowed))]
    if values.size == 0:
        return int(fallback)
    owners, counts = np.unique(values.astype(np.int64), return_counts=True)
    return int(owners[int(np.argmax(counts))])


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict):
    proposals, scored = discover(labels, raw, points, params)
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    candidate = labels.copy()
    applications: list[dict] = []
    frames: list[dict] = []
    for proposal in proposals[proposals.discovery_status.eq("eligible")].sort_values(
            ["first_frame", "left_track"]).itertuples(index=False):
        trial = candidate.copy()
        local_rows: list[dict] = []
        reason = ""
        for frame in range(int(proposal.first_frame), int(proposal.last_frame) + 1):
            targets: list[tuple[object, int]] = []
            for track, owner in (
                    (int(proposal.left_track), int(proposal.left_stable_owner)),
                    (int(proposal.right_track), int(proposal.right_stable_owner))):
                row = _point(groups[track], frame)
                if (row is not None and bool(row.physically_visible)
                        and int(row.accepted_owner) == int(proposal.foreign_owner)):
                    targets.append((row, owner))
            if not targets:
                continue
            component = invasion._component_near(
                trial[frame], int(proposal.foreign_owner), targets[0][0])
            if component is None:
                reason = "foreign_owner_component_missing"
                break
            residual_marker, peak_ratio = _residual_marker(
                raw[frame], component, [item[0] for item in targets], params)
            if residual_marker is None:
                reason = "residual_foreign_owner_core_missing"
                break
            markers = np.zeros(component.shape, np.int16)
            marker_details: list[tuple[int, tuple[int, int], object, int]] = []
            for index, (row, owner) in enumerate(targets, 1):
                marker = physical._marker_pixel(
                    component, float(row.x), float(row.y))
                if marker is None or markers[marker] > 0:
                    reason = "target_marker_unavailable"
                    break
                markers[marker] = index
                marker_details.append((index, marker, row, owner))
            if reason:
                break
            if markers[residual_marker] > 0:
                reason = "residual_marker_collides_with_target"
                break
            residual_index = len(targets) + 1
            markers[residual_marker] = residual_index
            elevation = -ndi.gaussian_filter(
                raw[frame].astype(np.float32),
                float(params.get("watershed_sigma_px", 1.0)))
            basins = watershed(elevation, markers=markers, mask=component)
            before = trial[frame].copy()
            for index, _, row, owner in marker_details:
                partition = basins == index
                radius = float(np.clip(
                    0.75 * float(row.radius_px), 2.0, 6.0))
                yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
                partition |= component & (
                    (xx - float(row.x)) ** 2
                    + (yy - float(row.y)) ** 2 <= radius ** 2)
                trial[frame][partition] = owner
            residual_parts, residual_count = ndi.label(
                component & (trial[frame] == int(proposal.foreign_owner)),
                STRUCTURE)
            retained_part = int(residual_parts[residual_marker])
            if retained_part <= 0:
                reason = "foreign_owner_residual_marker_lost"
                break
            reassigned_shards = 0
            target_owners = {int(item[3]) for item in marker_details}
            for part in range(1, residual_count + 1):
                if part == retained_part:
                    continue
                shard = residual_parts == part
                owner = _adjacent_owner(
                    trial[frame], component, shard, target_owners,
                    marker_details[0][3])
                trial[frame][shard] = owner
                reassigned_shards += int(np.count_nonzero(shard))
            target_shards_reassigned = 0
            for _, marker, _, owner in marker_details:
                target_parts, target_count = ndi.label(
                    component & (trial[frame] == owner), STRUCTURE)
                retained_target = int(target_parts[marker])
                if retained_target <= 0:
                    reason = "stable_owner_marker_lost"
                    break
                for part in range(1, target_count + 1):
                    if part == retained_target:
                        continue
                    shard = target_parts == part
                    allowed = (target_owners - {int(owner)}) | {
                        int(proposal.foreign_owner)}
                    replacement = _adjacent_owner(
                        trial[frame], component, shard, allowed,
                        int(proposal.foreign_owner))
                    trial[frame][shard] = replacement
                    target_shards_reassigned += int(np.count_nonzero(shard))
            if reason:
                break
            if not np.any(trial[frame] == int(proposal.foreign_owner)):
                reason = "foreign_owner_residual_extinguished"
                break
            for _, _, row, owner in marker_details:
                if int(physical._disk_owner(
                        trial[frame], float(row.x), float(row.y),
                        float(row.radius_px))) != owner:
                    reason = "stable_owner_not_dominant_at_track_core"
                    break
            if reason:
                break
            local_rows.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "left_track": int(proposal.left_track),
                "right_track": int(proposal.right_track),
                "foreign_owner": int(proposal.foreign_owner),
                "target_seats_restored": len(targets),
                "residual_peak_fraction": peak_ratio,
                "target_shard_pixels_reassigned": target_shards_reassigned,
                "residual_shard_pixels_reassigned": reassigned_shards,
                "changed_pixels": int(np.count_nonzero(trial[frame] != before))})
        changed = trial != candidate
        if not reason and _excess_components(trial) > _excess_components(candidate):
            reason = "new_duplicate_identity_component"
        if reason:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "left_track": int(proposal.left_track),
                "right_track": int(proposal.right_track),
                "outcome": "rejected_application", "reason": reason,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        candidate = trial
        frames.extend(local_rows)
        applications.append({
            "proposal_id": proposal.proposal_id,
            "left_track": int(proposal.left_track),
            "right_track": int(proposal.right_track), "outcome": "applied",
            "reason": "two_owner_seats_preserved_through_convergence",
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2))))})
    columns = ["proposal_id", "left_track", "right_track", "outcome",
               "reason", "changed_pixels", "changed_frames"]
    return (candidate, proposals, pd.DataFrame(applications, columns=columns),
            pd.DataFrame(frames), scored)


def summarize(labels: np.ndarray, candidate: np.ndarray, unclaimed: np.ndarray,
              raw: np.ndarray, points: pd.DataFrame, proposals: pd.DataFrame,
              applications: pd.DataFrame, mode: str) -> dict:
    changed = candidate != labels
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    named_losses = 0
    for identity in before_ids:
        before = np.any(labels == identity, axis=(1, 2))
        after = np.any(candidate == identity, axis=(1, 2))
        named_losses += int(np.count_nonzero(before & ~after))
    return {
        "mode": mode, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "tracks_audited": int(points.track_id.nunique()),
        "pair_proposals_audited": int(len(proposals)),
        "eligible_terminal_convergences": int(proposals.discovery_status.eq(
            "eligible").sum()) if len(proposals) else 0,
        "applied_terminal_convergences": int(applications.outcome.eq(
            "applied").sum()) if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "unclaimed_ledger_changed_pixels": 0,
        "zero_signal_additions": int(np.count_nonzero(
            changed & (labels == 0) & (candidate > 0) & (raw <= 0))),
        "old_identity_set_preserved": before_ids == after_ids,
        "donor_named_frame_losses": named_losses,
        "new_same_frame_identity_components": max(
            0, _excess_components(candidate) - _excess_components(labels)),
    }
