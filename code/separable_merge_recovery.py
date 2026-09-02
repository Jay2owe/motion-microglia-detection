"""Field-wide owner recovery after separable physical encounters.

The producer finds short A-B-A owner excursions on independently linked raw
signal tracks.  It has no recording, identity, frame, coordinate, event, or
review-case inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed


VISIBLE_STATES = {"observed", "latent_visible"}


def _mode(values: list[int]) -> tuple[int, int, float]:
    positive = np.asarray([int(value) for value in values if int(value) > 0])
    if not len(positive):
        return 0, 0, 0.0
    identities, counts = np.unique(positive, return_counts=True)
    index = int(np.argmax(counts))
    return (int(identities[index]), int(counts[index]),
            float(counts[index] / len(positive)))


def _disk_owner(frame: np.ndarray, x: float, y: float,
                radius: float) -> int:
    height, width = frame.shape
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(width, int(np.ceil(x + radius + 1)))
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(height, int(np.ceil(y + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    values = frame[y0:y1, x0:x1][
        (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2]
    positive = values[values > 0]
    if not len(positive):
        return 0
    identities, counts = np.unique(positive, return_counts=True)
    return int(identities[int(np.argmax(counts))])


def attach_owners(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach the locally dominant accepted owner to every physical point."""
    result = points.copy()
    result["physically_visible"] = result["state"].isin(VISIBLE_STATES)
    result["accepted_owner"] = [
        _disk_owner(labels[int(row.frame)], float(row.x), float(row.y),
                    float(row.radius_px))
        for row in result.itertuples(index=False)]
    return result


def canonical_owners(points: pd.DataFrame, minimum_support: int,
                     minimum_purity: float,
                     exclude_encounter_frames: bool = False,
                     ) -> dict[int, dict]:
    """Infer stable owners from the whole physical-track history."""
    stable = points[points["physically_visible"]]
    if exclude_encounter_frames:
        stable = stable[~stable["in_encounter"].astype(bool)]
    result: dict[int, dict] = {}
    for track_id, group in stable.groupby("track_id"):
        owner, support, purity = _mode(
            group["accepted_owner"].astype(int).tolist())
        if owner > 0 and support >= minimum_support and purity >= minimum_purity:
            result[int(track_id)] = {
                "owner": owner, "support": support, "purity": purity}
    return result


def _point_row(index: pd.DataFrame, track_id: int, frame: int):
    key = (int(track_id), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _component_at(component_labels: np.ndarray, x: float, y: float,
                  search_radius: int = 6) -> int:
    height, width = component_labels.shape
    cx, cy = int(round(x)), int(round(y))
    if 0 <= cy < height and 0 <= cx < width and component_labels[cy, cx] > 0:
        return int(component_labels[cy, cx])
    y0, y1 = max(0, cy - search_radius), min(height, cy + search_radius + 1)
    x0, x1 = max(0, cx - search_radius), min(width, cx + search_radius + 1)
    yy, xx = np.nonzero(component_labels[y0:y1, x0:x1] > 0)
    if not len(xx):
        return 0
    distances = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
    best = int(np.argmin(distances))
    return int(component_labels[yy[best] + y0, xx[best] + x0])


def _marker_pixel(mask: np.ndarray, x: float,
                  y: float) -> tuple[int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    best = int(np.argmin((xx - x) ** 2 + (yy - y) ** 2))
    return int(yy[best]), int(xx[best])


def _median_identity_area(labels: np.ndarray, identity: int) -> float:
    areas = np.count_nonzero(labels == int(identity), axis=(1, 2))
    positive = areas[areas > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def _valley_ratio(frame: np.ndarray, left: tuple[float, float],
                  right: tuple[float, float]) -> float:
    distance = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    samples = max(3, int(np.ceil(distance)) + 1)
    values = []
    for weight in np.linspace(0.0, 1.0, samples):
        x = int(round((1 - weight) * left[0] + weight * right[0]))
        y = int(round((1 - weight) * left[1] + weight * right[1]))
        y = int(np.clip(y, 0, frame.shape[0] - 1))
        x = int(np.clip(x, 0, frame.shape[1] - 1))
        values.append(float(frame[y, x]))
    endpoint = max(min(values[0], values[-1]), 1.0)
    interior = values[1:-1] if len(values) > 2 else values
    return float(np.clip(min(interior) / endpoint, 0.0, 2.0))


def _owner_runs(group: pd.DataFrame) -> list[dict]:
    rows = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.accepted_owner)
        if owner <= 0 or not bool(row.physically_visible):
            continue
        frame = int(row.frame)
        if rows and rows[-1]["owner"] == owner and frame == rows[-1]["end"] + 1:
            rows[-1]["end"] = frame
            rows[-1]["frames"].append(frame)
        else:
            rows.append({"owner": owner, "start": frame, "end": frame,
                         "frames": [frame]})
    return rows


def apply_post_encounter_memory(
        labels: np.ndarray, raw: np.ndarray, scored: pd.DataFrame,
        canonical: dict[int, dict], params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Repair bounded A-B-A owner excursions tied to a recent encounter."""
    candidate = labels.copy()
    visible = scored[scored["physically_visible"]].copy()
    index = visible.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    audit: list[dict] = []
    structure = np.ones((3, 3), np.uint8)
    lookback = int(params.get("post_encounter_lookback_frames", 7))
    minimum_pre = int(params.get("minimum_pre_owner_run_frames", 5))
    maximum_middle = int(params.get("maximum_post_owner_run_frames", 10))
    maximum_distance = float(params.get("maximum_companion_distance_px", 30.0))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.65))

    for track_id, group in visible.groupby("track_id"):
        runs = _owner_runs(group)
        for run_index in range(1, len(runs) - 1):
            before, middle, after = runs[run_index - 1:run_index + 2]
            base = {"track_id": int(track_id), "owner_before": before["owner"],
                    "owner_middle": middle["owner"], "owner_after": after["owner"],
                    "first_frame": middle["start"], "last_frame": middle["end"],
                    "applied": False, "changed_pixels": 0, "reason": ""}
            if before["owner"] != after["owner"] or before["owner"] == middle["owner"]:
                audit.append({**base, "reason": "not_bracketed_owner_excursion"})
                continue
            if len(before["frames"]) < minimum_pre or len(middle["frames"]) > maximum_middle:
                audit.append({**base, "reason": "owner_run_duration_out_of_range"})
                continue
            recent = group[
                group["frame"].between(middle["start"] - lookback,
                                       middle["start"] - 1) &
                group["in_encounter"].astype(bool)]
            if recent.empty:
                audit.append({**base, "reason": "no_recent_physical_encounter"})
                continue
            old_owner, new_owner = before["owner"], middle["owner"]
            if any(np.any(labels[frame] == old_owner) for frame in middle["frames"]):
                audit.append({**base, "reason": "old_owner_present_elsewhere"})
                continue

            frame_changes: list[tuple[int, np.ndarray, int, float]] = []
            refusal = ""
            for frame in middle["frames"]:
                row = _point_row(index, int(track_id), frame)
                if row is None:
                    refusal = "missing_excursion_track_point"
                    break
                host_mask = candidate[frame] == new_owner
                host_components, host_count_before = ndi.label(host_mask, structure)
                component = _component_at(host_components, row.x, row.y)
                if component <= 0:
                    refusal = "excursion_core_outside_new_owner"
                    break
                shared = host_components == component
                companions = []
                for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                    other_track = int(other.track_id)
                    if other_track == int(track_id):
                        continue
                    stable = canonical.get(other_track)
                    if stable is None or int(stable["owner"]) != new_owner:
                        continue
                    if _component_at(host_components, other.x, other.y) != component:
                        continue
                    distance = float(np.hypot(row.x - other.x, row.y - other.y))
                    if 4.0 <= distance <= maximum_distance:
                        companions.append((distance, other))
                if not companions:
                    refusal = "no_stable_companion_in_shared_mask"
                    break
                _, companion = min(companions, key=lambda value: value[0])
                valley = _valley_ratio(
                    raw[frame], (float(row.x), float(row.y)),
                    (float(companion.x), float(companion.y)))
                if valley > maximum_valley:
                    refusal = "raw_cores_not_separable"
                    break
                lost_marker = _marker_pixel(shared, row.x, row.y)
                host_marker = _marker_pixel(shared, companion.x, companion.y)
                if lost_marker is None or host_marker is None or lost_marker == host_marker:
                    refusal = "watershed_markers_unavailable"
                    break
                markers = np.zeros(labels.shape[1:], np.int16)
                markers[lost_marker] = 1
                markers[host_marker] = 2
                elevation = -ndi.gaussian_filter(
                    raw[frame].astype(np.float32),
                    float(params.get("watershed_sigma_px", 1.0)))
                partition = watershed(elevation, markers=markers, mask=shared)
                change = partition == 1
                trial = candidate[frame].copy()
                trial[change] = old_owner
                old_count = ndi.label(labels[frame] == old_owner, structure)[1]
                new_old_count = ndi.label(trial == old_owner, structure)[1]
                host_after, new_host_count = ndi.label(trial == new_owner, structure)
                core_components: set[int] = set()
                for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                    stable = canonical.get(int(other.track_id))
                    if stable is None or int(stable["owner"]) != new_owner:
                        continue
                    component_id = _component_at(host_after, other.x, other.y)
                    if component_id > 0:
                        core_components.add(component_id)
                maximum_fragment_ratio = float(
                    params.get("maximum_coreless_fragment_ratio", 0.25))
                absorbed = np.zeros_like(change)
                for component_id in range(1, new_host_count + 1):
                    if component_id in core_components:
                        continue
                    fragment = host_after == component_id
                    fragment_area = int(np.count_nonzero(fragment))
                    touches_recovery = bool(np.any(
                        ndi.binary_dilation(change, structure=structure) & fragment))
                    if (touches_recovery and fragment_area <=
                            maximum_fragment_ratio * max(
                                1, int(np.count_nonzero(change)))):
                        absorbed |= fragment
                if np.any(absorbed):
                    trial[absorbed] = old_owner
                    change |= absorbed
                    new_old_count = ndi.label(trial == old_owner, structure)[1]
                    new_host_count = ndi.label(trial == new_owner, structure)[1]
                if max(0, new_old_count - 1) > max(0, old_count - 1):
                    refusal = "new_old_owner_duplicate"
                    break
                if max(0, new_host_count - 1) > max(0, host_count_before - 1):
                    refusal = "new_host_duplicate"
                    break
                changed = int(np.count_nonzero(change))
                if changed < int(params.get("minimum_changed_pixels", 5)):
                    refusal = "recovered_region_too_small"
                    break
                frame_changes.append((frame, change, changed, valley))
            if refusal:
                audit.append({**base, "reason": refusal})
                continue
            for frame, change, _, _ in frame_changes:
                candidate[frame][change] = old_owner
            audit.append({
                **base, "applied": True,
                "changed_pixels": int(sum(item[2] for item in frame_changes)),
                "frames_changed": len(frame_changes),
                "maximum_valley_ratio": max(item[3] for item in frame_changes),
                "reason": "applied"})
    return candidate, pd.DataFrame(audit)


def recover_post_encounter_owners(
        labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
        params: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the accepted, target-free recovery rule."""
    scored = attach_owners(points, labels)
    canonical = canonical_owners(
        scored, int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_canonical_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    candidate, audit = apply_post_encounter_memory(
        labels, raw, scored, canonical, params)
    if len(audit):
        audit.insert(0, "proposal_type", "post_encounter_memory")
    return candidate, audit


def apply_per_frame_pair_recovery(
        labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
        encounters: pd.DataFrame, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Recover a missing stable owner when two raw cores remain separable."""
    candidate = labels.copy()
    scored = attach_owners(points, labels)
    canonical = canonical_owners(
        scored, int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_canonical_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", True)))
    index = scored.set_index(["track_id", "frame"], drop=False)
    audit: list[dict] = []
    processed: set[tuple[int, int, int]] = set()
    area_cache: dict[int, float] = {}
    structure = (ndi.generate_binary_structure(2, 1)
                 if bool(params.get("four_connected_components", False))
                 else np.ones((3, 3), np.uint8))

    if not bool(params.get("enable_pair_recovery", True)):
        encounters = encounters.iloc[0:0]
    for event in encounters.sort_values(
            ["frame", "encounter_id"]).itertuples(index=False):
        frame = int(event.frame)
        base = {
            "encounter_id": int(event.encounter_id), "frame": frame,
            "track_a": int(event.track_a), "track_b": int(event.track_b),
            "distance_px": float(event.distance_px),
            "valley_ratio": float(event.valley_ratio),
            "separable": bool(event.separable), "applied": False,
            "changed_pixels": 0, "reason": "",
        }
        if not bool(event.separable):
            audit.append({**base, "reason": "raw_pair_not_separable"})
            continue
        row_a = _point_row(index, int(event.track_a), frame)
        row_b = _point_row(index, int(event.track_b), frame)
        ca = canonical.get(int(event.track_a))
        cb = canonical.get(int(event.track_b))
        if row_a is None or row_b is None or ca is None or cb is None:
            audit.append({**base, "reason": "missing_stable_track_support"})
            continue
        if not bool(row_a.physically_visible) or not bool(
                row_b.physically_visible):
            audit.append({**base, "reason": "physical_core_not_visible"})
            continue
        if ca["owner"] == cb["owner"]:
            audit.append({**base, "reason": "same_canonical_owner"})
            continue

        owner_a = _disk_owner(
            candidate[frame], row_a.x, row_a.y, row_a.radius_px)
        owner_b = _disk_owner(
            candidate[frame], row_b.x, row_b.y, row_b.radius_px)
        if owner_a == owner_b == cb["owner"]:
            lost, host = row_a, row_b
            lost_track, host_track = int(event.track_a), int(event.track_b)
            lost_owner, host_owner = int(ca["owner"]), int(cb["owner"])
        elif owner_a == owner_b == ca["owner"]:
            lost, host = row_b, row_a
            lost_track, host_track = int(event.track_b), int(event.track_a)
            lost_owner, host_owner = int(cb["owner"]), int(ca["owner"])
        else:
            audit.append({
                **base, "reason": "owners_not_collapsed_to_companion"})
            continue
        detail = {
            **base, "lost_track": lost_track, "host_track": host_track,
            "lost_owner": lost_owner, "host_owner": host_owner,
            "lost_support": canonical[lost_track]["support"],
            "host_support": canonical[host_track]["support"],
        }
        if np.any(candidate[frame] == lost_owner):
            audit.append({**detail, "reason": "lost_owner_present_elsewhere"})
            continue

        host_components, _ = ndi.label(
            candidate[frame] == host_owner, structure)
        lost_component = _component_at(host_components, lost.x, lost.y)
        host_component = _component_at(host_components, host.x, host.y)
        if lost_component <= 0 or host_component <= 0:
            audit.append({**detail, "reason": "core_outside_host_mask"})
            continue
        key = (frame, host_owner, lost_component)
        if key in processed:
            audit.append({
                **detail, "reason": "host_component_already_processed"})
            continue
        target_mask = host_components == lost_component
        target_area = int(np.count_nonzero(target_mask))
        median_area = area_cache.setdefault(
            lost_owner, _median_identity_area(labels, lost_owner))
        ratio = target_area / max(median_area, 1.0)
        detail.update({
            "target_area": target_area,
            "lost_owner_median_area": median_area,
            "area_ratio": ratio,
        })
        if not (float(params.get("minimum_area_ratio", 0.25)) <= ratio <=
                float(params.get("maximum_area_ratio", 4.0))):
            audit.append({
                **detail, "reason": "component_area_out_of_range"})
            continue

        if lost_component != host_component:
            change = target_mask
            mode = "separate_component_relabel"
        else:
            if bool(params.get("separate_components_only", False)):
                audit.append({**detail, "reason": "shared_component_excluded"})
                continue
            lost_marker = _marker_pixel(target_mask, lost.x, lost.y)
            host_marker = _marker_pixel(target_mask, host.x, host.y)
            if (lost_marker is None or host_marker is None
                    or lost_marker == host_marker):
                audit.append({
                    **detail, "reason": "watershed_markers_unavailable"})
                continue
            markers = np.zeros(labels.shape[1:], np.int16)
            markers[lost_marker] = 1
            markers[host_marker] = 2
            elevation = -ndi.gaussian_filter(
                raw[frame].astype(np.float32),
                float(params.get("watershed_sigma_px", 1.0)))
            partition = watershed(
                elevation, markers=markers, mask=target_mask)
            change = partition == 1
            if not np.any(partition == 2):
                audit.append({**detail, "reason": "watershed_removed_host"})
                continue
            mode = "raw_watershed_partition"
        changed = int(np.count_nonzero(change))
        if changed < int(params.get("minimum_changed_pixels", 5)):
            audit.append({
                **detail, "reason": "recovered_region_too_small"})
            continue
        candidate[frame][change] = lost_owner
        processed.add(key)
        audit.append({
            **detail, "applied": True, "changed_pixels": changed,
            "application_mode": mode, "reason": "applied",
        })
    return candidate, pd.DataFrame(audit)


def _new_excess_components(before: np.ndarray, after: np.ndarray,
                           identity: int) -> int:
    structure = np.ones((3, 3), np.uint8)
    before_count = ndi.label(before == int(identity), structure)[1]
    after_count = ndi.label(after == int(identity), structure)[1]
    return max(0, max(0, after_count - 1) - max(0, before_count - 1))


def absorb_coreless_host_fragments(
        before: np.ndarray, after: np.ndarray, points: pd.DataFrame,
        audit: pd.DataFrame, maximum_fragment_ratio: float,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Absorb unsupported host fragments, or revert an unsafe frame."""
    candidate = after.copy()
    result = audit.copy()
    structure = np.ones((3, 3), np.uint8)
    point_index = points.set_index(["track_id", "frame"], drop=False)
    applied_rows = result.index[
        result.get("applied", pd.Series(False, index=result.index)).astype(bool)]
    for index in applied_rows:
        row = result.loc[index]
        frame = int(row.frame)
        lost_owner = int(float(row.lost_owner))
        host_owner = int(float(row.host_owner))
        host_track = int(float(row.host_track))
        key = (host_track, frame)
        if key not in point_index.index:
            candidate[frame] = before[frame]
            result.loc[index, ["applied", "changed_pixels", "reason"]] = [
                False, 0, "host_track_point_missing"]
            continue
        point = point_index.loc[key]
        if isinstance(point, pd.DataFrame):
            point = point.iloc[0]
        host_components, host_count = ndi.label(
            candidate[frame] == host_owner, structure)
        supported = _component_at(
            host_components, float(point.x), float(point.y))
        recovered = ((candidate[frame] == lost_owner)
                     & (before[frame] == host_owner))
        recovered_area = int(np.count_nonzero(recovered))
        absorbed = np.zeros_like(recovered)
        for component in range(1, host_count + 1):
            if component == supported:
                continue
            fragment = host_components == component
            area = int(np.count_nonzero(fragment))
            touches = bool(np.any(
                ndi.binary_dilation(recovered, structure=structure)
                & fragment))
            if (touches and area <= maximum_fragment_ratio
                    * max(1, recovered_area)):
                absorbed |= fragment
        if np.any(absorbed):
            candidate[frame][absorbed] = lost_owner
        unsafe = (
            _new_excess_components(
                before[frame], candidate[frame], host_owner)
            or _new_excess_components(
                before[frame], candidate[frame], lost_owner))
        if unsafe:
            candidate[frame] = before[frame]
            result.loc[index, ["applied", "changed_pixels", "reason"]] = [
                False, 0, "new_duplicate_component"]
            continue
        result.loc[index, "changed_pixels"] = int(
            np.count_nonzero(candidate[frame] != before[frame]))
        result.loc[index, "absorbed_coreless_pixels"] = int(
            np.count_nonzero(absorbed))
        result.loc[index, "reason"] = (
            "applied_fragment_safe" if np.any(absorbed) else "applied")
    return candidate, result


def exclude_crowded_pair_recoveries(
        before: np.ndarray, after: np.ndarray, points: pd.DataFrame,
        audit: pd.DataFrame, minimum_third_track_clearance: float,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Revert a pair episode when a third strong body is too close."""
    candidate = after.copy()
    result = audit.copy()
    if result.empty or "applied" not in result:
        return candidate, result
    applied = result[result["applied"].astype(bool)].copy()
    if applied.empty:
        return candidate, result
    applied["pair"] = applied.apply(
        lambda row: tuple(sorted((int(row.track_a), int(row.track_b)))),
        axis=1)
    point_index = points.set_index(["track_id", "frame"], drop=False)
    blocked_pair_frames: set[int] = set()
    for (track_a, track_b), rows in applied.groupby("pair"):
        pair_frames = sorted(set(rows.frame.astype(int)))
        crowded = False
        for frame in pair_frames:
            keys = ((track_a, frame), (track_b, frame))
            if any(key not in point_index.index for key in keys):
                crowded = True
                break
            pair_points = []
            for key in keys:
                point = point_index.loc[key]
                if isinstance(point, pd.DataFrame):
                    point = point.iloc[0]
                pair_points.append(point)
            first, second = pair_points
            first_xy = np.array([float(first.x), float(first.y)])
            second_xy = np.array([float(second.x), float(second.y)])
            pair_distance = float(np.linalg.norm(first_xy - second_xy))
            if pair_distance <= 0:
                crowded = True
                break
            midpoint = (first_xy + second_xy) / 2.0
            frame_points = points[
                points.frame.eq(frame)
                & ~points.track_id.isin((track_a, track_b))
                & points.strong.astype(bool)]
            if frame_points.empty:
                continue
            distances = np.linalg.norm(
                frame_points[["x", "y"]].to_numpy(float) - midpoint,
                axis=1)
            if float(np.min(distances)) < (
                    minimum_third_track_clearance * pair_distance):
                crowded = True
                break
        if crowded:
            blocked_pair_frames.update(pair_frames)

    if blocked_pair_frames:
        frames = list(sorted(blocked_pair_frames))
        candidate[frames] = before[frames]
        mask = (result["applied"].astype(bool)
                & result.frame.astype(int).isin(blocked_pair_frames))
        result.loc[mask, "applied"] = False
        result.loc[mask, "changed_pixels"] = 0
        result.loc[mask, "reason"] = "crowded_three_body_context"
    return candidate, result


def recover_isolated_pair_encounters(
        labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
        encounters: pd.DataFrame, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Run target-free pair recovery with fragmentation and crowding guards."""
    candidate, audit = apply_per_frame_pair_recovery(
        labels, raw, points, encounters, params)
    candidate, audit = absorb_coreless_host_fragments(
        labels, candidate, points, audit,
        float(params.get("maximum_coreless_fragment_ratio", 0.25)))
    candidate, audit = exclude_crowded_pair_recoveries(
        labels, candidate, points, audit,
        float(params.get(
            "minimum_third_strong_track_clearance_pair_distances", 2.0)))
    return candidate, audit
