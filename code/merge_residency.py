"""Recover stable physical residents displaced during connected merges."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

STRUCTURE = np.ones((3, 3), np.uint8)


def _disk_owner(frame: np.ndarray, x: float, y: float,
                radius: float) -> tuple[int, float, float]:
    height, width = frame.shape
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    x0, x1 = max(0, int(np.floor(x - radius))), min(
        width, int(np.ceil(x + radius + 1)))
    y0, y1 = max(0, int(np.floor(y - radius))), min(
        height, int(np.ceil(y + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    values = frame[y0:y1, x0:x1][disk]
    positive = values[values > 0]
    coverage = float(len(positive) / max(len(values), 1))
    if not len(positive):
        return 0, coverage, 0.0
    owners, counts = np.unique(positive, return_counts=True)
    best = int(np.argmax(counts))
    return int(owners[best]), coverage, float(counts[best] / len(positive))


def _attach_owners(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = points.copy()
    evidence = [_disk_owner(labels[int(row.frame)], float(row.x), float(row.y),
                            float(row.radius_px))
                for row in rows.itertuples(index=False)]
    rows["candidate_owner"] = [item[0] for item in evidence]
    rows["candidate_coverage_fraction"] = [item[1] for item in evidence]
    rows["candidate_owner_purity"] = [item[2] for item in evidence]
    rows["physically_visible"] = rows["state"].isin(
        ["observed", "latent_visible"])
    return rows


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.candidate_owner)
        if (runs and runs[-1]["owner"] == owner and
                int(row.frame) == runs[-1]["last_frame"] + 1):
            runs[-1]["last_frame"] = int(row.frame)
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first_frame": int(row.frame),
                         "last_frame": int(row.frame), "rows": [row]})
    for run in runs:
        run["frames"] = len(run["rows"])
    return runs


def _component_at(frame: np.ndarray, owner: int, x: float, y: float,
                  radius: float) -> np.ndarray | None:
    components, count = ndi.label(frame == owner, STRUCTURE)
    if not count:
        return None
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= max(2.0, radius) ** 2
    values = components[disk & (components > 0)]
    if not len(values):
        return None
    ids, counts = np.unique(values, return_counts=True)
    return components == int(ids[np.argmax(counts)])


def _marker(mask: np.ndarray, x: float, y: float) -> tuple[int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    index = int(np.argmin((xx - x) ** 2 + (yy - y) ** 2))
    return int(yy[index]), int(xx[index])


def _valley_ratio(frame: np.ndarray, left: tuple[float, float],
                  right: tuple[float, float]) -> float:
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


def _point_inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _canonical_owners(points: pd.DataFrame, minimum_purity: float,
                      minimum_coverage: float) -> tuple[dict[int, int], dict[int, int]]:
    usable = points[
        points["physically_visible"].astype(bool) &
        (points["candidate_owner"] > 0) &
        (points["candidate_owner_purity"] >= minimum_purity) &
        (points["candidate_coverage_fraction"] >= minimum_coverage)]
    owner: dict[int, int] = {}
    support: dict[int, int] = {}
    for track_id, group in usable.groupby("track_id"):
        counts = group.groupby("candidate_owner").size()
        best = min(counts.index, key=lambda item: (-int(counts[item]), int(item)))
        owner[int(track_id)] = int(best)
        support[int(track_id)] = int(counts[best])
    return owner, support


def _median_track_component_area(labels: np.ndarray, rows: list,
                                 owner: int) -> float:
    values = []
    for row in rows:
        component = _component_at(
            labels[int(row.frame)], owner, float(row.x), float(row.y),
            float(row.radius_px))
        if component is not None:
            values.append(int(component.sum()))
    return float(np.median(values)) if values else 0.0


def _conflicts_are_displaced(
        labels: np.ndarray, points: pd.DataFrame, canonical: dict[int, int],
        resident_track: int, resident_owner: int, first_frame: int,
        conflict_tracks: set[int], minimum_bookend_frames: int,
        ) -> bool:
    if not conflict_tracks:
        return True
    count = 0
    for frame in range(max(0, first_frame - minimum_bookend_frames), first_frame):
        frame_points = points[points["frame"] == frame]
        resident_rows = frame_points[frame_points["track_id"] == resident_track]
        if not len(resident_rows):
            continue
        resident = resident_rows.iloc[0]
        if int(resident.candidate_owner) != resident_owner:
            continue
        resident_component = _component_at(
            labels[frame], resident_owner, resident.x, resident.y,
            resident.radius_px)
        if resident_component is None:
            continue
        proved = True
        for track_id in conflict_tracks:
            rows = frame_points[frame_points["track_id"] == track_id]
            if not len(rows) or int(rows.iloc[0].candidate_owner) != resident_owner:
                proved = False
                break
            other = rows.iloc[0]
            component = _component_at(
                labels[frame], resident_owner, other.x, other.y,
                other.radius_px)
            if component is None or np.any(component & resident_component):
                proved = False
                break
            if canonical.get(track_id, resident_owner) == resident_owner:
                proved = False
                break
        count += int(proved)
    return count >= minimum_bookend_frames


def recover_merge_residents(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        physical_points: pd.DataFrame, params: dict,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply complete merge-resident transactions discovered from the field."""
    attached = _attach_owners(physical_points, labels)
    visible = attached[attached["physically_visible"].astype(bool)].copy()
    minimum_purity = float(params.get("minimum_owner_purity", 0.8))
    minimum_coverage = float(params.get("minimum_owner_coverage", 0.5))
    canonical, support = _canonical_owners(
        attached, minimum_purity, minimum_coverage)
    minimum_support = int(params.get("minimum_canonical_support_frames", 8))
    minimum_resident_support = int(
        params.get("minimum_resident_owner_support_frames", 5))
    minimum_pre = int(params.get("minimum_pre_merge_owner_frames", 5))
    maximum_run = max(1, int(np.ceil(
        len(labels) * float(params.get("maximum_merge_run_fraction", 0.05)))))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.9))
    minimum_area_ratio = float(params.get("minimum_partition_area_ratio", 0.25))
    maximum_area_ratio = float(params.get("maximum_partition_area_ratio", 2.5))
    minimum_combined = float(
        params.get("minimum_merged_to_combined_area_ratio", 0.5))
    bookend_frames = int(params.get("minimum_displaced_duplicate_bookend_frames", 3))
    sigma = float(params.get("watershed_sigma_px", 1.0))
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    audits: list[dict] = []

    for track_id, group in visible.groupby("track_id"):
        track_id = int(track_id)
        resident_owner = canonical.get(track_id, 0)
        if (resident_owner <= 0 or
                support.get(track_id, 0) < minimum_resident_support):
            continue
        runs = _runs(group)
        for index, run in enumerate(runs):
            source_owner = int(run["owner"])
            base = {
                "resident_track": track_id, "resident_owner": resident_owner,
                "host_owner": source_owner, "first_frame": run["first_frame"],
                "last_frame": run["last_frame"], "applied": False,
                "changed_pixels": 0, "released_pixels": 0, "reason": ""}
            if source_owner <= 0 or source_owner == resident_owner:
                continue
            if run["frames"] > maximum_run or index == 0:
                audits.append({**base, "reason": "source_run_out_of_range"})
                continue
            previous = runs[index - 1]
            if (int(previous["owner"]) != resident_owner or
                    int(previous["frames"]) < minimum_pre or
                    int(previous["last_frame"]) + 1 != int(run["first_frame"])):
                audits.append({**base, "reason": "no_stable_adjacent_pre_merge_owner"})
                continue
            resident_area = _median_track_component_area(
                labels, previous["rows"], resident_owner)
            frame_changes = []
            refusal = ""
            all_conflict_tracks: set[int] = set()
            for resident_row in run["rows"]:
                frame = int(resident_row.frame)
                frame_points = visible[visible["frame"] == frame]
                shared = _component_at(
                    labels[frame], source_owner, resident_row.x, resident_row.y,
                    resident_row.radius_px)
                if shared is None:
                    refusal = "resident_not_in_host_component"
                    break
                host_rows = []
                for other in frame_points.itertuples(index=False):
                    other_track = int(other.track_id)
                    if (other_track != track_id and
                            canonical.get(other_track, 0) == source_owner and
                            support.get(other_track, 0) >= minimum_support and
                            int(other.candidate_owner) == source_owner and
                            _point_inside(shared, other)):
                        host_rows.append(other)
                if not host_rows:
                    refusal = "no_stable_host_resident_in_shared_component"
                    break
                host = min(host_rows, key=lambda row: (
                    np.hypot(float(row.x) - float(resident_row.x),
                             float(row.y) - float(resident_row.y)),
                    int(row.track_id)))
                host_group = visible[visible["track_id"] == int(host.track_id)]
                host_area = _median_track_component_area(
                    labels, list(host_group.itertuples(index=False)), source_owner)
                if shared.sum() < minimum_combined * (resident_area + host_area):
                    refusal = "shared_component_too_small_for_residents"
                    break
                valley = _valley_ratio(
                    raw[frame], (resident_row.x, resident_row.y),
                    (host.x, host.y))
                if valley > maximum_valley:
                    refusal = "raw_valley_not_separable"
                    break
                resident_marker = _marker(shared, resident_row.x, resident_row.y)
                host_marker = _marker(shared, host.x, host.y)
                if (resident_marker is None or host_marker is None or
                        resident_marker == host_marker):
                    refusal = "resident_markers_unavailable"
                    break
                markers = np.zeros(labels.shape[1:], np.int16)
                markers[resident_marker] = 1
                markers[host_marker] = 2
                elevation = -ndi.gaussian_filter(
                    raw[frame].astype(np.float32), sigma)
                partition = watershed(
                    elevation, markers=markers, mask=shared,
                    connectivity=STRUCTURE)
                resident_part = partition == 1
                host_part = partition == 2
                if (ndi.label(resident_part, STRUCTURE)[1] != 1 or
                        ndi.label(host_part, STRUCTURE)[1] != 1):
                    refusal = "partition_not_two_connected_residents"
                    break
                resident_ratio = float(resident_part.sum() / max(resident_area, 1.0))
                host_ratio = float(host_part.sum() / max(host_area, 1.0))
                if not (minimum_area_ratio <= resident_ratio <= maximum_area_ratio and
                        minimum_area_ratio <= host_ratio <= maximum_area_ratio):
                    refusal = "partition_area_out_of_range"
                    break

                conflict_components, count = ndi.label(
                    labels[frame] == resident_owner, STRUCTURE)
                released = np.zeros(labels.shape[1:], bool)
                conflict_tracks: set[int] = set()
                for component_id in range(1, count + 1):
                    component = conflict_components == component_id
                    occupants = [
                        point for point in frame_points.itertuples(index=False)
                        if _point_inside(component, point)]
                    if not occupants:
                        refusal = "owner_conflict_has_no_physical_track"
                        break
                    divergent = [
                        int(point.track_id) for point in occupants
                        if canonical.get(int(point.track_id), resident_owner) !=
                        resident_owner]
                    if not divergent:
                        refusal = "owner_conflict_has_canonical_resident"
                        break
                    released |= component
                    conflict_tracks.update(divergent)
                if refusal:
                    break
                if not conflict_tracks:
                    refusal = "no_displaced_owner_conflict"
                    break
                if np.any(occupied[frame] & (resident_part | released)):
                    refusal = "proposal_overlaps_earlier_transaction"
                    break
                all_conflict_tracks.update(conflict_tracks)
                frame_changes.append((frame, resident_part, released, valley))
            if refusal or len(frame_changes) != int(run["frames"]):
                audits.append({**base, "reason": refusal or "incomplete_merge_run"})
                continue
            if not _conflicts_are_displaced(
                    labels, attached, canonical, track_id, resident_owner,
                    int(run["first_frame"]), all_conflict_tracks,
                    bookend_frames):
                audits.append({**base,
                               "reason": "conflict_not_preexisting_displaced_duplicate"})
                continue
            changed = released_pixels = 0
            for frame, resident_part, released, _ in frame_changes:
                candidate[frame][released] = 0
                candidate_unclaimed[frame][released] = resident_owner
                candidate[frame][resident_part] = resident_owner
                candidate_unclaimed[frame][resident_part] = 0
                occupied[frame] |= resident_part | released
                changed += int(resident_part.sum())
                released_pixels += int(released.sum())
            audits.append({
                **base, "applied": True,
                "changed_pixels": int(changed + released_pixels),
                "released_pixels": released_pixels,
                "changed_frames": int(run["frames"]),
                "maximum_raw_valley_ratio": float(max(
                    item[3] for item in frame_changes)),
                "displaced_track_count": int(len(all_conflict_tracks)),
                "reason": "applied"})
    return candidate, candidate_unclaimed, pd.DataFrame(audits)



