"""Allocate fresh identities to field-discovered persistent released bodies."""
from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), np.uint8)


def _disk_owner(frame: np.ndarray, x: float, y: float,
                radius: float) -> tuple[int, float, float]:
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    height, width = frame.shape
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


def _attach_evidence(points: pd.DataFrame, labels: np.ndarray,
                     unclaimed: np.ndarray) -> pd.DataFrame:
    rows = points.copy()
    owner_evidence = [
        _disk_owner(labels[int(row.frame)], float(row.x), float(row.y),
                    float(row.radius_px))
        for row in rows.itertuples(index=False)]
    rows["candidate_owner"] = [item[0] for item in owner_evidence]
    rows["candidate_coverage_fraction"] = [item[1] for item in owner_evidence]
    rows["candidate_owner_purity"] = [item[2] for item in owner_evidence]
    rows["physically_visible"] = rows["state"].isin(
        ["observed", "latent_visible"])
    unclaimed_owner = []
    for row in rows.itertuples(index=False):
        frame = int(row.frame)
        y = int(np.clip(round(float(row.y)), 0, labels.shape[1] - 1))
        x = int(np.clip(round(float(row.x)), 0, labels.shape[2] - 1))
        unclaimed_owner.append(int(unclaimed[frame, y, x]))
    rows["unclaimed_owner"] = unclaimed_owner
    return rows


def _runs(group: pd.DataFrame, column: str) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        value = int(getattr(row, column))
        if (runs and runs[-1]["value"] == value and
                int(row.frame) == runs[-1]["last_frame"] + 1):
            runs[-1]["last_frame"] = int(row.frame)
            runs[-1]["rows"].append(row)
        else:
            runs.append({"value": value, "first_frame": int(row.frame),
                         "last_frame": int(row.frame), "rows": [row]})
    for run in runs:
        run["frames"] = len(run["rows"])
    return runs


def _component_at(frame: np.ndarray, value: int, x: float, y: float,
                  radius: float) -> np.ndarray | None:
    components, count = ndi.label(frame == value, STRUCTURE)
    if not count:
        return None
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= max(2.0, radius) ** 2
    values = components[disk & (components > 0)]
    if not len(values):
        return None
    ids, counts = np.unique(values, return_counts=True)
    return components == int(ids[np.argmax(counts)])


def _point_inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _bookend_owner_runs(owner_runs: list[dict], first_frame: int,
                        last_frame: int, owner: int,
                        minimum_frames: int) -> tuple[dict, dict] | None:
    before = [run for run in owner_runs
              if run["last_frame"] == first_frame - 1]
    after = [run for run in owner_runs
             if run["first_frame"] == last_frame + 1]
    if not before or not after:
        return None
    left, right = before[-1], after[0]
    if (left["value"] != owner or right["value"] != owner or
            left["frames"] < minimum_frames or
            right["frames"] < minimum_frames):
        return None
    return left, right


def _takeover_with_separate_predecessor(
        group: pd.DataFrame, all_visible: pd.DataFrame, owner_runs: list[dict],
        after_frame: int, seed_owner: int, minimum_frames: int,
        minimum_separation_radii: float,
        ) -> tuple[dict | None, int]:
    for run in owner_runs:
        takeover_owner = int(run["value"])
        if (run["first_frame"] <= after_frame or takeover_owner <= 0 or
                takeover_owner == seed_owner or run["frames"] < minimum_frames):
            continue
        first = int(run["first_frame"])
        current = group[group["frame"] == first]
        if not len(current):
            continue
        current_row = current.iloc[0]
        predecessors = all_visible[
            (all_visible["frame"] == first - 1) &
            (all_visible["track_id"] != int(current_row.track_id)) &
            (all_visible["candidate_owner"] == takeover_owner)]
        for predecessor in predecessors.itertuples(index=False):
            distance = float(np.hypot(
                float(current_row.x) - float(predecessor.x),
                float(current_row.y) - float(predecessor.y)))
            scale = max(float(current_row.radius_px) +
                        float(predecessor.radius_px), 1.0)
            if distance / scale >= minimum_separation_radii:
                return run, int(predecessor.track_id)
    return None, 0


def _lineage_components(
        labels: np.ndarray, unclaimed: np.ndarray, group: pd.DataFrame,
        all_visible: pd.DataFrame,
        ) -> tuple[list[tuple[int, np.ndarray, str, int]], int]:
    changes: list[tuple[int, np.ndarray, str, int]] = []
    refused = 0
    for row in group.sort_values("frame").itertuples(index=False):
        frame = int(row.frame)
        source = ""
        value = int(row.candidate_owner)
        component = None
        if value > 0:
            component = _component_at(
                labels[frame], value, row.x, row.y, row.radius_px)
            source = "assigned"
        if component is None and int(row.unclaimed_owner) > 0:
            value = int(row.unclaimed_owner)
            component = _component_at(
                unclaimed[frame], value, row.x, row.y, row.radius_px)
            source = "unclaimed"
        if component is None:
            continue
        frame_points = all_visible[all_visible["frame"] == frame]
        occupants = [
            int(other.track_id) for other in frame_points.itertuples(index=False)
            if _point_inside(component, other)]
        if any(track != int(row.track_id) for track in occupants):
            refused += 1
            continue
        changes.append((frame, component, source, value))
    return changes, refused


def allocate_persistent_released_lineages(
        labels: np.ndarray, unclaimed: np.ndarray,
        physical_points: pd.DataFrame, params: dict,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Discover ledger-seeded persistent lineages and allocate fresh IDs."""
    attached = _attach_evidence(physical_points, labels, unclaimed)
    visible = attached[attached["physically_visible"].astype(bool)].copy()
    minimum_seed = int(params.get("minimum_unclaimed_seed_frames", 3))
    minimum_bookend = int(params.get("minimum_same_owner_bookend_frames", 3))
    minimum_lifetime = int(np.ceil(
        len(labels) * float(params.get("minimum_visible_lifetime_fraction", 0.5))))
    minimum_takeover = int(np.ceil(
        len(labels) * float(params.get("minimum_takeover_run_fraction", 0.25))))
    minimum_separation = float(
        params.get("minimum_predecessor_separation_radii", 3.0))
    minimum_component_fraction = float(
        params.get("minimum_exclusive_component_fraction", 0.8))
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    next_identity = int(max(np.max(labels), np.max(unclaimed))) + 1
    audits: list[dict] = []
    proposals: list[dict] = []

    for track_id, group in visible.groupby("track_id"):
        group = group.sort_values("frame")
        if len(group) < minimum_lifetime:
            continue
        owner_runs = _runs(group, "candidate_owner")
        unclaimed_runs = [
            run for run in _runs(group, "unclaimed_owner")
            if run["value"] > 0]
        for seed in unclaimed_runs:
            base = {
                "physical_track": int(track_id),
                "seed_owner": int(seed["value"]),
                "seed_first_frame": int(seed["first_frame"]),
                "seed_last_frame": int(seed["last_frame"]),
                "applied": False, "new_identity": 0,
                "changed_label_pixels": 0,
                "changed_unclaimed_pixels": 0,
                "changed_frames": 0, "reason": ""}
            if seed["frames"] < minimum_seed:
                audits.append({**base, "reason": "unclaimed_seed_too_short"})
                continue
            bookends = _bookend_owner_runs(
                owner_runs, int(seed["first_frame"]), int(seed["last_frame"]),
                int(seed["value"]), minimum_bookend)
            if bookends is None:
                audits.append({**base, "reason": "same_owner_bookends_missing"})
                continue
            takeover, predecessor = _takeover_with_separate_predecessor(
                group, visible, owner_runs, int(bookends[1]["last_frame"]),
                int(seed["value"]), minimum_takeover, minimum_separation)
            if takeover is None:
                audits.append({**base,
                               "reason": "no_long_takeover_from_separate_predecessor"})
                continue
            changes, refused = _lineage_components(
                labels, unclaimed, group, visible)
            fraction = float(len(changes) / max(len(group), 1))
            if fraction < minimum_component_fraction:
                audits.append({
                    **base, "exclusive_component_fraction": fraction,
                    "refused_shared_components": refused,
                    "reason": "insufficient_exclusive_component_support"})
                continue
            if any(np.any(occupied[frame] & component)
                   for frame, component, _, _ in changes):
                audits.append({**base, "reason": "overlapping_proposal"})
                continue
            proposals.append({
                "base": base, "group": group, "changes": changes,
                "refused": refused, "fraction": fraction,
                "takeover_owner": int(takeover["value"]),
                "takeover_first_frame": int(takeover["first_frame"]),
                "predecessor_track": predecessor})

    proposals.sort(key=lambda proposal: (
        int(proposal["group"]["frame"].min()),
        float(proposal["group"]["y"].median()),
        float(proposal["group"]["x"].median()),
        int(proposal["base"]["physical_track"])))
    for proposal in proposals:
        new_identity = next_identity
        next_identity += 1
        label_changes = unclaimed_changes = 0
        frames = set()
        for frame, component, source, _ in proposal["changes"]:
            frames.add(frame)
            before_labels = candidate[frame].copy()
            before_unclaimed = candidate_unclaimed[frame].copy()
            candidate[frame][component] = new_identity
            candidate_unclaimed[frame][component] = 0
            occupied[frame] |= component
            label_changes += int(np.count_nonzero(
                candidate[frame] != before_labels))
            unclaimed_changes += int(np.count_nonzero(
                candidate_unclaimed[frame] != before_unclaimed))
        audits.append({
            **proposal["base"], "applied": True,
            "new_identity": new_identity,
            "takeover_owner": proposal["takeover_owner"],
            "takeover_first_frame": proposal["takeover_first_frame"],
            "predecessor_track": proposal["predecessor_track"],
            "visible_frames": int(len(proposal["group"])),
            "exclusive_component_frames": int(len(proposal["changes"])),
            "exclusive_component_fraction": proposal["fraction"],
            "refused_shared_components": proposal["refused"],
            "changed_label_pixels": label_changes,
            "changed_unclaimed_pixels": unclaimed_changes,
            "changed_frames": int(len(frames)), "reason": "applied"})
    return candidate, candidate_unclaimed, pd.DataFrame(audits)



