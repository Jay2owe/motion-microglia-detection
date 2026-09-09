"""Preserve a long-lived physical core through a one-owner shared component.

Discovery is complete-field and target-free. Identity, track, frame, event and
coordinate values in the audit are measured outputs, never supplied selectors.
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
FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "forced_identity", "forced_interval", "allowed_pair",
)


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("distinct-history discovery must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "distinct-history discovery received forbidden targets: "
            + ", ".join(supplied))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _runs(group: pd.DataFrame, maximum_blank_frames: int) -> list[dict]:
    """Positive-owner runs, joining same-owner observations across short gaps."""
    rows: list[dict] = []
    visible = group[group.physically_visible.map(_truth)].sort_values("frame")
    for row in visible.itertuples(index=False):
        owner, frame = int(row.accepted_owner), int(row.frame)
        if owner <= 0:
            continue
        if (rows and rows[-1]["owner"] == owner
                and frame <= rows[-1]["end"] + maximum_blank_frames + 1):
            rows[-1]["end"] = frame
            rows[-1]["observations"].append(row)
        else:
            rows.append({"owner": owner, "start": frame, "end": frame,
                         "observations": [row]})
    return rows


def _component_at(parts: np.ndarray, row: object,
                  search_radius: int = 6) -> int:
    return physical._component_at(
        parts, float(row.x), float(row.y), search_radius)


def _marker(mask: np.ndarray, row: object) -> tuple[int, int] | None:
    return physical._marker_pixel(mask, float(row.x), float(row.y))


def _identity_area(labels: np.ndarray, owner: int) -> float:
    areas = np.count_nonzero(labels == int(owner), axis=(1, 2))
    positive = areas[areas > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def _seed_component(partition: np.ndarray,
                    seed: tuple[int, int]) -> np.ndarray:
    parts, _ = ndi.label(partition, STRUCTURE)
    component = int(parts[seed])
    return parts == component if component else np.zeros_like(partition)


def _valid_partition(mask: np.ndarray, victim: np.ndarray,
                     resident: np.ndarray, victim_seed: tuple[int, int],
                     resident_seed: tuple[int, int], expected_victim: float,
                     expected_resident: float, params: dict,
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    options: list[tuple[float, np.ndarray, np.ndarray]] = []
    minimum = int(params.get("minimum_basin_pixels", 5))
    lower = float(params.get("minimum_expected_area_ratio", 0.10))
    upper = float(params.get("maximum_expected_area_ratio", 4.0))

    def add(victim_part: np.ndarray, resident_part: np.ndarray) -> None:
        if not victim_part[victim_seed] or not resident_part[resident_seed]:
            return
        if victim_part.sum() < minimum or resident_part.sum() < minimum:
            return
        if ndi.label(victim_part, STRUCTURE)[1] != 1:
            return
        if ndi.label(resident_part, STRUCTURE)[1] != 1:
            return
        vr = float(victim_part.sum()) / max(expected_victim, 1.0)
        rr = float(resident_part.sum()) / max(expected_resident, 1.0)
        if not (lower <= vr <= upper and lower <= rr <= upper):
            return
        error = abs(np.log(max(vr, 1e-9))) + abs(np.log(max(rr, 1e-9)))
        options.append((float(error), victim_part, resident_part))

    victim_core = _seed_component(victim, victim_seed)
    add(victim_core, mask & ~victim_core)
    resident_core = _seed_component(resident, resident_seed)
    add(mask & ~resident_core, resident_core)
    if not options:
        return None
    _, best_victim, best_resident = min(options, key=lambda item: item[0])
    if not np.array_equal(best_victim | best_resident, mask):
        raise AssertionError("two-seed partition is not exhaustive")
    return best_victim, best_resident


def _partition(mask: np.ndarray, raw: np.ndarray, victim: object,
               resident: object, expected_victim: float,
               expected_resident: float, params: dict,
               ) -> tuple[np.ndarray, np.ndarray, str] | None:
    victim_seed, resident_seed = _marker(mask, victim), _marker(mask, resident)
    if victim_seed is None or resident_seed is None or victim_seed == resident_seed:
        return None
    if np.hypot(victim_seed[0] - resident_seed[0],
                victim_seed[1] - resident_seed[1]) < float(
                    params.get("minimum_seed_distance_px", 4.0)):
        return None
    markers = np.zeros(mask.shape, np.uint8)
    markers[victim_seed], markers[resident_seed] = 1, 2
    smooth = ndi.gaussian_filter(
        raw.astype(np.float32), float(params.get("watershed_sigma_px", 1.0)))
    raw_split = watershed(-smooth, markers=markers, mask=mask,
                          connectivity=STRUCTURE.astype(bool))
    connected = _valid_partition(
        mask, raw_split == 1, raw_split == 2, victim_seed, resident_seed,
        expected_victim, expected_resident, params)
    if connected is not None:
        return (*connected, "raw_watershed_connected")

    yy, xx = np.nonzero(mask)
    victim_radius = max(float(np.sqrt(expected_victim / np.pi)), 1.0)
    resident_radius = max(float(np.sqrt(expected_resident / np.pi)), 1.0)
    victim_distance = np.hypot(
        yy - float(victim.y), xx - float(victim.x)) / victim_radius
    resident_distance = np.hypot(
        yy - float(resident.y), xx - float(resident.x)) / resident_radius
    geometric_victim = np.zeros_like(mask)
    geometric_victim[yy[victim_distance <= resident_distance],
                     xx[victim_distance <= resident_distance]] = True
    connected = _valid_partition(
        mask, geometric_victim, mask & ~geometric_victim,
        victim_seed, resident_seed, expected_victim, expected_resident, params)
    return None if connected is None else (*connected, "geodesic_seed_fallback")


def _valley(raw: np.ndarray, left: object, right: object) -> float:
    return physical._valley_ratio(
        raw, (float(left.x), float(left.y)),
        (float(right.x), float(right.y)))


def _same_shared_component(frame: np.ndarray, owner: int, left: object,
                           right: object) -> tuple[np.ndarray | None, int]:
    parts, _ = ndi.label(frame == int(owner), STRUCTURE)
    left_part = _component_at(parts, left)
    right_part = _component_at(parts, right)
    if left_part <= 0 or left_part != right_part:
        return None, 0
    return parts == left_part, left_part


def _resident_candidates(frame: np.ndarray, points: pd.DataFrame,
                         victim: object, victim_track: int,
                         resident_owner: int, params: dict) -> list[dict]:
    rows: list[dict] = []
    victim_radius = float(params["victim_radius"])
    for row in points.itertuples(index=False):
        if (int(row.track_id) == int(victim_track)
                or int(row.accepted_owner) != int(resident_owner)
                or not _truth(row.physically_visible) or not _truth(row.strong)):
            continue
        shared, component = _same_shared_component(
            frame, resident_owner, victim, row)
        if shared is None:
            continue
        distance = float(np.hypot(
            float(victim.x) - float(row.x),
            float(victim.y) - float(row.y)))
        separation = distance / max(victim_radius + float(row.radius_px), 1.0)
        rows.append({"row": row, "component": component, "mask": shared,
                     "distance": distance, "separation": separation})
    return rows


def discover_and_apply(labels: np.ndarray, raw: np.ndarray,
                       points: pd.DataFrame, params: dict,
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Audit all owner transitions and atomically recover eligible core runs."""
    assert_target_free(params)
    candidate = labels.copy()
    scored = physical.attach_owners(points, labels)
    scored["physically_visible"] = scored.physically_visible.map(_truth)
    index = scored.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in scored.groupby("frame")}
    frame_count = len(labels)
    minimum_prior = max(2, int(np.ceil(
        frame_count * float(params.get("minimum_prior_movie_fraction", 0.08)))))
    maximum_blank = max(0, int(np.ceil(
        frame_count * float(params.get("maximum_prior_blank_fraction", 0.02)))))
    maximum_transition_blank = max(
        0, int(params.get("maximum_transition_blank_frames", 0)))
    maximum_valley = float(params.get("maximum_two_core_valley_ratio", 0.65))
    minimum_initial_separation = float(params.get(
        "minimum_initial_separation_sum_radii", 1.5))
    minimum_followup_separation = float(params.get(
        "minimum_followup_separation_sum_radii", 1.0))
    proposal_rows: list[dict] = []
    frame_rows: list[dict] = []
    reserved: set[tuple[int, int, int]] = set()
    proposal_number = 0

    for victim_track, group in scored.groupby("track_id", sort=True):
        runs = _runs(group, maximum_blank)
        for before, after in zip(runs[:-1], runs[1:]):
            if before["owner"] == after["owner"]:
                continue
            proposal_number += 1
            proposal_id = f"DH{proposal_number:04d}"
            victim_owner, resident_owner = int(before["owner"]), int(after["owner"])
            start = int(after["start"])
            prior = group[(group.frame.astype(int) < start)
                          & group.physically_visible.astype(bool)]
            positive = prior[prior.accepted_owner.astype(int) > 0]
            victim_prior = positive[
                positive.accepted_owner.astype(int).eq(victim_owner)]
            support = len(victim_prior)
            purity = support / max(len(positive), 1)
            strong_fraction = float(victim_prior.strong.map(_truth).mean()) \
                if support else 0.0
            prior_frames = np.sort(victim_prior.frame.astype(int).unique())
            maximum_gap = int(np.diff(prior_frames).max()) if len(prior_frames) > 1 else 0
            victim_radius = float(np.median(victim_prior.radius_px)) \
                if support else 0.0
            base = {
                "proposal_id": proposal_id, "victim_track": int(victim_track),
                "victim_owner": victim_owner, "resident_owner": resident_owner,
                "transition_frame": start, "prior_support": support,
                "prior_purity": purity, "prior_strong_fraction": strong_fraction,
                "prior_maximum_observation_gap": maximum_gap,
                "minimum_prior_support": minimum_prior,
                "eligible": False, "applied": False, "changed_pixels": 0,
                "changed_frames": 0, "reason": "",
            }
            reasons: list[str] = []
            if support < minimum_prior:
                reasons.append("insufficient_prior_tenure")
            if purity < float(params.get("minimum_prior_owner_purity", 0.90)):
                reasons.append("impure_prior_owner_history")
            if strong_fraction < float(params.get("minimum_prior_strong_fraction", 0.90)):
                reasons.append("weak_prior_raw_support")
            if maximum_gap > maximum_blank + 1:
                reasons.append("discontinuous_prior_core")
            if start - int(before["end"]) - 1 > maximum_transition_blank:
                reasons.append("transition_not_temporally_contiguous")
            first_victim = _point(index, int(victim_track), start)
            if first_victim is None or not _truth(first_victim.physically_visible):
                reasons.append("victim_core_not_visible_at_transition")
            if np.any(labels[start] == victim_owner):
                reasons.append("victim_owner_present_elsewhere")
            companions: list[dict] = []
            if first_victim is not None and not reasons:
                local = by_frame.get(start, scored.iloc[0:0])
                companions = _resident_candidates(
                    labels[start], local, first_victim, int(victim_track),
                    resident_owner, {**params, "victim_radius": victim_radius})
                companions = [item for item in companions
                              if item["separation"] >= minimum_initial_separation
                              and _valley(raw[start], first_victim, item["row"])
                              <= maximum_valley]
                if len(companions) != 1:
                    reasons.append("resident_core_not_unique_or_not_two_basin")
            if reasons:
                proposal_rows.append({**base, "reason": "|".join(reasons)})
                continue

            companion_track = int(companions[0]["row"].track_id)
            expected_victim = _identity_area(labels, victim_owner)
            expected_resident = _identity_area(labels, resident_owner)
            pending: list[dict] = []
            stop_reason = "victim_or_resident_core_disappeared"
            for frame in range(start, frame_count):
                victim = _point(index, int(victim_track), frame)
                resident = _point(index, companion_track, frame)
                if (victim is None or resident is None
                        or not _truth(victim.physically_visible)
                        or not _truth(resident.physically_visible)
                        or int(victim.accepted_owner) != resident_owner
                        or int(resident.accepted_owner) != resident_owner):
                    break
                if np.any(labels[frame] == victim_owner):
                    stop_reason = "victim_owner_reappeared_or_separated"
                    break
                shared, component = _same_shared_component(
                    labels[frame], resident_owner, victim, resident)
                if shared is None:
                    stop_reason = "accepted_component_separated"
                    break
                key = (frame, resident_owner, component)
                if key in reserved:
                    stop_reason = "shared_component_already_reserved"
                    break
                distance = float(np.hypot(
                    float(victim.x) - float(resident.x),
                    float(victim.y) - float(resident.y)))
                separation = distance / max(
                    victim_radius + float(resident.radius_px), 1.0)
                valley = _valley(raw[frame], victim, resident)
                if separation < minimum_followup_separation or valley > maximum_valley:
                    stop_reason = "two_basin_evidence_ended"
                    break
                partition = _partition(
                    shared, raw[frame], victim, resident,
                    expected_victim, expected_resident, params)
                if partition is None:
                    stop_reason = "connected_partition_unavailable"
                    break
                victim_basin, resident_basin, method = partition
                trial = candidate[frame].copy()
                trial[shared] = 0
                trial[victim_basin] = victim_owner
                trial[resident_basin] = resident_owner
                if owner_consensus.count_new_duplicate_components(
                        labels[frame:frame + 1], trial[None]) > 0:
                    stop_reason = "new_duplicate_component"
                    break
                pending.append({
                    "proposal_id": proposal_id, "frame": frame,
                    "victim_track": int(victim_track),
                    "resident_track": companion_track,
                    "victim_owner": victim_owner,
                    "resident_owner": resident_owner,
                    "separation_sum_radii": separation,
                    "valley_ratio": valley, "partition_method": method,
                    "victim_pixels": int(victim_basin.sum()),
                    "resident_pixels": int(resident_basin.sum()),
                    "changed_pixels": int(victim_basin.sum()),
                })
            if not pending:
                proposal_rows.append({
                    **base, "eligible": True, "resident_track": companion_track,
                    "initial_separation_sum_radii": companions[0]["separation"],
                    "reason": stop_reason})
                continue
            for item in pending:
                frame = int(item["frame"])
                victim = _point(index, int(victim_track), frame)
                resident = _point(index, companion_track, frame)
                shared, component = _same_shared_component(
                    labels[frame], resident_owner, victim, resident)
                partition = _partition(
                    shared, raw[frame], victim, resident,
                    expected_victim, expected_resident, params)
                victim_basin, resident_basin, _ = partition
                candidate[frame][shared] = 0
                candidate[frame][victim_basin] = victim_owner
                candidate[frame][resident_basin] = resident_owner
                reserved.add((frame, resident_owner, component))
                frame_rows.append(item)
            proposal_rows.append({
                **base, "eligible": True, "applied": True,
                "resident_track": companion_track,
                "initial_separation_sum_radii": companions[0]["separation"],
                "changed_pixels": int(sum(item["changed_pixels"] for item in pending)),
                "changed_frames": len(pending), "reason": "applied_until_" + stop_reason,
            })

    proposals = pd.DataFrame(proposal_rows)
    frames = pd.DataFrame(frame_rows)
    return candidate, proposals, frames


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    mode = str(params.get("mode", "candidate"))
    if mode == "baseline":
        candidate = labels.copy()
        proposals = pd.DataFrame()
        frames = pd.DataFrame()
    elif mode == "candidate":
        candidate, proposals, frames = discover_and_apply(
            labels, raw, points, params)
    else:
        raise ValueError("mode must be baseline or candidate")
    candidate_unclaimed = unclaimed.copy()
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("distinct-history recovery changed foreground")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("distinct-history recovery changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError(f"distinct-history recovery created {duplicates} duplicates")
    stem = str(params.get("output_stem", "95_A3"))
    labels_path = out.out / f"{stem}.tif"
    unclaimed_path = out.out / f"{stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(labels_path, candidate, compression="zlib")
    tifffile.imwrite(unclaimed_path, candidate_unclaimed, compression="zlib")
    proposal_path = out.out / "distinct_history_proposals.csv"
    frame_path = out.out / "distinct_history_frames.csv"
    proposals.to_csv(proposal_path, index=False)
    frames.to_csv(frame_path, index=False)
    for name in ("application_audit", "conflicted_lineage_audit"):
        source = params.get(f"{name}_path")
        if source:
            shutil.copy2(source, out.out / f"{name}.csv")
    applied = proposals[proposals.get(
        "applied", pd.Series(False, index=proposals.index)).astype(bool)] \
        if len(proposals) else proposals
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {name: 0 for name in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "transitions_audited": int(len(proposals)),
        "eligible_transitions": int(proposals.get(
            "eligible", pd.Series(dtype=bool)).sum()) if len(proposals) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.any(candidate != labels, axis=(1, 2)).sum()),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "unclaimed_changed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(duplicates),
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs = {"labels": labels_path, "unclaimed": unclaimed_path,
               "proposals": proposal_path, "frames": frame_path,
               "metrics": metrics_path}
    for name in ("application_audit", "conflicted_lineage_audit"):
        path = out.out / f"{name}.csv"
        if path.is_file():
            outputs[name] = path
    return {"outputs": outputs, "summary": metrics}

