"""Safely preserve distinct-history cores after terminal owner diversion.

Discovery is complete-field and target-free. The rule only repairs an
unambiguous terminal-newcomer topology, and every proposal is applied
atomically after accepted-audit, third-body, raw-core, and duplicate guards.
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

import distinct_history_transition_candidates as a001
import owner_consensus


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_EXACT = frozenset({
    "target", "targets", "identity", "identities", "identity_id",
    "identity_ids", "owner", "owners", "owner_id", "owner_ids", "cell",
    "cells", "cell_id", "cell_ids", "track", "tracks", "track_id",
    "track_ids", "frame", "frames", "frame_id", "frame_ids", "coordinate",
    "coordinates", "location", "locations", "event", "events", "event_id",
    "event_ids", "region", "regions", "roi", "rois", "review_case",
    "review_cases", "case_id", "case_ids", "bbox", "bounding_box",
})
FORBIDDEN_FRAGMENTS = (
    "target_", "_target", "selector", "problem_", "review_case", "case_id",
    "allowed_pair", "forced_identity", "forced_interval", "include_track",
    "exclude_track", "include_frame", "exclude_frame", "include_identity",
    "exclude_identity", "include_owner", "exclude_owner", "coordinate_list",
    "location_list", "crop_box", "included_", "excluded_",
    "track_id", "owner_id", "identity_id", "cell_id", "frame_id",
    "event_id", "roi",
)
FORBIDDEN_ROLE_SUFFIXES = (
    "_track", "_owner", "_identity", "_cell", "_frame", "_event",
    "_region", "_roi",
)


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("distinct-history discovery must be field-wide")
    supplied = []
    for key, value in params.items():
        if value in (None, "", [], {}):
            continue
        name = str(key).strip().lower()
        if (name in FORBIDDEN_EXACT
                or any(token in name for token in FORBIDDEN_FRAGMENTS)
                or name.endswith(FORBIDDEN_ROLE_SUFFIXES)):
            supplied.append(str(key))
    if supplied:
        raise ValueError("distinct-history discovery received forbidden "
                         "selectors: " + ", ".join(sorted(supplied)))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _protected(paths: dict) -> tuple[set[int], set[int]]:
    tracks: set[int] = set()
    identities: set[int] = set()
    application_path = paths.get("application_audit_path")
    if application_path:
        table = pd.read_csv(application_path)
        if "application_status" in table and "track_id" in table:
            applied = table[table.application_status.astype(str).eq("applied")]
            tracks.update(applied.track_id.dropna().astype(int))
            if "assigned_identity" in applied:
                identities.update(value for value in
                                  applied.assigned_identity.dropna().astype(int)
                                  if value > 0)
    conflict_path = paths.get("conflicted_lineage_audit_path")
    if conflict_path:
        table = pd.read_csv(conflict_path)
        if len(table):
            outcome = (table.outcome.astype(str).eq("applied")
                       if "outcome" in table else pd.Series(False, index=table.index))
            assigned = (pd.to_numeric(table.assigned_identity, errors="coerce") > 0
                        if "assigned_identity" in table else
                        pd.Series(False, index=table.index))
            accepted = table[outcome | assigned]
            if "physical_track" in accepted:
                tracks.update(accepted.physical_track.dropna().astype(int))
            if "assigned_identity" in accepted:
                identities.update(value for value in
                                  accepted.assigned_identity.dropna().astype(int)
                                  if value > 0)
    return tracks, identities


def protected_assignments(paths: dict) -> tuple[set[int], set[int]]:
    """Return accepted tracks and identities that proposals must not alter."""
    return _protected(paths)


def _sample_pair(image: np.ndarray, left: object, right: object,
                 sigma: float) -> dict:
    smooth = ndi.gaussian_filter(image.astype(np.float32), sigma)
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    samples = max(5, 2 * int(np.ceil(distance)) + 1)
    xs = np.linspace(float(left.x), float(right.x), samples)
    ys = np.linspace(float(left.y), float(right.y), samples)
    values = ndi.map_coordinates(smooth, [ys, xs], order=1, mode="nearest")
    low, high = float(min(values[0], values[-1])), float(max(values[0], values[-1]))
    interior = values[1:-1] if len(values) > 2 else values
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return {
        "separation_sum_radii": distance / scale,
        "valley_ratio": float(np.min(interior) / max(low, 1.0)),
        "endpoint_balance": low / max(high, 1.0),
        "both_strong": _truth(left.strong) and _truth(right.strong),
    }


def _raw_partition(mask: np.ndarray, raw: np.ndarray, victim: object,
                   resident: object, params: dict,
                   ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    victim_seed = a001._marker(mask, victim)
    resident_seed = a001._marker(mask, resident)
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
    split = watershed(-smooth, markers=markers, mask=mask,
                      connectivity=STRUCTURE.astype(bool))
    victim_basin, resident_basin = split == 1, split == 2
    minimum_pixels = int(params.get("minimum_basin_pixels", 5))
    if (victim_basin.sum() < minimum_pixels
            or resident_basin.sum() < minimum_pixels
            or ndi.label(victim_basin, STRUCTURE)[1] != 1
            or ndi.label(resident_basin, STRUCTURE)[1] != 1
            or not np.array_equal(victim_basin | resident_basin, mask)):
        return None
    victim_expected = float(victim.radius_px) ** 2
    resident_expected = float(resident.radius_px) ** 2
    expected_total = max(victim_expected + resident_expected, 1.0)
    actual_total = max(float(mask.sum()), 1.0)
    ratios = {
        "victim_radius_area_ratio": (
            float(victim_basin.sum()) / actual_total)
            / max(victim_expected / expected_total, 1e-9),
        "resident_radius_area_ratio": (
            float(resident_basin.sum()) / actual_total)
            / max(resident_expected / expected_total, 1e-9),
    }
    lower = float(params.get("minimum_radius_area_ratio", 0.15))
    upper = float(params.get("maximum_radius_area_ratio", 3.0))
    if not all(lower <= value <= upper for value in ratios.values()):
        return None
    return victim_basin, resident_basin, ratios


def _maximum_consecutive(values: list[int]) -> int:
    best = current = 0
    previous = None
    for value in sorted(set(values)):
        current = current + 1 if previous is not None and value == previous + 1 else 1
        best, previous = max(best, current), value
    return best


def discover_and_apply(labels: np.ndarray, raw: np.ndarray,
                       points: pd.DataFrame, params: dict,
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    _, broad_proposals, broad_frames = a001.discover_and_apply(
        labels, raw, points, params)
    scored = a001.physical.attach_owners(points, labels)
    scored["physically_visible"] = scored.physically_visible.map(_truth)
    indexed = scored.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in scored.groupby("frame")}
    owner_sets = {
        int(track): set(group.loc[
            group.physically_visible & group.accepted_owner.astype(int).gt(0),
            "accepted_owner"].astype(int))
        for track, group in scored.groupby("track_id")
    }
    protected_tracks, protected_identities = _protected(params)
    proposal_rows: list[dict] = []
    candidate_plans: list[dict] = []
    empty_audit = {
        "accepted_applied_track_conflict": False,
        "accepted_assigned_identity_conflict": False,
        "accepted_history_third_owner": False,
        "third_physical_track_in_component": False,
        "third_identity_in_component": False,
        "third_physical_tracks": "", "third_physical_track_count": 0,
        "third_identities": "", "third_identity_count": 0,
        "qualified_two_core_frames": 0,
        "maximum_consecutive_two_core_frames": 0,
        "minimum_endpoint_balance_observed": np.nan,
        "partition_method": "raw_watershed_only",
        "resident_pretransition_visible_frames": 0,
        "victim_postinterval_visible_frames": 0,
        "later_same_track_victim_owner_frames": 0,
        "pre_pair_bookend": False,
        "post_pair_bookend": False,
        "terminal_newcomer_topology": False,
        "global_atomic_conflict": False,
    }
    for proposal in broad_proposals.itertuples(index=False):
        base = {name: getattr(proposal, name) for name in broad_proposals.columns}
        base.update(empty_audit)
        base.update({"eligible": False, "applied": False,
                     "changed_pixels": 0, "changed_frames": 0})
        if not bool(getattr(proposal, "applied", False)):
            base["reason"] = "broad_precondition_rejected:" + str(proposal.reason)
            proposal_rows.append(base)
            continue
        victim_track = int(proposal.victim_track)
        resident_track = int(proposal.resident_track)
        victim_owner = int(proposal.victim_owner)
        resident_owner = int(proposal.resident_owner)
        reasons: list[str] = []
        track_conflict = bool({victim_track, resident_track} & protected_tracks)
        identity_conflict = bool(
            {victim_owner, resident_owner} & protected_identities)
        history_conflict = bool(
            owner_sets.get(victim_track, set()) != {victim_owner, resident_owner}
            or owner_sets.get(resident_track, set()) != {resident_owner})
        base["accepted_applied_track_conflict"] = track_conflict
        base["accepted_assigned_identity_conflict"] = identity_conflict
        base["accepted_history_third_owner"] = history_conflict
        if track_conflict:
            reasons.append("accepted_applied_track_conflict")
        if identity_conflict:
            reasons.append("accepted_assigned_identity_conflict")
        if history_conflict:
            reasons.append("accepted_history_third_owner")
        frame_table = broad_frames[
            broad_frames.proposal_id.astype(str).eq(str(proposal.proposal_id))]
        transition_start = int(frame_table.frame.min())
        interval_end = int(frame_table.frame.max())
        victim_history = scored[
            scored.track_id.astype(int).eq(victim_track)
            & scored.physically_visible]
        resident_history = scored[
            scored.track_id.astype(int).eq(resident_track)
            & scored.physically_visible]
        resident_pre = resident_history[
            resident_history.frame.astype(int).lt(transition_start)]
        victim_post = victim_history[
            victim_history.frame.astype(int).gt(interval_end)]
        later_victim_owner = victim_post[
            victim_post.accepted_owner.astype(int).eq(victim_owner)]
        pre_victim = a001._point(indexed, victim_track, transition_start - 1)
        pre_resident = a001._point(indexed, resident_track, transition_start - 1)
        post_victim = a001._point(indexed, victim_track, interval_end + 1)
        post_resident = a001._point(indexed, resident_track, interval_end + 1)
        base["resident_pretransition_visible_frames"] = int(
            resident_pre.frame.nunique())
        base["victim_postinterval_visible_frames"] = int(
            victim_post.frame.nunique())
        base["later_same_track_victim_owner_frames"] = int(
            later_victim_owner.frame.nunique())
        base["pre_pair_bookend"] = bool(
            pre_victim is not None and pre_resident is not None
            and _truth(pre_victim.physically_visible)
            and _truth(pre_resident.physically_visible)
            and int(pre_victim.accepted_owner) == victim_owner
            and int(pre_resident.accepted_owner) == resident_owner)
        base["post_pair_bookend"] = bool(
            post_victim is not None and post_resident is not None
            and _truth(post_victim.physically_visible)
            and _truth(post_resident.physically_visible)
            and int(post_victim.accepted_owner) == victim_owner
            and int(post_resident.accepted_owner) == resident_owner)
        resident_first = int(resident_history.frame.min()) \
            if len(resident_history) else -1
        victim_last = int(victim_history.frame.max()) \
            if len(victim_history) else -1
        terminal_newcomer = bool(
            resident_first == transition_start
            and victim_last == interval_end
            and not len(later_victim_owner))
        base["terminal_newcomer_topology"] = terminal_newcomer
        if not terminal_newcomer:
            reasons.append("not_terminal_newcomer_diversion")
            if base["pre_pair_bookend"] or base["post_pair_bookend"] or \
                    len(later_victim_owner):
                reasons.append("resolved_bookended_encounter")
        evidence: list[dict] = []
        partitions: list[dict] = []
        third_tracks: set[int] = set()
        third_identities: set[int] = set()
        for frame_row in frame_table.itertuples(index=False):
            frame = int(frame_row.frame)
            victim = a001._point(indexed, victim_track, frame)
            resident = a001._point(indexed, resident_track, frame)
            if victim is None or resident is None:
                reasons.append("missing_pair_point")
                continue
            foreground_parts, _ = ndi.label(labels[frame] > 0, STRUCTURE)
            victim_part = a001._component_at(foreground_parts, victim)
            resident_part = a001._component_at(foreground_parts, resident)
            if victim_part <= 0 or victim_part != resident_part:
                reasons.append("pair_not_in_one_foreground_component")
                continue
            foreground = foreground_parts == victim_part
            local_tracks = set()
            for row in by_frame.get(frame, scored.iloc[0:0]).itertuples(index=False):
                if not _truth(row.physically_visible):
                    continue
                if a001._component_at(foreground_parts, row) == victim_part:
                    local_tracks.add(int(row.track_id))
            third_tracks.update(local_tracks - {victim_track, resident_track})
            local_ids = set(map(int, np.unique(labels[frame][foreground]))) - {0}
            third_identities.update(local_ids - {victim_owner, resident_owner})
            evidence.append({"frame": frame, **_sample_pair(
                raw[frame], victim, resident,
                float(params.get("raw_evidence_sigma_px", 1.0)))})
            shared, _ = a001._same_shared_component(
                labels[frame], resident_owner, victim, resident)
            if shared is None:
                reasons.append("accepted_pair_not_in_one_owner_component")
                continue
            partition = _raw_partition(
                shared, raw[frame], victim, resident, params)
            if partition is None:
                reasons.append("raw_watershed_partition_unavailable")
                continue
            victim_basin, resident_basin, ratios = partition
            partitions.append({
                "proposal_id": str(proposal.proposal_id), "frame": frame,
                "victim_track": victim_track, "resident_track": resident_track,
                "victim_owner": victim_owner, "resident_owner": resident_owner,
                "victim_basin": victim_basin, "resident_basin": resident_basin,
                "shared": shared, **evidence[-1], **ratios,
                "partition_method": "raw_watershed_connected",
                "victim_pixels": int(victim_basin.sum()),
                "resident_pixels": int(resident_basin.sum()),
                "changed_pixels": int(victim_basin.sum()),
            })
        base["third_physical_tracks"] = "|".join(map(str, sorted(third_tracks)))
        base["third_physical_track_count"] = len(third_tracks)
        base["third_identities"] = "|".join(map(str, sorted(third_identities)))
        base["third_identity_count"] = len(third_identities)
        base["third_physical_track_in_component"] = bool(third_tracks)
        base["third_identity_in_component"] = bool(third_identities)
        if third_tracks:
            reasons.append("third_physical_track_in_component")
        if third_identities:
            reasons.append("third_identity_in_component")
        qualified = [item for item in evidence if
                     item["both_strong"]
                     and item["separation_sum_radii"] >= float(params.get(
                         "minimum_two_core_evidence_separation_sum_radii", 1.0))
                     and item["valley_ratio"] <= float(params.get(
                         "maximum_two_core_evidence_valley_ratio", 0.80))
                     and item["endpoint_balance"] >= float(params.get(
                         "minimum_two_core_endpoint_balance", 0.15))]
        qualified_frames = [int(item["frame"]) for item in qualified]
        base["qualified_two_core_frames"] = len(set(qualified_frames))
        base["maximum_consecutive_two_core_frames"] = _maximum_consecutive(
            qualified_frames)
        base["minimum_endpoint_balance_observed"] = (
            min((item["endpoint_balance"] for item in evidence), default=np.nan))
        required = int(params.get("minimum_two_core_evidence_frames", 2))
        if base["maximum_consecutive_two_core_frames"] < required:
            reasons.append("insufficient_multiframe_two_core_evidence")
        if len(partitions) != len(frame_table):
            reasons.append("not_all_frames_raw_partitionable")
        if reasons:
            base["reason"] = "|".join(sorted(set(reasons)))
            proposal_rows.append(base)
            continue
        base.update({
            "eligible": True, "reason": "eligible_pending_global_atomic_gate",
            "changed_pixels": int(sum(item["changed_pixels"] for item in partitions)),
            "changed_frames": len(partitions),
        })
        candidate_plans.append({"base": base, "partitions": partitions})
        proposal_rows.append(base)

    # Reject every proposal participating in a pixel/component claim conflict.
    conflicts: dict[str, set[str]] = {}
    claims: list[tuple[str, int, np.ndarray]] = []
    for plan in candidate_plans:
        proposal_id = str(plan["base"]["proposal_id"])
        for item in plan["partitions"]:
            for other_id, other_frame, other_mask in claims:
                if int(item["frame"]) == other_frame and np.any(
                        item["shared"] & other_mask):
                    conflicts.setdefault(proposal_id, set()).add(other_id)
                    conflicts.setdefault(other_id, set()).add(proposal_id)
            claims.append((proposal_id, int(item["frame"]), item["shared"]))
    candidate = labels.copy()
    applied_frames: list[dict] = []
    for plan in candidate_plans:
        proposal_id = str(plan["base"]["proposal_id"])
        row = next(item for item in proposal_rows
                   if str(item["proposal_id"]) == proposal_id)
        if proposal_id in conflicts:
            row.update({
                "eligible": False, "applied": False,
                "changed_pixels": 0, "changed_frames": 0,
                "global_atomic_conflict": True,
                "reason": "global_atomic_conflict_with:" + "|".join(
                    sorted(conflicts[proposal_id])),
            })
            continue
        for item in plan["partitions"]:
            frame = int(item["frame"])
            candidate[frame][item["shared"]] = 0
            candidate[frame][item["victim_basin"]] = int(item["victim_owner"])
            candidate[frame][item["resident_basin"]] = int(item["resident_owner"])
            applied_frames.append({key: value for key, value in item.items()
                                   if key not in {"victim_basin", "resident_basin",
                                                  "shared"}})
        row.update({"applied": True, "reason": "applied_after_global_atomic_gate"})

    invariant_failure = (
        not np.array_equal(candidate > 0, labels > 0)
        or (set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))))
        or owner_consensus.count_new_duplicate_components(labels, candidate) > 0)
    if invariant_failure:
        candidate = labels.copy()
        applied_frames = []
        for row in proposal_rows:
            if row.get("applied"):
                row.update({"eligible": False, "applied": False,
                            "changed_pixels": 0, "changed_frames": 0,
                            "global_atomic_conflict": True,
                            "reason": "global_atomic_invariant_rejection"})
    return candidate, pd.DataFrame(proposal_rows), pd.DataFrame(applied_frames)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, proposals, frames = discover_and_apply(labels, raw, points, params)
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("distinct-history preservation changed foreground")
    if before_ids != after_ids:
        raise AssertionError("distinct-history preservation changed the identity set")
    if duplicates:
        raise AssertionError("distinct-history preservation created a duplicate component")
    stem = str(params.get("output_stem", "95_A3"))
    labels_path = out.out / f"{stem}.tif"
    unclaimed_path = out.out / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(labels_path, candidate, compression="zlib")
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    proposal_path = out.out / "distinct_history_proposals.csv"
    frame_path = out.out / "distinct_history_frames.csv"
    proposals.to_csv(proposal_path, index=False)
    frames.to_csv(frame_path, index=False)
    for name in ("application_audit", "conflicted_lineage_audit"):
        source = params.get(f"{name}_path")
        if source:
            shutil.copy2(source, out.out / f"{name}.csv")
    applied = proposals[proposals.applied.fillna(False).astype(bool)] \
        if len(proposals) else proposals
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {name: 0 for name in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "transitions_audited": int(len(proposals)),
        "eligible_transitions": int(proposals.eligible.fillna(False).sum())
            if len(proposals) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.any(candidate != labels, axis=(1, 2)).sum()),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "unclaimed_changed_pixels": 0,
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(duplicates),
        "accepted_protected_tracks": len(_protected(params)[0]),
        "accepted_protected_identities": len(_protected(params)[1]),
        "globally_conflicted_proposals": int(
            proposals.global_atomic_conflict.fillna(False).sum()) if len(proposals) else 0,
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
