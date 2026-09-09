"""Preserve established physical seats through a persistent owner collapse.

Discovery is field-wide. Identities, tracks, frames, coordinates, events,
regions, and review cases are measured outputs and cannot select proposals.
An already-present old owner is allowed only when every such component is a
small, nearby, track-supported projection without a foreign owner history.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import distinct_history_transition_candidates as base
import owner_consensus
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
ALLOWED_PARAM_KEYS = {
    "mode", "targeting_mode", "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path", "output_stem",
    "minimum_prior_movie_fraction", "minimum_prior_owner_purity",
    "minimum_prior_strong_fraction", "maximum_prior_blank_fraction",
    "maximum_transition_blank_movie_fraction",
    "minimum_collapse_movie_fraction", "minimum_resident_movie_fraction",
    "minimum_resident_owner_purity", "minimum_resident_strong_fraction",
    "minimum_replacement_resident_strong_fraction",
    "minimum_collapsed_strong_fraction",
    "minimum_initial_separation_sum_radii",
    "minimum_followup_separation_sum_radii",
    "maximum_two_core_valley_ratio", "minimum_seed_distance_px",
    "watershed_sigma_px", "minimum_basin_pixels",
    "minimum_expected_area_ratio", "maximum_expected_area_ratio",
    "maximum_projection_area_fraction",
    "maximum_projection_distance_sum_radii",
}
REQUIRED_PATH_KEYS = {
    "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path",
}
AUDIT_COLUMNS = [
    "proposal_id", "victim_track", "victim_owner", "resident_track",
    "resident_owner", "transition_frame", "collapse_last_frame",
    "prior_support", "prior_purity", "prior_strong_fraction",
    "resident_support", "resident_purity", "resident_strong_fraction",
    "collapse_frames", "collapse_strong_fraction",
    "initial_separation_sum_radii", "maximum_valley_ratio",
    "projection_frames", "projection_components", "changed_pixels",
    "changed_frames", "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "victim_track", "victim_owner",
    "resident_track", "resident_owner", "separation_sum_radii",
    "valley_ratio", "partition_method", "victim_pixels",
    "resident_pixels", "changed_pixels", "projection_components",
]
PROJECTION_COLUMNS = [
    "proposal_id", "frame", "component", "pixels",
    "area_fraction_of_established_seat", "distance_sum_radii",
    "supporting_tracks", "prior_positive_owners",
    "lineage_compatible", "reason",
]


def selector_counts(params: dict) -> dict[str, int]:
    unexpected = [str(key) for key in params
                  if str(key) not in ALLOWED_PARAM_KEYS]
    roles = {
        "identities": ("identity", "identities", "cell", "cells"),
        "owners": ("owner", "owners"),
        "tracks": ("track", "tracks"),
        "frames": ("frame", "frames", "interval"),
        "coordinates": ("coordinate", "coordinates", "point", "points"),
        "events": ("event", "events"),
        "regions": ("region", "regions", "roi"),
        "review_cases": ("case", "cases", "review"),
    }
    return {role: sum(any(token in key.lower() for token in tokens)
                      for key in unexpected)
            for role, tokens in roles.items()}


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("seat preservation must be field-wide")
    supplied = sorted(str(key) for key in params
                      if str(key) not in ALLOWED_PARAM_KEYS)
    if supplied:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(supplied))
    missing = sorted(key for key in REQUIRED_PATH_KEYS
                     if params.get(key) in (None, ""))
    if missing:
        raise ValueError("missing evidence paths: " + ", ".join(missing))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _component_area(frame: np.ndarray, owner: int, point: object) -> int:
    parts, _ = ndi.label(frame == int(owner), STRUCTURE)
    component = base._component_at(parts, point)
    return int(np.count_nonzero(parts == component)) if component > 0 else 0


def _expected_area(labels: np.ndarray, group: pd.DataFrame, owner: int,
                   before_frame: int) -> float:
    areas = []
    for point in group[
            group.physically_visible.map(_truth)
            & group.frame.astype(int).lt(before_frame)
            & group.accepted_owner.astype(int).eq(int(owner))].itertuples(
                index=False):
        area = _component_area(labels[int(point.frame)], owner, point)
        if area:
            areas.append(area)
    return float(np.median(areas)) if areas else 0.0


def _profile(group: pd.DataFrame, owner: int) -> dict[str, float]:
    visible = group[
        group.physically_visible.map(_truth)
        & group.accepted_owner.astype(int).gt(0)]
    matching = visible[visible.accepted_owner.astype(int).eq(int(owner))]
    return {
        "support": int(len(matching)),
        "purity": float(len(matching) / max(len(visible), 1)),
        "strong_fraction": float(matching.strong.map(_truth).mean())
        if len(matching) else 0.0,
    }


def _projection_components(
        labels: np.ndarray, scored: pd.DataFrame, frame: int,
        victim_owner: int, victim_point: object, transition: int,
        expected_victim: float, params: dict, proposal_id: str,
        ) -> tuple[bool, list[dict]]:
    parts, count = ndi.label(labels[frame] == int(victim_owner), STRUCTURE)
    if not count:
        return True, []
    rows: list[dict] = []
    by_track = {int(track): group.sort_values("frame")
                for track, group in scored.groupby("track_id", sort=False)}
    local = scored[
        scored.frame.astype(int).eq(frame)
        & scored.physically_visible.map(_truth)]
    maximum_area = float(params.get("maximum_projection_area_fraction", 0.35))
    maximum_distance = float(params.get(
        "maximum_projection_distance_sum_radii", 5.0))
    all_valid = True
    for component in range(1, count + 1):
        mask = parts == component
        yy, xx = np.nonzero(mask)
        area = int(len(xx))
        equivalent_radius = math.sqrt(area / math.pi)
        distance = float(np.hypot(
            float(xx.mean()) - float(victim_point.x),
            float(yy.mean()) - float(victim_point.y)))
        distance_radii = distance / max(
            float(victim_point.radius_px) + equivalent_radius, 1.0)
        supporting: list[int] = []
        prior_owners: set[int] = set()
        for point in local.itertuples(index=False):
            # Only a point actually owned by this accepted component may
            # describe its history. Nearby ownerless tracks are not evidence
            # for or against a small projection.
            if int(point.accepted_owner) != int(victim_owner):
                continue
            if base._component_at(parts, point) != component:
                continue
            supporting.append(int(point.track_id))
            history = by_track[int(point.track_id)]
            prior_owners.update(map(int, history[
                history.frame.astype(int).lt(transition)
                & history.physically_visible.map(_truth)
                & history.accepted_owner.astype(int).gt(0)
            ].accepted_owner))
        reasons = []
        area_fraction = area / max(expected_victim, 1.0)
        if area_fraction > maximum_area:
            reasons.append("component_not_projection_sized")
        if distance_radii > maximum_distance:
            reasons.append("projection_not_near_established_seat")
        temporal_overlap = bool(
            frame > 0 and np.any(mask & (labels[frame - 1] == victim_owner)))
        if not supporting and not temporal_overlap:
            reasons.append("projection_lacks_physical_track")
        if prior_owners - {int(victim_owner)}:
            reasons.append("projection_track_has_foreign_history")
        valid = not reasons
        all_valid &= valid
        rows.append({
            "proposal_id": proposal_id, "frame": int(frame),
            "component": component, "pixels": area,
            "area_fraction_of_established_seat": area_fraction,
            "distance_sum_radii": distance_radii,
            "supporting_tracks": ("|".join(map(str, sorted(set(supporting))))
                                  if supporting else "temporal_overlap"),
            "prior_positive_owners": "|".join(map(str, sorted(prior_owners))),
            "lineage_compatible": valid,
            "reason": "projection_lineage_compatible" if valid else
            "|".join(reasons),
        })
    return all_valid, rows


def _precontact_bookend(labels: np.ndarray, index: pd.DataFrame,
                        victim_track: int, resident_track: int,
                        victim_owner: int, resident_owner: int,
                        transition: int) -> bool:
    victim = base._point(index, victim_track, transition - 1)
    resident = base._point(index, resident_track, transition - 1)
    if victim is None or resident is None:
        return False
    if not all((_truth(victim.physically_visible), _truth(resident.physically_visible),
                _truth(victim.strong), _truth(resident.strong))):
        return False
    if (int(victim.accepted_owner) != int(victim_owner)
            or int(resident.accepted_owner) != int(resident_owner)):
        return False
    foreground, _ = ndi.label(labels[transition - 1] > 0, STRUCTURE)
    victim_host = base._component_at(foreground, victim)
    resident_host = base._component_at(foreground, resident)
    return victim_host > 0 and victim_host == resident_host


def discover_and_apply(labels: np.ndarray, raw: np.ndarray,
                       points: pd.DataFrame, params: dict,
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame,
                                  pd.DataFrame, np.ndarray]:
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    scored["physically_visible"] = scored.physically_visible.map(_truth)
    frame_count = len(labels)
    maximum_blank = max(0, int(math.ceil(frame_count * _fraction(
        params, "maximum_prior_blank_fraction", 0.02))))
    maximum_transition_blank = max(0, int(math.ceil(frame_count * _fraction(
        params, "maximum_transition_blank_movie_fraction", 0.01))))
    minimum_prior = int(math.ceil(frame_count * _fraction(
        params, "minimum_prior_movie_fraction", 0.15)))
    minimum_collapse = int(math.ceil(frame_count * _fraction(
        params, "minimum_collapse_movie_fraction", 0.25)))
    minimum_resident = int(math.ceil(frame_count * _fraction(
        params, "minimum_resident_movie_fraction", 0.50)))
    index = scored.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in scored.groupby("frame")}
    by_track = {int(track): group.sort_values("frame")
                for track, group in scored.groupby("track_id", sort=True)}
    audits: list[dict] = []
    frame_rows: list[dict] = []
    projection_rows: list[dict] = []
    plans: list[dict] = []
    number = 0

    for victim_track, group in by_track.items():
        runs = base._runs(group, maximum_blank)
        for before, after in zip(runs[:-1], runs[1:]):
            if int(before["owner"]) == int(after["owner"]):
                continue
            number += 1
            proposal_id = f"PS{number:04d}"
            victim_owner = int(before["owner"])
            resident_owner = int(after["owner"])
            transition = int(after["start"])
            prior = group[
                group.frame.astype(int).lt(transition)
                & group.physically_visible]
            positive = prior[prior.accepted_owner.astype(int).gt(0)]
            victim_prior = positive[
                positive.accepted_owner.astype(int).eq(victim_owner)]
            prior_support = int(len(victim_prior))
            prior_purity = prior_support / max(len(positive), 1)
            prior_strong = float(victim_prior.strong.map(_truth).mean()) \
                if prior_support else 0.0
            prior_frames = np.sort(victim_prior.frame.astype(int).unique())
            prior_gap = int(np.diff(prior_frames).max()) \
                if len(prior_frames) > 1 else 0
            blank = transition - int(before["end"]) - 1
            reasons: list[str] = []
            if prior_support < minimum_prior:
                reasons.append("insufficient_prior_tenure")
            if prior_purity < _fraction(
                    params, "minimum_prior_owner_purity", 0.98):
                reasons.append("impure_prior_owner_history")
            if prior_strong < _fraction(
                    params, "minimum_prior_strong_fraction", 0.95):
                reasons.append("weak_prior_raw_support")
            if prior_gap > maximum_blank + 1:
                reasons.append("discontinuous_prior_core")
            if blank > maximum_transition_blank:
                reasons.append("transition_not_temporally_contiguous")
            first_victim = base._point(index, victim_track, transition)
            if first_victim is None or not _truth(first_victim.physically_visible):
                reasons.append("victim_core_not_visible_at_transition")

            companions = []
            victim_radius = float(np.median(victim_prior.radius_px)) \
                if prior_support else 0.0
            if first_victim is not None and not reasons:
                companions = base._resident_candidates(
                    labels[transition], by_frame.get(
                        transition, scored.iloc[0:0]), first_victim,
                    victim_track, resident_owner,
                    {**params, "victim_radius": victim_radius})
                companions = [item for item in companions
                              if item["separation"] >= float(params.get(
                                  "minimum_initial_separation_sum_radii", 1.25))
                              and base._valley(raw[transition], first_victim,
                                               item["row"]) <= float(params.get(
                                  "maximum_two_core_valley_ratio", 0.65))]
                if len(companions) != 1:
                    reasons.append("resident_core_not_unique_or_not_two_basin")
            resident_track = int(companions[0]["row"].track_id) \
                if len(companions) == 1 else 0
            resident_profile = _profile(
                by_track[resident_track], resident_owner) \
                if resident_track else {"support": 0, "purity": 0.0,
                                        "strong_fraction": 0.0}
            if resident_track:
                if resident_profile["support"] < minimum_resident:
                    reasons.append("resident_seat_not_movie_durable")
                if resident_profile["purity"] < _fraction(
                        params, "minimum_resident_owner_purity", 0.95):
                    reasons.append("resident_seat_owner_impure")
                if resident_profile["strong_fraction"] < _fraction(
                        params, "minimum_resident_strong_fraction", 0.95):
                    reasons.append("resident_seat_raw_support_weak")
                if not _precontact_bookend(
                        labels, index, victim_track, resident_track,
                        victim_owner, resident_owner, transition):
                    reasons.append("distinct_precontact_bookend_missing")

            expected_victim = _expected_area(
                labels, group, victim_owner, transition)
            pending: list[dict] = []
            proposal_projections: list[dict] = []
            stop_reason = "physical_seat_support_ended"
            resident_by_owner = ({resident_owner: resident_track}
                                 if resident_track else {})
            if not reasons:
                for frame in range(transition, frame_count):
                    victim = base._point(index, victim_track, frame)
                    if (victim is None
                            or not _truth(victim.physically_visible)):
                        break
                    frame_owner = int(victim.accepted_owner)
                    if frame_owner <= 0 or frame_owner == victim_owner:
                        break
                    frame_companions = base._resident_candidates(
                        labels[frame], by_frame.get(frame, scored.iloc[0:0]),
                        victim, victim_track, frame_owner,
                        {**params, "victim_radius": victim_radius})
                    frame_companions = [item for item in frame_companions
                                        if item["separation"] >= float(params.get(
                                            "minimum_followup_separation_sum_radii",
                                            1.0))
                                        and base._valley(
                                            raw[frame], victim, item["row"])
                                        <= float(params.get(
                                            "maximum_two_core_valley_ratio",
                                            0.65))]
                    expected_track = resident_by_owner.get(frame_owner)
                    if expected_track is not None:
                        frame_companions = [item for item in frame_companions
                                            if int(item["row"].track_id)
                                            == int(expected_track)]
                    if len(frame_companions) != 1:
                        stop_reason = "durable_resident_seat_not_unique"
                        break
                    resident = frame_companions[0]["row"]
                    frame_resident_track = int(resident.track_id)
                    frame_profile = _profile(
                        by_track[frame_resident_track], frame_owner)
                    if (frame_profile["support"] < minimum_resident
                            or frame_profile["purity"] < _fraction(
                                params, "minimum_resident_owner_purity", 0.95)
                            or frame_profile["strong_fraction"] < _fraction(
                                params,
                                "minimum_replacement_resident_strong_fraction",
                                0.90)):
                        stop_reason = "replacement_owner_lacks_durable_seat"
                        break
                    resident_by_owner.setdefault(
                        frame_owner, frame_resident_track)
                    shared, _ = base._same_shared_component(
                        labels[frame], frame_owner, victim, resident)
                    if shared is None:
                        stop_reason = "accepted_owner_component_separated"
                        break
                    separation = float(np.hypot(
                        float(victim.x) - float(resident.x),
                        float(victim.y) - float(resident.y))) / max(
                            float(victim.radius_px) + float(resident.radius_px), 1.0)
                    valley = base._valley(raw[frame], victim, resident)
                    if (separation < float(params.get(
                            "minimum_followup_separation_sum_radii", 1.0))
                            or valley > float(params.get(
                                "maximum_two_core_valley_ratio", 0.65))):
                        stop_reason = "two_core_evidence_ended"
                        break
                    compatible, projection_audit = _projection_components(
                        labels, scored, frame, victim_owner, victim,
                        transition, expected_victim, params, proposal_id)
                    proposal_projections.extend(projection_audit)
                    if not compatible:
                        stop_reason = "nonprojection_old_owner_present"
                        break
                    partition = base._partition(
                        shared, raw[frame], victim, resident,
                        expected_victim, _expected_area(
                            labels, by_track[frame_resident_track],
                            frame_owner, frame), params)
                    if partition is None:
                        stop_reason = "connected_partition_unavailable"
                        break
                    victim_basin, resident_basin, method = partition
                    pending.append({
                        "proposal_id": proposal_id, "frame": frame,
                        "victim_track": victim_track,
                        "victim_owner": victim_owner,
                        "resident_track": frame_resident_track,
                        "resident_owner": frame_owner,
                        "separation_sum_radii": separation,
                        "valley_ratio": valley,
                        "partition_method": method,
                        "victim_pixels": int(victim_basin.sum()),
                        "resident_pixels": int(resident_basin.sum()),
                        "changed_pixels": int(victim_basin.sum()),
                        "projection_components": len(projection_audit),
                        "both_strong": bool(
                            _truth(victim.strong) and _truth(resident.strong)),
                        "shared": shared, "victim_basin": victim_basin,
                        "resident_basin": resident_basin,
                    })
            collapse_strong = float(np.mean([
                bool(item["both_strong"]) for item in pending
            ])) if pending else 0.0
            if len(pending) < minimum_collapse:
                reasons.append("collapse_not_persistent")
            if collapse_strong < _fraction(
                    params, "minimum_collapsed_strong_fraction", 0.95):
                reasons.append("collapsed_cores_not_consistently_strong")
            if reasons:
                audits.append({
                    "proposal_id": proposal_id,
                    "victim_track": victim_track,
                    "victim_owner": victim_owner,
                    "resident_track": resident_track,
                    "resident_owner": resident_owner,
                    "transition_frame": transition,
                    "collapse_last_frame": pending[-1]["frame"]
                    if pending else transition,
                    "prior_support": prior_support,
                    "prior_purity": prior_purity,
                    "prior_strong_fraction": prior_strong,
                    "resident_support": resident_profile["support"],
                    "resident_purity": resident_profile["purity"],
                    "resident_strong_fraction": resident_profile["strong_fraction"],
                    "collapse_frames": len(pending),
                    "collapse_strong_fraction": collapse_strong,
                    "initial_separation_sum_radii": companions[0]["separation"]
                    if len(companions) == 1 else 0.0,
                    "maximum_valley_ratio": max(
                        (item["valley_ratio"] for item in pending), default=0.0),
                    "projection_frames": len(set(
                        item["frame"] for item in proposal_projections)),
                    "projection_components": len(proposal_projections),
                    "changed_pixels": 0, "changed_frames": 0,
                    "eligible": False, "applied": False,
                    "reason": "|".join(sorted(set(reasons))),
                })
                projection_rows.extend(proposal_projections)
                continue
            audit = {
                "proposal_id": proposal_id,
                "victim_track": victim_track,
                "victim_owner": victim_owner,
                "resident_track": resident_track,
                "resident_owner": resident_owner,
                "transition_frame": transition,
                "collapse_last_frame": pending[-1]["frame"],
                "prior_support": prior_support,
                "prior_purity": prior_purity,
                "prior_strong_fraction": prior_strong,
                "resident_support": resident_profile["support"],
                "resident_purity": resident_profile["purity"],
                "resident_strong_fraction": resident_profile["strong_fraction"],
                "collapse_frames": len(pending),
                "collapse_strong_fraction": collapse_strong,
                "initial_separation_sum_radii": companions[0]["separation"],
                "maximum_valley_ratio": max(
                    item["valley_ratio"] for item in pending),
                "projection_frames": len(set(
                    item["frame"] for item in proposal_projections)),
                "projection_components": len(proposal_projections),
                "changed_pixels": int(sum(
                    item["changed_pixels"] for item in pending)),
                "changed_frames": len(pending),
                "eligible": True, "applied": False,
                "reason": "eligible_until_" + stop_reason,
            }
            audits.append(audit)
            projection_rows.extend(proposal_projections)
            plans.append({"audit": audit, "frames": pending})

    conflicts: set[str] = set()
    claims: list[tuple[str, int, np.ndarray]] = []
    for plan in plans:
        proposal_id = str(plan["audit"]["proposal_id"])
        for item in plan["frames"]:
            for other_id, other_frame, other_mask in claims:
                if other_frame == int(item["frame"]) and np.any(
                        other_mask & item["shared"]):
                    conflicts.update((proposal_id, other_id))
            claims.append((proposal_id, int(item["frame"]), item["shared"]))

    candidate = labels.copy()
    proven_mask = np.zeros(labels.shape, dtype=bool)
    for plan in plans:
        audit = plan["audit"]
        if str(audit["proposal_id"]) in conflicts:
            audit.update({"eligible": False, "changed_pixels": 0,
                          "changed_frames": 0,
                          "reason": "global_atomic_component_conflict"})
            continue
        for item in plan["frames"]:
            frame = int(item["frame"])
            candidate[frame][item["shared"]] = 0
            candidate[frame][item["victim_basin"]] = int(item["victim_owner"])
            candidate[frame][item["resident_basin"]] = int(item["resident_owner"])
            proven_mask[frame][item["victim_basin"]] = True
            frame_rows.append({key: item[key] for key in FRAME_COLUMNS})
        audit.update({"applied": True,
                      "reason": "applied_projection_tolerant_seat_preservation"})
    return (candidate, pd.DataFrame(audits, columns=AUDIT_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS),
            pd.DataFrame(projection_rows, columns=PROJECTION_COLUMNS),
            proven_mask)


def _duplicate_proof(labels: np.ndarray, candidate: np.ndarray,
                     frames: pd.DataFrame, projections: pd.DataFrame,
                     proven_mask: np.ndarray) -> tuple[int, int, pd.DataFrame]:
    new_count = owner_consensus.count_new_duplicate_components(labels, candidate)
    proof_rows = []
    if not new_count:
        return 0, 0, pd.DataFrame(columns=[
            "proposal_id", "frame", "lineage_identity", "new_excess",
            "restored_pixels", "projection_components", "proof"])
    for row in frames.itertuples(index=False):
        frame, owner = int(row.frame), int(row.victim_owner)
        before = ndi.label(labels[frame] == owner, STRUCTURE)[1]
        after = ndi.label(candidate[frame] == owner, STRUCTURE)[1]
        added = max(0, max(0, after - 1) - max(0, before - 1))
        if not added:
            continue
        projection = projections[
            projections.proposal_id.astype(str).eq(str(row.proposal_id))
            & projections.frame.astype(int).eq(frame)
            & projections.lineage_compatible.astype(bool)]
        restored = int(np.count_nonzero(
            proven_mask[frame] & (candidate[frame] == owner)))
        if restored and len(projection) >= added:
            proof_rows.append({
                "proposal_id": str(row.proposal_id), "frame": frame,
                "lineage_identity": owner, "new_excess": added,
                "restored_pixels": restored,
                "projection_components": int(len(projection)),
                "proof": "established_soma_plus_compatible_projection",
            })
    proven = int(sum(row["new_excess"] for row in proof_rows))
    return int(new_count), proven, pd.DataFrame(proof_rows)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    if not labels.shape == unclaimed.shape == raw.shape:
        raise ValueError("labels, unclaimed, and raw stacks must align")
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        audit = pd.DataFrame(columns=AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
        projections = pd.DataFrame(columns=PROJECTION_COLUMNS)
        proven_mask = np.zeros(labels.shape, dtype=bool)
    else:
        candidate, audit, frames, projections, proven_mask = \
            discover_and_apply(labels, raw, points, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("seat preservation changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("seat preservation changed the identity set")
    duplicates, proven, duplicate_proof = _duplicate_proof(
        labels, candidate, frames, projections, proven_mask)
    if duplicates != proven:
        raise AssertionError("new duplicate lacks compatible-projection proof")
    output_dir = Path(out.out)
    stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(
            labels_path, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    audit_path = output_dir / "seat_preservation_audit.csv"
    frame_path = output_dir / "seat_preservation_frames.csv"
    projection_path = output_dir / "projection_lineage_audit.csv"
    duplicate_path = output_dir / "new_duplicate_lineage_proof.csv"
    audit.to_csv(audit_path, index=False)
    frames.to_csv(frame_path, index=False)
    projections.to_csv(projection_path, index=False)
    duplicate_proof.to_csv(duplicate_path, index=False)
    changed = candidate != labels
    changed_frames = np.flatnonzero(
        changed.reshape(len(labels), -1).any(axis=1)).astype(int).tolist()
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": selector_counts(params),
        "physical_tracks_audited": int(points.track_id.nunique()),
        "transitions_audited": int(len(audit)),
        "eligible_transitions": int(audit.eligible.astype(bool).sum())
        if len(audit) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": len(changed_frames),
        "changed_frame_indices": changed_frames,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": duplicates,
        "lineage_proven_new_duplicate_components": proven,
        "all_new_duplicates_projection_proven": duplicates == proven,
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {
        "outputs": {
            "labels": labels_path, "unclaimed": unclaimed_path,
            "audit": audit_path, "frames": frame_path,
            "projection_audit": projection_path,
            "duplicate_proof": duplicate_path, "metrics": metrics_path,
        },
        "summary": metrics,
    }

