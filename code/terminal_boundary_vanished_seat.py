"""Retain a proved cell seat through a short right-censored owner collapse."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import distinct_history_transition_candidates as partition
import owner_consensus
import ownerless_body_recovery as ownerless
import separable_merge_recovery as physical
from resident_takeover import attach_owners


FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "well", "review_case",
    "case_id", "allowed_", "forced_", "include_", "exclude_",
)
AUDIT_COLUMNS = [
    "proposal_id", "victim_owner", "resident_owner", "victim_track",
    "resident_track", "last_owned_frame", "repair_first", "repair_last",
    "repair_frames", "victim_identity_support_fraction",
    "resident_identity_support_fraction", "victim_track_recent_support",
    "victim_track_owner_purity", "resident_track_dominance",
    "minimum_predicted_disk_foreign_coverage",
    "minimum_separation_sum_radii", "minimum_raw_core_radius_ratio",
    "minimum_raw_core_containment", "minimum_shared_combined_area_ratio",
    "maximum_shared_combined_area_ratio", "geodesic_partition_frames",
    "minimum_partition_raw_core_retention",
    "raw_partition_frames", "changed_pixels", "changed_frames", "eligible",
    "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "victim_owner", "resident_owner",
    "victim_track", "resident_track", "predicted_x", "predicted_y",
    "predicted_disk_foreign_coverage",
    "separation_sum_radii", "raw_core_area", "raw_core_radius_ratio",
    "raw_core_containment", "shared_area", "shared_combined_area_ratio",
    "victim_basin_pixels", "resident_basin_pixels", "partition_method",
    "partition_raw_core_retention",
    "changed_pixels",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal boundary seat partition must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError("terminal boundary partition received forbidden "
                         "targets: " + ", ".join(supplied))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _presence(labels: np.ndarray, owner: int) -> np.ndarray:
    return np.any(labels == int(owner), axis=(1, 2))


def _component(labels: np.ndarray, owner: int, point: object) -> np.ndarray | None:
    components, _ = ndi.label(labels == int(owner), partition.STRUCTURE)
    component = physical._component_at(
        components, float(point.x), float(point.y),
        max(2, int(math.ceil(float(point.radius_px)))))
    return components == component if component > 0 else None


def _track_profile(group: pd.DataFrame, owner: int) -> tuple[int, float]:
    visible = group[group.physically_visible.map(_truth)]
    positive = visible[visible.candidate_owner.astype(int).gt(0)]
    matching = positive[positive.candidate_owner.astype(int).eq(int(owner))]
    return int(len(matching)), float(len(matching) / max(len(positive), 1))


def _dominant_point(rows: pd.DataFrame, minimum_ratio: float):
    if rows.empty:
        return None, 0.0
    ranked = rows.sort_values(
        ["radius_px", "track_id"], ascending=[False, True])
    first = ranked.iloc[0]
    second = float(ranked.iloc[1].radius_px) if len(ranked) > 1 else 0.0
    ratio = float(first.radius_px) / max(second, 1e-9) if second else math.inf
    return (first if ratio >= minimum_ratio else None), ratio


def _predict(group: pd.DataFrame, last_owned: int, frame: int,
             maximum_step_radius_fraction: float) -> SimpleNamespace:
    prior = group[
        group.frame.astype(int).le(int(last_owned))
        & group.physically_visible.map(_truth)].sort_values("frame")
    last = prior.iloc[-1]
    dx = dy = 0.0
    if len(prior) >= 2:
        previous = prior.iloc[-2]
        gap = max(int(last.frame) - int(previous.frame), 1)
        dx = (float(last.x) - float(previous.x)) / gap
        dy = (float(last.y) - float(previous.y)) / gap
    step = math.hypot(dx, dy)
    limit = maximum_step_radius_fraction * max(float(last.radius_px), 1.0)
    if step > limit > 0:
        dx, dy = dx * limit / step, dy * limit / step
    horizon = int(frame) - int(last_owned)
    return SimpleNamespace(
        frame=int(frame), x=float(last.x) + dx * horizon,
        y=float(last.y) + dy * horizon,
        radius_px=float(last.radius_px))


def _expected_area(labels: np.ndarray, group: pd.DataFrame, owner: int,
                   before: int, trailing: int) -> float:
    areas = []
    rows = group[
        group.frame.astype(int).between(max(0, before - trailing), before)
        & group.physically_visible.map(_truth)
        & group.candidate_owner.astype(int).eq(int(owner))]
    for point in rows.itertuples(index=False):
        component = _component(labels[int(point.frame)], owner, point)
        if component is not None:
            areas.append(int(component.sum()))
    return float(np.median(areas)) if areas else 0.0


def discover_and_apply(labels: np.ndarray, unclaimed: np.ndarray,
                       raw: np.ndarray, points: pd.DataFrame,
                       thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    if not labels.shape == unclaimed.shape == raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")
    movie = int(len(labels))
    scored = attach_owners(points, labels)
    scored["physically_visible"] = scored.state.astype(str).isin(
        ["observed", "latent_visible"])
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    by_frame = {int(frame): group for frame, group in scored.groupby("frame")}
    threshold_by_frame = thresholds.set_index("frame")
    minimum_identity_support = float(params.get(
        "minimum_identity_support_movie_fraction", 0.75))
    maximum_suffix = max(1, int(math.ceil(movie * float(params.get(
        "maximum_terminal_suffix_movie_fraction", 0.02)))))
    recent_frames = max(2, int(math.ceil(movie * float(params.get(
        "minimum_recent_track_support_movie_fraction", 0.05)))))
    minimum_recent_support = float(params.get(
        "minimum_recent_track_visible_fraction", 0.80))
    minimum_track_purity = float(params.get("minimum_recent_owner_purity", 0.90))
    minimum_foreign_coverage = float(params.get(
        "minimum_predicted_disk_foreign_coverage", 0.25))
    minimum_dominance = float(params.get(
        "minimum_resident_track_radius_dominance", 1.50))
    minimum_separation = float(params.get(
        "minimum_separation_sum_radii", 1.0))
    minimum_core_ratio = float(params.get(
        "minimum_raw_core_area_radius_ratio", 0.25))
    maximum_core_ratio = float(params.get(
        "maximum_raw_core_area_radius_ratio", 2.0))
    minimum_core_containment = float(params.get(
        "minimum_raw_core_shared_containment", 0.70))
    minimum_partition_core_retention = float(params.get(
        "minimum_partition_raw_core_retention", 0.60))
    minimum_combined = float(params.get(
        "minimum_shared_combined_area_ratio", 0.50))
    maximum_combined = float(params.get(
        "maximum_shared_combined_area_ratio", 1.50))
    maximum_step = float(params.get(
        "maximum_extrapolation_step_radius_fraction", 0.75))

    audits: list[dict] = []
    plans: list[dict] = []
    identities = sorted(set(map(int, np.unique(labels))) - {0})
    for victim in identities:
        presence = _presence(labels, victim)
        present = np.flatnonzero(presence)
        if not len(present) or int(present[-1]) >= movie - 1:
            continue
        last_owned = int(present[-1])
        suffix = list(range(last_owned + 1, movie))
        if len(suffix) > maximum_suffix:
            continue
        reasons: list[str] = []
        identity_support = float(len(present) / movie)
        internal_gaps = int(last_owned - int(present[0]) + 1 - len(present))
        if identity_support < minimum_identity_support:
            reasons.append("victim_identity_not_durable")
        if internal_gaps:
            reasons.append("victim_identity_has_prior_gaps")

        victim_rows = scored[
            scored.frame.astype(int).eq(last_owned)
            & scored.physically_visible.map(_truth)
            & scored.candidate_owner.astype(int).eq(victim)]
        victim_point, victim_dominance = _dominant_point(
            victim_rows, minimum_dominance)
        victim_track = int(victim_point.track_id) \
            if victim_point is not None else 0
        victim_group = groups.get(victim_track, scored.iloc[0:0])
        recent_support, recent_purity = _track_profile(victim_group, victim)
        recent_support = int((victim_group[
            victim_group.frame.astype(int).between(
                max(0, last_owned - recent_frames + 1), last_owned)
            & victim_group.physically_visible.map(_truth)
            & victim_group.candidate_owner.astype(int).eq(victim)]).shape[0])
        if victim_point is None:
            reasons.append("victim_terminal_track_not_unique")
        if recent_support < math.ceil(recent_frames * minimum_recent_support):
            reasons.append("victim_recent_track_incomplete")
        if recent_purity < minimum_track_purity:
            reasons.append("victim_recent_track_owner_impure")

        frame_plans = []
        resident_owner = resident_track = 0
        resident_support_fraction = 0.0
        resident_dominance_values = []
        foreign_coverage_values = []
        expected_victim = _expected_area(
            labels, victim_group, victim, last_owned, recent_frames) \
            if victim_point is not None else 0.0
        for frame in suffix if not reasons else []:
            predicted = _predict(
                victim_group, last_owned, frame, maximum_step)
            current_owner, foreign_coverage = ownerless._disk_coverage(
                labels[frame], predicted.x, predicted.y,
                predicted.radius_px)
            foreign_coverage_values.append(float(foreign_coverage))
            if (current_owner <= 0 or current_owner == victim
                    or foreign_coverage < minimum_foreign_coverage):
                reasons.append("predicted_seat_not_inside_foreign_foreground")
                break
            if resident_owner and current_owner != resident_owner:
                reasons.append("terminal_resident_owner_not_constant")
                break
            resident_owner = current_owner
            shared_components, _ = ndi.label(
                labels[frame] == resident_owner, partition.STRUCTURE)
            shared_id = physical._component_at(
                shared_components, predicted.x, predicted.y,
                max(2, int(math.ceil(predicted.radius_px))))
            shared = shared_components == shared_id if shared_id else None
            if shared is None:
                reasons.append("shared_component_missing")
                break
            residents = by_frame.get(frame, scored.iloc[0:0])
            residents = residents[
                residents.physically_visible.map(_truth)
                & residents.candidate_owner.astype(int).eq(resident_owner)]
            residents = residents[residents.apply(
                lambda row: bool(shared[
                    int(np.clip(round(float(row.y)), 0, shared.shape[0] - 1)),
                    int(np.clip(round(float(row.x)), 0, shared.shape[1] - 1))]),
                axis=1)]
            resident_point, dominance = _dominant_point(
                residents, minimum_dominance)
            resident_dominance_values.append(dominance)
            if resident_point is None:
                reasons.append("resident_terminal_track_not_unique")
                break
            if resident_track and int(resident_point.track_id) != resident_track:
                reasons.append("resident_terminal_track_not_constant")
                break
            resident_track = int(resident_point.track_id)
            resident_group = groups[resident_track]
            resident_presence = _presence(labels, resident_owner)
            resident_support_fraction = float(resident_presence.mean())
            if resident_support_fraction < minimum_identity_support:
                reasons.append("resident_identity_not_durable")
                break
            separation = float(math.hypot(
                predicted.x - float(resident_point.x),
                predicted.y - float(resident_point.y)) / max(
                    predicted.radius_px + float(resident_point.radius_px), 1.0))
            if separation < minimum_separation:
                reasons.append("terminal_seats_not_spatially_distinct")
                break
            expected_resident = _expected_area(
                labels, resident_group, resident_owner, last_owned,
                recent_frames)
            combined_ratio = float(shared.sum() / max(
                expected_victim + expected_resident, 1.0))
            if not minimum_combined <= combined_ratio <= maximum_combined:
                reasons.append("shared_component_area_inconsistent")
                break
            core, region, details = ownerless._connected_core(
                raw[frame], predicted,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            core_ratio = float(details["core_area_px"] / max(
                math.pi * predicted.radius_px ** 2, 1.0))
            full_core = np.zeros_like(shared)
            full_core[region] = core
            containment = float(np.count_nonzero(full_core & shared)
                                / max(int(full_core.sum()), 1))
            if not minimum_core_ratio <= core_ratio <= maximum_core_ratio:
                reasons.append("victim_raw_body_scale_unsupported")
                break
            if containment < minimum_core_containment:
                reasons.append("victim_raw_body_outside_shared_component")
                break
            split = partition._partition(
                shared, raw[frame], predicted, resident_point,
                expected_victim, expected_resident, params)
            if split is None:
                reasons.append("terminal_two_seed_partition_unavailable")
                break
            victim_basin, resident_basin, method = split
            basin_core = float(np.count_nonzero(victim_basin & full_core)
                               / max(int(np.count_nonzero(
                                   full_core & shared)), 1))
            if basin_core < minimum_partition_core_retention:
                reasons.append("partition_does_not_retain_victim_raw_core")
                break
            frame_plans.append({
                "frame": frame, "shared": shared,
                "victim_basin": victim_basin,
                "resident_basin": resident_basin, "predicted": predicted,
                "resident_point": resident_point, "separation": separation,
                "foreign_coverage": float(foreign_coverage),
                "core_area": int(details["core_area_px"]),
                "core_ratio": core_ratio, "containment": containment,
                "combined_ratio": combined_ratio, "method": method,
                "partition_core_retention": basin_core,
            })

        eligible = not reasons and len(frame_plans) == len(suffix)
        proposal_id = f"TB{len(plans) + 1:04d}" if eligible else ""
        row = {
            "proposal_id": proposal_id, "victim_owner": victim,
            "resident_owner": resident_owner, "victim_track": victim_track,
            "resident_track": resident_track, "last_owned_frame": last_owned,
            "repair_first": suffix[0], "repair_last": suffix[-1],
            "repair_frames": len(suffix),
            "victim_identity_support_fraction": identity_support,
            "resident_identity_support_fraction": resident_support_fraction,
            "victim_track_recent_support": recent_support,
            "victim_track_owner_purity": recent_purity,
            "resident_track_dominance": min(
                resident_dominance_values, default=0.0),
            "minimum_predicted_disk_foreign_coverage": min(
                foreign_coverage_values, default=0.0),
            "minimum_separation_sum_radii": min(
                (item["separation"] for item in frame_plans), default=0.0),
            "minimum_raw_core_radius_ratio": min(
                (item["core_ratio"] for item in frame_plans), default=0.0),
            "minimum_raw_core_containment": min(
                (item["containment"] for item in frame_plans), default=0.0),
            "minimum_shared_combined_area_ratio": min(
                (item["combined_ratio"] for item in frame_plans), default=0.0),
            "maximum_shared_combined_area_ratio": max(
                (item["combined_ratio"] for item in frame_plans), default=0.0),
            "minimum_partition_raw_core_retention": min(
                (item["partition_core_retention"] for item in frame_plans),
                default=0.0),
            "geodesic_partition_frames": sum(
                item["method"] == "geodesic_seed_fallback"
                for item in frame_plans),
            "raw_partition_frames": sum(
                item["method"] == "raw_watershed_connected"
                for item in frame_plans),
            "changed_pixels": int(sum(
                item["victim_basin"].sum() for item in frame_plans))
                if eligible else 0,
            "changed_frames": len(frame_plans) if eligible else 0,
            "eligible": eligible, "applied": False,
            "reason": ("eligible_terminal_boundary_vanished_seat"
                       if eligible else "|".join(dict.fromkeys(reasons))),
        }
        audits.append(row)
        if eligible:
            plans.append({**row, "frame_plans": frame_plans})

    candidate = labels.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    frame_rows = []
    for plan in plans:
        if any(np.any(occupied[item["frame"]] & item["shared"])
               for item in plan["frame_plans"]):
            continue
        trial = candidate.copy()
        for item in plan["frame_plans"]:
            frame = int(item["frame"])
            trial[frame][item["shared"]] = 0
            trial[frame][item["victim_basin"]] = int(plan["victim_owner"])
            trial[frame][item["resident_basin"]] = int(plan["resident_owner"])
        if owner_consensus.count_new_duplicate_components(labels, trial):
            continue
        for item in plan["frame_plans"]:
            frame = int(item["frame"])
            candidate[frame] = trial[frame]
            occupied[frame] |= item["shared"]
            frame_rows.append({
                "proposal_id": plan["proposal_id"], "frame": frame,
                "victim_owner": plan["victim_owner"],
                "resident_owner": plan["resident_owner"],
                "victim_track": plan["victim_track"],
                "resident_track": plan["resident_track"],
                "predicted_x": item["predicted"].x,
                "predicted_y": item["predicted"].y,
                "predicted_disk_foreign_coverage": item["foreign_coverage"],
                "separation_sum_radii": item["separation"],
                "raw_core_area": item["core_area"],
                "raw_core_radius_ratio": item["core_ratio"],
                "raw_core_containment": item["containment"],
                "shared_area": int(item["shared"].sum()),
                "shared_combined_area_ratio": item["combined_ratio"],
                "victim_basin_pixels": int(item["victim_basin"].sum()),
                "resident_basin_pixels": int(item["resident_basin"].sum()),
                "partition_method": item["method"],
                "partition_raw_core_retention":
                    item["partition_core_retention"],
                "changed_pixels": int(item["victim_basin"].sum()),
            })
        plan["applied"] = True
        plan["reason"] = "applied_terminal_boundary_vanished_seat"
        for audit in audits:
            if audit["proposal_id"] == plan["proposal_id"]:
                audit.update({key: plan[key] for key in (
                    "applied", "reason", "changed_pixels", "changed_frames")})

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("terminal boundary partition changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("terminal boundary partition overlaps unclaimed")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("terminal boundary partition changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("terminal boundary partition created duplicates")
    audit_table = pd.DataFrame(audits, columns=AUDIT_COLUMNS)
    frame_table = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    changed = candidate != labels
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "terminal_disappearances_audited": int(len(audit_table)),
        "eligible_proposals": int(audit_table.eligible.astype(bool).sum())
            if len(audit_table) else 0,
        "applied_proposals": int(audit_table.applied.astype(bool).sum())
            if len(audit_table) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(changed.reshape(movie, -1).any(axis=1).sum()),
        "foreground_exact": True, "unclaimed_exact": True,
        "identity_set_exact": True, "new_duplicate_components": 0,
    }
    return candidate, audit_table, frame_table, metrics


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    candidate, audit, frames, metrics = discover_and_apply(
        labels, unclaimed, raw, points, thresholds, params)
    stem = str(params.get("output_stem", "95_A3"))
    labels_out = out.out / f"{stem}.tif"
    unclaimed_out = out.out / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copyfile(labels_path, labels_out)
    else:
        tifffile.imwrite(labels_out, candidate, imagej=True,
                         compression="zlib", metadata={"axes": "TYX"})
    shutil.copyfile(unclaimed_path, unclaimed_out)
    audit_out = out.out / "terminal_boundary_seat_audit.csv"
    frames_out = out.out / "terminal_boundary_seat_frames.csv"
    metrics_out = out.out / "metrics.json"
    audit.to_csv(audit_out, index=False)
    frames.to_csv(frames_out, index=False)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n",
                           encoding="utf-8")
    return {"outputs": {
        "labels": labels_out, "unclaimed": unclaimed_out,
        "audit": audit_out, "frames": frames_out,
        "metrics": metrics_out}, "summary": metrics}

