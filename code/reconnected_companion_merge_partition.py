"""Preserve two identity seats through uniquely reconnected merge gaps."""
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

import owner_consensus
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_", "forced_", "include_", "exclude_",
)
AUDIT_COLUMNS = [
    "proposal_id", "reconnection_id", "track_before", "track_after",
    "resident_track", "companion_owner", "resident_owner", "gap_start",
    "gap_end", "gap_frames", "companion_before_support",
    "companion_after_support", "companion_before_purity",
    "companion_after_purity", "resident_owner_support",
    "resident_owner_purity", "resident_prior_support",
    "resident_post_support", "foreign_owner_fraction",
    "strong_gap_fraction", "minimum_separation_sum_radii",
    "maximum_separation_sum_radii", "reconnection_distance_px",
    "reconnection_patch_correlation", "reconnection_uniqueness_margin",
    "expected_resident_area", "expected_companion_area", "eligible",
    "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("reconnected companion partition must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "reconnected companion partition received forbidden targets: "
            + ", ".join(supplied))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _row(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return None if len(rows) != 1 else rows.iloc[0]


def _owner_stats(group: pd.DataFrame) -> tuple[int, int, float]:
    if group.empty:
        return 0, 0, 0.0
    owners = group.candidate_owner.astype(int)
    positive = owners[owners.gt(0)]
    if positive.empty:
        return 0, 0, 0.0
    counts = positive.value_counts()
    owner = int(counts.index[0])
    support = int(counts.iloc[0])
    return owner, support, float(support / len(group))


def _component_at_owner(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    if count <= 0:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    number = int(components[y, x])
    if number <= 0:
        radius = max(2.0, float(point.radius_px))
        yy, xx = np.nonzero(components > 0)
        distance = np.hypot(xx - float(point.x), yy - float(point.y))
        inside = distance <= radius
        if not inside.any():
            return None
        values = components[yy[inside], xx[inside]]
        ids, counts = np.unique(values[values > 0], return_counts=True)
        if not len(ids):
            return None
        number = int(ids[int(np.argmax(counts))])
    return components == number


def _expected_area(labels: np.ndarray, group: pd.DataFrame,
                   owner: int) -> float:
    values: list[int] = []
    for point in group.itertuples(index=False):
        if int(point.candidate_owner) != int(owner):
            continue
        component = _component_at_owner(
            labels[int(point.frame)], int(owner), point)
        if component is not None:
            values.append(int(component.sum()))
    return float(np.median(values)) if values else 0.0


def discover(labels: np.ndarray, points: pd.DataFrame,
             reconnections: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all reconnection gaps for a stable invaded resident seat."""
    assert_target_free(params)
    attached = attach_owners(points, labels)
    visible = attached[attached.physically_visible.astype(bool)].copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id", sort=True)}
    frame_count = len(labels)
    maximum_gap = max(1, int(math.ceil(frame_count * float(params.get(
        "maximum_reconnection_gap_movie_fraction", 0.15)))))
    minimum_companion_support = max(2, int(math.ceil(
        frame_count * float(params.get(
            "minimum_companion_support_movie_fraction", 0.04)))))
    minimum_resident_support = max(2, int(math.ceil(
        frame_count * float(params.get(
            "minimum_resident_support_movie_fraction", 0.40)))))
    minimum_flank_support = max(1, int(math.ceil(
        frame_count * float(params.get(
            "minimum_resident_flank_movie_fraction", 0.03)))))
    minimum_owner_purity = float(params.get("minimum_owner_purity", 0.75))
    minimum_gap_purity = float(params.get("minimum_gap_owner_purity", 0.55))
    minimum_gap_coverage = float(params.get("minimum_gap_owner_coverage", 0.50))
    minimum_foreign = float(params.get("minimum_foreign_owner_fraction", 0.30))
    minimum_strong = float(params.get("minimum_strong_gap_fraction", 0.80))
    minimum_separation = float(params.get(
        "minimum_separation_sum_radii", 1.20))
    maximum_separation = float(params.get(
        "maximum_separation_sum_radii", 3.50))
    minimum_patch = float(params.get(
        "minimum_reconnection_patch_correlation", 0.40))
    minimum_uniqueness = float(params.get(
        "minimum_reconnection_uniqueness_margin", 0.10))
    rows: list[dict] = []
    internals: list[dict] = []

    valid_reconnections = reconnections.dropna(
        subset=["track_before", "track_after", "end_frame", "start_frame"])
    for connection in valid_reconnections.sort_values(
            ["end_frame", "start_frame", "reconnection_id"]).itertuples(
                index=False):
        before_track = int(connection.track_before)
        after_track = int(connection.track_after)
        end = int(connection.end_frame)
        start = int(connection.start_frame)
        gap = list(range(end + 1, start))
        if (not gap or len(gap) > maximum_gap
                or before_track not in groups or after_track not in groups):
            continue
        before = groups[before_track]
        after = groups[after_track]
        before_owner, before_support, before_purity = _owner_stats(before)
        after_owner, after_support, after_purity = _owner_stats(after)
        if before_owner <= 0 or before_owner != after_owner:
            continue
        companion_owner = before_owner
        candidates: list[tuple[int, pd.DataFrame]] = []
        for track, group in groups.items():
            if track in {before_track, after_track}:
                continue
            middle = group[group.frame.astype(int).isin(gap)]
            if (len(middle) == len(gap)
                    and middle.candidate_owner.astype(int)
                    .eq(companion_owner).any()):
                candidates.append((track, group))
        for resident_track, resident in candidates:
            middle = resident[resident.frame.astype(int).isin(gap)]
            prior = resident[resident.frame.astype(int).le(end)]
            post = resident[resident.frame.astype(int).ge(start)]
            flanks = pd.concat([prior, post], ignore_index=True)
            resident_owner, resident_support, resident_purity = \
                _owner_stats(flanks)
            prior_support = int(prior.candidate_owner.astype(int)
                                .eq(resident_owner).sum())
            post_support = int(post.candidate_owner.astype(int)
                               .eq(resident_owner).sum())
            owners_in_gap = set(middle.candidate_owner.astype(int))
            foreign_fraction = float(middle.candidate_owner.astype(int)
                                     .eq(companion_owner).mean())
            strong_fraction = float(middle.strong.map(_truth).mean())
            before_point = _row(before, end)
            after_point = _row(after, start)
            resident_before = _row(resident, end)
            resident_after = _row(resident, start)
            separations: list[float] = []
            if before_point is not None and after_point is not None:
                for point in middle.itertuples(index=False):
                    fraction = ((int(point.frame) - end)
                                / max(start - end, 1))
                    x = ((1.0 - fraction) * float(before_point.x)
                         + fraction * float(after_point.x))
                    y = ((1.0 - fraction) * float(before_point.y)
                         + fraction * float(after_point.y))
                    radius = ((1.0 - fraction) * float(before_point.radius_px)
                              + fraction * float(after_point.radius_px))
                    separations.append(float(np.hypot(
                        float(point.x) - x, float(point.y) - y)
                        / max(float(point.radius_px) + radius, 1.0)))
            minimum_seen = min(separations, default=float("nan"))
            maximum_seen = max(separations, default=float("nan"))
            reasons: list[str] = []
            if min(before_support, after_support) < minimum_companion_support:
                reasons.append("insufficient_companion_support")
            if min(before_purity, after_purity) < minimum_owner_purity:
                reasons.append("companion_owner_not_stable")
            if resident_owner <= 0 or resident_owner == companion_owner:
                reasons.append("resident_owner_not_distinct")
            if (resident_support < minimum_resident_support
                    or resident_purity < minimum_owner_purity):
                reasons.append("resident_owner_not_stable")
            if min(prior_support, post_support) < minimum_flank_support:
                reasons.append("resident_flanks_too_short")
            if resident_before is None or resident_after is None:
                reasons.append("resident_not_bracketing_gap")
            elif (int(resident_before.candidate_owner) != resident_owner
                  or int(resident_after.candidate_owner) != resident_owner):
                reasons.append("resident_owner_not_restored_on_both_flanks")
            if foreign_fraction < minimum_foreign:
                reasons.append("companion_does_not_invade_resident_seat")
            if owners_in_gap - {resident_owner, companion_owner}:
                reasons.append("third_owner_on_resident_seat")
            if (float(middle.candidate_owner_purity.min()) < minimum_gap_purity
                    or float(middle.candidate_coverage_fraction.min())
                    < minimum_gap_coverage):
                reasons.append("weak_gap_owner_measurement")
            if strong_fraction < minimum_strong:
                reasons.append("resident_track_dim_in_gap")
            if (not separations or minimum_seen < minimum_separation
                    or maximum_seen > maximum_separation):
                reasons.append("two_seat_geometry_out_of_range")
            patch = float(connection.patch_correlation)
            uniqueness = float(connection.uniqueness_margin)
            if not np.isfinite(patch) or patch < minimum_patch:
                reasons.append("reconnection_patch_support_too_weak")
            if np.isfinite(uniqueness) and uniqueness < minimum_uniqueness:
                reasons.append("reconnection_not_unique")
            expected_resident = _expected_area(
                labels, flanks[flanks.candidate_owner.astype(int)
                               .eq(resident_owner)], resident_owner)
            expected_companion = float(np.median([
                _expected_area(labels, before, companion_owner),
                _expected_area(labels, after, companion_owner),
            ]))
            if min(expected_resident, expected_companion) <= 0:
                reasons.append("expected_seat_area_missing")
            public = {
                "proposal_id": "",
                "reconnection_id": int(connection.reconnection_id),
                "track_before": before_track,
                "track_after": after_track,
                "resident_track": int(resident_track),
                "companion_owner": int(companion_owner),
                "resident_owner": int(resident_owner),
                "gap_start": int(gap[0]), "gap_end": int(gap[-1]),
                "gap_frames": len(gap),
                "companion_before_support": before_support,
                "companion_after_support": after_support,
                "companion_before_purity": before_purity,
                "companion_after_purity": after_purity,
                "resident_owner_support": resident_support,
                "resident_owner_purity": resident_purity,
                "resident_prior_support": prior_support,
                "resident_post_support": post_support,
                "foreign_owner_fraction": foreign_fraction,
                "strong_gap_fraction": strong_fraction,
                "minimum_separation_sum_radii": minimum_seen,
                "maximum_separation_sum_radii": maximum_seen,
                "reconnection_distance_px": float(connection.distance_px),
                "reconnection_patch_correlation": patch,
                "reconnection_uniqueness_margin": uniqueness,
                "expected_resident_area": expected_resident,
                "expected_companion_area": expected_companion,
                "eligible": not reasons,
                "reason": ("eligible_reconnected_companion_merge"
                           if not reasons else "|".join(reasons)),
            }
            rows.append(public)
            internals.append({
                **public, "gap_values": gap, "resident_group": resident,
                "before_point": before_point, "after_point": after_point,
            })
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = \
                f"RCM{number:04d}"
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internals))


def _nearest(mask: np.ndarray, x: float, y: float
             ) -> tuple[tuple[int, int], float] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    distance = np.hypot(xx - float(x), yy - float(y))
    index = int(np.argmin(distance))
    return (int(yy[index]), int(xx[index])), float(distance[index])


def apply(labels: np.ndarray, internal: pd.DataFrame, params: dict
          ) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply each eligible gap atomically with a two-marker partition."""
    candidate = labels.copy()
    minimum_seed_separation = float(params.get(
        "minimum_seed_separation_sum_radii", 0.50))
    maximum_seed_distance = float(params.get(
        "maximum_seed_to_foreground_sum_radii", 2.00))
    minimum_area = float(params.get(
        "minimum_basin_expected_area_fraction", 0.15))
    maximum_area = float(params.get(
        "maximum_basin_expected_area_fraction", 2.50))
    used: set[tuple[int, int, int]] = set()
    applications: list[dict] = []
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for _, proposal in eligible.sort_values(
            ["gap_start", "resident_track"]).iterrows():
        resident_owner = int(proposal["resident_owner"])
        companion_owner = int(proposal["companion_owner"])
        conflict_key = (resident_owner, companion_owner)
        if any((int(frame), *sorted(conflict_key)) in used
               for frame in proposal["gap_values"]):
            applications.append({
                "proposal_id": proposal["proposal_id"], "applied": False,
                "changed_pixels": 0, "changed_frames": 0,
                "minimum_resident_basin_ratio": 0.0,
                "minimum_companion_basin_ratio": 0.0,
                "maximum_seed_distance_sum_radii": 0.0,
                "reason": "overlapping_atomic_proposal",
            })
            continue
        trial = candidate.copy()
        reasons: list[str] = []
        frame_changes: list[int] = []
        resident_ratios: list[float] = []
        companion_ratios: list[float] = []
        seed_distances: list[float] = []
        resident_group = proposal["resident_group"]
        before_point = proposal["before_point"]
        after_point = proposal["after_point"]
        end = int(proposal["gap_start"]) - 1
        start = int(proposal["gap_end"]) + 1
        for frame in proposal["gap_values"]:
            resident = _row(resident_group, int(frame))
            if resident is None:
                reasons.append(f"resident_point_missing_frame_{frame}")
                break
            fraction = (int(frame) - end) / max(start - end, 1)
            companion_x = ((1.0 - fraction) * float(before_point.x)
                           + fraction * float(after_point.x))
            companion_y = ((1.0 - fraction) * float(before_point.y)
                           + fraction * float(after_point.y))
            companion_radius = (
                (1.0 - fraction) * float(before_point.radius_px)
                + fraction * float(after_point.radius_px))
            foreground = trial[int(frame)] > 0
            components, _ = ndi.label(foreground, STRUCTURE)
            resident_seed = _nearest(
                foreground, float(resident.x), float(resident.y))
            companion_seed = _nearest(
                foreground, companion_x, companion_y)
            if resident_seed is None or companion_seed is None:
                reasons.append(f"foreground_seed_missing_frame_{frame}")
                break
            resident_pixel, resident_distance = resident_seed
            companion_pixel, companion_distance = companion_seed
            resident_component = int(components[resident_pixel])
            companion_component = int(components[companion_pixel])
            if (resident_component <= 0
                    or resident_component != companion_component):
                reasons.append(f"seats_not_one_shared_mask_frame_{frame}")
                break
            mask = components == resident_component
            owners = set(map(int, np.unique(trial[int(frame)][mask]))) - {0}
            if owners != {resident_owner, companion_owner}:
                reasons.append(f"shared_mask_has_other_owner_frame_{frame}")
                break
            scale = max(float(resident.radius_px) + companion_radius, 1.0)
            maximum_distance_seen = max(
                resident_distance, companion_distance) / scale
            seed_distance = float(np.hypot(
                resident_pixel[0] - companion_pixel[0],
                resident_pixel[1] - companion_pixel[1])) / scale
            seed_distances.append(maximum_distance_seen)
            if maximum_distance_seen > maximum_seed_distance:
                reasons.append(f"seat_too_far_from_shared_mask_frame_{frame}")
                break
            if (resident_pixel == companion_pixel
                    or seed_distance < minimum_seed_separation):
                reasons.append(f"partition_seeds_not_distinct_frame_{frame}")
                break
            markers = np.zeros(mask.shape, dtype=np.uint8)
            markers[resident_pixel] = 1
            markers[companion_pixel] = 2
            partition = watershed(
                np.zeros(mask.shape, dtype=np.float32), markers,
                mask=mask, connectivity=STRUCTURE.astype(bool),
                compactness=1.0)
            resident_basin = partition == 1
            companion_basin = partition == 2
            if (ndi.label(resident_basin, STRUCTURE)[1] != 1
                    or ndi.label(companion_basin, STRUCTURE)[1] != 1
                    or not np.array_equal(
                        resident_basin | companion_basin, mask)):
                reasons.append(f"partition_not_two_connected_basins_frame_{frame}")
                break
            resident_ratio = float(resident_basin.sum()) / max(
                float(proposal["expected_resident_area"]), 1.0)
            companion_ratio = float(companion_basin.sum()) / max(
                float(proposal["expected_companion_area"]), 1.0)
            resident_ratios.append(resident_ratio)
            companion_ratios.append(companion_ratio)
            if not (minimum_area <= resident_ratio <= maximum_area
                    and minimum_area <= companion_ratio <= maximum_area):
                reasons.append(f"partition_area_out_of_range_frame_{frame}")
                break
            before = trial[int(frame)].copy()
            trial[int(frame)][resident_basin] = resident_owner
            trial[int(frame)][companion_basin] = companion_owner
            frame_changes.append(int(np.count_nonzero(
                trial[int(frame)] != before)))
        if (not reasons and owner_consensus.count_new_duplicate_components(
                labels, trial)):
            reasons.append("new_duplicate_component")
        if not reasons and not np.array_equal(trial > 0, labels > 0):
            reasons.append("foreground_changed")
        if not reasons:
            candidate = trial
            for frame in proposal["gap_values"]:
                used.add((int(frame), *sorted(conflict_key)))
        applications.append({
            "proposal_id": proposal["proposal_id"],
            "applied": not reasons,
            "changed_pixels": int(sum(frame_changes)) if not reasons else 0,
            "changed_frames": int(sum(value > 0 for value in frame_changes))
                if not reasons else 0,
            "minimum_resident_basin_ratio": min(resident_ratios, default=0.0),
            "minimum_companion_basin_ratio": min(
                companion_ratios, default=0.0),
            "maximum_seed_distance_sum_radii": max(
                seed_distances, default=0.0),
            "reason": ("reconnection_guided_two_seat_partition"
                       if not reasons else "|".join(reasons)),
        })
    return candidate, pd.DataFrame(applications)


def produce(labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
            reconnections: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                       dict]:
    """Discover and apply only fully evidenced two-seat merge partitions."""
    assert_target_free(params)
    audit, internal = discover(labels, points, reconnections, params)
    candidate, applications = apply(labels, internal, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reconnected companion partition changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("reconnected companion partition overlapped ledger")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError(
            "reconnected companion partition changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "reconnected companion partition created duplicates")
    changed = candidate != labels
    applied = (applications[applications.applied.astype(bool)]
               if len(applications) else applications)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "reconnections_audited": int(len(reconnections)),
        "candidate_pairings_audited": int(len(audit)),
        "eligible_partitions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_partitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(labels), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(duplicates),
    }
    return candidate, unclaimed.copy(), audit, applications, summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    points = pd.read_csv(params["points_path"])
    reconnections = pd.read_csv(params["reconnections_path"])
    audit, internal = discover(labels, points, reconnections, params)
    candidate, applications = apply(labels, internal, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reconnected companion partition changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("reconnected companion partition overlapped ledger")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("reconnected companion partition changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("reconnected companion partition created duplicates")
    stem = str(params.get("output_stem", labels_path.stem))
    labels_output = out.out / f"{stem}.tif"
    ledger_output = out.out / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copyfile(labels_path, labels_output)
    else:
        tifffile.imwrite(
            labels_output, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(unclaimed_path, ledger_output)
    audit_path = out.out / "reconnected_companion_merge_audit.csv"
    application_path = out.out / "reconnected_companion_merge_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "reconnections_audited": int(len(reconnections)),
        "candidate_pairings_audited": int(len(audit)),
        "eligible_partitions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_partitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(labels), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(duplicates),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {
        "labels": labels_output, "unclaimed": ledger_output,
        "audit": audit_path, "applications": application_path,
        "metrics": metrics_path}, "summary": metrics}
