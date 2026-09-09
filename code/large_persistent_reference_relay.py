"""Preserve large, stable physical references through multi-owner relays.

Discovery is complete-field and scale-relative.  A reference must be visible
for almost the whole movie, lie in the upper field radius quantile, move
smoothly, begin with a durable owner run, return to that owner repeatedly, and
then be carried by several foreign owners.  No identity, track, frame,
coordinate, event, region, well, or review selector is accepted.

When the large reference is relabelled, a separate component is restored
directly and a shared component is divided by raw-signal watershed.  Existing
pixels carrying the restored owner are either exchanged with a mathematically
reciprocal reference or moved to a new unclaimed ledger entry.  This prevents
the repair from converting a small, remote mask into a duplicate of the large
cell.
"""
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

import component_accounting
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}
PATH_KEYS = {
    "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path",
}
CONTROL_KEYS = {"targeting_mode", "output_stem"}
PARAMETER_KEYS = {
    "minimum_visible_movie_fraction", "minimum_reference_radius_quantile",
    "maximum_anchor_onset_movie_fraction",
    "minimum_anchor_run_movie_fraction",
    "minimum_anchor_return_movie_fraction",
    "minimum_foreign_occupancy_movie_fraction",
    "minimum_distinct_foreign_owners", "maximum_step_sum_radii",
    "maximum_component_area_radius_ratio",
    "minimum_recovered_area_radius_ratio",
    "minimum_reciprocal_orientation_movie_fraction",
    "minimum_reciprocal_consistency",
    "minimum_canonical_support_movie_fraction",
    "minimum_canonical_purity", "minimum_core_separation_sum_radii",
    "maximum_separable_valley_ratio", "watershed_sigma_px",
    "minimum_changed_pixels",
}
ALLOWED_KEYS = PATH_KEYS | CONTROL_KEYS | PARAMETER_KEYS
TRACK_COLUMNS = [
    "proposal_id", "physical_track", "anchor_owner", "visible_frames",
    "visible_fraction", "median_radius_px", "radius_quantile",
    "maximum_step_sum_radii", "anchor_first", "anchor_last",
    "anchor_run_frames", "anchor_return_frames", "foreign_frames",
    "foreign_owner_count", "foreign_owners", "eligible", "applied",
    "applied_runs", "applied_frames", "changed_label_pixels",
    "changed_unclaimed_pixels", "assigned_unclaimed_id", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "physical_track", "frame", "anchor_owner",
    "source_owner", "source_component_pixels",
    "source_component_area_radius_ratio", "application_mode",
    "reciprocal_components", "unclaimed_orphan_pixels",
    "recovered_pixels", "changed_label_pixels",
    "changed_unclaimed_pixels", "valley_ratio", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported large-reference relay parameters: "
            + ", ".join(unsupported))
    supplied = sorted(
        key for key in FORBIDDEN_TARGETS
        if params.get(key) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "large-reference relay received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("large-reference relay must use field-wide discovery")
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing large-reference inputs: " + ", ".join(missing))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    positive = group[
        group.physically_visible.astype(bool)
        & group.accepted_owner.astype(int).gt(0)]
    for row in positive.sort_values("frame").itertuples(index=False):
        frame, owner = int(row.frame), int(row.accepted_owner)
        if (runs and runs[-1]["owner"] == owner
                and runs[-1]["last"] + 1 == frame):
            runs[-1]["last"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "frames": [frame]})
    return runs


def _steps(group: pd.DataFrame) -> list[float]:
    ordered = group.sort_values("frame")
    rows = list(ordered.itertuples(index=False))
    result = []
    for left, right in zip(rows, rows[1:]):
        if int(right.frame) != int(left.frame) + 1:
            continue
        result.append(float(np.hypot(
            float(right.x) - float(left.x),
            float(right.y) - float(left.y))) / max(
                float(left.radius_px) + float(right.radius_px), 1.0))
    return result


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Find large, nearly stationary references with multi-owner relays."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    visible = scored[scored.physically_visible.astype(bool)]
    movie = int(len(labels))
    radius_by_track = visible.groupby("track_id").radius_px.median()
    radius_cutoff = float(radius_by_track.quantile(_fraction(
        params, "minimum_reference_radius_quantile", 0.90)))
    minimum_visible = int(math.ceil(movie * _fraction(
        params, "minimum_visible_movie_fraction", 0.90)))
    maximum_onset = int(math.floor(movie * _fraction(
        params, "maximum_anchor_onset_movie_fraction", 0.10)))
    minimum_anchor = int(math.ceil(movie * _fraction(
        params, "minimum_anchor_run_movie_fraction", 0.15)))
    minimum_return = int(math.ceil(movie * _fraction(
        params, "minimum_anchor_return_movie_fraction", 0.10)))
    minimum_foreign = int(math.ceil(movie * _fraction(
        params, "minimum_foreign_occupancy_movie_fraction", 0.25)))
    minimum_owners = int(params.get("minimum_distinct_foreign_owners", 3))
    maximum_step = float(params.get("maximum_step_sum_radii", 0.75))
    rows: list[dict] = []
    internals: list[dict] = []

    for track_id, group in scored.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        track_visible = group[group.physically_visible.astype(bool)]
        runs = _positive_runs(group)
        first = runs[0] if runs else None
        median_radius = float(track_visible.radius_px.median()) \
            if len(track_visible) else 0.0
        radius_quantile = float((radius_by_track <= median_radius).mean()) \
            if len(radius_by_track) else 0.0
        largest_step = max(_steps(track_visible), default=float("inf"))
        anchor_owner = int(first["owner"]) if first else 0
        anchor_last = int(first["last"]) if first else -1
        later = track_visible[
            track_visible.frame.astype(int).gt(anchor_last)]
        returns = later[
            later.accepted_owner.astype(int).eq(anchor_owner)]
        foreign = later[
            later.accepted_owner.astype(int).gt(0)
            & later.accepted_owner.astype(int).ne(anchor_owner)]
        foreign_owners = sorted(set(map(
            int, foreign.accepted_owner.astype(int).tolist())))
        reasons = []
        if len(track_visible) < minimum_visible:
            reasons.append("insufficient_visible_history")
        if median_radius < radius_cutoff:
            reasons.append("reference_below_field_radius_quantile")
        if largest_step > maximum_step:
            reasons.append("reference_motion_discontinuous")
        if first is None:
            reasons.append("no_positive_starting_owner")
        elif int(first["first"]) > maximum_onset:
            reasons.append("anchor_owner_starts_too_late")
        elif len(first["frames"]) < minimum_anchor:
            reasons.append("anchor_owner_run_too_short")
        if len(returns) < minimum_return:
            reasons.append("anchor_owner_does_not_recur")
        if len(foreign) < minimum_foreign:
            reasons.append("foreign_occupancy_too_short")
        if len(foreign_owners) < minimum_owners:
            reasons.append("too_few_foreign_owners")
        public = {
            "proposal_id": "", "physical_track": int(track_id),
            "anchor_owner": anchor_owner,
            "visible_frames": int(len(track_visible)),
            "visible_fraction": float(len(track_visible) / movie),
            "median_radius_px": median_radius,
            "radius_quantile": radius_quantile,
            "maximum_step_sum_radii": largest_step,
            "anchor_first": int(first["first"]) if first else -1,
            "anchor_last": anchor_last,
            "anchor_run_frames": len(first["frames"]) if first else 0,
            "anchor_return_frames": int(len(returns)),
            "foreign_frames": int(len(foreign)),
            "foreign_owner_count": len(foreign_owners),
            "foreign_owners": "|".join(map(str, foreign_owners)),
            "eligible": not reasons, "applied": False,
            "applied_runs": 0, "applied_frames": 0,
            "changed_label_pixels": 0, "changed_unclaimed_pixels": 0,
            "assigned_unclaimed_id": 0,
            "reason": ("eligible_large_persistent_reference_relay"
                       if not reasons else "|".join(reasons)),
        }
        rows.append(public)
        internals.append({**public, "group": group, "foreign": foreign})

    number = 0
    for public, internal in zip(rows, internals):
        if bool(public["eligible"]):
            number += 1
            proposal_id = f"LPR{number:04d}"
            public["proposal_id"] = internal["proposal_id"] = proposal_id
    return pd.DataFrame(rows, columns=TRACK_COLUMNS), internals, scored


def _inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _component_at_owner(plane: np.ndarray, owner: int, row) -> np.ndarray | None:
    components, _ = ndi.label(plane == int(owner), STRUCTURE)
    component_id = physical._component_at(
        components, float(row.x), float(row.y), search_radius=6)
    return None if component_id <= 0 else components == component_id


def _reciprocal_exchange(scored: pd.DataFrame, target_track: int,
                         other_track: int, anchor_owner: int,
                         host_owner: int, movie: int, params: dict) -> bool:
    left = scored[scored.track_id.astype(int).eq(target_track)][
        ["frame", "physically_visible", "accepted_owner"]].rename(columns={
            "physically_visible": "left_visible",
            "accepted_owner": "left_owner"})
    right = scored[scored.track_id.astype(int).eq(other_track)][
        ["frame", "physically_visible", "accepted_owner"]].rename(columns={
            "physically_visible": "right_visible",
            "accepted_owner": "right_owner"})
    paired = left.merge(right, on="frame")
    paired = paired[
        paired.left_visible.astype(bool) & paired.right_visible.astype(bool)]
    forward = int(np.count_nonzero(
        paired.left_owner.astype(int).eq(anchor_owner)
        & paired.right_owner.astype(int).eq(host_owner)))
    reverse = int(np.count_nonzero(
        paired.left_owner.astype(int).eq(host_owner)
        & paired.right_owner.astype(int).eq(anchor_owner)))
    minimum = int(math.ceil(movie * _fraction(
        params, "minimum_reciprocal_orientation_movie_fraction", 0.04)))
    relevant = paired[
        paired.left_owner.astype(int).isin([anchor_owner, host_owner])
        & paired.right_owner.astype(int).isin([anchor_owner, host_owner])
        & paired.left_owner.astype(int).ne(
            paired.right_owner.astype(int))]
    consistency = float((forward + reverse) / len(relevant)) \
        if len(relevant) else 0.0
    return bool(forward >= minimum and reverse >= minimum
                and consistency >= float(params.get(
                    "minimum_reciprocal_consistency", 0.80)))


def _prepare_frame(candidate: np.ndarray, candidate_unclaimed: np.ndarray,
                   baseline_labels: np.ndarray,
                   baseline_unclaimed: np.ndarray, raw: np.ndarray,
                   scored: pd.DataFrame, by_frame: dict[int, pd.DataFrame],
                   canonical: dict[int, dict], target_row,
                   anchor_owner: int, unclaimed_id: int, params: dict,
                   ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    frame = int(target_row.frame)
    track_id = int(target_row.track_id)
    host_owner = int(target_row.accepted_owner)
    base = {
        "frame": frame, "source_owner": host_owner,
        "source_component_pixels": 0,
        "source_component_area_radius_ratio": 0.0,
        "application_mode": "", "reciprocal_components": 0,
        "unclaimed_orphan_pixels": 0, "recovered_pixels": 0,
        "changed_label_pixels": 0, "changed_unclaimed_pixels": 0,
        "valley_ratio": np.nan, "applied": False, "reason": "",
    }
    if host_owner <= 0 or host_owner == anchor_owner:
        return None, None, {**base, "reason": "not_positive_wrong_owner"}
    shared = _component_at_owner(candidate[frame], host_owner, target_row)
    if shared is None:
        return None, None, {
            **base, "reason": "reference_outside_source_component"}
    source_area = int(np.count_nonzero(shared))
    radius = max(float(target_row.radius_px), 1.0)
    area_ratio = float(source_area / (math.pi * radius ** 2))
    maximum_body_ratio = float(params.get(
        "maximum_component_area_radius_ratio", 4.0))
    companions = []
    if area_ratio > maximum_body_ratio:
        minimum_separation = float(params.get(
            "minimum_core_separation_sum_radii", 0.75))
        for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
            other_track = int(other.track_id)
            stable = canonical.get(other_track)
            if (other_track == track_id or stable is None
                    or int(stable["owner"]) != host_owner
                    or not _inside(shared, other)):
                continue
            separation = float(np.hypot(
                float(target_row.x) - float(other.x),
                float(target_row.y) - float(other.y))) / max(
                    radius + float(other.radius_px), 1.0)
            if separation >= minimum_separation:
                companions.append(other)
        if not companions:
            return None, None, {
                **base, "source_component_pixels": source_area,
                "source_component_area_radius_ratio": area_ratio,
                "reason": "large_shared_component_has_no_stable_host_core"}
        target_marker = physical._marker_pixel(
            shared, float(target_row.x), float(target_row.y))
        if target_marker is None:
            return None, None, {**base, "reason": "target_marker_unavailable"}
        markers = np.zeros(candidate.shape[1:], np.int16)
        markers[target_marker] = 1
        for other in companions:
            marker = physical._marker_pixel(
                shared, float(other.x), float(other.y))
            if marker is not None and marker != target_marker:
                markers[marker] = 2
        if not np.any(markers == 2):
            return None, None, {**base, "reason": "host_markers_unavailable"}
        nearest = min(companions, key=lambda other: float(np.hypot(
            float(target_row.x) - float(other.x),
            float(target_row.y) - float(other.y))))
        valley = physical._valley_ratio(
            raw[frame], (float(target_row.x), float(target_row.y)),
            (float(nearest.x), float(nearest.y)))
        if valley > float(params.get("maximum_separable_valley_ratio", 0.90)):
            return None, None, {
                **base, "valley_ratio": valley,
                "reason": "raw_cores_not_separable"}
        elevation = -ndi.gaussian_filter(
            raw[frame].astype(np.float32),
            float(params.get("watershed_sigma_px", 1.0)))
        recovered = watershed(
            elevation, markers=markers, mask=shared) == 1
        mode = "raw_watershed_partition"
    else:
        recovered = shared.copy()
        valley = np.nan
        mode = "separate_component_relabel"

    recovered_area = int(np.count_nonzero(recovered))
    recovered_ratio = float(recovered_area / (math.pi * radius ** 2))
    if not (float(params.get("minimum_recovered_area_radius_ratio", 0.25))
            <= recovered_ratio <= maximum_body_ratio):
        return None, None, {
            **base, "source_component_pixels": source_area,
            "source_component_area_radius_ratio": area_ratio,
            "recovered_pixels": recovered_area, "valley_ratio": valley,
            "reason": "recovered_body_area_out_of_range"}
    if recovered_area < int(params.get("minimum_changed_pixels", 5)):
        return None, None, {**base, "reason": "recovered_body_too_small"}

    trial = candidate[frame].copy()
    unclaimed_trial = candidate_unclaimed[frame].copy()
    anchor_components, anchor_count = ndi.label(
        trial == anchor_owner, STRUCTURE)
    reciprocal = 0
    orphan_pixels = 0
    for component_id in range(1, anchor_count + 1):
        component = anchor_components == component_id
        # An already-adjacent anchor fragment belongs to the same recovered
        # body.  It is retained only if the union is actually connected.
        if np.any(ndi.binary_dilation(recovered, structure=STRUCTURE) & component):
            continue
        witnesses = [
            other for other in by_frame.get(frame, pd.DataFrame()).itertuples(
                index=False)
            if int(other.track_id) != track_id and _inside(component, other)
            and _reciprocal_exchange(
                scored, track_id, int(other.track_id), anchor_owner,
                host_owner, len(candidate), params)]
        trial[component] = 0
        if witnesses:
            trial[component] = host_owner
            reciprocal += 1
        else:
            unclaimed_trial[component] = unclaimed_id
            orphan_pixels += int(np.count_nonzero(component))

    trial[recovered] = anchor_owner
    unclaimed_trial[recovered] = 0

    if mode == "raw_watershed_partition":
        host_after, host_count = ndi.label(trial == host_owner, STRUCTURE)
        retained_components = set()
        for other in companions:
            component_id = physical._component_at(
                host_after, float(other.x), float(other.y), search_radius=6)
            if component_id > 0:
                retained_components.add(component_id)
        for component_id in range(1, host_count + 1):
            if component_id not in retained_components:
                fragment = host_after == component_id
                if np.any(ndi.binary_dilation(recovered, structure=STRUCTURE)
                          & fragment):
                    trial[fragment] = anchor_owner

    if component_accounting.component_excess(
            trial, anchor_owner) > component_accounting.component_excess(
                baseline_labels[frame], anchor_owner):
        return None, None, {**base, "reason": "new_anchor_owner_duplicate"}
    if component_accounting.component_excess(
            trial, host_owner) > component_accounting.component_excess(
                baseline_labels[frame], host_owner):
        return None, None, {**base, "reason": "new_source_owner_duplicate"}
    if not np.array_equal(
            (trial > 0) | (unclaimed_trial > 0),
            (baseline_labels[frame] > 0) | (baseline_unclaimed[frame] > 0)):
        return None, None, {**base, "reason": "foreground_union_changed"}
    changed_labels = int(np.count_nonzero(trial != candidate[frame]))
    changed_unclaimed = int(np.count_nonzero(
        unclaimed_trial != candidate_unclaimed[frame]))
    return trial, unclaimed_trial, {
        **base, "source_component_pixels": source_area,
        "source_component_area_radius_ratio": area_ratio,
        "application_mode": mode, "reciprocal_components": reciprocal,
        "unclaimed_orphan_pixels": orphan_pixels,
        "recovered_pixels": int(np.count_nonzero(trial == anchor_owner)),
        "changed_label_pixels": changed_labels,
        "changed_unclaimed_pixels": changed_unclaimed,
        "valley_ratio": valley, "reason": "prepared",
    }


def apply(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
          scored: pd.DataFrame, audit: pd.DataFrame, internals: list[dict],
          params: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                                 pd.DataFrame]:
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    visible = scored[scored.physically_visible.astype(bool)]
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    canonical = physical.canonical_owners(
        scored, max(2, int(math.ceil(len(labels) * _fraction(
            params, "minimum_canonical_support_movie_fraction", 0.08)))),
        float(params.get("minimum_canonical_purity", 0.80)), False)
    occupied = np.zeros_like(labels, bool)
    next_unclaimed = int(np.max(unclaimed)) + 1
    frame_rows: list[dict] = []

    for proposal in sorted(
            (value for value in internals if bool(value["eligible"])),
            key=lambda value: (-float(value["median_radius_px"]),
                               int(value["physical_track"]))):
        track_id = int(proposal["physical_track"])
        anchor_owner = int(proposal["anchor_owner"])
        wrong = proposal["foreign"].sort_values("frame")
        run_groups: list[list] = []
        for row in wrong.itertuples(index=False):
            if (run_groups
                    and int(row.frame) == int(run_groups[-1][-1].frame) + 1
                    and int(row.accepted_owner)
                    == int(run_groups[-1][-1].accepted_owner)):
                run_groups[-1].append(row)
            else:
                run_groups.append([row])
        applied_runs = applied_frames = label_changes = unclaimed_changes = 0
        for run_index, run_rows in enumerate(run_groups):
            prepared = []
            refusal = ""
            for target_row in run_rows:
                trial, unclaimed_trial, detail = _prepare_frame(
                    candidate, candidate_unclaimed, labels, unclaimed, raw,
                    scored, by_frame, canonical, target_row, anchor_owner,
                    next_unclaimed, params)
                detail.update({
                    "proposal_id": proposal["proposal_id"],
                    "physical_track": track_id,
                    "anchor_owner": anchor_owner,
                    "run_index": run_index,
                })
                if trial is None or unclaimed_trial is None:
                    refusal = str(detail["reason"])
                    frame_rows.append(detail)
                    break
                frame = int(target_row.frame)
                change = ((trial != candidate[frame])
                          | (unclaimed_trial != candidate_unclaimed[frame]))
                if np.any(occupied[frame] & change):
                    refusal = "overlapping_proposal"
                    frame_rows.append({**detail, "reason": refusal})
                    break
                prepared.append((frame, trial, unclaimed_trial, detail, change))
            if refusal:
                for _, _, _, detail, _ in prepared:
                    frame_rows.append({**detail, "reason": "run_atomic_refusal"})
                continue
            for frame, trial, unclaimed_trial, detail, change in prepared:
                candidate[frame] = trial
                candidate_unclaimed[frame] = unclaimed_trial
                occupied[frame] |= change
                applied_frames += 1
                label_changes += int(detail["changed_label_pixels"])
                unclaimed_changes += int(detail["changed_unclaimed_pixels"])
                frame_rows.append({**detail, "applied": True,
                                   "reason": "applied"})
            if prepared:
                applied_runs += 1
        match = audit.index[
            audit.physical_track.astype(int).eq(track_id)]
        if len(match):
            index = match[0]
            audit.loc[index, "applied"] = applied_frames > 0
            audit.loc[index, "applied_runs"] = applied_runs
            audit.loc[index, "applied_frames"] = applied_frames
            audit.loc[index, "changed_label_pixels"] = label_changes
            audit.loc[index, "changed_unclaimed_pixels"] = unclaimed_changes
            audit.loc[index, "assigned_unclaimed_id"] = (
                next_unclaimed if applied_frames else 0)
            audit.loc[index, "reason"] = (
                "applied_large_persistent_reference_relay"
                if applied_frames else "no_complete_foreign_run_passed")
        if applied_frames:
            next_unclaimed += 1

    return (candidate, candidate_unclaimed, audit,
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict):
    audit, internals, scored = discover(labels, points, params)
    candidate, candidate_unclaimed, audit, frames = apply(
        labels, unclaimed, raw, scored, audit, internals, params)
    baseline_union = (labels > 0) | (unclaimed > 0)
    candidate_union = (candidate > 0) | (candidate_unclaimed > 0)
    if not np.array_equal(candidate_union, baseline_union):
        raise AssertionError("large-reference relay changed foreground union")
    if np.any((unclaimed > 0) & (candidate_unclaimed != unclaimed)):
        raise AssertionError("large-reference relay changed existing unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError("large-reference relay changed movie identity set")
    duplicates = component_accounting.new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"large-reference relay created {duplicates} duplicate components")
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "physical_tracks_audited": int(scored.track_id.nunique()),
        "eligible_large_references": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_large_references": int(len(applied)),
        "applied_runs": int(applied.applied_runs.sum()) if len(applied) else 0,
        "applied_frames": int(applied.applied_frames.sum())
            if len(applied) else 0,
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "changed_frames": int(np.count_nonzero(np.any(
            (candidate != labels) | (candidate_unclaimed != unclaimed),
            axis=(1, 2)))),
        "foreground_ledger_union_exact": True,
        "preexisting_unclaimed_exact": True,
        "movie_identity_set_exact": True,
        "new_duplicate_components": 0,
    }
    return candidate, candidate_unclaimed, audit, frames, summary


def run(_upstream: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, candidate_unclaimed, audit, frames, summary = produce(
        labels, unclaimed, raw, points, params)
    output_dir = Path(out.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": output_dir / f"{stem}.tif",
        "unclaimed": output_dir / f"{stem}_unclaimed_original_ids.tif",
        "audit": output_dir / "large_persistent_reference_relay_audit.csv",
        "frames": output_dir / "large_persistent_reference_relay_frames.csv",
        "metrics": output_dir / "producer_metrics.json",
    }
    tifffile.imwrite(
        outputs["labels"], candidate, imagej=True, compression="zlib",
        photometric="minisblack", metadata={
            "axes": "TYX", "finterval": 1800.0, "tunit": "sec",
            "unit": "pixel"})
    tifffile.imwrite(
        outputs["unclaimed"], candidate_unclaimed, imagej=True,
        compression="zlib", photometric="minisblack", metadata={
            "axes": "TYX", "finterval": 1800.0, "tunit": "sec",
            "unit": "pixel"})
    audit.to_csv(outputs["audit"], index=False)
    frames.to_csv(outputs["frames"], index=False)
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}


