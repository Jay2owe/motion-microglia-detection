"""Partition a right-censored owner collapse into established physical seats.

The detector is field-wide and target-free. It uses complete prior owner
histories, a durable resident seat, a short terminal collapse, and repeated
two-seed partitions. A short leading dormant interval is included only when
the following visible observations repeatedly prove the same two seats.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import component_accounting
import distinct_history_transition_candidates as partition
import separable_merge_recovery as physical


ALLOWED_KEYS = {
    "targeting_mode",
    "minimum_prior_support_movie_fraction", "minimum_prior_owner_purity",
    "minimum_prior_strong_fraction", "maximum_prior_blank_movie_fraction",
    "maximum_leading_dormant_movie_fraction",
    "require_leading_dormant_interval",
    "minimum_collapse_movie_fraction", "maximum_collapse_movie_fraction",
    "maximum_terminal_slack_movie_fraction",
    "minimum_resident_support_movie_fraction",
    "minimum_resident_owner_purity", "minimum_resident_strong_fraction",
    "minimum_separation_sum_radii", "minimum_raw_partition_fraction",
    "minimum_seed_distance_px", "watershed_sigma_px", "minimum_basin_pixels",
    "minimum_expected_area_ratio", "maximum_expected_area_ratio",
}
AUDIT_COLUMNS = [
    "proposal_id", "victim_track", "victim_owner", "resident_owner",
    "transition_frame", "repair_first", "repair_last", "repair_frames",
    "leading_dormant_frames", "prior_support", "prior_purity",
    "prior_strong_fraction", "resident_track", "resident_support",
    "resident_purity", "resident_strong_fraction",
    "minimum_separation_sum_radii", "raw_partition_fraction",
    "changed_pixels", "changed_frames", "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "victim_track", "resident_track",
    "victim_owner", "resident_owner", "separation_sum_radii",
    "partition_method", "victim_pixels", "resident_pixels", "changed_pixels",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("right-censored seat partition must be field-wide")
    unexpected = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unexpected:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(unexpected))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _profile(group: pd.DataFrame, owner: int) -> dict:
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


def _expected_area(labels: np.ndarray, group: pd.DataFrame,
                   owner: int, before: int) -> float:
    areas = []
    for point in group[
            group.frame.astype(int).lt(int(before))
            & group.physically_visible.map(_truth)
            & group.accepted_owner.astype(int).eq(int(owner))].itertuples(
                index=False):
        parts, _ = ndi.label(
            labels[int(point.frame)] == int(owner), partition.STRUCTURE)
        component = physical._component_at(
            parts, float(point.x), float(point.y),
            max(2, int(math.ceil(float(point.radius_px)))))
        if component > 0:
            areas.append(int(np.count_nonzero(parts == component)))
    return float(np.median(areas)) if areas else 0.0


def discover_and_apply(labels: np.ndarray, raw: np.ndarray,
                       points: pd.DataFrame, params: dict
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    index = scored.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in scored.groupby("frame")}
    frame_count = len(labels)
    minimum_prior = int(math.ceil(frame_count * _fraction(
        params, "minimum_prior_support_movie_fraction", 0.08)))
    minimum_prior_purity = _fraction(
        params, "minimum_prior_owner_purity", 0.90)
    minimum_prior_strong = _fraction(
        params, "minimum_prior_strong_fraction", 0.50)
    maximum_prior_blank = int(math.ceil(frame_count * _fraction(
        params, "maximum_prior_blank_movie_fraction", 0.02)))
    maximum_leading = int(math.ceil(frame_count * _fraction(
        params, "maximum_leading_dormant_movie_fraction", 0.02)))
    require_leading = bool(params.get("require_leading_dormant_interval", True))
    minimum_collapse = int(math.ceil(frame_count * _fraction(
        params, "minimum_collapse_movie_fraction", 0.03)))
    maximum_collapse = int(math.ceil(frame_count * _fraction(
        params, "maximum_collapse_movie_fraction", 0.08)))
    terminal_slack = int(math.ceil(frame_count * _fraction(
        params, "maximum_terminal_slack_movie_fraction", 0.02)))
    minimum_resident = int(math.ceil(frame_count * _fraction(
        params, "minimum_resident_support_movie_fraction", 0.12)))
    minimum_resident_purity = _fraction(
        params, "minimum_resident_owner_purity", 0.80)
    minimum_resident_strong = _fraction(
        params, "minimum_resident_strong_fraction", 0.80)
    minimum_separation = float(params.get(
        "minimum_separation_sum_radii", 1.0))
    minimum_raw_fraction = _fraction(
        params, "minimum_raw_partition_fraction", 0.50)
    proposals = []
    plans = []
    number = 0

    resident_profiles = {}
    for track, group in groups.items():
        visible = group[
            group.physically_visible.map(_truth)
            & group.accepted_owner.astype(int).gt(0)]
        counts = visible.accepted_owner.astype(int).value_counts()
        if counts.empty:
            continue
        owner = int(counts.index[0])
        detail = _profile(group, owner)
        if (detail["support"] >= minimum_resident
                and detail["purity"] >= minimum_resident_purity
                and detail["strong_fraction"] >= minimum_resident_strong):
            resident_profiles[int(track)] = {"owner": owner, **detail}

    for victim_track, group in groups.items():
        runs = partition._runs(group, maximum_prior_blank)
        for before, after in zip(runs[:-1], runs[1:]):
            victim_owner, resident_owner = int(before["owner"]), int(after["owner"])
            if (victim_owner <= 0 or resident_owner <= 0
                    or victim_owner == resident_owner):
                continue
            transition = int(after["start"])
            # A physical track can have a short interpolated dormant interval
            # immediately before its new visible owner. Treat that interval as
            # part of the same collapse only when the later frames supply the
            # repeated two-seat evidence required below.
            repair_start = int(before["end"]) + 1
            leading = transition - repair_start
            after_rows = group[group.frame.astype(int).between(
                repair_start, int(after["end"]))]
            usable = after_rows[after_rows.physically_visible.map(_truth)]
            repair_last = int(usable.frame.max()) if len(usable) else -1
            # Include a short leading dormant interval when the following
            # visible frames repeatedly prove the same two seats. Starting
            # after that interval would manufacture a new lineage gap.
            repair_rows = after_rows[after_rows.frame.astype(int).between(
                repair_start, repair_last)] if len(usable) else after_rows.iloc[0:0]
            repair_frames = list(map(int, repair_rows.frame))
            prior = group[
                group.frame.astype(int).lt(transition)
                & group.physically_visible.map(_truth)
                & group.accepted_owner.astype(int).gt(0)]
            matching = prior[prior.accepted_owner.astype(int).eq(victim_owner)]
            prior_support = int(len(matching))
            prior_purity = float(prior_support / max(len(prior), 1))
            prior_strong = float(matching.strong.map(_truth).mean()) \
                if len(matching) else 0.0
            prior_frames = np.sort(matching.frame.astype(int).unique())
            prior_gap = int(np.diff(prior_frames).max()) \
                if len(prior_frames) > 1 else 0
            reasons = []
            if prior_support < minimum_prior:
                reasons.append("insufficient_prior_victim_support")
            if prior_purity < minimum_prior_purity:
                reasons.append("prior_victim_owner_impure")
            if prior_strong < minimum_prior_strong:
                reasons.append("prior_victim_raw_support_weak")
            if prior_gap > maximum_prior_blank + 1:
                reasons.append("prior_victim_path_discontinuous")
            if leading < 0 or leading > maximum_leading:
                reasons.append("leading_dormant_interval_too_long")
            if require_leading and leading < 1:
                reasons.append("no_leading_dormant_interval")
            if not minimum_collapse <= len(repair_frames) <= maximum_collapse:
                reasons.append("visible_collapse_duration_outside_range")
            last_observed = int(group.frame.max())
            if (last_observed < frame_count - 1 - terminal_slack
                    or (repair_frames and repair_frames[-1]
                        < last_observed - terminal_slack)):
                reasons.append("collapse_not_right_censored")
            if any(np.any(labels[frame] == victim_owner)
                   for frame in repair_frames):
                reasons.append("victim_owner_present_during_repair")

            expected_victim = _expected_area(
                labels, group, victim_owner, transition)
            frame_plans = []
            chosen_resident = 0
            failure = ""
            if not reasons:
                for point in repair_rows.itertuples(index=False):
                    frame = int(point.frame)
                    candidates = []
                    for other in by_frame.get(frame, scored.iloc[0:0]).itertuples(
                            index=False):
                        profile = resident_profiles.get(int(other.track_id))
                        if (profile is None
                                or profile["owner"] != resident_owner
                                or not _truth(other.physically_visible)
                                or not _truth(other.strong)):
                            continue
                        shared, component = partition._same_shared_component(
                            labels[frame], resident_owner, point, other)
                        if shared is None:
                            continue
                        separation = float(np.hypot(
                            float(point.x) - float(other.x),
                            float(point.y) - float(other.y))) / max(
                                float(point.radius_px)
                                + float(other.radius_px), 1.0)
                        if separation >= minimum_separation:
                            candidates.append((int(other.track_id), other,
                                               shared, component, separation))
                    if chosen_resident:
                        candidates = [item for item in candidates
                                      if item[0] == chosen_resident]
                    if len(candidates) != 1:
                        failure = "durable_resident_core_not_unique"
                        break
                    resident_track, resident, shared, _, separation = candidates[0]
                    chosen_resident = resident_track
                    expected_resident = _expected_area(
                        labels, groups[resident_track], resident_owner,
                        transition)
                    split = partition._partition(
                        shared, raw[frame], point, resident,
                        expected_victim, expected_resident, params)
                    if split is None:
                        failure = "two_seed_partition_unavailable"
                        break
                    victim_basin, resident_basin, method = split
                    frame_plans.append({
                        "frame": frame, "shared": shared,
                        "victim_basin": victim_basin,
                        "resident_basin": resident_basin,
                        "resident_track": resident_track,
                        "separation": separation, "method": method,
                    })
            raw_fraction = (sum(plan["method"] == "raw_watershed_connected"
                                for plan in frame_plans)
                            / max(len(frame_plans), 1))
            if failure:
                reasons.append(failure)
            if frame_plans and raw_fraction < minimum_raw_fraction:
                reasons.append("insufficient_repeated_raw_partition_support")
            eligible = not reasons and len(frame_plans) == len(repair_frames)
            if eligible:
                number += 1
                proposal_id = f"RCSP{number:04d}"
            else:
                proposal_id = ""
            resident_detail = resident_profiles.get(chosen_resident, {
                "support": 0, "purity": 0.0, "strong_fraction": 0.0})
            public = {
                "proposal_id": proposal_id, "victim_track": victim_track,
                "victim_owner": victim_owner, "resident_owner": resident_owner,
                "transition_frame": transition,
                "repair_first": repair_frames[0] if repair_frames else -1,
                "repair_last": repair_frames[-1] if repair_frames else -1,
                "repair_frames": len(repair_frames),
                "leading_dormant_frames": leading,
                "prior_support": prior_support, "prior_purity": prior_purity,
                "prior_strong_fraction": prior_strong,
                "resident_track": chosen_resident,
                "resident_support": resident_detail["support"],
                "resident_purity": resident_detail["purity"],
                "resident_strong_fraction": resident_detail["strong_fraction"],
                "minimum_separation_sum_radii": min(
                    (plan["separation"] for plan in frame_plans), default=0.0),
                "raw_partition_fraction": raw_fraction,
                "changed_pixels": int(sum(plan["victim_basin"].sum()
                                           for plan in frame_plans))
                if eligible else 0,
                "changed_frames": len(frame_plans) if eligible else 0,
                "eligible": eligible, "applied": False,
                "reason": ("eligible_right_censored_seat_partition"
                           if eligible else "|".join(sorted(set(reasons)))),
            }
            proposals.append(public)
            plans.append({**public, "frame_plans": frame_plans})

    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    frame_rows = []
    for proposal in plans:
        if not proposal["eligible"]:
            continue
        if any(np.any(occupied[int(plan["frame"])] & plan["shared"])
               for plan in proposal["frame_plans"]):
            continue
        trial = candidate.copy()
        for plan in proposal["frame_plans"]:
            frame = int(plan["frame"])
            trial[frame][plan["shared"]] = 0
            trial[frame][plan["victim_basin"]] = int(proposal["victim_owner"])
            trial[frame][plan["resident_basin"]] = int(proposal["resident_owner"])
        if component_accounting.new_duplicate_components(labels, trial):
            continue
        for plan in proposal["frame_plans"]:
            frame = int(plan["frame"])
            candidate[frame][plan["shared"]] = 0
            candidate[frame][plan["victim_basin"]] = int(proposal["victim_owner"])
            candidate[frame][plan["resident_basin"]] = int(proposal["resident_owner"])
            occupied[frame] |= plan["shared"]
            frame_rows.append({
                "proposal_id": proposal["proposal_id"], "frame": frame,
                "victim_track": int(proposal["victim_track"]),
                "resident_track": int(proposal["resident_track"]),
                "victim_owner": int(proposal["victim_owner"]),
                "resident_owner": int(proposal["resident_owner"]),
                "separation_sum_radii": float(plan["separation"]),
                "partition_method": plan["method"],
                "victim_pixels": int(plan["victim_basin"].sum()),
                "resident_pixels": int(plan["resident_basin"].sum()),
                "changed_pixels": int(plan["victim_basin"].sum()),
            })
        proposal["applied"] = True
        proposal["reason"] = "applied_right_censored_seat_partition"
    audit = pd.DataFrame([
        {column: row.get(column, "") for column in AUDIT_COLUMNS}
        for row in plans], columns=AUDIT_COLUMNS)
    frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    return candidate, audit, frames


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                       dict]:
    """Discover and repair fully evidenced terminal two-seat collapses."""
    assert_target_free(params)
    if not labels.shape == unclaimed.shape == raw.shape:
        raise ValueError("labels, unclaimed, and raw stacks must align")
    candidate, audit, frames = discover_and_apply(labels, raw, points, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("right-censored partition changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("right-censored partition overlaps unclaimed data")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("right-censored partition changed identity set")
    duplicates = component_accounting.new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("right-censored partition created duplicates")
    changed = candidate != labels
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "transitions_audited": int(len(audit)),
        "eligible_transitions": int(audit.eligible.astype(bool).sum())
        if len(audit) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.any(changed, axis=(1, 2)).sum()),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_duplicate_components": 0,
    }
    return candidate, unclaimed.copy(), audit, frames, metrics
