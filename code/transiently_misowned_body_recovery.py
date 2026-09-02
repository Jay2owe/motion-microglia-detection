"""Recover isolated ownerless tails after a dormant physical-track bifurcation.

Discovery is field-wide. Identity values are treated as opaque labels and the
producer refuses identity, track, frame, coordinate, event, or review targets.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import ownerless_body_recovery


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals",
}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "track_id", "frame_id", "event_id", "coordinate",
    "review_case", "case_id", "target_identity", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
    "review_region",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and (name in FORBIDDEN_TARGETS or any(
            token in name.lower() for token in FORBIDDEN_TARGET_TOKENS)))
    if supplied:
        raise ValueError(
            "field-wide transient-misownership recovery received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "transient-misownership recovery must use field-wide discovery")


def _owner_runs(group: pd.DataFrame) -> list[dict]:
    rows = list(group.sort_values("frame").itertuples(index=False))
    runs: list[dict] = []
    start = 0
    while start < len(rows):
        owner = int(rows[start].assigned_owner)
        end = start
        while (end + 1 < len(rows)
               and int(rows[end + 1].assigned_owner) == owner
               and int(rows[end + 1].frame) == int(rows[end].frame) + 1):
            end += 1
        runs.append({
            "owner": owner,
            "first_frame": int(rows[start].frame),
            "last_frame": int(rows[end].frame),
            "frames": int(end - start + 1),
        })
        start = end + 1
    return runs


def _distance_to_owner(frame: np.ndarray, owner: int, x: float, y: float,
                       cache: dict[tuple[int, int], np.ndarray],
                       frame_index: int) -> float:
    key = (int(owner), int(frame_index))
    if key not in cache:
        mask = frame == int(owner)
        cache[key] = (ndi.distance_transform_edt(~mask)
                      if np.any(mask)
                      else np.full(frame.shape, np.inf, dtype=np.float32))
    iy = int(np.clip(round(float(y)), 0, frame.shape[0] - 1))
    ix = int(np.clip(round(float(x)), 0, frame.shape[1] - 1))
    return float(cache[key][iy, ix])


def discover(labels: np.ndarray, unclaimed: np.ndarray,
             points: pd.DataFrame, params: dict,
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all physical tracks and return the attached point table."""
    assert_target_free(params)
    if labels.shape != unclaimed.shape:
        raise ValueError("labels and unclaimed stacks must match")
    attached = ownerless_body_recovery.attach_coverage(
        points, labels, unclaimed)
    frame_count = int(len(labels))
    minimum_span_fraction = float(
        params.get("minimum_span_fraction_of_movie", 0.80))
    minimum_visible_fraction = float(
        params.get("minimum_visible_fraction_of_span", 0.90))
    minimum_observed_fraction = float(
        params.get("minimum_observed_fraction_of_span", 0.85))
    minimum_strong = int(params.get("minimum_strong_observations", 10))
    minimum_separation = float(
        params.get("minimum_median_separation_radii", 4.0))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.0))
    maximum_owner_fraction = float(
        params.get("maximum_transient_owner_fraction_of_span", 0.05))
    maximum_owner_start = float(
        params.get("maximum_owner_run_start_fraction_of_span", 0.05))
    minimum_owner_coverage = float(
        params.get("minimum_transient_owner_median_coverage", 0.50))
    minimum_ownerless_tail = float(
        params.get("minimum_ownerless_tail_fraction_of_span", 0.75))
    minimum_tail_presence = float(
        params.get("minimum_owner_tail_presence_fraction", 0.80))
    minimum_tail_median_distance = float(
        params.get("minimum_owner_tail_median_distance_radii", 4.0))
    minimum_tail_distance = float(
        params.get("minimum_owner_tail_minimum_distance_radii", 3.0))
    maximum_unclaimed = float(
        params.get("maximum_unclaimed_visible_fraction", 0.25))
    minimum_dormant_bridge = int(
        params.get("minimum_dormant_bridge_frames", 2))
    maximum_dormant_bridge = int(np.ceil(
        frame_count * float(params.get(
            "maximum_dormant_bridge_fraction_of_movie", 0.05))))
    distance_cache: dict[tuple[int, int], np.ndarray] = {}
    rows: list[dict] = []

    for track_id, group in attached.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        first = int(group.frame.min())
        last = int(group.frame.max())
        span = last - first + 1
        visible = group[group.physically_visible.astype(bool)].copy()
        observed = visible[visible.state == "observed"]
        owners = sorted(set(map(int, group.assigned_owner)) - {0})
        nonzero_runs = [run for run in _owner_runs(group)
                        if int(run["owner"]) > 0]
        transient_owner = owners[0] if len(owners) == 1 else 0
        owner_run = (nonzero_runs[0] if len(nonzero_runs) == 1 else
                     {"owner": 0, "first_frame": -1,
                      "last_frame": -1, "frames": 0})
        owner_rows = group[group.assigned_owner == transient_owner] \
            if transient_owner else group.iloc[:0]
        owner_fraction = float(len(owner_rows) / max(span, 1))
        owner_start_fraction = float(
            (int(owner_run["first_frame"]) - first) / max(span, 1)) \
            if transient_owner else 1.0
        ownerless_tail_fraction = float(
            (last - int(owner_run["last_frame"])) / max(span, 1)) \
            if transient_owner else 0.0
        tail = group[group.frame > int(owner_run["last_frame"])] \
            if transient_owner else group.iloc[:0]
        presence = []
        distances = []
        if transient_owner:
            for point in tail.itertuples(index=False):
                frame = int(point.frame)
                present = bool(np.any(labels[frame] == transient_owner))
                presence.append(present)
                if present:
                    distance = _distance_to_owner(
                        labels[frame], transient_owner, point.x, point.y,
                        distance_cache, frame)
                    distances.append(
                        distance / max(float(point.radius_px), 1.0))
        tail_presence = float(np.mean(presence)) if presence else 0.0
        tail_median_distance = float(np.median(distances)) \
            if distances else 0.0
        tail_minimum_distance = float(np.min(distances)) \
            if distances else 0.0
        radii = np.maximum(visible.radius_px.to_numpy(float), 1.0)
        separation = (visible.nearest_track_distance_px.to_numpy(float)
                      / radii)
        median_separation = float(np.median(separation)) \
            if len(separation) else 0.0
        encounter_fraction = float(
            visible.in_encounter.astype(bool).mean()) \
            if len(visible) else 1.0
        unclaimed_fraction = float(
            (visible.unclaimed_coverage_fraction > 0).mean()) \
            if len(visible) else 1.0
        owner_coverage = float(
            owner_rows.assigned_coverage_fraction.median()) \
            if len(owner_rows) else 0.0
        tail_visible = visible[
            visible.frame > int(owner_run["last_frame"])] \
            if transient_owner else visible.iloc[:0]
        first_tail_visible = int(tail_visible.frame.min()) \
            if len(tail_visible) else -1
        bridge = group[
            (group.frame > int(owner_run["last_frame"]))
            & (group.frame < first_tail_visible)] \
            if transient_owner and first_tail_visible >= 0 else group.iloc[:0]
        dormant_bridge_frames = int(len(bridge))
        bridge_all_dormant = bool(
            len(bridge) and (bridge.state == "dormant").all())
        reasons: list[str] = []
        expected_frames = np.arange(first, last + 1)
        if not np.array_equal(group.frame.to_numpy(int), expected_frames):
            reasons.append("physical_track_not_frame_continuous")
        if span / max(frame_count, 1) < minimum_span_fraction:
            reasons.append("track_span_too_short")
        if len(visible) / max(span, 1) < minimum_visible_fraction:
            reasons.append("insufficient_visible_fraction")
        if len(observed) / max(span, 1) < minimum_observed_fraction:
            reasons.append("insufficient_observed_fraction")
        if int(observed.strong.astype(bool).sum()) < minimum_strong:
            reasons.append("too_few_strong_observations")
        if median_separation < minimum_separation:
            reasons.append("insufficient_isolation")
        if encounter_fraction > maximum_encounter:
            reasons.append("encountered_other_body")
        if len(owners) != 1:
            reasons.append("not_exactly_one_transient_owner")
        if len(nonzero_runs) != 1:
            reasons.append("transient_owner_not_one_contiguous_run")
        if owner_fraction > maximum_owner_fraction:
            reasons.append("owner_run_not_transient")
        if owner_start_fraction > maximum_owner_start:
            reasons.append("owner_run_not_at_track_start")
        if owner_coverage < minimum_owner_coverage:
            reasons.append("transient_owner_coverage_too_weak")
        if dormant_bridge_frames < minimum_dormant_bridge:
            reasons.append("no_dormant_bifurcation_bridge")
        if dormant_bridge_frames > maximum_dormant_bridge:
            reasons.append("dormant_bridge_too_long")
        if not bridge_all_dormant:
            reasons.append("bifurcation_bridge_not_fully_dormant")
        if ownerless_tail_fraction < minimum_ownerless_tail:
            reasons.append("ownerless_tail_too_short")
        if tail_presence < minimum_tail_presence:
            reasons.append("transient_owner_does_not_continue_elsewhere")
        if tail_median_distance < minimum_tail_median_distance:
            reasons.append("owner_tail_not_spatially_distinct")
        if tail_minimum_distance < minimum_tail_distance:
            reasons.append("owner_tail_initially_ambiguous")
        if unclaimed_fraction > maximum_unclaimed:
            reasons.append("too_much_unclaimed_ledger_overlap")
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "track_id": int(track_id),
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "span_fraction_of_movie": float(span / max(frame_count, 1)),
            "visible_frames": int(len(visible)),
            "visible_fraction_of_span": float(len(visible) / max(span, 1)),
            "observed_frames": int(len(observed)),
            "observed_fraction_of_span": float(len(observed) / max(span, 1)),
            "strong_observations": int(observed.strong.astype(bool).sum()),
            "median_separation_radii": median_separation,
            "encounter_fraction": encounter_fraction,
            "distinct_assigned_owners": int(len(owners)),
            "transient_owner": transient_owner,
            "owner_run_first": int(owner_run["first_frame"]),
            "owner_run_last": int(owner_run["last_frame"]),
            "owner_run_frames": int(owner_run["frames"]),
            "owner_fraction_of_span": owner_fraction,
            "owner_run_start_fraction": owner_start_fraction,
            "owner_median_coverage": owner_coverage,
            "first_tail_visible_frame": first_tail_visible,
            "dormant_bridge_frames": dormant_bridge_frames,
            "bridge_all_dormant": bridge_all_dormant,
            "ownerless_tail_fraction": ownerless_tail_fraction,
            "owner_tail_presence_fraction": tail_presence,
            "owner_tail_median_distance_radii": tail_median_distance,
            "owner_tail_minimum_distance_radii": tail_minimum_distance,
            "unclaimed_visible_fraction": unclaimed_fraction,
            "median_x": float(visible.x.median()) if len(visible) else np.nan,
            "median_y": float(visible.y.median()) if len(visible) else np.nan,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows), attached


def _overlapping_components(frame: np.ndarray, values: Iterable[int],
                            seed: np.ndarray) -> np.ndarray:
    selected = np.zeros_like(frame, dtype=bool)
    for value in sorted(set(map(int, values)) - {0}):
        components, _ = ndi.label(frame == value, STRUCTURE)
        ids = set(map(int, np.unique(components[seed]))) - {0}
        for component_id in ids:
            selected |= components == component_id
    return selected


def _duplicate_components(labels: np.ndarray, identities: Iterable[int]) -> int:
    total = 0
    for identity in identities:
        for frame in labels:
            total += max(
                0, ndi.label(frame == int(identity), STRUCTURE)[1] - 1)
    return int(total)


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                       pd.DataFrame, pd.DataFrame]:
    """Apply each eligible proposal atomically using raw-connected cores."""
    proposals, attached = discover(labels, unclaimed, points, params)
    if labels.shape != raw.shape:
        raise ValueError("labels, unclaimed, and aligned raw stacks must match")
    threshold_by_frame = thresholds.set_index("frame")
    maximum_area_ratio = float(
        params.get("maximum_core_area_radius_ratio", 2.5))
    minimum_application_fraction = float(
        params.get("minimum_application_fraction_of_visible", 0.75))
    include_latent = bool(params.get("include_latent_visible", True))
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    next_identity = int(candidate.max()) + 1
    audit_rows: list[dict] = []

    for proposal in proposals.itertuples(index=False):
        base = proposal._asdict()
        if proposal.discovery_status != "eligible":
            audit_rows.append({
                **base, "assigned_identity": 0,
                "application_status": "rejected_discovery",
                "application_reason": proposal.discovery_reason,
                "changed_pixels": 0, "changed_frames": 0,
                "relabelled_owner_pixels": 0,
                "claimed_unclaimed_pixels": 0,
                "added_background_pixels": 0,
                "skipped_core_frames": 0,
            })
            continue
        identity = next_identity
        trial = candidate.copy()
        trial_unclaimed = candidate_unclaimed.copy()
        track = attached[
            (attached.track_id == int(proposal.track_id))
            & (attached.frame > int(proposal.owner_run_last))] \
            .sort_values("frame")
        allowed_states = {"observed", "latent_visible"} \
            if include_latent else {"observed"}
        track = track[track.state.isin(allowed_states)]
        changed_frames = 0
        relabelled = 0
        claimed = 0
        added = 0
        skipped = 0
        rejection = ""
        for point in track.itertuples(index=False):
            frame_index = int(point.frame)
            core, region, details = ownerless_body_recovery._connected_core(
                raw[frame_index], point,
                float(threshold_by_frame.loc[frame_index, "weak_threshold"]),
                params)
            expected = np.pi * max(float(point.radius_px), 1.0) ** 2
            area_ratio = details["core_area_px"] / max(expected, 1.0)
            if not details["core_area_px"] or area_ratio > maximum_area_ratio:
                skipped += 1
                continue
            seed = np.zeros_like(trial[frame_index], dtype=bool)
            seed[region] = core
            accepted_values = np.unique(trial[frame_index][seed])
            accepted_values = accepted_values[accepted_values > 0]
            if len(accepted_values):
                rejection = "ownerless_tail_raw_core_overlaps_accepted_identity"
                break
            else:
                owner_component = np.zeros_like(seed)
            ledger_values = np.unique(trial_unclaimed[frame_index][seed])
            ledger_values = ledger_values[ledger_values > 0]
            ledger_components = _overlapping_components(
                trial_unclaimed[frame_index], ledger_values, seed)
            if (np.count_nonzero(ledger_components) / max(expected, 1.0)
                    > maximum_area_ratio):
                rejection = "unclaimed_component_too_large"
                break
            proposed = seed | owner_component | ledger_components
            conflicting = (trial[frame_index] > 0) & proposed
            if np.any(conflicting):
                rejection = "proposed_mask_overlaps_other_identity"
                break
            blank_additions = proposed & (trial[frame_index] == 0)
            if np.any(blank_additions & (raw[frame_index] <= 0)):
                rejection = "proposed_background_has_zero_raw_signal"
                break
            claimed += int(np.count_nonzero(
                proposed & (trial_unclaimed[frame_index] > 0)))
            added += int(np.count_nonzero(
                proposed & (trial[frame_index] == 0)
                & (trial_unclaimed[frame_index] == 0)))
            trial[frame_index][proposed] = identity
            trial_unclaimed[frame_index][proposed] = 0
            changed_frames += 1
        required = int(np.ceil(
            minimum_application_fraction * int(proposal.visible_frames)))
        if not rejection and changed_frames < required:
            rejection = "too_few_raw_supported_application_frames"
        if rejection:
            audit_rows.append({
                **base, "assigned_identity": 0,
                "application_status": "rejected_application",
                "application_reason": rejection,
                "changed_pixels": 0, "changed_frames": 0,
                "relabelled_owner_pixels": 0,
                "claimed_unclaimed_pixels": 0,
                "added_background_pixels": 0,
                "skipped_core_frames": skipped,
            })
            continue
        changed = trial != candidate
        changed_pixels = int(np.count_nonzero(changed))
        candidate = trial
        candidate_unclaimed = trial_unclaimed
        next_identity += 1
        audit_rows.append({
            **base, "assigned_identity": identity,
            "application_status": "applied",
            "application_reason": "applied",
            "changed_pixels": changed_pixels,
            "changed_frames": changed_frames,
            "relabelled_owner_pixels": relabelled,
            "claimed_unclaimed_pixels": claimed,
            "added_background_pixels": added,
            "skipped_core_frames": skipped,
        })
    audit = pd.DataFrame(audit_rows)
    applied = audit[audit.application_status == "applied"] \
        if len(audit) else audit
    new_ids = list(map(int, applied.assigned_identity)) \
        if len(applied) else []
    if _duplicate_components(candidate, new_ids):
        raise AssertionError(
            "transient-misownership recovery created duplicate components")
    changed_existing = (labels > 0) & (candidate != labels)
    if np.any(changed_existing):
        raise AssertionError(
            "ownerless-tail recovery changed an accepted foreground pixel")
    cleared = (unclaimed > 0) & (candidate_unclaimed == 0)
    if np.any(cleared & (candidate == 0)):
        raise AssertionError("cleared unclaimed pixels were not assigned")
    if np.any((candidate_unclaimed != unclaimed) & ~cleared):
        raise AssertionError("unclaimed ledger changed outside claimed pixels")
    if np.any((candidate > 0) & (labels == 0) & (raw <= 0)):
        raise AssertionError("recovery added a zero-signal foreground pixel")
    return candidate, candidate_unclaimed, proposals, audit, attached


def summarize(labels: np.ndarray, unclaimed: np.ndarray,
              candidate: np.ndarray, candidate_unclaimed: np.ndarray,
              proposals: pd.DataFrame, audit: pd.DataFrame) -> dict:
    old_ids = set(map(int, np.unique(labels))) - {0}
    new_ids = set(map(int, np.unique(candidate))) - {0}
    changed = candidate != labels
    ledger_changed = candidate_unclaimed != unclaimed
    applied = audit[audit.application_status == "applied"] \
        if len(audit) else audit
    return {
        "audited_tracks": int(len(proposals)),
        "eligible_proposals": int(
            (proposals.discovery_status == "eligible").sum()),
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
        "relabelled_owner_pixels": int(
            applied.relabelled_owner_pixels.sum()) if len(applied) else 0,
        "claimed_unclaimed_pixels": int(
            applied.claimed_unclaimed_pixels.sum()) if len(applied) else 0,
        "added_background_pixels": int(
            applied.added_background_pixels.sum()) if len(applied) else 0,
        "unclaimed_ledger_changed_pixels": int(np.count_nonzero(ledger_changed)),
        "foreground_added_pixels": int(np.count_nonzero(
            (candidate > 0) & (labels == 0))),
        "foreground_removed_pixels": int(np.count_nonzero(
            (candidate == 0) & (labels > 0))),
        "old_identity_set_preserved": bool(old_ids <= new_ids),
        "new_identities": int(len(new_ids - old_ids)),
        "identity_targets": 0,
        "track_targets": 0,
        "frame_targets": 0,
        "coordinate_targets": 0,
        "event_targets": 0,
    }
