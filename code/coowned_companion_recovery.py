"""Recover a newly visible companion hidden under an established owner.

Discovery is field-wide.  A candidate must be a complete, isolated two-track
episode: both raw tracks share one accepted owner and one connected mask, the
shorter branch is repeatedly observed, the pair is sometimes raw-separable,
and the other branch continues well after the shorter branch converges.  The
producer accepts no identity, track, frame, coordinate, event, or review-case
targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import separable_merge_recovery as accepted_merge


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        name for name in FORBIDDEN_TARGETS
        if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide co-owned companion recovery received forbidden "
            "targets: " + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "co-owned companion recovery must use field-wide discovery")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _mode(values: pd.Series) -> tuple[int, int, float]:
    positive = values.astype(int)
    positive = positive[positive > 0]
    if positive.empty:
        return 0, 0, 0.0
    counts = positive.value_counts()
    owner = int(counts.index[0])
    support = int(counts.iloc[0])
    return owner, support, float(support / len(positive))


def _component_id(components: np.ndarray, row) -> int:
    return accepted_merge._component_at(
        components, float(row.x), float(row.y))


def _component_excess(frame: np.ndarray, identity: int) -> int:
    return max(0, int(ndi.label(frame == identity, STRUCTURE)[1]) - 1)


def new_duplicate_components(baseline: np.ndarray,
                             candidate: np.ndarray) -> int:
    added = 0
    for frame in range(len(candidate)):
        identities = (set(map(int, np.unique(baseline[frame]))) |
                      set(map(int, np.unique(candidate[frame])))) - {0}
        for identity in identities:
            added += max(
                0, _component_excess(candidate[frame], identity)
                - _component_excess(baseline[frame], identity))
    return int(added)


def _track_summary(group: pd.DataFrame) -> dict:
    visible = group[group.physically_visible.astype(bool)].sort_values("frame")
    observed = visible[visible.state == "observed"]
    owner, support, purity = _mode(visible.accepted_owner)
    return {
        "first_frame": int(visible.frame.min()) if len(visible) else -1,
        "last_frame": int(visible.frame.max()) if len(visible) else -1,
        "visible_frames": int(len(visible)),
        "observed_frames": int(len(observed)),
        "strong_observations": int(_as_bool(observed.strong).sum()),
        "modal_owner": owner,
        "owner_support": support,
        "owner_purity": purity,
    }


def discover(labels: np.ndarray, points: pd.DataFrame,
             encounter_frames: pd.DataFrame,
             reconnections: pd.DataFrame | None,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every encountered track pair and return field-wide proposals."""
    assert_target_free(params)
    scored = accepted_merge.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id")}
    summaries = {track: _track_summary(group)
                 for track, group in groups.items()}

    pairs = encounter_frames.copy()
    pairs["left"] = np.minimum(
        pairs.track_a.astype(int), pairs.track_b.astype(int))
    pairs["right"] = np.maximum(
        pairs.track_a.astype(int), pairs.track_b.astype(int))
    partners: dict[int, set[int]] = {track: set() for track in groups}
    for row in pairs[["left", "right"]].drop_duplicates().itertuples(index=False):
        partners.setdefault(int(row.left), set()).add(int(row.right))
        partners.setdefault(int(row.right), set()).add(int(row.left))

    minimum_overlap = int(params.get("minimum_shared_owner_frames", 12))
    minimum_separable = int(params.get("minimum_separable_frames", 3))
    minimum_observed = int(params.get("minimum_companion_observations", 8))
    minimum_strong = int(params.get("minimum_companion_strong_observations", 5))
    minimum_purity = float(params.get("minimum_owner_purity", 0.95))
    minimum_shared_component = float(
        params.get("minimum_shared_component_fraction", 1.0))
    minimum_tail = int(params.get("minimum_survivor_tail_frames", 20))
    minimum_anchor = int(params.get("minimum_owner_anchor_frames", 8))
    minimum_contiguous = float(
        params.get("minimum_overlap_contiguous_fraction", 1.0))
    require_exclusive = bool(params.get("require_exclusive_pair", True))
    require_no_reconnection = bool(
        params.get("require_no_companion_reconnection", True))

    owner_presence = {
        int(owner): np.any(labels == int(owner), axis=(1, 2))
        for owner in np.unique(labels) if int(owner) > 0}
    reconnection_before: set[int] = set()
    if reconnections is not None and len(reconnections) and \
            "track_before" in reconnections.columns:
        reconnection_before = set(map(
            int, reconnections.track_before.dropna().astype(int)))

    rows: list[dict] = []
    pair_groups = pairs.groupby(["left", "right"], sort=True)
    for proposal_number, ((left, right), encounters) in enumerate(
            pair_groups, start=1):
        left, right = int(left), int(right)
        if left not in groups or right not in groups:
            continue
        overlap = groups[left].merge(
            groups[right], on="frame", suffixes=("_left", "_right"))
        same = overlap[
            (overlap.accepted_owner_left > 0)
            & (overlap.accepted_owner_left == overlap.accepted_owner_right)]
        shared_owner, _, _ = _mode(same.accepted_owner_left)
        shared = same[same.accepted_owner_left == shared_owner].copy()
        left_summary, right_summary = summaries[left], summaries[right]
        if left_summary["last_frame"] < right_summary["last_frame"]:
            companion, survivor = left, right
        elif right_summary["last_frame"] < left_summary["last_frame"]:
            companion, survivor = right, left
        else:
            companion = survivor = 0
        companion_summary = summaries.get(companion, {
            "first_frame": -1, "last_frame": -1, "observed_frames": 0,
            "strong_observations": 0, "owner_purity": 0.0,
            "modal_owner": 0})
        survivor_summary = summaries.get(survivor, {
            "last_frame": -1, "owner_purity": 0.0, "modal_owner": 0})

        shared_component_frames = 0
        for row in shared.itertuples(index=False):
            components, _ = ndi.label(
                labels[int(row.frame)] == shared_owner, STRUCTURE)
            left_id = accepted_merge._component_at(
                components, float(row.x_left), float(row.y_left))
            right_id = accepted_merge._component_at(
                components, float(row.x_right), float(row.y_right))
            shared_component_frames += int(left_id > 0 and left_id == right_id)
        shared_component_fraction = (
            shared_component_frames / len(shared) if len(shared) else 0.0)
        separable = int(_as_bool(encounters.separable).sum())
        first_shared = int(shared.frame.min()) if len(shared) else -1
        last_shared = int(shared.frame.max()) if len(shared) else -1
        contiguous_fraction = (
            len(shared) / max(1, last_shared - first_shared + 1)
            if len(shared) else 0.0)
        tail_frames = (
            survivor_summary["last_frame"] - companion_summary["last_frame"]
            if companion and survivor else 0)
        presence = owner_presence.get(
            shared_owner, np.zeros(len(labels), dtype=bool))
        pre_anchor = int(np.count_nonzero(
            presence[:max(0, companion_summary["first_frame"])]))
        post_anchor = int(np.count_nonzero(
            presence[min(len(labels), companion_summary["last_frame"] + 1):]))
        exclusive = bool(
            companion and survivor
            and partners.get(companion, set()) == {survivor}
            and partners.get(survivor, set()) == {companion})
        reasons: list[str] = []
        if companion == 0:
            reasons.append("no_unique_ending_branch")
        if len(shared) < minimum_overlap:
            reasons.append("too_few_shared_owner_frames")
        if separable < minimum_separable:
            reasons.append("too_few_separable_frames")
        if companion_summary["observed_frames"] < minimum_observed:
            reasons.append("too_few_companion_observations")
        if companion_summary["strong_observations"] < minimum_strong:
            reasons.append("too_few_companion_strong_observations")
        if (companion_summary["modal_owner"] != shared_owner
                or survivor_summary["modal_owner"] != shared_owner
                or companion_summary["owner_purity"] < minimum_purity
                or survivor_summary["owner_purity"] < minimum_purity):
            reasons.append("owner_not_pure_on_both_branches")
        if shared_component_fraction < minimum_shared_component:
            reasons.append("not_one_shared_owner_component")
        if contiguous_fraction < minimum_contiguous:
            reasons.append("shared_owner_overlap_not_contiguous")
        if tail_frames < minimum_tail:
            reasons.append("survivor_tail_too_short")
        if pre_anchor < minimum_anchor or post_anchor < minimum_anchor:
            reasons.append("owner_not_anchored_on_both_sides")
        if require_exclusive and not exclusive:
            reasons.append("third_party_encounter_in_lineage")
        if (require_no_reconnection and companion
                and companion in reconnection_before):
            reasons.append("companion_has_later_reconnection")
        rows.append({
            "proposal_id": f"P{proposal_number:04d}",
            "track_left": left, "track_right": right,
            "companion_track": companion, "survivor_track": survivor,
            "shared_owner": shared_owner,
            "first_shared_frame": first_shared,
            "last_shared_frame": last_shared,
            "shared_owner_frames": int(len(shared)),
            "shared_component_frames": int(shared_component_frames),
            "shared_component_fraction": float(shared_component_fraction),
            "overlap_contiguous_fraction": float(contiguous_fraction),
            "separable_frames": separable,
            "companion_observed_frames": int(
                companion_summary["observed_frames"]),
            "companion_strong_observations": int(
                companion_summary["strong_observations"]),
            "companion_owner_purity": float(
                companion_summary["owner_purity"]),
            "survivor_owner_purity": float(
                survivor_summary["owner_purity"]),
            "survivor_tail_frames": int(tail_frames),
            "owner_pre_anchor_frames": pre_anchor,
            "owner_post_anchor_frames": post_anchor,
            "exclusive_pair": exclusive,
            "companion_reconnections": int(
                companion in reconnection_before if companion else 0),
            "median_x": float(
                groups[companion].x.median()) if companion else np.nan,
            "median_y": float(
                groups[companion].y.median()) if companion else np.nan,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })

    proposals = pd.DataFrame(rows)
    if len(proposals):
        eligible = proposals[proposals.discovery_status == "eligible"]
        counts: dict[int, int] = {}
        for row in eligible.itertuples(index=False):
            for track in (int(row.companion_track), int(row.survivor_track)):
                counts[track] = counts.get(track, 0) + 1
        ambiguous = {
            str(row.proposal_id) for row in eligible.itertuples(index=False)
            if counts[int(row.companion_track)] > 1
            or counts[int(row.survivor_track)] > 1}
        if ambiguous:
            mask = proposals.proposal_id.astype(str).isin(ambiguous)
            proposals.loc[mask, "discovery_status"] = "rejected"
            proposals.loc[mask, "discovery_reason"] = \
                "ambiguous_overlapping_proposal"
    return scored, proposals


def _marker(mask: np.ndarray, row) -> tuple[int, int] | None:
    return accepted_merge._marker_pixel(mask, float(row.x), float(row.y))


def _absorb_coreless_owner(
        trial: np.ndarray, shared: np.ndarray, owner: int,
        assigned_identity: int, resident_marker: tuple[int, int],
        ) -> tuple[np.ndarray, np.ndarray]:
    """Keep the resident owner component and absorb split-off coreless pieces."""
    owner_components, _ = ndi.label(trial == owner, STRUCTURE)
    resident_component = int(owner_components[resident_marker])
    if resident_component <= 0:
        raise ValueError("resident owner core was lost")
    coreless_owner = (
        shared & (trial == owner)
        & (owner_components != resident_component))
    result = trial.copy()
    result[coreless_owner] = assigned_identity
    return result, coreless_owner


def _frame_plan(labels: np.ndarray, raw: np.ndarray,
                visible: pd.DataFrame, proposal, params: dict) -> pd.DataFrame:
    owner = int(proposal.shared_owner)
    companion = visible[
        visible.track_id == int(proposal.companion_track)].sort_values("frame")
    rows: list[dict] = []
    for point in companion.itertuples(index=False):
        frame = int(point.frame)
        components, _ = ndi.label(labels[frame] == owner, STRUCTURE)
        component_id = _component_id(components, point)
        shared = components == component_id if component_id > 0 else \
            np.zeros(labels.shape[1:], bool)
        others = visible[
            (visible.frame == frame)
            & (visible.track_id != int(proposal.companion_track))
            & (visible.accepted_owner == owner)]
        resident_rows = [
            other for other in others.itertuples(index=False)
            if _component_id(components, other) == component_id]
        if len(resident_rows) == 1:
            resident = resident_rows[0]
            distance = float(np.hypot(
                float(point.x) - float(resident.x),
                float(point.y) - float(resident.y)))
            valley = accepted_merge._valley_ratio(
                raw[frame], (float(point.x), float(point.y)),
                (float(resident.x), float(resident.y)))
            companion_marker = _marker(shared, point)
            resident_marker = _marker(shared, resident)
            markers_distinct = (
                companion_marker is not None and resident_marker is not None
                and companion_marker != resident_marker)
            resident_track = int(resident.track_id)
        else:
            distance = 0.0
            valley = np.inf
            companion_marker = resident_marker = None
            markers_distinct = False
            resident_track = 0
        rows.append({
            "frame": frame,
            "resident_track": resident_track,
            "companion_x": float(point.x), "companion_y": float(point.y),
            "resident_x": (float(resident.x) if len(resident_rows) == 1
                           else np.nan),
            "resident_y": (float(resident.y) if len(resident_rows) == 1
                           else np.nan),
            "distance_px": distance, "valley_ratio": float(valley),
            "resident_count": int(len(resident_rows)),
            "markers_distinct": bool(markers_distinct),
            "component_area": int(shared.sum()),
            "companion_marker_y": (companion_marker[0]
                                   if companion_marker else -1),
            "companion_marker_x": (companion_marker[1]
                                   if companion_marker else -1),
            "resident_marker_y": (resident_marker[0]
                                  if resident_marker else -1),
            "resident_marker_x": (resident_marker[1]
                                  if resident_marker else -1),
        })
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    minimum_distance = float(params.get("minimum_marker_distance_px", 3.0))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.9))
    identifiable = (
        plan.resident_count.eq(1) & plan.markers_distinct.astype(bool)
        & plan.distance_px.ge(minimum_distance))
    mode = str(params.get("application_mode", "contiguous_to_convergence"))
    if mode == "separable_only":
        selected = identifiable & plan.valley_ratio.le(maximum_valley)
    elif mode == "all_identifiable":
        selected = identifiable
    elif mode == "contiguous_to_convergence":
        convergence = ~identifiable
        run = int(params.get("convergence_run_frames", 2))
        stop = len(plan)
        values = convergence.to_numpy(bool)
        for index in range(0, len(values) - run + 1):
            if bool(np.all(values[index:index + run])):
                stop = index
                break
        selected = pd.Series(False, index=plan.index)
        if stop:
            selected.iloc[:stop] = True
    else:
        raise ValueError(f"unknown application mode: {mode}")
    plan["selected"] = selected.astype(bool)
    return plan


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, encounter_frames: pd.DataFrame,
            reconnections: pd.DataFrame | None, params: dict,
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover and atomically apply every safe co-owned companion proposal."""
    assert_target_free(params)
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")
    scored, proposals = discover(
        labels, points, encounter_frames, reconnections, params)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    next_identity = int(labels.max()) + 1
    applications: list[dict] = []
    frame_audits: list[pd.DataFrame] = []
    minimum_frames = int(params.get("minimum_application_frames", 8))
    minimum_changed = int(params.get("minimum_changed_pixels_per_frame", 5))
    sigma = float(params.get("watershed_sigma_px", 1.0))
    partition_mode = str(params.get("partition_mode", "geodesic"))
    if partition_mode not in {"geodesic", "intensity"}:
        raise ValueError(f"unknown partition mode: {partition_mode}")

    eligible = proposals[proposals.discovery_status == "eligible"].sort_values(
        ["first_shared_frame", "median_y", "median_x", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        assigned_identity = next_identity
        next_identity += 1
        plan = _frame_plan(labels, raw, visible, proposal, params)
        plan.insert(0, "proposal_id", str(proposal.proposal_id))
        plan["assigned_identity"] = assigned_identity
        selected = plan[plan.selected.astype(bool)] if len(plan) else plan
        refusal = ""
        prepared: list[tuple[int, np.ndarray, int]] = []
        if len(selected) < minimum_frames:
            refusal = "too_few_identifiable_application_frames"
        elif (str(params.get("application_mode", "")) == "contiguous_to_convergence"
              and len(selected)
              and not np.all(np.diff(selected.frame.to_numpy(int)) == 1)):
            refusal = "application_interval_not_contiguous"
        if not refusal:
            for row in selected.itertuples(index=False):
                frame = int(row.frame)
                owner = int(proposal.shared_owner)
                components, _ = ndi.label(
                    candidate[frame] == owner, STRUCTURE)
                component_id = int(components[
                    int(row.companion_marker_y),
                    int(row.companion_marker_x)])
                shared = components == component_id if component_id > 0 else \
                    np.zeros(labels.shape[1:], bool)
                companion_marker = (
                    int(row.companion_marker_y), int(row.companion_marker_x))
                resident_marker = (
                    int(row.resident_marker_y), int(row.resident_marker_x))
                if (component_id <= 0 or not shared[resident_marker]
                        or companion_marker == resident_marker):
                    refusal = "application_markers_unavailable"
                    break
                markers = np.zeros(labels.shape[1:], np.int16)
                markers[companion_marker] = 1
                markers[resident_marker] = 2
                elevation = (
                    np.zeros_like(raw[frame], dtype=np.float32)
                    if partition_mode == "geodesic"
                    else -ndi.gaussian_filter(
                        raw[frame].astype(np.float32), sigma))
                partition = watershed(elevation, markers=markers, mask=shared)
                companion_region = partition == 1
                trial = candidate[frame].copy()
                trial[companion_region] = assigned_identity
                try:
                    trial, coreless_owner = _absorb_coreless_owner(
                        trial, shared, owner, assigned_identity,
                        resident_marker)
                except ValueError:
                    refusal = "resident_owner_core_lost"
                    break
                if np.any(coreless_owner):
                    companion_region |= coreless_owner
                changed = int(companion_region.sum())
                if changed < minimum_changed:
                    refusal = "companion_partition_too_small"
                    break
                if not np.any(trial == owner):
                    refusal = "established_owner_extinguished"
                    break
                if _component_excess(trial, owner) > \
                        _component_excess(labels[frame], owner):
                    refusal = "new_established_owner_duplicate"
                    break
                if int(ndi.label(
                        trial == assigned_identity, STRUCTURE)[1]) != 1:
                    refusal = "new_companion_identity_fragmented"
                    break
                if not np.array_equal(trial > 0, labels[frame] > 0):
                    refusal = "foreground_changed"
                    break
                change = trial != candidate[frame]
                if np.any(occupied[frame] & change):
                    refusal = "overlapping_proposal"
                    break
                prepared.append((frame, trial, changed))
        if refusal:
            outcome = "rejected_application"
            changed_pixels = changed_frames = 0
        else:
            for frame, trial, _ in prepared:
                change = trial != candidate[frame]
                candidate[frame] = trial
                occupied[frame] |= change
            outcome = "applied"
            changed_pixels = int(sum(item[2] for item in prepared))
            changed_frames = int(len(prepared))
        plan["application_outcome"] = outcome
        plan["application_reason"] = refusal or "applied"
        frame_audits.append(plan)
        applications.append({
            **proposal._asdict(), "assigned_identity": assigned_identity,
            "application_mode": str(params.get("application_mode")),
            "partition_mode": partition_mode,
            "outcome": outcome, "reason": refusal or "applied",
            "changed_pixels": changed_pixels,
            "changed_frames": changed_frames,
        })

    application_table = pd.DataFrame(applications)
    frame_table = (pd.concat(frame_audits, ignore_index=True)
                   if frame_audits else pd.DataFrame())
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("co-owned companion recovery changed foreground")
    if new_duplicate_components(labels, candidate):
        raise AssertionError(
            "co-owned companion recovery created duplicate components")
    return candidate, proposals, application_table, frame_table


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              params: dict) -> dict:
    changed = candidate != labels
    baseline_ids = set(map(int, np.unique(labels))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    applied = (applications[applications.outcome == "applied"]
               if len(applications) else applications)
    return {
        "targeting_mode": "field_wide_discovery",
        "identity_targets": 0, "track_targets": 0,
        "frame_targets": 0, "coordinate_targets": 0,
        "event_targets": 0,
        "pairs_audited": int(len(proposals)),
        "eligible_proposals": int(
            (proposals.discovery_status == "eligible").sum())
            if len(proposals) else 0,
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "relabelled_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate > 0))),
        "foreground_added_pixels": int(np.count_nonzero(
            (labels == 0) & (candidate > 0))),
        "removed_foreground_pixels": int(np.count_nonzero(
            (labels > 0) & (candidate == 0))),
        "foreground_ledger_changes": int(np.count_nonzero(
            (labels > 0) != (candidate > 0))),
        "unclaimed_overlap_additions": int(np.count_nonzero(
            changed & (candidate > 0) & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(
            (labels == 0) & (candidate > 0) & (raw == 0))),
        "new_same_frame_identity_components": new_duplicate_components(
            labels, candidate),
        "old_identity_set_preserved": bool(
            baseline_ids.issubset(candidate_ids)),
        "new_identities": int(len(candidate_ids - baseline_ids)),
        "active_identities_before": int(len(baseline_ids)),
        "active_identities_after": int(len(candidate_ids)),
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
