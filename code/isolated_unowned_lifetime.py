"""Allocate identities to complete isolated raw-signal lifetimes.

Discovery audits every physical track. The producer accepts arrays and
mathematical thresholds only; identities, tracks, frames, coordinates, events,
and review cases are forbidden as targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import ownerless_body_recovery
import component_accounting
from raw_physical_hypotheses import evidence_image


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "track_id", "frame_id", "event_id", "coordinate",
    "review_case", "case_id", "target_identity", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
    "review_region",
)
APPLICATION_COLUMNS = [
    "proposal_id", "physical_track", "assigned_identity", "outcome",
    "reason", "changed_pixels", "changed_frames",
    "claimed_unclaimed_pixels", "raw_supported_additions",
    "skipped_core_frames",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "raw_core_pixels", "changed_pixels",
    "claimed_unclaimed_pixels", "raw_supported_additions",
]


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and any(token in name.lower() for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "field-wide isolated-lifetime recovery received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "isolated-lifetime recovery must use field-wide discovery")


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    physical_track: int
    first_frame: int
    last_frame: int
    span_frames: int
    visible_frames: int
    observed_frames: int
    strong_observations: int
    strong_observation_fraction: float
    visible_fraction_of_span: float
    observed_fraction_of_span: float
    maximum_assigned_coverage: float
    positive_unclaimed_values: int
    encounter_fraction: float
    median_separation_radii: float
    terminal_to_initial_evidence_ratio: float
    final_visibility: float
    compatible_predecessors: int
    compatible_successors: int
    discovery_status: str
    discovery_reason: str


def _endpoint_compatibility(group: pd.DataFrame, all_points: pd.DataFrame,
                            params: dict) -> tuple[int, int]:
    gap_limit = int(params.get("maximum_endpoint_gap_frames", 2))
    step_limit = float(params.get(
        "maximum_endpoint_step_sum_radii", 2.0))
    first = group.iloc[0]
    last = group.iloc[-1]
    track = int(first.track_id)
    endpoints = all_points.sort_values("frame").groupby(
        "track_id", sort=True).agg(
            first_frame=("frame", "first"), last_frame=("frame", "last"),
            first_x=("x", "first"), first_y=("y", "first"),
            first_radius=("radius_px", "first"),
            last_x=("x", "last"), last_y=("y", "last"),
            last_radius=("radius_px", "last"))
    endpoints = endpoints[endpoints.index.astype(int) != track]
    predecessor_gap = int(first.frame) - endpoints.last_frame.astype(int)
    predecessor_distance = np.hypot(
        endpoints.last_x.to_numpy(float) - float(first.x),
        endpoints.last_y.to_numpy(float) - float(first.y))
    predecessor_scale = np.maximum(
        endpoints.last_radius.to_numpy(float) + float(first.radius_px), 1.0)
    predecessors = int(np.count_nonzero(
        (predecessor_gap.to_numpy() >= 1)
        & (predecessor_gap.to_numpy() <= gap_limit)
        & (predecessor_distance / predecessor_scale <= step_limit)))
    successor_gap = endpoints.first_frame.astype(int) - int(last.frame)
    successor_distance = np.hypot(
        endpoints.first_x.to_numpy(float) - float(last.x),
        endpoints.first_y.to_numpy(float) - float(last.y))
    successor_scale = np.maximum(
        endpoints.first_radius.to_numpy(float) + float(last.radius_px), 1.0)
    successors = int(np.count_nonzero(
        (successor_gap.to_numpy() >= 1)
        & (successor_gap.to_numpy() <= gap_limit)
        & (successor_distance / successor_scale <= step_limit)))
    return predecessors, successors


def discover(labels: np.ndarray, unclaimed: np.ndarray,
             points: pd.DataFrame, params: dict,
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    attached = ownerless_body_recovery.attach_coverage(
        points, labels, unclaimed)
    frame_count = int(len(labels))
    minimum_span = int(np.ceil(frame_count * float(
        params.get("minimum_span_fraction_of_movie", 0.18))))
    minimum_visible = float(params.get(
        "minimum_visible_fraction_of_span", 0.95))
    minimum_observed = float(params.get(
        "minimum_observed_fraction_of_span", 0.90))
    minimum_strong = int(params.get("minimum_strong_observations", 10))
    minimum_strong_fraction = float(params.get(
        "minimum_strong_observation_fraction", 0.60))
    maximum_coverage = float(params.get(
        "maximum_assigned_coverage_fraction", 0.0))
    maximum_ledger_values = int(params.get(
        "maximum_positive_unclaimed_values", 1))
    maximum_encounter = float(params.get(
        "maximum_encounter_fraction", 0.0))
    minimum_separation = float(params.get(
        "minimum_median_separation_radii", 8.0))
    maximum_fade_ratio = float(params.get(
        "maximum_terminal_to_initial_evidence_ratio", 0.25))
    maximum_final_visibility = float(params.get(
        "maximum_final_visibility", 0.50))
    allow_recording_boundary = bool(params.get(
        "allow_recording_boundary_lifetime", False))
    boundary_end = int(np.floor(
        (frame_count - 1) * float(params.get(
            "minimum_recording_boundary_end_fraction", 0.99))))
    rows: list[dict] = []

    for track, group in attached.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        first = int(group.frame.min())
        last = int(group.frame.max())
        span = last - first + 1
        visible = group[group.physically_visible.astype(bool)].copy()
        observed = visible[visible.state == "observed"]
        strong = int(observed.strong.astype(bool).sum())
        strong_fraction = strong / max(len(observed), 1)
        assigned_max = float(
            visible.assigned_coverage_fraction.max()) if len(visible) else 1.0
        ledger_values = set(visible.loc[
            visible.unclaimed_owner.astype(int) > 0,
            "unclaimed_owner"].astype(int))
        encounter_fraction = float(
            visible.in_encounter.astype(bool).mean()) if len(visible) else 1.0
        radii = np.maximum(visible.radius_px.to_numpy(float), 1.0)
        separations = (
            visible.nearest_track_distance_px.to_numpy(float) / radii)
        median_separation = float(np.median(separations)) \
            if len(separations) else 0.0
        evidence = visible.evidence.to_numpy(float)
        quarter = max(1, len(evidence) // 4)
        first_half = max(1, len(evidence) // 2)
        evidence_ratio = float(
            np.median(evidence[-quarter:])
            / max(float(np.median(evidence[:first_half])), 1.0)) \
            if len(evidence) else np.inf
        final_visibility = float(visible.visibility.iloc[-1]) \
            if len(visible) else np.inf
        predecessors, successors = _endpoint_compatibility(
            group, attached, params)
        reasons: list[str] = []
        expected = np.arange(first, last + 1)
        if not np.array_equal(group.frame.astype(int).to_numpy(), expected):
            reasons.append("physical_track_not_frame_continuous")
        if span < minimum_span:
            reasons.append("track_span_too_short")
        if len(visible) / max(span, 1) < minimum_visible:
            reasons.append("insufficient_visible_fraction")
        if len(observed) / max(span, 1) < minimum_observed:
            reasons.append("insufficient_observed_fraction")
        if strong < minimum_strong:
            reasons.append("too_few_strong_observations")
        if strong_fraction < minimum_strong_fraction:
            reasons.append("insufficient_strong_observation_fraction")
        if assigned_max > maximum_coverage:
            reasons.append("accepted_owner_already_present")
        if len(ledger_values) > maximum_ledger_values:
            reasons.append("multiple_unclaimed_lineages_overlap_lifetime")
        if encounter_fraction > maximum_encounter:
            reasons.append("physical_encounter_present")
        if median_separation < minimum_separation:
            reasons.append("insufficient_isolation")
        reaches_recording_boundary = last >= boundary_end
        if not (allow_recording_boundary and reaches_recording_boundary):
            if evidence_ratio > maximum_fade_ratio:
                reasons.append("lifetime_does_not_end_in_evidence_fade")
            if final_visibility > maximum_final_visibility:
                reasons.append("lifetime_does_not_end_at_low_visibility")
        if predecessors:
            reasons.append("compatible_predecessor_exists")
        if successors:
            reasons.append("compatible_successor_exists")
        rows.append(asdict(Proposal(
            proposal_id=f"P{len(rows) + 1:04d}",
            physical_track=int(track), first_frame=first, last_frame=last,
            span_frames=span, visible_frames=int(len(visible)),
            observed_frames=int(len(observed)), strong_observations=strong,
            strong_observation_fraction=float(strong_fraction),
            visible_fraction_of_span=float(len(visible) / max(span, 1)),
            observed_fraction_of_span=float(len(observed) / max(span, 1)),
            maximum_assigned_coverage=assigned_max,
            positive_unclaimed_values=int(len(ledger_values)),
            encounter_fraction=encounter_fraction,
            median_separation_radii=median_separation,
            terminal_to_initial_evidence_ratio=evidence_ratio,
            final_visibility=final_visibility,
            compatible_predecessors=predecessors,
            compatible_successors=successors,
            discovery_status="eligible" if not reasons else "rejected",
            discovery_reason="eligible" if not reasons else "|".join(reasons),
        )))
    return (pd.DataFrame(rows, columns=list(Proposal.__dataclass_fields__)),
            attached)


def _components_overlapping(frame: np.ndarray, seed: np.ndarray) -> np.ndarray:
    selected = np.zeros_like(seed)
    for value in sorted(set(map(int, np.unique(frame[seed]))) - {0}):
        components, _ = ndi.label(frame == value, STRUCTURE)
        for component in set(map(int, np.unique(components[seed]))) - {0}:
            selected |= components == component
    return selected


def _point_inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _duplicate_components(stack: np.ndarray) -> int:
    return component_accounting.total_component_excess(stack)


def _connected_core(frame: np.ndarray, point, weak_threshold: float,
                    params: dict) -> tuple[np.ndarray,
                                           tuple[slice, slice], dict]:
    """Use the track centre, then a bounded peak search for latent points."""
    core, region, details = ownerless_body_recovery._connected_core(
        frame, point, weak_threshold, params)
    details["seed_displacement_px"] = 0.0
    if details["core_area_px"]:
        return core, region, details
    evidence = evidence_image(frame, params)
    y = float(point.y)
    x = float(point.x)
    search_radius = max(
        1.0, float(params.get("maximum_seed_displacement_radius_ratio", 1.5))
        * max(float(point.radius_px), 1.0))
    y0 = max(0, int(np.floor(y - search_radius)))
    y1 = min(frame.shape[0], int(np.ceil(y + search_radius + 1)))
    x0 = max(0, int(np.floor(x - search_radius)))
    x1 = min(frame.shape[1], int(np.ceil(x + search_radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    allowed = ((xx - x) ** 2 + (yy - y) ** 2 <= search_radius ** 2) \
        & (frame[y0:y1, x0:x1] > 0)
    if not np.any(allowed):
        return core, region, details
    scores = np.where(allowed, evidence[y0:y1, x0:x1], -np.inf)
    local_y, local_x = np.unravel_index(int(np.argmax(scores)), scores.shape)

    class Seed:
        pass

    seed = Seed()
    seed.y = y0 + int(local_y)
    seed.x = x0 + int(local_x)
    fallback_core, fallback_region, fallback_details = \
        ownerless_body_recovery._connected_core(
            frame, seed, weak_threshold, params)
    fallback_details["seed_displacement_px"] = float(np.hypot(
        float(seed.x) - x, float(seed.y) - y))
    return fallback_core, fallback_region, fallback_details


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                       pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proposals, attached = discover(labels, unclaimed, points, params)
    threshold_by_frame = thresholds.set_index("frame")
    maximum_area_ratio = float(params.get(
        "maximum_core_area_radius_ratio", 2.5))
    minimum_application = float(params.get(
        "minimum_lifetime_application_fraction", 0.95))
    include_latent = bool(params.get("include_latent_visible", True))
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    next_identity = int(labels.max()) + 1
    applications: list[dict] = []
    frame_rows: list[dict] = []

    eligible = proposals[
        proposals.discovery_status == "eligible"].sort_values(
            ["first_frame", "physical_track", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        trial = candidate.copy()
        trial_unclaimed = candidate_unclaimed.copy()
        assigned_identity = next_identity
        states = VISIBLE_STATES if include_latent else {"observed"}
        track = attached[
            (attached.track_id.astype(int) == int(proposal.physical_track))
            & attached.state.isin(states)].sort_values("frame")
        changed_frames = 0
        claimed = 0
        added = 0
        skipped = 0
        rejection = ""
        local_rows: list[dict] = []
        for point in track.itertuples(index=False):
            frame = int(point.frame)
            core, region, details = _connected_core(
                raw[frame], point,
                float(threshold_by_frame.loc[frame, "weak_threshold"]),
                params)
            expected_area = np.pi * max(float(point.radius_px), 1.0) ** 2
            area_ratio = details["core_area_px"] / max(expected_area, 1.0)
            if not details["core_area_px"] or area_ratio > maximum_area_ratio:
                skipped += 1
                continue
            seed = np.zeros_like(trial[frame], dtype=bool)
            seed[region] = core
            ledger = _components_overlapping(trial_unclaimed[frame], seed)
            proposed = seed | ledger
            if np.any((trial[frame] > 0) & proposed):
                rejection = "proposed_component_overlaps_accepted_identity"
                break
            occupants = [
                int(other.track_id)
                for other in attached[
                    (attached.frame.astype(int) == frame)
                    & attached.physically_visible.astype(bool)]
                .itertuples(index=False)
                if int(other.track_id) != int(proposal.physical_track)
                and _point_inside(proposed, other)]
            if occupants:
                rejection = "proposed_component_contains_other_physical_body"
                break
            if np.any(proposed & (trial_unclaimed[frame] == 0)
                      & (raw[frame] <= 0)):
                rejection = "raw_supported_addition_has_zero_signal"
                break
            before = trial[frame].copy()
            before_unclaimed = trial_unclaimed[frame].copy()
            trial[frame][proposed] = assigned_identity
            trial_unclaimed[frame][proposed] = 0
            changed = trial[frame] != before
            if np.any(changed):
                changed_frames += 1
            frame_claimed = int(np.count_nonzero(
                (before_unclaimed > 0) & (trial_unclaimed[frame] == 0)))
            frame_added = int(np.count_nonzero(
                changed & (before == 0) & (before_unclaimed == 0)))
            claimed += frame_claimed
            added += frame_added
            local_rows.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "raw_core_pixels": int(details["core_area_px"]),
                "changed_pixels": int(np.count_nonzero(changed)),
                "claimed_unclaimed_pixels": frame_claimed,
                "raw_supported_additions": frame_added,
            })
        required = int(np.ceil(minimum_application * len(track)))
        if not rejection and changed_frames < required:
            rejection = "too_few_supported_lifetime_frames"
        if rejection:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "physical_track": int(proposal.physical_track),
                "assigned_identity": 0, "outcome": "rejected_application",
                "reason": rejection, "changed_pixels": 0,
                "changed_frames": 0, "claimed_unclaimed_pixels": 0,
                "raw_supported_additions": 0,
                "skipped_core_frames": skipped,
            })
            continue
        changed = trial != candidate
        candidate = trial
        candidate_unclaimed = trial_unclaimed
        next_identity += 1
        applications.append({
            "proposal_id": proposal.proposal_id,
            "physical_track": int(proposal.physical_track),
            "assigned_identity": assigned_identity, "outcome": "applied",
            "reason": "complete_isolated_lifetime_allocated",
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(
                np.any(changed, axis=(1, 2)))),
            "claimed_unclaimed_pixels": claimed,
            "raw_supported_additions": added,
            "skipped_core_frames": skipped,
        })
        frame_rows.extend(local_rows)

    applications_frame = pd.DataFrame(
        applications, columns=APPLICATION_COLUMNS)
    frames_frame = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    old_union = (labels > 0) | (unclaimed > 0)
    new_union = (candidate > 0) | (candidate_unclaimed > 0)
    if np.any(old_union & ~new_union):
        raise AssertionError("isolated-lifetime recovery removed foreground")
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    if np.any((candidate > 0) & ~old_union & (raw <= 0)):
        raise AssertionError("isolated-lifetime recovery added zero-signal pixels")
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError("isolated-lifetime recovery changed accepted pixels")
    if _duplicate_components(candidate) > _duplicate_components(labels):
        raise AssertionError(
            "isolated-lifetime recovery created duplicate components")
    return (candidate, candidate_unclaimed, proposals,
            applications_frame, frames_frame, attached)


def summarize(labels: np.ndarray, unclaimed: np.ndarray,
              candidate: np.ndarray, candidate_unclaimed: np.ndarray,
              proposals: pd.DataFrame, applications: pd.DataFrame) -> dict:
    changed = candidate != labels
    ledger_changed = candidate_unclaimed != unclaimed
    active = set(map(int, np.unique(labels))) - {0}
    after = set(map(int, np.unique(candidate))) - {0}
    old_union = (labels > 0) | (unclaimed > 0)
    new_union = (candidate > 0) | (candidate_unclaimed > 0)
    return {
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "proposals_audited": int(len(proposals)),
        "eligible_proposals": int(proposals.discovery_status.eq(
            "eligible").sum()) if len(proposals) else 0,
        "applied_proposals": int(applications.outcome.eq(
            "applied").sum()) if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
        "relabelled_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0))),
        "raw_supported_additions": int(np.count_nonzero(
            (candidate > 0) & ~old_union)),
        "claimed_unclaimed_pixels": int(np.count_nonzero(
            (unclaimed > 0) & (candidate_unclaimed == 0))),
        "unclaimed_ledger_changed_pixels": int(np.count_nonzero(
            ledger_changed)),
        "foreground_removed_pixels": int(np.count_nonzero(
            (labels > 0) & (candidate == 0))),
        "old_identity_set_preserved": active <= after,
        "new_identity_count": int(len(after - active)),
        "foreground_ledger_union_preserved_or_expanded": bool(np.all(
            ~old_union | new_union)),
        "new_same_frame_identity_components": max(
            0, _duplicate_components(candidate)
            - _duplicate_components(labels)),
    }
