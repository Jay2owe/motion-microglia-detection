"""Backfill a late-born stable owner through its complete physical lineage.

Discovery is field-wide. Identity values are opaque labels inferred from the
physical-track owner history; this producer accepts no identity, track, frame,
coordinate, event, or review-case target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from atomic_two_seat_recovery import _raw_core
from ownerless_body_recovery import attach_coverage
from resident_takeover import attach_owners
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
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
    """Refuse selected biological cases under singular or plural key names."""
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and (name in FORBIDDEN_TARGETS or any(
            token in name.lower() for token in FORBIDDEN_TARGET_TOKENS)))
    if supplied:
        raise ValueError(
            "field-wide late-owner backfill received forbidden targets: "
            + ", ".join(supplied))


def _runs(group: pd.DataFrame) -> list[dict]:
    ordered = group.sort_values("frame")
    rows = list(ordered.itertuples(index=False))
    result: list[dict] = []
    start = 0
    while start < len(rows):
        owner = int(rows[start].candidate_owner)
        end = start
        while (end + 1 < len(rows)
               and int(rows[end + 1].candidate_owner) == owner
               and int(rows[end + 1].frame) == int(rows[end].frame) + 1):
            end += 1
        result.append({
            "owner": owner,
            "first_frame": int(rows[start].frame),
            "last_frame": int(rows[end].frame),
            "frames": end - start + 1,
        })
        start = end + 1
    return result


def discover_lineages(attached: pd.DataFrame, labels: np.ndarray,
                      frame_count: int, params: dict) -> pd.DataFrame:
    minimum_span = int(np.ceil(
        frame_count * float(params.get("minimum_span_fraction_of_movie", 0.5))))
    minimum_visible_fraction = float(
        params.get("minimum_visible_fraction_of_track_span", 0.7))
    minimum_ownerless_fraction = float(
        params.get("minimum_ownerless_visible_fraction", 0.5))
    minimum_stable_run = int(params.get("minimum_stable_owner_run_frames", 8))
    minimum_foreign = int(params.get("minimum_distinct_prior_foreign_owners", 2))
    maximum_terminal_gap = int(np.ceil(
        frame_count * float(params.get(
            "maximum_terminal_gap_fraction_of_movie", 0.1))))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.2))
    maximum_unclaimed = float(
        params.get("maximum_unclaimed_coverage_fraction", 0.0))
    active = sorted(set(map(int, np.unique(labels))) - {0})
    first_frame = {
        identity: int(np.flatnonzero(np.any(labels == identity, axis=(1, 2)))[0])
        for identity in active
    }
    rows: list[dict] = []
    for track_id, group in attached.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        first = int(group.frame.min())
        last = int(group.frame.max())
        span = last - first + 1
        visible = group[group.physically_visible.astype(bool)].copy()
        runs = [run for run in _runs(visible) if int(run["owner"]) > 0]
        stable = max(
            runs,
            key=lambda run: (int(run["frames"]), int(run["last_frame"]),
                             -int(run["owner"])),
            default={"owner": 0, "first_frame": -1,
                     "last_frame": -1, "frames": 0})
        stable_owner = int(stable["owner"])
        owners = visible.candidate_owner.astype(int)
        ownerless_fraction = float((owners == 0).mean()) if len(visible) else 0.0
        foreign = set(map(int, owners)) - {0, stable_owner}
        competing = attached[
            (attached.track_id != int(track_id))
            & (attached.frame >= first)
            & (attached.frame <= last)
            & (attached.candidate_owner == stable_owner)
        ] if stable_owner else attached.iloc[:0]
        reasons: list[str] = []
        expected_frames = np.arange(first, last + 1)
        if not np.array_equal(group.frame.to_numpy(int), expected_frames):
            reasons.append("physical_track_not_frame_continuous")
        if span < minimum_span:
            reasons.append("track_span_too_short")
        if len(visible) / max(span, 1) < minimum_visible_fraction:
            reasons.append("insufficient_visible_fraction")
        if ownerless_fraction < minimum_ownerless_fraction:
            reasons.append("not_mostly_ownerless")
        if int(stable["frames"]) < minimum_stable_run:
            reasons.append("stable_owner_run_too_short")
        if stable_owner <= 0 or first_frame.get(stable_owner, -1) < first:
            reasons.append("stable_owner_has_earlier_global_claim")
        if stable_owner > 0 and first_frame.get(stable_owner, -1) > last:
            reasons.append("stable_owner_outside_track_lifespan")
        if int(stable["last_frame"]) < last - maximum_terminal_gap:
            reasons.append("stable_owner_not_terminal")
        if len(foreign) < minimum_foreign:
            reasons.append("insufficient_prior_owner_contamination")
        if float(visible.in_encounter.astype(bool).mean()) > maximum_encounter:
            reasons.append("too_much_encounter_time")
        if float(group.unclaimed_coverage_fraction.max()) > maximum_unclaimed:
            reasons.append("overlaps_unclaimed_ledger")
        if len(competing):
            reasons.append("stable_owner_supported_on_competing_track")
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "track_id": int(track_id),
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "visible_frames": int(len(visible)),
            "visible_fraction": float(len(visible) / max(span, 1)),
            "ownerless_visible_fraction": ownerless_fraction,
            "stable_owner": stable_owner,
            "stable_owner_first_global_frame": first_frame.get(stable_owner, -1),
            "stable_run_first": int(stable["first_frame"]),
            "stable_run_last": int(stable["last_frame"]),
            "stable_run_frames": int(stable["frames"]),
            "distinct_prior_foreign_owners": int(len(foreign)),
            "foreign_owners": "|".join(map(str, sorted(foreign))),
            "encounter_fraction": float(
                visible.in_encounter.astype(bool).mean()) if len(visible) else 1.0,
            "maximum_unclaimed_coverage": float(
                group.unclaimed_coverage_fraction.max()),
            "competing_owner_track_points": int(len(competing)),
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows)


def _dim_raw_core(raw_frame: np.ndarray, point, params: dict,
                  ) -> tuple[np.ndarray, tuple[slice, slice], dict]:
    radius = int(np.ceil(np.clip(float(point.radius_px), 2.0, 6.0)))
    y = int(np.clip(round(float(point.y)), 0, raw_frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, raw_frame.shape[1] - 1))
    y0, y1 = max(0, y - radius), min(raw_frame.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(raw_frame.shape[1], x + radius + 1)
    local = raw_frame[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
            <= radius ** 2)
    positive = disk & (local > 0)
    if not np.any(positive):
        return (np.zeros_like(local, bool), (slice(y0, y1), slice(x0, x1)),
                {"core_area_px": 0, "seed_mode": "no_positive_raw_in_radius"})
    values = np.where(positive, local.astype(np.float32), -np.inf)
    seed_y, seed_x = np.unravel_index(int(np.argmax(values)), values.shape)
    peak = float(local[seed_y, seed_x])
    threshold = float(params.get("dim_core_peak_fraction", 0.1)) * peak
    components, _ = ndi.label(positive & (local >= threshold), STRUCTURE)
    component_id = int(components[seed_y, seed_x])
    core = components == component_id
    return core, (slice(y0, y1), slice(x0, x1)), {
        "core_area_px": int(np.count_nonzero(core)),
        "seed_mode": "local_raw_relative_peak",
        "raw_peak": peak,
        "raw_threshold": threshold,
    }


def _excess_components(labels: np.ndarray) -> int:
    return component_accounting.total_component_excess(labels)


def apply_lineage(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
                  attached: pd.DataFrame, thresholds: pd.DataFrame,
                  proposal, params: dict,
                  ) -> tuple[np.ndarray | None, pd.DataFrame, str]:
    candidate = labels.copy()
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.5))
    minimum_donor_pixels = int(params.get("minimum_donor_remaining_pixels", 3))
    minimum_donor_fraction = float(
        params.get("minimum_donor_remaining_fraction", 0.1))
    target = int(proposal.stable_owner)
    points = attached[attached.track_id == int(proposal.track_id)].sort_values("frame")
    audit: list[dict] = []
    for point in points.itertuples(index=False):
        frame = int(point.frame)
        before_frame = candidate[frame].copy()
        core, region, details = _raw_core(
            raw[frame], point,
            float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
        if not details["core_area_px"]:
            core, region, details = _dim_raw_core(raw[frame], point, params)
        expected = np.pi * max(float(point.radius_px), 1.0) ** 2
        ratio = details["core_area_px"] / max(expected, 1.0)
        if not details["core_area_px"] or ratio > maximum_ratio:
            return None, pd.DataFrame(audit), "raw_core_incomplete"
        if np.any(core & (unclaimed[frame][region] > 0)):
            return None, pd.DataFrame(audit), "raw_core_overlaps_unclaimed"
        if np.any(core & (raw[frame][region] == 0)):
            return None, pd.DataFrame(audit), "raw_core_contains_zero_signal"
        existing_target = candidate[frame] == target
        if np.any(existing_target):
            target_parts, _ = ndi.label(existing_target, STRUCTURE)
            touched = set(map(int, np.unique(target_parts[region][core]))) - {0}
            if not touched or len(touched) != 1:
                return None, pd.DataFrame(audit), "target_owner_on_competing_component"
            if set(map(int, np.unique(target_parts))) - {0} != touched:
                return None, pd.DataFrame(audit), "target_owner_has_multiple_components"
        old_values = candidate[frame][region][core].copy()
        foreign = set(map(int, np.unique(old_values))) - {0, target}
        donor_components: dict[int, tuple[np.ndarray, set[int], int]] = {}
        for donor in foreign:
            donor_parts, donor_part_count = ndi.label(
                candidate[frame] == donor, STRUCTURE)
            touched_parts = set(map(
                int, np.unique(donor_parts[region][core]))) - {0}
            donor_components[donor] = (
                donor_parts, touched_parts, int(donor_part_count))
            baseline_count = int(np.count_nonzero(candidate[frame] == donor))
            removed = int(np.count_nonzero(old_values == donor))
            remaining = baseline_count - removed
            if (remaining < minimum_donor_pixels
                    or remaining / max(baseline_count, 1) < minimum_donor_fraction):
                return None, pd.DataFrame(audit), "foreign_donor_would_be_erased"
        candidate[frame][region][core] = target

        # Cutting a raw-supported moving core out of a merged donor mask must not
        # leave donor shards.  Within each touched donor component, retain its
        # largest remaining connected body and transfer only the disconnected
        # shards created by the cut.  This is identity-blind and preserves both
        # foreground and donor topology.
        cleanup_pixels = 0
        for donor, (donor_parts, touched_parts, before_part_count) in \
                donor_components.items():
            for part_id in touched_parts:
                original_part = donor_parts == part_id
                residual_parts, residual_count = ndi.label(
                    original_part & (candidate[frame] == donor), STRUCTURE)
                if residual_count <= 1:
                    continue
                ranked = []
                for residual_id in range(1, residual_count + 1):
                    piece = residual_parts == residual_id
                    ranked.append((
                        int(np.count_nonzero(piece)),
                        float(raw[frame][piece].sum()),
                        -residual_id,
                        residual_id,
                    ))
                keep_id = max(ranked)[3]
                shards = (residual_parts > 0) & (residual_parts != keep_id)
                cleanup_pixels += int(np.count_nonzero(shards))
                candidate[frame][shards] = target
            after_count = ndi.label(candidate[frame] == donor, STRUCTURE)[1]
            if after_count > before_part_count:
                return None, pd.DataFrame(audit), "donor_topology_not_preserved"
            after_pixels = int(np.count_nonzero(candidate[frame] == donor))
            before_pixels = int(np.count_nonzero(before_frame == donor))
            if (after_pixels < minimum_donor_pixels
                    or after_pixels / max(before_pixels, 1)
                    < minimum_donor_fraction):
                return None, pd.DataFrame(audit), "foreign_donor_would_be_erased"
        before_target_parts = ndi.label(before_frame == target, STRUCTURE)[1]
        after_target_parts = ndi.label(candidate[frame] == target, STRUCTURE)[1]
        if after_target_parts > max(1, before_target_parts):
            return None, pd.DataFrame(audit), "target_topology_not_preserved"
        audit.append({
            "proposal_id": proposal.proposal_id,
            "frame": frame,
            "track_state": str(point.state),
            "stable_owner": target,
            "seed_mode": details.get("seed_mode", "track_point"),
            "core_area_px": int(details["core_area_px"]),
            "core_area_radius_ratio": float(ratio),
            "donor_shard_cleanup_pixels": int(cleanup_pixels),
            "prior_owners": "|".join(map(str, sorted(
                set(map(int, np.unique(old_values))) - {0}))),
            "changed_pixels": int(np.count_nonzero(
                candidate[frame] != before_frame)),
        })
    # A nearby larger cell can legitimately dominate the scorer's whole search
    # disk.  Verify the raw component we actually recovered rather than asking
    # that the target also dominate unrelated signal in that disk.
    for point in points.itertuples(index=False):
        frame = int(point.frame)
        core, region, details = _raw_core(
            raw[frame], point,
            float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
        if not details["core_area_px"]:
            core, region, details = _dim_raw_core(raw[frame], point, params)
        values = candidate[frame][region][core]
        if not len(values) or not np.all(values == target):
            return None, pd.DataFrame(audit), "recovered_core_verification_failed"
    return candidate, pd.DataFrame(audit), "applied"


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("late-owner backfill must use field-wide discovery")
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed and raw stacks must align")
    attached = attach_coverage(points, labels, unclaimed)
    attached["candidate_owner"] = attach_owners(
        points, labels).candidate_owner.astype(int)
    proposals = discover_lineages(attached, labels, len(labels), params)
    candidate = labels.copy()
    applications: list[dict] = []
    frame_rows: list[pd.DataFrame] = []
    eligible = proposals[proposals.discovery_status == "eligible"].sort_values(
        ["first_frame", "track_id", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        proposed, audit, reason = apply_lineage(
            candidate, unclaimed, raw, attached, thresholds, proposal, params)
        if proposed is None:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "stable_owner": int(proposal.stable_owner),
                "outcome": "rejected_application",
                "reason": reason,
                "changed_pixels": 0,
                "changed_frames": 0,
            })
            continue
        changed = proposed != candidate
        candidate = proposed
        applications.append({
            "proposal_id": proposal.proposal_id,
            "track_id": int(proposal.track_id),
            "stable_owner": int(proposal.stable_owner),
            "outcome": "applied",
            "reason": reason,
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        })
        frame_rows.append(audit)
    return (candidate, proposals, pd.DataFrame(applications),
            pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame())


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray,
              points: pd.DataFrame, proposals: pd.DataFrame,
              applications: pd.DataFrame, mode: str) -> dict:
    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    relabelled = changed & (labels > 0) & (candidate > 0)
    active = set(map(int, np.unique(labels))) - {0}
    candidate_active = set(map(int, np.unique(candidate))) - {0}
    named_frame_losses = 0
    for identity in active:
        before = np.any(labels == identity, axis=(1, 2))
        after = np.any(candidate == identity, axis=(1, 2))
        named_frame_losses += int(np.count_nonzero(before & ~after))
    return {
        "mode": mode,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "review_case_targets_received": False,
        "tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(proposals)),
        "eligible_lineages": int(proposals.discovery_status.eq("eligible").sum()),
        "applied_lineages": int(applications.outcome.eq("applied").sum())
            if len(applications) else 0,
        "rejected_application_lineages": int(
            applications.outcome.eq("rejected_application").sum())
            if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "raw_supported_additions": int(np.count_nonzero(additions)),
        "relabelled_foreground_pixels": int(np.count_nonzero(relabelled)),
        "removed_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "unclaimed_overlap_additions": int(np.count_nonzero(
            (candidate > 0) & (labels == 0) & (unclaimed > 0))),
        "old_identity_set_preserved": active == candidate_active,
        "donor_named_frame_losses": named_frame_losses,
        "new_same_frame_identity_components": max(
            0, _excess_components(candidate) - _excess_components(labels)),
    }
