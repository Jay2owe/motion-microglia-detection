"""Give a complete short lifetime a fresh ID when its owner has a long donor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import concurrent_duplicate_invasion as invasion
import isolated_unowned_lifetime
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)


def assert_target_free(params: dict) -> None:
    invasion.assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("concurrent duplicate lifetime must be field-wide")


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int) == int(frame)]
    return rows.iloc[0] if len(rows) == 1 else None


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every track for one short complete lifetime and unique donor."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    frame_count = len(labels)
    minimum_span = int(params.get("minimum_lifetime_frames", 8))
    maximum_span = int(np.floor(frame_count * float(
        params.get("maximum_lifetime_fraction_of_movie", 0.15))))
    minimum_visible = float(params.get("minimum_visible_fraction", 0.90))
    minimum_observed = float(params.get("minimum_observed_fraction", 0.80))
    minimum_purity = float(params.get("minimum_owner_purity", 0.90))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.20))
    minimum_preexist = int(params.get("minimum_donor_preexist_frames", 5))
    minimum_outlive = int(params.get("minimum_donor_outlive_frames", 20))
    minimum_donor_span = int(np.ceil(frame_count * float(
        params.get("minimum_donor_span_fraction_of_movie", 0.50))))
    minimum_donor_support = float(
        params.get("minimum_donor_interval_support_fraction", 0.90))
    minimum_raw_separable = float(
        params.get("minimum_raw_separable_fraction", 0.80))
    minimum_separation = float(
        params.get("minimum_median_separation_sum_radii", 1.0))
    maximum_valley = float(params.get("maximum_raw_valley_ratio", 0.80))
    by_track = {int(track): group.sort_values("frame")
                for track, group in scored.groupby("track_id", sort=True)}
    rows: list[dict] = []
    for track, group in by_track.items():
        first, last = int(group.frame.min()), int(group.frame.max())
        span = last - first + 1
        visible = group[group.physically_visible.astype(bool)]
        positive = visible[visible.accepted_owner.astype(int) > 0]
        counts = positive.accepted_owner.astype(int).value_counts()
        owner = int(counts.index[0]) if len(counts) else 0
        owner_support = int(counts.iloc[0]) if len(counts) else 0
        purity = owner_support / max(len(visible), 1)
        predecessors, successors = isolated_unowned_lifetime._endpoint_compatibility(
            group, scored, params)
        donor_options: list[dict] = []
        if owner:
            for donor_track, donor in by_track.items():
                if donor_track == track:
                    continue
                donor_first, donor_last = (int(donor.frame.min()),
                                           int(donor.frame.max()))
                donor_span = donor_last - donor_first + 1
                if (first - donor_first < minimum_preexist
                        or donor_last - last < minimum_outlive
                        or donor_span < minimum_donor_span):
                    continue
                supports = separate = raw_separable = 0
                separations: list[float] = []
                valleys: list[float] = []
                valid = True
                for frame in range(first, last + 1):
                    target_row, donor_row = _point(group, frame), _point(donor, frame)
                    if target_row is None or donor_row is None:
                        valid = False
                        break
                    if (not bool(target_row.physically_visible)
                            or not bool(donor_row.physically_visible)
                            or int(target_row.accepted_owner) != owner
                            or int(donor_row.accepted_owner) != owner):
                        continue
                    supports += 1
                    target_component = invasion._component_near(
                        labels[frame], owner, target_row)
                    donor_component = invasion._component_near(
                        labels[frame], owner, donor_row)
                    if target_component is None or donor_component is None:
                        valid = False
                        break
                    separate += int(not np.any(target_component & donor_component))
                    distance = float(np.hypot(
                        float(target_row.x) - float(donor_row.x),
                        float(target_row.y) - float(donor_row.y)))
                    scale = max(float(target_row.radius_px)
                                + float(donor_row.radius_px), 1.0)
                    separations.append(distance / scale)
                    valley = physical._valley_ratio(
                        raw[frame], (float(target_row.x), float(target_row.y)),
                        (float(donor_row.x), float(donor_row.y)))
                    valleys.append(valley)
                    raw_separable += int(valley <= maximum_valley)
                support_fraction = supports / max(span, 1)
                separate_fraction = separate / max(supports, 1)
                raw_separable_fraction = raw_separable / max(supports, 1)
                median_separation = float(np.median(separations)) \
                    if separations else 0.0
                maximum_valley_seen = max(valleys) if valleys else np.inf
                if (valid and support_fraction >= minimum_donor_support
                        and raw_separable_fraction >= minimum_raw_separable
                        and median_separation >= minimum_separation
                        ):
                    donor_options.append({
                        "donor_track": donor_track,
                        "donor_first": donor_first, "donor_last": donor_last,
                        "donor_span_frames": donor_span,
                        "donor_interval_support_fraction": support_fraction,
                        "separate_component_fraction": separate_fraction,
                        "raw_separable_fraction": raw_separable_fraction,
                        "median_separation_sum_radii": median_separation,
                        "maximum_raw_valley_ratio": maximum_valley_seen,
                    })
        reasons: list[str] = []
        if not np.array_equal(group.frame.to_numpy(int),
                              np.arange(first, last + 1)):
            reasons.append("track_not_continuous")
        if span < minimum_span or span > maximum_span:
            reasons.append("lifetime_outside_duration_range")
        if len(visible) / max(span, 1) < minimum_visible:
            reasons.append("insufficient_visible_fraction")
        observed = int((visible.state == "observed").sum())
        if observed / max(span, 1) < minimum_observed:
            reasons.append("insufficient_observed_fraction")
        if owner <= 0 or purity < minimum_purity:
            reasons.append("owner_not_pure_over_lifetime")
        encounter_fraction = float(
            visible.in_encounter.astype(bool).mean()) if len(visible) else 1.0
        if encounter_fraction > maximum_encounter:
            reasons.append("too_many_encounter_frames")
        if first == 0 or last == frame_count - 1:
            reasons.append("lifetime_censored_by_movie_boundary")
        if predecessors or successors:
            reasons.append("compatible_endpoint_continuation_exists")
        if len(donor_options) != 1:
            reasons.append("long_concurrent_donor_not_unique")
        donor = donor_options[0] if len(donor_options) == 1 else {
            "donor_track": 0, "donor_first": -1, "donor_last": -1,
            "donor_span_frames": 0, "donor_interval_support_fraction": 0.0,
            "separate_component_fraction": 0.0,
            "raw_separable_fraction": 0.0,
            "median_separation_sum_radii": np.nan,
            "maximum_raw_valley_ratio": np.nan,
        }
        rows.append({
            "proposal_id": f"P{len(rows) + 1:04d}",
            "physical_track": track, "first_frame": first,
            "last_frame": last, "span_frames": span,
            "visible_frames": int(len(visible)), "observed_frames": observed,
            "shared_owner": owner, "owner_support_frames": owner_support,
            "owner_purity": float(purity),
            "encounter_fraction": encounter_fraction,
            "compatible_predecessors": predecessors,
            "compatible_successors": successors,
            **donor,
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows), scored


def _apply(labels: np.ndarray, raw: np.ndarray, scored: pd.DataFrame,
           proposal, identity: int, params: dict):
    candidate = labels.copy()
    target_group = scored[
        scored.track_id == int(proposal.physical_track)].sort_values("frame")
    donor_group = scored[
        scored.track_id == int(proposal.donor_track)].sort_values("frame")
    owner = int(proposal.shared_owner)
    sigma = float(params.get("watershed_sigma_px", 1.0))
    audit: list[dict] = []
    for frame in range(int(proposal.first_frame), int(proposal.last_frame) + 1):
        target, donor = _point(target_group, frame), _point(donor_group, frame)
        if target is None or donor is None:
            return None, pd.DataFrame(audit), "physical_point_missing"
        before = candidate[frame].copy()
        target_component = invasion._component_near(
            candidate[frame], owner, target)
        donor_component = invasion._component_near(
            candidate[frame], owner, donor)
        if target_component is None or donor_component is None:
            return None, pd.DataFrame(audit), "shared_owner_component_missing"
        frame_points = scored[(scored.frame == frame)
                              & scored.physically_visible.astype(bool)]
        component_tracks = set()
        for other in frame_points.itertuples(index=False):
            other_component = invasion._component_near(
                candidate[frame], owner, other)
            if (other_component is not None
                    and np.any(target_component & other_component)):
                component_tracks.add(int(other.track_id))
        shared = bool(np.any(target_component & donor_component))
        expected = ({int(proposal.physical_track), int(proposal.donor_track)}
                    if shared else {int(proposal.physical_track)})
        if component_tracks != expected:
            return None, pd.DataFrame(audit), "target_component_core_set_mismatch"
        if not shared:
            target_partition = target_component
            donor_shards = np.zeros_like(target_component, bool)
            method = "relabel_separate_duplicate_component"
        else:
            target_marker = physical._marker_pixel(
                target_component, float(target.x), float(target.y))
            donor_marker = physical._marker_pixel(
                target_component, float(donor.x), float(donor.y))
            if (target_marker is None or donor_marker is None
                    or target_marker == donor_marker):
                return None, pd.DataFrame(audit), "watershed_markers_unavailable"
            markers = np.zeros(target_component.shape, np.int16)
            markers[target_marker], markers[donor_marker] = 1, 2
            elevation = -ndi.gaussian_filter(raw[frame].astype(np.float32), sigma)
            target_partition = watershed(
                elevation, markers=markers, mask=target_component) == 1
            support_radius = float(np.clip(
                0.75 * float(target.radius_px), 2.0, 6.0))
            yy, xx = np.ogrid[:target_component.shape[0],
                              :target_component.shape[1]]
            target_partition |= target_component & (
                (xx - float(target.x)) ** 2
                + (yy - float(target.y)) ** 2 <= support_radius ** 2)
            method = "raw_seeded_shared_component_partition"
        candidate[frame][target_partition] = identity
        if shared:
            residual_parts, _ = ndi.label(
                target_component & (candidate[frame] == owner), STRUCTURE)
            donor_marker = physical._marker_pixel(
                residual_parts > 0, float(donor.x), float(donor.y))
            if donor_marker is None:
                return None, pd.DataFrame(audit), "donor_partition_missing"
            donor_part = int(residual_parts[donor_marker])
            if donor_part <= 0:
                return None, pd.DataFrame(audit), "donor_partition_missing"
            donor_shards = ((residual_parts > 0)
                            & (residual_parts != donor_part))
            candidate[frame][donor_shards] = identity
        if ndi.label(candidate[frame] == identity, STRUCTURE)[1] != 1:
            return None, pd.DataFrame(audit), "fresh_identity_not_one_component"
        if (ndi.label(candidate[frame] == owner, STRUCTURE)[1]
                > ndi.label(before == owner, STRUCTURE)[1]):
            return None, pd.DataFrame(audit), "new_donor_owner_component"
        if int(physical._disk_owner(
                candidate[frame], float(target.x), float(target.y),
                float(target.radius_px))) != identity:
            return None, pd.DataFrame(audit), "fresh_owner_not_dominant_at_core"
        if int(physical._disk_owner(
                candidate[frame], float(donor.x), float(donor.y),
                float(donor.radius_px))) != owner:
            return None, pd.DataFrame(audit), "donor_owner_not_preserved"
        audit.append({
            "proposal_id": proposal.proposal_id, "frame": frame,
            "assigned_identity": identity, "shared_owner": owner,
            "operation": method,
            "donor_shards_reassigned": int(np.count_nonzero(donor_shards)),
            "changed_pixels": int(np.count_nonzero(candidate[frame] != before)),
        })
    return candidate, pd.DataFrame(audit), "applied"


def recover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            params: dict):
    proposals, scored = discover(labels, raw, points, params)
    candidate = labels.copy()
    next_identity = int(labels.max()) + 1
    applications: list[dict] = []
    audits: list[pd.DataFrame] = []
    occupied = np.zeros_like(labels, bool)
    for proposal in proposals[proposals.discovery_status == "eligible"].sort_values(
            ["first_frame", "physical_track"]).itertuples(index=False):
        proposed, audit, reason = _apply(
            candidate, raw, scored, proposal, next_identity, params)
        if proposed is None:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "physical_track": int(proposal.physical_track),
                "donor_track": int(proposal.donor_track),
                "assigned_identity": 0, "outcome": "rejected_application",
                "reason": reason, "changed_pixels": 0, "changed_frames": 0})
            continue
        changed = proposed != candidate
        if np.any(changed & occupied):
            continue
        candidate, occupied = proposed, occupied | changed
        applications.append({
            "proposal_id": proposal.proposal_id,
            "physical_track": int(proposal.physical_track),
            "donor_track": int(proposal.donor_track),
            "assigned_identity": next_identity, "outcome": "applied",
            "reason": reason, "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2))))})
        audits.append(audit)
        next_identity += 1
    columns = ["proposal_id", "physical_track", "donor_track",
               "assigned_identity", "outcome", "reason", "changed_pixels",
               "changed_frames"]
    return (candidate, proposals, pd.DataFrame(applications, columns=columns),
            pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
            scored)


def summarize(labels: np.ndarray, candidate: np.ndarray, unclaimed: np.ndarray,
              raw: np.ndarray, points: pd.DataFrame, proposals: pd.DataFrame,
              applications: pd.DataFrame, mode: str) -> dict:
    metrics = invasion.summarize(
        labels, candidate, unclaimed, raw, points,
        proposals, applications.rename(columns={
            "physical_track": "resident_track"}), mode)
    metrics["eligible_duplicate_lifetimes"] = int(
        proposals.discovery_status.eq("eligible").sum())
    metrics["applied_duplicate_lifetimes"] = int(
        applications.outcome.eq("applied").sum()) if len(applications) else 0
    metrics["new_identities"] = sorted(
        set(map(int, np.unique(candidate))) - set(map(int, np.unique(labels))) - {0})
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    metrics["old_identity_set_preserved"] = before_ids <= after_ids
    return metrics

