"""Preserve a short-lived companion owner at a terminal assimilation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import concurrent_duplicate_invasion as invasion
import separable_merge_recovery as physical
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)


def assert_target_free(params: dict) -> None:
    invasion.assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal assimilation recovery must be field-wide")


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _component_centroid(mask: np.ndarray) -> tuple[float, float]:
    y, x = ndi.center_of_mass(mask)
    return float(x), float(y)


def _residual_marker(raw_frame: np.ndarray, component: np.ndarray,
                     target, params: dict):
    sigma = float(params.get("raw_sigma_px", 1.0))
    smooth = ndi.gaussian_filter(raw_frame.astype(np.float32), sigma)
    yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
    scale = max(float(target.radius_px), 1.0)
    allowed = component & (
        (xx - float(target.x)) ** 2 + (yy - float(target.y)) ** 2
        >= (float(params.get("minimum_residual_distance_radii", 1.5))
            * scale) ** 2)
    peaks = allowed & (smooth == ndi.maximum_filter(smooth, size=3)) \
        & (raw_frame > 0)
    if not np.any(peaks):
        return None, 0.0, 0.0
    scores = np.where(peaks, smooth, -np.inf)
    marker = np.unravel_index(int(np.argmax(scores)), scores.shape)
    peak = float(np.max(smooth[component]))
    residual_fraction = float(smooth[marker] / max(peak, 1.0))
    target_y, target_x = int(round(float(target.y))), int(round(float(target.x)))
    target_fraction = float(
        smooth[target_y, target_x] / max(peak, 1.0))
    if residual_fraction < float(params.get("minimum_residual_peak_fraction", 0.25)):
        return None, residual_fraction, target_fraction
    if target_fraction < float(params.get("minimum_target_peak_fraction", 0.5)):
        return None, residual_fraction, target_fraction
    return (int(marker[0]), int(marker[1])), residual_fraction, target_fraction


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    minimum_prior = int(params.get("minimum_prior_owner_frames", 3))
    minimum_purity = float(params.get("minimum_prior_owner_purity", 0.8))
    maximum_foreign = int(params.get("maximum_terminal_foreign_frames", 2))
    maximum_lookback = int(params.get("maximum_prior_lookback_frames", 10))
    maximum_step = float(params.get(
        "maximum_previous_owner_component_step_radii", 4.0))
    rows: list[dict] = []
    proposal = 0
    for track, group in scored.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        visible = group[group.physically_visible.astype(bool)]
        if len(visible) == 0:
            continue
        last = int(group.frame.max())
        last_row = _point(group, last)
        if (last_row is None or not bool(last_row.physically_visible)
                or int(last_row.accepted_owner) <= 0):
            continue
        foreign_owner = int(last_row.accepted_owner)
        foreign_frames = [last]
        cursor = last - 1
        while cursor >= int(group.frame.min()):
            row = _point(group, cursor)
            if (row is None or not bool(row.physically_visible)
                    or int(row.accepted_owner) != foreign_owner):
                break
            foreign_frames.append(cursor)
            cursor -= 1
        if len(foreign_frames) > maximum_foreign:
            continue
        first = min(foreign_frames)
        prior = visible[
            (visible.frame.astype(int) < first)
            & (visible.frame.astype(int) >= first - maximum_lookback)
            & (visible.accepted_owner.astype(int) > 0)]
        if len(prior) == 0:
            continue
        counts = prior.accepted_owner.astype(int).value_counts()
        prior_owner = int(counts.index[0])
        prior_frames = int(counts.iloc[0])
        purity = float(prior_frames / len(prior))
        component = invasion._component_near(
            labels[first], foreign_owner, last_row)
        previous_components, count = ndi.label(
            labels[first - 1] == foreign_owner, STRUCTURE)
        previous = None
        best_step = float("inf")
        if component is not None and first > 0:
            current_x, current_y = _component_centroid(component)
            for part in range(1, count + 1):
                mask = previous_components == part
                px, py = _component_centroid(mask)
                step = float(np.hypot(current_x - px, current_y - py)) / max(
                    float(last_row.radius_px), 1.0)
                if step < best_step:
                    previous, best_step = mask, step
        marker, residual_fraction, target_fraction = (None, 0.0, 0.0)
        if component is not None:
            marker, residual_fraction, target_fraction = _residual_marker(
                raw[first], component, last_row, params)
        complete_raw_frames = 0
        for frame in sorted(foreign_frames):
            row = _point(group, frame)
            frame_component = invasion._component_near(
                labels[frame], foreign_owner, row)
            if frame_component is None:
                continue
            frame_residual, _, _ = _residual_marker(
                raw[frame], frame_component, row, params)
            frame_target = physical._marker_pixel(
                frame_component, float(row.x), float(row.y))
            if (frame_residual is not None and frame_target is not None
                    and frame_residual != frame_target):
                complete_raw_frames += 1
        reasons: list[str] = []
        if prior_owner == foreign_owner:
            reasons.append("no_owner_change")
        if prior_frames < minimum_prior or purity < minimum_purity:
            reasons.append("insufficient_prior_owner_tenure")
        if component is None:
            reasons.append("foreign_owner_component_missing")
        if previous is None or best_step > maximum_step:
            reasons.append("foreign_owner_has_no_nearby_previous_seat")
        if marker is None:
            reasons.append("separate_residual_raw_core_missing")
        if complete_raw_frames != len(foreign_frames):
            reasons.append("separate_raw_cores_not_complete")
        if reasons:
            continue
        proposal += 1
        rows.append({
            "proposal_id": f"P{proposal:04d}", "physical_track": int(track),
            "prior_owner": prior_owner, "foreign_owner": foreign_owner,
            "first_frame": first, "last_frame": last,
            "prior_owner_frames": prior_frames,
            "prior_owner_purity": purity,
            "foreign_owner_frames": len(foreign_frames),
            "previous_component_step_radii": best_step,
            "residual_peak_fraction": residual_fraction,
            "target_peak_fraction": target_fraction,
            "discovery_status": "eligible", "discovery_reason": "eligible"})
    columns = ["proposal_id", "physical_track", "prior_owner", "foreign_owner",
               "first_frame", "last_frame", "prior_owner_frames",
               "prior_owner_purity", "foreign_owner_frames",
               "previous_component_step_radii", "residual_peak_fraction",
               "target_peak_fraction", "discovery_status", "discovery_reason"]
    return pd.DataFrame(rows, columns=columns), scored


def _excess_components(stack: np.ndarray) -> int:
    return component_accounting.total_component_excess(stack)


def recover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            params: dict):
    proposals, scored = discover(labels, raw, points, params)
    groups = {int(track): group.sort_values("frame")
              for track, group in scored.groupby("track_id", sort=True)}
    candidate = labels.copy()
    applications: list[dict] = []
    frames: list[dict] = []
    for proposal in proposals.itertuples(index=False):
        trial = candidate.copy()
        reason = ""
        local: list[dict] = []
        for frame in range(int(proposal.first_frame), int(proposal.last_frame) + 1):
            row = _point(groups[int(proposal.physical_track)], frame)
            component = invasion._component_near(
                trial[frame], int(proposal.foreign_owner), row)
            if component is None:
                reason = "foreign_owner_component_missing"
                break
            residual, residual_fraction, target_fraction = _residual_marker(
                raw[frame], component, row, params)
            target = physical._marker_pixel(
                component, float(row.x), float(row.y))
            if residual is None or target is None or residual == target:
                reason = "two_raw_markers_unavailable"
                break
            markers = np.zeros(component.shape, np.int16)
            markers[target] = 1
            markers[residual] = 2
            elevation = -ndi.gaussian_filter(
                raw[frame].astype(np.float32),
                float(params.get("raw_sigma_px", 1.0)))
            basins = watershed(elevation, markers=markers, mask=component,
                               connectivity=STRUCTURE)
            before = trial[frame].copy()
            trial[frame][basins == 1] = int(proposal.prior_owner)
            trial[frame][basins == 2] = int(proposal.foreign_owner)
            if not np.any(trial[frame] == int(proposal.foreign_owner)):
                reason = "foreign_owner_residual_extinguished"
                break
            if int(physical._disk_owner(
                    trial[frame], float(row.x), float(row.y),
                    float(row.radius_px))) != int(proposal.prior_owner):
                reason = "prior_owner_not_dominant_at_companion_core"
                break
            local.append({
                "proposal_id": str(proposal.proposal_id), "frame": frame,
                "physical_track": int(proposal.physical_track),
                "prior_owner": int(proposal.prior_owner),
                "foreign_owner": int(proposal.foreign_owner),
                "residual_peak_fraction": residual_fraction,
                "target_peak_fraction": target_fraction,
                "changed_pixels": int(np.count_nonzero(trial[frame] != before))})
        changed = trial != candidate
        if not reason and _excess_components(trial) > _excess_components(candidate):
            reason = "new_duplicate_identity_component"
        if reason:
            applications.append({
                "proposal_id": str(proposal.proposal_id),
                "physical_track": int(proposal.physical_track),
                "outcome": "rejected_application", "reason": reason,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        candidate = trial
        frames.extend(local)
        applications.append({
            "proposal_id": str(proposal.proposal_id),
            "physical_track": int(proposal.physical_track), "outcome": "applied",
            "reason": "terminal_companion_owner_preserved",
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2))))})
    columns = ["proposal_id", "physical_track", "outcome", "reason",
               "changed_pixels", "changed_frames"]
    return (candidate, proposals, pd.DataFrame(applications, columns=columns),
            pd.DataFrame(frames), scored)


def summarize(labels: np.ndarray, candidate: np.ndarray, unclaimed: np.ndarray,
              raw: np.ndarray, points: pd.DataFrame, proposals: pd.DataFrame,
              applications: pd.DataFrame, mode: str) -> dict:
    changed = candidate != labels
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    named_losses = 0
    for identity in before_ids:
        before = np.any(labels == identity, axis=(1, 2))
        after = np.any(candidate == identity, axis=(1, 2))
        named_losses += int(np.count_nonzero(before & ~after))
    return {
        "mode": mode, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "tracks_audited": int(points.track_id.nunique()),
        "terminal_assimilation_proposals": int(len(proposals)),
        "applied_terminal_assimilations": int(applications.outcome.eq(
            "applied").sum()),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=(1, 2)))),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "unclaimed_ledger_changed_pixels": 0,
        "zero_signal_additions": int(np.count_nonzero(
            changed & (labels == 0) & (candidate > 0) & (raw <= 0))),
        "old_identity_set_preserved": before_ids == after_ids,
        "donor_named_frame_losses": named_losses,
        "new_same_frame_identity_components": max(
            0, _excess_components(candidate) - _excess_components(labels)),
    }
