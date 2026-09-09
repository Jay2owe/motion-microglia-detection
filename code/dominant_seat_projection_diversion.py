"""Recover dominant physical seats while accounting for a detached projection.

This field-wide rule extends the production bracketed-flash core with two
independent proofs: the displaced owner dominates one nearly complete physical
seat, and the component on which its label survives is small, weakly associated
with the owner over the movie, and subordinate in core radius to the restored
soma.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import bounded_owner_excursion as bounded
import bracketed_mixed_owner_flash as bracketed
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "dominant_owner", "source_owner",
    "middle_first", "middle_last", "middle_frames", "dominant_support",
    "positive_observations", "dominant_purity", "visible_track_fraction",
    "episode_strong_fraction", "maximum_episode_step_sum_radii",
    "donor_track", "donor_support", "donor_purity",
    "preexisting_owner_components", "projection_track",
    "projection_owner_support", "projection_component_pixels",
    "restored_soma_pixels", "projection_component_fraction",
    "projection_radius_fraction", "projection_proved", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "track_id", "frame", "dominant_owner", "source_owner",
    "donor_track", "partition_method", "changed_pixels",
    "explained_projection_components", "unexplained_duplicate_components",
    "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    bracketed.assert_target_free(params)


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _component_value(components: np.ndarray, mask: np.ndarray, point) -> int:
    marker = bounded._marker(mask, point)
    return int(components[marker]) if marker is not None else 0


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict):
    """Audit dominant-seat single-owner flashes across the complete field."""
    assert_target_free(params)
    base_params = dict(params)
    base_params.update({
        "maximum_middle_movie_fraction": float(
            params.get("maximum_middle_movie_fraction", 0.02)),
        "minimum_total_bookend_movie_fraction": float(
            params.get("minimum_total_bookend_movie_fraction", 0.20)),
        "minimum_long_flank_movie_fraction": float(
            params.get("minimum_long_flank_movie_fraction", 0.10)),
        "minimum_bracket_owner_purity": float(
            params.get("minimum_dominant_owner_purity", 0.95)),
        "minimum_distinct_middle_owners": 1,
        "minimum_episode_strong_fraction": float(
            params.get("minimum_episode_strong_fraction", 0.90)),
        "maximum_episode_step_sum_radii": float(
            params.get("maximum_episode_step_sum_radii", 0.50)),
        "minimum_shared_mask_fraction": 1.0,
    })
    base_audit, base_internal, attached = bracketed.discover(
        labels, points, base_params)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    minimum_support = int(math.ceil(movie * float(
        params.get("minimum_dominant_support_movie_fraction", 0.75))))
    minimum_visible = float(params.get("minimum_visible_track_fraction", 0.95))
    minimum_donor_support = int(math.ceil(movie * float(
        params.get("minimum_donor_support_movie_fraction", 0.75))))
    minimum_donor_purity = float(params.get("minimum_donor_owner_purity", 0.90))
    maximum_projection_support = int(math.ceil(movie * float(
        params.get("maximum_projection_owner_support_movie_fraction", 0.05))))
    maximum_projection_size = float(params.get(
        "maximum_projection_component_soma_fraction", 0.25))
    maximum_projection_radius = float(params.get(
        "maximum_projection_core_radius_soma_fraction", 0.60))
    rows: list[dict] = []
    internals: list[dict] = []
    eligible_base = base_internal[base_internal.eligible.astype(bool)] \
        if len(base_internal) else base_internal
    for proposal in eligible_base.itertuples(index=False):
        target_group = groups[int(proposal.track_id)]
        visible_fraction = float(
            target_group.physically_visible.astype(bool).sum() / movie)
        middle_frames = list(map(int, proposal.frame_values))
        source_owners = [int(_point(target_group, frame).candidate_owner)
                         for frame in middle_frames]
        source_owner = source_owners[0] if len(set(source_owners)) == 1 else 0
        reasons: list[str] = []
        if int(proposal.bracket_support) < minimum_support:
            reasons.append("insufficient_dominant_owner_support")
        if float(proposal.bracket_purity) < float(
                params.get("minimum_dominant_owner_purity", 0.95)):
            reasons.append("insufficient_dominant_owner_purity")
        if visible_fraction < minimum_visible:
            reasons.append("dominant_seat_not_movie_long")
        if source_owner <= 0:
            reasons.append("middle_owner_not_unique")

        donor_options = []
        projection_options = []
        frame_details = []
        for frame_index in middle_frames:
            target = _point(target_group, frame_index)
            source_component = bounded._component_at(
                labels[frame_index], source_owner, target) \
                if target is not None and source_owner > 0 else None
            target_marker = (bounded._marker(source_component, target)
                             if source_component is not None else None)
            frame_attached = attached[
                attached.frame.astype(int).eq(frame_index)
                & attached.physically_visible.astype(bool)]
            donor_rows = []
            if source_component is not None and target_marker is not None:
                for other in frame_attached.itertuples(index=False):
                    if int(other.track_id) == int(proposal.track_id):
                        continue
                    marker = bounded._marker(source_component, other)
                    if marker is not None and marker != target_marker:
                        donor_rows.append(other)
            stable_donors = []
            for donor in donor_rows:
                donor_group = groups[int(donor.track_id)]
                support = int((
                    donor_group.candidate_owner.astype(int) == source_owner).sum())
                positive = donor_group[
                    donor_group.physically_visible.astype(bool)
                    & (donor_group.candidate_owner.astype(int) > 0)]
                purity = float(support / len(positive)) if len(positive) else 0.0
                if support >= minimum_donor_support and purity >= minimum_donor_purity:
                    stable_donors.append((support, purity, int(donor.track_id)))
            if len(stable_donors) != 1:
                reasons.append("established_source_donor_not_unique")
                donor = (0, 0.0, 0)
            else:
                donor = stable_donors[0]
                donor_options.append(donor)

            trial, method, changed, _ = bracketed._prepare_frame(
                labels[frame_index], raw[frame_index], frame_attached, target,
                int(proposal.bracket_owner), params)
            if trial is None:
                reasons.append("shared_component_partition_unavailable")
                continue
            before_components, before_count = ndi.label(
                labels[frame_index] == int(proposal.bracket_owner), STRUCTURE)
            trial_components, _ = ndi.label(
                trial == int(proposal.bracket_owner), STRUCTURE)
            restored_value = _component_value(
                trial_components,
                trial == int(proposal.bracket_owner), target)
            restored_pixels = int((trial_components == restored_value).sum())
            existing_values = list(range(1, int(before_count) + 1))
            if len(existing_values) != 1:
                reasons.append("preexisting_owner_component_not_unique")
                existing_value = 0
            else:
                existing_value = existing_values[0]
            existing_mask = before_components == existing_value \
                if existing_value else np.zeros_like(labels[frame_index], bool)
            projection_pixels = int(existing_mask.sum())
            projection_cores = []
            for other in frame_attached.itertuples(index=False):
                if int(other.track_id) == int(proposal.track_id):
                    continue
                marker = bounded._marker(existing_mask, other)
                if marker is not None:
                    projection_cores.append(other)
            if len(projection_cores) != 1:
                reasons.append("projection_core_not_unique")
                projection = None
                projection_support = 0
                projection_radius_fraction = float("inf")
            else:
                projection = projection_cores[0]
                projection_group = groups[int(projection.track_id)]
                projection_support = int((
                    projection_group.candidate_owner.astype(int)
                    == int(proposal.bracket_owner)).sum())
                projection_radius_fraction = float(
                    projection.radius_px / max(float(target.radius_px), 1.0))
                if projection_support > maximum_projection_support:
                    reasons.append("projection_owner_support_too_long")
                if projection_radius_fraction > maximum_projection_radius:
                    reasons.append("projection_core_too_large")
            projection_fraction = float(
                projection_pixels / max(restored_pixels, 1))
            if projection_fraction > maximum_projection_size:
                reasons.append("projection_component_too_large")
            duplicate_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.bracket_owner))
                - component_accounting.component_excess(
                    labels[frame_index], int(proposal.bracket_owner)))
            if duplicate_delta != 1:
                reasons.append("exactly_one_projection_component_not_created")
            projection_proved = not any(
                reason.startswith("projection_")
                or reason.startswith("preexisting_")
                or reason.startswith("exactly_one_") for reason in reasons)
            projection_options.append({
                "track": int(projection.track_id) if projection is not None else 0,
                "support": projection_support,
                "pixels": projection_pixels,
                "restored_pixels": restored_pixels,
                "size_fraction": projection_fraction,
                "radius_fraction": projection_radius_fraction,
                "proved": projection_proved,
            })
            frame_details.append({
                "frame": frame_index, "trial": trial, "method": method,
                "changed": changed, "source_owner": source_owner,
                "donor_track": int(donor[2]),
            })
        if len({item[2] for item in donor_options}) > 1:
            reasons.append("source_donor_changes_inside_interval")
        if len({item["track"] for item in projection_options}) > 1:
            reasons.append("projection_track_changes_inside_interval")
        projection = projection_options[0] if projection_options else {
            "track": 0, "support": 0, "pixels": 0, "restored_pixels": 0,
            "size_fraction": float("inf"), "radius_fraction": float("inf"),
            "proved": False,
        }
        donor = donor_options[0] if donor_options else (0, 0.0, 0)
        public = {
            "proposal_id": "", "track_id": int(proposal.track_id),
            "dominant_owner": int(proposal.bracket_owner),
            "source_owner": source_owner,
            "middle_first": int(proposal.middle_first),
            "middle_last": int(proposal.middle_last),
            "middle_frames": int(proposal.middle_frames),
            "dominant_support": int(proposal.bracket_support),
            "positive_observations": int(proposal.positive_observations),
            "dominant_purity": float(proposal.bracket_purity),
            "visible_track_fraction": visible_fraction,
            "episode_strong_fraction": float(proposal.episode_strong_fraction),
            "maximum_episode_step_sum_radii": float(
                proposal.maximum_episode_step_sum_radii),
            "donor_track": int(donor[2]), "donor_support": int(donor[0]),
            "donor_purity": float(donor[1]),
            "preexisting_owner_components": 1 if projection_options else 0,
            "projection_track": int(projection["track"]),
            "projection_owner_support": int(projection["support"]),
            "projection_component_pixels": int(projection["pixels"]),
            "restored_soma_pixels": int(projection["restored_pixels"]),
            "projection_component_fraction": float(projection["size_fraction"]),
            "projection_radius_fraction": float(projection["radius_fraction"]),
            "projection_proved": bool(projection["proved"] and not reasons),
            "eligible": not reasons,
            "reason": ("eligible_dominant_seat_projection_diversion"
                       if not reasons else "|".join(dict.fromkeys(reasons))),
        }
        rows.append(public)
        internals.append({**public, "frame_details": frame_details})
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"DSP{number:04d}"
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internals), attached, base_audit)


def apply(labels: np.ndarray, internal: pd.DataFrame):
    """Apply only complete proposals whose projection proof remains valid."""
    candidate = labels.copy()
    rows = []
    explained = applied = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending = list(proposal.frame_details)
        failure = ""
        for detail in pending:
            trial = detail["trial"]
            delta = component_accounting.component_excess(
                trial, int(proposal.dominant_owner)) - \
                component_accounting.component_excess(
                    candidate[int(detail["frame"])], int(proposal.dominant_owner))
            if delta != 1 or not bool(proposal.projection_proved):
                failure = "projection_proof_failed_at_application"
                break
        if failure:
            rows.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "frame": int(proposal.middle_first),
                "dominant_owner": int(proposal.dominant_owner),
                "source_owner": int(proposal.source_owner),
                "donor_track": int(proposal.donor_track),
                "partition_method": "atomic", "changed_pixels": 0,
                "explained_projection_components": 0,
                "unexplained_duplicate_components": 0, "applied": False,
                "reason": "atomic_refusal:" + failure,
            })
            continue
        for detail in pending:
            frame_index = int(detail["frame"])
            candidate[frame_index] = detail["trial"]
            explained += 1
            rows.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id), "frame": frame_index,
                "dominant_owner": int(proposal.dominant_owner),
                "source_owner": int(proposal.source_owner),
                "donor_track": int(proposal.donor_track),
                "partition_method": detail["method"],
                "changed_pixels": int(detail["changed"]),
                "explained_projection_components": 1,
                "unexplained_duplicate_components": 0, "applied": True,
                "reason": "dominant_seat_restored_with_explained_projection",
            })
        applied += 1
    changed = candidate != labels
    return candidate, pd.DataFrame(rows, columns=APPLICATION_COLUMNS), {
        "applied_proposals": applied,
        "new_explained_projection_components": explained,
        "new_duplicate_components": 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0,
        "removed_identity_count": 0,
    }
