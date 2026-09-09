"""Complete raw cores only for field-proved terminal projection chains.

Discovery accepts mathematical thresholds and complete arrays/tables only. A
candidate child must be ownerless, short, right-censored, continuously visible,
outward from a unique stable soma through a persistent same-owner projection,
and uniquely closest to that projection. No biological target can be supplied.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import bracketed_mixed_owner_flash as flash
import component_accounting
import ownerless_cohort_latent_gap_completion as gap_completion
from ownerless_body_recovery import evidence_image
from resident_takeover import attach_owners


AUDIT_COLUMNS = [
    "proposal_id", "child_track", "parent_track", "anchor_track",
    "anchor_owner", "child_first", "child_last", "child_frames",
    "child_visible_fraction", "child_strong_fraction",
    "parent_owner_purity", "parent_support_fraction",
    "parent_anchor_radius_ratio", "maximum_parent_anchor_distance_radii",
    "maximum_child_parent_distance_radii", "minimum_outward_cosine",
    "minimum_radial_extension_radii", "maximum_child_step_radii",
    "parent_lapse_fraction", "parent_distance_margin", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "child_track", "parent_track", "anchor_track",
    "anchor_owner", "frame", "changed_pixels", "core_offset_radii",
    "core_area_radius_ratio", "explained_projection_components", "applied",
    "reason",
]


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _distance(left, right) -> float:
    return float(np.hypot(float(left.x) - float(right.x),
                          float(left.y) - float(right.y)))


def _scaled_distance(left, right) -> float:
    return _distance(left, right) / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def _visible(group: pd.DataFrame) -> pd.DataFrame:
    return group[group.physically_visible.astype(bool)].sort_values("frame")


def _positive_owner_summary(group: pd.DataFrame) -> tuple[int, int, float]:
    positive = _visible(group)
    positive = positive[positive.candidate_owner.astype(int).gt(0)]
    if positive.empty:
        return 0, 0, 0.0
    counts = positive.candidate_owner.astype(int).value_counts()
    owner, support = int(counts.index[0]), int(counts.iloc[0])
    return owner, support, float(support / len(positive))


def _stable_anchors(groups: dict[int, pd.DataFrame], movie: int,
                    params: dict) -> dict[int, list[int]]:
    minimum_support = int(math.ceil(movie * float(params.get(
        "minimum_anchor_support_movie_fraction", 0.75))))
    minimum_purity = float(params.get("minimum_anchor_owner_purity", 0.98))
    anchors: dict[int, list[int]] = {}
    for track, group in groups.items():
        owner, support, purity = _positive_owner_summary(group)
        if owner > 0 and support >= minimum_support and purity >= minimum_purity:
            anchors.setdefault(owner, []).append(int(track))
    return anchors


def _parent_projections(groups: dict[int, pd.DataFrame], anchors: dict[int, list[int]],
                        movie: int, params: dict) -> list[dict]:
    minimum_support = int(math.ceil(movie * float(params.get(
        "minimum_parent_support_movie_fraction", 0.15))))
    minimum_purity = float(params.get("minimum_parent_owner_purity", 0.90))
    minimum_presence = float(params.get(
        "minimum_parent_anchor_presence_fraction", 0.90))
    maximum_radius_ratio = float(params.get(
        "maximum_parent_anchor_radius_ratio", 0.65))
    maximum_distance = float(params.get(
        "maximum_parent_anchor_distance_sum_radii", 2.75))
    parents = []
    for track, group in groups.items():
        owner, support, purity = _positive_owner_summary(group)
        if owner <= 0 or len(anchors.get(owner, [])) != 1:
            continue
        anchor_track = int(anchors[owner][0])
        if int(track) == anchor_track or support < minimum_support \
                or purity < minimum_purity:
            continue
        anchor_group = groups[anchor_track]
        visible = _visible(group)
        distances = []
        radius_ratios = []
        present = 0
        for point in visible.itertuples(index=False):
            anchor = _point(anchor_group, int(point.frame))
            if (anchor is None or not bool(anchor.physically_visible)
                    or int(anchor.candidate_owner) != owner):
                continue
            present += 1
            distances.append(_scaled_distance(point, anchor))
            radius_ratios.append(float(point.radius_px) / max(
                float(anchor.radius_px), 1.0))
        presence = float(present / len(visible)) if len(visible) else 0.0
        radius_ratio = float(np.median(radius_ratios)) \
            if radius_ratios else float("inf")
        distance = max(distances, default=float("inf"))
        if (presence >= minimum_presence and radius_ratio <= maximum_radius_ratio
                and distance <= maximum_distance):
            parents.append({
                "track": int(track), "owner": owner,
                "anchor": anchor_track, "purity": purity,
                "support_fraction": float(support / movie),
                "radius_ratio": radius_ratio,
                "anchor_distance": distance,
            })
    return parents


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict):
    flash.assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    anchors = _stable_anchors(groups, movie, params)
    parents = _parent_projections(groups, anchors, movie, params)
    maximum_span = int(math.ceil(movie * float(params.get(
        "maximum_child_span_movie_fraction", 0.08))))
    maximum_right_censor = int(math.ceil(movie * float(params.get(
        "maximum_right_censor_movie_fraction", 0.04))))
    minimum_frames = int(params.get("minimum_child_frames", 3))
    minimum_visible = float(params.get("minimum_child_visible_fraction", 0.80))
    minimum_strong = float(params.get("minimum_child_strong_fraction", 0.60))
    maximum_radius_ratio = float(params.get(
        "maximum_child_parent_radius_ratio", 1.20))
    maximum_distance = float(params.get(
        "maximum_child_parent_distance_sum_radii", 2.75))
    minimum_cosine = float(params.get("minimum_outward_cosine", 0.60))
    minimum_extension = float(params.get(
        "minimum_radial_extension_parent_radii", 0.50))
    maximum_step = float(params.get("maximum_child_step_sum_radii", 0.75))
    maximum_parent_lapse = float(params.get(
        "maximum_parent_lapse_fraction", 0.20))
    minimum_margin = float(params.get("minimum_parent_distance_margin", 1.10))
    public_rows = []
    internal_rows = []
    for child_track, child_group in groups.items():
        frames = child_group.frame.astype(int)
        visible = _visible(child_group)
        reasons = []
        if len(child_group) < minimum_frames:
            reasons.append("child_too_short_to_prove")
        if len(child_group) > maximum_span:
            reasons.append("child_not_short_terminal_body")
        if movie - 1 - int(frames.max()) > maximum_right_censor:
            reasons.append("child_not_right_censored")
        visible_fraction = float(len(visible) / len(child_group)) \
            if len(child_group) else 0.0
        strong_fraction = float(visible.strong.astype(bool).mean()) \
            if len(visible) else 0.0
        if visible_fraction < minimum_visible:
            reasons.append("child_visibility_too_low")
        if strong_fraction < minimum_strong:
            reasons.append("child_raw_support_too_weak")
        if len(visible) and np.any(visible.candidate_owner.astype(int) > 0):
            reasons.append("child_not_ownerless")

        matches = []
        for parent in parents:
            parent_group = groups[parent["track"]]
            anchor_group = groups[parent["anchor"]]
            distances = []
            cosines = []
            extensions = []
            radius_ratios = []
            parent_lapses = 0
            complete = True
            for child in visible.itertuples(index=False):
                parent_point = _point(parent_group, int(child.frame))
                anchor_point = _point(anchor_group, int(child.frame))
                if (parent_point is None or anchor_point is None
                        or not bool(anchor_point.physically_visible)
                        or int(anchor_point.candidate_owner) != parent["owner"]):
                    complete = False
                    break
                if not bool(parent_point.physically_visible):
                    parent_lapses += 1
                    if bool(child.strong):
                        complete = False
                        break
                elif int(parent_point.candidate_owner) != parent["owner"]:
                    complete = False
                    break
                ap = np.array([float(parent_point.x) - float(anchor_point.x),
                               float(parent_point.y) - float(anchor_point.y)])
                pc = np.array([float(child.x) - float(parent_point.x),
                               float(child.y) - float(parent_point.y)])
                denominator = float(np.linalg.norm(ap) * np.linalg.norm(pc))
                cosines.append(float(np.dot(ap, pc) / denominator)
                               if denominator else -1.0)
                radial_extension = (
                    _distance(child, anchor_point)
                    - _distance(parent_point, anchor_point)) / max(
                        float(parent_point.radius_px), 1.0)
                extensions.append(radial_extension)
                distances.append(_scaled_distance(child, parent_point))
                radius_ratios.append(float(child.radius_px) / max(
                    float(parent_point.radius_px), 1.0))
            if not complete or not distances:
                continue
            matches.append({
                **parent,
                "distance": max(distances),
                "median_distance": float(np.median(distances)),
                "cosine": min(cosines), "extension": min(extensions),
                "child_parent_radius_ratio": float(np.median(radius_ratios)),
                "parent_lapse_fraction": float(parent_lapses / len(visible)),
            })
        matches.sort(key=lambda item: item["median_distance"])
        match = matches[0] if matches else None
        distance_margin = (matches[1]["median_distance"]
                           / max(matches[0]["median_distance"], 1e-9)) \
            if len(matches) > 1 else float("inf")
        if match is None:
            reasons.append("no_persistent_projection_parent")
        else:
            if match["child_parent_radius_ratio"] > maximum_radius_ratio:
                reasons.append("child_not_projection_sized")
            if match["distance"] > maximum_distance:
                reasons.append("child_too_far_from_parent_projection")
            if match["cosine"] < minimum_cosine:
                reasons.append("child_not_outward_from_soma_chain")
            if match["extension"] < minimum_extension:
                reasons.append("child_not_radially_beyond_parent")
            if match["parent_lapse_fraction"] > maximum_parent_lapse:
                reasons.append("parent_projection_lapse_too_long")
            if distance_margin < minimum_margin:
                reasons.append("projection_parent_not_unique")
        steps = [_scaled_distance(left, right)
                 for left, right in zip(visible.itertuples(index=False),
                                        visible.iloc[1:].itertuples(index=False))
                 if int(right.frame) == int(left.frame) + 1]
        maximum_seen_step = max(steps, default=0.0)
        if maximum_seen_step > maximum_step:
            reasons.append("child_motion_not_continuous")
        row = {
            "proposal_id": "", "child_track": int(child_track),
            "parent_track": int(match["track"]) if match else 0,
            "anchor_track": int(match["anchor"]) if match else 0,
            "anchor_owner": int(match["owner"]) if match else 0,
            "child_first": int(frames.min()), "child_last": int(frames.max()),
            "child_frames": int(len(child_group)),
            "child_visible_fraction": visible_fraction,
            "child_strong_fraction": strong_fraction,
            "parent_owner_purity": float(match["purity"]) if match else 0.0,
            "parent_support_fraction": float(match["support_fraction"])
                if match else 0.0,
            "parent_anchor_radius_ratio": float(match["radius_ratio"])
                if match else 0.0,
            "maximum_parent_anchor_distance_radii": float(
                match["anchor_distance"]) if match else 0.0,
            "maximum_child_parent_distance_radii": float(match["distance"])
                if match else 0.0,
            "minimum_outward_cosine": float(match["cosine"])
                if match else -1.0,
            "minimum_radial_extension_radii": float(match["extension"])
                if match else 0.0,
            "maximum_child_step_radii": maximum_seen_step,
            "parent_lapse_fraction": float(match["parent_lapse_fraction"])
                if match else 0.0,
            "parent_distance_margin": distance_margin,
            "eligible": not reasons,
            "reason": "eligible_terminal_projection_chain" if not reasons
                else "|".join(dict.fromkeys(reasons)),
        }
        public_rows.append(row)
        internal_rows.append({**row, "child_visible_frames":
                              visible.frame.astype(int).tolist()})
    number = 0
    for public, internal in zip(public_rows, internal_rows):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"TPC{number:04d}"
    return (pd.DataFrame(public_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internal_rows), attached, len(parents))


def apply(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
          thresholds: pd.DataFrame, attached: pd.DataFrame,
          internal: pd.DataFrame, params: dict):
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    threshold_by_frame = thresholds.set_index("frame")
    candidate = labels.copy()
    rows = []
    explained = 0
    eligible = internal[internal.eligible.astype(bool)] if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        child_group = groups[int(proposal.child_track)]
        pending = []
        failure = ""
        for frame in list(proposal.child_visible_frames):
            child = _point(child_group, int(frame))
            found = gap_completion._nearest_raw_core(
                raw[int(frame)], evidence_image(raw[int(frame)], params), child,
                float(threshold_by_frame.loc[int(frame), "weak_threshold"]),
                params)
            if found is None:
                failure = "qualified_raw_core_missing"
                break
            core, region, details = found
            occupied = ((candidate[int(frame)][region] > 0)
                        | (unclaimed[int(frame)][region] > 0))
            if np.any(core & occupied):
                failure = "raw_core_overlaps_existing_ledger"
                break
            trial = candidate[int(frame)].copy()
            before = component_accounting.component_excess(
                trial, int(proposal.anchor_owner))
            trial[region][core] = int(proposal.anchor_owner)
            after = component_accounting.component_excess(
                trial, int(proposal.anchor_owner))
            delta = max(0, int(after - before))
            if delta not in (0, 1):
                failure = "more_than_one_projection_component_created"
                break
            pending.append({
                "frame": int(frame), "trial": trial,
                "pixels": int(core.sum()),
                "offset": float(details["core_offset_radii"]),
                "area_ratio": float(details["core_area_radius_ratio"]),
                "explained": delta,
            })
        if failure or len(pending) != len(proposal.child_visible_frames):
            rows.append({
                "proposal_id": proposal.proposal_id,
                "child_track": int(proposal.child_track),
                "parent_track": int(proposal.parent_track),
                "anchor_track": int(proposal.anchor_track),
                "anchor_owner": int(proposal.anchor_owner), "frame": -1,
                "changed_pixels": 0, "core_offset_radii": 0.0,
                "core_area_radius_ratio": 0.0,
                "explained_projection_components": 0, "applied": False,
                "reason": "atomic_refusal:" + (failure or "incomplete_child"),
            })
            continue
        for detail in pending:
            candidate[detail["frame"]] = detail["trial"]
            explained += detail["explained"]
            rows.append({
                "proposal_id": proposal.proposal_id,
                "child_track": int(proposal.child_track),
                "parent_track": int(proposal.parent_track),
                "anchor_track": int(proposal.anchor_track),
                "anchor_owner": int(proposal.anchor_owner),
                "frame": detail["frame"],
                "changed_pixels": detail["pixels"],
                "core_offset_radii": detail["offset"],
                "core_area_radius_ratio": detail["area_ratio"],
                "explained_projection_components": detail["explained"],
                "applied": True,
                "reason": "terminal_projection_chain_raw_core",
            })
    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    actual_duplicates = component_accounting.new_duplicate_components(
        labels, candidate)
    details = {
        "eligible_proposals": int(len(eligible)),
        "applied_proposals": int(pd.DataFrame(rows).applied.astype(bool).groupby(
            pd.DataFrame(rows).proposal_id).all().sum()) if rows else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_added_pixels": int(additions.sum()),
        "preexisting_assigned_changed_pixels": int(np.count_nonzero(
            changed & (labels > 0))),
        "unclaimed_overlap_pixels": int(np.count_nonzero(
            additions & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_explained_projection_components": int(explained),
        "new_duplicate_components": max(0, int(actual_duplicates) - explained),
    }
    return candidate, pd.DataFrame(rows, columns=APPLICATION_COLUMNS), details
