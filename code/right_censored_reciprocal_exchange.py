"""Repair decisive right-censored reciprocal owner exchanges field-wide.

The rule audits every co-visible physical-reference pair.  It accepts a pair
only when two durable owners reverse between nearby observations, same-track
motion is decisively cheaper than crossed motion, and both references end
shortly after the exchange.  Only frames carrying direct pair-reference
evidence are relabelled; no biological identity, frame, event, or region is
configured.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import component_accounting
from resident_takeover import attach_owners


ALLOWED_KEYS = {
    "maximum_observation_gap_movie_fraction",
    "maximum_reference_tail_movie_fraction",
    "minimum_owner_history_movie_fraction",
    "minimum_owner_purity",
    "minimum_owner_coverage",
    "minimum_leading_owner_coverage",
    "maximum_pair_separation_sum_radii",
    "minimum_crossed_to_same_motion_cost_ratio",
    "targeting_mode",
}
AUDIT_COLUMNS = [
    "proposal_id", "track_a", "track_b", "owner_a", "owner_b",
    "prior_frame", "exchange_frame", "track_a_last", "track_b_last",
    "prior_owner_a_frames", "prior_owner_b_frames",
    "prior_separation_sum_radii", "exchange_separation_sum_radii",
    "same_track_motion_cost", "crossed_motion_cost", "motion_cost_ratio",
    "repair_first", "repair_last", "repair_frames",
    "leading_repair_frames", "changed_pixels", "changed_frames",
    "new_duplicate_components", "eligible", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    """Reject any option outside the mathematical field-wide rule."""
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "right-censored reciprocal exchange must be field-wide")
    unexpected = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unexpected:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(unexpected))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _valid(point: object, purity: float, coverage: float) -> bool:
    return bool(
        bool(point.physically_visible)
        and int(point.candidate_owner) > 0
        and float(point.candidate_owner_purity) >= purity
        and float(point.candidate_coverage_fraction) >= coverage)


def _distance(left: object, right: object) -> float:
    return float(np.hypot(float(left.x) - float(right.x),
                          float(left.y) - float(right.y)))


def _scaled_distance(left: object, right: object) -> float:
    return _distance(left, right) / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def produce(labels: np.ndarray, unclaimed: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Return labels after every fully proved reciprocal exchange."""
    assert_target_free(params)
    if labels.shape != unclaimed.shape:
        raise ValueError("labels and unclaimed stacks must align")
    attached = attach_owners(points, labels).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)
    purity = _fraction(params, "minimum_owner_purity", 0.80)
    coverage = _fraction(params, "minimum_owner_coverage", 0.50)
    leading_coverage = _fraction(
        params, "minimum_leading_owner_coverage", 0.05)
    frame_count = len(labels)
    maximum_gap = max(1, int(math.ceil(frame_count * _fraction(
        params, "maximum_observation_gap_movie_fraction", 0.03))))
    maximum_tail = max(1, int(math.ceil(frame_count * _fraction(
        params, "maximum_reference_tail_movie_fraction", 0.06))))
    minimum_history = max(2, int(math.ceil(frame_count * _fraction(
        params, "minimum_owner_history_movie_fraction", 0.25))))
    maximum_separation = float(params.get(
        "maximum_pair_separation_sum_radii", 3.0))
    minimum_motion_ratio = float(params.get(
        "minimum_crossed_to_same_motion_cost_ratio", 1.5))

    valid = attached[attached.apply(
        lambda row: _valid(row, purity, coverage), axis=1)]
    owner_presence = {
        int(owner): set(map(int, frames))
        for owner, frames in ((owner, np.flatnonzero(np.any(
            labels == int(owner), axis=(1, 2))))
            for owner in np.unique(labels) if int(owner) > 0)}
    track_last = attached.groupby("track_id").frame.max().astype(int).to_dict()

    pair_observations: dict[tuple[int, int], list[tuple]] = {}
    for _, group in valid.groupby("frame", sort=True):
        rows = list(group.itertuples(index=False))
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                track_a, track_b = int(left.track_id), int(right.track_id)
                if track_a == track_b:
                    continue
                if track_a > track_b:
                    track_a, track_b, left, right = \
                        track_b, track_a, right, left
                pair_observations.setdefault((track_a, track_b), []).append(
                    (int(left.frame), left, right))

    proposals: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for (track_a, track_b), observations in pair_observations.items():
        for prior, exchange in zip(observations[:-1], observations[1:]):
            prior_frame, prior_a, prior_b = prior
            exchange_frame, exchange_a, exchange_b = exchange
            if exchange_frame - prior_frame > maximum_gap:
                continue
            owner_a = int(prior_a.candidate_owner)
            owner_b = int(prior_b.candidate_owner)
            if owner_a == owner_b:
                continue
            if not (int(exchange_a.candidate_owner) == owner_b
                    and int(exchange_b.candidate_owner) == owner_a):
                continue
            key = (owner_a, owner_b, prior_frame, exchange_frame)
            if key in seen:
                continue
            seen.add(key)
            support_a = sum(frame < prior_frame
                            for frame in owner_presence.get(owner_a, set()))
            support_b = sum(frame < prior_frame
                            for frame in owner_presence.get(owner_b, set()))
            prior_separation = _scaled_distance(prior_a, prior_b)
            exchange_separation = _scaled_distance(exchange_a, exchange_b)
            same_cost = (_distance(prior_a, exchange_a)
                         + _distance(prior_b, exchange_b))
            crossed_cost = (_distance(prior_a, exchange_b)
                            + _distance(prior_b, exchange_a))
            motion_ratio = crossed_cost / max(same_cost, 1e-9)
            reasons: list[str] = []
            if min(support_a, support_b) < minimum_history:
                reasons.append("owners_not_durable_before_exchange")
            if max(prior_separation, exchange_separation) > maximum_separation:
                reasons.append("reference_pair_not_local")
            if motion_ratio < minimum_motion_ratio:
                reasons.append("same_reference_motion_not_decisive")
            if (int(track_last[track_a]) - exchange_frame > maximum_tail
                    or int(track_last[track_b]) - exchange_frame
                    > maximum_tail):
                reasons.append("reference_pair_not_right_censored")
            proposals.append({
                "proposal_id": "", "track_a": track_a,
                "track_b": track_b, "owner_a": owner_a,
                "owner_b": owner_b, "prior_frame": prior_frame,
                "exchange_frame": exchange_frame,
                "track_a_last": int(track_last[track_a]),
                "track_b_last": int(track_last[track_b]),
                "prior_owner_a_frames": support_a,
                "prior_owner_b_frames": support_b,
                "prior_separation_sum_radii": prior_separation,
                "exchange_separation_sum_radii": exchange_separation,
                "same_track_motion_cost": same_cost,
                "crossed_motion_cost": crossed_cost,
                "motion_cost_ratio": motion_ratio,
                "changed_pixels": 0, "changed_frames": 0,
                "new_duplicate_components": 0,
                "eligible": not reasons, "applied": False,
                "reason": ("eligible_right_censored_reciprocal_exchange"
                           if not reasons else "|".join(reasons)),
            })

    candidate = labels.copy()
    occupied_owners: set[int] = set()
    number = 0
    for proposal in proposals:
        if not proposal["eligible"]:
            continue
        owner_a, owner_b = int(proposal["owner_a"]), int(proposal["owner_b"])
        if owner_a in occupied_owners or owner_b in occupied_owners:
            proposal["reason"] = "overlapping_owner_exchange"
            continue
        first = int(proposal["exchange_frame"])
        last = max(int(proposal["track_a_last"]),
                   int(proposal["track_b_last"]))
        pair_points = attached[
            attached.track_id.astype(int).isin((
                int(proposal["track_a"]), int(proposal["track_b"])))
            & attached.frame.astype(int).between(first, last)
            & attached.physically_visible.astype(bool)]
        prior = int(proposal["prior_frame"])
        leading_points = attached[
            attached.track_id.astype(int).isin((
                int(proposal["track_a"]), int(proposal["track_b"])))
            & attached.frame.astype(int).between(
                max(0, prior - maximum_gap), prior - 1)
            & attached.physically_visible.astype(bool)
            & (attached.candidate_owner_purity.astype(float) >= purity)
            & (attached.candidate_coverage_fraction.astype(float)
               >= leading_coverage)]
        reciprocal_leading = leading_points[
            ((leading_points.track_id.astype(int)
              == int(proposal["track_a"]))
             & (leading_points.candidate_owner.astype(int) == owner_b))
            | ((leading_points.track_id.astype(int)
                == int(proposal["track_b"]))
               & (leading_points.candidate_owner.astype(int) == owner_a))]
        leading_frames = sorted(set(map(int, reciprocal_leading.frame)))
        repair_frames = sorted(
            set(map(int, pair_points.frame)) | set(leading_frames))
        trial = candidate.copy()
        for frame in repair_frames:
            mask_a = trial[frame] == owner_a
            mask_b = trial[frame] == owner_b
            trial[frame][mask_a] = owner_b
            trial[frame][mask_b] = owner_a
        duplicate_count = component_accounting.new_duplicate_components(
            labels, trial)
        changed = trial != candidate
        candidate = trial
        number += 1
        proposal["proposal_id"] = f"RCRE{number:04d}"
        proposal["changed_pixels"] = int(changed.sum())
        proposal["changed_frames"] = int(np.any(
            changed, axis=(1, 2)).sum())
        proposal["repair_first"] = min(repair_frames)
        proposal["repair_last"] = max(repair_frames)
        proposal["repair_frames"] = len(repair_frames)
        proposal["leading_repair_frames"] = len(leading_frames)
        proposal["new_duplicate_components"] = duplicate_count
        proposal["applied"] = True
        proposal["reason"] = "applied_reference_bounded_owner_permutation"
        occupied_owners.update((owner_a, owner_b))

    audit = pd.DataFrame(proposals, columns=AUDIT_COLUMNS)
    changed = candidate != labels
    duplicate_count = component_accounting.new_duplicate_components(
        labels, candidate)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reciprocal exchange changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("reciprocal exchange overlaps unclaimed data")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("reciprocal exchange changed identity set")
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "pairs_audited": int(len(audit)),
        "eligible_exchanges": int(audit.eligible.astype(bool).sum())
        if len(audit) else 0,
        "applied_exchanges": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.any(changed, axis=(1, 2)).sum()),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(duplicate_count),
    }
    return candidate, unclaimed.copy(), audit, metrics
