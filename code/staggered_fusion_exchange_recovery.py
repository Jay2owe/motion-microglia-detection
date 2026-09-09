"""Recover reciprocal owner exchanges after a staggered close encounter.

Two physical paths need not carry their established owners simultaneously:
one can dim before the neighbour is first detected.  This field-wide rule
therefore establishes each path from its own earlier history, then requires
the paths to acquire each other's owners together after a local encounter.
Only components directly supported by the exchanged paths are relabelled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import component_accounting
import dormant_seat_successor_recovery as raw_recovery


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGET_TOKENS = (
    "identity", "owner_target", "track_target", "frame_target",
    "coordinate", "event", "region", "review", "case_id", "well",
    "forced", "allowed_pair",
)
AUDIT_COLUMNS = [
    "proposal_id", "track_a", "track_b", "owner_a", "owner_b",
    "owner_a_support", "owner_b_support", "wrong_b_support",
    "wrong_a_support", "owner_a_last", "owner_b_first",
    "track_a_exchange_first", "track_b_exchange_first",
    "owner_onset_gap_frames", "exchange_onset_gap_frames",
    "exchange_separation_sum_radii", "prefix_fraction_a",
    "prefix_fraction_b", "suffix_fraction_a", "suffix_fraction_b",
    "track_a_follower_count", "track_b_follower_count",
    "track_a_followers", "track_b_followers",
    "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "track_id", "expected_owner",
    "observed_owner", "operation", "component_area", "changed_pixels",
    "protecting_tracks",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("staggered fusion exchange must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "staggered fusion exchange received forbidden targets: "
            + ", ".join(supplied))


def selector_counts(params: dict) -> dict[str, int]:
    del params
    return {
        "identity_targets": 0, "owner_targets": 0, "track_targets": 0,
        "frame_targets": 0, "coordinate_targets": 0, "event_targets": 0,
        "region_targets": 0, "review_case_targets": 0, "well_targets": 0,
    }


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _valid_owner_rows(points: pd.DataFrame, params: dict) -> pd.DataFrame:
    purity = _fraction(params, "minimum_owner_purity", 0.80)
    coverage = _fraction(params, "minimum_owner_coverage", 0.15)
    return points[
        points.physically_visible.astype(bool)
        & (points.candidate_owner.astype(int) > 0)
        & (points.candidate_owner_purity.astype(float) >= purity)
        & (points.candidate_coverage_fraction.astype(float) >= coverage)
    ].copy()


def _directed_transitions(points: pd.DataFrame, frame_count: int,
                          params: dict) -> list[dict]:
    valid = _valid_owner_rows(points, params)
    minimum_prefix = max(3, int(math.ceil(frame_count * _fraction(
        params, "minimum_prior_owner_support_movie_fraction", 0.04))))
    minimum_suffix = max(2, int(math.ceil(frame_count * _fraction(
        params, "minimum_exchanged_owner_support_movie_fraction", 0.03))))
    minimum_prefix_fraction = _fraction(
        params, "minimum_prior_owner_fraction", 0.75)
    minimum_suffix_fraction = _fraction(
        params, "minimum_exchanged_owner_fraction", 0.60)
    transitions = []
    for track_id, group in valid.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        owners = sorted(set(map(int, group.candidate_owner)))
        for owner_from in owners:
            from_rows = group[group.candidate_owner.astype(int) == owner_from]
            if len(from_rows) < minimum_prefix:
                continue
            for owner_to in owners:
                if owner_to == owner_from:
                    continue
                to_rows = group[group.candidate_owner.astype(int) == owner_to]
                candidates = []
                for first_to in map(int, to_rows.frame):
                    prefix = group[group.frame.astype(int) < first_to]
                    suffix = group[group.frame.astype(int) >= first_to]
                    from_prefix = prefix[
                        prefix.candidate_owner.astype(int) == owner_from]
                    to_suffix = suffix[
                        suffix.candidate_owner.astype(int) == owner_to]
                    if (len(from_prefix) < minimum_prefix
                            or len(to_suffix) < minimum_suffix):
                        continue
                    prefix_fraction = len(from_prefix) / max(len(prefix), 1)
                    suffix_fraction = len(to_suffix) / max(len(suffix), 1)
                    if (prefix_fraction < minimum_prefix_fraction
                            or suffix_fraction < minimum_suffix_fraction):
                        continue
                    candidates.append({
                        "track": int(track_id), "owner_from": owner_from,
                        "owner_to": owner_to, "first_to": first_to,
                        "last_to": int(to_suffix.frame.max()),
                        "owner_support": int(len(from_prefix)),
                        "wrong_support": int(len(to_suffix)),
                        "owner_first": int(from_prefix.frame.min()),
                        "owner_last": int(from_prefix.frame.max()),
                        "prefix_fraction": float(prefix_fraction),
                        "suffix_fraction": float(suffix_fraction),
                    })
                if candidates:
                    transitions.append(min(
                        candidates, key=lambda item: item["first_to"]))
    return transitions


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    radii = float(left.radius_px) + float(right.radius_px)
    return distance / max(radii, 1.0)


def discover(points: pd.DataFrame, frame_count: int,
             params: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Audit every reciprocal pair of independently established paths."""
    assert_target_free(params)
    transitions = _directed_transitions(points, frame_count, params)
    index = points.set_index(["track_id", "frame"], drop=False)
    maximum_owner_gap = max(1, int(math.ceil(frame_count * _fraction(
        params, "maximum_staggered_owner_onset_gap_movie_fraction", 0.10))))
    maximum_exchange_gap = max(1, int(math.ceil(frame_count * _fraction(
        params, "maximum_exchange_onset_gap_movie_fraction", 0.04))))
    maximum_separation = float(params.get(
        "maximum_exchange_separation_sum_radii", 3.0))
    minimum_follower_support = max(3, int(math.ceil(
        frame_count * _fraction(
            params, "minimum_post_exchange_follower_support_fraction", 0.08))))
    maximum_follower_distance = float(params.get(
        "maximum_post_exchange_follower_distance_sum_radii", 3.0))
    valid = _valid_owner_rows(points, params)

    def followers(source_track: int, exchanged_owner: int,
                  exchange_frame: int) -> list[int]:
        result = []
        for track, group in valid.groupby("track_id", sort=True):
            track = int(track)
            if track == int(source_track):
                continue
            owner_rows = group[
                group.candidate_owner.astype(int) == int(exchanged_owner)]
            if len(owner_rows) < minimum_follower_support:
                continue
            first = int(owner_rows.frame.min())
            if first <= int(exchange_frame):
                continue
            # A follower cannot have an established accepted owner before the
            # exchange; that would be an independent cell, not a relay.
            earlier = group[group.frame.astype(int) < first]
            if len(earlier):
                continue
            source = _point(index, int(source_track), first)
            follower = _point(index, track, first)
            if source is None or follower is None:
                continue
            if _scaled_distance(source, follower) > maximum_follower_distance:
                continue
            result.append(track)
        return result
    audit = []
    eligible = []
    seen: set[tuple[int, int]] = set()
    for left in transitions:
        for right in transitions:
            if int(left["track"]) >= int(right["track"]):
                continue
            if not (int(left["owner_from"]) == int(right["owner_to"])
                    and int(left["owner_to"]) == int(right["owner_from"])):
                continue
            pair = (int(left["track"]), int(right["track"]))
            if pair in seen:
                continue
            seen.add(pair)
            exchange_frame = max(int(left["first_to"]),
                                 int(right["first_to"]))
            point_left = _point(index, int(left["track"]), exchange_frame)
            point_right = _point(index, int(right["track"]), exchange_frame)
            owner_gap = abs(int(left["owner_last"])
                            - int(right["owner_first"]))
            exchange_gap = abs(int(left["first_to"])
                               - int(right["first_to"]))
            separation = (_scaled_distance(point_left, point_right)
                          if point_left is not None and point_right is not None
                          else float("inf"))
            left_followers = followers(
                int(left["track"]), int(left["owner_to"]), exchange_frame)
            right_followers = followers(
                int(right["track"]), int(right["owner_to"]), exchange_frame)
            reasons = []
            if owner_gap > maximum_owner_gap:
                reasons.append("established_owner_windows_not_local")
            if exchange_gap > maximum_exchange_gap:
                reasons.append("reciprocal_exchange_onsets_not_local")
            if separation > maximum_separation:
                reasons.append("exchanged_paths_not_local")
            if point_left is None or point_right is None:
                reasons.append("exchange_frame_not_covisible")
            minimum_followers = max(0, int(params.get(
                "minimum_total_post_exchange_followers", 0)))
            if (not reasons
                    and len(left_followers) + len(right_followers)
                    < minimum_followers):
                reasons.append("insufficient_movie_new_exchange_followers")
            row = {
                "proposal_id": "", "track_a": int(left["track"]),
                "track_b": int(right["track"]),
                "owner_a": int(left["owner_from"]),
                "owner_b": int(right["owner_from"]),
                "owner_a_support": int(left["owner_support"]),
                "owner_b_support": int(right["owner_support"]),
                "wrong_b_support": int(left["wrong_support"]),
                "wrong_a_support": int(right["wrong_support"]),
                "owner_a_last": int(left["owner_last"]),
                "owner_b_first": int(right["owner_first"]),
                "track_a_exchange_first": int(left["first_to"]),
                "track_b_exchange_first": int(right["first_to"]),
                "owner_onset_gap_frames": owner_gap,
                "exchange_onset_gap_frames": exchange_gap,
                "exchange_separation_sum_radii": separation,
                "prefix_fraction_a": float(left["prefix_fraction"]),
                "prefix_fraction_b": float(right["prefix_fraction"]),
                "suffix_fraction_a": float(left["suffix_fraction"]),
                "suffix_fraction_b": float(right["suffix_fraction"]),
                "track_a_follower_count": len(left_followers),
                "track_b_follower_count": len(right_followers),
                "track_a_followers": "|".join(map(str, left_followers)),
                "track_b_followers": "|".join(map(str, right_followers)),
                "eligible": not reasons, "applied": False,
                "reason": ("eligible_staggered_reciprocal_exchange"
                           if not reasons else "|".join(reasons)),
            }
            audit.append(row)
            if not reasons:
                eligible.append({
                    "row": row, "left": left, "right": right,
                    "left_followers": left_followers,
                    "right_followers": right_followers,
                })
    for number, proposal in enumerate(eligible, 1):
        proposal["row"]["proposal_id"] = f"SFEX{number:04d}"
    return pd.DataFrame(audit, columns=AUDIT_COLUMNS), eligible


def _component_near(plane: np.ndarray, owner: int, point,
                    maximum_distance_radii: float) -> np.ndarray | None:
    components, count = ndi.label(plane == int(owner), STRUCTURE)
    if count <= 0:
        return None
    y = int(np.clip(round(float(point.y)), 0, plane.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, plane.shape[1] - 1))
    component_id = int(components[y, x])
    if component_id > 0:
        return components == component_id
    radius = max(float(point.radius_px), 1.0)
    choices = []
    for value in range(1, int(count) + 1):
        yy, xx = np.nonzero(components == value)
        if not len(xx):
            continue
        distance = float(np.sqrt(np.min(
            (yy - float(point.y)) ** 2 + (xx - float(point.x)) ** 2)))
        choices.append((distance / radius, value))
    if not choices or min(choices)[0] > maximum_distance_radii:
        return None
    return components == min(choices)[1]


def _initial_owner_profiles(points: pd.DataFrame, params: dict
                            ) -> dict[int, tuple[int, int]]:
    valid = _valid_owner_rows(points, params)
    profiles = {}
    for track, group in valid.groupby("track_id", sort=True):
        head = group.sort_values("frame").iloc[:max(3, len(group) // 2)]
        counts = head.candidate_owner.astype(int).value_counts()
        if len(counts):
            owner = int(counts.index[0])
            first = int(group[
                group.candidate_owner.astype(int) == owner].frame.min())
            profiles[int(track)] = (owner, first)
    return profiles


def _partition(component: np.ndarray, target, protectors: list) -> np.ndarray:
    yy, xx = np.nonzero(component)
    points = [target] + protectors
    costs = []
    for point in points:
        radius = max(float(point.radius_px), 1.0)
        costs.append(((yy - float(point.y)) ** 2
                      + (xx - float(point.x)) ** 2) / radius ** 2)
    selected = np.argmin(np.asarray(costs), axis=0) == 0
    result = np.zeros_like(component)
    result[yy[selected], xx[selected]] = True
    return result


def apply(labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
          eligible: list[dict], params: dict, raw: np.ndarray | None = None,
          thresholds: pd.DataFrame | None = None):
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    profiles = _initial_owner_profiles(points, params)
    visible_by_frame = {
        int(frame): group[group.physically_visible.astype(bool)]
        for frame, group in points.groupby("frame", sort=True)}
    maximum_distance = float(params.get(
        "maximum_component_point_distance_radii", 1.0))
    minimum_repair_purity = _fraction(
        params, "minimum_repair_owner_purity", 0.80)
    minimum_repair_coverage = _fraction(
        params, "minimum_repair_owner_coverage", 0.15)
    rows = []
    occupied_tracks: set[int] = set()
    applied = set()
    for proposal in eligible:
        row = proposal["row"]
        tracks = (int(row["track_a"]), int(row["track_b"]))
        if any(track in occupied_tracks for track in tracks):
            row["reason"] = "overlapping_exchange_paths"
            continue
        assignments = [
            (tracks[0], int(row["owner_a"]), int(row["owner_b"]),
             int(row["track_a_exchange_first"]),
             int(proposal["left"]["last_to"])),
            (tracks[1], int(row["owner_b"]), int(row["owner_a"]),
             int(row["track_b_exchange_first"]),
             int(proposal["right"]["last_to"])),
        ]
        exchange_first = max(int(row["track_a_exchange_first"]),
                             int(row["track_b_exchange_first"]))
        for track in proposal["left_followers"]:
            owner = int(row["owner_b"])
            owner_rows = points[
                (points.track_id.astype(int) == int(track))
                & (points.candidate_owner.astype(int) == owner)]
            assignments.append((int(track), int(row["owner_a"]), owner,
                                exchange_first,
                                int(owner_rows.frame.max())))
        for track in proposal["right_followers"]:
            owner = int(row["owner_a"])
            owner_rows = points[
                (points.track_id.astype(int) == int(track))
                & (points.candidate_owner.astype(int) == owner)]
            assignments.append((int(track), int(row["owner_b"]), owner,
                                exchange_first,
                                int(owner_rows.frame.max())))
        absorbed = set(proposal["left_followers"]) \
            | set(proposal["right_followers"])
        trial = candidate.copy()
        trial_unclaimed = candidate_unclaimed.copy()
        local_rows = []
        by_frame: dict[int, list[tuple[object, int, int]]] = {}
        for track, expected, observed, first, last in assignments:
            group = points[
                (points.track_id.astype(int) == track)
                & (points.frame.astype(int) >= first)
                & (points.frame.astype(int) <= last)
                & points.physically_visible.astype(bool)
                & (points.candidate_owner.astype(int) == observed)
                & (points.candidate_owner_purity.astype(float)
                   >= minimum_repair_purity)
                & (points.candidate_coverage_fraction.astype(float)
                   >= minimum_repair_coverage)]
            for point in group.sort_values("frame").itertuples(index=False):
                by_frame.setdefault(int(point.frame), []).append(
                    (point, expected, observed))

        lineage_tracks = set(tracks) | absorbed
        exchange_owners = {int(row["owner_a"]), int(row["owner_b"])}
        for frame, active in sorted(by_frame.items()):
            original = trial[frame].copy()
            touched = np.zeros_like(original, dtype=bool)
            active_components = []
            for point, expected, observed in active:
                component = _component_near(
                    original, observed, point, maximum_distance)
                if component is None:
                    continue
                touched |= component
                active_components.append((point, expected, component))
            if not active_components:
                continue

            # Existing paths established before the encounter can protect
            # their own portion of a touched component. Movie-new relays are
            # deliberately excluded because they belong to the exchange side.
            protectors = []
            active_track_ids = {
                int(item[0].track_id) for item in active_components}
            for other in visible_by_frame.get(
                    frame, pd.DataFrame()).itertuples(index=False):
                other_track = int(other.track_id)
                if (other_track in absorbed
                        or other_track in active_track_ids):
                    continue
                profile = profiles.get(other_track, (0, len(labels)))
                expected = int(profile[0])
                if (expected not in exchange_owners
                        or int(profile[1]) >= exchange_first):
                    continue
                # Before the second reciprocal path reaches its own exchange
                # onset, retain its established-owner seed if that ownership
                # is still directly observed. This permits staggered repair
                # without stealing the not-yet-exchanged neighbour.
                if (other_track in tracks
                        and (int(other.candidate_owner) != expected
                             or float(other.candidate_owner_purity)
                             < minimum_repair_purity
                             or float(other.candidate_coverage_fraction)
                             < minimum_repair_coverage)):
                    continue
                y = int(np.clip(round(float(other.y)), 0,
                                touched.shape[0] - 1))
                x = int(np.clip(round(float(other.x)), 0,
                                touched.shape[1] - 1))
                if touched[y, x]:
                    protectors.append((other, expected))

            before = trial[frame].copy()
            source_components = []
            for _, _, component in active_components:
                if not any(np.array_equal(component, existing)
                           for existing in source_components):
                    source_components.append(component)
            for source_component in source_components:
                yy, xx = np.nonzero(source_component)
                seeds = [
                    (point, expected)
                    for point, expected, component in active_components
                    if np.any(component & source_component)]
                for point, expected in protectors:
                    y = int(np.clip(round(float(point.y)), 0,
                                    source_component.shape[0] - 1))
                    x = int(np.clip(round(float(point.x)), 0,
                                    source_component.shape[1] - 1))
                    if source_component[y, x]:
                        seeds.append((point, expected))
                if not seeds:
                    continue
                owners = np.asarray(
                    [owner for _, owner in seeds], dtype=trial.dtype)
                if len(set(map(int, owners))) == 1:
                    trial[frame][source_component] = int(owners[0])
                    continue
                costs = []
                for point, _ in seeds:
                    radius = max(float(point.radius_px), 1.0)
                    costs.append(((yy - float(point.y)) ** 2
                                  + (xx - float(point.x)) ** 2) / radius ** 2)
                selected = np.argmin(np.asarray(costs), axis=0)
                trial[frame][yy, xx] = owners[selected]
            changed = int(np.count_nonzero(trial[frame] != before))
            if not changed:
                continue
            local_rows.append({
                "proposal_id": str(row["proposal_id"]), "frame": frame,
                "track_id": "|".join(map(str, sorted(set(
                    int(item[0].track_id) for item in active_components)))),
                "expected_owner": "|".join(map(str, sorted(set(
                    int(item[1]) for item in active_components)))),
                "observed_owner": "|".join(map(str, sorted(set(
                    int(item[0].candidate_owner)
                    for item in active_components)))),
                "operation": "atomic_source_component_exchange",
                "component_area": int(touched.sum()),
                "changed_pixels": changed,
                "protecting_tracks": "|".join(map(str, sorted(
                    int(item[0].track_id) for item in protectors))),
            })

        # If one exchange path has no movie-new follower, bridge its dim
        # ownerless interval only until its own established owner returns.
        # This prevents the opposite side's corrected body from becoming the
        # only carrier of that owner and creating another boundary swap.
        if raw is not None and thresholds is not None:
            maximum_bridge = max(1, int(math.ceil(len(labels) * _fraction(
                params, "maximum_reciprocal_bridge_movie_fraction", 0.25))))
            minimum_free = _fraction(
                params, "minimum_bridge_core_free_fraction", 0.65)
            threshold_index = thresholds.set_index("frame")
            evidence_cache: dict[int, np.ndarray] = {}
            bridge_sides = (
                (proposal["left"], proposal["left_followers"]),
                (proposal["right"], proposal["right_followers"]),
            )
            for transition, followers_for_side in bridge_sides:
                if followers_for_side:
                    continue
                track = int(transition["track"])
                expected = int(transition["owner_from"])
                wrong_last = int(transition["last_to"])
                group = points[
                    (points.track_id.astype(int) == track)
                    & (points.frame.astype(int) > wrong_last)] \
                    .sort_values("frame")
                returns = group[
                    (group.candidate_owner.astype(int) == expected)
                    & (group.candidate_owner_purity.astype(float)
                       >= minimum_repair_purity)
                    & (group.candidate_coverage_fraction.astype(float)
                       >= minimum_repair_coverage)]
                if not len(returns):
                    continue
                return_frame = int(returns.frame.min())
                if return_frame - wrong_last > maximum_bridge:
                    continue
                bridge = group[
                    group.frame.astype(int).between(
                        wrong_last + 1, return_frame - 1)
                    & group.physically_visible.astype(bool)]
                for point in bridge.itertuples(index=False):
                    frame = int(point.frame)
                    observed = int(point.candidate_owner)
                    if observed == expected:
                        continue
                    before = trial[frame].copy()
                    operation = ""
                    if observed > 0:
                        component = _component_near(
                            trial[frame], observed, point, maximum_distance)
                        if component is not None:
                            trial[frame][component] = expected
                            operation = "restore_foreign_bridge_component"
                    if not operation:
                        evidence = evidence_cache.setdefault(
                            frame, raw_recovery.evidence_image(
                                raw[frame], params))
                        found = raw_recovery._point_centered_raw_core(
                            raw[frame], evidence, point,
                            float(threshold_index.loc[
                                frame, "weak_threshold"]), params)
                        if found is None:
                            continue
                        core, region, details = found
                        free = core & (trial[frame][region] == 0)
                        if float(free.sum() / max(int(core.sum()), 1)) \
                                < minimum_free:
                            continue
                        view = trial[frame][region]
                        view[free] = expected
                        # An unclaimed mask is already detector-supported
                        # foreground, not a competing accepted owner.  When
                        # the independently tracked raw core proves this is
                        # the returning exchange path, transfer those pixels
                        # atomically from unclaimed to the expected owner.
                        unclaimed_view = trial_unclaimed[frame][region]
                        unclaimed_view[free] = 0
                        operation = str(details.get(
                            "method", "point_centered_raw_bridge"))
                    changed = int(np.count_nonzero(trial[frame] != before))
                    if changed:
                        local_rows.append({
                            "proposal_id": str(row["proposal_id"]),
                            "frame": frame, "track_id": track,
                            "expected_owner": expected,
                            "observed_owner": observed,
                            "operation": operation,
                            "component_area": changed,
                            "changed_pixels": changed,
                            "protecting_tracks": "",
                        })
        if not local_rows:
            row["reason"] = "eligible_but_no_supported_pixels"
            continue
        candidate = trial
        candidate_unclaimed = trial_unclaimed
        rows.extend(local_rows)
        occupied_tracks.update(tracks)
        applied.add(str(row["proposal_id"]))
        row["applied"] = True
        row["reason"] = "applied_component_local_reciprocal_restoration"

    changed = candidate != labels
    if np.any((labels > 0) & (candidate == 0)):
        raise AssertionError("staggered fusion exchange removed foreground")
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError("staggered fusion exchange overlaps unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("staggered fusion exchange changed identity set")
    details = {
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_added_pixels": int(np.count_nonzero(
            changed & (labels == 0) & (candidate > 0))),
        "foreground_removed_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "zero_signal_additions": int(np.count_nonzero(
            changed & (labels == 0) & (candidate > 0)
            & ((raw == 0) if raw is not None else False))),
        "unclaimed_overlap_pixels": int(np.count_nonzero(
            (candidate > 0) & (candidate_unclaimed > 0))),
        "unclaimed_reassigned_pixels": int(np.count_nonzero(
            (unclaimed > 0) & (candidate_unclaimed == 0)
            & (candidate > 0))),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "component_excess_before": int(
            component_accounting.total_component_excess(labels)),
        "component_excess_after": int(
            component_accounting.total_component_excess(candidate)),
        "duplicate_owner_frames_before": int(
            component_accounting.duplicate_owner_frames(labels)),
        "duplicate_owner_frames_after": int(
            component_accounting.duplicate_owner_frames(candidate)),
        "new_duplicate_components": int(
            component_accounting.new_duplicate_components(labels, candidate)),
        "applied_proposals": sorted(applied),
    }
    return (candidate, candidate_unclaimed,
            pd.DataFrame(rows, columns=FRAME_COLUMNS), details)


def produce(labels: np.ndarray, unclaimed: np.ndarray,
            points: pd.DataFrame, params: dict, raw: np.ndarray | None = None,
            thresholds: pd.DataFrame | None = None):
    assert_target_free(params)
    audit, eligible = discover(points, len(labels), params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
        details = {
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0, "foreground_removed_pixels": 0,
            "zero_signal_additions": 0,
            "unclaimed_overlap_pixels": 0, "new_identity_count": 0,
            "unclaimed_reassigned_pixels": 0,
            "removed_identity_count": 0,
            "component_excess_before": int(
                component_accounting.total_component_excess(labels)),
            "component_excess_after": int(
                component_accounting.total_component_excess(labels)),
            "duplicate_owner_frames_before": int(
                component_accounting.duplicate_owner_frames(labels)),
            "duplicate_owner_frames_after": int(
                component_accounting.duplicate_owner_frames(labels)),
            "new_duplicate_components": 0, "applied_proposals": [],
        }
    else:
        candidate, candidate_unclaimed, frames, details = apply(
            labels, unclaimed, points, eligible, params, raw, thresholds)
        applied = set(details["applied_proposals"])
        if len(audit):
            audit["applied"] = audit.proposal_id.astype(str).isin(applied)
            audit.loc[audit.applied.astype(bool), "reason"] = \
                "applied_component_local_reciprocal_restoration"
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": selector_counts(params),
        "physical_tracks_audited": int(points.track_id.nunique()),
        "reciprocal_pairs_audited": int(len(audit)),
        "eligible_pairs": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_pairs": int(len(details["applied_proposals"])),
        "frame_rows": int(len(frames)),
        "applied": bool(details["changed_pixels"]),
        **details,
    }
    return candidate, candidate_unclaimed, audit, frames, metrics


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    raw = tifffile.imread(params["raw_path"])[:len(labels)] \
        if params.get("raw_path") else None
    thresholds = pd.read_csv(params["thresholds_path"]) \
        if params.get("thresholds_path") else None
    candidate, candidate_unclaimed, audit, frames, metrics = produce(
        labels, unclaimed, points, params, raw, thresholds)
    output_dir = Path(out.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(
            labels_path, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    if np.array_equal(candidate_unclaimed, unclaimed):
        shutil.copy2(params["unclaimed_path"], unclaimed_path)
    else:
        tifffile.imwrite(
            unclaimed_path, candidate_unclaimed, imagej=True,
            compression="zlib", metadata={
                "axes": "TYX", "finterval": 1800.0,
                "tunit": "sec", "unit": "pixel"})
    outputs = {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": output_dir / "staggered_fusion_exchange_audit.csv",
        "frames": output_dir / "staggered_fusion_exchange_frames.csv",
        "metrics": output_dir / "metrics.json",
    }
    audit.to_csv(outputs["audit"], index=False)
    frames.to_csv(outputs["frames"], index=False)
    outputs["metrics"].write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": metrics}
