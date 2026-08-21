from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


BIG = 1e9


@dataclass
class IdentityState:
    identity: int
    t: int
    mask: np.ndarray
    position: np.ndarray
    area: int
    mean: float
    velocity: np.ndarray
    green_track_ids: set[int]
    age: int = 1
    reference_area: float = np.nan
    reference_mean: float = np.nan
    reference_shape: np.ndarray | None = None
    bounds: tuple[int, int, int, int] | None = None


def _shape_descriptor(mask: np.ndarray) -> np.ndarray:
    """Scale-normalised body axes and perimeter for lineage comparison."""
    points = np.column_stack(np.nonzero(mask)).astype(float)
    if len(points) < 2:
        return np.zeros(3, float)
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    scale = np.sqrt(max(len(points), 1))
    axes = np.sqrt(np.maximum(eigenvalues, 0.0) + 1e-6) / scale
    edge = mask ^ ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))
    return np.array([axes[0], axes[1], float(edge.sum()) / scale], float)


def frame_observations(labels: np.ndarray, raw: np.ndarray,
                       green_tracks: dict[int, set[int]] | None = None,
                       preferred_identities: dict[int, int] | None = None,
                       protected_preferred: dict[int, bool] | None = None,
                       ) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for local_id in np.unique(labels):
        local_id = int(local_id)
        if local_id <= 0:
            continue
        mask = labels == local_id
        yy, xx = np.nonzero(mask)
        result[local_id] = {
            "local_id": local_id,
            "mask": mask,
            "position": np.array([yy.mean(), xx.mean()], float),
            "bounds": (int(yy.min()), int(yy.max()) + 1,
                       int(xx.min()), int(xx.max()) + 1),
            "area": int(mask.sum()),
            "mean": float(raw[mask].mean()),
            "shape": _shape_descriptor(mask),
            "green_track_ids": set() if green_tracks is None
                               else set(green_tracks.get(local_id, set())),
            "preferred_identity": 0 if preferred_identities is None
                                  else int(preferred_identities.get(local_id, 0)),
            "preferred_identity_protected": (
                False if protected_preferred is None
                else bool(protected_preferred.get(local_id, False))),
        }
    return result


def motion_support(earlier: np.ndarray, later: np.ndarray, lag: np.ndarray,
                   earlier_bounds: tuple[int, int, int, int] | None = None,
                   later_bounds: tuple[int, int, int, int] | None = None,
                   ) -> dict:
    if earlier_bounds is not None and later_bounds is not None:
        y0 = min(earlier_bounds[0], later_bounds[0])
        y1 = max(earlier_bounds[1], later_bounds[1])
        x0 = min(earlier_bounds[2], later_bounds[2])
        x1 = max(earlier_bounds[3], later_bounds[3])
    else:
        union = earlier | later
        yy, xx = np.nonzero(union)
        if not len(yy):
            return {"motion_support": 0.0, "supportive_px": 0,
                    "contradictory_px": 0, "union_px": 0,
                    "source_loss_coverage": 0.0,
                    "destination_gain_coverage": 0.0}
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1
    if y0 >= y1 or x0 >= x1:
        return {"motion_support": 0.0, "supportive_px": 0,
                "contradictory_px": 0, "union_px": 0,
                "source_loss_coverage": 0.0,
                "destination_gain_coverage": 0.0}
    old = earlier[y0:y1, x0:x1]
    new = later[y0:y1, x0:x1]
    union = old | new
    local = lag[y0:y1, x0:x1]
    old_only, new_only = old & ~new, new & ~old
    shared = old & new
    loss_support = int(np.count_nonzero(old_only & (local <= -1.5)))
    gain_support = int(np.count_nonzero(new_only & (local >= 1.5)))
    stable_support = int(np.count_nonzero(shared & (np.abs(local) < 1.0)))
    supportive = loss_support + gain_support + stable_support
    contradictory = (
        np.count_nonzero(old_only & (local >= 1.5))
        + np.count_nonzero(new_only & (local <= -1.5))
    )
    measured_union = union & np.isfinite(local)
    total = int(measured_union.sum())
    return {
        "motion_support": float((supportive - contradictory) / max(total, 1)),
        "supportive_px": int(supportive),
        "contradictory_px": int(contradictory),
        "union_px": total,
        "source_loss_coverage": float(loss_support / max(int(old.sum()), 1)),
        "destination_gain_coverage": float(gain_support / max(int(new.sum()), 1)),
    }


def link_features(state: IdentityState, observation: dict, target_t: int,
                  lag: np.ndarray, direction: int, params: dict) -> dict:
    steps = abs(int(target_t) - int(state.t))
    predicted = state.position + state.velocity * max(steps, 1)
    preferred_identity = int(observation.get("preferred_identity", 0))
    hard_reserved = {
        int(value) for value in params.get("hard_reserved_identity_ids", [])}
    if state.identity in hard_reserved and state.identity == preferred_identity:
        # Reserved fast movers are located by their blue-to-red destination from
        # the last actual soma, not by a velocity extrapolation that can overshoot
        # after abrupt turns.
        predicted = state.position.copy()
    distance = float(np.linalg.norm(predicted - observation["position"]))
    observed_displacement = float(np.linalg.norm(
        state.position - observation["position"]))
    maximum_distance = float(params["max_link_px"]) * np.sqrt(max(steps, 1))
    maximum_recovery_distance = float(params.get(
        "maximum_recovery_distance_px", np.inf))
    area_ratio = float(observation["area"] / max(state.area, 1))
    area_change = abs(float(np.log2(max(area_ratio, 1e-12))))
    mean_change = abs(float(np.log2((observation["mean"] + 1.0)
                                    / (state.mean + 1.0))))
    reference_area = (float(state.reference_area)
                      if np.isfinite(state.reference_area) else float(state.area))
    reference_mean = (float(state.reference_mean)
                      if np.isfinite(state.reference_mean) else float(state.mean))
    reference_shape = (state.reference_shape if state.reference_shape is not None
                       else _shape_descriptor(state.mask))
    lineage_area_change = abs(float(np.log2(
        max(float(observation["area"]) / max(reference_area, 1.0), 1e-12))))
    lineage_mean_change = abs(float(np.log2(
        (float(observation["mean"]) + 1.0) / (reference_mean + 1.0))))
    lineage_shape_change = float(np.mean(np.abs(np.log2(
        (np.asarray(observation["shape"], float) + 1e-3)
        / (np.asarray(reference_shape, float) + 1e-3)))))
    step_velocity = (observation["position"] - state.position) / max(steps, 1)
    old_speed = float(np.linalg.norm(state.velocity))
    new_speed = float(np.linalg.norm(step_velocity))
    reversal_penalty = 0.0
    minimum_direction_speed = float(params.get(
        "minimum_direction_speed_px", 2.0))
    if old_speed >= minimum_direction_speed and new_speed >= minimum_direction_speed:
        cosine = float(np.dot(state.velocity, step_velocity)
                       / max(old_speed * new_speed, 1e-12))
        reversal_penalty = max(0.0, -cosine)
    shared_green = len(state.green_track_ids & observation["green_track_ids"])
    shared_distance_multiplier = float(
        params.get("shared_motion_distance_multiplier", 1.5))
    shared_reward = float(params.get("shared_motion_reward", 1.5))
    support = {"motion_support": 0.0, "supportive_px": 0,
               "contradictory_px": 0, "union_px": 0,
               "source_loss_coverage": 0.0,
               "destination_gain_coverage": 0.0}
    # A complete motion handoff can legitimately exceed the ordinary link radius, but
    # it cannot exceed the explicit fast-motion bound.  This cheap spatial prefilter
    # avoids comparing every cell with every distant cell across the full field.
    fast_motion_bound = float(params.get(
        "fast_motion_max_link_px", maximum_recovery_distance))
    if steps == 1 and observed_displacement <= fast_motion_bound:
        if direction > 0:
            support = motion_support(
                state.mask, observation["mask"], lag[state.t],
                state.bounds, observation["bounds"])
        else:
            support = motion_support(
                observation["mask"], state.mask, lag[target_t],
                observation["bounds"], state.bounds)
    fast_motion = bool(
        not bool(params.get("red_led_motion", False))
        and
        steps == 1
        and support["motion_support"] >= float(
            params.get("fast_motion_minimum_support", np.inf))
        and support["supportive_px"] >= int(
            params.get("fast_motion_minimum_px", 1))
        and support["source_loss_coverage"] >= float(
            params.get("fast_motion_minimum_source_loss_coverage", 0.0))
        and support["destination_gain_coverage"] >= float(
            params.get("fast_motion_minimum_destination_gain_coverage", 0.0)))
    reversal_aware = bool(
        params.get("reversal_aware_motion", False)
        and steps == 1
        and reversal_penalty > 0.0
        and observed_displacement <= float(params.get(
            "reversal_aware_max_link_px", params["max_link_px"]))
        and support["motion_support"] >= float(params.get(
            "reversal_aware_minimum_motion_support", 0.7))
        and support["source_loss_coverage"] >= float(params.get(
            "reversal_aware_minimum_source_loss_coverage", 0.7))
        and support["destination_gain_coverage"] >= float(params.get(
            "reversal_aware_minimum_destination_gain_coverage", 0.7))
        and support["supportive_px"] >= int(params.get(
            "reversal_aware_minimum_supportive_px", 8)))
    shared_can_extend = bool(
        shared_green and (
            not bool(params.get("shared_motion_requires_support", False))
            or support["motion_support"] >= float(
                params.get("shared_motion_minimum_support", 0.0))))
    if fast_motion or reversal_aware:
        # REGRESSION GUARD: complete blue-to-red evidence must be evaluated before
        # the ordinary distance gate, especially when a cell reverses direction.
        gate_distance = observed_displacement
        permitted_distance = (fast_motion_bound if fast_motion else float(
            params.get("reversal_aware_max_link_px", params["max_link_px"])))
    else:
        gate_distance = distance
        permitted_distance = maximum_distance * (
            shared_distance_multiplier if shared_can_extend else 1.0)
    allowed = (gate_distance <= permitted_distance
               and observed_displacement <= maximum_recovery_distance
               and float(params["minimum_area_ratio"]) <= area_ratio
               <= float(params["maximum_area_ratio"])
               and mean_change <= float(params["maximum_mean_log2_change"]))
    if state.identity in hard_reserved or preferred_identity in hard_reserved:
        reserved_distance = float(params.get(
            "hard_reserved_max_link_px", params["max_link_px"]))
        reserved_minimum_area = float(params.get(
            "hard_reserved_minimum_area_ratio", params["minimum_area_ratio"]))
        reserved_maximum_area = float(params.get(
            "hard_reserved_maximum_area_ratio", params["maximum_area_ratio"]))
        maximum_distance = max(
            maximum_distance, reserved_distance * np.sqrt(max(steps, 1)))
        allowed = bool(
            state.identity == preferred_identity
            and distance <= maximum_distance
            and observed_displacement <= float(params.get(
                "hard_reserved_max_link_px", maximum_recovery_distance))
            and reserved_minimum_area <= area_ratio <= reserved_maximum_area
            and mean_change <= float(params["maximum_mean_log2_change"]))
    effective_distance = (observed_displacement
                          if fast_motion or reversal_aware else distance)
    effective_reversal_penalty = 0.0 if reversal_aware else reversal_penalty
    high_ids = {int(value) for value in params.get(
        "high_confidence_identity_ids", [])}
    preferred_reward = float(params.get("preferred_identity_reward", 0.0))
    if high_ids and state.identity not in high_ids:
        preferred_reward = float(params.get(
            "low_confidence_preferred_identity_reward", preferred_reward))
    cost = (effective_distance / max(float(params["max_link_px"]), 1e-6)
            + float(params.get("area_weight", 0.45)) * area_change
            + float(params.get("mean_weight", 0.25)) * mean_change
            + float(params.get("lineage_area_weight", 0.0))
              * lineage_area_change
            + float(params.get("lineage_mean_weight", 0.0))
              * lineage_mean_change
            + float(params.get("lineage_shape_weight", 0.0))
              * lineage_shape_change
            + float(params.get("direction_reversal_weight", 0.0))
              * effective_reversal_penalty
            - float(params["motion_weight"]) * support["motion_support"]
            - (shared_reward if shared_green else 0.0)
            - (preferred_reward
               if preferred_identity == state.identity
               else 0.0)
            + 0.15 * max(steps - 1, 0))
    return {
        "allowed": bool(allowed), "cost": float(cost), "distance_px": distance,
        "observed_displacement_px": observed_displacement,
        "predicted_y": float(predicted[0]), "predicted_x": float(predicted[1]),
        "area_ratio": area_ratio, "mean_log2_change": mean_change,
        "lineage_area_log2_change": lineage_area_change,
        "lineage_mean_log2_change": lineage_mean_change,
        "lineage_shape_log2_change": lineage_shape_change,
        "direction_reversal_penalty": effective_reversal_penalty,
        "raw_direction_reversal_penalty": reversal_penalty,
        "shared_green_tracks": int(shared_green),
        "fast_motion_handoff": fast_motion,
        "reversal_aware_handoff": reversal_aware,
        "motion_prediction_model": (
            "current_position_after_supported_reversal" if reversal_aware
            else "constant_velocity"),
        "gap_steps": int(steps), **support,
    }


def _red_led_motion_reservations(
        states: list[IdentityState], observations: list[dict],
        features: dict[tuple[int, int], dict], direction: int, params: dict,
        ) -> list[tuple[int, int, dict]]:
    """Let each new red destination choose its most plausible blue source.

    REGRESSION GUARD: a blue source must not grab the nearest red edge on a
    neighbouring stationary cell. Only an unambiguous red arrival may reserve a
    complete blue source, after lineage size, shape and direction are considered.
    """
    if not bool(params.get("red_led_motion", False)):
        return []
    minimum_source = float(params.get(
        "red_led_minimum_source_loss_coverage", 0.7))
    minimum_destination = float(params.get(
        "red_led_minimum_destination_gain_coverage", 0.7))
    minimum_px = int(params.get("red_led_minimum_supportive_px", 12))
    maximum_distance = float(params.get(
        "red_led_maximum_link_px", params.get("fast_motion_max_link_px", 60.0)))
    maximum_cost = float(params.get("red_led_maximum_cost", np.inf))
    minimum_margin = float(params.get("red_led_minimum_margin", 0.2))
    minimum_age = int(params.get("red_led_minimum_state_age", 1))
    high_ids = {int(identity) for identity in params.get(
        "high_confidence_identity_ids", [])}
    protect_preferred_px = float(params.get(
        "red_led_protect_preferred_within_px", params.get("max_link_px", 24.0)))
    protect_preferred_gap = int(params.get(
        "red_led_protect_preferred_max_gap_frames", 3))
    cross_high_pairs: set[tuple[int, int]] = set()
    proposals: list[tuple[float, float, int, int, dict]] = []

    # Forward tracking: each observation is the new red destination. Backward
    # tracking: the current state is that same later red destination and each
    # earlier observation is a possible blue source.
    destination_indices = (range(len(observations)) if direction > 0
                           else range(len(states)))
    for destination_index in destination_indices:
        candidates: list[tuple[float, int, int, dict]] = []
        pairs = (((row, destination_index) for row in range(len(states)))
                 if direction > 0 else
                 ((destination_index, col) for col in range(len(observations))))
        for row, col in pairs:
            state = states[row]
            value = features[row, col]
            preferred = int(observations[col].get("preferred_identity", 0))
            if (state.identity != preferred
                    and bool(observations[col].get(
                        "preferred_identity_protected", False))):
                continue
            cross_high = bool(
                high_ids
                and (state.identity in high_ids or preferred in high_ids)
                and state.identity != preferred)
            if cross_high and not bool(params.get(
                    "red_led_allow_high_confidence_reassignment", False)):
                continue
            if cross_high and preferred > 0:
                # A lost identity may reclaim only unowned territory. If the
                # destination's established owner has a short, spatially
                # continuous route to it, red motion cannot steal that object.
                preferred_rows = [
                    preferred_row for preferred_row, preferred_state
                    in enumerate(states) if preferred_state.identity == preferred]
                preferred_is_continuous = any(
                    features[preferred_row, col]["gap_steps"]
                    <= protect_preferred_gap
                    and features[preferred_row, col]["observed_displacement_px"]
                    <= protect_preferred_px
                    * features[preferred_row, col]["gap_steps"]
                    for preferred_row in preferred_rows)
                if preferred_is_continuous:
                    continue
            source_threshold = (float(params.get(
                "red_led_high_minimum_source_loss_coverage", minimum_source))
                                if cross_high else minimum_source)
            destination_threshold = (float(params.get(
                "red_led_high_minimum_destination_gain_coverage",
                minimum_destination)) if cross_high else minimum_destination)
            if (state.age < minimum_age or value["gap_steps"] != 1
                    or value["source_loss_coverage"] < source_threshold
                    or value["destination_gain_coverage"] < destination_threshold
                    or value["supportive_px"] < minimum_px
                    or value["observed_displacement_px"] > maximum_distance):
                continue
            if cross_high:
                cross_high_pairs.add((row, col))
            preferred_reward = float(params.get(
                "red_led_preferred_identity_reward", 0.0)) \
                if preferred == state.identity else 0.0
            cost = (
                value["observed_displacement_px"]
                / max(float(params.get("red_led_distance_scale_px", 36.0)), 1e-6)
                + float(params.get("red_led_area_weight", 1.0))
                  * value["lineage_area_log2_change"]
                + float(params.get("red_led_mean_weight", 0.25))
                  * value["lineage_mean_log2_change"]
                + float(params.get("red_led_shape_weight", 0.75))
                  * value["lineage_shape_log2_change"]
                + float(params.get("red_led_reversal_weight", 1.0))
                  * value["direction_reversal_penalty"]
                - float(params.get("red_led_motion_reward", 2.0))
                  * min(value["source_loss_coverage"],
                        value["destination_gain_coverage"])
                - preferred_reward)
            candidates.append((float(cost), row, col, value))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], states[item[1]].identity,
                                           observations[item[2]]["local_id"]))
        best = candidates[0]
        second_cost = candidates[1][0] if len(candidates) > 1 else np.inf
        margin = float(second_cost - best[0])
        required_margin = (float(params.get(
            "red_led_high_minimum_margin", minimum_margin))
                           if (best[1], best[2]) in cross_high_pairs
                           else minimum_margin)
        if best[0] > maximum_cost or margin < required_margin:
            continue
        value = dict(best[3])
        value.update({
            "allowed": True,
            "cost": float(best[0]),
            "red_led_motion_handoff": True,
            "red_led_candidate_margin": margin,
            "red_led_direction": "red_destination_to_blue_source",
        })
        proposals.append((float(best[0]), -margin, best[1], best[2], value))

    # Conflicting red destinations cannot both claim the same blue source. The
    # strongest unambiguous destination wins; the other stays unresolved for the
    # ordinary, distance-limited assignment rather than forcing a swap.
    reservations: list[tuple[int, int, dict]] = []
    used_states: set[int] = set()
    used_observations: set[int] = set()
    for _, _, row, col, value in sorted(proposals):
        if row in used_states or col in used_observations:
            continue
        reservations.append((row, col, value))
        used_states.add(row)
        used_observations.add(col)
    return reservations


def _solve_subset(
        state_indices: list[int], observation_indices: list[int],
        features: dict[tuple[int, int], dict], params: dict,
        blocked: set[tuple[int, int]] | None = None,
        ) -> list[tuple[int, int, dict]]:
    """Solve one assignment layer while retaining original list indices."""
    if not state_indices or not observation_indices:
        return []
    n_states, n_obs = len(state_indices), len(observation_indices)
    matrix = np.full((n_states + n_obs, n_states + n_obs), BIG, float)
    for local_row, row in enumerate(state_indices):
        for local_col, col in enumerate(observation_indices):
            value = features[row, col]
            if value["allowed"] and (blocked is None or (row, col) not in blocked):
                matrix[local_row, local_col] = value["cost"]
        matrix[local_row, n_obs + local_row] = float(params["miss_cost"])
    for local_col in range(n_obs):
        matrix[n_states + local_col, local_col] = float(params["birth_cost"])
    matrix[n_states:, n_obs:] = 0.0
    rows, cols = linear_sum_assignment(matrix)
    result: list[tuple[int, int, dict]] = []
    for local_row, local_col in zip(rows, cols):
        if (local_row < n_states and local_col < n_obs
                and matrix[local_row, local_col] < BIG):
            row, col = state_indices[int(local_row)], observation_indices[int(local_col)]
            result.append((row, col, features[row, col]))
    return result


def _occupancy_reservations(
        states: list[IdentityState], observations: list[dict],
        features: dict[tuple[int, int], dict], params: dict,
        ) -> list[tuple[int, int, dict]]:
    """Reserve unambiguous one-frame mask continuity before global assignment.

    A long-gap identity must not displace a substantial soma whose previous mask
    still occupies the same pixels.  Mutual-best overlap and competitor gates keep
    true one-to-many and many-to-one contacts available to merge/split reasoning.
    """
    if not bool(params.get("occupancy_reservation", False)):
        return []
    n_states, n_obs = len(states), len(observations)
    if not n_states or not n_obs:
        return []
    intersection = np.zeros((n_states, n_obs), np.int64)
    iou = np.zeros((n_states, n_obs), float)
    source_coverage = np.zeros((n_states, n_obs), float)
    destination_coverage = np.zeros((n_states, n_obs), float)
    for row, state in enumerate(states):
        if int(features[row, 0]["gap_steps"]) != 1:
            continue
        for col, observation in enumerate(observations):
            state_bounds = state.bounds
            observation_bounds = observation.get("bounds")
            if state_bounds is None or observation_bounds is None:
                pixels = int(np.count_nonzero(
                    state.mask & observation["mask"]))
            else:
                y0 = max(int(state_bounds[0]), int(observation_bounds[0]))
                y1 = min(int(state_bounds[1]), int(observation_bounds[1]))
                x0 = max(int(state_bounds[2]), int(observation_bounds[2]))
                x1 = min(int(state_bounds[3]), int(observation_bounds[3]))
                if y0 >= y1 or x0 >= x1:
                    continue
                pixels = int(np.count_nonzero(
                    state.mask[y0:y1, x0:x1]
                    & observation["mask"][y0:y1, x0:x1]))
            if pixels <= 0:
                continue
            union = int(state.area) + int(observation["area"]) - pixels
            intersection[row, col] = pixels
            iou[row, col] = pixels / max(union, 1)
            source_coverage[row, col] = pixels / max(int(state.area), 1)
            destination_coverage[row, col] = pixels / max(
                int(observation["area"]), 1)

    minimum_intersection = int(params.get(
        "occupancy_minimum_intersection_px", 12))
    minimum_iou = float(params.get(
        "occupancy_minimum_intersection_over_union", 0.50))
    minimum_coverage = float(params.get(
        "occupancy_minimum_bidirectional_coverage", 0.70))
    minimum_ratio = float(params.get("occupancy_minimum_area_ratio", 0.50))
    maximum_ratio = float(params.get("occupancy_maximum_area_ratio", 2.00))
    minimum_margin = float(params.get("occupancy_minimum_iou_margin", 0.20))
    maximum_competitor = float(params.get(
        "occupancy_maximum_competitor_coverage", 0.20))

    proposals: list[tuple[float, int, int, dict]] = []
    for row, state in enumerate(states):
        if not np.any(intersection[row]):
            continue
        best_col = int(np.argmax(iou[row]))
        if int(np.argmax(iou[:, best_col])) != row:
            continue
        row_other = np.delete(iou[row], best_col)
        col_other = np.delete(iou[:, best_col], row)
        row_second = float(row_other.max()) if len(row_other) else 0.0
        col_second = float(col_other.max()) if len(col_other) else 0.0
        margin = min(float(iou[row, best_col]) - row_second,
                     float(iou[row, best_col]) - col_second)
        source_competitors = np.delete(source_coverage[row], best_col)
        destination_competitors = np.delete(
            destination_coverage[:, best_col], row)
        competitor = max(
            float(source_competitors.max()) if len(source_competitors) else 0.0,
            float(destination_competitors.max())
            if len(destination_competitors) else 0.0)
        area_ratio = float(observations[best_col]["area"] / max(state.area, 1))
        allowed = bool(
            intersection[row, best_col] >= minimum_intersection
            and iou[row, best_col] >= minimum_iou
            and source_coverage[row, best_col] >= minimum_coverage
            and destination_coverage[row, best_col] >= minimum_coverage
            and minimum_ratio <= area_ratio <= maximum_ratio
            and margin >= minimum_margin
            and competitor < maximum_competitor)
        if not allowed:
            continue
        value = dict(features[row, best_col])
        value.update({
            "allowed": True,
            "occupancy_reserved": True,
            "occupancy_intersection_px": int(intersection[row, best_col]),
            "occupancy_intersection_over_union": float(iou[row, best_col]),
            "occupancy_source_coverage": float(source_coverage[row, best_col]),
            "occupancy_destination_coverage": float(
                destination_coverage[row, best_col]),
            "occupancy_iou_margin": margin,
            "occupancy_maximum_competitor_coverage": competitor,
            "occupancy_rejection_reason": "",
            "motion_prediction_model": "mutual_same_territory_overlap",
        })
        proposals.append((-float(iou[row, best_col]), row, best_col, value))

    reservations: list[tuple[int, int, dict]] = []
    used_states: set[int] = set()
    used_observations: set[int] = set()
    for _, row, col, value in sorted(proposals):
        if row in used_states or col in used_observations:
            continue
        reservations.append((row, col, value))
        used_states.add(row)
        used_observations.add(col)
    return reservations


def _assign_legacy(states: list[IdentityState], observations: list[dict],
                   target_t: int, lag: np.ndarray, direction: int, params: dict,
                   ) -> tuple[list[tuple[int, int, dict]], set[int], set[int]]:
    n_states, n_obs = len(states), len(observations)
    if not n_states:
        return [], set(), set(range(n_obs))
    if not n_obs:
        return [], set(range(n_states)), set()
    features: dict[tuple[int, int], dict] = {}
    for row, state in enumerate(states):
        for col, observation in enumerate(observations):
            value = link_features(state, observation, target_t, lag, direction, params)
            features[row, col] = value
    links = _occupancy_reservations(
        states, observations, features, params)
    matched_states = {row for row, _, _ in links}
    matched_obs = {col for _, col, _ in links}
    motion_links = _red_led_motion_reservations(
        states, observations, features, direction, params)
    motion_links = [
        link for link in motion_links
        if link[0] not in matched_states and link[1] not in matched_obs]
    links.extend(motion_links)
    matched_states.update(row for row, _, _ in motion_links)
    matched_obs.update(col for _, col, _ in motion_links)

    high_ids = {int(value) for value in params.get(
        "high_confidence_identity_ids", [])}
    if high_ids:
        high_states = [row for row, state in enumerate(states)
                       if row not in matched_states and state.identity in high_ids]
        high_observations = [
            col for col, observation in enumerate(observations)
            if col not in matched_obs
            and int(observation.get("preferred_identity", 0)) in high_ids]
        # Established names are fixed before the flexible layer. Ordinary
        # distance/shape cost may not permute two names that are both available;
        # only the red-led reservation above can justify such a reassignment.
        high_blocked = {
            (row, col) for row in high_states for col in high_observations
            if int(observations[col].get("preferred_identity", 0))
            != states[row].identity}
        high_links = _solve_subset(
            high_states, high_observations, features, params, high_blocked)
        links.extend(high_links)
        matched_states.update(row for row, _, _ in high_links)
        matched_obs.update(col for _, col, _ in high_links)

    remaining_states = [row for row in range(n_states)
                        if row not in matched_states]
    remaining_observations = [col for col in range(n_obs)
                              if col not in matched_obs]
    if bool(params.get("small_cell_quarantine", False)) and high_ids:
        reclaim = _large_state_reclaims(
            states, observations, features, remaining_states,
            remaining_observations, high_ids, params)
        links.extend(reclaim)
        matched_states.update(row for row, _, _ in reclaim)
        matched_obs.update(col for _, col, _ in reclaim)
        remaining_states = [row for row in range(n_states)
                            if row not in matched_states]
        remaining_observations = [col for col in range(n_obs)
                                  if col not in matched_obs]
    blocked: set[tuple[int, int]] = set()
    if high_ids:
        for row in remaining_states:
            state = states[row]
            if state.identity not in high_ids:
                continue
            for col in remaining_observations:
                preferred = int(observations[col].get("preferred_identity", 0))
                if preferred != state.identity:
                    blocked.add((row, col))
        for col in remaining_observations:
            preferred = int(observations[col].get("preferred_identity", 0))
            if preferred not in high_ids:
                continue
            for row in remaining_states:
                if states[row].identity != preferred:
                    blocked.add((row, col))
    remaining_links = _solve_subset(
        remaining_states, remaining_observations, features, params, blocked)
    links.extend(remaining_links)
    matched_states.update(row for row, _, _ in remaining_links)
    matched_obs.update(col for _, col, _ in remaining_links)
    return links, set(range(n_states)) - matched_states, set(range(n_obs)) - matched_obs


def _subset_reservations(
        states: list[IdentityState], observations: list[dict],
        features: dict[tuple[int, int], dict], state_indices: list[int],
        observation_indices: list[int], direction: int, params: dict,
        method: str,
        ) -> list[tuple[int, int, dict]]:
    """Run a reservation method inside one confidence layer."""
    if not state_indices or not observation_indices:
        return []
    subset_states = [states[row] for row in state_indices]
    subset_observations = [observations[col] for col in observation_indices]
    subset_features = {
        (local_row, local_col): features[row, col]
        for local_row, row in enumerate(state_indices)
        for local_col, col in enumerate(observation_indices)
    }
    if method == "occupancy":
        local_links = _occupancy_reservations(
            subset_states, subset_observations, subset_features, params)
    elif method == "motion":
        local_links = _red_led_motion_reservations(
            subset_states, subset_observations, subset_features,
            direction, params)
    else:
        raise ValueError(f"unknown reservation method: {method}")
    return [
        (state_indices[row], observation_indices[col], value)
        for row, col, value in local_links]


def _large_state_reclaims(
        states: list[IdentityState], observations: list[dict],
        features: dict[tuple[int, int], dict], state_indices: list[int],
        observation_indices: list[int], large_identity_ids: set[int],
        params: dict,
        ) -> list[tuple[int, int, dict]]:
    """Give a missed large identity first refusal on nearby large territory.

    This is deliberately a one-frame, mutual-best rescue. It cannot pull a
    dormant large identity across the field, and it cannot reserve a small
    observation merely because that observation lies nearby.
    """
    minimum_observation_area = int(params.get(
        "quarantine_large_observation_minimum_area_px", 80))
    minimum_overlap = int(params.get(
        "quarantine_minimum_large_state_overlap_px", 8))
    maximum_distance = float(params.get(
        "quarantine_maximum_large_state_distance_px", 12.0))
    maximum_gap = int(params.get("quarantine_maximum_gap_frames", 1))
    minimum_ratio = float(params.get("quarantine_minimum_area_ratio", 0.4))
    maximum_ratio = float(params.get("quarantine_maximum_area_ratio", 2.5))
    candidates: list[tuple[float, int, int, int, dict]] = []
    for row in state_indices:
        state = states[row]
        if int(state.identity) not in large_identity_ids:
            continue
        for col in observation_indices:
            observation = observations[col]
            if int(observation["area"]) < minimum_observation_area:
                continue
            value = features[row, col]
            if int(value["gap_steps"]) > maximum_gap:
                continue
            ratio = float(observation["area"] / max(int(state.area), 1))
            if not minimum_ratio <= ratio <= maximum_ratio:
                continue
            overlap = int(np.count_nonzero(state.mask & observation["mask"]))
            distance = float(value["observed_displacement_px"])
            if overlap < minimum_overlap and distance > maximum_distance:
                continue
            overlap_fraction = overlap / max(
                min(int(state.area), int(observation["area"])), 1)
            score = overlap_fraction + max(
                0.0, 1.0 - distance / max(maximum_distance, 1e-6))
            updated = dict(value)
            updated.update({
                "allowed": True,
                "cost": float(min(
                    value.get("cost", np.inf),
                    float(params["miss_cost"]) + float(params["birth_cost"])
                    - min(score, 1.0))),
                "small_cell_quarantine_reclaim": True,
                "quarantine_overlap_px": overlap,
                "quarantine_overlap_fraction": overlap_fraction,
            })
            candidates.append((-score, -overlap, row, col, updated))
    if not candidates:
        return []
    best_for_state: dict[int, tuple] = {}
    best_for_observation: dict[int, tuple] = {}
    for candidate in sorted(candidates):
        _, _, row, col, _ = candidate
        best_for_state.setdefault(row, candidate)
        best_for_observation.setdefault(col, candidate)
    mutual = [candidate for candidate in best_for_state.values()
              if best_for_observation.get(candidate[3]) == candidate]
    results: list[tuple[int, int, dict]] = []
    used_states: set[int] = set()
    used_observations: set[int] = set()
    for _, _, row, col, value in sorted(mutual):
        if row in used_states or col in used_observations:
            continue
        results.append((row, col, value))
        used_states.add(row)
        used_observations.add(col)
    return results


def _assign_confidence_layers(
        states: list[IdentityState], observations: list[dict], target_t: int,
        lag: np.ndarray, direction: int, params: dict,
        ) -> tuple[list[tuple[int, int, dict]], set[int], set[int]]:
    """Assign complete confidence layers in order, with small cells last."""
    n_states, n_obs = len(states), len(observations)
    if not n_states:
        return [], set(), set(range(n_obs))
    if not n_obs:
        return [], set(range(n_states)), set()
    features = {
        (row, col): link_features(
            state, observation, target_t, lag, direction, params)
        for row, state in enumerate(states)
        for col, observation in enumerate(observations)
    }
    raw_map = params.get("confidence_layer_by_identity", {})
    layer_map = {int(identity): str(layer)
                 for identity, layer in raw_map.items()}
    order = [str(layer) for layer in params["confidence_layer_order"]]
    default_layer = str(params.get(
        "confidence_default_layer", "small_or_dim"))
    rank = {layer: index for index, layer in enumerate(order)}
    small_layer = str(params.get(
        "small_confidence_layer", "small_or_dim"))

    def identity_layer(identity: int) -> str:
        return layer_map.get(int(identity), default_layer)

    links: list[tuple[int, int, dict]] = []
    matched_states: set[int] = set()
    matched_observations: set[int] = set()
    large_ids = {identity for identity, layer in layer_map.items()
                 if layer != small_layer}

    def annotate(rows: list[tuple[int, int, dict]],
                 assignment_layer: str,
                 ) -> list[tuple[int, int, dict]]:
        annotated = []
        for row, col, value in rows:
            marked = dict(value)
            marked["confidence_assignment_layer"] = assignment_layer
            marked["confidence_identity_layer"] = identity_layer(
                states[row].identity)
            annotated.append((row, col, marked))
        return annotated

    for layer in order:
        if (layer == small_layer
                and bool(params.get("small_cell_quarantine", False))):
            remaining_states = [row for row in range(n_states)
                                if row not in matched_states]
            remaining_observations = [col for col in range(n_obs)
                                      if col not in matched_observations]
            reclaimed = _large_state_reclaims(
                states, observations, features, remaining_states,
                remaining_observations, large_ids, params)
            reclaimed = annotate(reclaimed, "large_reclaim_before_small")
            links.extend(reclaimed)
            matched_states.update(row for row, _, _ in reclaimed)
            matched_observations.update(col for _, col, _ in reclaimed)

        layer_states = [
            row for row, state in enumerate(states)
            if row not in matched_states
            and identity_layer(state.identity) == layer]
        if not layer_states:
            continue
        layer_rank = rank[layer]
        reservable_observations = []
        for col, observation in enumerate(observations):
            if col in matched_observations:
                continue
            preferred = int(observation.get("preferred_identity", 0))
            preferred_layer = identity_layer(preferred)
            if (preferred <= 0
                    or rank.get(preferred_layer, len(order)) >= layer_rank):
                reservable_observations.append(col)

        occupancy = _subset_reservations(
            states, observations, features, layer_states,
            reservable_observations, direction, params, "occupancy")
        occupancy = annotate(occupancy, layer)
        links.extend(occupancy)
        matched_states.update(row for row, _, _ in occupancy)
        matched_observations.update(col for _, col, _ in occupancy)

        layer_states = [row for row in layer_states
                        if row not in matched_states]
        reservable_observations = [col for col in reservable_observations
                                   if col not in matched_observations]
        motion = _subset_reservations(
            states, observations, features, layer_states,
            reservable_observations, direction, params, "motion")
        motion = annotate(motion, layer)
        links.extend(motion)
        matched_states.update(row for row, _, _ in motion)
        matched_observations.update(col for _, col, _ in motion)

        layer_states = [row for row in layer_states
                        if row not in matched_states]
        ordinary_observations = [
            col for col, observation in enumerate(observations)
            if col not in matched_observations
            and (int(observation.get("preferred_identity", 0)) <= 0
                 or identity_layer(int(observation.get(
                     "preferred_identity", 0))) == layer)]
        blocked = {
            (row, col) for row in layer_states
            for col in ordinary_observations
            if int(observations[col].get("preferred_identity", 0)) > 0
            and int(observations[col].get("preferred_identity", 0))
            != states[row].identity}
        ordinary = _solve_subset(
            layer_states, ordinary_observations, features, params, blocked)
        ordinary = annotate(ordinary, layer)
        links.extend(ordinary)
        matched_states.update(row for row, _, _ in ordinary)
        matched_observations.update(col for _, col, _ in ordinary)

    return (links, set(range(n_states)) - matched_states,
            set(range(n_obs)) - matched_observations)


def _assign(states: list[IdentityState], observations: list[dict], target_t: int,
            lag: np.ndarray, direction: int, params: dict,
            ) -> tuple[list[tuple[int, int, dict]], set[int], set[int]]:
    if params.get("confidence_layer_order"):
        return _assign_confidence_layers(
            states, observations, target_t, lag, direction, params)
    return _assign_legacy(
        states, observations, target_t, lag, direction, params)


def _new_observation_diagnostic(
        states: list[IdentityState], active: list[IdentityState],
        linked_identity_ids: set[int], observation: dict, target_t: int,
        lag: np.ndarray, direction: int, params: dict) -> dict:
    """Explain why the tracker paid for a new name instead of reusing an old one."""
    prefix = "diagnostic_"
    if not states:
        return {
            f"{prefix}cause": "no_previous_identity",
            f"{prefix}candidate_identity": 0,
            f"{prefix}candidate_active": False,
            f"{prefix}candidate_taken": False,
            f"{prefix}blockers": "no_previous_identity",
        }

    active_ids = {state.identity for state in active}
    evaluated: list[tuple[float, IdentityState, dict]] = []
    for state in states:
        feature = link_features(state, observation, target_t, lag, direction, params)
        # A blocked candidate still needs a finite ranking. This is the same link cost,
        # plus explicit penalties for each hard gate it failed.
        maximum_distance = float(params["max_link_px"]) * np.sqrt(
            max(feature["gap_steps"], 1))
        if feature["shared_green_tracks"]:
            maximum_distance *= float(
                params.get("shared_motion_distance_multiplier", 1.5))
        blockers = int(feature["distance_px"] > maximum_distance)
        blockers += int(feature["observed_displacement_px"] > float(
            params.get("maximum_recovery_distance_px", np.inf)))
        blockers += int(not (float(params["minimum_area_ratio"])
                            <= feature["area_ratio"]
                            <= float(params["maximum_area_ratio"])))
        blockers += int(feature["mean_log2_change"]
                        > float(params["maximum_mean_log2_change"]))
        evaluated.append((float(feature["cost"]) + 4.0 * blockers,
                          state, feature))
    _, best_state, best = min(evaluated, key=lambda row: (row[0], row[1].identity))

    maximum_distance = float(params["max_link_px"]) * np.sqrt(
        max(best["gap_steps"], 1))
    if best["shared_green_tracks"]:
        maximum_distance *= float(
            params.get("shared_motion_distance_multiplier", 1.5))
    blockers: list[str] = []
    if best["distance_px"] > maximum_distance:
        blockers.append("distance")
    if best["observed_displacement_px"] > float(
            params.get("maximum_recovery_distance_px", np.inf)):
        blockers.append("recovery_displacement")
    if not (float(params["minimum_area_ratio"]) <= best["area_ratio"]
            <= float(params["maximum_area_ratio"])):
        blockers.append("area")
    if best["mean_log2_change"] > float(params["maximum_mean_log2_change"]):
        blockers.append("brightness")

    active_candidate = best_state.identity in active_ids
    taken = best_state.identity in linked_identity_ids
    if taken:
        cause = "identity_taken_by_competing_observation"
    elif not active_candidate and best["allowed"]:
        cause = "identity_expired_before_reappearance"
    elif not active_candidate:
        cause = "expired_identity_also_failed_hard_gate"
    elif blockers:
        cause = "active_identity_failed_" + "_and_".join(blockers) + "_gate"
    elif best["cost"] >= float(params["miss_cost"]) + float(params["birth_cost"]):
        cause = "new_name_cheaper_than_persistence"
    else:
        cause = "global_assignment_competition"

    return {
        f"{prefix}cause": cause,
        f"{prefix}candidate_identity": int(best_state.identity),
        f"{prefix}candidate_active": bool(active_candidate),
        f"{prefix}candidate_taken": bool(taken),
        f"{prefix}blockers": ";".join(blockers),
        f"{prefix}candidate_from_t": int(best_state.t),
        f"{prefix}candidate_from_imagej_frame": int(best_state.t + 1),
        f"{prefix}gap_steps": int(best["gap_steps"]),
        f"{prefix}cost": float(best["cost"]),
        f"{prefix}distance_px": float(best["distance_px"]),
        f"{prefix}observed_displacement_px": float(
            best["observed_displacement_px"]),
        f"{prefix}maximum_distance_px": float(maximum_distance),
        f"{prefix}area_ratio": float(best["area_ratio"]),
        f"{prefix}mean_log2_change": float(best["mean_log2_change"]),
        f"{prefix}shared_green_tracks": int(best["shared_green_tracks"]),
        f"{prefix}motion_support": float(best["motion_support"]),
    }


def _anchor_states(observation_labels: np.ndarray, raw: np.ndarray,
                   anchor_t: int, green_lookup: dict[tuple[int, int], set[int]]
                   , anchor_identity_map: dict[int, int] | None = None,
                   preferred_lookup: dict[tuple[int, int], int] | None = None,
                   protected_lookup: dict[tuple[int, int], bool] | None = None,
                   ) -> tuple[dict[int, IdentityState], np.ndarray, int]:
    tracks = {local: value for (t, local), value in green_lookup.items() if t == anchor_t}
    preferred = {local: value for (t, local), value in
                 (preferred_lookup or {}).items() if t == anchor_t}
    protected = {local: value for (t, local), value in
                 (protected_lookup or {}).items() if t == anchor_t}
    observations = frame_observations(
        observation_labels[anchor_t], raw[anchor_t], tracks, preferred,
        protected)
    ordered = sorted(observations.values(), key=lambda row: (row["y"] if "y" in row else row["position"][0],
                                                              row["x"] if "x" in row else row["position"][1]))
    states: dict[int, IdentityState] = {}
    anchor = np.zeros(observation_labels.shape[1:], np.uint16)
    used: set[int] = set()
    next_unused = 1
    for sequential_identity, observation in enumerate(ordered, start=1):
        identity = int((anchor_identity_map or {}).get(
            int(observation["local_id"]), sequential_identity))
        if identity <= 0 or identity in used:
            raise ValueError("anchor identity map must contain unique positive names")
        used.add(identity)
        anchor[observation["mask"]] = identity
        states[identity] = IdentityState(
            identity, anchor_t, observation["mask"].copy(),
            observation["position"].copy(), observation["area"],
            observation["mean"], np.zeros(2, float),
            set(observation["green_track_ids"]), age=1,
            reference_area=float(observation["area"]),
            reference_mean=float(observation["mean"]),
            reference_shape=np.asarray(observation["shape"], float).copy(),
            bounds=tuple(observation["bounds"]))
    while next_unused in used:
        next_unused += 1
    return states, anchor, max(max(used, default=0) + 1, next_unused)


def track_direction(observation_labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                    anchor_t: int, direction: int, anchor_states: dict[int, IdentityState],
                    next_identity: int, params: dict,
                    green_lookup: dict[tuple[int, int], set[int]]
                    , preferred_lookup: dict[tuple[int, int], int] | None = None,
                    protected_lookup: dict[tuple[int, int], bool] | None = None,
                    reserved_identity_ids: set[int] | None = None,
                    ) -> tuple[np.ndarray, list[dict], int]:
    output = np.zeros(observation_labels.shape, np.uint16)
    states = {identity: IdentityState(
        row.identity, row.t, row.mask.copy(), row.position.copy(), row.area,
        row.mean, row.velocity.copy(), set(row.green_track_ids),
        age=int(row.age), reference_area=float(row.reference_area),
        reference_mean=float(row.reference_mean),
        reference_shape=(None if row.reference_shape is None
                         else np.asarray(row.reference_shape, float).copy()),
        bounds=None if row.bounds is None else tuple(row.bounds))
        for identity, row in anchor_states.items()}
    links_out: list[dict] = []
    frame_range = (range(anchor_t + 1, len(raw)) if direction > 0
                   else range(anchor_t - 1, -1, -1))
    direction_name = "forward" if direction > 0 else "backward"
    max_gap = int(params["max_gap_frames"])
    for t in frame_range:
        tracks = {local: value for (frame, local), value in green_lookup.items()
                  if frame == t}
        preferred = {local: value for (frame, local), value in
                     (preferred_lookup or {}).items() if frame == t}
        protected = {local: value for (frame, local), value in
                     (protected_lookup or {}).items() if frame == t}
        obs_dict = frame_observations(
            observation_labels[t], raw[t], tracks, preferred, protected)
        observations = [obs_dict[key] for key in sorted(obs_dict)]
        active = [row for row in states.values()
                  if abs(t - row.t) <= max_gap + 1]
        active.sort(key=lambda row: row.identity)
        links, missed, born = _assign(active, observations, t, lag, direction, params)
        prior_states = list(states.values())
        linked_identity_ids = {active[state_index].identity
                               for state_index, _, _ in links}
        for state_index, obs_index, feature in links:
            state = active[state_index]; observation = observations[obs_index]
            steps = max(abs(t - state.t), 1)
            step_velocity = (observation["position"] - state.position) / steps
            velocity = 0.65 * step_velocity + 0.35 * state.velocity
            alpha = float(params.get("lineage_reference_alpha", 0.15))
            old_reference_area = (float(state.reference_area)
                                  if np.isfinite(state.reference_area)
                                  else float(state.area))
            old_reference_mean = (float(state.reference_mean)
                                  if np.isfinite(state.reference_mean)
                                  else float(state.mean))
            old_reference_shape = (
                np.asarray(state.reference_shape, float)
                if state.reference_shape is not None
                else _shape_descriptor(state.mask))
            output[t][observation["mask"]] = state.identity
            links_out.append({
                "pass": direction_name, "kind": "link", "identity": state.identity,
                "from_t": int(state.t), "to_t": int(t),
                "from_imagej_frame": int(state.t + 1), "to_imagej_frame": int(t + 1),
                "observation_local_id": int(observation["local_id"]), **feature,
            })
            states[state.identity] = IdentityState(
                state.identity, t, observation["mask"].copy(),
                observation["position"].copy(), observation["area"],
                observation["mean"], velocity,
                set(state.green_track_ids) | set(observation["green_track_ids"]),
                age=int(state.age) + 1,
                reference_area=(1.0 - alpha) * old_reference_area
                               + alpha * float(observation["area"]),
                reference_mean=(1.0 - alpha) * old_reference_mean
                               + alpha * float(observation["mean"]),
                reference_shape=(1.0 - alpha) * old_reference_shape
                                + alpha * np.asarray(observation["shape"], float),
                bounds=tuple(observation["bounds"]))
        for state_index in missed:
            state = active[state_index]
            links_out.append({
                "pass": direction_name, "kind": "miss", "identity": state.identity,
                "from_t": int(state.t), "to_t": int(t),
                "from_imagej_frame": int(state.t + 1), "to_imagej_frame": int(t + 1),
                "observation_local_id": 0, "allowed": False,
                "cost": float(params["miss_cost"]), "gap_steps": abs(t - state.t),
            })
        for obs_index in sorted(born):
            observation = observations[obs_index]
            diagnostic = _new_observation_diagnostic(
                prior_states, active, linked_identity_ids, observation, t,
                lag, direction, params)
            requested = int(observation.get("preferred_identity", 0))
            if requested > 0 and requested not in states:
                identity = requested
            else:
                prohibited = set(states) | set(reserved_identity_ids or set())
                while next_identity in prohibited:
                    next_identity += 1
                identity = next_identity
                next_identity += 1
            output[t][observation["mask"]] = identity
            states[identity] = IdentityState(
                identity, t, observation["mask"].copy(),
                observation["position"].copy(), observation["area"],
                observation["mean"], np.zeros(2, float),
                set(observation["green_track_ids"]), age=1,
                reference_area=float(observation["area"]),
                reference_mean=float(observation["mean"]),
                reference_shape=np.asarray(observation["shape"], float).copy(),
                bounds=tuple(observation["bounds"]))
            links_out.append({
                "pass": direction_name, "kind": "new_observation",
                "identity": identity, "from_t": None, "to_t": int(t),
                "from_imagej_frame": None, "to_imagej_frame": int(t + 1),
                "observation_local_id": int(observation["local_id"]),
                "preferred_identity": requested,
                "allowed": True, "cost": float(params["birth_cost"]),
                "gap_steps": 0, **diagnostic,
            })
    return output, links_out, next_identity


def track_from_anchor(observation_labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                      anchor_t: int, params: dict,
                      observation_table: pd.DataFrame | None = None,
                      anchor_identity_map: dict[int, int] | None = None,
                      ) -> tuple[np.ndarray, pd.DataFrame]:
    green_lookup: dict[tuple[int, int], set[int]] = {}
    preferred_lookup: dict[tuple[int, int], int] = {}
    protected_lookup: dict[tuple[int, int], bool] = {}
    if observation_table is not None and len(observation_table):
        for row in observation_table.itertuples():
            text = str(row.green_track_ids)
            tracks = set() if text in ("", "nan") else {
                int(value) for value in text.split(";") if value}
            green_lookup[int(row.t), int(row.local_id)] = tracks
            if hasattr(row, "preferred_identity") and not pd.isna(
                    row.preferred_identity):
                preferred_lookup[int(row.t), int(row.local_id)] = int(
                    row.preferred_identity)
            if (hasattr(row, "preferred_identity_protected")
                    and not pd.isna(row.preferred_identity_protected)):
                protected_lookup[int(row.t), int(row.local_id)] = bool(
                    row.preferred_identity_protected)
    anchor_states, anchor_labels, next_identity = _anchor_states(
        observation_labels, raw, anchor_t, green_lookup, anchor_identity_map,
        preferred_lookup, protected_lookup)
    reserved_identity_ids = set(preferred_lookup.values()) | set(anchor_states)
    next_identity = max(next_identity, max(reserved_identity_ids, default=0) + 1)
    forward, forward_links, next_identity = track_direction(
        observation_labels, raw, lag, anchor_t, 1, anchor_states,
        next_identity, params, green_lookup, preferred_lookup,
        protected_lookup, reserved_identity_ids)
    backward, backward_links, next_identity = track_direction(
        observation_labels, raw, lag, anchor_t, -1, anchor_states,
        next_identity, params, green_lookup, preferred_lookup,
        protected_lookup, reserved_identity_ids)
    combined = forward | backward
    combined[anchor_t] = anchor_labels
    anchor_rows = [{
        "pass": "anchor", "kind": "seed", "identity": identity,
        "from_t": anchor_t, "to_t": anchor_t,
        "from_imagej_frame": anchor_t + 1, "to_imagej_frame": anchor_t + 1,
        "observation_local_id": 0, "allowed": True, "cost": 0.0,
        "gap_steps": 0,
    } for identity in sorted(anchor_states)]
    return combined, pd.DataFrame(anchor_rows + forward_links + backward_links)
