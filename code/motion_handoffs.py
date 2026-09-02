from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from persistent_merges import (_candidate_seeds, _identity_shape,
                               _partition_score, _split_interval)
from tracking import frame_observations


BIG = 1e9


@dataclass
class MotionLobe:
    component: int
    area: int
    position: np.ndarray
    shape: np.ndarray


def _shape_descriptor(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask)).astype(float)
    if len(points) < 2:
        return np.zeros(3, float)
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    scale = np.sqrt(max(len(points), 1))
    axes = np.sqrt(np.maximum(eigenvalues, 0.0) + 1e-6) / scale
    edge = mask ^ ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))
    perimeter = float(edge.sum()) / scale
    return np.array([axes[0], axes[1], perimeter], float)


def _motion_lobes(binary: np.ndarray, minimum_px: int
                  ) -> tuple[np.ndarray, list[MotionLobe]]:
    components, count = ndi.label(binary, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    rows: list[MotionLobe] = []
    objects = ndi.find_objects(components)
    for component in range(1, count + 1):
        if int(sizes[component]) < int(minimum_px):
            continue
        bounds = objects[component - 1]
        if bounds is None:
            continue
        local_mask = components[bounds] == component
        local_position = np.array(ndi.center_of_mass(local_mask), float)
        position = local_position + np.array(
            [bounds[0].start, bounds[1].start], float)
        rows.append(MotionLobe(
            component=component, area=int(sizes[component]),
            position=position, shape=_shape_descriptor(local_mask)))
    return components, rows


def _motion_pair_cost_arrays(
        lost: list[MotionLobe], gained: list[MotionLobe], params: dict,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all pair features with the scalar formula applied by array axes."""
    old_positions = np.asarray([row.position for row in lost], float)
    new_positions = np.asarray([row.position for row in gained], float)
    distances = np.linalg.norm(
        old_positions[:, None, :] - new_positions[None, :, :], axis=2)
    old_areas = np.asarray([row.area for row in lost], float)
    new_areas = np.asarray([row.area for row in gained], float)
    area_changes = np.abs(np.log2(
        (new_areas[None, :] + 1.0) / (old_areas[:, None] + 1.0)))
    old_shapes = np.asarray([row.shape for row in lost], float)
    new_shapes = np.asarray([row.shape for row in gained], float)
    shape_changes = np.mean(np.abs(np.log2(
        (new_shapes[None, :, :] + 1e-3)
        / (old_shapes[:, None, :] + 1e-3))), axis=2)

    maximum_distance = float(params["maximum_motion_pair_distance_px"])
    maximum_area = float(params["maximum_motion_area_log2_change"])
    values = (
        float(params["motion_pair_distance_weight"])
        * distances / max(maximum_distance, 1e-6)
        + float(params["motion_pair_area_weight"]) * area_changes
        + float(params["motion_pair_shape_weight"]) * shape_changes)
    allowed = ((distances <= maximum_distance)
               & (area_changes <= maximum_area))
    return np.where(allowed, values, BIG), distances, area_changes, shape_changes


def pair_blue_red_regions(lag_frame: np.ndarray, params: dict
                          ) -> tuple[np.ndarray, np.ndarray,
                                     dict[int, tuple[MotionLobe, dict]]]:
    """Globally pair every blue loss with one red gain at minimum total cost."""
    threshold = float(params["motion_lobe_log2"])
    finite = np.isfinite(lag_frame)
    lost_labels, lost = _motion_lobes(
        (lag_frame <= -threshold) & finite,
        int(params["minimum_motion_lobe_px"]))
    gained_labels, gained = _motion_lobes(
        (lag_frame >= threshold) & finite,
        int(params["minimum_motion_lobe_px"]))
    if not lost or not gained:
        return lost_labels, gained_labels, {}

    cost, distances, area_changes, shape_changes = _motion_pair_cost_arrays(
        lost, gained, params)

    old_rows, new_rows = linear_sum_assignment(cost)
    result: dict[int, tuple[MotionLobe, dict]] = {}
    maximum_cost = float(params["maximum_motion_pair_cost"])
    for old_index, new_index in zip(old_rows, new_rows):
        value = cost[old_index, new_index]
        if value >= BIG or value > maximum_cost:
            continue
        features = {
            "pair_distance_px": float(distances[old_index, new_index]),
            "pair_area_log2_change": float(
                area_changes[old_index, new_index]),
            "pair_shape_change": float(shape_changes[old_index, new_index]),
            "pair_cost": float(value),
        }
        result[lost[old_index].component] = (
            gained[new_index], features)
    return lost_labels, gained_labels, result


def build_motion_pair_cache(lag: np.ndarray, params: dict) -> list[tuple]:
    """Compute each transition's global blue-to-red pairing once per candidate."""
    return [pair_blue_red_regions(frame, params) for frame in lag]


def motion_pair_evidence_for_fixed_objects(
        labels: np.ndarray, lag: np.ndarray, params: dict,
        existing: pd.DataFrame | None = None,
        transitions: set[int] | None = None,
        pair_cache: list[tuple] | None = None) -> pd.DataFrame:
    """Attach one shared evidence name to objects joined by a blue-to-red pair.

    The tracking stage treats these names as a strong reason to reuse an established
    identity. Names are transition-specific, so an unrelated cell cannot inherit the
    evidence later merely by passing through the same location.
    """
    if len(lag) + 1 != len(labels):
        raise ValueError("labels and lag have incompatible frame counts")
    evidence: dict[tuple[int, int], set[int]] = defaultdict(set)
    if existing is not None and len(existing):
        for row in existing.itertuples():
            text = str(row.green_track_ids)
            if text not in ("", "nan"):
                evidence[int(row.t), int(row.local_id)].update(
                    int(value) for value in text.split(";") if value)

    next_track = int(params.get("motion_track_id_offset", 1_000_000))
    minimum_px = int(params["minimum_motion_identity_px"])
    minimum_coverage = float(params["minimum_motion_object_coverage"])
    for t in range(len(lag)):
        if transitions is not None and t not in transitions:
            continue
        lost_labels, gained_labels, pairs = (
            pair_cache[t] if pair_cache is not None
            else pair_blue_red_regions(lag[t], params))
        for lost_component, (destination, _) in sorted(pairs.items()):
            loss_mask = lost_labels == int(lost_component)
            gain_mask = gained_labels == int(destination.component)
            source: list[int] = []
            destination_ids: list[int] = []
            for identity in set(map(int, np.unique(labels[t][loss_mask]))) - {0}:
                mask = labels[t] == identity
                overlap = int(np.count_nonzero(mask & loss_mask))
                if (overlap >= minimum_px
                        and overlap / max(int(mask.sum()), 1) >= minimum_coverage):
                    source.append(identity)
            for identity in set(map(int, np.unique(labels[t + 1][gain_mask]))) - {0}:
                mask = labels[t + 1] == identity
                overlap = int(np.count_nonzero(mask & gain_mask))
                if (overlap >= minimum_px
                        and overlap / max(int(mask.sum()), 1) >= minimum_coverage):
                    destination_ids.append(identity)
            if not source or not destination_ids:
                continue
            for identity in source:
                evidence[t, identity].add(next_track)
            for identity in destination_ids:
                evidence[t + 1, identity].add(next_track)
            next_track += 1

    rows: list[dict] = []
    for t in range(len(labels)):
        for identity in sorted(set(map(int, np.unique(labels[t]))) - {0}):
            rows.append({
                "t": int(t), "local_id": int(identity),
                "green_track_ids": ";".join(
                    map(str, sorted(evidence.get((t, identity), set())))),
            })
    return pd.DataFrame(rows)


def _destination_host(destination: np.ndarray, current: dict[int, dict],
                      continuing: set[int], params: dict
                      ) -> tuple[int, dict] | None:
    area = max(int(destination.sum()), 1)
    overlaps = sorted((
        (int(np.count_nonzero(destination & current[identity]["mask"])) / area,
         identity)
        for identity in continuing
    ), reverse=True)
    if not overlaps:
        return None
    best_fraction, host = overlaps[0]
    second_fraction = overlaps[1][0] if len(overlaps) > 1 else 0.0
    margin = best_fraction - second_fraction
    if (best_fraction < float(params["minimum_gain_host_overlap"])
            or margin < float(params["minimum_gain_host_margin"])):
        return None
    return int(host), {
        "gain_host_overlap": float(best_fraction),
        "gain_host_second_overlap": float(second_fraction),
        "gain_host_margin": float(margin),
    }


def _stable_bookend(labels: np.ndarray, start_t: int, identity: int,
                    host: int, old_area: int, params: dict) -> int | None:
    """Find proof that both cells remain separate, not a one-frame fragment."""
    maximum = min(len(labels), start_t + int(params["maximum_bookend_frames"]) + 1)
    stable_frames = int(params["minimum_stable_bookend_frames"])
    minimum_ratio = float(params["minimum_bookend_area_ratio"])
    maximum_ratio = float(params["maximum_bookend_area_ratio"])
    structure = np.ones((3, 3), np.uint8)
    for future_t in range(start_t + 1, maximum):
        present = set(map(int, np.unique(labels[future_t]))) - {0}
        if host not in present:
            return None
        if identity not in present or future_t + stable_frames > len(labels):
            continue
        valid = True
        for frame in range(future_t, future_t + stable_frames):
            frame_present = set(map(int, np.unique(labels[frame]))) - {0}
            if identity not in frame_present or host not in frame_present:
                valid = False
                break
            target = labels[frame] == identity
            ratio = int(target.sum()) / max(int(old_area), 1)
            if (ndi.label(target, structure=structure)[1] != 1
                    or not minimum_ratio <= ratio <= maximum_ratio):
                valid = False
                break
        if valid:
            return future_t
    return None


def carry_identities_through_motion_handoffs(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray, params: dict,
        pair_cache: list[tuple] | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Recover fast cells whose blue-to-red jump lands inside another identity.

    A repair requires the full chain of custody: the old identity overlaps a blue
    loss, global one-to-one matching assigns that loss a red gain, the gain is owned
    by a continuing identity, and both identities are visible separately at a later
    bookend. The merged foreground is partitioned without adding or deleting pixels.
    """
    if len(raw) != len(labels) or len(lag) + 1 != len(labels):
        raise ValueError("labels, raw and lag have incompatible frame counts")
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    events: list[dict] = []
    audit: list[dict] = []
    frame_rows: list[dict] = []

    for t in range(1, len(fixed)):
        previous = frame_observations(fixed[t - 1], raw[t - 1])
        current = frame_observations(fixed[t], raw[t])
        vanished = sorted(set(previous) - set(current))
        continuing = set(previous) & set(current)
        if not vanished:
            continue
        lost_labels, gained_labels, pairs = (
            pair_cache[t - 1] if pair_cache is not None
            else pair_blue_red_regions(lag[t - 1], params))
        used_hosts: set[int] = set()
        for identity in vanished:
            row = {
                "identity": int(identity),
                "last_t": int(t - 1),
                "last_imagej_frame": int(t),
                "missing_t": int(t),
                "missing_imagej_frame": int(t + 1),
                "old_area_px": int(previous[identity]["area"]),
                "decision": "unclassified",
            }
            old_mask = previous[identity]["mask"]
            y, x = np.nonzero(old_mask)
            border = bool(np.any(
                (y < int(params["border_px"]))
                | (y >= labels.shape[1] - int(params["border_px"]))
                | (x < int(params["border_px"]))
                | (x >= labels.shape[2] - int(params["border_px"]))))
            row["old_touches_border"] = border

            history_start = max(0, t - int(params["minimum_pre_event_frames"]))
            history_complete = all(
                np.any(fixed[frame] == identity)
                for frame in range(history_start, t))
            row["pre_event_history_frames"] = int(t - history_start)
            if not history_complete:
                row["decision"] = "insufficient_pre_event_history"
                audit.append(row)
                continue

            overlap_counts = np.bincount(
                lost_labels[old_mask].ravel(),
                minlength=int(lost_labels.max()) + 1)
            if len(overlap_counts):
                overlap_counts[0] = 0
            lost_component = (int(np.argmax(overlap_counts))
                              if len(overlap_counts) > 1 else 0)
            lost_overlap = (float(overlap_counts[lost_component]
                                  / max(previous[identity]["area"], 1))
                            if lost_component else 0.0)
            row.update({
                "lost_component": lost_component,
                "lost_identity_overlap": lost_overlap,
            })
            if lost_component == 0:
                row["decision"] = "no_blue_loss_region"
                audit.append(row)
                continue
            if lost_overlap < float(params["minimum_lost_identity_overlap"]):
                row["decision"] = "blue_overlap_below_gate"
                audit.append(row)
                continue
            if lost_component not in pairs:
                row["decision"] = "no_global_red_match"
                audit.append(row)
                continue

            destination, pair_features = pairs[lost_component]
            destination_mask = gained_labels == destination.component
            row.update({
                "gain_component": int(destination.component),
                "gain_area_px": int(destination.area),
                **pair_features,
            })
            host_result = _destination_host(
                destination_mask, current, continuing, params)
            if host_result is None:
                row["decision"] = "red_destination_not_owned_unambiguously"
                audit.append(row)
                continue
            host, host_features = host_result
            row.update({"host_identity": host, **host_features})
            if host in used_hosts:
                row["decision"] = "destination_host_already_partitioned"
                audit.append(row)
                continue

            future_t = _stable_bookend(
                fixed, t, identity, host, previous[identity]["area"], params)
            if future_t is None:
                row["decision"] = "no_stable_separated_bookend"
                audit.append(row)
                continue
            row.update({
                "bookend_t": int(future_t),
                "bookend_imagej_frame": int(future_t + 1),
                "missing_frames": int(future_t - t),
            })

            partition = _split_interval(
                fixed, raw, lag, t, future_t, identity, host, params,
                destination_masks={t: destination_mask})
            if partition is None:
                row["decision"] = "bookended_partition_failed"
                audit.append(row)
                continue
            candidate, partition_rows = partition
            original_support = fixed > 0
            fixed = candidate
            if not np.array_equal(fixed > 0, original_support):
                raise AssertionError(
                    "motion handoff changed foreground support")
            for partition_row in partition_rows:
                frame = int(partition_row["t"])
                inferred[frame][(fixed[frame] == identity)
                                | (fixed[frame] == host)] = True
            event_id = f"MH{len(events) + 1:04d}"
            event = {
                "event_id": event_id,
                "kind": "fast_motion_identity_handoff",
                "status": "resolved_inferred",
                "identity": int(identity),
                "host_identity": int(host),
                "start_t": int(t),
                "start_imagej_frame": int(t + 1),
                "end_t": int(future_t - 1),
                "end_imagej_frame": int(future_t),
                "bookend_t": int(future_t),
                "bookend_imagej_frame": int(future_t + 1),
                "partitioned_frames": int(future_t - t),
                "lost_identity_overlap": lost_overlap,
                **pair_features,
                **host_features,
                "mean_motion_support": float(np.mean([
                    part["a_motion_support"] for part in partition_rows])),
                "mean_destination_margin": float(np.mean([
                    part["destination_margin"] for part in partition_rows])),
                "reason": "blue loss globally matched to a red destination owned by "
                          "another identity; both identities later separate",
            }
            events.append(event)
            for partition_row in partition_rows:
                frame_rows.append({
                    "event_id": event_id,
                    "identity": int(identity),
                    "host_identity": int(host),
                    **partition_row,
                })
            row.update({
                "decision": "resolved_inferred",
                "event_id": event_id,
            })
            audit.append(row)
            used_hosts.add(host)
            current = frame_observations(fixed[t], raw[t])
            continuing = set(previous) & set(current)

    event_table = pd.DataFrame(events)
    event_table.attrs["frame_rows"] = pd.DataFrame(frame_rows)
    return fixed, event_table, inferred, pd.DataFrame(audit)


def _segment_age(labels: np.ndarray, identity: int, t: int) -> int:
    """Number of consecutive frames this name has occupied up to ``t``."""
    age = 0
    for frame in range(int(t), -1, -1):
        if not np.any(labels[frame] == int(identity)):
            break
        age += 1
    return age


def _recent_identity_state(labels: np.ndarray, raw: np.ndarray, identity: int,
                           t: int, frames: int) -> dict | None:
    observations: list[dict] = []
    for frame in range(max(0, int(t) - int(frames) + 1), int(t) + 1):
        row = frame_observations(labels[frame], raw[frame]).get(int(identity))
        if row is not None:
            observations.append(row)
    if not observations:
        return None
    positions = [row["position"] for row in observations]
    velocity = (positions[-1] - positions[-2]
                if len(positions) >= 2 else np.zeros(2, float))
    return {
        "position": positions[-1],
        "predicted": positions[-1] + velocity,
        "area": float(np.median([row["area"] for row in observations])),
        "mean": float(np.median([row["mean"] for row in observations])),
        "shape": _shape_descriptor(observations[-1]["mask"]),
    }


def _neighbour_fallback(frame: np.ndarray, mask: np.ndarray,
                        forbidden: set[int]) -> int | None:
    """Return the identity sharing the longest boundary with ``mask``."""
    ring = ndi.binary_dilation(mask, structure=np.ones((3, 3), bool)) & ~mask
    values, counts = np.unique(frame[ring], return_counts=True)
    choices = sorted(
        ((int(count), int(value)) for value, count in zip(values, counts)
         if int(value) > 0 and int(value) not in forbidden),
        reverse=True)
    return choices[0][1] if choices else None


def seed_first_handoff_identities(
        labels: np.ndarray, guide_labels: np.ndarray, handoffs: pd.DataFrame,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Restore an established name on the first resolved post-jump outline.

    Long-memory retracking may keep the right geometry but give its first inferred
    outline a new name. The pre-event frame identifies the established name; the
    handoff guide identifies the correct outline one frame later. Only that first
    frame is seeded, then blue-to-red trajectory evidence controls later frames.
    """
    if labels.shape != guide_labels.shape:
        raise ValueError("labels and handoff guide have incompatible shapes")
    fixed = labels.copy()
    rows: list[dict] = []
    if handoffs.empty:
        return fixed, pd.DataFrame()
    for handoff in handoffs.sort_values("start_t").itertuples():
        t = int(handoff.start_t)
        guide_identity = int(handoff.identity)
        if t <= 0:
            continue
        before_mask = guide_labels[t - 1] == guide_identity
        first_mask = guide_labels[t] == guide_identity
        if not before_mask.any() or not first_mask.any():
            continue
        before_values, before_counts = np.unique(
            fixed[t - 1][before_mask], return_counts=True)
        before_choices = [(int(count), int(value))
                          for value, count in zip(before_values, before_counts)
                          if int(value) > 0]
        first_values, first_counts = np.unique(
            fixed[t][first_mask], return_counts=True)
        first_choices = [(int(count), int(value))
                         for value, count in zip(first_values, first_counts)
                         if int(value) > 0]
        if not before_choices or not first_choices:
            continue
        target = max(before_choices)[1]
        source = max(first_choices)[1]
        if source == target:
            continue
        source_mask = fixed[t] == source
        target_mask = fixed[t] == target
        fixed[t][target_mask] = source
        fixed[t][source_mask] = target
        rows.append({
            "event_id": str(handoff.event_id),
            "t": t, "imagej_frame": t + 1,
            "guide_identity": guide_identity,
            "persistent_identity": target,
            "displaced_identity": source,
            "persistent_pixels": int(source_mask.sum()),
            "displaced_pixels": int(target_mask.sum()),
        })
    if not np.array_equal(fixed > 0, labels > 0):
        raise AssertionError("handoff identity seeding changed foreground support")
    return fixed, pd.DataFrame(rows)


def build_motion_reservation_requests(
        seeded_labels: np.ndarray, seed_events: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build zero or more host-carry requests from field-discovered seed events.

    Each seed describes an established identity temporarily travelling under another
    name. The established name becomes reservable when that temporary alias is no
    longer present. Seeds whose alias never ends are rejected with an audit reason;
    they do not abort unrelated reservations.
    """
    request_columns = [
        "event_id", "kind", "identity", "displaced_identity", "to_t"]
    audit_columns = [
        "event_id", "identity", "displaced_identity", "start_search_t",
        "to_t", "decision", "reason"]
    if seed_events.empty:
        return (pd.DataFrame(columns=request_columns),
                pd.DataFrame(columns=audit_columns))

    required = {"event_id", "t", "persistent_identity", "displaced_identity"}
    missing = required - set(seed_events.columns)
    if missing:
        raise ValueError(f"motion seed events missing columns: {sorted(missing)}")

    requests: list[dict] = []
    audit: list[dict] = []
    ordered = seed_events.sort_values(["t", "event_id"], kind="stable")
    for seed in ordered.itertuples(index=False):
        target = int(seed.persistent_identity)
        alias = int(seed.displaced_identity)
        start_t = int(seed.t) + 1
        while (start_t < len(seeded_labels)
               and np.any(seeded_labels[start_t] == alias)):
            start_t += 1
        common = {
            "event_id": str(seed.event_id),
            "identity": target,
            "displaced_identity": alias,
            "start_search_t": int(seed.t) + 1,
            "to_t": start_t if start_t < len(seeded_labels) else pd.NA,
        }
        if start_t >= len(seeded_labels):
            audit.append({**common, "decision": "rejected",
                          "reason": "displaced_identity_never_ended"})
            continue
        requests.append({
            "event_id": str(seed.event_id),
            "kind": "motion_alias_ended_target_reserved",
            "identity": target,
            "displaced_identity": alias,
            "to_t": start_t,
        })
        audit.append({**common, "decision": "accepted",
                      "reason": "displaced_identity_ended"})
    return (pd.DataFrame(requests, columns=request_columns),
            pd.DataFrame(audit, columns=audit_columns))


def carry_established_identities_along_motion(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray, params: dict,
        pair_cache: list[tuple] | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Keep an established identity on its complete blue-to-red trajectory.

    This is the identity-swap counterpart to ``carry_identities_through_motion_handoffs``.
    It handles a moving cell that is still present under the wrong, newly created name.
    A complete blue loss and its globally paired red gain must agree with size, shape,
    brightness and the recent direction of travel. A smaller nearby remnant is never
    allowed to keep the established name merely because it is closer.
    """
    if len(raw) != len(labels) or len(lag) + 1 != len(labels):
        raise ValueError("labels, raw and lag have incompatible frame counts")
    fixed = labels.copy()
    events: list[dict] = []
    audit: list[dict] = []
    displaced_names: dict[int, int] = {}
    active_aliases: dict[int, tuple[int, int]] = {}
    reserved_names: dict[int, int] = {}
    reservation_starts: dict[int, int] = {}
    recent_frames = int(params["identity_reference_frames"])

    for t in range(len(fixed) - 1):
        for target, alias in reserved_names.items():
            target_mask = fixed[t] == target
            if target_mask.any():
                fixed[t][target_mask] = alias
        # Once a red destination has been proved, keep its contiguous object segment
        # under the established name. Otherwise the next quiet frame would undo the
        # correction simply because it contains no complete blue/red handoff.
        for target, (alias, first_t) in list(active_aliases.items()):
            if t <= first_t:
                continue
            alias_mask = fixed[t] == alias
            if not alias_mask.any():
                target_mask = fixed[t] == target
                if target_mask.any():
                    fixed[t][target_mask] = alias
                    events.append({
                        "event_id": f"MT{len(events) + 1:04d}",
                        "kind": "motion_alias_ended_target_reserved",
                        "identity": int(target),
                        "from_t": int(t - 1), "from_imagej_frame": int(t),
                        "to_t": int(t), "to_imagej_frame": int(t + 1),
                        "displaced_identity": int(alias),
                        "wrong_target_fallback": int(alias),
                        "source_fragment_identities": "",
                        "source_fragment_pixels": 0,
                    })
                reserved_names[target] = alias
                reservation_starts[target] = int(t)
                del active_aliases[target]
                continue
            target_mask = fixed[t] == target
            fallback = 0
            if target_mask.any():
                fallback_value = _neighbour_fallback(
                    fixed[t], target_mask, {target, alias})
                if fallback_value is None:
                    fallback_value = displaced_names.get(target, alias)
                fallback = int(fallback_value)
                fixed[t][target_mask] = fallback
            fixed[t][alias_mask] = target
            events.append({
                "event_id": f"MT{len(events) + 1:04d}",
                "kind": "motion_alias_continuation",
                "identity": int(target),
                "from_t": int(t - 1), "from_imagej_frame": int(t),
                "to_t": int(t), "to_imagej_frame": int(t + 1),
                "displaced_identity": int(alias),
                "wrong_target_fallback": fallback,
                "source_fragment_identities": "",
                "source_fragment_pixels": 0,
            })
        lost_labels, gained_labels, pairs = (
            pair_cache[t] if pair_cache is not None
            else pair_blue_red_regions(lag[t], params))
        if not pairs:
            continue
        current = frame_observations(fixed[t], raw[t])
        following = frame_observations(fixed[t + 1], raw[t + 1])
        used_targets: set[int] = set()
        used_destinations: set[int] = set()

        for lost_component, (destination, pair_features) in sorted(pairs.items()):
            loss_mask = lost_labels == int(lost_component)
            gain_mask = gained_labels == int(destination.component)
            loss_values, loss_counts = np.unique(fixed[t][loss_mask], return_counts=True)
            source_rows: list[dict] = []
            for value, count in zip(loss_values, loss_counts):
                identity = int(value)
                if identity <= 0 or identity not in current:
                    continue
                coverage = int(count) / max(int(current[identity]["area"]), 1)
                if (int(count) >= int(params["minimum_motion_identity_px"])
                        and coverage >= float(params["minimum_source_loss_coverage"])):
                    source_rows.append({
                        "identity": identity,
                        "loss_px": int(count),
                        "coverage": float(coverage),
                        "age": _segment_age(fixed, identity, t),
                    })
            established = [row for row in source_rows
                           if row["age"] >= int(params["minimum_established_frames"])]
            if not established:
                continue
            target_row = max(established, key=lambda row: (row["age"], row["loss_px"],
                                                            -row["identity"]))
            target = int(target_row["identity"])
            if target in used_targets:
                continue
            state = _recent_identity_state(fixed, raw, target, t, recent_frames)
            if state is None:
                continue

            fragment_rows = [
                row for row in source_rows
                if row["identity"] != target
                and row["age"] <= int(params["maximum_fragment_age_frames"])
            ]
            source_ids = {target} | {int(row["identity"]) for row in fragment_rows}
            source_mask = np.isin(fixed[t], list(source_ids)) & loss_mask
            source_area = int(source_mask.sum())
            source_ratio = source_area / max(float(state["area"]), 1.0)
            if not (float(params["minimum_source_area_ratio"]) <= source_ratio
                    <= float(params["maximum_source_area_ratio"])):
                continue

            gain_values, gain_counts = np.unique(fixed[t + 1][gain_mask],
                                                  return_counts=True)
            destinations: list[dict] = []
            for value, count in zip(gain_values, gain_counts):
                identity = int(value)
                if (identity <= 0 or identity == target
                        or identity in used_destinations or identity not in following):
                    continue
                coverage = int(count) / max(int(following[identity]["area"]), 1)
                if (int(count) < int(params["minimum_motion_identity_px"])
                        or coverage < float(params["minimum_destination_gain_coverage"])):
                    continue
                age = _segment_age(fixed, identity, t + 1)
                if age > int(params["maximum_destination_age_frames"]):
                    continue
                observation = following[identity]
                predicted_distance = float(np.linalg.norm(
                    state["predicted"] - observation["position"]))
                area_change = abs(float(np.log2(
                    observation["area"] / max(source_area, 1))))
                mean_change = abs(float(np.log2(
                    (observation["mean"] + 1.0) / (state["mean"] + 1.0))))
                shape_change = float(np.mean(np.abs(np.log2(
                    (_shape_descriptor(observation["mask"]) + 1e-3)
                    / (state["shape"] + 1e-3)))))
                cost = (
                    predicted_distance / max(float(params["trajectory_scale_px"]), 1e-6)
                    + float(params["trajectory_area_weight"]) * area_change
                    + float(params["trajectory_mean_weight"]) * mean_change
                    + float(params["trajectory_shape_weight"]) * shape_change
                )
                destinations.append({
                    "identity": identity, "coverage": float(coverage),
                    "age": age, "predicted_distance_px": predicted_distance,
                    "area_change": area_change, "mean_change": mean_change,
                    "shape_change": shape_change, "cost": cost,
                })
            if not destinations:
                continue
            destinations.sort(key=lambda row: (row["cost"], row["identity"]))
            best = destinations[0]
            second_cost = destinations[1]["cost"] if len(destinations) > 1 else np.inf
            if (best["cost"] > float(params["maximum_trajectory_cost"])
                    or second_cost - best["cost"]
                    < float(params["minimum_trajectory_margin"])):
                continue

            destination_identity = int(best["identity"])
            assigned = following.get(target)
            assigned_gain_coverage = 0.0
            assigned_distance = np.inf
            if assigned is not None:
                assigned_gain_coverage = float(np.mean(gain_mask[assigned["mask"]]))
                assigned_distance = float(np.linalg.norm(
                    state["predicted"] - assigned["position"]))
                if (assigned_gain_coverage
                        >= float(params["maximum_assigned_gain_coverage"])):
                    continue
                if (assigned_distance - float(best["predicted_distance_px"])
                        < float(params["minimum_trajectory_improvement_px"])):
                    continue

            original_support = fixed > 0
            fragment_pixels = 0
            for fragment in fragment_rows:
                fragment_identity = int(fragment["identity"])
                mask = (fixed[t] == fragment_identity) & loss_mask
                fragment_pixels += int(mask.sum())
                fixed[t][mask] = target

            destination_mask = fixed[t + 1] == destination_identity
            wrong_target_mask = fixed[t + 1] == target
            fallback = None
            if wrong_target_mask.any():
                fallback = _neighbour_fallback(
                    fixed[t + 1], wrong_target_mask,
                    {target, destination_identity})
                if fallback is None:
                    fallback = displaced_names.get(target, destination_identity)
                fixed[t + 1][wrong_target_mask] = int(fallback)
            fixed[t + 1][destination_mask] = target
            displaced_names.setdefault(target, destination_identity)
            active_aliases[target] = (destination_identity, int(t + 1))
            if not np.array_equal(fixed > 0, original_support):
                raise AssertionError("motion trajectory carry changed foreground support")

            event_id = f"MT{len(events) + 1:04d}"
            row = {
                "event_id": event_id,
                "kind": "established_identity_motion_reassignment",
                "identity": target,
                "from_t": int(t), "from_imagej_frame": int(t + 1),
                "to_t": int(t + 1), "to_imagej_frame": int(t + 2),
                "displaced_identity": destination_identity,
                "wrong_target_fallback": int(fallback) if fallback is not None else 0,
                "source_fragment_identities": ";".join(
                    map(str, sorted(source_ids - {target}))),
                "source_fragment_pixels": int(fragment_pixels),
                "source_area_px": source_area,
                "destination_area_px": int(destination_mask.sum()),
                "source_area_ratio": float(source_ratio),
                "destination_gain_coverage": float(best["coverage"]),
                "assigned_gain_coverage": float(assigned_gain_coverage),
                "destination_predicted_distance_px": float(
                    best["predicted_distance_px"]),
                "assigned_predicted_distance_px": float(assigned_distance),
                "trajectory_improvement_px": float(
                    assigned_distance - best["predicted_distance_px"]),
                "trajectory_cost": float(best["cost"]),
                **pair_features,
            }
            events.append(row)
            audit.append({**row, "decision": "reassigned_to_red_destination"})
            used_targets.add(target)
            used_destinations.add(destination_identity)
            current = frame_observations(fixed[t], raw[t])
            following = frame_observations(fixed[t + 1], raw[t + 1])

    for target, alias in reserved_names.items():
        for frame in range(reservation_starts[target], len(fixed)):
            fixed[frame][fixed[frame] == target] = alias
    return fixed, pd.DataFrame(events), pd.DataFrame(audit)


def carry_reserved_identities_through_motion_hosts(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
        trajectory_events: pd.DataFrame, params: dict,
        pair_cache: list[tuple] | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Partition a proven host after a reserved moving identity enters it.

    Once a complete blue-to-red jump ends inside a continuing host, the established
    identity is not allowed to vanish or return to its old remnant. Its previous size,
    shape and position are carried through the host on every later frame while the host
    keeps the complementary connected body. Foreground support is never changed.
    """
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    rows: list[dict] = []
    if trajectory_events.empty:
        return fixed, pd.DataFrame(), inferred
    reservations = trajectory_events[
        trajectory_events.kind == "motion_alias_ended_target_reserved"]
    for reservation in reservations.itertuples():
        target = int(reservation.identity)
        start_t = int(reservation.to_t)
        if start_t <= 0 or start_t >= len(fixed):
            continue
        previous = frame_observations(fixed[start_t - 1], raw[start_t - 1])
        current = frame_observations(fixed[start_t], raw[start_t])
        if target not in previous:
            continue
        lost_labels, gained_labels, pairs = (
            pair_cache[start_t - 1] if pair_cache is not None
            else pair_blue_red_regions(lag[start_t - 1], params))
        old_mask = previous[target]["mask"]
        counts = np.bincount(
            lost_labels[old_mask].ravel(), minlength=int(lost_labels.max()) + 1)
        if len(counts):
            counts[0] = 0
        component = int(np.argmax(counts)) if len(counts) > 1 else 0
        overlap = (float(counts[component] / max(previous[target]["area"], 1))
                   if component else 0.0)
        if (component not in pairs
                or overlap < float(params["minimum_source_loss_coverage"])):
            continue
        destination, pair_features = pairs[component]
        destination_mask = gained_labels == int(destination.component)
        continuing = set(previous) & set(current) - {target}
        host_result = _destination_host(
            destination_mask, current, continuing, params)
        if host_result is None:
            continue
        host, host_features = host_result
        if host not in previous:
            continue

        state_target = previous[target]
        state_host = previous[host]
        velocity_target = np.zeros(2, float)
        velocity_host = np.zeros(2, float)
        for t in range(start_t, len(fixed)):
            observations = frame_observations(fixed[t], raw[t])
            if host not in observations:
                break
            if target in observations:
                state_target = observations[target]
                state_host = observations[host]
                continue
            merged = observations[host]["mask"]
            predicted_target = state_target["position"] + velocity_target
            predicted_host = state_host["position"] + velocity_host
            seeds = _candidate_seeds(
                merged, raw[t], predicted_target, predicted_host, params)
            candidates: list[dict] = []
            for seed_target in seeds:
                for seed_host in seeds:
                    if seed_target == seed_host:
                        continue
                    scored = _partition_score(
                        merged, raw[t], state_target, state_host,
                        predicted_target, predicted_host,
                        state_target["area"], state_host["area"],
                        _identity_shape(state_target["mask"]),
                        _identity_shape(state_host["mask"]), lag[t - 1],
                        seed_target, seed_host, params,
                        destination_mask if t == start_t else None)
                    if scored is not None:
                        scored["partition_method"] = "intensity_watershed"
                        candidates.append(scored)
            if not candidates:
                # A genuinely merged rounded signal can have no intensity valley at
                # all. On a flat surface, watershed becomes a shortest-path division
                # from the predicted target and host positions. The usual connected,
                # size, shape, position and motion terms still decide whether it is
                # acceptable; only the missing intensity valley is waived.
                flat = np.zeros_like(raw[t])
                for seed_target in seeds:
                    for seed_host in seeds:
                        if seed_target == seed_host:
                            continue
                        scored = _partition_score(
                            merged, flat, state_target, state_host,
                            predicted_target, predicted_host,
                            state_target["area"], state_host["area"],
                            _identity_shape(state_target["mask"]),
                            _identity_shape(state_host["mask"]), lag[t - 1],
                            seed_target, seed_host, params,
                            destination_mask if t == start_t else None)
                        if scored is not None:
                            scored["partition_method"] = "geodesic_fallback"
                            candidates.append(scored)
            if not candidates:
                break
            best = min(candidates, key=lambda row: row["cost"])
            original_support = fixed[t] > 0
            fixed[t][merged] = 0
            fixed[t][best["part_a"]] = target
            fixed[t][best["part_b"]] = host
            if not np.array_equal(fixed[t] > 0, original_support):
                raise AssertionError("reserved motion carry changed foreground support")
            inferred[t][merged] = True
            updated = frame_observations(fixed[t], raw[t])
            velocity_target = (0.65 * (updated[target]["position"]
                                       - state_target["position"])
                               + 0.35 * velocity_target)
            velocity_host = (0.65 * (updated[host]["position"]
                                     - state_host["position"])
                             + 0.35 * velocity_host)
            state_target, state_host = updated[target], updated[host]
            rows.append({
                "event_id": f"MR{len(rows) + 1:04d}",
                "identity": target, "host_identity": host,
                "t": t, "imagej_frame": t + 1,
                "target_area_px": int(best["area_a_px"]),
                "host_area_px": int(best["area_b_px"]),
                "target_motion_support": float(best["a_motion_support"]),
                "host_motion_support": float(best["b_motion_support"]),
                "body_separation_support": float(best["body_separation_support"]),
                "partition_method": str(best["partition_method"]),
                "cost": float(best["cost"]),
                "lost_identity_overlap": overlap,
                **pair_features, **host_features,
            })
    return fixed, pd.DataFrame(rows), inferred
