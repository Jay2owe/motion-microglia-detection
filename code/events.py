from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from tracking import frame_observations, motion_support


def _mask_distance(source: dict, target: dict) -> float:
    pixels = np.column_stack(np.nonzero(target["mask"]))
    return float(cKDTree(pixels).query(source["position"], k=1)[0])


def _delayed_features(old: dict, new: dict, lag: np.ndarray,
                      maximum_distance: float) -> dict:
    distance = float(np.linalg.norm(old["position"] - new["position"]))
    area_ratio = float(new["area"] / max(old["area"], 1))
    area_change = abs(float(np.log2(max(area_ratio, 1e-12))))
    mean_change = abs(float(np.log2((new["mean"] + 1.0) / (old["mean"] + 1.0))))
    if distance > maximum_distance:
        return {
            "cost": float(1e6 + distance), "distance_px": distance,
            "area_ratio": area_ratio, "mean_log2_change": mean_change,
            "motion_support": 0.0, "supportive_px": 0,
            "contradictory_px": 0, "union_px": 0,
        }
    support = motion_support(old["mask"], new["mask"], lag)
    cost = (distance / max(maximum_distance, 1e-6) + 0.4 * area_change
            + 0.2 * mean_change - 1.5 * support["motion_support"])
    return {
        "cost": float(cost), "distance_px": distance,
        "area_ratio": area_ratio, "mean_log2_change": mean_change, **support,
    }


def _bookend_assignment(before: list[dict], future: list[dict],
                        old_raw: np.ndarray, new_raw: np.ndarray,
                        horizon: int, minimum_margin: float,
                        minimum_separation: float) -> dict | None:
    if len(before) != 2 or len(future) < 2:
        return None
    maximum_distance = 18.0 * np.sqrt(max(horizon, 1))
    future = [row for row in future
              if min(float(np.linalg.norm(row["position"] - old["position"]))
                     for old in before) <= maximum_distance]
    if len(future) < 2:
        return None
    lag = np.log2((new_raw.astype(np.float32) + 1.0)
                  / (old_raw.astype(np.float32) + 1.0))
    features = [[_delayed_features(old, new, lag, maximum_distance)
                 for new in future] for old in before]
    candidates: list[dict] = []
    for left, right in itertools.permutations(range(len(future)), 2):
        if left == right:
            continue
        a, b = future[left], future[right]
        separation = float(np.linalg.norm(a["position"] - b["position"]))
        if separation < float(minimum_separation):
            continue
        fa = features[0][left]
        fb = features[1][right]
        if max(fa["distance_px"], fb["distance_px"]) > maximum_distance:
            continue
        candidates.append({
            "cost": float(fa["cost"] + fb["cost"]),
            "future_a": a, "future_b": b,
            "a_motion_support": fa["motion_support"],
            "b_motion_support": fb["motion_support"],
            "a_distance_px": fa["distance_px"],
            "b_distance_px": fb["distance_px"],
            "separation_px": separation,
        })
    if not candidates:
        return None
    candidates.sort(key=lambda row: row["cost"])
    best = candidates[0]
    second_cost = candidates[1]["cost"] if len(candidates) > 1 else np.inf
    margin = float(second_cost - best["cost"])
    if margin < float(minimum_margin):
        return None
    return {**best, "margin": margin}


def _find_bookends(labels: np.ndarray, raw: np.ndarray,
                   observation_cache: list[dict[int, dict]], before: list[dict],
                   start_t: int, params: dict) -> list[dict]:
    results: list[dict] = []
    pre_t = start_t - 1
    for horizon in params["lag_horizons"]:
        future_t = pre_t + int(horizon)
        if future_t <= start_t or future_t >= len(labels):
            continue
        future = list(observation_cache[future_t].values())
        assignment = _bookend_assignment(
            before, future, raw[pre_t], raw[future_t], int(horizon),
            params["minimum_evidence_margin"],
            params["minimum_bookend_separation_px"])
        if assignment is not None:
            results.append({"horizon": int(horizon), "future_t": int(future_t),
                            **assignment})
    return results


def _stable_bookend(bookends: list[dict]) -> tuple[dict | None, bool]:
    if not bookends:
        return None, False
    first = bookends[0]
    first_pair = (int(first["future_a"]["local_id"]),
                  int(first["future_b"]["local_id"]))
    # Identity numbers, not frame-local numbers, are stored as local_id here because
    # frame_observations receives an identity-labelled movie.
    for later in bookends[1:]:
        later_pair = (int(later["future_a"]["local_id"]),
                      int(later["future_b"]["local_id"]))
        if later_pair == first_pair:
            return first, True
    return first, False


def _suffix_mapping_safe(labels: np.ndarray, start_t: int,
                         mapping: dict[int, int]) -> bool:
    sources = set(mapping)
    for frame in labels[start_t:]:
        present = set(map(int, np.unique(frame))) - {0}
        for source, target in mapping.items():
            if source == target or source not in present:
                continue
            if target in present and target not in sources:
                return False
    return True


def _apply_suffix_mapping(labels: np.ndarray, start_t: int,
                          mapping: dict[int, int]) -> None:
    for t in range(start_t, len(labels)):
        original = labels[t].copy()
        for source, target in mapping.items():
            if source != target:
                labels[t][original == source] = target


def _partition_interval(labels: np.ndarray, start_t: int, future_t: int,
                        before: list[dict], future_a: dict, future_b: dict,
                        identity_a: int, identity_b: int,
                        source_a: int, source_b: int, params: dict
                        ) -> tuple[np.ndarray, np.ndarray] | None:
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    sources = {identity_a, identity_b, source_a, source_b}
    pre_t = start_t - 1
    span = max(future_t - pre_t, 1)
    for t in range(start_t, future_t):
        alpha = (t - pre_t) / span
        seed_a = (1.0 - alpha) * before[0]["position"] + alpha * future_a["position"]
        seed_b = (1.0 - alpha) * before[1]["position"] + alpha * future_b["position"]
        union = np.isin(candidate[t], list(sources))
        if not union.any():
            observations = list(frame_observations(candidate[t], candidate[t]).values())
            if not observations:
                return None
            midpoint = (seed_a + seed_b) / 2.0
            nearest = min(observations,
                          key=lambda row: float(np.linalg.norm(row["position"] - midpoint)))
            if float(np.linalg.norm(nearest["position"] - midpoint)) > 2.0 * float(params["merge_reach_px"]):
                return None
            union = nearest["mask"]
        yy, xx = np.nonzero(union)
        da = (yy - seed_a[0]) ** 2 + (xx - seed_a[1]) ** 2
        db = (yy - seed_b[0]) ** 2 + (xx - seed_b[1]) ** 2
        part_a = np.zeros(union.shape, bool); part_b = np.zeros(union.shape, bool)
        part_a[yy[da <= db], xx[da <= db]] = True
        part_b[yy[db < da], xx[db < da]] = True
        minimum = int(params["minimum_partition_px"])
        if int(part_a.sum()) < minimum or int(part_b.sum()) < minimum:
            return None
        if ndi.label(part_a, structure=np.ones((3, 3), np.uint8))[1] != 1:
            return None
        if ndi.label(part_b, structure=np.ones((3, 3), np.uint8))[1] != 1:
            return None
        candidate[t][union] = 0
        candidate[t][part_a] = identity_a
        candidate[t][part_b] = identity_b
        inferred[t][union] = True
    mapping = {source_a: identity_a, source_b: identity_b}
    if not _suffix_mapping_safe(candidate, future_t, mapping):
        return None
    _apply_suffix_mapping(candidate, future_t, mapping)
    return candidate, inferred


def resolve_events(labels: np.ndarray, raw: np.ndarray, params: dict
                   ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Resolve only bookended two-to-one-to-two events; expose every other roster change."""
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    unresolved = np.zeros(labels.shape, bool)
    rows: list[dict] = []
    covered_transitions: set[int] = set()
    event_number = 0
    observation_cache = [frame_observations(fixed[t], raw[t])
                         for t in range(len(fixed))]

    for t in range(1, len(fixed)):
        if t in covered_transitions:
            continue
        previous = observation_cache[t - 1]
        current = observation_cache[t]
        vanished = sorted(set(previous) - set(current))
        continuing = sorted(set(previous) & set(current))
        if len(current) >= len(previous):
            continue
        host_geometry = {}
        for identity_b in continuing:
            pixels = np.column_stack(np.nonzero(current[identity_b]["mask"]))
            host_geometry[identity_b] = {
                "tree": cKDTree(pixels),
                "expanded": ndi.binary_dilation(current[identity_b]["mask"], iterations=2),
                "growth": current[identity_b]["area"] - previous[identity_b]["area"],
            }
        for identity_a in vanished:
            history_start = max(0, t - int(params["minimum_pre_event_frames"]))
            required_history = range(history_start, t)
            if (len(list(required_history)) < int(params["minimum_pre_event_frames"])
                    or any(identity_a not in observation_cache[frame]
                           for frame in required_history)
                    or previous[identity_a]["area"]
                    < int(params["minimum_merge_identity_px"])):
                continue
            hosts = []
            for identity_b in continuing:
                if any(identity_b not in observation_cache[frame]
                       for frame in required_history):
                    continue
                geometry = host_geometry[identity_b]
                distance = float(geometry["tree"].query(
                    previous[identity_a]["position"], k=1)[0])
                absorbed = float(np.mean(
                    geometry["expanded"][previous[identity_a]["mask"]]))
                growth = geometry["growth"]
                growth_fraction = growth / max(previous[identity_a]["area"], 1)
                if (distance <= float(params["merge_reach_px"])
                        and absorbed >= float(params["minimum_absorbed_fraction"])
                        and float(params["minimum_host_growth_fraction"])
                        <= growth_fraction <= float(params["maximum_host_growth_fraction"])):
                    hosts.append((identity_b, distance, absorbed, growth_fraction))
            if not hosts:
                continue
            identity_b, host_distance, absorbed_fraction, growth_fraction = min(
                hosts, key=lambda row: row[1])
            before = [previous[identity_a], previous[identity_b]]
            bookends = _find_bookends(fixed, raw, observation_cache, before, t, params)
            bookend, stable = _stable_bookend(bookends)
            event_number += 1
            event_id = f"E{event_number:04d}"
            if bookend is None or not stable:
                unresolved[t][current[identity_b]["mask"]] = True
                rows.append({
                    "event_id": event_id, "kind": "possible_merge",
                    "status": "unresolved", "start_t": t,
                    "start_imagej_frame": t + 1, "end_t": None,
                    "end_imagej_frame": None, "identity_a": identity_a,
                    "identity_b": identity_b, "host_distance_px": host_distance,
                    "absorbed_fraction": absorbed_fraction,
                    "host_growth_fraction": growth_fraction,
                    "lag_horizon": bookend["horizon"] if bookend else None,
                    "evidence_margin": bookend["margin"] if bookend else None,
                    "reason": "no stable separated bookend at two tested lag horizons",
                })
                continue
            source_a = int(bookend["future_a"]["local_id"])
            source_b = int(bookend["future_b"]["local_id"])
            partition = _partition_interval(
                fixed, t, bookend["future_t"], before,
                bookend["future_a"], bookend["future_b"], identity_a, identity_b,
                source_a, source_b, params)
            if partition is None:
                unresolved[t][current[identity_b]["mask"]] = True
                rows.append({
                    "event_id": event_id, "kind": "possible_merge",
                    "status": "unresolved", "start_t": t,
                    "start_imagej_frame": t + 1,
                    "end_t": int(bookend["future_t"]),
                    "end_imagej_frame": int(bookend["future_t"] + 1),
                    "identity_a": identity_a, "identity_b": identity_b,
                    "host_distance_px": host_distance,
                    "absorbed_fraction": absorbed_fraction,
                    "host_growth_fraction": growth_fraction,
                    "lag_horizon": bookend["horizon"],
                    "evidence_margin": bookend["margin"],
                    "reason": "bookend passed but exclusive union partition failed",
                })
                continue
            fixed, event_inferred = partition
            inferred |= event_inferred
            covered_transitions.update(range(t, int(bookend["future_t"]) + 1))
            future_t = int(bookend["future_t"])
            for frame in range(t, future_t):
                observation_cache[frame] = frame_observations(fixed[frame], raw[frame])
            mapping = {source_a: identity_a, source_b: identity_b}
            for frame in range(future_t, len(fixed)):
                remapped: dict[int, dict] = {}
                for source_identity, observation in observation_cache[frame].items():
                    target_identity = mapping.get(source_identity, source_identity)
                    updated = dict(observation)
                    updated["local_id"] = int(target_identity)
                    remapped[int(target_identity)] = updated
                observation_cache[frame] = remapped
            rows.append({
                "event_id": event_id, "kind": "temporary_merge",
                "status": "resolved_inferred", "start_t": t,
                "start_imagej_frame": t + 1,
                "end_t": int(bookend["future_t"]),
                "end_imagej_frame": int(bookend["future_t"] + 1),
                "identity_a": identity_a, "identity_b": identity_b,
                "host_distance_px": host_distance,
                "absorbed_fraction": absorbed_fraction,
                "host_growth_fraction": growth_fraction,
                "lag_horizon": int(bookend["horizon"]),
                "evidence_margin": float(bookend["margin"]),
                "a_motion_support": float(bookend["a_motion_support"]),
                "b_motion_support": float(bookend["b_motion_support"]),
                "reason": "two identities separated before and at two delayed bookends",
            })
            break

    # Preserve every remaining identity-roster change as an explicit unresolved event.
    recorded_starts = {int(row["start_t"]) for row in rows}
    for t in range(1, len(fixed)):
        if t in recorded_starts or t in covered_transitions:
            continue
        old_ids = set(map(int, np.unique(fixed[t - 1]))) - {0}
        new_ids = set(map(int, np.unique(fixed[t]))) - {0}
        appeared, disappeared = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
        if not appeared and not disappeared:
            continue
        event_number += 1
        affected = np.isin(fixed[t], appeared)
        if disappeared:
            prior = np.isin(fixed[t - 1], disappeared)
            affected |= ndi.binary_dilation(prior, iterations=2)
        unresolved[t] |= affected
        rows.append({
            "event_id": f"E{event_number:04d}", "kind": "roster_change",
            "status": "unresolved", "start_t": t,
            "start_imagej_frame": t + 1, "end_t": t,
            "end_imagej_frame": t + 1,
            "identity_a": ";".join(map(str, disappeared)),
            "identity_b": ";".join(map(str, appeared)),
            "host_distance_px": None, "lag_horizon": None,
            "evidence_margin": None,
            "reason": "appearance or disappearance not proven to be birth, death, entry, exit, split, merge, or miss",
        })
    events = pd.DataFrame(rows)
    if events.empty:
        events = pd.DataFrame(columns=["status"])
    return fixed, events, inferred, unresolved
