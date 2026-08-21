from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from tracking import frame_observations, motion_support


def _identity_shape(mask: np.ndarray) -> np.ndarray:
    """Area-normalized axes and boundary length for identity matching."""
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


def _shape_change(mask: np.ndarray, expected: np.ndarray) -> float:
    measured = _identity_shape(mask)
    return float(np.mean(np.abs(np.log2(
        (measured + 1e-3) / (expected + 1e-3)))))


def _previous_overlap(previous: np.ndarray, current: np.ndarray) -> float:
    expanded = ndi.binary_dilation(current, structure=np.ones((3, 3), bool))
    return float(np.count_nonzero(previous & expanded) / max(
        min(int(previous.sum()), int(current.sum())), 1))


def _nearest_host_pixel(host: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    pixels = np.column_stack(np.nonzero(host))
    if not len(pixels):
        raise ValueError("cannot place an identity seed in an empty merged object")
    index = int(np.argmin(np.sum((pixels - point[None, :]) ** 2, axis=1)))
    return int(pixels[index, 0]), int(pixels[index, 1])


def _candidate_seeds(host: np.ndarray, raw: np.ndarray,
                     predicted_a: np.ndarray, predicted_b: np.ndarray,
                     params: dict,
                     destination_a: np.ndarray | None = None
                     ) -> list[tuple[int, int]]:
    smooth = ndi.gaussian_filter(raw.astype(np.float32),
                                 sigma=float(params["peak_sigma_px"]))
    host_values = smooth[host]
    threshold = float(np.percentile(host_values, 40)) if host_values.size else 0.0
    peaks = peak_local_max(
        smooth, min_distance=int(params["peak_min_distance_px"]),
        threshold_abs=threshold, labels=host.astype(np.uint8),
        num_peaks=int(params["maximum_peaks"]), exclude_border=False)
    seeds = [tuple(map(int, row)) for row in peaks]
    # Interpolated bookend positions are always candidates. This makes the operation
    # forceable even when one dim cell has no distinct intensity maximum.
    seeds += [_nearest_host_pixel(host, predicted_a),
              _nearest_host_pixel(host, predicted_b)]
    if destination_a is not None and np.any(destination_a):
        destination_position = np.mean(
            np.column_stack(np.nonzero(destination_a)), axis=0)
        seeds.append(_nearest_host_pixel(host, destination_position))
    return list(dict.fromkeys(seeds))


def _partition_score(host: np.ndarray, raw: np.ndarray,
                     previous_a: dict, previous_b: dict,
                     predicted_a: np.ndarray, predicted_b: np.ndarray,
                     expected_area_a: float, expected_area_b: float,
                     expected_shape_a: np.ndarray, expected_shape_b: np.ndarray,
                     lag: np.ndarray, seed_a: tuple[int, int],
                     seed_b: tuple[int, int], params: dict,
                     destination_a: np.ndarray | None = None
                     ) -> dict | None:
    if seed_a == seed_b:
        return None
    separation = float(np.linalg.norm(np.subtract(seed_a, seed_b)))
    if separation < float(params["minimum_seed_separation_px"]):
        return None
    markers = np.zeros(host.shape, np.uint8)
    markers[seed_a] = 1
    markers[seed_b] = 2
    smooth = ndi.gaussian_filter(raw.astype(np.float32),
                                 sigma=float(params["peak_sigma_px"]))
    # Microglial processes and the detector both use eight-neighbour connectivity.
    # Four-neighbour watershed can silently leave diagonally attached host pixels at 0.
    split = watershed(-smooth, markers=markers, mask=host,
                      connectivity=np.ones((3, 3), bool))
    part_a, part_b = split == 1, split == 2
    minimum = int(params["minimum_partition_px"])
    if int(part_a.sum()) < minimum or int(part_b.sum()) < minimum:
        return None
    if ndi.label(part_a, structure=np.ones((3, 3), np.uint8))[1] != 1:
        return None
    if ndi.label(part_b, structure=np.ones((3, 3), np.uint8))[1] != 1:
        return None
    if not np.array_equal(part_a | part_b, host) or np.any(part_a & part_b):
        return None

    position_a = np.mean(np.column_stack(np.nonzero(part_a)), axis=0)
    position_b = np.mean(np.column_stack(np.nonzero(part_b)), axis=0)
    distance_a = float(np.linalg.norm(position_a - predicted_a))
    distance_b = float(np.linalg.norm(position_b - predicted_b))
    area_change_a = abs(float(np.log2(part_a.sum() / max(expected_area_a, 1.0))))
    area_change_b = abs(float(np.log2(part_b.sum() / max(expected_area_b, 1.0))))
    shape_change_a = _shape_change(part_a, expected_shape_a)
    shape_change_b = _shape_change(part_b, expected_shape_b)
    previous_overlap_a = _previous_overlap(previous_a["mask"], part_a)
    previous_overlap_b = _previous_overlap(previous_b["mask"], part_b)
    support_a = motion_support(previous_a["mask"], part_a, lag)
    support_b = motion_support(previous_b["mask"], part_b, lag)
    # Two touching cells often retain two bright bodies with a dim waist between them.
    # Reward partitions that place their shared boundary through that waist. Motion and
    # identity history still work when the bodies overlap too strongly for a clear dip.
    boundary = ((part_a & ndi.binary_dilation(part_b))
                | (part_b & ndi.binary_dilation(part_a)))
    peak_a = float(np.percentile(smooth[part_a], 95))
    peak_b = float(np.percentile(smooth[part_b], 95))
    waist = float(np.median(smooth[boundary])) if boundary.any() else min(peak_a, peak_b)
    body_separation_support = float(np.clip(
        np.log2((min(peak_a, peak_b) + 1.0) / (waist + 1.0)), 0.0, 2.0))
    destination_overlap_a = 0.0
    destination_overlap_b = 0.0
    destination_margin = 0.0
    destination_inside_host_px = 0
    if destination_a is not None:
        measured_destination = destination_a & host
        destination_inside_host_px = int(measured_destination.sum())
        if destination_inside_host_px:
            destination_overlap_a = float(
                np.mean(part_a[measured_destination]))
            destination_overlap_b = float(
                np.mean(part_b[measured_destination]))
            destination_margin = destination_overlap_a - destination_overlap_b
    cost = (
        float(params["position_weight"])
        * (distance_a + distance_b) / max(float(params["position_scale_px"]), 1e-6)
        + float(params["area_weight"]) * (area_change_a + area_change_b)
        + float(params.get("identity_shape_weight", 0.0))
        * (shape_change_a + shape_change_b)
        - float(params.get("identity_overlap_weight", 0.0))
        * (previous_overlap_a + previous_overlap_b)
        - float(params["motion_weight"])
        * (support_a["motion_support"] + support_b["motion_support"])
        - float(params.get("body_shape_weight", 0.0)) * body_separation_support
        - float(params.get("motion_destination_weight", 0.0))
        * destination_margin
    )
    return {
        "cost": float(cost), "part_a": part_a, "part_b": part_b,
        "distance_a_px": distance_a, "distance_b_px": distance_b,
        "area_a_px": int(part_a.sum()), "area_b_px": int(part_b.sum()),
        "a_identity_shape_change": shape_change_a,
        "b_identity_shape_change": shape_change_b,
        "a_previous_overlap": previous_overlap_a,
        "b_previous_overlap": previous_overlap_b,
        "a_motion_support": float(support_a["motion_support"]),
        "b_motion_support": float(support_b["motion_support"]),
        "a_supportive_px": int(support_a["supportive_px"]),
        "b_supportive_px": int(support_b["supportive_px"]),
        "a_contradictory_px": int(support_a["contradictory_px"]),
        "b_contradictory_px": int(support_b["contradictory_px"]),
        "body_peak_a": peak_a, "body_peak_b": peak_b,
        "body_waist_intensity": waist,
        "body_separation_support": body_separation_support,
        "destination_inside_host_px": destination_inside_host_px,
        "destination_overlap_a": destination_overlap_a,
        "destination_overlap_b": destination_overlap_b,
        "destination_margin": destination_margin,
        "seed_a_y": int(seed_a[0]), "seed_a_x": int(seed_a[1]),
        "seed_b_y": int(seed_b[0]), "seed_b_x": int(seed_b[1]),
        "seed_separation_px": separation,
    }


def _split_interval(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                    start_t: int, future_t: int, identity_a: int,
                    identity_b: int, params: dict,
                    destination_masks: dict[int, np.ndarray] | None = None,
                    ) -> tuple[np.ndarray, list[dict]] | None:
    candidate = labels.copy()
    before = frame_observations(candidate[start_t - 1], raw[start_t - 1])
    future = frame_observations(candidate[future_t], raw[future_t])
    if identity_a not in before or identity_b not in before \
            or identity_a not in future or identity_b not in future:
        return None
    before_a, before_b = before[identity_a], before[identity_b]
    future_a, future_b = future[identity_a], future[identity_b]
    before_shape_a = _identity_shape(before_a["mask"])
    before_shape_b = _identity_shape(before_b["mask"])
    future_shape_a = _identity_shape(future_a["mask"])
    future_shape_b = _identity_shape(future_b["mask"])
    previous_a, previous_b = before_a, before_b
    span = future_t - (start_t - 1)
    rows: list[dict] = []

    for t in range(start_t, future_t):
        if np.any(candidate[t] == identity_a) or not np.any(candidate[t] == identity_b):
            return None
        host = candidate[t] == identity_b
        alpha = (t - (start_t - 1)) / span
        predicted_a = ((1.0 - alpha) * before_a["position"]
                       + alpha * future_a["position"])
        predicted_b = ((1.0 - alpha) * before_b["position"]
                       + alpha * future_b["position"])
        expected_area_a = ((1.0 - alpha) * before_a["area"]
                           + alpha * future_a["area"])
        expected_area_b = ((1.0 - alpha) * before_b["area"]
                           + alpha * future_b["area"])
        expected_shape_a = ((1.0 - alpha) * before_shape_a
                            + alpha * future_shape_a)
        expected_shape_b = ((1.0 - alpha) * before_shape_b
                            + alpha * future_shape_b)
        destination_a = (None if destination_masks is None
                         else destination_masks.get(t))
        seeds = _candidate_seeds(
            host, raw[t], predicted_a, predicted_b, params, destination_a)
        candidates: list[dict] = []
        for first, second in itertools.permutations(seeds, 2):
            scored = _partition_score(
                host, raw[t], previous_a, previous_b, predicted_a, predicted_b,
                expected_area_a, expected_area_b, expected_shape_a,
                expected_shape_b, lag[t - 1], first, second, params,
                destination_a)
            if scored is not None:
                candidates.append(scored)
        if not candidates:
            return None
        candidates.sort(key=lambda row: row["cost"])
        best = candidates[0]
        if destination_a is not None:
            if (best["destination_overlap_a"]
                    < float(params.get("minimum_destination_overlap", 0.0))
                    or best["destination_margin"]
                    < float(params.get("minimum_destination_margin", -1.0))):
                return None
        second_cost = candidates[1]["cost"] if len(candidates) > 1 else np.inf
        original_host = host.copy()
        candidate[t][original_host] = 0
        candidate[t][best["part_a"]] = identity_a
        candidate[t][best["part_b"]] = identity_b
        if not np.array_equal(candidate[t] > 0, labels[t] > 0):
            raise AssertionError("merge partition changed foreground support")
        current = frame_observations(candidate[t], raw[t])
        previous_a, previous_b = current[identity_a], current[identity_b]
        rows.append({
            "t": t, "imagej_frame": t + 1,
            "candidate_count": len(candidates),
            "second_best_margin": float(second_cost - best["cost"]),
            **{key: value for key, value in best.items()
               if key not in ("part_a", "part_b")},
        })
    return candidate, rows


def _first_bookend(labels: np.ndarray, start_t: int, identity_a: int,
                   identity_b: int, maximum_frames: int) -> int | None:
    last = min(len(labels), start_t + int(maximum_frames) + 1)
    for future_t in range(start_t + 1, last):
        present = set(map(int, np.unique(labels[future_t]))) - {0}
        if identity_a in present and identity_b in present:
            return future_t
        if identity_b not in present:
            return None
    return None


def carry_identities_through_merges(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Force persistent identities through proven two-to-one-to-two intervals.

    Identity continuity is resolved before this function. A cell that disappears into
    a continuing neighbour and reappears with the same identity at a later separated
    bookend is not treated as a birth or death. The unchanged merged foreground is
    partitioned with globally exclusive watershed candidates scored by position, area,
    and blue/red lag motion support.
    """
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    events: list[dict] = []
    event_number = 0
    for t in range(1, len(fixed)):
        previous = frame_observations(fixed[t - 1], raw[t - 1])
        current = frame_observations(fixed[t], raw[t])
        vanished = sorted(set(previous) - set(current))
        continuing = sorted(set(previous) & set(current))
        for identity_a in vanished:
            history_start = max(0, t - int(params["minimum_pre_event_frames"]))
            if any(np.all(fixed[frame] != identity_a)
                   for frame in range(history_start, t)):
                continue
            hosts: list[tuple[float, int, float, float]] = []
            for identity_b in continuing:
                expanded = ndi.binary_dilation(current[identity_b]["mask"], iterations=2)
                absorbed = float(np.mean(expanded[previous[identity_a]["mask"]]))
                pixels = np.column_stack(np.nonzero(current[identity_b]["mask"]))
                distance = float(np.min(np.linalg.norm(
                    pixels - previous[identity_a]["position"][None, :], axis=1)))
                growth = current[identity_b]["area"] - previous[identity_b]["area"]
                growth_fraction = growth / max(previous[identity_a]["area"], 1)
                if (distance <= float(params["merge_reach_px"])
                        and absorbed >= float(params["minimum_absorbed_fraction"])
                        and float(params["minimum_host_growth_fraction"])
                        <= growth_fraction <= float(params["maximum_host_growth_fraction"])):
                    hosts.append((distance, identity_b, absorbed, growth_fraction))
            for host_distance, identity_b, absorbed, growth_fraction in sorted(hosts):
                future_t = _first_bookend(
                    fixed, t, identity_a, identity_b,
                    int(params["maximum_bookend_frames"]))
                if future_t is None:
                    continue
                partition = _split_interval(
                    fixed, raw, lag, t, future_t, identity_a, identity_b, params)
                if partition is None:
                    continue
                candidate, frame_rows = partition
                original_support = fixed > 0
                fixed = candidate
                if not np.array_equal(fixed > 0, original_support):
                    raise AssertionError("merge carry-forward changed foreground support")
                for row in frame_rows:
                    inferred[row["t"]][(fixed[row["t"]] == identity_a)
                                       | (fixed[row["t"]] == identity_b)] = True
                event_number += 1
                events.append({
                    "event_id": f"PM{event_number:04d}",
                    "kind": "persistent_temporary_merge",
                    "status": "resolved_inferred",
                    "start_t": t, "start_imagej_frame": t + 1,
                    "end_t": future_t - 1, "end_imagej_frame": future_t,
                    "bookend_t": future_t,
                    "bookend_imagej_frame": future_t + 1,
                    "identity_a": identity_a, "identity_b": identity_b,
                    "host_distance_px": host_distance,
                    "absorbed_fraction": absorbed,
                    "host_growth_fraction": growth_fraction,
                    "lag_horizon": future_t - (t - 1),
                    "partitioned_frames": future_t - t,
                    "mean_a_motion_support": float(np.mean(
                        [row["a_motion_support"] for row in frame_rows])),
                    "mean_b_motion_support": float(np.mean(
                        [row["b_motion_support"] for row in frame_rows])),
                    "mean_second_best_margin": float(np.mean(
                        [row["second_best_margin"] for row in frame_rows])),
                    "reason": "same two persistent identities separated at both bookends; merged foreground partitioned using raw peaks and blue/red lag support",
                    "frame_rows": frame_rows,
                })
                break

    flat_rows: list[dict] = []
    for event in events:
        frame_rows = event.pop("frame_rows")
        for row in frame_rows:
            flat_rows.append({
                "event_id": event["event_id"],
                "identity_a": event["identity_a"],
                "identity_b": event["identity_b"], **row,
            })
    event_table = pd.DataFrame(events)
    event_table.attrs["frame_rows"] = pd.DataFrame(flat_rows)
    return fixed, event_table, inferred
