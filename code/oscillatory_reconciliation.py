from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from model_review import eroded_cores
from persistent_merges import _split_interval
from tracking import motion_support


def _conflict_free_selection(
        scores: np.ndarray, incidence: np.ndarray,
        ) -> np.ndarray | None:
    """Return the exact independent optimum without starting a MILP solver."""
    values = np.asarray(scores, float)
    if np.any(np.sum(incidence, axis=1) > 1.0) or np.any(values == 0.0):
        return None
    return np.flatnonzero(values > 0.0)


def _ids(labels: np.ndarray) -> list[int]:
    return sorted(set(map(int, np.unique(labels))) - {0})


def _frames(labels: np.ndarray) -> dict[int, np.ndarray]:
    return {
        identity: np.flatnonzero(np.any(labels == identity, axis=(1, 2)))
        for identity in _ids(labels)
    }


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = np.column_stack(np.nonzero(mask))
    if not len(points):
        return 0, 0, 0, 0
    low = points.min(axis=0)
    high = points.max(axis=0) + 1
    return int(low[0]), int(high[0]), int(low[1]), int(high[1])


def _centre(mask: np.ndarray) -> np.ndarray:
    return np.mean(np.column_stack(np.nonzero(mask)), axis=0)


def _mask_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_points = np.column_stack(np.nonzero(first))
    second_points = np.column_stack(np.nonzero(second))
    if not len(first_points) or not len(second_points):
        return float("inf")
    if len(first_points) > len(second_points):
        first_points, second_points = second_points, first_points
    distances, _ = cKDTree(second_points).query(first_points, k=1)
    return float(np.min(distances))


def _substantial_core_count(mask: np.ndarray, minimum_core_px: int) -> int:
    return sum(area >= int(minimum_core_px) for area, _ in eroded_cores(mask, 1))


def _transition_evidence(old: np.ndarray, new: np.ndarray,
                         lag_frame: np.ndarray) -> dict[str, float]:
    support = motion_support(old, new, lag_frame, _bounds(old), _bounds(new))
    blue = np.column_stack(np.nonzero(old & (lag_frame <= -1.5)))
    red = np.column_stack(np.nonzero(new & (lag_frame >= 1.5)))
    displacement = _centre(new) - _centre(old)
    cosine = float("nan")
    if len(blue) >= 8 and len(red) >= 8:
        motion_vector = red.mean(axis=0) - blue.mean(axis=0)
        denominator = float(np.linalg.norm(motion_vector)
                            * np.linalg.norm(displacement))
        if denominator > 1e-12:
            cosine = float(np.dot(motion_vector, displacement) / denominator)
    return {**support, "direction_cosine": cosine}


def _relay_orientation(first: int, second: int,
                       identity_frames: dict[int, np.ndarray],
                       minimum_solo_frames: int) -> tuple[int, int] | None:
    a = identity_frames[first]
    b = identity_frames[second]
    if not len(a) or not len(b):
        return None
    for persistent, relay, outer, inner in ((first, second, a, b),
                                             (second, first, b, a)):
        before = int(np.count_nonzero(outer < int(inner.min())))
        after = int(np.count_nonzero(outer > int(inner.max())))
        relay_solo = int(np.count_nonzero(~np.isin(inner, outer)))
        if before >= minimum_solo_frames and after >= minimum_solo_frames \
                and relay_solo >= minimum_solo_frames:
            return persistent, relay
    return None


def _relay_metrics(labels: np.ndarray, lag: np.ndarray, persistent: int,
                   relay: int, identity_frames: dict[int, np.ndarray],
                   minimum_core_px: int) -> dict:
    persistent_frames = identity_frames[persistent]
    relay_frames = identity_frames[relay]
    relay_start, relay_end = int(relay_frames.min()), int(relay_frames.max())
    overlap = np.intersect1d(persistent_frames, relay_frames)

    reference_frames = np.concatenate((
        persistent_frames[(persistent_frames < relay_start)
                          & (persistent_frames >= relay_start - 4)],
        persistent_frames[(persistent_frames > relay_end)
                          & (persistent_frames <= relay_end + 4)]))
    reference_areas = [int(np.count_nonzero(labels[t] == persistent))
                       for t in reference_frames]
    reference_area = float(np.median(reference_areas)) if reference_areas else 0.0

    path: dict[int, np.ndarray] = {}
    for t in range(max(0, relay_start - 1), min(len(labels), relay_end + 2)):
        mask = (labels[t] == persistent) | (labels[t] == relay)
        if np.any(mask):
            path[t] = mask
    path_areas = [int(mask.sum()) for mask in path.values()]
    area_ratio = (float(np.median(path_areas)) / reference_area
                  if reference_area else float("inf"))

    distances = [_mask_distance(labels[t] == persistent, labels[t] == relay)
                 for t in overlap]
    core_counts = [_substantial_core_count(
        (labels[t] == persistent) | (labels[t] == relay), minimum_core_px)
                   for t in overlap]

    transitions: list[dict[str, float]] = []
    times = sorted(path)
    for earlier, later in zip(times, times[1:]):
        if later != earlier + 1 or earlier >= len(lag):
            continue
        transitions.append(_transition_evidence(
            path[earlier], path[later], lag[earlier]))
    measured_cosines = [row["direction_cosine"] for row in transitions
                        if np.isfinite(row["direction_cosine"])]
    supported_cosines = [value for value in measured_cosines if value > 0.5]
    motion_fraction = (len(supported_cosines) / len(measured_cosines)
                       if measured_cosines else 0.0)
    support_values = [float(row["motion_support"]) for row in transitions]

    interval = np.arange(relay_start, relay_end + 1)
    covered = np.isin(interval, persistent_frames) | np.isin(interval, relay_frames)
    return {
        "persistent_identity": int(persistent),
        "relay_identity": int(relay),
        "persistent_first_imagej_frame": int(persistent_frames.min() + 1),
        "persistent_last_imagej_frame": int(persistent_frames.max() + 1),
        "relay_first_imagej_frame": int(relay_start + 1),
        "relay_last_imagej_frame": int(relay_end + 1),
        "relay_frames": int(len(relay_frames)),
        "relay_solo_frames": int(np.count_nonzero(
            ~np.isin(relay_frames, persistent_frames))),
        "overlap_frames": int(len(overlap)),
        "overlap_imagej_frames": ";".join(map(str, overlap + 1)),
        "maximum_pair_distance_px": (float(max(distances))
                                      if distances else float("inf")),
        "median_pair_distance_px": (float(np.median(distances))
                                     if distances else float("inf")),
        "reference_area_px": reference_area,
        "median_path_area_px": float(np.median(path_areas)) if path_areas else 0.0,
        "median_union_area_ratio": area_ratio,
        "maximum_substantial_union_cores": int(max(core_counts))
            if core_counts else 0,
        "one_core_overlap_fraction": float(np.mean(np.asarray(core_counts) <= 1))
            if core_counts else 0.0,
        "path_coverage_fraction": float(np.mean(covered)) if len(covered) else 0.0,
        "measured_motion_transitions": int(len(measured_cosines)),
        "supported_motion_transitions": int(len(supported_cosines)),
        "motion_direction_support_fraction": motion_fraction,
        "median_motion_direction_cosine": (float(np.median(measured_cosines))
                                            if measured_cosines else float("nan")),
        "median_motion_support": (float(np.median(support_values))
                                  if support_values else 0.0),
    }


def overlapping_relay_candidates(labels: np.ndarray, lag: np.ndarray,
                                  params: dict,
                                  minimum_core_px: int = 12) -> pd.DataFrame:
    """Find A -> A+B -> B -> A+B -> A sequences that behave as one soma."""
    identity_frames = _frames(labels)
    rows: list[dict] = []
    identities = sorted(identity_frames)
    for index, first in enumerate(identities):
        for second in identities[index + 1:]:
            orientation = _relay_orientation(
                first, second, identity_frames,
                int(params.get("minimum_solo_frames", 3)))
            if orientation is None:
                continue
            persistent, relay = orientation
            overlap = np.intersect1d(
                identity_frames[persistent], identity_frames[relay])
            if not len(overlap) or len(overlap) > int(
                    params.get("maximum_overlap_frames", 4)):
                continue
            metrics = _relay_metrics(
                labels, lag, persistent, relay, identity_frames, minimum_core_px)
            allowed = bool(
                metrics["maximum_pair_distance_px"] <= float(params.get(
                    "maximum_pair_distance_px", 3.0))
                and float(params.get("minimum_union_area_ratio", 0.65))
                <= metrics["median_union_area_ratio"]
                <= float(params.get("maximum_union_area_ratio", 1.6))
                and metrics["maximum_substantial_union_cores"] <= int(params.get(
                    "maximum_substantial_union_cores", 1))
                and metrics["measured_motion_transitions"] >= int(params.get(
                    "minimum_motion_supported_transitions", 3))
                and metrics["motion_direction_support_fraction"] >= float(
                    params.get("minimum_motion_support_fraction", 0.7))
                and metrics["median_motion_direction_cosine"] >= float(
                    params.get("minimum_motion_direction_cosine", 0.5))
                and metrics["path_coverage_fraction"] == 1.0)
            rows.append({**metrics, "allowed": allowed})
    columns = [
        "persistent_identity", "relay_identity", "persistent_first_imagej_frame",
        "persistent_last_imagej_frame", "relay_first_imagej_frame",
        "relay_last_imagej_frame", "relay_frames", "relay_solo_frames",
        "overlap_frames", "overlap_imagej_frames", "maximum_pair_distance_px",
        "median_pair_distance_px", "reference_area_px", "median_path_area_px",
        "median_union_area_ratio", "maximum_substantial_union_cores",
        "one_core_overlap_fraction", "path_coverage_fraction",
        "measured_motion_transitions", "supported_motion_transitions",
        "motion_direction_support_fraction", "median_motion_direction_cosine",
        "median_motion_support", "allowed"]
    return pd.DataFrame(rows, columns=columns)


def reconcile_overlapping_relays(
        labels: np.ndarray, lag: np.ndarray, params: dict,
        minimum_core_px: int = 12,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidates = overlapping_relay_candidates(
        labels, lag, params, minimum_core_px)
    accepted = candidates[candidates.allowed].copy()
    if accepted.empty:
        return labels.copy(), accepted, candidates
    accepted = accepted.sort_values([
        "motion_direction_support_fraction", "median_motion_support",
        "relay_frames"], ascending=[False, False, False])
    result = labels.copy()
    decisions: list[dict] = []
    used: set[int] = set()
    for row in accepted.itertuples():
        persistent, relay = int(row.persistent_identity), int(row.relay_identity)
        if persistent in used or relay in used:
            continue
        renamed = int(np.count_nonzero(result == relay))
        result[result == relay] = persistent
        decisions.append({
            **row._asdict(),
            "decision_id": f"OR{len(decisions) + 1:04d}",
            "renamed_pixels": renamed,
            "method": "overlapping_relay_reunion",
        })
        used.update((persistent, relay))
    if not np.array_equal(result > 0, labels > 0):
        raise AssertionError("overlapping relay reunion changed foreground support")
    return result, pd.DataFrame(decisions), candidates


def graph_lineage_candidates(labels: np.ndarray, lag: np.ndarray, params: dict,
                             minimum_core_px: int = 12) -> pd.DataFrame:
    """Score every nested identity pair over its complete movie history."""
    relaxed = {
        "minimum_solo_frames": int(params.get("minimum_identity_frames", 3)),
        "maximum_overlap_frames": int(params.get("maximum_overlap_frames", 5)),
        "maximum_pair_distance_px": float(params.get(
            "maximum_endpoint_distance_px", 18.0)),
        "minimum_union_area_ratio": 0.25,
        "maximum_union_area_ratio": 4.0,
        "minimum_motion_supported_transitions": 0,
        "minimum_motion_support_fraction": 0.0,
        "minimum_motion_direction_cosine": -1.0,
        "maximum_substantial_union_cores": 1,
    }
    table = overlapping_relay_candidates(
        labels, lag, relaxed, minimum_core_px)
    if table.empty:
        table["graph_score"] = pd.Series(dtype=float)
        table["graph_allowed"] = pd.Series(dtype=bool)
        return table
    distance_scale = float(params.get("distance_scale_px", 12.0))
    area_fit = np.maximum(
        0.0, 1.0 - np.abs(np.log2(np.maximum(
            table.median_union_area_ratio.astype(float), 1e-12))))
    table["graph_score"] = (
        float(params.get("continuity_weight", 2.0))
        * table.path_coverage_fraction.astype(float)
        + float(params.get("motion_weight", 2.0))
        * table.motion_direction_support_fraction.astype(float)
        + float(params.get("area_weight", 1.0)) * area_fit
        + float(params.get("one_core_weight", 2.0))
        * table.one_core_overlap_fraction.astype(float)
        - table.median_pair_distance_px.astype(float) / max(distance_scale, 1e-6)
        - float(params.get("overlap_penalty", 0.5))
        * np.maximum(table.overlap_frames.astype(float) - 1.0, 0.0))
    table["graph_allowed"] = (
        (table.maximum_substantial_union_cores <= 1)
        & (table.path_coverage_fraction == 1.0)
        & (table.median_pair_distance_px <= float(params.get(
            "maximum_corridor_distance_px", 12.0)))
        & (table.graph_score >= float(params.get("minimum_edge_score", 3.0))))
    return table


def optimise_lineage_graph(
        labels: np.ndarray, lag: np.ndarray, params: dict,
        minimum_core_px: int = 12,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Select a maximum-score, non-conflicting set of whole-movie lineage joins."""
    candidates = graph_lineage_candidates(labels, lag, params, minimum_core_px)
    edges = candidates[candidates.graph_allowed].reset_index(drop=True)
    if edges.empty:
        return labels.copy(), edges, candidates
    identities = sorted(set(edges.persistent_identity.astype(int))
                        | set(edges.relay_identity.astype(int)))
    identity_index = {identity: index for index, identity in enumerate(identities)}
    incidence = np.zeros((len(identities), len(edges)), float)
    for column, row in enumerate(edges.itertuples()):
        incidence[identity_index[int(row.persistent_identity)], column] = 1.0
        incidence[identity_index[int(row.relay_identity)], column] = 1.0
    direct = _conflict_free_selection(
        edges.graph_score.to_numpy(float), incidence)
    if direct is None:
        solution = milp(
            c=-edges.graph_score.to_numpy(float),
            integrality=np.ones(len(edges), int),
            bounds=Bounds(np.zeros(len(edges)), np.ones(len(edges))),
            constraints=LinearConstraint(
                csr_matrix(incidence),
                np.full(len(identities), -np.inf), np.ones(len(identities))),
            options={"time_limit": 30.0})
        if not solution.success or solution.x is None:
            raise RuntimeError(
                f"whole-movie lineage graph failed: {solution.message}")
        chosen = edges[np.asarray(solution.x) > 0.5].copy()
    else:
        chosen = edges.iloc[direct].copy()
    chosen["decision_id"] = [f"GO{index + 1:04d}"
                             for index in range(len(chosen))]
    chosen["method"] = "whole_movie_graph_optimisation"
    result = labels.copy()
    for row in chosen.itertuples():
        result[result == int(row.relay_identity)] = int(row.persistent_identity)
    if not np.array_equal(result > 0, labels > 0):
        raise AssertionError("whole-movie graph optimisation changed foreground support")
    return result, chosen, candidates


def host_merge_graph_candidates(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
        params: dict, partition_params: dict,
        ) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Score bracketed identity gaps that can be partitioned from a grown host."""
    identity_frames = _frames(labels)
    rows: list[dict] = []
    partitions: dict[int, np.ndarray] = {}
    minimum_frames = int(params.get("minimum_identity_frames", 8))
    minimum_area = float(params.get("minimum_identity_area_px", 80.0))
    for identity, frames in identity_frames.items():
        if len(frames) < minimum_frames:
            continue
        areas = np.asarray([
            np.count_nonzero(labels[t] == identity) for t in frames], float)
        expected_identity_area = float(np.median(areas))
        if expected_identity_area < minimum_area:
            continue
        present = set(map(int, frames))
        for t in range(int(frames.min()) + 1, int(frames.max())):
            if t in present or t - 1 not in present or t + 1 not in present:
                continue
            before_position = _centre(labels[t - 1] == identity)
            after_position = _centre(labels[t + 1] == identity)
            predicted = (before_position + after_position) / 2.0
            for host in sorted(set(map(int, np.unique(labels[t]))) - {0}):
                if (not np.any(labels[t - 1] == host)
                        or not np.any(labels[t + 1] == host)):
                    continue
                host_mask = labels[t] == host
                host_distance = float(cKDTree(
                    np.column_stack(np.nonzero(host_mask))).query(predicted)[0])
                expected_host_area = float(np.median([
                    np.count_nonzero(labels[t - 1] == host),
                    np.count_nonzero(labels[t + 1] == host)]))
                host_growth = float(host_mask.sum() - expected_host_area)
                growth_ratio = host_growth / max(expected_identity_area, 1.0)
                if (host_distance > float(params.get(
                        "maximum_host_distance_px", 8.0))
                        or growth_ratio < float(params.get(
                            "minimum_host_growth_ratio", 0.5))
                        or growth_ratio > float(params.get(
                            "maximum_host_growth_ratio", 1.8))):
                    continue
                partition = _split_interval(
                    labels, raw, lag, t, t + 1, identity, host,
                    partition_params)
                if partition is None:
                    continue
                partitioned, partition_rows = partition
                evidence = partition_rows[0]
                identity_area_ratio = float(
                    evidence["area_a_px"] / max(expected_identity_area, 1.0))
                host_area_ratio = float(
                    evidence["area_b_px"] / max(expected_host_area, 1.0))
                minimum_motion = min(float(evidence["a_motion_support"]),
                                     float(evidence["b_motion_support"]))
                minimum_overlap = min(float(evidence["a_previous_overlap"]),
                                      float(evidence["b_previous_overlap"]))
                maximum_shape = max(float(evidence["a_identity_shape_change"]),
                                    float(evidence["b_identity_shape_change"]))
                area_minimum = float(params.get("minimum_partition_area_ratio", 0.4))
                area_maximum = float(params.get("maximum_partition_area_ratio", 2.5))
                allowed = bool(
                    minimum_motion >= float(params.get(
                        "minimum_partition_motion_support", 0.75))
                    and minimum_overlap >= float(params.get(
                        "minimum_partition_previous_overlap", 0.7))
                    and maximum_shape <= float(params.get(
                        "maximum_partition_shape_change", 0.35))
                    and area_minimum <= identity_area_ratio <= area_maximum
                    and area_minimum <= host_area_ratio <= area_maximum)
                score = (
                    float(params.get("motion_weight", 2.0)) * minimum_motion
                    + float(params.get("overlap_weight", 1.5)) * minimum_overlap
                    + float(params.get("shape_weight", 1.0))
                    * max(0.0, 1.0 - maximum_shape)
                    + float(params.get("growth_weight", 1.0))
                    * max(0.0, 1.0 - abs(np.log2(max(growth_ratio, 1e-12))))
                    - host_distance / max(float(params.get(
                        "distance_scale_px", 8.0)), 1e-6))
                row_number = len(rows)
                rows.append({
                    "candidate_index": row_number,
                    "lost_identity": int(identity),
                    "host_identity": int(host),
                    "t": int(t),
                    "imagej_frame": int(t + 1),
                    "expected_lost_area_px": expected_identity_area,
                    "expected_host_area_px": expected_host_area,
                    "host_growth_px": host_growth,
                    "host_growth_ratio": growth_ratio,
                    "host_distance_px": host_distance,
                    "lost_partition_area_ratio": identity_area_ratio,
                    "host_partition_area_ratio": host_area_ratio,
                    "minimum_motion_support": minimum_motion,
                    "minimum_previous_overlap": minimum_overlap,
                    "maximum_shape_change": maximum_shape,
                    "graph_score": float(score),
                    "graph_allowed": allowed,
                    **{key: value for key, value in evidence.items()
                       if key not in {"t", "imagej_frame"}},
                })
                partitions[row_number] = partitioned
    return pd.DataFrame(rows), partitions


def optimise_host_merge_graph(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
        params: dict, partition_params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Select non-conflicting one-frame host partitions across the whole movie."""
    candidates, partitions = host_merge_graph_candidates(
        labels, raw, lag, params, partition_params)
    if candidates.empty:
        return labels.copy(), candidates, candidates
    allowed = candidates[candidates.graph_allowed].reset_index(drop=True)
    if allowed.empty:
        return labels.copy(), allowed, candidates
    resources = sorted({
        (int(row.t), int(identity))
        for row in allowed.itertuples()
        for identity in (row.lost_identity, row.host_identity)})
    resource_index = {resource: index for index, resource in enumerate(resources)}
    incidence = np.zeros((len(resources), len(allowed)), float)
    for column, row in enumerate(allowed.itertuples()):
        incidence[resource_index[(int(row.t), int(row.lost_identity))], column] = 1.0
        incidence[resource_index[(int(row.t), int(row.host_identity))], column] = 1.0
    direct = _conflict_free_selection(
        allowed.graph_score.to_numpy(float), incidence)
    if direct is None:
        solution = milp(
            c=-allowed.graph_score.to_numpy(float),
            integrality=np.ones(len(allowed), int),
            bounds=Bounds(np.zeros(len(allowed)), np.ones(len(allowed))),
            constraints=LinearConstraint(
                csr_matrix(incidence), np.full(len(resources), -np.inf),
                np.ones(len(resources))),
            options={"time_limit": 30.0})
        if not solution.success or solution.x is None:
            raise RuntimeError(f"host-merge graph failed: {solution.message}")
        chosen = allowed[np.asarray(solution.x) > 0.5].copy()
    else:
        chosen = allowed.iloc[direct].copy()
    chosen["decision_id"] = [f"HP{index + 1:04d}"
                             for index in range(len(chosen))]
    chosen["method"] = "whole_movie_host_partition_graph"
    result = labels.copy()
    for row in chosen.sort_values("t").itertuples():
        candidate_index = int(row.candidate_index)
        partitioned = partitions[candidate_index]
        t = int(row.t)
        result[t] = partitioned[t]
    if not np.array_equal(result > 0, labels > 0):
        raise AssertionError("host-merge graph changed foreground support")
    return result, chosen, candidates
