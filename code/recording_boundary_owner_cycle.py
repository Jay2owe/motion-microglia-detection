"""Repair closed owner permutations formed near a recording boundary.

Discovery is complete-field and identity-agnostic. Biological identities,
tracks, frames, coordinates, regions, events, and review cases cannot be
provided as producer targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

import separable_merge_recovery as physical
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "identity_ids", "track_id", "track_ids", "frame_id",
    "frame_ids", "event_id", "event_ids", "coordinate", "coordinates",
    "review_case", "review_cases", "target_identity", "target_track",
    "target_frame", "target_event", "review_region", "forced_interval",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        key for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in key.lower() for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "recording-boundary owner-cycle recovery received forbidden "
            f"targets: {supplied}")
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("owner-cycle recovery must use field-wide discovery")


def resolve_data_relative_params(points: pd.DataFrame, frame_count: int,
                                 params: dict) -> dict:
    """Resolve physical and temporal units without recording-specific IDs."""
    resolved = dict(params)
    visible = points[points.state.isin(VISIBLE_STATES)]
    if visible.empty:
        raise ValueError("owner-cycle recovery requires visible physical points")
    radii = visible.radius_px.astype(float).to_numpy()
    median_area = float(np.median(np.pi * radii ** 2))
    median_radius = float(np.median(radii))
    if "minimum_component_area_fraction_of_median_soma" in resolved:
        resolved["minimum_component_area_px"] = max(1, int(round(
            median_area * float(resolved[
                "minimum_component_area_fraction_of_median_soma"]))))
    temporal = {
        "minimum_transition_run_fraction_of_movie":
            "minimum_transition_run_points",
        "maximum_transition_blank_fraction_of_movie":
            "maximum_transition_blank_frames",
        "maximum_anchor_reference_gap_fraction_of_movie":
            "maximum_anchor_reference_gap_frames",
        "maximum_duplicate_bracket_fraction_of_movie":
            "maximum_duplicate_bracket_frames",
    }
    for fraction_name, frame_name in temporal.items():
        if fraction_name in resolved:
            resolved[frame_name] = max(1, int(np.ceil(
                frame_count * float(resolved[fraction_name]))))
    if "watershed_sigma_fraction_of_median_soma_radius" in resolved:
        resolved["watershed_sigma_px"] = max(0.5, float(round(
            median_radius * float(resolved[
                "watershed_sigma_fraction_of_median_soma_radius"]))))
    return resolved


def _center(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    return float(xx.mean()), float(yy.mean())


def _radius(area: float) -> float:
    return float(np.sqrt(max(float(area), 1.0) / np.pi))


def _components(frame: np.ndarray, owner: int,
                minimum_area: int) -> list[dict]:
    labels, count = ndi.label(frame == int(owner), STRUCTURE)
    result = []
    for component_id in range(1, count + 1):
        mask = labels == component_id
        area = int(np.count_nonzero(mask))
        if area < minimum_area:
            continue
        result.append({
            "owner": int(owner), "component": component_id, "mask": mask,
            "area": area, "center": _center(mask), "radius": _radius(area),
        })
    return result


def _frame_components(frame: np.ndarray, minimum_area: int) -> list[dict]:
    result: list[dict] = []
    labels, owners, areas = component_accounting.label_value_components(frame)
    for owner in sorted(set(map(int, owners)) - {0}):
        component_ids = np.flatnonzero(owners == owner) + 1
        for component_id, value_component in enumerate(component_ids, start=1):
            area = int(areas[int(value_component) - 1])
            if area < minimum_area:
                continue
            mask = labels == int(value_component)
            result.append({
                "owner": owner, "component": component_id, "mask": mask,
                "area": area, "center": _center(mask),
                "radius": _radius(area),
            })
    return result


def _point_owner_runs(group: pd.DataFrame, boundary_last: int,
                      maximum_blank: int) -> list[dict]:
    rows = group[
        group.physically_visible.astype(bool)
        & group.frame.astype(int).between(0, boundary_last)
    ].sort_values("frame")
    runs: list[dict] = []
    for row in rows.itertuples(index=False):
        owner = int(row.accepted_owner)
        if owner <= 0:
            continue
        frame = int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame <= runs[-1]["last"] + maximum_blank + 1):
            runs[-1]["last"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "rows": [row]})
    return runs


def _direct_edges(scored: pd.DataFrame, frame_count: int,
                  params: dict) -> list[dict]:
    boundary_last = max(2, int(np.ceil(
        frame_count * float(params.get("boundary_fraction", 0.15)))) - 1)
    minimum_run = int(params.get("minimum_transition_run_points", 3))
    maximum_blank = int(params.get("maximum_transition_blank_frames", 2))
    maximum_step = float(params.get(
        "maximum_transition_step_sum_radii", 3.0))
    edges: list[dict] = []
    for track, group in scored.groupby("track_id", sort=True):
        runs = _point_owner_runs(group, boundary_last, maximum_blank)
        for left, right in zip(runs[:-1], runs[1:]):
            if left["owner"] == right["owner"]:
                continue
            if min(len(left["rows"]), len(right["rows"])) < minimum_run:
                continue
            before = left["rows"][-1]
            after = right["rows"][0]
            distance = float(np.hypot(
                float(before.x) - float(after.x),
                float(before.y) - float(after.y)))
            scale = max(
                float(before.radius_px) + float(after.radius_px), 1.0)
            normalized = distance / scale
            if normalized > maximum_step:
                continue
            edges.append({
                "source_owner": int(left["owner"]),
                "target_owner": int(right["owner"]),
                "transition_frame": int(right["first"]),
                "x": 0.5 * (float(before.x) + float(after.x)),
                "y": 0.5 * (float(before.y) + float(after.y)),
                "scale_px": scale,
                "evidence_kind": "physical_track_owner_transition",
                "physical_track": int(track),
                "support": int(min(len(left["rows"]), len(right["rows"]))),
                "normalized_cost": normalized,
            })
    return edges


def _component_edges(labels: np.ndarray, params: dict) -> list[dict]:
    minimum_area = int(params.get("minimum_component_area_px", 8))
    boundary_last = max(2, int(np.ceil(
        len(labels) * float(params.get("boundary_fraction", 0.15)))) - 1)
    maximum_cost = float(params.get(
        "maximum_component_transition_cost", 1.9))
    area_weight = float(params.get("component_area_log_weight", 0.20))
    edges: list[dict] = []
    for frame in range(min(boundary_last, len(labels) - 2) + 1):
        left = _frame_components(labels[frame], minimum_area)
        right = _frame_components(labels[frame + 1], minimum_area)
        if not left or not right:
            continue
        costs = np.full((len(left), len(right)), 1e6, dtype=float)
        for i, before in enumerate(left):
            for j, after in enumerate(right):
                distance = float(np.hypot(
                    before["center"][0] - after["center"][0],
                    before["center"][1] - after["center"][1]))
                scale = max(before["radius"] + after["radius"], 1.0)
                cost = (distance / scale + area_weight * abs(np.log(
                    max(after["area"], 1) / max(before["area"], 1))))
                if cost <= maximum_cost:
                    costs[i, j] = cost
        rows, columns = linear_sum_assignment(costs)
        for i, j in zip(rows, columns):
            if costs[i, j] >= 1e5:
                continue
            before, after = left[i], right[j]
            if before["owner"] == after["owner"]:
                continue
            edges.append({
                "source_owner": before["owner"],
                "target_owner": after["owner"],
                "transition_frame": frame + 1,
                "x": 0.5 * (before["center"][0] + after["center"][0]),
                "y": 0.5 * (before["center"][1] + after["center"][1]),
                "scale_px": before["radius"] + after["radius"],
                "evidence_kind": "one_to_one_component_continuation",
                "physical_track": -1, "support": 1,
                "normalized_cost": float(costs[i, j]),
            })
    return edges


def _deduplicate_edges(edges: Iterable[dict]) -> list[dict]:
    selected: dict[tuple[int, int], dict] = {}
    for edge in edges:
        key = (int(edge["source_owner"]), int(edge["target_owner"]))
        priority = (
            edge["evidence_kind"] != "physical_track_owner_transition",
            -int(edge["support"]), float(edge["normalized_cost"]),
            int(edge["transition_frame"]), int(edge["physical_track"]),
        )
        current = selected.get(key)
        if current is None:
            selected[key] = edge
            continue
        current_priority = (
            current["evidence_kind"] != "physical_track_owner_transition",
            -int(current["support"]), float(current["normalized_cost"]),
            int(current["transition_frame"]), int(current["physical_track"]),
        )
        if priority < current_priority:
            selected[key] = edge
    return list(selected.values())


def _canonical_cycle(nodes: list[int]) -> tuple[int, ...]:
    rotations = [tuple(nodes[index:] + nodes[:index])
                 for index in range(len(nodes))]
    return min(rotations)


def _simple_cycles(edges: list[dict], minimum: int,
                   maximum: int) -> list[tuple[int, ...]]:
    adjacency: dict[int, list[int]] = {}
    for edge in edges:
        adjacency.setdefault(int(edge["source_owner"]), []).append(
            int(edge["target_owner"]))
    found: set[tuple[int, ...]] = set()
    for start in sorted(adjacency):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for target in adjacency.get(node, []):
                if target == start and minimum <= len(path) <= maximum:
                    found.add(_canonical_cycle(path))
                elif target not in path and len(path) < maximum:
                    stack.append((target, path + [target]))
    return sorted(found)


def _events_connected(cycle_edges: list[dict], maximum_radii: float) -> bool:
    if not cycle_edges:
        return False
    visited = {0}
    pending = [0]
    while pending:
        left = pending.pop()
        for right in range(len(cycle_edges)):
            if right in visited:
                continue
            distance = float(np.hypot(
                cycle_edges[left]["x"] - cycle_edges[right]["x"],
                cycle_edges[left]["y"] - cycle_edges[right]["y"]))
            scale = max(0.5 * (
                cycle_edges[left]["scale_px"]
                + cycle_edges[right]["scale_px"]), 1.0)
            if distance / scale <= maximum_radii:
                visited.add(right)
                pending.append(right)
    return len(visited) == len(cycle_edges)


def discover_cycles(labels: np.ndarray, points: pd.DataFrame, params: dict,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return every field-discovered edge and qualifying closed cycle."""
    params = resolve_data_relative_params(points, len(labels), params)
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    edges = _deduplicate_edges(
        _direct_edges(scored, len(labels), params)
        + _component_edges(labels, params))
    by_pair = {(int(edge["source_owner"]), int(edge["target_owner"])): edge
               for edge in edges}
    minimum = int(params.get("minimum_cycle_owners", 4))
    maximum = int(params.get("maximum_cycle_owners", 8))
    minimum_direct_fraction = float(params.get(
        "minimum_direct_edge_fraction", 0.75))
    maximum_component_edges = int(params.get("maximum_component_edges", 1))
    maximum_event_link = float(params.get(
        "maximum_cycle_event_link_radii", 6.0))
    cycle_rows = []
    memberships = []
    for cycle_index, cycle in enumerate(
            _simple_cycles(edges, minimum, maximum), start=1):
        chosen = [by_pair[(cycle[index], cycle[(index + 1) % len(cycle)])]
                  for index in range(len(cycle))]
        direct = sum(edge["evidence_kind"]
                     == "physical_track_owner_transition" for edge in chosen)
        component = len(chosen) - direct
        reasons = []
        if direct / len(chosen) < minimum_direct_fraction:
            reasons.append("insufficient_direct_physical_edges")
        if component > maximum_component_edges:
            reasons.append("too_many_component_only_edges")
        if not _events_connected(chosen, maximum_event_link):
            reasons.append("transition_events_not_spatially_connected")
        cycle_id = f"C{cycle_index:03d}"
        cycle_rows.append({
            "cycle_id": cycle_id, "owners": "|".join(map(str, cycle)),
            "owner_count": len(cycle), "direct_edges": direct,
            "component_edges": component,
            "first_transition_frame": min(
                int(edge["transition_frame"]) for edge in chosen),
            "closure_frame": max(
                int(edge["transition_frame"]) for edge in chosen),
            "minimum_edge_support": min(
                int(edge["support"]) for edge in chosen),
            "total_normalized_cost": float(sum(
                edge["normalized_cost"] for edge in chosen)),
            "discovery_status": "eligible" if not reasons else "rejected",
            "discovery_reason": "eligible" if not reasons else "|".join(reasons),
        })
        for order, edge in enumerate(chosen):
            memberships.append({
                "cycle_id": cycle_id, "cycle_order": order, **edge})
    edge_table = pd.DataFrame(edges)
    cycle_table = pd.DataFrame(cycle_rows)
    member_table = pd.DataFrame(memberships)
    return edge_table, cycle_table, member_table


def _track_summaries(scored: pd.DataFrame) -> dict[int, dict]:
    result = {}
    visible = scored[scored.physically_visible.astype(bool)]
    for track, group in visible.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        first, last = group.iloc[0], group.iloc[-1]
        result[int(track)] = {
            "track": int(track), "rows": group,
            "first_frame": int(first.frame), "last_frame": int(last.frame),
            "first_x": float(first.x), "first_y": float(first.y),
            "last_x": float(last.x), "last_y": float(last.y),
            "first_radius": float(first.radius_px),
            "last_radius": float(last.radius_px),
        }
    return result


def _virtual_chains(seeds: Iterable[int], summaries: dict[int, dict],
                    params: dict) -> dict[int, list[int]]:
    """Join broken references using a global one-to-one assignment.

    Simultaneous track endings and beginnings are resolved together so two
    nearby anchors cannot independently claim the same successor.
    """
    maximum_gap = int(params.get("maximum_anchor_reference_gap_frames", 5))
    maximum_distance = float(params.get(
        "maximum_anchor_handoff_sum_radii", 1.5))
    seed_list = sorted({int(seed) for seed in seeds if int(seed) in summaries})
    chains = {seed: [seed] for seed in seed_list}
    used = set(seed_list)
    while True:
        active = sorted(chains)
        available = sorted(set(summaries) - used)
        if not active or not available:
            break
        costs = np.full((len(active), len(available)), 1e6, dtype=float)
        for row, seed in enumerate(active):
            tail = summaries[chains[seed][-1]]
            for column, track in enumerate(available):
                candidate = summaries[track]
                gap = candidate["first_frame"] - tail["last_frame"] - 1
                if not 0 <= gap <= maximum_gap:
                    continue
                distance = float(np.hypot(
                    candidate["first_x"] - tail["last_x"],
                    candidate["first_y"] - tail["last_y"]))
                scale = max(
                    candidate["first_radius"] + tail["last_radius"], 1.0)
                normalized = distance / scale
                if normalized <= maximum_distance:
                    # A tiny gap term makes equally close assignments prefer
                    # temporally adjacent references without identity order.
                    costs[row, column] = normalized + 1e-6 * gap
        rows, columns = linear_sum_assignment(costs)
        assignments = [
            (active[row], available[column])
            for row, column in zip(rows, columns)
            if costs[row, column] < 1e5]
        if not assignments:
            break
        for seed, track in assignments:
            chains[seed].append(track)
            used.add(track)
    return chains


def _component_near(frame: np.ndarray, owner: int, row,
                    search_radius: int = 6) -> np.ndarray | None:
    components, _ = ndi.label(frame == int(owner), STRUCTURE)
    component = physical._component_at(
        components, float(row.x), float(row.y), search_radius)
    return None if component <= 0 else components == component


def discover_anchors(labels: np.ndarray, scored: pd.DataFrame,
                     eligible_members: pd.DataFrame, params: dict,
                     ) -> pd.DataFrame:
    summaries = _track_summaries(scored)
    minimum_span = float(params.get("minimum_anchor_span_fraction", 0.80))
    maximum_dispersion = float(params.get(
        "maximum_anchor_dispersion_radii", 2.0))
    rows = []
    direct = eligible_members[
        eligible_members.evidence_kind == "physical_track_owner_transition"]
    chains = _virtual_chains(direct.physical_track.astype(int), summaries,
                             params)
    for member in direct.itertuples(index=False):
        seed = int(member.physical_track)
        if seed not in summaries:
            continue
        chain = chains[seed]
        virtual = pd.concat([summaries[track]["rows"] for track in chain]) \
            .sort_values("frame")
        first = int(virtual.frame.min())
        last = int(virtual.frame.max())
        span_fraction = (last - first + 1) / max(len(labels), 1)
        median_x = float(virtual.x.median())
        median_y = float(virtual.y.median())
        median_radius = max(float(virtual.radius_px.median()), 1.0)
        dispersion = float(np.quantile(np.hypot(
            virtual.x.to_numpy(float) - median_x,
            virtual.y.to_numpy(float) - median_y), 0.90) / median_radius)
        source = int(member.source_owner)
        area_samples = []
        seed_rows = summaries[seed]["rows"]
        for row in seed_rows.itertuples(index=False):
            if (int(row.frame) >= int(member.transition_frame)
                    or int(row.accepted_owner) != source):
                continue
            component = _component_near(labels[int(row.frame)], source, row)
            if component is not None:
                area_samples.append(int(np.count_nonzero(component)))
        reasons = []
        if span_fraction < minimum_span:
            reasons.append("virtual_reference_span_too_short")
        if dispersion > maximum_dispersion:
            reasons.append("virtual_reference_not_recurrent")
        if not area_samples:
            reasons.append("ordinary_anchor_area_unavailable")
        rows.append({
            "cycle_id": member.cycle_id, "source_owner": source,
            "target_owner": int(member.target_owner),
            "transition_frame": int(member.transition_frame),
            "seed_track": seed, "virtual_tracks": "|".join(map(str, chain)),
            "virtual_first_frame": first, "virtual_last_frame": last,
            "span_fraction": span_fraction,
            "dispersion_radii": dispersion,
            "expected_area_px": float(np.median(area_samples))
            if area_samples else np.nan,
            "anchor_status": "eligible" if not reasons else "rejected",
            "anchor_reason": "eligible" if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows)


def _marker(mask: np.ndarray, x: float, y: float) -> tuple[int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    best = int(np.argmin((xx - x) ** 2 + (yy - y) ** 2))
    return int(yy[best]), int(xx[best])


def _bounded_core(mask: np.ndarray, point: tuple[int, int],
                  expected_area: float, scale: float) -> np.ndarray:
    radius = scale * _radius(expected_area)
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    core = mask & ((xx - point[1]) ** 2 + (yy - point[0]) ** 2
                   <= radius ** 2)
    parts, _ = ndi.label(core, STRUCTURE)
    component = int(parts[point])
    if component <= 0:
        return np.zeros_like(mask)
    core = parts == component
    remainder, count = ndi.label(mask & ~core, STRUCTURE)
    if count <= 1:
        return core
    values, counts = np.unique(remainder[remainder > 0], return_counts=True)
    keep = int(values[np.argmax(counts)])
    safe = mask & (remainder != keep)
    return safe if ndi.label(safe, STRUCTURE)[1] == 1 else core


def _anchor_observations(labels: np.ndarray, scored: pd.DataFrame,
                         anchors: pd.DataFrame, params: dict,
                         ) -> dict[int, list[dict]]:
    observations: dict[int, list[dict]] = {}
    search_scale = float(params.get("anchor_component_search_radius_scale", 2.0))
    for anchor in anchors[anchors.anchor_status == "eligible"].itertuples(
            index=False):
        tracks = {int(value) for value in str(anchor.virtual_tracks).split("|")}
        rows = scored[
            scored.track_id.astype(int).isin(tracks)
            & scored.physically_visible.astype(bool)]
        for row in rows.itertuples(index=False):
            frame = int(row.frame)
            plane = labels[frame]
            components, _ = ndi.label(plane > 0, STRUCTURE)
            component_id = physical._component_at(
                components, float(row.x), float(row.y),
                max(2, int(np.ceil(search_scale * float(row.radius_px)))))
            if component_id <= 0:
                continue
            host = components == component_id
            accepted = physical._disk_owner(
                plane, float(row.x), float(row.y), float(row.radius_px))
            if accepted <= 0:
                continue
            owner_components, _ = ndi.label(plane == accepted, STRUCTURE)
            owner_component = physical._component_at(
                owner_components, float(row.x), float(row.y),
                max(2, int(np.ceil(search_scale * float(row.radius_px)))))
            if owner_component <= 0:
                continue
            observations.setdefault(frame, []).append({
                "cycle_id": anchor.cycle_id,
                "origin": int(anchor.source_owner),
                "expected_area": float(anchor.expected_area_px),
                "accepted": int(accepted),
                "component_id": int(owner_component),
                "mask": owner_components == owner_component,
                "foreground_host": host,
                "x": float(row.x), "y": float(row.y),
                "radius_px": float(row.radius_px),
            })
    return observations


def _apply_anchors(candidate: np.ndarray, baseline: np.ndarray,
                   raw: np.ndarray, observations: dict[int, list[dict]],
                   params: dict) -> tuple[np.ndarray, list[dict]]:
    whole_ratio = float(params.get("maximum_isolated_anchor_area_ratio", 2.5))
    core_scale = float(params.get("anchor_core_area_scale", 1.2))
    sigma = float(params.get("watershed_sigma_px", 1.0))
    audits = []
    for frame, frame_observations in observations.items():
        groups: dict[tuple[int, int], list[dict]] = {}
        for observation in frame_observations:
            groups.setdefault((observation["accepted"],
                               observation["component_id"]), []).append(
                                   observation)
        for group in groups.values():
            host = group[0]["mask"]
            unique = {item["origin"]: item for item in group}
            before = candidate[frame].copy()
            if len(unique) == 1:
                item = next(iter(unique.values()))
                point = _marker(host, item["x"], item["y"])
                if point is None:
                    continue
                area_ratio = (np.count_nonzero(host)
                              / max(item["expected_area"], 1.0))
                recovered = (host if area_ratio <= whole_ratio else
                             _bounded_core(host, point,
                                           item["expected_area"], core_scale))
                candidate[frame][recovered] = item["origin"]
                method = ("isolated_anchor_component" if area_ratio <= whole_ratio
                          else "bounded_anchor_core")
            else:
                marker_labels = np.zeros(host.shape, np.int16)
                index_to_origin = {}
                for index, item in enumerate(unique.values(), start=1):
                    point = _marker(host, item["x"], item["y"])
                    if point is None or marker_labels[point] > 0:
                        continue
                    marker_labels[point] = index
                    index_to_origin[index] = item["origin"]
                if len(index_to_origin) != len(unique):
                    continue
                elevation = -ndi.gaussian_filter(
                    raw[frame].astype(np.float32), sigma)
                partition = watershed(
                    elevation, marker_labels, mask=host,
                    connectivity=STRUCTURE)
                for index, origin in index_to_origin.items():
                    candidate[frame][partition == index] = origin
                method = "raw_multi_anchor_partition"
            audits.append({
                "frame": frame, "origins": "|".join(
                    map(str, sorted(unique))), "operation": method,
                "changed_pixels": int(np.count_nonzero(
                    candidate[frame] != before)),
            })
    return candidate, audits


def _substantial_components(frame: np.ndarray, owner: int,
                            minimum_area: int) -> list[np.ndarray]:
    return [item["mask"] for item in _components(
        frame, owner, minimum_area)]


def _bracket_owner(labels: np.ndarray, group: pd.DataFrame, frame: int,
                   transient: int) -> int:
    observations = []
    for row in group.sort_values("frame").itertuples(index=False):
        if str(row.state) not in VISIBLE_STATES:
            continue
        owner = physical._disk_owner(
            labels[int(row.frame)], float(row.x), float(row.y),
            float(row.radius_px))
        if owner > 0 and owner != transient:
            observations.append((int(row.frame), owner))
    before = [item for item in observations if item[0] < frame]
    after = [item for item in observations if item[0] > frame]
    left = before[-1][1] if before else 0
    right = after[0][1] if after else 0
    return left if left > 0 and left == right else 0


def _geometric_side_owner(candidate: np.ndarray, start_mask: np.ndarray,
                          frame: int, transient: int, direction: int,
                          params: dict) -> int:
    mask = start_mask
    x, y = _center(mask)
    area = float(np.count_nonzero(mask))
    maximum_frames = int(params.get("maximum_duplicate_bracket_frames", 8))
    maximum_cost = float(params.get("maximum_duplicate_path_cost", 2.5))
    ambiguity = float(params.get("minimum_duplicate_path_margin", 1.15))
    minimum_area = int(params.get("minimum_component_area_px", 8))
    for step in range(1, maximum_frames + 1):
        other_frame = frame + direction * step
        if other_frame < 0 or other_frame >= len(candidate):
            return -1
        choices = []
        for item in _frame_components(candidate[other_frame], minimum_area):
            distance = float(np.hypot(
                item["center"][0] - x, item["center"][1] - y))
            cost = distance / max(_radius(area) + item["radius"], 1.0)
            if cost <= maximum_cost:
                choices.append((cost, item))
        choices.sort(key=lambda item: item[0])
        if not choices:
            continue
        if len(choices) > 1 and choices[1][0] <= ambiguity * choices[0][0]:
            return 0
        item = choices[0][1]
        mask = item["mask"]
        x, y = item["center"]
        area = float(item["area"])
        if int(item["owner"]) != transient:
            return int(item["owner"])
    return 0


def _release_duplicates(candidate: np.ndarray, baseline: np.ndarray,
                        points: pd.DataFrame,
                        observations: dict[int, list[dict]],
                        anchor_origins: set[int], inverse_cycle: dict[int, int],
                        params: dict) -> tuple[np.ndarray, list[dict]]:
    minimum_area = int(params.get("minimum_component_area_px", 8))
    track_groups = {int(track): group
                    for track, group in points.groupby("track_id")}
    audits = []
    for frame in range(len(candidate)):
        frame_points = points[
            (points.frame.astype(int) == frame)
            & points.state.isin(VISIBLE_STATES)]
        for origin in sorted(anchor_origins):
            parts = _substantial_components(
                candidate[frame], origin, minimum_area)
            if len(parts) <= 1:
                continue
            records = []
            for component in parts:
                is_protected = False
                distance = ndi.distance_transform_edt(~component)
                for observation in observations.get(frame, []):
                    if int(observation["origin"]) != origin:
                        continue
                    y = int(np.clip(round(float(observation["y"])), 0,
                                    component.shape[0] - 1))
                    x = int(np.clip(round(float(observation["x"])), 0,
                                    component.shape[1] - 1))
                    if float(distance[y, x]) <= max(
                            2.0, float(observation["radius_px"])):
                        is_protected = True
                        break
                tracks = []
                for row in frame_points.itertuples(index=False):
                    y = int(np.clip(round(float(row.y)), 0,
                                    component.shape[0] - 1))
                    x = int(np.clip(round(float(row.x)), 0,
                                    component.shape[1] - 1))
                    if float(distance[y, x]) <= max(2.0, float(row.radius_px)):
                        tracks.append(int(row.track_id))
                inferred = {
                    _bracket_owner(baseline, track_groups[track], frame, origin)
                    for track in tracks if track in track_groups}
                inferred.discard(0)
                inferred = {inverse_cycle.get(owner, owner)
                            for owner in inferred}
                records.append((component, inferred, is_protected))
            if sum(record[2] for record in records) != 1:
                continue
            for component, inferred, protected in records:
                if protected:
                    continue
                replacement = next(iter(inferred)) if len(inferred) == 1 else 0
                if replacement <= 0:
                    left = _geometric_side_owner(
                        candidate, component, frame, origin, -1, params)
                    right = _geometric_side_owner(
                        candidate, component, frame, origin, 1, params)
                    if left > 0 and right > 0 and left == right:
                        replacement = left
                    elif left == 0 and right > 0:
                        replacement = right
                    elif right == 0 and left > 0:
                        replacement = left
                    elif left > 0 and right == -1:
                        replacement = left
                    elif right > 0 and left == -1:
                        replacement = right
                if replacement > 0 and replacement != origin:
                    before = candidate[frame].copy()
                    candidate[frame][component] = replacement
                    audits.append({
                        "frame": frame, "duplicate_owner": origin,
                        "restored_owner": replacement,
                        "changed_pixels": int(np.count_nonzero(
                            candidate[frame] != before)),
                    })
    return candidate, audits


def significant_duplicate_excess(labels: np.ndarray,
                                 minimum_area: int) -> int:
    return component_accounting.total_component_excess(
        labels, minimum_area=minimum_area)


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                       pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Discover and atomically repair every uniquely supported boundary cycle."""
    params = resolve_data_relative_params(points, len(labels), params)
    assert_target_free(params)
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed, and raw stacks must align")
    edges, cycles, members = discover_cycles(labels, points, params)
    eligible = cycles[cycles.discovery_status == "eligible"] \
        if len(cycles) else cycles
    scored = physical.attach_owners(points, labels)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    application_rows = []
    anchor_rows = []
    anchor_audits: list[dict] = []
    duplicate_audits: list[dict] = []

    # Distinct connected cycles cannot share an owner. Refuse overlaps rather
    # than resolving them by numeric identity order.
    occupied_owners: set[int] = set()
    ordered_eligible = (eligible.sort_values(
        ["owner_count", "direct_edges", "total_normalized_cost"],
        ascending=[False, False, True]) if not eligible.empty else eligible)
    for cycle in ordered_eligible.itertuples(index=False):
        cycle_members = members[members.cycle_id == cycle.cycle_id].copy()
        owners = set(cycle_members.source_owner.astype(int))
        if owners & occupied_owners:
            application_rows.append({
                "cycle_id": cycle.cycle_id, "outcome": "rejected_application",
                "reason": "cycle_owner_overlap", "changed_pixels": 0,
                "changed_frames": 0,
            })
            continue
        anchors = discover_anchors(labels, scored, cycle_members, params)
        anchor_rows.extend(anchors.to_dict(orient="records"))
        eligible_anchors = anchors[anchors.anchor_status == "eligible"]
        minimum_anchors = int(params.get("minimum_recurrent_anchors", 2))
        minimum_nonanchors = int(params.get("minimum_moving_cycle_branches", 2))
        if (len(eligible_anchors) < minimum_anchors
                or len(cycle_members) - len(eligible_anchors) < minimum_nonanchors):
            application_rows.append({
                "cycle_id": cycle.cycle_id, "outcome": "rejected_application",
                "reason": "anchor_moving_branch_balance_not_supported",
                "changed_pixels": 0, "changed_frames": 0,
            })
            continue
        anchor_sources = set(eligible_anchors.source_owner.astype(int))
        inverse = {
            int(row.target_owner): int(row.source_owner)
            for row in cycle_members.itertuples(index=False)}
        before_cycle = candidate.copy()
        original = labels
        # Non-recurrent branches have unique durable owner lineages; apply each
        # inverse edge simultaneously from its observed transition onward.
        for member in cycle_members.itertuples(index=False):
            source = int(member.source_owner)
            if source in anchor_sources:
                continue
            target = int(member.target_owner)
            for frame in range(int(member.transition_frame), len(labels)):
                candidate[frame][original[frame] == target] = source
        observations = _anchor_observations(
            labels, scored, eligible_anchors, params)
        candidate, applied_anchors = _apply_anchors(
            candidate, labels, raw, observations, params)
        candidate, released = _release_duplicates(
            candidate, labels, points, observations, anchor_sources,
            inverse, params)
        anchor_audits.extend({"cycle_id": cycle.cycle_id, **row}
                             for row in applied_anchors)
        duplicate_audits.extend({"cycle_id": cycle.cycle_id, **row}
                                for row in released)
        if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
            candidate = before_cycle
            application_rows.append({
                "cycle_id": cycle.cycle_id, "outcome": "rejected_application",
                "reason": "active_identity_set_changed", "changed_pixels": 0,
                "changed_frames": 0,
            })
            continue
        occupied_owners |= owners
        changed = candidate != before_cycle
        application_rows.append({
            "cycle_id": cycle.cycle_id, "outcome": "applied",
            "reason": "closed_boundary_permutation_reconciled",
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(
                np.any(changed, axis=(1, 2)))),
        })

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("owner-cycle recovery changed foreground")
    if not np.array_equal(candidate_unclaimed, unclaimed):
        raise AssertionError("owner-cycle recovery changed unclaimed ledger")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("owner-cycle recovery changed active identities")
    minimum_area = int(params.get("minimum_component_area_px", 8))
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "identity_targets": 0, "track_targets": 0, "frame_targets": 0,
        "coordinate_targets": 0, "event_targets": 0,
        "review_case_targets_received": False,
        "audited_edges": int(len(edges)),
        "audited_cycles": int(len(cycles)),
        "eligible_cycles": int(len(eligible)),
        "applied_cycles": int(sum(
            row["outcome"] == "applied" for row in application_rows)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "unclaimed_changed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "active_identity_set_preserved": bool(
            set(map(int, np.unique(candidate)))
            == set(map(int, np.unique(labels)))),
        "significant_duplicate_excess_before": significant_duplicate_excess(
            labels, minimum_area),
        "significant_duplicate_excess_after": significant_duplicate_excess(
            candidate, minimum_area),
    }
    return (candidate, candidate_unclaimed, edges, cycles, members,
            pd.DataFrame(anchor_rows), pd.DataFrame(application_rows),
            {"metrics": metrics,
             "anchor_frame_audit": pd.DataFrame(anchor_audits),
             "duplicate_release_audit": pd.DataFrame(duplicate_audits)})
