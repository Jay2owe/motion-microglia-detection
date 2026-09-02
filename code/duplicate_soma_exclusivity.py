"""Capacity-one identity projection for independently visible soma bodies.

The implementation is field-wide and equivariant to identity renaming. It has
no identity, track, frame, coordinate, event, region, or review-case inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import separable_merge_recovery as physical
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
VISIBLE_STATES = {"observed", "latent_visible"}
FORBIDDEN_PARAMETER_PARTS = (
    "identity", "track_target", "frame_target", "coordinate", "region",
    "event_target", "review_case")


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("duplicate-soma exclusivity must be field-wide")
    for key in params:
        lowered = str(key).lower()
        if any(part in lowered for part in FORBIDDEN_PARAMETER_PARTS):
            raise ValueError(f"target-like producer parameter is forbidden: {key}")


@dataclass
class Component:
    frame: int
    owner: int
    index: int
    component_map: np.ndarray
    component_id: int
    area: int
    x: float
    y: float
    radius: float
    tracks: tuple[int, ...]
    evidence: float
    core_x: float
    core_y: float
    core_radius: float

    @cached_property
    def mask(self) -> np.ndarray:
        """Materialize a full-frame mask only for a component being changed."""
        return self.component_map == self.component_id


def _minimum_area(points: pd.DataFrame, params: dict) -> int:
    visible = points[points.state.isin(VISIBLE_STATES)]
    typical = float(np.median(np.pi * visible.radius_px.astype(float) ** 2))
    fraction = float(params.get(
        "minimum_component_area_fraction_of_median_soma", 0.15))
    return max(1, int(np.ceil(typical * fraction)))


def _component_at(parts: np.ndarray, x: float, y: float,
                  search_radius: float) -> int:
    height, width = parts.shape
    cx, cy = int(round(x)), int(round(y))
    if 0 <= cy < height and 0 <= cx < width and parts[cy, cx] > 0:
        return int(parts[cy, cx])
    radius = max(1, int(np.ceil(search_radius)))
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    yy, xx = np.nonzero(parts[y0:y1, x0:x1] > 0)
    if not len(xx):
        return 0
    distances = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
    best = int(np.argmin(distances))
    return int(parts[yy[best] + y0, xx[best] + x0])


def _components(labels: np.ndarray, scored: pd.DataFrame,
                minimum_area: int,
                frame_ids: np.ndarray | None = None,
                ) -> dict[tuple[int, int], list[Component]]:
    visible = scored[
        scored.physically_visible.astype(bool)
        & (scored.accepted_owner.astype(int) > 0)]
    # A pandas groupby followed by ``itertuples`` constructs one temporary
    # frame and one named-tuple class for every owner-frame pair.  The component
    # lookup needs only stable row order, so index the rows in one pass instead.
    grouped_points: dict[tuple[int, int], list] = {}
    for row in visible.itertuples(index=False):
        key = (int(row.frame), int(row.accepted_owner))
        grouped_points.setdefault(key, []).append(row)
    by_frame_owner = {key: tuple(rows)
                      for key, rows in grouped_points.items()}
    result: dict[tuple[int, int], list[Component]] = {}
    frames = (range(len(labels)) if frame_ids is None
              else map(int, np.asarray(frame_ids).tolist()))
    for frame in frames:
        value_parts, component_owners, component_areas = \
            component_accounting.label_value_components(labels[frame])
        component_slices = ndi.find_objects(value_parts)
        for owner in map(int, np.unique(labels[frame])):
            if owner <= 0:
                continue
            component_ids = np.flatnonzero(component_owners == owner) + 1
            parts = np.where(labels[frame] == owner, value_parts, 0)
            points = by_frame_owner.get((frame, owner), ())
            rows: list[Component] = []
            for index, component_id in enumerate(component_ids, start=1):
                area = int(component_areas[int(component_id) - 1])
                if area < minimum_area:
                    continue
                component_slice = component_slices[int(component_id) - 1]
                if component_slice is None:
                    raise AssertionError("labelled component has no bounding box")
                local_y, local_x = np.nonzero(
                    value_parts[component_slice] == int(component_id))
                yy = local_y + int(component_slice[0].start)
                xx = local_x + int(component_slice[1].start)
                support = []
                for row in points:
                    if _component_at(parts, float(row.x), float(row.y),
                                     max(2.0, float(row.radius_px))) \
                            == int(component_id):
                        support.append(row)
                if support:
                    core = max(support, key=lambda row: float(row.evidence))
                    tracks = tuple(sorted({int(row.track_id) for row in support}))
                    evidence = float(sum(float(row.evidence) for row in support))
                    core_x, core_y = float(core.x), float(core.y)
                    core_radius = float(core.radius_px)
                else:
                    tracks, evidence = (), 0.0
                    core_x, core_y = float(xx.mean()), float(yy.mean())
                    core_radius = float(np.sqrt(area / np.pi))
                rows.append(Component(
                    frame=frame, owner=owner, index=index,
                    component_map=value_parts, component_id=int(component_id),
                    area=area, x=float(xx.mean()), y=float(yy.mean()),
                    radius=float(np.sqrt(area / np.pi)), tracks=tracks,
                    evidence=evidence, core_x=core_x, core_y=core_y,
                    core_radius=core_radius))
            result[(frame, owner)] = rows
    return result


def _pair_metrics(left: Component, right: Component,
                  raw_frame: np.ndarray) -> tuple[float, float]:
    distance = float(np.hypot(left.core_x - right.core_x,
                              left.core_y - right.core_y))
    scale = max(left.core_radius + right.core_radius, 1.0)
    valley = physical._valley_ratio(
        raw_frame, (left.core_x, left.core_y),
        (right.core_x, right.core_y))
    return distance / scale, valley


def audit(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
          params: dict, frame_ids: np.ndarray | None = None,
          ) -> tuple[pd.DataFrame, pd.DataFrame,
                               dict[tuple[int, int], list[Component]]]:
    """Find physically distinct same-owner soma components in every frame."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels)
    minimum_area = _minimum_area(scored, params)
    components = _components(labels, scored, minimum_area, frame_ids)
    minimum_separation = float(params.get(
        "minimum_two_core_separation_sum_radii", 1.0))
    maximum_valley = float(params.get("maximum_two_core_valley_ratio", 0.80))
    rows: list[dict] = []
    for (frame, owner), values in components.items():
        supported = [value for value in values if value.tracks]
        qualifying_pairs = []
        for left, right in combinations(supported, 2):
            separation, valley = _pair_metrics(left, right, raw[frame])
            if separation >= minimum_separation and valley <= maximum_valley:
                qualifying_pairs.append((left, right, separation, valley))
        if not qualifying_pairs:
            continue
        involved = sorted({part.index for pair in qualifying_pairs
                           for part in pair[:2]})
        rows.append({
            "frame": frame, "owner": owner,
            "substantial_components": len(values),
            "physical_components": len(supported),
            "duplicate_physical_components": len(involved),
            "duplicate_excess": len(involved) - 1,
            "component_indices": "|".join(map(str, involved)),
            "physical_tracks": "|".join(map(str, sorted({
                track for value in supported if value.index in involved
                for track in value.tracks}))),
            "minimum_separation_sum_radii": min(
                pair[2] for pair in qualifying_pairs),
            "maximum_valley_ratio": max(pair[3] for pair in qualifying_pairs),
            "minimum_component_area_px": minimum_area,
        })
    columns = [
        "frame", "owner", "substantial_components", "physical_components",
        "duplicate_physical_components", "duplicate_excess",
        "component_indices", "physical_tracks",
        "minimum_separation_sum_radii", "maximum_valley_ratio",
        "minimum_component_area_px"]
    return pd.DataFrame(rows, columns=columns), scored, components


def _match_cost(left: Component, right: Component, gap: int) -> float:
    distance = float(np.hypot(left.x - right.x, left.y - right.y))
    scale = max(left.radius + right.radius, 1.0)
    area_cost = abs(np.log(max(left.area, 1) / max(right.area, 1)))
    return distance / scale / max(gap, 1) + 0.15 * area_cost


def _trace(seed: Component, components: dict[tuple[int, int], list[Component]],
           frame_count: int, params: dict) -> list[Component]:
    maximum_gap = max(1, int(np.ceil(frame_count * float(
        params.get("maximum_path_gap_fraction_of_movie", 0.04)))))
    maximum_cost = float(params.get("maximum_path_step_cost", 2.75))
    path = {seed.frame: seed}
    for direction in (-1, 1):
        current = seed
        while True:
            options = []
            for gap in range(1, maximum_gap + 1):
                frame = current.frame + direction * gap
                if frame < 0 or frame >= frame_count:
                    break
                for candidate in components.get((frame, seed.owner), []):
                    cost = _match_cost(current, candidate, gap)
                    if cost <= maximum_cost:
                        options.append((gap, cost, candidate))
                if options:
                    break
            if not options:
                break
            _, _, current = min(options, key=lambda item: (item[0], item[1]))
            if current.frame in path:
                break
            path[current.frame] = current
    return [path[frame] for frame in sorted(path)]


def _path_score(seed: Component,
                components: dict[tuple[int, int], list[Component]],
                duplicate_keys: set[tuple[int, int]], frame_count: int,
                track_owner_support: dict[tuple[int, int], int],
                params: dict) -> tuple[tuple[float, ...], list[Component]]:
    path = _trace(seed, components, frame_count, params)
    supported = [part for part in path if part.tracks]
    exclusive = [part for part in supported
                 if (part.frame, part.owner) not in duplicate_keys]
    span = path[-1].frame - path[0].frame + 1
    evidence = sum(part.evidence for part in supported)
    area = sum(part.area for part in supported)
    seed_owner_support = max(
        (track_owner_support.get((seed.owner, track), 0)
         for track in seed.tracks), default=0)
    # The last terms are physical evidence only. Component index, owner value,
    # track number, and image location are intentionally absent.
    return (float(len(exclusive) + seed_owner_support),
            float(len(exclusive)), float(seed_owner_support),
            float(len(supported)), float(span), float(evidence),
            float(area)), path


def _alternative_owner(scored: pd.DataFrame, tracks: tuple[int, ...],
                       transient_owner: int, frame: int,
                       candidate_frame: np.ndarray, params: dict) -> int:
    if not tracks:
        return 0
    rows = scored[
        scored.track_id.astype(int).isin(tracks)
        & scored.physically_visible.astype(bool)]
    other = rows[(rows.accepted_owner.astype(int) > 0)
                 & (rows.accepted_owner.astype(int) != transient_owner)]
    counts = other.accepted_owner.astype(int).value_counts()
    minimum = max(2, int(np.ceil(len(candidate_frame.shape) * 0.0 +
                                 float(params.get(
                                     "minimum_alternative_owner_support", 2)))))
    if not len(counts) or int(counts.iloc[0]) < minimum:
        return 0
    if len(counts) > 1 and int(counts.iloc[0]) == int(counts.iloc[1]):
        return 0
    owner = int(counts.index[0])
    purity = float(counts.iloc[0] / max(int(counts.sum()), 1))
    if purity < float(params.get("minimum_alternative_owner_purity", 0.60)):
        return 0
    # Capacity one also applies to the alternative owner.
    if np.any(candidate_frame == owner):
        return 0
    return owner


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict):
    before, scored, components = audit(labels, raw, points, params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    applications: list[dict] = []
    if before.empty:
        return (candidate, candidate_unclaimed, before, before.copy(),
                pd.DataFrame(), scored)
    duplicate_keys = set(zip(before.frame.astype(int), before.owner.astype(int)))
    visible_owned = scored[
        scored.physically_visible.astype(bool)
        & (scored.accepted_owner.astype(int) > 0)]
    track_owner_support = {
        (int(owner), int(track)): int(len(group))
        for (owner, track), group in visible_owned.groupby(
            ["accepted_owner", "track_id"])}
    for row in before.sort_values(["frame", "owner"]).itertuples(index=False):
        key = (int(row.frame), int(row.owner))
        involved = {int(value) for value in str(row.component_indices).split("|")}
        candidates = [part for part in components[key] if part.index in involved]
        ranked = []
        for part in candidates:
            score, path = _path_score(
                part, components, duplicate_keys, len(labels),
                track_owner_support, params)
            ranked.append((score, part, path))
        ranked.sort(key=lambda item: item[0], reverse=True)
        winner_score, winner, _ = ranked[0]
        for loser_score, loser, _ in ranked[1:]:
            replacement = _alternative_owner(
                scored, loser.tracks, int(row.owner), int(row.frame),
                candidate[int(row.frame)], params)
            mask = loser.mask & (candidate[int(row.frame)] == int(row.owner))
            changed = int(np.count_nonzero(mask))
            if replacement > 0:
                candidate[int(row.frame)][mask] = replacement
                candidate_unclaimed[int(row.frame)][mask] = 0
                method = "restore_unique_existing_owner"
            else:
                candidate[int(row.frame)][mask] = 0
                candidate_unclaimed[int(row.frame)][mask] = int(row.owner)
                method = "release_duplicate_to_unclaimed_ledger"
            applications.append({
                "frame": int(row.frame), "owner": int(row.owner),
                "winner_tracks": "|".join(map(str, winner.tracks)),
                "loser_tracks": "|".join(map(str, loser.tracks)),
                "replacement_owner": replacement, "method": method,
                "changed_pixels": changed,
                "winner_combined_tenure": winner_score[0],
                "loser_combined_tenure": loser_score[0],
                "winner_exclusive_path_tenure": winner_score[1],
                "loser_exclusive_path_tenure": loser_score[1],
                "winner_seed_track_owner_tenure": winner_score[2],
                "loser_seed_track_owner_tenure": loser_score[2],
                "winner_total_path_support": winner_score[3],
                "loser_total_path_support": loser_score[3],
            })
    # Only changed frames can gain or lose a duplicate component. Recompute
    # those frames and retain the byte-equivalent rows from the baseline audit
    # everywhere else.
    changed_frames = np.flatnonzero(np.any(
        candidate != labels, axis=(1, 2)))
    if len(changed_frames):
        changed_after, _, _ = audit(
            candidate, raw, points, params, frame_ids=changed_frames)
        unchanged_before = before[
            ~before.frame.astype(int).isin(changed_frames)]
        after = pd.concat(
            [unchanged_before, changed_after], ignore_index=True
        ).sort_values(["frame", "owner"], kind="stable").reset_index(drop=True)
    else:
        after = before.copy()
    return (candidate, candidate_unclaimed, before, after,
            pd.DataFrame(applications), scored)


def summarize(labels: np.ndarray, unclaimed: np.ndarray,
              candidate: np.ndarray, candidate_unclaimed: np.ndarray,
              before: pd.DataFrame, after: pd.DataFrame,
              applications: pd.DataFrame, mode: str) -> dict:
    union_before = (labels > 0) | (unclaimed > 0)
    union_after = (candidate > 0) | (candidate_unclaimed > 0)
    ids_before = set(map(int, np.unique(labels))) - {0}
    ids_after = set(map(int, np.unique(candidate))) - {0}
    presence_before = {owner: np.any(labels == owner, axis=(1, 2))
                       for owner in ids_before}
    presence_preserved = all(np.all(
        presence_before[owner] <= np.any(candidate == owner, axis=(1, 2)))
        for owner in ids_before)
    return {
        "mode": mode, "targeting_mode": "field_wide_discovery",
        "identity_targets": 0, "track_targets": 0, "frame_targets": 0,
        "coordinate_targets": 0, "event_targets": 0, "region_targets": 0,
        "review_case_targets": 0,
        "physical_duplicate_rows_before": int(len(before)),
        "physical_duplicate_excess_before": int(
            before.duplicate_excess.sum()) if len(before) else 0,
        "physical_duplicate_rows_after": int(len(after)),
        "physical_duplicate_excess_after": int(
            after.duplicate_excess.sum()) if len(after) else 0,
        "applications": int(len(applications)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_ledger_union_exact": bool(np.array_equal(
            union_before, union_after)),
        "active_identity_set_exact": ids_before == ids_after,
        "baseline_owner_frame_presence_preserved": bool(presence_preserved),
        "new_identities": sorted(ids_after - ids_before),
    }
