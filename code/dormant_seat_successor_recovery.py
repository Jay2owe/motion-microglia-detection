"""Consolidate a movie-new successor onto a dormant established seat.

The detector searches every physical track for a durable owner followed by a
long, raw-visible terminal dropout.  It accepts a continuation only when one
movie-new successor starts at the old endpoint, is substantially closer than
every alternative, and has durable internally consistent support.  Uniquely
linked, mostly ownerless track fragments extend the same physical lineage.

Discovery is field-wide.  Identity, track, frame, coordinate, event, region,
well and review selectors are forbidden.  Identity numbers are outputs only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import component_accounting
from ownerless_body_recovery import evidence_image


STRUCTURE = np.ones((3, 3), np.uint8)
CONTROL_KEYS = {"mode", "targeting_mode", "output_stem"}
PATH_KEYS = {
    "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path", "thresholds_path",
}
PARAMETER_KEYS = {
    "minimum_anchor_support_movie_fraction",
    "minimum_anchor_owner_purity",
    "minimum_owner_movie_presence_fraction",
    "minimum_terminal_dropout_movie_fraction",
    "minimum_terminal_visibility_fraction",
    "maximum_successor_gap_frames",
    "maximum_successor_endpoint_sum_radii",
    "minimum_successor_margin_sum_radii",
    "minimum_successor_support_movie_fraction",
    "minimum_successor_owner_purity",
    "minimum_successor_visibility_fraction",
    "maximum_chain_gap_frames",
    "maximum_chain_endpoint_sum_radii",
    "minimum_chain_margin_sum_radii",
    "minimum_chain_visibility_fraction",
    "maximum_chain_foreign_owner_fraction",
    "minimum_host_support_movie_fraction",
    "minimum_host_owner_purity",
    "maximum_host_point_distance_radii",
    "maximum_projection_component_area_ratio",
    "maximum_projection_gap_radii",
    "minimum_projection_host_margin_radii",
    "maximum_component_point_distance_radii",
    "minimum_raw_core_area_radius_ratio",
    "maximum_raw_core_area_radius_ratio",
    "maximum_core_offset_radii",
    "core_window_radius_px",
    "core_peak_fraction",
    "weak_threshold_fraction",
    "point_core_radius_scale",
    "minimum_point_core_free_fraction",
}
ALLOWED_KEYS = CONTROL_KEYS | PATH_KEYS | PARAMETER_KEYS
AUDIT_COLUMNS = [
    "proposal_id", "predecessor_track", "predecessor_owner",
    "successor_track", "successor_owner", "predecessor_first_frame",
    "predecessor_last_owner_frame", "predecessor_last_frame",
    "anchor_support_frames", "anchor_owner_purity",
    "terminal_dropout_frames", "terminal_visibility_fraction",
    "successor_first_frame", "successor_last_frame",
    "successor_support_frames", "successor_owner_purity",
    "successor_movie_novel", "successor_endpoint_sum_radii",
    "successor_next_best_sum_radii", "successor_margin_sum_radii",
    "chain_tracks", "chain_last_frame", "eligible", "reason", "applied",
]
LINEAGE_COLUMNS = [
    "proposal_id", "track_id", "role", "first_frame", "last_frame",
    "visible_frames", "positive_owner_frames", "dominant_owner",
    "dominant_owner_support", "dominant_owner_purity",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "track_id", "predecessor_owner",
    "observed_owner", "operation", "changed_pixels",
    "raw_core_area_radius_ratio", "proof",
]
PROJECTION_COLUMNS = [
    "proposal_id", "frame", "source_owner", "host_owner", "host_track",
    "source_component_area", "host_component_area", "component_area_ratio",
    "component_gap_radii", "host_owner_support", "host_owner_purity",
    "proof",
]


@dataclass(frozen=True)
class InterpolatedPoint:
    frame: int
    x: float
    y: float
    radius_px: float
    candidate_owner: int = 0
    physically_visible: bool = True


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {
        "1", "true", "yes"}


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("dormant-seat successor recovery must be field-wide")
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported parameters (selectors are forbidden): "
            + ", ".join(unsupported))
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing input paths: " + ", ".join(missing))


def selector_counts(params: dict) -> dict[str, int]:
    unsupported = [str(key).lower() for key in params if key not in ALLOWED_KEYS]
    roles = {
        "identities": ("identity", "identities", "cell", "cells"),
        "owners": ("owner", "owners"),
        "tracks": ("track", "tracks"),
        "frames": ("frame", "frames", "interval"),
        "coordinates": ("coordinate", "coordinates", "point", "points"),
        "events": ("event", "events"),
        "regions": ("region", "regions", "roi"),
        "wells": ("well", "video", "recording"),
        "review_cases": ("case", "cases", "review"),
    }
    return {role: sum(any(token in key for token in tokens)
                      for key in unsupported)
            for role, tokens in roles.items()}


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _visible(group: pd.DataFrame) -> pd.DataFrame:
    if "physically_visible" in group:
        mask = group.physically_visible.map(_truth)
    else:
        mask = group.state.astype(str).isin(["observed", "latent_visible"])
    return group[mask].sort_values("frame").copy()


def _profile(group: pd.DataFrame) -> tuple[int, int, float]:
    positive = group[group.candidate_owner.astype(int).gt(0)]
    if positive.empty:
        return 0, 0, 0.0
    counts = positive.candidate_owner.astype(int).value_counts()
    owner = int(counts.index[0])
    support = int(counts.iloc[0])
    return owner, support, float(support / len(positive))


def _presence(labels: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for frame, plane in enumerate(labels):
        for owner in map(int, np.unique(plane)):
            if owner > 0:
                result[owner].add(int(frame))
    return dict(result)


def _endpoint_distance(left: object, right: object) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(
        float(left.radius_px) + float(right.radius_px), 1.0)


def _consecutive(group: pd.DataFrame) -> bool:
    frames = group.frame.astype(int).to_numpy()
    return len(frames) <= 1 or bool(np.all(np.diff(frames) == 1))


def _starting_candidates(
        by_track: dict[int, pd.DataFrame], used: set[int], endpoint: object,
        maximum_gap: int) -> list[tuple[float, int, pd.DataFrame]]:
    rows = []
    endpoint_frame = int(endpoint.frame)
    for track_id, group in by_track.items():
        if track_id in used or group.empty:
            continue
        start = group.iloc[0]
        gap = int(start.frame) - endpoint_frame
        if 1 <= gap <= maximum_gap:
            rows.append((_endpoint_distance(endpoint, start), track_id, group))
    return sorted(rows, key=lambda item: (item[0], item[1]))


def _extend_chain(
        by_track: dict[int, pd.DataFrame], root_tracks: list[int],
        params: dict) -> list[int]:
    maximum_gap = int(params.get("maximum_chain_gap_frames", 2))
    maximum_distance = float(params.get(
        "maximum_chain_endpoint_sum_radii", 1.50))
    minimum_margin = float(params.get(
        "minimum_chain_margin_sum_radii", 1.00))
    minimum_visibility = _fraction(
        params, "minimum_chain_visibility_fraction", 0.80)
    maximum_foreign = _fraction(
        params, "maximum_chain_foreign_owner_fraction", 0.10)
    result = list(root_tracks)
    used = set(result)
    while True:
        current = by_track[result[-1]]
        choices = _starting_candidates(
            by_track, used, current.iloc[-1], maximum_gap)
        if not choices:
            break
        distance, track_id, group = choices[0]
        next_distance = choices[1][0] if len(choices) > 1 else float("inf")
        positive_fraction = float(
            group.candidate_owner.astype(int).gt(0).mean())
        visible_fraction = float(len(_visible(group)) / max(len(group), 1))
        if (distance > maximum_distance
                or next_distance - distance < minimum_margin
                or visible_fraction < minimum_visibility
                or positive_fraction > maximum_foreign):
            break
        result.append(int(track_id))
        used.add(int(track_id))
    return result


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Audit all tracks and return uniquely proved dormant-seat lineages."""
    frame_count = len(labels)
    presence = _presence(labels)
    visible = _visible(points)
    by_track = {int(track): group.sort_values("frame").copy()
                for track, group in visible.groupby("track_id", sort=True)}
    minimum_anchor = int(math.ceil(frame_count * _fraction(
        params, "minimum_anchor_support_movie_fraction", 0.20)))
    minimum_owner_presence = _fraction(
        params, "minimum_owner_movie_presence_fraction", 0.25)
    minimum_tail = int(math.ceil(frame_count * _fraction(
        params, "minimum_terminal_dropout_movie_fraction", 0.08)))
    minimum_tail_visibility = _fraction(
        params, "minimum_terminal_visibility_fraction", 0.90)
    minimum_anchor_purity = _fraction(
        params, "minimum_anchor_owner_purity", 0.90)
    maximum_gap = int(params.get("maximum_successor_gap_frames", 2))
    maximum_distance = float(params.get(
        "maximum_successor_endpoint_sum_radii", 1.50))
    minimum_margin = float(params.get(
        "minimum_successor_margin_sum_radii", 1.00))
    minimum_successor = int(math.ceil(frame_count * _fraction(
        params, "minimum_successor_support_movie_fraction", 0.10)))
    minimum_successor_purity = _fraction(
        params, "minimum_successor_owner_purity", 0.95)
    minimum_successor_visibility = _fraction(
        params, "minimum_successor_visibility_fraction", 0.90)

    audits: list[dict] = []
    lineages: list[dict] = []
    lineage_rows: list[dict] = []
    proposal_number = 0
    for track_id, group in by_track.items():
        owner, support, purity = _profile(group)
        if owner <= 0:
            continue
        owner_rows = group[group.candidate_owner.astype(int).eq(owner)]
        last_owner = int(owner_rows.frame.max())
        tail = group[group.frame.astype(int).gt(last_owner)]
        # Only audit terminal ownerless runs; internal gaps are projections or
        # temporary segmentation failures and are outside this rule.
        is_terminal_dropout = (
            len(tail) >= minimum_tail
            and bool(tail.candidate_owner.astype(int).eq(0).all())
            and int(tail.frame.min()) == last_owner + 1
            and int(tail.frame.max()) == int(group.frame.max())
            and _consecutive(tail))
        if not is_terminal_dropout:
            continue
        proposal_number += 1
        proposal_id = f"DS{proposal_number:04d}"
        choices = _starting_candidates(
            by_track, {track_id}, group.iloc[-1], maximum_gap)
        closest = choices[0] if choices else None
        next_distance = choices[1][0] if len(choices) > 1 else float("inf")
        reasons = []
        if support < minimum_anchor:
            reasons.append("anchor_support_too_short")
        if purity < minimum_anchor_purity:
            reasons.append("anchor_owner_impure")
        if len(presence.get(owner, set())) / frame_count < minimum_owner_presence:
            reasons.append("anchor_owner_not_movie_durable")
        tail_visibility = float(len(tail) / max(
            int(tail.frame.max()) - int(tail.frame.min()) + 1, 1))
        if tail_visibility < minimum_tail_visibility:
            reasons.append("terminal_dropout_not_consistently_visible")
        successor_track = successor_owner = successor_support = 0
        successor_purity = 0.0
        successor_first = successor_last = -1
        successor_distance = float("inf")
        successor_novel = False
        successor_visibility = 0.0
        successor_group = None
        if closest is None:
            reasons.append("no_successor_start_near_endpoint")
        else:
            successor_distance, successor_track, successor_group = closest
            successor_owner, successor_support, successor_purity = _profile(
                successor_group)
            successor_first = int(successor_group.frame.min())
            successor_last = int(successor_group.frame.max())
            successor_visibility = float(
                len(_visible(successor_group)) / max(len(successor_group), 1))
            successor_novel = (
                successor_owner > 0
                and min(presence.get(successor_owner, {frame_count}))
                == successor_first)
            if successor_distance > maximum_distance:
                reasons.append("successor_endpoint_too_distant")
            if next_distance - successor_distance < minimum_margin:
                reasons.append("successor_endpoint_not_unique")
            if successor_owner <= 0:
                reasons.append("successor_has_no_durable_owner")
            elif successor_owner == owner:
                reasons.append("successor_already_has_anchor_owner")
            if successor_support < minimum_successor:
                reasons.append("successor_support_too_short")
            if successor_purity < minimum_successor_purity:
                reasons.append("successor_owner_impure")
            if successor_visibility < minimum_successor_visibility:
                reasons.append("successor_not_consistently_visible")
            if not successor_novel:
                reasons.append("successor_not_movie_novel_at_start")
        eligible = not reasons
        chain_tracks: list[int] = []
        if eligible and successor_group is not None:
            chain_tracks = _extend_chain(
                by_track, [int(successor_track)], params)
            chain_last = int(by_track[chain_tracks[-1]].frame.max())
            lineages.append({
                "proposal_id": proposal_id,
                "predecessor_track": int(track_id),
                "predecessor_owner": int(owner),
                "successor_owner": int(successor_owner),
                "tail_first_frame": int(tail.frame.min()),
                "tracks": [int(track_id)] + chain_tracks,
                "groups": [tail] + [by_track[value] for value in chain_tracks],
            })
            members = [(track_id, "terminal_dropout", tail)] + [
                (value, "movie_new_successor" if index == 0
                 else "unique_ownerless_continuation", by_track[value])
                for index, value in enumerate(chain_tracks)]
            for member_track, role, member_group in members:
                member_owner, member_support, member_purity = _profile(
                    member_group)
                lineage_rows.append({
                    "proposal_id": proposal_id, "track_id": member_track,
                    "role": role,
                    "first_frame": int(member_group.frame.min()),
                    "last_frame": int(member_group.frame.max()),
                    "visible_frames": int(len(member_group)),
                    "positive_owner_frames": int(
                        member_group.candidate_owner.astype(int).gt(0).sum()),
                    "dominant_owner": member_owner,
                    "dominant_owner_support": member_support,
                    "dominant_owner_purity": member_purity,
                })
        else:
            chain_last = successor_last
        audits.append({
            "proposal_id": proposal_id,
            "predecessor_track": int(track_id),
            "predecessor_owner": int(owner),
            "successor_track": int(successor_track),
            "successor_owner": int(successor_owner),
            "predecessor_first_frame": int(group.frame.min()),
            "predecessor_last_owner_frame": last_owner,
            "predecessor_last_frame": int(group.frame.max()),
            "anchor_support_frames": support,
            "anchor_owner_purity": purity,
            "terminal_dropout_frames": int(len(tail)),
            "terminal_visibility_fraction": tail_visibility,
            "successor_first_frame": successor_first,
            "successor_last_frame": successor_last,
            "successor_support_frames": successor_support,
            "successor_owner_purity": successor_purity,
            "successor_movie_novel": successor_novel,
            "successor_endpoint_sum_radii": successor_distance,
            "successor_next_best_sum_radii": next_distance,
            "successor_margin_sum_radii": next_distance - successor_distance,
            "chain_tracks": "|".join(map(str, chain_tracks)),
            "chain_last_frame": chain_last,
            "eligible": eligible,
            "reason": "eligible" if eligible else ";".join(sorted(set(reasons))),
            "applied": False,
        })
    return (pd.DataFrame(audits, columns=AUDIT_COLUMNS),
            pd.DataFrame(lineage_rows, columns=LINEAGE_COLUMNS), lineages)


def _component_near(frame: np.ndarray, owner: int, point: object,
                    maximum_distance_radii: float) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    radius = max(float(point.radius_px), 1.0)
    choices = []
    for component_id in range(1, int(count) + 1):
        yy, xx = np.nonzero(components == component_id)
        if not len(xx):
            continue
        distance = float(np.sqrt(np.min(
            (yy - float(point.y)) ** 2 + (xx - float(point.x)) ** 2)))
        if distance <= maximum_distance_radii * radius:
            choices.append((distance, -len(xx), component_id))
    if not choices:
        return None
    return components == min(choices)[2]


def _component_distance(left: np.ndarray, right: np.ndarray) -> float:
    if not np.any(left) or not np.any(right):
        return float("inf")
    return float(ndi.distance_transform_edt(~left)[right].min())


def _dominant_by_track(points: pd.DataFrame) -> dict[int, tuple[int, int, float]]:
    return {int(track): _profile(_visible(group))
            for track, group in points.groupby("track_id", sort=True)}


def _remote_source_components(
        plane: np.ndarray, source_owner: int, point: object,
        maximum_distance_radii: float) -> list[np.ndarray]:
    components, count = ndi.label(plane == int(source_owner), STRUCTURE)
    local = _component_near(
        plane, source_owner, point, maximum_distance_radii)
    rows = []
    for component_id in range(1, int(count) + 1):
        mask = components == component_id
        if local is not None and np.array_equal(mask, local):
            continue
        rows.append(mask)
    return rows


def _host_for_remote_component(
        plane: np.ndarray, component: np.ndarray, source_owner: int,
        frame_points: pd.DataFrame,
        profiles: dict[int, tuple[int, int, float]], frame_count: int,
        params: dict) -> tuple[int, int, str, dict] | None:
    minimum_support = int(math.ceil(frame_count * _fraction(
        params, "minimum_host_support_movie_fraction", 0.20)))
    minimum_purity = _fraction(params, "minimum_host_owner_purity", 0.90)
    maximum_point_distance = float(params.get(
        "maximum_host_point_distance_radii", 1.0))
    maximum_area_ratio = float(params.get(
        "maximum_projection_component_area_ratio", 0.35))
    maximum_gap = float(params.get("maximum_projection_gap_radii", 1.20))
    minimum_margin = float(params.get(
        "minimum_projection_host_margin_radii", 0.50))
    yy, xx = np.nonzero(component)
    if not len(xx):
        return None

    # Strongest proof: the foreign physical body is inside this component,
    # while its track is durably owned by another identity.
    contained = []
    for row in frame_points.itertuples(index=False):
        track_id = int(row.track_id)
        owner, support, purity = profiles.get(track_id, (0, 0, 0.0))
        if owner <= 0 or owner == source_owner:
            continue
        distance = float(np.sqrt(np.min(
            (yy - float(row.y)) ** 2 + (xx - float(row.x)) ** 2)))
        radius = max(float(row.radius_px), 1.0)
        if (support >= minimum_support and purity >= minimum_purity
                and distance / radius <= maximum_point_distance):
            contained.append((distance / radius, track_id, owner,
                              support, purity))
    if len(contained) == 1:
        distance, track_id, owner, support, purity = contained[0]
        return owner, track_id, "durable_foreign_body_inside_component", {
            "host_component_area": 0,
            "component_area_ratio": float("nan"),
            "component_gap_radii": distance,
            "host_owner_support": support,
            "host_owner_purity": purity,
        }
    if len(contained) > 1:
        return None

    # Detached projections may sit just beyond the host's physical core.  The
    # component must be small relative to a unique durable host component.
    adjacent = []
    for row in frame_points.itertuples(index=False):
        track_id = int(row.track_id)
        owner, support, purity = profiles.get(track_id, (0, 0, 0.0))
        if owner <= 0 or owner == source_owner:
            continue
        host = _component_near(plane, owner, row, 1.0)
        if host is None:
            continue
        radius = max(float(row.radius_px), 1.0)
        gap = _component_distance(component, host) / radius
        ratio = float(component.sum() / max(int(host.sum()), 1))
        if (support >= minimum_support and purity >= minimum_purity
                and gap <= maximum_gap and ratio <= maximum_area_ratio):
            adjacent.append((gap, track_id, owner, int(host.sum()), ratio,
                             support, purity))
    adjacent.sort(key=lambda item: (item[0], item[1]))
    if not adjacent:
        return None
    if len(adjacent) > 1 and adjacent[1][0] - adjacent[0][0] < minimum_margin:
        return None
    gap, track_id, owner, host_area, ratio, support, purity = adjacent[0]
    return owner, track_id, "small_component_adjacent_to_durable_host", {
        "host_component_area": host_area,
        "component_area_ratio": ratio,
        "component_gap_radii": gap,
        "host_owner_support": support,
        "host_owner_purity": purity,
    }


def _interpolate(left: object, right: object, frame: int) -> InterpolatedPoint:
    fraction = ((frame - int(left.frame))
                / max(int(right.frame) - int(left.frame), 1))
    return InterpolatedPoint(
        frame=frame,
        x=float(left.x) + fraction * (float(right.x) - float(left.x)),
        y=float(left.y) + fraction * (float(right.y) - float(left.y)),
        radius_px=float(left.radius_px) + fraction * (
            float(right.radius_px) - float(left.radius_px)))


def _point_centered_raw_core(
        frame: np.ndarray, evidence: np.ndarray, point: object,
        weak_threshold: float, params: dict,
        ) -> tuple[np.ndarray, tuple[slice, slice], dict] | None:
    """Rebuild the physical hypothesis at its measured point.

    Observed track points are raw-detector maxima, so this reproduces the
    detector's own connected core.  Interpolated or latent points use an
    area-constrained local rank mask, preventing a brighter neighbouring cell
    from attracting the core away from the measured seat.
    """
    radius = max(float(point.radius_px), 2.0)
    cy = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    cx = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    observed = str(getattr(point, "state", "interpolated")) == "observed"
    if observed:
        peak_y, peak_x = cy, cx
        half_window = max(int(params.get("core_window_radius_px", 7)),
                          int(math.ceil(2.0 * radius)))
        y0, y1 = max(0, peak_y - half_window), min(
            frame.shape[0], peak_y + half_window + 1)
        x0, x1 = max(0, peak_x - half_window), min(
            frame.shape[1], peak_x + half_window + 1)
        peak = float(evidence[peak_y, peak_x])
        level = max(
            float(params.get("core_peak_fraction", 0.35)) * peak,
            float(params.get("weak_threshold_fraction", 0.5))
            * float(weak_threshold))
        mask = ((evidence[y0:y1, x0:x1] >= level)
                & (frame[y0:y1, x0:x1] > 0))
        components, _ = ndi.label(mask, STRUCTURE)
        component_id = int(components[peak_y - y0, peak_x - x0])
        if component_id > 0:
            core = components == component_id
            ratio = float(core.sum() / max(math.pi * radius ** 2, 1.0))
            if ratio <= float(params.get(
                    "maximum_raw_core_area_radius_ratio", 2.0)):
                return core, (slice(y0, y1), slice(x0, x1)), {
                    "core_area_px": int(core.sum()),
                    "core_area_radius_ratio": ratio,
                    "core_offset_radii": 0.0,
                    "core_threshold": level,
                    "method": "reconstructed_observed_hypothesis",
                }

    scale = float(params.get("point_core_radius_scale", 1.75))
    half_window = int(math.ceil(scale * radius))
    y0, y1 = max(0, cy - half_window), min(
        frame.shape[0], cy + half_window + 1)
    x0, x1 = max(0, cx - half_window), min(
        frame.shape[1], cx + half_window + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = ((yy - float(point.y)) ** 2 + (xx - float(point.x)) ** 2
            <= (scale * radius) ** 2)
    eligible = disk & (frame[y0:y1, x0:x1] > 0)
    values = evidence[y0:y1, x0:x1][eligible]
    if not len(values):
        return None
    wanted = min(int(round(math.pi * radius ** 2)), len(values))
    level_index = max(0, len(values) - wanted)
    level = float(np.partition(values, level_index)[level_index])
    mask = eligible & (evidence[y0:y1, x0:x1] >= level)
    components, count = ndi.label(mask, STRUCTURE)
    choices = []
    for component_id in range(1, int(count) + 1):
        local_y, local_x = np.nonzero(components == component_id)
        if not len(local_x):
            continue
        distance = float(np.sqrt(np.min(
            (local_y + y0 - float(point.y)) ** 2
            + (local_x + x0 - float(point.x)) ** 2)))
        choices.append((distance, -len(local_x), component_id))
    if not choices:
        return None
    distance, negative_area, component_id = min(choices)
    core = components == int(component_id)
    ratio = float(core.sum() / max(math.pi * radius ** 2, 1.0))
    return core, (slice(y0, y1), slice(x0, x1)), {
        "core_area_px": int(-negative_area),
        "core_area_radius_ratio": ratio,
        "core_offset_radii": float(distance / radius),
        "core_threshold": level,
        "method": "area_constrained_point_core",
    }


def _lineage_points(lineage: dict) -> list[tuple[int, object]]:
    groups = lineage["groups"]
    points: list[tuple[int, object]] = []
    previous = None
    for group in groups:
        first = group.iloc[0]
        if previous is not None:
            for frame in range(int(previous.frame) + 1, int(first.frame)):
                points.append((0, _interpolate(previous, first, frame)))
        track_id = int(group.track_id.iloc[0])
        points.extend((track_id, row)
                      for row in group.itertuples(index=False))
        previous = group.iloc[-1]
    return points


def _nearest_foreground_component(
        plane: np.ndarray, point: object, maximum_distance_radii: float
        ) -> tuple[int, np.ndarray] | None:
    y = int(np.clip(round(float(point.y)), 0, plane.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, plane.shape[1] - 1))
    if int(plane[y, x]) > 0:
        owner = int(plane[y, x])
        return owner, _component_near(
            plane, owner, point, maximum_distance_radii)
    radius = max(float(point.radius_px), 1.0)
    choices = []
    for owner in map(int, np.unique(plane)):
        if owner <= 0:
            continue
        mask = _component_near(
            plane, owner, point, maximum_distance_radii)
        if mask is None:
            continue
        yy, xx = np.nonzero(mask)
        distance = float(np.sqrt(np.min(
            (yy - float(point.y)) ** 2 + (xx - float(point.x)) ** 2)))
        choices.append((distance / radius, owner, mask))
    if not choices:
        return None
    _, owner, mask = min(choices, key=lambda item: (item[0], item[1]))
    return owner, mask


def _partition(mask: np.ndarray, target: object,
               protectors: list[object]) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.zeros_like(mask)
    members = [target] + protectors
    costs = []
    for point in members:
        radius = max(float(point.radius_px), 1.0)
        costs.append(((yy - float(point.y)) ** 2
                      + (xx - float(point.x)) ** 2) / radius ** 2)
    chosen = np.argmin(np.asarray(costs), axis=0) == 0
    result = np.zeros_like(mask)
    result[yy[chosen], xx[chosen]] = True
    return result


def apply(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
          points: pd.DataFrame, thresholds: pd.DataFrame,
          lineages: list[dict], params: dict
          ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    candidate = labels.copy()
    thresholds_by_frame = thresholds.set_index("frame")
    visible = _visible(points)
    frame_points = {int(frame): group.copy()
                    for frame, group in visible.groupby("frame", sort=True)}
    profiles = _dominant_by_track(points)
    maximum_distance = float(params.get(
        "maximum_component_point_distance_radii", 1.0))
    minimum_raw_ratio = float(params.get(
        "minimum_raw_core_area_radius_ratio", 0.20))
    maximum_raw_ratio = float(params.get(
        "maximum_raw_core_area_radius_ratio", 2.0))
    minimum_free_fraction = _fraction(
        params, "minimum_point_core_free_fraction", 0.65)
    frames = []
    projections = []
    evidence_cache: dict[int, np.ndarray] = {}
    applied_proposals = set()

    for lineage in lineages:
        proposal_id = str(lineage["proposal_id"])
        source_owner = int(lineage["predecessor_owner"])
        for track_id, point in _lineage_points(lineage):
            frame = int(point.frame)
            trial = candidate[frame].copy()
            local_points = frame_points.get(frame, pd.DataFrame())

            # If the source name is already elsewhere, retain genuine source
            # projections but repair components with a unique durable host.
            remote = _remote_source_components(
                trial, source_owner, point, maximum_distance)
            for component in remote:
                host = _host_for_remote_component(
                    trial, component, source_owner, local_points,
                    profiles, len(labels), params)
                if host is None:
                    continue
                host_owner, host_track, proof, details = host
                before = trial.copy()
                trial[component] = int(host_owner)
                changed = int(np.count_nonzero(trial != before))
                if not changed:
                    continue
                frames.append({
                    "proposal_id": proposal_id, "frame": frame,
                    "track_id": host_track,
                    "predecessor_owner": source_owner,
                    "observed_owner": source_owner,
                    "operation": "restore_remote_durable_host",
                    "changed_pixels": changed,
                    "raw_core_area_radius_ratio": float("nan"),
                    "proof": proof,
                })
                projections.append({
                    "proposal_id": proposal_id, "frame": frame,
                    "source_owner": source_owner,
                    "host_owner": host_owner, "host_track": host_track,
                    "source_component_area": int(component.sum()),
                    **details, "proof": proof,
                })

            current = _nearest_foreground_component(
                trial, point, maximum_distance)
            if current is not None:
                observed, component = current
                if component is None:
                    continue
                if observed == source_owner:
                    continue
                protectors = []
                if not local_points.empty:
                    for other in local_points.itertuples(index=False):
                        if int(other.track_id) == int(track_id):
                            continue
                        dominant, support, purity = profiles.get(
                            int(other.track_id), (0, 0, 0.0))
                        if dominant != observed or support <= 0 or purity < 0.90:
                            continue
                        y = int(np.clip(round(float(other.y)), 0,
                                        trial.shape[0] - 1))
                        x = int(np.clip(round(float(other.x)), 0,
                                        trial.shape[1] - 1))
                        if component[y, x]:
                            protectors.append(other)
                before = trial.copy()
                if protectors:
                    target_portion = _partition(component, point, protectors)
                    trial[target_portion] = source_owner
                    operation = "partition_foreign_component_by_physical_seats"
                else:
                    trial[component] = source_owner
                    operation = "consolidate_successor_component"
                changed = int(np.count_nonzero(trial != before))
                if changed:
                    frames.append({
                        "proposal_id": proposal_id, "frame": frame,
                        "track_id": track_id,
                        "predecessor_owner": source_owner,
                        "observed_owner": observed,
                        "operation": operation, "changed_pixels": changed,
                        "raw_core_area_radius_ratio": float("nan"),
                        "proof": "unique_endpoint_lineage",
                    })
                    applied_proposals.add(proposal_id)
                candidate[frame] = trial
                continue

            evidence = evidence_cache.setdefault(
                frame, evidence_image(raw[frame], params))
            found = _point_centered_raw_core(
                raw[frame], evidence, point,
                float(thresholds_by_frame.loc[frame, "weak_threshold"]), params)
            if found is None:
                candidate[frame] = trial
                continue
            core, region, details = found
            ratio = float(details["core_area_radius_ratio"])
            if not minimum_raw_ratio <= ratio <= maximum_raw_ratio:
                candidate[frame] = trial
                continue
            if np.any(core & (unclaimed[frame][region] > 0)):
                candidate[frame] = trial
                continue
            free = core & (trial[region] == 0)
            if float(free.sum() / max(int(core.sum()), 1)) < minimum_free_fraction:
                candidate[frame] = trial
                continue
            free_components, free_count = ndi.label(free, STRUCTURE)
            if free_count > 1:
                y0, x0 = region[0].start, region[1].start
                choices = []
                for component_id in range(1, int(free_count) + 1):
                    local_y, local_x = np.nonzero(
                        free_components == component_id)
                    if not len(local_x):
                        continue
                    distance = float(np.sqrt(np.min(
                        (local_y + y0 - float(point.y)) ** 2
                        + (local_x + x0 - float(point.x)) ** 2)))
                    choices.append((distance, -len(local_x), component_id))
                free = free_components == min(choices)[2]
            before = trial.copy()
            view = trial[region]
            view[free] = source_owner
            changed = int(np.count_nonzero(trial != before))
            if changed:
                frames.append({
                    "proposal_id": proposal_id, "frame": frame,
                    "track_id": track_id,
                    "predecessor_owner": source_owner,
                    "observed_owner": 0,
                    "operation": "recover_ownerless_successor_core",
                    "changed_pixels": changed,
                    "raw_core_area_radius_ratio": ratio,
                    "proof": str(details.get("method", "point_centered_raw_core")),
                })
                applied_proposals.add(proposal_id)
            candidate[frame] = trial

    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    successor_aliases = {int(lineage["successor_owner"])
                         for lineage in lineages}
    removed = before_ids - after_ids
    unexplained_removed = removed - successor_aliases
    before_excess = int(component_accounting.total_component_excess(labels))
    after_excess = int(component_accounting.total_component_excess(candidate))
    raw_new_duplicates = int(
        component_accounting.new_duplicate_components(labels, candidate))
    projection_frames = set(
        zip(pd.DataFrame(projections).frame.astype(int),
            pd.DataFrame(projections).host_owner.astype(int))) \
        if projections else set()
    unexplained_duplicate_increase = 0
    for frame in range(len(labels)):
        owners = set(map(int, np.unique(labels[frame]))) \
            | set(map(int, np.unique(candidate[frame])))
        for owner in owners - {0}:
            before_value = component_accounting.component_excess(
                labels[frame], owner)
            after_value = component_accounting.component_excess(
                candidate[frame], owner)
            increase = max(0, after_value - before_value)
            if increase and (frame, owner) not in projection_frames:
                unexplained_duplicate_increase += int(increase)
    details = {
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_added_pixels": int(additions.sum()),
        "foreground_removed_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "unclaimed_overlap_pixels": int(np.count_nonzero(
            (candidate > 0) & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(removed)),
        "retired_successor_aliases": sorted(map(int, removed)),
        "unexplained_removed_identities": sorted(map(int, unexplained_removed)),
        "raw_new_duplicate_components": raw_new_duplicates,
        "component_excess_before": before_excess,
        "component_excess_after": after_excess,
        "duplicate_owner_frames_before": int(
            component_accounting.duplicate_owner_frames(labels)),
        "duplicate_owner_frames_after": int(
            component_accounting.duplicate_owner_frames(candidate)),
        "projection_proof_rows": int(len(projections)),
        "unexplained_new_duplicate_components": int(
            unexplained_duplicate_increase),
        "applied_proposals": sorted(applied_proposals),
    }
    return (candidate, pd.DataFrame(frames, columns=FRAME_COLUMNS),
            pd.DataFrame(projections, columns=PROJECTION_COLUMNS), details)


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    audit, lineage_members, lineages = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
        projections = pd.DataFrame(columns=PROJECTION_COLUMNS)
        details = {
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0, "foreground_removed_pixels": 0,
            "unclaimed_overlap_pixels": 0, "zero_signal_additions": 0,
            "new_identity_count": 0, "removed_identity_count": 0,
            "retired_successor_aliases": [],
            "unexplained_removed_identities": [],
            "raw_new_duplicate_components": 0,
            "component_excess_before": int(
                component_accounting.total_component_excess(labels)),
            "component_excess_after": int(
                component_accounting.total_component_excess(labels)),
            "duplicate_owner_frames_before": int(
                component_accounting.duplicate_owner_frames(labels)),
            "duplicate_owner_frames_after": int(
                component_accounting.duplicate_owner_frames(labels)),
            "projection_proof_rows": 0,
            "unexplained_new_duplicate_components": 0,
            "applied_proposals": [],
        }
    else:
        candidate, frames, projections, details = apply(
            labels, unclaimed, raw, points, thresholds, lineages, params)
    if details["foreground_removed_pixels"]:
        raise AssertionError("dormant-seat recovery removed foreground")
    if details["unclaimed_overlap_pixels"] or details["zero_signal_additions"]:
        raise AssertionError("dormant-seat recovery violated source ledgers")
    if details["new_identity_count"]:
        raise AssertionError("dormant-seat recovery created an identity")
    if details["unexplained_removed_identities"]:
        raise AssertionError("dormant-seat recovery removed an unproved identity")
    if details["unexplained_new_duplicate_components"]:
        raise AssertionError("dormant-seat recovery created an unproved duplicate")
    audit = audit.copy()
    applied = set(details["applied_proposals"])
    if len(audit):
        audit["applied"] = audit.proposal_id.astype(str).isin(applied)
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": selector_counts(params),
        "physical_tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "lineage_members": int(len(lineage_members)),
        "applied": bool(details["changed_pixels"]),
        **details,
    }
    return (candidate, unclaimed.copy(), audit, lineage_members, frames,
            projections, metrics)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])[:len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    result = produce(labels, unclaimed, raw, points, thresholds, params)
    candidate, candidate_unclaimed, audit, members, frames, projections, metrics = result
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
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    outputs = {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": output_dir / "dormant_seat_successor_audit.csv",
        "lineage_members": output_dir / "dormant_seat_successor_members.csv",
        "frames": output_dir / "dormant_seat_successor_frames.csv",
        "projection_proof": output_dir / "projection_host_proof.csv",
        "metrics": output_dir / "metrics.json",
    }
    audit.to_csv(outputs["audit"], index=False)
    members.to_csv(outputs["lineage_members"], index=False)
    frames.to_csv(outputs["frames"], index=False)
    projections.to_csv(outputs["projection_proof"], index=False)
    outputs["metrics"].write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": metrics}


