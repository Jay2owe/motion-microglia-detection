"""Recover long successor aliases and mature-track owner takeovers.

The producer is field-wide.  It accepts only image/evidence paths and
dimensionless thresholds.  Identities, tracks, frames, coordinates, regions,
events, and review cases are measured outputs and cannot select proposals.

Two complementary proofs are applied atomically:

* a later identity succeeds an already established identity while their union
  continuously covers one physical lineage and they overlap only rarely; and
* a value-connected component is carried by a mature physical track whose
  dominant owner was already observed for a large fraction of the movie before
  the foreign owner appeared on that component.

Only owner values of existing foreground pixels change.
"""
from __future__ import annotations

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
from resident_takeover import attach_owners


ALLOWED_PARAM_KEYS = {
    "mode", "targeting_mode", "labels_path", "unclaimed_path",
    "physical_track_points_path", "output_stem",
    "minimum_nested_predecessor_movie_fraction",
    "minimum_nested_source_runs",
    "minimum_nested_joint_coverage",
    "maximum_nested_cooccurrence_movie_fraction",
    "minimum_nested_strong_fraction",
    "maximum_motion_step_quantile",
    "minimum_terminal_predecessor_track_fraction",
    "maximum_terminal_seed_movie_fraction",
    "minimum_terminal_target_movie_fraction",
    "minimum_terminal_source_hiatus_movie_fraction",
    "minimum_terminal_tail_coverage",
    "minimum_terminal_pure_host_fraction",
    "minimum_terminal_overlap_fraction",
    "minimum_terminal_strong_fraction",
    "maximum_terminal_gap_movie_fraction",
    "minimum_mature_track_movie_fraction",
    "minimum_mature_track_owner_fraction",
    "minimum_prior_lineage_movie_fraction",
    "minimum_takeover_source_movie_fraction",
}
REQUIRED_PATH_KEYS = {
    "labels_path", "unclaimed_path", "physical_track_points_path",
}
AUDIT_COLUMNS = [
    "proposal_id", "atomic_group_id", "proposal_kind",
    "source_identity", "lineage_identity", "first_frame", "last_frame",
    "span_frames", "component_count", "supporting_tracks",
    "established_prefix_fraction", "prefix_coverage", "union_coverage",
    "cooccurrence_frames", "cooccurrence_fraction", "strong_fraction",
    "onset_boundary_step_radii", "maximum_motion_step_radii",
    "field_motion_step_radii", "source_runs", "dual_strong_frames",
    "tail_coverage", "pure_host_fraction", "component_overlap_fraction",
    "dominant_track_owner_fraction",
    "prior_lineage_fraction", "component_track_matching_injective",
    "component_lineage_unanimous", "locally_qualified", "eligible",
    "reason",
]
APPLICATION_COLUMNS = [
    "atomic_group_id", "member_proposals", "proposal_count",
    "changed_pixels", "changed_frames", "applied", "reason",
]


@dataclass(frozen=True)
class ComponentKey:
    frame: int
    component: int


def selector_counts(params: dict) -> dict[str, int]:
    unexpected = [str(key) for key in params
                  if str(key) not in ALLOWED_PARAM_KEYS]
    roles = {
        "identities": ("identity", "identities", "cell", "cells"),
        "owners": ("owner", "owners"),
        "tracks": ("track", "tracks"),
        "frames": ("frame", "frames", "interval"),
        "coordinates": ("coordinate", "coordinates", "point", "points"),
        "events": ("event", "events"),
        "regions": ("region", "regions", "roi"),
        "review_cases": ("case", "cases", "review"),
    }
    return {role: sum(any(token in key.lower() for token in tokens)
                      for key in unexpected)
            for role, tokens in roles.items()}


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("complete-flip recovery must be field-wide")
    supplied = sorted(str(key) for key in params
                      if str(key) not in ALLOWED_PARAM_KEYS)
    if supplied:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(supplied))
    missing = sorted(key for key in REQUIRED_PATH_KEYS
                     if params.get(key) in (None, ""))
    if missing:
        raise ValueError("missing evidence paths: " + ", ".join(missing))


def _setting(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _presence(labels: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for frame_index, frame in enumerate(labels):
        for identity in np.unique(frame):
            if int(identity) > 0:
                result.setdefault(int(identity), set()).add(int(frame_index))
    return result


def _visible_points(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    attached = attach_owners(points, labels)
    return attached[
        attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).gt(0)].copy()


def _runs(frames: set[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for frame in sorted(frames):
        if result and frame == result[-1][-1] + 1:
            result[-1].append(frame)
        else:
            result.append([frame])
    return result


def _normalised_point_step(left: object, right: object) -> float:
    distance = float(np.hypot(float(right.x) - float(left.x),
                              float(right.y) - float(left.y)))
    gap = max(int(right.frame) - int(left.frame), 1)
    radius = math.sqrt(max(
        (float(left.radius_px) ** 2 + float(right.radius_px) ** 2) / 2.0,
        np.finfo(float).eps))
    return distance / (radius * gap)


def _field_step_threshold(physical: pd.DataFrame, quantile: float,
                          maximum_gap: int) -> float:
    values: list[float] = []
    for _, group in physical.groupby("track_id", sort=False):
        rows = list(group.sort_values("frame").itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            if (int(right.frame) - int(left.frame) <= maximum_gap
                    and bool(left.strong) and bool(right.strong)):
                values.append(_normalised_point_step(left, right))
    if not values:
        raise ValueError("physical evidence has no strong motion steps")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _mask_step(left: np.ndarray, right: np.ndarray) -> float:
    if not left.any() or not right.any():
        return float("inf")
    left_yx = np.mean(np.argwhere(left), axis=0)
    right_yx = np.mean(np.argwhere(right), axis=0)
    radius = math.sqrt(max(
        ((left.sum() / math.pi) + (right.sum() / math.pi)) / 2.0,
        np.finfo(float).eps))
    return float(np.linalg.norm(left_yx - right_yx)) / radius


def _mask_boundary_step(left: np.ndarray, right: np.ndarray) -> float:
    """Minimum component-to-component step, with overlap as exact continuity."""
    if not left.any() or not right.any():
        return float("inf")
    if np.any(left & right):
        return 0.0
    structure = np.ones((3, 3), dtype=np.uint8)
    left_labels, left_count = ndi.label(left, structure)
    right_labels, right_count = ndi.label(right, structure)
    best = float("inf")
    for left_id in range(1, left_count + 1):
        left_mask = left_labels == left_id
        left_yx = np.mean(np.argwhere(left_mask), axis=0)
        for right_id in range(1, right_count + 1):
            right_mask = right_labels == right_id
            right_yx = np.mean(np.argwhere(right_mask), axis=0)
            radius = math.sqrt(max(
                ((left_mask.sum() / math.pi)
                 + (right_mask.sum() / math.pi)) / 2.0,
                np.finfo(float).eps))
            best = min(best, float(np.linalg.norm(left_yx - right_yx))
                       / radius)
    return best


def _component_keys(labels: np.ndarray, source: int,
                    frames: set[int]) -> list[ComponentKey]:
    keys: list[ComponentKey] = []
    for frame in sorted(frames):
        components, owners, _ = \
            component_accounting.label_value_components(labels[frame])
        del components
        keys.extend(ComponentKey(frame, component)
                    for component, owner in enumerate(owners, start=1)
                    if int(owner) == int(source))
    return keys


def _nested_alias_proposals(labels: np.ndarray, visible: pd.DataFrame,
                            params: dict, field_step: float) -> list[dict]:
    """Find recurrent later names wholly nested in an older lineage."""
    frame_count = len(labels)
    presence = _presence(labels)
    minimum_predecessor = _setting(
        params, "minimum_nested_predecessor_movie_fraction", 0.50)
    minimum_runs = max(2, int(round(float(params.get(
        "minimum_nested_source_runs", 2)))))
    minimum_joint = _setting(params, "minimum_nested_joint_coverage", 0.98)
    maximum_cooccurrence = _setting(
        params, "maximum_nested_cooccurrence_movie_fraction", 0.06)
    minimum_strong = _setting(
        params, "minimum_nested_strong_fraction", 0.70)
    strong_owner_frames = {
        identity: set(group.loc[group.strong.astype(bool), "frame"].astype(int))
        for identity, group in visible.groupby(
            visible.candidate_owner.astype(int), sort=False)}
    proposals: list[dict] = []
    for predecessor in sorted(presence):
        predecessor_frames = presence[predecessor]
        predecessor_first, predecessor_last = (
            min(predecessor_frames), max(predecessor_frames))
        for source in sorted(presence):
            if source == predecessor:
                continue
            source_frames = presence[source]
            source_first, source_last = min(source_frames), max(source_frames)
            if not (predecessor_first < source_first
                    and source_last < predecessor_last):
                continue
            source_runs = _runs(source_frames)
            window = set(range(predecessor_first, predecessor_last + 1))
            overlap = predecessor_frames & source_frames
            union_coverage = len(window & (predecessor_frames | source_frames)) \
                / len(window)
            cooccurrence_fraction = len(overlap) / frame_count
            source_points = visible[
                visible.candidate_owner.astype(int).eq(source)]
            strong_fraction = float(source_points.strong.astype(bool).mean()) \
                if len(source_points) else 0.0
            globally_bracketed = all(
                run[0] > predecessor_first and run[-1] < predecessor_last
                and run[0] - 1 in predecessor_frames
                and run[-1] + 1 in predecessor_frames
                for run in source_runs)
            dual_strong = sorted(
                overlap & strong_owner_frames.get(predecessor, set())
                & strong_owner_frames.get(source, set()))
            boundary_steps: list[float] = []
            for run in source_runs:
                boundary_steps.append(_mask_boundary_step(
                    labels[run[0] - 1] == predecessor,
                    labels[run[0]] == source))
                boundary_steps.append(_mask_boundary_step(
                    labels[run[-1]] == source,
                    labels[run[-1] + 1] == predecessor))
            maximum_step = max(boundary_steps, default=float("inf"))
            reasons: list[str] = []
            if len(predecessor_frames) / frame_count < minimum_predecessor:
                reasons.append("predecessor_not_movie_durable")
            if len(source_runs) < minimum_runs:
                reasons.append("successor_not_recurrent")
            if not globally_bracketed:
                reasons.append("successor_runs_not_globally_bracketed")
            if union_coverage < minimum_joint:
                reasons.append("nested_joint_coverage_too_low")
            if cooccurrence_fraction > maximum_cooccurrence:
                reasons.append("nested_names_cooccur_too_often")
            if strong_fraction < minimum_strong:
                reasons.append("nested_successor_raw_support_too_weak")
            if dual_strong:
                reasons.append("cooccurrence_has_two_strong_lineages")
            if maximum_step > field_step:
                reasons.append("nested_boundary_motion_is_field_outlier")
            keys = _component_keys(labels, source, source_frames) \
                if not reasons else []
            proposals.append({
                "proposal_kind": "complete_complement",
                "source_identity": source,
                "lineage_identity": predecessor,
                "first_frame": source_first, "last_frame": source_last,
                "span_frames": source_last - source_first + 1,
                "component_count": len(keys), "supporting_tracks": "",
                "established_prefix_fraction": (
                    source_first - predecessor_first) / frame_count,
                "prefix_coverage": len(set(range(
                    predecessor_first, source_first)) & predecessor_frames)
                / max(source_first - predecessor_first, 1),
                "union_coverage": union_coverage,
                "cooccurrence_frames": len(overlap),
                "cooccurrence_fraction": cooccurrence_fraction,
                "strong_fraction": strong_fraction,
                "onset_boundary_step_radii": boundary_steps[0]
                if boundary_steps else float("inf"),
                "maximum_motion_step_radii": maximum_step,
                "field_motion_step_radii": field_step,
                "source_runs": len(source_runs),
                "dual_strong_frames": "|".join(map(str, dual_strong)),
                "tail_coverage": 0.0, "pure_host_fraction": 0.0,
                "component_overlap_fraction": 0.0,
                "dominant_track_owner_fraction": 0.0,
                "prior_lineage_fraction": len(predecessor_frames) / frame_count,
                "component_track_matching_injective": True,
                "component_lineage_unanimous": not dual_strong,
                "_component_keys": keys, "_local_reasons": reasons,
                "_proof": "nested_recurrent_physical_lineage",
            })
    return proposals


def _terminal_successor_proposals(
        labels: np.ndarray, visible: pd.DataFrame, params: dict,
        field_step: float) -> list[dict]:
    """Propagate a one-point terminal owner handoff through one component."""
    frame_count = len(labels)
    presence = _presence(labels)
    minimum_track_fraction = _setting(
        params, "minimum_terminal_predecessor_track_fraction", 0.80)
    maximum_seed = math.ceil(frame_count * _setting(
        params, "maximum_terminal_seed_movie_fraction", 0.03))
    minimum_target = _setting(
        params, "minimum_terminal_target_movie_fraction", 0.50)
    minimum_hiatus = math.ceil(frame_count * _setting(
        params, "minimum_terminal_source_hiatus_movie_fraction", 0.05))
    minimum_tail = _setting(params, "minimum_terminal_tail_coverage", 0.98)
    minimum_pure = _setting(
        params, "minimum_terminal_pure_host_fraction", 0.98)
    minimum_overlap = _setting(
        params, "minimum_terminal_overlap_fraction", 0.98)
    minimum_strong = _setting(
        params, "minimum_terminal_strong_fraction", 0.95)
    maximum_gap = math.ceil(frame_count * _setting(
        params, "maximum_terminal_gap_movie_fraction", 0.03))
    seeds: list[dict] = []
    for track, group in visible.groupby("track_id", sort=False):
        rows = list(group.sort_values("frame").itertuples(index=False))
        if len(rows) < 2:
            continue
        runs: list[list[object]] = []
        for row in rows:
            if runs and int(runs[-1][-1].candidate_owner) == int(
                    row.candidate_owner):
                runs[-1].append(row)
            else:
                runs.append([row])
        if len(runs) < 2:
            continue
        terminal, preceding = runs[-1], runs[-2]
        source = int(terminal[0].candidate_owner)
        target = int(preceding[-1].candidate_owner)
        if source == target:
            continue
        target_count = sum(int(row.candidate_owner) == target for row in rows)
        target_fraction = target_count / len(rows)
        step = _normalised_point_step(preceding[-1], terminal[0])
        reasons: list[str] = []
        if len(terminal) > maximum_seed:
            reasons.append("terminal_seed_exceeds_movie_fraction")
        if target_fraction < minimum_track_fraction:
            reasons.append("predecessor_not_dominant_on_seed_track")
        if len(presence.get(target, set())) / frame_count < minimum_target:
            reasons.append("predecessor_identity_not_movie_durable")
        if step > field_step:
            reasons.append("terminal_handoff_motion_is_field_outlier")
        seed_frame = int(terminal[0].frame)
        prior_source = sorted(frame for frame in presence.get(source, set())
                              if frame < seed_frame)
        hiatus = (seed_frame - prior_source[-1] - 1) if prior_source \
            else frame_count
        if hiatus < minimum_hiatus:
            reasons.append("source_lacks_preseed_hiatus")
        tail_frames = set(range(seed_frame, frame_count))
        source_tail = tail_frames & presence.get(source, set())
        tail_coverage = len(source_tail) / len(tail_frames)
        if tail_coverage < minimum_tail:
            reasons.append("source_does_not_cover_recording_tail")
        if tail_frames & presence.get(target, set()):
            reasons.append("predecessor_returns_or_coexists_in_tail")
        missing = sorted(tail_frames - source_tail)
        if len(missing) > maximum_gap:
            reasons.append("terminal_tail_detector_gap_too_long")
        masks: list[np.ndarray] = []
        keys: list[ComponentKey] = []
        pure_hosts = 0
        strong_frames = 0
        component_steps: list[float] = []
        overlap_links = 0
        for frame in sorted(source_tail):
            components, owners, _ = \
                component_accounting.label_value_components(labels[frame])
            source_components = [index for index, owner in
                                 enumerate(owners, start=1)
                                 if int(owner) == source]
            if len(source_components) != 1:
                continue
            component = source_components[0]
            mask = components == component
            keys.append(ComponentKey(frame, component))
            masks.append(mask)
            host_labels, _ = ndi.label(
                labels[frame] > 0, np.ones((3, 3), dtype=np.uint8))
            host = int(host_labels[np.argwhere(mask)[0][0],
                                   np.argwhere(mask)[0][1]])
            host_owners = set(map(int, np.unique(labels[frame][
                host_labels == host]))) - {0}
            pure_hosts += int(host_owners == {source})
            frame_points = visible[
                visible.frame.astype(int).eq(frame)
                & visible.candidate_owner.astype(int).eq(source)]
            strong_frames += int(frame_points.strong.astype(bool).any())
        for (left_key, left), (right_key, right) in zip(
                zip(keys, masks), zip(keys[1:], masks[1:])):
            if right_key.frame != left_key.frame + 1:
                component_steps.append(float("inf"))
                continue
            overlap_links += int(np.any(left & right))
            component_steps.append(_mask_step(left, right))
        pure_fraction = pure_hosts / max(len(source_tail), 1)
        overlap_fraction = overlap_links / max(len(source_tail) - 1, 1)
        strong_fraction = strong_frames / max(len(source_tail), 1)
        maximum_step = max(component_steps, default=float("inf"))
        if len(keys) / max(len(source_tail), 1) < minimum_tail:
            reasons.append("tail_not_one_unambiguous_component")
        if pure_fraction < minimum_pure:
            reasons.append("terminal_component_host_not_pure")
        if overlap_fraction < minimum_overlap:
            reasons.append("terminal_component_not_overlap_continuous")
        if strong_fraction < minimum_strong:
            reasons.append("terminal_component_raw_support_too_weak")
        if maximum_step > field_step:
            reasons.append("terminal_component_motion_is_field_outlier")
        seeds.append({
            "proposal_kind": "terminal_component_successor",
            "source_identity": source, "lineage_identity": target,
            "first_frame": seed_frame, "last_frame": frame_count - 1,
            "span_frames": frame_count - seed_frame,
            "component_count": len(keys), "supporting_tracks": str(track),
            "established_prefix_fraction": target_count / frame_count,
            "prefix_coverage": target_fraction,
            "union_coverage": 0.0, "cooccurrence_frames": 0,
            "cooccurrence_fraction": 0.0,
            "strong_fraction": strong_fraction,
            "onset_boundary_step_radii": step,
            "maximum_motion_step_radii": maximum_step,
            "field_motion_step_radii": field_step, "source_runs": 1,
            "dual_strong_frames": "", "tail_coverage": tail_coverage,
            "pure_host_fraction": pure_fraction,
            "component_overlap_fraction": overlap_fraction,
            "dominant_track_owner_fraction": target_fraction,
            "prior_lineage_fraction": len(presence.get(target, set()))
            / frame_count,
            "component_track_matching_injective": len(keys) == len(source_tail),
            "component_lineage_unanimous": pure_fraction >= minimum_pure,
            "_component_keys": keys if not reasons else [],
            "_local_reasons": reasons,
            "_proof": "terminal_overlap_continuous_physical_lineage",
        })
    local = [row for row in seeds if not row["_local_reasons"]]
    targets_by_cohort: dict[tuple[int, int], set[int]] = {}
    for row in local:
        cohort = (int(row["source_identity"]), int(row["first_frame"]))
        targets_by_cohort.setdefault(cohort, set()).add(
            int(row["lineage_identity"]))
    for row in local:
        cohort = (int(row["source_identity"]), int(row["first_frame"]))
        if len(targets_by_cohort[cohort]) > 1:
            row["_local_reasons"].append(
                "terminal_cohort_has_multiple_lineage_targets")
    unique: dict[tuple[int, int, int], dict] = {}
    for row in seeds:
        key = (int(row["source_identity"]), int(row["lineage_identity"]),
               int(row["first_frame"]))
        previous = unique.get(key)
        if previous is None or (len(row["_local_reasons"])
                                < len(previous["_local_reasons"])):
            unique[key] = row
    return list(unique.values())


def _track_profiles(visible: pd.DataFrame, frame_count: int,
                    params: dict) -> dict[int, dict]:
    minimum_observations = math.ceil(frame_count * _setting(
        params, "minimum_mature_track_movie_fraction", 0.50))
    minimum_owner_fraction = _setting(
        params, "minimum_mature_track_owner_fraction", 0.90)
    profiles: dict[int, dict] = {}
    for track, group in visible.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        counts = group.candidate_owner.astype(int).value_counts()
        owner = int(counts.index[0])
        fraction = float(counts.iloc[0] / counts.sum())
        profiles[int(track)] = {
            "group": group, "owner": owner, "owner_fraction": fraction,
            "observations": int(counts.sum()),
            "mature": bool(counts.sum() >= minimum_observations
                           and fraction >= minimum_owner_fraction),
        }
    return profiles


def _takeover_proposals(labels: np.ndarray, visible: pd.DataFrame,
                        params: dict, field_step: float) -> list[dict]:
    frame_count = len(labels)
    minimum_prior = math.ceil(frame_count * _setting(
        params, "minimum_prior_lineage_movie_fraction", 0.50))
    profiles = _track_profiles(visible, frame_count, params)
    proposals: list[dict] = []
    for frame_index, frame in enumerate(labels):
        components, owners, areas = \
            component_accounting.label_value_components(frame)
        direct: dict[int, set[int]] = {}
        frame_points = visible[visible.frame.astype(int).eq(frame_index)]
        for point in frame_points.itertuples(index=False):
            y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
            x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
            component = int(components[y, x])
            if component > 0:
                direct.setdefault(component, set()).add(int(point.track_id))
        for component, current_owner in enumerate(owners, start=1):
            current_owner = int(current_owner)
            tracks = sorted(direct.get(component, set()))
            votes: dict[int, list[int]] = {}
            current_prior_tracks: list[int] = []
            fractions: list[float] = []
            prior_fractions: list[float] = []
            for track in tracks:
                profile = profiles[track]
                if not profile["mature"]:
                    continue
                lineage = int(profile["owner"])
                prior = profile["group"][
                    profile["group"].frame.astype(int).lt(frame_index)
                    & profile["group"].candidate_owner.astype(int).eq(lineage)]
                if len(prior) < minimum_prior:
                    continue
                if lineage == current_owner:
                    current_prior_tracks.append(track)
                    continue
                votes.setdefault(lineage, []).append(track)
                fractions.append(float(profile["owner_fraction"]))
                prior_fractions.append(len(prior) / frame_count)
            reasons: list[str] = []
            if not votes:
                continue
            if len(votes) != 1:
                reasons.append("mature_tracks_disagree_on_lineage")
            if current_prior_tracks:
                reasons.append("component_has_mature_current_owner_seat")
            lineage = min(votes) if len(votes) == 1 else 0
            supporting = sorted(votes.get(lineage, []))
            proposals.append({
                "proposal_kind": "bounded_component_excursion",
                "source_identity": current_owner,
                "lineage_identity": int(lineage),
                "first_frame": frame_index, "last_frame": frame_index,
                "span_frames": 1, "component_count": 1,
                "supporting_tracks": "|".join(map(str, supporting)),
                "established_prefix_fraction": max(prior_fractions,
                                                     default=0.0),
                "prefix_coverage": 1.0,
                "union_coverage": 0.0,
                "cooccurrence_frames": 0,
                "cooccurrence_fraction": 0.0,
                "strong_fraction": 1.0,
                "onset_boundary_step_radii": 0.0,
                "maximum_motion_step_radii": 0.0,
                "field_motion_step_radii": field_step,
                "source_runs": 1, "dual_strong_frames": "",
                "tail_coverage": 0.0, "pure_host_fraction": 0.0,
                "component_overlap_fraction": 0.0,
                "dominant_track_owner_fraction": min(fractions,
                                                       default=0.0),
                "prior_lineage_fraction": min(prior_fractions,
                                               default=0.0),
                "component_track_matching_injective": bool(supporting),
                "component_lineage_unanimous": len(votes) == 1,
                "_component_keys": [ComponentKey(frame_index, component)],
                "_local_reasons": reasons,
                "_proof": "mature_prior_same_track_lineage",
                "_pixels": int(areas[component - 1]),
            })
    minimum_cohort = math.ceil(frame_count * _setting(
        params, "minimum_takeover_source_movie_fraction", 0.05))
    cohorts: dict[tuple[int, int, str], list[dict]] = {}
    for row in proposals:
        if row["_local_reasons"]:
            continue
        key = (int(row["source_identity"]), int(row["lineage_identity"]),
               str(row["supporting_tracks"]))
        cohorts.setdefault(key, []).append(row)
    for rows in cohorts.values():
        if len(rows) < minimum_cohort:
            for row in rows:
                row["_local_reasons"].append(
                    "takeover_cohort_below_movie_fraction")
    return proposals


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict,
             ) -> tuple[pd.DataFrame, list[dict]]:
    assert_target_free(params)
    attached = attach_owners(points, labels)
    visible = attached[
        attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).gt(0)].copy()
    physical = attached[attached.physically_visible.astype(bool)].copy()
    field_step = _field_step_threshold(
        physical, _setting(params, "maximum_motion_step_quantile", 0.90),
        max(1, math.ceil(len(labels) * _setting(
            params, "maximum_terminal_gap_movie_fraction", 0.03))))
    proposals = _nested_alias_proposals(
        labels, visible, params, field_step)
    proposals.extend(_terminal_successor_proposals(
        labels, visible, params, field_step))
    proposals.extend(_takeover_proposals(labels, visible, params, field_step))
    for number, row in enumerate(proposals, start=1):
        row["proposal_id"] = f"CF{number:04d}"
        row["atomic_group_id"] = f"CG{number:04d}"
        row["locally_qualified"] = not row["_local_reasons"]
        row["eligible"] = bool(row["locally_qualified"])
        row["reason"] = ("eligible_complete_flip_recovery"
                         if row["eligible"] else
                         "|".join(row["_local_reasons"]))
    public = pd.DataFrame([
        {column: row.get(column, "") for column in AUDIT_COLUMNS}
        for row in proposals], columns=AUDIT_COLUMNS)
    return public, proposals


def apply(labels: np.ndarray, proposals: list[dict],
          ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    candidate = labels.copy()
    component_maps = [
        component_accounting.label_value_components(frame)[0]
        for frame in labels]
    assignments: dict[ComponentKey, tuple[int, str]] = {}
    conflicted: set[ComponentKey] = set()
    for row in proposals:
        if not row["eligible"]:
            continue
        target = int(row["lineage_identity"])
        for key in row["_component_keys"]:
            previous = assignments.get(key)
            if previous is not None and previous[0] != target:
                conflicted.add(key)
            assignments[key] = (target, str(row["proposal_id"]))
    proven_mask = np.zeros(labels.shape, dtype=bool)
    applications: list[dict] = []
    for row in proposals:
        if not row["eligible"]:
            continue
        keys = list(row["_component_keys"])
        if any(key in conflicted for key in keys):
            applications.append({
                "atomic_group_id": row["atomic_group_id"],
                "member_proposals": row["proposal_id"], "proposal_count": 1,
                "changed_pixels": 0, "changed_frames": 0, "applied": False,
                "reason": "component_target_conflict",
            })
            continue
        changed_pixels = 0
        changed_frames: set[int] = set()
        source = int(row["source_identity"])
        target = int(row["lineage_identity"])
        for key in keys:
            mask = component_maps[key.frame] == int(key.component)
            if not np.any(mask) or not np.all(labels[key.frame][mask] == source):
                raise AssertionError("component provenance changed during apply")
            changed_pixels += int(mask.sum())
            changed_frames.add(int(key.frame))
            candidate[key.frame][mask] = target
            proven_mask[key.frame][mask] = True
        applications.append({
            "atomic_group_id": row["atomic_group_id"],
            "member_proposals": row["proposal_id"], "proposal_count": 1,
            "changed_pixels": changed_pixels,
            "changed_frames": len(changed_frames), "applied": True,
            "reason": str(row["_proof"]),
        })
    return candidate, pd.DataFrame(applications,
                                   columns=APPLICATION_COLUMNS), proven_mask


def _proven_new_duplicates(baseline: np.ndarray, candidate: np.ndarray,
                           proven_mask: np.ndarray,
                           ) -> tuple[int, int, pd.DataFrame]:
    new = component_accounting.new_duplicate_components(baseline, candidate)
    if not new:
        return 0, 0, pd.DataFrame(columns=[
            "frame", "lineage_identity", "component", "pixels",
            "source_owners", "contains_preexisting_lineage",
            "changed_pixels_lineage_proven", "proof"])
    proven = 0
    proof_rows: list[dict] = []
    changed_frames = np.flatnonzero(
        (baseline != candidate).reshape(len(baseline), -1).any(axis=1))
    for frame_index in changed_frames:
        before, after = baseline[frame_index], candidate[frame_index]
        targets = set(map(int, np.unique(after[before != after]))) - {0}
        for target in targets:
            before_excess = component_accounting.component_excess(
                before, target)
            after_excess = component_accounting.component_excess(after, target)
            added = max(0, after_excess - before_excess)
            if not added:
                continue
            components, count = ndi.label(
                after == target, np.ones((3, 3), dtype=np.uint8))
            proven_components: list[tuple[int, np.ndarray]] = []
            for component in range(1, count + 1):
                mask = components == component
                if not np.any(before[mask] == target) and np.any(
                        proven_mask[frame_index][mask]):
                    proven_components.append((component, mask))
            for component, mask in proven_components[:added]:
                sources = sorted(set(map(int, np.unique(before[mask]))) - {0})
                proof_rows.append({
                    "frame": int(frame_index),
                    "lineage_identity": int(target),
                    "component": int(component), "pixels": int(mask.sum()),
                    "source_owners": "|".join(map(str, sources)),
                    "contains_preexisting_lineage": False,
                    "changed_pixels_lineage_proven": bool(
                        np.any(proven_mask[frame_index][mask])),
                    "proof": "eligible_field_wide_physical_lineage_proposal",
                })
            proven += min(added, len(proven_components))
    return int(new), int(proven), pd.DataFrame(proof_rows)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    output_dir = Path(out.out)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internals = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame(columns=APPLICATION_COLUMNS)
        proven_mask = np.zeros(labels.shape, dtype=bool)
    else:
        candidate, applications, proven_mask = apply(labels, internals)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("complete-flip recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids - before_ids:
        raise AssertionError("complete-flip recovery created an identity")
    new_duplicates, proven_duplicates, duplicate_proof = _proven_new_duplicates(
        labels, candidate, proven_mask)
    if new_duplicates != proven_duplicates:
        raise AssertionError(
            "new duplicate component lacks physical-lineage proof")
    stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    audit_path = output_dir / "branched_motion_lineage_audit.csv"
    application_path = output_dir / "branched_motion_lineage_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    duplicate_proof_path = output_dir / "new_duplicate_lineage_proof.csv"
    duplicate_proof.to_csv(duplicate_proof_path, index=False)
    changed = candidate != labels
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    eligible = audit[audit.eligible.astype(bool)] if len(audit) else audit
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": selector_counts(params),
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique()),
        "identity_pairs_audited": int(audit.proposal_kind.eq(
            "complete_complement").sum()),
        "takeover_components_audited": int(audit.proposal_kind.eq(
            "bounded_component_excursion").sum()),
        "eligible_proposals": int(len(eligible)),
        "applied_atomic_groups": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": len(changed_frames),
        "changed_frame_indices": changed_frames,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": len(after_ids - before_ids),
        "removed_identity_count": len(before_ids - after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": bool(np.array_equal(
            tifffile.imread(unclaimed_path), unclaimed)),
        "new_duplicate_components": new_duplicates,
        "lineage_proven_new_duplicate_components": proven_duplicates,
        "all_new_duplicates_injectively_unanimous": bool(
            new_duplicates == proven_duplicates),
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {
        "outputs": {"labels": labels_path, "unclaimed": unclaimed_path,
                    "audit": audit_path, "applications": application_path,
                    "duplicate_proof": duplicate_proof_path,
                    "metrics": metrics_path},
        "summary": metrics,
    }

