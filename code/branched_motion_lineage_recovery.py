"""Recover bounded owner excursions on independently tracked physical bodies.

This isolated producer is field-wide.  It accepts image/evidence paths and
dimensionless thresholds only; identities, tracks, frames, events, coordinates,
regions, and review cases are measured outputs and can never be selectors.

Two forms of the same physical-lineage error are supported:

* a short identity whose components all hand back to one durable owner and
  whose presence exactly complements that owner; and
* a short, component-level foreign-owner run bracketed by the same established
  owner on one physical track.

The second form is refused when the foreign owner has an independent physical
lineage at the same time or a recent reciprocal owner exchange.  A bounded
single-source excursion must also persist for a data-relative minimum duration;
shorter events are allowed only when multiple independent sources converge on
the same lineage over the same frames.  This rejects isolated flashes without
discarding a one-frame multi-branch collapse.  Simultaneous proposals connected
through an owner or component are decided atomically.  Only ownership of
existing connected foreground components can change.
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


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
ALLOWED_PARAM_KEYS = {
    "mode", "targeting_mode", "labels_path", "unclaimed_path",
    "physical_track_points_path", "application_audit_path",
    "conflicted_lineage_audit_path", "distinct_history_proposals_path",
    "distinct_history_frames_path", "output_stem",
    "maximum_complete_alias_movie_fraction",
    "minimum_complete_alias_union_coverage",
    "minimum_complete_alias_strong_fraction",
    "minimum_durable_movie_fraction",
    "maximum_short_to_durable_track_ratio",
    "maximum_excursion_movie_fraction",
    "maximum_flank_gap_movie_fraction",
    "minimum_stable_owner_fraction",
    "minimum_excursion_strong_fraction",
    "maximum_excursion_encounter_fraction",
    "minimum_single_source_excursion_movie_fraction",
    "maximum_motion_step_quantile",
    "maximum_reciprocal_window_movie_fraction",
    "maximum_fuzzy_distance_radii", "minimum_fuzzy_unique_margin_radii",
}
REQUIRED_PATH_KEYS = {
    "labels_path", "unclaimed_path", "physical_track_points_path",
    "application_audit_path", "conflicted_lineage_audit_path",
    "distinct_history_proposals_path", "distinct_history_frames_path",
}
AUDIT_COLUMNS = [
    "proposal_id", "atomic_group_id", "proposal_kind", "source_identity",
    "lineage_identity", "first_frame", "last_frame", "span_frames",
    "component_count", "supporting_tracks", "stable_owner_fraction",
    "strong_fraction", "encounter_fraction", "maximum_step_radii",
    "field_step_threshold_radii", "union_coverage", "cooccurrence_frames",
    "direct_track_matches", "fuzzy_track_matches",
    "minimum_fuzzy_unique_margin_radii", "ambiguous_track_matches",
    "unmatched_components", "lineage_disagreement_tracks",
    "component_track_matching_injective", "component_lineage_unanimous",
    "concurrent_source_lineage_tracks", "reciprocal_lineage_tracks",
    "accepted_track_frame_overlaps",
    "locally_qualified", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "atomic_group_id", "member_proposals", "proposal_count",
    "changed_pixels", "changed_frames", "applied", "reason",
]


@dataclass(frozen=True)
class ComponentKey:
    frame: int
    component: int


@dataclass(frozen=True)
class ComponentMatch:
    key: ComponentKey | None
    track: int
    mode: str
    distance_radii: float
    unique_margin_radii: float
    candidate_keys: tuple[ComponentKey, ...] = ()


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
    return {
        role: sum(any(token in key.lower() for token in tokens)
                  for key in unexpected)
        for role, tokens in roles.items()
    }


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("branched-lineage recovery must be field-wide")
    supplied = [str(key) for key in params
                if str(key) not in ALLOWED_PARAM_KEYS]
    if supplied:
        raise ValueError(
            "branched-lineage recovery received unsupported parameters "
            "(selectors are forbidden): " + ", ".join(sorted(supplied)))
    missing = sorted(key for key in REQUIRED_PATH_KEYS
                     if params.get(key) in (None, ""))
    if missing:
        raise ValueError("missing required evidence paths: " + ", ".join(missing))


def _setting(params: dict, key: str, default: float,
             lower: float = 0.0, upper: float = 1.0) -> float:
    value = float(params.get(key, default))
    if not lower <= value <= upper:
        raise ValueError(f"{key} must be in [{lower}, {upper}]")
    return value


def _presence(labels: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for frame_index, frame in enumerate(labels):
        for identity in np.unique(frame):
            if int(identity) > 0:
                result.setdefault(int(identity), set()).add(int(frame_index))
    return result


def _runs(frames: set[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for frame in sorted(map(int, frames)):
        if result and frame == result[-1][-1] + 1:
            result[-1].append(frame)
        else:
            result.append([frame])
    return result


def _owner_runs(group: pd.DataFrame) -> list[dict]:
    result: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.candidate_owner)
        if (result and result[-1]["owner"] == owner
                and int(row.frame) == int(result[-1]["rows"][-1].frame) + 1):
            result[-1]["rows"].append(row)
        else:
            result.append({"owner": owner, "rows": [row]})
    return result


def _normalised_step(left: object, right: object) -> float:
    distance = float(np.hypot(float(right.x) - float(left.x),
                              float(right.y) - float(left.y)))
    frame_gap = max(abs(int(right.frame) - int(left.frame)), 1)
    radius = math.sqrt(max(
        (float(left.radius_px) ** 2 + float(right.radius_px) ** 2) / 2.0,
        1.0))
    return distance / (radius * frame_gap)


def _field_step_threshold(visible: pd.DataFrame, quantile: float,
                          maximum_gap: int) -> float:
    values: list[float] = []
    for _, group in visible.groupby("track_id", sort=False):
        rows = list(group.sort_values("frame").itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            gap = int(right.frame) - int(left.frame)
            if (gap <= maximum_gap and bool(left.strong)
                    and bool(right.strong)):
                values.append(_normalised_step(left, right))
    if not values:
        raise ValueError("physical evidence has no strong motion steps")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _component_maps(labels: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    components: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for frame in labels:
        labelled, component_owners, _ = \
            component_accounting.label_value_components(frame)
        components.append(labelled)
        owners.append(component_owners)
    return components, owners


def _match_point_to_component(
        point: object, source: int, frame: np.ndarray,
        components: np.ndarray, component_owners: np.ndarray,
        maximum_fuzzy_distance: float, minimum_fuzzy_margin: float,
        ) -> ComponentMatch:
    """Assign one physical point to at most one same-owner component.

    A rounded point inside a component is always a direct match.  Otherwise the
    nearest component is accepted only when both its distance and its lead over
    the runner-up are expressed in the point's own radius and pass the supplied
    dimensionless gates.
    """
    height, width = frame.shape
    y = int(np.clip(round(float(point.y)), 0, height - 1))
    x = int(np.clip(round(float(point.x)), 0, width - 1))
    direct = int(components[y, x])
    if (direct > 0 and int(component_owners[direct - 1]) == int(source)):
        return ComponentMatch(
            ComponentKey(int(point.frame), direct), int(point.track_id),
            "direct", 0.0, float("inf"),
            (ComponentKey(int(point.frame), direct),))
    candidates: list[tuple[float, ComponentKey]] = []
    radius = max(float(point.radius_px), np.finfo(float).eps)
    for offset, owner in enumerate(component_owners, start=1):
        if int(owner) != int(source):
            continue
        yy, xx = np.where(components == int(offset))
        if not len(xx):
            continue
        distance = float(np.sqrt(np.min(
            (xx.astype(float) - float(point.x)) ** 2
            + (yy.astype(float) - float(point.y)) ** 2))) / radius
        candidates.append((distance, ComponentKey(int(point.frame), offset)))
    candidates.sort(key=lambda item: (item[0], item[1].component))
    if not candidates:
        return ComponentMatch(None, int(point.track_id), "unmatched",
                              float("inf"), 0.0)
    best_distance, best_key = candidates[0]
    second_distance = candidates[1][0] if len(candidates) > 1 else float("inf")
    margin = float(second_distance - best_distance)
    nearby = tuple(key for distance, key in candidates
                   if distance <= best_distance + minimum_fuzzy_margin)
    if best_distance > maximum_fuzzy_distance:
        return ComponentMatch(None, int(point.track_id), "unmatched",
                              best_distance, margin, nearby)
    if margin < minimum_fuzzy_margin:
        return ComponentMatch(None, int(point.track_id), "ambiguous_fuzzy",
                              best_distance, margin, nearby)
    return ComponentMatch(best_key, int(point.track_id), "fuzzy",
                          best_distance, margin, (best_key,))


def _component_matches(
        labels: np.ndarray, visible: pd.DataFrame,
        components: list[np.ndarray], component_owners: list[np.ndarray],
        maximum_fuzzy_distance: float, minimum_fuzzy_margin: float,
        ) -> tuple[dict[tuple[int, int], ComponentMatch],
                   dict[ComponentKey, list[ComponentMatch]],
                   dict[ComponentKey, set[int]]]:
    """Build one injective track-to-component assignment per owner-frame."""
    by_track_frame: dict[tuple[int, int], ComponentMatch] = {}
    by_component: dict[ComponentKey, list[ComponentMatch]] = {}
    ambiguous: dict[ComponentKey, set[int]] = {}
    for point in visible.itertuples(index=False):
        track_frame = (int(point.track_id), int(point.frame))
        if track_frame in by_track_frame:
            previous = by_track_frame[track_frame]
            by_track_frame[track_frame] = ComponentMatch(
                None, int(point.track_id), "duplicate_track_frame", 0.0, 0.0,
                previous.candidate_keys)
            for key in previous.candidate_keys:
                ambiguous.setdefault(key, set()).add(int(point.track_id))
            continue
        match = _match_point_to_component(
            point, int(point.candidate_owner), labels[int(point.frame)],
            components[int(point.frame)], component_owners[int(point.frame)],
            maximum_fuzzy_distance, minimum_fuzzy_margin)
        by_track_frame[track_frame] = match
        if match.key is not None:
            by_component.setdefault(match.key, []).append(match)
        elif match.mode in {"ambiguous_fuzzy", "duplicate_track_frame"}:
            for key in match.candidate_keys:
                ambiguous.setdefault(key, set()).add(int(point.track_id))
    return by_track_frame, by_component, ambiguous


def _active_rows(table: pd.DataFrame, conflict_table: bool = False) -> pd.DataFrame:
    if table.empty:
        return table
    active = np.full(len(table), bool(conflict_table), dtype=bool)
    if "applied" in table:
        active |= table.applied.astype(str).str.lower().isin(
            {"true", "1", "yes"}).to_numpy()
    for column in ("application_status", "outcome", "status"):
        if column in table:
            active |= table[column].astype(str).str.lower().str.contains(
                r"applied|accepted|changed|conflict", regex=True).to_numpy()
    if "changed_pixels" in table:
        active |= (pd.to_numeric(table.changed_pixels, errors="coerce")
                   .fillna(0).to_numpy() > 0)
    return table.loc[active]


def _integer_values(row: object, columns: tuple[str, ...]) -> set[int]:
    result: set[int] = set()
    for column in columns:
        if not hasattr(row, column):
            continue
        value = pd.to_numeric(pd.Series([getattr(row, column)]),
                              errors="coerce").iloc[0]
        if pd.notna(value) and int(value) >= 0:
            result.add(int(value))
    return result


def _accepted_track_frame_vetoes(
        application_audit: pd.DataFrame, conflicted_audit: pd.DataFrame,
        proposal_ledger: pd.DataFrame, frame_ledger: pd.DataFrame,
        frame_count: int,
        ) -> set[tuple[int, int]]:
    """Return only accepted track/frame claims, never blanket identities."""
    protected: set[tuple[int, int]] = set()
    active_proposals = _active_rows(proposal_ledger)
    active_ids = set(active_proposals.proposal_id.astype(str)) \
        if "proposal_id" in active_proposals else set()
    if len(frame_ledger):
        active_frames = frame_ledger[
            frame_ledger.proposal_id.astype(str).isin(active_ids)] \
            if "proposal_id" in frame_ledger else frame_ledger.iloc[0:0]
        if "changed_pixels" in frame_ledger:
            active_frames = pd.concat([
                active_frames,
                frame_ledger[pd.to_numeric(
                    frame_ledger.changed_pixels, errors="coerce").fillna(0) > 0],
            ]).drop_duplicates()
        for row in active_frames.itertuples(index=False):
            frames = _integer_values(row, ("frame",))
            tracks = _integer_values(
                row, ("track_id", "physical_track", "victim_track",
                      "resident_track"))
            protected.update((track, frame) for track in tracks for frame in frames)
    active_application = _active_rows(application_audit)
    for row in active_application.itertuples(index=False):
        tracks = _integer_values(
            row, ("track_id", "physical_track", "victim_track",
                  "resident_track"))
        frames = _integer_values(row, ("frame",))
        if not frames and hasattr(row, "first_frame") and hasattr(row, "last_frame"):
            first = int(getattr(row, "first_frame"))
            last = int(getattr(row, "last_frame"))
            frames = set(range(max(first, 0), min(last, frame_count - 1) + 1))
        protected.update((track, frame) for track in tracks for frame in frames)
    active_conflicts = _active_rows(conflicted_audit, conflict_table=True)
    for row in active_conflicts.itertuples(index=False):
        tracks = _integer_values(
            row, ("track_id", "physical_track", "victim_track",
                  "resident_track"))
        frames = _integer_values(row, ("frame",))
        if not frames:
            frames = set(range(frame_count))
        protected.update((track, frame) for track in tracks for frame in frames)
    return protected


def _track_profiles(visible: pd.DataFrame) -> dict[int, dict]:
    profiles: dict[int, dict] = {}
    for track, group in visible.groupby("track_id", sort=False):
        counts = group.candidate_owner.astype(int).value_counts()
        largest = int(counts.iloc[0])
        second = int(counts.iloc[1]) if len(counts) > 1 else 0
        profiles[int(track)] = {
            "group": group.sort_values("frame"),
            "counts": {int(key): int(value) for key, value in counts.items()},
            "strict_owner": int(counts.index[0]) if largest > second else 0,
        }
    return profiles


def _direct_handoff(point: object, group: pd.DataFrame, lineage: int,
                    maximum_gap: int, maximum_step: float) -> bool:
    rows = list(group.sort_values("frame").itertuples(index=False))
    position = next((index for index, row in enumerate(rows)
                     if int(row.frame) == int(point.frame)), None)
    if position is None:
        return False
    for neighbour_position in (position - 1, position + 1):
        if not 0 <= neighbour_position < len(rows):
            continue
        neighbour = rows[neighbour_position]
        if (int(neighbour.candidate_owner) == int(lineage)
                and abs(int(neighbour.frame) - int(point.frame)) <= maximum_gap
                and _normalised_step(point, neighbour) <= maximum_step):
            return True
    return False


def _track_supports_lineage(track: int, frame: int, source: int, lineage: int,
                            profiles: dict[int, dict], maximum_gap: int,
                            ) -> bool:
    """Require a track's local non-source evidence to name one lineage."""
    profile = profiles[int(track)]
    group = profile["group"]
    local = group[
        group.frame.astype(int).between(int(frame) - maximum_gap,
                                        int(frame) + maximum_gap)
        & group.candidate_owner.astype(int).ne(int(source))]
    local_owners = set(local.candidate_owner.astype(int)) - {0}
    if local_owners:
        return local_owners == {int(lineage)}
    return int(profile["strict_owner"]) == int(lineage)


def _component_evidence(
        keys: set[ComponentKey], source: int, lineage: int,
        by_component: dict[ComponentKey, list[ComponentMatch]],
        ambiguous: dict[ComponentKey, set[int]],
        profiles: dict[int, dict], maximum_gap: int,
        protected_track_frames: set[tuple[int, int]],
        ) -> dict:
    direct = fuzzy = 0
    fuzzy_margins: list[float] = []
    ambiguous_tracks: set[int] = set()
    disagreements: set[int] = set()
    matched_tracks: set[int] = set()
    unmatched_components = 0
    protected: set[tuple[int, int]] = set()
    seen_track_frames: set[tuple[int, int]] = set()
    injective = True
    for key in sorted(keys, key=lambda item: (item.frame, item.component)):
        matches = by_component.get(key, [])
        if not matches:
            unmatched_components += 1
        ambiguous_tracks.update(ambiguous.get(key, set()))
        for match in matches:
            track_frame = (int(match.track), int(key.frame))
            if track_frame in seen_track_frames:
                injective = False
            seen_track_frames.add(track_frame)
            matched_tracks.add(int(match.track))
            if match.mode == "direct":
                direct += 1
            elif match.mode == "fuzzy":
                fuzzy += 1
                fuzzy_margins.append(float(match.unique_margin_radii))
            if not _track_supports_lineage(
                    int(match.track), int(key.frame), source, lineage,
                    profiles, maximum_gap):
                disagreements.add(int(match.track))
            if track_frame in protected_track_frames:
                protected.add(track_frame)
    injective = bool(injective and not ambiguous_tracks)
    unanimous = bool(not disagreements and not unmatched_components
                     and not ambiguous_tracks)
    return {
        "tracks": matched_tracks,
        "direct": int(direct), "fuzzy": int(fuzzy),
        "minimum_fuzzy_margin": min(fuzzy_margins, default=float("inf")),
        "ambiguous_tracks": ambiguous_tracks,
        "unmatched_components": int(unmatched_components),
        "disagreements": disagreements,
        "injective": injective, "unanimous": unanimous,
        "protected": protected,
    }


def _component_text(keys: set[ComponentKey]) -> str:
    return "|".join(f"{key.frame}:{key.component}"
                    for key in sorted(keys, key=lambda item: (item.frame,
                                                              item.component)))


def _base_row(kind: str, source: int, lineage: int, first: int, last: int,
              keys: set[ComponentKey], tracks: set[int], threshold: float,
              ) -> dict:
    return {
        "proposal_id": "", "atomic_group_id": "", "proposal_kind": kind,
        "source_identity": int(source), "lineage_identity": int(lineage),
        "first_frame": int(first), "last_frame": int(last),
        "span_frames": int(last - first + 1),
        "component_count": int(len(keys)),
        "supporting_tracks": "|".join(map(str, sorted(tracks))),
        "stable_owner_fraction": 0.0, "strong_fraction": 0.0,
        "encounter_fraction": 0.0, "maximum_step_radii": float("inf"),
        "field_step_threshold_radii": float(threshold),
        "union_coverage": 0.0, "cooccurrence_frames": 0,
        "direct_track_matches": 0, "fuzzy_track_matches": 0,
        "minimum_fuzzy_unique_margin_radii": float("inf"),
        "ambiguous_track_matches": 0, "unmatched_components": 0,
        "lineage_disagreement_tracks": "",
        "component_track_matching_injective": False,
        "component_lineage_unanimous": False,
        "concurrent_source_lineage_tracks": "",
        "reciprocal_lineage_tracks": "",
        "accepted_track_frame_overlaps": 0,
        "locally_qualified": False, "eligible": False, "reason": "",
        "_component_keys": keys, "_tracks": tracks,
        "_component_evidence": {},
        "_local_reasons": [], "_safety_reasons": [],
    }


def _complete_alias_proposals(
        labels: np.ndarray, visible: pd.DataFrame, profiles: dict[int, dict],
        components: list[np.ndarray], component_owners: list[np.ndarray],
        presence: dict[int, set[int]], params: dict, maximum_gap: int,
        step_threshold: float,
        by_component: dict[ComponentKey, list[ComponentMatch]],
        ambiguous: dict[ComponentKey, set[int]],
        protected_track_frames: set[tuple[int, int]],
        ) -> list[dict]:
    frame_count = int(len(labels))
    maximum_short = int(math.ceil(frame_count * _setting(
        params, "maximum_complete_alias_movie_fraction", 0.06)))
    minimum_union = _setting(
        params, "minimum_complete_alias_union_coverage", 0.98)
    minimum_strong = _setting(
        params, "minimum_complete_alias_strong_fraction", 0.70)
    minimum_durable = int(math.ceil(frame_count * _setting(
        params, "minimum_durable_movie_fraction", 0.50)))
    maximum_ratio = _setting(
        params, "maximum_short_to_durable_track_ratio", 1.0, 0.0, 10.0)
    proposals: list[dict] = []
    for source in sorted(presence):
        source_presence = presence[source]
        source_rows = visible[visible.candidate_owner.astype(int) == int(source)]
        related_tracks = set(map(int, source_rows.track_id.unique()))
        alternatives: set[int] = set()
        ratios: list[float] = []
        reasons: list[str] = []
        if len(source_presence) > maximum_short:
            reasons.append("identity_exceeds_complete_alias_movie_fraction")
        for track in sorted(related_tracks):
            counts = profiles[track]["counts"]
            other = {owner: count for owner, count in counts.items()
                     if owner != int(source)}
            if len(other) != 1:
                reasons.append("alias_track_lacks_unique_lineage_owner")
                continue
            lineage, lineage_count = next(iter(other.items()))
            alternatives.add(int(lineage))
            ratios.append(float(counts.get(int(source), 0)
                                / max(int(lineage_count), 1)))
        lineage = next(iter(alternatives)) if len(alternatives) == 1 else 0
        if not related_tracks:
            reasons.append("alias_lacks_physical_track_evidence")
        if len(alternatives) != 1 or lineage <= 0:
            reasons.append("alias_tracks_do_not_share_one_lineage_owner")
        if max(ratios, default=float("inf")) > maximum_ratio:
            reasons.append("alias_dominates_a_physical_track")
        durable_presence = presence.get(int(lineage), set())
        joint = source_presence | durable_presence
        lifespan = set(range(min(joint), max(joint) + 1)) if joint else set()
        coverage = float(len(joint) / len(lifespan)) if lifespan else 0.0
        cooccurrence = source_presence & durable_presence
        if len(durable_presence) < minimum_durable:
            reasons.append("alias_lineage_owner_not_durable")
        if cooccurrence:
            reasons.append("alias_and_lineage_owner_cooccur")
        if coverage < minimum_union:
            reasons.append("alias_union_not_complete")
        if not lifespan or any(
                run[0] == min(lifespan) or run[-1] == max(lifespan)
                or run[0] - 1 not in durable_presence
                or run[-1] + 1 not in durable_presence
                for run in _runs(source_presence)):
            reasons.append("alias_run_not_globally_bracketed")
        strong_fraction = float(source_rows.strong.astype(bool).mean()) \
            if len(source_rows) else 0.0
        if strong_fraction < minimum_strong:
            reasons.append("alias_raw_support_too_weak")
        keys: set[ComponentKey] = set()
        maximum_seen_step = 0.0
        for frame in sorted(source_presence):
            source_components = [int(component + 1)
                                 for component, owner in enumerate(
                                     component_owners[frame])
                                 if int(owner) == int(source)]
            for component in source_components:
                key = ComponentKey(int(frame), int(component))
                keys.add(key)
        evidence = _component_evidence(
            keys, source, lineage, by_component, ambiguous, profiles,
            maximum_gap, protected_track_frames) if lineage > 0 else {
                "tracks": set(), "direct": 0, "fuzzy": 0,
                "minimum_fuzzy_margin": float("inf"),
                "ambiguous_tracks": set(),
                "unmatched_components": len(keys), "disagreements": set(),
                "injective": False, "unanimous": False, "protected": set(),
            }
        if not evidence["injective"]:
            reasons.append("alias_component_track_matching_not_injective")
        if not evidence["unanimous"]:
            reasons.append("alias_component_lineage_not_unanimous")
        for key in keys:
            for match in by_component.get(key, []):
                point_rows = profiles[int(match.track)]["group"]
                point = point_rows[point_rows.frame.astype(int).eq(
                    int(key.frame))]
                if len(point):
                    item = next(point.itertuples(index=False))
                    for neighbour in point_rows.itertuples(index=False):
                        if (int(neighbour.candidate_owner) == int(lineage)
                                and abs(int(neighbour.frame) - key.frame)
                                <= maximum_gap):
                            maximum_seen_step = max(
                                maximum_seen_step,
                                _normalised_step(item, neighbour))
        safety: list[str] = []
        if evidence["protected"]:
            safety.append("accepted_track_frame_veto")
        first = min(source_presence) if source_presence else -1
        last = max(source_presence) if source_presence else -1
        row = _base_row("complete_complement", source, lineage,
                        first, last, keys, related_tracks, step_threshold)
        row.update({
            "stable_owner_fraction": float(1.0 / (1.0 + max(
                ratios, default=float("inf")))) if ratios else 0.0,
            "strong_fraction": strong_fraction,
            "encounter_fraction": float(source_rows.in_encounter.astype(bool).mean())
                if len(source_rows) else 0.0,
            "maximum_step_radii": float(maximum_seen_step),
            "union_coverage": coverage,
            "cooccurrence_frames": int(len(cooccurrence)),
            "supporting_tracks": "|".join(map(str, sorted(evidence["tracks"]))),
            "direct_track_matches": int(evidence["direct"]),
            "fuzzy_track_matches": int(evidence["fuzzy"]),
            "minimum_fuzzy_unique_margin_radii": float(
                evidence["minimum_fuzzy_margin"]),
            "ambiguous_track_matches": int(len(evidence["ambiguous_tracks"])),
            "unmatched_components": int(evidence["unmatched_components"]),
            "lineage_disagreement_tracks": "|".join(map(
                str, sorted(evidence["disagreements"]))),
            "component_track_matching_injective": bool(evidence["injective"]),
            "component_lineage_unanimous": bool(evidence["unanimous"]),
            "accepted_track_frame_overlaps": int(len(evidence["protected"])),
            "_component_evidence": evidence,
            "_local_reasons": list(dict.fromkeys(reasons)),
            "_safety_reasons": list(dict.fromkeys(safety)),
        })
        row["locally_qualified"] = not row["_local_reasons"]
        proposals.append(row)
    return proposals


def _lineage_conflicts(source: int, lineage: int, first: int, last: int,
                       profiles: dict[int, dict], reciprocal_window: int,
                       ) -> tuple[set[int], set[int]]:
    concurrent: set[int] = set()
    reciprocal: set[int] = set()
    for track, profile in profiles.items():
        if int(profile["strict_owner"]) != int(source):
            continue
        group = profile["group"]
        source_frames = set(group.loc[
            group.candidate_owner.astype(int).eq(int(source)),
            "frame"].astype(int))
        if source_frames & set(range(int(first), int(last) + 1)):
            concurrent.add(int(track))
        lineage_frames = set(group.loc[
            group.candidate_owner.astype(int).eq(int(lineage)),
            "frame"].astype(int))
        if any(int(first) - reciprocal_window <= frame
               <= int(last) + reciprocal_window for frame in lineage_frames):
            reciprocal.add(int(track))
    return concurrent, reciprocal


def _bounded_component_proposals(
        labels: np.ndarray, visible: pd.DataFrame, profiles: dict[int, dict],
        components: list[np.ndarray], component_owners: list[np.ndarray],
        params: dict, step_threshold: float, maximum_gap: int,
        by_track_frame: dict[tuple[int, int], ComponentMatch],
        by_component: dict[ComponentKey, list[ComponentMatch]],
        ambiguous: dict[ComponentKey, set[int]],
        protected_track_frames: set[tuple[int, int]],
        ) -> list[dict]:
    frame_count = int(len(labels))
    maximum_span = int(math.ceil(frame_count * _setting(
        params, "maximum_excursion_movie_fraction", 0.06)))
    minimum_stable = _setting(
        params, "minimum_stable_owner_fraction", 0.60)
    minimum_strong = _setting(
        params, "minimum_excursion_strong_fraction", 0.70)
    maximum_encounter = _setting(
        params, "maximum_excursion_encounter_fraction", 0.60)
    reciprocal_window = int(math.ceil(frame_count * _setting(
        params, "maximum_reciprocal_window_movie_fraction", 0.15)))
    proposals: list[dict] = []
    for track, profile in sorted(profiles.items()):
        group = profile["group"]
        runs = _owner_runs(group)
        for index, run in enumerate(runs):
            run = runs[index]
            source = int(run["owner"])
            run_rows = list(run["rows"])
            first = int(run_rows[0].frame)
            last = int(run_rows[-1].frame)
            pre = runs[index - 1]["rows"][-1] if index > 0 else None
            post = runs[index + 1]["rows"][0] \
                if index + 1 < len(runs) else None
            lineage = int(pre.candidate_owner) if pre is not None else 0
            bounded = bool(
                pre is not None and post is not None and lineage > 0
                and source > 0 and source != lineage
                and int(post.candidate_owner) == lineage)
            counts = profile["counts"]
            stable_fraction = float(counts.get(lineage, 0) / max(
                counts.get(lineage, 0) + counts.get(source, 0), 1)) \
                if lineage > 0 else 0.0
            strong_fraction = float(np.mean([bool(row.strong)
                                             for row in run_rows]))
            encounter_fraction = float(np.mean([bool(row.in_encounter)
                                                for row in run_rows]))
            chain = [pre, *run_rows, post] if bounded else run_rows
            steps = [_normalised_step(left, right)
                     for left, right in zip(chain, chain[1:])]
            maximum_step = max(steps, default=float("inf"))
            reasons: list[str] = []
            if not bounded:
                reasons.append("track_run_not_bounded_by_same_lineage")
            if bounded and int(profile["strict_owner"]) != int(lineage):
                reasons.append("bracketing_owner_not_track_lineage")
            if bounded and (not bool(pre.strong) or not bool(post.strong)):
                reasons.append("lineage_flanks_lack_strong_raw_support")
            if last - first + 1 > maximum_span:
                reasons.append("excursion_exceeds_movie_fraction")
            if bounded and (first - int(pre.frame) > maximum_gap
                            or int(post.frame) - last > maximum_gap):
                reasons.append("lineage_flank_gap_too_long")
            if bounded and stable_fraction < minimum_stable:
                reasons.append("lineage_owner_not_stable_on_track")
            if strong_fraction < minimum_strong:
                reasons.append("excursion_raw_support_too_weak")
            if encounter_fraction > maximum_encounter:
                reasons.append("excursion_too_encounter_dominated")
            if bounded and maximum_step > step_threshold:
                reasons.append("excursion_motion_step_is_field_outlier")
            keys: set[ComponentKey] = set()
            missing = 0
            for point in run_rows:
                match = by_track_frame.get(
                    (int(point.track_id), int(point.frame)))
                if match is None or match.key is None:
                    missing += 1
                else:
                    keys.add(match.key)
            if missing or len({key.frame for key in keys}) != len(run_rows):
                reasons.append("excursion_component_mapping_incomplete")
            evidence = _component_evidence(
                keys, source, lineage, by_component, ambiguous, profiles,
                maximum_gap, protected_track_frames) if bounded else {
                    "tracks": set(), "direct": 0, "fuzzy": 0,
                    "minimum_fuzzy_margin": float("inf"),
                    "ambiguous_tracks": set(),
                    "unmatched_components": len(keys), "disagreements": set(),
                    "injective": False, "unanimous": False, "protected": set(),
                }
            if bounded and not evidence["injective"]:
                reasons.append("component_track_matching_not_injective")
            if bounded and not evidence["unanimous"]:
                reasons.append("component_lineage_not_unanimous")
            concurrent, reciprocal = (set(), set()) if not bounded else \
                _lineage_conflicts(
                    source, lineage, first, last, profiles, reciprocal_window)
            safety: list[str] = []
            if concurrent:
                safety.append("concurrent_independent_source_lineage")
            if reciprocal:
                safety.append("nearby_reciprocal_lineage_exchange")
            if evidence["protected"]:
                safety.append("accepted_track_frame_veto")
            kind = "bounded_component_excursion" if bounded else \
                "excluded_track_owner_run"
            row = _base_row(kind, source, lineage,
                            first, last, keys, {int(track)}, step_threshold)
            row.update({
                "stable_owner_fraction": stable_fraction,
                "strong_fraction": strong_fraction,
                "encounter_fraction": encounter_fraction,
                "maximum_step_radii": maximum_step,
                "supporting_tracks": "|".join(map(
                    str, sorted(evidence["tracks"]))),
                "direct_track_matches": int(evidence["direct"]),
                "fuzzy_track_matches": int(evidence["fuzzy"]),
                "minimum_fuzzy_unique_margin_radii": float(
                    evidence["minimum_fuzzy_margin"]),
                "ambiguous_track_matches": int(len(
                    evidence["ambiguous_tracks"])),
                "unmatched_components": int(evidence["unmatched_components"]),
                "lineage_disagreement_tracks": "|".join(map(
                    str, sorted(evidence["disagreements"]))),
                "component_track_matching_injective": bool(
                    evidence["injective"]),
                "component_lineage_unanimous": bool(evidence["unanimous"]),
                "concurrent_source_lineage_tracks": "|".join(
                    map(str, sorted(concurrent))),
                "reciprocal_lineage_tracks": "|".join(
                    map(str, sorted(reciprocal))),
                "accepted_track_frame_overlaps": int(len(
                    evidence["protected"])),
                "_component_evidence": evidence,
                "_local_reasons": list(dict.fromkeys(reasons)),
                "_safety_reasons": list(dict.fromkeys(safety)),
            })
            row["locally_qualified"] = not row["_local_reasons"]
            proposals.append(row)
    return proposals


def _overlap(left: dict, right: dict) -> bool:
    first = max(int(left["first_frame"]), int(right["first_frame"]))
    last = min(int(left["last_frame"]), int(right["last_frame"]))
    if left["_component_keys"] & right["_component_keys"]:
        return True
    if first > last:
        return False
    left_owners = {int(left["source_identity"]),
                   int(left["lineage_identity"])} - {0}
    right_owners = {int(right["source_identity"]),
                    int(right["lineage_identity"])} - {0}
    if left_owners & right_owners:
        return True
    return bool(first == last
                and int(left["lineage_identity"]) > 0
                and int(left["lineage_identity"])
                == int(right["lineage_identity"]))


def _atomic_groups(proposals: list[dict]) -> list[list[int]]:
    candidates = [index for index, row in enumerate(proposals)
                  if bool(row["locally_qualified"])]
    adjacency = {index: set() for index in candidates}
    for offset, left in enumerate(candidates):
        for right in candidates[offset + 1:]:
            if _overlap(proposals[left], proposals[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
    groups: list[list[int]] = []
    unseen = set(candidates)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        group: list[int] = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in sorted(adjacency[current] & unseen):
                unseen.remove(neighbour)
                stack.append(neighbour)
        groups.append(sorted(group))
    return groups


def _parallel_short_excursion_is_supported(rows: list[dict]) -> bool:
    """Return whether a short event has independent synchronous support.

    Complete-alias rows may duplicate the same evidence, so only bounded
    component proposals count as independent branches.
    """
    bounded = [row for row in rows
               if row["proposal_kind"] == "bounded_component_excursion"]
    if len(bounded) < 2:
        return False
    return bool(
        len({int(row["source_identity"]) for row in bounded}) == len(bounded)
        and len({int(row["lineage_identity"]) for row in bounded}) == 1
        and len({(int(row["first_frame"]), int(row["last_frame"]))
                 for row in bounded}) == 1
    )


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict,
             application_audit: pd.DataFrame,
             conflicted_audit: pd.DataFrame,
             proposal_ledger: pd.DataFrame,
             frame_ledger: pd.DataFrame,
             ) -> tuple[pd.DataFrame, list[dict], float]:
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    visible = attached[
        attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).gt(0)].copy()
    frame_count = int(len(labels))
    maximum_gap = max(1, int(math.ceil(frame_count * _setting(
        params, "maximum_flank_gap_movie_fraction", 0.03))))
    step_quantile = _setting(
        params, "maximum_motion_step_quantile", 0.90)
    physical_visible = attached[attached.physically_visible.astype(bool)].copy()
    step_threshold = _field_step_threshold(
        physical_visible, step_quantile, maximum_gap)
    components, component_owners = _component_maps(labels)
    profiles = _track_profiles(visible)
    presence = _presence(labels)
    maximum_fuzzy_distance = _setting(
        params, "maximum_fuzzy_distance_radii", 1.0, 0.0, 10.0)
    minimum_fuzzy_margin = _setting(
        params, "minimum_fuzzy_unique_margin_radii", 0.50, 0.0, 10.0)
    by_track_frame, by_component, ambiguous = _component_matches(
        labels, visible, components, component_owners,
        maximum_fuzzy_distance, minimum_fuzzy_margin)
    protected_track_frames = _accepted_track_frame_vetoes(
        application_audit, conflicted_audit, proposal_ledger, frame_ledger,
        frame_count)
    proposals = _complete_alias_proposals(
        labels, visible, profiles, components, component_owners, presence,
        params, maximum_gap, step_threshold, by_component, ambiguous,
        protected_track_frames)
    proposals.extend(_bounded_component_proposals(
        labels, visible, profiles, components, component_owners, params,
        step_threshold, maximum_gap, by_track_frame, by_component, ambiguous,
        protected_track_frames))
    for number, row in enumerate(proposals, start=1):
        row["proposal_id"] = f"BL{number:04d}"
    minimum_single_source_span = max(1, int(math.ceil(
        frame_count * _setting(
            params, "minimum_single_source_excursion_movie_fraction",
            0.05))))
    for number, members in enumerate(_atomic_groups(proposals), start=1):
        group_id = f"AG{number:04d}"
        group_rows = [proposals[index] for index in members]
        group_safety = list(dict.fromkeys(
            reason for index in members
            for reason in proposals[index]["_safety_reasons"]))
        short_bounded = [
            row for row in group_rows
            if row["proposal_kind"] == "bounded_component_excursion"
            and int(row["span_frames"]) < minimum_single_source_span
        ]
        if (short_bounded
                and not _parallel_short_excursion_is_supported(group_rows)):
            group_safety.append("isolated_excursion_below_movie_fraction")
        component_targets: dict[ComponentKey, set[int]] = {}
        for index in members:
            row = proposals[index]
            for key in row["_component_keys"]:
                component_targets.setdefault(key, set()).add(
                    int(row["lineage_identity"]))
        if any(len(targets) > 1 for targets in component_targets.values()):
            group_safety.append("atomic_component_has_multiple_targets")
        for index in members:
            row = proposals[index]
            row["atomic_group_id"] = group_id
            row["eligible"] = not group_safety
            reasons = [*row["_local_reasons"], *row["_safety_reasons"]]
            if group_safety and not row["_safety_reasons"]:
                reasons.append("atomic_peer_safety_refusal")
            row["reason"] = ("eligible_bounded_physical_lineage_recovery"
                             if not reasons else
                             "|".join(dict.fromkeys(reasons)))
    for row in proposals:
        if not row["locally_qualified"]:
            row["reason"] = "|".join(row["_local_reasons"])
    public = pd.DataFrame([
        {column: row.get(column, "") for column in AUDIT_COLUMNS}
        for row in proposals], columns=AUDIT_COLUMNS)
    return public, proposals, step_threshold


def apply(labels: np.ndarray, proposals: list[dict],
          ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    components, component_owners = _component_maps(labels)
    applications: list[dict] = []
    groups: dict[str, list[dict]] = {}
    for row in proposals:
        if row["locally_qualified"]:
            groups.setdefault(str(row["atomic_group_id"]), []).append(row)
    for group_id in sorted(groups):
        members = groups[group_id]
        proposal_ids = "|".join(str(row["proposal_id"]) for row in members)
        if not all(bool(row["eligible"]) for row in members):
            applications.append({
                "atomic_group_id": group_id,
                "member_proposals": proposal_ids,
                "proposal_count": int(len(members)), "changed_pixels": 0,
                "changed_frames": 0, "applied": False,
                "reason": "atomic_group_refused_by_safety_gate",
            })
            continue
        assignments: dict[ComponentKey, int] = {}
        conflict = False
        for row in members:
            target = int(row["lineage_identity"])
            for key in row["_component_keys"]:
                if key in assignments and assignments[key] != target:
                    conflict = True
                assignments[key] = target
        trial = candidate.copy()
        if not conflict:
            for key, target in assignments.items():
                component = components[key.frame] == int(key.component)
                source = int(component_owners[key.frame][key.component - 1])
                if (not np.any(component)
                        or not np.all(candidate[key.frame][component] == source)):
                    conflict = True
                    break
                trial[key.frame][component] = int(target)
        if conflict:
            applications.append({
                "atomic_group_id": group_id,
                "member_proposals": proposal_ids,
                "proposal_count": int(len(members)), "changed_pixels": 0,
                "changed_frames": 0, "applied": False,
                "reason": "atomic_group_component_conflict",
            })
            continue
        changed = trial != candidate
        candidate = trial
        applications.append({
            "atomic_group_id": group_id,
            "member_proposals": proposal_ids,
            "proposal_count": int(len(members)),
            "changed_pixels": int(changed.sum()),
            "changed_frames": int(np.count_nonzero(
                changed.reshape(len(changed), -1).any(axis=1))),
            "applied": True,
            "reason": "atomic_existing_component_relabel",
        })
    return candidate, pd.DataFrame(applications, columns=APPLICATION_COLUMNS)


def _duplicate_lineage_verification(
        baseline: np.ndarray, candidate: np.ndarray, proposals: list[dict],
        applications: pd.DataFrame,
        ) -> tuple[int, int]:
    applied_groups = set(applications.loc[
        applications.applied.astype(bool), "atomic_group_id"].astype(str)) \
        if len(applications) else set()
    proven_assignments: dict[tuple[int, int], set[ComponentKey]] = {}
    for row in proposals:
        if str(row["atomic_group_id"]) not in applied_groups:
            continue
        evidence = row["_component_evidence"]
        if not evidence.get("injective") or not evidence.get("unanimous"):
            continue
        target = int(row["lineage_identity"])
        for key in row["_component_keys"]:
            proven_assignments.setdefault((int(key.frame), target), set()).add(key)
    new_duplicates = 0
    proven_duplicates = 0
    changed_frames = np.flatnonzero(
        (baseline != candidate).reshape(len(baseline), -1).any(axis=1))
    for frame in changed_frames:
        before = baseline[int(frame)]
        after = candidate[int(frame)]
        owners = set(map(int, np.unique(after[before != after]))) - {0}
        for owner in owners:
            before_excess = component_accounting.component_excess(before, owner)
            after_excess = component_accounting.component_excess(after, owner)
            added = max(0, after_excess - before_excess)
            new_duplicates += added
            proven = len(proven_assignments.get((int(frame), int(owner)), set()))
            proven_duplicates += min(added, proven)
    return int(new_duplicates), int(proven_duplicates)


def _read_audit(path_value: object) -> pd.DataFrame:
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"accepted audit not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _copy_audits(params: dict, output_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for key, name in (("application_audit_path", "application_audit.csv"),
                      ("conflicted_lineage_audit_path",
                       "conflicted_lineage_audit.csv"),
                      ("distinct_history_proposals_path",
                       "distinct_history_proposals.csv"),
                      ("distinct_history_frames_path",
                       "distinct_history_frames.csv")):
        source = Path(str(params[key]))
        target = output_dir / name
        shutil.copyfile(source, target)
        outputs[name.removesuffix(".csv")] = target
    return outputs


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    output_dir = Path(out.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    application_audit = _read_audit(params["application_audit_path"])
    conflicted_audit = _read_audit(params["conflicted_lineage_audit_path"])
    proposal_ledger = _read_audit(params["distinct_history_proposals_path"])
    frame_ledger = _read_audit(params["distinct_history_frames_path"])
    audit, internals, step_threshold = discover(
        labels, points, params, application_audit, conflicted_audit,
        proposal_ledger, frame_ledger)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame(columns=APPLICATION_COLUMNS)
    else:
        candidate, applications = apply(labels, internals)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("branched-lineage recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids - before_ids:
        raise AssertionError("branched-lineage recovery created an identity")
    output_stem = str(params.get("output_stem",
                                 Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{output_stem}.tif"
    unclaimed_path = output_dir / f"{output_stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    changed = candidate != labels
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    applied_group_ids = set(applied.atomic_group_id.astype(str)) \
        if len(applied) else set()
    applied_audit = audit[audit.atomic_group_id.astype(str).isin(
        applied_group_ids)] if len(audit) else audit
    protected_track_frames = _accepted_track_frame_vetoes(
        application_audit, conflicted_audit, proposal_ledger, frame_ledger,
        len(labels))
    new_duplicates, proven_duplicates = _duplicate_lineage_verification(
        labels, candidate, internals, applications)
    if new_duplicates != proven_duplicates:
        raise AssertionError(
            "new duplicate component lacks injective unanimous lineage evidence")
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": selector_counts(params),
        "parameter_keys_audited": int(len(params)),
        "physical_tracks_audited": int(points.track_id.nunique()),
        "identity_audit_rows": int(
            audit.proposal_kind.eq("complete_complement").sum()),
        "track_run_audit_rows": int(
            audit.proposal_kind.isin({
                "bounded_component_excursion", "excluded_track_owner_run"}).sum()),
        "detected_proposals": int(len(audit)),
        "locally_qualified_proposals": int(
            audit.locally_qualified.astype(bool).sum()) if len(audit) else 0,
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "atomic_groups": int(audit.loc[
            audit.locally_qualified.astype(bool), "atomic_group_id"].nunique())
            if len(audit) else 0,
        "applied_atomic_groups": int(len(applied)),
        "applied_proposals": int(len(applied_audit)),
        "complete_complement_proposals_applied": int(
            applied_audit.proposal_kind.eq("complete_complement").sum())
            if len(applied_audit) else 0,
        "bounded_component_proposals_applied": int(
            applied_audit.proposal_kind.eq(
                "bounded_component_excursion").sum())
            if len(applied_audit) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(len(changed_frames)),
        "changed_frame_indices": changed_frames,
        "field_step_threshold_radii": float(step_threshold),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": bool(np.array_equal(candidate > 0,
                                                         labels > 0)),
        "preexisting_unclaimed_exact": bool(np.array_equal(
            tifffile.imread(unclaimed_path), unclaimed)),
        "new_duplicate_components": int(new_duplicates),
        "lineage_proven_new_duplicate_components": int(proven_duplicates),
        "all_new_duplicates_injectively_unanimous": bool(
            new_duplicates == proven_duplicates),
        "accepted_application_audit_rows": int(len(application_audit)),
        "accepted_conflict_audit_rows": int(len(conflicted_audit)),
        "accepted_proposal_ledger_rows": int(len(proposal_ledger)),
        "accepted_frame_ledger_rows": int(len(frame_ledger)),
        "accepted_protected_track_frames": int(len(protected_track_frames)),
    }
    audit_path = output_dir / "branched_motion_lineage_audit.csv"
    applications_path = output_dir / "branched_motion_lineage_applications.csv"
    metrics_path = output_dir / "metrics.json"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(applications_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    outputs = {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "applications": applications_path,
        "metrics": metrics_path,
    }
    outputs.update(_copy_audits(params, output_dir))
    return {"outputs": outputs, "summary": metrics}
