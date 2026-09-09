"""Preserve two established identities through a terminal shared mask.

Start with the conservative resolved-split inheritance rule.  If its two
components later rejoin, require a unique companion physical track that was
inside the displaced component immediately before contact, remains strongly
visible throughout the shared interval, and forms two spatially separated raw
cores with a low valley in every frame.  A seeded watershed then partitions
the existing shared foreground.  The resolved suffix and terminal interval
are applied atomically.

No identity, track, frame, coordinate, event, region, or review-case selector
is accepted.  All identities and locations in audits are measured outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile

import distinct_history_transition_candidates as separation
import area_conservative_split_inheritance as base
import owner_consensus
from resident_takeover import attach_owners


EXTRA_PATH_KEYS = {"raw_path"}
EXTRA_NUMERIC_KEYS = {
    "minimum_terminal_prior_support_movie_fraction",
    "minimum_terminal_companion_strong_fraction",
    "minimum_terminal_core_separation_sum_radii",
    "maximum_terminal_core_valley_ratio", "minimum_seed_distance_px",
    "watershed_sigma_px", "minimum_basin_pixels",
    "minimum_expected_area_ratio", "maximum_expected_area_ratio",
}
ALLOWED_KEYS = base.ALLOWED_KEYS | EXTRA_PATH_KEYS | EXTRA_NUMERIC_KEYS


def selector_counts(params: dict) -> dict[str, int]:
    unexpected = [str(key) for key in params if str(key) not in ALLOWED_KEYS]
    roles = {
        "identities": ("identity", "identities", "cell", "cells"),
        "owners": ("owner", "owners"), "tracks": ("track", "tracks"),
        "frames": ("frame", "frames", "interval"),
        "coordinates": ("coordinate", "coordinates", "point", "points"),
        "events": ("event", "events"), "regions": ("region", "regions", "roi"),
        "review_cases": ("case", "cases", "review"),
    }
    return {role: sum(any(token in key.lower() for token in tokens)
                      for key in unexpected)
            for role, tokens in roles.items()}


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal split inheritance must be field-wide")
    unexpected = sorted(str(key) for key in params
                        if str(key) not in ALLOWED_KEYS)
    if unexpected:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(unexpected))
    missing = sorted(key for key in base.PATH_KEYS | EXTRA_PATH_KEYS
                     if not params.get(key))
    if missing:
        raise ValueError("missing evidence paths: " + ", ".join(missing))
    for key in EXTRA_NUMERIC_KEYS & set(params):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be one finite scalar")
    base.assert_target_free({key: value for key, value in params.items()
                             if key in base.ALLOWED_KEYS})


def _point_inside(mask: np.ndarray, point: object) -> bool:
    y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _restore_pair(candidate: np.ndarray, labels: np.ndarray,
                  established: int, incoming: int) -> None:
    changed = candidate != labels
    pair = np.isin(labels, (established, incoming)) \
        & np.isin(candidate, (established, incoming))
    candidate[changed & pair] = labels[changed & pair]


def _terminal_plan(labels: np.ndarray, raw: np.ndarray,
                   scored: pd.DataFrame, proposal: pd.Series,
                   terminal_frames: list[int], params: dict,
                   ) -> tuple[list[tuple[int, np.ndarray, int, int]], int,
                              list[str]]:
    established = int(proposal.established_owner)
    incoming = int(proposal.incoming_owner)
    seat_track = int(proposal.seat_track)
    first = min(terminal_frames)
    minimum_prior = max(1, int(math.ceil(len(labels) * float(params.get(
        "minimum_terminal_prior_support_movie_fraction", 0.03)))))
    prior_first = max(int(proposal.mutation_first_frame), first - minimum_prior * 3)
    strong_min = float(params.get(
        "minimum_terminal_companion_strong_fraction", 0.80))
    separation_min = float(params.get(
        "minimum_terminal_core_separation_sum_radii", 1.50))
    valley_max = float(params.get("maximum_terminal_core_valley_ratio", 0.65))
    seat_rows = scored[
        scored.track_id.astype(int).eq(seat_track)
        & scored.frame.astype(int).isin(terminal_frames)
        & scored.physically_visible.map(base._truth)]
    if set(seat_rows.frame.astype(int)) != set(terminal_frames):
        return [], 0, ["terminal_established_seat_incomplete"]
    candidates: list[tuple[int, pd.DataFrame]] = []
    for track, group in scored.groupby("track_id", sort=True):
        track = int(track)
        if track == seat_track:
            continue
        prior = group[
            group.frame.astype(int).between(prior_first, first - 1)
            & group.candidate_owner.astype(int).eq(established)
            & group.physically_visible.map(base._truth)]
        terminal = group[
            group.frame.astype(int).isin(terminal_frames)
            & group.physically_visible.map(base._truth)]
        if (len(prior) < minimum_prior
                or set(terminal.frame.astype(int)) != set(terminal_frames)
                or float(terminal.strong.map(base._truth).mean()) < strong_min):
            continue
        candidates.append((track, terminal.sort_values("frame")))
    valid: list[tuple[int, list[tuple[int, np.ndarray, int, int]]]] = []
    for track, companion_rows in candidates:
        items: list[tuple[int, np.ndarray, int, int]] = []
        failed = False
        for frame in terminal_frames:
            seat = seat_rows[seat_rows.frame.astype(int).eq(frame)].iloc[0]
            companion = companion_rows[
                companion_rows.frame.astype(int).eq(frame)].iloc[0]
            shared = base._component_at(labels[frame], incoming, seat)
            if shared is None or not _point_inside(shared[0], companion):
                failed = True
                break
            distance = float(np.hypot(float(seat.x) - float(companion.x),
                                      float(seat.y) - float(companion.y)))
            scale = max(float(seat.radius_px) + float(companion.radius_px), 1.0)
            if distance / scale < separation_min:
                failed = True
                break
            if separation._valley(raw[frame], seat, companion) > valley_max:
                failed = True
                break
            partition = separation._partition(
                shared[0], raw[frame], seat, companion,
                float(proposal.expected_seat_area),
                float(proposal.expected_incoming_area), params)
            if partition is None:
                failed = True
                break
            proposed = labels[frame].copy()
            proposed[partition[0]] = established
            proposed[partition[1]] = incoming
            items.append((frame, proposed, int(partition[0].sum()),
                          int(partition[1].sum())))
        if not failed and len(items) == len(terminal_frames):
            valid.append((track, items))
    if len(valid) != 1:
        return [], 0, ["terminal_companion_lineage_not_unique"]
    return valid[0][1], valid[0][0], []


def discover_and_apply(labels: np.ndarray, unclaimed: np.ndarray,
                       points: pd.DataFrame, raw: np.ndarray, params: dict,
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    base_params = {key: value for key, value in params.items()
                   if key in base.ALLOWED_KEYS}
    candidate, audit, frames = base.discover_and_apply(
        labels, unclaimed, points, base_params)
    if not len(audit):
        return candidate, audit, frames
    scored = attach_owners(points, labels)
    scored["physically_visible"] = scored.physically_visible.map(base._truth)
    for index, proposal in audit.iterrows():
        if not base._truth(proposal.applied):
            continue
        selected = frames[
            frames.proposal_id.astype(str).eq(str(proposal.proposal_id))
            & frames.action.astype(str).eq("terminal_remerge_held_out")]
        terminal_frames = sorted(selected.frame.astype(int).unique().tolist())
        if not terminal_frames:
            continue
        plans, companion_track, failures = _terminal_plan(
            labels, raw, scored, proposal, terminal_frames, params)
        if failures:
            _restore_pair(candidate, labels, int(proposal.established_owner),
                          int(proposal.incoming_owner))
            audit.at[index, "eligible"] = False
            audit.at[index, "applied"] = False
            audit.at[index, "changed_pixels"] = 0
            audit.at[index, "changed_frames"] = 0
            audit.at[index, "reason"] = "|".join(failures)
            continue
        for frame, proposed, established_pixels, incoming_pixels in plans:
            candidate[frame] = proposed
            mask = (frames.proposal_id.astype(str).eq(str(proposal.proposal_id))
                    & frames.frame.astype(int).eq(frame)
                    & frames.action.astype(str).eq("terminal_remerge_held_out"))
            frames.loc[mask, "seat_component_area"] = (
                established_pixels + incoming_pixels)
            frames.loc[mask, "displaced_component_area"] = incoming_pixels
            frames.loc[mask, "action"] = "terminal_two_core_partition"
            frames.loc[mask, "changed_pixels"] = established_pixels
            frames.loc[mask, "eligible"] = True
            frames.loc[mask, "reason"] = \
                "unique_companion_raw_core_partition"
        pair_changed = (candidate != labels) \
            & np.isin(labels, (int(proposal.established_owner),
                               int(proposal.incoming_owner)))
        audit.at[index, "changed_pixels"] = int(pair_changed.sum())
        audit.at[index, "changed_frames"] = int(np.count_nonzero(
            pair_changed.reshape(len(pair_changed), -1).any(axis=1)))
        audit.at[index, "reason"] = \
            "applied_resolved_split_and_terminal_two_core_partition"
        audit.at[index, "terminal_companion_track"] = companion_track
    if (not np.array_equal(candidate > 0, labels > 0)
            or np.any((candidate > 0) & (unclaimed > 0))
            or set(map(int, np.unique(candidate)))
            != set(map(int, np.unique(labels)))
            or owner_consensus.count_new_duplicate_components(
                labels, candidate) > 0):
        raise AssertionError("terminal two-core split violated invariants")
    return candidate, audit, frames


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, audit, frames = discover_and_apply(
        labels, unclaimed, points, raw, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("terminal split inheritance changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("terminal split inheritance claimed unclaimed pixels")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("terminal split inheritance changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("terminal split inheritance created duplicates")
    output_dir = Path(out.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", "95_A3"))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(labels_path, candidate, compression="zlib")
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    audit_path = output_dir / "terminal_two_core_split_audit.csv"
    frames_path = output_dir / "terminal_two_core_split_frames.csv"
    audit.to_csv(audit_path, index=False)
    frames.to_csv(frames_path, index=False)
    changed = candidate != labels
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {name: 0 for name in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "physical_tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.map(base._truth).sum()),
        "applied_proposals": int(audit.applied.map(base._truth).sum()),
        "terminal_partitions": int((frames.action.astype(str)
                                     == "terminal_two_core_partition").sum()),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_exact": True, "unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(duplicates),
        "parameters": {key: value for key, value in params.items()
                       if key not in base.PATH_KEYS | EXTRA_PATH_KEYS},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {"labels": labels_path, "unclaimed": unclaimed_path,
                        "audit": audit_path, "frames": frames_path,
                        "metrics": metrics_path}, "summary": metrics}



