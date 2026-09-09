"""Preserve a long-established physical seat through a two-body split.

The producer audits every physical track and every positive owner transition.
It recognises a reciprocal area exchange: the incoming owner takes the
established seat-sized component while the established owner moves to a
nearby component whose size matches the incoming owner's pre-contact mask.
Those two components are swapped together.  A short terminal re-merge is
audited but held out because one shared mask cannot preserve both identities.

No identity, track, frame, coordinate, event, region, or review-case value can
be supplied as a selector.  All such values in the audit are measured output.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import owner_consensus
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
PATH_KEYS = {
    "labels_path", "unclaimed_path", "physical_track_points_path",
    "parent_lineage_audit_path", "parent_lineage_applications_path",
}
CONTROL_KEYS = {"mode", "targeting_mode", "output_stem"}
NUMERIC_KEYS = {
    "minimum_prefix_movie_fraction", "minimum_track_coverage",
    "minimum_prefix_owner_purity", "minimum_prefix_strong_fraction",
    "minimum_suffix_incoming_owner_fraction",
    "maximum_track_gap_movie_fraction",
    "minimum_local_incoming_reference_movie_fraction",
    "maximum_exchange_distance_sum_radii",
    "maximum_seat_log_area_ratio", "maximum_displaced_log_area_ratio",
    "minimum_displaced_area_fraction", "minimum_exchange_fraction",
    "maximum_terminal_merge_movie_fraction",
    "maximum_terminal_combined_log_area_ratio",
    "minimum_unique_cost_margin", "maximum_motion_step_quantile",
}
ALLOWED_KEYS = PATH_KEYS | CONTROL_KEYS | NUMERIC_KEYS
AUDIT_COLUMNS = [
    "proposal_id", "seat_track", "established_owner", "incoming_owner",
    "transition_frame", "prefix_frames", "track_coverage",
    "prefix_owner_purity", "prefix_strong_fraction",
    "suffix_incoming_owner_fraction", "transition_step_radii",
    "field_motion_step_radii", "expected_seat_area",
    "expected_incoming_area", "incoming_reference_frames",
    "reacquisition_frame", "mutation_first_frame", "suffix_frames",
    "reciprocal_swap_frames", "seat_owner_frames",
    "terminal_merge_frames", "exchange_fraction",
    "parent_lineage_conflict", "eligible", "applied", "changed_pixels",
    "changed_frames", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "seat_observed_owner", "seat_component_area",
    "displaced_component_area", "component_distance_sum_radii",
    "seat_log_area_ratio", "displaced_log_area_ratio", "action",
    "changed_pixels", "eligible", "reason",
]


def selector_counts(params: dict) -> dict[str, int]:
    unexpected = [str(key) for key in params if str(key) not in ALLOWED_KEYS]
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
        raise ValueError("split inheritance must be field-wide")
    if params.get("mode", "candidate") not in {"baseline", "candidate", "accepted"}:
        raise ValueError("mode must be baseline, candidate, or accepted")
    unexpected = sorted(str(key) for key in params
                        if str(key) not in ALLOWED_KEYS)
    if unexpected:
        raise ValueError("unsupported parameters (selectors forbidden): "
                         + ", ".join(unexpected))
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing evidence paths: " + ", ".join(missing))
    for key in NUMERIC_KEYS & set(params):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be one finite scalar")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _component_at(frame: np.ndarray, owner: int, row: object):
    components, _ = ndi.label(frame == int(owner), STRUCTURE)
    x = min(max(int(round(float(row.x))), 0), frame.shape[1] - 1)
    y = min(max(int(round(float(row.y))), 0), frame.shape[0] - 1)
    number = int(components[y, x])
    if number <= 0:
        return None
    mask = components == number
    yx = np.mean(np.argwhere(mask), axis=0)
    return mask, int(mask.sum()), float(yx[1]), float(yx[0])


def _owner_components(frame: np.ndarray, owner: int) -> list[tuple]:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    result = []
    for number in range(1, count + 1):
        mask = components == number
        yx = np.mean(np.argwhere(mask), axis=0)
        result.append((mask, int(mask.sum()), float(yx[1]), float(yx[0])))
    return result


def _normalised_step(left: object, right: object) -> float:
    distance = float(np.hypot(float(right.x) - float(left.x),
                              float(right.y) - float(left.y)))
    gap = max(int(right.frame) - int(left.frame), 1)
    radius = math.sqrt(max(
        (float(left.radius_px) ** 2 + float(right.radius_px) ** 2) / 2.0,
        np.finfo(float).eps))
    return distance / (radius * gap)


def _field_motion_threshold(points: pd.DataFrame, quantile: float,
                            maximum_gap: int) -> float:
    values: list[float] = []
    visible = points[points.physically_visible.map(_truth)]
    for _, group in visible.groupby("track_id", sort=False):
        rows = list(group.sort_values("frame").itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            if (int(right.frame) - int(left.frame) <= maximum_gap
                    and _truth(left.strong) and _truth(right.strong)):
                values.append(_normalised_step(left, right))
    if not values:
        raise ValueError("physical evidence has no strong motion steps")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _split_members(value: object) -> set[int]:
    result: set[int] = set()
    for token in str(value).split("|"):
        try:
            result.add(int(float(token.strip())))
        except (TypeError, ValueError):
            pass
    return result


def _protected_parent(params: dict) -> tuple[set[int], set[int]]:
    audit = pd.read_csv(params["parent_lineage_audit_path"])
    applications = pd.read_csv(params["parent_lineage_applications_path"])
    if not len(audit) or not len(applications):
        return set(), set()
    applied_groups = set(applications.loc[
        applications.applied.map(_truth), "atomic_group_id"].astype(str))
    applied = audit[audit.atomic_group_id.astype(str).isin(applied_groups)]
    owners: set[int] = set()
    tracks: set[int] = set()
    for name in ("source_identity", "lineage_identity"):
        if name in applied:
            owners.update(pd.to_numeric(
                applied[name], errors="coerce").dropna().astype(int))
    if "supporting_tracks" in applied:
        for value in applied.supporting_tracks:
            tracks.update(_split_members(value))
    return owners, tracks


def _maximum_gap(frames: list[int]) -> int:
    return max((right - left - 1 for left, right in zip(frames, frames[1:])),
               default=0)


def _local_incoming_areas(labels: np.ndarray, scored: pd.DataFrame,
                          seat_track: int, incoming_owner: int,
                          transition: int, maximum_distance: float,
                          ) -> tuple[list[int], set[int]]:
    seat_rows = scored[(scored.track_id.astype(int) == int(seat_track))
                       & (scored.frame.astype(int) < int(transition))]
    areas: list[int] = []
    frames: set[int] = set()
    for row in seat_rows.itertuples(index=False):
        seat = _component_at(labels[int(row.frame)],
                             int(row.candidate_owner), row)
        if seat is None:
            continue
        _, seat_area, seat_x, seat_y = seat
        for _, area, x, y in _owner_components(
                labels[int(row.frame)], incoming_owner):
            scale = math.sqrt(seat_area / math.pi) + math.sqrt(area / math.pi)
            distance = float(np.hypot(x - seat_x, y - seat_y)) / max(scale, 1.0)
            if distance <= maximum_distance:
                areas.append(int(area))
                frames.add(int(row.frame))
    return areas, frames


def _transition_candidates(labels: np.ndarray, scored: pd.DataFrame,
                           params: dict, field_motion: float) -> list[dict]:
    frame_count = len(labels)
    maximum_gap = max(1, int(math.ceil(frame_count * float(params.get(
        "maximum_track_gap_movie_fraction", 0.02)))))
    minimum_prefix = int(math.ceil(frame_count * float(params.get(
        "minimum_prefix_movie_fraction", 0.40))))
    minimum_references = int(math.ceil(frame_count * float(params.get(
        "minimum_local_incoming_reference_movie_fraction", 0.02))))
    maximum_distance = float(params.get(
        "maximum_exchange_distance_sum_radii", 3.0))
    proposals: list[dict] = []
    proposal_number = 0
    visible = scored[scored.physically_visible.map(_truth)
                     & scored.candidate_owner.astype(int).gt(0)]
    for track_id, group in visible.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        rows = list(group.itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            established = int(left.candidate_owner)
            incoming = int(right.candidate_owner)
            transition = int(right.frame)
            if (established == incoming
                    or transition - int(left.frame) > maximum_gap + 1):
                continue
            prefix = group[group.frame.astype(int) < transition]
            suffix = group[group.frame.astype(int) >= transition]
            if len(prefix) < minimum_prefix or not len(suffix):
                continue
            prefix_purity = float(np.mean(
                prefix.candidate_owner.astype(int) == established))
            prefix_strong = float(np.mean(prefix.strong.map(_truth)))
            track_coverage = float(len(group.frame.astype(int).unique())
                                   / frame_count)
            suffix_purity = float(np.mean(
                suffix.candidate_owner.astype(int) == incoming))
            transition_step = _normalised_step(left, right)
            if (track_coverage < float(params.get(
                    "minimum_track_coverage", 0.95))
                    or prefix_purity < float(params.get(
                        "minimum_prefix_owner_purity", 0.98))
                    or prefix_strong < float(params.get(
                        "minimum_prefix_strong_fraction", 0.75))
                    or suffix_purity < float(params.get(
                        "minimum_suffix_incoming_owner_fraction", 0.95))
                    or _maximum_gap(list(group.frame.astype(int))) > maximum_gap
                    or transition_step > field_motion):
                continue
            seat_areas: list[int] = []
            for row in prefix[prefix.candidate_owner.astype(int).eq(
                    established)].itertuples(index=False):
                component = _component_at(
                    labels[int(row.frame)], established, row)
                if component is not None:
                    seat_areas.append(int(component[1]))
            incoming_areas, reference_frames = _local_incoming_areas(
                labels, scored, int(track_id), incoming, transition,
                maximum_distance)
            if not seat_areas or len(reference_frames) < minimum_references:
                continue
            proposal_number += 1
            proposals.append({
                "proposal_id": f"AS{proposal_number:04d}",
                "seat_track": int(track_id),
                "established_owner": established,
                "incoming_owner": incoming,
                "transition_frame": transition,
                "prefix_frames": int(len(prefix)),
                "track_coverage": track_coverage,
                "prefix_owner_purity": prefix_purity,
                "prefix_strong_fraction": prefix_strong,
                "suffix_incoming_owner_fraction": suffix_purity,
                "transition_step_radii": transition_step,
                "field_motion_step_radii": field_motion,
                "expected_seat_area": float(np.median(seat_areas)),
                "expected_incoming_area": float(np.median(incoming_areas)),
                "incoming_reference_frames": int(len(reference_frames)),
                "suffix": suffix,
            })
    return proposals


def _plan_proposal(labels: np.ndarray, proposal: dict, params: dict,
                   ) -> tuple[list[dict], list[tuple[int, np.ndarray]], dict]:
    established = int(proposal["established_owner"])
    incoming = int(proposal["incoming_owner"])
    expected_seat = float(proposal["expected_seat_area"])
    expected_incoming = float(proposal["expected_incoming_area"])
    maximum_distance = float(params.get(
        "maximum_exchange_distance_sum_radii", 3.0))
    maximum_seat_ratio = float(params.get("maximum_seat_log_area_ratio", 1.0))
    maximum_displaced_ratio = float(params.get(
        "maximum_displaced_log_area_ratio", 1.4))
    minimum_displaced = expected_incoming * float(params.get(
        "minimum_displaced_area_fraction", 0.25))
    unique_margin = float(params.get("minimum_unique_cost_margin", 0.05))
    frame_count = len(labels)
    maximum_terminal = int(math.ceil(frame_count * float(params.get(
        "maximum_terminal_merge_movie_fraction", 0.10))))
    terminal_start = frame_count
    for frame in range(frame_count - 1, -1, -1):
        if np.any(labels[frame] == established):
            break
        terminal_start = frame
    terminal_length = frame_count - terminal_start
    complete_suffix = proposal.pop("suffix")
    reacquired = complete_suffix[
        complete_suffix.candidate_owner.astype(int).eq(established)]
    reacquisition_frame = int(reacquired.frame.astype(int).max()) \
        if len(reacquired) else -1
    mutation_first_frame = reacquisition_frame + 1
    suffix = complete_suffix[
        complete_suffix.frame.astype(int) >= mutation_first_frame]
    rows: list[dict] = []
    plans: list[tuple[int, np.ndarray]] = []
    actions: list[str] = []
    failures: list[str] = []
    if reacquisition_frame < int(proposal["transition_frame"]):
        failures.append("no_post_contact_established_owner_reacquisition")
        suffix = suffix.iloc[0:0]
    for seat in suffix.itertuples(index=False):
        frame = int(seat.frame)
        observed = int(seat.candidate_owner)
        row = {
            "proposal_id": proposal["proposal_id"], "frame": frame,
            "seat_observed_owner": observed, "seat_component_area": 0,
            "displaced_component_area": 0,
            "component_distance_sum_radii": np.nan,
            "seat_log_area_ratio": np.nan,
            "displaced_log_area_ratio": np.nan, "action": "",
            "changed_pixels": 0, "eligible": False, "reason": "",
        }
        if observed == established:
            row.update({"action": "established_owner_retained",
                        "eligible": True, "reason": "already_correct"})
            actions.append(row["action"])
            rows.append(row)
            continue
        if observed != incoming:
            row["reason"] = "third_owner_on_established_seat"
            failures.append(row["reason"])
            rows.append(row)
            continue
        seat_component = _component_at(labels[frame], incoming, seat)
        if seat_component is None:
            row["reason"] = "incoming_owner_component_missing"
            failures.append(row["reason"])
            rows.append(row)
            continue
        seat_mask, seat_area, seat_x, seat_y = seat_component
        seat_log_ratio = abs(math.log(max(seat_area, 1) / expected_seat))
        row.update({"seat_component_area": seat_area,
                    "seat_log_area_ratio": seat_log_ratio})
        candidates: list[tuple[float, tuple, float, float]] = []
        for component in _owner_components(labels[frame], established):
            _, area, x, y = component
            if area < minimum_displaced:
                continue
            scale = math.sqrt(seat_area / math.pi) + math.sqrt(area / math.pi)
            distance = float(np.hypot(x - seat_x, y - seat_y)) / max(scale, 1.0)
            area_ratio = abs(math.log(max(area, 1) / expected_incoming))
            if (distance <= maximum_distance
                    and seat_log_ratio <= maximum_seat_ratio
                    and area_ratio <= maximum_displaced_ratio):
                candidates.append((area_ratio + 0.15 * distance,
                                   component, distance, area_ratio))
        candidates.sort(key=lambda item: item[0])
        if candidates and (len(candidates) == 1
                           or candidates[1][0] - candidates[0][0]
                           >= unique_margin):
            _, displaced, distance, displaced_ratio = candidates[0]
            displaced_mask, displaced_area, _, _ = displaced
            proposed = labels[frame].copy()
            proposed[seat_mask] = established
            proposed[displaced_mask] = incoming
            changed = int(np.count_nonzero(proposed != labels[frame]))
            row.update({
                "displaced_component_area": displaced_area,
                "component_distance_sum_radii": distance,
                "displaced_log_area_ratio": displaced_ratio,
                "action": "reciprocal_component_swap",
                "changed_pixels": changed, "eligible": True,
                "reason": "area_conservative_reciprocal_exchange",
            })
            actions.append(row["action"])
            plans.append((frame, proposed))
            rows.append(row)
            continue
        terminal = (terminal_length > 0 and terminal_length <= maximum_terminal
                    and frame >= terminal_start
                    and not np.any(labels[frame] == established))
        combined_log_ratio = abs(math.log(
            max(seat_area, 1) / (expected_seat + expected_incoming)))
        if (terminal and seat_log_ratio <= maximum_seat_ratio
                and combined_log_ratio <= float(params.get(
                    "maximum_terminal_combined_log_area_ratio", 1.0))):
            row.update({
                "action": "terminal_remerge_held_out",
                "changed_pixels": 0, "eligible": True,
                "reason": "ambiguous_shared_mask_left_unchanged",
            })
            actions.append(row["action"])
        else:
            row["reason"] = "reciprocal_area_exchange_not_proven"
            failures.append(row["reason"])
        rows.append(row)
    reciprocal = actions.count("reciprocal_component_swap")
    incoming_frames = sum(int(value in {
        "reciprocal_component_swap", "terminal_remerge_held_out"})
        for value in actions)
    exchange_fraction = reciprocal / max(incoming_frames, 1)
    summary = {
        "reacquisition_frame": int(reacquisition_frame),
        "mutation_first_frame": int(mutation_first_frame),
        "suffix_frames": int(len(suffix)),
        "reciprocal_swap_frames": int(reciprocal),
        "seat_owner_frames": int(actions.count("established_owner_retained")),
        "terminal_merge_frames": int(actions.count(
            "terminal_remerge_held_out")),
        "exchange_fraction": float(exchange_fraction),
        "failures": failures,
    }
    return rows, plans, summary


def discover_and_apply(labels: np.ndarray, unclaimed: np.ndarray,
                       points: pd.DataFrame, params: dict,
                       ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    assert_target_free(params)
    scored = attach_owners(points, labels)
    scored["physically_visible"] = scored.physically_visible.map(_truth)
    maximum_gap = max(1, int(math.ceil(len(labels) * float(params.get(
        "maximum_track_gap_movie_fraction", 0.02)))))
    field_motion = _field_motion_threshold(
        scored, float(params.get("maximum_motion_step_quantile", 0.90)),
        maximum_gap + 1)
    protected_owners, protected_tracks = _protected_parent(params)
    proposals = _transition_candidates(labels, scored, params, field_motion)
    audit_rows: list[dict] = []
    frame_rows: list[dict] = []
    accepted_plans: list[dict] = []
    for proposal in proposals:
        suffix = proposal["suffix"]
        plan_input = dict(proposal)
        plan_input["suffix"] = suffix
        rows, plans, summary = _plan_proposal(labels, plan_input, params)
        frame_rows.extend(rows)
        conflict = (int(proposal["seat_track"]) in protected_tracks
                    or {int(proposal["established_owner"]),
                        int(proposal["incoming_owner"])} & protected_owners)
        eligible = (not summary["failures"]
                    and summary["reciprocal_swap_frames"] > 0
                    and summary["exchange_fraction"] >= float(params.get(
                        "minimum_exchange_fraction", 0.80))
                    and not conflict)
        reason = "eligible_area_conservative_split_inheritance" if eligible \
            else ("parent_lineage_conflict" if conflict
                  else "|".join(dict.fromkeys(summary["failures"]))
                  or "insufficient_reciprocal_exchange")
        row = {key: proposal.get(key, "") for key in AUDIT_COLUMNS}
        row.update(summary)
        row.update({"parent_lineage_conflict": bool(conflict),
                    "eligible": bool(eligible), "applied": False,
                    "changed_pixels": 0, "changed_frames": 0,
                    "reason": reason})
        audit_rows.append(row)
        if eligible:
            accepted_plans.append({"row": row, "frames": plans})

    # Proposals sharing pixels, owners, or physical seats are not independent.
    conflicts: set[str] = set()
    for index, left in enumerate(accepted_plans):
        left_row = left["row"]
        left_owners = {int(left_row["established_owner"]),
                       int(left_row["incoming_owner"])}
        left_changes = {frame: proposed != labels[frame]
                        for frame, proposed in left["frames"]}
        for right in accepted_plans[index + 1:]:
            right_row = right["row"]
            right_owners = {int(right_row["established_owner"]),
                            int(right_row["incoming_owner"])}
            right_changes = {frame: proposed != labels[frame]
                             for frame, proposed in right["frames"]}
            overlap = any(frame in right_changes
                          and np.any(mask & right_changes[frame])
                          for frame, mask in left_changes.items())
            if (left_owners & right_owners or overlap
                    or int(left_row["seat_track"])
                    == int(right_row["seat_track"])):
                conflicts.update({str(left_row["proposal_id"]),
                                  str(right_row["proposal_id"])})

    candidate = labels.copy()
    for plan in accepted_plans:
        row = plan["row"]
        if str(row["proposal_id"]) in conflicts:
            row.update({"eligible": False, "reason": "global_atomic_conflict"})
            continue
        trial = candidate.copy()
        for frame, proposed in plan["frames"]:
            trial[frame] = proposed
        before_ids = set(map(int, np.unique(labels)))
        after_ids = set(map(int, np.unique(trial)))
        if (not np.array_equal(trial > 0, labels > 0)
                or np.any((trial > 0) & (unclaimed > 0))
                or before_ids != after_ids
                or owner_consensus.count_new_duplicate_components(
                    labels, trial) > 0):
            row.update({"eligible": False,
                        "reason": "global_atomic_invariant_rejection"})
            continue
        changed = trial != candidate
        candidate = trial
        row.update({
            "applied": True,
            "changed_pixels": int(changed.sum()),
            "changed_frames": int(np.count_nonzero(
                changed.reshape(len(changed), -1).any(axis=1))),
            "reason": "applied_area_conservative_split_inheritance",
        })
    return (candidate,
            pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(frame_rows, columns=FRAME_COLUMNS))


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, audit, frames = discover_and_apply(
        labels, unclaimed, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("split inheritance changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("split inheritance claimed unclaimed pixels")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("split inheritance changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError("split inheritance created duplicate components")

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
    audit_path = output_dir / "area_conservative_split_audit.csv"
    frames_path = output_dir / "area_conservative_split_frames.csv"
    metrics_path = output_dir / "metrics.json"
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
        "eligible_proposals": int(audit.eligible.map(_truth).sum())
            if len(audit) else 0,
        "applied_proposals": int(audit.applied.map(_truth).sum())
            if len(audit) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_exact": bool(np.array_equal(candidate > 0, labels > 0)),
        "unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(duplicates),
        "parameters": {key: value for key, value in params.items()
                       if key not in PATH_KEYS},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "frames": frames_path,
        "metrics": metrics_path,
    }, "summary": metrics}


