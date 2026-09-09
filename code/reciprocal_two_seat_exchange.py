"""Repair isolated one-frame reciprocal owner exchanges field-wide.

The producer discovers A-to-B/B-to-A transitions from current-run physical
tracks and accepted labels. It accepts mathematical thresholds only;
identities, tracks, frames, coordinates, events, regions, and review cases are
forbidden parameters.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import owner_consensus
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_pair", "forced_interval", "failure_target",
)
AUDIT_COLUMNS = [
    "proposal_id", "frame", "track_a", "track_b", "owner_a", "owner_b",
    "prior_run_a", "prior_run_b", "minimum_prior_run",
    "previous_separation_sum_radii", "transition_separation_sum_radii",
    "track_a_returned_immediately", "track_b_returned_immediately",
    "third_pair_owner_cores", "endpoint_status_a", "endpoint_status_b",
    "endpoint_successor_a", "endpoint_successor_b", "eligible", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("reciprocal two-seat exchange must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "reciprocal two-seat exchange received forbidden targets: "
            + ", ".join(supplied))


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _valid_point(point, minimum_purity: float,
                 minimum_coverage: float) -> bool:
    return bool(
        point is not None
        and bool(point.physically_visible)
        and int(point.candidate_owner) > 0
        and float(point.candidate_owner_purity) >= minimum_purity
        and float(point.candidate_coverage_fraction) >= minimum_coverage)


def _prior_run(index: pd.DataFrame, track: int, transition: int,
               owner: int, minimum_purity: float,
               minimum_coverage: float) -> int:
    frames = 0
    frame = int(transition) - 1
    while frame >= 0:
        point = _point(index, track, frame)
        if (not _valid_point(point, minimum_purity, minimum_coverage)
                or int(point.candidate_owner) != int(owner)):
            break
        frames += 1
        frame -= 1
    return frames


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    radii = float(left.radius_px) + float(right.radius_px)
    return distance / max(radii, 1.0)


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    if count <= 0:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    value = int(components[y, x])
    if value <= 0:
        radius = max(2.0, 0.75 * float(point.radius_px))
        yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
        disk = (xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2 \
            <= radius ** 2
        values = components[disk & (components > 0)]
        if not len(values):
            return None
        ids, counts = np.unique(values, return_counts=True)
        value = int(ids[int(np.argmax(counts))])
    return components == value


def _endpoint_link(points: pd.DataFrame, index: pd.DataFrame, track: int,
                   transition: int, maximum_distance: float,
                   minimum_purity: float, minimum_coverage: float,
                   movie_frames: int) -> tuple[str, int]:
    """Audit whether a post-transition raw-track endpoint is unambiguous."""
    group = points[
        (points.track_id.astype(int) == int(track))
        & (points.frame.astype(int) >= int(transition))
        & points.physically_visible.astype(bool)].sort_values("frame")
    if group.empty:
        return "missing_post_transition_lineage", 0
    last = group.iloc[-1]
    last_frame = int(last.frame)
    if last_frame >= int(movie_frames) - 1:
        return "continuous_to_recording_end", int(track)
    successor_frame = last_frame + 1
    options: list[int] = []
    for successor in points[
            (points.frame.astype(int) == successor_frame)
            & points.physically_visible.astype(bool)].itertuples(index=False):
        successor_track = int(successor.track_id)
        if successor_track == int(track):
            continue
        lineage = points[
            (points.track_id.astype(int) == successor_track)
            & points.physically_visible.astype(bool)]
        if lineage.empty or int(lineage.frame.min()) != successor_frame:
            continue
        if not _valid_point(successor, minimum_purity, minimum_coverage):
            continue
        if _scaled_distance(last, successor) <= maximum_distance:
            options.append(successor_track)
    if len(options) == 1:
        return "unique_endpoint_successor", int(options[0])
    if not options:
        return "no_endpoint_successor", 0
    return "ambiguous_endpoint_successor", 0


def _third_pair_owner_cores(attached: pd.DataFrame, frame: int,
                            excluded_tracks: set[int], owners: set[int],
                            component_a: np.ndarray,
                            component_b: np.ndarray) -> int:
    count = 0
    rows = attached[
        (attached.frame.astype(int) == int(frame))
        & attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).isin(owners)]
    for point in rows.itertuples(index=False):
        if int(point.track_id) in excluded_tracks:
            continue
        y = int(np.clip(round(float(point.y)), 0, component_a.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, component_a.shape[1] - 1))
        count += int(bool(component_a[y, x] or component_b[y, x]))
    return count


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all reciprocal transitions and their field-wide gate evidence."""
    assert_target_free(params)
    attached = attach_owners(points, labels)
    attached = attached.sort_values(["frame", "track_id"]).reset_index(drop=True)
    index = attached.set_index(["track_id", "frame"], drop=False)
    minimum_purity = float(params.get("minimum_owner_purity", 0.8))
    minimum_coverage = float(params.get("minimum_owner_coverage", 0.5))
    minimum_prior = max(
        int(params.get("minimum_prior_owner_frames", 4)),
        int(math.ceil(float(params.get(
            "minimum_prior_movie_fraction", 0.08)) * len(labels))))
    maximum_separation = float(params.get(
        "maximum_seat_separation_sum_radii", 3.0))
    maximum_endpoint = float(params.get(
        "maximum_endpoint_distance_sum_radii", 1.5))
    transition_rows: dict[int, list[dict]] = {}
    for current in attached.itertuples(index=False):
        frame = int(current.frame)
        if frame <= 0 or frame >= len(labels) - 1:
            continue
        previous = _point(index, int(current.track_id), frame - 1)
        if (previous is None or not bool(previous.physically_visible)
                or not bool(current.physically_visible)
                or int(previous.candidate_owner) <= 0
                or int(current.candidate_owner) <= 0):
            continue
        old_owner = int(previous.candidate_owner)
        new_owner = int(current.candidate_owner)
        if old_owner == new_owner:
            continue
        transition_rows.setdefault(frame, []).append({
            "track": int(current.track_id), "old_owner": old_owner,
            "new_owner": new_owner, "previous": previous,
            "current": current,
            "prior_run": _prior_run(
                index, int(current.track_id), frame, old_owner,
                minimum_purity, minimum_coverage),
        })

    audits: list[dict] = []
    for frame, transitions in sorted(transition_rows.items()):
        for left_index, left in enumerate(transitions):
            for right in transitions[left_index + 1:]:
                if not (left["old_owner"] == right["new_owner"]
                        and left["new_owner"] == right["old_owner"]):
                    continue
                pair_left, pair_right = left, right
                if int(pair_left["track"]) > int(pair_right["track"]):
                    pair_left, pair_right = pair_right, pair_left
                owner_a = int(pair_left["old_owner"])
                owner_b = int(pair_right["old_owner"])
                previous_separation = _scaled_distance(
                    pair_left["previous"], pair_right["previous"])
                transition_separation = _scaled_distance(
                    pair_left["current"], pair_right["current"])
                next_a = _point(index, int(pair_left["track"]), frame + 1)
                next_b = _point(index, int(pair_right["track"]), frame + 1)
                returned_a = bool(
                    _valid_point(next_a, minimum_purity, minimum_coverage)
                    and int(next_a.candidate_owner) == owner_a)
                returned_b = bool(
                    _valid_point(next_b, minimum_purity, minimum_coverage)
                    and int(next_b.candidate_owner) == owner_b)
                component_a = _component_at(
                    labels[frame], owner_b, pair_left["current"])
                component_b = _component_at(
                    labels[frame], owner_a, pair_right["current"])
                reasons: list[str] = []
                if not all(_valid_point(
                        point, minimum_purity, minimum_coverage)
                        for point in (pair_left["previous"],
                                      pair_left["current"],
                                      pair_right["previous"],
                                      pair_right["current"])):
                    reasons.append("low_owner_evidence")
                if min(int(pair_left["prior_run"]),
                       int(pair_right["prior_run"])) < minimum_prior:
                    reasons.append("insufficient_prior_owner_tenure")
                if max(previous_separation, transition_separation) \
                        > maximum_separation:
                    reasons.append("distant_reciprocal_transition")
                if not (returned_a or returned_b):
                    reasons.append("no_immediate_return")
                if component_a is None or component_b is None:
                    reasons.append("swappable_component_missing")
                    third = 0
                else:
                    if np.any(component_a & component_b):
                        reasons.append("swappable_components_not_distinct")
                    third = _third_pair_owner_cores(
                        attached, frame,
                        {int(pair_left["track"]), int(pair_right["track"])},
                        {owner_a, owner_b}, component_a, component_b)
                    if third:
                        reasons.append("third_pair_owner_core_in_components")
                endpoint_a, successor_a = _endpoint_link(
                    attached, index, int(pair_left["track"]), frame,
                    maximum_endpoint, minimum_purity, minimum_coverage,
                    len(labels))
                endpoint_b, successor_b = _endpoint_link(
                    attached, index, int(pair_right["track"]), frame,
                    maximum_endpoint, minimum_purity, minimum_coverage,
                    len(labels))
                if endpoint_a == "ambiguous_endpoint_successor" or \
                        endpoint_b == "ambiguous_endpoint_successor":
                    reasons.append("ambiguous_endpoint_link")
                audits.append({
                    "proposal_id": "", "frame": int(frame),
                    "track_a": int(pair_left["track"]),
                    "track_b": int(pair_right["track"]),
                    "owner_a": owner_a, "owner_b": owner_b,
                    "prior_run_a": int(pair_left["prior_run"]),
                    "prior_run_b": int(pair_right["prior_run"]),
                    "minimum_prior_run": minimum_prior,
                    "previous_separation_sum_radii": previous_separation,
                    "transition_separation_sum_radii": transition_separation,
                    "track_a_returned_immediately": returned_a,
                    "track_b_returned_immediately": returned_b,
                    "third_pair_owner_cores": int(third),
                    "endpoint_status_a": endpoint_a,
                    "endpoint_status_b": endpoint_b,
                    "endpoint_successor_a": successor_a,
                    "endpoint_successor_b": successor_b,
                    "eligible": not reasons,
                    "reason": ("eligible_isolated_one_frame_reciprocal_exchange"
                               if not reasons else "|".join(reasons)),
                    "_component_a": component_a,
                    "_component_b": component_b,
                })
    for number, row in enumerate(
            (row for row in audits if bool(row["eligible"])), 1):
        row["proposal_id"] = f"RX{number:04d}"
    public = pd.DataFrame([
        {key: row[key] for key in AUDIT_COLUMNS} for row in audits],
        columns=AUDIT_COLUMNS)
    internal = pd.DataFrame(audits)
    return public, internal


def apply(labels: np.ndarray, internal: pd.DataFrame
          ) -> tuple[np.ndarray, pd.DataFrame]:
    """Atomically swap only the transition-frame pair components."""
    candidate = labels.copy()
    applications: list[dict] = []
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for _, proposal in eligible.iterrows():
        frame = int(proposal["frame"])
        component_a = proposal["_component_a"]
        component_b = proposal["_component_b"]
        owner_a = int(proposal["owner_a"])
        owner_b = int(proposal["owner_b"])
        trial = candidate[frame].copy()
        trial[component_a] = owner_a
        trial[component_b] = owner_b
        changed = int(np.count_nonzero(trial != candidate[frame]))
        candidate[frame] = trial
        applications.append({
            "proposal_id": str(proposal["proposal_id"]), "frame": frame,
            "track_a": int(proposal["track_a"]),
            "track_b": int(proposal["track_b"]),
            "owner_a": owner_a, "owner_b": owner_b,
            "applied": bool(changed), "changed_pixels": changed,
            "reason": ("atomic_transition_frame_component_swap"
                       if changed else "already_consistent"),
        })
    return candidate, pd.DataFrame(applications, columns=[
        "proposal_id", "frame", "track_a", "track_b", "owner_a", "owner_b",
        "applied", "changed_pixels", "reason"])


def _copy_audits(upstream_dir: Path | None, params: dict, out) -> dict:
    outputs = {}
    for name in ("application_audit.csv", "conflicted_lineage_audit.csv"):
        source = upstream_dir / name if upstream_dir is not None else None
        configured = params.get(name.removesuffix(".csv") + "_path")
        if source is None or not source.is_file():
            source = Path(configured) if configured else None
        target = out.out / name
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text("\n", encoding="utf-8")
        outputs[name.removesuffix(".csv")] = target
    return outputs


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis-tuner adapter around the production-owned algorithm."""
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame()
    else:
        candidate, applications = apply(labels, internal)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reciprocal exchange changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("reciprocal exchange changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError("reciprocal exchange created duplicate components")

    output_stem = str(params.get(
        "output_stem", Path(params["labels_path"]).stem))
    labels_path = out.out / f"{output_stem}.tif"
    unclaimed_path = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    audit_path = out.out / "reciprocal_two_seat_exchange_audit.csv"
    applications_path = out.out / "reciprocal_two_seat_exchange_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(applications_path, index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "reciprocal_transitions_audited": int(len(audit)),
        "eligible_exchanges": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_exchanges": int(len(applied)),
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    outputs = {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "applications": applications_path,
        "metrics": metrics_path,
    }
    outputs.update(_copy_audits(upstream_dir, params, out))
    return {"outputs": outputs, "summary": metrics}
