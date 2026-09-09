"""Calibrate false owner transactions created by resolved onset splits.

The field-wide rule removes a transaction only when a physical reference is
first visible for a movie-relative short donor-owned run inside a donor
component that contains another donor reference, then continues durably under
a different owner while the donor also continues. Biological masks and event
catalogues are never changed.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
ALGORITHM_KEYS = {
    "targeting_mode", "maximum_leading_run_movie_fraction",
    "minimum_successor_run_movie_fraction",
    "minimum_donor_presence_movie_fraction",
    "minimum_successor_strong_fraction",
    "maximum_transition_step_sum_radii",
}
RUNTIME_KEYS = {
    "transactions_path", "labels_path", "points_path", "output_stem",
    "enabled",
}
AUDIT_COLUMNS = [
    "transaction_index", "family", "physical_track", "first_frame",
    "last_frame", "donor_owner", "successor_owner", "leading_frames",
    "successor_frames", "successor_strong_fraction",
    "transition_step_sum_radii", "donor_presence_frames",
    "shared_donor_component_references", "donor_continues_after_split",
    "calibrated", "reason",
]


def assert_target_free(params: dict[str, Any], *, runtime: bool = False) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "shared-donor onset-split calibration must be field-wide")
    allowed = ALGORITHM_KEYS | (RUNTIME_KEYS if runtime else set())
    unexpected = sorted(str(key) for key in params if key not in allowed)
    if unexpected:
        raise ValueError(
            "unsupported parameters (selectors forbidden): "
            + ", ".join(unexpected))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _fraction(params: dict[str, Any], key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _integer_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return [int(item) for item in parsed] \
        if isinstance(parsed, (list, tuple)) else []


def _runs(group: pd.DataFrame) -> list[dict[str, Any]]:
    visible = group[
        group.physically_visible.map(_truth)
        & group.candidate_owner.astype(int).gt(0)
    ].sort_values("frame")
    runs: list[dict[str, Any]] = []
    for point in visible.itertuples(index=False):
        owner, frame = int(point.candidate_owner), int(point.frame)
        if (runs and runs[-1]["owner"] == owner
                and runs[-1]["last"] + 1 == frame):
            runs[-1]["last"] = frame
            runs[-1]["points"].append(point)
        else:
            runs.append({"owner": owner, "first": frame, "last": frame,
                         "points": [point]})
    return runs


def _point_inside(mask: np.ndarray, point: object) -> bool:
    y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _component_at(frame: np.ndarray, owner: int,
                  point: object) -> np.ndarray | None:
    components, _ = ndi.label(frame == int(owner), STRUCTURE)
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    component = int(components[y, x])
    return components == component if component else None


def _step(left: object, right: object) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    return distance / max(float(left.radius_px) + float(right.radius_px), 1.0)


def _summary(transactions: pd.DataFrame) -> dict[str, float | int]:
    return {
        "owner_transactions": int(len(transactions)),
        "owner_transaction_burden": (
            float(transactions.impact.astype(float).sum())
            if len(transactions) else 0.0),
    }


def calibrate(transactions: pd.DataFrame, labels: np.ndarray,
              points: pd.DataFrame, params: dict[str, Any]
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return filtered owner transactions and a decision for every input."""
    assert_target_free(params)
    movie = int(len(labels))
    maximum_leading = max(1, math.ceil(movie * _fraction(
        params, "maximum_leading_run_movie_fraction", 0.01)))
    minimum_successor = max(2, math.ceil(movie * _fraction(
        params, "minimum_successor_run_movie_fraction", 0.08)))
    minimum_donor = max(2, math.ceil(movie * _fraction(
        params, "minimum_donor_presence_movie_fraction", 0.50)))
    minimum_strong = _fraction(
        params, "minimum_successor_strong_fraction", 0.75)
    maximum_step = float(params.get(
        "maximum_transition_step_sum_radii", 1.0))
    attached = points.copy()
    attached["frame"] = attached.frame.astype(int)
    attached["track_id"] = attached.track_id.astype(int)
    attached["candidate_owner"] = attached.candidate_owner.astype(int)
    owner_presence = {
        int(owner): int(np.count_nonzero(np.any(labels == int(owner), axis=(1, 2))))
        for owner in np.unique(labels) if int(owner) > 0
    }
    calibrated_indices: set[int] = set()
    audit_rows: list[dict[str, Any]] = []

    for index, transaction in transactions.reset_index(drop=True).iterrows():
        donor = int(transaction.get("accepted_before", 0))
        successor = int(transaction.get("accepted_after", 0))
        first = int(transaction.get("first_frame", -1))
        last = int(transaction.get("last_frame", -1))
        tracks = _integer_list(transaction.get("physical_tracks", []))
        reasons: list[str] = []
        leading_frames = successor_frames = shared = 0
        successor_strong = 0.0
        transition_step = float("inf")
        donor_continues = False
        track = tracks[0] if len(tracks) == 1 else 0

        if str(transaction.get("family", "")) != "identity_swap_or_takeover":
            reasons.append("not_owner_takeover_family")
        if len(tracks) != 1:
            reasons.append("transaction_not_single_physical_track")
        group = attached[attached.track_id.eq(track)].sort_values("frame")
        runs = _runs(group) if track else []
        pair = None
        for left, right in zip(runs, runs[1:]):
            if (int(left["owner"]) == donor
                    and int(right["owner"]) == successor
                    and int(left["last"]) == first
                    and int(right["first"]) == last):
                pair = left, right
                break
        if pair is None:
            reasons.append("matching_consecutive_owner_runs_missing")
        else:
            leading, following = pair
            leading_frames = len(leading["points"])
            successor_frames = len(following["points"])
            successor_strong = float(np.mean([
                _truth(point.strong) for point in following["points"]]))
            transition_step = _step(
                leading["points"][-1], following["points"][0])
            visible = group[group.physically_visible.map(_truth)]
            if len(visible) == 0 or int(leading["first"]) != int(
                    visible.frame.min()):
                reasons.append("donor_run_not_at_first_visible_frame")
            if leading_frames > maximum_leading:
                reasons.append("leading_owner_run_too_long")
            if successor_frames < minimum_successor:
                reasons.append("successor_owner_run_too_short")
            if successor_strong < minimum_strong:
                reasons.append("successor_owner_run_too_dim")
            if transition_step > maximum_step:
                reasons.append("transition_motion_not_scale_local")
            target = leading["points"][-1]
            component = _component_at(labels[first], donor, target)
            if component is None:
                reasons.append("donor_component_missing")
            else:
                frame_points = attached[
                    attached.frame.eq(first)
                    & attached.physically_visible.map(_truth)
                    & attached.candidate_owner.eq(donor)]
                shared = int(sum(
                    int(point.track_id) != track
                    and _point_inside(component, point)
                    for point in frame_points.itertuples(index=False)))
                if shared < 1:
                    reasons.append("donor_component_lacks_companion_reference")
        if owner_presence.get(donor, 0) < minimum_donor:
            reasons.append("donor_owner_not_movie_durable")
        donor_continues = bool(len(attached[
            attached.frame.eq(last)
            & attached.physically_visible.map(_truth)
            & attached.candidate_owner.eq(donor)
            & attached.track_id.ne(track)]))
        if not donor_continues:
            reasons.append("donor_does_not_continue_after_split")

        calibrated = not reasons
        if calibrated:
            calibrated_indices.add(index)
        audit_rows.append({
            "transaction_index": index,
            "family": str(transaction.get("family", "")),
            "physical_track": track, "first_frame": first,
            "last_frame": last, "donor_owner": donor,
            "successor_owner": successor, "leading_frames": leading_frames,
            "successor_frames": successor_frames,
            "successor_strong_fraction": successor_strong,
            "transition_step_sum_radii": transition_step,
            "donor_presence_frames": owner_presence.get(donor, 0),
            "shared_donor_component_references": shared,
            "donor_continues_after_split": donor_continues,
            "calibrated": calibrated,
            "reason": ("calibrated_resolved_onset_split" if calibrated
                       else "|".join(sorted(set(reasons)))),
        })

    filtered = transactions.reset_index(drop=True).drop(
        index=sorted(calibrated_indices)).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    before, after = _summary(transactions), _summary(filtered)
    summary: dict[str, Any] = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "transactions_audited": int(len(transactions)),
        "resolved_onset_splits_calibrated": int(len(calibrated_indices)),
        "before": before, "after": after,
        "owner_transaction_count_delta": int(
            after["owner_transactions"] - before["owner_transactions"]),
        "owner_transaction_burden_delta": float(
            after["owner_transaction_burden"]
            - before["owner_transaction_burden"]),
        "labels_exact": True,
    }
    return filtered, audit, summary


def run(upstream_dir: Path | None, params: dict[str, Any], out) -> dict[str, Any]:
    """Analysis-tuner adapter for the accepted score-only calibration."""
    assert_target_free(params, runtime=True)
    algorithm = {key: value for key, value in params.items()
                 if key in ALGORITHM_KEYS}
    transactions = pd.read_csv(params["transactions_path"])
    labels = tifffile.imread(params["labels_path"])
    points = pd.read_csv(params["points_path"])
    if bool(params.get("enabled", True)):
        filtered, audit, summary = calibrate(
            transactions, labels, points, algorithm)
    else:
        filtered = transactions.copy()
        audit = pd.DataFrame(columns=AUDIT_COLUMNS)
        summary = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "review_cases")},
            "transactions_audited": 0,
            "resolved_onset_splits_calibrated": 0,
            "before": _summary(transactions), "after": _summary(transactions),
            "owner_transaction_count_delta": 0,
            "owner_transaction_burden_delta": 0.0, "labels_exact": True,
        }
    transactions_path = out.out / "candidate_owner_transactions.csv"
    audit_path = out.out / "shared_donor_onset_split_audit.csv"
    metrics_path = out.out / "metrics.json"
    filtered.to_csv(transactions_path, index=False)
    audit.to_csv(audit_path, index=False)
    metrics_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {"transactions": transactions_path,
                        "audit": audit_path, "metrics": metrics_path},
            "summary": summary}
