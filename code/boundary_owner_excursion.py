"""Recover left-censored A-B-A owner excursions without named targets."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import bounded_owner_excursion as bounded
import component_accounting
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region_target", "review_case",
    "case_id", "forced_identity", "forced_track", "forced_frame",
    "include_track", "exclude_track", "include_identity",
    "exclude_identity",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower() for token in FORBIDDEN))
    if supplied:
        raise ValueError(
            "boundary owner excursion received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("boundary owner excursion must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return distance / scale


def _raw_pair(raw: np.ndarray, left, right, sigma: float) -> dict:
    smooth = ndi.gaussian_filter(raw.astype(np.float32), sigma)
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    samples = max(5, 2 * int(math.ceil(distance)) + 1)
    xs = np.linspace(float(left.x), float(right.x), samples)
    ys = np.linspace(float(left.y), float(right.y), samples)
    values = ndi.map_coordinates(smooth, [ys, xs], order=1, mode="nearest")
    low, high = float(min(values[0], values[-1])), \
        float(max(values[0], values[-1]))
    return {
        "distance_sum_radii": _scaled_distance(left, right),
        "valley_ratio": float(np.min(values[1:-1]) / max(low, 1.0)),
        "endpoint_balance": low / max(high, 1.0),
        "both_strong": _truth(left.strong) and _truth(right.strong),
    }


def _event_local_companions(
        attached: pd.DataFrame, raw: np.ndarray, resident_group: pd.DataFrame,
        owner_b: int, middle_start: int, middle_end: int, params: dict,
        ) -> list[dict]:
    resident_index = resident_group.set_index("frame", drop=False)
    maximum_distance = float(params.get(
        "maximum_companion_distance_sum_radii", 2.50))
    maximum_valley = float(params.get("maximum_two_core_valley_ratio", 0.80))
    minimum_balance = float(params.get("minimum_two_core_endpoint_balance", 0.25))
    minimum_raw = int(params.get("minimum_two_core_frames", 2))
    sigma = float(params.get("pair_profile_sigma_px", 1.0))
    movie = int(len(raw))
    onset_slack = max(1, int(math.ceil(movie * float(
        params.get("maximum_companion_onset_slack_movie_fraction", 0.02)))))
    end_slack = max(1, int(math.ceil(movie * float(
        params.get("maximum_companion_end_slack_movie_fraction", 0.02)))))
    candidates: list[dict] = []
    for track, group in attached[
            attached.physically_visible.astype(bool)
            & attached.candidate_owner.astype(int).eq(int(owner_b))
            & ~attached.track_id.astype(int).isin(
                resident_group.track_id.astype(int).unique())] \
            .groupby("track_id", sort=True):
        visible_all = attached[
            attached.track_id.astype(int).eq(int(track))
            & attached.physically_visible.astype(bool)]
        first, last = (int(visible_all.frame.min()),
                       int(visible_all.frame.max()))
        if not (middle_start - onset_slack <= first <= middle_start
                and middle_start <= last <= middle_end + end_slack):
            continue
        overlap = group[group.frame.astype(int).between(
            middle_start, middle_end)]
        measurements: list[dict] = []
        for point in overlap.itertuples(index=False):
            frame = int(point.frame)
            if frame not in resident_index.index:
                continue
            resident = resident_index.loc[frame]
            if isinstance(resident, pd.DataFrame):
                resident = resident.iloc[0]
            if not _truth(resident.physically_visible):
                continue
            measurements.append(_raw_pair(
                raw[frame], resident, point, sigma))
        qualified = [item for item in measurements
                     if item["distance_sum_radii"] <= maximum_distance
                     and item["valley_ratio"] <= maximum_valley
                     and item["endpoint_balance"] >= minimum_balance
                     and item["both_strong"]]
        if len(qualified) < minimum_raw:
            continue
        candidates.append({
            "track": int(track), "first": first, "last": last,
            "owner_support": int(len(group)),
            "overlap_frames": int(len(measurements)),
            "qualified_two_core_frames": int(len(qualified)),
            "maximum_distance_sum_radii": float(max(
                item["distance_sum_radii"] for item in qualified)),
            "maximum_valley_ratio": float(max(
                item["valley_ratio"] for item in qualified)),
            "minimum_endpoint_balance": float(min(
                item["endpoint_balance"] for item in qualified)),
        })
    return candidates


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit all left-censored A-B-A triples in the complete field."""
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    movie = len(labels)
    boundary_slack = max(0, int(math.ceil(movie * float(
        params.get("maximum_boundary_start_movie_fraction", 0.01)))))
    maximum_middle = max(1, int(math.ceil(movie * float(
        params.get("maximum_excursion_movie_fraction", 0.06)))))
    minimum_return = max(2, int(math.ceil(movie * float(
        params.get("minimum_return_anchor_movie_fraction", 0.08)))))
    maximum_step = float(params.get("maximum_step_sum_radii", 0.60))
    rows: list[dict] = []
    internals: list[dict] = []
    for track, group in attached.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        visible = group[group.physically_visible.astype(bool)]
        if visible.empty or int(visible.frame.min()) > boundary_slack:
            continue
        runs = bounded._positive_runs(group)
        if len(runs) < 3:
            continue
        for position in range(1, len(runs) - 1):
            before, middle, after = runs[position - 1:position + 2]
            if (int(before["start"]) > boundary_slack
                    or int(before["owner"]) != int(after["owner"])
                    or int(before["owner"]) == int(middle["owner"])):
                continue
            owner_a, owner_b = int(before["owner"]), int(middle["owner"])
            middle_frames = list(map(int, middle["frames"]))
            reasons: list[str] = []
            if len(before["frames"]) < 1:
                reasons.append("boundary_anchor_missing")
            if len(middle_frames) > maximum_middle:
                reasons.append("excursion_too_long")
            if len(after["frames"]) < minimum_return:
                reasons.append("return_anchor_too_short")
            span = group[group.frame.astype(int).between(
                int(before["start"]), int(after["end"]))]
            span_visible = span[span.physically_visible.astype(bool)]
            expected = list(range(int(before["start"]), int(after["end"]) + 1))
            if span_visible.frame.astype(int).tolist() != expected:
                reasons.append("physical_lineage_not_continuous")
            steps = [bounded._scaled_step(left, right)
                     for left, right in zip(
                         span_visible.itertuples(index=False),
                         span_visible.iloc[1:].itertuples(index=False))]
            maximum_seen_step = max(steps, default=0.0)
            if maximum_seen_step > maximum_step:
                reasons.append("physical_lineage_motion_too_large")
            companions = _event_local_companions(
                attached, raw, group, owner_b, int(middle["start"]),
                int(middle["end"]), params)
            if len(companions) != 1:
                reasons.append("event_local_companion_not_unique")
            companion = companions[0] if len(companions) == 1 else None
            public = {
                "proposal_id": "", "track_id": int(track),
                "owner_a": owner_a, "owner_b": owner_b,
                "boundary_start": int(before["start"]),
                "boundary_anchor_frames": int(len(before["frames"])),
                "excursion_start": int(middle["start"]),
                "excursion_end": int(middle["end"]),
                "excursion_frames": int(len(middle_frames)),
                "return_start": int(after["start"]),
                "return_anchor_frames": int(len(after["frames"])),
                "maximum_step_sum_radii": float(maximum_seen_step),
                "companion_track": int(companion["track"]) if companion else 0,
                "companion_first_frame": int(companion["first"])
                    if companion else -1,
                "companion_last_frame": int(companion["last"])
                    if companion else -1,
                "qualified_two_core_frames": int(
                    companion["qualified_two_core_frames"])
                    if companion else 0,
                "maximum_two_core_distance_sum_radii": float(
                    companion["maximum_distance_sum_radii"])
                    if companion else float("nan"),
                "maximum_two_core_valley_ratio": float(
                    companion["maximum_valley_ratio"])
                    if companion else float("nan"),
                "minimum_two_core_endpoint_balance": float(
                    companion["minimum_endpoint_balance"])
                    if companion else float("nan"),
                "eligible": not reasons,
                "reason": "eligible_boundary_anchored_owner_excursion"
                if not reasons else "|".join(reasons),
            }
            rows.append(public)
            internals.append({**public, "frame_values": middle_frames})
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"BOE{number:04d}"
    return pd.DataFrame(rows), pd.DataFrame(internals), attached


def _excess(frame: np.ndarray, owner: int) -> int:
    return component_accounting.component_excess(frame, int(owner))


def apply(labels: np.ndarray, raw: np.ndarray, attached: pd.DataFrame,
          internal: pd.DataFrame, params: dict
          ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    candidate = labels.copy()
    index = attached.sort_values(["frame", "track_id"]).set_index(
        ["track_id", "frame"], drop=False)
    applications: list[dict] = []
    applied = explained = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending: list[tuple[int, np.ndarray, str, int, int]] = []
        failure = ""
        for frame in list(proposal.frame_values):
            trial, method, changed = bounded._prepare_frame(
                candidate[int(frame)], raw[int(frame)], index, proposal,
                int(frame), params)
            if trial is None:
                failure = method
                break
            owner_a_delta = (_excess(trial, int(proposal.owner_a))
                             - _excess(candidate[int(frame)],
                                       int(proposal.owner_a)))
            owner_b_delta = (_excess(trial, int(proposal.owner_b))
                             - _excess(candidate[int(frame)],
                                       int(proposal.owner_b)))
            if owner_a_delta > 1:
                failure = "multiple_projection_components_created"
                break
            if owner_b_delta > 0:
                failure = "foreign_owner_fragmented"
                break
            pending.append((int(frame), trial, method, int(changed),
                            max(0, int(owner_a_delta))))
        if failure:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "frame": int(proposal.excursion_start),
                "partition_method": "atomic", "changed_pixels": 0,
                "explained_projection_components": 0,
                "applied": False, "reason": "atomic_refusal:" + failure})
            continue
        for frame, trial, method, changed, projection in pending:
            candidate[frame] = trial
            explained += projection
            applications.append({
                "proposal_id": proposal.proposal_id, "frame": frame,
                "partition_method": method, "changed_pixels": changed,
                "explained_projection_components": projection,
                "applied": True,
                "reason": "atomic_boundary_owner_excursion_recovery"})
        applied += 1
    changed = candidate != labels
    details = {
        "run_triples_audited": int(len(internal)),
        "eligible_excursions": int(internal.eligible.astype(bool).sum())
            if len(internal) else 0,
        "applied_excursions": int(applied),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0,
        "new_duplicate_components": 0,
        "new_explained_projection_components": int(explained),
    }
    return candidate, pd.DataFrame(applications), details


def _save(source: Path, target: Path, before: np.ndarray,
          after: np.ndarray) -> None:
    if np.array_equal(before, after):
        shutil.copyfile(source, target)
    else:
        tifffile.imwrite(target, after, imagej=True, compression="zlib",
                         photometric="minisblack", metadata={
                             "axes": "TYX", "finterval": 1800.0,
                             "tunit": "sec", "unit": "pixel"})


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal, attached = discover(labels, raw, points, params)
    candidate, applications, details = apply(
        labels, raw, attached, internal, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("boundary owner excursion changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("boundary owner excursion changed identity set")

    stem = str(params.get("output_stem", labels_path.stem))
    labels_out = out.out / f"{stem}.tif"
    unclaimed_out = out.out / f"{stem}_unclaimed_original_ids.tif"
    _save(labels_path, labels_out, labels, candidate)
    shutil.copyfile(unclaimed_path, unclaimed_out)
    audit_path = out.out / "boundary_owner_excursion_audit.csv"
    application_path = out.out / "boundary_owner_excursion_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    outputs = {"labels": labels_out, "unclaimed": unclaimed_out,
               "audit": audit_path, "applications": application_path}
    for name in ("application_audit.csv", "conflicted_lineage_audit.csv"):
        source = upstream_dir / name if upstream_dir is not None else None
        configured = params.get(name.removesuffix(".csv") + "_path")
        if source is None or not source.is_file():
            source = Path(configured) if configured else None
        target = out.out / name
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text(
                "proposal_id,physical_track,assigned_identity,outcome,reason,"
                "changed_pixels,changed_frames\n", encoding="utf-8")
        outputs[name.removesuffix(".csv")] = target
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        **details, "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    outputs["metrics"] = metrics_path
    return {"outputs": outputs, "summary": summary}

