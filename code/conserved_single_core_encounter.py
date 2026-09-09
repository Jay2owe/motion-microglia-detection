"""Reclassify conserved encounter sources supported by one raw-signal core.

The calibration is field-wide and target-free. Event, track, owner and frame
values are measured outputs; none may be supplied as selectors.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


FAMILY_DISRUPTION = {"single_cell_multi_core": 5.0}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "forced_identity", "forced_interval",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        key for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in key.lower() for token in FORBIDDEN_TARGET_TOKENS)
    )
    if supplied:
        raise ValueError(
            "single-core encounter calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("single-core encounter calibration must be field-wide")


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _owners(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def _tracks(rows: pd.DataFrame) -> tuple[int, ...]:
    text = "|".join(rows.source_ref.astype(str))
    return tuple(sorted({int(value) for value in re.findall(r"T(\d+)", text)}))


def _sample_pair(image: np.ndarray, left: object, right: object,
                 sigma: float) -> dict:
    smooth = ndi.gaussian_filter(image.astype(np.float32), sigma)
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    samples = max(5, 2 * int(np.ceil(distance)) + 1)
    xs = np.linspace(float(left.x), float(right.x), samples)
    ys = np.linspace(float(left.y), float(right.y), samples)
    values = ndi.map_coordinates(smooth, [ys, xs], order=1, mode="nearest")
    endpoint_low = float(min(values[0], values[-1]))
    endpoint_high = float(max(values[0], values[-1]))
    interior = values[1:-1] if len(values) > 2 else values
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return {
        "separation_sum_radii": distance / scale,
        "valley_ratio": float(np.min(interior) / max(endpoint_low, 1.0)),
        "endpoint_balance": endpoint_low / max(endpoint_high, 1.0),
        "both_strong": _truth(left.strong) and _truth(right.strong),
    }


def _source_measurements(source_rows: pd.DataFrame, indexed: pd.DataFrame,
                         raw: np.ndarray, sigma: float) -> list[dict]:
    tracks = _tracks(source_rows)
    if len(tracks) != 2:
        return []
    first = int(source_rows.source_first_frame.iloc[0])
    last = int(source_rows.source_last_frame.iloc[0])
    records: list[dict] = []
    for frame in range(first, last + 1):
        local = []
        for track in tracks:
            key = (track, frame)
            if key not in indexed.index:
                continue
            point = indexed.loc[key]
            if isinstance(point, pd.DataFrame):
                point = point.iloc[0]
            if _truth(point.physically_visible):
                local.append(point)
        for left, right in combinations(local, 2):
            records.append({"frame": frame, **_sample_pair(
                raw[frame], left, right, sigma)})
    return records


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, raw: np.ndarray, params: dict,
              enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame, dict]:
    """Return the calibrated catalogue, members, complete audit and summary."""
    assert_target_free(params)
    sigma = float(params.get("single_core_raw_sigma_px", 1.0))
    maximum_two_core_valley = float(params.get(
        "single_core_maximum_two_core_valley_ratio", 0.80))
    minimum_two_core_balance = float(params.get(
        "single_core_minimum_two_core_endpoint_balance", 0.25))
    minimum_two_core_separation = float(params.get(
        "single_core_minimum_two_core_separation_sum_radii", 1.0))
    minimum_shared_core_valley = float(params.get(
        "single_core_minimum_shared_core_valley_ratio", 0.90))
    maximum_shared_core_separation = float(params.get(
        "single_core_maximum_shared_core_separation_sum_radii", 1.5))
    minimum_shared_core_frames = int(params.get(
        "single_core_minimum_shared_core_frames", 2))

    indexed = points.set_index(["track_id", "frame"], drop=False)
    calibrated_events = events.copy()
    calibrated_members = members.copy()
    audit_rows: list[dict] = []
    calibrated_event_ids: set[str] = set()
    calibrated_source_count = 0

    for event_id, event_members in members.groupby("event_id", sort=False):
        event_id = str(event_id)
        sources = event_members.drop_duplicates("source_id")
        source_results: dict[int, dict] = {}
        event_reasons: list[str] = []
        has_merge_source = False
        for source in sources.itertuples(index=False):
            source_id = int(source.source_id)
            rows = event_members[
                event_members.source_id.astype(int).eq(source_id)]
            family = str(source.source_family)
            before = _owners(source.source_accepted_before)
            after = _owners(source.source_accepted_after)
            records = (_source_measurements(rows, indexed, raw, sigma)
                       if family == "separable_cells_merged" else [])
            separable = [
                record for record in records
                if record["both_strong"]
                and record["separation_sum_radii"] >= minimum_two_core_separation
                and record["endpoint_balance"] >= minimum_two_core_balance
                and record["valley_ratio"] <= maximum_two_core_valley]
            median_valley = float(np.median([
                record["valley_ratio"] for record in records])) \
                if records else float("nan")
            maximum_separation = float(max([
                record["separation_sum_radii"] for record in records],
                default=float("nan")))
            reasons: list[str] = []
            eligible = False
            if family == "separable_cells_merged":
                has_merge_source = True
                if before != after or len(before) != 1:
                    reasons.append("owner_not_conserved")
                if len(_tracks(rows)) != 2:
                    reasons.append("not_exactly_two_references")
                if not records:
                    reasons.append("no_joint_visible_observation")
                if len({record["frame"] for record in records}) \
                        < minimum_shared_core_frames:
                    reasons.append("insufficient_shared_core_frames")
                if separable:
                    reasons.append("positive_two_core_raw_evidence")
                if records and median_valley < minimum_shared_core_valley:
                    reasons.append("raw_valley_not_shared_core")
                if records and maximum_separation > maximum_shared_core_separation:
                    reasons.append("references_too_separated")
                eligible = not reasons
            elif family == "single_cell_multi_core":
                reasons.append("already_single_cell_multi_core")
            else:
                reasons.append("incompatible_constituent_family")
            source_results[source_id] = {
                "eligible": eligible, "family": family,
                "records": records, "separable": separable,
                "median_valley": median_valley,
                "maximum_separation": maximum_separation,
                "reasons": reasons,
            }
            audit_rows.append({
                "event_id": event_id,
                "source_id": source_id,
                "source_family": family,
                "physical_tracks": "|".join(map(str, _tracks(rows))),
                "conserved_owner_count": len(before) if before == after else 0,
                "pair_observations": len(records),
                "raw_separable_frames": len({r["frame"] for r in separable}),
                "median_valley_ratio": median_valley,
                "maximum_separation_sum_radii": maximum_separation,
                "eligible_source": eligible,
                "reason": "eligible_shared_core" if eligible else "|".join(reasons),
            })
        incompatible = [
            value for value in source_results.values()
            if value["family"] not in {
                "separable_cells_merged", "single_cell_multi_core"}]
        unqualified = [
            value for value in source_results.values()
            if value["family"] == "separable_cells_merged"
            and not value["eligible"]]
        event_eligible = bool(has_merge_source and not incompatible and not unqualified)
        if not event_eligible:
            if incompatible:
                event_reasons.append("event_has_incompatible_constituent")
            if unqualified:
                event_reasons.append("event_has_unqualified_merge_constituent")
            continue
        if not enabled:
            continue
        calibrated_event_ids.add(event_id)
        for source_id, value in source_results.items():
            if value["family"] != "separable_cells_merged":
                continue
            mask = (
                calibrated_members.event_id.astype(str).eq(event_id)
                & calibrated_members.source_id.astype(int).eq(source_id))
            duration = int(calibrated_members.loc[mask, "source_last_frame"].iloc[0]) \
                - int(calibrated_members.loc[mask, "source_first_frame"].iloc[0]) + 1
            calibrated_members.loc[mask, "source_family"] = \
                "single_cell_multi_core"
            calibrated_members.loc[mask, "source_impact"] = min(
                25.0, 10.0 + duration / 2.0)
            calibrated_source_count += 1
        updated_sources = calibrated_members[
            calibrated_members.event_id.astype(str).eq(event_id)] \
            .drop_duplicates("source_id")
        impact = float(updated_sources.source_impact.astype(float).max())
        disruption = (0.65 * FAMILY_DISRUPTION["single_cell_multi_core"]
                      + 0.35 * impact)
        event_mask = calibrated_events.event_id.astype(str).eq(event_id)
        calibrated_events.loc[event_mask, "family"] = "single_cell_multi_core"
        calibrated_events.loc[event_mask, "impact"] = impact
        calibrated_events.loc[event_mask, "disruption_score"] = disruption
        calibrated_events.loc[event_mask, "fidelity"] = 100.0 - disruption
        calibrated_events.loc[event_mask, "evidence"] = (
            calibrated_events.loc[event_mask, "evidence"].astype(str)
            + " ; conserved owner and shared raw core")

    calibrated_events = calibrated_events.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated_events["impact_rank"] = np.arange(1, len(calibrated_events) + 1)
    columns = list(events.columns)
    calibrated_events = calibrated_events[columns]
    audit = pd.DataFrame(audit_rows)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(events.event_id.nunique()),
        "sources_audited": int(members[["event_id", "source_id"]]
                               .drop_duplicates().shape[0]),
        "events_reclassified": int(len(calibrated_event_ids)),
        "sources_reclassified": int(calibrated_source_count),
        "enabled": bool(enabled),
    }
    return calibrated_events, calibrated_members, audit, summary


def _event_summary(events: pd.DataFrame) -> dict:
    owner = events[events.family.eq("identity_swap_or_takeover")]
    return {
        "events": int(len(events)),
        "owner_change_events": int(len(owner)),
        "owner_change_disruption_burden": float(owner.disruption_score.sum()),
        "high_disruption_events": int((events.disruption_score >= 50).sum()),
        "total_disruption_burden": float(events.disruption_score.sum()),
        "events_by_family": {str(key): int(value) for key, value in
                             events.groupby("family").size().items()},
    }


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis-tuner stage adapter for one complete recording."""
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    points = pd.read_csv(params["points_path"])
    raw = tifffile.imread(params["raw_path"])
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, points, raw, params,
        enabled=bool(params.get("enabled", True)))
    labels_path = Path(params["labels_path"])
    ledger_path = Path(params["unclaimed_path"])
    stem = str(params["stem"])
    outputs = {
        "candidate_events": out.out / "candidate_disruptive_events.csv",
        "candidate_event_members": out.out / "candidate_event_members.csv",
        "audit": out.out / "conserved_single_core_encounter_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["candidate_events"], index=False)
    calibrated_members.to_csv(outputs["candidate_event_members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(ledger_path, outputs["unclaimed"])
    payload = {
        **summary,
        "before": _event_summary(events),
        "after": _event_summary(calibrated),
        "label_bytes_unchanged": outputs["labels"].read_bytes()
        == labels_path.read_bytes(),
        "unclaimed_bytes_unchanged": outputs["unclaimed"].read_bytes()
        == ledger_path.read_bytes(),
    }
    outputs["metrics"].write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": payload}
