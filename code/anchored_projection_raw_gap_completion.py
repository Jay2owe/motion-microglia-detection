"""Add raw-supported cores for ownerless gaps in proved projection tracks.

The upstream relay audit supplies the field-discovered branch/anchor relation.
This stage never accepts a biological target.  It adds only nonzero-signal
pixels from an adaptive raw core, only where both accepted ledgers are empty,
and only for visible ownerless rows inside that proved projection's lifespan.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import bracketed_mixed_owner_flash as flash
import component_accounting
import ownerless_cohort_latent_gap_completion as gap_completion
from ownerless_body_recovery import evidence_image
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
AUDIT_COLUMNS = [
    "proposal_id", "relay_proposal_id", "track_id", "anchor_owner",
    "anchor_track", "first_frame", "last_frame", "gap_frames",
    "minimum_visibility", "maximum_core_offset_radii",
    "maximum_core_area_radius_ratio", "maximum_anchor_gap_target_radii",
    "changed_pixels", "explained_projection_components", "applied", "reason",
]


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def complete(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
             points: pd.DataFrame, thresholds: pd.DataFrame,
             relay_audit: pd.DataFrame, relay_applications: pd.DataFrame,
             params: dict):
    flash.assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    threshold_by_frame = thresholds.set_index("frame")
    proven_ids = set(relay_applications[
        relay_applications.applied.astype(bool)].proposal_id.astype(str))
    proven = relay_audit[
        relay_audit.eligible.astype(bool)
        & relay_audit.proposal_id.astype(str).isin(proven_ids)]
    candidate = labels.copy()
    maximum_gap = float(params.get("maximum_raw_core_anchor_gap_target_radii", 3.0))
    rows = []
    explained = 0
    for number, proposal in enumerate(proven.itertuples(index=False), start=1):
        target_group = groups.get(int(proposal.track_id), pd.DataFrame())
        anchor_group = groups.get(int(proposal.anchor_track), pd.DataFrame())
        ownerless = target_group[
            target_group.physically_visible.astype(bool)
            & target_group.candidate_owner.astype(int).eq(0)
            & target_group.frame.astype(int).gt(int(proposal.initial_last))]
        proposal_id = f"ARG{number:04d}"
        pending = []
        failure = ""
        for target in ownerless.itertuples(index=False):
            frame = int(target.frame)
            anchor = _point(anchor_group, frame)
            if (anchor is None or not bool(anchor.physically_visible)
                    or int(anchor.candidate_owner) != int(proposal.anchor_owner)):
                failure = "durable_anchor_absent"
                break
            evidence = evidence_image(raw[frame], params)
            found = gap_completion._nearest_raw_core(
                raw[frame], evidence, target,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            if found is None:
                failure = "qualified_raw_core_missing"
                break
            core, region, details = found
            occupied = ((candidate[frame][region] > 0)
                        | (unclaimed[frame][region] > 0))
            if np.any(core & occupied):
                failure = "raw_core_overlaps_existing_ledger"
                break
            anchor_mask = candidate[frame] == int(proposal.anchor_owner)
            local_full = np.zeros_like(anchor_mask)
            local_full[region] = core
            gap = float(ndi.distance_transform_edt(~anchor_mask)[local_full].min()) \
                / max(float(target.radius_px), 1.0)
            if gap > maximum_gap:
                failure = "raw_core_too_far_from_anchor_owner"
                break
            trial = candidate[frame].copy()
            before_excess = component_accounting.component_excess(
                trial, int(proposal.anchor_owner))
            trial[region][core] = int(proposal.anchor_owner)
            after_excess = component_accounting.component_excess(
                trial, int(proposal.anchor_owner))
            delta = max(0, after_excess - before_excess)
            if delta not in (0, 1):
                failure = "more_than_one_projection_component_created"
                break
            pending.append({
                "frame": frame, "trial": trial,
                "visibility": float(target.visibility),
                "core_offset": float(details["core_offset_radii"]),
                "core_area_ratio": float(details["core_area_radius_ratio"]),
                "anchor_gap": gap, "pixels": int(core.sum()),
                "explained": int(delta),
            })
        if failure or not pending:
            rows.append({
                "proposal_id": proposal_id,
                "relay_proposal_id": str(proposal.proposal_id),
                "track_id": int(proposal.track_id),
                "anchor_owner": int(proposal.anchor_owner),
                "anchor_track": int(proposal.anchor_track),
                "first_frame": -1, "last_frame": -1,
                "gap_frames": int(len(ownerless)),
                "minimum_visibility": 0.0,
                "maximum_core_offset_radii": 0.0,
                "maximum_core_area_radius_ratio": 0.0,
                "maximum_anchor_gap_target_radii": 0.0,
                "changed_pixels": 0, "explained_projection_components": 0,
                "applied": False,
                "reason": "atomic_refusal:" + (failure or "no_ownerless_gap"),
            })
            continue
        for detail in pending:
            candidate[int(detail["frame"])] = detail["trial"]
            explained += int(detail["explained"])
        rows.append({
            "proposal_id": proposal_id,
            "relay_proposal_id": str(proposal.proposal_id),
            "track_id": int(proposal.track_id),
            "anchor_owner": int(proposal.anchor_owner),
            "anchor_track": int(proposal.anchor_track),
            "first_frame": min(item["frame"] for item in pending),
            "last_frame": max(item["frame"] for item in pending),
            "gap_frames": int(len(pending)),
            "minimum_visibility": min(item["visibility"] for item in pending),
            "maximum_core_offset_radii": max(
                item["core_offset"] for item in pending),
            "maximum_core_area_radius_ratio": max(
                item["core_area_ratio"] for item in pending),
            "maximum_anchor_gap_target_radii": max(
                item["anchor_gap"] for item in pending),
            "changed_pixels": sum(item["pixels"] for item in pending),
            "explained_projection_components": sum(
                item["explained"] for item in pending),
            "applied": True,
            "reason": "adaptive_raw_core_completed_proved_projection_gap",
        })
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    changed = labels != candidate
    additions = changed & (labels == 0) & (candidate > 0)
    actual_duplicates = component_accounting.new_duplicate_components(
        labels, candidate)
    details = {
        "proved_relays_consumed": int(len(proven)),
        "applied_proposals": int(audit.applied.astype(bool).sum())
            if len(audit) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_added_pixels": int(additions.sum()),
        "preexisting_assigned_changed_pixels": int(np.count_nonzero(
            changed & (labels > 0))),
        "unclaimed_overlap_pixels": int(np.count_nonzero(
            additions & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_explained_projection_components": int(explained),
        "new_duplicate_components": max(
            0, int(actual_duplicates) - int(explained)),
    }
    return candidate, audit, details


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("anchored raw-gap completion requires upstream labels")
    flash.assert_target_free(params)
    labels_path = Path(upstream_dir) / str(params["labels_name"])
    unclaimed_path = Path(upstream_dir) / str(params["unclaimed_name"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    raw = raw[int(params.get("raw_frame_offset", 0)):]
    raw = raw[:len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    relay_audit = pd.read_csv(params["relay_audit_path"])
    relay_applications = pd.read_csv(params["relay_applications_path"])
    candidate, audit, details = complete(
        labels, unclaimed, raw, points, thresholds,
        relay_audit, relay_applications, params)
    if details["preexisting_assigned_changed_pixels"]:
        raise AssertionError("anchored raw-gap completion changed assigned pixels")
    if details["unclaimed_overlap_pixels"] or details["zero_signal_additions"]:
        raise AssertionError("anchored raw-gap completion violated input ledgers")
    if details["new_duplicate_components"]:
        raise AssertionError("unexplained duplicate component was created")

    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
        "audit": out.out / "anchored_projection_raw_gap_completion_audit.csv",
        "metrics": out.out / "producer_metrics.json",
    }
    tifffile.imwrite(outputs["labels"], candidate, compression="zlib")
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    audit.to_csv(outputs["audit"], index=False)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        **details,
        "preexisting_unclaimed_exact": (
            outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes()),
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

