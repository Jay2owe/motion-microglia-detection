"""Calibrate repeated same-pair shared-core merge evidence field-wide.

The rule accepts no identity, owner, track, frame, event, coordinate, region
or review-case selector.  It changes only the disruption catalogue.
"""
from __future__ import annotations

import re
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import conserved_single_core_encounter as shared_core


FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate_target", "event_target", "region_target",
    "review_case_target", "forced_identity", "forced_interval",
    "include_track", "exclude_track",
)
STRUCTURE = np.ones((3, 3), np.uint8)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "aggregate shared-core calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("aggregate shared-core calibration must be field-wide")


def _owners(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def event_summary(events: pd.DataFrame) -> dict:
    owner = events[events.family.eq("identity_swap_or_takeover")]
    return {
        "events": int(len(events)),
        "owner_change_events": int(len(owner)),
        "owner_change_disruption_burden": float(
            owner.disruption_score.astype(float).sum()),
        "high_disruption_events": int(
            (events.disruption_score.astype(float) >= 50).sum()),
        "total_disruption_burden": float(
            events.disruption_score.astype(float).sum()),
        "events_by_family": {str(key): int(value) for key, value in
                             events.groupby("family").size().items()},
    }


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, raw: np.ndarray, labels: np.ndarray,
              params: dict,
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, dict]:
    """Reclassify only unanimous aggregate shared-core merge events."""
    assert_target_free(params)
    if "physically_visible" not in points.columns:
        raise ValueError(
            "aggregate shared-core calibration requires the scorer's "
            "physically_visible point field")
    _, _, source_audit, _ = shared_core.calibrate(
        events, members, points, raw, params, enabled=False)
    frame_count = int(len(labels))
    maximum_duration = int(np.ceil(frame_count * float(
        params.get("maximum_event_movie_fraction", 0.10))))
    minimum_sources = int(params.get("minimum_merge_sources", 3))
    minimum_observations = int(params.get("minimum_pair_observations", 3))
    minimum_eligible_fraction = float(params.get(
        "minimum_eligible_source_fraction", 0.25))
    minimum_owner_movie = float(params.get(
        "minimum_owner_movie_fraction", 0.90))
    minimum_component = float(params.get(
        "minimum_largest_component_fraction", 0.90))
    owner_presence = {
        int(owner): np.any(labels == int(owner), axis=(1, 2))
        for owner in set(map(int, np.unique(labels))) - {0}}
    calibrated_events = events.copy()
    calibrated_members = members.copy()
    event_audit: list[dict] = []
    reclassified: set[str] = set()
    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        sources = event_members.drop_duplicates("source_id")
        audit = source_audit[source_audit.event_id.astype(str).eq(event_id)]
        before, after = (_owners(event.accepted_before),
                         _owners(event.accepted_after))
        owner = next(iter(before)) if len(before) == 1 and before == after else 0
        first, last = int(event.first_frame), int(event.last_frame)
        duration = last - first + 1
        source_families = set(sources.source_family.astype(str))
        pairs = set(audit.physical_tracks.astype(str)) if len(audit) else set()
        eligible_fraction = float(audit.eligible_source.astype(bool).mean()) \
            if len(audit) else 0.0
        pair_observations = int(audit.pair_observations.astype(int).sum()) \
            if len(audit) else 0
        raw_separable_frames = int(
            audit.raw_separable_frames.astype(int).sum()) if len(audit) else 0
        owner_movie_fraction = float(owner_presence.get(
            owner, np.zeros(frame_count, bool)).mean()) if owner else 0.0
        component_fractions: list[float] = []
        if owner:
            for frame in range(first, last + 1):
                mask = labels[frame] == owner
                components, count = ndi.label(mask, STRUCTURE)
                sizes = np.bincount(components.ravel())
                component_fractions.append(
                    float(sizes[1:].max() / max(int(mask.sum()), 1))
                    if int(count) else 0.0)
        minimum_component_fraction = float(min(component_fractions)) \
            if component_fractions else 0.0
        reasons: list[str] = []
        if str(event.family) != "separable_cells_merged":
            reasons.append("not_merge_family")
        if duration > maximum_duration:
            reasons.append("event_too_long")
        if len(sources) < minimum_sources:
            reasons.append("too_few_merge_sources")
        if source_families != {"separable_cells_merged"}:
            reasons.append("sources_not_all_merge_alerts")
        if len(pairs) != 1 or any(len(re.findall(r"\d+", pair)) != 2
                                 for pair in pairs):
            reasons.append("sources_do_not_share_one_track_pair")
        if not owner:
            reasons.append("accepted_owner_not_conserved")
        if pair_observations < minimum_observations:
            reasons.append("too_few_joint_track_observations")
        if eligible_fraction < minimum_eligible_fraction:
            reasons.append("no_independent_shared_core_source")
        if raw_separable_frames > 0:
            reasons.append("positive_two_core_raw_evidence")
        if owner_movie_fraction < minimum_owner_movie:
            reasons.append("owner_not_movie_long")
        if minimum_component_fraction < minimum_component:
            reasons.append("owner_not_one_dominant_component")
        eligible = not reasons
        if eligible:
            reclassified.add(event_id)
            mask = calibrated_members.event_id.astype(str).eq(event_id)
            merge = mask & calibrated_members.source_family.astype(str).eq(
                "separable_cells_merged")
            calibrated_members.loc[merge, "source_family"] = \
                "single_cell_multi_core"
            durations = (calibrated_members.loc[merge, "source_last_frame"]
                         .astype(int).to_numpy()
                         - calibrated_members.loc[merge, "source_first_frame"]
                         .astype(int).to_numpy() + 1)
            calibrated_members.loc[merge, "source_impact"] = np.minimum(
                25.0, 10.0 + durations / 2.0)
            updated = calibrated_members[mask].drop_duplicates("source_id")
            impact = float(updated.source_impact.astype(float).max())
            disruption = 0.65 * 5.0 + 0.35 * impact
            event_mask = calibrated_events.event_id.astype(str).eq(event_id)
            calibrated_events.loc[event_mask, "family"] = \
                "single_cell_multi_core"
            calibrated_events.loc[event_mask, "impact"] = impact
            calibrated_events.loc[event_mask, "disruption_score"] = disruption
            calibrated_events.loc[event_mask, "fidelity"] = 100.0 - disruption
            calibrated_events.loc[event_mask, "evidence"] = (
                calibrated_events.loc[event_mask, "evidence"].astype(str)
                + " ; aggregate same-pair shared raw core")
        event_audit.append({
            "event_id": event_id, "family": str(event.family),
            "duration_frames": duration, "merge_sources": int(len(sources)),
            "shared_track_pairs": "|".join(sorted(pairs)),
            "measured_owner": owner,
            "pair_observations": pair_observations,
            "eligible_source_fraction": eligible_fraction,
            "raw_separable_frames": raw_separable_frames,
            "owner_movie_fraction": owner_movie_fraction,
            "minimum_largest_component_fraction": minimum_component_fraction,
            "eligible": eligible,
            "reason": ("aggregate_same_pair_shared_core" if eligible
                       else "|".join(reasons)),
        })
    calibrated_events = calibrated_events.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated_events["impact_rank"] = np.arange(1, len(calibrated_events) + 1)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(len(events)),
        "events_reclassified": int(len(reclassified)),
        "reclassified_event_ids": sorted(reclassified),
        "before": event_summary(events),
        "after": event_summary(calibrated_events),
    }
    return (calibrated_events, calibrated_members,
            source_audit, pd.DataFrame(event_audit), summary)


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis adapter using only explicit current-run evidence paths."""
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    points = pd.read_csv(params["points_path"])
    raw = tifffile.imread(params["raw_path"])
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    calibrated, calibrated_members, source_audit, event_audit, summary = \
        calibrate(events, members, points, raw, labels, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "source_audit": out.out / "aggregate_shared_core_source_audit.csv",
        "event_audit": out.out / "aggregate_shared_core_event_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["events"], index=False)
    calibrated_members.to_csv(outputs["members"], index=False)
    source_audit.to_csv(outputs["source_audit"], index=False)
    event_audit.to_csv(outputs["event_audit"], index=False)
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    summary["label_bytes_unchanged"] = (
        outputs["labels"].read_bytes() == labels_path.read_bytes())
    summary["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
