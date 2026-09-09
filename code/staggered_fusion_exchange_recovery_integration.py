"""Production adapter for staggered reciprocal owner-exchange recovery."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import staggered_fusion_exchange_recovery as recovery


CONFIG_KEY = "field_wide_staggered_fusion_exchange_recovery"
STAGE_NAME = "117_staggered_fusion_exchange_recovery"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "targeting_mode",
    "reviewed_candidate_labels_sha256",
    "reviewed_candidate_unclaimed_sha256", "parameters",
}
PROJECTION_COLUMNS = [
    "analysis_frame", "source_imagej_frame", "owner",
    "components_before", "components_after", "increase",
    "duplicate_increase",
    "changed_component_area_px", "changed_centroid_x",
    "changed_centroid_y", "proof_track", "proof_track_state",
    "point_distance_radii", "proof_track_visible_frames",
    "minimum_lineage_support_frames", "raw_positive_fraction",
    "preexisting_component_retained", "classification",
    "passes_projection_proof",
]


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "staggered fusion exchange production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "staggered fusion exchange recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "mode": "candidate",
        "targeting_mode": "field_wide_discovery",
        # The standalone core validates its file contract; production supplies
        # the corresponding current-run arrays and tables in memory.
        "labels_path": "current-run labels array",
        "unclaimed_path": "current-run unclaimed array",
        "raw_path": "current-run raw array",
        "physical_track_points_path": "current-run physical points table",
        "thresholds_path": "current-run evidence thresholds table",
    })
    recovery.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def _components(plane: np.ndarray, owner: int) -> tuple[np.ndarray, int]:
    return ndi.label(plane == int(owner), recovery.STRUCTURE)


def _projection_proof(
        before: np.ndarray, after: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, frames: pd.DataFrame, params: dict[str, Any],
        ) -> pd.DataFrame:
    """Prove every new component from field-wide physical-track evidence."""
    minimum_support = max(3, int(math.ceil(
        len(before) * float(params.get(
            "minimum_post_exchange_follower_support_fraction", 0.08)))))
    maximum_distance = float(params.get(
        "maximum_post_exchange_follower_distance_sum_radii", 3.0))
    visible_support = points[points.physically_visible.astype(bool)] \
        .groupby("track_id").size().to_dict()
    rows: list[dict[str, Any]] = []
    for frame in range(len(before)):
        owners = sorted(set(map(int, np.unique(before[frame])))
                        | set(map(int, np.unique(after[frame]))))
        for owner in owners:
            if owner <= 0:
                continue
            _, count_before = _components(before[frame], owner)
            components_after, count_after = _components(after[frame], owner)
            duplicate_increase = (
                max(int(count_after) - 1, 0)
                - max(int(count_before) - 1, 0))
            if duplicate_increase <= 0:
                continue
            changed_to_owner = ((after[frame] == owner)
                                & (before[frame] != owner))
            for component_id in range(1, int(count_after) + 1):
                component = components_after == component_id
                changed = component & changed_to_owner
                if not np.any(changed):
                    continue
                yy, xx = np.nonzero(changed)
                centroid_x, centroid_y = float(xx.mean()), float(yy.mean())
                active = frames[
                    (frames.frame.astype(int) == frame)
                    & frames.expected_owner.astype(str).str.split("|").map(
                        lambda values: str(owner) in values)]
                candidates = []
                for item in active.itertuples(index=False):
                    for token in str(item.track_id).split("|"):
                        if not token:
                            continue
                        track = int(token)
                        match = points[
                            (points.track_id.astype(int) == track)
                            & (points.frame.astype(int) == frame)]
                        if match.empty:
                            continue
                        point = match.iloc[0]
                        distance = float(np.hypot(
                            centroid_x - float(point.x),
                            centroid_y - float(point.y)))
                        scaled = distance / max(float(point.radius_px), 1.0)
                        candidates.append((scaled, track, point))
                if candidates:
                    scaled, track, point = min(
                        candidates, key=lambda item: item[0])
                    support = int(visible_support.get(track, 0))
                    state = str(point.state)
                else:
                    scaled, track, support, state = (
                        float("inf"), 0, 0, "no_physical_path")
                raw_positive = float(np.mean(raw[frame][changed] > 0))
                passed = bool(
                    scaled <= maximum_distance
                    and support >= minimum_support
                    and raw_positive == 1.0)
                rows.append({
                    "analysis_frame": frame,
                    "source_imagej_frame": frame + 2,
                    "owner": owner,
                    "components_before": int(count_before),
                    "components_after": int(count_after),
                    "increase": int(count_after - count_before),
                    "duplicate_increase": int(duplicate_increase),
                    "changed_component_area_px": int(changed.sum()),
                    "changed_centroid_x": centroid_x,
                    "changed_centroid_y": centroid_y,
                    "proof_track": int(track),
                    "proof_track_state": state,
                    "point_distance_radii": float(scaled),
                    "proof_track_visible_frames": support,
                    "minimum_lineage_support_frames": minimum_support,
                    "raw_positive_fraction": raw_positive,
                    "preexisting_component_retained": bool(count_before > 0),
                    "classification": (
                        (("retained_accepted_projection_plus_"
                          "durable_raw_supported_body")
                         if count_before > 0 else
                         "multiple_durable_raw_supported_body_parts")
                        if passed
                        else "unproved_component_increase"),
                    "passes_projection_proof": passed,
                })
    return pd.DataFrame(rows, columns=PROJECTION_COLUMNS)


def run(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Apply the reviewed target-free rule to current-run evidence."""
    enabled, params = _configured(config_values)
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, candidate_unclaimed, audit, frames, metrics = \
            recovery.produce(
                labels, unclaimed, points, params, raw, thresholds)
        projections = _projection_proof(
            labels, candidate, raw, points, frames, params)
        unproved = int((~projections.passes_projection_proof.astype(bool)).sum()) \
            if len(projections) else 0
        proved_instances = int(projections.drop_duplicates(
            ["analysis_frame", "owner"]).duplicate_increase.sum()) \
            if len(projections) else 0
        if (proved_instances != int(metrics["new_duplicate_components"])
                or unproved):
            raise AssertionError(
                "staggered fusion exchange created an unproved component")
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=recovery.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=recovery.FRAME_COLUMNS)
        projections = pd.DataFrame(columns=PROJECTION_COLUMNS)
        unproved = 0
        proved_instances = 0
        metrics = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": recovery.selector_counts(params),
            "physical_tracks_audited": 0, "reciprocal_pairs_audited": 0,
            "eligible_pairs": 0, "applied_pairs": 0, "frame_rows": 0,
            "applied": False, "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0, "foreground_removed_pixels": 0,
            "zero_signal_additions": 0, "unclaimed_overlap_pixels": 0,
            "unclaimed_reassigned_pixels": 0, "new_identity_count": 0,
            "removed_identity_count": 0, "component_excess_before": 0,
            "component_excess_after": 0,
            "duplicate_owner_frames_before": 0,
            "duplicate_owner_frames_after": 0,
            "new_duplicate_components": 0, "applied_proposals": [],
        }

    audit.to_csv(stage / "staggered_fusion_exchange_audit.csv", index=False)
    frames.to_csv(stage / "staggered_fusion_exchange_frames.csv", index=False)
    projections.to_csv(
        stage / "staggered_fusion_exchange_projection_proof.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **metrics,
        "projection_proof_rows": int(len(projections)),
        "projection_proved_component_instances": proved_instances,
        "unproved_new_duplicate_components": unproved,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, audit, frames, projections,
            metrics)
