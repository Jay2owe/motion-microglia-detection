"""Production adapter for left-censored fragmented ownerless lineages."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import temporal_core_path_lineage as lineage


CONFIG_KEY = "field_wide_temporal_core_path_lineage"
STAGE_NAME = "111_temporal_core_path_lineage"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "targeting_mode",
    "reviewed_candidate_labels_sha256",
    "reviewed_candidate_unclaimed_sha256", "parameters",
}


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "temporal core-path production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("temporal core-path lineage must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    lineage.assert_target_free(params)
    if float(params.get(
            "maximum_existing_unclaimed_coverage_fraction", 1.0)) != 0.0:
        raise ValueError(
            "accepted temporal core-path lineage requires zero existing "
            "unclaimed representation")
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        events: pd.DataFrame, config_values: dict[str, Any],
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Allocate one complete identity to each proved absent lineage."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    root = Path(output_root)
    stage = root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = root / "26_latent_body_tracks/out/latent_track_points.csv"
    thresholds_path = (
        root / "25_raw_physical_hypotheses/out/"
        "frame_evidence_thresholds.csv")
    if enabled:
        missing = [path for path in (points_path, thresholds_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "temporal core-path lineage requires current-run evidence: "
                + ", ".join(map(str, missing)))
        points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, candidate_unclaimed, audit, applications, \
            frame_audit, details = lineage.produce(
                labels, unclaimed, raw, events, points, thresholds, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=lineage.AUDIT_COLUMNS)
        applications = pd.DataFrame()
        frame_audit = pd.DataFrame()
        identity_count = len(set(map(int, np.unique(labels))) - {0})
        details = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames",
                "coordinates", "events", "regions", "wells",
                "review_cases")},
            "events_audited": 0, "eligible_proposals": 0,
            "applied_proposals": 0, "changed_pixels": 0,
            "changed_frames": 0, "existing_label_changes": 0,
            "unclaimed_overlap_pixels": 0, "unclaimed_additions": 0,
            "unclaimed_pixels_transferred": 0,
            "zero_signal_additions": 0,
            "new_duplicate_components": 0,
            "input_identity_count": identity_count,
            "output_identity_count": identity_count,
            "new_identities": [], "new_identity_count": 0,
        }

    changed = candidate != before
    if not np.array_equal(candidate_unclaimed, before_unclaimed):
        raise AssertionError(
            "temporal core-path lineage changed the unclaimed ledger")
    if np.any(changed & (before > 0)):
        raise AssertionError(
            "temporal core-path lineage changed assigned pixels")
    if np.any(changed & (before_unclaimed > 0)):
        raise AssertionError(
            "temporal core-path lineage overlapped unclaimed pixels")
    if np.any(changed & (raw == 0)):
        raise AssertionError(
            "temporal core-path lineage added zero-signal pixels")
    if int(details.get("new_duplicate_components", 0)):
        raise AssertionError(
            "temporal core-path lineage created duplicate components")

    audit.to_csv(stage / "temporal_core_path_lineage_audit.csv", index=False)
    applications.to_csv(
        stage / "temporal_core_path_lineage_applications.csv", index=False)
    frame_audit.to_csv(
        stage / "temporal_core_path_frame_audit.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **details,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_post_score_label_allocation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "unclaimed_array_exact": bool(np.array_equal(
            candidate_unclaimed, before_unclaimed)),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, audit, applications,
            frame_audit, metrics)
