"""Production adapter for field-wide body-scale ownerless allocation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import body_scale_ownerless_allocation as allocation


CONFIG_KEY = "field_wide_body_scale_ownerless_allocation"
STAGE_NAME = "105_body_scale_ownerless_allocation"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "targeting_mode",
    "reviewed_candidate_labels_sha256", "reviewed_candidate_unclaimed_sha256",
    "parameters",
}


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "body-scale ownerless production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("body-scale ownerless allocation must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    allocation.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Allocate complete durable bodies using current-run field evidence."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    root = Path(output_root)
    stage = root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = root / "26_latent_body_tracks/out/latent_track_points.csv"
    thresholds_path = (
        root / "25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv")
    if enabled:
        missing = [path for path in (points_path, thresholds_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "body-scale ownerless allocation requires current-run "
                "evidence: " + ", ".join(map(str, missing)))
        points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, audit, applications, details = allocation.produce(
            labels, unclaimed, raw, points, thresholds, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=allocation.AUDIT_COLUMNS)
        applications = pd.DataFrame()
        details = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "tracks_audited": 0, "eligible_proposals": 0,
            "applied_proposals": 0, "changed_pixels": 0,
            "changed_frames": 0, "existing_label_changes": 0,
            "unclaimed_overlap_pixels": 0, "zero_signal_additions": 0,
            "new_duplicate_components": 0,
            "input_identity_count": int(len(set(map(int, np.unique(labels))) - {0})),
            "output_identity_count": int(len(set(map(int, np.unique(labels))) - {0})),
            "new_identities": [], "new_identity_count": 0,
        }
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError("body-scale allocation changed unclaimed data")
    if np.any((before > 0) & (candidate != before)):
        raise AssertionError("body-scale allocation changed assigned pixels")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("body-scale allocation overlaps accepted ledgers")
    if np.any((candidate > 0) & (before == 0) & (raw == 0)):
        raise AssertionError("body-scale allocation added zero-signal pixels")
    if int(details["new_duplicate_components"]):
        raise AssertionError("body-scale allocation created duplicates")

    audit.to_csv(stage / "body_scale_ownerless_audit.csv", index=False)
    applications.to_csv(
        stage / "body_scale_ownerless_applications.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **details,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "preexisting_unclaimed_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, before_unclaimed, audit, applications, metrics
