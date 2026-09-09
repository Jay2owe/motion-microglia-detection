"""Production adapter for proved-lineage raw projection-gap completion."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import anchored_projection_raw_gap_completion as completion


CONFIG_KEY = "field_wide_anchored_projection_raw_gap_completion"
STAGE_NAME = "101_anchored_projection_raw_gap_completion"
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
            "anchored raw-gap production configuration contains unsupported "
            "keys (selectors are forbidden): " + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("anchored raw-gap completion must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    completion.flash.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        relay_audit: pd.DataFrame, relay_applications: pd.DataFrame,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Complete only ownerless raw cores inside a proved relay lineage."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (Path(output_root)
                   / "26_latent_body_tracks/out/latent_track_points.csv")
    thresholds_path = (Path(output_root)
                       / "25_raw_physical_hypotheses/out"
                       / "frame_evidence_thresholds.csv")
    for evidence_path in (points_path, thresholds_path):
        if not evidence_path.is_file():
            raise FileNotFoundError(
                "anchored raw-gap evidence is missing: "
                + str(evidence_path))
    points = pd.read_csv(points_path)
    thresholds = pd.read_csv(thresholds_path)
    if enabled:
        candidate, audit, details = completion.complete(
            labels, unclaimed, raw, points, thresholds,
            relay_audit, relay_applications, params)
    else:
        audit = pd.DataFrame(columns=completion.AUDIT_COLUMNS)
        candidate = labels.copy()
        details = {
            "proved_relays_consumed": 0, "applied_proposals": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0,
            "preexisting_assigned_changed_pixels": 0,
            "unclaimed_overlap_pixels": 0, "zero_signal_additions": 0,
            "new_identity_count": 0, "removed_identity_count": 0,
            "new_explained_projection_components": 0,
            "new_duplicate_components": 0,
        }
    if int(details["preexisting_assigned_changed_pixels"]):
        raise AssertionError("anchored raw-gap completion changed assigned pixels")
    if int(details["unclaimed_overlap_pixels"]) or int(
            details["zero_signal_additions"]):
        raise AssertionError("anchored raw-gap completion violated input ledgers")
    if int(details["new_duplicate_components"]):
        raise AssertionError(
            "anchored raw-gap completion created unexplained duplicates")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("anchored raw-gap completion changed identity set")

    audit.to_csv(
        stage / "anchored_projection_raw_gap_completion_audit.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "owners", "identities", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        **details,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, metrics
