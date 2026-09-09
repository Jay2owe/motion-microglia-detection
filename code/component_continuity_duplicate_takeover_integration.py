"""Production adapter for direct component-continuity owner takeover repair."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import component_continuity_duplicate_takeover as takeover


CONFIG_KEY = "field_wide_component_continuity_duplicate_takeover"
STAGE_NAME = "92_component_continuity_duplicate_takeover"
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
            "component-continuity takeover production configuration "
            "contains unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "component-continuity takeover production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    takeover.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Run the accepted direct-takeover producer on current-run evidence."""
    enabled, params = _configured(config_values)
    evidence = Path(output_root) / "26_latent_body_tracks/out"
    points_path = evidence / "latent_track_points.csv"
    if not points_path.is_file():
        raise FileNotFoundError(
            "component-continuity takeover evidence is missing: "
            + str(points_path))
    points = pd.read_csv(points_path)
    if enabled:
        candidate, candidate_unclaimed, audit, applications, summary = \
            takeover.produce(labels, unclaimed, raw, points, params)
    else:
        candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
        audit = pd.DataFrame(columns=takeover.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=takeover.APPLICATION_COLUMNS)
        summary = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "review_cases")},
            "transitions_audited": 0,
            "eligible_takeovers": 0,
            "applied_takeovers": 0,
            "application_rows": 0,
            "changed_pixels": 0,
            "changed_frames": 0,
            "foreground_ledger_exact": True,
            "preexisting_unclaimed_exact": True,
            "identity_set_exact": True,
            "new_identity_count": 0,
            "removed_identity_count": 0,
            "new_duplicate_components": 0,
        }
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "component-continuity takeover changed foreground")
    if not np.array_equal(candidate_unclaimed, unclaimed):
        raise AssertionError(
            "component-continuity takeover changed unclaimed ledger")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError(
            "component-continuity takeover changed active identity set")
    if int(summary["new_duplicate_components"]):
        raise AssertionError(
            "component-continuity takeover introduced duplicates")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage / "component_continuity_takeover_audit.csv", index=False)
    applications.to_csv(
        stage / "component_continuity_takeover_applications.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **summary,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "label_foreground_exact": True,
        "unclaimed_array_exact": True,
        "identity_set_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, applications, metrics
