"""Production adapter for field-wide dormant-owner reciprocal partition."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import dormant_owner_reciprocal_partition as partition


CONFIG_KEY = "field_wide_dormant_owner_reciprocal_partition"
STAGE_NAME = "119_dormant_owner_reciprocal_partition"
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
            "dormant-owner reciprocal-partition production configuration "
            "contains unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "dormant-owner reciprocal partition must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "targeting_mode": "field_wide_discovery",
        "labels_path": "current-run labels array",
        "unclaimed_path": "current-run unclaimed array",
        "raw_path": "current-run raw array",
        "physical_track_points_path": "current-run physical points table",
        "apply_recovery": True,
    })
    partition.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, config_values: dict[str, Any],
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the reviewed rule to current-run field-wide evidence."""
    enabled, params = _configured(config_values)
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, candidate_unclaimed, audit, applications, metrics = \
            partition.produce(labels, unclaimed, raw, points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=partition.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=partition.APPLICATION_COLUMNS)
        metrics = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "components_audited": 0,
            "owner_transitions_audited": 0,
            "eligible_takeovers": 0,
            "applied_takeovers": 0,
            "reciprocal_partitions": 0,
            "reciprocal_pair_frames": 0,
            "changed_pixels": 0,
            "changed_frames": 0,
            "foreground_exact": True,
            "unclaimed_ledger_exact": True,
            "movie_identity_set_exact": True,
            "new_component_excess": 0,
            "new_unexplained_component_excess": 0,
            "partition_explanation":
                "continuous_takeover_body_plus_unique_durable_separate_body",
        }

    audit.to_csv(
        stage / "reciprocal_separable_partition_audit.csv", index=False)
    applications.to_csv(
        stage / "reciprocal_separable_partition_applications.csv",
        index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **metrics,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, applications, metrics
