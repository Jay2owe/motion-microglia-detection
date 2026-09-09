"""Production adapter for reconnection-guided two-seat merge partition."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import reconnected_companion_merge_partition as partition


CONFIG_KEY = "field_wide_reconnected_companion_merge_partition"
STAGE_NAME = "91_reconnected_companion_merge_partition"
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
            "reconnected companion partition production configuration "
            "contains unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "reconnected companion partition production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    partition.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Run the accepted producer from current latent-track evidence."""
    enabled, params = _configured(config_values)
    evidence = Path(output_root) / "26_latent_body_tracks/out"
    points_path = evidence / "latent_track_points.csv"
    reconnections_path = evidence / "reconnection_hypotheses.csv"
    missing = [path for path in (points_path, reconnections_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "reconnected companion partition evidence is missing: "
            + ", ".join(map(str, missing)))
    points = pd.read_csv(points_path)
    reconnections = pd.read_csv(reconnections_path)
    if enabled:
        candidate, candidate_unclaimed, audit, applications, summary = \
            partition.produce(
                labels, unclaimed, points, reconnections, params)
    else:
        candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
        audit = pd.DataFrame(columns=partition.AUDIT_COLUMNS)
        applications = pd.DataFrame()
        summary = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "review_cases")},
            "reconnections_audited": int(len(reconnections)),
            "candidate_pairings_audited": 0,
            "eligible_partitions": 0,
            "applied_partitions": 0,
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
            "reconnected companion partition changed foreground")
    if not np.array_equal(candidate_unclaimed, unclaimed):
        raise AssertionError(
            "reconnected companion partition changed unclaimed")
    if int(summary["new_duplicate_components"]):
        raise AssertionError(
            "reconnected companion partition introduced duplicates")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage / "reconnected_companion_merge_audit.csv", index=False)
    applications.to_csv(
        stage / "reconnected_companion_merge_applications.csv", index=False)
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
