"""Production adapter for field-wide large persistent-reference relay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import large_persistent_reference_relay as relay


CONFIG_KEY = "field_wide_large_persistent_reference_relay"
STAGE_NAME = "118_large_persistent_reference_relay"
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
            "large persistent-reference relay production configuration "
            "contains unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "large persistent-reference relay must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "targeting_mode": "field_wide_discovery",
        # The reusable core validates the standalone file contract. Production
        # supplies the equivalent current-run arrays and table in memory.
        "labels_path": "current-run labels array",
        "unclaimed_path": "current-run unclaimed array",
        "raw_path": "current-run raw array",
        "physical_track_points_path": "current-run physical points table",
    })
    relay.assert_target_free(params)
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
        candidate, candidate_unclaimed, audit, frames, metrics = relay.produce(
            labels, unclaimed, raw, points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=relay.TRACK_COLUMNS)
        frames = pd.DataFrame(columns=relay.FRAME_COLUMNS)
        metrics = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "physical_tracks_audited": 0,
            "eligible_large_references": 0,
            "applied_large_references": 0,
            "applied_runs": 0,
            "applied_frames": 0,
            "changed_label_pixels": 0,
            "changed_unclaimed_pixels": 0,
            "changed_frames": 0,
            "foreground_ledger_union_exact": True,
            "preexisting_unclaimed_exact": True,
            "movie_identity_set_exact": True,
            "new_duplicate_components": 0,
        }

    audit.to_csv(
        stage / "large_persistent_reference_relay_audit.csv", index=False)
    frames.to_csv(
        stage / "large_persistent_reference_relay_frames.csv", index=False)
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
    return candidate, candidate_unclaimed, audit, frames, metrics
