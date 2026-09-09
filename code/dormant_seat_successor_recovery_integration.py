"""Production adapter for dormant-seat movie-new-successor recovery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import dormant_seat_successor_recovery as recovery


CONFIG_KEY = "field_wide_dormant_seat_successor_recovery"
STAGE_NAME = "116_dormant_seat_successor_recovery"
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
            "dormant-seat successor production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "dormant-seat successor recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "mode": "candidate",
        "targeting_mode": "field_wide_discovery",
        # The core validates its standalone file contract even though this
        # adapter supplies in-memory arrays and tables.
        "labels_path": "current-run labels array",
        "unclaimed_path": "current-run unclaimed array",
        "raw_path": "current-run raw array",
        "physical_track_points_path": "current-run physical points table",
        "thresholds_path": "current-run evidence thresholds table",
    })
    recovery.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the reviewed target-free rule to current-run evidence."""
    enabled, params = _configured(config_values)
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, candidate_unclaimed, audit, members, frames, \
            projections, metrics = recovery.produce(
                labels, unclaimed, raw, points, thresholds, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=recovery.AUDIT_COLUMNS)
        members = pd.DataFrame(columns=recovery.LINEAGE_COLUMNS)
        frames = pd.DataFrame(columns=recovery.FRAME_COLUMNS)
        projections = pd.DataFrame(columns=recovery.PROJECTION_COLUMNS)
        metrics = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "physical_tracks_audited": 0, "proposals_audited": 0,
            "eligible_proposals": 0, "lineage_members": 0,
            "applied": False, "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0, "foreground_removed_pixels": 0,
            "unclaimed_overlap_pixels": 0, "zero_signal_additions": 0,
            "new_identity_count": 0, "removed_identity_count": 0,
            "retired_successor_aliases": [],
            "unexplained_removed_identities": [],
            "raw_new_duplicate_components": 0,
            "component_excess_before": 0, "component_excess_after": 0,
            "duplicate_owner_frames_before": 0,
            "duplicate_owner_frames_after": 0,
            "projection_proof_rows": 0,
            "unexplained_new_duplicate_components": 0,
            "applied_proposals": [],
        }

    audit.to_csv(stage / "dormant_seat_successor_audit.csv", index=False)
    members.to_csv(stage / "dormant_seat_successor_members.csv", index=False)
    frames.to_csv(stage / "dormant_seat_successor_frames.csv", index=False)
    projections.to_csv(stage / "projection_host_proof.csv", index=False)
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
    return (candidate, candidate_unclaimed, audit, members, frames,
            projections, metrics)
