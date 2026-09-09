"""Production adapter for established multi-owner seat-cycle recovery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import established_seat_cycle_recovery as recovery


CONFIG_KEY = "field_wide_established_seat_cycle_recovery"
STAGE_NAME = "115_established_seat_cycle_recovery"
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
            "established seat-cycle production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("established seat-cycle recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "mode": "candidate",
        "targeting_mode": "field_wide_discovery",
        # The reusable core validates its standalone file contract even when
        # called in-process. These sentinels are never opened by ``produce``.
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
                   pd.DataFrame, dict[str, Any]]:
    """Apply the reviewed target-free rule to current-run evidence."""
    enabled, params = _configured(config_values)
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, candidate_unclaimed, audit, seats, frames, metrics = \
            recovery.produce(
                labels, unclaimed, raw, points, thresholds, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=recovery.AUDIT_COLUMNS)
        seats = pd.DataFrame(columns=recovery.SEAT_COLUMNS)
        frames = pd.DataFrame(columns=recovery.FRAME_COLUMNS)
        metrics = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "proposals_audited": 0, "eligible_proposals": 0,
            "seat_members": 0, "applied": False,
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0, "foreground_removed_pixels": 0,
            "unclaimed_overlap_pixels": 0, "zero_signal_additions": 0,
            "new_identity_count": 0, "removed_identity_count": 0,
            "new_duplicate_components": 0,
            "component_excess_before": 0, "component_excess_after": 0,
            "duplicate_owner_frames_before": 0,
            "duplicate_owner_frames_after": 0,
        }

    audit.to_csv(stage / "established_seat_cycle_audit.csv", index=False)
    seats.to_csv(stage / "established_seat_cycle_members.csv", index=False)
    frames.to_csv(stage / "established_seat_cycle_frames.csv", index=False)
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
    return candidate, candidate_unclaimed, audit, seats, frames, metrics
