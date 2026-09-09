"""Production adapter for recording-start two-seat inheritance."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import recording_start_two_seat_inheritance as inheritance


CONFIG_KEY = "field_wide_recording_start_two_seat_inheritance"
STAGE_NAME = "87_recording_start_two_seat_inheritance"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "targeting_mode",
    "reviewed_candidate_labels_sha256", "parameters",
}


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "recording-start two-seat production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "recording-start two-seat production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    inheritance.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Run the producer from current arrays and accepted physical evidence."""
    enabled, params = _configured(config_values)
    points_path = (Path(output_root) / "26_latent_body_tracks/out"
                   / "latent_track_points.csv")
    encounters_path = (Path(output_root) / "26_latent_body_tracks/out"
                       / "encounter_frames.csv")
    missing = [str(path) for path in (points_path, encounters_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "recording-start two-seat evidence is missing: "
            + ", ".join(missing))
    points = pd.read_csv(points_path)
    encounters = pd.read_csv(encounters_path)
    if enabled:
        candidate, candidate_unclaimed, audit, applications, summary = \
            inheritance.produce(
                labels, unclaimed, raw, points, encounters, params)
    else:
        candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
        audit, applications = pd.DataFrame(), pd.DataFrame()
        summary = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "review_cases")},
            "boundary_encounters_audited": 0,
            "eligible_proposals": 0,
            "applied_proposals": 0,
            "changed_pixels": 0,
            "changed_frames": 0,
            "explained_projection_components_added": 0,
            "unexplained_duplicate_components_added": 0,
            "foreground_ledger_exact": True,
            "preexisting_unclaimed_exact": True,
            "old_identity_set_preserved": True,
            "new_identity_count": 0,
            "removed_identity_count": 0,
        }
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("recording-start two-seat changed foreground")
    if not np.array_equal(candidate_unclaimed, unclaimed):
        raise AssertionError("recording-start two-seat changed unclaimed data")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("recording-start two-seat changed identity set")
    if int(summary["unexplained_duplicate_components_added"]):
        raise AssertionError(
            "recording-start two-seat introduced unexplained duplicates")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage / "recording_start_two_seat_audit.csv", index=False)
    applications.to_csv(
        stage / "recording_start_two_seat_applications.csv", index=False)
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
