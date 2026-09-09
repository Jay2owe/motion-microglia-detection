"""Production adapter for bracketed ownerless-seat completion."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bracketed_ownerless_seat_completion as completion


CONFIG_KEY = "field_wide_bracketed_ownerless_seat_completion"
STAGE_NAME = "109_bracketed_ownerless_seat_completion"
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
            "bracketed ownerless-seat production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "bracketed ownerless-seat completion must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    completion.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Complete only uniquely bracketed raw-visible ownerless seats."""
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
                "bracketed ownerless-seat completion requires current-run "
                "evidence: " + ", ".join(map(str, missing)))
        points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, audit, frame_audit, details = completion.complete(
            labels, unclaimed, raw, points, thresholds, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame()
        frame_audit = pd.DataFrame()
        details = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames",
                "coordinates", "events", "regions", "wells",
                "review_cases")},
            "identity_gaps_audited": 0, "eligible_gaps": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "preexisting_assigned_changed_pixels": 0,
            "preexisting_unclaimed_changed_pixels": 0,
            "zero_signal_additions": 0, "new_identity_count": 0,
            "new_duplicate_components": 0,
        }
    changed = candidate != before
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError(
            "bracketed ownerless-seat completion changed unclaimed data")
    if np.any(changed & (before > 0)):
        raise AssertionError(
            "bracketed ownerless-seat completion changed assigned pixels")
    if np.any(changed & (before_unclaimed > 0)):
        raise AssertionError(
            "bracketed ownerless-seat completion overlapped unclaimed data")
    if np.any(changed & (raw == 0)):
        raise AssertionError(
            "bracketed ownerless-seat completion added zero-signal pixels")
    if int(details.get("new_identity_count", 0)):
        raise AssertionError(
            "bracketed ownerless-seat completion created an identity")
    if int(details.get("new_duplicate_components", 0)):
        raise AssertionError(
            "bracketed ownerless-seat completion created a duplicate")

    audit.to_csv(stage / "bracketed_ownerless_seat_audit.csv", index=False)
    frame_audit.to_csv(
        stage / "bracketed_ownerless_seat_frames.csv", index=False)
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
        "unclaimed_array_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, before_unclaimed, audit, frame_audit, metrics
