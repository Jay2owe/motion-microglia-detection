"""Production adapter for gap-tolerant reciprocal seat exchange."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gap_tolerant_reciprocal_seat_exchange as exchange


CONFIG_KEY = "field_wide_gap_tolerant_reciprocal_seat_exchange"
STAGE_NAME = "90_gap_tolerant_reciprocal_seat_exchange"
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
            "gap-tolerant reciprocal exchange production configuration "
            "contains unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "gap-tolerant reciprocal exchange production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    exchange.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Run the accepted producer from current labels and physical evidence."""
    enabled, params = _configured(config_values)
    points_path = (Path(output_root) / "26_latent_body_tracks/out"
                   / "latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "gap-tolerant reciprocal exchange evidence is missing: "
            f"{points_path}")
    points = pd.read_csv(points_path)
    if enabled:
        candidate, candidate_unclaimed, audit, applications, summary = \
            exchange.produce(labels, unclaimed, points, params)
    else:
        candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
        audit = pd.DataFrame(columns=exchange.AUDIT_COLUMNS)
        applications = pd.DataFrame()
        summary = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "review_cases")},
            "pairs_audited": 0,
            "eligible_exchanges": 0,
            "applied_exchanges": 0,
            "changed_pixels": 0,
            "changed_frames": 0,
            "input_identity_count": len(set(map(int, np.unique(labels))) - {0}),
            "output_identity_count": len(set(map(int, np.unique(labels))) - {0}),
            "new_identity_count": 0,
            "removed_identity_count": 0,
            "foreground_ledger_exact": True,
            "preexisting_unclaimed_exact": True,
            "identity_set_exact": True,
            "new_duplicate_components": 0,
        }
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("gap-tolerant reciprocal exchange changed foreground")
    if not np.array_equal(candidate_unclaimed, unclaimed):
        raise AssertionError("gap-tolerant reciprocal exchange changed unclaimed")
    if int(summary["new_duplicate_components"]):
        raise AssertionError(
            "gap-tolerant reciprocal exchange introduced duplicates")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage / "gap_tolerant_reciprocal_exchange_audit.csv",
                 index=False)
    applications.to_csv(
        stage / "gap_tolerant_reciprocal_exchange_applications.csv",
        index=False)
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
