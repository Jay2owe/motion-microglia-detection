"""Production adapter for persistent single-owner flash recovery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bracketed_mixed_owner_flash as flash


CONFIG_KEY = "field_wide_persistent_single_owner_flash"
STAGE_NAME = "95_persistent_single_owner_flash"
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
            "persistent single-owner flash production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "persistent single-owner flash production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    flash.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run evidence."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (Path(output_root)
                   / "26_latent_body_tracks/out/latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "persistent single-owner flash evidence is missing: "
            + str(points_path))
    points = pd.read_csv(points_path)

    if enabled:
        audit, internal, attached = flash.discover(labels, points, params)
        candidate, applications, details = flash.apply(
            labels, raw, attached, internal, params)
    else:
        audit = pd.DataFrame(columns=flash.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=flash.APPLICATION_COLUMNS)
        candidate = labels.copy()
        details = {
            "applied_proposals": 0, "changed_pixels": 0,
            "changed_frames": 0, "new_identity_count": 0,
            "removed_identity_count": 0, "new_duplicate_components": 0,
        }

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError(
            "persistent single-owner flash changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "persistent single-owner flash overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError(
            "persistent single-owner flash changed identity set")
    if int(details["new_duplicate_components"]):
        raise AssertionError(
            "persistent single-owner flash created unexplained duplicates")

    audit.to_csv(stage / "persistent_single_owner_flash_audit.csv", index=False)
    applications.to_csv(
        stage / "persistent_single_owner_flash_applications.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "owners", "identities", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "physical_tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        **details,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, applications, metrics
