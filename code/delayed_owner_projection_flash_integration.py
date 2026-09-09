"""Production adapter for delayed-owner projection-flash recovery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import delayed_owner_projection_flash as projection


CONFIG_KEY = "field_wide_delayed_owner_projection_flash"
STAGE_NAME = "97_delayed_owner_projection_flash"
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
            "delayed-owner projection-flash production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "delayed-owner projection-flash production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    projection.flash.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run evidence."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (Path(output_root)
                   / "26_latent_body_tracks/out/latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "delayed-owner projection-flash evidence is missing: "
            + str(points_path))
    points = pd.read_csv(points_path)

    if enabled:
        audit, internal, _attached, base_audit = projection.discover(
            labels, raw, points, params)
        candidate, applications, details = projection.apply(labels, internal)
    else:
        audit = pd.DataFrame(columns=projection.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=projection.APPLICATION_COLUMNS)
        base_audit = pd.DataFrame(columns=projection.flash.AUDIT_COLUMNS)
        candidate = labels.copy()
        details = {
            "applied_proposals": 0, "changed_pixels": 0,
            "changed_frames": 0, "new_identity_count": 0,
            "removed_identity_count": 0,
            "new_explained_projection_components": 0,
            "new_duplicate_components": 0,
        }

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError(
            "delayed-owner projection flash changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "delayed-owner projection flash overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError(
            "delayed-owner projection flash changed identity set")
    if int(details["new_duplicate_components"]):
        raise AssertionError(
            "delayed-owner projection flash created unexplained duplicates")

    audit.to_csv(
        stage / "delayed_owner_projection_flash_audit.csv", index=False)
    applications.to_csv(
        stage / "delayed_owner_projection_flash_applications.csv", index=False)
    base_audit.to_csv(stage / "medium_window_base_audit.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "owners", "identities", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "physical_tracks_audited": int(points.track_id.nunique()),
        "base_proposals_audited": int(len(base_audit)),
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
    return (candidate, unclaimed.copy(), audit, applications,
            base_audit, metrics)
