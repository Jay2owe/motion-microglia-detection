"""Production adapter for field-wide terminal projection-chain completion."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import terminal_projection_chain_completion as completion


CONFIG_KEY = "field_wide_terminal_projection_chain_completion"
STAGE_NAME = "103_terminal_projection_chain_completion"
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
            "terminal projection-chain production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal projection-chain completion must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    completion.flash.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Complete only unassigned raw cores in fully proved terminal chains."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (Path(output_root)
                   / "26_latent_body_tracks/out/latent_track_points.csv")
    thresholds_path = (Path(output_root)
                       / "25_raw_physical_hypotheses/out"
                       / "frame_evidence_thresholds.csv")
    for evidence_path in (points_path, thresholds_path):
        if not evidence_path.is_file():
            raise FileNotFoundError(
                "terminal projection-chain evidence is missing: "
                + str(evidence_path))
    points = pd.read_csv(points_path)
    thresholds = pd.read_csv(thresholds_path)
    if enabled:
        audit, internal, attached, parent_count = completion.discover(
            labels, points, params)
        candidate, applications, details = completion.apply(
            labels, unclaimed, raw, thresholds, attached, internal, params)
    else:
        audit = pd.DataFrame(columns=completion.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=completion.APPLICATION_COLUMNS)
        candidate = labels.copy()
        parent_count = 0
        details = {
            "eligible_proposals": 0, "applied_proposals": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_added_pixels": 0,
            "preexisting_assigned_changed_pixels": 0,
            "unclaimed_overlap_pixels": 0, "zero_signal_additions": 0,
            "new_identity_count": 0, "removed_identity_count": 0,
            "new_explained_projection_components": 0,
            "new_duplicate_components": 0,
        }
    if int(details["preexisting_assigned_changed_pixels"]):
        raise AssertionError("terminal chain changed assigned pixels")
    if int(details["unclaimed_overlap_pixels"]) or int(
            details["zero_signal_additions"]):
        raise AssertionError("terminal chain violated accepted ledgers")
    if int(details["new_duplicate_components"]):
        raise AssertionError("terminal chain created unexplained duplicates")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("terminal chain changed identity set")

    audit.to_csv(stage / "terminal_projection_chain_audit.csv", index=False)
    applications.to_csv(
        stage / "terminal_projection_chain_applications.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "physical_tracks_audited": int(points.track_id.nunique()),
        "projection_parents_audited": int(parent_count),
        **details,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, applications, metrics
