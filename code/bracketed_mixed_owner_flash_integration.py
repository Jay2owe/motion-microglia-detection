"""Production adapter for bracketed mixed-owner flash recovery.

The accepted target-free detector restores one physical owner across a very
short A-[B,C,...]-A excursion only when raw shared-mask evidence proves that
at least one foreign mask contains another physical core. The whole interval
is committed atomically and proposals that would create a duplicate owner are
refused.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bracketed_mixed_owner_flash as flash


CONFIG_KEY = "field_wide_bracketed_mixed_owner_flash"
STAGE_NAME = "76_bracketed_mixed_owner_flash"
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
            "bracketed mixed-owner flash production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "bracketed mixed-owner flash production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    flash.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def _empty_details() -> dict[str, int]:
    return {
        "applied_proposals": 0,
        "changed_pixels": 0,
        "changed_frames": 0,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": 0,
    }


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run evidence."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (Path(output_root)
                   / "26_latent_body_tracks/out/latent_track_points.csv")

    if enabled:
        if not points_path.is_file():
            raise FileNotFoundError(
                "bracketed mixed-owner flash requires current-run physical "
                f"evidence: {points_path}")
        points = pd.read_csv(points_path)
        audit, internal, attached = flash.discover(labels, points, params)
        candidate, applications, details = flash.apply(
            labels, raw, attached, internal, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        audit = pd.DataFrame(columns=flash.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=flash.APPLICATION_COLUMNS)
        candidate = labels.copy()
        details = _empty_details()

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError("bracketed mixed-owner flash changed foreground")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError("bracketed mixed-owner flash changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "bracketed mixed-owner flash overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("bracketed mixed-owner flash changed identity set")
    if int(details["new_duplicate_components"]):
        raise AssertionError(
            "bracketed mixed-owner flash created unexplained duplicates")

    audit.to_csv(stage / "bracketed_mixed_owner_flash_audit.csv", index=False)
    applications.to_csv(
        stage / "bracketed_mixed_owner_flash_applications.csv", index=False)
    changed = candidate != before
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "owners", "identities", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique())
            if "track_id" in points else 0,
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        **details,
        "changed_frame_indices": changed_frames,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, before_unclaimed, audit, applications, metrics
