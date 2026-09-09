"""Production adapter for accepted A5 Issue 004 split inheritance.

The detector restores two established identities through a resolved split and
a short terminal shared mask using only current labels, raw signal, physical
tracks, and current-run accepted-lineage audits.  It has no case selectors and
enforces foreground, unclaimed, identity, and component-count invariants.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import owner_consensus
import terminal_two_core_split_inheritance as inheritance


CONFIG_KEY = "field_wide_terminal_two_core_split_inheritance"
STAGE_NAME = "70_terminal_two_core_split_inheritance"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "targeting_mode",
    "reviewed_candidate_labels_sha256",
    "reviewed_candidate_unclaimed_sha256", "parameters",
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"current-run evidence is missing: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "terminal two-core production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal two-core production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run evidence."""
    enabled, params = _configured(config_values)
    labels_before = labels.copy()
    unclaimed_before = unclaimed.copy()
    output_root = Path(output_root)
    stage = output_root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (
        output_root / "26_latent_body_tracks/out/latent_track_points.csv")
    lineage_stage = output_root / "67_complete_flip_recovery/out"
    parent_audit_path = lineage_stage / "branched_motion_lineage_audit.csv"
    parent_applications_path = (
        lineage_stage / "branched_motion_lineage_applications.csv")
    params.update({
        "labels_path": "<current-run-label-array>",
        "unclaimed_path": "<current-run-unclaimed-array>",
        "raw_path": "<current-run-raw-array>",
        "physical_track_points_path": str(points_path),
        "parent_lineage_audit_path": str(parent_audit_path),
        "parent_lineage_applications_path": str(parent_applications_path),
        "output_stem": "current_run",
        "mode": "accepted",
    })
    inheritance.assert_target_free(params)

    if enabled:
        points = _read_table(points_path)
        _read_table(parent_audit_path)
        _read_table(parent_applications_path)
        candidate, audit, frames = inheritance.discover_and_apply(
            labels, unclaimed, points, raw, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        candidate = labels.copy()
        audit = pd.DataFrame(columns=inheritance.base.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=inheritance.base.FRAME_COLUMNS)

    if not np.array_equal(candidate > 0, labels_before > 0):
        raise AssertionError("terminal two-core stage changed foreground")
    if not np.array_equal(unclaimed, unclaimed_before):
        raise AssertionError("terminal two-core stage changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "terminal two-core stage overlaps assigned and unclaimed pixels")
    before_ids = set(map(int, np.unique(labels_before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("terminal two-core stage changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels_before, candidate)
    if duplicates:
        raise AssertionError("terminal two-core stage created duplicates")

    audit.to_csv(stage / "terminal_two_core_split_audit.csv", index=False)
    frames.to_csv(stage / "terminal_two_core_split_frames.csv", index=False)
    changed = candidate != labels_before
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    metrics = {
        "version": config_values["accepted_postprocessing"].get(
            CONFIG_KEY, {}).get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": inheritance.selector_counts(params),
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique())
            if "track_id" in points else 0,
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.map(
            inheritance.base._truth).sum()) if len(audit) else 0,
        "applied_proposals": int(audit.applied.map(
            inheritance.base._truth).sum()) if len(audit) else 0,
        "terminal_partitions": int((frames.action.astype(str)
                                     == "terminal_two_core_partition").sum())
            if len(frames) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": len(changed_frames),
        "changed_frame_indices": changed_frames,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": len(after_ids - before_ids),
        "removed_identity_count": len(before_ids - after_ids),
        "foreground_exact": True,
        "unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(duplicates),
        "parameters": {key: value for key, value in params.items()
                       if key not in inheritance.base.PATH_KEYS
                       | inheritance.EXTRA_PATH_KEYS},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed_before, audit, frames, metrics
