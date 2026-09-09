"""Production adapter for accepted A5 Issue 005 two-seat preservation.

The field-wide detector preserves a durable pre-contact seat through an owner
collapse while allowing independently proved same-lineage projections.  This
adapter supplies only current-run arrays and physical-track evidence and
enforces the accepted foreground, unclaimed, identity, and projection proofs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import projection_tolerant_seat_preservation as preservation


CONFIG_KEY = "field_wide_projection_tolerant_seat_preservation"
STAGE_NAME = "68_projection_tolerant_seat_preservation"
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
            "projection-tolerant seat production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "projection-tolerant seat production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run evidence."""
    enabled, params = _configured(config_values)
    labels_before = labels.copy()
    unclaimed_before = unclaimed.copy()
    output_root = Path(output_root)
    stage = output_root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (
        output_root / "26_latent_body_tracks/out/latent_track_points.csv")
    params.update({
        "labels_path": "<current-run-label-array>",
        "unclaimed_path": "<current-run-unclaimed-array>",
        "raw_path": "<current-run-raw-array>",
        "physical_track_points_path": str(points_path),
        "output_stem": "current_run",
        "mode": "accepted",
    })
    preservation.assert_target_free(params)

    if enabled:
        points = _read_table(points_path)
        candidate, audit, frames, projections, proven_mask = \
            preservation.discover_and_apply(labels, raw, points, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        candidate = labels.copy()
        audit = pd.DataFrame(columns=preservation.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=preservation.FRAME_COLUMNS)
        projections = pd.DataFrame(columns=preservation.PROJECTION_COLUMNS)
        proven_mask = np.zeros(labels.shape, dtype=bool)

    if not np.array_equal(candidate > 0, labels_before > 0):
        raise AssertionError("projection-tolerant seat stage changed foreground")
    if not np.array_equal(unclaimed, unclaimed_before):
        raise AssertionError("projection-tolerant seat stage changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "projection-tolerant seat stage overlaps assigned and unclaimed pixels")
    before_ids = set(map(int, np.unique(labels_before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("projection-tolerant seat stage changed identity set")
    new_duplicates, proven_duplicates, duplicate_proof = \
        preservation._duplicate_proof(
            labels_before, candidate, frames, projections, proven_mask)
    if new_duplicates != proven_duplicates:
        raise AssertionError(
            "new duplicate component lacks same-lineage projection proof")

    audit.to_csv(stage / "seat_preservation_audit.csv", index=False)
    frames.to_csv(stage / "seat_preservation_frames.csv", index=False)
    projections.to_csv(stage / "projection_lineage_audit.csv", index=False)
    duplicate_proof.to_csv(
        stage / "new_duplicate_lineage_proof.csv", index=False)
    changed = candidate != labels_before
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "version": config_values["accepted_postprocessing"].get(
            CONFIG_KEY, {}).get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": preservation.selector_counts(params),
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique())
            if "track_id" in points else 0,
        "transitions_audited": int(len(audit)),
        "eligible_transitions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": len(changed_frames),
        "changed_frame_indices": changed_frames,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": len(after_ids - before_ids),
        "removed_identity_count": len(before_ids - after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(new_duplicates),
        "lineage_proven_new_duplicate_components": int(proven_duplicates),
        "all_new_duplicates_projection_proven": bool(
            new_duplicates == proven_duplicates),
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")
                       and key not in {
                           "labels_path", "unclaimed_path", "raw_path"}},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed_before, audit, frames, projections,
            duplicate_proof, metrics)
