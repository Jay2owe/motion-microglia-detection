"""Production adapter for accepted A5 Issue 006 successor separation.

The field-wide detector backfills a movie-novel successor over a proved
continuing physical track when the displaced predecessor has one independent,
durable seat.  This adapter supplies only current-run arrays and track evidence
and enforces foreground, unclaimed, identity, and component-count invariants.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import owner_consensus
import retroactive_successor_separation as separation


CONFIG_KEY = "field_wide_retroactive_successor_separation"
STAGE_NAME = "69_retroactive_successor_separation"
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
            "retroactive successor production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("retroactive successor production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
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
    separation.assert_target_free(params)

    if enabled:
        points = _read_table(points_path)
        candidate, audit, frames = separation.discover_and_apply(
            labels, points, raw, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        candidate = labels.copy()
        audit = pd.DataFrame(columns=separation.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=separation.FRAME_COLUMNS)

    if not np.array_equal(candidate > 0, labels_before > 0):
        raise AssertionError("retroactive successor stage changed foreground")
    if not np.array_equal(unclaimed, unclaimed_before):
        raise AssertionError("retroactive successor stage changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "retroactive successor stage overlaps assigned and unclaimed pixels")
    before_ids = set(map(int, np.unique(labels_before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("retroactive successor stage changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels_before, candidate)
    if duplicates:
        raise AssertionError(
            "retroactive successor stage created duplicate components")

    audit.to_csv(stage / "retroactive_successor_audit.csv", index=False)
    frames.to_csv(stage / "retroactive_successor_frames.csv", index=False)
    duplicate_proof = pd.DataFrame(columns=[
        "proposal_id", "frame", "successor_owner", "new_excess", "proof"])
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
        "target_counts": separation.selector_counts(params),
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
        "new_duplicate_components": int(duplicates),
        "lineage_proven_new_duplicate_components": 0,
        "all_new_duplicates_projection_proven": duplicates == 0,
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")
                       and key not in {
                           "labels_path", "unclaimed_path", "raw_path"}},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed_before, audit, frames, duplicate_proof,
            metrics)
