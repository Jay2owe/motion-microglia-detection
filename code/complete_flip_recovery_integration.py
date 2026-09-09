"""Production adapter for accepted A5 Issue 003 complete-flip recovery.

The algorithm discovers every eligible successor alias and owner takeover from
current-run labels and physical tracks.  This adapter supplies only those
arrays and tables and enforces the accepted foreground, unclaimed-ledger,
identity, and physical-lineage projection invariants.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import complete_flip_recovery as recovery


CONFIG_KEY = "field_wide_complete_flip_recovery"
STAGE_NAME = "67_complete_flip_recovery"
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
            "complete-flip production configuration contains unsupported "
            "keys (selectors are forbidden): " + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("complete-flip production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run track evidence."""
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
        "physical_track_points_path": str(points_path),
        "output_stem": "current_run",
        "mode": "accepted",
    })
    recovery.assert_target_free(params)

    if enabled:
        points = _read_table(points_path)
        audit, internal = recovery.discover(labels, points, params)
        candidate, applications, proven_mask = recovery.apply(
            labels, internal)
    else:
        points = pd.DataFrame(columns=["track_id"])
        candidate = labels.copy()
        audit = pd.DataFrame(columns=recovery.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=recovery.APPLICATION_COLUMNS)
        proven_mask = np.zeros(labels.shape, dtype=bool)

    if not np.array_equal(candidate > 0, labels_before > 0):
        raise AssertionError("complete-flip recovery changed foreground")
    if not np.array_equal(unclaimed, unclaimed_before):
        raise AssertionError("complete-flip recovery changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "complete-flip recovery overlaps assigned and unclaimed pixels")
    before_ids = set(map(int, np.unique(labels_before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids - before_ids:
        raise AssertionError("complete-flip recovery created an identity")
    new_duplicates, proven_duplicates, duplicate_proof = \
        recovery._proven_new_duplicates(
            labels_before, candidate, proven_mask)
    if new_duplicates != proven_duplicates:
        raise AssertionError(
            "new duplicate component lacks physical-lineage proof")

    audit.to_csv(stage / "branched_motion_lineage_audit.csv", index=False)
    applications.to_csv(
        stage / "branched_motion_lineage_applications.csv", index=False)
    duplicate_proof.to_csv(
        stage / "new_duplicate_lineage_proof.csv", index=False)
    changed = candidate != labels_before
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    eligible = audit[audit.eligible.astype(bool)] if len(audit) else audit
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "version": config_values["accepted_postprocessing"].get(
            CONFIG_KEY, {}).get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": recovery.selector_counts(params),
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique())
            if "track_id" in points else 0,
        "identity_pairs_audited": int(audit.proposal_kind.eq(
            "complete_complement").sum()) if len(audit) else 0,
        "takeover_components_audited": int(audit.proposal_kind.eq(
            "bounded_component_excursion").sum()) if len(audit) else 0,
        "eligible_proposals": int(len(eligible)),
        "applied_atomic_groups": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": len(changed_frames),
        "changed_frame_indices": changed_frames,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": len(after_ids - before_ids),
        "removed_identity_count": len(before_ids - after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(new_duplicates),
        "lineage_proven_new_duplicate_components": int(proven_duplicates),
        "all_new_duplicates_injectively_unanimous": bool(
            new_duplicates == proven_duplicates),
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")
                       and key not in {"labels_path", "unclaimed_path"}},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed_before, audit, applications,
            duplicate_proof, metrics)
