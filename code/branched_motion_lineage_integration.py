"""Production adapter for accepted A5 Issue 003 branched-lineage recovery.

The recovery algorithm remains responsible for discovery and application.  This
adapter supplies only evidence produced by the current accepted-history run and
enforces the production invariants around it.  It has no identity, track, frame,
coordinate, event, region, or review-case selectors.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

import branched_motion_lineage_recovery as recovery


CONFIG_KEY = "field_wide_branched_motion_lineage_recovery"
STAGE_NAME = "66_branched_motion_lineage_recovery"
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
            "branched-lineage production configuration contains unsupported "
            "keys (selectors are forbidden): " + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("branched-lineage production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the field-wide rule using only evidence from ``output_root``."""
    enabled, params = _configured(config_values)
    unclaimed_before = unclaimed.copy()
    output_root = Path(output_root)
    stage = output_root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    evidence = {
        "physical_track_points_path": (
            output_root / "26_latent_body_tracks/out/latent_track_points.csv"),
        "application_audit_path": (
            output_root /
            "65_distinct_history_core_preservation/out/application_audit.csv"),
        "conflicted_lineage_audit_path": (
            output_root /
            "65_distinct_history_core_preservation/out/"
            "conflicted_lineage_audit.csv"),
        "distinct_history_proposals_path": (
            output_root /
            "65_distinct_history_core_preservation/out/"
            "distinct_history_proposals.csv"),
        "distinct_history_frames_path": (
            output_root /
            "65_distinct_history_core_preservation/out/"
            "distinct_history_frames.csv"),
    }

    if enabled:
        params.update({key: str(path) for key, path in evidence.items()})
        # The array API does not consume label paths, but the production
        # algorithm's closed parameter schema requires explicit provenance for
        # all array inputs. These tokens are descriptive, never selectors.
        params.update({
            "labels_path": "<current-run-label-array>",
            "unclaimed_path": "<current-run-unclaimed-array>",
            "output_stem": "current_run",
            "mode": "accepted",
        })
        recovery.assert_target_free(params)
        points = _read_table(evidence["physical_track_points_path"])
        application = _read_table(evidence["application_audit_path"])
        conflicted = _read_table(evidence["conflicted_lineage_audit_path"])
        proposals = _read_table(evidence["distinct_history_proposals_path"])
        frames = _read_table(evidence["distinct_history_frames_path"])
        audit, internal, step_threshold = recovery.discover(
            labels, points, params, application, conflicted, proposals, frames)
        candidate, applications = recovery.apply(labels, internal)
        for name, source in evidence.items():
            if name == "physical_track_points_path":
                continue
            shutil.copyfile(source, stage / Path(source).name)
    else:
        recovery.assert_target_free({
            **params,
            "labels_path": "<disabled-current-run-label-array>",
            "unclaimed_path": "<disabled-current-run-unclaimed-array>",
            "physical_track_points_path": "<disabled-current-run-tracks>",
            "application_audit_path": "<disabled-current-run-audit>",
            "conflicted_lineage_audit_path": "<disabled-current-run-conflicts>",
            "distinct_history_proposals_path": "<disabled-current-run-proposals>",
            "distinct_history_frames_path": "<disabled-current-run-frames>",
            "output_stem": "current_run", "mode": "accepted",
        })
        candidate = labels.copy()
        audit = pd.DataFrame(columns=recovery.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=recovery.APPLICATION_COLUMNS)
        step_threshold = 0.0

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("branched-lineage recovery changed foreground")
    if not np.array_equal(unclaimed, unclaimed_before):
        raise AssertionError("branched-lineage recovery changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "branched-lineage recovery overlaps assigned and unclaimed pixels")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids - before_ids:
        raise AssertionError("branched-lineage recovery created an identity")
    new_duplicates, proven_duplicates = \
        recovery._duplicate_lineage_verification(
            labels, candidate, internal if enabled else [], applications)
    if new_duplicates != proven_duplicates:
        raise AssertionError(
            "a new duplicate component lacks injective unanimous lineage evidence")

    audit.to_csv(stage / "branched_motion_lineage_audit.csv", index=False)
    applications.to_csv(
        stage / "branched_motion_lineage_applications.csv", index=False)
    changed = candidate != labels
    applied = (applications[applications.applied.astype(bool)]
               if len(applications) else applications)
    metrics = {
        "version": config_values["accepted_postprocessing"].get(
            CONFIG_KEY, {}).get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_atomic_groups": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "field_step_threshold_radii": float(step_threshold),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
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
    return candidate, unclaimed_before, audit, applications, metrics
