"""Production adapter for bracketed established-owner invasion repair.

The field-wide detector restores a physical track's bookending owner through
a short A-B-A excursion only when B remains established on an independent
nearby track before, during, and after the excursion.  This adapter supplies
only current-run arrays and physical-track evidence and enforces the accepted
foreground, unclaimed, identity, and component-count contracts.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

import bracketed_established_owner_invasion as invasion
import owner_consensus


CONFIG_KEY = "field_wide_bracketed_established_owner_invasion"
STAGE_NAME = "71_bracketed_established_owner_invasion"
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
            "bracketed owner-invasion production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "bracketed owner-invasion production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the accepted target-free rule using current-run evidence."""
    enabled, params = _configured(config_values)
    labels_before = labels.copy()
    unclaimed_before = unclaimed.copy()
    output_root = Path(output_root)
    stage = output_root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (
        output_root / "26_latent_body_tracks/out/latent_track_points.csv")
    parent_stage = output_root / "65_distinct_history_core_preservation/out"
    application_audit_path = parent_stage / "application_audit.csv"
    conflicted_audit_path = parent_stage / "conflicted_lineage_audit.csv"
    params.update({
        "labels_path": "<current-run-label-array>",
        "unclaimed_path": "<current-run-unclaimed-array>",
        "raw_path": "<current-run-raw-array>",
        "physical_track_points_path": str(points_path),
        "application_audit_path": str(application_audit_path),
        "conflicted_lineage_audit_path": str(conflicted_audit_path),
        "output_stem": "current_run",
        "mode": "accepted",
    })
    invasion.assert_target_free(params)

    if enabled:
        points = _read_table(points_path)
        _read_table(application_audit_path)
        _read_table(conflicted_audit_path)
        audit, internal = invasion.discover(labels, points, params)
        candidate, applications = invasion.bounded.apply(
            labels, raw, points, internal, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        candidate = labels.copy()
        audit = pd.DataFrame(columns=invasion.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=invasion.bounded.APPLICATION_COLUMNS)

    if not np.array_equal(candidate > 0, labels_before > 0):
        raise AssertionError("bracketed owner-invasion stage changed foreground")
    if not np.array_equal(unclaimed, unclaimed_before):
        raise AssertionError("bracketed owner-invasion stage changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "bracketed owner-invasion stage overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(labels_before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("bracketed owner-invasion stage changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels_before, candidate)
    if duplicates:
        raise AssertionError("bracketed owner-invasion stage created duplicates")

    audit.to_csv(
        stage / "bracketed_established_owner_invasion_audit.csv", index=False)
    applications.to_csv(
        stage / "bracketed_established_owner_invasion_applications.csv",
        index=False)
    empty_header = (
        "proposal_id,physical_track,assigned_identity,outcome,reason,"
        "changed_pixels,changed_frames\n")
    for source, name in (
            (application_audit_path, "application_audit.csv"),
            (conflicted_audit_path, "conflicted_lineage_audit.csv")):
        target = stage / name
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text(empty_header, encoding="utf-8")

    changed = candidate != labels_before
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    producer_targets = {key: 0 for key in (
        "identities", "tracks", "frames", "coordinates", "events",
        "regions", "review_cases")}
    metrics = {
        "version": config_values["accepted_postprocessing"].get(
            CONFIG_KEY, {}).get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {"owners": 0, **producer_targets},
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique())
            if "track_id" in points else 0,
        "run_triples_audited": int(len(audit)),
        "eligible_invasions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_invasions": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
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
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")
                       and key not in {
                           "labels_path", "unclaimed_path", "raw_path"}},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed_before, audit, applications, metrics
