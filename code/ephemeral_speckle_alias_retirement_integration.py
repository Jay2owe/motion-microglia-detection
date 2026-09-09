"""Production adapter for producer-proved ephemeral alias retirement."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ephemeral_speckle_alias_retirement as retirement


CONFIG_KEY = "field_wide_ephemeral_speckle_alias_retirement"
STAGE_NAME = "100_ephemeral_speckle_alias_retirement"
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
            "ephemeral-alias retirement production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("ephemeral-alias retirement must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    retirement.flash.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Retire only complete identities proved by the field-wide producer."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (Path(output_root)
                   / "26_latent_body_tracks/out/latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "ephemeral-alias retirement evidence is missing: "
            + str(points_path))
    points = pd.read_csv(points_path)
    if enabled:
        audit, internal, _attached = retirement.discover(
            labels, points, params)
        candidate, applications, details = retirement.apply(labels, internal)
    else:
        audit = pd.DataFrame(columns=retirement.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=retirement.APPLICATION_COLUMNS)
        candidate = labels.copy()
        details = {
            "applied_proposals": 0, "changed_pixels": 0,
            "changed_frames": 0, "new_identity_count": 0,
            "removed_identity_count": 0,
            "new_explained_projection_components": 0,
            "new_duplicate_components": 0,
        }
    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError("ephemeral-alias retirement changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("ephemeral-alias retirement overlaps unclaimed")
    if int(details["new_duplicate_components"]):
        raise AssertionError(
            "ephemeral-alias retirement created unexplained duplicates")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    removed = before_ids - after_ids
    new_ids = after_ids - before_ids
    proved = set(audit[audit.eligible.astype(bool)].alias_owner.astype(int)) \
        if len(audit) else set()
    fully_applied = set(applications[
        applications.applied.astype(bool)].alias_owner.astype(int)) \
        if len(applications) else set()
    proof_exact = bool(not new_ids and removed == proved == fully_applied)
    if not proof_exact:
        raise AssertionError(
            "retired identities do not exactly match complete producer proof")

    audit.to_csv(
        stage / "ephemeral_speckle_alias_retirement_audit.csv", index=False)
    applications.to_csv(
        stage / "ephemeral_speckle_alias_retirement_applications.csv",
        index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "owners", "identities", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "identities_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        **details,
        "retired_aliases": sorted(map(int, removed)),
        "producer_proved_aliases": sorted(map(int, proved)),
        "fully_applied_aliases": sorted(map(int, fully_applied)),
        "retirement_proof_exact": proof_exact,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, applications, metrics
