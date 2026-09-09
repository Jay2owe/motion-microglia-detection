"""Production adapter for delayed projection-owner reclaim.

The accepted field-wide detector separates an asymmetric two-core overlap when
one durable lineage ends and the continuing lineage shortly reclaims an
already-established nearby owner.  It consumes only current-run physical and
raw evidence, preserves foreground and the identity set, and allocates no new
biological identity.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import delayed_projection_reclaim as reclaim
import owner_consensus


CONFIG_KEY = "field_wide_delayed_projection_reclaim"
STAGE_NAME = "73_delayed_projection_reclaim"
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
            "delayed projection-reclaim production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "delayed projection-reclaim production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    reclaim.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def _empty_details() -> dict[str, int]:
    return {
        "proposals_audited": 0, "eligible_proposals": 0,
        "applied_proposals": 0, "changed_pixels": 0,
        "changed_frames": 0, "new_identity_count": 0,
        "new_duplicate_components": 0,
        "new_explained_projection_components": 0,
    }


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the accepted target-free rule using current-run evidence."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    output_root = Path(output_root)
    stage = output_root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    physical_stage = output_root / "26_latent_body_tracks/out"
    points_path = physical_stage / "latent_track_points.csv"
    encounters_path = physical_stage / "encounter_frames.csv"

    if enabled:
        missing = [path for path in (points_path, encounters_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "delayed projection-reclaim recovery requires current-run "
                "physical evidence: " + ", ".join(map(str, missing)))
        points = pd.read_csv(points_path)
        encounters = pd.read_csv(encounters_path)
        proposals, scored = reclaim.discover(
            labels, raw, points, encounters, params)
        candidate, applications, details = reclaim.apply(
            labels, raw, proposals, scored, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        proposals = pd.DataFrame()
        applications = pd.DataFrame()
        candidate = labels.copy()
        details = _empty_details()

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError("delayed projection-reclaim changed foreground")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError("delayed projection-reclaim changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "delayed projection-reclaim overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("delayed projection-reclaim changed identity set")
    if int(details["new_duplicate_components"]):
        raise AssertionError(
            "delayed projection-reclaim created unexplained duplicates")
    raw_duplicate_excess = owner_consensus.count_new_duplicate_components(
        before, candidate)

    proposals.to_csv(stage / "delayed_projection_reclaim_audit.csv", index=False)
    applications.to_csv(
        stage / "delayed_projection_reclaim_applications.csv", index=False)
    changed = candidate != before
    changed_frames = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1)).astype(int).tolist()
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "owners", "identities", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "parameter_keys_audited": len(params),
        "physical_tracks_audited": int(points.track_id.nunique())
            if "track_id" in points else 0,
        **details,
        "changed_frame_indices": changed_frames,
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "removed_identity_count": len(before_ids - after_ids),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "raw_new_component_excess": int(raw_duplicate_excess),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, before_unclaimed, proposals, applications, metrics
