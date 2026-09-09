"""Production adapter for ownerless-cohort latent-gap completion."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ownerless_cohort_latent_gap_completion as completion


CONFIG_KEY = "field_wide_ownerless_cohort_latent_gap_completion"
STAGE_NAME = "80_ownerless_cohort_latent_gap_completion"
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
            "ownerless-cohort gap production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "ownerless-cohort gap production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    completion.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Apply the accepted field-wide rule using current-run evidence."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    root = Path(output_root)
    stage = root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = root / "26_latent_body_tracks/out/latent_track_points.csv"
    cohorts_path = root / "34_ownerless_cohort_recovery/out/producer_audit.csv"
    thresholds_path = (
        root / "25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv")
    if enabled:
        missing = [path for path in (points_path, cohorts_path, thresholds_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "ownerless-cohort gap completion requires current-run "
                "evidence: " + ", ".join(map(str, missing)))
        points = pd.read_csv(points_path)
        cohorts = pd.read_csv(cohorts_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, audit, details = completion.complete(
            labels, unclaimed, raw, points, cohorts, thresholds, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame()
        details = {
            "maximum_gap_frames": max(1, int(np.ceil(
                len(labels) * float(params.get(
                    "maximum_gap_movie_fraction", 0.04))))),
            "cohort_identities_discovered": 0,
            "proposals_audited": 0, "applied_proposals": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "changed_identities": 0,
            "preexisting_assigned_changed_pixels": 0,
            "unclaimed_overlap_pixels": 0,
            "zero_signal_additions": 0,
            "new_identity_count": 0, "removed_identity_count": 0,
            "new_duplicate_components": 0,
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "review_cases")},
        }
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("ownerless-cohort gap completion overlaps ledgers")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError("ownerless-cohort gap completion changed unclaimed")
    if np.any((before > 0) & (candidate != before)):
        raise AssertionError(
            "ownerless-cohort gap completion changed assigned pixels")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("ownerless-cohort gap completion changed identity set")

    audit.to_csv(stage / "ownerless_cohort_latent_gap_audit.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **details,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "parameter_keys_audited": len(params),
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, before_unclaimed, audit, metrics
