"""Production adapter for terminal projection-diversion recovery.

The field-wide detector repairs an end-of-recording three-seat diversion only
when the original physical track remains visible, the stolen identity lands on
a newly born reference, and the resulting neighbouring pair has repeated raw
two-core evidence.  The extra reference is retained as a projection of the
established resident; the adapter never allocates a new biological identity.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import owner_consensus
import terminal_projection_diversion as diversion


CONFIG_KEY = "field_wide_terminal_projection_diversion"
STAGE_NAME = "72_terminal_projection_diversion"
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
            "terminal projection-diversion production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "terminal projection-diversion production must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    diversion.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


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
    points_path = (
        output_root / "26_latent_body_tracks/out/latent_track_points.csv")

    if enabled:
        if not points_path.is_file():
            raise FileNotFoundError(
                "terminal projection-diversion recovery requires current-run "
                f"physical evidence: {points_path}")
        points = pd.read_csv(points_path)
        candidate, proposals, applications, details = \
            diversion.discover_and_apply(labels, raw, points, params)
    else:
        points = pd.DataFrame(columns=["track_id"])
        candidate = labels.copy()
        proposals = pd.DataFrame()
        applications = pd.DataFrame()
        details = {
            "proposals_audited": 0, "eligible_proposals": 0,
            "applied_proposals": 0, "changed_pixels": 0,
            "changed_frames": 0, "new_identity_count": 0,
            "new_duplicate_components": 0,
            "new_explained_projection_components": 0,
        }

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError(
            "terminal projection-diversion stage changed foreground")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError(
            "terminal projection-diversion stage changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "terminal projection-diversion stage overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError(
            "terminal projection-diversion stage changed identity set")
    unexplained_duplicates = int(details["new_duplicate_components"])
    if unexplained_duplicates:
        raise AssertionError(
            "terminal projection-diversion stage created unexplained duplicates")
    # Keep the established production-wide count as an independent guard.  The
    # only permitted excess components are the projection lobes proved by this
    # stage's own complete-field physical evidence.
    raw_duplicate_excess = owner_consensus.count_new_duplicate_components(
        before, candidate)

    proposals.to_csv(stage / "terminal_diversion_triples.csv", index=False)
    applications.to_csv(
        stage / "terminal_diversion_applications.csv", index=False)
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
