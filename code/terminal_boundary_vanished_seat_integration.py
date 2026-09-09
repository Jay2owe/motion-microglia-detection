"""Production adapter for a proved terminal vanished-seat partition."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import owner_consensus
import terminal_boundary_vanished_seat as partition


CONFIG_KEY = "field_wide_terminal_boundary_vanished_seat_partition"
STAGE_NAME = "106_terminal_boundary_vanished_seat_partition"
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
            "terminal boundary partition configuration contains unsupported "
            "keys (selectors are forbidden): " + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("terminal boundary partition must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    partition.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def run(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Partition only complete-field, right-censored vanished seats."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    root = Path(output_root)
    stage = root / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = root / "26_latent_body_tracks/out/latent_track_points.csv"
    thresholds_path = (
        root / "25_raw_physical_hypotheses/out/frame_evidence_thresholds.csv")
    if enabled:
        missing = [path for path in (points_path, thresholds_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "terminal boundary partition requires current-run evidence: "
                + ", ".join(map(str, missing)))
        candidate, audit, frames, details = partition.discover_and_apply(
            labels, unclaimed, raw, pd.read_csv(points_path),
            pd.read_csv(thresholds_path), params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=partition.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=partition.FRAME_COLUMNS)
        details = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "terminal_disappearances_audited": 0,
            "eligible_proposals": 0, "applied_proposals": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_exact": True, "unclaimed_exact": True,
            "identity_set_exact": True, "new_duplicate_components": 0,
        }

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError("terminal boundary partition changed foreground")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError("terminal boundary partition changed unclaimed data")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("terminal boundary partition overlaps unclaimed")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(before))):
        raise AssertionError("terminal boundary partition changed identity set")
    duplicates = int(owner_consensus.count_new_duplicate_components(
        before, candidate))
    if duplicates:
        raise AssertionError("terminal boundary partition created duplicates")

    audit.to_csv(stage / "terminal_boundary_seat_audit.csv", index=False)
    frames.to_csv(stage / "terminal_boundary_seat_frames.csv", index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **details,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "foreground_exact": True, "unclaimed_exact": True,
        "identity_set_exact": True,
        "new_duplicate_components": duplicates,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, before_unclaimed, audit, frames, metrics

