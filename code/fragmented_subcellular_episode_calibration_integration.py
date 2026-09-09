"""Production adapter for fragmented subcellular-reference calibration."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import fragmented_subcellular_episode_calibration as calibration


CONFIG_KEY = "field_wide_fragmented_subcellular_episode_calibration"
STAGE_NAME = "108_fragmented_subcellular_episode_calibration"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "score_only",
    "targeting_mode", "reviewed_event_catalogue_sha256", "parameters",
}


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "fragmented subcellular production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "fragmented subcellular calibration must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    calibration.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        raw: np.ndarray, thresholds: pd.DataFrame, events: pd.DataFrame,
        members: pd.DataFrame, config_values: dict[str, Any],
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove only field-proved subcellular reference alerts."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    if enabled:
        calibrated, calibrated_members, audit, summary = calibration.calibrate(
            events, members, points, raw, thresholds, params)
    else:
        calibrated, calibrated_members = events.copy(), members.copy()
        audit = pd.DataFrame()
        summary = {
            "targeting_mode": "field_wide_discovery",
            "target_counts": {key: 0 for key in (
                "identities", "owners", "tracks", "frames", "coordinates",
                "events", "regions", "wells", "review_cases")},
            "events_audited": int(len(events)),
            "events_calibrated_nonbiological": 0,
            "removed_event_ids": [],
            "before": calibration.event_summary(events),
            "after": calibration.event_summary(events),
        }
    if not np.array_equal(labels, before):
        raise AssertionError(
            "fragmented subcellular calibration changed labels")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError(
            "fragmented subcellular calibration changed unclaimed data")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": stage / "candidate_disruptive_events.csv",
        "members": stage / "candidate_event_members.csv",
        "audit": stage / "fragmented_subcellular_episode_audit.csv",
        "metrics": stage / "producer_metrics.json",
    }
    calibrated.to_csv(paths["events"], index=False)
    calibrated_members.to_csv(paths["members"], index=False)
    audit.to_csv(paths["audit"], index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **summary,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_score_calibration",
        "score_only": True,
        "label_array_exact": True,
        "unclaimed_array_exact": True,
        "event_catalogue_sha256": _sha256(paths["events"]),
        "event_members_sha256": _sha256(paths["members"]),
        "audit_sha256": _sha256(paths["audit"]),
        "parameters": params,
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (before, before_unclaimed, calibrated, calibrated_members,
            audit, metrics)
