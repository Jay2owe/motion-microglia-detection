"""Production adapter for producer-ledger projection-event calibration."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import explained_projection_event_calibration as calibration


CONFIG_KEY = "field_wide_explained_projection_event_calibration"
STAGE_NAME = "98_explained_projection_event_calibration"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "score_only",
    "targeting_mode", "reviewed_event_catalogue_sha256", "parameters",
}


def _configured(values: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(configured) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(
            "explained-projection production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "explained-projection production must be field-wide")
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


def run(labels: np.ndarray, unclaimed: np.ndarray,
        events: pd.DataFrame, members: pd.DataFrame,
        applications: pd.DataFrame, config_values: dict[str, Any],
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Calibrate events while preserving both biological ledgers exactly."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    if enabled:
        calibrated, calibrated_members, audit, removed = calibration.calibrate(
            events, members, applications)
    else:
        calibrated, calibrated_members = events.copy(), members.copy()
        audit = pd.DataFrame(columns=calibration.AUDIT_COLUMNS)
        removed = []
    if not np.array_equal(labels, before):
        raise AssertionError("explained-projection calibration changed labels")
    if not np.array_equal(unclaimed, before_unclaimed):
        raise AssertionError(
            "explained-projection calibration changed unclaimed data")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": stage / "candidate_disruptive_events.csv",
        "members": stage / "candidate_event_members.csv",
        "audit": stage / "explained_projection_event_audit.csv",
        "metrics": stage / "producer_metrics.json",
    }
    calibrated.to_csv(paths["events"], index=False)
    calibrated_members.to_csv(paths["members"], index=False)
    audit.to_csv(paths["audit"], index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "events_audited": int(len(events)),
        "producer_proposals_audited": int(
            applications.proposal_id.astype(str).nunique())
            if len(applications) else 0,
        "events_removed_as_explained_projections": int(len(removed)),
        "removed_event_ids": removed,
        "labels_changed": False,
        "unclaimed_changed": False,
        "before": calibration._summary(events),
        "after": calibration._summary(calibrated),
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
