"""Production adapter for complete foreground-conservative ownership episodes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import conservative_episode_ownership as ownership

CONFIG_KEY = "field_wide_conservative_episode_ownership"
STAGE_NAME = "120_conservative_episode_ownership"
TOP_LEVEL_KEYS = {
    "version", "enabled", "enabled_by_default", "targeting_mode",
    "reviewed_candidate_labels_sha256", "reviewed_candidate_unclaimed_sha256",
    "parameters",
}


def configured(values):
    config = values["accepted_postprocessing"].get(CONFIG_KEY, {})
    unexpected = sorted(set(config) - TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError("Unsupported conservative ownership configuration: " + ", ".join(unexpected))
    if config.get("targeting_mode", "field_wide_discovery") != "field_wide_discovery":
        raise ValueError("Conservative ownership must be field-wide")
    params = deepcopy(config.get("parameters", {}))
    params.update(targeting_mode="field_wide_discovery", apply_recovery=True,
                  labels_path="current-run labels array",
                  unclaimed_path="current-run unclaimed array",
                  raw_path="current-run raw array",
                  physical_track_points_path="current-run physical points table")
    ownership.check_params(params)
    return bool(config.get("enabled", False)), params


def run(labels, unclaimed, raw, points, config_values, output_root):
    enabled, params = configured(config_values)
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, ledger, audit, frames, allocations, metrics = ownership.produce(
            labels, unclaimed, raw, points, params)
    else:
        candidate, ledger = labels.copy(), unclaimed.copy()
        audit, frames, allocations = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        metrics = dict(targeting_mode="field_wide_discovery",
                       changed_label_pixels=0, changed_frames=0,
                       suppressed_foreground_pixels=0, added_foreground_pixels=0,
                       named_foreground_preserved=True, new_duplicate_components=0,
                       new_identities=[], allocations=[])
    metrics = {**metrics,
               "applied_proposals": int(frames.loc[frames.applied.astype(bool), "proposal"].nunique())
                   if len(frames) else 0,
               "version": config_values["accepted_postprocessing"].get(CONFIG_KEY, {}).get("version"),
               "enabled": enabled, "parameters": params,
               "mode": "accepted_label_reconciliation"}
    audit.to_csv(stage / "discovery_audit.csv", index=False)
    frames.to_csv(stage / "episode_frames.csv", index=False)
    allocations.to_csv(stage / "displaced_body_allocations.csv", index=False)
    (stage / "producer_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, ledger, audit, frames, allocations, metrics
