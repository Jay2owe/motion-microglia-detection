"""Production adapter for recurrent dominant-body owner relays."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import owner_consensus
import recurrent_dominant_body_relay as relay


CONFIG_KEY = "field_wide_recurrent_dominant_body_relay"
STAGE_NAME = "114_recurrent_dominant_body_relay"
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
            "recurrent dominant-body relay production configuration "
            "contains unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("dominant-body relay must use field-wide discovery")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "mode": "candidate",
        "targeting_mode": "field_wide_discovery",
        # The reusable core validates its standalone file contract even when
        # called in-process. These sentinels are never opened by ``produce``.
        "labels_path": "current-run labels array",
        "unclaimed_path": "current-run unclaimed array",
    })
    relay.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def _closed_recurrent_exchange_proof(internal: dict[str, Any]) -> bool:
    """Accept only a repeatedly observed, closed three-owner exchange.

    A single trajectory that visits many unrelated names can be a crowding
    ambiguity rather than evidence that one physical body retained its owner.
    Production therefore requires the established owner to reclaim the body
    after entry and exactly two foreign owners to recur independently. This is
    categorical complete-field evidence, not an identity or location selector.
    """
    resident = int(internal["resident_owner"])
    observed = [
        int(row["observed_owner"]) for row in internal["trajectory"]]
    foreign = [owner for owner in observed if owner != resident]
    foreign_owners = set(foreign)
    foreign_runs: list[int] = []
    for owner in foreign:
        if not foreign_runs or foreign_runs[-1] != owner:
            foreign_runs.append(owner)
    run_counts = {
        owner: foreign_runs.count(owner) for owner in foreign_owners}
    return bool(
        resident in observed
        and len(foreign_owners) == 2
        and min(run_counts.values(), default=0) >= 2)


def _produce_with_closed_exchange_proof(
        labels: np.ndarray, unclaimed: np.ndarray, params: dict[str, Any],
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    audit, internals = relay.discover(labels, params)
    rejected = 0
    for index, internal in enumerate(internals):
        if (bool(internal["eligible"])
                and not _closed_recurrent_exchange_proof(internal)):
            audit.loc[index, "eligible"] = False
            audit.loc[index, "reason"] = \
                "rejected_without_closed_recurrent_exchange_proof"
            internal["eligible"] = False
            internal["reason"] = \
                "rejected_without_closed_recurrent_exchange_proof"
            rejected += 1
    candidate, audit, frames = relay.apply(labels, audit, internals)

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("dominant-body relay changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("dominant-body relay changed identity set")
    changed = candidate != labels
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    new_duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    metrics = {
        "mode": "accepted_label_reconciliation",
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_proposals": int(len(applied)),
        "closed_exchange_rejections": int(rejected),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "frame_object_bijection_preserved": bool(new_duplicates == 0),
        "new_duplicate_components": int(new_duplicates),
    }
    return candidate, unclaimed.copy(), audit, frames, metrics


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the reviewed target-free rule to current-run label arrays."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, candidate_unclaimed, audit, frames, details = \
            _produce_with_closed_exchange_proof(labels, unclaimed, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=relay.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=relay.FRAME_COLUMNS)
        details = {
            "proposals_audited": 0, "eligible_proposals": 0,
            "applied_proposals": 0, "closed_exchange_rejections": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "foreground_ledger_exact": True,
            "preexisting_unclaimed_exact": True,
            "identity_set_exact": True, "new_identity_count": 0,
            "removed_identity_count": 0,
            "frame_object_bijection_preserved": True,
            "new_duplicate_components": 0,
        }

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError("dominant-body relay changed foreground")
    if not np.array_equal(candidate_unclaimed, before_unclaimed):
        raise AssertionError("dominant-body relay changed unclaimed data")
    if (set(map(int, np.unique(candidate))) - {0}) != \
            (set(map(int, np.unique(before))) - {0}):
        raise AssertionError("dominant-body relay changed identity set")

    audit.to_csv(stage / "recurrent_dominant_body_relay_audit.csv",
                 index=False)
    frames.to_csv(stage / "recurrent_dominant_body_relay_frames.csv",
                  index=False)
    configured = config_values["accepted_postprocessing"].get(CONFIG_KEY, {})
    metrics = {
        **details,
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": "accepted_label_reconciliation",
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, frames, metrics
