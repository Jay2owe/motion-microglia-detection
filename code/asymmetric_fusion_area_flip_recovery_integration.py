"""Production adapter for field-wide asymmetric fusion and area-flip repair."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import m1b_asymmetric_fusion_area_flip_recovery as recovery


CONFIG_KEY = "field_wide_asymmetric_fusion_area_flip_recovery"
STAGE_NAME = "113_asymmetric_fusion_area_flip_recovery"
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
            "asymmetric fusion/area-flip production configuration contains "
            "unsupported keys (selectors are forbidden): "
            + ", ".join(unexpected))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "asymmetric fusion/area-flip recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params.update({
        "mode": "candidate",
        "targeting_mode": "field_wide_discovery",
        # The reusable core validates its standalone file-input contract even
        # when called in-process. These runtime sentinels are not opened by
        # ``produce`` and avoid embedding any historical artifact path.
        "labels_path": "current-run labels array",
        "unclaimed_path": "current-run unclaimed array",
    })
    recovery.assert_target_free(params)
    return bool(configured.get("enabled", False)), params


def _produce_with_atomic_pair_proof(
        labels: np.ndarray, unclaimed: np.ndarray, params: dict[str, Any],
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Require an area-flip pair to have an applied short-fusion proof.

    Area alone is not sufficient evidence that two separately visible cells
    exchanged identities: morphology can change reciprocally for long periods.
    The production rule therefore treats the two reviewed mechanisms as one
    atomic owner-pair proof. This remains complete-field and identity-blind.
    """
    audit = recovery.discover(labels, params)
    fusion_trial = audit.copy()
    area_rows = fusion_trial.proposal_kind.eq("reciprocal_area_flip") \
        & fusion_trial.eligible.astype(bool)
    fusion_trial.loc[area_rows, "eligible"] = False
    fusion_trial.loc[area_rows, "reason"] = "held_for_pair_fusion_proof"
    _, fusion_trial, _ = recovery.apply(labels, fusion_trial, params)
    applied_fusions = fusion_trial[
        fusion_trial.applied.astype(bool)
        & fusion_trial.proposal_kind.eq("short_asymmetric_fusion")]
    proved_pairs = {
        tuple(sorted((int(row.owner_a), int(row.owner_b))))
        for row in applied_fusions.itertuples(index=False)
    }
    unpaired_rejections = 0
    for index, row in audit[
            audit.eligible.astype(bool)
            & audit.proposal_kind.eq("reciprocal_area_flip")].iterrows():
        pair = tuple(sorted((int(row.owner_a), int(row.owner_b))))
        if pair not in proved_pairs:
            audit.loc[index, "eligible"] = False
            audit.loc[index, "reason"] = \
                "area_flip_without_applied_pair_fusion_proof"
            unpaired_rejections += 1
    candidate, audit, frames = recovery.apply(labels, audit, params)

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("fusion/area-flip recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("fusion/area-flip recovery changed identity set")
    strict_duplicates = recovery.owner_consensus.count_new_duplicate_components(
        labels, candidate)
    unproven = recovery.fusion._unproven_duplicate_components(
        labels, candidate,
        float(params.get("maximum_secondary_component_area_fraction", 0.15)))
    if unproven:
        raise AssertionError("fusion/area-flip recovery created duplicates")
    changed = candidate != labels
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_proposals": int(len(applied)),
        "applied_fusion_restorations": int(
            applied.proposal_kind.eq("short_asymmetric_fusion").sum())
            if len(applied) else 0,
        "applied_area_flips": int(
            applied.proposal_kind.eq("reciprocal_area_flip").sum())
            if len(applied) else 0,
        "unpaired_area_flips_rejected": int(unpaired_rejections),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "new_duplicate_components": int(strict_duplicates),
        "projection_proven_new_duplicate_components": int(
            strict_duplicates - unproven),
        "unproven_new_duplicate_components": int(unproven),
    }
    return candidate, unclaimed.copy(), audit, frames, metrics


def run(labels: np.ndarray, unclaimed: np.ndarray,
        config_values: dict[str, Any], output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Apply the reviewed general rule to current-run label arrays."""
    enabled, params = _configured(config_values)
    before = labels.copy()
    before_unclaimed = unclaimed.copy()
    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, candidate_unclaimed, audit, frames, details = \
            _produce_with_atomic_pair_proof(labels, unclaimed, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=recovery.AUDIT_COLUMNS)
        frames = pd.DataFrame(columns=recovery.FRAME_COLUMNS)
        details = {
            "proposals_audited": 0, "eligible_proposals": 0,
            "applied_proposals": 0, "applied_fusion_restorations": 0,
            "applied_area_flips": 0, "changed_pixels": 0,
            "changed_frames": 0, "foreground_ledger_exact": True,
            "preexisting_unclaimed_exact": True, "identity_set_exact": True,
            "new_identity_count": 0, "removed_identity_count": 0,
            "new_duplicate_components": 0,
            "projection_proven_new_duplicate_components": 0,
            "unproven_new_duplicate_components": 0,
            "unpaired_area_flips_rejected": 0,
        }

    if not np.array_equal(candidate > 0, before > 0):
        raise AssertionError(
            "asymmetric fusion/area-flip recovery changed foreground")
    if not np.array_equal(candidate_unclaimed, before_unclaimed):
        raise AssertionError(
            "asymmetric fusion/area-flip recovery changed unclaimed data")
    before_ids = set(map(int, np.unique(before))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError(
            "asymmetric fusion/area-flip recovery changed identity set")
    if int(details.get("new_duplicate_components", 0)):
        raise AssertionError(
            "asymmetric fusion/area-flip recovery created duplicates")

    audit.to_csv(stage / "fusion_area_flip_audit.csv", index=False)
    frames.to_csv(stage / "fusion_area_flip_frames.csv", index=False)
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
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, frames, metrics
