"""Preserve a takeover body and reciprocally name its separate counterpart.

Discovery remains complete-field and identity-blind. A proven dormant-owner
takeover keeps the established owner along its complete continuous component
chain. If that owner later appears on one durable, spatially separate body, the
two components are treated as an atomic reciprocal partition: the continuous
chain keeps the established owner and the separate body receives the displaced
incoming owner.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile

import component_accounting
import dormant_owner_separable_reappearance as evidence


ALLOWED_KEYS = evidence.ALLOWED_KEYS
AUDIT_COLUMNS = evidence.AUDIT_COLUMNS + [
    "reciprocal_partition_mode",
    "pre_reappearance_source_frames",
    "paired_reappearance_frames",
    "maximum_source_components_after_reappearance",
    "maximum_incoming_components_on_terminal_chain",
]
APPLICATION_COLUMNS = [
    "proposal_id", "frame", "source_owner", "incoming_owner", "role",
    "component_pixels", "changed_pixels", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    evidence.assert_target_free(params)


def _owner_components_by_frame(components: dict, owner: int) -> dict:
    result: dict[int, list] = {}
    for component in components.values():
        if component.owner == int(owner):
            result.setdefault(component.frame, []).append(component)
    return result


def discover(labels: np.ndarray, params: dict):
    """Find safe complete or reciprocal recovery for every takeover."""
    assert_target_free(params)
    audit, internal, indexed, components = evidence.discover(labels, params)
    public = audit.copy()
    detailed = internal.copy()
    additions = []
    final_eligible = []
    final_reasons = []
    for proposal in detailed.itertuples(index=False):
        onset = int(proposal.separable_reappearance_frame)
        terminal = [components[int(key)]
                    for key in proposal.terminal_chain]
        terminal_frames = {component.frame for component in terminal}
        source_by_frame = _owner_components_by_frame(
            components, int(proposal.source_owner))
        incoming_by_frame = _owner_components_by_frame(
            components, int(proposal.incoming_owner))
        pre_source = [
            frame for frame in terminal_frames
            if onset >= 0 and frame < onset and source_by_frame.get(frame)
        ]
        paired = [
            frame for frame in terminal_frames
            if onset >= 0 and frame >= onset and source_by_frame.get(frame)
        ]
        maximum_source = max(
            (len(source_by_frame.get(frame, [])) for frame in paired),
            default=0)
        maximum_incoming = max(
            (len(incoming_by_frame.get(frame, []))
             for frame in terminal_frames), default=0)
        reasons = []
        if bool(proposal.eligible) and onset >= 0:
            if pre_source:
                reasons.append("source_present_before_durable_reappearance")
            if not paired:
                reasons.append("no_paired_reappearance_frames")
            if maximum_source != 1:
                reasons.append("parallel_source_partition_not_unique")
            if maximum_incoming != 1:
                reasons.append("terminal_incoming_partition_not_unique")
        accepted = bool(proposal.eligible) and not reasons
        if not bool(proposal.eligible):
            reason = str(proposal.reason)
        elif reasons:
            reason = "|".join(reasons)
        elif onset >= 0:
            reason = "eligible_reciprocal_separable_takeover_partition"
        else:
            reason = "eligible_complete_dormant_takeover_recovery"
        additions.append({
            "reciprocal_partition_mode": onset >= 0,
            "pre_reappearance_source_frames": len(pre_source),
            "paired_reappearance_frames": len(paired),
            "maximum_source_components_after_reappearance": maximum_source,
            "maximum_incoming_components_on_terminal_chain": maximum_incoming,
        })
        final_eligible.append(accepted)
        final_reasons.append(reason)
    addition_table = pd.DataFrame(additions)
    for column in AUDIT_COLUMNS[len(evidence.AUDIT_COLUMNS):]:
        values = addition_table[column] if len(addition_table) else []
        public[column] = values
        detailed[column] = values
    if len(public):
        public["eligible"] = final_eligible
        detailed["eligible"] = final_eligible
        public["reason"] = final_reasons
        detailed["reason"] = final_reasons
    return public[AUDIT_COLUMNS], detailed, indexed, components


def apply(labels: np.ndarray, internal: pd.DataFrame, indexed: np.ndarray,
          components: dict, enabled: bool):
    """Apply each selected two-owner partition as one validated transaction."""
    candidate = labels.copy()
    occupied = np.zeros(labels.shape, bool)
    rows = []
    eligible = internal[internal.eligible.astype(bool)] if len(internal) \
        else internal
    for proposal in eligible.itertuples(index=False):
        source_owner = int(proposal.source_owner)
        incoming_owner = int(proposal.incoming_owner)
        onset = int(proposal.separable_reappearance_frame)
        terminal = [components[int(key)] for key in proposal.terminal_chain]
        terminal_frames = {component.frame for component in terminal}
        source_by_frame = _owner_components_by_frame(components, source_owner)
        separate = [
            component
            for frame in sorted(terminal_frames)
            if onset >= 0 and frame >= onset
            for component in source_by_frame.get(frame, [])
        ]
        pending = []
        refused = ""
        for component in terminal:
            mask = indexed[component.frame] == component.key
            if np.any(occupied[component.frame] & mask):
                refused = "overlapping_eligible_proposal"
                break
            if not np.all(labels[component.frame][mask] == incoming_owner):
                refused = "incoming_component_provenance_changed"
                break
            pending.append((component, mask, source_owner,
                            "continuous_takeover_body"))
        if not refused:
            for component in separate:
                mask = indexed[component.frame] == component.key
                if np.any(occupied[component.frame] & mask):
                    refused = "overlapping_eligible_proposal"
                    break
                if not np.all(labels[component.frame][mask] == source_owner):
                    refused = "separate_component_provenance_changed"
                    break
                pending.append((component, mask, incoming_owner,
                                "separate_reappearance_body"))
        if refused:
            rows.append({
                "proposal_id": proposal.proposal_id, "frame": -1,
                "source_owner": source_owner,
                "incoming_owner": incoming_owner, "role": "transaction",
                "component_pixels": 0, "changed_pixels": 0,
                "applied": False, "reason": refused,
            })
            continue
        for component, mask, new_owner, role in pending:
            changed = int(mask.sum()) if enabled else 0
            if enabled:
                candidate[component.frame][mask] = new_owner
                occupied[component.frame] |= mask
            rows.append({
                "proposal_id": proposal.proposal_id,
                "frame": component.frame,
                "source_owner": source_owner,
                "incoming_owner": incoming_owner, "role": role,
                "component_pixels": component.area,
                "changed_pixels": changed, "applied": bool(enabled),
                "reason": ("reciprocal_separable_takeover_partition"
                           if onset >= 0 and enabled
                           else "complete_dormant_takeover_recovery"
                           if enabled else "audit_only"),
            })
    return candidate, pd.DataFrame(rows, columns=APPLICATION_COLUMNS)


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict):
    del points
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")
    audit, internal, indexed, components = discover(labels, params)
    candidate, applications = apply(
        labels, internal, indexed, components,
        bool(params.get("apply_recovery", True)))
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reciprocal partition changed assigned foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("reciprocal partition overlapped unclaimed ledger")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("reciprocal partition changed the identity set")
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    new_excess = component_accounting.new_duplicate_components(
        labels, candidate)
    reciprocal = audit[
        audit.eligible.astype(bool)
        & audit.reciprocal_partition_mode.astype(bool)] if len(audit) else audit
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "components_audited": int(len(components)),
        "owner_transitions_audited": int(len(audit)),
        "eligible_takeovers": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_takeovers": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "reciprocal_partitions": int(len(reciprocal)),
        "reciprocal_pair_frames": int(
            reciprocal.paired_reappearance_frames.astype(int).sum())
            if len(reciprocal) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_exact": True,
        "unclaimed_ledger_exact": True,
        "movie_identity_set_exact": True,
        "new_component_excess": int(new_excess),
        "new_unexplained_component_excess": int(new_excess),
        "partition_explanation":
            "continuous_takeover_body_plus_unique_durable_separate_body",
    }
    return candidate, unclaimed.copy(), audit, applications, summary


def run(_upstream: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, candidate_unclaimed, audit, applications, summary = produce(
        labels, unclaimed, raw, points, params)
    output = Path(out.out)
    output.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": output / f"{stem}.tif",
        "unclaimed": output / f"{stem}_unclaimed_original_ids.tif",
        "audit": output / "reciprocal_separable_partition_audit.csv",
        "applications": output /
            "reciprocal_separable_partition_applications.csv",
        "metrics": output / "producer_metrics.json",
    }
    tifffile.imwrite(
        outputs["labels"], candidate, imagej=True, compression="zlib",
        photometric="minisblack", metadata={
            "axes": "TYX", "finterval": 1800.0, "tunit": "sec",
            "unit": "pixel"})
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    audit.to_csv(outputs["audit"], index=False)
    applications.to_csv(outputs["applications"], index=False)
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}
