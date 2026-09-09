"""Repair a short fusion owner loss followed by a reciprocal area flip.

This field-wide producer combines two independent geometric proofs:

* during a short fusion, a small neighbour's mask jumps to the scale of a
  nearby lost established body, so only the fusion interval is restored to the
  lost owner; and
* two locally adjacent owners exchange large- and small-body areas between
  adjacent frames, after which the name on the large body persists, so their
  complete suffix is reciprocally exchanged.

The proofs use movie-relative support and dimensionless ratios only. Identity,
track, frame, coordinate, event, region, well, and review selectors are absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile

import m1_asymmetric_fusion_suffix_exchange as fusion
import owner_consensus


CONTROL_KEYS = {"mode", "targeting_mode", "output_stem"}
PATH_KEYS = {"labels_path", "unclaimed_path"}
AREA_KEYS = {
    "minimum_area_flip_prior_support_movie_fraction",
    "minimum_area_flip_suffix_support_movie_fraction",
    "minimum_area_flip_body_ratio",
    "minimum_area_flip_cost_margin",
    "maximum_area_flip_cross_cost",
    "maximum_area_flip_separation_sum_radii",
}
ALLOWED_KEYS = fusion.ALLOWED_KEYS | AREA_KEYS
AUDIT_COLUMNS = [
    "proposal_id", "proposal_kind", "owner_a", "owner_b",
    "first_frame", "last_frame", "prior_support_a", "prior_support_b",
    "suffix_support", "prior_area_a", "prior_area_b", "current_area_a",
    "current_area_b", "prior_body_ratio", "current_body_ratio",
    "same_area_cost", "cross_area_cost", "area_cost_margin",
    "prior_separation_sum_radii", "current_separation_sum_radii",
    "changed_pixels", "changed_frames", "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "proposal_kind", "frame", "operation", "owner_a",
    "owner_b", "changed_pixels",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("fusion/area-flip recovery must be field-wide")
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported parameters (selectors are forbidden): "
            + ", ".join(unsupported))
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing input paths: " + ", ".join(missing))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _body(frame: np.ndarray, owner: int,
          secondary_fraction: float) -> np.ndarray | None:
    return fusion._body_component(frame, owner, secondary_fraction)


def _distance_in_sum_radii(left: np.ndarray, right: np.ndarray) -> float:
    distance = float(np.linalg.norm(
        fusion._centroid(left) - fusion._centroid(right)))
    return distance / max(fusion._radius(left) + fusion._radius(right), 1.0)


def _fusion_proposals(labels: np.ndarray, params: dict) -> list[dict]:
    base_audit, internal = fusion.discover(labels, params)
    by_id = base_audit.set_index("proposal_id", drop=False)
    proposals: list[dict] = []
    for row in internal:
        if not bool(row["eligible"]):
            continue
        public = by_id.loc[str(row["proposal_id"])]
        if isinstance(public, pd.DataFrame):
            public = public.iloc[0]
        proposals.append({
            "proposal_id": f"FG{len(proposals) + 1:04d}",
            "proposal_kind": "short_asymmetric_fusion",
            "owner_a": int(row["small_survivor"]),
            "owner_b": int(row["lost_large_owner"]),
            "first_frame": int(row["fusion_start"]),
            "last_frame": int(row["split_frame"]) - 1,
            "prior_support_a": int(row["prior_support_small"]),
            "prior_support_b": int(row["prior_support_large"]),
            "suffix_support": int(row["postsplit_support_frames"]),
            "prior_area_a": float(row["prior_small_area"]),
            "prior_area_b": float(row["prior_large_area"]),
            "current_area_a": float(row["fused_area"]),
            "current_area_b": 0.0,
            "prior_body_ratio": float(row["large_to_small_area_ratio"]),
            "current_body_ratio": float(row["survivor_area_jump_ratio"]),
            "same_area_cost": 0.0, "cross_area_cost": 0.0,
            "area_cost_margin": 0.0,
            "prior_separation_sum_radii": float(
                row["prior_separation_sum_radii"]),
            "current_separation_sum_radii": 0.0,
            "changed_pixels": 0, "changed_frames": 0,
            "eligible": True, "applied": False,
            "reason": "eligible_short_asymmetric_fusion",
        })
    return proposals


def _area_flip_proposals(labels: np.ndarray, params: dict) -> list[dict]:
    frame_count = len(labels)
    presence = fusion._presence(labels)
    owners = sorted(presence)
    minimum_prior = max(3, math.ceil(frame_count * _fraction(
        params, "minimum_area_flip_prior_support_movie_fraction", 0.25)))
    minimum_suffix = max(2, math.ceil(frame_count * _fraction(
        params, "minimum_area_flip_suffix_support_movie_fraction", 0.10)))
    minimum_ratio = float(params.get("minimum_area_flip_body_ratio", 2.0))
    minimum_margin = float(params.get("minimum_area_flip_cost_margin", 0.75))
    maximum_cross = float(params.get("maximum_area_flip_cross_cost", 1.25))
    maximum_separation = float(params.get(
        "maximum_area_flip_separation_sum_radii", 3.0))
    secondary_fraction = float(params.get(
        "maximum_secondary_component_area_fraction", 0.15))
    proposals: list[dict] = []

    for frame_index in range(1, frame_count):
        common = [owner for owner in owners
                  if presence[owner][frame_index - 1]
                  and presence[owner][frame_index]]
        bodies: dict[tuple[int, int], np.ndarray] = {}
        for owner in common:
            prior = _body(labels[frame_index - 1], owner, secondary_fraction)
            current = _body(labels[frame_index], owner, secondary_fraction)
            if prior is not None and current is not None:
                bodies[(owner, 0)] = prior
                bodies[(owner, 1)] = current
        valid = [owner for owner in common if (owner, 0) in bodies]
        for left_index, left_owner in enumerate(valid):
            for right_owner in valid[left_index + 1:]:
                small_owner, large_owner = left_owner, right_owner
                prior_a = bodies[(small_owner, 0)]
                prior_b = bodies[(large_owner, 0)]
                current_a = bodies[(small_owner, 1)]
                current_b = bodies[(large_owner, 1)]
                pa, pb = float(prior_a.sum()), float(prior_b.sum())
                ca, cb = float(current_a.sum()), float(current_b.sum())
                if pa == pb or ca == cb:
                    continue
                # Keep small_owner as the previously small owner and
                # large_owner as the previously large owner.
                if pa > pb:
                    small_owner, large_owner = large_owner, small_owner
                    prior_a, prior_b = prior_b, prior_a
                    current_a, current_b = current_b, current_a
                    pa, pb, ca, cb = pb, pa, cb, ca
                if ca <= cb:
                    continue
                prior_ratio = pb / max(pa, 1.0)
                current_ratio = ca / max(cb, 1.0)
                same_cost = abs(math.log(ca / max(pa, 1.0))) + abs(
                    math.log(cb / max(pb, 1.0)))
                cross_cost = abs(math.log(ca / max(pb, 1.0))) + abs(
                    math.log(cb / max(pa, 1.0)))
                margin = same_cost - cross_cost
                prior_separation = _distance_in_sum_radii(prior_a, prior_b)
                current_separation = _distance_in_sum_radii(
                    current_a, current_b)
                prior_support_a = int(
                    presence[small_owner][:frame_index].sum())
                prior_support_b = int(
                    presence[large_owner][:frame_index].sum())
                suffix_support = int(
                    presence[small_owner][frame_index:].sum())
                reasons: list[str] = []
                if min(prior_support_a, prior_support_b) < minimum_prior:
                    reasons.append("insufficient_bilateral_prior_support")
                if suffix_support < minimum_suffix:
                    reasons.append("large_body_name_lacks_suffix_support")
                if min(prior_ratio, current_ratio) < minimum_ratio:
                    reasons.append("body_area_order_not_strongly_reversed")
                if margin < minimum_margin:
                    reasons.append("reciprocal_area_cost_not_preferred")
                if cross_cost > maximum_cross:
                    reasons.append("crossed_body_areas_not_conserved")
                if max(prior_separation, current_separation) > maximum_separation:
                    reasons.append("owner_bodies_not_local")
                proposals.append({
                    "proposal_id": f"AR{len(proposals) + 1:04d}",
                    "proposal_kind": "reciprocal_area_flip",
                    "owner_a": int(small_owner),
                    "owner_b": int(large_owner),
                    "first_frame": frame_index,
                    "last_frame": frame_count - 1,
                    "prior_support_a": prior_support_a,
                    "prior_support_b": prior_support_b,
                    "suffix_support": suffix_support,
                    "prior_area_a": pa, "prior_area_b": pb,
                    "current_area_a": ca, "current_area_b": cb,
                    "prior_body_ratio": prior_ratio,
                    "current_body_ratio": current_ratio,
                    "same_area_cost": same_cost,
                    "cross_area_cost": cross_cost,
                    "area_cost_margin": margin,
                    "prior_separation_sum_radii": prior_separation,
                    "current_separation_sum_radii": current_separation,
                    "changed_pixels": 0, "changed_frames": 0,
                    "eligible": not reasons, "applied": False,
                    "reason": "eligible_reciprocal_area_flip" if not reasons
                    else "|".join(reasons),
                })
    return proposals


def discover(labels: np.ndarray, params: dict) -> pd.DataFrame:
    proposals = _fusion_proposals(labels, params)
    proposals.extend(_area_flip_proposals(labels, params))
    return pd.DataFrame(proposals, columns=AUDIT_COLUMNS)


def apply(labels: np.ndarray, audit: pd.DataFrame, params: dict,
          ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate = labels.copy()
    secondary_fraction = float(params.get(
        "maximum_secondary_component_area_fraction", 0.15))
    eligible = audit[audit.eligible.astype(bool)].copy() if len(audit) else audit
    # Multiple proposals may share a pair only when fusion ends before the flip.
    frames: list[dict] = []
    for index, row in eligible.sort_values(
            ["first_frame", "proposal_kind"]).iterrows():
        owner_a, owner_b = int(row.owner_a), int(row.owner_b)
        trial = candidate.copy()
        if row.proposal_kind == "short_asymmetric_fusion":
            operation = "fusion_owner_restore"
            for frame in range(int(row.first_frame), int(row.last_frame) + 1):
                mask = trial[frame] == owner_a
                pixels = int(mask.sum())
                trial[frame][mask] = owner_b
                frames.append({
                    "proposal_id": row.proposal_id,
                    "proposal_kind": row.proposal_kind, "frame": frame,
                    "operation": operation, "owner_a": owner_a,
                    "owner_b": owner_b, "changed_pixels": pixels,
                })
        else:
            operation = "reciprocal_suffix_exchange"
            for frame in range(int(row.first_frame), len(trial)):
                mask_a = trial[frame] == owner_a
                mask_b = trial[frame] == owner_b
                pixels = int(mask_a.sum() + mask_b.sum())
                trial[frame][mask_a] = owner_b
                trial[frame][mask_b] = owner_a
                if pixels:
                    frames.append({
                        "proposal_id": row.proposal_id,
                        "proposal_kind": row.proposal_kind, "frame": frame,
                        "operation": operation, "owner_a": owner_a,
                        "owner_b": owner_b, "changed_pixels": pixels,
                    })
        unproven = fusion._unproven_duplicate_components(
            labels, trial, secondary_fraction)
        changed = trial != candidate
        audit_index = int(audit.index[audit.proposal_id.eq(row.proposal_id)][0])
        if unproven or not np.any(changed):
            audit.loc[audit_index, "eligible"] = False
            audit.loc[audit_index, "reason"] = (
                "new_unproven_duplicate_components" if unproven
                else "already_consistent")
            frames = [item for item in frames
                      if item["proposal_id"] != row.proposal_id]
            continue
        candidate = trial
        audit.loc[audit_index, "applied"] = True
        audit.loc[audit_index, "reason"] = f"applied_{operation}"
        audit.loc[audit_index, "changed_pixels"] = int(changed.sum())
        audit.loc[audit_index, "changed_frames"] = int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1)))
    return candidate, audit, pd.DataFrame(frames, columns=FRAME_COLUMNS)


def produce(labels: np.ndarray, unclaimed: np.ndarray, params: dict):
    assert_target_free(params)
    audit = discover(labels, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
    else:
        candidate, audit, frames = apply(labels, audit, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("fusion/area-flip recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("fusion/area-flip recovery changed identity set")
    strict_duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    unproven = fusion._unproven_duplicate_components(
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
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_duplicate_components": int(strict_duplicates),
        "projection_proven_new_duplicate_components": int(
            strict_duplicates - unproven),
        "unproven_new_duplicate_components": int(unproven),
    }
    return candidate, unclaimed.copy(), audit, frames, metrics


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    candidate, candidate_unclaimed, audit, frames, metrics = produce(
        labels, unclaimed, params)
    output_dir = Path(out.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(
            labels_path, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    audit_path = output_dir / "fusion_area_flip_audit.csv"
    frames_path = output_dir / "fusion_area_flip_frames.csv"
    metrics_path = output_dir / "metrics.json"
    audit.to_csv(audit_path, index=False)
    frames.to_csv(frames_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "frames": frames_path,
        "metrics": metrics_path,
    }, "summary": metrics}
