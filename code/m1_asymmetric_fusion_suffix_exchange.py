"""Repair asymmetric fusion owner swaps across the complete field.

An established large body can disappear into a much smaller neighbour while
the neighbour's labelled area expands to the large-body scale. If both names
return after a short fusion but remain exchanged, the fusion interval belongs
to the lost large owner and the complete post-split suffix is reciprocally
renamed. Discovery uses only movie-relative support, component geometry, area
ratios, and temporal continuity. Reviewed identities and locations cannot be
provided.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import owner_consensus


STRUCTURE = np.ones((3, 3), dtype=np.uint8)
PATH_KEYS = {"labels_path", "unclaimed_path"}
CONTROL_KEYS = {"mode", "targeting_mode", "output_stem"}
NUMERIC_KEYS = {
    "minimum_prior_support_movie_fraction",
    "maximum_fusion_gap_movie_fraction",
    "minimum_postsplit_support_movie_fraction",
    "minimum_large_to_small_area_ratio",
    "minimum_survivor_area_jump_ratio",
    "minimum_fused_to_lost_area_ratio",
    "maximum_fused_to_lost_area_ratio",
    "maximum_prior_separation_sum_radii",
    "minimum_fused_overlap_fraction_of_lost",
    "minimum_postsplit_union_overlap_fraction",
    "maximum_secondary_component_area_fraction",
}
ALLOWED_KEYS = PATH_KEYS | CONTROL_KEYS | NUMERIC_KEYS
AUDIT_COLUMNS = [
    "proposal_id", "small_survivor", "lost_large_owner", "fusion_start",
    "split_frame", "fusion_frames", "prior_support_small",
    "prior_support_large", "postsplit_support_frames", "prior_small_area",
    "prior_large_area", "fused_area", "large_to_small_area_ratio",
    "survivor_area_jump_ratio", "fused_to_lost_area_ratio",
    "prior_separation_sum_radii", "fused_overlap_fraction_of_lost",
    "postsplit_union_overlap_fraction", "changed_pixels", "changed_frames",
    "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "operation", "small_survivor",
    "lost_large_owner", "changed_pixels",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("asymmetric fusion recovery must be field-wide")
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


def _components(frame: np.ndarray, owner: int) -> list[np.ndarray]:
    labels, count = ndi.label(frame == int(owner), STRUCTURE)
    return [labels == index for index in range(1, count + 1)]


def _body_component(frame: np.ndarray, owner: int,
                    maximum_secondary_fraction: float) -> np.ndarray | None:
    """Return one body while tolerating only sub-body projection fragments."""
    parts = _components(frame, owner)
    if not parts:
        return None
    parts.sort(key=lambda mask: int(mask.sum()), reverse=True)
    body = parts[0]
    secondary = sum(int(mask.sum()) for mask in parts[1:])
    if secondary > maximum_secondary_fraction * int(body.sum()):
        return None
    return body


def _centroid(mask: np.ndarray) -> np.ndarray:
    return np.mean(np.argwhere(mask), axis=0)


def _radius(mask: np.ndarray) -> float:
    return math.sqrt(float(mask.sum()) / math.pi)


def _presence(labels: np.ndarray) -> dict[int, np.ndarray]:
    identities = sorted(set(map(int, np.unique(labels))) - {0})
    return {
        owner: np.asarray([np.any(frame == owner) for frame in labels], bool)
        for owner in identities
    }


def _recent_area(labels: np.ndarray, owner: int, end: int,
                 support: int) -> float:
    values = [int(np.count_nonzero(labels[frame] == owner))
              for frame in range(max(0, end - support), end)
              if np.any(labels[frame] == owner)]
    return float(np.median(values)) if values else 0.0


def _first_return(flags: np.ndarray, start: int, maximum_gap: int) -> int:
    stop = min(len(flags), start + maximum_gap + 1)
    indices = np.flatnonzero(flags[start:stop])
    return int(start + indices[0]) if len(indices) else -1


def discover(labels: np.ndarray, params: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Audit every short owner loss that has an asymmetric fusion signature."""
    frame_count = len(labels)
    presence = _presence(labels)
    minimum_prior = max(3, math.ceil(frame_count * _fraction(
        params, "minimum_prior_support_movie_fraction", 0.25)))
    maximum_gap = max(1, math.ceil(frame_count * _fraction(
        params, "maximum_fusion_gap_movie_fraction", 0.05)))
    minimum_post = max(2, math.ceil(frame_count * _fraction(
        params, "minimum_postsplit_support_movie_fraction", 0.10)))
    minimum_size_ratio = float(params.get(
        "minimum_large_to_small_area_ratio", 3.0))
    minimum_jump = float(params.get(
        "minimum_survivor_area_jump_ratio", 3.0))
    minimum_fused_ratio = float(params.get(
        "minimum_fused_to_lost_area_ratio", 0.50))
    maximum_fused_ratio = float(params.get(
        "maximum_fused_to_lost_area_ratio", 2.0))
    maximum_separation = float(params.get(
        "maximum_prior_separation_sum_radii", 3.0))
    minimum_lost_overlap = float(params.get(
        "minimum_fused_overlap_fraction_of_lost", 0.10))
    minimum_split_overlap = float(params.get(
        "minimum_postsplit_union_overlap_fraction", 0.45))
    maximum_secondary = float(params.get(
        "maximum_secondary_component_area_fraction", 0.15))

    rows: list[dict] = []
    internal: list[dict] = []
    for fusion_start in range(1, frame_count - 1):
        prior_owners = {owner for owner, flags in presence.items()
                        if flags[fusion_start - 1]}
        current_owners = {owner for owner, flags in presence.items()
                          if flags[fusion_start]}
        lost_owners = sorted(prior_owners - current_owners)
        survivors = sorted(prior_owners & current_owners)
        for lost_owner in lost_owners:
            lost_flags = presence[lost_owner]
            split_frame = _first_return(lost_flags, fusion_start, maximum_gap)
            if split_frame < 0:
                continue
            fusion_frames = split_frame - fusion_start
            if fusion_frames <= 0:
                continue
            prior_large = _body_component(
                labels[fusion_start - 1], lost_owner, maximum_secondary)
            if prior_large is None:
                continue
            prior_support_large = int(lost_flags[:fusion_start].sum())
            for small_owner in survivors:
                small_flags = presence[small_owner]
                if not bool(small_flags[fusion_start:split_frame].all()):
                    continue
                prior_small = _body_component(
                    labels[fusion_start - 1], small_owner, maximum_secondary)
                fused_first = _body_component(
                    labels[fusion_start], small_owner, maximum_secondary)
                fused_last = _body_component(
                    labels[split_frame - 1], small_owner, maximum_secondary)
                split_small = _body_component(
                    labels[split_frame], small_owner, maximum_secondary)
                split_large = _body_component(
                    labels[split_frame], lost_owner, maximum_secondary)
                if any(mask is None for mask in (
                        prior_small, fused_first, fused_last,
                        split_small, split_large)):
                    continue
                prior_support_small = int(small_flags[:fusion_start].sum())
                postsplit_support = int(np.count_nonzero(
                    small_flags[split_frame:] & lost_flags[split_frame:]))
                recent_window = max(3, minimum_prior // 2)
                prior_small_area = _recent_area(
                    labels, small_owner, fusion_start, recent_window)
                prior_large_area = _recent_area(
                    labels, lost_owner, fusion_start, recent_window)
                fused_area = float(fused_first.sum())
                size_ratio = prior_large_area / max(prior_small_area, 1.0)
                jump_ratio = fused_area / max(prior_small_area, 1.0)
                fused_ratio = fused_area / max(prior_large_area, 1.0)
                distance = float(np.linalg.norm(
                    _centroid(prior_small) - _centroid(prior_large)))
                separation = distance / max(
                    _radius(prior_small) + _radius(prior_large), 1.0)
                lost_overlap = float(np.count_nonzero(
                    fused_first & prior_large) / max(prior_large.sum(), 1))
                split_overlap = float(np.count_nonzero(
                    fused_last & (split_small | split_large))
                    / max(fused_last.sum(), 1))
                reasons: list[str] = []
                if min(prior_support_small, prior_support_large) < minimum_prior:
                    reasons.append("insufficient_bilateral_prior_support")
                if postsplit_support < minimum_post:
                    reasons.append("insufficient_bilateral_postsplit_support")
                if size_ratio < minimum_size_ratio:
                    reasons.append("prior_bodies_not_asymmetric")
                if jump_ratio < minimum_jump:
                    reasons.append("survivor_area_did_not_jump")
                if not minimum_fused_ratio <= fused_ratio <= maximum_fused_ratio:
                    reasons.append("fused_area_not_lost_body_scale")
                if separation > maximum_separation:
                    reasons.append("prior_bodies_not_local")
                if lost_overlap < minimum_lost_overlap:
                    reasons.append("fused_mask_does_not_inherit_lost_body")
                if split_overlap < minimum_split_overlap:
                    reasons.append("postsplit_union_not_fused_descendant")
                proposal_id = f"AF{len(rows) + 1:04d}"
                public = {
                    "proposal_id": proposal_id,
                    "small_survivor": int(small_owner),
                    "lost_large_owner": int(lost_owner),
                    "fusion_start": int(fusion_start),
                    "split_frame": int(split_frame),
                    "fusion_frames": int(fusion_frames),
                    "prior_support_small": prior_support_small,
                    "prior_support_large": prior_support_large,
                    "postsplit_support_frames": postsplit_support,
                    "prior_small_area": prior_small_area,
                    "prior_large_area": prior_large_area,
                    "fused_area": fused_area,
                    "large_to_small_area_ratio": size_ratio,
                    "survivor_area_jump_ratio": jump_ratio,
                    "fused_to_lost_area_ratio": fused_ratio,
                    "prior_separation_sum_radii": separation,
                    "fused_overlap_fraction_of_lost": lost_overlap,
                    "postsplit_union_overlap_fraction": split_overlap,
                    "changed_pixels": 0,
                    "changed_frames": 0,
                    "eligible": not reasons,
                    "applied": False,
                    "reason": "eligible_asymmetric_fusion_exchange" \
                        if not reasons else "|".join(reasons),
                }
                rows.append(public)
                internal.append({**public})
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS), internal


def _unproven_duplicate_components(
        before: np.ndarray, after: np.ndarray,
        maximum_secondary_fraction: float) -> int:
    """Count excess components that are too large to be projection fragments."""
    unproven = 0
    owners = sorted(set(map(int, np.unique(after))) - {0})
    for frame_index, frame in enumerate(after):
        for owner in owners:
            old_count = len(_components(before[frame_index], owner))
            parts = _components(frame, owner)
            excess = max(0, len(parts) - max(old_count, 1))
            if not excess:
                continue
            parts.sort(key=lambda mask: int(mask.sum()), reverse=True)
            secondary = sum(int(mask.sum()) for mask in parts[1:])
            if secondary > maximum_secondary_fraction * int(parts[0].sum()):
                unproven += excess
    return int(unproven)


def apply(labels: np.ndarray, audit: pd.DataFrame,
          internal: list[dict], maximum_secondary_fraction: float,
          ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Apply disjoint identity-pair exchanges atomically."""
    candidate = labels.copy()
    eligible = [row for row in internal if bool(row["eligible"])]
    owner_counts: dict[int, int] = {}
    for row in eligible:
        for owner in (int(row["small_survivor"]),
                      int(row["lost_large_owner"])):
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
    frames: list[dict] = []
    for row in eligible:
        proposal_id = str(row["proposal_id"])
        small_owner = int(row["small_survivor"])
        large_owner = int(row["lost_large_owner"])
        index = int(audit.index[audit.proposal_id.eq(proposal_id)][0])
        if owner_counts[small_owner] != 1 or owner_counts[large_owner] != 1:
            audit.loc[index, "eligible"] = False
            audit.loc[index, "reason"] = "overlapping_identity_pair_proposals"
            continue
        trial = candidate.copy()
        fusion_start = int(row["fusion_start"])
        split_frame = int(row["split_frame"])
        for frame in range(fusion_start, split_frame):
            mask = trial[frame] == small_owner
            pixels = int(mask.sum())
            trial[frame][mask] = large_owner
            frames.append({
                "proposal_id": proposal_id, "frame": frame,
                "operation": "fusion_owner_restore",
                "small_survivor": small_owner,
                "lost_large_owner": large_owner,
                "changed_pixels": pixels,
            })
        for frame in range(split_frame, len(trial)):
            small_mask = trial[frame] == small_owner
            large_mask = trial[frame] == large_owner
            pixels = int(small_mask.sum() + large_mask.sum())
            trial[frame][small_mask] = large_owner
            trial[frame][large_mask] = small_owner
            if pixels:
                frames.append({
                    "proposal_id": proposal_id, "frame": frame,
                    "operation": "reciprocal_suffix_exchange",
                    "small_survivor": small_owner,
                    "lost_large_owner": large_owner,
                    "changed_pixels": pixels,
                })
        unproven_duplicates = _unproven_duplicate_components(
            labels, trial, maximum_secondary_fraction)
        if unproven_duplicates:
            audit.loc[index, "eligible"] = False
            audit.loc[index, "reason"] = "new_unproven_duplicate_components"
            frames = [item for item in frames
                      if item["proposal_id"] != proposal_id]
            continue
        changed = trial != candidate
        if not np.any(changed):
            audit.loc[index, "eligible"] = False
            audit.loc[index, "reason"] = "already_consistent"
            frames = [item for item in frames
                      if item["proposal_id"] != proposal_id]
            continue
        candidate = trial
        audit.loc[index, "applied"] = True
        audit.loc[index, "reason"] = "applied_asymmetric_fusion_suffix_exchange"
        audit.loc[index, "changed_pixels"] = int(changed.sum())
        audit.loc[index, "changed_frames"] = int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1)))
    return candidate, audit, pd.DataFrame(frames, columns=FRAME_COLUMNS)


def produce(labels: np.ndarray, unclaimed: np.ndarray,
            params: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                                   pd.DataFrame, dict]:
    assert_target_free(params)
    audit, internal = discover(labels, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
    else:
        candidate, audit, frames = apply(
            labels, audit, internal,
            float(params.get("maximum_secondary_component_area_fraction",
                             0.15)))
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("asymmetric fusion exchange changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("asymmetric fusion exchange changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    unproven_duplicates = _unproven_duplicate_components(
        labels, candidate,
        float(params.get("maximum_secondary_component_area_fraction", 0.15)))
    if unproven_duplicates:
        raise AssertionError("asymmetric fusion exchange created duplicates")
    changed = candidate != labels
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    summary = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "wells", "review_cases")},
        "owner_losses_audited": int(len(audit)),
        "eligible_exchanges": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_exchanges": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_duplicate_components": int(duplicates),
        "projection_proven_new_duplicate_components": int(
            duplicates - unproven_duplicates),
        "unproven_new_duplicate_components": int(unproven_duplicates),
    }
    return candidate, unclaimed.copy(), audit, frames, summary


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    assert_target_free(params)
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
    audit_path = output_dir / "asymmetric_fusion_exchange_audit.csv"
    frame_path = output_dir / "asymmetric_fusion_exchange_frames.csv"
    metrics_path = output_dir / "metrics.json"
    audit.to_csv(audit_path, index=False)
    frames.to_csv(frame_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "frames": frame_path, "metrics": metrics_path,
    }, "summary": metrics}
