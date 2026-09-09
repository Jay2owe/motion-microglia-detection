"""Censor dormant-owner recovery when two durable bodies become separable.

The underlying takeover detector remains complete-field and identity-blind.
This refinement prevents a recovered owner from being painted onto a second
body after that owner has independently reappeared as a durable, spatially
separate and eventually comparable component.
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

import component_accounting
import dormant_owner_takeover as base


NEW_PARAMETER_KEYS = {
    "minimum_separable_reappearance_support_movie_fraction",
    "minimum_separable_reappearance_remaining_fraction",
    "minimum_separable_reappearance_run_movie_fraction",
    "minimum_separable_reappearance_maximum_area_ratio",
    "minimum_separable_reappearance_median_distance_sum_radii",
}
ALLOWED_KEYS = base.ALLOWED_KEYS | NEW_PARAMETER_KEYS
AUDIT_COLUMNS = base.AUDIT_COLUMNS + [
    "separable_reappearance_frame",
    "reappearance_support_frames",
    "reappearance_remaining_fraction",
    "reappearance_maximum_run_frames",
    "reappearance_maximum_area_ratio",
    "reappearance_median_distance_sum_radii",
    "recovery_last_frame",
]
APPLICATION_COLUMNS = base.APPLICATION_COLUMNS + [
    "recovery_censored", "censor_frame",
]


def assert_target_free(params: dict) -> None:
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError("unsupported conflict-censored takeover parameters: "
                         + ", ".join(unsupported))
    supplied = []
    for key, value in params.items():
        if value in (None, "", [], {}):
            continue
        name = str(key).strip().lower()
        if name in base.FORBIDDEN_KEYS or any(
                token in name for token in base.FORBIDDEN_FRAGMENTS):
            supplied.append(str(key))
    if supplied:
        raise ValueError("conflict-censored takeover received forbidden "
                         "selectors: " + ", ".join(sorted(supplied)))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("conflict-censored takeover must be field-wide")
    missing = sorted(key for key in base.PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing conflict-censored takeover inputs: "
                         + ", ".join(missing))


def _base_params(params: dict) -> dict:
    return {key: value for key, value in params.items()
            if key in base.ALLOWED_KEYS}


def _maximum_consecutive_run(frames: list[int]) -> int:
    maximum = current = 0
    previous = None
    for frame in sorted(set(frames)):
        current = current + 1 if previous is not None and frame == previous + 1 \
            else 1
        maximum = max(maximum, current)
        previous = frame
    return maximum


def _separable_reappearance(
        chain: tuple[int, ...], source_owner: int,
        components: dict[int, base.Component], movie_frames: int,
        params: dict) -> dict:
    targets = [components[int(key)] for key in chain]
    by_frame = {}
    for component in components.values():
        if component.owner == source_owner:
            by_frame.setdefault(component.frame, []).append(component)
    observations = []
    for target in targets:
        extras = by_frame.get(target.frame, [])
        if not extras:
            continue
        extra = max(extras, key=lambda item: item.area)
        observations.append({
            "frame": target.frame,
            "area_ratio": float(extra.area / max(target.area, 1)),
            "distance": float(math.hypot(target.x - extra.x,
                                          target.y - extra.y)
                              / max(target.radius + extra.radius, 1.0)),
        })

    minimum_support = int(math.ceil(movie_frames * float(params.get(
        "minimum_separable_reappearance_support_movie_fraction", 0.05))))
    minimum_remaining = float(params.get(
        "minimum_separable_reappearance_remaining_fraction", 0.25))
    minimum_run = int(math.ceil(movie_frames * float(params.get(
        "minimum_separable_reappearance_run_movie_fraction", 0.05))))
    minimum_area_ratio = float(params.get(
        "minimum_separable_reappearance_maximum_area_ratio", 0.50))
    minimum_distance = float(params.get(
        "minimum_separable_reappearance_median_distance_sum_radii", 1.50))

    for onset in sorted({row["frame"] for row in observations}):
        later = [row for row in observations if row["frame"] >= onset]
        remaining = sum(target.frame >= onset for target in targets)
        support_frames = len({row["frame"] for row in later})
        remaining_fraction = float(support_frames / max(remaining, 1))
        maximum_run = _maximum_consecutive_run(
            [int(row["frame"]) for row in later])
        maximum_area_ratio = max(
            (float(row["area_ratio"]) for row in later), default=0.0)
        median_distance = float(np.median(
            [float(row["distance"]) for row in later])) if later else 0.0
        if (support_frames >= minimum_support
                and remaining_fraction >= minimum_remaining
                and maximum_run >= minimum_run
                and maximum_area_ratio >= minimum_area_ratio
                and median_distance >= minimum_distance):
            return {
                "separable_reappearance_frame": int(onset),
                "reappearance_support_frames": int(support_frames),
                "reappearance_remaining_fraction": remaining_fraction,
                "reappearance_maximum_run_frames": int(maximum_run),
                "reappearance_maximum_area_ratio": maximum_area_ratio,
                "reappearance_median_distance_sum_radii": median_distance,
            }
    return {
        "separable_reappearance_frame": -1,
        "reappearance_support_frames": 0,
        "reappearance_remaining_fraction": 0.0,
        "reappearance_maximum_run_frames": 0,
        "reappearance_maximum_area_ratio": 0.0,
        "reappearance_median_distance_sum_radii": 0.0,
    }


def discover(labels: np.ndarray, params: dict):
    """Augment every base proposal with a field-derived conflict boundary."""
    assert_target_free(params)
    audit, internal, indexed, components = base.discover(
        labels, _base_params(params))
    public = audit.copy()
    detailed = internal.copy()
    additions = []
    for proposal in detailed.itertuples(index=False):
        evidence = _separable_reappearance(
            tuple(proposal.terminal_chain), int(proposal.source_owner),
            components, len(labels), params)
        last_frame = int(proposal.terminal_last)
        if evidence["separable_reappearance_frame"] >= 0:
            last_frame = evidence["separable_reappearance_frame"] - 1
        additions.append({**evidence, "recovery_last_frame": last_frame})
    addition_table = pd.DataFrame(additions)
    for column in AUDIT_COLUMNS[len(base.AUDIT_COLUMNS):]:
        values = addition_table[column] if len(addition_table) else []
        public[column] = values
        detailed[column] = values
    if len(public):
        censored = (public.eligible.astype(bool)
                    & (public.separable_reappearance_frame.astype(int) >= 0))
        public.loc[censored, "reason"] = \
            "eligible_conflict_censored_large_terminal_dormant_takeover"
        detailed.loc[censored, "reason"] = \
            "eligible_conflict_censored_large_terminal_dormant_takeover"
    return public[AUDIT_COLUMNS], detailed, indexed, components


def apply(labels: np.ndarray, internal: pd.DataFrame, indexed: np.ndarray,
          components: dict[int, base.Component], enabled: bool):
    candidate = labels.copy()
    rows = []
    occupied = np.zeros(labels.shape, bool)
    eligible = internal[internal.eligible.astype(bool)] if len(internal) \
        else internal
    for proposal in eligible.itertuples(index=False):
        censor = int(proposal.separable_reappearance_frame)
        selected = [int(key) for key in proposal.terminal_chain
                    if censor < 0 or components[int(key)].frame < censor]
        pending = []
        refused = "" if selected else "no_preconflict_recovery_interval"
        for key in selected:
            component = components[key]
            mask = indexed[component.frame] == component.key
            if np.any(occupied[component.frame] & mask):
                refused = "overlapping_eligible_proposal"
                break
            if not np.all(candidate[component.frame][mask]
                          == int(proposal.incoming_owner)):
                refused = "incoming_component_provenance_changed"
                break
            before_components = ndi.label(
                candidate[component.frame] == int(proposal.source_owner),
                base.STRUCTURE)[1]
            pending.append((component, mask, before_components))
        if refused:
            rows.append({
                "proposal_id": proposal.proposal_id, "frame": -1,
                "source_owner": int(proposal.source_owner),
                "incoming_owner": int(proposal.incoming_owner),
                "component_pixels": 0, "preexisting_source_components": 0,
                "explained_projection_components": 0, "changed_pixels": 0,
                "applied": False, "reason": refused,
                "recovery_censored": censor >= 0, "censor_frame": censor,
            })
            continue
        for component, mask, before_components in pending:
            changed = int(mask.sum()) if enabled else 0
            if enabled:
                candidate[component.frame][mask] = int(proposal.source_owner)
                occupied[component.frame] |= mask
            rows.append({
                "proposal_id": proposal.proposal_id,
                "frame": component.frame,
                "source_owner": int(proposal.source_owner),
                "incoming_owner": int(proposal.incoming_owner),
                "component_pixels": component.area,
                "preexisting_source_components": int(before_components),
                "explained_projection_components": 0,
                "changed_pixels": changed,
                "applied": bool(enabled),
                "reason": ("conflict_censored_dormant_takeover_recovery"
                           if enabled else "audit_only"),
                "recovery_censored": censor >= 0,
                "censor_frame": censor,
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
        raise AssertionError("conflict censor changed assigned foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("conflict censor overlapped unclaimed ledger")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("conflict censor changed the movie identity set")
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    new_excess = component_accounting.new_duplicate_components(
        labels, candidate)
    censored = audit[
        audit.eligible.astype(bool)
        & (audit.separable_reappearance_frame.astype(int) >= 0)] \
        if len(audit) else audit
    conflicts_avoided = int(sum(
        int(row.reappearance_support_frames) for row in
        censored.itertuples(index=False))) if len(censored) else 0
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
        "censored_takeovers": int(len(censored)),
        "simultaneous_source_owner_conflicts_avoided": conflicts_avoided,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_exact": True,
        "unclaimed_ledger_exact": True,
        "movie_identity_set_exact": True,
        "new_component_excess": int(new_excess),
        "new_unexplained_component_excess": int(new_excess),
        "projection_explanation":
            "durable_separable_reappearance_censors_recovery",
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
        "audit": output / "conflict_censored_dormant_takeover_audit.csv",
        "applications": output /
            "conflict_censored_dormant_takeover_applications.csv",
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
