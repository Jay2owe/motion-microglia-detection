"""Backfill a late owner only after a duplicated-owner physical split.

Discovery is field-wide. Biological identities, tracks, frames, coordinates,
events, and review cases are inferred from complete stack evidence and cannot
be supplied as producer targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile


import separable_merge_recovery as accepted_merge
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "track_id", "frame_id", "event_id", "coordinate",
    "review_case", "case_id", "target_identity", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
    "review_region",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        name for name, value in params.items()
        if value not in (None, "", [], {})
        and (name in FORBIDDEN_TARGETS or any(
            token in name.lower() for token in FORBIDDEN_TARGET_TOKENS)))
    if supplied:
        raise ValueError(
            "field-wide post-split backfill received forbidden targets: "
            + ", ".join(supplied))


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    target_track: int
    predecessor_track: int
    donor_track: int
    successor_track: int
    foreign_owner: int
    stable_owner: int
    target_first: int
    split_first: int
    correction_last: int
    stable_first: int
    target_last: int
    foreign_run_frames: int
    correction_frames: int
    stable_run_frames: int
    successor_support_frames: int
    maximum_valley_ratio: float
    discovery_status: str
    discovery_reason: str


def _runs(group: pd.DataFrame) -> list[dict]:
    rows = list(group.sort_values("frame").itertuples(index=False))
    result: list[dict] = []
    for row in rows:
        owner = int(row.accepted_owner)
        frame = int(row.frame)
        if (result and result[-1]["owner"] == owner
                and frame == result[-1]["last"] + 1):
            result[-1]["last"] = frame
            result[-1]["frames"] += 1
        else:
            result.append({"owner": owner, "first": frame,
                           "last": frame, "frames": 1})
    return result


def _endpoint(group: pd.DataFrame, first: bool):
    ordered = group.sort_values("frame")
    return ordered.iloc[0] if first else ordered.iloc[-1]


def _normalised_step(left, right) -> float:
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    return distance / max(float(left.radius_px) + float(right.radius_px), 1.0)


def _owner_support(group: pd.DataFrame, owner: int) -> int:
    visible = group[group.physically_visible.astype(bool)]
    return int(np.count_nonzero(visible.accepted_owner.to_numpy(int) == owner))


def discover(scored: pd.DataFrame, labels: np.ndarray, raw: np.ndarray,
             params: dict) -> pd.DataFrame:
    visible = scored[scored.physically_visible.astype(bool)].copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id", sort=True)}
    active = sorted(set(map(int, np.unique(labels))) - {0})
    first_claim = {
        identity: int(np.flatnonzero(np.any(
            labels == identity, axis=(1, 2)))[0]) for identity in active}
    minimum_foreign = int(params.get("minimum_foreign_prefix_frames", 6))
    minimum_stable = int(params.get("minimum_terminal_stable_frames", 4))
    minimum_successor = int(params.get("minimum_successor_support_frames", 8))
    minimum_correction = int(params.get("minimum_correction_frames", 5))
    maximum_step = float(params.get("maximum_handoff_step_sum_radii", 1.75))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.65))
    proposals: list[dict] = []

    for target_track, group in groups.items():
        runs = _runs(group)
        positive = [run for run in runs if int(run["owner"]) > 0]
        if len(positive) < 2:
            continue
        foreign_run, stable_run = positive[-2:]
        foreign_owner = int(foreign_run["owner"])
        stable_owner = int(stable_run["owner"])
        if foreign_owner == stable_owner:
            continue
        reasons: list[str] = []
        target_first = int(group.frame.min())
        target_last = int(group.frame.max())
        if (int(foreign_run["first"]) != target_first
                or int(foreign_run["last"]) + 1 != int(stable_run["first"])):
            reasons.append("not_two_contiguous_terminal_owner_runs")
        if int(foreign_run["frames"]) < minimum_foreign:
            reasons.append("foreign_prefix_too_short")
        if int(stable_run["frames"]) < minimum_stable:
            reasons.append("stable_run_too_short")
        if int(stable_run["last"]) != target_last:
            reasons.append("stable_run_not_terminal")
        expected = np.arange(target_first, target_last + 1)
        if not np.array_equal(group.frame.to_numpy(int), expected):
            reasons.append("physical_track_not_visible_continuous")
        if first_claim.get(stable_owner, -1) != int(stable_run["first"]):
            reasons.append("stable_owner_not_born_on_target_run")

        predecessor_matches = []
        successor_matches = []
        donor_matches = []
        target_start_point = _endpoint(group, True)
        target_end_point = _endpoint(group, False)
        for other_track, other in groups.items():
            if other_track == target_track:
                continue
            other_first = int(other.frame.min())
            other_last = int(other.frame.max())
            if other_last == target_first - 1:
                point = _endpoint(other, False)
                if (int(point.accepted_owner) == foreign_owner
                        and _normalised_step(point, target_start_point)
                        <= maximum_step):
                    predecessor_matches.append(other_track)
            if other_first == target_last + 1:
                point = _endpoint(other, True)
                if (int(point.accepted_owner) == stable_owner
                        and _owner_support(other, stable_owner) >= minimum_successor
                        and _normalised_step(target_end_point, point) <= maximum_step):
                    successor_matches.append(other_track)
            if (target_first < other_first <= int(foreign_run["last"])
                    and other_last >= int(stable_run["first"])
                    and _owner_support(other, foreign_owner)
                    == len(other[other.physically_visible.astype(bool)])):
                overlap = group[group.frame.between(
                    other_first, int(foreign_run["last"]))]
                donor_overlap = other[other.frame.between(
                    other_first, int(foreign_run["last"]))]
                if len(overlap) != len(donor_overlap) or not len(overlap):
                    continue
                valleys = []
                valid = True
                for target_point, donor_point in zip(
                        overlap.itertuples(index=False),
                        donor_overlap.itertuples(index=False)):
                    if (int(target_point.frame) != int(donor_point.frame)
                            or int(target_point.accepted_owner) != foreign_owner
                            or int(donor_point.accepted_owner) != foreign_owner):
                        valid = False
                        break
                    valleys.append(accepted_merge._valley_ratio(
                        raw[int(target_point.frame)],
                        (float(target_point.x), float(target_point.y)),
                        (float(donor_point.x), float(donor_point.y))))
                if valid and valleys and max(valleys) <= maximum_valley:
                    donor_matches.append((other_track, other_first, max(valleys)))

        if len(predecessor_matches) != 1:
            reasons.append("predecessor_not_unique")
        if len(successor_matches) != 1:
            reasons.append("successor_not_unique")
        if len(donor_matches) != 1:
            reasons.append("post_split_donor_not_unique")
        split_first = donor_matches[0][1] if len(donor_matches) == 1 else -1
        correction_frames = (int(foreign_run["last"]) - split_first + 1
                             if split_first >= 0 else 0)
        if correction_frames < minimum_correction:
            reasons.append("correction_interval_too_short")
        if stable_owner > 0 and split_first >= 0 and np.any(
                labels[split_first:int(stable_run["first"])] == stable_owner):
            reasons.append("stable_owner_already_present_before_stable_run")

        proposal = Proposal(
            proposal_id=f"P{len(proposals) + 1:04d}",
            target_track=target_track,
            predecessor_track=(predecessor_matches[0]
                               if len(predecessor_matches) == 1 else -1),
            donor_track=(donor_matches[0][0]
                         if len(donor_matches) == 1 else -1),
            successor_track=(successor_matches[0]
                             if len(successor_matches) == 1 else -1),
            foreign_owner=foreign_owner,
            stable_owner=stable_owner,
            target_first=target_first,
            split_first=split_first,
            correction_last=int(foreign_run["last"]),
            stable_first=int(stable_run["first"]),
            target_last=target_last,
            foreign_run_frames=int(foreign_run["frames"]),
            correction_frames=correction_frames,
            stable_run_frames=int(stable_run["frames"]),
            successor_support_frames=(
                _owner_support(groups[successor_matches[0]], stable_owner)
                if len(successor_matches) == 1 else 0),
            maximum_valley_ratio=(float(donor_matches[0][2])
                                  if len(donor_matches) == 1 else float("nan")),
            discovery_status="eligible" if not reasons else "rejected",
            discovery_reason="eligible" if not reasons else "|".join(reasons),
        )
        proposals.append(asdict(proposal))
    return pd.DataFrame(proposals)


def _component_excess(frame: np.ndarray, identity: int) -> int:
    return max(0, int(ndi.label(frame == identity, STRUCTURE)[1]) - 1)


def _apply(labels: np.ndarray, raw: np.ndarray, scored: pd.DataFrame,
           proposal, params: dict):
    candidate = labels.copy()
    visible = scored[scored.physically_visible.astype(bool)]
    sigma = float(params.get("watershed_sigma_px", 1.0))
    audit = []
    for frame in range(int(proposal.split_first),
                       int(proposal.correction_last) + 1):
        target_rows = visible[(visible.track_id == int(proposal.target_track))
                              & (visible.frame == frame)]
        donor_rows = visible[(visible.track_id == int(proposal.donor_track))
                             & (visible.frame == frame)]
        if len(target_rows) != 1 or len(donor_rows) != 1:
            return None, pd.DataFrame(audit), "missing_unique_track_points"
        target_point = target_rows.iloc[0]
        donor_point = donor_rows.iloc[0]
        foreign = int(proposal.foreign_owner)
        target = int(proposal.stable_owner)
        before = candidate[frame].copy()
        components, _ = ndi.label(before == foreign, STRUCTURE)
        target_component = accepted_merge._component_at(
            components, float(target_point.x), float(target_point.y))
        donor_component = accepted_merge._component_at(
            components, float(donor_point.x), float(donor_point.y))
        if target_component <= 0 or donor_component <= 0:
            return None, pd.DataFrame(audit), "track_outside_foreign_owner"
        source = components == target_component
        method = "whole_separated_component"
        if donor_component == target_component:
            target_marker = accepted_merge._marker_pixel(
                source, float(target_point.x), float(target_point.y))
            donor_marker = accepted_merge._marker_pixel(
                source, float(donor_point.x), float(donor_point.y))
            if (target_marker is None or donor_marker is None
                    or target_marker == donor_marker):
                return None, pd.DataFrame(audit), "watershed_markers_unavailable"
            markers = np.zeros(source.shape, np.int16)
            markers[target_marker] = 1
            markers[donor_marker] = 2
            elevation = -ndi.gaussian_filter(
                raw[frame].astype(np.float32), sigma)
            recovered = watershed(
                elevation, markers=markers, mask=source) == 1
            method = "raw_seeded_watershed"
        else:
            recovered = source
        if not np.any(recovered):
            return None, pd.DataFrame(audit), "empty_recovered_partition"
        trial = before.copy()
        trial[recovered] = target
        y = int(np.clip(round(float(donor_point.y)), 0, trial.shape[0] - 1))
        x = int(np.clip(round(float(donor_point.x)), 0, trial.shape[1] - 1))
        cleanup_pixels = 0
        if donor_component == target_component:
            residual_parts, _ = ndi.label(
                source & (trial == foreign), STRUCTURE)
            donor_residual = int(residual_parts[y, x])
            if donor_residual <= 0:
                return None, pd.DataFrame(audit), "donor_core_not_preserved"
            shards = ((residual_parts > 0)
                      & (residual_parts != donor_residual))
            cleanup_pixels = int(np.count_nonzero(shards))
            trial[shards] = target
        if trial[y, x] != foreign:
            return None, pd.DataFrame(audit), "donor_core_not_preserved"
        if _component_excess(trial, foreign) > _component_excess(before, foreign):
            return None, pd.DataFrame(audit), "new_foreign_owner_duplicate"
        if _component_excess(trial, target) > max(
                0, _component_excess(before, target)):
            return None, pd.DataFrame(audit), "new_target_owner_duplicate"
        candidate[frame] = trial
        audit.append({
            "proposal_id": proposal.proposal_id,
            "frame": frame,
            "partition_method": method,
            "changed_pixels": int(np.count_nonzero(trial != before)),
            "target_partition_pixels": int(np.count_nonzero(recovered)),
            "donor_shard_cleanup_pixels": cleanup_pixels,
            "donor_pixels_remaining": int(np.count_nonzero(trial == foreign)),
        })
    return candidate, pd.DataFrame(audit), "applied"


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict):
    assert_target_free(params)
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("post-split backfill must use field-wide discovery")
    if not (labels.shape == unclaimed.shape == raw.shape):
        raise ValueError("labels, unclaimed and raw stacks must align")
    scored = accepted_merge.attach_owners(points, labels)
    proposals = discover(scored, labels, raw, params)
    candidate = labels.copy()
    occupied = np.zeros_like(labels, bool)
    applications = []
    frame_audits = []
    eligible = proposals[proposals.discovery_status == "eligible"].sort_values(
        ["split_first", "target_track", "proposal_id"])
    for proposal in eligible.itertuples(index=False):
        proposed, audit, reason = _apply(candidate, raw, scored, proposal, params)
        if proposed is None:
            applications.append({
                "proposal_id": proposal.proposal_id,
                "target_track": int(proposal.target_track),
                "outcome": "rejected_application", "reason": reason,
                "changed_pixels": 0, "changed_frames": 0})
            continue
        changed = proposed != candidate
        if np.any(occupied & changed):
            applications.append({
                "proposal_id": proposal.proposal_id,
                "target_track": int(proposal.target_track),
                "outcome": "rejected_application",
                "reason": "overlapping_proposal", "changed_pixels": 0,
                "changed_frames": 0})
            continue
        candidate = proposed
        occupied |= changed
        applications.append({
            "proposal_id": proposal.proposal_id,
            "target_track": int(proposal.target_track),
            "outcome": "applied", "reason": reason,
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_frames": int(np.count_nonzero(
                np.any(changed, axis=(1, 2))))})
        frame_audits.append(audit)
    return (candidate, proposals, pd.DataFrame(applications),
            pd.concat(frame_audits, ignore_index=True)
            if frame_audits else pd.DataFrame())


def _excess_components(stack: np.ndarray) -> int:
    return component_accounting.total_component_excess(stack)


def summarize(labels: np.ndarray, candidate: np.ndarray,
              unclaimed: np.ndarray, raw: np.ndarray,
              points: pd.DataFrame,
              proposals: pd.DataFrame, applications: pd.DataFrame,
              mode: str) -> dict:
    changed = candidate != labels
    active = set(map(int, np.unique(labels))) - {0}
    after_active = set(map(int, np.unique(candidate))) - {0}
    named_losses = 0
    for identity in active:
        before = np.any(labels == identity, axis=(1, 2))
        after = np.any(candidate == identity, axis=(1, 2))
        named_losses += int(np.count_nonzero(before & ~after))
    return {
        "mode": mode,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "review_case_targets_received": False,
        "tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(proposals)),
        "eligible_proposals": int(proposals.discovery_status.eq(
            "eligible").sum()) if len(proposals) else 0,
        "applied_proposals": int(applications.outcome.eq("applied").sum())
            if len(applications) else 0,
        "rejected_application_proposals": int(applications.outcome.eq(
            "rejected_application").sum()) if len(applications) else 0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
        "relabelled_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate > 0))),
        "raw_supported_additions": int(np.count_nonzero(
            changed & (labels == 0) & (candidate > 0))),
        "removed_foreground_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "preexisting_unclaimed_changed_pixels": 0,
        "unclaimed_overlap_additions": int(np.count_nonzero(
            changed & (labels == 0) & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(
            changed & (labels == 0) & (candidate > 0) & (raw == 0))),
        "old_identity_set_preserved": active == after_active,
        "donor_named_frame_losses": named_losses,
        "new_same_frame_identity_components": max(
            0, _excess_components(candidate) - _excess_components(labels)),
    }


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    mode = str(params.get("mode", "candidate"))
    candidate, proposals, applications, frames = recover(
        labels, unclaimed, raw, points, params)
    if mode == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame(columns=[
            "proposal_id", "target_track", "outcome", "reason",
            "changed_pixels", "changed_frames"])
        frames = pd.DataFrame()
    elif mode != "candidate":
        raise ValueError(f"unknown mode: {mode}")
    metrics = summarize(
        labels, candidate, unclaimed, raw, points, proposals, applications,
        mode)
    paths = {
        "labels": out.out / "95_A3.tif",
        "unclaimed": out.out / "95_A3_unclaimed_original_ids.tif",
        "changed": out.out / "changed_pixels.tif",
        "proposals": out.out / "post_split_proposals.csv",
        "applications": out.out / "post_split_applications.csv",
        "frames": out.out / "post_split_frame_audit.csv",
        "metrics": out.out / "producer_metrics.json",
    }
    if mode == "baseline":
        shutil.copyfile(params["labels_path"], paths["labels"])
    else:
        tifffile.imwrite(paths["labels"], candidate, compression="zlib")
    shutil.copyfile(params["unclaimed_path"], paths["unclaimed"])
    tifffile.imwrite(paths["changed"], candidate != labels,
                     compression="zlib")
    proposals.to_csv(paths["proposals"], index=False)
    applications.to_csv(paths["applications"], index=False)
    frames.to_csv(paths["frames"], index=False)
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": paths, "summary": metrics}

