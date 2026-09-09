"""Recover a bracketed flash borrowed from an owner's later true body.

The ordinary mixed-owner rule rejects any correction that would create a
disconnected same-owner component.  That is normally the right safeguard.
This extension accounts for the one safe exception: a continuously tracked
branch lies beside a movie-long anchor of owner A, briefly borrows owner B,
and B's unique stable physical seat does not acquire B until later.

Discovery is complete-field and parameter-target-free.  Identity and track
numbers are emitted only to the audit ledger; none are accepted as settings.
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

import bounded_owner_excursion as bounded
import bracketed_mixed_owner_flash as flash
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "bracket_owner", "source_owner",
    "middle_first", "middle_last", "middle_frames", "anchor_track",
    "anchor_support", "anchor_purity", "maximum_anchor_step_sum_radii",
    "delayed_owner_track", "delayed_owner_support", "delayed_owner_purity",
    "delayed_owner_first_frame", "maximum_projection_gap_radius",
    "maximum_projection_component_fraction", "explained_components",
    "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "track_id", "frame", "bracket_owner", "source_owner",
    "anchor_track", "delayed_owner_track", "partition_method",
    "changed_pixels", "explained_projection_components",
    "unexplained_duplicate_components", "applied", "reason",
]


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _positive_owner_stats(group: pd.DataFrame, owner: int) -> tuple[int, float]:
    positive = group[
        group.physically_visible.astype(bool)
        & (group.candidate_owner.astype(int) > 0)]
    support = int((positive.candidate_owner.astype(int) == int(owner)).sum())
    purity = float(support / len(positive)) if len(positive) else 0.0
    return support, purity


def _stable_tracks(groups: dict[int, pd.DataFrame], owner: int,
                   minimum_support: int, minimum_purity: float,
                   excluded_track: int) -> list[tuple[int, int, float]]:
    matches = []
    for track, group in groups.items():
        if int(track) == int(excluded_track):
            continue
        support, purity = _positive_owner_stats(group, owner)
        if support >= minimum_support and purity >= minimum_purity:
            matches.append((int(track), support, purity))
    return matches


def discover(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
             params: dict):
    """Audit every general medium-window proposal for the safe exception."""
    flash.assert_target_free(params)
    base_audit, base_internal, attached = flash.discover(
        labels, points, params)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    minimum_anchor_support = int(math.ceil(movie * float(
        params.get("minimum_anchor_support_movie_fraction", 0.75))))
    minimum_delayed_support = int(math.ceil(movie * float(
        params.get("minimum_delayed_owner_support_movie_fraction", 0.20))))
    minimum_anchor_purity = float(params.get(
        "minimum_anchor_owner_purity", 0.95))
    minimum_delayed_purity = float(params.get(
        "minimum_delayed_owner_purity", 0.90))
    maximum_anchor_step = float(params.get(
        "maximum_anchor_step_sum_radii", 1.75))
    maximum_projection_gap = float(params.get(
        "maximum_projection_gap_target_radii", 0.75))
    maximum_projection_fraction = float(params.get(
        "maximum_projection_component_anchor_fraction", 0.75))

    rows: list[dict] = []
    internals: list[dict] = []
    eligible_base = base_internal[base_internal.eligible.astype(bool)] \
        if len(base_internal) else base_internal
    for proposal in eligible_base.itertuples(index=False):
        target_group = groups[int(proposal.track_id)]
        frames = list(map(int, proposal.frame_values))
        targets = [_point(target_group, frame) for frame in frames]
        source_owners = [int(target.candidate_owner)
                         for target in targets if target is not None]
        source_owner = source_owners[0] \
            if len(source_owners) == len(frames) \
            and len(set(source_owners)) == 1 else 0
        reasons: list[str] = []
        if source_owner <= 0:
            reasons.append("middle_owner_not_unique")

        anchors = _stable_tracks(
            groups, int(proposal.bracket_owner), minimum_anchor_support,
            minimum_anchor_purity, int(proposal.track_id))
        if len(anchors) != 1:
            reasons.append("movie_long_bracket_anchor_not_unique")
            anchor = (0, 0, 0.0)
        else:
            anchor = anchors[0]

        delayed = _stable_tracks(
            groups, source_owner, minimum_delayed_support,
            minimum_delayed_purity, int(proposal.track_id)) \
            if source_owner > 0 else []
        delayed = [item for item in delayed if int(item[0]) != int(anchor[0])]
        if len(delayed) != 1:
            reasons.append("delayed_owner_stable_seat_not_unique")
            donor = (0, 0, 0.0)
            donor_first = -1
        else:
            donor = delayed[0]
            donor_group = groups[int(donor[0])]
            owned = donor_group[
                donor_group.physically_visible.astype(bool)
                & donor_group.candidate_owner.astype(int).eq(source_owner)]
            donor_first = int(owned.frame.astype(int).min()) \
                if len(owned) else -1
            if donor_first <= int(proposal.middle_last):
                reasons.append("source_owner_not_delayed_until_after_flash")

        pending = []
        anchor_steps = []
        projection_gaps = []
        projection_fractions = []
        explained = 0
        for frame, target in zip(frames, targets):
            if target is None or int(anchor[0]) <= 0:
                reasons.append("physical_point_missing")
                continue
            anchor_point = _point(groups[int(anchor[0])], frame)
            if (anchor_point is None
                    or not bool(anchor_point.physically_visible)
                    or int(anchor_point.candidate_owner)
                    != int(proposal.bracket_owner)):
                reasons.append("movie_long_anchor_absent_inside_flash")
                continue
            step = float(np.hypot(
                float(target.x) - float(anchor_point.x),
                float(target.y) - float(anchor_point.y))) / max(
                    float(target.radius_px) + float(anchor_point.radius_px), 1.0)
            anchor_steps.append(step)
            if step > maximum_anchor_step:
                reasons.append("projection_not_near_movie_long_anchor")

            frame_attached = attached[
                attached.frame.astype(int).eq(frame)
                & attached.physically_visible.astype(bool)]
            source_component = bounded._component_at(
                labels[frame], source_owner, target)
            if source_component is None:
                reasons.append("source_component_missing")
                continue
            owner_mask = labels[frame] == int(proposal.bracket_owner)
            _, owner_count = ndi.label(owner_mask, STRUCTURE)
            if int(owner_count) != 1:
                reasons.append("bracket_anchor_component_not_unique")
                continue
            gap = float(ndi.distance_transform_edt(~owner_mask)[
                source_component].min()) / max(float(target.radius_px), 1.0)
            fraction = float(source_component.sum() / max(owner_mask.sum(), 1))
            projection_gaps.append(gap)
            projection_fractions.append(fraction)
            if gap > maximum_projection_gap:
                reasons.append("projection_component_too_far_from_anchor_mask")
            if fraction > maximum_projection_fraction:
                reasons.append("projection_component_too_large_for_anchor")

            trial, method, changed, _ = flash._prepare_frame(
                labels[frame], raw[frame], frame_attached, target,
                int(proposal.bracket_owner), params)
            if trial is None:
                reasons.append("projection_partition_unavailable")
                continue
            duplicate_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.bracket_owner))
                - component_accounting.component_excess(
                    labels[frame], int(proposal.bracket_owner)))
            if duplicate_delta not in (0, 1):
                reasons.append("more_than_one_close_projection_created")
            explained += max(0, duplicate_delta)
            pending.append({
                "frame": frame, "trial": trial, "method": method,
                "changed": int(changed), "explained": max(0, duplicate_delta),
            })

        if len(pending) != len(frames):
            reasons.append("flash_not_fully_prepared")
        if explained < 1:
            reasons.append("ordinary_rule_can_account_for_all_frames")
        public = {
            "proposal_id": "", "track_id": int(proposal.track_id),
            "bracket_owner": int(proposal.bracket_owner),
            "source_owner": source_owner,
            "middle_first": int(proposal.middle_first),
            "middle_last": int(proposal.middle_last),
            "middle_frames": int(proposal.middle_frames),
            "anchor_track": int(anchor[0]),
            "anchor_support": int(anchor[1]),
            "anchor_purity": float(anchor[2]),
            "maximum_anchor_step_sum_radii": max(anchor_steps, default=float("inf")),
            "delayed_owner_track": int(donor[0]),
            "delayed_owner_support": int(donor[1]),
            "delayed_owner_purity": float(donor[2]),
            "delayed_owner_first_frame": donor_first,
            "maximum_projection_gap_radius": max(
                projection_gaps, default=float("inf")),
            "maximum_projection_component_fraction": max(
                projection_fractions, default=float("inf")),
            "explained_components": explained,
            "eligible": not reasons,
            "reason": ("eligible_delayed_owner_projection_flash"
                       if not reasons else "|".join(dict.fromkeys(reasons))),
        }
        rows.append(public)
        internals.append({**public, "frame_details": pending})

    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"DOP{number:04d}"
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internals), attached, base_audit)


def apply(labels: np.ndarray, internal: pd.DataFrame):
    """Apply complete proofs atomically and ledger explained components."""
    candidate = labels.copy()
    rows = []
    applied = explained = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending = list(proposal.frame_details)
        valid = len(pending) == int(proposal.middle_frames)
        for detail in pending:
            frame = int(detail["frame"])
            delta = (
                component_accounting.component_excess(
                    detail["trial"], int(proposal.bracket_owner))
                - component_accounting.component_excess(
                    candidate[frame], int(proposal.bracket_owner)))
            valid = valid and delta == int(detail["explained"])
        if not valid:
            rows.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "frame": int(proposal.middle_first),
                "bracket_owner": int(proposal.bracket_owner),
                "source_owner": int(proposal.source_owner),
                "anchor_track": int(proposal.anchor_track),
                "delayed_owner_track": int(proposal.delayed_owner_track),
                "partition_method": "atomic", "changed_pixels": 0,
                "explained_projection_components": 0,
                "unexplained_duplicate_components": 0,
                "applied": False,
                "reason": "atomic_refusal:projection_proof_changed",
            })
            continue
        for detail in pending:
            frame = int(detail["frame"])
            candidate[frame] = detail["trial"]
            frame_explained = int(detail["explained"])
            explained += frame_explained
            rows.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id), "frame": frame,
                "bracket_owner": int(proposal.bracket_owner),
                "source_owner": int(proposal.source_owner),
                "anchor_track": int(proposal.anchor_track),
                "delayed_owner_track": int(proposal.delayed_owner_track),
                "partition_method": detail["method"],
                "changed_pixels": int(detail["changed"]),
                "explained_projection_components": frame_explained,
                "unexplained_duplicate_components": 0,
                "applied": True,
                "reason": "delayed_owner_removed_from_bracketed_projection",
            })
        applied += 1
    changed = candidate != labels
    actual_duplicates = int(component_accounting.new_duplicate_components(
        labels, candidate))
    unexplained = max(0, actual_duplicates - explained)
    return candidate, pd.DataFrame(rows, columns=APPLICATION_COLUMNS), {
        "applied_proposals": applied,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_explained_projection_components": explained,
        "new_duplicate_components": unexplained,
    }


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    """Analysis-tuner adapter."""
    if upstream_dir is None:
        raise ValueError("delayed-owner stage requires upstream labels")
    flash.assert_target_free(params)
    labels_path = Path(upstream_dir) / str(params["labels_name"])
    unclaimed_path = Path(upstream_dir) / str(params["unclaimed_name"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    offset = int(params.get("raw_frame_offset", 0))
    raw = raw[offset:offset + len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal, attached, base_audit = discover(
        labels, raw, points, params)
    candidate, applications, details = apply(labels, internal)

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("delayed-owner projection flash changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("delayed-owner projection overlaps unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("delayed-owner projection changed identity set")
    if int(details["new_duplicate_components"]):
        raise AssertionError("unexplained duplicate component was created")

    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
        "audit": out.out / "delayed_owner_projection_flash_audit.csv",
        "applications": out.out / "delayed_owner_projection_flash_applications.csv",
        "base_audit": out.out / "medium_window_base_audit.csv",
        "metrics": out.out / "producer_metrics.json",
    }
    tifffile.imwrite(outputs["labels"], candidate, compression="zlib")
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    audit.to_csv(outputs["audit"], index=False)
    applications.to_csv(outputs["applications"], index=False)
    base_audit.to_csv(outputs["base_audit"], index=False)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "physical_tracks_audited": int(points.track_id.nunique()),
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        **details,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": (
            outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes()),
        "identity_set_exact": before_ids == after_ids,
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

