"""Recover small branches that lose the owner of their durable anchor.

Discovery is complete-field and accepts mathematical thresholds only.  A
branch is learned from a long initial interval in which it shares one mask
component with a unique movie-long owner anchor.  Later positive foreign-owner
components may be returned to that anchor when the branch remains small and
nearby.  Ownerless frames are deliberately left unchanged.
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
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "anchor_owner", "anchor_track",
    "track_frames", "visible_frames", "initial_first", "initial_last",
    "initial_frames", "initial_shared_component_frames",
    "initial_shared_component_fraction", "anchor_support", "anchor_purity",
    "anchor_presence_fraction", "anchor_owner_plurality",
    "foreign_positive_frames", "foreign_owners", "target_anchor_radius_ratio",
    "maximum_anchor_distance_sum_radii", "maximum_target_step_sum_radii",
    "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "track_id", "frame", "anchor_owner", "source_owner",
    "anchor_track", "source_stable_track", "partition_method",
    "source_component_pixels", "source_component_disk_ratio",
    "anchor_gap_target_radii", "changed_pixels",
    "explained_projection_components", "unexplained_duplicate_components",
    "applied", "reason",
]


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int).eq(int(frame))]
    return rows.iloc[0] if len(rows) == 1 else None


def _scaled_distance(left, right) -> float:
    return float(np.hypot(
        float(left.x) - float(right.x),
        float(left.y) - float(right.y))) / max(
            float(left.radius_px) + float(right.radius_px), 1.0)


def _evidence_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Include observed rows and only tightly bracketed interpolated rows."""
    ordered = group.sort_values("frame").copy()
    allowed = ordered.physically_visible.astype(bool).to_numpy()
    owners = ordered.candidate_owner.astype(int).to_numpy()
    frames = ordered.frame.astype(int).to_numpy()
    for index in range(1, len(ordered) - 1):
        if allowed[index] or owners[index] <= 0:
            continue
        if (allowed[index - 1] and allowed[index + 1]
                and frames[index - 1] + 1 == frames[index]
                and frames[index] + 1 == frames[index + 1]
                and owners[index - 1] == owners[index] == owners[index + 1]):
            allowed[index] = True
    return ordered.loc[allowed].copy()


def _owner_support(group: pd.DataFrame, owner: int) -> tuple[int, float]:
    rows = _evidence_rows(group)
    positive = rows[rows.candidate_owner.astype(int) > 0]
    support = int(positive.candidate_owner.astype(int).eq(int(owner)).sum())
    return support, float(support / len(positive)) if len(positive) else 0.0


def _stable_tracks(groups: dict[int, pd.DataFrame], owner: int,
                   minimum_support: int, minimum_purity: float,
                   excluded: int) -> list[tuple[int, int, float]]:
    found = []
    for track, group in groups.items():
        if int(track) == int(excluded):
            continue
        support, purity = _owner_support(group, owner)
        if support >= minimum_support and purity >= minimum_purity:
            found.append((int(track), support, purity))
    return found


def _initial_positive_run(group: pd.DataFrame) -> tuple[int, list[int]]:
    evidence = _evidence_rows(group)
    positive = evidence[evidence.candidate_owner.astype(int) > 0]
    if positive.empty:
        return 0, []
    first = positive.iloc[0]
    owner = int(first.candidate_owner)
    frames = [int(first.frame)]
    for row in positive.iloc[1:].itertuples(index=False):
        if (int(row.frame) != frames[-1] + 1
                or int(row.candidate_owner) != owner):
            break
        frames.append(int(row.frame))
    return owner, frames


def _same_component(frame: np.ndarray, owner: int, left, right) -> bool:
    a = bounded._component_at(frame, owner, left)
    b = bounded._component_at(frame, owner, right)
    return a is not None and b is not None and bool(np.any(a & b))


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict):
    """Audit all physical tracks for the anchored-projection signature."""
    flash.assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    evidence_groups = {track: _evidence_rows(group)
                       for track, group in groups.items()}
    movie = int(len(labels))
    minimum_anchor_support = int(math.ceil(movie * float(
        params.get("minimum_anchor_support_movie_fraction", 0.75))))
    minimum_initial = int(math.ceil(movie * float(
        params.get("minimum_initial_shared_tenure_movie_fraction", 0.08))))
    minimum_visible = int(math.ceil(movie * float(
        params.get("minimum_target_visible_movie_fraction", 0.15))))
    minimum_foreign = int(math.ceil(movie * float(
        params.get("minimum_foreign_positive_movie_fraction", 0.02))))
    minimum_anchor_purity = float(params.get("minimum_anchor_purity", 0.98))
    minimum_shared = float(params.get(
        "minimum_initial_shared_component_fraction", 0.75))
    minimum_presence = float(params.get(
        "minimum_anchor_presence_target_span_fraction", 0.90))
    minimum_plurality = float(params.get("minimum_anchor_owner_plurality", 0.50))
    maximum_radius_ratio = float(params.get(
        "maximum_target_anchor_radius_ratio", 0.65))
    maximum_anchor_distance = float(params.get(
        "maximum_anchor_distance_sum_radii", 2.25))
    maximum_step = float(params.get("maximum_target_step_sum_radii", 1.0))

    # Build the stable-anchor index once. Recomputing support for every
    # target/anchor pair is quadratic in the number of physical tracks.
    stable_by_owner: dict[int, list[tuple[int, int, float]]] = {}
    for candidate_track, evidence_group in evidence_groups.items():
        positive = evidence_group[
            evidence_group.candidate_owner.astype(int) > 0]
        if len(positive) < minimum_anchor_support:
            continue
        counts = positive.candidate_owner.astype(int).value_counts()
        for owner, support in counts.items():
            purity = float(int(support) / len(positive))
            if int(support) >= minimum_anchor_support \
                    and purity >= minimum_anchor_purity:
                stable_by_owner.setdefault(int(owner), []).append(
                    (int(candidate_track), int(support), purity))

    public_rows: list[dict] = []
    internal_rows: list[dict] = []
    for track, group in groups.items():
        evidence = evidence_groups[track]
        visible = group[group.physically_visible.astype(bool)]
        anchor_owner, initial_frames = _initial_positive_run(group)
        reasons: list[str] = []
        if anchor_owner <= 0:
            reasons.append("no_initial_positive_owner")
        first_track_frame = int(group.frame.astype(int).min())
        if initial_frames and initial_frames[0] != first_track_frame:
            reasons.append("initial_owner_does_not_begin_with_track")
        if len(initial_frames) < minimum_initial:
            reasons.append("initial_same_owner_tenure_too_short")
        if len(visible) < minimum_visible:
            reasons.append("target_visible_tenure_too_short")

        stable = [item for item in stable_by_owner.get(anchor_owner, [])
                  if int(item[0]) != int(track)]
        if len(stable) != 1:
            reasons.append("unique_movie_long_anchor_not_found")
            anchor = (0, 0, 0.0)
        else:
            anchor = stable[0]
        anchor_group = groups.get(int(anchor[0]), pd.DataFrame())

        shared = 0
        for frame in initial_frames:
            target_point = _point(group, frame)
            anchor_point = _point(anchor_group, frame) if len(anchor_group) else None
            if (target_point is not None and anchor_point is not None
                    and int(anchor_point.candidate_owner) == anchor_owner
                    and _same_component(labels[frame], anchor_owner,
                                        target_point, anchor_point)):
                shared += 1
        shared_fraction = float(shared / len(initial_frames)) \
            if initial_frames else 0.0
        if shared_fraction < minimum_shared:
            reasons.append("initial_shared_component_evidence_too_weak")

        positive = evidence[evidence.candidate_owner.astype(int) > 0]
        anchor_count = int(positive.candidate_owner.astype(int).eq(
            anchor_owner).sum()) if anchor_owner > 0 else 0
        plurality = float(anchor_count / len(positive)) if len(positive) else 0.0
        if plurality < minimum_plurality:
            reasons.append("initial_owner_not_positive_plurality")
        initial_last = max(initial_frames, default=-1)
        foreign = positive[
            (positive.frame.astype(int) > initial_last)
            & positive.candidate_owner.astype(int).ne(anchor_owner)]
        if len(foreign) < minimum_foreign:
            reasons.append("too_few_later_foreign_owner_frames")

        anchor_presence = 0
        anchor_distances: list[float] = []
        radius_ratios: list[float] = []
        evidence_span = evidence[evidence.frame.astype(int) >= min(
            initial_frames, default=first_track_frame)]
        for point in evidence_span.itertuples(index=False):
            anchor_point = _point(anchor_group, int(point.frame)) \
                if len(anchor_group) else None
            if (anchor_point is None
                    or not bool(anchor_point.physically_visible)
                    or int(anchor_point.candidate_owner) != anchor_owner):
                continue
            anchor_presence += 1
            anchor_distances.append(_scaled_distance(point, anchor_point))
            radius_ratios.append(float(point.radius_px) / max(
                float(anchor_point.radius_px), 1.0))
        presence_fraction = float(anchor_presence / len(evidence_span)) \
            if len(evidence_span) else 0.0
        radius_ratio = float(np.median(radius_ratios)) \
            if radius_ratios else float("inf")
        maximum_distance = max(anchor_distances, default=float("inf"))
        if presence_fraction < minimum_presence:
            reasons.append("anchor_not_continuous_beside_target")
        if radius_ratio > maximum_radius_ratio:
            reasons.append("target_not_smaller_than_anchor")
        if maximum_distance > maximum_anchor_distance:
            reasons.append("target_moves_too_far_from_anchor")

        steps = [_scaled_distance(left, right)
                 for left, right in zip(
                     evidence_span.itertuples(index=False),
                     evidence_span.iloc[1:].itertuples(index=False))
                 if int(right.frame) == int(left.frame) + 1]
        maximum_seen_step = max(steps, default=0.0)
        if maximum_seen_step > maximum_step:
            reasons.append("target_motion_not_continuous")

        public = {
            "proposal_id": "", "track_id": int(track),
            "anchor_owner": int(anchor_owner), "anchor_track": int(anchor[0]),
            "track_frames": int(len(group)), "visible_frames": int(len(visible)),
            "initial_first": min(initial_frames, default=-1),
            "initial_last": initial_last, "initial_frames": len(initial_frames),
            "initial_shared_component_frames": shared,
            "initial_shared_component_fraction": shared_fraction,
            "anchor_support": int(anchor[1]), "anchor_purity": float(anchor[2]),
            "anchor_presence_fraction": presence_fraction,
            "anchor_owner_plurality": plurality,
            "foreign_positive_frames": int(len(foreign)),
            "foreign_owners": "|".join(map(str, sorted(set(
                foreign.candidate_owner.astype(int).tolist())))),
            "target_anchor_radius_ratio": radius_ratio,
            "maximum_anchor_distance_sum_radii": maximum_distance,
            "maximum_target_step_sum_radii": maximum_seen_step,
            "eligible": not reasons,
            "reason": ("eligible_anchored_projection_owner_relay"
                       if not reasons else "|".join(dict.fromkeys(reasons))),
        }
        public_rows.append(public)
        internal_rows.append({
            **public,
            "foreign_frames": foreign.frame.astype(int).tolist(),
        })

    number = 0
    for public, internal in zip(public_rows, internal_rows):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"APR{number:04d}"
    return (pd.DataFrame(public_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internal_rows), attached)


def _anchor_component(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    return bounded._component_at(frame, owner, point)


def _component_gap(component: np.ndarray, anchor: np.ndarray,
                   radius: float) -> float:
    if np.any(component & anchor):
        return 0.0
    return float(ndi.distance_transform_edt(~anchor)[component].min()) / max(
        float(radius), 1.0)


def apply(labels: np.ndarray, raw: np.ndarray, attached: pd.DataFrame,
          internal: pd.DataFrame, params: dict):
    """Apply each complete relay proof atomically; keep ownerless rows intact."""
    candidate = labels.copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    movie = int(len(labels))
    minimum_source_support = int(math.ceil(movie * float(params.get(
        "minimum_source_anchor_support_movie_fraction", 0.75))))
    minimum_source_purity = float(params.get("minimum_source_anchor_purity", 0.98))
    maximum_disk_ratio = float(params.get(
        "maximum_isolated_component_expected_disk_ratio", 3.0))
    maximum_gap = float(params.get("maximum_anchor_gap_target_radii", 2.0))
    rows: list[dict] = []
    applied = explained = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending: list[dict] = []
        failure = ""
        target_group = groups[int(proposal.track_id)]
        anchor_group = groups[int(proposal.anchor_track)]
        for frame in list(proposal.foreign_frames):
            target = _point(target_group, int(frame))
            anchor_point = _point(anchor_group, int(frame))
            if target is None or anchor_point is None:
                failure = "physical_point_missing"
                break
            source_owner = int(target.candidate_owner)
            if source_owner <= 0:
                failure = "ownerless_frame_would_be_modified"
                break
            source_component = bounded._component_at(
                candidate[int(frame)], source_owner, target)
            anchor_component = _anchor_component(
                candidate[int(frame)], int(proposal.anchor_owner), anchor_point)
            if source_component is None or anchor_component is None:
                failure = "source_or_anchor_component_missing"
                break
            gap = _component_gap(
                source_component, anchor_component, float(target.radius_px))
            if gap > maximum_gap:
                failure = "foreign_component_too_far_from_anchor"
                break
            area = int(source_component.sum())
            disk_ratio = float(area / max(
                math.pi * float(target.radius_px) ** 2, 1.0))

            frame_rows = attached[
                attached.frame.astype(int).eq(int(frame))
                & attached.physically_visible.astype(bool)]
            companion_tracks = []
            for point in frame_rows[
                    frame_rows.candidate_owner.astype(int).eq(source_owner)] \
                    .itertuples(index=False):
                if int(point.track_id) == int(proposal.track_id):
                    continue
                marker = bounded._marker(source_component, point)
                if marker is not None:
                    companion_tracks.append(int(point.track_id))
            source_stable_track = 0
            if companion_tracks:
                stable_source = _stable_tracks(
                    groups, source_owner, minimum_source_support,
                    minimum_source_purity, int(proposal.track_id))
                stable_inside = [item for item in stable_source
                                 if item[0] in companion_tracks]
                if len(stable_inside) != 1:
                    failure = "merged_source_lacks_unique_stable_companion"
                    break
                source_stable_track = int(stable_inside[0][0])
            elif disk_ratio > maximum_disk_ratio:
                failure = "isolated_foreign_component_not_projection_sized"
                break

            trial, method, changed, companions = flash._prepare_frame(
                candidate[int(frame)], raw[int(frame)], frame_rows, target,
                int(proposal.anchor_owner), params)
            if trial is None:
                failure = method
                break
            if companion_tracks and companions <= 0:
                failure = "merged_source_companion_not_partitioned"
                break
            duplicate_delta = (
                component_accounting.component_excess(
                    trial, int(proposal.anchor_owner))
                - component_accounting.component_excess(
                    candidate[int(frame)], int(proposal.anchor_owner)))
            if duplicate_delta not in (0, 1):
                failure = "more_than_one_projection_component_would_be_created"
                break
            pending.append({
                "frame": int(frame), "trial": trial,
                "source_owner": source_owner,
                "source_stable_track": source_stable_track,
                "method": method, "area": area, "disk_ratio": disk_ratio,
                "gap": gap, "changed": int(changed),
                "explained": max(0, int(duplicate_delta)),
            })

        if failure or len(pending) != int(proposal.foreign_positive_frames):
            rows.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "frame": int(proposal.initial_last),
                "anchor_owner": int(proposal.anchor_owner),
                "source_owner": 0, "anchor_track": int(proposal.anchor_track),
                "source_stable_track": 0, "partition_method": "atomic",
                "source_component_pixels": 0,
                "source_component_disk_ratio": 0.0,
                "anchor_gap_target_radii": 0.0, "changed_pixels": 0,
                "explained_projection_components": 0,
                "unexplained_duplicate_components": 0, "applied": False,
                "reason": "atomic_refusal:" + (failure or "incomplete_relay"),
            })
            continue
        for detail in pending:
            candidate[detail["frame"]] = detail["trial"]
            explained += int(detail["explained"])
            rows.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id),
                "frame": detail["frame"],
                "anchor_owner": int(proposal.anchor_owner),
                "source_owner": detail["source_owner"],
                "anchor_track": int(proposal.anchor_track),
                "source_stable_track": detail["source_stable_track"],
                "partition_method": detail["method"],
                "source_component_pixels": detail["area"],
                "source_component_disk_ratio": detail["disk_ratio"],
                "anchor_gap_target_radii": detail["gap"],
                "changed_pixels": detail["changed"],
                "explained_projection_components": detail["explained"],
                "unexplained_duplicate_components": 0, "applied": True,
                "reason": "anchored_projection_owner_relay",
            })
        applied += 1

    changed = candidate != labels
    actual_duplicates = component_accounting.new_duplicate_components(
        labels, candidate)
    unexplained = max(0, int(actual_duplicates) - int(explained))
    details = {
        "applied_proposals": applied,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": 0, "removed_identity_count": 0,
        "new_explained_projection_components": int(explained),
        "new_duplicate_components": int(unexplained),
    }
    return candidate, pd.DataFrame(rows, columns=APPLICATION_COLUMNS), details


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    flash.assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    offset = int(params.get("raw_frame_offset", 0))
    raw = raw[offset:offset + len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal, attached = discover(labels, points, params)
    candidate, applications, details = apply(
        labels, raw, attached, internal, params)

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("anchored projection relay changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("anchored projection relay overlaps unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("anchored projection relay changed identity set")
    if int(details["new_duplicate_components"]):
        raise AssertionError("unexplained duplicate component was created")

    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
        "audit": out.out / "anchored_projection_owner_relay_audit.csv",
        "applications": out.out / "anchored_projection_owner_relay_applications.csv",
        "metrics": out.out / "producer_metrics.json",
    }
    tifffile.imwrite(outputs["labels"], candidate, compression="zlib")
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    audit.to_csv(outputs["audit"], index=False)
    applications.to_csv(outputs["applications"], index=False)
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

