"""Retire short identities made only of nearby projection-sized speckles.

An identity is retired only when it has no persistent physical seat, its whole
label lifespan is short, most of its owner-frames are fragmented, and every
component is a small neighbour of an owner supported by a unique movie-long
physical anchor.  Every pixel is reassigned component-by-component to the
nearest proved durable owner; foreground is unchanged.
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

import bracketed_mixed_owner_flash as flash
import component_accounting
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
AUDIT_COLUMNS = [
    "proposal_id", "alias_owner", "first_frame", "last_frame", "span_frames",
    "named_frames", "component_count", "fragmented_owner_frames",
    "fragmented_owner_frame_fraction", "maximum_physical_track_support",
    "durable_destination_owners", "maximum_component_destination_gap_radii",
    "maximum_component_destination_area_fraction", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "alias_owner", "frame", "component_index",
    "destination_owner", "component_pixels", "destination_component_pixels",
    "destination_gap_radii", "component_destination_area_fraction",
    "explained_projection_components", "applied", "reason",
]


def _evidence_rows(group: pd.DataFrame) -> pd.DataFrame:
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


def _durable_owners(attached: pd.DataFrame, movie: int,
                    params: dict) -> dict[int, tuple[int, int, float]]:
    minimum_support = int(math.ceil(movie * float(params.get(
        "minimum_durable_support_movie_fraction", 0.75))))
    minimum_purity = float(params.get("minimum_durable_owner_purity", 0.98))
    by_owner: dict[int, list[tuple[int, int, float]]] = {}
    for track, group in attached.groupby("track_id", sort=True):
        positive = _evidence_rows(group)
        positive = positive[positive.candidate_owner.astype(int) > 0]
        if len(positive) < minimum_support:
            continue
        counts = positive.candidate_owner.astype(int).value_counts()
        for owner, support in counts.items():
            purity = float(int(support) / len(positive))
            if int(support) >= minimum_support and purity >= minimum_purity:
                by_owner.setdefault(int(owner), []).append(
                    (int(track), int(support), purity))
    return {owner: rows[0] for owner, rows in by_owner.items()
            if len(rows) == 1}


def _nearest_destination(frame: np.ndarray, component: np.ndarray,
                         durable: dict[int, tuple[int, int, float]],
                         cache: dict):
    """Find the nearest durable-owner pixel with one EDT per movie frame."""
    radius = math.sqrt(float(component.sum()) / math.pi)
    if "distance" not in cache:
        mask = np.isin(frame, np.asarray(sorted(durable), dtype=frame.dtype))
        if not np.any(mask):
            return None
        distance, indices = ndi.distance_transform_edt(
            ~mask, return_indices=True)
        cache["distance"] = distance
        cache["indices"] = indices
        cache["components"] = {}
    values = cache["distance"][component]
    if not len(values):
        return None
    component_positions = np.argwhere(component)
    selected = component_positions[int(np.argmin(values))]
    nearest_y = int(cache["indices"][0, selected[0], selected[1]])
    nearest_x = int(cache["indices"][1, selected[0], selected[1]])
    owner = int(frame[nearest_y, nearest_x])
    if owner <= 0 or owner not in durable:
        return None
    if owner not in cache["components"]:
        cache["components"][owner] = ndi.label(frame == owner, STRUCTURE)[0]
    owner_components = cache["components"][owner]
    component_id = int(owner_components[nearest_y, nearest_x])
    destination_area = int(np.count_nonzero(owner_components == component_id))
    gap = float(np.min(values))
    return (gap / max(radius, 1.0), gap, owner, destination_area)


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict):
    flash.assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    movie = int(len(labels))
    durable = _durable_owners(attached, movie, params)
    maximum_span = int(math.ceil(movie * float(params.get(
        "maximum_alias_span_movie_fraction", 0.15))))
    maximum_support = int(math.ceil(movie * float(params.get(
        "maximum_alias_physical_support_movie_fraction", 0.02))))
    minimum_fragmented = float(params.get(
        "minimum_fragmented_owner_frame_fraction", 0.50))
    maximum_gap = float(params.get(
        "maximum_component_destination_gap_equivalent_radii", 2.0))
    maximum_area_fraction = float(params.get(
        "maximum_component_destination_area_fraction", 0.40))

    support_by_owner: dict[int, int] = {}
    for _, group in attached.groupby("track_id", sort=True):
        positive = _evidence_rows(group)
        positive = positive[positive.candidate_owner.astype(int) > 0]
        if positive.empty:
            continue
        counts = positive.candidate_owner.astype(int).value_counts()
        for owner, support in counts.items():
            support_by_owner[int(owner)] = max(
                support_by_owner.get(int(owner), 0), int(support))

    public_rows: list[dict] = []
    internal_rows: list[dict] = []
    destination_cache: dict[int, dict] = {}
    for alias_owner in sorted(set(map(int, np.unique(labels))) - {0}):
        present = np.flatnonzero(np.any(
            labels == int(alias_owner), axis=tuple(range(1, labels.ndim))))
        first, last = int(present[0]), int(present[-1])
        span = last - first + 1
        component_rows = []
        fragmented = 0
        maximum_seen_gap = 0.0
        maximum_seen_fraction = 0.0
        destinations = set()
        reasons: list[str] = []
        if span > maximum_span:
            reasons.append("identity_lifespan_not_short")
        owner_support = int(support_by_owner.get(alias_owner, 0))
        if owner_support > maximum_support:
            reasons.append("identity_has_persistent_physical_seat")
        # Long-lived identities and identities with a genuine physical seat
        # cannot become aliases, so avoid unnecessary image-distance work.
        if reasons:
            public = {
                "proposal_id": "", "alias_owner": alias_owner,
                "first_frame": first, "last_frame": last,
                "span_frames": span, "named_frames": int(len(present)),
                "component_count": 0, "fragmented_owner_frames": 0,
                "fragmented_owner_frame_fraction": 0.0,
                "maximum_physical_track_support": owner_support,
                "durable_destination_owners": "",
                "maximum_component_destination_gap_radii": 0.0,
                "maximum_component_destination_area_fraction": 0.0,
                "eligible": False,
                "reason": "|".join(dict.fromkeys(reasons)),
            }
            public_rows.append(public)
            internal_rows.append({**public, "component_rows": []})
            continue
        for frame_index in present:
            components, count = ndi.label(
                labels[int(frame_index)] == alias_owner, STRUCTURE)
            fragmented += int(count > 1)
            for component_index in range(1, int(count) + 1):
                component = components == component_index
                destination = _nearest_destination(
                    labels[int(frame_index)], component, durable,
                    destination_cache.setdefault(int(frame_index), {}))
                if destination is None:
                    reasons.append("no_durable_destination_owner")
                    continue
                gap_radii, _, destination_owner, destination_area = destination
                area_fraction = float(component.sum() / max(destination_area, 1))
                maximum_seen_gap = max(maximum_seen_gap, float(gap_radii))
                maximum_seen_fraction = max(
                    maximum_seen_fraction, area_fraction)
                destinations.add(int(destination_owner))
                if gap_radii > maximum_gap:
                    reasons.append("component_too_far_from_durable_owner")
                if area_fraction > maximum_area_fraction:
                    reasons.append("component_not_projection_sized")
                component_rows.append({
                    "frame": int(frame_index),
                    "component_index": int(component_index),
                    "component": component,
                    "destination_owner": int(destination_owner),
                    "component_pixels": int(component.sum()),
                    "destination_component_pixels": int(destination_area),
                    "destination_gap_radii": float(gap_radii),
                    "component_destination_area_fraction": area_fraction,
                })
        fragmented_fraction = float(fragmented / len(present))
        if fragmented_fraction < minimum_fragmented:
            reasons.append("identity_not_recurrently_fragmented")
        if not component_rows:
            reasons.append("no_components_prepared")
        public = {
            "proposal_id": "", "alias_owner": alias_owner,
            "first_frame": first, "last_frame": last, "span_frames": span,
            "named_frames": int(len(present)),
            "component_count": int(len(component_rows)),
            "fragmented_owner_frames": int(fragmented),
            "fragmented_owner_frame_fraction": fragmented_fraction,
            "maximum_physical_track_support": owner_support,
            "durable_destination_owners": "|".join(map(str, sorted(destinations))),
            "maximum_component_destination_gap_radii": maximum_seen_gap,
            "maximum_component_destination_area_fraction": maximum_seen_fraction,
            "eligible": not reasons,
            "reason": ("eligible_ephemeral_speckle_alias_retirement"
                       if not reasons else "|".join(dict.fromkeys(reasons))),
        }
        public_rows.append(public)
        internal_rows.append({**public, "component_rows": component_rows})

    number = 0
    for public, internal in zip(public_rows, internal_rows):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"ESR{number:04d}"
    return (pd.DataFrame(public_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internal_rows), attached)


def apply(labels: np.ndarray, internal: pd.DataFrame):
    candidate = labels.copy()
    applications = []
    applied = explained = 0
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending = []
        valid = True
        working = candidate.copy()
        for detail in list(proposal.component_rows):
            frame = int(detail["frame"])
            component = detail["component"]
            if not np.all(working[frame][component] == int(proposal.alias_owner)):
                valid = False
                break
            trial = working[frame].copy()
            before_excess = component_accounting.component_excess(
                trial, int(detail["destination_owner"]))
            trial[component] = int(detail["destination_owner"])
            after_excess = component_accounting.component_excess(
                trial, int(detail["destination_owner"]))
            working[frame] = trial
            pending.append({**detail,
                            "explained": max(0, after_excess - before_excess)})
        if not valid or len(pending) != int(proposal.component_count):
            continue
        candidate = working
        for detail in pending:
            frame = int(detail["frame"])
            explained += int(detail["explained"])
            applications.append({
                "proposal_id": proposal.proposal_id,
                "alias_owner": int(proposal.alias_owner), "frame": frame,
                "component_index": int(detail["component_index"]),
                "destination_owner": int(detail["destination_owner"]),
                "component_pixels": int(detail["component_pixels"]),
                "destination_component_pixels": int(
                    detail["destination_component_pixels"]),
                "destination_gap_radii": float(detail["destination_gap_radii"]),
                "component_destination_area_fraction": float(
                    detail["component_destination_area_fraction"]),
                "explained_projection_components": int(detail["explained"]),
                "applied": True,
                "reason": "ephemeral_speckle_alias_retired_to_nearest_durable_owner",
            })
        applied += 1
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    actual_duplicates = component_accounting.new_duplicate_components(
        labels, candidate)
    changed = labels != candidate
    return candidate, pd.DataFrame(applications, columns=APPLICATION_COLUMNS), {
        "applied_proposals": int(applied),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "new_explained_projection_components": int(explained),
        "new_duplicate_components": max(0, int(actual_duplicates) - int(explained)),
    }


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("ephemeral alias retirement requires upstream labels")
    flash.assert_target_free(params)
    labels_path = Path(upstream_dir) / str(params["labels_name"])
    unclaimed_path = Path(upstream_dir) / str(params["unclaimed_name"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal, _ = discover(labels, points, params)
    candidate, applications, details = apply(labels, internal)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("ephemeral alias retirement changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("ephemeral alias retirement overlaps unclaimed")
    if int(details["new_duplicate_components"]):
        raise AssertionError("unexplained duplicate component was created")

    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
        "audit": out.out / "ephemeral_speckle_alias_retirement_audit.csv",
        "applications": out.out / "ephemeral_speckle_alias_retirement_applications.csv",
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
        "identities_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        **details,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": (
            outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes()),
    }
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

