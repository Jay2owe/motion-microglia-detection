"""Production correction of field-wide physical-body owner substitutions."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import component_accounting
import tifffile


FORBIDDEN_TARGETS = (
    "review_cases_path", "event_ids", "identity_ids", "frame_ids",
    "coordinates", "forced_identity_ids", "forced_intervals",
)


def _disk_owner(labels: np.ndarray, x: float, y: float,
                radius: float) -> tuple[int, float, float]:
    height, width = labels.shape
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    x0, x1 = max(0, int(np.floor(x - radius))), min(
        width, int(np.ceil(x + radius + 1)))
    y0, y1 = max(0, int(np.floor(y - radius))), min(
        height, int(np.ceil(y + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    values = labels[y0:y1, x0:x1][disk]
    positive = values[values > 0]
    coverage = float(len(positive) / max(len(values), 1))
    if not len(positive):
        return 0, coverage, 0.0
    owners, counts = np.unique(positive, return_counts=True)
    best = int(np.argmax(counts))
    return int(owners[best]), coverage, float(counts[best] / len(positive))


def attach_owners(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = points.copy()
    owner, coverage, purity = [], [], []
    for row in rows.itertuples(index=False):
        values = _disk_owner(labels[int(row.frame)], float(row.x), float(row.y),
                             float(row.radius_px))
        owner.append(values[0]); coverage.append(values[1]); purity.append(values[2])
    rows["candidate_owner"] = owner
    rows["candidate_coverage_fraction"] = coverage
    rows["candidate_owner_purity"] = purity
    rows["physically_visible"] = rows["state"].isin(
        ["observed", "latent_visible"])
    return rows


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.candidate_owner)
        if (runs and runs[-1]["owner"] == owner and
                int(row.frame) == runs[-1]["last_frame"] + 1):
            runs[-1]["last_frame"] = int(row.frame)
            runs[-1]["rows"].append(row)
        else:
            runs.append({"owner": owner, "first_frame": int(row.frame),
                         "last_frame": int(row.frame), "rows": [row]})
    for run in runs:
        run["frames"] = len(run["rows"])
    return runs


def _near_positive(runs: list[dict], index: int, direction: int,
                   maximum_gap: int) -> dict | None:
    source = runs[index]
    cursor = index + direction
    while 0 <= cursor < len(runs):
        candidate = runs[cursor]
        if direction < 0:
            gap = source["first_frame"] - candidate["last_frame"] - 1
        else:
            gap = candidate["first_frame"] - source["last_frame"] - 1
        if gap > maximum_gap:
            return None
        if candidate["owner"] > 0:
            return candidate
        cursor += direction
    return None


def discover_proposals(points: pd.DataFrame, params: dict) -> pd.DataFrame:
    visible = points[points["physically_visible"].astype(bool)].copy()
    rows: list[dict] = []
    minimum_support = int(params.get("minimum_canonical_support_frames", 8))
    minimum_ratio = float(params.get("minimum_support_ratio", 2.0))
    maximum_source = int(params.get("maximum_source_run_frames", 12))
    maximum_gap = int(params.get("maximum_adjoining_gap_frames", 6))
    minimum_purity = float(params.get("minimum_owner_purity", 0.8))
    minimum_coverage = float(params.get("minimum_owner_coverage", 0.5))
    minimum_isolation = float(params.get("minimum_track_isolation_px", 10.0))

    for track_id, group in visible.groupby("track_id"):
        runs = _runs(group)
        positive = [run for run in runs if run["owner"] > 0]
        support: dict[int, int] = {}
        for run in positive:
            support[run["owner"]] = support.get(run["owner"], 0) + run["frames"]
        if not support:
            continue
        canonical = min(support, key=lambda owner: (-support[owner], owner))
        canonical_support = support[canonical]
        for index, run in enumerate(runs):
            source = int(run["owner"])
            if source <= 0 or source == canonical:
                continue
            before = _near_positive(runs, index, -1, maximum_gap)
            after = _near_positive(runs, index, +1, maximum_gap)
            before_owner = int(before["owner"]) if before else 0
            after_owner = int(after["owner"]) if after else 0
            ratio = canonical_support / max(support.get(source, 0), 1)
            reasons: list[str] = []
            if canonical_support < minimum_support:
                reasons.append("insufficient_canonical_support")
            if ratio < minimum_ratio:
                reasons.append("insufficient_support_ratio")
            if run["frames"] > maximum_source:
                reasons.append("source_run_too_long")
            if before_owner == canonical and after_owner == canonical:
                reasons.append("bracketed_owner_excursion")
            elif before_owner != canonical and after_owner != canonical:
                reasons.append("not_adjoining_canonical_owner")
            values = pd.DataFrame([row._asdict() for row in run["rows"]])
            if bool(values["in_encounter"].astype(bool).any()):
                reasons.append("encounter_associated")
            if float(values["candidate_owner_purity"].median()) < minimum_purity:
                reasons.append("low_owner_purity")
            if float(values["candidate_coverage_fraction"].median()) < minimum_coverage:
                reasons.append("low_owner_coverage")
            if float(values["nearest_track_distance_px"].median()) < minimum_isolation:
                reasons.append("insufficient_raw_track_isolation")
            rows.append({
                "proposal_id": f"P{len(rows) + 1:04d}",
                "track_id": int(track_id), "source_owner": source,
                "proposed_owner": int(canonical),
                "first_frame": int(run["first_frame"]),
                "last_frame": int(run["last_frame"]),
                "source_run_frames": int(run["frames"]),
                "source_total_support": int(support.get(source, 0)),
                "proposed_owner_support": int(canonical_support),
                "support_ratio": float(ratio),
                "before_owner": before_owner, "after_owner": after_owner,
                "median_owner_purity": float(
                    values["candidate_owner_purity"].median()),
                "median_owner_coverage": float(
                    values["candidate_coverage_fraction"].median()),
                "median_track_isolation_px": float(
                    values["nearest_track_distance_px"].median()),
                "discovery_status": "eligible" if not reasons else "rejected",
                "discovery_reason": "eligible" if not reasons else "|".join(reasons),
            })
    return pd.DataFrame(rows)


def _component_at(frame: np.ndarray, owner: int, x: float, y: float,
                  radius: float) -> np.ndarray | None:
    components, count = ndi.label(frame == owner,
                                  structure=np.ones((3, 3), np.uint8))
    if not count:
        return None
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= max(2.0, radius) ** 2
    values = components[disk & (components > 0)]
    if not len(values):
        return None
    labels, counts = np.unique(values, return_counts=True)
    return components == int(labels[np.argmax(counts)])


def _identity_median_area(labels: np.ndarray, owner: int) -> float:
    areas = [int(np.count_nonzero(frame == owner)) for frame in labels]
    positive = [area for area in areas if area > 0]
    return float(np.median(positive)) if positive else 0.0


def count_new_duplicate_components(
        baseline: np.ndarray, candidate: np.ndarray) -> int:
    """Count only new same-frame components beyond an identity's first."""
    return component_accounting.new_duplicate_components(
        baseline, candidate)


def _core_count(component: np.ndarray, frame_points: pd.DataFrame) -> int:
    """Count candidate-independent physical-track centres in a component."""
    count = 0
    for point in frame_points.itertuples(index=False):
        y = int(np.clip(round(float(point.y)), 0, component.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, component.shape[1] - 1))
        count += int(component[y, x])
    return count


def _orphan_absorption(
        labels: np.ndarray, frame: int, source_owner: int,
        occupied: np.ndarray, frame_points: pd.DataFrame,
        maximum_area_ratio: float) -> tuple[np.ndarray, int] | None:
    """Return a safe owner for a zero-core fragment touching one owned body.

    The operation is deliberately narrow: the whole blocking component must have
    no physical core, touch exactly one other connected labelled component, and
    that neighbour must contain exactly one physical core whose disk owner agrees
    with the neighbour. Absorption therefore joins an already-touching component
    and cannot create another component of the receiving identity.
    """
    components, count = ndi.label(
        labels[frame] == source_owner, structure=np.ones((3, 3), np.uint8))
    values = np.unique(components[occupied & (components > 0)])
    if len(values) != 1:
        return None
    component = components == int(values[0])
    if np.any(component & ~occupied):
        return None
    if _core_count(component, frame_points) != 0:
        return None
    border = ndi.binary_dilation(component, structure=np.ones((3, 3), bool)) & ~component
    neighbour_ids = set(map(int, np.unique(labels[frame][border]))) - {0, source_owner}
    if len(neighbour_ids) != 1:
        return None
    neighbour_owner = int(next(iter(neighbour_ids)))
    neighbour_components, _ = ndi.label(
        labels[frame] == neighbour_owner, structure=np.ones((3, 3), np.uint8))
    neighbour_values = np.unique(neighbour_components[border & (neighbour_components > 0)])
    if len(neighbour_values) != 1:
        return None
    neighbour = neighbour_components == int(neighbour_values[0])
    neighbour_points = []
    for point in frame_points.itertuples(index=False):
        y = int(np.clip(round(float(point.y)), 0, component.shape[0] - 1))
        x = int(np.clip(round(float(point.x)), 0, component.shape[1] - 1))
        if neighbour[y, x]:
            neighbour_points.append(point)
    if len(neighbour_points) != 1:
        return None
    if int(neighbour_points[0].candidate_owner) != neighbour_owner:
        return None
    ratio = float(component.sum() / max(neighbour.sum(), 1))
    if ratio > maximum_area_ratio:
        return None
    return component, neighbour_owner


def apply_proposals(labels: np.ndarray, points: pd.DataFrame,
                    proposals: pd.DataFrame, params: dict) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    audit_rows: list[dict] = []
    point_index = points.set_index(["track_id", "frame"], drop=False)
    visible = points[points["physically_visible"].astype(bool)]
    minimum_area_ratio = float(params.get("minimum_component_area_ratio", 0.25))
    maximum_area_ratio = float(params.get("maximum_component_area_ratio", 4.0))
    medians: dict[int, float] = {}

    for proposal in proposals.itertuples(index=False):
        if proposal.discovery_status != "eligible":
            audit_rows.append({**proposal._asdict(), "outcome": "rejected_discovery",
                               "changed_pixels": 0})
            continue
        owner = int(proposal.proposed_owner)
        source = int(proposal.source_owner)
        medians.setdefault(owner, _identity_median_area(labels, owner))
        changes: list[tuple[int, np.ndarray]] = []
        reasons: list[str] = []
        for frame_index in range(int(proposal.first_frame),
                                 int(proposal.last_frame) + 1):
            key = (int(proposal.track_id), frame_index)
            if key not in point_index.index:
                reasons.append("missing_track_point")
                continue
            row = point_index.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if np.any(candidate[frame_index] == owner):
                reasons.append("proposed_owner_already_present")
                continue
            component = _component_at(
                candidate[frame_index], source, float(row.x), float(row.y),
                float(row.radius_px))
            if component is None:
                reasons.append("source_component_not_found")
                continue
            frame_points = visible[visible["frame"] == frame_index]
            cores = 0
            for point in frame_points.itertuples(index=False):
                y = int(np.clip(round(float(point.y)), 0, component.shape[0] - 1))
                x = int(np.clip(round(float(point.x)), 0, component.shape[1] - 1))
                cores += int(component[y, x])
            if cores != 1:
                reasons.append("component_not_single_raw_core")
                continue
            area_ratio = float(component.sum() / max(medians[owner], 1.0))
            if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
                reasons.append("component_area_incompatible")
                continue
            changes.append((frame_index, component))
        expected = int(proposal.last_frame) - int(proposal.first_frame) + 1
        if reasons or len(changes) != expected:
            audit_rows.append({
                **proposal._asdict(), "outcome": "rejected_application:" +
                "|".join(sorted(set(reasons or ["incomplete_run"]))),
                "changed_pixels": 0})
            continue
        changed = 0
        for frame_index, component in changes:
            changed += int(component.sum())
            candidate[frame_index][component] = owner
        audit_rows.append({**proposal._asdict(), "outcome": "applied",
                           "changed_pixels": changed})
    return candidate, pd.DataFrame(audit_rows)


def apply_dependency_ordered_proposals(
        labels: np.ndarray, points: pd.DataFrame, proposals: pd.DataFrame,
        params: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply a simultaneous ownership chain only when every owner is released.

    A proposed owner may already occupy another component if another eligible
    proposal releases *all* of that owner's pixels in the same frame. This solves a
    chain such as 50->68 while 68->70, without ever creating duplicate 68 masks.
    """
    visible = points[points["physically_visible"].astype(bool)]
    point_index = points.set_index(["track_id", "frame"], drop=False)
    minimum_area_ratio = float(params.get("minimum_component_area_ratio", 0.25))
    maximum_area_ratio = float(params.get("maximum_component_area_ratio", 4.0))
    medians: dict[int, float] = {}
    audit: dict[str, dict] = {}
    prepared: dict[str, dict] = {}
    absorb_orphans = bool(params.get("absorb_zero_core_blockers", False))
    maximum_orphan_ratio = float(
        params.get("maximum_orphan_to_owner_area_ratio", 0.75))

    for proposal in proposals.itertuples(index=False):
        base = proposal._asdict()
        proposal_id = str(proposal.proposal_id)
        if proposal.discovery_status != "eligible":
            audit[proposal_id] = {**base, "outcome": "rejected_discovery",
                                  "changed_pixels": 0}
            continue
        source = int(proposal.source_owner)
        owner = int(proposal.proposed_owner)
        outside = np.ones(len(labels), bool)
        outside[int(proposal.first_frame):int(proposal.last_frame) + 1] = False
        if not np.any(labels[outside] == source):
            audit[proposal_id] = {
                **base, "outcome": "rejected_application:source_identity_extinguished",
                "changed_pixels": 0}
            continue
        medians.setdefault(owner, _identity_median_area(labels, owner))
        changes: list[tuple[int, np.ndarray]] = []
        reasons: list[str] = []
        for frame_index in range(int(proposal.first_frame),
                                 int(proposal.last_frame) + 1):
            key = (int(proposal.track_id), frame_index)
            if key not in point_index.index:
                reasons.append("missing_track_point"); continue
            row = point_index.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            component = _component_at(
                labels[frame_index], source, float(row.x), float(row.y),
                float(row.radius_px))
            if component is None:
                reasons.append("source_component_not_found"); continue
            frame_points = visible[visible["frame"] == frame_index]
            cores = _core_count(component, frame_points)
            if cores != 1:
                reasons.append("component_not_single_raw_core"); continue
            ratio = float(component.sum() / max(medians[owner], 1.0))
            if not minimum_area_ratio <= ratio <= maximum_area_ratio:
                reasons.append("component_area_incompatible"); continue
            changes.append((frame_index, component))
        expected = int(proposal.last_frame) - int(proposal.first_frame) + 1
        if reasons or len(changes) != expected:
            audit[proposal_id] = {
                **base, "outcome": "rejected_application:" +
                "|".join(sorted(set(reasons or ["incomplete_run"]))),
                "changed_pixels": 0}
            continue
        # Overlapping source masks make the simultaneous ownership graph ambiguous.
        if any(frame == other_frame and np.any(mask & other_mask)
               for item in prepared.values()
               for frame, mask in changes
               for other_frame, other_mask in item["changes"]):
            audit[proposal_id] = {
                **base, "outcome": "rejected_application:overlapping_proposal",
                "changed_pixels": 0}
            continue
        prepared[proposal_id] = {"proposal": proposal, "changes": changes}

    # A requested owner can also be occupied by a zero-core fragment that has
    # become attached to one independently owned physical body. Such fragments
    # are proposed from topology and physical cores, never from review targets.
    orphan_releases: dict[tuple[int, int, int], dict] = {}
    if absorb_orphans:
        for proposal_id, item in prepared.items():
            owner = int(item["proposal"].proposed_owner)
            for frame, _ in item["changes"]:
                occupied = labels[frame] == owner
                if not np.any(occupied):
                    continue
                regular_release = np.zeros_like(occupied)
                for other_id, other in prepared.items():
                    if other_id == proposal_id or int(
                            other["proposal"].source_owner) != owner:
                        continue
                    for other_frame, other_mask in other["changes"]:
                        if other_frame == frame:
                            regular_release |= other_mask
                uncovered = occupied & ~regular_release
                components, count = ndi.label(
                    uncovered, structure=np.ones((3, 3), np.uint8))
                frame_points = visible[visible["frame"] == frame]
                for component_id in range(1, count + 1):
                    fragment = components == component_id
                    result = _orphan_absorption(
                        labels, frame, owner, fragment, frame_points,
                        maximum_orphan_ratio)
                    if result is None:
                        continue
                    mask, new_owner = result
                    key = (frame, owner, int(np.flatnonzero(mask)[0]))
                    release = orphan_releases.setdefault(key, {
                        "frame": frame, "source_owner": owner,
                        "proposed_owner": new_owner, "mask": mask,
                        "dependents": set(),
                    })
                    release["dependents"].add(proposal_id)

    # Fixed point: removing an invalid releaser can invalidate proposals depending on it.
    changed = True
    while changed:
        changed = False
        invalid: dict[str, str] = {}
        for proposal_id, item in prepared.items():
            proposal = item["proposal"]
            owner = int(proposal.proposed_owner)
            for frame, _ in item["changes"]:
                occupied = labels[frame] == owner
                if not np.any(occupied):
                    continue
                released = np.zeros_like(occupied)
                for other_id, other in prepared.items():
                    if other_id == proposal_id or int(
                            other["proposal"].source_owner) != owner:
                        continue
                    for other_frame, other_mask in other["changes"]:
                        if other_frame == frame:
                            released |= other_mask
                for release in orphan_releases.values():
                    if (release["frame"] == frame and
                            release["source_owner"] == owner and
                            proposal_id in release["dependents"]):
                        released |= release["mask"]
                if np.any(occupied & ~released):
                    invalid[proposal_id] = "proposed_owner_not_fully_released"
                    break
        for proposal_id, reason in invalid.items():
            item = prepared.pop(proposal_id)
            audit[proposal_id] = {
                **item["proposal"]._asdict(),
                "outcome": "rejected_application:" + reason,
                "changed_pixels": 0}
            changed = True

    candidate = labels.copy()
    for proposal_id, item in prepared.items():
        owner = int(item["proposal"].proposed_owner)
        pixels = 0
        for frame, mask in item["changes"]:
            candidate[frame][mask] = owner
            pixels += int(mask.sum())
        audit[proposal_id] = {**item["proposal"]._asdict(), "outcome": "applied",
                              "changed_pixels": pixels}
    orphan_audit: list[dict] = []
    for index, release in enumerate(orphan_releases.values(), start=1):
        active_dependents = sorted(
            release["dependents"] & set(prepared))
        applied = bool(active_dependents)
        if applied:
            candidate[int(release["frame"])][release["mask"]] = int(
                release["proposed_owner"])
        orphan_audit.append({
            "proposal_id": f"O{index:04d}", "track_id": "",
            "source_owner": int(release["source_owner"]),
            "proposed_owner": int(release["proposed_owner"]),
            "first_frame": int(release["frame"]),
            "last_frame": int(release["frame"]), "source_run_frames": 1,
            "source_total_support": "", "proposed_owner_support": "",
            "support_ratio": "", "before_owner": "", "after_owner": "",
            "median_owner_purity": "", "median_owner_coverage": "",
            "median_track_isolation_px": "",
            "discovery_status": "eligible",
            "discovery_reason": "zero_core_fragment_touching_single_core_owner",
            "outcome": ("applied" if applied else
                        "rejected_application:dependent_proposal_rejected"),
            "changed_pixels": int(release["mask"].sum()) if applied else 0,
            "dependent_proposals": "|".join(active_dependents),
        })
    ordered = [audit[str(value)] for value in proposals["proposal_id"]]
    return candidate, pd.DataFrame(ordered + orphan_audit)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    supplied = [name for name in FORBIDDEN_TARGETS if params.get(name)]
    if supplied:
        raise ValueError("field-wide owner consensus received forbidden targets: " +
                         ", ".join(supplied))
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    mode = str(params.get("mode", "candidate"))
    if mode == "baseline":
        candidate = labels.copy()
        audit = pd.DataFrame(columns=["proposal_id", "outcome", "changed_pixels"])
    elif mode == "candidate":
        physical = pd.read_csv(params["physical_track_points_path"])
        points = attach_owners(physical, labels)
        proposals = discover_proposals(points, params)
        if str(params.get("application_mode", "sequential")) == "dependency_ordered":
            candidate, audit = apply_dependency_ordered_proposals(
                labels, points, proposals, params)
        else:
            candidate, audit = apply_proposals(labels, points, proposals, params)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("owner consensus changed foreground segmentation")
    labels_out = out.out / "95_A3.tif"
    unclaimed_out = out.out / "95_A3_unclaimed_original_ids.tif"
    if mode == "baseline":
        shutil.copyfile(labels_path, labels_out)
    else:
        tifffile.imwrite(labels_out, candidate, imagej=True, compression="zlib",
                         metadata={"axes": "TYX", "finterval": 1800.0,
                                   "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(unclaimed_path, unclaimed_out)
    audit_path = out.out / "owner_consensus_audit.csv"
    audit.to_csv(audit_path, index=False)
    applied = audit[audit.get("outcome", pd.Series(dtype=str)) == "applied"] \
        if len(audit) else audit
    summary = {
        "targeting_mode": "field_wide_discovery",
        "review_case_targets_received": False,
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "event_target_count": 0,
        "mode": mode, "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "active_identities": int(len(set(map(int, np.unique(candidate))) - {0})),
        "applied_orphan_absorptions": int(
            audit.get("proposal_id", pd.Series(dtype=str)).astype(str).str.startswith("O")
            .where(audit.get("outcome", pd.Series(dtype=str)).eq("applied"), False)
            .sum()) if len(audit) else 0,
        "parameters": {key: value for key, value in params.items()
                       if not key.endswith("_path")},
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {"labels": labels_out, "unclaimed": unclaimed_out,
                         "audit": audit_path, "metrics": metrics_path},
            "summary": summary}
