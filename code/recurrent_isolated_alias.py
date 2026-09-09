"""Production repair for recurrent isolated aliases on stable physical seats.

Discovery is complete-field and identity-blind: identities are measured from the
supplied label stack and never supplied as targets.  A proposal is a compact
cluster of at least two returns to the same transient owner inside a long,
high-purity dominant-owner history.  Every structural frame gate is rechecked
before the complete cluster is applied atomically.
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
from resident_takeover import attach_owners


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_pair", "forced_interval", "failure_target",
)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "owner_a", "owner_b", "cluster_start",
    "cluster_end", "cluster_span_frames", "transient_runs",
    "transient_frames", "maximum_transient_run_frames",
    "maximum_internal_bridge_frames", "pre_flank_frames",
    "post_flank_frames", "visible_observations", "strong_observations",
    "encounter_observations", "stable_owner_support",
    "positive_owner_observations", "stable_owner_purity",
    "maximum_step_median_radius", "stable_area_q1", "stable_area_q3",
    "stable_area_iqr", "stable_area_lower_fence", "stable_area_upper_fence",
    "transient_area_min", "transient_area_max",
    "maximum_component_raw_cores", "retained_transient_owner_cores",
    "competing_stable_owner_cores", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "track_id", "frame", "owner_a", "owner_b",
    "component_area", "changed_pixels", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("recurrent isolated alias must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "recurrent isolated alias received forbidden targets: "
            + ", ".join(supplied))


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for point in group.sort_values("frame").itertuples(index=False):
        if not bool(point.physically_visible):
            continue
        owner = int(point.candidate_owner)
        if owner <= 0:
            continue
        frame = int(point.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame == runs[-1]["end"] + 1):
            runs[-1]["end"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "start": frame, "end": frame,
                         "frames": [frame]})
    return runs


def _point(index: pd.DataFrame, track: int, frame: int):
    key = (int(track), int(frame))
    if key not in index.index:
        return None
    row = index.loc[key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    if count <= 0:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    value = int(components[y, x])
    if value <= 0:
        radius = max(2.0, 0.75 * float(point.radius_px))
        yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
        values = components[
            ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
             <= radius ** 2) & (components > 0)]
        if not len(values):
            return None
        ids, counts = np.unique(values, return_counts=True)
        value = int(ids[int(np.argmax(counts))])
    return components == value


def _inside(mask: np.ndarray, point) -> bool:
    y = int(np.clip(round(float(point.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, mask.shape[1] - 1))
    if bool(mask[y, x]):
        return True
    radius = max(2.0, 0.75 * float(point.radius_px))
    y0 = max(0, int(np.floor(float(point.y) - radius)))
    y1 = min(mask.shape[0], int(np.ceil(float(point.y) + radius + 1)))
    x0 = max(0, int(np.floor(float(point.x) - radius)))
    x1 = min(mask.shape[1], int(np.ceil(float(point.x) + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = ((xx - float(point.x)) ** 2 + (yy - float(point.y)) ** 2
            <= radius ** 2)
    return bool(np.any(mask[y0:y1, x0:x1] & disk))


def _maximum_step_median_radius(visible: pd.DataFrame) -> float:
    ordered = list(visible.sort_values("frame").itertuples(index=False))
    if len(ordered) < 2:
        return 0.0
    median_radius = float(np.median(
        [max(float(point.radius_px), 1.0) for point in ordered]))
    steps = [
        float(np.hypot(float(right.x) - float(left.x),
                       float(right.y) - float(left.y))) / median_radius
        for left, right in zip(ordered, ordered[1:])
        if int(right.frame) == int(left.frame) + 1
    ]
    return max(steps, default=0.0)


def _clusters(runs: list[dict], owner_a: int, owner_b: int,
              maximum_run: int, maximum_bridge: int,
              maximum_cluster: int) -> list[dict]:
    """Return maximal compact B-A-B clusters for one measured owner pair."""
    foreign = [index for index, run in enumerate(runs)
               if int(run["owner"]) == int(owner_b)
               and len(run["frames"]) <= maximum_run]
    clusters: list[dict] = []
    cursor = 0
    while cursor < len(foreign):
        selected = [foreign[cursor]]
        probe = cursor + 1
        while probe < len(foreign):
            previous, current = selected[-1], foreign[probe]
            between = runs[previous + 1:current]
            bridge_frames = (int(runs[current]["start"])
                             - int(runs[previous]["end"]) - 1)
            span = (int(runs[current]["end"])
                    - int(runs[selected[0]]["start"]) + 1)
            if (len(between) == 1
                    and int(between[0]["owner"]) == int(owner_a)
                    and len(between[0]["frames"]) == bridge_frames
                    and bridge_frames <= maximum_bridge
                    and span <= maximum_cluster):
                selected.append(current)
                probe += 1
                continue
            break
        if len(selected) >= 2:
            first_index, last_index = selected[0], selected[-1]
            transient = [frame for index in selected
                         for frame in runs[index]["frames"]]
            bridges = [
                int(runs[right]["start"]) - int(runs[left]["end"]) - 1
                for left, right in zip(selected, selected[1:])]
            clusters.append({
                "run_indices": selected,
                "first_run_index": first_index,
                "last_run_index": last_index,
                "start": int(runs[first_index]["start"]),
                "end": int(runs[last_index]["end"]),
                "frames": list(map(int, transient)),
                "runs": len(selected),
                "maximum_run": max(len(runs[index]["frames"])
                                   for index in selected),
                "maximum_bridge": max(bridges, default=0),
            })
            cursor = probe
        else:
            cursor += 1
    return clusters


def _component_structure(labels: np.ndarray, attached_by_frame: dict[int, pd.DataFrame],
                         track: int, owner_a: int, owner_b: int,
                         point_index: pd.DataFrame, frames: list[int]
                         ) -> tuple[list[int], int, int, int, str]:
    areas: list[int] = []
    maximum_cores = 0
    retained_b = 0
    competing_a = 0
    for frame_index in frames:
        resident = _point(point_index, track, frame_index)
        if resident is None or not bool(resident.physically_visible):
            return areas, maximum_cores, retained_b, competing_a, \
                "resident_point_missing"
        component = _component_at(labels[frame_index], owner_b, resident)
        if component is None:
            return areas, maximum_cores, retained_b, competing_a, \
                "resident_component_missing"
        areas.append(int(component.sum()))
        frame_rows = attached_by_frame.get(frame_index, pd.DataFrame())
        visible_rows = (frame_rows[frame_rows.physically_visible.astype(bool)]
                        if len(frame_rows) else frame_rows)
        component_tracks = {
            int(row.track_id) for row in visible_rows.itertuples(index=False)
            if _inside(component, row)}
        maximum_cores = max(maximum_cores, len(component_tracks))
        if component_tracks != {int(track)}:
            return areas, maximum_cores, retained_b, competing_a, \
                "foreign_component_not_single_resident_core"
        retained_b += int(sum(
            int(row.track_id) != int(track)
            and int(row.candidate_owner) == int(owner_b)
            for row in visible_rows.itertuples(index=False)))
        competing_a += int(sum(
            int(row.track_id) != int(track)
            and int(row.candidate_owner) == int(owner_a)
            for row in visible_rows.itertuples(index=False)))
    if retained_b:
        return areas, maximum_cores, retained_b, competing_a, \
            "retained_transient_owner_core_present"
    if competing_a:
        return areas, maximum_cores, retained_b, competing_a, \
            "competing_stable_owner_core_present"
    return areas, maximum_cores, retained_b, competing_a, "eligible"


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all recurrent same-owner blip clusters in the complete field."""
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    point_index = attached.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in attached.groupby("frame")}
    frame_count = int(len(labels))
    minimum_visible = int(math.ceil(
        float(params.get("minimum_visible_movie_fraction", 0.20))
        * frame_count))
    minimum_support = int(math.ceil(
        float(params.get("minimum_stable_support_movie_fraction", 0.25))
        * frame_count))
    minimum_flank = int(math.ceil(
        float(params.get("minimum_external_flank_movie_fraction", 0.10))
        * frame_count))
    maximum_run = int(math.ceil(
        float(params.get("maximum_transient_run_movie_fraction", 0.02))
        * frame_count))
    maximum_bridge = int(math.ceil(
        float(params.get("maximum_internal_bridge_movie_fraction", 0.04))
        * frame_count))
    maximum_cluster = int(math.ceil(
        float(params.get("maximum_cluster_movie_fraction", 0.06))
        * frame_count))
    minimum_purity = float(params.get("minimum_stable_owner_purity", 0.95))
    maximum_step = float(params.get("maximum_step_median_radius", 0.75))
    fence_multiplier = float(params.get("tukey_fence_multiplier", 1.5))
    rows: list[dict] = []
    internals: list[dict] = []
    for track, group in attached.groupby("track_id"):
        track = int(track)
        visible = group[group.physically_visible.astype(bool)]
        positive = visible[visible.candidate_owner.astype(int) > 0]
        if positive.empty:
            continue
        counts = positive.candidate_owner.astype(int).value_counts()
        owner_a = int(counts.index[0])
        support = int(counts.iloc[0])
        purity = float(support / len(positive))
        runs = _positive_runs(group)
        foreign_owners = sorted({int(run["owner"]) for run in runs
                                 if int(run["owner"]) != owner_a})
        maximum_seen_step = _maximum_step_median_radius(visible)
        for owner_b in foreign_owners:
            for cluster in _clusters(
                    runs, owner_a, owner_b, maximum_run,
                    maximum_bridge, maximum_cluster):
                first_index = int(cluster["first_run_index"])
                last_index = int(cluster["last_run_index"])
                pre = runs[first_index - 1] if first_index > 0 else None
                post = runs[last_index + 1] \
                    if last_index + 1 < len(runs) else None
                pre_frames = (len(pre["frames"]) if pre is not None
                              and int(pre["owner"]) == owner_a
                              and int(pre["end"]) == int(cluster["start"]) - 1
                              else 0)
                post_frames = (len(post["frames"]) if post is not None
                               and int(post["owner"]) == owner_a
                               and int(post["start"]) == int(cluster["end"]) + 1
                               else 0)
                stable_areas: list[int] = []
                for point in positive[
                        (positive.candidate_owner.astype(int) == owner_a)
                        & ~positive.frame.astype(int).between(
                            int(cluster["start"]), int(cluster["end"]))] \
                        .itertuples(index=False):
                    component = _component_at(
                        labels[int(point.frame)], owner_a, point)
                    if component is not None:
                        stable_areas.append(int(component.sum()))
                if stable_areas:
                    q1, q3 = np.percentile(stable_areas, [25, 75])
                    iqr = float(q3 - q1)
                    lower = max(1.0, float(q1 - fence_multiplier * iqr))
                    upper = float(q3 + fence_multiplier * iqr)
                else:
                    q1 = q3 = iqr = lower = upper = float("nan")
                transient_areas, maximum_cores, retained_b, competing_a, \
                    structural_reason = _component_structure(
                        labels, by_frame, track, owner_a, owner_b,
                        point_index, cluster["frames"])
                reasons: list[str] = []
                if len(visible) < minimum_visible:
                    reasons.append("insufficient_visible_history")
                if support < minimum_support:
                    reasons.append("insufficient_stable_owner_support")
                if purity < minimum_purity:
                    reasons.append("insufficient_stable_owner_purity")
                if int(visible.strong.astype(bool).sum()) != len(visible):
                    reasons.append("visible_history_not_all_strong")
                if int(visible.in_encounter.astype(bool).sum()):
                    reasons.append("visible_history_contains_encounter")
                if pre_frames < minimum_flank or post_frames < minimum_flank:
                    reasons.append("insufficient_external_stable_flanks")
                if maximum_seen_step > maximum_step:
                    reasons.append("seat_motion_exceeds_radius_gate")
                if not stable_areas or len(transient_areas) != len(cluster["frames"]):
                    reasons.append("component_area_evidence_incomplete")
                elif any(area < lower or area > upper
                         for area in transient_areas):
                    reasons.append("transient_area_outside_tukey_fence")
                if structural_reason != "eligible":
                    reasons.append(structural_reason)
                public = {
                    "proposal_id": "", "track_id": track,
                    "owner_a": owner_a, "owner_b": owner_b,
                    "cluster_start": int(cluster["start"]),
                    "cluster_end": int(cluster["end"]),
                    "cluster_span_frames": (int(cluster["end"])
                                            - int(cluster["start"]) + 1),
                    "transient_runs": int(cluster["runs"]),
                    "transient_frames": len(cluster["frames"]),
                    "maximum_transient_run_frames": int(cluster["maximum_run"]),
                    "maximum_internal_bridge_frames": int(cluster["maximum_bridge"]),
                    "pre_flank_frames": pre_frames,
                    "post_flank_frames": post_frames,
                    "visible_observations": int(len(visible)),
                    "strong_observations": int(visible.strong.astype(bool).sum()),
                    "encounter_observations": int(visible.in_encounter.astype(bool).sum()),
                    "stable_owner_support": support,
                    "positive_owner_observations": int(len(positive)),
                    "stable_owner_purity": purity,
                    "maximum_step_median_radius": maximum_seen_step,
                    "stable_area_q1": float(q1), "stable_area_q3": float(q3),
                    "stable_area_iqr": float(iqr),
                    "stable_area_lower_fence": float(lower),
                    "stable_area_upper_fence": float(upper),
                    "transient_area_min": min(transient_areas, default=0),
                    "transient_area_max": max(transient_areas, default=0),
                    "maximum_component_raw_cores": maximum_cores,
                    "retained_transient_owner_cores": retained_b,
                    "competing_stable_owner_cores": competing_a,
                    "eligible": not reasons,
                    "reason": ("eligible_recurrent_isolated_transient_alias"
                               if not reasons else "|".join(reasons)),
                }
                rows.append(public)
                internals.append({**public,
                                  "frame_values": list(cluster["frames"])})
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            value = f"RA{number:04d}"
            public["proposal_id"] = internal["proposal_id"] = value
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internals))


def _prepare_frame(frame: np.ndarray, by_frame: dict[int, pd.DataFrame],
                   point_index: pd.DataFrame, proposal, frame_index: int,
                   ) -> tuple[np.ndarray | None, str, int, int]:
    resident = _point(point_index, int(proposal.track_id), frame_index)
    if resident is None or not bool(resident.physically_visible):
        return None, "resident_point_missing", 0, 0
    component = _component_at(frame, int(proposal.owner_b), resident)
    if component is None:
        return None, "resident_component_missing", 0, 0
    visible = by_frame.get(frame_index, pd.DataFrame())
    visible = visible[visible.physically_visible.astype(bool)] \
        if len(visible) else visible
    component_tracks = {
        int(row.track_id) for row in visible.itertuples(index=False)
        if _inside(component, row)}
    if component_tracks != {int(proposal.track_id)}:
        return None, "foreign_component_not_single_resident_core", 0, 0
    if any(int(row.track_id) != int(proposal.track_id)
           and int(row.candidate_owner) == int(proposal.owner_b)
           for row in visible.itertuples(index=False)):
        return None, "retained_transient_owner_core_present", 0, 0
    if any(int(row.track_id) != int(proposal.track_id)
           and int(row.candidate_owner) == int(proposal.owner_a)
           for row in visible.itertuples(index=False)):
        return None, "competing_stable_owner_core_present", 0, 0
    trial = frame.copy()
    trial[component] = int(proposal.owner_a)
    changed = int(np.count_nonzero(trial != frame))
    if changed != int(component.sum()) or changed <= 0:
        return None, "component_transfer_incomplete", 0, 0
    if component_accounting.new_duplicate_components(
            frame[None], trial[None]):
        return None, "new_duplicate_component", 0, 0
    before_components = int(component_accounting.label_value_components(frame)[0].max())
    after_components = int(component_accounting.label_value_components(trial)[0].max())
    if after_components != before_components:
        return None, "label_component_ledger_changed", 0, 0
    return trial, "isolated_component_relabel", changed, int(component.sum())


def apply(labels: np.ndarray, points: pd.DataFrame,
          internal: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    attached = attach_owners(points, labels).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)
    point_index = attached.set_index(["track_id", "frame"], drop=False)
    by_frame = {int(frame): group for frame, group in attached.groupby("frame")}
    applications: list[dict] = []
    eligible = internal[internal.eligible.astype(bool)] \
        if len(internal) else internal
    for proposal in eligible.itertuples(index=False):
        pending: list[tuple[int, np.ndarray, str, int, int]] = []
        refusal = ""
        for frame_index in list(proposal.frame_values):
            trial, reason, changed, area = _prepare_frame(
                candidate[int(frame_index)], by_frame, point_index,
                proposal, int(frame_index))
            if trial is None:
                refusal = reason
                break
            pending.append((int(frame_index), trial, reason, changed, area))
        if refusal:
            for frame_index in list(proposal.frame_values):
                applications.append({
                    "proposal_id": proposal.proposal_id,
                    "track_id": int(proposal.track_id),
                    "frame": int(frame_index), "owner_a": int(proposal.owner_a),
                    "owner_b": int(proposal.owner_b), "component_area": 0,
                    "changed_pixels": 0, "applied": False,
                    "reason": "atomic_refusal:" + refusal,
                })
            continue
        trial_stack = candidate.copy()
        for frame_index, trial, _, _, _ in pending:
            trial_stack[frame_index] = trial
        if component_accounting.new_duplicate_components(labels, trial_stack):
            refusal = "new_duplicate_component_across_cluster"
        if refusal:
            for frame_index in list(proposal.frame_values):
                applications.append({
                    "proposal_id": proposal.proposal_id,
                    "track_id": int(proposal.track_id),
                    "frame": int(frame_index), "owner_a": int(proposal.owner_a),
                    "owner_b": int(proposal.owner_b), "component_area": 0,
                    "changed_pixels": 0, "applied": False,
                    "reason": "atomic_refusal:" + refusal,
                })
            continue
        for frame_index, trial, reason, changed, area in pending:
            candidate[frame_index] = trial
            applications.append({
                "proposal_id": proposal.proposal_id,
                "track_id": int(proposal.track_id), "frame": frame_index,
                "owner_a": int(proposal.owner_a), "owner_b": int(proposal.owner_b),
                "component_area": area, "changed_pixels": changed,
                "applied": True, "reason": reason,
            })
    return candidate, pd.DataFrame(applications, columns=APPLICATION_COLUMNS)


def _copy_audits(upstream_dir: Path | None, params: dict, out) -> dict:
    outputs = {}
    for name in ("application_audit.csv", "conflicted_lineage_audit.csv"):
        source = upstream_dir / name if upstream_dir is not None else None
        configured = params.get(name.removesuffix(".csv") + "_path")
        if source is None or not source.is_file():
            source = Path(configured) if configured else None
        target = out.out / name
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text("\n", encoding="utf-8")
        outputs[name.removesuffix(".csv")] = target
    return outputs


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame(columns=APPLICATION_COLUMNS)
    else:
        candidate, applications = apply(labels, points, internal)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("recurrent isolated alias changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("recurrent isolated alias changed identity set")
    duplicates = component_accounting.new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("recurrent isolated alias created duplicate components")

    output_stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = out.out / f"{output_stem}.tif"
    unclaimed_path = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    audit_path = out.out / "recurrent_isolated_alias_audit.csv"
    applications_path = out.out / "recurrent_isolated_alias_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(applications_path, index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "recurrent_clusters_audited": int(len(audit)),
        "eligible_clusters": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_clusters": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": len(before_ids),
        "output_identity_count": len(after_ids),
        "new_identity_count": 0, "removed_identity_count": 0,
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "label_component_ledger_exact": True,
        "new_duplicate_components": int(duplicates),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    outputs = {"labels": labels_path, "unclaimed": unclaimed_path,
               "audit": audit_path, "applications": applications_path,
               "metrics": metrics_path}
    outputs.update(_copy_audits(upstream_dir, params, out))
    return {"outputs": outputs, "summary": metrics}
