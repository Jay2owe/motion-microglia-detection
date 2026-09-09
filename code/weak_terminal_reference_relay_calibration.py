"""Calibrate terminal relays from a weak reference to a strong reference.

Discovery is complete-field and target-free.  Measured identifiers are emitted
only in the audit; none can be supplied as selectors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGET_TOKENS = (
    "identity_id", "owner_id", "track_id", "frame_id", "event_id",
    "coordinate", "region", "review_case", "case_id", "target_identity",
    "target_owner", "target_track", "target_frame", "target_event",
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate_target", "event_target", "region_target",
    "review_case_target",
    "forced_identity", "forced_interval", "include_track", "exclude_track",
)


def assert_target_free(params: dict) -> None:
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in str(key).lower()
                for token in FORBIDDEN_TARGET_TOKENS))
    if supplied:
        raise ValueError(
            "weak terminal relay calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("weak terminal relay calibration must be field-wide")


def _owners(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def _tracks(value: object) -> frozenset[int]:
    if pd.isna(value):
        return frozenset()
    return frozenset(map(int, re.findall(r"\d+", str(value))))


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def event_summary(events: pd.DataFrame) -> dict:
    owner = events[events.family.eq("identity_swap_or_takeover")]
    return {
        "events": int(len(events)),
        "owner_change_events": int(len(owner)),
        "owner_change_disruption_burden": float(
            owner.disruption_score.astype(float).sum()),
        "high_disruption_events": int(
            (events.disruption_score.astype(float) >= 50).sum()),
        "total_disruption_burden": float(
            events.disruption_score.astype(float).sum()),
        "events_by_family": {str(key): int(value) for key, value in
                             events.groupby("family").size().items()},
    }


def _runs(group: pd.DataFrame, owner: int) -> list[dict]:
    rows = group[
        group.candidate_owner.astype(int).eq(int(owner))
    ].sort_values("frame")
    runs: list[dict] = []
    for row in rows.itertuples(index=False):
        frame = int(row.frame)
        if runs and frame == runs[-1]["last"] + 1:
            runs[-1]["last"] = frame
            runs[-1]["frames"] += 1
        else:
            runs.append({"first": frame, "last": frame, "frames": 1})
    return runs


def _scaled_distance(left, right) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    return distance / max(float(left.radius_px) + float(right.radius_px), 1.0)


def _two_core_raw_evidence(raw_frame: np.ndarray, left, right,
                           params: dict) -> bool:
    if not (_truth(left.strong) and _truth(right.strong)):
        return False
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    scaled = _scaled_distance(left, right)
    if scaled < float(params.get("minimum_two_core_separation_sum_radii", 1.0)):
        return False
    smooth = ndi.gaussian_filter(
        raw_frame.astype(np.float32),
        float(params.get("raw_profile_sigma_px", 1.0)))
    samples = max(5, 2 * int(math.ceil(distance)) + 1)
    xs = np.linspace(float(left.x), float(right.x), samples)
    ys = np.linspace(float(left.y), float(right.y), samples)
    values = ndi.map_coordinates(smooth, [ys, xs], order=1, mode="nearest")
    low, high = float(min(values[0], values[-1])), \
        float(max(values[0], values[-1]))
    valley = float(np.min(values[1:-1]) / max(low, 1.0))
    balance = low / max(high, 1.0)
    return (valley <= float(params.get("maximum_two_core_valley_ratio", 0.80))
            and balance >= float(params.get(
                "minimum_two_core_endpoint_balance", 0.25)))


def calibrate(events: pd.DataFrame, members: pd.DataFrame,
              points: pd.DataFrame, raw: np.ndarray, labels: np.ndarray,
              params: dict):
    assert_target_free(params)
    required = {"candidate_owner", "visibility", "strong"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(
            "weak terminal relay calibration requires scored point fields: "
            + ", ".join(missing))
    movie = int(len(labels))
    terminal_first = int(math.floor(movie * (
        1.0 - float(params.get("terminal_movie_fraction", 0.12)))))
    maximum_duration = max(1, int(math.ceil(movie * float(
        params.get("maximum_event_movie_fraction", 0.08)))))
    minimum_gap_span = int(math.ceil(movie * float(
        params.get("minimum_gap_track_span_movie_fraction", 0.50))))
    maximum_gap_visibility = float(params.get(
        "maximum_gap_visibility", 0.35))
    minimum_successor_strong = float(params.get(
        "minimum_successor_strong_fraction", 0.50))
    minimum_successor_frames = int(params.get("minimum_successor_frames", 3))
    minimum_pre_anchor = int(params.get("minimum_pre_owner_anchor_frames", 3))
    maximum_post_delay = int(params.get("maximum_post_owner_delay_frames", 2))
    minimum_component = float(params.get(
        "minimum_largest_component_fraction", 0.90))

    groups = {int(track): group.sort_values("frame")
              for track, group in points.groupby("track_id", sort=True)}
    calibrated_events = events.copy()
    calibrated_members = members.copy()
    audit_rows: list[dict] = []
    reclassified: set[str] = set()

    for event in events.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        sources = event_members.drop_duplicates("source_id")
        gap_sources = sources[sources.source_family.astype(str).eq(
            "ordinary_gap")]
        merge_sources = sources[sources.source_family.astype(str).eq(
            "separable_cells_merged")]
        first, last = int(event.first_frame), int(event.last_frame)
        duration = last - first + 1
        before, after = (_owners(event.accepted_before),
                         _owners(event.accepted_after))
        owner = next(iter(before)) if len(before) == 1 and before == after else 0
        tracks = _tracks(event.physical_tracks)
        gap_track = 0
        if len(gap_sources) == 1:
            parsed = _tracks(gap_sources.iloc[0].source_ref)
            if len(parsed) == 1:
                gap_track = next(iter(parsed))
        successor_tracks = tracks - {gap_track} if gap_track else frozenset()
        successor_track = next(iter(successor_tracks)) \
            if len(successor_tracks) == 1 else 0
        gap_group = groups.get(gap_track, pd.DataFrame())
        successor_group = groups.get(successor_track, pd.DataFrame())
        reasons: list[str] = []

        if str(event.family) != "separable_cells_merged":
            reasons.append("not_merge_family")
        if len(sources) != 2 or len(gap_sources) != 1 or len(merge_sources) != 1:
            reasons.append("not_one_gap_plus_one_merge")
        if first < terminal_first or last < terminal_first:
            reasons.append("not_recording_terminal")
        if duration > maximum_duration:
            reasons.append("event_too_long")
        if not owner:
            reasons.append("accepted_owner_not_conserved")
        if len(tracks) != 2 or not gap_track or not successor_track:
            reasons.append("not_exactly_two_references")

        gap_first = int(gap_sources.iloc[0].source_first_frame) \
            if len(gap_sources) == 1 else first
        gap_last = int(gap_sources.iloc[0].source_last_frame) \
            if len(gap_sources) == 1 else last
        gap_rows = gap_group[
            gap_group.frame.astype(int).between(gap_first, gap_last)
        ] if len(gap_group) else gap_group
        gap_span = (int(gap_group.frame.max()) - int(gap_group.frame.min()) + 1
                    if len(gap_group) else 0)
        max_visibility = float(gap_rows.visibility.astype(float).max()) \
            if len(gap_rows) else 1.0
        maximum_step = max(
            (_scaled_distance(left, right) for left, right in zip(
                gap_group.itertuples(index=False),
                list(gap_group.itertuples(index=False))[1:])), default=0.0
        ) if len(gap_group) > 1 else 0.0
        if gap_span < minimum_gap_span:
            reasons.append("gap_reference_not_long_lived")
        if len(gap_rows) != gap_last - gap_first + 1:
            reasons.append("gap_reference_not_complete")
        if max_visibility > maximum_gap_visibility:
            reasons.append("gap_reference_not_uniformly_weak")
        if maximum_step > float(params.get(
                "maximum_gap_reference_step_sum_radii", 1.0)):
            reasons.append("gap_reference_not_spatially_stable")

        pre_runs = _runs(gap_group, owner) if owner and len(gap_group) else []
        pre_support = max((run["frames"] for run in pre_runs
                           if run["last"] < gap_first), default=0)
        post_delay = min((run["first"] - gap_last for run in pre_runs
                          if run["first"] > gap_last), default=movie + 1)
        if pre_support < minimum_pre_anchor:
            reasons.append("gap_reference_lacks_pre_owner_anchor")
        if post_delay > maximum_post_delay + 1:
            reasons.append("gap_reference_lacks_prompt_post_owner_anchor")

        successor_rows = successor_group[
            successor_group.frame.astype(int).between(gap_first, last)
        ] if len(successor_group) else successor_group
        successor_first = int(successor_group.frame.min()) \
            if len(successor_group) else -1
        successor_strong = float(successor_rows.strong.map(_truth).mean()) \
            if len(successor_rows) else 0.0
        successor_owner = float(successor_rows.candidate_owner.astype(int)
                                .eq(owner).mean()) \
            if owner and len(successor_rows) else 0.0
        if not gap_first <= successor_first <= gap_last + 1:
            reasons.append("companion_not_born_during_gap")
        if len(successor_rows) < minimum_successor_frames:
            reasons.append("companion_too_short")
        if successor_strong < minimum_successor_strong:
            reasons.append("companion_not_strong")
        if successor_owner < 0.95:
            reasons.append("companion_does_not_carry_conserved_owner")

        joint = sorted(set(gap_rows.frame.astype(int))
                       & set(successor_rows.frame.astype(int))) \
            if len(gap_rows) and len(successor_rows) else []
        gap_index = gap_group.set_index("frame", drop=False) \
            if len(gap_group) else gap_group
        successor_index = successor_group.set_index("frame", drop=False) \
            if len(successor_group) else successor_group
        two_core = 0
        for frame in joint:
            left, right = gap_index.loc[frame], successor_index.loc[frame]
            if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
                continue
            two_core += int(_two_core_raw_evidence(
                raw[int(frame)], left, right, params))
        if two_core:
            reasons.append("positive_two_core_raw_evidence")

        component_fractions: list[float] = []
        if owner:
            for frame in range(first, last + 1):
                mask = labels[frame] == owner
                components, count = ndi.label(mask, STRUCTURE)
                sizes = np.bincount(components.ravel())
                component_fractions.append(
                    float(sizes[1:].max() / max(int(mask.sum()), 1))
                    if int(count) else 0.0)
        minimum_component_fraction = min(component_fractions, default=0.0)
        if minimum_component_fraction < minimum_component:
            reasons.append("owner_not_one_dominant_component")

        eligible = not reasons
        if eligible:
            reclassified.add(event_id)
            mask = calibrated_members.event_id.astype(str).eq(event_id)
            calibrated_members.loc[mask, "source_family"] = \
                "reference_fragmentation"
            calibrated_members.loc[mask, "source_impact"] = 10.0
            event_mask = calibrated_events.event_id.astype(str).eq(event_id)
            disruption = 0.65 * 5.0 + 0.35 * 10.0
            calibrated_events.loc[event_mask, "family"] = \
                "reference_fragmentation"
            calibrated_events.loc[event_mask, "impact"] = 10.0
            calibrated_events.loc[event_mask, "disruption_score"] = disruption
            calibrated_events.loc[event_mask, "fidelity"] = 100.0 - disruption
            calibrated_events.loc[event_mask, "evidence"] = (
                calibrated_events.loc[event_mask, "evidence"].astype(str)
                + " ; weak terminal reference relay")
        audit_rows.append({
            "event_id": event_id, "family": str(event.family),
            "measured_owner": owner, "gap_track": gap_track,
            "successor_track": successor_track, "event_duration": duration,
            "gap_track_span": gap_span, "gap_maximum_visibility": max_visibility,
            "gap_maximum_step_sum_radii": maximum_step,
            "pre_owner_anchor_frames": pre_support,
            "post_owner_delay_frames": post_delay,
            "successor_frames": int(len(successor_rows)),
            "successor_strong_fraction": successor_strong,
            "successor_owner_fraction": successor_owner,
            "joint_frames": int(len(joint)),
            "two_core_raw_frames": int(two_core),
            "minimum_largest_component_fraction": minimum_component_fraction,
            "eligible": eligible,
            "reason": ("weak_terminal_reference_relay" if eligible
                       else "|".join(reasons)),
        })

    calibrated_events = calibrated_events.sort_values(
        ["disruption_score", "impact", "first_frame"],
        ascending=[False, False, True]).reset_index(drop=True)
    calibrated_events["impact_rank"] = np.arange(1, len(calibrated_events) + 1)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "events_audited": int(len(events)),
        "events_reclassified": int(len(reclassified)),
        "reclassified_event_ids": sorted(reclassified),
        "before": event_summary(events),
        "after": event_summary(calibrated_events),
    }
    return calibrated_events, calibrated_members, pd.DataFrame(audit_rows), summary


def run(_upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    events = pd.read_csv(params["events_path"])
    members = pd.read_csv(params["members_path"])
    points = pd.read_csv(params["points_path"])
    raw = tifffile.imread(params["raw_path"])
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    calibrated, calibrated_members, audit, summary = calibrate(
        events, members, points, raw, labels, params)
    stem = str(params.get("stem", labels_path.stem))
    outputs = {
        "events": out.out / "candidate_disruptive_events.csv",
        "members": out.out / "candidate_event_members.csv",
        "audit": out.out / "weak_terminal_reference_relay_audit.csv",
        "metrics": out.out / "metrics.json",
        "labels": out.out / f"{stem}.tif",
        "unclaimed": out.out / f"{stem}_unclaimed_original_ids.tif",
    }
    calibrated.to_csv(outputs["events"], index=False)
    calibrated_members.to_csv(outputs["members"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    shutil.copyfile(labels_path, outputs["labels"])
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    summary["label_bytes_unchanged"] = (
        outputs["labels"].read_bytes() == labels_path.read_bytes())
    summary["unclaimed_bytes_unchanged"] = (
        outputs["unclaimed"].read_bytes() == unclaimed_path.read_bytes())
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

