"""Identity-blind terminal physical-core lineage discovery.

Ported unchanged discovery helpers from the reviewed Issue003 R02-A004.
Rejected foreground-suppressing application paths are intentionally not shipped.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import large_persistent_reference_relay as relay
import separable_merge_recovery as physical


PATH_KEYS = {
    "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path",
}
CONTROL_KEYS = {
    "targeting_mode", "output_stem", "apply_recovery",
    "enable_endpoint_linking", "allow_unopposed_large_component",
}
PARAMETER_KEYS = {
    "maximum_endpoint_gap_frames", "maximum_endpoint_step_sum_radii",
    "minimum_endpoint_margin_sum_radii",
    "minimum_visible_movie_fraction", "minimum_reference_radius_quantile",
    "maximum_anchor_onset_movie_fraction",
    "minimum_anchor_run_movie_fraction",
    "minimum_foreign_occupancy_movie_fraction",
    "minimum_foreign_prior_tenure_movie_fraction",
    "maximum_foreign_canonical_core_multiplicity",
    "maximum_step_sum_radii",
    "minimum_unopposed_foreign_support_movie_fraction",
    "maximum_component_area_radius_ratio",
    "minimum_recovered_area_radius_ratio",
    "minimum_reciprocal_orientation_movie_fraction",
    "minimum_reciprocal_consistency",
    "minimum_canonical_support_movie_fraction", "minimum_canonical_purity",
    "minimum_core_separation_sum_radii",
    "maximum_separable_valley_ratio", "watershed_sigma_px",
    "minimum_changed_pixels",
}
ALLOWED_KEYS = PATH_KEYS | CONTROL_KEYS | PARAMETER_KEYS
FORBIDDEN_KEYS = {
    "identity", "identities", "identity_id", "identity_ids",
    "owner", "owners", "owner_id", "owner_ids",
    "track", "tracks", "track_id", "track_ids",
    "frame", "frames", "frame_id", "frame_ids",
    "coordinate", "coordinates", "event", "events", "event_id",
    "event_ids", "region", "regions", "well", "wells", "review_case",
    "review_cases", "case_id", "case_ids", "bbox", "roi", "target",
    "targets",
}
FORBIDDEN_FRAGMENTS = (
    "target_", "_target", "selector", "review_case", "case_id",
    "forced_", "include_", "exclude_", "allowed_pair", "track_id",
    "owner_id", "identity_id", "frame_id", "event_id", "coordinate",
    "region", "roi",
)
AUDIT_COLUMNS = [
    "proposal_id", "lineage_tracks", "terminal_track", "endpoint_links",
    "anchor_owner", "visible_frames", "visible_fraction",
    "median_radius_px", "radius_quantile", "maximum_step_sum_radii",
    "anchor_first", "anchor_last", "anchor_run_frames",
    "anchor_support_frames", "foreign_frames", "foreign_owner_count",
    "foreign_owners", "established_foreign_frames",
    "established_foreign_owners", "unestablished_foreign_owners",
    "single_core_foreign_owners", "multicore_foreign_owners",
    "selected_foreign_runs", "rejected_multicore_runs",
    "eligible", "applied", "applied_runs",
    "applied_frames", "changed_label_pixels", "changed_unclaimed_pixels",
    "assigned_unclaimed_id", "reason",
]


def assert_target_free(params: dict) -> None:
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported terminal-core-lineage parameters: "
            + ", ".join(unsupported))
    supplied = []
    for key, value in params.items():
        if value in (None, "", [], {}):
            continue
        name = str(key).strip().lower()
        if name in FORBIDDEN_KEYS or any(
                token in name for token in FORBIDDEN_FRAGMENTS):
            supplied.append(str(key))
    if supplied:
        raise ValueError(
            "terminal physical-core lineage discovery received forbidden "
            "selectors: " + ", ".join(sorted(supplied)))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "terminal physical-core lineage discovery must be field-wide")
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError(
            "missing terminal-core-lineage inputs: " + ", ".join(missing))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _track_groups(scored: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(track): group.sort_values("frame").reset_index(drop=True)
        for track, group in scored.groupby("track_id", sort=True)
    }


def _endpoint_links(groups: dict[int, pd.DataFrame], params: dict
                    ) -> list[dict]:
    if not bool(params.get("enable_endpoint_linking", True)):
        return []
    maximum_gap = int(params.get("maximum_endpoint_gap_frames", 1))
    maximum_step = float(params.get(
        "maximum_endpoint_step_sum_radii", 1.0))
    minimum_margin = float(params.get(
        "minimum_endpoint_margin_sum_radii", 0.20))
    candidates: list[dict] = []
    starts: dict[int, list[tuple[int, object]]] = {}
    for right_id, right in groups.items():
        right_row = right.iloc[0]
        if bool(right_row.physically_visible):
            starts.setdefault(int(right_row.frame), []).append(
                (right_id, right_row))
    for left_id, left in groups.items():
        left_row = left.iloc[-1]
        if not bool(left_row.physically_visible):
            continue
        last_frame = int(left_row.frame)
        for gap in range(maximum_gap + 1):
            for right_id, right_row in starts.get(last_frame + gap, []):
                if right_id == left_id:
                    continue
                step = float(np.hypot(
                    float(right_row.x) - float(left_row.x),
                    float(right_row.y) - float(left_row.y)) / max(
                        float(left_row.radius_px)
                        + float(right_row.radius_px), 1.0))
                if step <= maximum_step:
                    candidates.append({
                        "left": left_id, "right": right_id,
                        "gap_frames": gap, "step_sum_radii": step,
                    })
    by_left: dict[int, list[dict]] = {}
    by_right: dict[int, list[dict]] = {}
    for edge in candidates:
        by_left.setdefault(edge["left"], []).append(edge)
        by_right.setdefault(edge["right"], []).append(edge)
    result = []
    for edge in candidates:
        outgoing = sorted(
            by_left[edge["left"]],
            key=lambda item: (item["step_sum_radii"], item["right"]))
        incoming = sorted(
            by_right[edge["right"]],
            key=lambda item: (item["step_sum_radii"], item["left"]))
        if outgoing[0] is not edge or incoming[0] is not edge:
            continue
        left_margin = (outgoing[1]["step_sum_radii"]
                       - edge["step_sum_radii"]) \
            if len(outgoing) > 1 else float("inf")
        right_margin = (incoming[1]["step_sum_radii"]
                        - edge["step_sum_radii"]) \
            if len(incoming) > 1 else float("inf")
        if min(left_margin, right_margin) < minimum_margin:
            continue
        result.append({
            **edge, "left_margin_sum_radii": left_margin,
            "right_margin_sum_radii": right_margin,
        })
    return sorted(result, key=lambda item: (item["left"], item["right"]))


def _lineages(groups: dict[int, pd.DataFrame], links: list[dict]
              ) -> list[list[int]]:
    successor = {int(edge["left"]): int(edge["right"]) for edge in links}
    predecessor = {int(edge["right"]): int(edge["left"]) for edge in links}
    result = []
    for track in sorted(groups):
        if track in predecessor:
            continue
        lineage = [track]
        while (lineage[-1] in successor
               and successor[lineage[-1]] not in lineage):
            lineage.append(successor[lineage[-1]])
        result.append(lineage)
    return result


def _lineage_group(lineage: list[int], groups: dict[int, pd.DataFrame]
                   ) -> pd.DataFrame:
    parts = []
    previous_last = -1
    for track in lineage:
        group = groups[track]
        if previous_last >= 0:
            group = group[group.frame.astype(int).gt(previous_last)]
        parts.append(group)
        previous_last = int(groups[track].frame.max())
    return pd.concat(parts).sort_values("frame").reset_index(drop=True)


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    rows = group[
        group.physically_visible.astype(bool)
        & group.accepted_owner.astype(int).gt(0)]
    result: list[dict] = []
    for row in rows.sort_values("frame").itertuples(index=False):
        frame, owner = int(row.frame), int(row.accepted_owner)
        if (result and result[-1]["owner"] == owner
                and result[-1]["last"] + 1 == frame):
            result[-1]["last"] = frame
            result[-1]["frames"].append(frame)
        else:
            result.append({
                "owner": owner, "first": frame, "last": frame,
                "frames": [frame],
            })
    return result


def _steps(group: pd.DataFrame) -> list[float]:
    rows = list(group.sort_values("frame").itertuples(index=False))
    values = []
    for left, right in zip(rows, rows[1:]):
        if int(right.frame) != int(left.frame) + 1:
            continue
        values.append(float(np.hypot(
            float(right.x) - float(left.x),
            float(right.y) - float(left.y)) / max(
                float(left.radius_px) + float(right.radius_px), 1.0)))
    return values


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, list[dict]]:
    """Discover qualified terminal owner loss over physical-core lineages."""
    assert_target_free(params)
    scored = physical.attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = _track_groups(scored)
    links = _endpoint_links(groups, params)
    lineages = _lineages(groups, links)
    movie = len(labels)
    visible = scored[scored.physically_visible.astype(bool)]
    radius_by_track = visible.groupby("track_id").radius_px.median()
    minimum_visible = int(math.ceil(movie * _fraction(
        params, "minimum_visible_movie_fraction", 0.90)))
    minimum_radius_quantile = _fraction(
        params, "minimum_reference_radius_quantile", 0.95)
    maximum_anchor_onset = int(math.floor(movie * _fraction(
        params, "maximum_anchor_onset_movie_fraction", 0.10)))
    minimum_anchor_run = int(math.ceil(movie * _fraction(
        params, "minimum_anchor_run_movie_fraction", 0.15)))
    minimum_foreign = int(math.ceil(movie * _fraction(
        params, "minimum_foreign_occupancy_movie_fraction", 0.25)))
    minimum_prior_tenure = int(math.ceil(movie * _fraction(
        params, "minimum_foreign_prior_tenure_movie_fraction", 0.05)))
    maximum_step = float(params.get("maximum_step_sum_radii", 1.25))
    owner_presence_frames: dict[int, list[int]] = {}
    for frame, plane in enumerate(labels):
        for owner in np.unique(plane):
            if int(owner) > 0:
                owner_presence_frames.setdefault(int(owner), []).append(frame)
    link_index = {
        (int(edge["left"]), int(edge["right"])): edge for edge in links}
    rows: list[dict] = []
    internals: list[dict] = []

    for lineage in lineages:
        group = _lineage_group(lineage, groups)
        track_visible = group[group.physically_visible.astype(bool)]
        runs = _positive_runs(group)
        durable = [
            run for run in runs
            if int(run["first"]) <= maximum_anchor_onset
            and len(run["frames"]) >= minimum_anchor_run]
        anchor = max(
            durable,
            key=lambda run: (len(run["frames"]), -int(run["first"])),
            default=None)
        anchor_owner = int(anchor["owner"]) if anchor else 0
        anchor_rows = track_visible[
            track_visible.accepted_owner.astype(int).eq(anchor_owner)] \
            if anchor_owner > 0 else track_visible.iloc[0:0]
        anchor_last = int(anchor_rows.frame.max()) \
            if len(anchor_rows) else -1
        foreign = track_visible[
            track_visible.frame.astype(int).gt(anchor_last)
            & track_visible.accepted_owner.astype(int).gt(0)
            & track_visible.accepted_owner.astype(int).ne(anchor_owner)]
        foreign_owners = sorted(set(map(
            int, foreign.accepted_owner.astype(int).tolist())))
        established_owners = []
        unestablished_owners = []
        for owner in foreign_owners:
            first_invasion = int(foreign[
                foreign.accepted_owner.astype(int).eq(owner)].frame.min())
            prior_tenure = int(np.searchsorted(
                owner_presence_frames.get(owner, []), first_invasion))
            destination = (established_owners if
                           prior_tenure >= minimum_prior_tenure else
                           unestablished_owners)
            destination.append(owner)
        established_foreign = foreign[
            foreign.accepted_owner.astype(int).isin(established_owners)]
        median_radius = float(track_visible.radius_px.median()) \
            if len(track_visible) else 0.0
        radius_quantile = float((radius_by_track <= median_radius).mean()) \
            if len(radius_by_track) else 0.0
        largest_step = max(_steps(track_visible), default=float("inf"))
        reasons = []
        if len(track_visible) < minimum_visible:
            reasons.append("insufficient_visible_history")
        if radius_quantile < minimum_radius_quantile:
            reasons.append("reference_below_field_radius_quantile")
        if largest_step > maximum_step:
            reasons.append("reference_motion_discontinuous")
        if anchor is None:
            reasons.append("no_durable_early_owner")
        if len(foreign) < minimum_foreign:
            reasons.append("terminal_foreign_occupancy_too_short")
        if len(established_foreign) < minimum_foreign:
            reasons.append("insufficient_preestablished_foreign_occupancy")
        edge_text = []
        for left, right in zip(lineage, lineage[1:]):
            edge = link_index[(left, right)]
            edge_text.append(
                f"{left}>{right}:gap={edge['gap_frames']}:"
                f"step={edge['step_sum_radii']:.6f}")
        public = {
            "proposal_id": "", "lineage_tracks": "|".join(map(str, lineage)),
            "terminal_track": int(lineage[-1]),
            "endpoint_links": "|".join(edge_text),
            "anchor_owner": anchor_owner,
            "visible_frames": int(len(track_visible)),
            "visible_fraction": float(len(track_visible) / movie),
            "median_radius_px": median_radius,
            "radius_quantile": radius_quantile,
            "maximum_step_sum_radii": largest_step,
            "anchor_first": int(anchor["first"]) if anchor else -1,
            "anchor_last": anchor_last,
            "anchor_run_frames": len(anchor["frames"]) if anchor else 0,
            "anchor_support_frames": int(len(anchor_rows)),
            "foreign_frames": int(len(foreign)),
            "foreign_owner_count": len(foreign_owners),
            "foreign_owners": "|".join(map(str, foreign_owners)),
            "established_foreign_frames": int(len(established_foreign)),
            "established_foreign_owners": "|".join(
                map(str, established_owners)),
            "unestablished_foreign_owners": "|".join(
                map(str, unestablished_owners)),
            "single_core_foreign_owners": "",
            "multicore_foreign_owners": "",
            "selected_foreign_runs": 0,
            "rejected_multicore_runs": 0,
            "eligible": not reasons, "applied": False,
            "applied_runs": 0, "applied_frames": 0,
            "changed_label_pixels": 0, "changed_unclaimed_pixels": 0,
            "assigned_unclaimed_id": 0,
            "reason": ("eligible_terminal_physical_core_lineage"
                       if not reasons else "|".join(reasons)),
        }
        rows.append(public)
        internals.append({**public, "group": group, "foreign": foreign})

    number = 0
    for public, internal in zip(rows, internals):
        if bool(public["eligible"]):
            number += 1
            proposal_id = f"TPC{number:04d}"
            public["proposal_id"] = internal["proposal_id"] = proposal_id
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS), internals, scored,
            links)


def _inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])
