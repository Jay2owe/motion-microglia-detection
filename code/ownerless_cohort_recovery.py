"""Recover medium-lived, never-owned physical-body cohorts field-wide.

Seeds are selected relative to the accepted physical detector's observation
requirements. Reciprocal endpoint links join adjacent detector fragments before
one deterministic identity is allocated to each physical seat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ownerless_body_recovery


FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals",
}


def _summary_rows(attached: pd.DataFrame, params: dict) -> pd.DataFrame:
    detector_minimum_observations = int(
        params.get("detector_minimum_track_observations", 3))
    detector_minimum_strong = int(
        params.get("detector_minimum_strong_observations", 2))
    minimum_seed_observations = int(np.ceil(
        float(params.get("minimum_seed_observation_multiple", 2.5))
        * detector_minimum_observations))
    minimum_seed_strong = int(np.ceil(
        float(params.get("minimum_seed_strong_multiple", 1.5))
        * detector_minimum_strong))
    minimum_follower_observations = int(np.ceil(
        float(params.get("minimum_follower_observation_multiple", 1.0))
        * detector_minimum_observations))
    minimum_follower_strong = int(np.ceil(
        float(params.get("minimum_follower_strong_multiple", 1.0))
        * detector_minimum_strong))
    minimum_separation = float(
        params.get("minimum_median_separation_radii", 4.0))
    maximum_encounter = float(params.get("maximum_encounter_fraction", 0.0))
    maximum_coverage = float(
        params.get("maximum_existing_coverage_fraction", 0.0))
    rows: list[dict] = []
    for track_id, group in attached.groupby("track_id", sort=True):
        visible = group[group["physically_visible"].astype(bool)].copy()
        observed = visible[visible["state"] == "observed"]
        radii = np.maximum(visible["radius_px"].to_numpy(float), 1.0)
        separation = (visible["nearest_track_distance_px"].to_numpy(float)
                      / radii) if len(visible) else np.array([])
        values = {
            "proposal_id": f"P{len(rows) + 1:04d}",
            "track_id": int(track_id),
            "first_visible_frame": int(visible["frame"].min()) if len(visible) else -1,
            "last_visible_frame": int(visible["frame"].max()) if len(visible) else -1,
            "observed_frames": int(len(observed)),
            "visible_frames": int(len(visible)),
            "strong_observations": int(observed["strong"].astype(bool).sum()),
            "median_separation_radii": float(np.median(separation)) if len(separation) else 0.0,
            "encounter_fraction": float(visible["in_encounter"].astype(bool).mean()) if len(visible) else 1.0,
            "maximum_assigned_coverage": float(visible["assigned_coverage_fraction"].max()) if len(visible) else 1.0,
            "maximum_unclaimed_coverage": float(visible["unclaimed_coverage_fraction"].max()) if len(visible) else 1.0,
            "median_x": float(visible["x"].median()) if len(visible) else np.nan,
            "median_y": float(visible["y"].median()) if len(visible) else np.nan,
        }
        common = (
            values["median_separation_radii"] >= minimum_separation
            and values["encounter_fraction"] <= maximum_encounter
            and values["maximum_assigned_coverage"] <= maximum_coverage
            and values["maximum_unclaimed_coverage"] <= maximum_coverage)
        seed = (common
                and values["observed_frames"] >= minimum_seed_observations
                and values["strong_observations"] >= minimum_seed_strong)
        follower = (common
                    and values["observed_frames"] >= minimum_follower_observations
                    and values["strong_observations"] >= minimum_follower_strong)
        reasons: list[str] = []
        if values["maximum_assigned_coverage"] > maximum_coverage:
            reasons.append("has_assigned_mask_coverage")
        if values["maximum_unclaimed_coverage"] > maximum_coverage:
            reasons.append("has_unclaimed_mask_coverage")
        if values["encounter_fraction"] > maximum_encounter:
            reasons.append("encountered_other_body")
        if values["median_separation_radii"] < minimum_separation:
            reasons.append("insufficient_isolation")
        if values["observed_frames"] < minimum_follower_observations:
            reasons.append("too_few_direct_observations")
        if values["strong_observations"] < minimum_follower_strong:
            reasons.append("too_few_strong_observations")
        role = "seed" if seed else "follower_candidate" if follower else "rejected"
        values.update({
            "discovery_role": role,
            "discovery_reason": role if role != "rejected" else "|".join(reasons),
        })
        rows.append(values)
    return pd.DataFrame(rows)


def _endpoint(group: pd.DataFrame, first: bool):
    ordered = group[group["physically_visible"].astype(bool)].sort_values("frame")
    return ordered.iloc[0] if first else ordered.iloc[-1]


def _fragment_edges(attached: pd.DataFrame, summaries: pd.DataFrame,
                    params: dict) -> pd.DataFrame:
    eligible = summaries[summaries["discovery_role"].isin(
        ["seed", "follower_candidate"])]
    maximum_gap = int(params.get("maximum_fragment_gap_frames", 1))
    maximum_distance = float(
        params.get("maximum_endpoint_distance_sum_radii", 2.0))
    candidates: list[dict] = []
    groups = {int(tid): group for tid, group in attached.groupby("track_id")}
    for left in eligible.itertuples(index=False):
        end = _endpoint(groups[int(left.track_id)], False)
        for right in eligible.itertuples(index=False):
            if int(left.track_id) == int(right.track_id):
                continue
            start = _endpoint(groups[int(right.track_id)], True)
            gap = int(start.frame) - int(end.frame) - 1
            if gap < 0 or gap > maximum_gap:
                continue
            distance = float(np.hypot(float(start.x) - float(end.x),
                                      float(start.y) - float(end.y)))
            distance_radii = distance / max(
                float(start.radius_px) + float(end.radius_px), 1.0)
            if distance_radii > maximum_distance:
                continue
            candidates.append({
                "track_before": int(left.track_id),
                "track_after": int(right.track_id),
                "end_frame": int(end.frame), "start_frame": int(start.frame),
                "gap_frames": gap, "distance_px": distance,
                "distance_sum_radii": distance_radii,
            })
    columns = [
        "edge_id", "track_before", "track_after", "end_frame",
        "start_frame", "gap_frames", "distance_px",
        "distance_sum_radii", "edge_status"]
    if not candidates:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(candidates).sort_values(
        ["distance_sum_radii", "gap_frames", "track_before", "track_after"])
    best_after = table.groupby("track_before")["distance_sum_radii"].transform("min")
    best_before = table.groupby("track_after")["distance_sum_radii"].transform("min")
    table["edge_status"] = np.where(
        np.isclose(table["distance_sum_radii"], best_after)
        & np.isclose(table["distance_sum_radii"], best_before),
        "reciprocal", "rejected_nonreciprocal")
    table = table.sort_values(["track_before", "track_after"]).reset_index(drop=True)
    table.insert(0, "edge_id", [f"E{i:04d}" for i in range(1, len(table) + 1)])
    return table[columns]


def discover_seats(attached: pd.DataFrame, params: dict,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = _summary_rows(attached, params)
    edges = _fragment_edges(attached, summaries, params)
    seeds = set(map(int, summaries.loc[
        summaries.discovery_role == "seed", "track_id"]))
    eligible_tracks = set(map(int, summaries.loc[
        summaries.discovery_role.isin(["seed", "follower_candidate"]),
        "track_id"]))
    neighbours = {track: set() for track in eligible_tracks}
    for edge in edges[edges.edge_status == "reciprocal"].itertuples(index=False):
        neighbours[int(edge.track_before)].add(int(edge.track_after))
        neighbours[int(edge.track_after)].add(int(edge.track_before))
    components: list[set[int]] = []
    unseen = set(eligible_tracks)
    while unseen:
        component_seed = unseen.pop()
        component = {component_seed}
        stack = [component_seed]
        while stack:
            item = stack.pop()
            new = neighbours[item] & unseen
            unseen -= new
            component |= new
            stack.extend(new)
        if component & seeds:
            components.append(component)
    components.sort(key=lambda component: (
        int(summaries[summaries.track_id.isin(component)].first_visible_frame.min()),
        float(summaries[summaries.track_id.isin(component)].median_y.median()),
        float(summaries[summaries.track_id.isin(component)].median_x.median())))
    seat_rows: list[dict] = []
    for index, component in enumerate(components, start=1):
        group = summaries[summaries.track_id.isin(component)]
        seat_rows.append({
            "seat_id": f"S{index:04d}",
            "member_tracks": "|".join(map(str, sorted(component))),
            "track_count": len(component),
            "seed_tracks": int(group.discovery_role.eq("seed").sum()),
            "first_frame": int(group.first_visible_frame.min()),
            "last_frame": int(group.last_visible_frame.max()),
            "observed_frames": int(group.observed_frames.sum()),
            "strong_observations": int(group.strong_observations.sum()),
            "median_x": float(group.median_x.median()),
            "median_y": float(group.median_y.median()),
        })
    seats = pd.DataFrame(seat_rows)
    seat_by_track = {}
    for seat in seat_rows:
        for track in map(int, str(seat["member_tracks"]).split("|")):
            seat_by_track[track] = seat["seat_id"]
    summaries["seat_id"] = summaries.track_id.map(seat_by_track).fillna("")
    summaries["discovery_status"] = np.where(
        summaries.seat_id != "", "eligible", "rejected")
    return summaries, edges, seats


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame,
                       pd.DataFrame, pd.DataFrame]:
    supplied = sorted(name for name in FORBIDDEN_TARGETS
                      if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError("field-wide ownerless cohort received forbidden targets: "
                         + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("ownerless cohort must use field-wide discovery")
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")
    attached = ownerless_body_recovery.attach_coverage(points, labels, unclaimed)
    summaries, edges, seats = discover_seats(attached, params)
    candidate = labels.copy()
    threshold_by_frame = thresholds.set_index("frame")
    maximum_ratio = float(params.get("maximum_core_area_radius_ratio", 2.0))
    next_identity = int(candidate.max()) + 1
    application_rows: list[dict] = []
    for seat in seats.itertuples(index=False):
        identity = next_identity
        next_identity += 1
        changed_pixels = rejected_overlap = rejected_core = zero_signal = 0
        member_tracks = set(map(int, str(seat.member_tracks).split("|")))
        group = attached[(attached.track_id.isin(member_tracks))
                         & (attached.state == "observed")].sort_values(
                             ["frame", "track_id"])
        frames_changed: set[int] = set()
        for point in group.itertuples(index=False):
            frame_index = int(point.frame)
            core, region, details = ownerless_body_recovery._connected_core(
                raw[frame_index], point,
                float(threshold_by_frame.loc[frame_index, "weak_threshold"]),
                params)
            expected_area = np.pi * max(float(point.radius_px), 1.0) ** 2
            area_ratio = details["core_area_px"] / max(expected_area, 1.0)
            if not details["core_area_px"] or area_ratio > maximum_ratio:
                rejected_core += 1
                continue
            occupied = ((candidate[frame_index][region] > 0)
                        | (unclaimed[frame_index][region] > 0))
            if np.any(core & occupied):
                rejected_overlap += 1
                continue
            candidate[frame_index][region][core] = identity
            changed_pixels += int(core.sum())
            frames_changed.add(frame_index)
            zero_signal += int(np.count_nonzero(
                core & (raw[frame_index][region] == 0)))
        application_rows.append({
            **seat._asdict(), "assigned_identity": identity,
            "outcome": "applied" if frames_changed else "rejected_application",
            "changed_pixels": changed_pixels,
            "changed_frames": len(frames_changed),
            "rejected_overlap_frames": rejected_overlap,
            "rejected_core_frames": rejected_core,
            "zero_signal_additions": zero_signal,
        })
    applications = pd.DataFrame(application_rows)
    identity_by_seat = ({str(row.seat_id): int(row.assigned_identity)
                         for row in applications.itertuples(index=False)}
                        if len(applications) else {})
    summaries["assigned_identity"] = summaries.seat_id.map(
        identity_by_seat).fillna(0).astype(int)
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError("ownerless cohort changed an existing label pixel")
    if np.any((unclaimed > 0) & (candidate > 0)):
        raise AssertionError("ownerless cohort overlapped the unclaimed ledger")
    if np.any((candidate > 0) & (labels == 0) & (raw == 0)):
        raise AssertionError("ownerless cohort added a zero-signal pixel")
    return candidate, summaries, edges, applications, attached
