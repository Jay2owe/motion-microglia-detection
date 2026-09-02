"""Recover established residents when one owner invades separated body tracks.

Discovery is field-wide. The producer accepts mathematical thresholds only;
identities, tracks, frames, coordinates, events and review cases are forbidden.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals",
}


def _disk_owner(frame: np.ndarray, x: float, y: float,
                radius: float) -> tuple[int, float, float]:
    height, width = frame.shape
    radius = float(np.clip(0.75 * radius, 2.0, 6.0))
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(width, int(np.ceil(x + radius + 1)))
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(height, int(np.ceil(y + radius + 1)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    values = frame[y0:y1, x0:x1][disk]
    positive = values[values > 0]
    coverage = float(len(positive) / max(len(values), 1))
    if not len(positive):
        return 0, coverage, 0.0
    owners, counts = np.unique(positive, return_counts=True)
    best = int(np.argmax(counts))
    return int(owners[best]), coverage, float(counts[best] / len(positive))


def attach_owners(points: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    attached = points.copy()
    evidence = [
        _disk_owner(labels[int(row.frame)], float(row.x), float(row.y),
                    float(row.radius_px))
        for row in attached.itertuples(index=False)]
    attached["candidate_owner"] = [item[0] for item in evidence]
    attached["candidate_coverage_fraction"] = [item[1] for item in evidence]
    attached["candidate_owner_purity"] = [item[2] for item in evidence]
    attached["physically_visible"] = attached["state"].isin(
        ["observed", "latent_visible"])
    return attached


def _runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        owner = int(row.candidate_owner)
        if (runs and runs[-1]["owner"] == owner
                and int(row.frame) == runs[-1]["last_frame"] + 1):
            runs[-1]["last_frame"] = int(row.frame)
            runs[-1]["points"].append(row)
        else:
            runs.append({
                "owner": owner, "first_frame": int(row.frame),
                "last_frame": int(row.frame), "points": [row],
            })
    for run in runs:
        run["frames"] = len(run["points"])
    return runs


def _previous_positive(runs: list[dict], index: int,
                       maximum_gap: int) -> dict | None:
    source = runs[index]
    cursor = index - 1
    while cursor >= 0:
        candidate = runs[cursor]
        gap = source["first_frame"] - candidate["last_frame"] - 1
        if gap > maximum_gap:
            return None
        if candidate["owner"] > 0:
            return candidate
        cursor -= 1
    return None


def _point_at(proposal: pd.Series, frame: int):
    for point in proposal["points"]:
        if int(point.frame) == frame:
            return point
    return None


def _pair_is_separated(left: pd.Series, right: pd.Series,
                       params: dict) -> tuple[bool, float]:
    first = max(int(left.first_frame), int(right.first_frame))
    last = min(int(left.last_frame), int(right.last_frame))
    frame = max(int(left.first_frame), int(right.first_frame)) \
        if first > last else first
    left_point = _point_at(left, frame)
    right_point = _point_at(right, frame)
    if left_point is None or right_point is None:
        return False, 0.0
    distance = float(np.hypot(
        float(left_point.x) - float(right_point.x),
        float(left_point.y) - float(right_point.y)))
    separation = distance / max(
        float(left_point.radius_px) + float(right_point.radius_px), 1.0)
    required = float(params.get("minimum_pair_separation_sum_radii", 1.5))
    if bool(left_point.in_encounter) or bool(right_point.in_encounter):
        return False, separation
    return separation >= required, separation


def discover_proposals(attached: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Find source owners that invade multiple established separated tracks."""
    visible = attached[attached["physically_visible"].astype(bool)].copy()
    minimum_resident = int(params.get("minimum_resident_run_frames", 8))
    minimum_takeover = int(params.get("minimum_takeover_run_frames", 2))
    maximum_gap = int(params.get("maximum_adjoining_blank_frames", 2))
    maximum_step_radii = float(params.get("maximum_endpoint_step_radii", 1.5))
    minimum_purity = float(params.get("minimum_owner_purity", 0.8))
    minimum_coverage = float(params.get("minimum_owner_coverage", 0.5))
    rows: list[dict] = []

    for track_id, group in visible.groupby("track_id"):
        runs = _runs(group)
        for index, source in enumerate(runs):
            if source["owner"] <= 0:
                continue
            resident = _previous_positive(runs, index, maximum_gap)
            if resident is None or resident["owner"] == source["owner"]:
                continue
            values = pd.DataFrame([
                point._asdict() for point in source["points"]])
            endpoint = resident["points"][-1]
            first = source["points"][0]
            step = float(np.hypot(
                float(endpoint.x) - float(first.x),
                float(endpoint.y) - float(first.y)))
            step_radii = step / max(
                0.5 * (float(endpoint.radius_px) + float(first.radius_px)),
                1.0)
            reasons: list[str] = []
            if int(resident["frames"]) < minimum_resident:
                reasons.append("resident_run_too_short")
            if int(source["frames"]) < minimum_takeover:
                reasons.append("takeover_run_too_short")
            if step_radii > maximum_step_radii:
                reasons.append("physical_endpoint_discontinuity")
            if float(values["candidate_owner_purity"].median()) < minimum_purity:
                reasons.append("low_owner_purity")
            if float(values["candidate_coverage_fraction"].median()) < minimum_coverage:
                reasons.append("low_owner_coverage")
            rows.append({
                "proposal_id": f"P{len(rows) + 1:04d}",
                "track_id": int(track_id),
                "resident_owner": int(resident["owner"]),
                "source_owner": int(source["owner"]),
                "first_frame": int(source["first_frame"]),
                "last_frame": int(source["last_frame"]),
                "resident_run_frames": int(resident["frames"]),
                "takeover_run_frames": int(source["frames"]),
                "endpoint_step_radii": step_radii,
                "median_owner_purity": float(
                    values["candidate_owner_purity"].median()),
                "median_owner_coverage": float(
                    values["candidate_coverage_fraction"].median()),
                "points": source["points"],
                "precohort_status": "eligible" if not reasons else "rejected",
                "precohort_reason": (
                    "eligible" if not reasons else "|".join(reasons)),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "proposal_id", "track_id", "resident_owner", "source_owner",
            "first_frame", "last_frame", "resident_run_frames",
            "takeover_run_frames", "endpoint_step_radii",
            "median_owner_purity", "median_owner_coverage", "points",
            "precohort_status", "precohort_reason", "cohort_id",
            "cohort_tracks", "minimum_pair_separation_sum_radii",
            "discovery_status", "discovery_reason",
        ])
    proposals = pd.DataFrame(rows)
    proposals["cohort_id"] = ""
    proposals["cohort_tracks"] = 0
    proposals["minimum_pair_separation_sum_radii"] = np.nan
    proposals["discovery_status"] = "rejected"
    proposals["discovery_reason"] = proposals["precohort_reason"].where(
        proposals["precohort_status"] != "eligible", "no_separated_cohort")

    maximum_start_gap = int(params.get("maximum_cohort_start_gap_frames", 3))
    minimum_tracks = int(params.get("minimum_cohort_tracks", 2))
    cohort_number = 0
    eligible = proposals[proposals["precohort_status"] == "eligible"]
    for _, group in eligible.groupby("source_owner"):
        indices = list(group.index)
        neighbours: dict[int, set[int]] = {index: set() for index in indices}
        separations: dict[tuple[int, int], float] = {}
        for offset, left_index in enumerate(indices):
            for right_index in indices[offset + 1:]:
                left = proposals.loc[left_index]
                right = proposals.loc[right_index]
                if abs(int(left.first_frame) - int(right.first_frame)) > \
                        maximum_start_gap:
                    continue
                separated, value = _pair_is_separated(left, right, params)
                if not separated:
                    continue
                neighbours[left_index].add(right_index)
                neighbours[right_index].add(left_index)
                pair = (min(left_index, right_index),
                        max(left_index, right_index))
                separations[pair] = value
        unseen = set(indices)
        while unseen:
            seed = unseen.pop()
            component = {seed}
            stack = [seed]
            while stack:
                item = stack.pop()
                new = neighbours[item] & unseen
                unseen -= new
                component |= new
                stack.extend(new)
            residents = set(map(
                int, proposals.loc[list(component), "resident_owner"]))
            if len(component) < minimum_tracks or \
                    len(residents) < minimum_tracks:
                continue
            cohort_number += 1
            cohort_id = f"C{cohort_number:04d}"
            pair_values = [
                value for pair, value in separations.items()
                if pair[0] in component and pair[1] in component]
            proposals.loc[list(component), "cohort_id"] = cohort_id
            proposals.loc[list(component), "cohort_tracks"] = len(component)
            proposals.loc[list(component),
                          "minimum_pair_separation_sum_radii"] = min(pair_values)
            proposals.loc[list(component), "discovery_status"] = "eligible"
            proposals.loc[list(component), "discovery_reason"] = "eligible"
    return proposals


def _component_at(frame: np.ndarray, owner: int, point) -> np.ndarray | None:
    components, count = ndi.label(
        frame == owner, structure=np.ones((3, 3), np.uint8))
    if not count:
        return None
    y = int(np.clip(round(float(point.y)), 0, frame.shape[0] - 1))
    x = int(np.clip(round(float(point.x)), 0, frame.shape[1] - 1))
    value = int(components[y, x])
    if value <= 0:
        return None
    return components == value


def _geodesic_partition(component: np.ndarray,
                        member_points: list) -> np.ndarray:
    """Connected multi-source partition constrained to one mask component."""
    height, width = component.shape
    distance = np.full(component.shape, np.inf, dtype=float)
    assignment = np.full(component.shape, -1, dtype=np.int32)
    queue: list[tuple[float, int, int, int]] = []
    for index, point in enumerate(member_points):
        y = int(np.clip(round(float(point.y)), 0, height - 1))
        x = int(np.clip(round(float(point.x)), 0, width - 1))
        if not component[y, x]:
            raise ValueError("partition seed is outside its source component")
        distance[y, x] = 0.0
        assignment[y, x] = index
        heapq.heappush(queue, (0.0, index, y, x))
    neighbours = [
        (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)), (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)), (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
    ]
    while queue:
        current, owner, y, x = heapq.heappop(queue)
        if current != distance[y, x] or owner != assignment[y, x]:
            continue
        for dy, dx, cost in neighbours:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width
                    and component[ny, nx]):
                continue
            proposed = current + cost
            if (proposed < distance[ny, nx]
                    or (np.isclose(proposed, distance[ny, nx])
                        and owner < assignment[ny, nx])):
                distance[ny, nx] = proposed
                assignment[ny, nx] = owner
                heapq.heappush(queue, (proposed, owner, ny, nx))
    if np.any(component & (assignment < 0)):
        raise AssertionError(
            "geodesic partition left component pixels unassigned")
    return assignment


def apply_proposals(labels: np.ndarray, attached: pd.DataFrame,
                    proposals: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply eligible cohorts with absent-resident and exclusive-core gates."""
    candidate = labels.copy()
    audit: dict[str, dict] = {}
    for proposal in proposals.itertuples(index=False):
        row = proposal._asdict()
        row.pop("points", None)
        row.update({
            "outcome": ("pending" if proposal.discovery_status == "eligible"
                        else "rejected_discovery"),
            "changed_pixels": 0, "changed_frames": 0,
            "skipped_resident_present_frames": 0,
            "skipped_unmodelled_core_frames": 0,
        })
        audit[str(proposal.proposal_id)] = row

    eligible = proposals[proposals["discovery_status"] == "eligible"]
    if not len(eligible):
        return candidate, pd.DataFrame(audit.values())
    attached_index = attached.set_index(["track_id", "frame"], drop=False)
    changed_by_proposal: dict[str, set[int]] = {
        str(value): set() for value in eligible["proposal_id"]}

    for (_, source_owner), group in eligible.groupby(
            ["cohort_id", "source_owner"]):
        first = int(group["first_frame"].min())
        last = int(group["last_frame"].max())
        for frame in range(first, last + 1):
            active = group[(group["first_frame"] <= frame)
                           & (group["last_frame"] >= frame)]
            if not len(active):
                continue
            seen_components: set[bytes] = set()
            for proposal in active.itertuples(index=False):
                key = (int(proposal.track_id), frame)
                if key not in attached_index.index:
                    continue
                point = attached_index.loc[key]
                if isinstance(point, pd.DataFrame):
                    point = point.iloc[0]
                component = _component_at(
                    labels[frame], int(source_owner), point)
                if component is None:
                    continue
                token = np.packbits(component, axis=None).tobytes()
                if token in seen_components:
                    continue
                seen_components.add(token)

                component_proposals = []
                for member in active.itertuples(index=False):
                    member_key = (int(member.track_id), frame)
                    if member_key not in attached_index.index:
                        continue
                    member_point = attached_index.loc[member_key]
                    if isinstance(member_point, pd.DataFrame):
                        member_point = member_point.iloc[0]
                    y = int(np.clip(round(float(member_point.y)),
                                    0, component.shape[0] - 1))
                    x = int(np.clip(round(float(member_point.x)),
                                    0, component.shape[1] - 1))
                    if component[y, x]:
                        component_proposals.append((member, member_point))
                available = []
                for member, member_point in component_proposals:
                    if np.any(labels[frame] == int(member.resident_owner)):
                        audit[str(member.proposal_id)][
                            "skipped_resident_present_frames"] += 1
                    else:
                        available.append((member, member_point))
                if not available:
                    continue
                if len(available) != len(component_proposals):
                    continue

                modelled_tracks = {
                    int(member.track_id) for member, _ in component_proposals}
                component_tracks: set[int] = set()
                for other in attached[
                        attached["frame"] == frame].itertuples(index=False):
                    if not bool(other.physically_visible):
                        continue
                    y = int(np.clip(round(float(other.y)),
                                    0, component.shape[0] - 1))
                    x = int(np.clip(round(float(other.x)),
                                    0, component.shape[1] - 1))
                    if component[y, x]:
                        component_tracks.add(int(other.track_id))
                if component_tracks - modelled_tracks:
                    for member, _ in available:
                        audit[str(member.proposal_id)][
                            "skipped_unmodelled_core_frames"] += 1
                    continue

                assignments_full = _geodesic_partition(
                    component, [point for _, point in available])
                yy, xx = np.where(component)
                assignments = assignments_full[yy, xx]
                before = candidate[frame, yy, xx].copy()
                for index, (member, _) in enumerate(available):
                    selected = assignments == index
                    candidate[frame, yy[selected], xx[selected]] = int(
                        member.resident_owner)
                    changed = int(np.count_nonzero(
                        before[selected] != int(member.resident_owner)))
                    if changed:
                        proposal_id = str(member.proposal_id)
                        audit[proposal_id]["changed_pixels"] += changed
                        changed_by_proposal[proposal_id].add(frame)

    for proposal_id, frames in changed_by_proposal.items():
        audit[proposal_id]["changed_frames"] = len(frames)
        audit[proposal_id]["outcome"] = (
            "applied" if frames else "rejected_application")
    return candidate, pd.DataFrame(audit.values())


def recover(labels: np.ndarray, points: pd.DataFrame,
            params: dict) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    supplied = sorted(
        name for name in FORBIDDEN_TARGETS
        if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide resident takeover received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("resident takeover must use field-wide discovery")
    attached = attach_owners(points, labels)
    proposals = discover_proposals(attached, params)
    candidate, audit = apply_proposals(labels, attached, proposals)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("resident takeover changed foreground segmentation")
    return candidate, audit, attached
