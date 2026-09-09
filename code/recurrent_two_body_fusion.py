"""Carry two established identities through recurrent two-body fusions.

Discovery is field-wide. A pair must demonstrate a short two-to-one-to-two
calibration interval and then repeat the same directed absorption. Spatially
reused identities and crowded reciprocal track traffic are rejected. No
identity, track, frame, coordinate, event, region or review case is supplied.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile

import owner_consensus
import separable_merge_recovery as physical


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "forced_interval",
    "case_id", "allowed_pair",
)


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("recurrent fusion discovery must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "recurrent fusion discovery received forbidden targets: "
            + ", ".join(supplied))


def _components(frame: np.ndarray, owner: int) -> list[dict]:
    parts, count = ndi.label(frame == int(owner), STRUCTURE)
    rows: list[dict] = []
    for component in range(1, count + 1):
        mask = parts == component
        yy, xx = np.nonzero(mask)
        if not len(xx):
            continue
        rows.append({
            "mask": mask, "area": int(len(xx)),
            "x": float(xx.mean()), "y": float(yy.mean()),
            "radius": float(np.sqrt(len(xx) / np.pi)),
        })
    return rows


def _largest(frame: np.ndarray, owner: int) -> dict | None:
    rows = _components(frame, owner)
    return max(rows, key=lambda row: row["area"]) if rows else None


def _presence(labels: np.ndarray, owner: int) -> np.ndarray:
    return np.any(labels == int(owner), axis=(1, 2))


def _history_stats(labels: np.ndarray, owner: int) -> dict:
    rows = []
    for frame in range(len(labels)):
        component = _largest(labels[frame], owner)
        if component is not None:
            rows.append(component)
    if not rows:
        return {"frames": 0, "median_area": 0.0,
                "maximum_dispersion_radii": np.inf}
    centres = np.array([[row["x"], row["y"]] for row in rows], float)
    centre = np.median(centres, axis=0)
    radii = np.array([row["radius"] for row in rows], float)
    dispersion = np.linalg.norm(centres - centre[None, :], axis=1)
    return {
        "frames": int(len(rows)),
        "median_area": float(np.median([row["area"] for row in rows])),
        "maximum_dispersion_radii": float(
            dispersion.max() / max(float(np.median(radii)), 1.0)),
    }


def _absorption_geometry(previous: np.ndarray, current: np.ndarray,
                         lost: int, host: int) -> dict | None:
    before_lost = _largest(previous, lost)
    before_host = _largest(previous, host)
    after_host = _largest(current, host)
    if before_lost is None or before_host is None or after_host is None:
        return None
    expanded = ndi.binary_dilation(after_host["mask"], structure=STRUCTURE,
                                   iterations=2)
    absorbed = float(np.mean(expanded[before_lost["mask"]]))
    pixels = np.column_stack(np.nonzero(after_host["mask"]))
    point = np.array([before_lost["y"], before_lost["x"]], float)
    distance = float(np.min(np.linalg.norm(pixels - point[None, :], axis=1)))
    growth = ((float(after_host["area"]) - float(before_host["area"]))
              / max(float(before_lost["area"]), 1.0))
    return {
        "lost_area": int(before_lost["area"]),
        "host_area_before": int(before_host["area"]),
        "host_area_after": int(after_host["area"]),
        "host_distance_px": distance,
        "absorbed_fraction": absorbed,
        "host_growth_fraction": growth,
    }


def _geometry_passes(detail: dict | None, params: dict,
                     *, calibrated: bool = False) -> bool:
    if detail is None:
        return False
    maximum_growth = float(params[
        "maximum_calibrated_host_growth_fraction" if calibrated
        else "maximum_host_growth_fraction"])
    return bool(
        detail["host_distance_px"] <= float(params["merge_reach_px"])
        and detail["absorbed_fraction"] >= float(
            params["minimum_absorbed_fraction"])
        and float(params["minimum_host_growth_fraction"])
        <= detail["host_growth_fraction"] <= maximum_growth)


def _next_bookend(labels: np.ndarray, start: int, lost: int,
                  host: int) -> int | None:
    for frame in range(start + 1, len(labels)):
        present = set(map(int, np.unique(labels[frame]))) - {0}
        if lost in present and host in present:
            return frame
        if host not in present:
            return None
    return None


def _absorption_events(labels: np.ndarray, params: dict) -> pd.DataFrame:
    minimum_pre = int(params["minimum_pre_event_frames"])
    rows: list[dict] = []
    for frame in range(1, len(labels)):
        before_ids = set(map(int, np.unique(labels[frame - 1]))) - {0}
        after_ids = set(map(int, np.unique(labels[frame]))) - {0}
        vanished = sorted(before_ids - after_ids)
        continuing = sorted(before_ids & after_ids)
        for lost in vanished:
            first = max(0, frame - minimum_pre)
            if any(not np.any(labels[index] == lost)
                   for index in range(first, frame)):
                continue
            for host in continuing:
                detail = _absorption_geometry(
                    labels[frame - 1], labels[frame], lost, host)
                if not _geometry_passes(detail, params):
                    continue
                bookend = _next_bookend(labels, frame, lost, host)
                rows.append({
                    "lost_owner": int(lost), "host_owner": int(host),
                    "start_frame": int(frame),
                    "bookend_frame": -1 if bookend is None else int(bookend),
                    "bookend_gap_frames": (-1 if bookend is None
                                            else int(bookend - frame)),
                    **detail,
                })
    return pd.DataFrame(rows)


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find repeated directed absorptions with a self-validating bookend."""
    assert_target_free(params)
    events = _absorption_events(labels, params)
    if events.empty:
        return pd.DataFrame(), events, physical.attach_owners(points, labels)
    scored = physical.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)]
    history = {
        owner: _history_stats(labels, int(owner))
        for owner in sorted(set(events.lost_owner.astype(int))
                            | set(events.host_owner.astype(int)))
    }
    rows: list[dict] = []
    for (lost, host), group in events.groupby(
            ["lost_owner", "host_owner"], sort=True):
        group = group.sort_values("start_frame")
        first = group.iloc[0]
        lost_stats, host_stats = history[int(lost)], history[int(host)]
        transition_tracks = []
        for track, track_rows in visible.groupby("track_id", sort=True):
            owners = set(track_rows.accepted_owner.astype(int)) - {0}
            if {int(lost), int(host)} <= owners:
                transition_tracks.append(int(track))
        base = {
            "lost_owner": int(lost), "host_owner": int(host),
            "absorption_events": int(len(group)),
            "absorption_start_frames": "|".join(map(
                str, group.start_frame.astype(int))),
            "first_bookend_frame": int(first.bookend_frame),
            "first_bookend_gap_frames": int(first.bookend_gap_frames),
            "lost_presence_frames": int(lost_stats["frames"]),
            "host_presence_frames": int(host_stats["frames"]),
            "lost_median_area": float(lost_stats["median_area"]),
            "host_median_area": float(host_stats["median_area"]),
            "lost_maximum_dispersion_radii": float(
                lost_stats["maximum_dispersion_radii"]),
            "host_maximum_dispersion_radii": float(
                host_stats["maximum_dispersion_radii"]),
            "transition_tracks": "|".join(map(str, transition_tracks)),
            "eligible": False, "reason": "",
        }
        reasons = []
        if len(group) < int(params["minimum_recurrent_absorptions"]):
            reasons.append("not_recurrent")
        if int(first.bookend_frame) < 0 or int(first.bookend_gap_frames) > int(
                params["maximum_calibration_bookend_frames"]):
            reasons.append("no_short_calibration_bookend")
        if min(lost_stats["frames"], host_stats["frames"]) < int(
                params["minimum_compact_history_frames"]):
            reasons.append("insufficient_identity_history")
        if min(lost_stats["median_area"], host_stats["median_area"]) < float(
                params["minimum_median_component_area_px"]):
            reasons.append("substantial_body_missing")
        if max(lost_stats["maximum_dispersion_radii"],
               host_stats["maximum_dispersion_radii"]) > float(
                   params["maximum_identity_dispersion_radii"]):
            reasons.append("spatially_reused_or_nonlocal_identity")
        if not transition_tracks:
            reasons.append("no_physical_transition_track")
        rows.append({
            **base,
            "eligible": not reasons,
            "reason": ("eligible_recurrent_bookended_fusion"
                       if not reasons else "|".join(reasons)),
        })
    audit = pd.DataFrame(rows)
    audit.insert(0, "proposal_id", "")
    eligible_indices = list(audit.index[audit.eligible.astype(bool)])
    for number, index in enumerate(eligible_indices, 1):
        audit.loc[index, "proposal_id"] = f"RF{number:04d}"
    return audit, events, scored


def _nearest_pixel(mask: np.ndarray, y: float, x: float) -> tuple[int, int]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("cannot seed an empty host component")
    index = int(np.argmin((yy - float(y)) ** 2 + (xx - float(x)) ** 2))
    return int(yy[index]), int(xx[index])


def _host_component(frame: np.ndarray, host: int,
                    predicted: tuple[np.ndarray, np.ndarray],
                    params: dict) -> np.ndarray | None:
    options = _components(frame, host)
    if not options:
        return None
    midpoint = (predicted[0] + predicted[1]) / 2.0
    selected = min(options, key=lambda row: float(np.hypot(
        row["y"] - midpoint[0], row["x"] - midpoint[1])))
    distances = []
    pixels = np.column_stack(np.nonzero(selected["mask"]))
    for point in predicted:
        distances.append(float(np.min(np.linalg.norm(
            pixels - point[None, :], axis=1))))
    if max(distances) > float(params["maximum_seed_to_host_px"]):
        return None
    return selected["mask"]


def _basin_valid(part_lost: np.ndarray, part_host: np.ndarray,
                 expected_lost: float, expected_host: float,
                 params: dict) -> bool:
    lost_area, host_area = int(part_lost.sum()), int(part_host.sum())
    total = max(lost_area + host_area, 1)
    fraction = lost_area / total
    minimum = int(params["minimum_partition_pixels"])
    minimum_ratio = float(params["minimum_expected_area_ratio"])
    maximum_ratio = float(params["maximum_expected_area_ratio"])
    return bool(
        lost_area >= minimum and host_area >= minimum
        and float(params["minimum_lost_basin_fraction"]) <= fraction
        <= float(params["maximum_lost_basin_fraction"])
        and minimum_ratio <= lost_area / max(expected_lost, 1.0)
        <= maximum_ratio
        and minimum_ratio <= host_area / max(expected_host, 1.0)
        <= maximum_ratio)


def _seed_component(partition: np.ndarray,
                    seed: tuple[int, int]) -> np.ndarray:
    """Return the single 8-connected basin containing ``seed``."""
    components, _ = ndi.label(partition, STRUCTURE)
    component = int(components[seed])
    return components == component if component else np.zeros_like(partition)


def _connected_partition(mask: np.ndarray, part_lost: np.ndarray,
                         part_host: np.ndarray,
                         lost_seed: tuple[int, int],
                         host_seed: tuple[int, int], expected_lost: float,
                         expected_host: float, params: dict,
                         ) -> tuple[np.ndarray, np.ndarray] | None:
    """Remove partition slivers while keeping both physical basins connected.

    A two-marker split of a non-convex object can leave a tiny disconnected
    island on either side. Test both seed-preserving complements and retain the
    valid option closest to the original basin area. Ownership changes only
    inside the already merged foreground object.
    """
    options: list[tuple[int, np.ndarray, np.ndarray]] = []
    lost_core = _seed_component(part_lost, lost_seed)
    host_complement = mask & ~lost_core
    if (host_complement[host_seed]
            and int(ndi.label(host_complement, STRUCTURE)[1]) == 1
            and _basin_valid(lost_core, host_complement,
                             expected_lost, expected_host, params)):
        options.append((
            abs(int(lost_core.sum()) - int(part_lost.sum())),
            lost_core, host_complement))
    host_core = _seed_component(part_host, host_seed)
    lost_complement = mask & ~host_core
    if (lost_complement[lost_seed]
            and int(ndi.label(lost_complement, STRUCTURE)[1]) == 1
            and _basin_valid(lost_complement, host_core,
                             expected_lost, expected_host, params)):
        options.append((
            abs(int(lost_complement.sum()) - int(part_lost.sum())),
            lost_complement, host_core))
    if not options:
        return None
    _, connected_lost, connected_host = min(options, key=lambda row: row[0])
    return connected_lost, connected_host


def _partition(mask: np.ndarray, raw: np.ndarray,
               predicted_lost: np.ndarray, predicted_host: np.ndarray,
               expected_lost: float, expected_host: float,
               params: dict) -> tuple[np.ndarray, np.ndarray, str] | None:
    lost_seed = _nearest_pixel(
        mask, float(predicted_lost[0]), float(predicted_lost[1]))
    host_seed = _nearest_pixel(
        mask, float(predicted_host[0]), float(predicted_host[1]))
    if lost_seed == host_seed or float(np.linalg.norm(
            np.subtract(lost_seed, host_seed))) < float(
                params["minimum_seed_separation_px"]):
        return None
    markers = np.zeros(mask.shape, np.uint8)
    markers[lost_seed] = 1
    markers[host_seed] = 2
    smooth = ndi.gaussian_filter(
        raw.astype(np.float32), float(params["watershed_sigma_px"]))
    split = watershed(-smooth, markers=markers, mask=mask,
                      connectivity=STRUCTURE.astype(bool))
    raw_lost, raw_host = split == 1, split == 2
    if _basin_valid(raw_lost, raw_host, expected_lost, expected_host, params):
        connected = _connected_partition(
            mask, raw_lost, raw_host, lost_seed, host_seed,
            expected_lost, expected_host, params)
        if connected is not None:
            return (*connected, "raw_signal_watershed_connected")

    yy, xx = np.nonzero(mask)
    lost_radius = max(float(np.sqrt(expected_lost / np.pi)), 1.0)
    host_radius = max(float(np.sqrt(expected_host / np.pi)), 1.0)
    lost_distance = np.hypot(
        yy - float(predicted_lost[0]), xx - float(predicted_lost[1])) \
        / lost_radius
    host_distance = np.hypot(
        yy - float(predicted_host[0]), xx - float(predicted_host[1])) \
        / host_radius
    geometry_lost = np.zeros_like(mask)
    geometry_lost[yy[lost_distance <= host_distance],
                  xx[lost_distance <= host_distance]] = True
    geometry_host = mask & ~geometry_lost
    if not _basin_valid(
            geometry_lost, geometry_host, expected_lost, expected_host, params):
        return None
    connected = _connected_partition(
        mask, geometry_lost, geometry_host, lost_seed, host_seed,
        expected_lost, expected_host, params)
    if connected is None:
        return None
    return (*connected, "radius_normalised_geometry_connected")


def _state(frame: np.ndarray, owner: int) -> dict | None:
    component = _largest(frame, owner)
    if component is None:
        return None
    return {**component, "position": np.array(
        [component["y"], component["x"]], float)}


def _split_bookended(labels: np.ndarray, raw: np.ndarray, start: int,
                     future: int, lost: int, host: int, params: dict,
                     proposal_id: str, lost_track_rows: pd.DataFrame,
                     ) -> tuple[np.ndarray, list[dict]] | None:
    before_lost = _state(labels[start - 1], lost)
    before_host = _state(labels[start - 1], host)
    after_lost = _state(labels[future], lost)
    after_host = _state(labels[future], host)
    if any(row is None for row in (
            before_lost, before_host, after_lost, after_host)):
        return None
    candidate = labels.copy()
    rows: list[dict] = []
    span = future - (start - 1)
    for frame in range(start, future):
        if np.any(candidate[frame] == lost):
            return None
        alpha = (frame - (start - 1)) / span
        predicted_lost = ((1.0 - alpha) * before_lost["position"]
                          + alpha * after_lost["position"])
        predicted_host = ((1.0 - alpha) * before_host["position"]
                          + alpha * after_host["position"])
        expected_lost = ((1.0 - alpha) * before_lost["area"]
                         + alpha * after_lost["area"])
        expected_host = ((1.0 - alpha) * before_host["area"]
                         + alpha * after_host["area"])
        observed = lost_track_rows[
            lost_track_rows.frame.astype(int).eq(int(frame))]
        mask = _host_component(
            candidate[frame], host,
            (predicted_lost, predicted_host), params)
        partition = (None if mask is None else _partition(
            mask, raw[frame], predicted_lost, predicted_host,
            expected_lost, expected_host, params))
        used_track_seed = False
        if partition is None and len(observed):
            point = min(
                observed.itertuples(index=False),
                key=lambda row: float(np.hypot(
                    float(row.y) - predicted_lost[0],
                    float(row.x) - predicted_lost[1])))
            track_position = np.array([float(point.y), float(point.x)])
            track_mask = _host_component(
                candidate[frame], host,
                (track_position, predicted_host), params)
            partition = (None if track_mask is None else _partition(
                track_mask, raw[frame], track_position, predicted_host,
                expected_lost, expected_host, params))
            if partition is not None:
                mask = track_mask
                used_track_seed = True
        if partition is None:
            return None
        part_lost, part_host, method = partition
        if used_track_seed:
            method = "physical_track_fallback_" + method
        candidate[frame][mask] = 0
        candidate[frame][part_lost] = lost
        candidate[frame][part_host] = host
        rows.append({
            "proposal_id": proposal_id, "interval_type": "bookended",
            "frame": frame, "lost_owner": lost, "host_owner": host,
            "partition_method": method,
            "lost_pixels": int(part_lost.sum()),
            "host_pixels": int(part_host.sum()),
            "changed_pixels": int(part_lost.sum()),
        })
    return candidate, rows


def _split_terminal(labels: np.ndarray, raw: np.ndarray, start: int,
                    lost: int, host: int, params: dict,
                    proposal_id: str) -> tuple[np.ndarray, list[dict]] | None:
    candidate = labels.copy()
    previous_lost = _state(candidate[start - 1], lost)
    previous_host = _state(candidate[start - 1], host)
    older_lost = _state(candidate[max(0, start - 2)], lost)
    older_host = _state(candidate[max(0, start - 2)], host)
    if any(row is None for row in (
            previous_lost, previous_host, older_lost, older_host)):
        return None
    rows: list[dict] = []
    damping = float(params["terminal_velocity_damping"])
    for frame in range(start, len(candidate)):
        if np.any(candidate[frame] == lost):
            older_lost, older_host = previous_lost, previous_host
            previous_lost = _state(candidate[frame], lost)
            previous_host = _state(candidate[frame], host)
            if previous_lost is None or previous_host is None:
                return None
            continue
        predicted_lost = (previous_lost["position"] + damping * (
            previous_lost["position"] - older_lost["position"]))
        predicted_host = (previous_host["position"] + damping * (
            previous_host["position"] - older_host["position"]))
        expected_lost = float(previous_lost["area"])
        expected_host = float(previous_host["area"])
        mask = _host_component(
            candidate[frame], host,
            (predicted_lost, predicted_host), params)
        if mask is None:
            return None
        partition = _partition(
            mask, raw[frame], predicted_lost, predicted_host,
            expected_lost, expected_host, params)
        if partition is None:
            if len(rows) >= int(params["minimum_terminal_partition_frames"]):
                break
            return None
        part_lost, part_host, method = partition
        candidate[frame][mask] = 0
        candidate[frame][part_lost] = lost
        candidate[frame][part_host] = host
        older_lost, older_host = previous_lost, previous_host
        previous_lost = {**_state(candidate[frame], lost)}
        previous_host = {**_state(candidate[frame], host)}
        rows.append({
            "proposal_id": proposal_id, "interval_type": "right_censored",
            "frame": frame, "lost_owner": lost, "host_owner": host,
            "partition_method": method,
            "lost_pixels": int(part_lost.sum()),
            "host_pixels": int(part_host.sum()),
            "changed_pixels": int(part_lost.sum()),
        })
    return candidate, rows


def apply(labels: np.ndarray, raw: np.ndarray, pair_audit: pd.DataFrame,
          scored: pd.DataFrame, params: dict,
          ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate = labels.copy()
    applications: list[dict] = []
    verdicts: list[dict] = []
    eligible = pair_audit[pair_audit.eligible.astype(bool)] \
        if len(pair_audit) else pair_audit
    for proposal in eligible.itertuples(index=False):
        lost, host = int(proposal.lost_owner), int(proposal.host_owner)
        proposal_id = str(proposal.proposal_id)
        transition_tracks = {
            int(value) for value in str(proposal.transition_tracks).split("|")
            if value}
        lost_track_rows = scored[
            scored.physically_visible.astype(bool)
            & scored.track_id.astype(int).isin(transition_tracks)]
        before_pair = candidate.copy()
        completed_bookends = 0
        pair_rows: list[dict] = []
        failure = ""
        frame = 1
        while frame < len(candidate):
            previous_has_both = bool(
                np.any(candidate[frame - 1] == lost)
                and np.any(candidate[frame - 1] == host))
            current_lost = bool(np.any(candidate[frame] == lost))
            current_host = bool(np.any(candidate[frame] == host))
            if not previous_has_both or current_lost or not current_host:
                frame += 1
                continue
            detail = _absorption_geometry(
                candidate[frame - 1], candidate[frame], lost, host)
            if not _geometry_passes(
                    detail, params, calibrated=completed_bookends > 0):
                frame += 1
                continue
            future = _next_bookend(candidate, frame, lost, host)
            if future is not None and future - frame <= int(
                    params["maximum_recurrent_bookend_frames"]):
                result = _split_bookended(
                    candidate, raw, frame, future, lost, host, params,
                    proposal_id, lost_track_rows)
                if result is None:
                    failure = "bookended_partition_failed"
                    break
                candidate, rows = result
                pair_rows.extend(rows)
                completed_bookends += 1
                frame = future + 1
                continue
            if (future is None and completed_bookends >= int(
                    params["minimum_bookended_intervals_before_terminal"])):
                result = _split_terminal(
                    candidate, raw, frame, lost, host, params, proposal_id)
                if result is None:
                    failure = "terminal_partition_failed"
                    break
                candidate, rows = result
                pair_rows.extend(rows)
                frame = len(candidate)
                continue
            frame += 1
        if failure or completed_bookends < int(
                params["minimum_bookended_intervals_before_terminal"]):
            candidate = before_pair
            verdicts.append({
                "proposal_id": proposal_id, "lost_owner": lost,
                "host_owner": host, "outcome": "rejected_application",
                "reason": failure or "insufficient_applied_bookends",
                "bookended_intervals": completed_bookends,
                "changed_pixels": 0, "changed_frames": 0,
            })
            continue
        applications.extend(pair_rows)
        changed = candidate != before_pair
        verdicts.append({
            "proposal_id": proposal_id, "lost_owner": lost,
            "host_owner": host, "outcome": "applied",
            "reason": "recurrent_bookended_two_body_fusion",
            "bookended_intervals": completed_bookends,
            "changed_pixels": int(changed.sum()),
            "changed_frames": int(np.count_nonzero(
                changed.reshape(len(changed), -1).any(axis=1))),
        })
    return candidate, pd.DataFrame(verdicts), pd.DataFrame(applications)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    pair_audit, absorption_events, scored = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        verdicts, applications = pd.DataFrame(), pd.DataFrame()
    else:
        candidate, verdicts, applications = apply(
            labels, raw, pair_audit, scored, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("recurrent fusion recovery changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("recurrent fusion recovery changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"recurrent fusion recovery created {duplicates} duplicate components")

    output_stem = str(params.get(
        "output_stem", Path(params["labels_path"]).stem))
    labels_path = out.out / f"{output_stem}.tif"
    unclaimed_path = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    tifffile.imwrite(
        labels_path, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    application_audit = out.out / "application_audit.csv"
    conflicted_audit = out.out / "conflicted_lineage_audit.csv"
    application_source = (upstream_dir / "application_audit.csv"
                          if upstream_dir is not None else Path(
                              params["accepted_application_audit_path"]))
    conflict_source = (upstream_dir / "conflicted_lineage_audit.csv"
                       if upstream_dir is not None else Path(
                           params["accepted_conflicted_lineage_audit_path"]))
    shutil.copyfile(application_source, application_audit)
    shutil.copyfile(conflict_source, conflicted_audit)
    pair_path = out.out / "recurrent_fusion_pair_audit.csv"
    absorption_path = out.out / "absorption_event_audit.csv"
    verdict_path = out.out / "recurrent_fusion_verdicts.csv"
    application_path = out.out / "recurrent_fusion_frame_applications.csv"
    pair_audit.to_csv(pair_path, index=False)
    absorption_events.to_csv(absorption_path, index=False)
    verdicts.to_csv(verdict_path, index=False)
    applications.to_csv(application_path, index=False)
    changed = candidate != labels
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "pairs_audited": int(len(pair_audit)),
        "eligible_pairs": int(pair_audit.eligible.astype(bool).sum())
            if len(pair_audit) else 0,
        "applied_pairs": int(verdicts.outcome.eq("applied").sum())
            if len(verdicts) else 0,
        "absorption_events_audited": int(len(absorption_events)),
        "application_rows": int(len(applications)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": bool(np.array_equal(
            candidate > 0, labels > 0)),
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {
        "outputs": {
            "labels": labels_path, "unclaimed": unclaimed_path,
            "application_audit": application_audit,
            "conflicted_lineage_audit": conflicted_audit,
            "pair_audit": pair_path, "absorption_events": absorption_path,
            "verdicts": verdict_path, "applications": application_path,
            "metrics": metrics_path,
        },
        "summary": metrics,
    }
