"""Production linker for candidate-independent latent physical-body tracks."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import tifffile

import raw_physical_hypotheses


@dataclass
class Track:
    track_id: int
    observations: list[dict] = field(default_factory=list)


def _patch(frame: np.ndarray, x: float, y: float, radius: int = 4) -> np.ndarray:
    cy, cx = int(round(y)), int(round(x))
    padded = np.pad(frame.astype(np.float32, copy=False), radius)
    patch = padded[cy:cy + 2 * radius + 1, cx:cx + 2 * radius + 1]
    patch = patch - float(np.mean(patch))
    norm = float(np.linalg.norm(patch))
    return patch.ravel() / norm if norm > 0 else patch.ravel()


def _predicted(track: Track, frame: int, damping: float) -> tuple[float, float]:
    last = track.observations[-1]
    if len(track.observations) < 2:
        return float(last["x"]), float(last["y"])
    previous = track.observations[-2]
    elapsed = max(int(last["frame"]) - int(previous["frame"]), 1)
    gap = frame - int(last["frame"])
    vx = (float(last["x"]) - float(previous["x"])) / elapsed
    vy = (float(last["y"]) - float(previous["y"])) / elapsed
    return (float(last["x"]) + damping * vx * gap,
            float(last["y"]) + damping * vy * gap)


def _motion_support(
        motion: np.ndarray | None, frame: int,
        x0: float, y0: float, x1: float, y1: float) -> float:
    if motion is None or not len(motion):
        return 0.0
    transition = min(max(frame, 0), len(motion) - 1)
    plane = np.max(motion[transition, 1:], axis=0).astype(np.float32)
    samples = []
    for weight in np.linspace(0.0, 1.0, 7):
        x = int(round((1 - weight) * x0 + weight * x1))
        y = int(round((1 - weight) * y0 + weight * y1))
        y0s, y1s = max(0, y - 1), min(plane.shape[0], y + 2)
        x0s, x1s = max(0, x - 1), min(plane.shape[1], x + 2)
        if y0s < y1s and x0s < x1s:
            samples.append(float(np.max(plane[y0s:y1s, x0s:x1s])))
    return float(np.clip(max(samples, default=0.0) / 65535.0, 0.0, 1.0))


def _motion_support_plane(
        plane: np.ndarray | None,
        x0: float, y0: float, x1: float, y1: float) -> float:
    if plane is None:
        return 0.0
    samples = []
    for weight in np.linspace(0.0, 1.0, 7):
        x = int(round((1 - weight) * x0 + weight * x1))
        y = int(round((1 - weight) * y0 + weight * y1))
        y0s, y1s = max(0, y - 1), min(plane.shape[0], y + 2)
        x0s, x1s = max(0, x - 1), min(plane.shape[1], x + 2)
        if y0s < y1s and x0s < x1s:
            samples.append(float(np.max(plane[y0s:y1s, x0s:x1s])))
    return float(np.clip(max(samples, default=0.0) / 65535.0, 0.0, 1.0))


def link_observations(
        hypotheses: pd.DataFrame, raw: np.ndarray,
        motion: np.ndarray | None, params: dict) -> list[Track]:
    maximum_gap = int(params.get("maximum_track_gap_frames", 16))
    base_distance = float(params.get("base_link_distance_px", 7.0))
    gap_distance = float(params.get("additional_link_px_per_gap", 1.25))
    maximum_distance = float(params.get("maximum_link_distance_px", 18.0))
    damping = float(params.get("velocity_prediction_damping", 0.5))
    area_weight = float(params.get("area_cost_weight", 0.25))
    appearance_weight = float(params.get("appearance_cost_weight", 0.50))
    motion_weight = float(params.get("motion_cost_reward", 0.15))
    motion_extension = float(params.get("motion_link_extension_px", 0.0))
    maximum_gap_bridge_speed_radii = float(
        params.get("maximum_gap_bridge_speed_sum_radii_per_frame", np.inf))
    tracks: list[Track] = []
    next_id = 1
    motion_planes = (np.max(motion[:, 1:], axis=1) if motion is not None
                     and motion.ndim == 4 and motion.shape[1] > 1 else None)
    by_frame = {int(frame): group.to_dict("records")
                for frame, group in hypotheses.groupby("frame")}
    for frame in range(len(raw)):
        detections = by_frame.get(frame, [])
        motion_plane = (motion_planes[min(frame, len(motion_planes) - 1)]
                        if motion_planes is not None and len(motion_planes)
                        else None)
        active = [index for index, track in enumerate(tracks)
                  if 0 < frame - int(track.observations[-1]["frame"])
                  <= maximum_gap]
        cost = np.full((len(active), len(detections)), 1e6, np.float64)
        for ai, track_index in enumerate(active):
            track = tracks[track_index]
            last = track.observations[-1]
            gap = frame - int(last["frame"])
            px, py = _predicted(track, frame, damping)
            allowed = min(maximum_distance,
                          base_distance + gap_distance * max(0, gap - 1))
            last_patch = _patch(raw[int(last["frame"])], last["x"], last["y"])
            for di, detection in enumerate(detections):
                endpoint_distance = float(np.hypot(
                    float(detection["x"]) - float(last["x"]),
                    float(detection["y"]) - float(last["y"])))
                endpoint_radii = max(
                    float(last["radius_px"])
                    + float(detection["radius_px"]), 1e-6)
                if (gap > 1
                        and endpoint_distance / (endpoint_radii * gap)
                        > maximum_gap_bridge_speed_radii):
                    continue
                distance = float(np.hypot(
                    float(detection["x"]) - px,
                    float(detection["y"]) - py))
                if distance > maximum_distance:
                    continue
                motion_support = _motion_support_plane(
                    motion_plane, last["x"], last["y"],
                    detection["x"], detection["y"])
                adaptive_allowed = min(
                    maximum_distance, allowed + motion_extension * motion_support)
                if distance > adaptive_allowed:
                    continue
                area_change = abs(np.log(
                    max(float(detection["core_area_px"]), 1.0)
                    / max(float(last["core_area_px"]), 1.0)))
                current_patch = _patch(
                    raw[frame], detection["x"], detection["y"])
                correlation = float(np.clip(
                    np.dot(last_patch, current_patch), -1.0, 1.0))
                cost[ai, di] = (
                    distance / max(adaptive_allowed, 1e-6)
                    + area_weight * area_change
                    + appearance_weight * (1.0 - correlation)
                    - motion_weight * motion_support)
        used_detections: set[int] = set()
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            for ai, di in zip(rows, columns):
                if cost[ai, di] >= 1e5:
                    continue
                tracks[active[int(ai)]].observations.append(detections[int(di)])
                used_detections.add(int(di))
        for di, detection in enumerate(detections):
            if di in used_detections or not bool(detection["strong"]):
                continue
            tracks.append(Track(next_id, [detection]))
            next_id += 1
    return tracks


def _sample_evidence(
        evidence: np.ndarray, x: float, y: float, radius: int = 2) -> float:
    cy, cx = int(round(y)), int(round(x))
    y0, y1 = max(0, cy - radius), min(evidence.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(evidence.shape[1], cx + radius + 1)
    return float(np.max(evidence[y0:y1, x0:x1])) if y0 < y1 and x0 < x1 else 0.0


def expand_track_points(
        tracks: list[Track], raw: np.ndarray, thresholds: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum_observations = int(params.get("minimum_track_observations", 4))
    minimum_strong = int(params.get("minimum_strong_observations", 2))
    latent_fraction = float(params.get("latent_weak_threshold_fraction", 0.5))
    minimum_latent_visibility = float(params.get("minimum_latent_visibility", 0.05))
    threshold_map = thresholds.set_index("frame").to_dict("index")
    evidence_cache = [raw_physical_hypotheses.evidence_image(frame, params)
                      for frame in raw]
    point_rows: list[dict] = []
    track_rows: list[dict] = []
    retained_id = 1
    for track in tracks:
        observations = sorted(track.observations, key=lambda row: int(row["frame"]))
        strong_count = sum(bool(row["strong"]) for row in observations)
        if (len(observations) < minimum_observations
                or strong_count < minimum_strong):
            continue
        by_frame = {int(row["frame"]): row for row in observations}
        first, last = min(by_frame), max(by_frame)
        observed_frames = sorted(by_frame)
        for frame in range(first, last + 1):
            if frame in by_frame:
                observation = by_frame[frame]
                x, y = float(observation["x"]), float(observation["y"])
                radius = float(observation["radius_px"])
                visibility = max(float(observation["visibility"]), 0.05)
                state = "observed"
                evidence_value = float(observation["evidence"])
                strong = bool(observation["strong"])
            else:
                left = max(value for value in observed_frames if value < frame)
                right = min(value for value in observed_frames if value > frame)
                weight = (frame - left) / (right - left)
                a, b = by_frame[left], by_frame[right]
                x = float((1 - weight) * float(a["x"]) + weight * float(b["x"]))
                y = float((1 - weight) * float(a["y"]) + weight * float(b["y"]))
                radius = float((1 - weight) * float(a["radius_px"])
                               + weight * float(b["radius_px"]))
                evidence_value = _sample_evidence(evidence_cache[frame], x, y)
                threshold = threshold_map[frame]
                weak = latent_fraction * float(threshold["weak_threshold"])
                strong_threshold = float(threshold["strong_threshold"])
                visibility = float(np.clip(
                    (evidence_value - weak)
                    / max(strong_threshold - weak, 1e-6), 0.0, 1.0))
                state = ("latent_visible" if visibility >= minimum_latent_visibility
                         else "dormant")
                strong = False
            point_rows.append({
                "track_id": retained_id, "frame": frame,
                "x": x, "y": y, "radius_px": radius,
                "evidence": evidence_value, "visibility": visibility,
                "state": state, "strong": strong,
            })
        track_rows.append({
            "track_id": retained_id,
            "first_frame": first, "last_frame": last,
            "observations": len(observations),
            "strong_observations": strong_count,
            "lifespan_frames": last - first + 1,
            "latent_frames": last - first + 1 - len(observations),
            "median_radius_px": float(np.median(
                [float(row["radius_px"]) for row in observations])),
        })
        retained_id += 1
    points = pd.DataFrame(point_rows)
    summaries = pd.DataFrame(track_rows)
    if points.empty:
        return points, summaries
    nearest = np.full(len(points), np.inf, np.float64)
    encounter = np.zeros(len(points), bool)
    for _, indices in points.groupby("frame").groups.items():
        index = np.asarray(list(indices), int)
        xy = points.loc[index, ["x", "y"]].to_numpy(float)
        if len(index) < 2:
            continue
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        local_nearest = distances.min(axis=1)
        nearest[index] = local_nearest
        radii = points.loc[index, "radius_px"].to_numpy(float)
        limits = np.maximum(10.0, 1.5 * (radii[:, None] + radii[None, :]))
        np.fill_diagonal(limits, 0.0)
        encounter[index] = np.any(distances <= limits, axis=1)
    points["nearest_track_distance_px"] = nearest
    points["in_encounter"] = encounter
    return points, summaries


def _valley_ratio(
        frame: np.ndarray, left: tuple[float, float],
        right: tuple[float, float]) -> float:
    distance = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    samples = max(3, int(np.ceil(distance)) + 1)
    values = []
    for weight in np.linspace(0.0, 1.0, samples):
        x = int(round((1 - weight) * left[0] + weight * right[0]))
        y = int(round((1 - weight) * left[1] + weight * right[1]))
        y = int(np.clip(y, 0, frame.shape[0] - 1))
        x = int(np.clip(x, 0, frame.shape[1] - 1))
        values.append(float(frame[y, x]))
    endpoint = max(min(values[0], values[-1]), 1.0)
    interior = values[1:-1] if len(values) > 2 else values
    return float(np.clip(min(interior) / endpoint, 0.0, 2.0))


def encounter_analysis(
        points: pd.DataFrame, raw: np.ndarray | None = None,
        maximum_valley_ratio: float = 0.65,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if points.empty:
        episodes = pd.DataFrame(columns=[
            "encounter_id", "track_a", "track_b", "start_frame",
            "end_frame", "frames", "minimum_distance_px",
            "separable_frames", "separable_fraction", "median_valley_ratio"])
        frames = pd.DataFrame(columns=[
            "track_a", "track_b", "frame", "distance_px",
            "valley_ratio", "separable"])
        return episodes, frames
    pair_frames: dict[tuple[int, int], list[dict]] = {}
    visible = points[(points["state"] == "observed")
                     | (points["state"] == "latent_visible")]
    for frame, group in visible.groupby("frame"):
        rows = list(group.itertuples(index=False))
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                distance = float(np.hypot(left.x - right.x, left.y - right.y))
                limit = max(10.0, 1.5 * (left.radius_px + right.radius_px))
                if distance <= limit:
                    pair = tuple(sorted((int(left.track_id), int(right.track_id))))
                    ratio = (_valley_ratio(
                        raw[int(frame)], (left.x, left.y), (right.x, right.y))
                        if raw is not None else 0.0)
                    pair_frames.setdefault(pair, []).append({
                        "track_a": pair[0], "track_b": pair[1],
                        "frame": int(frame), "distance_px": distance,
                        "valley_ratio": ratio,
                        "separable": bool(distance >= 4.0
                                          and ratio <= maximum_valley_ratio),
                    })
    rows: list[dict] = []
    frame_rows: list[dict] = []
    encounter_id = 1
    for pair, values in sorted(pair_frames.items()):
        run: list[dict] = []
        for item in sorted(values, key=lambda row: row["frame"]):
            if run and item["frame"] > run[-1]["frame"] + 1:
                separable = sum(bool(value["separable"]) for value in run)
                rows.append({
                    "encounter_id": encounter_id, "track_a": pair[0],
                    "track_b": pair[1], "start_frame": run[0]["frame"],
                    "end_frame": run[-1]["frame"], "frames": len(run),
                    "minimum_distance_px": min(
                        value["distance_px"] for value in run),
                    "separable_frames": separable,
                    "separable_fraction": separable / len(run),
                    "median_valley_ratio": float(np.median(
                        [value["valley_ratio"] for value in run])),
                })
                frame_rows.extend({**value, "encounter_id": encounter_id}
                                  for value in run)
                encounter_id += 1
                run = []
            run.append(item)
        if run:
            separable = sum(bool(value["separable"]) for value in run)
            rows.append({
                "encounter_id": encounter_id, "track_a": pair[0],
                "track_b": pair[1], "start_frame": run[0]["frame"],
                "end_frame": run[-1]["frame"], "frames": len(run),
                "minimum_distance_px": min(
                    value["distance_px"] for value in run),
                "separable_frames": separable,
                "separable_fraction": separable / len(run),
                "median_valley_ratio": float(np.median(
                    [value["valley_ratio"] for value in run])),
            })
            frame_rows.extend({**value, "encounter_id": encounter_id}
                              for value in run)
            encounter_id += 1
    return pd.DataFrame(rows), pd.DataFrame(frame_rows)


def encounter_episodes(points: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible episode-only view used by focused tests."""
    episodes, _ = encounter_analysis(points)
    return episodes


def reconnection_hypotheses(
        points: pd.DataFrame, raw: np.ndarray, params: dict) -> pd.DataFrame:
    """Find unique raw-only links between conservative tracklet endpoints."""
    columns = [
        "reconnection_id", "track_before", "track_after", "end_frame",
        "start_frame", "gap_frames", "distance_px", "patch_correlation",
        "link_cost", "uniqueness_margin"]
    if points.empty:
        return pd.DataFrame(columns=columns)
    visible = points[points["state"].isin(["observed", "latent_visible"])]
    grouped = {int(track_id): group.sort_values("frame")
               for track_id, group in visible.groupby("track_id")}
    maximum_gap = int(params.get("maximum_reconnection_gap_frames", 16))
    base_distance = float(params.get("reconnection_base_distance_px", 5.0))
    gap_distance = float(params.get("reconnection_distance_px_per_gap", 0.5))
    maximum_distance = float(params.get("maximum_reconnection_distance_px", 12.0))
    maximum_cost = float(params.get("maximum_reconnection_cost", 1.15))
    minimum_margin = float(params.get("minimum_reconnection_margin", 0.15))
    candidates: list[dict] = []
    track_ids = sorted(grouped)
    for before_id in track_ids:
        before = grouped[before_id]
        endpoint = before.iloc[-1]
        end_frame = int(endpoint.frame)
        for after_id in track_ids:
            if before_id == after_id:
                continue
            after = grouped[after_id]
            startpoint = after.iloc[0]
            start_frame = int(startpoint.frame)
            gap = start_frame - end_frame - 1
            if gap < 0 or gap > maximum_gap:
                continue
            allowed = min(maximum_distance,
                          base_distance + gap_distance * (gap + 1))
            distance = float(np.hypot(
                float(endpoint.x) - float(startpoint.x),
                float(endpoint.y) - float(startpoint.y)))
            if distance > allowed:
                continue
            patch_before = _patch(raw[end_frame], endpoint.x, endpoint.y)
            patch_after = _patch(raw[start_frame], startpoint.x, startpoint.y)
            correlation = float(np.clip(
                np.dot(patch_before, patch_after), -1.0, 1.0))
            cost = distance / max(allowed, 1e-6) + 0.5 * (1.0 - correlation)
            candidates.append({
                "track_before": before_id, "track_after": after_id,
                "end_frame": end_frame, "start_frame": start_frame,
                "gap_frames": gap, "distance_px": distance,
                "patch_correlation": correlation, "link_cost": cost,
            })
    if not candidates:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(candidates)
    best_after = table.loc[table.groupby("track_after")["link_cost"].idxmin()]
    rows: list[dict] = []
    for before_id, group in table.groupby("track_before"):
        ordered = group.sort_values("link_cost")
        best = ordered.iloc[0]
        second = float(ordered.iloc[1].link_cost) if len(ordered) > 1 else np.inf
        margin = second - float(best.link_cost)
        mutual = bool(((best_after["track_before"] == before_id) &
                       (best_after["track_after"] == int(best.track_after))).any())
        if (not mutual or float(best.link_cost) > maximum_cost
                or margin < minimum_margin):
            continue
        rows.append({**best.to_dict(), "uniqueness_margin": margin})
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values(["end_frame", "track_before"]).reset_index(drop=True)
    result.insert(0, "reconnection_id", np.arange(1, len(result) + 1))
    return result[columns]


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("m1 raw hypotheses are required")
    if params.get("targeting_mode") != "field_wide_discovery":
        raise ValueError("latent tracking must declare field_wide_discovery")
    forbidden = ("labels_path", "review_cases_path", "identity_ids",
                 "frame_ids", "coordinates")
    supplied = [name for name in forbidden if params.get(name)]
    if supplied:
        raise ValueError("latent tracking received forbidden targets: "
                         + ", ".join(supplied))
    hypotheses = pd.read_csv(upstream_dir / "raw_physical_hypotheses.csv")
    thresholds = pd.read_csv(upstream_dir / "frame_evidence_thresholds.csv")
    raw = raw_physical_hypotheses.aligned_raw(params)
    motion = None
    if params.get("motion_path"):
        full_motion = tifffile.imread(params["motion_path"])
        offset = int(params.get("motion_frame_offset", 2))
        motion = full_motion[offset:offset + len(raw)]
    tracks = link_observations(hypotheses, raw, motion, params)
    points, summaries = expand_track_points(tracks, raw, thresholds, params)
    encounters, encounter_frames = encounter_analysis(
        points, raw, float(params.get("maximum_separable_valley_ratio", 0.65)))
    reconnections = reconnection_hypotheses(points, raw, params)
    points_path = out.out / "latent_track_points.csv"
    tracks_path = out.out / "physical_tracks.csv"
    encounters_path = out.out / "encounter_episodes.csv"
    encounter_frames_path = out.out / "encounter_frames.csv"
    reconnections_path = out.out / "reconnection_hypotheses.csv"
    points.to_csv(points_path, index=False)
    summaries.to_csv(tracks_path, index=False)
    encounters.to_csv(encounters_path, index=False)
    encounter_frames.to_csv(encounter_frames_path, index=False)
    reconnections.to_csv(reconnections_path, index=False)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "candidate_labels_received": False,
        "identity_targets": 0, "frame_targets": 0, "coordinate_targets": 0,
        "physical_tracks": int(len(summaries)),
        "track_points": int(len(points)),
        "observed_points": int((points["state"] == "observed").sum()),
        "latent_visible_points": int((points["state"] == "latent_visible").sum()),
        "dormant_points": int((points["state"] == "dormant").sum()),
        "encounter_episodes": int(len(encounters)),
        "reconnection_hypotheses": int(len(reconnections)),
        "parameters": {key: params[key] for key in sorted(params)
                       if key not in {"raw_path", "motion_path"}},
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "track_points": points_path, "tracks": tracks_path,
        "encounters": encounters_path,
        "encounter_frames": encounter_frames_path,
        "reconnections": reconnections_path,
        "metrics": metrics_path,
    }, "summary": summary}
