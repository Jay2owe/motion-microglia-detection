"""Production detector for candidate-independent physical-cell cores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


def aligned_raw(params: dict) -> np.ndarray:
    raw = tifffile.imread(params["raw_path"])
    offset = int(params.get("source_frame_offset", 2))
    frames = int(params.get("frames", len(raw) - offset))
    result = raw[offset:offset + frames]
    if len(result) != frames:
        raise ValueError("registered raw stack is shorter than requested alignment")
    return result


def evidence_image(frame: np.ndarray, params: dict) -> np.ndarray:
    image = frame.astype(np.float32, copy=False)
    core_sigma = float(params.get("core_sigma_px", 1.2))
    background_sigma = float(params.get("background_sigma_px", 8.0))
    return (ndi.gaussian_filter(image, core_sigma)
            - ndi.gaussian_filter(image, background_sigma))


def _greedy_separated_peaks(
        evidence: np.ndarray, candidates: np.ndarray,
        minimum_distance: float) -> list[tuple[int, int]]:
    ordered = sorted(
        ((float(evidence[y, x]), int(y), int(x)) for y, x in candidates),
        reverse=True)
    accepted: list[tuple[int, int]] = []
    distance2 = float(minimum_distance) ** 2
    for _, y, x in ordered:
        if all((y - ay) ** 2 + (x - ax) ** 2 >= distance2
               for ay, ax in accepted):
            accepted.append((y, x))
    return accepted


def detect_frame(
        frame: np.ndarray, frame_index: int, params: dict,
        ) -> tuple[list[dict], dict]:
    evidence = evidence_image(frame, params)
    positive = evidence[evidence > 0]
    if not positive.size:
        return [], {
            "frame": frame_index, "weak_threshold": 0.0,
            "strong_threshold": 0.0, "positive_evidence_pixels": 0,
        }
    weak_q = float(params.get("weak_peak_quantile", 0.50))
    strong_q = float(params.get("strong_peak_quantile", 0.70))
    if not 0 < weak_q < strong_q < 1:
        raise ValueError("peak quantiles must satisfy 0 < weak < strong < 1")
    weak = float(np.quantile(positive, weak_q))
    strong = float(np.quantile(positive, strong_q))
    minimum_distance = float(params.get("minimum_peak_distance_px", 5.0))
    maximum = ndi.maximum_filter(
        evidence, size=2 * int(np.ceil(minimum_distance)) + 1,
        mode="nearest")
    candidates = np.argwhere(
        (evidence == maximum) & (evidence >= weak) & (frame > 0))
    peaks = _greedy_separated_peaks(evidence, candidates, minimum_distance)
    half_window = int(params.get("core_window_radius_px", 7))
    rows: list[dict] = []
    span = max(strong - weak, np.finfo(np.float32).eps)
    for hypothesis_id, (y, x) in enumerate(peaks, start=1):
        y0, y1 = max(0, y - half_window), min(frame.shape[0], y + half_window + 1)
        x0, x1 = max(0, x - half_window), min(frame.shape[1], x + half_window + 1)
        local_evidence = evidence[y0:y1, x0:x1]
        local_raw = frame[y0:y1, x0:x1]
        peak = float(evidence[y, x])
        core_level = max(0.35 * peak, 0.5 * weak)
        core = (local_evidence >= core_level) & (local_raw > 0)
        labelled, _ = ndi.label(core)
        local_y, local_x = y - y0, x - x0
        component_id = int(labelled[local_y, local_x])
        area = int(np.count_nonzero(labelled == component_id)) \
            if component_id else 1
        radius = float(np.clip(np.sqrt(max(area, 1) / np.pi), 2.0, 7.0))
        visibility = float(np.clip((peak - weak) / span, 0.0, 1.0))
        rows.append({
            "frame": int(frame_index),
            "hypothesis_id": int(hypothesis_id),
            "x": float(x), "y": float(y),
            "evidence": peak,
            "visibility": visibility,
            "strong": bool(peak >= strong),
            "core_area_px": area,
            "radius_px": radius,
            "raw_peak": int(frame[y, x]),
            "local_positive_fraction": float(np.mean(local_raw > 0)),
        })
    threshold = {
        "frame": int(frame_index),
        "weak_threshold": weak,
        "strong_threshold": strong,
        "positive_evidence_pixels": int(positive.size),
    }
    return rows, threshold


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    if params.get("targeting_mode") != "field_wide_discovery":
        raise ValueError("raw hypothesis stage must declare field_wide_discovery")
    forbidden = ("labels_path", "review_cases_path", "identity_ids",
                 "frame_ids", "coordinates")
    supplied = [name for name in forbidden if params.get(name)]
    if supplied:
        raise ValueError("raw hypothesis stage received forbidden targets: "
                         + ", ".join(supplied))
    raw = aligned_raw(params)
    rows: list[dict] = []
    thresholds: list[dict] = []
    for frame_index, frame in enumerate(raw):
        detected, threshold = detect_frame(frame, frame_index, params)
        rows.extend(detected)
        thresholds.append(threshold)
    hypotheses = pd.DataFrame(rows)
    threshold_table = pd.DataFrame(thresholds)
    hypotheses_path = out.out / "raw_physical_hypotheses.csv"
    thresholds_path = out.out / "frame_evidence_thresholds.csv"
    hypotheses.to_csv(hypotheses_path, index=False)
    threshold_table.to_csv(thresholds_path, index=False)
    per_frame = hypotheses.groupby("frame").size().reindex(
        range(len(raw)), fill_value=0)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "candidate_labels_received": False,
        "identity_targets": 0, "frame_targets": 0, "coordinate_targets": 0,
        "frames": int(len(raw)),
        "hypotheses": int(len(hypotheses)),
        "strong_hypotheses": int(hypotheses["strong"].sum()),
        "hypotheses_per_frame_min": int(per_frame.min()),
        "hypotheses_per_frame_median": float(per_frame.median()),
        "hypotheses_per_frame_max": int(per_frame.max()),
        "parameters": {key: params[key] for key in sorted(params)
                       if key not in {"raw_path"}},
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "hypotheses": hypotheses_path,
        "thresholds": thresholds_path,
        "metrics": metrics_path,
    }, "summary": summary}
