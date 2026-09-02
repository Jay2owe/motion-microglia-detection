from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi


ASSISTED_NEW_CELL_PROVENANCE = 2
MANUAL_NEW_CELL_PROVENANCE = 1
NEW_CELL_METHODS = (
    "soma_seed_expansion",
    "fixed_local_threshold",
    "adaptive_local_threshold",
    "manual_outline",
)
START_REASONS = ("movie_start", "birth", "border_entry")
END_REASONS = ("movie_end", "border_exit")
STRUCTURE = np.ones((3, 3), bool)


@dataclass(frozen=True)
class NewCellRequest:
    identity: int
    start_imagej_frame: int
    end_imagej_frame: int
    anchor_points: dict[int, tuple[int, int]]
    method: str = "soma_seed_expansion"
    crop_side_px: int = 96
    threshold_z: float = 2.0
    threshold_smoothing: float = 0.55
    threshold_keyframes: dict[int, float] = field(default_factory=dict)
    minimum_area_px: int = 4
    maximum_area_fraction: float = 0.75
    start_reason: str = "movie_start"
    end_reason: str = "movie_end"
    manual_masks: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class NewCellProposal:
    request: NewCellRequest
    masks: np.ndarray
    provenance: np.ndarray
    crop_centres: list[dict[str, Any]]
    per_frame: list[dict[str, Any]]
    warnings: list[str]
    lifetime_errors: list[str]
    parent_state_sha256: str

    @property
    def valid(self) -> bool:
        return (not self.lifetime_errors
                and bool(self.per_frame)
                and all(bool(row.get("valid")) for row in self.per_frame))


def state_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def allocate_new_identity(labels: np.ndarray,
                          reserved: Iterable[int] = ()) -> int:
    """Return a deterministic unused positive value representable by labels."""
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("new identities require an integer label array")
    maximum = int(np.iinfo(labels.dtype).max)
    used = set(map(int, np.unique(labels))) | {
        int(value) for value in reserved}
    used.discard(0)
    candidate = max(used, default=0) + 1
    if candidate <= maximum:
        return candidate
    for candidate in range(1, maximum + 1):
        if candidate not in used:
            return candidate
    if maximum < int(np.iinfo(np.uint64).max):
        return maximum + 1
    raise ValueError("no unused identity can be represented by uint64 labels")


def promote_labels_for_identity(labels: np.ndarray, identity: int) -> np.ndarray:
    """Return labels with the smallest safe unsigned type for an identity."""
    labels = np.asarray(labels)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("new identities require an integer TYX label array")
    identity = int(identity)
    if identity <= 0:
        raise ValueError("new identity must be positive")
    if identity <= int(np.iinfo(labels.dtype).max):
        return labels
    if np.any(labels < 0):
        raise ValueError("cannot promote a label array containing negative values")
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        candidate = np.dtype(dtype)
        if identity <= int(np.iinfo(candidate).max) \
                and candidate.itemsize > labels.dtype.itemsize:
            return labels.astype(candidate)
    raise ValueError(
        f"identity {identity} exceeds the supported uint64 label capacity")


def default_crop_side_px(labels: np.ndarray) -> int:
    """Choose a data-relative square side from observed cell areas."""
    areas: list[int] = []
    for frame in labels:
        values, counts = np.unique(frame[frame > 0], return_counts=True)
        if len(values):
            areas.extend(int(value) for value in counts)
    if not areas:
        return min(96, max(labels.shape[1:]))
    equivalent_diameter = 2.0 * math.sqrt(float(np.median(areas)) / math.pi)
    side = max(64, int(math.ceil(2.5 * equivalent_diameter)))
    return min(side, max(labels.shape[1:]))


def new_identity_scope_state_sha256(
        labels: np.ndarray, identity: int, start_imagej_frame: int,
        end_imagej_frame: int, mask: np.ndarray) -> str:
    """Fingerprint only the pixels a proposal will claim plus identity absence."""
    start, end = int(start_imagej_frame), int(end_imagej_frame)
    if mask.shape != (end - start + 1, *labels.shape[1:]):
        raise ValueError("new-cell scope mask has an incompatible shape")
    digest = hashlib.sha256()
    digest.update(labels.dtype.str.encode("ascii"))
    digest.update(np.asarray(labels.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray([identity, start, end], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    scoped = labels[start - 1:end][mask]
    digest.update(np.ascontiguousarray(scoped).tobytes())
    digest.update(np.asarray(
        [int(np.count_nonzero(labels == int(identity)))], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _validate_request(labels: np.ndarray, raw: np.ndarray,
                      request: NewCellRequest) -> None:
    if labels.ndim != 3 or raw.shape != labels.shape:
        raise ValueError("new-cell labels and registered raw must be same-shaped TYX")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("new-cell labels must use an integer data type")
    identity = int(request.identity)
    if identity <= 0 or identity > int(np.iinfo(labels.dtype).max):
        raise ValueError(
            f"new identity must fit {labels.dtype.name} and be positive")
    if np.any(labels == identity):
        raise ValueError(f"new identity {identity} already exists")
    start, end = int(request.start_imagej_frame), int(request.end_imagej_frame)
    if start < 1 or end < start or end > len(labels):
        raise ValueError(
            f"new-cell frame range must be within 1-{len(labels)}")
    if request.start_reason not in START_REASONS:
        raise ValueError(f"unsupported new-cell start reason: {request.start_reason!r}")
    if request.end_reason not in END_REASONS:
        raise ValueError(f"unsupported new-cell end reason: {request.end_reason!r}")
    if request.start_reason == "movie_start" and start != 1:
        raise ValueError("movie_start requires ImageJ frame 1")
    if request.end_reason == "movie_end" and end != len(labels):
        raise ValueError(
            f"movie_end requires ImageJ frame {len(labels)}")
    if request.method not in NEW_CELL_METHODS:
        raise ValueError(f"unsupported new-cell method: {request.method!r}")
    if isinstance(request.crop_side_px, bool) or request.crop_side_px < 9:
        raise ValueError("new-cell crop side must be an integer of at least 9 pixels")
    if request.crop_side_px > max(labels.shape[1:]):
        raise ValueError(
            "new-cell crop side cannot exceed the larger image dimension")
    if not np.isfinite(request.threshold_z):
        raise ValueError("new-cell threshold must be finite")
    if not np.isfinite(request.threshold_smoothing) \
            or not 0 <= request.threshold_smoothing <= 1:
        raise ValueError("threshold smoothing must be between 0 and 1")
    for frame, threshold in request.threshold_keyframes.items():
        if frame < start or frame > end or not np.isfinite(threshold):
            raise ValueError("new-cell threshold keyframe is invalid")
    if request.minimum_area_px < 1:
        raise ValueError("minimum new-cell area must be positive")
    if not 0 < request.maximum_area_fraction <= 1:
        raise ValueError("maximum crop-area fraction must be within (0, 1]")
    if not request.anchor_points:
        raise ValueError("new-cell tracking requires at least one anchor click")
    height, width = labels.shape[1:]
    for frame, point in request.anchor_points.items():
        if frame < start or frame > end:
            raise ValueError("new-cell anchor lies outside the declared lifetime")
        if len(point) != 2:
            raise ValueError("new-cell anchor must contain y and x")
        y, x = map(int, point)
        if y < 0 or x < 0 or y >= height or x >= width:
            raise ValueError("new-cell anchor lies outside the image")
        if labels[frame - 1, y, x] > 0:
            raise ValueError(
                f"new-cell anchor on ImageJ frame {frame} touches existing "
                f"identity {int(labels[frame - 1, y, x])}; use forced splitting")
    for frame, mask in request.manual_masks.items():
        if frame < start or frame > end:
            raise ValueError("manual new-cell mask lies outside the lifetime")
        if np.asarray(mask).shape != labels.shape[1:]:
            raise ValueError("manual new-cell mask and label frame differ in shape")


def _crop_bounds(center: tuple[float, float], side: int,
                 shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    half = side / 2.0
    y0 = max(0, int(math.floor(center[0] - half)))
    y1 = min(height, int(math.ceil(center[0] + half)))
    x0 = max(0, int(math.floor(center[1] - half)))
    x1 = min(width, int(math.ceil(center[1] + half)))
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    return y0, y1, x0, x1


def _robust_local_z(frame: np.ndarray, center: tuple[float, float], side: int,
                    protected: np.ndarray | None = None
                    ) -> tuple[np.ndarray, tuple[int, int, int, int], float, float]:
    bounds = _crop_bounds(center, side, frame.shape)
    y0, y1, x0, x1 = bounds
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.indices(crop.shape)
    local_y, local_x = center[0] - y0, center[1] - x0
    seed_radius = max(2.0, side * 0.10)
    outside_seed = (yy - local_y) ** 2 + (xx - local_x) ** 2 > seed_radius ** 2
    eligible = outside_seed.copy()
    if protected is not None:
        eligible &= ~protected[y0:y1, x0:x1]
    values = crop[eligible]
    if values.size:
        upper = float(np.percentile(values, 70))
        values = values[values <= upper]
    if values.size:
        background = float(np.median(values))
        mad = 1.4826 * float(np.median(np.abs(values - background)))
    else:
        background, mad = 0.0, 0.0
    noise = max(mad, math.sqrt(max(abs(background), 1.0)), 1.0)
    return (crop - background) / noise, bounds, background, noise


def _locate_cell(frame: np.ndarray, protected: np.ndarray,
                 predicted: tuple[float, float], side: int,
                 unclaimed: np.ndarray | None = None,
                 motion: np.ndarray | None = None
                 ) -> tuple[tuple[float, float] | None, float, float, float]:
    z, (y0, y1, x0, x1), _background, _noise = _robust_local_z(
        frame, predicted, side, protected)
    allowed = ~protected[y0:y1, x0:x1]
    score = ndi.gaussian_filter(z, 1.2)
    if unclaimed is not None:
        score = score + 0.45 * (unclaimed[y0:y1, x0:x1] > 0)
    motion_score = None
    if motion is not None:
        motion_crop = np.abs(np.asarray(motion[y0:y1, x0:x1], np.float32))
        finite = motion_crop[np.isfinite(motion_crop)]
        if finite.size:
            scale = max(float(np.percentile(finite, 90)), 1e-6)
            motion_score = np.clip(motion_crop / scale, 0.0, 1.0)
            score = score + 0.35 * ndi.gaussian_filter(motion_score, 1.0)
    yy, xx = np.indices(score.shape)
    py, px = predicted[0] - y0, predicted[1] - x0
    search_radius = max(4.0, side * 0.28)
    distance = np.hypot(yy - py, xx - px)
    allowed &= distance <= search_radius
    merit = score - 0.40 * distance / search_radius
    merit[~allowed] = -np.inf
    if not np.any(np.isfinite(merit)):
        return None, -np.inf, 0.0, 0.0
    maximum = float(np.max(merit))
    if not np.isfinite(maximum) or maximum < 0.75:
        return None, maximum, 0.0, 0.0
    peak_y, peak_x = np.unravel_index(int(np.argmax(merit)), merit.shape)
    # Use a compact high-signal component around the maximum to avoid locking to
    # one noisy pixel on a broad soma plateau.
    support = allowed & (score >= max(0.5, float(score[peak_y, peak_x]) - 0.8))
    components, _count = ndi.label(support, structure=STRUCTURE)
    component = components == int(components[peak_y, peak_x])
    if np.any(component):
        weights = np.maximum(score[component], 0.05)
        cy = float(np.average(yy[component], weights=weights))
        cx = float(np.average(xx[component], weights=weights))
    else:
        cy, cx = float(peak_y), float(peak_x)
    suppressed = merit.copy()
    suppress = (yy - cy) ** 2 + (xx - cx) ** 2 <= max(3.0, side * 0.08) ** 2
    suppressed[suppress] = -np.inf
    runner = float(np.max(suppressed)) if np.any(np.isfinite(suppressed)) else -np.inf
    margin = maximum - runner if np.isfinite(runner) else maximum
    confidence = float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, maximum)))))
    motion_support = (0.0 if motion_score is None else float(
        motion_score[min(motion_score.shape[0] - 1, max(0, int(round(cy)))),
                     min(motion_score.shape[1] - 1, max(0, int(round(cx))))]))
    return (cy + y0, cx + x0), confidence, float(margin), motion_support


def track_moving_crop(labels: np.ndarray, raw: np.ndarray,
                      request: NewCellRequest,
                      unclaimed: np.ndarray | None = None,
                      lag: np.ndarray | None = None
                      ) -> list[dict[str, Any]]:
    """Propagate crop centres from user anchors in both temporal directions."""
    _validate_request(labels, raw, request)
    if lag is not None and np.asarray(lag).shape != labels.shape:
        raise ValueError("lag evidence and new-cell labels differ in shape")
    start, end = request.start_imagej_frame, request.end_imagej_frame
    anchors = {int(frame): tuple(map(int, point))
               for frame, point in request.anchor_points.items()}
    centres: dict[int, tuple[float, float] | None] = {
        frame: (float(point[0]), float(point[1]))
        for frame, point in anchors.items()}
    metadata: dict[int, dict[str, Any]] = {}
    for frame, point in centres.items():
        metadata[frame] = {
            "source": "anchor", "confidence": 1.0,
            "runner_up_margin": None, "motion_support": None,
            "needs_recenter": False}

    anchor_frames = sorted(anchors)
    # Between two user anchors, the linear keyframe path is a safe fallback if local
    # appearance becomes dim. Local evidence refines it but never creates a hidden gap.
    for left, right in zip(anchor_frames, anchor_frames[1:]):
        ly, lx = centres[left]
        ry, rx = centres[right]
        for frame in range(left + 1, right):
            fraction = (frame - left) / (right - left)
            predicted = (ly + fraction * (ry - ly), lx + fraction * (rx - lx))
            located, confidence, margin, motion_support = _locate_cell(
                raw[frame - 1], labels[frame - 1] > 0, predicted,
                request.crop_side_px,
                None if unclaimed is None else unclaimed[frame - 1],
                None if lag is None else lag[frame - 1])
            centres[frame] = predicted if located is None else located
            metadata[frame] = {
                "source": "interpolated" if located is None else "propagated",
                "confidence": 0.0 if located is None else confidence,
                "runner_up_margin": margin,
                "motion_support": motion_support,
                "needs_recenter": False,
                "warning": "weak local evidence between anchor keyframes"
                if located is None else None,
            }

    def propagate(origin: int, stop: int, step: int) -> None:
        previous = centres[origin]
        velocity = np.zeros(2, float)
        frame = origin + step
        while (frame >= stop if step < 0 else frame <= stop):
            if frame in centres:
                new = centres[frame]
                if previous is not None and new is not None:
                    velocity = np.asarray(new) - np.asarray(previous)
                previous = new
                frame += step
                continue
            if previous is None:
                centres[frame] = None
                metadata[frame] = {
                    "source": "unresolved", "confidence": 0.0,
                    "runner_up_margin": None, "motion_support": 0.0,
                    "needs_recenter": True}
                frame += step
                continue
            predicted_array = np.asarray(previous) + velocity
            predicted = (float(predicted_array[0]), float(predicted_array[1]))
            located, confidence, margin, motion_support = _locate_cell(
                raw[frame - 1], labels[frame - 1] > 0, predicted,
                request.crop_side_px,
                None if unclaimed is None else unclaimed[frame - 1],
                None if lag is None else lag[frame - 1])
            if located is None:
                centres[frame] = None
                metadata[frame] = {
                    "source": "unresolved", "confidence": 0.0,
                    "runner_up_margin": margin,
                    "motion_support": motion_support,
                    "needs_recenter": True}
                previous = None
            else:
                velocity = np.asarray(located) - np.asarray(previous)
                maximum_step = max(2.0, request.crop_side_px * 0.20)
                speed = float(np.linalg.norm(velocity))
                if speed > maximum_step:
                    velocity *= maximum_step / speed
                centres[frame] = located
                metadata[frame] = {
                    "source": "propagated", "confidence": confidence,
                    "runner_up_margin": margin,
                    "motion_support": motion_support,
                    "needs_recenter": False}
                previous = located
            frame += step

    first, last = anchor_frames[0], anchor_frames[-1]
    propagate(first, start, -1)
    propagate(last, end, 1)

    records: list[dict[str, Any]] = []
    for frame in range(start, end + 1):
        centre = centres.get(frame)
        row = dict(metadata.get(frame, {
            "source": "unresolved", "confidence": 0.0,
            "runner_up_margin": None, "motion_support": 0.0,
            "needs_recenter": True}))
        row["imagej_frame"] = frame
        if centre is None:
            row.update({"y": None, "x": None, "bounds_yx": None,
                        "clipped": False})
        else:
            bounds = _crop_bounds(centre, request.crop_side_px, labels.shape[1:])
            row.update({
                "y": float(centre[0]), "x": float(centre[1]),
                "bounds_yx": list(map(int, bounds)),
                "clipped": bool(
                    bounds[0] == 0 or bounds[1] == labels.shape[1]
                    or bounds[2] == 0 or bounds[3] == labels.shape[2]),
            })
        if row.get("warning") is None:
            row.pop("warning", None)
        records.append(row)
    return records


def _seed_component(binary: np.ndarray, seed: tuple[float, float],
                    maximum_snap_px: float) -> np.ndarray:
    if not np.any(binary):
        return np.zeros(binary.shape, bool)
    y = int(round(seed[0])); x = int(round(seed[1]))
    y = min(binary.shape[0] - 1, max(0, y))
    x = min(binary.shape[1] - 1, max(0, x))
    components, count = ndi.label(binary, structure=STRUCTURE)
    selected = int(components[y, x])
    if selected <= 0:
        points = np.column_stack(np.nonzero(binary))
        distances = np.linalg.norm(points - np.array([y, x]), axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > float(maximum_snap_px):
            return np.zeros(binary.shape, bool)
        py, px = points[index]
        selected = int(components[py, px])
    if selected <= 0 or selected > count:
        return np.zeros(binary.shape, bool)
    return components == selected


def _fixed_threshold_mask(frame: np.ndarray, labels: np.ndarray,
                          centre: tuple[float, float], side: int,
                          threshold: float,
                          unclaimed: np.ndarray | None = None) -> np.ndarray:
    z, (y0, y1, x0, x1), _background, _noise = _robust_local_z(
        frame, centre, side, labels > 0)
    if unclaimed is not None:
        z = z + 0.35 * (unclaimed[y0:y1, x0:x1] > 0)
    allowed = labels[y0:y1, x0:x1] == 0
    binary = (z >= float(threshold)) & allowed
    local_seed = (centre[0] - y0, centre[1] - x0)
    component = _seed_component(binary, local_seed, max(3.0, side * 0.12))
    result = np.zeros(frame.shape, bool)
    result[y0:y1, x0:x1] = component
    return result


def _soma_expansion_mask(frame: np.ndarray, labels: np.ndarray,
                         centre: tuple[float, float], side: int,
                         threshold: float,
                         unclaimed: np.ndarray | None = None) -> np.ndarray:
    z, (y0, y1, x0, x1), _background, _noise = _robust_local_z(
        frame, centre, side, labels > 0)
    if unclaimed is not None:
        z = z + 0.35 * (unclaimed[y0:y1, x0:x1] > 0)
    allowed = labels[y0:y1, x0:x1] == 0
    positive_ridge = np.maximum(-ndi.gaussian_laplace(z, 1.1), 0.0)
    ridge_scale = float(np.percentile(positive_ridge, 98)) \
        if positive_ridge.size else 0.0
    ridge = (np.clip(positive_ridge / ridge_scale, 0, 1)
             if ridge_scale > 0 else np.zeros(z.shape, np.float32))
    intensity = 1.0 / (1.0 + np.exp(-np.clip((z - threshold) / 1.25, -30, 30)))
    score = np.maximum(intensity, 0.38 * intensity + 0.72 * ridge)
    seed_binary = (z >= max(float(threshold) + 0.75, 2.0)) & allowed
    local_seed = (centre[0] - y0, centre[1] - x0)
    core = _seed_component(seed_binary, local_seed, max(4.0, side * 0.15))
    if not np.any(core):
        core = _seed_component((z >= float(threshold)) & allowed, local_seed,
                               max(4.0, side * 0.15))
    traversable = ((z >= float(threshold) - 0.85) | (score >= 0.52)) & allowed
    grown = (ndi.binary_propagation(core, structure=STRUCTURE, mask=traversable)
             if np.any(core) else np.zeros(z.shape, bool))
    grown = ndi.binary_closing(grown, structure=STRUCTURE)
    grown &= allowed
    grown = _seed_component(grown, local_seed, max(4.0, side * 0.15))
    result = np.zeros(frame.shape, bool)
    result[y0:y1, x0:x1] = grown
    return result


def _mask_descriptor(mask: np.ndarray, frame: np.ndarray,
                     crop_bounds: list[int] | None) -> dict[str, Any]:
    area = int(np.count_nonzero(mask))
    if not area:
        return {"area_px": 0, "y": None, "x": None,
                "mean_intensity": None, "connected_components": 0,
                "perimeter_px": 0, "circularity": None,
                "crop_edge_contact": False,
                "crop_safety_margin_contact": False,
                "image_border_contact": False}
    yy, xx = np.nonzero(mask)
    _components, count = ndi.label(mask, structure=STRUCTURE)
    crop_edge = False
    crop_margin = False
    if crop_bounds is not None:
        y0, y1, x0, x1 = crop_bounds
        crop_edge = bool(
            np.any(mask[y0, x0:x1]) or np.any(mask[y1 - 1, x0:x1])
            or np.any(mask[y0:y1, x0]) or np.any(mask[y0:y1, x1 - 1]))
        margin_y = max(1, int(math.ceil((y1 - y0) * 0.15)))
        margin_x = max(1, int(math.ceil((x1 - x0) * 0.15)))
        safe = np.zeros(mask.shape, bool)
        if y1 - y0 > 2 * margin_y and x1 - x0 > 2 * margin_x:
            safe[y0 + margin_y:y1 - margin_y,
                 x0 + margin_x:x1 - margin_x] = True
            crop_margin = bool(np.any(mask & ~safe))
    perimeter = int(np.count_nonzero(
        mask & ~ndi.binary_erosion(mask, structure=STRUCTURE)))
    circularity = float(4.0 * math.pi * area / max(perimeter ** 2, 1))
    return {
        "area_px": area,
        "y": float(yy.mean()), "x": float(xx.mean()),
        "mean_intensity": float(frame[mask].mean()),
        "connected_components": int(count),
        "perimeter_px": perimeter,
        "circularity": circularity,
        "crop_edge_contact": crop_edge,
        "crop_safety_margin_contact": crop_margin,
        "image_border_contact": bool(
            yy.min() == 0 or xx.min() == 0
            or yy.max() == mask.shape[0] - 1
            or xx.max() == mask.shape[1] - 1),
    }


def _candidate_quality(mask: np.ndarray, frame: np.ndarray,
                       z_threshold: float, previous: np.ndarray | None,
                       crop_side: int) -> float:
    if not np.any(mask):
        return -1e6
    descriptor = _mask_descriptor(mask, frame, None)
    score = math.log1p(descriptor["area_px"])
    score += min(3.0, descriptor["mean_intensity"] / max(float(np.std(frame)), 1.0))
    if previous is not None and np.any(previous):
        union = int(np.count_nonzero(mask | previous))
        overlap = int(np.count_nonzero(mask & previous)) / max(union, 1)
        area_change = abs(math.log2(
            (descriptor["area_px"] + 1) / (int(previous.sum()) + 1)))
        py, px = ndi.center_of_mass(previous)
        displacement = math.hypot(descriptor["y"] - py, descriptor["x"] - px)
        score += 3.0 * overlap - area_change - displacement / max(crop_side, 1)
    score -= 0.12 * abs(float(z_threshold))
    return float(score)


def threshold_for_frame(request: NewCellRequest, imagej_frame: int) -> float:
    """Interpolate explicit threshold keyframes around the shared baseline."""
    points = {int(frame): float(value)
              for frame, value in request.threshold_keyframes.items()}
    if not points:
        return float(request.threshold_z)
    if imagej_frame in points:
        return points[imagej_frame]
    earlier = [frame for frame in points if frame < imagej_frame]
    later = [frame for frame in points if frame > imagej_frame]
    if not earlier:
        return points[min(later)]
    if not later:
        return points[max(earlier)]
    left, right = max(earlier), min(later)
    fraction = (imagej_frame - left) / (right - left)
    return points[left] + fraction * (points[right] - points[left])


def _adaptive_masks(labels: np.ndarray, raw: np.ndarray,
                    centres: list[dict[str, Any]], request: NewCellRequest,
                    unclaimed: np.ndarray | None
                    ) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
    offsets = np.linspace(-1.5, 1.5, 9)
    frames = list(range(request.start_imagej_frame, request.end_imagej_frame + 1))
    threshold_grids = [[
        float(threshold_for_frame(request, frame) + value)
        for value in offsets] for frame in frames]
    candidates: list[list[np.ndarray]] = []
    unary: list[np.ndarray] = []
    for offset, frame in enumerate(frames):
        centre_row = centres[offset]
        if centre_row["y"] is None:
            candidates.append([
                np.zeros(labels.shape[1:], bool)
                for _ in threshold_grids[offset]])
            unary.append(np.full(len(threshold_grids[offset]), -1e6, float))
            continue
        centre = (float(centre_row["y"]), float(centre_row["x"]))
        frame_candidates = [
            _fixed_threshold_mask(
                raw[frame - 1], labels[frame - 1], centre,
                request.crop_side_px, threshold,
                None if unclaimed is None else unclaimed[frame - 1])
            for threshold in threshold_grids[offset]]
        candidates.append(frame_candidates)
        unary.append(np.asarray([
            _candidate_quality(mask, raw[frame - 1], threshold, None,
                               request.crop_side_px)
            - 8.0 * float(np.any(mask & (labels[frame - 1] > 0)))
            for mask, threshold in zip(
                frame_candidates, threshold_grids[offset])], float))

    count, width = len(frames), len(offsets)
    scores = np.full((count, width), -np.inf, float)
    back = np.zeros((count, width), np.intp)
    scores[0] = unary[0]
    smoothing = 0.25 + 2.0 * float(request.threshold_smoothing)
    for frame_index in range(1, count):
        for current in range(width):
            transition_scores = np.empty(width, float)
            current_mask = candidates[frame_index][current]
            for previous in range(width):
                previous_mask = candidates[frame_index - 1][previous]
                transition = _candidate_quality(
                    current_mask, raw[frames[frame_index] - 1],
                    threshold_grids[frame_index][current], previous_mask,
                    request.crop_side_px)
                transition -= smoothing * abs(
                    threshold_grids[frame_index][current]
                    - threshold_grids[frame_index - 1][previous])
                transition_scores[previous] = scores[frame_index - 1, previous] \
                    + transition
            best_previous = int(np.argmax(transition_scores))
            back[frame_index, current] = best_previous
            scores[frame_index, current] = (
                unary[frame_index][current] + transition_scores[best_previous])
    selected = [0] * count
    selected[-1] = int(np.argmax(scores[-1]))
    for frame_index in range(count - 1, 0, -1):
        selected[frame_index - 1] = int(back[frame_index, selected[frame_index]])

    # A reverse dynamic pass supplies an honest directional disagreement flag.
    reverse_selected = [0] * count
    reverse_score = unary[-1].copy()
    reverse_selected[-1] = int(np.argmax(reverse_score))
    for frame_index in range(count - 2, -1, -1):
        next_index = reverse_selected[frame_index + 1]
        values = []
        for current in range(width):
            transition = _candidate_quality(
                candidates[frame_index + 1][next_index],
                raw[frames[frame_index + 1] - 1],
                threshold_grids[frame_index + 1][next_index],
                candidates[frame_index][current], request.crop_side_px)
            values.append(unary[frame_index][current] + transition
                          - smoothing * abs(
                              threshold_grids[frame_index][current]
                              - threshold_grids[frame_index + 1][next_index]))
        reverse_selected[frame_index] = int(np.argmax(values))

    audit: list[dict[str, Any]] = []
    for index in range(count):
        ordered = np.sort(scores[index])
        margin = float(ordered[-1] - ordered[-2]) if width > 1 else None
        audit.append({
            "selected_threshold_z": threshold_grids[index][selected[index]],
            "runner_up_margin": margin,
            "forward_backward_disagreement": bool(
                selected[index] != reverse_selected[index]),
            "reverse_threshold_z": threshold_grids[index][reverse_selected[index]],
        })
    return ([candidates[index][selected[index]] for index in range(count)],
            [threshold_grids[index][value]
             for index, value in enumerate(selected)], audit)


def _validate_lifetime(request: NewCellRequest, masks: np.ndarray,
                       frame_rows: list[dict[str, Any]], total_frames: int
                       ) -> list[str]:
    errors: list[str] = []
    empty = [row["imagej_frame"] for row in frame_rows if row["area_px"] == 0]
    if empty:
        errors.append("new identity has empty lifetime frame(s): "
                      + ", ".join(map(str, empty)))
    if request.start_reason == "movie_start" and request.start_imagej_frame != 1:
        errors.append("movie_start requires the first movie frame")
    if request.end_reason == "movie_end" and request.end_imagej_frame != total_frames:
        errors.append("movie_end requires the final movie frame")
    if request.start_reason == "border_entry" and frame_rows \
            and not frame_rows[0]["image_border_contact"]:
        errors.append("border_entry requires the first accepted outline to touch "
                      "the full-image border")
    if request.end_reason == "border_exit" and frame_rows \
            and not frame_rows[-1]["image_border_contact"]:
        errors.append("border_exit requires the last accepted outline to touch "
                      "the full-image border")
    return errors


def _build_frame_rows(labels: np.ndarray, raw: np.ndarray,
                      request: NewCellRequest, masks: np.ndarray,
                      provenance: np.ndarray,
                      centres: list[dict[str, Any]],
                      thresholds: list[float | None],
                      adaptive_audit: list[dict[str, Any]] | None = None,
                      raw_frame_indices: Iterable[int] | None = None,
                      lag: np.ndarray | None = None
                      ) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    source_indices = (list(range(len(labels))) if raw_frame_indices is None
                      else [int(value) for value in raw_frame_indices])
    if len(source_indices) != len(labels):
        raise ValueError("raw-frame mapping and new-cell labels differ in length")
    for offset, imagej_frame in enumerate(range(
            request.start_imagej_frame, request.end_imagej_frame + 1)):
        mask = masks[offset]
        centre = centres[offset]
        descriptor = _mask_descriptor(
            mask, raw[imagej_frame - 1], centre.get("bounds_yx"))
        collision_ids = sorted(set(map(int, np.unique(
            labels[imagej_frame - 1][mask]))) - {0})
        manual_pixels = int(np.count_nonzero(
            provenance[offset] == MANUAL_NEW_CELL_PROVENANCE))
        errors: list[str] = []
        if centre.get("needs_recenter"):
            errors.append("crop tracking needs a new centroid anchor")
        if descriptor["area_px"] < request.minimum_area_px:
            errors.append(
                f"area {descriptor['area_px']} px is below minimum "
                f"{request.minimum_area_px} px")
        if descriptor["connected_components"] > 1:
            errors.append("outline is not eight-connected")
        if request.method != "manual_outline" and descriptor["area_px"] \
                and centre.get("y") is not None and centre.get("x") is not None:
            cy, cx = float(centre["y"]), float(centre["x"])
            yy, xx = np.nonzero(mask)
            if float(np.min(np.hypot(yy - cy, xx - cx))) \
                    > max(3.0, request.crop_side_px * 0.15):
                errors.append("outline is not connected to the propagated seed")
        if collision_ids:
            errors.append("outline overlaps existing identit"
                          + ("y " if len(collision_ids) == 1 else "ies ")
                          + ", ".join(map(str, collision_ids)))
        crop_area = request.crop_side_px ** 2
        if descriptor["area_px"] > request.maximum_area_fraction * crop_area:
            warnings.append(
                f"ImageJ frame {imagej_frame}: outline occupies more than "
                f"{100 * request.maximum_area_fraction:g}% of the crop")
        if descriptor["crop_edge_contact"]:
            warnings.append(
                f"ImageJ frame {imagej_frame}: outline touches the moving crop edge")
        elif descriptor["crop_safety_margin_contact"]:
            warnings.append(
                f"ImageJ frame {imagej_frame}: outline enters the outer 15% "
                "crop safety margin")
        if centre.get("runner_up_margin") is not None \
                and float(centre["runner_up_margin"]) < 0.08:
            warnings.append(
                f"ImageJ frame {imagej_frame}: crop location has a weak "
                "best-versus-runner-up margin")
        audit = {} if adaptive_audit is None else adaptive_audit[offset]
        if audit.get("forward_backward_disagreement"):
            warnings.append(
                f"ImageJ frame {imagej_frame}: forward and backward adaptive "
                "threshold choices disagree")
        motion_support = centre.get("motion_support")
        if motion_support is None and lag is not None and np.any(mask):
            motion_values = np.abs(np.asarray(
                lag[imagej_frame - 1][mask], np.float32))
            frame_values = np.abs(np.asarray(lag[imagej_frame - 1], np.float32))
            finite = frame_values[np.isfinite(frame_values)]
            scale = max(float(np.percentile(finite, 90)), 1e-6) \
                if finite.size else 1.0
            motion_support = float(np.clip(
                np.mean(motion_values) / scale, 0.0, 1.0))
        if lag is not None and motion_support is not None \
                and float(motion_support) < 0.05:
            warnings.append(
                f"ImageJ frame {imagej_frame}: weak local motion support")
        rows.append({
            "imagej_frame": imagej_frame,
            "source_raw_frame_index": source_indices[imagej_frame - 1],
            **descriptor,
            "threshold_z": thresholds[offset],
            "tracking_confidence": float(centre.get("confidence", 0.0)),
            "tracking_source": centre.get("source"),
            "tracking_runner_up_margin": centre.get("runner_up_margin"),
            "motion_support": motion_support,
            "adaptive_runner_up_margin": audit.get("runner_up_margin"),
            "forward_backward_disagreement": bool(
                audit.get("forward_backward_disagreement", False)),
            "collision_identities": collision_ids,
            "manual_pixels": manual_pixels,
            "assisted_pixels": int(descriptor["area_px"] - manual_pixels),
            "errors": errors,
            "valid": not errors,
        })
        if len(rows) > 1 and descriptor["area_px"] \
                and rows[-2]["area_px"]:
            previous = rows[-2]
            area_ratio = max(
                descriptor["area_px"] / previous["area_px"],
                previous["area_px"] / descriptor["area_px"])
            displacement = math.hypot(
                descriptor["y"] - previous["y"],
                descriptor["x"] - previous["x"])
            intersection = int(np.count_nonzero(mask & masks[offset - 1]))
            union = int(np.count_nonzero(mask | masks[offset - 1]))
            rows[-1]["temporal_iou"] = intersection / max(union, 1)
            rows[-1]["centroid_displacement_px"] = displacement
            rows[-1]["area_ratio_to_previous"] = area_ratio
            if area_ratio > 2.0:
                warnings.append(
                    f"ImageJ frame {imagej_frame}: outline area changes by more "
                    "than twofold")
            if displacement > request.crop_side_px * 0.20:
                warnings.append(
                    f"ImageJ frame {imagej_frame}: centroid movement exceeds 20% "
                    "of the crop side")
            previous_threshold = previous.get("threshold_z")
            if previous_threshold is not None and thresholds[offset] is not None \
                    and abs(float(thresholds[offset])
                            - float(previous_threshold)) > 0.75:
                warnings.append(
                    f"ImageJ frame {imagej_frame}: automatic threshold jumps by "
                    "more than 0.75 z")
        else:
            rows[-1]["temporal_iou"] = None
            rows[-1]["centroid_displacement_px"] = None
            rows[-1]["area_ratio_to_previous"] = None
    return rows, warnings


def propose_new_identity(labels: np.ndarray, raw: np.ndarray,
                         request: NewCellRequest,
                         unclaimed: np.ndarray | None = None,
                         lag: np.ndarray | None = None,
                         raw_frame_indices: Iterable[int] | None = None
                         ) -> NewCellProposal:
    """Generate a method-neutral, non-mutating new-identity proposal."""
    labels = promote_labels_for_identity(
        np.asarray(labels), int(request.identity))
    raw = np.asarray(raw)
    _validate_request(labels, raw, request)
    if unclaimed is not None and np.asarray(unclaimed).shape != labels.shape:
        raise ValueError("unclaimed labels and new-cell labels differ in shape")
    if lag is not None and np.asarray(lag).shape != labels.shape:
        raise ValueError("lag evidence and new-cell labels differ in shape")
    centres = track_moving_crop(labels, raw, request, unclaimed, lag)
    frame_count = request.end_imagej_frame - request.start_imagej_frame + 1
    masks = np.zeros((frame_count, *labels.shape[1:]), bool)
    provenance = np.zeros(masks.shape, np.uint8)
    thresholds: list[float | None] = [None] * frame_count
    adaptive_audit: list[dict[str, Any]] | None = None

    if request.method == "adaptive_local_threshold":
        generated, selected_thresholds, adaptive_audit = _adaptive_masks(
            labels, raw, centres, request, unclaimed)
        masks[:] = np.asarray(generated, bool)
        thresholds = [float(value) for value in selected_thresholds]
    else:
        for offset, imagej_frame in enumerate(range(
                request.start_imagej_frame, request.end_imagej_frame + 1)):
            centre_row = centres[offset]
            if request.method == "manual_outline":
                manual = request.manual_masks.get(imagej_frame)
                if manual is not None:
                    masks[offset] = np.asarray(manual, bool)
                continue
            if centre_row["y"] is None:
                continue
            centre = (float(centre_row["y"]), float(centre_row["x"]))
            unclaimed_frame = None if unclaimed is None else unclaimed[imagej_frame - 1]
            if request.method == "fixed_local_threshold":
                frame_threshold = threshold_for_frame(request, imagej_frame)
                masks[offset] = _fixed_threshold_mask(
                    raw[imagej_frame - 1], labels[imagej_frame - 1], centre,
                    request.crop_side_px, frame_threshold, unclaimed_frame)
                thresholds[offset] = float(frame_threshold)
            elif request.method == "soma_seed_expansion":
                frame_threshold = threshold_for_frame(request, imagej_frame)
                masks[offset] = _soma_expansion_mask(
                    raw[imagej_frame - 1], labels[imagej_frame - 1], centre,
                    request.crop_side_px, frame_threshold, unclaimed_frame)
                thresholds[offset] = float(frame_threshold)

    if request.method == "manual_outline":
        provenance[masks] = MANUAL_NEW_CELL_PROVENANCE
    else:
        provenance[masks] = ASSISTED_NEW_CELL_PROVENANCE
        # Manual corrections are exact per-frame overrides even when the job's
        # default generator remains automatic.
        for imagej_frame, manual in request.manual_masks.items():
            offset = imagej_frame - request.start_imagej_frame
            masks[offset] = np.asarray(manual, bool)
            provenance[offset] = np.where(
                masks[offset], MANUAL_NEW_CELL_PROVENANCE, 0).astype(np.uint8)
    rows, warnings = _build_frame_rows(
        labels, raw, request, masks, provenance, centres,
        thresholds, adaptive_audit, raw_frame_indices, lag)
    lifetime_errors = _validate_lifetime(request, masks, rows, len(labels))
    parent_hash = new_identity_scope_state_sha256(
        labels, request.identity, request.start_imagej_frame,
        request.end_imagej_frame, masks)
    return NewCellProposal(
        request=request, masks=masks, provenance=provenance,
        crop_centres=centres, per_frame=rows, warnings=warnings,
        lifetime_errors=lifetime_errors, parent_state_sha256=parent_hash)


def revalidate_new_identity_proposal(labels: np.ndarray, raw: np.ndarray,
                                     proposal: NewCellProposal,
                                     lag: np.ndarray | None = None
                                     ) -> NewCellProposal:
    labels = promote_labels_for_identity(labels, proposal.request.identity)
    thresholds = [row.get("threshold_z") for row in proposal.per_frame]
    raw_frame_indices = list(range(len(labels)))
    for row in proposal.per_frame:
        imagej_frame = int(row["imagej_frame"])
        raw_frame_indices[imagej_frame - 1] = int(
            row.get("source_raw_frame_index", imagej_frame - 1))
    rows, warnings = _build_frame_rows(
        labels, raw, proposal.request, proposal.masks, proposal.provenance,
        proposal.crop_centres, thresholds,
        raw_frame_indices=raw_frame_indices, lag=lag)
    lifetime_errors = _validate_lifetime(
        proposal.request, proposal.masks, rows, len(labels))
    proposal.per_frame = rows
    proposal.warnings = warnings
    proposal.lifetime_errors = lifetime_errors
    proposal.parent_state_sha256 = new_identity_scope_state_sha256(
        labels, proposal.request.identity,
        proposal.request.start_imagej_frame,
        proposal.request.end_imagej_frame, proposal.masks)
    return proposal


def paint_new_identity(proposal: NewCellProposal, labels: np.ndarray,
                       raw: np.ndarray, imagej_frame: int,
                       points: Iterable[tuple[int, int]], radius_px: int,
                       add: bool = True) -> NewCellProposal:
    """Paint exact new-cell foreground/background and invalidate frame review."""
    request = proposal.request
    if imagej_frame < request.start_imagej_frame \
            or imagej_frame > request.end_imagej_frame:
        raise ValueError("paint frame lies outside the new-cell lifetime")
    if radius_px < 0:
        raise ValueError("new-cell brush radius cannot be negative")
    offset = imagej_frame - request.start_imagej_frame
    height, width = proposal.masks.shape[1:]
    yy, xx = np.indices((height, width))
    changed = False
    for y, x in points:
        if y < 0 or x < 0 or y >= height or x >= width:
            continue
        brush = (yy - int(y)) ** 2 + (xx - int(x)) ** 2 <= radius_px ** 2
        if add:
            proposal.masks[offset][brush] = True
            proposal.provenance[offset][brush] = MANUAL_NEW_CELL_PROVENANCE
        else:
            proposal.masks[offset][brush] = False
            proposal.provenance[offset][brush] = 0
        changed = True
    if not changed:
        raise ValueError("new-cell brush did not touch the image")
    return revalidate_new_identity_proposal(labels, raw, proposal)


def interpolate_manual_outlines(
        first: np.ndarray, second: np.ndarray, intermediate_count: int
        ) -> list[np.ndarray]:
    """Interpolate binary outlines as signed-distance proposals."""
    first = np.asarray(first, bool)
    second = np.asarray(second, bool)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("manual outline keyframes must be same-shaped 2D masks")
    if intermediate_count < 0:
        raise ValueError("manual outline interpolation count cannot be negative")
    if not np.any(first) or not np.any(second):
        raise ValueError("manual outline keyframes cannot be empty")
    signed_first = ndi.distance_transform_edt(first) \
        - ndi.distance_transform_edt(~first)
    signed_second = ndi.distance_transform_edt(second) \
        - ndi.distance_transform_edt(~second)
    result = []
    for index in range(1, intermediate_count + 1):
        fraction = index / (intermediate_count + 1)
        result.append(((1.0 - fraction) * signed_first
                       + fraction * signed_second) >= 0)
    return result


def _positive_runs(array: np.ndarray) -> list[list[int]]:
    flat = np.ascontiguousarray(array).ravel()
    indices = np.flatnonzero(flat > 0)
    if not len(indices):
        return []
    runs: list[list[int]] = []
    start = int(indices[0]); previous = start; value = int(flat[start])
    for raw_index in indices[1:]:
        index = int(raw_index); current = int(flat[index])
        if index == previous + 1 and current == value:
            previous = index
            continue
        runs.append([start, previous - start + 1, value])
        start = previous = index; value = current
    runs.append([start, previous - start + 1, value])
    return runs


def _decode_positive_runs(shape: tuple[int, ...], runs: list,
                          dtype: np.dtype) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64))
    flat = np.zeros(size, dtype=dtype)
    previous_end = 0
    for row in runs:
        if not isinstance(row, list) or len(row) != 3 \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in row):
            raise ValueError("new-cell patch contains an invalid run")
        start, length, value = row
        end = start + length
        if start < previous_end or length < 1 or end > size or value <= 0:
            raise ValueError("new-cell patch run is out of bounds or overlapping")
        flat[start:end] = value
        previous_end = end
    return flat.reshape(shape)


def encode_new_identity_patch(proposal: NewCellProposal) -> dict[str, Any]:
    spatial = np.any(proposal.masks, axis=0)
    yy, xx = np.nonzero(spatial)
    if not len(yy):
        raise ValueError("cannot encode an empty new-cell proposal")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    mask = np.ascontiguousarray(proposal.masks[:, y0:y1, x0:x1], np.uint8)
    provenance = np.ascontiguousarray(
        proposal.provenance[:, y0:y1, x0:x1], np.uint8)
    return {
        "schema": "motion.add-identity-patch",
        "schema_version": 1,
        "origin_yx": [y0, x0],
        "shape": list(map(int, mask.shape)),
        "mask_runs": _positive_runs(mask),
        "provenance_runs": _positive_runs(provenance),
        "mask_sha256": state_sha256(mask.astype(bool)),
        "provenance_sha256": state_sha256(provenance),
    }


def _decode_new_identity_patch(patch: dict[str, Any]
                               ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not isinstance(patch, dict) \
            or patch.get("schema") != "motion.add-identity-patch" \
            or patch.get("schema_version") != 1:
        raise ValueError("new-cell operation has no supported accepted patch")
    shape_value = patch.get("shape")
    origin_value = patch.get("origin_yx")
    if not isinstance(shape_value, list) or len(shape_value) != 3 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in shape_value):
        raise ValueError("new-cell patch shape is invalid")
    if not isinstance(origin_value, list) or len(origin_value) != 2 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in origin_value):
        raise ValueError("new-cell patch origin is invalid")
    shape = tuple(map(int, shape_value))
    mask = _decode_positive_runs(shape, patch.get("mask_runs", []), np.uint8) > 0
    provenance = _decode_positive_runs(
        shape, patch.get("provenance_runs", []), np.uint8)
    if state_sha256(mask) != patch.get("mask_sha256"):
        raise ValueError("new-cell mask patch fingerprint differs")
    if state_sha256(provenance) != patch.get("provenance_sha256"):
        raise ValueError("new-cell provenance patch fingerprint differs")
    if np.any((provenance > 0) & ~np.isin(
            provenance, [MANUAL_NEW_CELL_PROVENANCE,
                         ASSISTED_NEW_CELL_PROVENANCE])):
        raise ValueError("new-cell provenance patch contains invalid values")
    if np.any((provenance > 0) != mask):
        raise ValueError("new-cell provenance support differs from its mask")
    return mask, provenance, (int(origin_value[0]), int(origin_value[1]))


def build_add_identity_operation(
        proposal: NewCellProposal,
        reviewed_imagej_frames: Iterable[int],
        linked_split_frames: Iterable[int] = (),
        parent_labels: np.ndarray | None = None) -> dict[str, Any]:
    """Freeze a fully reviewed proposal into a deterministic edit request."""
    request = proposal.request
    reviewed = sorted(set(map(int, reviewed_imagej_frames)))
    required = list(range(
        request.start_imagej_frame, request.end_imagej_frame + 1))
    if reviewed != required:
        raise ValueError("new-cell operation requires review of every lifetime frame")
    delegated = sorted(set(map(int, linked_split_frames)))
    if any(frame < request.start_imagej_frame
           or frame > request.end_imagej_frame for frame in delegated):
        raise ValueError("linked split frame lies outside the new-cell lifetime")
    blocking_rows = [
        row for row in proposal.per_frame
        if not row["valid"] and row["imagej_frame"] not in delegated]
    remaining_lifetime_errors = [
        error for error in proposal.lifetime_errors
        if not (delegated and error.startswith(
            "new identity has empty lifetime frame(s):"))]
    if blocking_rows or remaining_lifetime_errors:
        details = [*remaining_lifetime_errors]
        details.extend(
            f"ImageJ frame {row['imagej_frame']}: " + "; ".join(row["errors"])
            for row in blocking_rows)
        raise ValueError("cannot commit new identity; " + "; ".join(details))
    patch_proposal = proposal
    parent_hash = proposal.parent_state_sha256
    if delegated:
        if parent_labels is None:
            raise ValueError(
                "linked new-cell operation requires its parent label stack")
        parent_labels = promote_labels_for_identity(
            np.asarray(parent_labels), request.identity)
        masks = proposal.masks.copy()
        patch_provenance = proposal.provenance.copy()
        for imagej_frame in delegated:
            offset = imagej_frame - request.start_imagej_frame
            eligible = parent_labels[imagej_frame - 1] == 0
            masks[offset] &= eligible
            patch_provenance[offset][~masks[offset]] = 0
        patch_proposal = replace(
            proposal, masks=masks, provenance=patch_provenance)
        parent_hash = new_identity_scope_state_sha256(
            parent_labels, request.identity, request.start_imagej_frame,
            request.end_imagej_frame, masks)
    thresholds = [{
        "imagej_frame": int(row["imagej_frame"]),
        "threshold_z": row.get("threshold_z"),
    } for row in proposal.per_frame]
    return {
        "type": "add_identity_track",
        "identity": int(request.identity),
        "start_imagej_frame": int(request.start_imagej_frame),
        "end_imagej_frame": int(request.end_imagej_frame),
        "start_reason": request.start_reason,
        "end_reason": request.end_reason,
        "crop_side_px": int(request.crop_side_px),
        "anchor_clicks": [{
            "imagej_frame": int(frame), "y": int(point[0]), "x": int(point[1])}
            for frame, point in sorted(request.anchor_points.items())],
        "method": request.method,
        "method_version": 1,
        "parent_state_sha256": parent_hash,
        "accepted_patch": encode_new_identity_patch(patch_proposal),
        "proposal_warnings": list(proposal.warnings),
        "proposal_metrics": proposal.per_frame,
        "crop_centres": proposal.crop_centres,
        "thresholds": thresholds,
        "reviewed_imagej_frames": reviewed,
        "linked_split_frames": delegated,
    }


def normalize_add_identity_request(request: dict[str, Any],
                                   total_frames: int) -> dict[str, Any]:
    allowed = {
        "type", "user", "identity", "start_imagej_frame", "end_imagej_frame",
        "start_reason", "end_reason", "crop_side_px", "anchor_clicks", "method",
        "method_version", "parent_state_sha256", "accepted_patch",
        "proposal_warnings", "proposal_metrics", "crop_centres", "thresholds",
        "reviewed_imagej_frames", "linked_split_frames",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError("unsupported fields for add_identity_track: "
                         + ", ".join(unknown))

    def integer(key: str, minimum: int = 1) -> int:
        value = request.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"batch operation {key} must be an integer >= {minimum}")
        return int(value)

    identity = integer("identity")
    start = integer("start_imagej_frame")
    end = integer("end_imagej_frame")
    if end < start or end > total_frames:
        raise ValueError(f"new-cell frame range must be within 1-{total_frames}")
    start_reason = request.get("start_reason")
    end_reason = request.get("end_reason")
    if start_reason not in START_REASONS or end_reason not in END_REASONS:
        raise ValueError("new-cell operation has an invalid lifetime reason")
    if start_reason == "movie_start" and start != 1:
        raise ValueError("movie_start requires ImageJ frame 1")
    if end_reason == "movie_end" and end != total_frames:
        raise ValueError(f"movie_end requires ImageJ frame {total_frames}")
    crop_side = integer("crop_side_px", 9)
    method = request.get("method")
    if method not in NEW_CELL_METHODS:
        raise ValueError(f"unsupported new-cell method: {method!r}")
    if integer("method_version") != 1:
        raise ValueError("unsupported new-cell method version")
    parent_hash = request.get("parent_state_sha256")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise ValueError("new-cell parent-state fingerprint is invalid")
    anchors = request.get("anchor_clicks")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("new-cell operation has no anchor clicks")
    normalized_anchors = []
    seen_anchor_frames: set[int] = set()
    for row in anchors:
        if not isinstance(row, dict) or set(row) != {"imagej_frame", "y", "x"}:
            raise ValueError("new-cell anchor click is invalid")
        values = (row["imagej_frame"], row["y"], row["x"])
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values) or row["imagej_frame"] < start \
                or row["imagej_frame"] > end or row["y"] < 0 or row["x"] < 0:
            raise ValueError("new-cell anchor click is out of range")
        if row["imagej_frame"] in seen_anchor_frames:
            raise ValueError("new-cell operation has duplicate anchor frames")
        seen_anchor_frames.add(row["imagej_frame"])
        normalized_anchors.append(dict(row))
    normalized_anchors.sort(key=lambda row: row["imagej_frame"])
    warnings = request.get("proposal_warnings", [])
    metrics = request.get("proposal_metrics", [])
    centres = request.get("crop_centres", [])
    thresholds = request.get("thresholds", [])
    if not isinstance(warnings, list) or any(not isinstance(value, str)
                                             for value in warnings):
        raise ValueError("new-cell proposal warnings must be text")
    if any(not isinstance(value, list) for value in (metrics, centres, thresholds)) \
            or any(not isinstance(row, dict)
                   for rows in (metrics, centres, thresholds) for row in rows):
        raise ValueError("new-cell proposal audit fields are invalid")
    expected_frames = list(range(start, end + 1))
    reviewed = request.get("reviewed_imagej_frames")
    if reviewed != expected_frames:
        raise ValueError("new-cell operation must record review of every lifetime frame")
    delegated = request.get("linked_split_frames", [])
    if not isinstance(delegated, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            or value < start or value > end for value in delegated) \
            or delegated != sorted(set(delegated)):
        raise ValueError("new-cell linked split frames are invalid")
    patch = request.get("accepted_patch")
    mask, provenance, _origin = _decode_new_identity_patch(patch)
    if mask.shape[0] != end - start + 1:
        raise ValueError("new-cell patch frame count differs from its lifetime")
    empty_frames = [start + offset for offset in range(len(mask))
                    if not np.any(mask[offset])]
    if any(frame not in delegated for frame in empty_frames):
        raise ValueError(
            "new-cell empty patch frames must be delegated to forced splitting")
    normalized = {
        "type": "add_identity_track", "identity": identity,
        "start_imagej_frame": start, "end_imagej_frame": end,
        "start_reason": start_reason, "end_reason": end_reason,
        "crop_side_px": crop_side, "anchor_clicks": normalized_anchors,
        "method": method, "method_version": 1,
        "parent_state_sha256": parent_hash, "accepted_patch": patch,
        "proposal_warnings": warnings, "proposal_metrics": metrics,
        "crop_centres": centres, "thresholds": thresholds,
        "reviewed_imagej_frames": reviewed,
        "linked_split_frames": delegated,
    }
    if "user" in request:
        normalized["user"] = request["user"]
    return normalized


def apply_add_identity_operation(
        labels: np.ndarray, provenance: np.ndarray,
        operation: dict[str, Any], excluded_identities: set[int] | None = None
        ) -> dict[str, Any]:
    identity = int(operation["identity"])
    start = int(operation["start_imagej_frame"])
    end = int(operation["end_imagej_frame"])
    if int(operation["crop_side_px"]) > max(labels.shape[1:]):
        raise ValueError("new-cell crop side exceeds the image dimensions")
    if identity <= 0 or identity > int(np.iinfo(labels.dtype).max):
        raise ValueError(f"new identity must fit {labels.dtype.name}")
    if excluded_identities and identity in excluded_identities:
        raise ValueError(f"new identity {identity} is excluded")
    if np.any(labels == identity):
        raise ValueError(f"new identity {identity} already exists")
    patch_mask, patch_provenance, (y0, x0) = _decode_new_identity_patch(
        operation["accepted_patch"])
    if patch_mask.shape[0] != end - start + 1:
        raise ValueError("new-cell patch frame count differs from its lifetime")
    y1, x1 = y0 + patch_mask.shape[1], x0 + patch_mask.shape[2]
    if y1 > labels.shape[1] or x1 > labels.shape[2]:
        raise ValueError("new-cell patch lies outside the label field")
    full_mask = np.zeros((len(patch_mask), *labels.shape[1:]), bool)
    full_mask[:, y0:y1, x0:x1] = patch_mask
    if new_identity_scope_state_sha256(
            labels, identity, start, end, full_mask) \
            != operation["parent_state_sha256"]:
        raise ValueError(
            "new-cell proposal is stale relative to the replayed parent state")
    collisions = labels[start - 1:end][full_mask]
    if np.any(collisions > 0):
        collision_ids = sorted(set(map(int, np.unique(collisions))) - {0})
        raise ValueError("new-cell patch overlaps existing identit"
                         + ("y " if len(collision_ids) == 1 else "ies ")
                         + ", ".join(map(str, collision_ids)))
    delegated = set(map(int, operation.get("linked_split_frames", [])))
    per_frame: list[dict[str, Any]] = []
    added_pixels = 0
    manual_pixels = 0
    assisted_pixels = 0
    for offset, imagej_frame in enumerate(range(start, end + 1)):
        mask = full_mask[offset]
        if not np.any(mask):
            if imagej_frame not in delegated:
                raise ValueError(
                    f"new identity has an empty ImageJ frame {imagej_frame}")
            per_frame.append({
                "imagej_frame": imagej_frame, "added_pixels": 0,
                "manual_pixels": 0, "assisted_pixels": 0,
                "delegated_to_forced_split": True,
            })
            continue
        local_provenance = np.zeros(labels.shape[1:], np.uint8)
        local_provenance[y0:y1, x0:x1] = patch_provenance[offset]
        labels[imagej_frame - 1][mask] = identity
        provenance[imagej_frame - 1][mask] = local_provenance[mask]
        frame_added = int(np.count_nonzero(mask))
        frame_manual = int(np.count_nonzero(
            local_provenance[mask] == MANUAL_NEW_CELL_PROVENANCE))
        frame_assisted = int(np.count_nonzero(
            local_provenance[mask] == ASSISTED_NEW_CELL_PROVENANCE))
        added_pixels += frame_added
        manual_pixels += frame_manual
        assisted_pixels += frame_assisted
        per_frame.append({
            "imagej_frame": imagej_frame, "added_pixels": frame_added,
            "manual_pixels": frame_manual,
            "assisted_pixels": frame_assisted,
            "delegated_to_forced_split": False,
        })
    if not added_pixels:
        raise ValueError("new-cell operation did not add any pixels")
    return {
        "affected_imagej_frames": list(range(start, end + 1)),
        "added_pixels": added_pixels,
        "manual_pixels": manual_pixels,
        "assisted_pixels": assisted_pixels,
        "changed_pixels": added_pixels,
        "per_frame": per_frame,
    }


def _encode_2d_mask(mask: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(mask, np.uint8)
    if value.ndim != 2:
        raise ValueError("draft outline must be two-dimensional")
    return {
        "shape": list(map(int, value.shape)),
        "runs": _positive_runs(value),
        "sha256": state_sha256(value.astype(bool)),
    }


def _decode_2d_mask(record: dict[str, Any]) -> np.ndarray:
    if not isinstance(record, dict) or not isinstance(record.get("shape"), list) \
            or len(record["shape"]) != 2:
        raise ValueError("draft outline record is invalid")
    shape = tuple(map(int, record["shape"]))
    mask = _decode_positive_runs(shape, record.get("runs", []), np.uint8) > 0
    if state_sha256(mask) != record.get("sha256"):
        raise ValueError("draft outline fingerprint differs")
    return mask


def request_to_draft_record(request: NewCellRequest) -> dict[str, Any]:
    return {
        "identity": int(request.identity),
        "start_imagej_frame": int(request.start_imagej_frame),
        "end_imagej_frame": int(request.end_imagej_frame),
        "anchor_points": [{
            "imagej_frame": int(frame), "y": int(point[0]), "x": int(point[1])}
            for frame, point in sorted(request.anchor_points.items())],
        "method": request.method,
        "crop_side_px": int(request.crop_side_px),
        "threshold_z": float(request.threshold_z),
        "threshold_smoothing": float(request.threshold_smoothing),
        "threshold_keyframes": [{
            "imagej_frame": int(frame), "threshold_z": float(value)}
            for frame, value in sorted(request.threshold_keyframes.items())],
        "minimum_area_px": int(request.minimum_area_px),
        "maximum_area_fraction": float(request.maximum_area_fraction),
        "start_reason": request.start_reason,
        "end_reason": request.end_reason,
        "manual_masks": [{
            "imagej_frame": int(frame), "mask": _encode_2d_mask(mask)}
            for frame, mask in sorted(request.manual_masks.items())],
    }


def request_from_draft_record(record: dict[str, Any]) -> NewCellRequest:
    if not isinstance(record, dict):
        raise ValueError("draft new-cell request is invalid")
    anchors = record.get("anchor_points")
    thresholds = record.get("threshold_keyframes", [])
    manual = record.get("manual_masks", [])
    if not isinstance(anchors, list) or not isinstance(thresholds, list) \
            or not isinstance(manual, list):
        raise ValueError("draft new-cell request lists are invalid")
    return NewCellRequest(
        identity=int(record["identity"]),
        start_imagej_frame=int(record["start_imagej_frame"]),
        end_imagej_frame=int(record["end_imagej_frame"]),
        anchor_points={
            int(row["imagej_frame"]): (int(row["y"]), int(row["x"]))
            for row in anchors},
        method=str(record["method"]),
        crop_side_px=int(record["crop_side_px"]),
        threshold_z=float(record["threshold_z"]),
        threshold_smoothing=float(record["threshold_smoothing"]),
        threshold_keyframes={
            int(row["imagej_frame"]): float(row["threshold_z"])
            for row in thresholds},
        minimum_area_px=int(record["minimum_area_px"]),
        maximum_area_fraction=float(record["maximum_area_fraction"]),
        start_reason=str(record["start_reason"]),
        end_reason=str(record["end_reason"]),
        manual_masks={
            int(row["imagej_frame"]): _decode_2d_mask(row["mask"])
            for row in manual},
    )


def proposal_to_draft_job(proposal: NewCellProposal | None,
                          request: NewCellRequest,
                          accepted_frames: Iterable[int],
                          linked_split_frames: Iterable[int] = (),
                          linked_split_operations: list[dict[str, Any]] | None = None
                          ) -> dict[str, Any]:
    """Serialize configured and partially reviewed work for crash recovery."""
    row: dict[str, Any] = {
        "request": request_to_draft_record(request),
        "accepted_imagej_frames": sorted(set(map(int, accepted_frames))),
        "linked_split_frames": sorted(set(map(int, linked_split_frames))),
        "linked_split_operations": list(linked_split_operations or []),
    }
    if proposal is not None and np.any(proposal.masks):
        row["proposal"] = {
            "accepted_patch": encode_new_identity_patch(proposal),
            "crop_centres": proposal.crop_centres,
            "per_frame": proposal.per_frame,
            "warnings": proposal.warnings,
            "lifetime_errors": proposal.lifetime_errors,
            "parent_state_sha256": proposal.parent_state_sha256,
        }
    return row


def proposal_from_draft_job(
        labels: np.ndarray, raw: np.ndarray, row: dict[str, Any]
        ) -> tuple[NewCellRequest, NewCellProposal | None, set[int],
                   set[int], list[dict[str, Any]]]:
    if not isinstance(row, dict):
        raise ValueError("draft new-cell job is invalid")
    request = request_from_draft_record(row.get("request"))
    labels = promote_labels_for_identity(labels, request.identity)
    _validate_request(labels, raw, request)
    accepted = {int(value) for value in row.get("accepted_imagej_frames", [])}
    linked = {int(value) for value in row.get("linked_split_frames", [])}
    operations = row.get("linked_split_operations", [])
    if not isinstance(operations, list) or any(
            not isinstance(value, dict) for value in operations):
        raise ValueError("draft linked split operations are invalid")
    proposal_record = row.get("proposal")
    if proposal_record is None:
        return request, None, accepted, linked, operations
    if not isinstance(proposal_record, dict):
        raise ValueError("draft new-cell proposal is invalid")
    patch_mask, patch_provenance, (y0, x0) = _decode_new_identity_patch(
        proposal_record.get("accepted_patch"))
    expected_frames = request.end_imagej_frame - request.start_imagej_frame + 1
    if patch_mask.shape[0] != expected_frames:
        raise ValueError("draft new-cell proposal frame count differs")
    masks = np.zeros((expected_frames, *labels.shape[1:]), bool)
    provenance = np.zeros(masks.shape, np.uint8)
    y1, x1 = y0 + patch_mask.shape[1], x0 + patch_mask.shape[2]
    if y1 > labels.shape[1] or x1 > labels.shape[2]:
        raise ValueError("draft new-cell proposal lies outside the image")
    masks[:, y0:y1, x0:x1] = patch_mask
    provenance[:, y0:y1, x0:x1] = patch_provenance
    proposal = NewCellProposal(
        request=request, masks=masks, provenance=provenance,
        crop_centres=list(proposal_record.get("crop_centres", [])),
        per_frame=list(proposal_record.get("per_frame", [])),
        warnings=list(proposal_record.get("warnings", [])),
        lifetime_errors=list(proposal_record.get("lifetime_errors", [])),
        parent_state_sha256=str(proposal_record.get("parent_state_sha256", "")),
    )
    expected_hash = new_identity_scope_state_sha256(
        labels, request.identity, request.start_imagej_frame,
        request.end_imagej_frame, masks)
    if proposal.parent_state_sha256 != expected_hash:
        raise ValueError("draft new-cell proposal is stale relative to its parent")
    return request, proposal, accepted, linked, operations


def save_new_cell_draft(path: Path, parent_state_sha256: str,
                        jobs: list[dict[str, Any]]) -> Path:
    """Atomically replace a mutable, explicitly non-authoritative draft."""
    if not isinstance(parent_state_sha256, str) or len(parent_state_sha256) != 64:
        raise ValueError("draft parent fingerprint is invalid")
    if not isinstance(jobs, list):
        raise ValueError("draft jobs must be a list")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "motion.manual-new-cell-draft", "schema_version": 1,
        "authoritative": False, "parent_state_sha256": parent_state_sha256,
        "jobs": jobs,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_new_cell_draft(path: Path,
                        expected_parent_state_sha256: str) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) \
            or document.get("schema") != "motion.manual-new-cell-draft" \
            or document.get("schema_version") != 1 \
            or document.get("authoritative") is not False \
            or not isinstance(document.get("jobs"), list):
        raise ValueError("new-cell draft is invalid")
    if document.get("parent_state_sha256") != expected_parent_state_sha256:
        raise ValueError("new-cell draft is stale relative to the active parent")
    return document


def editing_state_sha256(labels: np.ndarray) -> str:
    """Public fingerprint used to bind non-authoritative drafts to one checkpoint."""
    return state_sha256(labels)
