from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.segmentation import watershed


STRUCTURE = np.ones((3, 3), bool)
SPLIT_METHODS = (
    "marker_controlled_watershed",
    "soma_seed_expansion",
    "manual_drawing",
    "temporal_persistence",
)
ASSISTED_SPLIT_PROVENANCE = 3
MANUAL_SPLIT_PROVENANCE = 4


@dataclass(frozen=True)
class SplitRequest:
    host_identity: int
    start_imagej_frame: int
    end_imagej_frame: int
    child_identities: tuple[int, ...]
    method: str = "marker_controlled_watershed"
    seeds: dict[int, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    manual_assignments: dict[int, np.ndarray] = field(default_factory=dict)
    minimum_child_area_px: int = 3


@dataclass
class SplitProposal:
    request: SplitRequest
    labels: np.ndarray
    provenance: np.ndarray
    per_frame: list[dict[str, Any]]
    warnings: list[str]
    boundary_errors: list[str]
    parent_state_sha256: str

    @property
    def valid(self) -> bool:
        return not self.boundary_errors and all(
            row["valid"] for row in self.per_frame)


def state_sha256(labels: np.ndarray) -> str:
    value = np.ascontiguousarray(labels)
    header = json.dumps({
        "dtype": value.dtype.name,
        "shape": list(map(int, value.shape)),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def split_scope_state_sha256(
        labels: np.ndarray, host_identity: int, start_imagej_frame: int,
        end_imagej_frame: int, child_identities: tuple[int, ...]) -> str:
    """Fingerprint only identities whose state can make this proposal stale."""
    scope_start = max(0, start_imagej_frame - 2)
    scope_end = min(len(labels), end_imagej_frame + 1)
    selected = labels[scope_start:scope_end]
    scoped = np.zeros(selected.shape, labels.dtype)
    relevant = {int(host_identity), *map(int, child_identities)}
    for identity in relevant:
        scoped[selected == identity] = identity
    return state_sha256(scoped)


def allocate_child_identities(
        labels: np.ndarray, host_identity: int, child_count: int,
        reserved: set[int] | None = None) -> tuple[int, ...]:
    """Keep the host name and reserve collision-free positive child names."""
    if child_count < 2 or child_count > 4:
        raise ValueError("forced splitting supports between 2 and 4 children")
    if host_identity <= 0 or not np.any(labels == host_identity):
        raise ValueError(f"host identity {host_identity} is absent")
    used = {int(value) for value in np.unique(labels) if value > 0}
    used.update(set() if reserved is None else {int(value) for value in reserved})
    maximum = int(np.iinfo(labels.dtype).max)
    children = [int(host_identity)]
    candidate = max(used, default=0) + 1
    while len(children) < child_count:
        while candidate in used:
            candidate += 1
        if candidate > maximum:
            raise ValueError(
                f"no unused identity fits the {labels.dtype.name} label data type")
        children.append(candidate)
        used.add(candidate)
        candidate += 1
    return tuple(children)


def suggest_child_identities(
        labels: np.ndarray, host_identity: int, start_imagej_frame: int,
        end_imagej_frame: int, child_count: int = 2,
        reserved: set[int] | None = None) -> tuple[int, ...]:
    """Prefer nearby identities at clean adjacent frames, then allocate new names."""
    if start_imagej_frame < 1 or end_imagej_frame < start_imagej_frame \
            or end_imagej_frame > len(labels):
        raise ValueError(f"split frame range must be within 1-{len(labels)}")
    reserved_values = set() if reserved is None else set(map(int, reserved))
    selected = labels[start_imagej_frame - 1:end_imagej_frame]
    candidates: dict[int, float] = {}
    for adjacent_index, host_index in (
            (start_imagej_frame - 2, start_imagej_frame - 1),
            (end_imagej_frame, end_imagej_frame - 1)):
        if adjacent_index < 0 or adjacent_index >= len(labels):
            continue
        host = labels[host_index] == host_identity
        if not np.any(host):
            continue
        distance = ndi.distance_transform_edt(~host)
        frame = labels[adjacent_index]
        for value in np.unique(frame):
            identity = int(value)
            if identity <= 0 or identity == host_identity \
                    or identity in reserved_values \
                    or np.any(selected == identity):
                continue
            candidate_mask = frame == identity
            separation = float(np.min(distance[candidate_mask]))
            candidates[identity] = min(candidates.get(identity, np.inf), separation)
    children = [int(host_identity)]
    children.extend(
        identity for identity, _distance in sorted(
            candidates.items(), key=lambda row: (row[1], row[0]))
        if len(children) < child_count)
    if len(children) < child_count:
        allocated = allocate_child_identities(
            labels, host_identity, child_count,
            reserved_values | set(children[1:]))
        for identity in allocated[1:]:
            if identity not in children:
                children.append(identity)
            if len(children) == child_count:
                break
    return tuple(children)


def _validate_request(
        labels: np.ndarray, raw: np.ndarray, request: SplitRequest) -> None:
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("forced splitting requires integer TYX labels")
    if raw.shape != labels.shape:
        raise ValueError("split raw images and labels differ in shape")
    if request.host_identity <= 0:
        raise ValueError("split host identity must be positive")
    if request.method not in SPLIT_METHODS:
        raise ValueError(f"unsupported split method: {request.method!r}")
    if request.start_imagej_frame < 1 \
            or request.end_imagej_frame < request.start_imagej_frame \
            or request.end_imagej_frame > len(labels):
        raise ValueError(
            f"split frame range must be within 1-{len(labels)}")
    children = tuple(int(value) for value in request.child_identities)
    if len(children) < 2 or len(children) > 4:
        raise ValueError("forced splitting requires between 2 and 4 children")
    if len(set(children)) != len(children) or any(value <= 0 for value in children):
        raise ValueError("split child identities must be unique positive integers")
    if request.host_identity not in children:
        raise ValueError("one split child must retain the host identity")
    if max(children) > int(np.iinfo(labels.dtype).max):
        raise ValueError(
            f"a split child identity exceeds the {labels.dtype.name} capacity")
    if request.minimum_child_area_px < 1:
        raise ValueError("minimum split child area must be positive")
    selected = labels[
        request.start_imagej_frame - 1:request.end_imagej_frame]
    absent_frames = [
        request.start_imagej_frame + offset
        for offset, frame in enumerate(selected)
        if not np.any(frame == request.host_identity)
    ]
    if absent_frames:
        joined = ", ".join(map(str, absent_frames[:8]))
        suffix = "..." if len(absent_frames) > 8 else ""
        raise ValueError(
            f"host identity {request.host_identity} is absent from ImageJ "
            f"frame(s) {joined}{suffix}")
    for child in children:
        if child == request.host_identity:
            continue
        if np.any(selected == child):
            raise ValueError(
                f"child identity {child} already has pixels inside the split interval")


def _snap_to_mask(mask: np.ndarray, point: tuple[float, float]) -> tuple[int, int]:
    pixels = np.column_stack(np.nonzero(mask))
    if not len(pixels):
        raise ValueError("cannot place a split seed in an empty host")
    target = np.asarray(point, float)
    index = int(np.argmin(np.sum((pixels - target[None, :]) ** 2, axis=1)))
    return int(pixels[index, 0]), int(pixels[index, 1])


def _spread_points(mask: np.ndarray, points: list[tuple[int, int]],
                   count: int) -> list[tuple[int, int]]:
    pixels = np.column_stack(np.nonzero(mask))
    if not len(pixels):
        return points
    result = list(dict.fromkeys(points))
    if not result:
        distance = ndi.distance_transform_edt(mask)
        first = np.unravel_index(int(np.argmax(distance)), mask.shape)
        result.append((int(first[0]), int(first[1])))
    while len(result) < count:
        chosen = np.asarray(result, float)
        distances = np.min(
            np.sum((pixels[:, None, :] - chosen[None, :, :]) ** 2, axis=2),
            axis=1)
        candidate = tuple(map(int, pixels[int(np.argmax(distances))]))
        if candidate in result:
            break
        result.append(candidate)
    return result[:count]


def default_split_seeds(
        host: np.ndarray, raw: np.ndarray, child_count: int
        ) -> tuple[tuple[int, int], ...]:
    """Find intensity bodies, then fall back to maximally separated host points."""
    smooth = ndi.gaussian_filter(raw.astype(np.float32), 1.0)
    values = smooth[host]
    threshold = float(np.percentile(values, 40)) if values.size else 0.0
    minimum_distance = max(1, int(round(np.sqrt(max(int(host.sum()), 1)) / 6)))
    peaks = peak_local_max(
        smooth, labels=host.astype(np.uint8), num_peaks=child_count,
        min_distance=minimum_distance, threshold_abs=threshold,
        exclude_border=False)
    points = [tuple(map(int, point)) for point in peaks]
    points = _spread_points(host, points, child_count)
    if len(points) != child_count:
        raise ValueError("the host is too small to place distinct child seeds")
    return tuple(points)


def _interpolated_seeds(
        request: SplitRequest, imagej_frame: int, host: np.ndarray,
        raw: np.ndarray) -> tuple[tuple[int, int], ...]:
    count = len(request.child_identities)
    supplied = {
        int(frame): tuple(tuple(map(int, point)) for point in points)
        for frame, points in request.seeds.items()
        if len(points) == count
    }
    if imagej_frame in supplied:
        points = supplied[imagej_frame]
    elif supplied:
        earlier = [frame for frame in supplied if frame < imagej_frame]
        later = [frame for frame in supplied if frame > imagej_frame]
        if earlier and later:
            left, right = max(earlier), min(later)
            alpha = (imagej_frame - left) / (right - left)
            points = tuple(
                tuple((1.0 - alpha) * np.asarray(supplied[left][index], float)
                      + alpha * np.asarray(supplied[right][index], float))
                for index in range(count))
        else:
            nearest = min(supplied, key=lambda value: abs(value - imagej_frame))
            points = supplied[nearest]
    else:
        return default_split_seeds(host, raw, count)
    snapped = [_snap_to_mask(host, point) for point in points]
    snapped = _spread_points(host, snapped, count)
    if len(set(snapped)) != count:
        raise ValueError(
            f"ImageJ frame {imagej_frame} does not have distinct child seeds")
    return tuple(snapped)


def _manual_markers(
        assignment: np.ndarray | None, host: np.ndarray,
        child_identities: tuple[int, ...]) -> np.ndarray:
    markers = np.zeros(host.shape, np.int32)
    if assignment is None:
        return markers
    values = np.asarray(assignment)
    if values.shape != host.shape:
        raise ValueError("manual split assignment and host frame differ in shape")
    for index, child in enumerate(child_identities, start=1):
        markers[host & ((values == index) | (values == child))] = index
    return markers


def _soma_core_markers(
        markers: np.ndarray, host: np.ndarray, raw: np.ndarray,
        seeds: tuple[tuple[int, int], ...]) -> np.ndarray:
    result = markers.copy()
    host_values = raw[host].astype(np.float32)
    floor = float(np.percentile(host_values, 55)) if host_values.size else 0.0
    yy, xx = np.ogrid[:host.shape[0], :host.shape[1]]
    for index, (seed_y, seed_x) in enumerate(seeds, start=1):
        seed_value = float(raw[seed_y, seed_x])
        threshold = max(floor, seed_value * 0.65)
        local = (host & (raw >= threshold)
                 & (((yy - seed_y) ** 2 + (xx - seed_x) ** 2) <= 5 ** 2))
        components, _count = ndi.label(local, structure=STRUCTURE)
        component_id = int(components[seed_y, seed_x])
        if component_id > 0:
            core = components == component_id
            # A different child's existing manual core wins contested pixels.
            result[core & (result == 0)] = index
        result[seed_y, seed_x] = index
    return result


def _partition_frame(
        host: np.ndarray, raw: np.ndarray, method: str,
        seeds: tuple[tuple[int, int], ...], child_identities: tuple[int, ...],
        manual_assignment: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    markers = _manual_markers(manual_assignment, host, child_identities)
    for index, seed in enumerate(seeds, start=1):
        if not np.any(markers == index):
            markers[seed] = index
    if method == "soma_seed_expansion":
        markers = _soma_core_markers(markers, host, raw, seeds)
    if any(not np.any(markers == index)
           for index in range(1, len(child_identities) + 1)):
        raise ValueError("every child requires a seed or painted ownership")

    smooth = ndi.gaussian_filter(raw.astype(np.float32), 1.0)
    if method == "marker_controlled_watershed":
        elevation = -smooth
    elif method == "soma_seed_expansion":
        local = smooth.copy()
        values = local[host]
        scale = float(np.percentile(values, 95) - np.percentile(values, 10)) \
            if values.size else 1.0
        normalised = (local - (float(np.percentile(values, 10))
                               if values.size else 0.0)) / max(scale, 1e-6)
        elevation = sobel(np.clip(normalised, 0, 1)) - 0.15 * np.clip(
            normalised, 0, 1)
    elif method == "temporal_persistence":
        elevation = sobel(smooth)
    elif method == "manual_drawing":
        if manual_assignment is not None:
            claimed = host & (markers > 0)
            if np.array_equal(claimed, host):
                partition = np.zeros(host.shape, np.int32)
                partition[host] = markers[host]
                return partition, claimed
        elevation = sobel(smooth)
    else:  # validated by the public entry point
        raise ValueError(f"unsupported split method: {method!r}")

    partition = watershed(
        elevation, markers=markers, mask=host, connectivity=STRUCTURE)
    manually_claimed = host & (_manual_markers(
        manual_assignment, host, child_identities) > 0)
    return partition.astype(np.int32, copy=False), manually_claimed


def validate_partition_frame(
        host: np.ndarray, partition: np.ndarray,
        child_identities: tuple[int, ...], minimum_child_area_px: int,
        seeds: tuple[tuple[int, int], ...] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    child_rows: list[dict[str, Any]] = []
    owned = partition > 0
    if not np.array_equal(owned, host):
        errors.append("child masks do not conserve the exact host union")
    for index, child in enumerate(child_identities, start=1):
        mask = partition == index
        area = int(mask.sum())
        components = int(ndi.label(mask, structure=STRUCTURE)[1]) if area else 0
        seed_retained = (
            None if seeds is None else bool(mask[seeds[index - 1]]))
        if area < minimum_child_area_px:
            errors.append(
                f"child identity {child} has {area} pixels; minimum is "
                f"{minimum_child_area_px}")
        if components != 1:
            errors.append(
                f"child identity {child} has {components} connected components")
        if seed_retained is False:
            errors.append(f"child identity {child} does not retain its seed")
        child_rows.append({
            "identity": int(child),
            "area_px": area,
            "connected_components": components,
            "seed_retained": seed_retained,
        })
    return {
        "valid": not errors,
        "errors": errors,
        "host_pixels": int(host.sum()),
        "assigned_pixels": int(np.count_nonzero(owned & host)),
        "outside_pixels": int(np.count_nonzero(owned & ~host)),
        "children": child_rows,
    }


def _proposal_confidence(
        raw: np.ndarray, partition: np.ndarray, child_count: int) -> float:
    if child_count < 2:
        return 0.0
    boundary = np.zeros(partition.shape, bool)
    for index in range(1, child_count + 1):
        mask = partition == index
        boundary |= mask & ndi.binary_dilation(
            (partition > 0) & (partition != index), structure=STRUCTURE)
    if not np.any(boundary):
        return 0.0
    body_values = [
        float(np.percentile(raw[partition == index], 90))
        for index in range(1, child_count + 1)
        if np.any(partition == index)
    ]
    if not body_values:
        return 0.0
    waist = float(np.median(raw[boundary]))
    scale = max(float(np.ptp(raw[partition > 0])), 1.0)
    return float(np.clip((min(body_values) - waist) / scale, 0.0, 1.0))


def _shape_descriptor(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask)).astype(float)
    if len(points) < 2:
        return np.zeros(3, float)
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    axes = np.sqrt(np.maximum(
        np.sort(np.linalg.eigvalsh(covariance))[::-1], 0.0) + 1e-6)
    scale = np.sqrt(max(len(points), 1))
    edge = mask ^ ndi.binary_erosion(mask, structure=STRUCTURE)
    return np.array([axes[0] / scale, axes[1] / scale,
                     float(edge.sum()) / scale])


def _identity_descriptor(mask: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    points = np.column_stack(np.nonzero(mask))
    return {
        "position": points.mean(axis=0),
        "area": float(mask.sum()),
        "mean": float(raw[mask].mean()),
        "shape": _shape_descriptor(mask),
    }


def _temporal_expected_descriptors(
        labels: np.ndarray, raw: np.ndarray, request: SplitRequest,
        imagej_frame: int) -> dict[int, dict[str, Any]]:
    bookends: list[tuple[int, dict[int, dict[str, Any]]]] = []
    for frame in (
            request.start_imagej_frame - 1,
            request.end_imagej_frame + 1):
        if frame < 1 or frame > len(labels):
            continue
        descriptors = {}
        for child in request.child_identities:
            mask = labels[frame - 1] == child
            if np.any(mask):
                descriptors[child] = _identity_descriptor(mask, raw[frame - 1])
        if len(descriptors) == len(request.child_identities):
            bookends.append((frame, descriptors))
    if not bookends:
        return {}
    if len(bookends) == 1:
        return bookends[0][1]
    left_frame, left = bookends[0]
    right_frame, right = bookends[-1]
    alpha = float(np.clip(
        (imagej_frame - left_frame) / max(right_frame - left_frame, 1),
        0.0, 1.0))
    return {
        child: {
            key: ((1.0 - alpha) * left[child][key]
                  + alpha * right[child][key])
            for key in ("position", "area", "mean", "shape")
        }
        for child in request.child_identities
    }


def _temporal_partition_score(
        partition: np.ndarray, child_identities: tuple[int, ...],
        raw: np.ndarray, expected: dict[int, dict[str, Any]],
        previous_masks: dict[int, np.ndarray]) -> float:
    score = 0.0
    for index, child in enumerate(child_identities, start=1):
        mask = partition == index
        if not np.any(mask):
            return np.inf
        measured = _identity_descriptor(mask, raw)
        reference = expected.get(child)
        if reference is not None:
            score += float(np.linalg.norm(
                measured["position"] - reference["position"])) / 12.0
            score += abs(float(np.log2(
                measured["area"] / max(float(reference["area"]), 1.0))))
            score += 1.5 * abs(float(np.log2(
                (measured["mean"] + 1.0)
                / (float(reference["mean"]) + 1.0))))
            score += float(np.mean(np.abs(np.log2(
                (measured["shape"] + 1e-3)
                / (np.asarray(reference["shape"]) + 1e-3)))))
        previous = previous_masks.get(child)
        if previous is not None:
            expanded = ndi.binary_dilation(previous, structure=STRUCTURE)
            overlap = np.count_nonzero(mask & expanded) / max(
                min(int(mask.sum()), int(previous.sum())), 1)
            score -= float(overlap)
    return float(score)


def _temporal_partition_candidates(
        host: np.ndarray, raw: np.ndarray,
        seeds: tuple[tuple[int, int], ...],
        child_identities: tuple[int, ...],
        manual_assignment: np.ndarray | None,
        expected: dict[int, dict[str, Any]],
        previous_masks: dict[int, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray, float | None]:
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    for method in (
            "temporal_persistence", "marker_controlled_watershed",
            "soma_seed_expansion"):
        partition, manual = _partition_frame(
            host, raw, method, seeds, child_identities, manual_assignment)
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(partition).view(np.uint8)).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        score = _temporal_partition_score(
            partition, child_identities, raw, expected, previous_masks)
        candidates.append((score, partition, manual))
    candidates.sort(key=lambda row: row[0])
    best_score, best_partition, best_manual = candidates[0]
    margin = (float(candidates[1][0] - best_score)
              if len(candidates) > 1 else None)
    return best_partition, best_manual, margin


def validate_boundary_continuity(
        labels: np.ndarray, request: SplitRequest) -> list[str]:
    """Reject child lifetimes that silently stop or resume beside the interval."""
    errors: list[str] = []
    start = request.start_imagej_frame
    end = request.end_imagej_frame
    for child in request.child_identities:
        existed_before = bool(np.any(labels[:start - 1] == child))
        continues_before = start == 1 or bool(np.any(labels[start - 2] == child))
        if existed_before and not continues_before:
            errors.append(
                f"child identity {child} exists before the interval but not in "
                f"the adjacent ImageJ frame {start - 1}")
        if end < len(labels) and not np.any(labels[end] == child):
            errors.append(
                f"child identity {child} would stop inside the movie after "
                f"ImageJ frame {end}; extend the interval or reconnect an "
                "identity present in the next frame")
    return errors


def _automatic_bookend_seeds(
        labels: np.ndarray, request: SplitRequest
        ) -> dict[int, tuple[tuple[int, int], ...]]:
    """Use declared child identities in adjacent clean frames as temporal seeds."""
    result: dict[int, tuple[tuple[int, int], ...]] = {}
    for imagej_frame in (
            request.start_imagej_frame - 1,
            request.end_imagej_frame + 1):
        if imagej_frame < 1 or imagej_frame > len(labels):
            continue
        frame = labels[imagej_frame - 1]
        points = []
        for child in request.child_identities:
            pixels = np.column_stack(np.nonzero(frame == child))
            if not len(pixels):
                break
            centre = pixels.mean(axis=0)
            points.append((int(round(centre[0])), int(round(centre[1]))))
        if len(points) == len(request.child_identities):
            result[imagej_frame] = tuple(points)
    return result


def propose_forced_split(
        labels: np.ndarray, raw: np.ndarray, request: SplitRequest
        ) -> SplitProposal:
    """Generate an in-memory, strict-host split proposal without mutating inputs."""
    labels = np.asarray(labels)
    raw = np.asarray(raw)
    _validate_request(labels, raw, request)
    if request.method == "temporal_persistence":
        combined_seeds = {
            **_automatic_bookend_seeds(labels, request),
            **request.seeds,
        }
        request = replace(request, seeds=combined_seeds)
    frame_count = request.end_imagej_frame - request.start_imagej_frame + 1
    proposal = np.zeros((frame_count, *labels.shape[1:]), labels.dtype)
    provenance = np.zeros(proposal.shape, np.uint8)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    previous_centres: tuple[tuple[int, int], ...] | None = None
    previous_masks: dict[int, np.ndarray] = {}

    for offset, imagej_frame in enumerate(range(
            request.start_imagej_frame, request.end_imagej_frame + 1)):
        t = imagej_frame - 1
        host = labels[t] == request.host_identity
        frame_seeds = _interpolated_seeds(request, imagej_frame, host, raw[t])
        if request.method == "temporal_persistence" \
                and imagej_frame not in request.seeds \
                and previous_centres is not None:
            frame_seeds = tuple(
                _snap_to_mask(host, point) for point in previous_centres)
        manual = request.manual_assignments.get(imagej_frame)
        runner_up_margin = None
        if request.method == "temporal_persistence":
            partition, manually_claimed, runner_up_margin = \
                _temporal_partition_candidates(
                    host, raw[t], frame_seeds, request.child_identities,
                    manual, _temporal_expected_descriptors(
                        labels, raw, request, imagej_frame), previous_masks)
        else:
            partition, manually_claimed = _partition_frame(
                host, raw[t], request.method, frame_seeds,
                request.child_identities, manual)
        validation = validate_partition_frame(
            host, partition, request.child_identities,
            request.minimum_child_area_px, frame_seeds)
        for index, child in enumerate(request.child_identities, start=1):
            proposal[offset][partition == index] = child
        provenance[offset][host] = ASSISTED_SPLIT_PROVENANCE
        provenance[offset][manually_claimed] = MANUAL_SPLIT_PROVENANCE
        confidence = _proposal_confidence(
            raw[t], partition, len(request.child_identities))
        if confidence < 0.05:
            warnings.append(
                f"ImageJ frame {imagej_frame} has weak visible boundary evidence")
        row = {
            "imagej_frame": imagej_frame,
            "seeds_yx": [[int(y), int(x)] for y, x in frame_seeds],
            "boundary_confidence": confidence,
            "runner_up_margin": runner_up_margin,
            "manual_pixels": int(np.count_nonzero(manually_claimed)),
            **validation,
        }
        rows.append(row)
        if validation["valid"]:
            centres: list[tuple[int, int]] = []
            for child in request.child_identities:
                points = np.column_stack(np.nonzero(proposal[offset] == child))
                centre = points.mean(axis=0)
                centres.append((int(round(centre[0])), int(round(centre[1]))))
            previous_centres = tuple(centres)
            previous_masks = {
                child: proposal[offset] == child
                for child in request.child_identities
            }

    return SplitProposal(
        request=request,
        labels=proposal,
        provenance=provenance,
        per_frame=rows,
        warnings=warnings,
        boundary_errors=validate_boundary_continuity(labels, request),
        parent_state_sha256=split_scope_state_sha256(
            labels, request.host_identity, request.start_imagej_frame,
            request.end_imagej_frame, request.child_identities),
    )


def revalidate_split_proposal(
        source_labels: np.ndarray, proposal: SplitProposal) -> SplitProposal:
    """Recompute hard gates after an interactive ownership correction."""
    rows: list[dict[str, Any]] = []
    request = proposal.request
    for offset, imagej_frame in enumerate(range(
            request.start_imagej_frame, request.end_imagej_frame + 1)):
        host = source_labels[imagej_frame - 1] == request.host_identity
        indexed = np.zeros(host.shape, np.int32)
        for index, child in enumerate(request.child_identities, start=1):
            indexed[proposal.labels[offset] == child] = index
        previous = proposal.per_frame[offset]
        validation = validate_partition_frame(
            host, indexed, request.child_identities,
            request.minimum_child_area_px)
        rows.append({
            **previous,
            **validation,
            "manual_pixels": int(np.count_nonzero(
                proposal.provenance[offset] == MANUAL_SPLIT_PROVENANCE)),
        })
    proposal.per_frame = rows
    return proposal


def paint_split_ownership(
        source_labels: np.ndarray, proposal: SplitProposal,
        imagej_frame: int, child_identity: int,
        points_yx: list[tuple[int, int]], radius_px: int = 1) -> SplitProposal:
    """Paint exact ownership inside the host and mark it as manual provenance."""
    request = proposal.request
    if imagej_frame < request.start_imagej_frame \
            or imagej_frame > request.end_imagej_frame:
        raise ValueError("painted frame is outside the split interval")
    if child_identity not in request.child_identities:
        raise ValueError("painted child is not part of this split")
    if radius_px < 0:
        raise ValueError("paint radius cannot be negative")
    offset = imagej_frame - request.start_imagej_frame
    host = source_labels[imagej_frame - 1] == request.host_identity
    brush = np.zeros(host.shape, bool)
    yy, xx = np.ogrid[:host.shape[0], :host.shape[1]]
    for y, x in points_yx:
        if 0 <= y < host.shape[0] and 0 <= x < host.shape[1]:
            brush |= (yy - y) ** 2 + (xx - x) ** 2 <= radius_px ** 2
    brush &= host
    if not np.any(brush):
        raise ValueError("manual split brush did not touch the host identity")
    proposal.labels[offset][brush] = child_identity
    proposal.provenance[offset][brush] = MANUAL_SPLIT_PROVENANCE
    return revalidate_split_proposal(source_labels, proposal)


def _positive_runs(array: np.ndarray) -> list[list[int]]:
    flat = np.asarray(array).reshape(-1)
    runs: list[list[int]] = []
    index = 0
    while index < len(flat):
        value = int(flat[index])
        if value == 0:
            index += 1
            continue
        end = index + 1
        while end < len(flat) and int(flat[end]) == value:
            end += 1
        runs.append([int(index), int(end - index), value])
        index = end
    return runs


def decode_positive_runs(
        shape: tuple[int, ...], runs: list[list[int]], dtype: np.dtype
        ) -> np.ndarray:
    size = int(np.prod(shape))
    flat = np.zeros(size, dtype=dtype)
    previous_end = 0
    for row in runs:
        if not isinstance(row, list) or len(row) != 3 \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in row):
            raise ValueError("split patch contains an invalid run")
        start, length, value = row
        end = start + length
        if start < previous_end or length < 1 or end > size or value <= 0:
            raise ValueError("split patch run is out of bounds or overlapping")
        flat[start:end] = value
        previous_end = end
    return flat.reshape(shape)


def encode_split_patch(proposal: SplitProposal) -> dict[str, Any]:
    request = proposal.request
    union = proposal.labels > 0
    spatial = np.any(union, axis=0)
    yy, xx = np.nonzero(spatial)
    if not len(yy):
        raise ValueError("cannot encode an empty split proposal")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    cropped_labels = np.ascontiguousarray(proposal.labels[:, y0:y1, x0:x1])
    cropped_provenance = np.ascontiguousarray(
        proposal.provenance[:, y0:y1, x0:x1])
    return {
        "schema": "motion.force-split-patch",
        "schema_version": 1,
        "origin_yx": [y0, x0],
        "shape": list(map(int, cropped_labels.shape)),
        "label_dtype": cropped_labels.dtype.name,
        "label_runs": _positive_runs(cropped_labels),
        "provenance_runs": _positive_runs(cropped_provenance),
        "labels_sha256": state_sha256(cropped_labels),
        "provenance_sha256": state_sha256(cropped_provenance),
    }


def build_force_split_operation(proposal: SplitProposal) -> dict[str, Any]:
    """Freeze a reviewed proposal into a deterministic edit request."""
    if not proposal.valid:
        failed = [
            str(row["imagej_frame"]) for row in proposal.per_frame
            if not row["valid"]]
        details = []
        if failed:
            details.append("invalid frame(s): " + ", ".join(failed))
        details.extend(proposal.boundary_errors)
        raise ValueError("cannot commit split; " + "; ".join(details))
    request = proposal.request
    return {
        "type": "force_split_identity_interval",
        "host_identity": int(request.host_identity),
        "start_imagej_frame": int(request.start_imagej_frame),
        "end_imagej_frame": int(request.end_imagej_frame),
        "child_identities": list(map(int, request.child_identities)),
        "method": request.method,
        "method_version": 1,
        "parent_state_sha256": proposal.parent_state_sha256,
        "accepted_patch": encode_split_patch(proposal),
        "proposal_warnings": list(proposal.warnings),
        "proposal_metrics": proposal.per_frame,
        "reviewed_imagej_frames": list(range(
            request.start_imagej_frame, request.end_imagej_frame + 1)),
    }


def _decode_patch(patch: dict[str, Any], dtype: np.dtype
                  ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not isinstance(patch, dict) \
            or patch.get("schema") != "motion.force-split-patch" \
            or patch.get("schema_version") != 1:
        raise ValueError("forced-split operation has no supported accepted patch")
    shape_value = patch.get("shape")
    origin_value = patch.get("origin_yx")
    if not isinstance(shape_value, list) or len(shape_value) != 3 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in shape_value):
        raise ValueError("forced-split patch shape is invalid")
    if not isinstance(origin_value, list) or len(origin_value) != 2 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in origin_value):
        raise ValueError("forced-split patch origin is invalid")
    shape = tuple(shape_value)
    label_runs = patch.get("label_runs")
    provenance_runs = patch.get("provenance_runs")
    if not isinstance(label_runs, list) or not isinstance(provenance_runs, list):
        raise ValueError("forced-split patch runs are invalid")
    patch_dtype = np.dtype(str(patch.get("label_dtype", "")))
    labels = decode_positive_runs(shape, label_runs, patch_dtype)
    provenance = decode_positive_runs(shape, provenance_runs, np.uint8)
    if state_sha256(labels) != patch.get("labels_sha256"):
        raise ValueError("forced-split label patch fingerprint differs")
    if state_sha256(provenance) != patch.get("provenance_sha256"):
        raise ValueError("forced-split provenance patch fingerprint differs")
    if not np.can_cast(patch_dtype, dtype, casting="safe"):
        raise ValueError("forced-split patch cannot be represented by current labels")
    return labels.astype(dtype, copy=False), provenance, (
        int(origin_value[0]), int(origin_value[1]))


def normalize_force_split_request(
        request: dict[str, Any], total_frames: int) -> dict[str, Any]:
    """Validate JSON-facing split fields while retaining the frozen patch."""
    allowed = {
        "type", "user", "host_identity", "start_imagej_frame",
        "end_imagej_frame", "child_identities", "method", "method_version",
        "parent_state_sha256", "accepted_patch", "proposal_warnings",
        "proposal_metrics", "reviewed_imagej_frames",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(
            "unsupported fields for force_split_identity_interval: "
            + ", ".join(unknown))

    def integer(key: str, minimum: int = 1) -> int:
        value = request.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"batch operation {key} must be an integer >= {minimum}")
        return int(value)

    host = integer("host_identity")
    start = integer("start_imagej_frame")
    end = integer("end_imagej_frame")
    if end < start or end > total_frames:
        raise ValueError(
            f"forced-split frame range must be within 1-{total_frames}")
    children = request.get("child_identities")
    if not isinstance(children, list) or len(children) < 2 or len(children) > 4 \
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in children) \
            or len(set(children)) != len(children) or host not in children:
        raise ValueError(
            "forced-split child identities must be 2-4 unique positive integers "
            "including the host")
    method = request.get("method")
    if method not in SPLIT_METHODS:
        raise ValueError(f"unsupported split method: {method!r}")
    if integer("method_version") != 1:
        raise ValueError("unsupported forced-split method version")
    parent_hash = request.get("parent_state_sha256")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise ValueError("forced-split parent-state fingerprint is invalid")
    warnings = request.get("proposal_warnings", [])
    metrics = request.get("proposal_metrics", [])
    if not isinstance(warnings, list) or any(
            not isinstance(value, str) for value in warnings):
        raise ValueError("forced-split proposal warnings must be text")
    if not isinstance(metrics, list) or any(
            not isinstance(value, dict) for value in metrics):
        raise ValueError("forced-split proposal metrics must be objects")
    reviewed = request.get("reviewed_imagej_frames")
    if reviewed != list(range(start, end + 1)):
        raise ValueError(
            "forced-split operation must record review of every interval frame")
    patch = request.get("accepted_patch")
    if not isinstance(patch, dict):
        raise ValueError("forced-split accepted patch must be an object")
    # Structural and fingerprint validation does not need the eventual label dtype.
    _decode_patch(patch, np.dtype(str(patch.get("label_dtype", "uint16"))))
    normalized = {
        "type": "force_split_identity_interval",
        "host_identity": host,
        "start_imagej_frame": start,
        "end_imagej_frame": end,
        "child_identities": list(map(int, children)),
        "method": method,
        "method_version": 1,
        "parent_state_sha256": parent_hash,
        "accepted_patch": patch,
        "proposal_warnings": warnings,
        "proposal_metrics": metrics,
        "reviewed_imagej_frames": reviewed,
    }
    if "user" in request:
        normalized["user"] = request["user"]
    return normalized


def apply_force_split_operation(
        labels: np.ndarray, provenance: np.ndarray,
        operation: dict[str, Any], excluded_identities: set[int] | None = None
        ) -> dict[str, Any]:
    """Apply one frozen accepted patch and return deterministic audit fields."""
    host = int(operation["host_identity"])
    start = int(operation["start_imagej_frame"])
    end = int(operation["end_imagej_frame"])
    children = tuple(map(int, operation["child_identities"]))
    if excluded_identities and any(child in excluded_identities for child in children):
        raise ValueError("a forced-split child identity is excluded")
    if split_scope_state_sha256(
            labels, host, start, end, children) \
            != operation["parent_state_sha256"]:
        raise ValueError(
            "forced-split proposal is stale relative to the replayed parent state")
    patch_labels, patch_provenance, (y0, x0) = _decode_patch(
        operation["accepted_patch"], labels.dtype)
    expected_frames = end - start + 1
    if patch_labels.shape[0] != expected_frames:
        raise ValueError("forced-split patch frame count differs from its interval")
    y1, x1 = y0 + patch_labels.shape[1], x0 + patch_labels.shape[2]
    if y1 > labels.shape[1] or x1 > labels.shape[2]:
        raise ValueError("forced-split patch lies outside the label field")
    if set(map(int, np.unique(patch_labels))) - {0, *children}:
        raise ValueError("forced-split patch contains an undeclared child identity")
    if np.any((patch_provenance > 0)
              & ~np.isin(patch_provenance, [
                  ASSISTED_SPLIT_PROVENANCE, MANUAL_SPLIT_PROVENANCE])):
        raise ValueError("forced-split patch provenance is invalid")

    per_child = {child: 0 for child in children}
    per_frame: list[dict[str, Any]] = []
    changed_pixels = 0
    manual_pixels = 0
    for offset, imagej_frame in enumerate(range(start, end + 1)):
        frame = labels[imagej_frame - 1]
        current_host = frame == host
        candidate = np.zeros(frame.shape, labels.dtype)
        candidate[y0:y1, x0:x1] = patch_labels[offset]
        patch_support = candidate > 0
        if not np.array_equal(patch_support, current_host):
            raise ValueError(
                f"forced-split patch no longer matches host {host} in ImageJ "
                f"frame {imagej_frame}")
        for child in children:
            if child != host and np.any(frame == child):
                raise ValueError(
                    f"child identity {child} already exists in ImageJ frame "
                    f"{imagej_frame}")
        before = frame[current_host].copy()
        frame[current_host] = candidate[current_host]
        local_provenance = np.zeros(frame.shape, np.uint8)
        local_provenance[y0:y1, x0:x1] = patch_provenance[offset]
        provenance[imagej_frame - 1][current_host] = \
            local_provenance[current_host]
        changed = int(np.count_nonzero(before != frame[current_host]))
        changed_pixels += changed
        manual = int(np.count_nonzero(
            local_provenance[current_host] == MANUAL_SPLIT_PROVENANCE))
        manual_pixels += manual
        child_rows = []
        for child in children:
            count = int(np.count_nonzero(candidate == child))
            per_child[child] += count
            child_rows.append({"identity": child, "pixels": count})
        per_frame.append({
            "imagej_frame": imagej_frame,
            "host_pixels": int(current_host.sum()),
            "changed_pixels": changed,
            "manual_corrected_pixels": manual,
            "per_child": child_rows,
        })
    if not changed_pixels:
        raise ValueError("forced split did not assign any host pixels to another child")
    return {
        "affected_imagej_frames": list(range(start, end + 1)),
        "split_pixels": int(sum(per_child.values())),
        "changed_pixels": changed_pixels,
        "manual_corrected_pixels": manual_pixels,
        "per_child": [
            {"identity": child, "pixels": per_child[child]}
            for child in children
        ],
        "per_frame": per_frame,
    }
