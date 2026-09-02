from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import utc_now, write_json
from manual_editing import (EditingHistoryController, _create_output,
                            _path_record, _record_failed_output,
                            _validate_file_record, save_edit_batch_session)


FILTER_SCHEMA = "motion.manual-editing-filter-set"
FILTER_SCHEMA_VERSION = 1
FILTER_REPORT_SCHEMA = "motion.manual-editing-filter-report"

FILTERABLE_METRICS = {
    "observed_frames": "frames containing the identity",
    "span_frames": "frames from the first through last observation",
    "track_span_min": "elapsed minutes from first through last observation",
    "missing_frames": "missing frames inside the track span",
    "gap_runs": "separate missing-frame gaps inside the track span",
    "longest_gap_frames": "longest missing-frame gap",
    "coverage_fraction": "observed frames divided by the track span",
    "flicker_score": "(missing frames + gap runs) divided by span frames",
    "median_area": "median area in the bundle's calibrated area unit",
    "min_area": "minimum area in the bundle's calibrated area unit",
    "max_area": "maximum area in the bundle's calibrated area unit",
    "median_area_px": "median pixel count",
    "area_cv": "area coefficient of variation",
    "median_intensity": "median within-outline raw intensity",
    "intensity_cv": "raw-intensity coefficient of variation",
    "total_distance": "centroid path length in the calibrated distance unit",
    "net_displacement": "first-to-last centroid displacement",
    "furthest_point_distance": (
        "largest centroid separation between any two observed frames"),
    "mobility_radius": "root-mean-square distance from the median centroid",
    "mean_speed": "time-weighted mean centroid speed",
    "median_speed": "median observed centroid speed",
    "max_speed": "maximum observed centroid speed",
    "total_distance_px": "centroid path length in pixels",
    "mean_speed_px_per_frame": "mean centroid displacement per frame",
    "directionality": "net displacement divided by path length",
    "merge_contact_frames": "frames sharing a label boundary with another identity",
    "merge_contact_fraction": "fraction of observed frames with shared boundaries",
    "merge_contact_partners": "number of identities sharing a boundary",
    "border_touch_frames": "frames touching an image border",
    "border_touch_fraction": "fraction of observed frames touching a border",
    "fragmented_mask_frames": "frames with disconnected regions under one identity",
    "fragmented_mask_fraction": "fraction of observed frames with disconnected regions",
    "manual_pixel_fraction": "fraction of pixels created by manual drawing",
    "centroid_assisted_pixel_fraction": (
        "fraction of pixels created by centroid-assisted detection"),
    "derived_quality_score": (
        "non-probabilistic mean of duration support, continuity, area stability, "
        "and mask integrity"),
}

_OPERATORS = {
    "lt": lambda values, threshold: values < threshold,
    "lte": lambda values, threshold: values <= threshold,
    "gt": lambda values, threshold: values > threshold,
    "gte": lambda values, threshold: values >= threshold,
    "eq": lambda values, threshold: values == threshold,
}


def load_filter_set(path: Path) -> dict[str, Any]:
    return validate_filter_set(json.loads(path.resolve().read_text(encoding="utf-8")))


def validate_filter_set(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("filter set must be an object")
    if document.get("schema") != FILTER_SCHEMA:
        raise ValueError(f"unsupported filter-set schema: {document.get('schema')!r}")
    if document.get("schema_version") != FILTER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported filter-set schema version: "
            f"{document.get('schema_version')!r}")
    combine = document.get("combine", "all")
    if combine not in {"all", "any"}:
        raise ValueError("filter-set combine must be 'all' or 'any'")
    filters = document.get("filters")
    if not isinstance(filters, list) or not filters:
        raise ValueError("filter set must contain at least one filter")
    normalized_filters: list[dict[str, Any]] = []
    for index, condition in enumerate(filters, start=1):
        if not isinstance(condition, dict):
            raise ValueError(f"filter {index} must be an object")
        unknown = set(condition) - {"name", "metric", "operator", "value"}
        if unknown:
            raise ValueError(
                f"filter {index} has unsupported fields: {', '.join(sorted(unknown))}")
        metric = condition.get("metric")
        if metric not in FILTERABLE_METRICS:
            raise ValueError(f"filter {index} uses unsupported metric: {metric!r}")
        operator = condition.get("operator")
        if operator not in {*_OPERATORS, "between", "outside"}:
            raise ValueError(f"filter {index} uses unsupported operator: {operator!r}")
        value = condition.get("value")
        if operator in {"between", "outside"}:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(
                    f"filter {index} {operator} value must contain two numbers")
            values = [_finite_number(item, f"filter {index} value") for item in value]
            if values[0] > values[1]:
                raise ValueError(f"filter {index} lower value exceeds upper value")
            normalized_value: float | list[float] = values
        else:
            normalized_value = _finite_number(value, f"filter {index} value")
        name = condition.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError(f"filter {index} name must be non-empty text")
        normalized_filters.append({
            "name": (name.strip() if isinstance(name, str)
                     else f"filter_{index}"),
            "metric": metric,
            "operator": operator,
            "value": normalized_value,
        })
    return {
        "schema": FILTER_SCHEMA,
        "schema_version": FILTER_SCHEMA_VERSION,
        "combine": combine,
        "filters": normalized_filters,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _source_state(source_path: Path, operation_count: int | None = None):
    controller = EditingHistoryController(source_path)
    if operation_count is not None:
        controller.restore(operation_count)
    labels, provenance, excluded = controller.materialize()
    return controller, labels, provenance, excluded


def _calibration(bundle) -> tuple[str, float, float]:
    row = bundle.manifest.get("spatial_calibration", {})
    unit = row.get("unit", "pixel")
    size_y = row.get("pixel_size_y", 1.0)
    size_x = row.get("pixel_size_x", 1.0)
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("editing bundle has no valid spatial unit")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value <= 0 for value in (size_y, size_x)):
        raise ValueError("editing bundle has invalid pixel sizes")
    return unit.strip(), float(size_y), float(size_x)


def _furthest_point_distance(
        y: np.ndarray, x: np.ndarray,
        pixel_size_y: float, pixel_size_x: float) -> float:
    """Return the greatest calibrated separation between observed centroids."""
    if len(y) < 2:
        return 0.0
    points = np.column_stack((y * pixel_size_y, x * pixel_size_x))
    furthest_squared = 0.0
    for index, point in enumerate(points[:-1]):
        offsets = points[index + 1:] - point
        furthest_squared = max(
            furthest_squared,
            float(np.max(np.einsum("ij,ij->i", offsets, offsets))))
    return float(np.sqrt(furthest_squared))


def _contact_evidence(frame: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for first, second in (
            (frame[:, :-1], frame[:, 1:]),
            (frame[:-1, :], frame[1:, :])):
        mask = (first > 0) & (second > 0) & (first != second)
        if not np.any(mask):
            continue
        values = np.column_stack((first[mask], second[mask])).astype(np.int64)
        values.sort(axis=1)
        pairs.update(map(tuple, np.unique(values, axis=0).tolist()))
    return pairs


def build_track_filter_metrics(
        source_path: Path, operation_count: int | None = None
        ) -> tuple[pd.DataFrame, dict[str, Any]]:
    controller, labels, provenance, _excluded = _source_state(
        source_path, operation_count)
    bundle = controller.bundle
    raw = bundle.registered_raw
    unit, pixel_size_y, pixel_size_x = _calibration(bundle)
    frame_interval_min = float(bundle.manifest["time"]["frame_interval_min"])
    rows: list[dict[str, Any]] = []
    contact_frames: dict[int, set[int]] = {}
    contact_partners: dict[int, set[int]] = {}
    structure = ndi.generate_binary_structure(2, 1)

    for t, frame in enumerate(labels):
        for identity_a, identity_b in _contact_evidence(frame):
            for identity, partner in (
                    (identity_a, identity_b), (identity_b, identity_a)):
                contact_frames.setdefault(identity, set()).add(t)
                contact_partners.setdefault(identity, set()).add(partner)
        objects = ndi.find_objects(frame)
        for identity_value in np.unique(frame):
            identity = int(identity_value)
            if identity == 0:
                continue
            region = objects[identity - 1]
            if region is None:
                continue
            mask = frame[region] == identity
            local_y, local_x = np.nonzero(mask)
            y = local_y.astype(float) + region[0].start
            x = local_x.astype(float) + region[1].start
            pixel_count = int(len(y))
            provenance_values = provenance[t][region][mask]
            rows.append({
                "identity": identity,
                "t": t,
                "area_px": pixel_count,
                "area": pixel_count * pixel_size_y * pixel_size_x,
                "y": float(y.mean()),
                "x": float(x.mean()),
                "mean_intensity": float(raw[t][region][mask].mean()),
                "component_count": int(ndi.label(mask, structure=structure)[1]),
                "touches_border": bool(
                    region[0].start == 0 or region[1].start == 0
                    or region[0].stop == frame.shape[0]
                    or region[1].stop == frame.shape[1]),
                "manual_pixels": int(np.count_nonzero(np.isin(
                    provenance_values, [1, 3, 4]))),
                "centroid_assisted_pixels": int(np.count_nonzero(np.isin(
                    provenance_values, [2, 5]))),
            })

    columns = ["identity", *FILTERABLE_METRICS]
    if not rows:
        return pd.DataFrame(columns=columns), _metric_metadata(
            unit, frame_interval_min, pixel_size_y, pixel_size_x)
    frame_table = pd.DataFrame(rows)
    metrics: list[dict[str, Any]] = []
    for identity, group in frame_table.groupby("identity", sort=True):
        group = group.sort_values("t")
        frames = group.t.to_numpy(dtype=int)
        frame_steps = np.diff(frames)
        missing_by_gap = np.maximum(frame_steps - 1, 0)
        observed_frames = int(len(group))
        span_frames = int(frames[-1] - frames[0] + 1)
        missing_frames = int(missing_by_gap.sum())
        gap_runs = int(np.count_nonzero(missing_by_gap))
        longest_gap = int(missing_by_gap.max()) if len(missing_by_gap) else 0
        areas = group.area.to_numpy(dtype=float)
        areas_px = group.area_px.to_numpy(dtype=float)
        intensities = group.mean_intensity.to_numpy(dtype=float)
        y = group.y.to_numpy(dtype=float)
        x = group.x.to_numpy(dtype=float)
        if len(group) > 1:
            dy = np.diff(y)
            dx = np.diff(x)
            distances = np.sqrt(
                (dy * pixel_size_y) ** 2 + (dx * pixel_size_x) ** 2)
            distances_px = np.sqrt(dy ** 2 + dx ** 2)
            elapsed_min = frame_steps.astype(float) * frame_interval_min
            speeds = distances / elapsed_min
            speeds_px_per_frame = distances_px / frame_steps
            total_distance = float(distances.sum())
            total_distance_px = float(distances_px.sum())
            total_elapsed_min = float(elapsed_min.sum())
            net_displacement = float(np.hypot(
                (y[-1] - y[0]) * pixel_size_y,
                (x[-1] - x[0]) * pixel_size_x))
        else:
            speeds = np.array([], dtype=float)
            speeds_px_per_frame = np.array([], dtype=float)
            total_distance = 0.0
            total_distance_px = 0.0
            total_elapsed_min = 0.0
            net_displacement = 0.0
        centre_y = float(np.median(y))
        centre_x = float(np.median(x))
        furthest_point_distance = _furthest_point_distance(
            y, x, pixel_size_y, pixel_size_x)
        mobility_radius = float(np.sqrt(np.mean(
            ((y - centre_y) * pixel_size_y) ** 2
            + ((x - centre_x) * pixel_size_x) ** 2)))
        area_cv = _coefficient_of_variation(areas)
        intensity_cv = _coefficient_of_variation(intensities)
        fragmented_frames = int(np.count_nonzero(group.component_count > 1))
        border_frames = int(np.count_nonzero(group.touches_border))
        identity_int = int(identity)
        merge_frames = len(contact_frames.get(identity_int, set()))
        manual_pixels = int(group.manual_pixels.sum())
        assisted_pixels = int(group.centroid_assisted_pixels.sum())
        total_pixels = int(group.area_px.sum())
        coverage = observed_frames / span_frames
        integrity = 1.0 - fragmented_frames / observed_frames
        quality = float(np.mean((
            min(observed_frames / 8.0, 1.0),
            coverage,
            1.0 / (1.0 + area_cv),
            integrity,
        )))
        metrics.append({
            "identity": identity_int,
            "observed_frames": observed_frames,
            "span_frames": span_frames,
            "track_span_min": float(max(span_frames - 1, 0) * frame_interval_min),
            "missing_frames": missing_frames,
            "gap_runs": gap_runs,
            "longest_gap_frames": longest_gap,
            "coverage_fraction": float(coverage),
            "flicker_score": float(min(
                (missing_frames + gap_runs) / span_frames, 1.0)),
            "median_area": float(np.median(areas)),
            "min_area": float(areas.min()),
            "max_area": float(areas.max()),
            "median_area_px": float(np.median(areas_px)),
            "area_cv": area_cv,
            "median_intensity": float(np.median(intensities)),
            "intensity_cv": intensity_cv,
            "total_distance": total_distance,
            "net_displacement": net_displacement,
            "furthest_point_distance": furthest_point_distance,
            "mobility_radius": mobility_radius,
            "mean_speed": (0.0 if not total_elapsed_min
                           else total_distance / total_elapsed_min),
            "median_speed": (0.0 if not len(speeds)
                             else float(np.median(speeds))),
            "max_speed": (0.0 if not len(speeds) else float(speeds.max())),
            "total_distance_px": total_distance_px,
            "mean_speed_px_per_frame": (
                0.0 if not len(speeds_px_per_frame)
                else float(np.mean(speeds_px_per_frame))),
            "directionality": (0.0 if not total_distance
                               else net_displacement / total_distance),
            "merge_contact_frames": merge_frames,
            "merge_contact_fraction": merge_frames / observed_frames,
            "merge_contact_partners": len(contact_partners.get(identity_int, set())),
            "border_touch_frames": border_frames,
            "border_touch_fraction": border_frames / observed_frames,
            "fragmented_mask_frames": fragmented_frames,
            "fragmented_mask_fraction": fragmented_frames / observed_frames,
            "manual_pixel_fraction": manual_pixels / total_pixels,
            "centroid_assisted_pixel_fraction": assisted_pixels / total_pixels,
            "derived_quality_score": quality,
        })
    return pd.DataFrame(metrics, columns=columns), _metric_metadata(
        unit, frame_interval_min, pixel_size_y, pixel_size_x)


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return 0.0 if mean == 0 else float(np.std(values) / abs(mean))


def _metric_metadata(unit: str, frame_interval_min: float,
                     pixel_size_y: float, pixel_size_x: float) -> dict[str, Any]:
    return {
        "distance_unit": unit,
        "area_unit": f"{unit}^2",
        "speed_unit": f"{unit}/min",
        "frame_interval_min": frame_interval_min,
        "pixel_size_y": pixel_size_y,
        "pixel_size_x": pixel_size_x,
        "pipeline_global_confidence_available": False,
        "derived_quality_score_is_probability": False,
        "derived_quality_score_formula": (
            "mean(min(observed_frames/8,1), coverage_fraction, "
            "1/(1+area_cv), 1-fragmented_mask_fraction)"),
        "merge_contact_definition": (
            "a geometric proxy: identities share a four-connected label boundary; "
            "the accepted pipeline does not bundle final per-identity merge events"),
        "metrics": FILTERABLE_METRICS,
    }


def evaluate_track_filters(
        metrics: pd.DataFrame, filter_set: dict[str, Any]
        ) -> pd.DataFrame:
    spec = validate_filter_set(filter_set)
    conditions: list[np.ndarray] = []
    labels: list[str] = []
    for condition in spec["filters"]:
        metric = condition["metric"]
        if metric not in metrics.columns:
            raise ValueError(f"metric table does not contain {metric}")
        values = metrics[metric].to_numpy(dtype=float)
        operator = condition["operator"]
        threshold = condition["value"]
        if operator in _OPERATORS:
            result = _OPERATORS[operator](values, float(threshold))
        else:
            lower, upper = map(float, threshold)
            inside = (values >= lower) & (values <= upper)
            result = inside if operator == "between" else ~inside
        conditions.append(np.asarray(result, dtype=bool))
        labels.append(condition["name"])
    matrix = np.column_stack(conditions)
    selected = matrix.all(axis=1) if spec["combine"] == "all" else matrix.any(axis=1)
    result = metrics.copy()
    result["selected"] = selected
    result["matched_filter_count"] = matrix.sum(axis=1).astype(int)
    result["matched_filters"] = [
        ";".join(label for label, matched in zip(labels, row) if matched)
        for row in matrix
    ]
    return result


def select_identities_by_filters(
        source_path: Path, filter_set: dict[str, Any],
        operation_count: int | None = None
        ) -> tuple[list[int], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    spec = validate_filter_set(filter_set)
    metrics, metadata = build_track_filter_metrics(source_path, operation_count)
    matches = evaluate_track_filters(metrics, spec)
    selected = matches.loc[matches.selected, "identity"].astype(int).tolist()
    return selected, matches, metadata, spec


def preview_track_filters(
        source_path: Path, filter_set: dict[str, Any], output_dir: Path,
        operation_count: int | None = None) -> Path:
    selected, matches, metadata, spec = select_identities_by_filters(
        source_path, filter_set, operation_count)
    controller = EditingHistoryController(source_path)
    output = _create_output(output_dir)
    try:
        metrics_path = output / "track_filter_metrics.csv"
        matches_path = output / "filter_matches.csv"
        selected_path = output / "selected_identities.csv"
        filters_path = output / "filter-set.json"
        matches.drop(columns=[
            "selected", "matched_filter_count", "matched_filters"
        ]).to_csv(metrics_path, index=False)
        matches.to_csv(matches_path, index=False)
        pd.DataFrame({"identity": selected}).to_csv(selected_path, index=False)
        write_json(filters_path, spec)
        manifest = {
            "schema": FILTER_REPORT_SCHEMA,
            "schema_version": 1,
            "status": "done",
            "created_at_utc": utc_now(),
            "source": _path_record(controller.source_manifest, output, False),
            "source_operation_count": (
                len(controller.operations) if operation_count is None
                else int(operation_count)),
            "filter_set": _path_record(filters_path, output, True),
            "outputs": {
                "track_metrics": _path_record(metrics_path, output, True),
                "filter_matches": _path_record(matches_path, output, True),
                "selected_identities": _path_record(selected_path, output, True),
            },
            "metric_metadata": metadata,
            "summary": {
                "candidate_identities": int(len(matches)),
                "selected_identities": int(len(selected)),
                "selected_identity_values": selected,
            },
        }
        manifest_path = output / "motion-editing-filter-report.json"
        write_json(manifest_path, manifest)
        load_filter_report(manifest_path)
    except BaseException as error:
        _record_failed_output(output, error)
        raise
    return output / "motion-editing-filter-report.json"


def load_filter_report(path: Path) -> dict[str, Any]:
    manifest_path = path.resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "motion-editing-filter-report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) \
            or manifest.get("schema") != FILTER_REPORT_SCHEMA \
            or manifest.get("schema_version") != 1 \
            or manifest.get("status") != "done":
        raise ValueError("filter report manifest is invalid")
    _validate_file_record(manifest_path, manifest.get("source", {}), True)
    filter_path = _validate_file_record(
        manifest_path, manifest.get("filter_set", {}), True)
    load_filter_set(filter_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("filter report has no outputs")
    resolved = {
        name: _validate_file_record(manifest_path, outputs.get(name, {}), True)
        for name in ("track_metrics", "filter_matches", "selected_identities")
    }
    metrics = pd.read_csv(resolved["track_metrics"])
    matches = pd.read_csv(resolved["filter_matches"])
    selected = pd.read_csv(resolved["selected_identities"])
    if "identity" not in metrics or not {
            "identity", "selected", "matched_filters"}.issubset(matches.columns) \
            or list(selected.columns) != ["identity"]:
        raise ValueError("filter report tables are invalid")
    selected_values = selected.identity.astype(int).tolist()
    summary = manifest.get("summary", {})
    if summary.get("candidate_identities") != len(metrics) \
            or summary.get("selected_identities") != len(selected_values) \
            or summary.get("selected_identity_values") != selected_values:
        raise ValueError("filter report summary differs from its tables")
    return manifest


def save_filtered_operation_session(
        source_path: Path, output_dir: Path, filter_set: dict[str, Any],
        operation_type: str, operation_parameters: dict[str, Any] | None = None,
        user: str | None = None, operation_count: int | None = None) -> Path:
    """Apply one operation to all identities selected by shared filters."""
    selected, matches, metadata, spec = select_identities_by_filters(
        source_path, filter_set, operation_count)
    if not selected:
        raise ValueError("filter set selected no identities; no session was written")
    selected_rows = matches.loc[matches.selected]
    reasons = dict(zip(
        selected_rows.identity.astype(int), selected_rows.matched_filters))
    canonical_spec = json.dumps(
        spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    context = {
        "selection_method": "track_filters",
        "operation_type": operation_type,
        "operation_parameters": operation_parameters or {},
        "filter_set": spec,
        "filter_set_sha256": hashlib.sha256(canonical_spec).hexdigest(),
        "selected_identities": selected,
        "selected_metrics": json.loads(selected_rows.to_json(orient="records")),
        "metric_metadata": metadata,
    }
    parameters = operation_parameters or {}
    if not isinstance(parameters, dict):
        raise ValueError("filtered operation parameters must be an object")
    if operation_type == "exclude_identity":
        if parameters:
            raise ValueError("identity removal takes no operation parameters")
        requests = [{
            "type": "exclude_identity",
            "identity": identity,
            "reason": f"selected by track filters: {reasons[identity]}",
        } for identity in selected]
        mode = "filtered_identity_exclusion"
    elif operation_type == "expand_identities":
        requests = [{
            "type": "expand_identities",
            "identities": selected,
            **parameters,
        }]
        mode = "filtered_identity_expansion"
    elif operation_type == "delete_identity_interval":
        requests = [{
            "type": "delete_identity_interval",
            "identities": selected,
            **parameters,
        }]
        mode = "filtered_identity_interval_deletion"
    else:
        raise ValueError(f"unsupported filtered operation: {operation_type!r}")
    return save_edit_batch_session(
        source_path, output_dir, requests, user, operation_count,
        mode=mode, batch_context=context)


def save_filtered_removal_session(
        source_path: Path, output_dir: Path, filter_set: dict[str, Any],
        user: str | None = None, operation_count: int | None = None) -> Path:
    return save_filtered_operation_session(
        source_path, output_dir, filter_set, "exclude_identity", {}, user,
        operation_count)


def save_filtered_expansion_session(
        source_path: Path, output_dir: Path, filter_set: dict[str, Any],
        radius: float, radius_unit: str = "calibrated",
        start_imagej_frame: int = 1,
        end_imagej_frame: int | None = None,
        user: str | None = None, operation_count: int | None = None) -> Path:
    return save_filtered_operation_session(
        source_path, output_dir, filter_set, "expand_identities", {
            "radius": radius,
            "radius_unit": radius_unit,
            "start_imagej_frame": start_imagej_frame,
            "end_imagej_frame": end_imagej_frame,
        }, user, operation_count)


def save_filtered_interval_deletion_session(
        source_path: Path, output_dir: Path, filter_set: dict[str, Any],
        start_imagej_frame: int, end_imagej_frame: int,
        user: str | None = None, operation_count: int | None = None) -> Path:
    return save_filtered_operation_session(
        source_path, output_dir, filter_set, "delete_identity_interval", {
            "start_imagej_frame": start_imagej_frame,
            "end_imagej_frame": end_imagej_frame,
        }, user, operation_count)
