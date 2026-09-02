from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from common import (Config, ROOT, change_overlay, outline_overlay, save_rgb_stack,
                    save_stack, sha256, utc_now, validate_labels, write_json)


BUNDLE_SCHEMA = "motion.manual-editing-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILENAME = "motion-editing-bundle.json"
SESSION_SCHEMA = "motion.manual-editing-session"
SESSION_SCHEMA_VERSION = 1
SESSION_FILENAME = "motion-editing-session.json"
BATCH_SCHEMA = "motion.manual-editing-batch"
BATCH_SCHEMA_VERSION = 1
MANUAL_PROVENANCE_VALUES = {
    "0": "automatic parent",
    "1": "manually drawn",
    "2": "centroid-assisted detection",
    "3": "accepted assisted split",
    "4": "manual split correction",
    "5": "accepted assisted mask retuning",
}


def _portable_source(path: Path) -> bool:
    from manual_portable_package import is_portable_package

    return is_portable_package(Path(path))


def _editing_manifest_path(path: Path) -> Path:
    """Resolve a bundle, session, package directory, or package archive entry."""
    resolved = Path(path).resolve()
    if _portable_source(resolved):
        from manual_portable_package import resolve_portable_entry

        return resolve_portable_entry(resolved)
    if resolved.is_dir():
        session_path = resolved / SESSION_FILENAME
        bundle_path = resolved / BUNDLE_FILENAME
        manifest_path = (session_path if session_path.is_file()
                         else bundle_path if bundle_path.is_file() else None)
        if manifest_path is None:
            raise FileNotFoundError(
                f"no editing bundle or session manifest in {resolved}")
        return manifest_path
    return resolved


@dataclass(frozen=True)
class EditingBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    canonical_labels: np.ndarray
    registered_raw: np.ndarray
    registered_raw_indices: np.ndarray
    unclaimed_labels: np.ndarray | None
    evidence_paths: dict[str, Path]


@dataclass(frozen=True)
class EditingSession:
    manifest_path: Path
    manifest: dict[str, Any]
    bundle: EditingBundle
    curated_labels: np.ndarray
    manual_provenance: np.ndarray
    excluded_identities: set[int]
    edits: list[dict[str, Any]]
    frame_identities: pd.DataFrame
    tracks: pd.DataFrame


@dataclass(frozen=True)
class EditingCheckpoint:
    operation_count: int
    action: str
    scope: str
    user: str
    created_at_utc: str
    changed_pixels: int


class EditingHistoryController:
    """Navigate the replayable edit log without mutating saved sessions."""

    def __init__(self, source_path: Path):
        resolved = Path(source_path).resolve()
        self.source_reference = resolved
        self.portable_source = _portable_source(resolved)
        manifest_path = _editing_manifest_path(resolved)
        document = _json_object(manifest_path)
        schema = document.get("schema")
        if schema == SESSION_SCHEMA:
            session = load_editing_session(manifest_path)
            self.bundle = session.bundle
            self.source_manifest = session.manifest_path
            self.operations = list(session.edits)
        elif schema == BUNDLE_SCHEMA:
            self.bundle = load_editing_bundle(manifest_path)
            self.source_manifest = self.bundle.manifest_path
            self.operations = []
        else:
            raise ValueError(f"unsupported editing source schema: {schema!r}")
        self.position = len(self.operations)
        self.checkpoints = [_automatic_checkpoint(self.bundle), *[
            _operation_checkpoint(operation)
            for operation in self.operations
        ]]

    @property
    def can_undo(self) -> bool:
        return self.position > 0

    @property
    def can_redo(self) -> bool:
        return self.position < len(self.operations)

    @property
    def current_checkpoint(self) -> EditingCheckpoint:
        return self.checkpoints[self.position]

    def restore(self, operation_count: int) -> EditingCheckpoint:
        if operation_count < 0 or operation_count > len(self.operations):
            raise ValueError(
                f"checkpoint must be between 0 and {len(self.operations)}")
        self.position = operation_count
        return self.current_checkpoint

    def undo(self) -> EditingCheckpoint:
        if not self.can_undo:
            raise ValueError("already at the original automatic output")
        return self.restore(self.position - 1)

    def redo(self) -> EditingCheckpoint:
        if not self.can_redo:
            raise ValueError("already at the latest edit")
        return self.restore(self.position + 1)

    def materialize(self) -> tuple[np.ndarray, np.ndarray, set[int]]:
        return replay_edit_log(self.bundle, self.operations[:self.position])

    def continue_from_here(
            self, output_dir: Path | None = None,
            user: str | None = None) -> Path:
        if self.position == len(self.operations):
            return (self.source_reference if self.portable_source
                    else self.source_manifest)
        if output_dir is None:
            raise ValueError(
                "an output folder is required when continuing from an earlier edit")
        if self.portable_source:
            from manual_portable_package import save_portable_history_checkpoint

            return save_portable_history_checkpoint(
                self.source_reference, output_dir, self.position, user)
        return save_history_checkpoint(
            self.source_manifest, output_dir, self.position, user)

    def remove_identity(
            self, identity: int, output_dir: Path,
            user: str | None = None) -> Path:
        return self.remove_identities([identity], output_dir, user)

    def swap_identities(
            self, identity_a: int, identity_b: int,
            start_imagej_frame: int, output_dir: Path,
            end_imagej_frame: int | None = None,
            user: str | None = None) -> Path:
        return self.apply_batch([{
            "type": "swap_identities",
            "identity_a": int(identity_a),
            "identity_b": int(identity_b),
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": (
                None if end_imagej_frame is None else int(end_imagej_frame)),
        }], output_dir, user)

    def reassign_identities(
            self, source_identities: list[int], target_identity: int,
            start_imagej_frame: int, end_imagej_frame: int,
            output_dir: Path, user: str | None = None) -> Path:
        return self.apply_batch([{
            "type": "reassign_identities",
            "source_identities": [int(identity) for identity in source_identities],
            "target_identity": int(target_identity),
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": int(end_imagej_frame),
        }], output_dir, user)

    def remove_identities(
            self, identities: list[int], output_dir: Path,
            user: str | None = None) -> Path:
        return self.apply_batch([
            {"type": "exclude_identity", "identity": int(identity)}
            for identity in identities
        ], output_dir, user)

    def delete_identities_between_frames(
            self, identities: list[int], start_imagej_frame: int,
            end_imagej_frame: int, output_dir: Path,
            user: str | None = None) -> Path:
        return self.apply_batch([{
            "type": "delete_identity_interval",
            "identities": [int(identity) for identity in identities],
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": int(end_imagej_frame),
        }], output_dir, user)

    def expand_identities(
            self, identities: list[int], radius: float, output_dir: Path,
            radius_unit: str = "calibrated",
            start_imagej_frame: int = 1,
            end_imagej_frame: int | None = None,
            user: str | None = None) -> Path:
        return self.apply_batch([{
            "type": "expand_identities",
            "identities": [int(identity) for identity in identities],
            "radius": float(radius),
            "radius_unit": radius_unit,
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": (
                None if end_imagej_frame is None else int(end_imagej_frame)),
        }], output_dir, user)

    def apply_batch(
            self, operations: list[dict[str, Any]], output_dir: Path,
            user: str | None = None) -> Path:
        if self.portable_source:
            from manual_portable_package import save_portable_edit_batch

            return save_portable_edit_batch(
                self.source_reference, output_dir, operations, user,
                operation_count=self.position)
        return save_edit_batch_session(
            self.source_manifest, output_dir, operations, user,
            operation_count=self.position)

    def add_identity_tracks(
            self, operations: list[dict[str, Any]], output_dir: Path,
            user: str | None = None) -> Path:
        if not operations or any(
                operation.get("type") != "add_identity_track"
                for operation in operations):
            raise ValueError("new-cell save requires add_identity_track operations")
        return self.apply_batch(operations, output_dir, user)

    def retune_masks_after_edit(
            self, operation: dict[str, Any], output_dir: Path,
            user: str | None = None) -> Path:
        if operation.get("type") != "retune_masks_after_edit":
            raise ValueError(
                "mask-retuning save requires a retune_masks_after_edit operation")
        return self.apply_batch([operation], output_dir, user)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _manifest_path(path: Path, filename: str) -> Path:
    resolved = path.resolve()
    return resolved / filename if resolved.is_dir() else resolved


def _tiff_summary(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return {
            "shape": [int(value) for value in series.shape],
            "dtype": np.dtype(series.dtype).name,
            "stored_axes": str(series.axes),
        }


def _read_tiff(path: Path) -> np.ndarray:
    # A real array, rather than a possible memory map, leaves no Windows file
    # handle behind when an editing run is validated inside a Dropbox folder.
    return np.array(tifffile.imread(path), copy=True)


def _path_record(path: Path, manifest_dir: Path,
                 bundled: bool) -> dict[str, Any]:
    resolved = path.resolve()
    if bundled:
        try:
            stored_path = resolved.relative_to(manifest_dir.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"bundled file is outside its bundle: {resolved}") from error
        kind = "bundle_relative"
    else:
        stored_path = str(resolved)
        kind = "external_absolute"
    return {
        "path": stored_path,
        "path_kind": kind,
        "sha256": sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _tiff_record(path: Path, manifest_dir: Path, bundled: bool,
                 semantic_axes: str,
                 label_frame_to_source: list[int] | None = None
                 ) -> dict[str, Any]:
    record = {
        **_path_record(path, manifest_dir, bundled),
        **_tiff_summary(path),
        "semantic_axes": semantic_axes,
    }
    if label_frame_to_source is not None:
        record["label_frame_to_source"] = [
            int(value) for value in label_frame_to_source]
    return record


def _resolve_record(manifest_path: Path, record: dict[str, Any]) -> Path:
    kind = record.get("path_kind")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("file record has no path")
    if kind == "bundle_relative":
        root = manifest_path.parent.resolve()
        result = (root / raw_path).resolve()
        try:
            result.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"bundle-relative path escapes its bundle: {raw_path}") from error
        return result
    if kind == "external_absolute":
        result = Path(raw_path)
        if not result.is_absolute():
            raise ValueError(f"external path is not absolute: {raw_path}")
        return result.resolve()
    raise ValueError(f"unsupported path_kind: {kind!r}")


def _validate_file_record(manifest_path: Path, record: dict[str, Any],
                          verify_hashes: bool) -> Path:
    path = _resolve_record(manifest_path, record)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = record.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"invalid byte count for {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"file size differs for {path}")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"invalid SHA-256 fingerprint for {path}")
    if verify_hashes and sha256(path) != expected_hash:
        raise ValueError(f"SHA-256 fingerprint differs for {path}")
    return path


def _validate_tiff_record(manifest_path: Path, record: dict[str, Any],
                          verify_hashes: bool) -> Path:
    path = _validate_file_record(manifest_path, record, verify_hashes)
    actual = _tiff_summary(path)
    for field in ("shape", "dtype", "stored_axes"):
        if actual[field] != record.get(field):
            raise ValueError(
                f"TIFF {field} differs for {path}: "
                f"{actual[field]!r} != {record.get(field)!r}")
    return path


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _create_output(output_dir: Path) -> Path:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _record_failed_output(output: Path, error: BaseException) -> None:
    write_json(output / "failed.json", {
        "status": "failed",
        "failed_at_utc": utc_now(),
        "error": f"{type(error).__name__}: {error}",
    })


def _validate_source_indices(indices: list[int], label_frames: int,
                             source_frames: int, label: str) -> None:
    if len(indices) != label_frames:
        raise ValueError(
            f"{label} frame mapping has {len(indices)} entries; "
            f"expected {label_frames}")
    if any(not isinstance(value, int) for value in indices):
        raise ValueError(f"{label} frame mapping must contain integers")
    if any(value < 0 or value >= source_frames for value in indices):
        raise ValueError(f"{label} frame mapping is outside the source stack")
    if any(current < previous for previous, current in zip(indices, indices[1:])):
        raise ValueError(f"{label} frame mapping must be nondecreasing")


def _validate_spatial_shape(shape: list[int], labels_shape: tuple[int, ...],
                            label: str) -> None:
    if len(shape) < 3 or tuple(shape[-2:]) != tuple(labels_shape[-2:]):
        raise ValueError(f"{label} and canonical labels differ in field shape")


def _spatial_calibration(
        source: Path, unit: str | None = None,
        pixel_size_y: float | None = None,
        pixel_size_x: float | None = None) -> dict[str, Any]:
    explicit = (unit, pixel_size_y, pixel_size_x)
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise ValueError(
                "spatial unit and both pixel sizes must be supplied together")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("spatial unit cannot be empty")
        sizes = (pixel_size_y, pixel_size_x)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) or value <= 0 for value in sizes):
            raise ValueError("spatial pixel sizes must be positive")
        return {
            "axes": "YX", "unit": unit.strip(),
            "pixel_size_y": float(pixel_size_y),
            "pixel_size_x": float(pixel_size_x),
            "source": "explicit",
        }

    with tifffile.TiffFile(source) as tif:
        metadata = tif.imagej_metadata or {}
        raw_unit = str(metadata.get("unit", "pixel")).strip() or "pixel"
        page = tif.pages[0]
        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")
        resolution_unit_tag = page.tags.get("ResolutionUnit")
        def resolution_value(tag) -> float:
            if tag is None:
                return 1.0
            value = tag.value
            if isinstance(value, tuple) and len(value) == 2:
                return float(value[0]) / float(value[1])
            return float(value)
        x_resolution = resolution_value(x_tag)
        y_resolution = resolution_value(y_tag)
        resolution_unit = (
            int(resolution_unit_tag.value) if resolution_unit_tag is not None else 1)
    if raw_unit.lower() in {"pixel", "pixels", "px"}:
        if resolution_unit == 2:
            raw_unit, scale = "µm", 25400.0
        elif resolution_unit == 3:
            raw_unit, scale = "µm", 10000.0
        else:
            return {
                "axes": "YX", "unit": "pixel",
                "pixel_size_y": 1.0, "pixel_size_x": 1.0,
                "source": "TIFF has no physical calibration",
            }
        size_x = scale / x_resolution
        size_y = scale / y_resolution
    else:
        normalized_units = {
            "micron": "µm", "microns": "µm", "um": "µm", "µm": "µm",
        }
        raw_unit = normalized_units.get(raw_unit.lower(), raw_unit)
        size_x = 1.0 / x_resolution
        size_y = 1.0 / y_resolution
    if any(not np.isfinite(value) or value <= 0 for value in (size_y, size_x)):
        raise ValueError("TIFF spatial calibration is invalid")
    return {
        "axes": "YX", "unit": raw_unit,
        "pixel_size_y": float(size_y), "pixel_size_x": float(size_x),
        "source": "registered raw TIFF metadata",
    }


def prepare_editing_bundle(
        output_dir: Path, stem: str, canonical_labels: Path,
        registered_raw: Path, motion_composite: Path,
        label_frame_to_raw: list[int], frame_interval_min: float,
        unclaimed_labels: Path | None = None, lag_float: Path | None = None,
        parent_manifest: Path | None = None,
        parent: dict[str, Any] | None = None,
        spatial_unit: str | None = None,
        pixel_size_y: float | None = None,
        pixel_size_x: float | None = None) -> Path:
    """Create one immutable, validated input bundle for the manual editor."""
    canonical_labels = canonical_labels.resolve()
    registered_raw = registered_raw.resolve()
    motion_composite = motion_composite.resolve()
    unclaimed_labels = (None if unclaimed_labels is None
                        else unclaimed_labels.resolve())
    lag_float = None if lag_float is None else lag_float.resolve()
    parent_manifest = (None if parent_manifest is None
                       else parent_manifest.resolve())
    for path in (canonical_labels, registered_raw, motion_composite,
                 unclaimed_labels, lag_float, parent_manifest):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    labels = _read_tiff(canonical_labels)
    validate_labels(labels)
    raw_summary = _tiff_summary(registered_raw)
    motion_summary = _tiff_summary(motion_composite)
    _validate_spatial_shape(raw_summary["shape"], labels.shape, "registered raw")
    _validate_spatial_shape(
        motion_summary["shape"], labels.shape, "motion composite")
    _validate_source_indices(
        label_frame_to_raw, len(labels), raw_summary["shape"][0],
        "registered raw")
    motion_indices = [
        min(value, motion_summary["shape"][0] - 1)
        for value in label_frame_to_raw]
    _validate_source_indices(
        motion_indices, len(labels), motion_summary["shape"][0],
        "motion composite")

    unclaimed: np.ndarray | None = None
    if unclaimed_labels is not None:
        unclaimed = _read_tiff(unclaimed_labels)
        validate_labels(unclaimed)
        if unclaimed.shape != labels.shape:
            raise ValueError("unclaimed labels and canonical labels differ in shape")
        if np.any((labels > 0) & (unclaimed > 0)):
            raise ValueError("canonical and unclaimed labels overlap")

    lag_summary: dict[str, Any] | None = None
    lag_indices: list[int] | None = None
    if lag_float is not None:
        lag_summary = _tiff_summary(lag_float)
        _validate_spatial_shape(lag_summary["shape"], labels.shape, "lag evidence")
        lag_indices = [min(value, lag_summary["shape"][0] - 1)
                       for value in label_frame_to_raw]
        _validate_source_indices(
            lag_indices, len(labels), lag_summary["shape"][0], "lag evidence")

    if not stem:
        raise ValueError("stem cannot be empty")
    if not np.isfinite(frame_interval_min) or frame_interval_min <= 0:
        raise ValueError("frame_interval_min must be positive")
    calibration = _spatial_calibration(
        registered_raw, spatial_unit, pixel_size_y, pixel_size_x)

    output = _create_output(output_dir)
    try:
        bundled_labels = _copy_file(
            canonical_labels, output / "inputs" / canonical_labels.name)
        bundled_unclaimed = None
        if unclaimed_labels is not None:
            bundled_unclaimed = _copy_file(
                unclaimed_labels, output / "inputs" / unclaimed_labels.name)
        bundled_parent = None
        if parent_manifest is not None:
            bundled_parent = _copy_file(
                parent_manifest, output / "provenance" / parent_manifest.name)

        assets: dict[str, dict[str, Any]] = {
            "canonical_labels": _tiff_record(
                bundled_labels, output, True, "TYX"),
            "registered_raw": _tiff_record(
                registered_raw, output, False, "TYX",
                label_frame_to_raw),
            "motion_composite": _tiff_record(
                motion_composite, output, False, "TCYX",
                motion_indices),
        }
        if bundled_unclaimed is not None:
            assets["unclaimed_labels"] = _tiff_record(
                bundled_unclaimed, output, True, "TYX")
        if lag_float is not None and lag_indices is not None:
            assets["lag_float"] = _tiff_record(
                lag_float, output, False, "TYX", lag_indices)

        parent_record = dict(parent or {})
        if bundled_parent is not None:
            parent_record["manifest"] = _path_record(
                bundled_parent, output, True)

        manifest = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "status": "done",
            "created_at_utc": utc_now(),
            "stem": stem,
            "parent": parent_record,
            "assets": assets,
            "time": {
                "frame_interval_min": float(frame_interval_min),
                "label_frames": int(len(labels)),
                "mapping_rule": (
                    "each asset records its zero-based source index for every "
                    "zero-based label frame"),
            },
            "spatial_calibration": calibration,
            "manual_provenance_values": MANUAL_PROVENANCE_VALUES,
            "software": {
                "python": platform.python_version(),
                "manual_editing_schema_version": BUNDLE_SCHEMA_VERSION,
                "manual_editing_module_sha256": sha256(Path(__file__).resolve()),
                "pipeline_module_sha256": sha256(ROOT / "code" / "pipeline.py"),
            },
            "summary": {
                "shape": [int(value) for value in labels.shape],
                "dtype": labels.dtype.name,
                "active_identities": int(len(set(map(int, np.unique(labels))) - {0})),
                "assigned_pixels": int(np.count_nonzero(labels)),
                "unclaimed_pixels": (None if unclaimed is None
                                     else int(np.count_nonzero(unclaimed))),
            },
        }
        manifest_path = output / BUNDLE_FILENAME
        write_json(manifest_path, manifest)
        load_editing_bundle(manifest_path)
    except BaseException as error:
        _record_failed_output(output, error)
        raise
    final_manifest = output / BUNDLE_FILENAME
    load_editing_bundle(final_manifest)
    return final_manifest


def _pinned_paths(cfg: Config, stem: str) -> dict[str, Path]:
    if stem not in cfg.stems:
        raise ValueError(f"stem {stem!r} is not configured: {cfg.stems}")
    registered = cfg.resolve("registered_input_dir")
    motion = cfg.resolve("motion_input_dir")
    result: dict[str, Path] = {}
    for name, row in cfg.values["pinned_files"][stem].items():
        if "relative_to_registered_input_dir" in row:
            result[name] = registered / row["relative_to_registered_input_dir"]
        elif "relative_to_motion_input_dir" in row:
            result[name] = motion / row["relative_to_motion_input_dir"]
        else:
            raise ValueError(f"pinned file {name!r} has no input-directory path")
    return result


def prepare_accepted_editing_bundle(
        output_dir: Path, accepted_base_path: Path | None = None,
        config_path: Path | None = None, stem: str | None = None,
        spatial_unit: str | None = None,
        pixel_size_y: float | None = None,
        pixel_size_x: float | None = None) -> Path:
    """Resolve the current accepted result and prepare it for manual editing."""
    accepted_path = (accepted_base_path or ROOT / "accepted_base.json").resolve()
    accepted = _json_object(accepted_path)
    labels_row = accepted.get("labels")
    if not isinstance(labels_row, dict) or not isinstance(labels_row.get("path"), str):
        raise ValueError("accepted base has no labels.path")
    labels_path = (accepted_path.parent / labels_row["path"]).resolve()
    if sha256(labels_path) != labels_row.get("sha256"):
        raise ValueError("accepted label fingerprint differs")

    selected_stem = stem or labels_path.stem
    cfg = Config.load(config_path)
    sources = _pinned_paths(cfg, selected_stem)
    for name in ("registered_raw", "motion_composite", "lag_float"):
        path = sources.get(name)
        if path is None or not path.is_file():
            raise FileNotFoundError(path or name)
        expected = cfg.values["pinned_files"][selected_stem][name]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"configured {name} fingerprint differs")

    raw_frames = _tiff_summary(sources["registered_raw"])["shape"][0]
    removed = accepted.get("source_frames_removed", [])
    if not isinstance(removed, list) or any(
            not isinstance(value, int) for value in removed):
        raise ValueError("source_frames_removed must be a list of ImageJ frame numbers")
    removed_zero_based = {value - 1 for value in removed}
    if any(value < 0 or value >= raw_frames for value in removed_zero_based):
        raise ValueError("source_frames_removed contains an unavailable frame")
    raw_indices = [index for index in range(raw_frames)
                   if index not in removed_zero_based]
    label_frames = _tiff_summary(labels_path)["shape"][0]
    if len(raw_indices) != label_frames:
        raise ValueError(
            "accepted source-frame removal does not explain the label frame count")

    unclaimed_path = None
    unclaimed_row = accepted.get("unclaimed")
    if isinstance(unclaimed_row, dict) and isinstance(unclaimed_row.get("path"), str):
        unclaimed_path = (accepted_path.parent / unclaimed_row["path"]).resolve()
        if sha256(unclaimed_path) != unclaimed_row.get("sha256"):
            raise ValueError("accepted unclaimed-label fingerprint differs")

    configured_calibration = cfg.values.get("spatial_calibration", {})
    if not any(value is not None for value in (
            spatial_unit, pixel_size_y, pixel_size_x)) \
            and isinstance(configured_calibration, dict) \
            and configured_calibration:
        spatial_unit = configured_calibration.get("unit")
        pixel_size_y = configured_calibration.get("pixel_size_y")
        pixel_size_x = configured_calibration.get("pixel_size_x")

    return prepare_editing_bundle(
        output_dir=output_dir,
        stem=selected_stem,
        canonical_labels=labels_path,
        registered_raw=sources["registered_raw"],
        motion_composite=sources["motion_composite"],
        label_frame_to_raw=raw_indices,
        frame_interval_min=float(cfg.values["frame_interval_min"]),
        unclaimed_labels=unclaimed_path,
        lag_float=sources["lag_float"],
        parent_manifest=accepted_path,
        parent={
            "kind": "accepted_base",
            "accepted_run": accepted.get("accepted_run"),
            "final_stage": accepted.get("final_stage"),
            "accepted_on": accepted.get("accepted_on"),
        },
        spatial_unit=spatial_unit,
        pixel_size_y=pixel_size_y,
        pixel_size_x=pixel_size_x,
    )


def load_editing_bundle(path: Path, verify_hashes: bool = True) -> EditingBundle:
    """Load and strictly validate a versioned manual-editing input bundle."""
    manifest_path = _editing_manifest_path(path)
    if manifest_path.name != BUNDLE_FILENAME:
        document = _json_object(manifest_path)
        if document.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("portable package entry is not an editing bundle")
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported bundle schema: {manifest.get('schema')!r}")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema version: {manifest.get('schema_version')!r}")
    if manifest.get("status") != "done":
        raise ValueError("editing bundle is not complete")
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("editing bundle has no assets object")
    for name in ("canonical_labels", "registered_raw", "motion_composite"):
        if not isinstance(assets.get(name), dict):
            raise ValueError(f"editing bundle has no {name} asset")

    label_record = assets["canonical_labels"]
    if label_record.get("semantic_axes") != "TYX":
        raise ValueError("canonical labels must declare TYX semantic axes")
    label_path = _validate_tiff_record(
        manifest_path, label_record, verify_hashes)
    labels = _read_tiff(label_path)
    validate_labels(labels)
    if list(labels.shape) != label_record.get("shape"):
        raise ValueError("canonical label shape differs after loading")

    raw_record = assets["registered_raw"]
    if raw_record.get("semantic_axes") != "TYX":
        raise ValueError("registered raw must declare TYX semantic axes")
    raw_path = _validate_tiff_record(manifest_path, raw_record, verify_hashes)
    raw_indices = raw_record.get("label_frame_to_source")
    if not isinstance(raw_indices, list):
        raise ValueError("registered raw has no label-frame mapping")
    raw_shape = raw_record.get("shape")
    _validate_spatial_shape(raw_shape, labels.shape, "registered raw")
    _validate_source_indices(
        raw_indices, len(labels), raw_shape[0], "registered raw")
    raw_source = _read_tiff(raw_path)
    registered_raw = raw_source[np.asarray(raw_indices, dtype=np.intp)]
    if registered_raw.shape != labels.shape:
        raise ValueError("aligned registered raw and canonical labels differ in shape")

    motion_record = assets["motion_composite"]
    if motion_record.get("semantic_axes") != "TCYX":
        raise ValueError("motion composite must declare TCYX semantic axes")
    motion_path = _validate_tiff_record(
        manifest_path, motion_record, verify_hashes)
    motion_indices = motion_record.get("label_frame_to_source")
    motion_shape = motion_record.get("shape")
    if not isinstance(motion_indices, list):
        raise ValueError("motion composite has no label-frame mapping")
    _validate_spatial_shape(motion_shape, labels.shape, "motion composite")
    _validate_source_indices(
        motion_indices, len(labels), motion_shape[0], "motion composite")

    evidence_paths = {"motion_composite": motion_path}
    lag_record = assets.get("lag_float")
    if lag_record is not None:
        if not isinstance(lag_record, dict) or lag_record.get("semantic_axes") != "TYX":
            raise ValueError("lag evidence must declare TYX semantic axes")
        lag_path = _validate_tiff_record(
            manifest_path, lag_record, verify_hashes)
        lag_indices = lag_record.get("label_frame_to_source")
        lag_shape = lag_record.get("shape")
        if not isinstance(lag_indices, list):
            raise ValueError("lag evidence has no label-frame mapping")
        _validate_spatial_shape(lag_shape, labels.shape, "lag evidence")
        _validate_source_indices(
            lag_indices, len(labels), lag_shape[0], "lag evidence")
        evidence_paths["lag_float"] = lag_path

    unclaimed = None
    unclaimed_record = assets.get("unclaimed_labels")
    if unclaimed_record is not None:
        if not isinstance(unclaimed_record, dict) \
                or unclaimed_record.get("semantic_axes") != "TYX":
            raise ValueError("unclaimed labels must declare TYX semantic axes")
        unclaimed_path = _validate_tiff_record(
            manifest_path, unclaimed_record, verify_hashes)
        unclaimed = _read_tiff(unclaimed_path)
        validate_labels(unclaimed)
        if unclaimed.shape != labels.shape:
            raise ValueError("unclaimed and canonical labels differ in shape")
        if np.any((labels > 0) & (unclaimed > 0)):
            raise ValueError("canonical and unclaimed labels overlap")

    time = manifest.get("time")
    if not isinstance(time, dict) or time.get("label_frames") != len(labels):
        raise ValueError("bundle time metadata differs from canonical labels")
    interval = time.get("frame_interval_min")
    if not isinstance(interval, (int, float)) or not np.isfinite(interval) \
            or interval <= 0:
        raise ValueError("bundle frame interval must be positive")

    calibration = manifest.get("spatial_calibration")
    if not isinstance(calibration, dict) or calibration.get("axes") != "YX" \
            or not isinstance(calibration.get("unit"), str) \
            or not calibration["unit"].strip():
        raise ValueError("bundle spatial calibration is invalid")
    for field in ("pixel_size_y", "pixel_size_x"):
        value = calibration.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not np.isfinite(value) or value <= 0:
            raise ValueError(f"bundle spatial calibration {field} is invalid")

    parent = manifest.get("parent", {})
    if not isinstance(parent, dict):
        raise ValueError("bundle parent must be an object")
    parent_manifest = parent.get("manifest")
    if parent_manifest is not None:
        if not isinstance(parent_manifest, dict):
            raise ValueError("parent manifest file record must be an object")
        _validate_file_record(manifest_path, parent_manifest, verify_hashes)

    return EditingBundle(
        manifest_path=manifest_path,
        manifest=manifest,
        canonical_labels=labels,
        registered_raw=registered_raw,
        registered_raw_indices=np.asarray(raw_indices, dtype=np.intp),
        unclaimed_labels=unclaimed,
        evidence_paths=evidence_paths,
    )


def rebuild_identity_tables(
        labels: np.ndarray, raw: np.ndarray, raw_indices: np.ndarray,
        manual_provenance: np.ndarray | None = None,
        excluded_identities: set[int] | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild derived views from labels; neither CSV is an editing authority."""
    validate_labels(labels)
    if raw.shape != labels.shape:
        raise ValueError("registered raw and labels differ in shape")
    if len(raw_indices) != len(labels):
        raise ValueError("registered raw frame mapping and labels differ in length")
    provenance = (np.zeros(labels.shape, np.uint8) if manual_provenance is None
                  else np.asarray(manual_provenance))
    if provenance.shape != labels.shape:
        raise ValueError("manual provenance and labels differ in shape")
    if not np.issubdtype(provenance.dtype, np.integer) \
            or np.any((provenance < 0) | (provenance > 5)):
        raise ValueError("manual provenance values must be between 0 and 5")
    excluded = set() if excluded_identities is None else {
        int(value) for value in excluded_identities}

    frame_columns = [
        "t", "imagej_frame", "source_imagej_frame", "identity", "y", "x",
        "area_px", "mean_intensity", "manual_pixels",
        "centroid_assisted_pixels", "excluded",
    ]
    rows: list[dict[str, Any]] = []
    for t in range(len(labels)):
        frame = labels[t]
        y, x = np.nonzero(frame)
        if not len(y):
            continue
        values = frame[y, x]
        identities, inverse, counts = np.unique(
            values, return_inverse=True, return_counts=True)
        y_sums = np.bincount(inverse, weights=y)
        x_sums = np.bincount(inverse, weights=x)
        intensity_sums = np.bincount(inverse, weights=raw[t, y, x])
        provenance_values = provenance[t, y, x]
        drawn_counts = np.bincount(
            inverse, weights=np.isin(
                provenance_values, [1, 3, 4]).astype(np.uint8),
            minlength=len(identities))
        assisted_counts = np.bincount(
            inverse, weights=np.isin(
                provenance_values, [2, 5]).astype(np.uint8),
            minlength=len(identities))
        for index, identity_value in enumerate(identities):
            identity = int(identity_value)
            count = int(counts[index])
            rows.append({
                "t": int(t),
                "imagej_frame": int(t + 1),
                "source_imagej_frame": int(raw_indices[t] + 1),
                "identity": identity,
                "y": float(y_sums[index] / count),
                "x": float(x_sums[index] / count),
                "area_px": count,
                "mean_intensity": float(intensity_sums[index] / count),
                "manual_pixels": int(drawn_counts[index]),
                "centroid_assisted_pixels": int(assisted_counts[index]),
                "excluded": bool(identity in excluded),
            })
    frame_table = pd.DataFrame(rows, columns=frame_columns)

    track_columns = [
        "identity", "first_imagej_frame", "last_imagej_frame",
        "first_source_imagej_frame", "last_source_imagej_frame",
        "observed_frames", "manual_frames", "centroid_assisted_frames",
        "median_area_px", "median_mean_intensity", "excluded",
    ]
    track_rows: list[dict[str, Any]] = []
    for identity, group in frame_table.groupby("identity", sort=True):
        track_rows.append({
            "identity": int(identity),
            "first_imagej_frame": int(group.imagej_frame.min()),
            "last_imagej_frame": int(group.imagej_frame.max()),
            "first_source_imagej_frame": int(group.source_imagej_frame.min()),
            "last_source_imagej_frame": int(group.source_imagej_frame.max()),
            "observed_frames": int(group.imagej_frame.nunique()),
            "manual_frames": int((group.manual_pixels > 0).sum()),
            "centroid_assisted_frames": int(
                (group.centroid_assisted_pixels > 0).sum()),
            "median_area_px": float(group.area_px.median()),
            "median_mean_intensity": float(group.mean_intensity.median()),
            "excluded": bool(identity in excluded),
        })
    return frame_table, pd.DataFrame(track_rows, columns=track_columns)


def _assert_table_matches(actual: pd.DataFrame, expected: pd.DataFrame,
                          label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    except AssertionError as error:
        raise ValueError(f"stored {label} table differs from curated labels") from error


def _required_operation_integer(operation: dict[str, Any], key: str,
                                minimum: int = 1) -> int:
    value = operation.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"edit operation {key} must be an integer >= {minimum}")
    return value


def _swap_identities_in_place(labels: np.ndarray, identity_a: int,
                              identity_b: int, start_frame: int,
                              end_frame: int) -> dict[str, Any]:
    if identity_a <= 0 or identity_b <= 0:
        raise ValueError("swapped identities must be positive")
    if identity_a == identity_b:
        raise ValueError("identity swap requires two different identities")
    if start_frame < 1 or end_frame < start_frame or end_frame > len(labels):
        raise ValueError(
            f"identity-swap frame range must be within 1-{len(labels)}")

    a_total = 0
    b_total = 0
    affected: list[int] = []
    per_frame: list[dict[str, int]] = []
    for imagej_frame in range(start_frame, end_frame + 1):
        frame = labels[imagej_frame - 1]
        a_mask = frame == identity_a
        b_mask = frame == identity_b
        a_pixels = int(np.count_nonzero(a_mask))
        b_pixels = int(np.count_nonzero(b_mask))
        if a_pixels or b_pixels:
            affected.append(imagej_frame)
            per_frame.append({
                "imagej_frame": imagej_frame,
                "identity_a_to_b_pixels": a_pixels,
                "identity_b_to_a_pixels": b_pixels,
            })
        frame[a_mask] = identity_b
        frame[b_mask] = identity_a
        a_total += a_pixels
        b_total += b_pixels
    if not a_total:
        raise ValueError(
            f"identity {identity_a} is absent from the selected frame range")
    if not b_total:
        raise ValueError(
            f"identity {identity_b} is absent from the selected frame range")
    return {
        "affected_imagej_frames": affected,
        "identity_a_to_b_pixels": a_total,
        "identity_b_to_a_pixels": b_total,
        "changed_pixels": a_total + b_total,
        "per_frame": per_frame,
    }


def _reassign_identities_in_place(
        labels: np.ndarray, source_identities: list[int], target_identity: int,
        start_frame: int, end_frame: int) -> dict[str, Any]:
    if not source_identities or any(identity <= 0 for identity in source_identities):
        raise ValueError("reassigned source identities must be positive")
    if len(set(source_identities)) != len(source_identities):
        raise ValueError("reassigned source identities must not contain duplicates")
    if target_identity <= 0:
        raise ValueError("reassignment target identity must be positive")
    if target_identity in source_identities:
        raise ValueError("reassignment target must differ from every source identity")
    if start_frame < 1 or end_frame < start_frame or end_frame > len(labels):
        raise ValueError(
            f"identity-reassignment frame range must be within 1-{len(labels)}")

    if not np.any(labels == target_identity):
        raise ValueError(
            f"target identity {target_identity} is absent from the current labels")
    selected = labels[start_frame - 1:end_frame]
    absent = [
        identity for identity in source_identities
        if not np.any(selected == identity)
    ]
    if absent:
        joined = ", ".join(str(identity) for identity in absent)
        raise ValueError(
            f"source identity values absent from the selected frame range: {joined}")

    per_source = {identity: 0 for identity in source_identities}
    affected: list[int] = []
    per_frame: list[dict[str, Any]] = []
    for imagej_frame in range(start_frame, end_frame + 1):
        frame = labels[imagej_frame - 1]
        source_counts = []
        frame_total = 0
        for identity in source_identities:
            mask = frame == identity
            count = int(np.count_nonzero(mask))
            if count:
                frame[mask] = target_identity
                per_source[identity] += count
                frame_total += count
            source_counts.append({
                "source_identity": identity,
                "reassigned_pixels": count,
            })
        if frame_total:
            affected.append(imagej_frame)
            per_frame.append({
                "imagej_frame": imagej_frame,
                "reassigned_pixels": frame_total,
                "per_source": source_counts,
            })

    changed_pixels = sum(per_source.values())
    return {
        "affected_imagej_frames": affected,
        "reassigned_pixels": changed_pixels,
        "changed_pixels": changed_pixels,
        "per_source": [
            {"source_identity": identity,
             "reassigned_pixels": per_source[identity]}
            for identity in source_identities
        ],
        "per_frame": per_frame,
    }


def _exclude_identity_in_place(
        labels: np.ndarray, provenance: np.ndarray, excluded: set[int],
        identity: int) -> dict[str, Any]:
    if identity <= 0:
        raise ValueError("removed identity must be positive")
    if identity in excluded:
        raise ValueError(f"identity {identity} is already removed")
    affected: list[int] = []
    per_frame: list[dict[str, int]] = []
    removed_pixels = 0
    for t, frame in enumerate(labels):
        mask = frame == identity
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        imagej_frame = t + 1
        affected.append(imagej_frame)
        per_frame.append({
            "imagej_frame": imagej_frame,
            "removed_pixels": count,
        })
        frame[mask] = 0
        provenance[t][mask] = 0
        removed_pixels += count
    if not removed_pixels:
        raise ValueError(f"identity {identity} is absent from the current labels")
    excluded.add(identity)
    return {
        "affected_imagej_frames": affected,
        "removed_pixels": removed_pixels,
        "changed_pixels": removed_pixels,
        "per_frame": per_frame,
    }


def _expand_identities_in_place(
        bundle: EditingBundle, labels: np.ndarray, provenance: np.ndarray,
        identities: list[int], radius: float, radius_unit: str,
        start_frame: int, end_frame: int) -> dict[str, Any]:
    if not identities or any(identity <= 0 for identity in identities):
        raise ValueError("expanded identities must be positive")
    if len(set(identities)) != len(identities):
        raise ValueError("expanded identities must not contain duplicates")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("expansion radius must be a positive finite number")
    if radius_unit not in {"calibrated", "pixel"}:
        raise ValueError("expansion radius unit must be 'calibrated' or 'pixel'")
    if start_frame < 1 or end_frame < start_frame or end_frame > len(labels):
        raise ValueError(
            f"identity-expansion frame range must be within 1-{len(labels)}")

    calibration = bundle.manifest["spatial_calibration"]
    if radius_unit == "calibrated":
        sampling = (
            float(calibration["pixel_size_y"]),
            float(calibration["pixel_size_x"]),
        )
        effective_unit = str(calibration["unit"])
    else:
        sampling = (1.0, 1.0)
        effective_unit = "pixel"

    selected = labels[start_frame - 1:end_frame]
    absent = [
        identity for identity in identities
        if not np.any(selected == identity)
    ]
    if absent:
        joined = ", ".join(str(identity) for identity in absent)
        raise ValueError(
            f"expanded identity values absent from the selected frame range: {joined}")

    per_identity = {identity: 0 for identity in identities}
    affected: list[int] = []
    per_frame: list[dict[str, Any]] = []
    for imagej_frame in range(start_frame, end_frame + 1):
        frame = labels[imagej_frame - 1]
        present = [identity for identity in identities if np.any(frame == identity)]
        if not present or not np.any(frame == 0):
            continue
        best_distance = np.full(frame.shape, np.inf, dtype=float)
        best_identity = np.zeros(frame.shape, dtype=frame.dtype)
        tied = np.zeros(frame.shape, dtype=bool)
        for identity in present:
            identity_y, identity_x = np.nonzero(frame == identity)
            margin_y = int(np.ceil(radius / sampling[0]))
            margin_x = int(np.ceil(radius / sampling[1]))
            y0 = max(0, int(identity_y.min()) - margin_y)
            y1 = min(frame.shape[0], int(identity_y.max()) + margin_y + 1)
            x0 = max(0, int(identity_x.min()) - margin_x)
            x1 = min(frame.shape[1], int(identity_x.max()) + margin_x + 1)
            region = np.s_[y0:y1, x0:x1]
            distance = ndi.distance_transform_edt(
                frame[region] != identity, sampling=sampling)
            region_best_distance = best_distance[region]
            region_best_identity = best_identity[region]
            region_tied = tied[region]
            equal = np.isclose(
                distance, region_best_distance, rtol=1e-10, atol=1e-12)
            closer = (distance < region_best_distance) & ~equal
            region_best_distance[closer] = distance[closer]
            region_best_identity[closer] = identity
            region_tied[closer] = False
            region_tied[equal] = True
        additions = (frame == 0) & (best_distance <= radius) & ~tied
        if not np.any(additions):
            continue
        identity_rows: list[dict[str, int]] = []
        for identity in identities:
            identity_mask = additions & (best_identity == identity)
            count = int(np.count_nonzero(identity_mask))
            if count:
                frame[identity_mask] = identity
                provenance[imagej_frame - 1][identity_mask] = 1
                per_identity[identity] += count
                identity_rows.append({
                    "identity": identity,
                    "added_pixels": count,
                })
        added = sum(row["added_pixels"] for row in identity_rows)
        affected.append(imagej_frame)
        per_frame.append({
            "imagej_frame": imagej_frame,
            "added_pixels": added,
            "per_identity": identity_rows,
        })
    added_pixels = sum(per_identity.values())
    if not added_pixels:
        raise ValueError(
            "identity expansion added no background pixels in the selected frame range")
    return {
        "effective_distance_unit": effective_unit,
        "affected_imagej_frames": affected,
        "added_pixels": added_pixels,
        "changed_pixels": added_pixels,
        "per_identity": [
            {"identity": identity, "added_pixels": per_identity[identity]}
            for identity in identities
        ],
        "per_frame": per_frame,
    }


def _delete_identity_interval_in_place(
        labels: np.ndarray, provenance: np.ndarray, identities: list[int],
        start_frame: int, end_frame: int) -> dict[str, Any]:
    if not identities or any(identity <= 0 for identity in identities):
        raise ValueError("interval-deleted identities must be positive")
    if len(set(identities)) != len(identities):
        raise ValueError("interval-deleted identities must not contain duplicates")
    if start_frame < 1 or end_frame < start_frame or end_frame > len(labels):
        raise ValueError(
            f"identity-deletion frame range must be within 1-{len(labels)}")
    absent = [
        identity for identity in identities
        if not np.any(labels == identity)
    ]
    if absent:
        joined = ", ".join(str(identity) for identity in absent)
        raise ValueError(
            f"interval-deleted identity values absent from the current labels: "
            f"{joined}")

    per_identity = {identity: 0 for identity in identities}
    affected: list[int] = []
    per_frame: list[dict[str, Any]] = []
    for imagej_frame in range(start_frame, end_frame + 1):
        frame = labels[imagej_frame - 1]
        identity_rows: list[dict[str, int]] = []
        for identity in identities:
            mask = frame == identity
            count = int(np.count_nonzero(mask))
            if not count:
                continue
            frame[mask] = 0
            provenance[imagej_frame - 1][mask] = 0
            per_identity[identity] += count
            identity_rows.append({
                "identity": identity,
                "deleted_pixels": count,
            })
        if identity_rows:
            deleted = sum(row["deleted_pixels"] for row in identity_rows)
            affected.append(imagej_frame)
            per_frame.append({
                "imagej_frame": imagej_frame,
                "deleted_pixels": deleted,
                "per_identity": identity_rows,
            })
    deleted_pixels = sum(per_identity.values())
    if not deleted_pixels:
        raise ValueError(
            "frame-range deletion found no selected identity pixels in the interval")
    return {
        "affected_imagej_frames": affected,
        "deleted_pixels": deleted_pixels,
        "changed_pixels": deleted_pixels,
        "per_identity": [
            {"identity": identity, "deleted_pixels": per_identity[identity]}
            for identity in identities
        ],
        "per_frame": per_frame,
    }


def _request_integer(request: dict[str, Any], key: str) -> int:
    value = request.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"batch operation {key} must be an integer")
    return value


def _request_positive_number(request: dict[str, Any], key: str) -> float:
    value = request.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(f"batch operation {key} must be a positive finite number")
    return float(value)


def _request_identities(request: dict[str, Any]) -> list[int]:
    values = request.get("identities")
    if not isinstance(values, list) or not values:
        raise ValueError("batch operation identities must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise ValueError("batch operation identities must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("batch operation identities must not contain duplicates")
    return sorted(values)


def _declared_operation_integer(operation: dict[str, Any], key: str) -> int:
    value = operation.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"edit operation {key} must be an integer")
    return value


def _declared_operation_positive_number(
        operation: dict[str, Any], key: str) -> float:
    value = operation.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"edit operation {key} must be a positive finite number")
    return float(value)


def _normalize_operation_request(
        request: dict[str, Any], total_frames: int) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("every batch operation must be an object")
    operation_type = request.get("type")
    common_fields = {"type", "user"}
    if operation_type == "swap_identities":
        allowed = common_fields | {
            "identity_a", "identity_b", "start_imagej_frame",
            "end_imagej_frame"}
        normalized = {
            "type": operation_type,
            "identity_a": _request_integer(request, "identity_a"),
            "identity_b": _request_integer(request, "identity_b"),
            "start_imagej_frame": _request_integer(
                request, "start_imagej_frame"),
            "end_imagej_frame": (
                total_frames if request.get("end_imagej_frame") is None
                else _request_integer(request, "end_imagej_frame")),
        }
    elif operation_type == "reassign_identities":
        allowed = common_fields | {
            "source_identities", "target_identity", "start_imagej_frame",
            "end_imagej_frame"}
        source_request = {
            "identities": request.get("source_identities")}
        normalized = {
            "type": operation_type,
            "source_identities": _request_identities(source_request),
            "target_identity": _request_integer(request, "target_identity"),
            "start_imagej_frame": _request_integer(
                request, "start_imagej_frame"),
            "end_imagej_frame": (
                total_frames if request.get("end_imagej_frame") is None
                else _request_integer(request, "end_imagej_frame")),
        }
    elif operation_type == "exclude_identity":
        allowed = common_fields | {"identity", "reason"}
        normalized = {
            "type": operation_type,
            "identity": _request_integer(request, "identity"),
        }
        if "reason" in request:
            reason = request["reason"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("batch operation reason must be non-empty text")
            normalized["reason"] = reason.strip()
    elif operation_type == "expand_identities":
        allowed = common_fields | {
            "identities", "radius", "radius_unit", "start_imagej_frame",
            "end_imagej_frame"}
        radius_unit = request.get("radius_unit", "calibrated")
        if radius_unit not in {"calibrated", "pixel"}:
            raise ValueError(
                "batch operation radius_unit must be 'calibrated' or 'pixel'")
        normalized = {
            "type": operation_type,
            "identities": _request_identities(request),
            "radius": _request_positive_number(request, "radius"),
            "radius_unit": radius_unit,
            "start_imagej_frame": (
                1 if request.get("start_imagej_frame") is None
                else _request_integer(request, "start_imagej_frame")),
            "end_imagej_frame": (
                total_frames if request.get("end_imagej_frame") is None
                else _request_integer(request, "end_imagej_frame")),
        }
    elif operation_type == "delete_identity_interval":
        allowed = common_fields | {
            "identities", "start_imagej_frame", "end_imagej_frame"}
        normalized = {
            "type": operation_type,
            "identities": _request_identities(request),
            "start_imagej_frame": _request_integer(
                request, "start_imagej_frame"),
            "end_imagej_frame": _request_integer(
                request, "end_imagej_frame"),
        }
    elif operation_type == "force_split_identity_interval":
        from manual_splitting import normalize_force_split_request

        return normalize_force_split_request(request, total_frames)
    elif operation_type == "add_identity_track":
        from manual_new_cells import normalize_add_identity_request

        return normalize_add_identity_request(request, total_frames)
    elif operation_type == "retune_masks_after_edit":
        from manual_retuning import normalize_retune_request

        return normalize_retune_request(request, total_frames)
    else:
        raise ValueError(f"unsupported batch operation: {operation_type!r}")
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported fields for {operation_type}: {', '.join(unknown)}")
    if "user" in request:
        normalized["user"] = request["user"]
    return normalized


def _apply_operation_to_state(
        bundle: EditingBundle, labels: np.ndarray, provenance: np.ndarray,
        excluded: set[int], operation: dict[str, Any],
        released_background: np.ndarray | None = None,
        ) -> tuple[dict[str, Any], str]:
    operation_type = operation.get("type")
    if operation_type == "swap_identities":
        details = _swap_identities_in_place(
            labels,
            _declared_operation_integer(operation, "identity_a"),
            _declared_operation_integer(operation, "identity_b"),
            _declared_operation_integer(operation, "start_imagej_frame"),
            _declared_operation_integer(operation, "end_imagej_frame"),
        )
        return details, "identity-swap"
    if operation_type == "reassign_identities":
        source_identities = operation.get("source_identities")
        if not isinstance(source_identities, list):
            raise ValueError("edit operation source_identities must be a list")
        details = _reassign_identities_in_place(
            labels,
            [_declared_operation_integer(
                {"identity": identity}, "identity")
             for identity in source_identities],
            _declared_operation_integer(operation, "target_identity"),
            _declared_operation_integer(operation, "start_imagej_frame"),
            _declared_operation_integer(operation, "end_imagej_frame"),
        )
        return details, "identity-reassignment"
    if operation_type == "exclude_identity":
        details = _exclude_identity_in_place(
            labels, provenance, excluded,
            _declared_operation_integer(operation, "identity"))
        return details, "identity-removal"
    if operation_type == "expand_identities":
        identities = operation.get("identities")
        if not isinstance(identities, list):
            raise ValueError("edit operation identities must be a list")
        details = _expand_identities_in_place(
            bundle, labels, provenance,
            [_declared_operation_integer(
                {"identity": identity}, "identity") for identity in identities],
            _declared_operation_positive_number(operation, "radius"),
            str(operation.get("radius_unit", "")),
            _declared_operation_integer(operation, "start_imagej_frame"),
            _declared_operation_integer(operation, "end_imagej_frame"),
        )
        return details, "identity-expansion"
    if operation_type == "delete_identity_interval":
        identities = operation.get("identities")
        if not isinstance(identities, list):
            raise ValueError("edit operation identities must be a list")
        details = _delete_identity_interval_in_place(
            labels, provenance,
            [_declared_operation_integer(
                {"identity": identity}, "identity") for identity in identities],
            _declared_operation_integer(operation, "start_imagej_frame"),
            _declared_operation_integer(operation, "end_imagej_frame"),
        )
        return details, "identity-interval-deletion"
    if operation_type == "force_split_identity_interval":
        from manual_splitting import apply_force_split_operation

        details = apply_force_split_operation(
            labels, provenance, operation, excluded)
        return details, "forced-identity-split"
    if operation_type == "add_identity_track":
        from manual_new_cells import apply_add_identity_operation

        details = apply_add_identity_operation(
            labels, provenance, operation, excluded)
        return details, "new-identity-track"
    if operation_type == "retune_masks_after_edit":
        from manual_retuning import apply_retune_operation

        if released_background is None:
            raise ValueError("retuning operation has no reconstructed source vacancy")
        details = apply_retune_operation(
            labels, provenance, operation, released_background, excluded)
        return details, "post-edit-mask-retuning"
    raise ValueError(f"unsupported edit operation: {operation_type!r}")


def _operation_identity_values(operation: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for key in ("identity", "identity_a", "identity_b", "target_identity",
                "host_identity"):
        value = operation.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            values.append(int(value))
    for key in ("identities", "source_identities", "child_identities",
                "recipient_identities"):
        sequence = operation.get(key, [])
        if isinstance(sequence, (list, tuple)):
            values.extend(int(value) for value in sequence
                          if isinstance(value, int)
                          and not isinstance(value, bool) and value > 0)
    return values


def _promote_labels_for_operation(
        labels: np.ndarray, operation: dict[str, Any]) -> np.ndarray:
    """Promote non-negative labels only when this operation needs more capacity."""
    required = max(_operation_identity_values(operation), default=0)
    if required <= int(np.iinfo(labels.dtype).max):
        return labels
    if np.any(labels < 0):
        raise ValueError("cannot promote a label array containing negative values")
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        candidate = np.dtype(dtype)
        if required <= int(np.iinfo(candidate).max) \
                and candidate.itemsize > labels.dtype.itemsize:
            return labels.astype(candidate)
    raise ValueError(
        f"identity {required} exceeds the supported uint64 label capacity")


def replay_edit_log(bundle: EditingBundle, operations: list[dict[str, Any]]
                    ) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Materialize the editor state from the canonical labels and edit log."""
    labels = bundle.canonical_labels.copy()
    provenance = np.zeros(labels.shape, np.uint8)
    excluded: set[int] = set()
    released_by_operation: dict[int, np.ndarray] = {}
    for expected_id, operation in enumerate(operations, start=1):
        if not isinstance(operation, dict):
            raise ValueError("every edit operation must be an object")
        operation_id = _required_operation_integer(operation, "operation_id")
        if operation_id != expected_id:
            raise ValueError("edit operation IDs must be consecutive from 1")
        if not isinstance(operation.get("created_at_utc"), str) \
                or not operation["created_at_utc"]:
            raise ValueError("edit operation has no creation time")
        if not isinstance(operation.get("user"), str) or not operation["user"]:
            raise ValueError("edit operation has no user")
        labels = _promote_labels_for_operation(labels, operation)
        operation_type = operation.get("type")
        before = (labels.copy() if operation_type in {
            "exclude_identity", "delete_identity_interval"} else None)
        released_background = None
        if operation_type == "retune_masks_after_edit":
            from manual_retuning import _validate_source_operation_ids

            source_ids = _validate_source_operation_ids(
                operation.get("source_operation_ids", []),
                operations[:expected_id - 1])
            released_background = np.zeros(labels.shape, bool)
            for source_id in source_ids:
                released_background |= released_by_operation[source_id]
        details, audit_label = _apply_operation_to_state(
            bundle, labels, provenance, excluded, operation,
            released_background)
        if before is not None:
            released_by_operation[expected_id] = (before > 0) & (labels == 0)
        for key, actual in details.items():
            if operation.get(key) != actual:
                raise ValueError(
                    f"{audit_label} audit field {key} does not reproduce")
    return labels, provenance, excluded


def _automatic_checkpoint(bundle: EditingBundle) -> EditingCheckpoint:
    return EditingCheckpoint(
        operation_count=0,
        action="Original automatic output",
        scope="All frames",
        user="Motion pipeline",
        created_at_utc=str(bundle.manifest.get("created_at_utc", "")),
        changed_pixels=0,
    )


def _operation_checkpoint(operation: dict[str, Any]) -> EditingCheckpoint:
    operation_type = operation.get("type")
    if operation_type == "swap_identities":
        action = (
            f"Swap identity {operation['identity_a']} with identity "
            f"{operation['identity_b']}")
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']}–"
            f"{operation['end_imagej_frame']}")
    elif operation_type == "reassign_identities":
        joined = ", ".join(
            str(value) for value in operation["source_identities"])
        action = (
            f"Assign identities {joined} to identity "
            f"{operation['target_identity']}")
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']}–"
            f"{operation['end_imagej_frame']}")
    elif operation_type == "exclude_identity":
        action = f"Remove identity {operation['identity']}"
        scope = "Entire track"
    elif operation_type == "expand_identities":
        joined = ", ".join(str(value) for value in operation["identities"])
        action = (
            f"Expand identities {joined} by {operation['radius']:g} "
            f"{operation['effective_distance_unit']}")
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']}â€“"
            f"{operation['end_imagej_frame']}")
    elif operation_type == "delete_identity_interval":
        joined = ", ".join(str(value) for value in operation["identities"])
        action = f"Delete identities {joined} between frames"
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']}–"
            f"{operation['end_imagej_frame']}")
    elif operation_type == "force_split_identity_interval":
        joined = ", ".join(
            str(value) for value in operation["child_identities"])
        action = (
            f"Split identity {operation['host_identity']} into identities "
            f"{joined}")
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']} to "
            f"{operation['end_imagej_frame']}")
    elif operation_type == "add_identity_track":
        action = f"Add new identity {operation['identity']}"
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']} to "
            f"{operation['end_imagej_frame']}")
    elif operation_type == "retune_masks_after_edit":
        joined = ", ".join(
            str(value) for value in operation["recipient_identities"])
        action = f"Retune masks for identities {joined}"
        scope = (
            f"ImageJ frames {operation['start_imagej_frame']} to "
            f"{operation['end_imagej_frame']}")
    else:
        action = str(operation_type).replace("_", " ").capitalize()
        scope = "Recorded scope"
    return EditingCheckpoint(
        operation_count=int(operation["operation_id"]),
        action=action,
        scope=scope,
        user=str(operation["user"]),
        created_at_utc=str(operation["created_at_utc"]),
        changed_pixels=int(operation.get("changed_pixels", 0)),
    )


def _editing_source(path: Path) -> tuple[
        EditingBundle, np.ndarray, np.ndarray, set[int],
        list[dict[str, Any]], Path | None]:
    manifest_path = _editing_manifest_path(path)
    document = _json_object(manifest_path)
    if document.get("schema") == BUNDLE_SCHEMA:
        bundle = load_editing_bundle(manifest_path)
        return (
            bundle, bundle.canonical_labels.copy(),
            np.zeros(bundle.canonical_labels.shape, np.uint8), set(), [], None)
    if document.get("schema") == SESSION_SCHEMA:
        session = load_editing_session(manifest_path)
        return (
            session.bundle, session.curated_labels.copy(),
            session.manual_provenance.copy(),
            set(session.excluded_identities), list(session.edits),
            session.manifest_path)
    raise ValueError(f"unsupported editing source schema: {document.get('schema')!r}")


def _save_editing_session(
        bundle: EditingBundle, curated: np.ndarray, provenance: np.ndarray,
        excluded: set[int], edits: list[dict[str, Any]], output_dir: Path,
        mode: str, parent_session: Path | None = None,
        history_action: dict[str, Any] | None = None,
        batch_action: dict[str, Any] | None = None) -> Path:
    replayed, replayed_provenance, replayed_excluded = replay_edit_log(
        bundle, edits)
    if not np.array_equal(curated, replayed):
        raise ValueError("curated labels do not match the edit log")
    if not np.array_equal(provenance, replayed_provenance):
        raise ValueError("manual provenance does not match the edit log")
    if excluded != replayed_excluded:
        raise ValueError("excluded identities do not match the edit log")

    output = _create_output(output_dir)
    try:
        out_dir = output / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        interval = float(bundle.manifest["time"]["frame_interval_min"])
        curated_path = out_dir / f"{bundle.manifest['stem']}.tif"
        provenance_path = out_dir / "manual_pixel_provenance.tif"
        exclusions_path = out_dir / "excluded_identities.csv"
        edits_path = out_dir / "edits.json"
        frame_path = out_dir / "frame_identities.csv"
        tracks_path = out_dir / "tracks.csv"

        save_stack(curated_path, curated, interval)
        save_stack(provenance_path, provenance, interval)
        pd.DataFrame([
            {"identity": identity, "reason": "excluded by replayed edit log"}
            for identity in sorted(excluded)
        ], columns=["identity", "reason"]).to_csv(exclusions_path, index=False)
        write_json(edits_path, {
            "schema_version": 1,
            "operations": edits,
        })
        frame_table, tracks = rebuild_identity_tables(
            curated, bundle.registered_raw,
            bundle.registered_raw_indices, provenance, excluded)
        frame_table.to_csv(frame_path, index=False)
        tracks.to_csv(tracks_path, index=False)

        manifest = {
            "schema": SESSION_SCHEMA,
            "schema_version": SESSION_SCHEMA_VERSION,
            "status": "done",
            "created_at_utc": utc_now(),
            "mode": mode,
            "stem": bundle.manifest["stem"],
            "parent_bundle": _path_record(
                bundle.manifest_path, output, False),
            "parent_canonical_sha256": bundle.manifest[
                "assets"]["canonical_labels"]["sha256"],
            "manual_provenance_values": MANUAL_PROVENANCE_VALUES,
            "outputs": {
                "curated_labels": _tiff_record(
                    curated_path, output, True, "TYX"),
                "manual_provenance": _tiff_record(
                    provenance_path, output, True, "TYX"),
                "excluded_identities": _path_record(
                    exclusions_path, output, True),
                "edits": _path_record(edits_path, output, True),
                "frame_identities": _path_record(frame_path, output, True),
                "tracks": _path_record(tracks_path, output, True),
            },
            "summary": {
                "edit_operations": len(edits),
                "excluded_identities": len(excluded),
                "changed_pixels": int(np.count_nonzero(
                    curated != bundle.canonical_labels)),
                "array_equal_to_parent": bool(np.array_equal(
                    curated, bundle.canonical_labels)),
                "round_trip_verified": False,
            },
        }
        if parent_session is not None:
            manifest["parent_session"] = _path_record(
                parent_session, output, False)
        if history_action is not None:
            manifest["history_action"] = history_action
        if batch_action is not None:
            manifest["batch_action"] = batch_action
        manifest_path = output / SESSION_FILENAME
        write_json(manifest_path, manifest)
        load_editing_session(manifest_path)
        manifest["summary"]["round_trip_verified"] = True
        write_json(manifest_path, manifest)
    except BaseException as error:
        _record_failed_output(output, error)
        raise
    final_manifest = output / SESSION_FILENAME
    load_editing_session(final_manifest)
    return final_manifest


def save_zero_edit_session(bundle_path: Path, output_dir: Path) -> Path:
    """Save and reimport an unchanged session, proving the editing boundary."""
    bundle = load_editing_bundle(bundle_path)
    labels, provenance, excluded = replay_edit_log(bundle, [])
    return _save_editing_session(
        bundle, labels, provenance, excluded, [], output_dir,
        mode="zero_edit_round_trip")


def save_edit_batch_session(
        source_path: Path, output_dir: Path,
        operation_requests: list[dict[str, Any]], user: str | None = None,
        operation_count: int | None = None,
        mode: str = "edit_batch",
        batch_context: dict[str, Any] | None = None) -> Path:
    """Apply an ordered edit batch atomically and save one immutable session."""
    if _portable_source(source_path):
        from manual_portable_package import save_portable_edit_batch

        return save_portable_edit_batch(
            source_path, output_dir, operation_requests, user,
            operation_count=operation_count, mode=mode,
            batch_context=batch_context)
    if not isinstance(operation_requests, list) or not operation_requests:
        raise ValueError("an edit batch must contain at least one operation")
    bundle, _labels, _provenance, _excluded, edits, parent_session = \
        _editing_source(source_path)
    selected_count = len(edits) if operation_count is None else operation_count
    if selected_count < 0 or selected_count > len(edits):
        raise ValueError(
            f"checkpoint must be between 0 and {len(edits)}")
    default_user = getpass.getuser() if user is None else user.strip()
    if not default_user:
        raise ValueError("batch user cannot be empty")
    selected_edits = edits[:selected_count]
    labels, provenance, excluded = replay_edit_log(bundle, selected_edits)
    new_operations: list[dict[str, Any]] = []
    for offset, request in enumerate(operation_requests, start=1):
        normalized = _normalize_operation_request(request, len(labels))
        request_user = normalized.pop("user", default_user)
        if not isinstance(request_user, str) or not request_user.strip():
            raise ValueError(f"batch operation {offset} user cannot be empty")
        operation = {
            "operation_id": selected_count + offset,
            **normalized,
            "created_at_utc": utc_now(),
            "user": request_user.strip(),
        }
        labels = _promote_labels_for_operation(labels, operation)
        released_background = None
        if operation.get("type") == "retune_masks_after_edit":
            from manual_retuning import released_background_from_history

            released_background = released_background_from_history(
                bundle, selected_edits + new_operations,
                operation["source_operation_ids"])
        details, _audit_label = _apply_operation_to_state(
            bundle, labels, provenance, excluded, operation,
            released_background)
        operation.update(details)
        new_operations.append(operation)
    history_action = None
    if selected_count < len(edits):
        history_action = {
            "type": "edit_from_checkpoint",
            "created_at_utc": new_operations[0]["created_at_utc"],
            "user": default_user,
            "selected_operation_count": int(selected_count),
            "source_operation_count": int(len(edits)),
            "later_operations_preserved_in_parent": int(
                len(edits) - selected_count),
        }
    batch_action = {
        "schema": BATCH_SCHEMA,
        "schema_version": BATCH_SCHEMA_VERSION,
        "atomic": True,
        "operation_count": len(new_operations),
        "first_operation_id": new_operations[0]["operation_id"],
        "last_operation_id": new_operations[-1]["operation_id"],
    }
    if batch_context is not None:
        batch_action["context"] = batch_context
    return _save_editing_session(
        bundle, labels, provenance, excluded,
        [*selected_edits, *new_operations], output_dir, mode=mode,
        parent_session=parent_session, history_action=history_action,
        batch_action=batch_action)


def save_identity_swap_session(
        source_path: Path, output_dir: Path, identity_a: int, identity_b: int,
        start_imagej_frame: int, end_imagej_frame: int | None = None,
        user: str | None = None) -> Path:
    """Swap two identity values over an inclusive ImageJ frame interval."""
    return save_edit_batch_session(
        source_path, output_dir, [{
            "type": "swap_identities",
            "identity_a": int(identity_a),
            "identity_b": int(identity_b),
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": (
                None if end_imagej_frame is None else int(end_imagej_frame)),
        }], user, mode="identity_swap")


def save_identity_reassignment_session(
        source_path: Path, output_dir: Path, source_identities: list[int],
        target_identity: int, start_imagej_frame: int,
        end_imagej_frame: int | None = None, user: str | None = None,
        operation_count: int | None = None) -> Path:
    """Relabel source identities as one target over an inclusive frame range."""
    return save_edit_batch_session(
        source_path, output_dir, [{
            "type": "reassign_identities",
            "source_identities": [
                int(identity) for identity in source_identities],
            "target_identity": int(target_identity),
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": (
                None if end_imagej_frame is None else int(end_imagej_frame)),
        }], user, operation_count, mode="identity_reassignment")


def save_identity_exclusions_session(
        source_path: Path, output_dir: Path, identities: list[int],
        user: str | None = None, operation_count: int | None = None) -> Path:
    """Remove complete identity tracks together in one atomic session."""
    return save_edit_batch_session(
        source_path, output_dir, [{
            "type": "exclude_identity", "identity": int(identity),
        } for identity in identities], user, operation_count,
        mode="identity_exclusion_batch")


def save_identity_exclusion_session(
        source_path: Path, output_dir: Path, identity: int,
        user: str | None = None, operation_count: int | None = None) -> Path:
    """Remove one complete identity track and preserve it in the edit log."""
    return save_identity_exclusions_session(
        source_path, output_dir, [identity], user, operation_count)


def save_identity_expansion_session(
        source_path: Path, output_dir: Path, identities: list[int],
        radius: float, radius_unit: str = "calibrated",
        start_imagej_frame: int = 1,
        end_imagej_frame: int | None = None,
        user: str | None = None, operation_count: int | None = None) -> Path:
    """Expand one or more identities simultaneously into background pixels."""
    return save_edit_batch_session(
        source_path, output_dir, [{
            "type": "expand_identities",
            "identities": [int(identity) for identity in identities],
            "radius": float(radius),
            "radius_unit": radius_unit,
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": (
                None if end_imagej_frame is None else int(end_imagej_frame)),
        }], user, operation_count, mode="identity_expansion_batch")


def save_identity_interval_deletion_session(
        source_path: Path, output_dir: Path, identities: list[int],
        start_imagej_frame: int, end_imagej_frame: int,
        user: str | None = None, operation_count: int | None = None) -> Path:
    """Delete selected identities only inside an inclusive frame interval."""
    return save_edit_batch_session(
        source_path, output_dir, [{
            "type": "delete_identity_interval",
            "identities": [int(identity) for identity in identities],
            "start_imagej_frame": int(start_imagej_frame),
            "end_imagej_frame": int(end_imagej_frame),
        }], user, operation_count, mode="identity_interval_deletion_batch")


def load_edit_batch(path: Path) -> list[dict[str, Any]]:
    document = _json_object(path.resolve())
    if document.get("schema") != BATCH_SCHEMA:
        raise ValueError(f"unsupported edit-batch schema: {document.get('schema')!r}")
    if document.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise ValueError(
            "unsupported edit-batch schema version: "
            f"{document.get('schema_version')!r}")
    operations = document.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("an edit batch must contain at least one operation")
    if any(not isinstance(operation, dict) for operation in operations):
        raise ValueError("every batch operation must be an object")
    return operations


def save_history_checkpoint(
        source_path: Path, output_dir: Path, operation_count: int,
        user: str | None = None) -> Path:
    """Save an earlier edit-log position while preserving the later branch."""
    if _portable_source(source_path):
        from manual_portable_package import save_portable_history_checkpoint

        return save_portable_history_checkpoint(
            source_path, output_dir, operation_count, user)
    bundle, _labels, _provenance, _excluded, edits, parent_session = \
        _editing_source(source_path)
    if parent_session is None:
        raise ValueError("history checkpoints require an editing session source")
    if operation_count < 0 or operation_count > len(edits):
        raise ValueError(
            f"checkpoint must be between 0 and {len(edits)}")
    edit_user = getpass.getuser() if user is None else user.strip()
    if not edit_user:
        raise ValueError("checkpoint user cannot be empty")
    selected_edits = edits[:operation_count]
    labels, provenance, excluded = replay_edit_log(bundle, selected_edits)
    return _save_editing_session(
        bundle, labels, provenance, excluded, selected_edits, output_dir,
        mode="history_checkpoint", parent_session=parent_session,
        history_action={
            "type": "continue_from_checkpoint",
            "created_at_utc": utc_now(),
            "user": edit_user,
            "selected_operation_count": int(operation_count),
            "source_operation_count": int(len(edits)),
            "later_operations_preserved_in_parent": int(
                len(edits) - operation_count),
        })


def editing_history_summary(source_path: Path) -> dict[str, Any]:
    controller = EditingHistoryController(source_path)
    return {
        "source": str(
            controller.source_reference if controller.portable_source
            else controller.source_manifest),
        "current_operation_count": controller.position,
        "checkpoints": [{
            "operation_count": checkpoint.operation_count,
            "action": checkpoint.action,
            "scope": checkpoint.scope,
            "user": checkpoint.user,
            "created_at_utc": checkpoint.created_at_utc,
            "changed_pixels": checkpoint.changed_pixels,
        } for checkpoint in controller.checkpoints],
    }


def _number_review_identities(stack: np.ndarray, labels: np.ndarray,
                              identities: set[int]) -> np.ndarray:
    result = stack.copy()
    for t in range(len(result)):
        image = Image.fromarray(result[t])
        draw = ImageDraw.Draw(image)
        for identity in sorted(identities):
            points = np.argwhere(labels[t] == identity)
            if not len(points):
                continue
            y, x = points.mean(axis=0)
            text = str(identity)
            left = max(0, int(round(x)) - 2)
            top = max(0, int(round(y)) - 9)
            box = draw.textbbox((left, top), text)
            draw.rectangle(
                (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1),
                fill=(0, 0, 0))
            draw.text((left, top), text, fill=(255, 255, 255))
        result[t] = np.asarray(image)
    return result


def _annotate_edit_review(stack: np.ndarray,
                          imagej_frames: list[int]) -> np.ndarray:
    header_height = 24
    result = np.zeros(
        (len(stack), stack.shape[1] + header_height, stack.shape[2], 3),
        np.uint8)
    result[:, header_height:] = stack
    panel_width = stack.shape[2] // 3
    names = ("before", "after", "changed pixels")
    compact = panel_width < 180
    for t, imagej_frame in enumerate(imagej_frames):
        image = Image.fromarray(result[t])
        draw = ImageDraw.Draw(image)
        for panel, name in enumerate(names):
            title = (f"F{imagej_frame} "
                     f"{'CHANGE' if panel == 2 else name.upper()}"
                     if compact else f"ImageJ frame {imagej_frame}: {name}")
            draw.text(
                (panel * panel_width + 5, 5),
                title,
                fill=(255, 255, 255))
        result[t] = np.asarray(image)
    return result


def _editing_review_stack(raw: np.ndarray, before: np.ndarray,
                          after: np.ndarray, imagej_frames: list[int],
                          identities: set[int]) -> np.ndarray:
    before_view = _number_review_identities(
        outline_overlay(raw, before, thick=2), before, identities)
    after_view = _number_review_identities(
        outline_overlay(raw, after, thick=2), after, identities)
    changed_view = _number_review_identities(
        change_overlay(raw, before, after, thick=2), after, identities)
    return _annotate_edit_review(
        np.concatenate([before_view, after_view, changed_view], axis=2),
        imagej_frames)


def _padded_square_rgb(frame: np.ndarray, centre: tuple[float, float],
                       side: int) -> np.ndarray:
    """Extract a fixed-size display crop without treating padding as image data."""
    side = max(1, int(side))
    top = int(round(float(centre[0]) - (side - 1) / 2.0))
    left = int(round(float(centre[1]) - (side - 1) / 2.0))
    bottom, right = top + side, left + side
    source_y0, source_y1 = max(0, top), min(frame.shape[0], bottom)
    source_x0, source_x1 = max(0, left), min(frame.shape[1], right)
    result = np.zeros((side, side, 3), np.uint8)
    target_y0, target_x0 = source_y0 - top, source_x0 - left
    if source_y1 > source_y0 and source_x1 > source_x0:
        result[
            target_y0:target_y0 + source_y1 - source_y0,
            target_x0:target_x0 + source_x1 - source_x0,
        ] = frame[source_y0:source_y1, source_x0:source_x1]
    return result


def _new_cell_moving_crop_review(
        raw: np.ndarray, before: np.ndarray, after: np.ndarray,
        operation: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    start = int(operation["start_imagej_frame"])
    end = int(operation["end_imagej_frame"])
    frames = list(range(start, end + 1))
    indices = np.asarray([frame - 1 for frame in frames], dtype=np.intp)
    identity = int(operation["identity"])
    before_view = _number_review_identities(
        outline_overlay(raw[indices], before[indices], thick=2),
        before[indices], {identity})
    after_view = _number_review_identities(
        outline_overlay(raw[indices], after[indices], thick=2),
        after[indices], {identity})
    changed_view = _number_review_identities(
        change_overlay(raw[indices], before[indices], after[indices], thick=2),
        after[indices], {identity})
    centres_by_frame = {
        int(row["imagej_frame"]): row
        for row in operation.get("crop_centres", [])
        if isinstance(row, dict) and "imagej_frame" in row}
    side = int(operation["crop_side_px"])
    rows = []
    for offset, imagej_frame in enumerate(frames):
        centre_row = centres_by_frame.get(imagej_frame, {})
        if centre_row.get("y") is None or centre_row.get("x") is None:
            yy, xx = np.nonzero(after[imagej_frame - 1] == identity)
            if not len(yy):
                raise ValueError(
                    f"new identity {identity} has no crop centre or pixels on "
                    f"ImageJ frame {imagej_frame}")
            centre = (float(np.mean(yy)), float(np.mean(xx)))
        else:
            centre = (float(centre_row["y"]), float(centre_row["x"]))
        rows.append(np.concatenate([
            _padded_square_rgb(before_view[offset], centre, side),
            _padded_square_rgb(after_view[offset], centre, side),
            _padded_square_rgb(changed_view[offset], centre, side),
        ], axis=1))
    return _annotate_edit_review(np.asarray(rows), frames), frames


def _forced_split_boundary_stack(
        raw: np.ndarray, labels: np.ndarray, imagej_frames: list[int],
        operations: list[dict[str, Any]], identities: set[int]) -> np.ndarray:
    """Render accepted inferred child boundaries for forced-split review."""
    view = _number_review_identities(
        outline_overlay(raw, labels, thick=2), labels, identities)
    for index, imagej_frame in enumerate(imagej_frames):
        internal = np.zeros(labels[index].shape, bool)
        frame_labels = labels[index]
        for operation in operations:
            if operation.get("type") != "force_split_identity_interval" \
                    or imagej_frame < int(operation["start_imagej_frame"]) \
                    or imagej_frame > int(operation["end_imagej_frame"]):
                continue
            children = tuple(map(int, operation["child_identities"]))
            for child in children:
                own = frame_labels == child
                other = np.isin(frame_labels, [
                    value for value in children if value != child])
                internal |= own & ndi.binary_dilation(
                    other, structure=np.ones((3, 3), bool))
        view[index][internal] = np.array([255, 210, 45], np.uint8)
    header = np.zeros(
        (len(view), view.shape[1] + 24, view.shape[2], 3), np.uint8)
    header[:, 24:] = view
    for index, imagej_frame in enumerate(imagej_frames):
        image = Image.fromarray(header[index])
        ImageDraw.Draw(image).text(
            (5, 5),
            f"ImageJ frame {imagej_frame}: accepted inferred split boundary",
            fill=(255, 255, 255))
        header[index] = np.asarray(image)
    return header


def _retuned_mask_ownership_stack(
        raw: np.ndarray, labels: np.ndarray, imagej_frames: list[int],
        operations: list[dict[str, Any]], identities: set[int]) -> np.ndarray:
    """Render reclaimed, unresolved, and contested post-edit mask pixels."""
    from manual_retuning import decode_retune_patch

    view = _number_review_identities(
        outline_overlay(raw, labels, thick=2), labels, identities)
    for operation in operations:
        if operation.get("type") != "retune_masks_after_edit":
            continue
        assignments, eligible, _released, contested, (y0, x0) = \
            decode_retune_patch(operation["accepted_patch"], labels.dtype)
        start = int(operation["start_imagej_frame"])
        y1, x1 = y0 + assignments.shape[1], x0 + assignments.shape[2]
        for review_index, imagej_frame in enumerate(imagej_frames):
            offset = imagej_frame - start
            if offset < 0 or offset >= len(assignments):
                continue
            assignment = assignments[offset]
            domain = eligible[offset]
            contested_view = contested[offset]
            unresolved = domain & (assignment == 0)
            region = view[review_index, y0:y1, x0:x1]
            region[unresolved] = np.rint(
                0.35 * region[unresolved]
                + 0.65 * np.array([235, 155, 35])
            ).astype(np.uint8)
            region[contested_view] = np.array([235, 55, 210], np.uint8)
            reclaimed = assignment > 0
            region[reclaimed] = np.rint(
                0.20 * region[reclaimed]
                + 0.80 * np.array([75, 235, 120])
            ).astype(np.uint8)
    header = np.zeros(
        (len(view), view.shape[1] + 24, view.shape[2], 3), np.uint8)
    header[:, 24:] = view
    for index, imagej_frame in enumerate(imagej_frames):
        image = Image.fromarray(header[index])
        ImageDraw.Draw(image).text(
            (5, 5),
            f"ImageJ frame {imagej_frame}: green reclaimed; orange unresolved; "
            "magenta contested",
            fill=(255, 255, 255))
        header[index] = np.asarray(image)
    return header


def _crop_bounds(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    y, x = np.nonzero(np.any(mask, axis=0))
    if not len(y):
        raise ValueError("editing session has no changed pixels to review")
    height, width = mask.shape[1:]
    return (
        max(0, int(y.min()) - padding),
        min(height, int(y.max()) + padding + 1),
        max(0, int(x.min()) - padding),
        min(width, int(x.max()) + padding + 1),
    )


def _representative_review_frames(operation: dict[str, Any],
                                  changed_frames: list[int]) -> list[int]:
    selected = {changed_frames[0], changed_frames[-1]}
    per_frame = operation.get("per_frame", [])
    for previous, current in zip(per_frame, per_frame[1:]):
        previous_presence = (
            previous.get("identity_a_to_b_pixels", 0) > 0,
            previous.get("identity_b_to_a_pixels", 0) > 0)
        current_presence = (
            current.get("identity_a_to_b_pixels", 0) > 0,
            current.get("identity_b_to_a_pixels", 0) > 0)
        if previous_presence != current_presence:
            selected.add(int(previous["imagej_frame"]))
            selected.add(int(current["imagej_frame"]))
    if len(selected) < min(4, len(changed_frames)):
        evenly_spaced = np.linspace(
            0, len(changed_frames) - 1,
            num=min(4, len(changed_frames)), dtype=int)
        selected.update(changed_frames[int(index)] for index in evenly_spaced)
    return sorted(selected)


def build_editing_review(session_path: Path, output_dir: Path,
                         padding_px: int = 24) -> Path:
    """Build immutable before/after/change reviews for an edited session."""
    if padding_px < 0:
        raise ValueError("review padding cannot be negative")
    session = load_editing_session(session_path)
    parent_record = session.manifest.get("parent_session")
    if parent_record is None:
        batch_action = session.manifest.get("batch_action", {})
        first_operation_id = int(batch_action.get(
            "first_operation_id", len(session.edits)))
        if session.manifest.get("mode") == "portable_checkpoint" \
                and session.edits and first_operation_id > 1:
            before, _provenance, _excluded = replay_edit_log(
                session.bundle, session.edits[:first_operation_id - 1])
        else:
            before = session.bundle.canonical_labels
    else:
        parent_path = _validate_file_record(
            session.manifest_path, parent_record, True)
        before = load_editing_session(parent_path).curated_labels
    after = session.curated_labels
    changed = before != after
    changed_indices = np.flatnonzero(
        changed.reshape(len(changed), -1).any(axis=1))
    if not len(changed_indices):
        raise ValueError("editing session has no changed frames to review")
    imagej_frames = [int(value + 1) for value in changed_indices]
    latest_operation = session.edits[-1]
    batch_action = session.manifest.get("batch_action", {})
    first_operation_id = int(batch_action.get(
        "first_operation_id", latest_operation["operation_id"]))
    review_operations = session.edits[first_operation_id - 1:]
    identities: set[int] = set()
    for operation in review_operations:
        if operation.get("type") == "swap_identities":
            identities.update((
                int(operation["identity_a"]),
                int(operation["identity_b"]),
            ))
        elif operation.get("type") == "reassign_identities":
            identities.update(
                int(value) for value in operation["source_identities"])
            identities.add(int(operation["target_identity"]))
        elif operation.get("type") == "exclude_identity":
            identities.add(int(operation["identity"]))
        elif operation.get("type") == "expand_identities":
            identities.update(int(value) for value in operation["identities"])
        elif operation.get("type") == "delete_identity_interval":
            identities.update(int(value) for value in operation["identities"])
        elif operation.get("type") == "force_split_identity_interval":
            identities.update(
                int(value) for value in operation["child_identities"])
        elif operation.get("type") == "add_identity_track":
            identities.add(int(operation["identity"]))
        elif operation.get("type") == "retune_masks_after_edit":
            identities.update(
                int(value) for value in operation["recipient_identities"])
    y0, y1, x0, x1 = _crop_bounds(changed, padding_px)
    representative_frames = _representative_review_frames(
        latest_operation, imagej_frames)
    representative_lookup = {
        imagej_frame: index
        for index, imagej_frame in enumerate(imagej_frames)}

    output = _create_output(output_dir)
    try:
        interval = float(session.bundle.manifest["time"]["frame_interval_min"])
        full_path = output / "full_field_before_after_change.tif"
        focus_path = output / "focused_before_after_change.tif"
        preview_path = output / "representative_before_after_change.png"
        split_operations = [
            operation for operation in review_operations
            if operation.get("type") == "force_split_identity_interval"]
        new_cell_operations = [
            operation for operation in review_operations
            if operation.get("type") == "add_identity_track"]
        retune_operations = [
            operation for operation in review_operations
            if operation.get("type") == "retune_masks_after_edit"]

        raw = session.bundle.registered_raw[changed_indices]
        before_changed = before[changed_indices]
        after_changed = after[changed_indices]
        full_review = _editing_review_stack(
            raw, before_changed, after_changed, imagej_frames, identities)
        save_rgb_stack(full_path, full_review, interval)

        focus_review = _editing_review_stack(
            raw[:, y0:y1, x0:x1],
            before_changed[:, y0:y1, x0:x1],
            after_changed[:, y0:y1, x0:x1],
            imagej_frames, identities)
        save_rgb_stack(focus_path, focus_review, interval)
        preview_rows = [
            focus_review[representative_lookup[imagej_frame]]
            for imagej_frame in representative_frames]
        preview = Image.fromarray(np.concatenate(preview_rows, axis=0))
        preview_scale = 3
        preview.resize(
            (preview.width * preview_scale, preview.height * preview_scale),
            Image.Resampling.NEAREST).save(preview_path)

        split_boundary_path = None
        if split_operations:
            split_boundary_path = output / "forced_split_boundaries.tif"
            split_review = _forced_split_boundary_stack(
                raw, after_changed, imagej_frames,
                split_operations, identities)
            save_rgb_stack(split_boundary_path, split_review, interval)

        retune_ownership_path = None
        if retune_operations:
            retune_ownership_path = output / "retuned_mask_ownership.tif"
            retune_review = _retuned_mask_ownership_stack(
                raw, after_changed, imagej_frames,
                retune_operations, identities)
            save_rgb_stack(retune_ownership_path, retune_review, interval)

        new_cell_outputs: dict[str, dict[str, Any]] = {}
        if new_cell_operations:
            centre_rows: list[dict[str, Any]] = []
            threshold_rows: list[dict[str, Any]] = []
            warning_rows: list[dict[str, Any]] = []
            summary_rows: list[dict[str, Any]] = []
            for operation in new_cell_operations:
                operation_id = int(operation["operation_id"])
                identity = int(operation["identity"])
                for row in operation.get("crop_centres", []):
                    centre_rows.append({
                        "operation_id": operation_id, "identity": identity,
                        **row})
                for row in operation.get("thresholds", []):
                    threshold_rows.append({
                        "operation_id": operation_id, "identity": identity,
                        **row})
                for warning in operation.get("proposal_warnings", []):
                    imagej_frame = None
                    prefix = "ImageJ frame "
                    if warning.startswith(prefix):
                        token = warning[len(prefix):].split(":", 1)[0]
                        try:
                            imagej_frame = int(token)
                        except ValueError:
                            pass
                    warning_rows.append({
                        "operation_id": operation_id, "identity": identity,
                        "imagej_frame": imagej_frame, "warning": warning})
                summary_rows.append({
                    "operation_id": operation_id,
                    "identity": identity,
                    "start_imagej_frame": operation["start_imagej_frame"],
                    "end_imagej_frame": operation["end_imagej_frame"],
                    "method": operation["method"],
                    "anchor_click_count": len(operation["anchor_clicks"]),
                    "linked_split_frames": operation.get(
                        "linked_split_frames", []),
                })
                crop_review, crop_frames = _new_cell_moving_crop_review(
                    session.bundle.registered_raw, before, after, operation)
                crop_path = output / (
                    f"new_identity_{identity}_moving_crop_review.tif")
                save_rgb_stack(crop_path, crop_review, interval)
                new_cell_outputs[
                    f"new_identity_{identity}_moving_crop_review"] = \
                    _tiff_record(crop_path, output, True, "TYXS")
                new_cell_outputs[
                    f"new_identity_{identity}_moving_crop_review"][
                        "imagej_frames"] = crop_frames

            centre_path = output / "new_cell_crop_centres.csv"
            threshold_path = output / "new_cell_threshold_trace.csv"
            warning_path = output / "new_cell_warning_index.csv"
            summary_path = output / "new_cell_operation_summary.json"
            pd.DataFrame(centre_rows).to_csv(centre_path, index=False)
            pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)
            pd.DataFrame(
                warning_rows,
                columns=("operation_id", "identity", "imagej_frame", "warning")
            ).to_csv(warning_path, index=False)
            write_json(summary_path, {
                "schema": "motion.manual-new-cell-review-summary",
                "schema_version": 1, "operations": summary_rows})
            new_cell_outputs.update({
                "new_cell_crop_centres": _path_record(
                    centre_path, output, True),
                "new_cell_threshold_trace": _path_record(
                    threshold_path, output, True),
                "new_cell_warning_index": _path_record(
                    warning_path, output, True),
                "new_cell_operation_summary": _path_record(
                    summary_path, output, True),
            })

        manifest = {
            "schema": "motion.manual-editing-review",
            "schema_version": 1,
            "status": "done",
            "created_at_utc": utc_now(),
            "source_session": _path_record(
                session.manifest_path, output, False),
            "comparison": "immediate parent session to edited session",
            "imagej_frames": imagej_frames,
            "representative_imagej_frames": representative_frames,
            "representative_preview_scale": preview_scale,
            "focus_crop_yx": {
                "y_start": y0, "y_end_exclusive": y1,
                "x_start": x0, "x_end_exclusive": x1,
                "padding_px": padding_px,
            },
            "outputs": {
                "full_field_review": _tiff_record(
                    full_path, output, True, "TYXS"),
                "focused_review": _tiff_record(
                    focus_path, output, True, "TYXS"),
                "representative_preview": _path_record(
                    preview_path, output, True),
            },
            "summary": {
                "changed_frames": len(imagej_frames),
                "changed_pixels": int(np.count_nonzero(changed)),
                "reviewed_identities": sorted(identities),
                "reviewed_operations": len(review_operations),
            },
        }
        if split_boundary_path is not None:
            manifest["outputs"]["forced_split_boundaries"] = _tiff_record(
                split_boundary_path, output, True, "TYXS")
        if retune_ownership_path is not None:
            manifest["outputs"]["retuned_mask_ownership"] = _tiff_record(
                retune_ownership_path, output, True, "TYXS")
        manifest["outputs"].update(new_cell_outputs)
        manifest_path = output / "motion-editing-review.json"
        write_json(manifest_path, manifest)
        for record in manifest["outputs"].values():
            _validate_file_record(manifest_path, record, True)
    except BaseException as error:
        _record_failed_output(output, error)
        raise
    return output / "motion-editing-review.json"


def load_editing_session(path: Path, verify_hashes: bool = True
                          ) -> EditingSession:
    """Reimport and validate a saved manual-editing session."""
    manifest_path = _editing_manifest_path(path)
    if manifest_path.name != SESSION_FILENAME:
        document = _json_object(manifest_path)
        if document.get("schema") != SESSION_SCHEMA:
            raise ValueError("portable package entry is not an editing session")
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"unsupported session schema: {manifest.get('schema')!r}")
    if manifest.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported session schema version: {manifest.get('schema_version')!r}")
    if manifest.get("status") != "done":
        raise ValueError("editing session is not complete")

    parent_record = manifest.get("parent_bundle")
    if not isinstance(parent_record, dict):
        raise ValueError("editing session has no parent bundle")
    parent_path = _validate_file_record(
        manifest_path, parent_record, verify_hashes)
    bundle = load_editing_bundle(parent_path, verify_hashes=verify_hashes)
    if manifest.get("parent_canonical_sha256") != bundle.manifest[
            "assets"]["canonical_labels"]["sha256"]:
        raise ValueError("session parent canonical fingerprint differs")
    parent_session_record = manifest.get("parent_session")
    if parent_session_record is not None:
        if not isinstance(parent_session_record, dict):
            raise ValueError("parent session file record must be an object")
        parent_session_path = _validate_file_record(
            manifest_path, parent_session_record, verify_hashes)
        if _json_object(parent_session_path).get("schema") != SESSION_SCHEMA:
            raise ValueError("parent session does not use the session schema")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("editing session has no outputs object")
    for name in ("curated_labels", "manual_provenance",
                 "excluded_identities", "edits", "frame_identities", "tracks"):
        if not isinstance(outputs.get(name), dict):
            raise ValueError(f"editing session has no {name} output")

    curated_record = outputs["curated_labels"]
    if curated_record.get("semantic_axes") != "TYX":
        raise ValueError("curated labels must declare TYX semantic axes")
    curated_path = _validate_tiff_record(
        manifest_path, curated_record, verify_hashes)
    curated = _read_tiff(curated_path)
    validate_labels(curated)
    if curated.shape != bundle.canonical_labels.shape:
        raise ValueError("curated and parent labels differ in shape")

    provenance_record = outputs["manual_provenance"]
    if provenance_record.get("semantic_axes") != "TYX":
        raise ValueError("manual provenance must declare TYX semantic axes")
    provenance_path = _validate_tiff_record(
        manifest_path, provenance_record, verify_hashes)
    provenance = _read_tiff(provenance_path)
    if provenance.shape != curated.shape \
            or not np.issubdtype(provenance.dtype, np.integer) \
            or np.any((provenance < 0) | (provenance > 5)):
        raise ValueError("manual provenance is incompatible with curated labels")

    exclusions_path = _validate_file_record(
        manifest_path, outputs["excluded_identities"], verify_hashes)
    exclusions = pd.read_csv(exclusions_path)
    if list(exclusions.columns) != ["identity", "reason"]:
        raise ValueError("excluded identities must have identity and reason columns")
    if exclusions.identity.isna().any():
        raise ValueError("excluded identity cannot be empty")
    excluded = {int(value) for value in exclusions.identity}
    if any(value <= 0 for value in excluded):
        raise ValueError("excluded identities must be positive")

    edits_path = _validate_file_record(
        manifest_path, outputs["edits"], verify_hashes)
    edits_document = _json_object(edits_path)
    edits = edits_document.get("operations")
    if edits_document.get("schema_version") != 1 or not isinstance(edits, list) \
            or any(not isinstance(value, dict) for value in edits):
        raise ValueError("edit log is invalid")

    replayed, replayed_provenance, replayed_excluded = replay_edit_log(
        bundle, edits)
    if not np.array_equal(curated, replayed):
        raise ValueError("curated labels do not match the replayed edit log")
    if not np.array_equal(provenance, replayed_provenance):
        raise ValueError("manual provenance does not match the replayed edit log")
    if excluded != replayed_excluded:
        raise ValueError("excluded identities do not match the replayed edit log")

    frame_path = _validate_file_record(
        manifest_path, outputs["frame_identities"], verify_hashes)
    tracks_path = _validate_file_record(
        manifest_path, outputs["tracks"], verify_hashes)
    frame_table = pd.read_csv(frame_path)
    tracks = pd.read_csv(tracks_path)
    expected_frames, expected_tracks = rebuild_identity_tables(
        curated, bundle.registered_raw, bundle.registered_raw_indices,
        provenance, excluded)
    _assert_table_matches(frame_table, expected_frames, "frame identities")
    _assert_table_matches(tracks, expected_tracks, "tracks")

    summary = manifest.get("summary")
    expected_summary = {
        "edit_operations": len(edits),
        "excluded_identities": len(excluded),
        "changed_pixels": int(np.count_nonzero(
            curated != bundle.canonical_labels)),
        "array_equal_to_parent": bool(np.array_equal(
            curated, bundle.canonical_labels)),
    }
    if not isinstance(summary, dict):
        raise ValueError("editing session has no summary")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"editing session summary field {key} differs")
    if not isinstance(summary.get("round_trip_verified"), bool):
        raise ValueError("editing session round-trip status must be true or false")

    batch_action = manifest.get("batch_action")
    if batch_action is not None:
        if not isinstance(batch_action, dict) \
                or batch_action.get("schema") != BATCH_SCHEMA \
                or batch_action.get("schema_version") != BATCH_SCHEMA_VERSION \
                or batch_action.get("atomic") is not True:
            raise ValueError("editing session batch record is invalid")
        batch_count = batch_action.get("operation_count")
        first_id = batch_action.get("first_operation_id")
        last_id = batch_action.get("last_operation_id")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (batch_count, first_id, last_id)) \
                or batch_count < 1 or first_id < 1 \
                or last_id - first_id + 1 != batch_count \
                or last_id > len(edits):
            raise ValueError("editing session batch operation range is invalid")
        context = batch_action.get("context")
        if context is not None:
            if not isinstance(context, dict) \
                    or context.get("selection_method") != "track_filters":
                raise ValueError("editing session batch context is invalid")
            filter_set = context.get("filter_set")
            if not isinstance(filter_set, dict):
                raise ValueError("filtered batch has no filter set")
            canonical_filter_set = json.dumps(
                filter_set, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(canonical_filter_set).hexdigest() != context.get(
                    "filter_set_sha256"):
                raise ValueError("filtered batch filter-set fingerprint differs")
            selected = context.get("selected_identities")
            selected_metrics = context.get("selected_metrics")
            batch_operations = edits[first_id - 1:last_id]
            operation_type = context.get("operation_type", "exclude_identity")
            if operation_type == "exclude_identity":
                operation_selection = [
                    operation.get("identity") for operation in batch_operations]
                operation_types_valid = all(
                    operation.get("type") == "exclude_identity"
                    for operation in batch_operations)
            elif operation_type == "expand_identities":
                operation_selection = (
                    batch_operations[0].get("identities")
                    if len(batch_operations) == 1 else None)
                operation_parameters = context.get("operation_parameters")
                operation_types_valid = (
                    len(batch_operations) == 1
                    and batch_operations[0].get("type") == "expand_identities"
                    and isinstance(operation_parameters, dict)
                    and set(operation_parameters) <= {
                        "radius", "radius_unit", "start_imagej_frame",
                        "end_imagej_frame"}
                    and all(
                        value is None or batch_operations[0].get(key) == value
                        for key, value in operation_parameters.items()))
            elif operation_type == "delete_identity_interval":
                operation_selection = (
                    batch_operations[0].get("identities")
                    if len(batch_operations) == 1 else None)
                operation_parameters = context.get("operation_parameters")
                operation_types_valid = (
                    len(batch_operations) == 1
                    and batch_operations[0].get("type")
                    == "delete_identity_interval"
                    and isinstance(operation_parameters, dict)
                    and set(operation_parameters) <= {
                        "start_imagej_frame", "end_imagej_frame"}
                    and all(
                        batch_operations[0].get(key) == value
                        for key, value in operation_parameters.items()))
            else:
                operation_selection = None
                operation_types_valid = False
            if not isinstance(selected, list) or not selected \
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or value <= 0 for value in selected) \
                    or selected != operation_selection \
                    or not operation_types_valid \
                    or not isinstance(selected_metrics, list) \
                    or [row.get("identity") for row in selected_metrics
                        if isinstance(row, dict)] != selected:
                raise ValueError("filtered batch selection record differs")

    return EditingSession(
        manifest_path=manifest_path,
        manifest=manifest,
        bundle=bundle,
        curated_labels=curated,
        manual_provenance=provenance,
        excluded_identities=excluded,
        edits=edits,
        frame_identities=frame_table,
        tracks=tracks,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and validate Motion manual-editing inputs")
    commands = parser.add_subparsers(dest="command", required=True)

    accepted = commands.add_parser(
        "prepare-accepted", help="prepare the current accepted Motion result")
    accepted.add_argument("--output", type=Path, required=True)
    accepted.add_argument("--accepted-base", type=Path)
    accepted.add_argument("--config", type=Path)
    accepted.add_argument("--stem")
    accepted.add_argument("--spatial-unit")
    accepted.add_argument("--pixel-size-y", type=float)
    accepted.add_argument("--pixel-size-x", type=float)

    prepare = commands.add_parser(
        "prepare", help="prepare explicitly selected Motion outputs")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--stem", required=True)
    prepare.add_argument("--canonical-labels", type=Path, required=True)
    prepare.add_argument("--registered-raw", type=Path, required=True)
    prepare.add_argument("--motion-composite", type=Path, required=True)
    prepare.add_argument("--lag-float", type=Path)
    prepare.add_argument("--unclaimed-labels", type=Path)
    prepare.add_argument("--parent-manifest", type=Path)
    prepare.add_argument("--source-frame-offset", type=int, default=0)
    prepare.add_argument("--frame-interval-min", type=float, required=True)
    prepare.add_argument("--spatial-unit")
    prepare.add_argument("--pixel-size-y", type=float)
    prepare.add_argument("--pixel-size-x", type=float)

    zero = commands.add_parser(
        "zero-edit", help="save and reimport an unchanged editing session")
    zero.add_argument("--bundle", type=Path, required=True)
    zero.add_argument("--output", type=Path, required=True)

    swap = commands.add_parser(
        "swap-identities",
        help="swap two identities over an inclusive ImageJ frame range")
    swap.add_argument("--source", type=Path, required=True)
    swap.add_argument("--output", type=Path, required=True)
    swap.add_argument("--identity-a", type=int, required=True)
    swap.add_argument("--identity-b", type=int, required=True)
    swap.add_argument("--start-frame", type=int, required=True,
                      help="first ImageJ frame, inclusive and one-based")
    swap.add_argument("--end-frame", type=int,
                      help="last ImageJ frame, inclusive; defaults to the movie end")
    swap.add_argument("--user", help="person making the edit; defaults to the OS user")

    assign = commands.add_parser(
        "assign-identities",
        help="relabel source identities as one target over a frame range")
    assign.add_argument("--source", type=Path, required=True)
    assign.add_argument("--output", type=Path, required=True)
    assign.add_argument(
        "--source-identity", type=int, action="append", required=True,
        help="identity to relabel; repeat for several sources")
    assign.add_argument("--target-identity", type=int, required=True)
    assign.add_argument("--start-frame", type=int, required=True)
    assign.add_argument("--end-frame", type=int)
    assign.add_argument(
        "--user", help="person making the edit; defaults to the OS user")

    remove = commands.add_parser(
        "remove-identity",
        help="remove one or more identities in one atomic editing session")
    remove.add_argument("--source", type=Path, required=True)
    remove.add_argument("--output", type=Path, required=True)
    remove.add_argument(
        "--identity", type=int, action="append", required=True,
        help="identity to remove; repeat this option for a batch")
    remove.add_argument(
        "--user", help="person removing the identity; defaults to the OS user")

    delete_interval = commands.add_parser(
        "delete-between-frames",
        help="delete selected identity outlines only inside a frame range")
    delete_interval.add_argument("--source", type=Path, required=True)
    delete_interval.add_argument("--output", type=Path, required=True)
    delete_interval.add_argument(
        "--identity", type=int, action="append", required=True,
        help="identity to delete; repeat this option for a simultaneous batch")
    delete_interval.add_argument("--start-frame", type=int, required=True)
    delete_interval.add_argument("--end-frame", type=int, required=True)
    delete_interval.add_argument(
        "--user", help="person deleting the interval; defaults to the OS user")

    expand = commands.add_parser(
        "expand-identities",
        help="expand one or more identities simultaneously into background")
    expand.add_argument("--source", type=Path, required=True)
    expand.add_argument("--output", type=Path, required=True)
    expand.add_argument(
        "--identity", type=int, action="append", required=True,
        help="identity to expand; repeat this option for a simultaneous batch")
    expand.add_argument("--radius", type=float, required=True)
    expand.add_argument(
        "--radius-unit", choices=("calibrated", "pixel"),
        default="calibrated")
    expand.add_argument("--start-frame", type=int, default=1)
    expand.add_argument("--end-frame", type=int)
    expand.add_argument(
        "--user", help="person expanding the identities; defaults to the OS user")

    batch = commands.add_parser(
        "apply-batch", help="apply an ordered JSON edit batch atomically")
    batch.add_argument("--source", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--batch", type=Path, required=True)
    batch.add_argument(
        "--user", help="default user for operations without their own user")

    propose_cell = commands.add_parser(
        "propose-new-cell",
        help="generate a non-authoritative missed-cell draft for review")
    propose_cell.add_argument("--source", type=Path, required=True)
    propose_cell.add_argument("--output", type=Path, required=True)
    propose_cell.add_argument(
        "--anchor", action="append", required=True, metavar="FRAME,Y,X",
        help="ImageJ frame, y, and x centroid; repeat after tracking loss")
    propose_cell.add_argument("--identity", type=int)
    propose_cell.add_argument(
        "--method", choices=("soma_seed_expansion", "fixed_local_threshold",
                             "adaptive_local_threshold", "manual_outline"),
        default="soma_seed_expansion")
    propose_cell.add_argument("--crop-side-px", type=int)
    propose_cell.add_argument("--threshold-z", type=float, default=2.0)
    propose_cell.add_argument("--threshold-smoothing", type=float, default=0.55)
    propose_cell.add_argument("--minimum-area-px", type=int, default=4)
    propose_cell.add_argument("--start-frame", type=int, default=1)
    propose_cell.add_argument("--end-frame", type=int)
    propose_cell.add_argument(
        "--start-reason", choices=("movie_start", "birth", "border_entry"),
        default="movie_start")
    propose_cell.add_argument(
        "--end-reason", choices=("movie_end", "border_exit"),
        default="movie_end")

    commit_cell = commands.add_parser(
        "commit-new-cell-draft",
        help="commit fully reviewed missed-cell draft jobs atomically")
    commit_cell.add_argument("--source", type=Path, required=True)
    commit_cell.add_argument("--draft", type=Path, required=True)
    commit_cell.add_argument("--output", type=Path, required=True)
    commit_cell.add_argument(
        "--user", help="person committing the reviewed new-cell jobs")

    preview_filters = commands.add_parser(
        "preview-filters",
        help="calculate track metrics and preview selected identities")
    preview_filters.add_argument("--source", type=Path, required=True)
    preview_filters.add_argument("--filters", type=Path, required=True)
    preview_filters.add_argument("--output", type=Path, required=True)
    preview_filters.add_argument("--operation-count", type=int)

    remove_filters = commands.add_parser(
        "remove-by-filters",
        help="atomically remove identities selected by a validated filter set")
    remove_filters.add_argument("--source", type=Path, required=True)
    remove_filters.add_argument("--filters", type=Path, required=True)
    remove_filters.add_argument("--output", type=Path, required=True)
    remove_filters.add_argument("--operation-count", type=int)
    remove_filters.add_argument(
        "--user", help="person applying the filters; defaults to the OS user")

    apply_filters = commands.add_parser(
        "apply-by-filters",
        help="apply an operation to identities selected by validated filters")
    apply_filters.add_argument("--source", type=Path, required=True)
    apply_filters.add_argument("--filters", type=Path, required=True)
    apply_filters.add_argument("--output", type=Path, required=True)
    apply_filters.add_argument(
        "--operation", choices=("remove", "delete-frames", "expand"),
        required=True)
    apply_filters.add_argument("--radius", type=float)
    apply_filters.add_argument(
        "--radius-unit", choices=("calibrated", "pixel"),
        default="calibrated")
    apply_filters.add_argument("--start-frame", type=int, default=1)
    apply_filters.add_argument("--end-frame", type=int)
    apply_filters.add_argument("--operation-count", type=int)
    apply_filters.add_argument(
        "--user", help="person applying the filters; defaults to the OS user")

    review = commands.add_parser(
        "build-review", help="build before/after/change images for a session")
    review.add_argument("--session", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--padding-px", type=int, default=24)

    history = commands.add_parser(
        "list-history", help="list every replayable edit checkpoint")
    history.add_argument("--source", type=Path, required=True)

    restore = commands.add_parser(
        "restore-checkpoint",
        help="save an earlier checkpoint without deleting later edits")
    restore.add_argument("--source", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    restore.add_argument("--operation-count", type=int, required=True)
    restore.add_argument(
        "--user", help="person restoring the checkpoint; defaults to the OS user")

    controls = commands.add_parser(
        "history-controls", help="open graphical Undo, Redo, and history controls")
    controls.add_argument("--source", type=Path, required=True)

    editor = commands.add_parser(
        "edit", help="open the graphical manual track-editing workspace")
    editor.add_argument("--source", type=Path, required=True)

    validate_bundle = commands.add_parser(
        "validate-bundle", help="validate an editing input bundle")
    validate_bundle.add_argument("--bundle", type=Path, required=True)

    validate_session = commands.add_parser(
        "validate-session", help="validate a saved editing session")
    validate_session.add_argument("--session", type=Path, required=True)

    freeze_package = commands.add_parser(
        "freeze-package",
        help="freeze a bundle or session as a self-contained editing package")
    freeze_package.add_argument("--source", type=Path, required=True)
    freeze_package.add_argument("--output", type=Path, required=True)
    freeze_package.add_argument("--operation-count", type=int)
    freeze_package.add_argument(
        "--container", choices=("archive", "directory"), default="archive")

    validate_package = commands.add_parser(
        "validate-package", help="validate a portable editing package")
    validate_package.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-accepted":
        path = prepare_accepted_editing_bundle(
            args.output, args.accepted_base, args.config, args.stem,
            args.spatial_unit, args.pixel_size_y, args.pixel_size_x)
        result = {"bundle": str(path), "status": "validated"}
    elif args.command == "prepare":
        label_frames = _tiff_summary(args.canonical_labels.resolve())["shape"][0]
        raw_indices = [args.source_frame_offset + value
                       for value in range(label_frames)]
        path = prepare_editing_bundle(
            output_dir=args.output,
            stem=args.stem,
            canonical_labels=args.canonical_labels,
            registered_raw=args.registered_raw,
            motion_composite=args.motion_composite,
            label_frame_to_raw=raw_indices,
            frame_interval_min=args.frame_interval_min,
            unclaimed_labels=args.unclaimed_labels,
            lag_float=args.lag_float,
            parent_manifest=args.parent_manifest,
            parent={"kind": "explicit_motion_output"},
            spatial_unit=args.spatial_unit,
            pixel_size_y=args.pixel_size_y,
            pixel_size_x=args.pixel_size_x,
        )
        result = {"bundle": str(path), "status": "validated"}
    elif args.command == "zero-edit":
        path = save_zero_edit_session(args.bundle, args.output)
        result = {"session": str(path), "status": "round_trip_verified"}
    elif args.command == "swap-identities":
        path = save_identity_swap_session(
            args.source, args.output, args.identity_a, args.identity_b,
            args.start_frame, args.end_frame, args.user)
        session = load_editing_session(path)
        operation = session.edits[-1]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation": "swap_identities",
            "identity_a": operation["identity_a"],
            "identity_b": operation["identity_b"],
            "start_imagej_frame": operation["start_imagej_frame"],
            "end_imagej_frame": operation["end_imagej_frame"],
            "operation_changed_pixels": operation["changed_pixels"],
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "assign-identities":
        path = save_identity_reassignment_session(
            args.source, args.output, args.source_identity,
            args.target_identity, args.start_frame, args.end_frame, args.user)
        session = load_editing_session(path)
        operation = session.edits[-1]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation": "reassign_identities",
            "source_identities": operation["source_identities"],
            "target_identity": operation["target_identity"],
            "start_imagej_frame": operation["start_imagej_frame"],
            "end_imagej_frame": operation["end_imagej_frame"],
            "reassigned_pixels": operation["reassigned_pixels"],
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "remove-identity":
        path = save_identity_exclusions_session(
            args.source, args.output, args.identity, args.user)
        session = load_editing_session(path)
        operation_count = len(args.identity)
        operations = session.edits[-operation_count:]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation": "exclude_identity_batch",
            "operation_count": operation_count,
            "identities": [operation["identity"] for operation in operations],
            "removed_pixels": sum(
                operation["removed_pixels"] for operation in operations),
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "delete-between-frames":
        path = save_identity_interval_deletion_session(
            args.source, args.output, args.identity,
            args.start_frame, args.end_frame, args.user)
        session = load_editing_session(path)
        operation = session.edits[-1]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation": "delete_identity_interval",
            "identities": operation["identities"],
            "start_imagej_frame": operation["start_imagej_frame"],
            "end_imagej_frame": operation["end_imagej_frame"],
            "deleted_pixels": operation["deleted_pixels"],
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "expand-identities":
        path = save_identity_expansion_session(
            args.source, args.output, args.identity, args.radius,
            args.radius_unit, args.start_frame, args.end_frame, args.user)
        session = load_editing_session(path)
        operation = session.edits[-1]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation": "expand_identities",
            "identities": operation["identities"],
            "radius": operation["radius"],
            "effective_distance_unit": operation["effective_distance_unit"],
            "added_pixels": operation["added_pixels"],
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "apply-batch":
        requests = load_edit_batch(args.batch)
        path = save_edit_batch_session(
            args.source, args.output, requests, args.user)
        session = load_editing_session(path)
        operations = session.edits[-len(requests):]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation_count": len(operations),
            "operation_types": [
                operation["type"] for operation in operations],
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "propose-new-cell":
        from manual_new_cells import (
            NewCellRequest, allocate_new_identity, default_crop_side_px,
            editing_state_sha256, proposal_to_draft_job,
            propose_new_identity, save_new_cell_draft)

        bundle, labels, _provenance, excluded, _edits, _parent = \
            _editing_source(args.source)
        anchors: dict[int, tuple[int, int]] = {}
        for token in args.anchor:
            parts = token.split(",")
            if len(parts) != 3:
                raise ValueError(
                    "--anchor must use FRAME,Y,X, for example 24,183,271")
            try:
                imagej_frame, y, x = map(int, parts)
            except ValueError as error:
                raise ValueError(
                    "--anchor FRAME,Y,X values must be integers") from error
            if imagej_frame in anchors:
                raise ValueError(
                    f"duplicate new-cell anchor on ImageJ frame {imagej_frame}")
            anchors[imagej_frame] = (y, x)
        identity = (allocate_new_identity(labels, excluded)
                    if args.identity is None
                    else int(args.identity))
        if identity in excluded:
            raise ValueError(
                f"new identity {identity} is reserved by the exclusion history")
        end_frame = len(labels) if args.end_frame is None else args.end_frame
        request = NewCellRequest(
            identity=identity,
            start_imagej_frame=args.start_frame,
            end_imagej_frame=end_frame,
            anchor_points=anchors,
            method=args.method,
            crop_side_px=(default_crop_side_px(labels)
                          if args.crop_side_px is None else args.crop_side_px),
            threshold_z=args.threshold_z,
            threshold_smoothing=args.threshold_smoothing,
            minimum_area_px=args.minimum_area_px,
            start_reason=args.start_reason,
            end_reason=args.end_reason,
        )
        lag = None
        lag_path = bundle.evidence_paths.get("lag_float")
        if lag_path is not None:
            lag_source = _read_tiff(lag_path)
            lag_indices = np.asarray(bundle.manifest["assets"]["lag_float"][
                "label_frame_to_source"], dtype=np.intp)
            lag = lag_source[lag_indices]
        proposal = propose_new_identity(
            labels, bundle.registered_raw, request, bundle.unclaimed_labels,
            lag, bundle.registered_raw_indices)
        job = proposal_to_draft_job(proposal, request, set())
        path = save_new_cell_draft(
            args.output, editing_state_sha256(labels), [job])
        result = {
            "draft": str(path),
            "status": "needs_review",
            "identity": identity,
            "lifetime_imagej_frames": [args.start_frame, end_frame],
            "blocked_imagej_frames": [
                row["imagej_frame"] for row in proposal.per_frame
                if not row["valid"]],
            "warning_count": len(proposal.warnings),
        }
    elif args.command == "commit-new-cell-draft":
        from manual_new_cells import (
            build_add_identity_operation, editing_state_sha256,
            load_new_cell_draft, proposal_from_draft_job)

        bundle, labels, _provenance, _excluded, _edits, _parent = \
            _editing_source(args.source)
        document = load_new_cell_draft(
            args.draft, editing_state_sha256(labels))
        operations: list[dict[str, Any]] = []
        identities: list[int] = []
        for index, row in enumerate(document["jobs"], start=1):
            request, proposal, accepted, linked, split_operations = \
                proposal_from_draft_job(
                    labels, bundle.registered_raw, row)
            if proposal is None:
                raise ValueError(
                    f"new-cell draft job {index} has no generated proposal")
            operations.append(build_add_identity_operation(
                proposal, accepted, linked, parent_labels=labels))
            operations.extend(split_operations)
            identities.append(request.identity)
        if not operations:
            raise ValueError("new-cell draft has no jobs to commit")
        path = save_edit_batch_session(
            args.source, args.output, operations, args.user,
            mode="new_identity_batch")
        session = load_editing_session(path)
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "new_identities": identities,
            "operation_count": len(operations),
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "preview-filters":
        from manual_editing_filters import load_filter_set, preview_track_filters
        filter_set = load_filter_set(args.filters)
        path = preview_track_filters(
            args.source, filter_set, args.output, args.operation_count)
        report = _json_object(path)
        result = {
            "filter_report": str(path),
            "status": "previewed",
            **report["summary"],
            "area_unit": report["metric_metadata"]["area_unit"],
            "speed_unit": report["metric_metadata"]["speed_unit"],
        }
    elif args.command == "remove-by-filters":
        from manual_editing_filters import (load_filter_set,
                                            save_filtered_removal_session)
        filter_set = load_filter_set(args.filters)
        path = save_filtered_removal_session(
            args.source, args.output, filter_set, args.user,
            args.operation_count)
        session = load_editing_session(path)
        context = session.manifest["batch_action"]["context"]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "selected_identities": context["selected_identities"],
            "selected_identity_count": len(context["selected_identities"]),
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "apply-by-filters":
        from manual_editing_filters import (load_filter_set,
                                            save_filtered_operation_session)
        filter_set = load_filter_set(args.filters)
        operation_type = {
            "remove": "exclude_identity",
            "delete-frames": "delete_identity_interval",
            "expand": "expand_identities",
        }[args.operation]
        if operation_type == "expand_identities":
            if args.radius is None:
                raise ValueError("--radius is required for expansion")
            parameters = {
                "radius": args.radius,
                "radius_unit": args.radius_unit,
                "start_imagej_frame": args.start_frame,
                "end_imagej_frame": args.end_frame,
            }
        elif operation_type == "delete_identity_interval":
            if args.end_frame is None:
                raise ValueError("--end-frame is required for frame-range deletion")
            if args.radius is not None:
                raise ValueError("--radius only applies to expansion")
            parameters = {
                "start_imagej_frame": args.start_frame,
                "end_imagej_frame": args.end_frame,
            }
        else:
            if args.radius is not None:
                raise ValueError("--radius only applies to expansion")
            parameters = {}
        path = save_filtered_operation_session(
            args.source, args.output, filter_set, operation_type, parameters,
            args.user, args.operation_count)
        session = load_editing_session(path)
        context = session.manifest["batch_action"]["context"]
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "operation": operation_type,
            "selected_identities": context["selected_identities"],
            "selected_identity_count": len(context["selected_identities"]),
            "session_changed_pixels": session.manifest[
                "summary"]["changed_pixels"],
        }
    elif args.command == "build-review":
        path = build_editing_review(
            args.session, args.output, args.padding_px)
        review = _json_object(path)
        result = {
            "review": str(path),
            "status": "validated",
            "imagej_frames": review["imagej_frames"],
            "representative_imagej_frames": review[
                "representative_imagej_frames"],
            "changed_pixels": review["summary"]["changed_pixels"],
        }
    elif args.command == "list-history":
        result = editing_history_summary(args.source)
        result["status"] = "valid"
    elif args.command == "restore-checkpoint":
        source_history = EditingHistoryController(args.source)
        path = save_history_checkpoint(
            args.source, args.output, args.operation_count, args.user)
        session = load_editing_session(path)
        history_action = session.manifest.get("history_action", {})
        result = {
            "session": str(path),
            "status": "round_trip_verified",
            "restored_operation_count": len(session.edits),
            "later_operations_preserved_in_parent": history_action.get(
                "later_operations_preserved_in_parent",
                len(source_history.operations) - args.operation_count),
        }
    elif args.command == "history-controls":
        from manual_editing_history_ui import launch_history_controls
        selected = launch_history_controls(args.source)
        result = {
            "status": "continued" if selected is not None else "cancelled",
            "selected_source": None if selected is None else str(selected),
        }
    elif args.command == "edit":
        from manual_editing_ui import launch_manual_editor
        selected = launch_manual_editor(args.source)
        result = {
            "status": "closed",
            "selected_source": str(selected),
        }
    elif args.command == "validate-bundle":
        bundle = load_editing_bundle(args.bundle)
        result = {
            "bundle": str(bundle.manifest_path), "status": "valid",
            "shape": list(bundle.canonical_labels.shape),
            "active_identities": bundle.manifest["summary"]["active_identities"],
        }
    elif args.command == "validate-session":
        session = load_editing_session(args.session)
        result = {
            "session": str(session.manifest_path), "status": "valid",
            "edit_operations": len(session.edits),
            "changed_pixels": int(np.count_nonzero(
                session.curated_labels != session.bundle.canonical_labels)),
        }
    elif args.command == "freeze-package":
        from manual_portable_package import (freeze_portable_package,
                                             load_portable_package)
        path = freeze_portable_package(
            args.source, args.output, args.operation_count, args.container)
        package = load_portable_package(path)
        result = {
            "package": str(path),
            "status": "validated",
            "package_id": package.manifest["package_id"],
            "profile": package.manifest["profile"],
            "archive_sha256": package.archive_sha256,
            **package.manifest["summary"],
        }
    elif args.command == "validate-package":
        from manual_portable_package import load_portable_package
        package = load_portable_package(args.package)
        result = {
            "package": str(package.source_path),
            "status": "valid",
            "package_id": package.manifest["package_id"],
            "profile": package.manifest["profile"],
            "archive_sha256": package.archive_sha256,
            **package.manifest["summary"],
        }
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
