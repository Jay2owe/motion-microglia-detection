from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from common import ROOT, save_stack, sha256, utc_now, write_json
from manual_editing import (BUNDLE_FILENAME, BUNDLE_SCHEMA,
                            BUNDLE_SCHEMA_VERSION, MANUAL_PROVENANCE_VALUES,
                            SESSION_FILENAME, SESSION_SCHEMA,
                            SESSION_SCHEMA_VERSION, EditingBundle,
                            _editing_source, _json_object, _path_record,
                            _read_tiff, _tiff_record, load_editing_bundle,
                            load_editing_session, rebuild_identity_tables,
                            replay_edit_log, save_edit_batch_session,
                            save_history_checkpoint)


PACKAGE_SCHEMA = "motion.portable-editing-package"
PACKAGE_SCHEMA_VERSION = 1
PACKAGE_FILENAME = "motion-editing-package.json"
PACKAGE_EXTENSION = ".motionpkg"
PACKAGE_PROFILE = "full_editing"


@dataclass(frozen=True)
class PortableEditingPackage:
    source_path: Path
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    entry_manifest: Path
    archive_sha256: str | None


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(raw: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise ValueError("portable package path must be non-empty text")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.anchor or ".." in path.parts \
            or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"portable package path is unsafe: {raw!r}")
    if ":" in path.parts[0] or "\\" in raw:
        raise ValueError(f"portable package path is not portable: {raw!r}")
    return path


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _remove_staging(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or not _within(resolved, parent):
        raise ValueError(f"refusing to remove staging path outside {parent}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def _write_motion_stack(
        path: Path, array: np.ndarray, frame_interval_min: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path, np.asarray(array), imagej=True, compression="zlib",
        metadata={
            "axes": "TCYX", "finterval": frame_interval_min * 60.0,
            "tunit": "sec", "unit": "pixel",
        })


def _source_lineage(manifest_path: Path, operation_count: int,
                    source_operation_count: int) -> dict[str, Any]:
    document = _json_object(manifest_path)
    lineage: dict[str, Any] = {
        "source_schema": document.get("schema"),
        "source_schema_version": document.get("schema_version"),
        "source_manifest_name": manifest_path.name,
        "source_manifest_sha256": sha256(manifest_path),
        "selected_operation_count": int(operation_count),
        "source_operation_count": int(source_operation_count),
    }
    parent = document.get("parent_session")
    ancestor_hashes: list[str] = []
    if isinstance(parent, dict) and isinstance(parent.get("sha256"), str):
        ancestor_hashes.append(parent["sha256"])
    if ancestor_hashes:
        lineage["declared_ancestor_session_sha256"] = ancestor_hashes
    batch_action = document.get("batch_action")
    if isinstance(batch_action, dict):
        first_id = batch_action.get("first_operation_id")
        last_id = batch_action.get("last_operation_id")
        if isinstance(first_id, int) and isinstance(last_id, int) \
                and 1 <= first_id <= last_id <= operation_count:
            lineage["source_batch_action"] = copy.deepcopy(batch_action)
    return lineage


def _write_portable_bundle(bundle: EditingBundle, package_root: Path) -> Path:
    bundle_dir = package_root / "payload" / "bundle"
    inputs = bundle_dir / "inputs"
    evidence = bundle_dir / "evidence"
    inputs.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    interval = float(bundle.manifest["time"]["frame_interval_min"])
    frames = len(bundle.canonical_labels)
    internal_indices = list(range(frames))

    canonical_path = inputs / "canonical_labels.tif"
    raw_path = evidence / "registered_raw_aligned.tif"
    motion_path = evidence / "motion_composite_aligned.tif"
    save_stack(canonical_path, bundle.canonical_labels, interval)
    save_stack(raw_path, bundle.registered_raw, interval)

    motion_source = _read_tiff(bundle.evidence_paths["motion_composite"])
    motion_indices = np.asarray(
        bundle.manifest["assets"]["motion_composite"][
            "label_frame_to_source"], dtype=np.intp)
    motion_aligned = motion_source[motion_indices]
    if len(motion_aligned) != frames:
        raise ValueError("portable motion evidence did not align to label frames")
    _write_motion_stack(motion_path, motion_aligned, interval)

    assets: dict[str, dict[str, Any]] = {
        "canonical_labels": _tiff_record(
            canonical_path, bundle_dir, True, "TYX"),
        "registered_raw": _tiff_record(
            raw_path, bundle_dir, True, "TYX", internal_indices),
        "motion_composite": _tiff_record(
            motion_path, bundle_dir, True, "TCYX", internal_indices),
    }
    if bundle.unclaimed_labels is not None:
        unclaimed_path = inputs / "unclaimed_labels.tif"
        save_stack(unclaimed_path, bundle.unclaimed_labels, interval)
        assets["unclaimed_labels"] = _tiff_record(
            unclaimed_path, bundle_dir, True, "TYX")
    lag_source_path = bundle.evidence_paths.get("lag_float")
    if lag_source_path is not None:
        lag_source = _read_tiff(lag_source_path)
        lag_indices = np.asarray(
            bundle.manifest["assets"]["lag_float"][
                "label_frame_to_source"], dtype=np.intp)
        lag_aligned = lag_source[lag_indices]
        if lag_aligned.shape != bundle.canonical_labels.shape:
            raise ValueError("portable lag evidence did not align to label frames")
        lag_path = evidence / "lag_float_aligned.tif"
        save_stack(lag_path, lag_aligned, interval)
        assets["lag_float"] = _tiff_record(
            lag_path, bundle_dir, True, "TYX", internal_indices)

    parent = copy.deepcopy(bundle.manifest.get("parent", {}))
    if not isinstance(parent, dict):
        parent = {}
    parent_manifest = parent.pop("manifest", None)
    if isinstance(parent_manifest, dict) \
            and isinstance(parent_manifest.get("sha256"), str):
        parent["origin_manifest_sha256"] = parent_manifest["sha256"]
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "status": "done",
        "created_at_utc": utc_now(),
        "stem": bundle.manifest["stem"],
        "parent": parent,
        "assets": assets,
        "time": {
            "frame_interval_min": interval,
            "label_frames": frames,
            "mapping_rule": (
                "portable evidence contains one internal source frame per label frame"),
        },
        "spatial_calibration": copy.deepcopy(
            bundle.manifest["spatial_calibration"]),
        "manual_provenance_values": MANUAL_PROVENANCE_VALUES,
        "software": {
            **copy.deepcopy(bundle.manifest.get("software", {})),
            "portable_package_schema_version": PACKAGE_SCHEMA_VERSION,
        },
        "summary": {
            "shape": list(map(int, bundle.canonical_labels.shape)),
            "dtype": bundle.canonical_labels.dtype.name,
            "active_identities": int(len(
                set(map(int, np.unique(bundle.canonical_labels))) - {0})),
            "assigned_pixels": int(np.count_nonzero(bundle.canonical_labels)),
            "unclaimed_pixels": (
                None if bundle.unclaimed_labels is None
                else int(np.count_nonzero(bundle.unclaimed_labels))),
        },
    }
    manifest_path = bundle_dir / BUNDLE_FILENAME
    write_json(manifest_path, manifest)
    load_editing_bundle(manifest_path)
    return manifest_path


def _write_portable_session(
        portable_bundle_path: Path, operations: list[dict[str, Any]],
        package_root: Path, lineage: dict[str, Any]) -> Path:
    bundle = load_editing_bundle(portable_bundle_path)
    curated, provenance, excluded = replay_edit_log(bundle, operations)
    session_dir = package_root / "payload"
    out_dir = session_dir / "session" / "out"
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
    pd.DataFrame([{
        "identity": identity,
        "reason": "excluded by replayed edit log",
    } for identity in sorted(excluded)], columns=(
        "identity", "reason")).to_csv(exclusions_path, index=False)
    write_json(edits_path, {"schema_version": 1, "operations": operations})
    frame_table, tracks = rebuild_identity_tables(
        curated, bundle.registered_raw, bundle.registered_raw_indices,
        provenance, excluded)
    frame_table.to_csv(frame_path, index=False)
    tracks.to_csv(tracks_path, index=False)
    manifest = {
        "schema": SESSION_SCHEMA,
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "done",
        "created_at_utc": utc_now(),
        "mode": "portable_checkpoint",
        "stem": bundle.manifest["stem"],
        "parent_bundle": _path_record(
            portable_bundle_path, session_dir, True),
        "parent_canonical_sha256": bundle.manifest[
            "assets"]["canonical_labels"]["sha256"],
        "manual_provenance_values": MANUAL_PROVENANCE_VALUES,
        "portable_lineage": lineage,
        "outputs": {
            "curated_labels": _tiff_record(
                curated_path, session_dir, True, "TYX"),
            "manual_provenance": _tiff_record(
                provenance_path, session_dir, True, "TYX"),
            "excluded_identities": _path_record(
                exclusions_path, session_dir, True),
            "edits": _path_record(edits_path, session_dir, True),
            "frame_identities": _path_record(frame_path, session_dir, True),
            "tracks": _path_record(tracks_path, session_dir, True),
        },
        "summary": {
            "edit_operations": len(operations),
            "excluded_identities": len(excluded),
            "changed_pixels": int(np.count_nonzero(
                curated != bundle.canonical_labels)),
            "array_equal_to_parent": bool(np.array_equal(
                curated, bundle.canonical_labels)),
            "round_trip_verified": False,
        },
    }
    source_batch = lineage.get("source_batch_action")
    if isinstance(source_batch, dict):
        manifest["batch_action"] = copy.deepcopy(source_batch)
    manifest_path = session_dir / SESSION_FILENAME
    write_json(manifest_path, manifest)
    load_editing_session(manifest_path)
    manifest["summary"]["round_trip_verified"] = True
    write_json(manifest_path, manifest)
    load_editing_session(manifest_path)
    return manifest_path


def _write_software_snapshot(package_root: Path) -> Path:
    software_dir = package_root / "software"
    source_dir = software_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    module_names = (
        "common.py", "manual_editing.py", "manual_editing_ui.py",
        "manual_editing_theme.py", "manual_editing_filters.py",
        "manual_editing_filter_ui.py", "manual_editing_history_ui.py",
        "manual_splitting.py", "manual_splitting_ui.py",
        "manual_new_cells.py", "manual_new_cells_ui.py",
        "manual_retuning.py", "manual_retuning_ui.py",
        "manual_portable_package.py",
    )
    records: list[dict[str, Any]] = []
    for name in module_names:
        source = ROOT / "code" / name
        if not source.is_file():
            continue
        destination = source_dir / name
        shutil.copy2(source, destination)
        records.append({
            "name": name,
            "sha256": sha256(destination),
            "bytes": int(destination.stat().st_size),
        })
    inventory_path = software_dir / "software-inventory.json"
    write_json(inventory_path, {
        "schema": "motion.portable-software-snapshot",
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "execution_policy": (
            "archival source only; importing a package never executes bundled code"),
        "modules": records,
    })
    return inventory_path


def _role_for_path(path: str) -> str:
    if path.startswith("payload/bundle/inputs/"):
        return "canonical_input"
    if path.startswith("payload/bundle/evidence/"):
        return "editing_evidence"
    if path.startswith("payload/session/out/"):
        return "session_output"
    if path.startswith("software/source/"):
        return "software_source_snapshot"
    if path.startswith("software/"):
        return "software_inventory"
    if path.endswith(BUNDLE_FILENAME):
        return "embedded_bundle_manifest"
    if path.endswith(SESSION_FILENAME):
        return "embedded_session_manifest"
    return "package_payload"


def _build_inventory(package_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name == PACKAGE_FILENAME:
            continue
        relative = path.relative_to(package_root).as_posix()
        _safe_relative_path(relative)
        rows.append({
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
            "role": _role_for_path(relative),
        })
    return rows


def _build_package_directory(
        source_path: Path, package_root: Path,
        operation_count: int | None) -> Path:
    bundle, _labels, _provenance, _excluded, edits, parent_session = \
        _editing_source(source_path)
    count = len(edits) if operation_count is None else int(operation_count)
    if count < 0 or count > len(edits):
        raise ValueError(f"portable checkpoint must be between 0 and {len(edits)}")
    selected_operations = copy.deepcopy(edits[:count])
    source_manifest = parent_session or bundle.manifest_path
    lineage = _source_lineage(source_manifest, count, len(edits))
    portable_bundle_path = _write_portable_bundle(bundle, package_root)
    has_session = parent_session is not None
    entry_path = (portable_bundle_path if not has_session
                  else _write_portable_session(
                      portable_bundle_path, selected_operations,
                      package_root, lineage))
    _write_software_snapshot(package_root)
    inventory = _build_inventory(package_root)
    entry_relative = entry_path.relative_to(package_root).as_posix()
    entry_schema = SESSION_SCHEMA if has_session else BUNDLE_SCHEMA
    package_manifest = {
        "schema": PACKAGE_SCHEMA,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "done",
        "created_at_utc": utc_now(),
        "package_id": str(uuid.uuid4()),
        "profile": PACKAGE_PROFILE,
        "immutable": True,
        "entry": {
            "schema": entry_schema,
            "path": entry_relative,
            "sha256": sha256(entry_path),
        },
        "lineage": lineage,
        "inventory": inventory,
        "inventory_sha256": _canonical_json_sha256(inventory),
        "summary": {
            "stem": bundle.manifest["stem"],
            "label_frames": int(len(bundle.canonical_labels)),
            "operation_count": count,
            "payload_files": len(inventory),
            "payload_bytes": int(sum(row["bytes"] for row in inventory)),
            "entry_schema": entry_schema,
        },
    }
    manifest_path = package_root / PACKAGE_FILENAME
    write_json(manifest_path, package_manifest)
    load_portable_package(package_root)
    return manifest_path


def _validate_inventory(
        root: Path, manifest: dict[str, Any], verify_hashes: bool) -> None:
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or any(
            not isinstance(row, dict) for row in inventory):
        raise ValueError("portable package inventory is invalid")
    if _canonical_json_sha256(inventory) != manifest.get("inventory_sha256"):
        raise ValueError("portable package inventory fingerprint differs")
    declared: set[str] = set()
    for row in inventory:
        raw_path = row.get("path")
        relative = _safe_relative_path(raw_path)
        if raw_path in declared:
            raise ValueError(f"duplicate portable package inventory path: {raw_path}")
        declared.add(raw_path)
        path = (root / Path(*relative.parts)).resolve()
        if not _within(path, root) or not path.is_file() or path.is_symlink():
            raise ValueError(f"portable package payload is missing or unsafe: {raw_path}")
        expected_bytes = row.get("bytes")
        expected_hash = row.get("sha256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) \
                or expected_bytes < 0 or path.stat().st_size != expected_bytes:
            raise ValueError(f"portable package byte count differs: {raw_path}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"portable package fingerprint is invalid: {raw_path}")
        if verify_hashes and sha256(path) != expected_hash:
            raise ValueError(f"portable package fingerprint differs: {raw_path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != PACKAGE_FILENAME}
    if actual != declared:
        missing = sorted(declared - actual)
        extra = sorted(actual - declared)
        raise ValueError(
            f"portable package inventory membership differs; missing={missing}, "
            f"extra={extra}")


def _load_package_directory(
        root: Path, source_path: Path, verify_hashes: bool,
        archive_hash: str | None = None) -> PortableEditingPackage:
    resolved = root.resolve()
    manifest_path = resolved / PACKAGE_FILENAME
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != PACKAGE_SCHEMA \
            or manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported portable editing package schema")
    if manifest.get("status") != "done" or manifest.get("immutable") is not True \
            or manifest.get("profile") != PACKAGE_PROFILE:
        raise ValueError("portable editing package is incomplete or unsupported")
    _validate_inventory(resolved, manifest, verify_hashes)
    entry = manifest.get("entry")
    if not isinstance(entry, dict):
        raise ValueError("portable editing package has no entry manifest")
    relative = _safe_relative_path(entry.get("path"))
    entry_path = (resolved / Path(*relative.parts)).resolve()
    if not _within(entry_path, resolved) or not entry_path.is_file():
        raise ValueError("portable editing package entry is missing")
    if verify_hashes and sha256(entry_path) != entry.get("sha256"):
        raise ValueError("portable editing package entry fingerprint differs")
    entry_document = _json_object(entry_path)
    if entry_document.get("schema") != entry.get("schema"):
        raise ValueError("portable editing package entry schema differs")
    if entry.get("schema") == BUNDLE_SCHEMA:
        loaded = load_editing_bundle(entry_path, verify_hashes=verify_hashes)
        operation_count = 0
        frames = len(loaded.canonical_labels)
    elif entry.get("schema") == SESSION_SCHEMA:
        loaded_session = load_editing_session(
            entry_path, verify_hashes=verify_hashes)
        operation_count = len(loaded_session.edits)
        frames = len(loaded_session.curated_labels)
    else:
        raise ValueError("portable editing package entry schema is unsupported")
    summary = manifest.get("summary")
    if not isinstance(summary, dict) \
            or summary.get("operation_count") != operation_count \
            or summary.get("label_frames") != frames \
            or summary.get("payload_files") != len(manifest["inventory"]) \
            or summary.get("payload_bytes") != sum(
                row["bytes"] for row in manifest["inventory"]):
        raise ValueError("portable editing package summary differs")
    return PortableEditingPackage(
        source_path=source_path.resolve(), root=resolved,
        manifest_path=manifest_path, manifest=manifest,
        entry_manifest=entry_path, archive_sha256=archive_hash)


def _safe_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    names: set[str] = set()
    for member in members:
        raw = member.filename.rstrip("/")
        if not raw:
            continue
        relative = _safe_relative_path(raw)
        normalized = relative.as_posix()
        if normalized in names:
            raise ValueError(f"portable archive contains a duplicate path: {normalized}")
        names.add(normalized)
        mode = (member.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"portable archive contains a symbolic link: {normalized}")
    return members


def _extract_archive_to_cache(path: Path, archive_hash: str) -> Path:
    cache_parent = Path(tempfile.gettempdir()) / "motion-portable-package-cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    target = cache_parent / archive_hash
    if target.is_dir():
        return target
    staging = cache_parent / f".{archive_hash}.{uuid.uuid4().hex}.partial"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
            members = _safe_zip_members(archive)
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"portable archive integrity failed at {bad}")
            for member in members:
                raw = member.filename.rstrip("/")
                if not raw:
                    continue
                relative = _safe_relative_path(raw)
                destination = staging / Path(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
        _load_package_directory(
            staging, path, verify_hashes=True, archive_hash=archive_hash)
        try:
            os.replace(staging, target)
        except FileExistsError:
            _remove_staging(staging, cache_parent)
    except BaseException:
        if staging.exists():
            _remove_staging(staging, cache_parent)
        raise
    return target


def is_portable_package(path: Path) -> bool:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        manifest = resolved / PACKAGE_FILENAME
        if not manifest.is_file():
            return False
        try:
            return _json_object(manifest).get("schema") == PACKAGE_SCHEMA
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    return resolved.is_file() and (
        resolved.suffix.lower() == PACKAGE_EXTENSION
        or zipfile.is_zipfile(resolved))


def load_portable_package(
        path: Path, verify_hashes: bool = True) -> PortableEditingPackage:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        return _load_package_directory(
            resolved, resolved, verify_hashes=verify_hashes)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not zipfile.is_zipfile(resolved):
        raise ValueError("portable editing package is not a ZIP64 archive")
    archive_hash = sha256(resolved)
    cache_root = _extract_archive_to_cache(resolved, archive_hash)
    return _load_package_directory(
        cache_root, resolved, verify_hashes=verify_hashes,
        archive_hash=archive_hash)


def resolve_portable_entry(path: Path) -> Path:
    return load_portable_package(path).entry_manifest


def _write_archive(package_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
            archive_path, "w", allowZip64=True) as archive:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package_root).as_posix()
            compression = (
                zipfile.ZIP_STORED if path.suffix.lower() in {
                    ".tif", ".tiff", ".zip"} else zipfile.ZIP_DEFLATED)
            archive.write(path, arcname=relative, compress_type=compression)
    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
        _safe_zip_members(archive)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"portable archive integrity failed at {bad}")
        expected = {
            path.relative_to(package_root).as_posix()
            for path in package_root.rglob("*") if path.is_file()}
        actual = {member.filename.rstrip("/") for member in archive.infolist()
                  if member.filename.rstrip("/")}
        if actual != expected:
            raise ValueError("portable archive members differ from staged package")


def freeze_portable_package(
        source_path: Path, output_path: Path,
        operation_count: int | None = None,
        container: str = "archive") -> Path:
    """Freeze an editing source into a fully self-contained immutable package."""
    if container not in {"archive", "directory"}:
        raise ValueError("portable package container must be archive or directory")
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    if source.is_dir() and (source / PACKAGE_FILENAME).is_file() \
            and _within(destination, source):
        raise ValueError(
            "a child package cannot be saved inside its immutable source package")
    if destination.exists():
        raise FileExistsError(
            f"immutable portable package already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", suffix=".partial",
        dir=destination.parent))
    temporary_archive = destination.parent / (
        f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        _build_package_directory(
            source, staging, operation_count)
        if container == "directory":
            os.replace(staging, destination)
        else:
            _write_archive(staging, temporary_archive)
            os.replace(temporary_archive, destination)
            _remove_staging(staging, destination.parent)
    except BaseException:
        if staging.exists():
            _remove_staging(staging, destination.parent)
        if temporary_archive.exists():
            _remove_staging(temporary_archive, destination.parent)
        raise
    return destination


def save_portable_edit_batch(
        source_package: Path, output_package: Path,
        operations: list[dict[str, Any]], user: str | None = None,
        operation_count: int | None = None,
        mode: str = "edit_batch",
        batch_context: dict[str, Any] | None = None) -> Path:
    """Apply edits to a read-only package and freeze the child atomically."""
    package = load_portable_package(source_package)
    destination = Path(output_package).resolve()
    if package.source_path.is_dir() and _within(
            destination, package.source_path):
        raise ValueError(
            "a child package cannot be saved inside its immutable source package")
    destination.parent.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.edit.", dir=destination.parent))
    try:
        session_path = save_edit_batch_session(
            package.entry_manifest, working / "session", operations, user,
            operation_count=operation_count, mode=mode,
            batch_context=batch_context)
        return freeze_portable_package(session_path, destination)
    finally:
        if working.exists():
            _remove_staging(working, destination.parent)


def save_portable_history_checkpoint(
        source_package: Path, output_package: Path, operation_count: int,
        user: str | None = None) -> Path:
    """Freeze an earlier package checkpoint without retaining cache dependencies."""
    package = load_portable_package(source_package)
    destination = Path(output_package).resolve()
    if package.source_path.is_dir() and _within(
            destination, package.source_path):
        raise ValueError(
            "a child package cannot be saved inside its immutable source package")
    destination.parent.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.branch.", dir=destination.parent))
    try:
        checkpoint = save_history_checkpoint(
            package.entry_manifest, working / "session", operation_count, user)
        return freeze_portable_package(checkpoint, destination)
    finally:
        if working.exists():
            _remove_staging(working, destination.parent)
