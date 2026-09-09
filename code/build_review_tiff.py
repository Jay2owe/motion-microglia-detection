"""Canonical, issue-independent full-field review TIFF renderer.

This is the only command new issue workflows should use to lay out review
panels.  The rendering conventions themselves live in ``review_rendering``
and ``common`` so the command and regression tests share the same code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import tifffile

from common import (change_counts, change_overlay, save_rgb_stack, sha256,
                    validate_labels, write_json)
from review_rendering import header, unique_outline


RENDERER_VERSION = 4
HEADER_HEIGHT = 24
CHANGE_RED = np.array([255, 45, 45], np.uint8)


def _align_source(stack: np.ndarray, frames: int, offset: int,
                  name: str, *, allow_transition_stack: bool = False) -> np.ndarray:
    """Apply the established review-frame alignment without changing it."""
    if offset < 0:
        raise ValueError(f"{name} frame offset cannot be negative")
    if len(stack) == frames:
        return stack
    if allow_transition_stack and frames > 1 and len(stack) == frames - 1:
        # Motion evidence may encode the T-1 transitions of an already trimmed
        # T-frame movie. Associate transition i with frame i and repeat the last
        # transition for the boundary frame, matching the existing clipped-end
        # convention used for longer full-source motion stacks.
        return np.concatenate((stack, stack[-1:]), axis=0)
    if len(stack) >= frames:
        indices = np.minimum(offset + np.arange(frames), len(stack) - 1)
        return stack[indices]
    raise ValueError(
        f"cannot align {name} ({len(stack)} frames) to {frames} review frames")


def _validate_inputs(baseline: np.ndarray, candidate: np.ndarray,
                     raw: np.ndarray, motion: np.ndarray) -> None:
    validate_labels(baseline)
    validate_labels(candidate)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"baseline {baseline.shape} and candidate {candidate.shape} differ")
    if raw.ndim != 3 or raw.shape != baseline.shape:
        raise ValueError(
            f"aligned raw must be {baseline.shape}, found {raw.shape}")
    if motion.ndim != 4 or motion.shape[0] != len(candidate):
        raise ValueError(
            "aligned motion must be a (T,C,Y,X) array with one entry per frame")
    if motion.shape[1] < 5 or motion.shape[2:] != candidate.shape[1:]:
        raise ValueError(
            "motion must contain at least five channels at the label resolution")


def render_review_arrays(
        baseline: np.ndarray,
        candidate: np.ndarray,
        raw: np.ndarray,
        motion: np.ndarray,
        *,
        baseline_title: str,
        candidate_title: str,
        source_frame_offset: int,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render the standard four-panel movie and candidate-only companion."""
    _validate_inputs(baseline, candidate, raw, motion)
    before = unique_outline(raw, baseline)
    after = unique_outline(raw, candidate)
    difference = change_overlay(raw, baseline, candidate)
    motion_view = unique_outline(raw, candidate, motion)
    counts = change_counts(baseline, candidate)
    notes = [f"{pixels} px changed" if pixels else "unchanged"
             for pixels in counts["per_frame"]]

    after_with_header = header(
        after, candidate_title, source_frame_offset)
    full = np.concatenate([
        header(before, baseline_title, source_frame_offset),
        after_with_header,
        header(difference, f"CHANGE: RED = {counts['pixels']} px MOVIE",
               source_frame_offset, notes),
        header(motion_view, f"{candidate_title} + MOTION",
               source_frame_offset),
    ], axis=2)
    return full, after_with_header, counts


def _validate_render(full: np.ndarray, candidate_outline: np.ndarray,
                     baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    frames, height, width = candidate.shape
    expected_full = (frames, height + HEADER_HEIGHT, width * 4, 3)
    expected_outline = (frames, height + HEADER_HEIGHT, width, 3)
    if full.shape != expected_full:
        raise AssertionError(f"full review shape {full.shape} != {expected_full}")
    if candidate_outline.shape != expected_outline:
        raise AssertionError(
            f"candidate outline shape {candidate_outline.shape} != {expected_outline}")

    change_field = full[:, HEADER_HEIGHT:, width * 2:width * 3]
    red = np.all(change_field == CHANGE_RED, axis=-1)
    changed = baseline != candidate
    if not np.array_equal(red, changed):
        raise AssertionError("red change pixels do not exactly equal changed labels")
    non_red = change_field[~red]
    if non_red.size and not np.all(
            (non_red[:, 0] == non_red[:, 1])
            & (non_red[:, 1] == non_red[:, 2])):
        raise AssertionError("the change field contains a non-grey, non-red pixel")
    return {
        "full_review_shape": list(expected_full),
        "candidate_outline_shape": list(expected_outline),
        "red_pixels_equal_changed_labels": True,
        "non_red_change_field_is_greyscale": True,
    }


def _code_hashes() -> dict[str, str]:
    code_dir = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), code_dir / "review_rendering.py",
             code_dir / "common.py", code_dir / "pipeline.py"]
    return {path.name: sha256(path) for path in paths}


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _staging_parent(output_dir: Path,
                    system_temp: Path | None = None) -> Path | None:
    """Choose same-volume staging when system temp is on another drive."""
    temp = (Path(tempfile.gettempdir()) if system_temp is None
            else Path(system_temp)).resolve()
    output = Path(output_dir).resolve()
    return None if temp.drive.casefold() == output.drive.casefold() \
        else output.parent


def _verified_existing(output_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    manifest_path = output_dir / "review_manifest.json"
    if not output_dir.exists():
        return None
    if not manifest_path.is_file():
        raise FileExistsError(
            f"review output exists without a completion manifest: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise FileExistsError(
            f"review output belongs to a different render: {output_dir}")
    for record in manifest.get("outputs", {}).values():
        path = output_dir / record["relative_path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"completed review output failed verification: {path}")
    return manifest


def build_review_bundle(
        *,
        baseline_labels_path: Path,
        candidate_labels_path: Path,
        raw_path: Path,
        motion_path: Path,
        output_dir: Path,
        stem: str | None = None,
        baseline_title: str = "CURRENT ACCEPTED BASE",
        candidate_title: str = "CANDIDATE",
        source_frame_offset: int = 0,
        motion_frame_offset: int | None = None,
        frame_interval_min: float = 30.0,
        expected_baseline_sha256: str | None = None,
        expected_candidate_sha256: str | None = None,
        dry_run: bool = False,
        ) -> dict[str, Any]:
    """Build an immutable, self-verifying review bundle."""
    paths = {
        "baseline_labels": Path(baseline_labels_path).resolve(),
        "candidate_labels": Path(candidate_labels_path).resolve(),
        "raw": Path(raw_path).resolve(),
        "motion": Path(motion_path).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    if (expected_baseline_sha256 is not None
            and input_hashes["baseline_labels"] != expected_baseline_sha256):
        raise ValueError("baseline label SHA-256 does not match the expected value")
    if (expected_candidate_sha256 is not None
            and input_hashes["candidate_labels"] != expected_candidate_sha256):
        raise ValueError("candidate label SHA-256 does not match the expected value")
    if frame_interval_min <= 0:
        raise ValueError("frame interval must be positive")

    baseline = np.asarray(tifffile.imread(paths["baseline_labels"]))
    candidate = np.asarray(tifffile.imread(paths["candidate_labels"]))
    validate_labels(baseline)
    validate_labels(candidate)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"baseline {baseline.shape} and candidate {candidate.shape} differ")
    frames = len(candidate)
    raw = _align_source(
        np.asarray(tifffile.imread(paths["raw"])), frames,
        source_frame_offset, "raw")
    effective_motion_offset = (source_frame_offset if motion_frame_offset is None
                               else motion_frame_offset)
    motion_source = np.asarray(tifffile.imread(paths["motion"]))
    motion_is_transition_stack = frames > 1 and len(motion_source) == frames - 1
    motion = _align_source(
        motion_source, frames, effective_motion_offset, "motion",
        allow_transition_stack=True)
    _validate_inputs(baseline, candidate, raw, motion)

    output_dir = Path(output_dir).resolve()
    stem = stem or paths["candidate_labels"].stem
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename stem")
    params = {
        "stem": stem,
        "baseline_title": baseline_title,
        "candidate_title": candidate_title,
        "source_frame_offset": int(source_frame_offset),
        "motion_frame_offset": int(effective_motion_offset),
        "motion_alignment": ("trimmed_transition_stack_terminal_repeat"
                             if motion_is_transition_stack
                             else "frame_stack_offset_with_terminal_clip"),
        "frame_interval_min": float(frame_interval_min),
    }
    producer = {
        "renderer": "code/build_review_tiff.py",
        "renderer_version": RENDERER_VERSION,
        "code_sha256": _code_hashes(),
    }
    fingerprint_payload = {
        "input_sha256": input_hashes,
        "params": params,
        "producer": producer,
    }
    fingerprint = _fingerprint(fingerprint_payload)
    existing = _verified_existing(output_dir, fingerprint)
    if existing is not None:
        return {
            "status": "reused", "output_dir": str(output_dir),
            "fingerprint": fingerprint,
            "manifest": str(output_dir / "review_manifest.json"),
        }
    if dry_run:
        return {
            "status": "planned", "output_dir": str(output_dir),
            "fingerprint": fingerprint, "frames": frames,
            "label_shape": list(candidate.shape),
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage outside the synchronized output tree. Dropbox can open a newly written
    # multi-page TIFF for indexing before the directory is published, which prevents
    # the atomic directory rename on Windows. Prefer system temp when it is on the
    # output volume; otherwise use the output parent because Windows cannot atomically
    # rename a directory across drives.
    staging_parent = _staging_parent(output_dir)
    temp_options = {} if staging_parent is None else {"dir": staging_parent}
    with tempfile.TemporaryDirectory(
            prefix=f"motion-review-{output_dir.name}-staging-",
            **temp_options) as temp:
        bundle = Path(temp) / "bundle"
        tiffs = bundle / "tiffs"
        tiffs.mkdir(parents=True)
        full, candidate_outline, counts = render_review_arrays(
            baseline, candidate, raw, motion,
            baseline_title=baseline_title,
            candidate_title=candidate_title,
            source_frame_offset=source_frame_offset)
        validation = _validate_render(
            full, candidate_outline, baseline, candidate)
        full_path = tiffs / (
            f"{stem}_full_field_before_after_change_motion_"
            "NUMBERED_DISPLAY_ONLY.tif")
        outline_path = tiffs / (
            f"{stem}_candidate_outlines_NUMBERED_DISPLAY_ONLY.tif")
        save_rgb_stack(full_path, full, frame_interval_min)
        save_rgb_stack(outline_path, candidate_outline, frame_interval_min)
        expected_shapes = {
            full_path.name: tuple(validation["full_review_shape"]),
            outline_path.name: tuple(validation["candidate_outline_shape"]),
        }
        for path in (full_path, outline_path):
            with tifffile.TiffFile(path) as opened:
                actual_shape = tuple(opened.series[0].shape)
            if actual_shape != expected_shapes[path.name]:
                raise AssertionError(
                    f"saved TIFF shape {actual_shape} != {expected_shapes[path.name]}")

        output_records = {
            "full_field_review": {
                "relative_path": str(full_path.relative_to(bundle)),
                "sha256": sha256(full_path),
                "bytes": full_path.stat().st_size,
            },
            "candidate_outlines": {
                "relative_path": str(outline_path.relative_to(bundle)),
                "sha256": sha256(outline_path),
                "bytes": outline_path.stat().st_size,
            },
        }
        manifest = {
            "status": "complete",
            "fingerprint": fingerprint,
            "scope": "all frames, complete field, native resolution",
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name],
                       "bytes": path.stat().st_size}
                for name, path in paths.items()
            },
            "params": params,
            "producer": producer,
            "render": {
                "frames": frames,
                "native_resolution_per_panel": [int(candidate.shape[1]),
                                                  int(candidate.shape[2])],
                "fixed_movie_contrast": True,
                "panels": [
                    "accepted raw + coloured outlines + numbers",
                    "candidate raw + coloured outlines + numbers",
                    "red changed pixels + grey candidate outlines; no numbers",
                    "candidate coloured outlines + numbers over aligned motion",
                ],
                "changed_pixels": counts["pixels"],
                "changed_frames": counts["frames"],
                "validation": validation,
            },
            "outputs": output_records,
            "operation": {
                "trigger": "explicit local command",
                "concurrency": "one immutable output directory per render",
                "mutations": "new files inside output_dir only",
                "staging": ("system temp on the output volume"
                            if staging_parent is None
                            else "temporary sibling on the output volume"),
                "idempotency": "same fingerprint is verified and reused",
                "retry": "none; rerun the same command after correcting the failure",
                "partial_success": "staging is discarded; no bundle is published",
                "rollback": "move or remove this isolated output directory",
            },
        }
        write_json(bundle / "review_manifest.json", manifest)
        bundle.rename(output_dir)

    verified = _verified_existing(output_dir, fingerprint)
    if verified is None:
        raise AssertionError("review bundle was not published")
    return {
        "status": "complete", "output_dir": str(output_dir),
        "fingerprint": fingerprint,
        "manifest": str(output_dir / "review_manifest.json"),
        "changed_pixels": verified["render"]["changed_pixels"],
        "changed_frames": verified["render"]["changed_frames"],
        "outputs": verified["outputs"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the canonical four-panel full-field review TIFFs")
    parser.add_argument("--baseline-labels", required=True, type=Path)
    parser.add_argument("--candidate-labels", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem")
    parser.add_argument("--baseline-title", default="CURRENT ACCEPTED BASE")
    parser.add_argument("--candidate-title", default="CANDIDATE")
    parser.add_argument("--source-frame-offset", type=int, default=0)
    parser.add_argument("--motion-frame-offset", type=int)
    parser.add_argument("--frame-interval-min", type=float, default=30.0)
    parser.add_argument("--expected-baseline-sha256")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = build_review_bundle(
        baseline_labels_path=args.baseline_labels,
        candidate_labels_path=args.candidate_labels,
        raw_path=args.raw,
        motion_path=args.motion,
        output_dir=args.output_dir,
        stem=args.stem,
        baseline_title=args.baseline_title,
        candidate_title=args.candidate_title,
        source_frame_offset=args.source_frame_offset,
        motion_frame_offset=args.motion_frame_offset,
        frame_interval_min=args.frame_interval_min,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
