"""Canonical full-field review renderer for scorer-only event calibrations.

The biological label renderer in :mod:`build_review_tiff` shows mask changes.  A
scorer-only round changes no masks, so its useful review is the complete event field
before and after calibration.  This command renders accepted identities, baseline
flags, calibrated flags, and flags removed from the score.  Event markers are
diagnostics, never proposed ground-truth identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import tifffile

from common import display_raw, label_edges, save_rgb_stack, sha256, write_json
from review_rendering import header, unique_outline


RENDERER_VERSION = 2
HEADER_HEIGHT = 24
RED = (255, 45, 45)
ORANGE = (255, 170, 20)
BLUE = (80, 180, 255)
REMOVED_GREEN = (80, 235, 130)
GREY = np.array([150, 150, 150], np.uint8)


def _align_source(stack: np.ndarray, frames: int, offset: int,
                  name: str) -> np.ndarray:
    if offset < 0:
        raise ValueError(f"{name} frame offset cannot be negative")
    if len(stack) == frames:
        return stack
    if len(stack) >= offset + frames:
        return stack[offset:offset + frames]
    raise ValueError(
        f"cannot align {name} ({len(stack)} frames) to {frames} review frames")


def _event_colour(score: float) -> tuple[int, int, int]:
    if score >= 50:
        return RED
    if score >= 35:
        return ORANGE
    return BLUE


def _accepted_grey(raw: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Raw signal with neutral accepted outlines and accepted identity numbers."""
    result = display_raw(raw)
    result[label_edges(labels, thick=2)] = GREY
    for frame in range(len(result)):
        image = Image.fromarray(result[frame])
        draw = ImageDraw.Draw(image)
        for identity in sorted(set(map(int, np.unique(labels[frame]))) - {0}):
            parts, count = ndi.label(
                labels[frame] == identity,
                structure=np.ones((3, 3), np.uint8))
            for component in range(1, count + 1):
                mask = parts == component
                if int(mask.sum()) < 8:
                    continue
                y, x = ndi.center_of_mass(mask)
                draw.text((round(x), round(y)), str(identity),
                          fill=(230, 230, 230), stroke_width=2,
                          stroke_fill=(0, 0, 0))
        result[frame] = np.asarray(image)
    return result


def _source_rows(events: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id", "source_id", "source_first_frame", "source_last_frame",
        "source_x", "source_y",
    }
    missing = required - set(members.columns)
    if missing:
        raise ValueError(f"event members are missing columns: {sorted(missing)}")
    ids = set(events.event_id.astype(str))
    rows = members[members.event_id.astype(str).isin(ids)].copy()
    rows["event_id"] = rows.event_id.astype(str)
    return rows.drop_duplicates(["event_id", "source_id"])


def _render_locations(raw: np.ndarray, labels: np.ndarray,
                      events: pd.DataFrame, members: pd.DataFrame,
                      *, removed: bool = False) -> tuple[np.ndarray, list[str]]:
    result = _accepted_grey(raw, labels)
    sources = _source_rows(events, members)
    lookup = events.copy()
    lookup["event_id"] = lookup.event_id.astype(str)
    lookup = lookup.set_index("event_id")
    notes: list[str] = []
    for frame in range(len(result)):
        active = sources[
            (pd.to_numeric(sources.source_first_frame) <= frame)
            & (pd.to_numeric(sources.source_last_frame) >= frame)].copy()
        if not active.empty:
            active["rank"] = active.event_id.map(lookup.impact_rank)
            active = active.sort_values(["rank", "event_id"])
        notes.append(f"{active.event_id.nunique()} active score flags")
        image = Image.fromarray(result[frame])
        draw = ImageDraw.Draw(image)
        labelled: set[str] = set()
        for row in active.itertuples(index=False):
            event_id = str(row.event_id)
            event = lookup.loc[event_id]
            colour = (REMOVED_GREEN if removed
                      else _event_colour(float(event.disruption_score)))
            x, y = float(row.source_x), float(row.source_y)
            draw.rectangle((x - 7, y - 7, x + 7, y + 7),
                           outline=colour, width=2)
            draw.line((x - 4, y, x + 4, y), fill=colour, width=1)
            draw.line((x, y - 4, x, y + 4), fill=colour, width=1)
            if event_id not in labelled:
                prefix = "REMOVED " if removed else ""
                text = (f"{prefix}#{int(event.impact_rank):03d} {event_id} "
                        f"S{float(event.disruption_score):.0f}")
                draw.text((x + 9, y - 7), text, fill=colour,
                          stroke_width=1, stroke_fill=(0, 0, 0))
                labelled.add(event_id)
        result[frame] = np.asarray(image)
    return result, notes


def event_review_index(baseline_events: pd.DataFrame,
                       candidate_events: pd.DataFrame,
                       source_frame_offset: int) -> pd.DataFrame:
    """Return a stable before/after table for every baseline or candidate flag."""
    baseline = baseline_events.copy()
    candidate = candidate_events.copy()
    baseline["event_id"] = baseline.event_id.astype(str)
    candidate["event_id"] = candidate.event_id.astype(str)
    before = baseline.set_index("event_id", drop=False)
    after = candidate.set_index("event_id", drop=False)
    rows: list[dict[str, Any]] = []
    for event_id in sorted(set(before.index) | set(after.index)):
        in_before = event_id in before.index
        in_after = event_id in after.index
        source = after.loc[event_id] if in_after else before.loc[event_id]
        status = ("retained" if in_before and in_after
                  else "removed" if in_before else "added")
        first = int(source.first_frame)
        last = int(source.last_frame)
        rows.append({
            "event_id": event_id,
            "calibration_status": status,
            "baseline_rank": (int(before.loc[event_id].impact_rank)
                              if in_before else ""),
            "candidate_rank": (int(after.loc[event_id].impact_rank)
                               if in_after else ""),
            "family": str(source.family),
            "disruption_score": float(source.disruption_score),
            "review_frame_start": first + 1,
            "review_frame_end": last + 1,
            "source_imagej_frame_start": first + source_frame_offset + 1,
            "source_imagej_frame_end": last + source_frame_offset + 1,
            "accepted_before": source.get("accepted_before", ""),
            "accepted_after": source.get("accepted_after", ""),
            "evidence": source.get("evidence", ""),
        })
    return pd.DataFrame(rows)


def render_event_comparison_arrays(
        labels: np.ndarray, raw: np.ndarray,
        baseline_events: pd.DataFrame, candidate_events: pd.DataFrame,
        members: pd.DataFrame, *, source_frame_offset: int,
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Render accepted identities and the full score field before/after."""
    if labels.ndim != 3 or raw.shape != labels.shape:
        raise ValueError("labels and aligned raw must have identical TYX shape")
    required = {"event_id", "impact_rank", "first_frame", "last_frame",
                "disruption_score", "family"}
    for name, table in (("baseline", baseline_events),
                        ("candidate", candidate_events)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} events are missing columns: {sorted(missing)}")
        if table.event_id.astype(str).duplicated().any():
            raise ValueError(f"{name} event identifiers must be unique")
    before_ids = set(baseline_events.event_id.astype(str))
    after_ids = set(candidate_events.event_id.astype(str))
    removed = baseline_events[
        baseline_events.event_id.astype(str).isin(before_ids - after_ids)].copy()
    accepted = header(
        unique_outline(raw, labels), "ACCEPTED BIOLOGICAL IDENTITIES",
        source_frame_offset)
    baseline_panel, baseline_notes = _render_locations(
        raw, labels, baseline_events, members)
    candidate_panel, candidate_notes = _render_locations(
        raw, labels, candidate_events, members)
    removed_panel, removed_notes = _render_locations(
        raw, labels, removed, members, removed=True)
    full = np.concatenate([
        accepted,
        header(baseline_panel, "SCORE FLAGS BEFORE CALIBRATION",
               source_frame_offset, baseline_notes),
        header(candidate_panel, "SCORE FLAGS AFTER CALIBRATION",
               source_frame_offset, candidate_notes),
        header(removed_panel,
               "GREEN = REMOVED SCORE FLAG; NOT A MASK OR IDENTITY EDIT",
               source_frame_offset, removed_notes),
    ], axis=2)
    expected_shape = (len(labels), labels.shape[1] + HEADER_HEIGHT,
                      labels.shape[2] * 4, 3)
    if full.shape != expected_shape:
        raise AssertionError(f"event review shape {full.shape} != {expected_shape}")
    return full, {
        "events_before": len(before_ids),
        "events_after": len(after_ids),
        "removed_event_ids": sorted(before_ids - after_ids),
        "added_event_ids": sorted(after_ids - before_ids),
        "retained_event_ids": sorted(before_ids & after_ids),
        "full_review_shape": list(expected_shape),
    }


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


def _verified_existing(output_dir: Path,
                       fingerprint: str) -> dict[str, Any] | None:
    manifest_path = output_dir / "event_review_manifest.json"
    if not output_dir.exists():
        return None
    if not manifest_path.is_file():
        raise FileExistsError(
            f"event review exists without completion manifest: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise FileExistsError(
            f"event review belongs to a different render: {output_dir}")
    for record in manifest.get("outputs", {}).values():
        path = output_dir / record["relative_path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"event review output failed verification: {path}")
    return manifest


def build_event_review_bundle(
        *, labels_path: Path, raw_path: Path,
        baseline_events_path: Path, candidate_events_path: Path,
        event_members_path: Path, output_dir: Path,
        stem: str = "95_A3", source_frame_offset: int = 0,
        frame_interval_min: float = 30.0,
        expected_labels_sha256: str | None = None,
        dry_run: bool = False,
        ) -> dict[str, Any]:
    """Build an immutable, self-verifying scorer-calibration review bundle."""
    paths = {
        "labels": Path(labels_path).resolve(),
        "raw": Path(raw_path).resolve(),
        "baseline_events": Path(baseline_events_path).resolve(),
        "candidate_events": Path(candidate_events_path).resolve(),
        "event_members": Path(event_members_path).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    if expected_labels_sha256 and hashes["labels"] != expected_labels_sha256:
        raise ValueError("labels SHA-256 does not match the expected value")
    code_path = Path(__file__).resolve()
    params = {
        "stem": stem,
        "source_frame_offset": int(source_frame_offset),
        "frame_interval_min": float(frame_interval_min),
    }
    producer = {
        "renderer": "code/build_event_review_tiff.py",
        "renderer_version": RENDERER_VERSION,
        "code_sha256": {
            code_path.name: sha256(code_path),
            "review_rendering.py": sha256(code_path.parent / "review_rendering.py"),
            "common.py": sha256(code_path.parent / "common.py"),
        },
    }
    fingerprint = _fingerprint({
        "input_sha256": hashes, "params": params, "producer": producer})
    output_dir = Path(output_dir).resolve()
    existing = _verified_existing(output_dir, fingerprint)
    if existing is not None:
        return {
            "status": "reused", "output_dir": str(output_dir),
            "manifest": str(output_dir / "event_review_manifest.json"),
            "fingerprint": fingerprint,
        }
    labels = np.asarray(tifffile.imread(paths["labels"]))
    raw = _align_source(
        np.asarray(tifffile.imread(paths["raw"])), len(labels),
        source_frame_offset, "raw")
    baseline = pd.read_csv(paths["baseline_events"])
    candidate = pd.read_csv(paths["candidate_events"])
    members = pd.read_csv(paths["event_members"], keep_default_na=False)
    if dry_run:
        return {
            "status": "planned", "output_dir": str(output_dir),
            "fingerprint": fingerprint, "frames": len(labels),
            "events_before": len(baseline), "events_after": len(candidate),
        }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = _staging_parent(output_dir)
    temp_options = {} if staging_parent is None else {"dir": staging_parent}
    with tempfile.TemporaryDirectory(
            prefix=f"motion-event-review-{output_dir.name}-staging-",
            **temp_options) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        full, summary = render_event_comparison_arrays(
            labels, raw, baseline, candidate, members,
            source_frame_offset=source_frame_offset)
        tiff_path = bundle / f"{stem}_event_score_before_after_DISPLAY_ONLY.tif"
        save_rgb_stack(tiff_path, full, frame_interval_min)
        before_path = bundle / "baseline_disruptive_events.csv"
        after_path = bundle / "candidate_disruptive_events.csv"
        index_path = bundle / "event_calibration_review_index.csv"
        removed_members_path = bundle / "removed_event_members.csv"
        baseline.to_csv(before_path, index=False)
        candidate.to_csv(after_path, index=False)
        index = event_review_index(
            baseline, candidate, source_frame_offset)
        index.to_csv(index_path, index=False)
        removed_ids = set(summary["removed_event_ids"])
        members[members.event_id.astype(str).isin(removed_ids)].to_csv(
            removed_members_path, index=False)
        outputs: dict[str, dict[str, Any]] = {}
        for label, path in {
                "event_review_tiff": tiff_path,
                "baseline_events": before_path,
                "candidate_events": after_path,
                "review_index": index_path,
                "removed_event_members": removed_members_path,
                }.items():
            outputs[label] = {
                "relative_path": str(path.relative_to(bundle)),
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
        manifest = {
            "status": "complete", "fingerprint": fingerprint,
            "scope": "all frames, complete field, native resolution",
            "inputs": {
                name: {"path": str(path), "sha256": hashes[name],
                       "bytes": path.stat().st_size}
                for name, path in paths.items()
            },
            "params": params, "producer": producer,
            "render": {
                **summary,
                "accepted_identity_numbers_rendered": True,
                "raw_track_ids_rendered_as_identities": False,
                "candidate_labels_modified": False,
                "panels": [
                    "accepted raw + coloured identity outlines + numbers",
                    "baseline score flags over grey accepted identities",
                    "calibrated score flags over grey accepted identities",
                    "green removed score flags over grey accepted identities",
                ],
                "marker_interpretation": (
                    "diagnostic inconsistency location, not ground-truth identity"),
            },
            "outputs": outputs,
            "operation": {
                "mutations": "new files inside output_dir only",
                "staging": ("system temp on the output volume"
                            if staging_parent is None
                            else "temporary sibling on the output volume"),
                "idempotency": "same fingerprint is verified and reused",
                "partial_success": "staging is discarded",
            },
        }
        write_json(bundle / "event_review_manifest.json", manifest)
        bundle.rename(output_dir)
    verified = _verified_existing(output_dir, fingerprint)
    if verified is None:
        raise AssertionError("event review bundle was not published")
    return {
        "status": "complete", "output_dir": str(output_dir),
        "manifest": str(output_dir / "event_review_manifest.json"),
        "fingerprint": fingerprint,
        "removed_event_ids": verified["render"]["removed_event_ids"],
        "outputs": verified["outputs"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the canonical scorer-calibration review TIFF")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--baseline-events", required=True, type=Path)
    parser.add_argument("--candidate-events", required=True, type=Path)
    parser.add_argument("--event-members", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem", default="95_A3")
    parser.add_argument("--source-frame-offset", type=int, default=0)
    parser.add_argument("--frame-interval-min", type=float, default=30.0)
    parser.add_argument("--expected-labels-sha256")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = build_event_review_bundle(
        labels_path=args.labels, raw_path=args.raw,
        baseline_events_path=args.baseline_events,
        candidate_events_path=args.candidate_events,
        event_members_path=args.event_members,
        output_dir=args.output_dir, stem=args.stem,
        source_frame_offset=args.source_frame_offset,
        frame_interval_min=args.frame_interval_min,
        expected_labels_sha256=args.expected_labels_sha256,
        dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
