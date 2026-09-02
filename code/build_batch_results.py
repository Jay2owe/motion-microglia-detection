from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

from common import OUTLINE_COLOURS, display_raw, label_edges


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_by_stem(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["stem"]: row for row in csv.DictReader(handle)}


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def median_track_duration(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        values = [int(row["observed_frames"]) for row in csv.DictReader(handle)]
    return round(float(np.median(values)), 1) if values else 0.0


def add_outline(raw: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = display_raw(raw[None, ...])[0]
    edge = label_edges(labels[None, ...], thick=2)[0]
    identities = labels[edge].astype(np.int64)
    rgb[edge] = OUTLINE_COLOURS[(identities - 1) % len(OUTLINE_COLOURS)]
    return rgb


def panel(image: np.ndarray, text: str, width: int = 330) -> Image.Image:
    source = Image.fromarray(image)
    height = round(source.height * width / source.width)
    source = source.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 30), "white")
    canvas.paste(source, (0, 30))
    ImageDraw.Draw(canvas).text((7, 7), text, fill="black", font=ImageFont.load_default())
    return canvas


def build_sheet(rows: list[dict], destination: Path) -> None:
    panels: list[list[Image.Image]] = []
    for row in rows:
        raw = tifffile.imread(row["registered_input"])
        labels = tifffile.imread(row["authoritative_final_labels"])
        movie_panels = []
        for index in (0, 49, 98):
            identities = int(np.unique(labels[index][labels[index] > 0]).size)
            title = (
                f"{row['stem']}  retained {index + 1} / source {index + 3}  "
                f"cells {identities}"
            )
            movie_panels.append(panel(add_outline(raw[index], labels[index]), title))
        panels.append(movie_panels)

    gap = 8
    row_heights = [max(item.height for item in row) for row in panels]
    sheet_width = sum(item.width for item in panels[0]) + gap * 2
    sheet_height = sum(row_heights) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (225, 225, 225))
    y = 0
    for movie_panels, row_height in zip(panels, row_heights):
        x = 0
        for item in movie_panels:
            sheet.paste(item, (x, y))
            x += item.width + gap
        y += row_height + gap
    sheet.save(destination, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--registered-root", type=Path, required=True)
    parser.add_argument("--motion-summary", type=Path, required=True)
    parser.add_argument("--run-map", type=Path, required=True,
                        help="JSON object mapping each stem to its immutable run name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stems", nargs="+", required=True)
    args = parser.parse_args()

    motion_root = args.motion_root.resolve()
    registered_root = args.registered_root.resolve()
    output_root = (args.output_root or registered_root / "accepted detection results").resolve()
    if output_root.exists():
        raise FileExistsError(f"result package already exists: {output_root}")
    labels_root = output_root / "labels"
    unclaimed_root = output_root / "unclaimed"
    manual_root = output_root / "manual review"
    labels_root.mkdir(parents=True)
    unclaimed_root.mkdir()
    manual_root.mkdir()

    registration = read_csv_by_stem(registered_root / "summary.csv")
    motion = read_csv_by_stem(args.motion_summary.resolve())
    run_map = read_json(args.run_map.resolve())
    records: list[dict] = []

    for stem in args.stems:
        run_spec = run_map[stem]
        if isinstance(run_spec, str):
            accepted_run = upstream_run = run_spec
        else:
            accepted_run = run_spec["accepted_run"]
            upstream_run = run_spec.get("upstream_run", accepted_run)
        final_manifest = read_json(
            motion_root / "m22_accepted_history" / accepted_run / "run.json")
        upstream_manifest = read_json(
            motion_root / "m20_oscillatory_lineage" / upstream_run / "run.json")
        if final_manifest.get("status") != "done":
            raise RuntimeError(f"m22_accepted_history/{accepted_run} is not done")
        if upstream_manifest.get("status") != "done":
            raise RuntimeError(f"m20_oscillatory_lineage/{upstream_run} is not done")
        final_path = Path(final_manifest["outputs"]["labels"]["path"])
        unclaimed_path = Path(final_manifest["outputs"]["unclaimed"]["path"])
        registered_path = registered_root / f"{stem}.tif"
        upstream_path = Path(upstream_manifest["outputs"]["labels"]["path"])
        tracks_path = Path(final_manifest["outputs"]["tracks"]["path"])
        manual_source = (
            motion_root / "m22_accepted_history" / accepted_run / "mid" /
            "accepted_history" / "18_seat_identity" / "out" /
            "blob_residency_manual_review.csv")
        manual_count = row_count(manual_source) if manual_source.is_file() else 0
        packaged_manual = ""
        if manual_count:
            manual_destination = manual_root / f"{stem}_blob_residency_manual_review.csv"
            shutil.copy2(manual_source, manual_destination)
            packaged_manual = str(manual_destination)

        with tifffile.TiffFile(registered_path) as tif:
            registered_shape = tuple(tif.series[0].shape)
        labels = tifffile.imread(final_path)
        unclaimed = tifffile.imread(unclaimed_path)
        upstream = tifffile.imread(upstream_path)
        if tuple(labels.shape) != registered_shape:
            raise ValueError(f"{stem}: labels {labels.shape} != input {registered_shape}")
        if labels.shape[0] != 99 or unclaimed.shape != labels.shape:
            raise ValueError(f"{stem}: expected matching 99-frame labels and unclaimed")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError(f"{stem}: invalid label type or negative labels")
        if np.any((labels > 0) & (unclaimed > 0)):
            raise ValueError(f"{stem}: assigned and unclaimed foreground overlap")
        upstream = upstream[-99:]
        final_foreground = (labels > 0) | (unclaimed > 0)
        foreground_removed = int(np.count_nonzero((upstream > 0) & ~final_foreground))
        foreground_recovered = int(np.count_nonzero(final_foreground & ~(upstream > 0)))
        if foreground_removed:
            raise ValueError(f"{stem}: accepted endpoint removed {foreground_removed} pixels")

        packaged_path = labels_root / f"{stem}.tif"
        packaged_unclaimed_path = unclaimed_root / f"{stem}_unclaimed_original_ids.tif"
        shutil.copy2(final_path, packaged_path)
        shutil.copy2(unclaimed_path, packaged_unclaimed_path)
        authoritative_sha = sha256(final_path)
        packaged_sha = sha256(packaged_path)
        if authoritative_sha != packaged_sha:
            raise RuntimeError(f"{stem}: packaged labels are not byte-identical")
        if sha256(unclaimed_path) != sha256(packaged_unclaimed_path):
            raise RuntimeError(f"{stem}: packaged unclaimed TIFF is not byte-identical")

        per_frame = [int(np.unique(frame[frame > 0]).size) for frame in labels]
        summary = final_manifest["summary"]
        record = {
            "stem": stem,
            "accepted_run": accepted_run,
            "upstream_run": upstream_run,
            "frames": labels.shape[0],
            "height": labels.shape[1],
            "width": labels.shape[2],
            "global_identities": row_count(tracks_path),
            "identities_per_frame_min": min(per_frame),
            "identities_per_frame_median": round(float(np.median(per_frame)), 1),
            "identities_per_frame_max": max(per_frame),
            "median_track_observed_frames": median_track_duration(tracks_path),
            "motion_trails": int(motion[stem]["motion_trails"]),
            "registration_gross_phase_events": int(registration[stem]["gross_phase_events"]),
            "motion_reservation_events": int(summary["motion_reservation_events"]),
            "stationary_events": int(summary["stationary_events"]),
            "foreground_pixels_removed": foreground_removed,
            "field_recovered_foreground_pixels": foreground_recovered,
            "identity_target_count": int(summary["identity_target_count"]),
            "coordinate_target_count": int(summary["coordinate_target_count"]),
            "targeting_mode": summary["targeting_mode"],
            "manual_review_candidates": manual_count,
            "accepted_stage_elapsed_seconds": float(final_manifest["elapsed_seconds"]),
            "final_labels_sha256": authoritative_sha,
            "unclaimed_sha256": sha256(unclaimed_path),
            "registered_input": str(registered_path),
            "authoritative_final_labels": str(final_path),
            "authoritative_unclaimed": str(unclaimed_path),
            "packaged_final_labels": str(packaged_path),
            "packaged_unclaimed": str(packaged_unclaimed_path),
            "manual_review_csv": packaged_manual,
        }
        records.append(record)
        print(f"validated {stem}: {record['global_identities']} identities, "
              f"{foreground_recovered} field-recovered pixels")

    summary_path = output_root / "batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    sheet_path = output_root / "result_sheet.png"
    build_sheet(records, sheet_path)
    manifest = {
        "status": "done",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "95_A1 through 95_B5",
        "source_frames_excluded_before_detection": [1, 2],
        "retained_source_frames": [3, 101],
        "retained_frame_count": 99,
        "historical_registered_output_used": False,
        "motion_evidence": str(args.motion_summary.resolve()),
        "detection_run_map": run_map,
        "authoritative_stage": "m22_accepted_history",
        "all_authoritative_manifests_done": True,
        "all_packaged_labels_byte_identical": True,
        "all_packaged_unclaimed_byte_identical": True,
        "all_prior_foreground_retained": True,
        "all_identity_target_counts_zero": all(r["identity_target_count"] == 0 for r in records),
        "all_coordinate_target_counts_zero": all(r["coordinate_target_count"] == 0 for r in records),
        "manual_review_candidates": sum(r["manual_review_candidates"] for r in records),
        "summary_csv": str(summary_path),
        "result_sheet": str(sheet_path),
        "records": records,
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = [
        "# Detection results: 95_A1 through 95_B5",
        "",
        "These are outputs from the accepted generalized pipeline, including Issue 23.",
        "Source frames 1 and 2 were excluded; each result contains source frames 3 through 101 (99 frames).",
        "The registered inputs are exact frame-3-through-101 slices of the experiment registration.",
        "",
        "`labels/` contains byte-identical copies of the authoritative final M22 label TIFFs.",
        "`unclaimed/` contains byte-identical copies of the remaining foreground TIFFs.",
        "`result_sheet.png` shows retained frames 1, 50, and 99 (source frames 3, 52, and 101).",
        "`batch_summary.csv` contains per-recording counts and validation metrics.",
        "`batch_manifest.json` records every authoritative output path and checksum.",
        "`manual review/` contains evidence-led events left unchanged because an automatic correction precondition was absent.",
    ]
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
