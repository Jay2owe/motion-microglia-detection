from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import tifffile

from common import OUTLINE_COLOURS, label_edges


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def identity_positions(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Place each identity number on its largest connected piece."""
    positions = []
    objects = ndi.find_objects(labels)
    for identity, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        local = labels[bounds] == identity
        components, count = ndi.label(local)
        if count > 1:
            sizes = np.bincount(components.ravel())
            component = int(np.argmax(sizes[1:]) + 1)
            local = components == component
        y_local, x_local = ndi.center_of_mass(local)
        y = int(round(bounds[0].start + y_local))
        x = int(round(bounds[1].start + x_local))
        positions.append((identity, y, x))
    return positions


def render_frame(raw: np.ndarray, labels: np.ndarray, high: float,
                 label_font: ImageFont.ImageFont) -> np.ndarray:
    grey = np.clip(raw.astype(np.float32) / max(high, 1.0), 0, 1) ** 0.7
    rgb = np.repeat((grey[..., None] * 180).astype(np.uint8), 3, axis=-1)
    edge = label_edges(labels[None, ...], thick=2)[0]
    identities = labels[edge].astype(np.int64)
    rgb[edge] = OUTLINE_COLOURS[(identities - 1) % len(OUTLINE_COLOURS)]

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for identity, y, x in identity_positions(labels):
        text = str(identity)
        bounds = draw.textbbox((0, 0), text, font=label_font, stroke_width=2)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        position = (
            min(max(1, x - text_width // 2), image.width - text_width - 1),
            min(max(1, y - text_height // 2), image.height - text_height - 1),
        )
        draw.text(position, text, font=label_font, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(image)


def make_preview(frames: list[tuple[str, np.ndarray]], destination: Path) -> None:
    header = 25
    gap = 8
    cell_width = max(frame.shape[1] for _, frame in frames)
    cell_height = max(frame.shape[0] for _, frame in frames) + header
    sheet = Image.new("RGB", (cell_width * 2 + gap, cell_height * 5 + gap * 4),
                      (225, 225, 225))
    label_font = font(15)
    for index, (stem, frame) in enumerate(frames):
        cell = Image.new("RGB", (cell_width, cell_height), "white")
        ImageDraw.Draw(cell).text(
            (6, 4), f"{stem}: retained frame 50 / source frame 52",
            font=label_font, fill="black")
        cell.paste(Image.fromarray(frame), (0, header))
        x = (index % 2) * (cell_width + gap)
        y = (index // 2) * (cell_height + gap)
        sheet.paste(cell, (x, y))
    sheet.save(destination, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    manifest_path = result_root / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = result_root / "numbered outlines"
    if output_root.exists():
        raise FileExistsError(f"numbered-outline folder already exists: {output_root}")
    output_root.mkdir(parents=True)

    output_records = []
    preview_frames = []
    label_font = font(13)
    for record in manifest["records"]:
        stem = record["stem"]
        raw_path = Path(record["registered_input"])
        labels_path = Path(record["authoritative_final_labels"])
        if sha256(labels_path) != record["final_labels_sha256"]:
            raise RuntimeError(f"{stem}: authoritative final labels changed")
        raw = tifffile.imread(raw_path)
        labels = tifffile.imread(labels_path)
        if raw.shape != labels.shape or len(labels) != 99:
            raise ValueError(f"{stem}: expected matching 99-frame raw and labels")
        positive = raw[raw > 0]
        high = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        overlay = np.empty((*raw.shape, 3), dtype=np.uint8)
        for index in range(len(raw)):
            overlay[index] = render_frame(raw[index], labels[index], high, label_font)

        output_path = output_root / f"{stem}_numbered_outlines.tif"
        tifffile.imwrite(
            output_path, overlay, imagej=True, compression="zlib", photometric="rgb",
            metadata={"axes": "TYXS", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"},
        )
        with tifffile.TiffFile(output_path) as tif:
            output_shape = tuple(tif.series[0].shape)
        if output_shape != overlay.shape:
            raise RuntimeError(f"{stem}: saved TIFF shape {output_shape} != {overlay.shape}")
        output_records.append({
            "stem": stem,
            "frames": 99,
            "height": labels.shape[1],
            "width": labels.shape[2],
            "path": str(output_path),
            "sha256": sha256(output_path),
            "bytes": output_path.stat().st_size,
            "labels_sha256": record["final_labels_sha256"],
        })
        preview_frames.append((stem, overlay[49].copy()))
        print(f"wrote {stem}: {output_shape}")

    preview_path = output_root / "preview_source_frame_52.png"
    make_preview(preview_frames, preview_path)
    output_manifest = {
        "status": "done",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rendering": "registered raw + coloured final identity outlines + identity numbers",
        "frames_per_tiff": 99,
        "retained_source_frames": [3, 101],
        "authoritative_label_source": "accepted M22 Issue 23 outputs",
        "historical_output_used": False,
        "preview": str(preview_path),
        "outputs": output_records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
