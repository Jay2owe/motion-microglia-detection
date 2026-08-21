from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]

OUTLINE_COLOURS = np.array([
    [255, 70, 70], [80, 210, 255], [110, 255, 100], [255, 210, 70],
    [220, 100, 255], [255, 145, 60], [80, 255, 210], [160, 160, 255],
    [255, 100, 190], [180, 255, 70], [90, 170, 255], [255, 180, 190],
], dtype=np.uint8)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Config:
    values: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        source = (path or ROOT / "config.json").resolve()
        return cls(read_json(source), source)

    def resolve(self, key: str) -> Path:
        return (self.path.parent / self.values[key]).resolve()

    @property
    def stems(self) -> list[str]:
        return list(self.values["stems"])


class Run:
    """One immutable stage run with an honest running/done/failed manifest."""

    def __init__(self, stage: str, name: str, params: dict[str, Any],
                 upstream: str | None = None):
        self.stage = stage
        self.name = name
        self.dir = ROOT / stage / name
        if self.dir.exists():
            raise FileExistsError(f"immutable run already exists: {self.dir}")
        for sub in ("out", "mid", "qc"):
            (self.dir / sub).mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.manifest: dict[str, Any] = {
            "stage": stage,
            "run": name,
            "status": "running",
            "upstream": upstream,
            "params": params,
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "elapsed_seconds": None,
            "outputs": {},
        }
        write_json(self.dir / "run.json", self.manifest)

    def record(self, label: str, path: Path) -> None:
        self.manifest["outputs"][label] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        write_json(self.dir / "run.json", self.manifest)

    def finish(self, summary: dict[str, Any]) -> None:
        self.manifest.update({
            "status": "done",
            "summary": summary,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
        })
        write_json(self.dir / "run.json", self.manifest)

    def fail(self, error: BaseException) -> None:
        self.manifest.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
        })
        write_json(self.dir / "run.json", self.manifest)


def load_stack(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


def save_stack(path: Path, array: np.ndarray, frame_interval_min: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path, np.asarray(array), imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": frame_interval_min * 60.0,
                  "tunit": "sec", "unit": "pixel"},
    )


def save_rgb_stack(path: Path, array: np.ndarray,
                   frame_interval_min: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path, np.asarray(array, np.uint8), imagej=True, compression="zlib",
        photometric="rgb",
        metadata={"axes": "TYXS", "finterval": frame_interval_min * 60.0,
                  "tunit": "sec", "unit": "pixel"},
    )


def display_raw(raw: np.ndarray) -> np.ndarray:
    values = raw[raw > 0]
    high = float(np.percentile(values, 99.5)) if values.size else 1.0
    grey = np.clip(raw.astype(np.float32) / max(high, 1.0), 0, 1) ** 0.7
    return np.repeat((grey[..., None] * 180).astype(np.uint8), 3, axis=-1)


def label_edges(labels: np.ndarray, thick: int = 1) -> np.ndarray:
    edges = np.zeros(labels.shape, bool)
    for axis in (-2, -1):
        shifted = np.roll(labels, 1, axis=axis)
        edges |= (labels != shifted) & (labels > 0)
    if thick > 1:
        edges = ndi.binary_dilation(edges, iterations=thick - 1)
    return edges & (labels > 0)


def outline_overlay(raw: np.ndarray, labels: np.ndarray, thick: int = 2,
                    inferred: np.ndarray | None = None,
                    unresolved: np.ndarray | None = None) -> np.ndarray:
    out = display_raw(raw)
    edge = label_edges(labels, thick=thick)
    identities = labels[edge].astype(np.int64)
    out[edge] = OUTLINE_COLOURS[(identities - 1) % len(OUTLINE_COLOURS)]
    if inferred is not None:
        inferred_edge = label_edges(np.where(inferred, labels, 0), thick=thick)
        out[inferred_edge] = np.array([0, 255, 255], np.uint8)
    if unresolved is not None:
        unresolved_edge = ndi.binary_dilation(unresolved.astype(bool), iterations=1)
        unresolved_edge ^= ndi.binary_erosion(unresolved.astype(bool), iterations=1)
        out[unresolved_edge] = np.array([255, 0, 255], np.uint8)
    return out


def changed_pixels(base: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Where the two label stacks disagree about which cell owns a pixel."""
    if base.shape != candidate.shape:
        raise ValueError("base and candidate must be the same shape")
    return base != candidate


def change_overlay(raw: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                   thick: int = 2) -> np.ndarray:
    """The change made visible, on the same field as the other review panels.

    Required by `WORKFLOW_CONTRACT.md`. It is meant to sit beside the base and
    candidate panels, not on its own: read across the three and the yellow tells
    you which cell just gained ground and from whom.

    Yellow is ground the candidate gave to a different name than the base did.
    Blue is ground the base named and the candidate leaves unclaimed. The
    candidate's own outlines stay, so a change can be read against the cell it
    belongs to rather than floating in the dark.
    """
    changed = changed_pixels(base, candidate)
    out = outline_overlay(raw, candidate, thick=thick)
    out[changed & (candidate > 0)] = np.array([255, 210, 60], np.uint8)
    out[changed & (candidate == 0) & (base > 0)] = np.array([80, 170, 255], np.uint8)
    return out


def change_counts(base: np.ndarray, candidate: np.ndarray) -> dict:
    """How much changed, in the words a review header needs."""
    changed = changed_pixels(base, candidate)
    per_frame = changed.reshape(len(changed), -1).sum(axis=1)
    return {
        "pixels": int(changed.sum()),
        "frames": int((per_frame > 0).sum()),
        "total_frames": int(len(changed)),
        "per_frame": [int(value) for value in per_frame],
    }


def validate_labels(labels: np.ndarray) -> None:
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer (T,Y,X) array")
    if np.any(labels < 0):
        raise ValueError("labels cannot be negative")
