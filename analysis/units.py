"""Spatial and temporal scale, resolved once and stamped on every table.

Nothing in this package multiplies by a pixel size directly. Every spatial
number goes through a :class:`Scale`, so an uncalibrated dataset reports pixels
and a calibrated one reports micrometres with no other code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tifffile

# TIFF ResolutionUnit tag values that carry a real physical length.
_RESOLUTION_UNIT_TO_MICRONS = {2: 25400.0, 3: 10000.0}  # inch, centimetre


@dataclass(frozen=True)
class Scale:
    """How one pixel and one frame map onto physical units."""

    minutes_per_frame: float
    microns_per_pixel: float | None = None
    source: str = "uncalibrated"

    @property
    def calibrated(self) -> bool:
        return self.microns_per_pixel is not None and self.microns_per_pixel > 0

    @property
    def length_unit(self) -> str:
        return "um" if self.calibrated else "px"

    @property
    def area_unit(self) -> str:
        return "um2" if self.calibrated else "px2"

    @property
    def speed_unit(self) -> str:
        return f"{self.length_unit}_per_min"

    def length(self, pixels: float) -> float:
        return pixels * self.microns_per_pixel if self.calibrated else pixels

    def area(self, square_pixels: float) -> float:
        if not self.calibrated:
            return square_pixels
        return square_pixels * self.microns_per_pixel ** 2

    def hours(self, frames: float) -> float:
        return frames * self.minutes_per_frame / 60.0

    def speed(self, pixels_per_frame: float) -> float:
        """Displacement per frame converted to length per minute."""
        return self.length(pixels_per_frame) / self.minutes_per_frame

    def describe(self) -> dict:
        return {
            "minutes_per_frame": self.minutes_per_frame,
            "microns_per_pixel": self.microns_per_pixel,
            "calibrated": self.calibrated,
            "source": self.source,
            "length_unit": self.length_unit,
            "area_unit": self.area_unit,
        }


def _microns_per_pixel_from_tiff(path: Path) -> float | None:
    """Recover a pixel size from TIFF resolution tags, or None if absent."""
    try:
        with tifffile.TiffFile(path) as handle:
            page = handle.pages[0]
            tags = page.tags
            if "XResolution" not in tags or "ResolutionUnit" not in tags:
                return None
            unit = int(tags["ResolutionUnit"].value)
            factor = _RESOLUTION_UNIT_TO_MICRONS.get(unit)
            if factor is None:
                return None  # ResolutionUnit 1 means "no absolute unit"
            numerator, denominator = tags["XResolution"].value
            if not denominator or not numerator:
                return None
            pixels_per_unit = numerator / denominator
            if pixels_per_unit <= 0:
                return None
            microns = factor / pixels_per_unit
            # An uncalibrated ImageJ stack often writes 1/1; reject the identity.
            return None if microns == factor else float(microns)
    except Exception:
        return None


def resolve_scale(
    minutes_per_frame: float,
    configured_microns_per_pixel: float | None = None,
    candidate_tiffs: list[Path] | None = None,
) -> Scale:
    """Resolve spatial scale in a fixed, auditable order.

    1. an explicit value in the analysis configuration;
    2. resolution tags on any candidate TIFF, in the order given
       (registered stack first, original acquisition stack next);
    3. uncalibrated - everything is reported in pixels.
    """
    if configured_microns_per_pixel:
        return Scale(minutes_per_frame, float(configured_microns_per_pixel), "config")

    for path in candidate_tiffs or []:
        path = Path(path)
        if not path.exists():
            continue
        value = _microns_per_pixel_from_tiff(path)
        if value:
            return Scale(minutes_per_frame, value, f"tiff:{path.name}")

    return Scale(minutes_per_frame, None, "uncalibrated")
