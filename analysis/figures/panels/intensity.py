"""Panels for reporter level, texture and puncta."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult


CORRECTED_COLUMNS = frozenset({
    "corrected_mean", "corrected_integrated_density", "dff",
})

RATIO_COLUMNS = frozenset({
    "signal_to_background", "bright_fraction_over_background", "punctate_fraction",
    "punctateness", "signal_cv", "dff",
})

__all__ = [
    "CORRECTED_COLUMNS", "RATIO_COLUMNS", "reporter_traces", "texture_plane",
    "puncta_strip", "presentation_stack",
]


def presentation_stack(
    stack: np.ndarray,
    *,
    image_filter: str = "auto-organotypic",
    black_percentile: float = 50.0,
    white_percentile: float = 99.8,
    gamma: float = 0.7,
    time_gain: float = 6.0,
    spatial_sigma_px: float = 1.0,
    pool_px: float = 4.0,
    sharpness: float = 3.0,
    noise_multiple: float = 1.0,
    pad_frames: int = 64,
    range_frames: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare intensity images for presentation without producing measurements.

    This is the Auto-Organotypic display path: optional adaptive temporal
    denoising, one range shared by the whole stack, a monotone gamma curve, and
    optional amplification of each frame's departure from the stack mean. The
    returned metadata always marks the result display-only.
    """
    from auto_organotypic import display as organotypic_display
    from auto_organotypic.render import screen as organotypic_screen

    values = np.asarray(stack, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("presentation_stack needs frames x rows x columns")
    method = str(image_filter).strip().lower().replace("_", "-")
    if method in ("none", "raw"):
        filtered = values.copy()
        method = "none"
        method_version = "not-applied"
    elif method in ("auto-organotypic", "adaptive", "adaptive-wiener"):
        filtered = organotypic_display.display_process(
            values,
            spatial_sigma=float(spatial_sigma_px),
            pool_px=float(pool_px),
            sharpness=float(sharpness),
            noise_multiple=float(noise_multiple),
            pad_frames=int(pad_frames),
        )
        method = "auto-organotypic adaptive temporal filter"
        method_version = organotypic_display.DISPLAY_METHOD_VERSION
    else:
        raise ValueError(
            f"unknown image_filter {image_filter!r}; choose auto-organotypic or none")

    if not 0 <= float(black_percentile) < float(white_percentile) <= 100:
        raise ValueError("display percentiles need 0 <= black < white <= 100")
    if float(gamma) <= 0 or float(time_gain) <= 0:
        raise ValueError("display_gamma and display_gain must be positive")
    if range_frames is None:
        range_source = filtered
        range_indices = np.arange(filtered.shape[0], dtype=int)
        range_scope = "all frames"
    else:
        range_indices = np.asarray(list(range_frames), dtype=int)
        if range_indices.ndim != 1 or range_indices.size == 0:
            raise ValueError("range_frames must name at least one frame")
        if np.any(range_indices < 0) or np.any(range_indices >= filtered.shape[0]):
            raise ValueError("range_frames contains a frame outside the stack")
        range_source = filtered[range_indices]
        range_scope = "displayed frames"
    mean_image = range_source.mean(axis=0)
    black, white = organotypic_display.display_range(
        range_source, mean_image, float(black_percentile), float(white_percentile))
    screen = organotypic_screen.scale_to_screen(filtered, black, white, float(gamma))
    if float(time_gain) != 1.0:
        screen = organotypic_screen.amplify_time(
            screen, screen.mean(axis=0), float(time_gain))
        screen = organotypic_screen.soft_limit(screen)
    else:
        screen = np.clip(screen, 0.0, 1.0)
    settings = {
        "display_only": True,
        "image_filter": method,
        "filter_method_version": method_version,
        "black_percentile": float(black_percentile),
        "white_percentile": float(white_percentile),
        "black_value": float(black),
        "white_value": float(white),
        "gamma": float(gamma),
        "time_gain": float(time_gain),
        "spatial_sigma_px": float(spatial_sigma_px),
        "pool_px": float(pool_px),
        "sharpness": float(sharpness),
        "noise_multiple": float(noise_multiple),
        "pad_frames": int(pad_frames),
        "display_range_scope": range_scope,
        "display_range_frame_count": int(range_indices.size),
        "display_range_frame_indices": ",".join(str(int(i)) for i in range_indices),
    }
    return np.asarray(screen, dtype=float), settings


def reporter_traces(
    ax: Any,
    hours: Sequence[float],
    frame: pd.DataFrame,
    theme: Any,
    *,
    columns: Sequence[str],
    looks: dict[str, common.Look] | None = None,
    normalise: str | None = None,
    hour_ticks: float | None = None,
    x_label: str = "Hours from start of recording",
) -> PanelResult:
    """Reporter channels over time without double background correction."""
    hours = np.asarray(hours, dtype=float)
    looks = looks or {}
    result = pd.DataFrame({"hours": hours})
    for column in columns:
        values = np.asarray(frame[column], dtype=float)
        if normalise == "z":
            spread = float(np.nanstd(values)) or 1.0
            values = (values - np.nanmean(values)) / spread
        elif normalise == "minmax":
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
            values = (values - low) / (high - low) if high > low else np.zeros_like(values)
        elif normalise not in (None, "none"):
            raise ValueError("normalise must be None, z or minmax")
        common.trace(
            ax, hours, values, theme, look=looks.get(column), role="reporter",
            label=column.replace("_", " "), hour_ticks=hour_ticks, x_label=x_label,
        )
        result[column] = values
    ax.set_ylabel("Reporter readout" + (" (normalised)" if normalise not in (None, "none") else ""))
    return PanelResult(data=result, axes=ax)


def texture_plane(
    figure: Any,
    rect: tuple[float, float, float, float],
    level: Sequence[float],
    texture: Sequence[float],
    theme: Any,
    *,
    look: common.Look | None = None,
    bins: int = 24,
    margins: bool = True,
    marks: Sequence[common.Mark] = (),
    level_label: str = "",
    texture_label: str = "",
) -> PanelResult:
    """Bulk reporter level against within-cell clustering.

    ``axes`` is (main, top, right), as the scatter it is built on gives them.
    """
    level = np.asarray(level, dtype=float)
    texture = np.asarray(texture, dtype=float)
    usable = np.isfinite(level) & np.isfinite(texture)
    level, texture = level[usable], texture[usable]
    drawn = common.scatter_with_margins(
        figure, rect, level, texture, theme, role="reporter", look=look,
        bins=bins, margins=margins,
    )
    ax, top, right = drawn.axes
    if marks:
        common.reference_lines(ax, marks, theme)
    ax.set_xlabel(level_label)
    ax.set_ylabel(texture_label)
    return PanelResult(
        data=pd.DataFrame({"level": level, "texture": texture}),
        axes=(ax, top, right),
    )


def puncta_strip(
    figure: Any,
    rect: tuple[float, float, float, float],
    images: Sequence[np.ndarray],
    theme: Any,
    *,
    masks: Sequence[np.ndarray] | None = None,
    titles: Sequence[str] | None = None,
    look: common.Look | None = None,
    percentiles: tuple[float, float] = (2.0, 99.5),
) -> PanelResult:
    """Reporter tiles on one shared contrast scale."""
    return common.image_strip(
        figure, rect, images, theme, masks=masks, titles=titles, look=look,
        percentiles=percentiles,
    )

