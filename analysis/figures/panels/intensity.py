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
    "puncta_strip",
]


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

