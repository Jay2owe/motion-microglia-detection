"""Chart grammar: panels that know how to draw and nothing about the biology.

A scatter with a histogram along each edge is the same panel whether the dots
are cells or countries. Everything in this file is that kind of panel - it is
given an axes, some numbers and the theme, and it draws. It never opens a file,
never picks a cell and never decides where on the page it sits, because those
are the choices the user's options reach, and they belong with the caller.

What lives here and what lives next door: a panel moves into a measurement
module's file the moment it has to *know* something about that measurement -
which colours its states carry, which rows are allowed to be joined by a line,
what has to be subtracted first. A panel that would draw the same picture for
any numbers stays here.

Colour is taken as a *look*, which may be either:

* a flat colour - a theme role (``"reporter"``), a palette name, or a hex value;
* a colour map name (``"viridis"``, ``"magma"``), in which case a trace is drawn
  as a line whose colour runs along the map with elapsed time, and an image tile
  is displayed through it.

Both are resolved by :func:`resolve_look`, so "the LUT for this graph" and "the
colour of this graph" are one option rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import textwrap

import numpy as np
import pandas as pd

from analysis import circadian as workbench

from ._contract import PanelResult

if TYPE_CHECKING:                       # names used only in annotations
    from matplotlib.colorbar import Colorbar
    from matplotlib.legend import Legend
    from matplotlib.patches import Rectangle

__all__ = [
    "Look", "Mark", "PanelResult", "resolve_look",
    "image_tile", "image_strip", "paired_image_strip",
    "trace", "trace_overlay", "cosinor_curve", "harmonic_curve", "trace_stack",
    "raster", "category_strip", "colour_bar", "inset_colour_bar", "semantic_legend",
    "histogram", "reference_lines",
    "scatter", "scatter_with_margins",
    "dumbbell", "lollipop", "ranked_rate",
    "phase_map", "rose", "vectors", "matrix", "surface", "stacked_histogram",
    "ridgeline", "stacked_area",
    "flow", "chord", "paired_slopes", "event_average", "hexbin", "dendrogram",
]


# ------------------------------------------------------------------ colour

@dataclass(frozen=True)
class Look:
    """How one series is coloured: a flat colour, or a map run along time."""

    colour: str | None = None
    cmap: Any = None

    @property
    def is_map(self) -> bool:
        return self.cmap is not None


@dataclass(frozen=True)
class Mark:
    """A value worth a line across a panel: a median, a null level, a cut-off.

    ``at`` is where the label sits as a fraction of the axis, so two marks close
    together can be labelled without the words landing on top of each other.
    """

    value: float
    label: str = ""
    role: str = "reference"
    dashed: bool = False
    at: float = 0.94


def series_colours(theme: Any, count: int) -> list[str]:
    """One distinguishable colour per series, however many there are.

    The theme's role cycle is the right answer up to its length and refuses
    beyond it, on the ground that repeating a colour draws two series the
    reader will read as one. That refusal is correct and unhelpful to a page
    with seven fitted signals on it, so past the cycle this samples the theme's
    sequential map instead.

    The two are not interchangeable and the switch is visible on the page: role
    colours are categorical and unordered, a sampled map is ordered. A figure
    using this must therefore have already put its series in a stated order, or
    the reader will infer one from the colours that is not there.
    """
    import matplotlib

    if count <= len(theme.cycle()):
        return theme.cycle(count)
    cmap = matplotlib.colormaps[theme.style["sequential_cmap"]]
    return [matplotlib.colors.to_hex(cmap(position))
            for position in np.linspace(0.08, 0.92, count)]


#: One line break, named so that source edits cannot turn it into a real one.
NEWLINE = chr(10)


def resolve_look(theme: Any, wanted: str | None, fallback_role: str = "morphology") -> Look:
    """Turn one user string into either a flat colour or a colour map.

    A name matplotlib knows as a colour map becomes a map; anything else is
    asked of the theme, which accepts a role, a palette name or a hex value and
    refuses a typo rather than quietly drawing in the wrong colour.
    """
    import matplotlib

    if not wanted:
        return Look(colour=theme.colour(fallback_role))
    try:
        return Look(cmap=matplotlib.colormaps[wanted])
    except (KeyError, AttributeError):
        pass
    # Auto-Organotypic owns the laboratory's established bioluminescence
    # lookup tables.  Asking for one here should paint a still exactly as that
    # package paints a movie, rather than approximating it with a theme colour.
    try:
        from auto_organotypic.render import luts as organotypic_luts

        try:
            return Look(cmap=organotypic_luts.colormap(wanted))
        except ValueError:
            # ImageJ-style single-colour LUTs are ramps rather than anchored
            # maps. Convert their exact 256-level table into a Matplotlib map
            # instead of falling through to a flat theme colour, which image
            # panels cannot use to encode intensity.
            if wanted in organotypic_luts.LUT_RAMPS:
                levels = np.linspace(0.0, 1.0, 256, dtype=np.float32)
                rgb = organotypic_luts.paint(levels, wanted).astype(float) / 255.0
                return Look(cmap=matplotlib.colors.ListedColormap(rgb, name=wanted))
    except ImportError:
        pass
    return Look(colour=theme.colour(wanted))


def _bare_axes(ax: Any) -> None:
    """No ticks, no frame - for a panel whose whole content is its colour."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------- image tiles

def image_tile(
    ax: Any,
    image: np.ndarray,
    theme: Any,
    *,
    mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    look: Look | None = None,
    title: str = "",
    outline_role: str = "outline",
) -> PanelResult:
    """One image tile, optionally with a mask boundary drawn over it.

    ``vmin``/``vmax`` are the caller's, deliberately: a strip of tiles from one
    cell must share one contrast or the reader sees brightness changes that are
    display, not signal.
    """
    from matplotlib.colors import to_rgba

    look = look or Look(cmap=None, colour=None)
    cmap = look.cmap if look.is_map else theme["image_cmap"]
    handle = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    if mask is not None and mask.any():
        from skimage.segmentation import find_boundaries

        overlay = np.zeros((*mask.shape, 4))
        overlay[find_boundaries(mask, mode="outer")] = to_rgba(theme.colour(outline_role))
        ax.imshow(overlay, interpolation="nearest")

    _bare_axes(ax)
    if title:
        ax.set_title(
            title, fontsize=theme.size("tile"), pad=5, color=theme.colour("caption")
        )
    # The pixels are not the table. They are already copied and hashed under the
    # bundle's data/src, and a CSV of them would be a second, worse copy; what a
    # reader needs recorded is how they were displayed.
    return PanelResult(
        data=pd.DataFrame([{
            "title": title,
            "rows": int(image.shape[0]) if hasattr(image, "shape") else 0,
            "columns": int(image.shape[1]) if hasattr(image, "shape") else 0,
            "vmin": float(vmin) if vmin is not None else float("nan"),
            "vmax": float(vmax) if vmax is not None else float("nan"),
            "mask_px": int(np.count_nonzero(mask)) if mask is not None else 0,
        }]),
        axes=ax, extra={"handle": handle},
    )


def image_strip(
    figure: Any,
    rect: tuple[float, float, float, float],
    images: Sequence[np.ndarray],
    theme: Any,
    *,
    masks: Sequence[np.ndarray] | None = None,
    titles: Sequence[str] | None = None,
    look: Look | None = None,
    outline_role: str = "outline",
    gap: float = 0.0022,
    percentiles: tuple[float, float] = (2.0, 99.5),
    vmin: float | None = None,
    vmax: float | None = None,
) -> PanelResult:
    """A row of image tiles across ``rect`` = (left, bottom, width, height).

    One row per tile - which tile, titled what, at what contrast - with the tile
    axes beside it and the contrast limits in ``extra``, because a caller
    matching a second strip to this one needs the numbers rather than a lookup.
    One contrast for the whole strip, taken from the pooled pixels, because the
    strip's job is to show what changed and a per-tile stretch would hide
    exactly that.
    """
    images = list(images)
    if not images:
        return PanelResult(
            data=pd.DataFrame(columns=["index", "title", "vmin", "vmax"]),
            axes=[], extra={"vmin": float("nan"), "vmax": float("nan")},
        )

    if vmin is None or vmax is None:
        pooled = np.concatenate([np.asarray(image, dtype=float).ravel() for image in images])
        low, high = np.percentile(pooled, percentiles)
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax

    left, bottom, width, height = rect
    each = (width - gap * (len(images) - 1)) / len(images)
    axes = []
    for position, image in enumerate(images):
        ax = figure.add_axes([left + position * (each + gap), bottom, each, height])
        image_tile(
            ax, image, theme,
            mask=None if masks is None else masks[position],
            vmin=vmin, vmax=vmax, look=look, outline_role=outline_role,
            title="" if titles is None else titles[position],
        )
        axes.append(ax)
    return PanelResult(
        data=pd.DataFrame({
            "index": range(len(images)),
            "title": ["" if titles is None else titles[i] for i in range(len(images))],
            "vmin": float(vmin), "vmax": float(vmax),
        }),
        axes=axes, extra={"vmin": float(vmin), "vmax": float(vmax)},
    )


def paired_image_strip(
    figure: Any,
    rect: tuple[float, float, float, float],
    before_images: Sequence[np.ndarray],
    after_images: Sequence[np.ndarray],
    theme: Any,
    *,
    before_masks: Sequence[np.ndarray] | None = None,
    after_masks: Sequence[np.ndarray] | None = None,
    titles: Sequence[tuple[str, str]] | None = None,
    before_role: str = "motility",
    after_role: str = "outline",
    within_pair_gap: float = 0.0022,
    between_pair_gap: float = 0.018,
    percentiles: tuple[float, float] = (2.0, 99.5),
    vmin: float | None = None,
    vmax: float | None = None,
) -> PanelResult:
    """Matched before/after crops with one contrast and visible pair gaps.

    The before outline is repeated over the after image, so retained and newly
    occupied regions can be read spatially rather than inferred from two nearby
    pictures. The caller supplies equally scaled crops, keeping selection and
    scientific interpretation outside this generic image-grid panel.
    """
    from matplotlib.colors import to_rgba
    from skimage.segmentation import find_boundaries

    before_images = list(before_images)
    after_images = list(after_images)
    if len(before_images) != len(after_images):
        raise ValueError("paired_image_strip needs one after image for every before image")
    count = len(before_images)
    if before_masks is not None and len(before_masks) != count:
        raise ValueError("paired_image_strip needs one before mask per image pair")
    if after_masks is not None and len(after_masks) != count:
        raise ValueError("paired_image_strip needs one after mask per image pair")
    if titles is not None and len(titles) != count:
        raise ValueError("paired_image_strip needs one title pair per image pair")
    if not count:
        return PanelResult(
            data=pd.DataFrame(columns=["pair", "state", "title", "vmin", "vmax"]),
            axes=[], extra={"vmin": float("nan"), "vmax": float("nan")},
        )

    images = [image for pair in zip(before_images, after_images) for image in pair]
    if vmin is None or vmax is None:
        pooled = np.concatenate([np.asarray(image, dtype=float).ravel() for image in images])
        low, high = np.percentile(pooled, percentiles)
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax

    left, bottom, width, height = rect
    available = width - count * within_pair_gap - (count - 1) * between_pair_gap
    each = available / (2 * count)
    if each <= 0:
        raise ValueError("paired-image gaps leave no room for the image tiles")

    axes = []
    rows: list[dict[str, Any]] = []
    x = left
    for pair in range(count):
        pair_titles = ("Before", "After") if titles is None else titles[pair]
        pair_images = (before_images[pair], after_images[pair])
        pair_masks = (
            None if before_masks is None else before_masks[pair],
            None if after_masks is None else after_masks[pair],
        )
        for state, image, mask, title in zip(
            ("before", "after"), pair_images, pair_masks, pair_titles
        ):
            ax = figure.add_axes([x, bottom, each, height])
            role = before_role if state == "before" else after_role
            drawn = image_tile(
                ax, image, theme, mask=mask, vmin=vmin, vmax=vmax,
                title=title, outline_role=role,
            )
            reference_px = 0
            if state == "after" and before_masks is not None:
                reference = np.asarray(before_masks[pair], dtype=bool)
                if reference.any():
                    overlay = np.zeros((*reference.shape, 4))
                    overlay[find_boundaries(reference, mode="outer")] = to_rgba(
                        theme.colour(before_role)
                    )
                    ax.imshow(overlay, interpolation="nearest")
                    reference_px = int(np.count_nonzero(reference))
            row = drawn.data.iloc[0].to_dict()
            row.update({
                "pair": pair,
                "state": state,
                "outline_role": role,
                "before_reference_px": reference_px,
            })
            rows.append(row)
            axes.append(ax)
            x += each
            if state == "before":
                x += within_pair_gap
        if pair < count - 1:
            x += between_pair_gap

    return PanelResult(
        data=pd.DataFrame(rows), axes=axes,
        extra={"vmin": float(vmin), "vmax": float(vmax)},
    )


# ----------------------------------------------------------------- traces

def trace(
    ax: Any,
    hours: Sequence[float],
    values: Sequence[float],
    theme: Any,
    *,
    look: Look | None = None,
    role: str = "morphology",
    label: str = "",
    linewidth: float | None = None,
    hour_ticks: float | None = None,
    show_x: bool = True,
    x_label: str = "Hours from start of recording",
    y_label: str = "",
) -> PanelResult:
    """One metric against time.

    With a flat colour this is a plain line. With a colour map it is the same
    line coloured along the map by elapsed time, which is how a "LUT for this
    graph" is honoured without changing what is plotted.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    look = look or Look(colour=theme.colour(role))
    width = theme["line_width"] if linewidth is None else linewidth

    if look.is_map and hours.size > 1:
        from matplotlib.collections import LineCollection

        points = np.stack([hours, values], axis=1)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(
            segments, cmap=look.cmap, array=hours[:-1], linewidths=width,
            capstyle="round", label=label or None,
        )
        collection.set_clim(float(np.nanmin(hours)), float(np.nanmax(hours)))
        ax.add_collection(collection)
        ax.set_xlim(float(np.nanmin(hours)), float(np.nanmax(hours)))
        finite = values[np.isfinite(values)]
        if finite.size:
            pad = (finite.max() - finite.min()) * 0.08 or 1.0
            ax.set_ylim(finite.min() - pad, finite.max() + pad)
    else:
        ax.plot(hours, values, color=look.colour, linewidth=width, label=label or None)
        ax.set_xlim(float(np.nanmin(hours)), float(np.nanmax(hours)))

    if y_label:
        ax.set_ylabel(y_label, fontsize=theme.size("panel"))
    ax.set_xticks(theme.hour_ticks(float(np.nanmin(hours)), float(np.nanmax(hours)), hour_ticks))
    if show_x:
        if x_label:
            ax.set_xlabel(x_label)
    else:
        ax.tick_params(labelbottom=False)
    return PanelResult(
        data=pd.DataFrame({"hours": hours, "value": values}), axes=ax)


def trace_overlay(
    ax: Any,
    hours: Sequence[float],
    series: dict[str, Sequence[float]],
    theme: Any,
    *,
    looks: dict[str, Look] | None = None,
    labels: dict[str, str] | None = None,
    normalise: str = "z",
    hour_ticks: float | None = None,
    x_label: str = "Hours from start of recording",
) -> PanelResult:
    """Several named traces on one axis, normally in within-series SD units.

    Measurements with different units cannot honestly share one vertical axis.
    The default therefore centres each trace and divides it by its own standard
    deviation. ``normalise='none'`` remains available when the caller knows the
    series already share a unit.
    """
    hours = np.asarray(hours, dtype=float)
    looks = looks or {}
    labels = labels or {}
    rows = []
    for index, (name, source) in enumerate(series.items()):
        values = np.asarray(source, dtype=float)
        if values.shape != hours.shape:
            raise ValueError(
                f"trace {name!r} has {values.size} values for {hours.size} times")
        if normalise == "z":
            spread = float(np.nanstd(values))
            plotted = ((values - float(np.nanmean(values))) / spread
                       if np.isfinite(spread) and spread > 0
                       else np.zeros_like(values))
        elif normalise == "none":
            plotted = values.copy()
        else:
            raise ValueError("normalise must be z or none")
        look = looks.get(name) or Look(
            colour=series_colours(theme, len(series))[index])
        trace(
            ax, hours, plotted, theme, look=look,
            label=labels.get(name, name.replace("_", " ")),
            hour_ticks=hour_ticks, x_label=x_label,
        )
        rows.extend({
            "series": name,
            "hours": float(hour),
            "source_value": float(raw),
            "value": float(shown),
            "normalise": normalise,
        } for hour, raw, shown in zip(hours, values, plotted))
    ax.set_ylabel("Within-trace standard deviations" if normalise == "z" else "Value")
    ax.legend(frameon=False, ncol=max(1, min(3, len(series))))
    return PanelResult(data=pd.DataFrame(rows), axes=ax)


def cosinor_curve(
    ax: Any,
    hours: Sequence[float],
    values: Sequence[float],
    theme: Any,
    *,
    period_hours: float = 24.0,
    detrend: str = str(workbench.DETREND_DEFAULTS["detrend"]),
    detrend_window_hours: float = float(
        workbench.DETREND_DEFAULTS["detrend_window_hours"]),
    detrend_options: dict | None = None,
    role: str = "fit",
    linestyle: tuple = (0, (6, 3)),
) -> PanelResult:
    """A fixed-period cosine over the selected Workbench baseline.

    Refitted here rather than read from ``rhythms.csv`` because the table stores
    the fit's parameters, not the curve; this is the same model the rhythms
    module fits, evaluated at the plotted hours. The fitted values come back so a
    caller can write them out beside the figure.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    usable = np.isfinite(hours) & np.isfinite(values)
    if usable.sum() < 4:
        empty = np.full(hours.shape, np.nan)
        return PanelResult(
            data=pd.DataFrame({"hours": hours, "fitted": empty}),
            axes=ax, extra={"fitted": empty},
        )

    result = workbench.detrend_trace(
        hours[usable], values[usable],
        {**dict(detrend_options or {}), "detrend": detrend,
         "detrend_window_hours": detrend_window_hours},
    )
    residual = np.full(hours.shape, np.nan, dtype=float)
    baseline = np.full(hours.shape, np.nan, dtype=float)
    residual[usable] = np.asarray(result["values"], dtype=float)
    baseline[usable] = np.asarray(result["baseline"], dtype=float)
    panel = harmonic_curve(
        ax, hours, residual, theme, period_hours=period_hours,
        baseline=baseline, role=role, linestyle=linestyle,
    )
    return PanelResult(
        data=panel.data.assign(
            baseline=baseline,
            detrend=detrend,
            detrend_window_hours=float(detrend_window_hours),
        ),
        axes=ax,
        extra={**panel.extra, "baseline": baseline, "detrended": residual},
    )


def harmonic_curve(
    ax: Any,
    hours: Sequence[float],
    values: Sequence[float],
    theme: Any,
    *,
    period_hours: float,
    baseline: Sequence[float] | None = None,
    role: str = "fit",
    colour: str | None = None,
    linestyle: tuple = (0, (6, 3)),
) -> PanelResult:
    """An explicit descriptive Workbench cosinor at an estimated period."""
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    usable = np.isfinite(hours) & np.isfinite(values)
    fitted = np.full(hours.shape, np.nan, dtype=float)
    if usable.sum() < 4 or not np.isfinite(period_hours) or period_hours <= 0:
        return PanelResult(
            data=pd.DataFrame({"hours": hours, "fitted": fitted}),
            axes=ax, extra={"fitted": fitted},
        )
    fitted = workbench.descriptive_cosinor_fitted_values(
        hours, values, float(period_hours)
    )
    if baseline is not None:
        baseline_values = np.asarray(baseline, dtype=float)
        if baseline_values.shape != hours.shape:
            raise ValueError("harmonic baseline must have one value per hour")
        fitted = fitted + baseline_values
    ax.plot(
        hours, fitted, color=colour or theme.colour(role),
        linewidth=theme.stroke("emphasis"), linestyle=linestyle,
    )
    return PanelResult(
        data=pd.DataFrame({"hours": hours, "fitted": fitted}),
        axes=ax, extra={"fitted": fitted},
    )


def trace_stack(
    figure: Any,
    rect: tuple[float, float, float, float],
    hours: Sequence[float],
    columns: Iterable[str],
    frame: Any,
    theme: Any,
    *,
    looks: dict[str, Look] | None = None,
    interval_minutes: float | None = None,
    hour_ticks: float | None = None,
    spacing: float = 0.045,
    x_label: str = "Hours from start of recording",
) -> PanelResult:
    """A column of traces sharing one time axis, top to bottom.

    ``rect`` is the whole block, (left, bottom, width, height); the panels divide
    it evenly. Only the bottom panel carries tick labels and the x-axis label,
    because the stack is read as one axis - and they all get the same day-aligned
    ticks, so a peak in one lines up with a peak in another.
    """
    from _metrics import axis_label, role_for

    columns = list(columns)
    if not columns:
        return PanelResult(
            data=pd.DataFrame(columns=["series", "hours", "value"]), axes=[])
    looks = looks or {}
    left, bottom, width, height = rect
    each = (height - spacing * (len(columns) - 1)) / len(columns)

    axes = []
    for index, column in enumerate(columns):
        top_down = len(columns) - 1 - index          # index 0 is the top panel
        ax = figure.add_axes([left, bottom + top_down * (each + spacing), width, each])
        trace(
            ax, hours, frame[column], theme,
            look=looks.get(column) or Look(colour=theme.colour(role_for(column))),
            hour_ticks=hour_ticks,
            show_x=(index == len(columns) - 1),
            x_label=x_label,
            y_label=axis_label(column, interval_minutes),
        )
        axes.append(ax)
    drawn = pd.concat(
        [pd.DataFrame({"series": column,
                       "hours": np.asarray(hours, dtype=float),
                       "value": np.asarray(frame[column], dtype=float)})
         for column in columns],
        ignore_index=True,
    )
    return PanelResult(data=drawn, axes=axes)


# ----------------------------------------------------------------- rasters

def raster(
    ax: Any,
    matrix: Any,
    theme: Any,
    *,
    hours: Sequence[float] | None = None,
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    missing_role: str = "missing",
    interpolation: str = "nearest",
    aspect: str = "auto",
) -> PanelResult:
    """One row per thing, one column per frame, colour is the value.

    Rows and hours rather than rows and columns: the x axis is drawn in the
    movie's own hours so it can be read against every other time axis in the
    package. Cells with no value take the background colour, which is why a
    missing measurement looks like nothing rather than like zero.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows = matrix.shape[0]
    if hours is not None:
        hours = np.asarray(hours, dtype=float)
        extent = (float(hours.min()), float(hours.max()), rows - 0.5, -0.5)
    else:
        extent = (-0.5, matrix.shape[1] - 0.5, rows - 0.5, -0.5)
    handle = ax.imshow(
        matrix, aspect=aspect, cmap=theme["sequential_cmap"] if cmap is None else cmap,
        vmin=vmin, vmax=vmax, extent=extent, interpolation=interpolation,
    )
    ax.set_facecolor(theme.colour(missing_role))
    # Long form, one row per drawn cell: a wide matrix in a CSV cannot say which
    # column is which hour, and that is the axis the figure is read against.
    row_index, column_index = np.indices(matrix.shape)
    across = (np.asarray(hours, dtype=float)[column_index.ravel()]
              if hours is not None else column_index.ravel().astype(float))
    return PanelResult(
        data=pd.DataFrame({
            "row": row_index.ravel(),
            "hours" if hours is not None else "column": across,
            "value": matrix.ravel(),
        }),
        axes=ax, extra={"handle": handle},
    )


def category_strip(ax: Any, blocks: Sequence[tuple[int, str]], theme: Any) -> PanelResult:
    """A thin bar of solid blocks beside a raster, saying which rows are which.

    ``blocks`` is (how many rows, which colour role), top to bottom. It carries
    no scale and no ticks because it is not a measurement - it is the key to the
    rows, drawn where the rows are rather than in a legend the eye has to travel
    to.
    """
    from matplotlib.colors import ListedColormap

    blocks = [(int(count), role) for count, role in blocks]
    total = sum(count for count, _ in blocks)
    drawn = pd.DataFrame({
        "index": range(len(blocks)),
        "rows": [count for count, _ in blocks],
        "role": [role for _, role in blocks],
    })
    if not total:
        _bare_axes(ax)
        return PanelResult(data=drawn, axes=ax)
    codes = np.concatenate(
        [np.full((count, 1), index, dtype=float) for index, (count, _) in enumerate(blocks)]
    )
    handle = ax.imshow(
        codes, aspect="auto", interpolation="nearest",
        cmap=ListedColormap([theme.colour(role) for _, role in blocks]),
        vmin=0, vmax=max(len(blocks) - 1, 1),
        extent=(0, 1, total - 0.5, -0.5),
    )
    _bare_axes(ax)
    return PanelResult(data=drawn, axes=ax, extra={"handle": handle})


def colour_bar(
    figure: Any,
    mappable: Any,
    rect: tuple[float, float, float, float],
    theme: Any,
    *,
    label: str = "",
) -> "Colorbar":
    """The key to a raster or a coloured collection, in its own rectangle.

    A key rather than a panel: it records nothing the mappable it points at has
    not already recorded, so it gives back the colour bar itself.
    """
    ax = figure.add_axes(rect)
    bar = figure.colorbar(mappable, cax=ax)
    if label:
        bar.set_label(label, fontsize=theme.size("annotation"))
    bar.ax.tick_params(labelsize=theme.size("caption"))
    return bar


def inset_colour_bar(
    ax: Any,
    mappable: Any,
    theme: Any,
    *,
    label: str,
    ticks: Sequence[float] | None = None,
    ticklabels: Sequence[str] | None = None,
    width_inches: float | None = None,
    height_inches: float | None = None,
    gap_inches: float | None = None,
) -> "Colorbar":
    """The canonical vertical colour-map key, in physical rather than relative size."""
    figure = ax.figure
    position = ax.get_position()
    scale = theme.canvas_scale
    width = float(theme["colour_bar_width_inches"] if width_inches is None
                  else width_inches)
    height = float(theme["colour_bar_height_inches"] if height_inches is None
                   else height_inches)
    gap = float(theme["colour_bar_gap_inches"] if gap_inches is None else gap_inches)
    if not all(np.isfinite(value) for value in (width, height, gap)):
        raise ValueError("colour-bar dimensions must be finite")
    if width <= 0 or height <= 0 or gap < 0:
        raise ValueError(
            "colour-bar width and height must be positive and its gap non-negative"
        )
    width_fraction = width * scale / figure.get_figwidth()
    height_fraction = min(height * scale / figure.get_figheight(), position.height)
    gap_fraction = gap * scale / figure.get_figwidth()
    colour_ax = figure.add_axes([
        position.x1 + gap_fraction,
        position.y0 + (position.height - height_fraction) / 2,
        width_fraction,
        height_fraction,
    ])
    bar = figure.colorbar(mappable, cax=colour_ax, ticks=ticks)
    bar.set_label(label, fontsize=theme.size("annotation"))
    bar.ax.tick_params(
        labelsize=theme.size("caption"), width=theme.stroke("hairline"),
        length=theme["tick_mark_length"] * 0.65,
    )
    bar.outline.set_linewidth(theme.stroke("hairline"))
    if ticklabels is not None:
        bar.ax.set_yticklabels(list(ticklabels))
    return bar


def semantic_legend(
    ax: Any,
    theme: Any,
    *,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    location: str = "above",
    columns: int | None = None,
) -> "Legend | None":
    """Draw one deduplicated key for every labelled colour or mark."""
    # A figure-level key can cover this axes.  Record that the semantic key has
    # been considered so the finishing pass does not silently add a duplicate.
    ax._semantic_legend_handled = True
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and not str(label).startswith("_"):
            unique.setdefault(str(label), handle)
    if not unique:
        return None
    return theme.legend(
        ax, list(unique.values()), list(unique), location=location,
        columns=columns,
    )


# -------------------------------------------------------------- histograms

def histogram(
    ax: Any,
    values: Sequence[float],
    theme: Any,
    *,
    bins: int | Sequence[float] = 24,
    role: str = "morphology",
    look: Look | None = None,
    log_x: bool = False,
    orientation: str = "vertical",
    alpha: float = 1.0,
    edge: bool = True,
    x_label: str = "",
    y_label: str = "",
    label: str = "",
) -> PanelResult:
    """How many observations fall in each band: one row per band.

    The bands come back because the plotted table has to exist as a CSV beside
    the figure, and rebuilding the bins a second time in the builder is how the
    two quietly stop matching.

    ``log_x`` builds the bands evenly on a log axis. A distribution spanning
    three decades put in even bands is one tall bar and forty empty ones.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    look = look or Look(colour=theme.colour(role))
    colour = look.colour or theme.colour(role)

    if isinstance(bins, (int, np.integer)):
        if log_x:
            positive = values[values > 0]
            low = float(positive.min()) if positive.size else 1e-3
            high = float(positive.max()) if positive.size else 1.0
            edges = np.logspace(np.log10(low), np.log10(high), int(bins) + 1)
            # np.logspace goes through log10 and back, so its end points land
            # within about 1e-14 of low and high rather than on them - and
            # np.histogram drops an observation above the last edge. Whether
            # the single largest value was counted therefore came down to which
            # side of that 1e-14 it fell on, silently, half the time. Pin the
            # ends, which is what np.linspace already guarantees for the even
            # bands below.
            edges[0], edges[-1] = low, high
        else:
            edges = np.linspace(float(values.min()), float(values.max()), int(bins) + 1) \
                if values.size else np.linspace(0.0, 1.0, int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)

    ax.hist(
        values, bins=edges, color=colour, alpha=alpha, orientation=orientation,
        label=label or None,
        edgecolor=theme.colour("page") if edge else "none",
        linewidth=theme.stroke("hairline") if edge else 0.0,
    )
    if log_x:
        (ax.set_yscale if orientation == "horizontal" else ax.set_xscale)("log")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    counts, _ = np.histogram(values, bins=edges)
    return PanelResult(
        data=pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:],
                           "count": counts}),
        axes=ax, extra={"counts": counts, "edges": edges},
    )


def reference_lines(
    ax: Any,
    marks: Iterable[Mark],
    theme: Any,
    *,
    orientation: str = "vertical",
    offset: float = 1.09,
) -> PanelResult:
    """Lines across a panel at values read off the data - and only those.

    Every mark here comes from the distribution being drawn. A threshold that
    lives somewhere else - in the tracker's configuration, in a paper - is not
    drawn by this package at all: a quoted number keeps being quoted after
    somebody changes it.
    """
    draw = ax.axvline if orientation == "vertical" else ax.axhline
    low, high = (ax.get_ylim() if orientation == "vertical" else ax.get_xlim())
    marks = list(marks)
    for mark in marks:
        colour = theme.colour(mark.role)
        draw(
            mark.value, color=colour, linewidth=theme.stroke("emphasis"),
            linestyle=(0, (5, 3)) if mark.dashed else "-",
        )
        if not mark.label:
            continue
        along = low + (high - low) * mark.at
        if orientation == "vertical":
            ax.text(
                mark.value * offset if ax.get_xscale() == "log" else mark.value,
                along, mark.label, fontsize=theme.size("annotation"),
                color=colour, va="top", ha="left",
            )
        else:
            ax.text(
                along, mark.value, mark.label, fontsize=theme.size("annotation"),
                color=colour, va="bottom", ha="right",
            )
    # Where a line was drawn is exactly the part of a figure a reader questions,
    # so it is recorded rather than left to the caption.
    return PanelResult(
        data=pd.DataFrame([{"value": mark.value, "label": mark.label,
                            "role": mark.role, "dashed": mark.dashed,
                            "orientation": orientation} for mark in marks]),
        axes=ax,
    )


# ---------------------------------------------------------------- scatters

def scatter(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    theme: Any,
    *,
    sizes: Sequence[float] | None = None,
    role: str = "morphology",
    look: Look | None = None,
    colours: Any = None,
    alpha: float = 0.55,
    edge_role: str = "page",
    label: str = "",
) -> PanelResult:
    """One dot per thing. ``sizes`` is a column, already scaled by the caller."""
    look = look or Look(colour=theme.colour(role))
    handle = ax.scatter(
        x, y,
        s=theme.point_area(1.2) if sizes is None else sizes,
        color=None if colours is not None else (look.colour or theme.colour(role)),
        c=colours,
        cmap=look.cmap if look.is_map else None,
        alpha=alpha,
        edgecolor=theme.colour(edge_role) if edge_role else "none",
        linewidth=theme.stroke("hairline") * 2 if edge_role else 0.0,
        zorder=3, label=label or None,
    )
    drawn = pd.DataFrame({"x": np.asarray(x, dtype=float),
                          "y": np.asarray(y, dtype=float)})
    if sizes is not None:
        drawn["size"] = np.asarray(sizes, dtype=float)
    return PanelResult(data=drawn, axes=ax, extra={"handle": handle})


def scatter_with_margins(
    figure: Any,
    rect: tuple[float, float, float, float],
    x: Sequence[float],
    y: Sequence[float],
    theme: Any,
    *,
    sizes: Sequence[float] | None = None,
    role: str = "morphology",
    look: Look | None = None,
    bins: int = 20,
    log_x: bool = False,
    margins: bool = True,
    margin: float = 0.15,
    gap: float = 0.025,
    alpha: float = 0.55,
) -> PanelResult:
    """A scatter with the distribution of each axis drawn along its edge.

    The two edge histograms are the same numbers as the dots, seen one axis at a
    time - which is what makes a cloud with no visible structure readable: the
    reader can still see where most of the cells are on each axis.

    ``axes`` is (main, top, right); the last two are ``None`` when ``margins``
    is off, which is how a caller drops them without a second layout.
    """
    left, bottom, width, height = rect
    main_w = width * (1 - margin - gap) if margins else width
    main_h = height * (1 - margin - gap) if margins else height

    ax = figure.add_axes([left, bottom, main_w, main_h])
    ax_top = ax_right = None
    if margins:
        ax_top = figure.add_axes(
            [left, bottom + main_h + height * gap, main_w, height * margin], sharex=ax
        )
        ax_right = figure.add_axes(
            [left + main_w + width * gap, bottom, width * margin, main_h], sharey=ax
        )

    drawn = scatter(ax, x, y, theme, sizes=sizes, role=role, look=look, alpha=alpha)
    if log_x:
        ax.set_xscale("log")

    if ax_top is not None:
        histogram(ax_top, x, theme, bins=bins, role=role, log_x=log_x,
                  alpha=0.65, edge=False)
        ax_top.set_yticks([])
        ax_top.tick_params(labelbottom=False)
        for side in ("left", "top", "right"):
            ax_top.spines[side].set_visible(False)
    if ax_right is not None:
        histogram(ax_right, y, theme, bins=bins, role=role,
                  orientation="horizontal", alpha=0.65, edge=False)
        ax_right.set_xticks([])
        ax_right.tick_params(labelleft=False)
        for side in ("bottom", "top", "right"):
            ax_right.spines[side].set_visible(False)
    return PanelResult(data=drawn.data, axes=(ax, ax_top, ax_right))


# ------------------------------------------------------- one row per thing

def dumbbell(
    ax: Any,
    left: Sequence[float],
    right: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str],
    right_colours: Sequence[str] | None = None,
    connector_role: str = "reference",
    hollow_role: str = "reference",
    left_marker: str = "o",
    right_marker: str = "o",
    annotations: Sequence[str] | None = None,
    annotation_colours: Sequence[str] | None = None,
    left_label: str = "",
    right_label: str = "",
) -> PanelResult:
    """Two dots per row joined by a line: where a thing was, where it is.

    The line is the point of the panel. Two bar charts side by side ask the
    reader to subtract; a dumbbell draws the difference as a length, so the rows
    where it matters are the long ones.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    positions = np.arange(len(labels))

    for y, (start, end) in enumerate(zip(left, right)):
        ax.plot(
            [start, end], [y, y], color=theme.colour(connector_role),
            linewidth=theme.stroke("emphasis"), zorder=1, solid_capstyle="round",
        )
    ax.scatter(
        left, positions, s=theme.point_area(1.5), color=theme.colour("page"),
        edgecolor=theme.colour(hollow_role), linewidth=theme.stroke("emphasis"),
        marker=left_marker, zorder=3, label=left_label or None,
    )
    ax.scatter(
        right, positions, s=theme.point_area(1.5),
        color=list(right_colours) if right_colours is not None else theme.colour("significant"),
        edgecolor="none", marker=right_marker, zorder=4, label=right_label or None,
    )

    if annotations is not None:
        for y, text in enumerate(annotations):
            ax.text(
                max(left[y], right[y]) + 0.012, y, text, va="center",
                fontsize=theme.size("subtitle"),
                color=(annotation_colours[y] if annotation_colours is not None
                       else theme.colour("ink")),
                fontweight="bold",
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=theme.size("annotation"))
    ax.set_ylim(-0.7, len(labels) - 0.3)
    return PanelResult(
        data=pd.DataFrame({"row": positions, "label": list(labels),
                           "left": left, "right": right}),
        axes=ax,
    )


def lollipop(
    ax: Any,
    values: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str],
    colours: Sequence[str] | None = None,
    role: str = "morphology",
    annotations: Sequence[str] | None = None,
    baseline: float = 0.0,
) -> PanelResult:
    """One dot per row on a stem from a baseline: a ranked list with a scale.

    A bar chart of the same numbers spends most of its ink on the part of the
    bar nobody reads. The stem carries the eye to the value and the dot is the
    value, so a row can be compared to its neighbour and to the scale at once.
    """
    values = np.asarray(values, dtype=float)
    positions = np.arange(len(labels))
    palette = list(colours) if colours is not None else [theme.colour(role)] * len(values)

    for y, (value, colour) in enumerate(zip(values, palette)):
        ax.plot(
            [baseline, value], [y, y], color=colour, linewidth=theme.stroke("emphasis"),
            alpha=0.45, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            value, y, s=theme.point_area(1.7), color=colour, zorder=3, edgecolor="none",
        )
    if annotations is not None:
        # Away from the baseline, not always to the right, and offset by a share
        # of the data rather than by a fixed 0.045. A right-hand annotation on a
        # negative row sits on top of the stem it is annotating, and a fixed
        # offset is either invisible or a mile off as soon as an axis is not
        # roughly unit-scaled.
        span = float(np.nanmax(np.abs(values - baseline))) if len(values) else 0.0
        pad = 0.035 * (span or 1.0)
        widest = max((len(str(text)) for text in annotations), default=0)
        for y, (value, text, colour) in enumerate(zip(values, annotations, palette)):
            beyond = not (value < baseline)
            ax.text(
                value + (pad if beyond else -pad), y, text, va="center",
                ha="left" if beyond else "right", fontsize=theme.size("emphasis"),
                color=colour, fontweight="bold",
            )
        # Room for the words. Matplotlib sizes an axes to its marks and not to
        # the text beside them, so without this the longest annotation runs off
        # the panel and lands on the row labels of the one next to it.
        low, high = ax.get_xlim()
        room = 0.055 * widest * (high - low)
        ax.set_xlim(low - (room if any(v < baseline for v in values) else 0.0),
                    high + (room if any(not (v < baseline) for v in values) else 0.0))

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=theme.size("panel"))
    ax.set_ylim(-0.65, len(labels) - 0.35)
    return PanelResult(
        data=pd.DataFrame({"row": positions, "label": list(labels),
                           "value": values, "baseline": float(baseline)}),
        axes=ax,
    )


def ranked_rate(
    ax: Any,
    frame: pd.DataFrame,
    theme: Any,
    *,
    numerator_column: str,
    denominator_column: str,
    unit_column: str = "identity",
    role: str = "highlight",
    reference: float | None = None,
    reference_label: str = "All observations",
    x_label: str = "Units, ranked by rate",
    y_label: str = "Detected observations (%)",
) -> PanelResult:
    """One rate per unit, ranked without confusing exposure with frequency."""
    needed = {unit_column, numerator_column, denominator_column}
    missing = needed - set(frame.columns)
    if missing:
        raise KeyError(f"ranked_rate is missing columns: {', '.join(sorted(missing))}")
    data = frame[[unit_column, numerator_column, denominator_column]].copy()
    data[numerator_column] = pd.to_numeric(data[numerator_column], errors="coerce")
    data[denominator_column] = pd.to_numeric(data[denominator_column], errors="coerce")
    data = data[
        data[numerator_column].notna()
        & data[denominator_column].notna()
        & (data[denominator_column] > 0)
    ].copy()
    data["rate"] = data[numerator_column] / data[denominator_column]
    data = data[np.isfinite(data["rate"])].copy()
    data = data.sort_values(
        ["rate", denominator_column, unit_column],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    data["rank"] = np.arange(1, len(data) + 1)
    data["rate_percent"] = 100.0 * data["rate"]

    if data.empty:
        ax.text(0.5, 0.5, "No measurable rates", ha="center", va="center",
                transform=ax.transAxes, color=theme.colour("caption"))
        ax.set_axis_off()
        return PanelResult(data=data, axes=ax)

    colour = theme.colour(role)
    ax.plot(
        data["rank"], data["rate_percent"], color=colour,
        linewidth=theme.stroke("guide"), alpha=0.38, zorder=2,
    )
    ax.scatter(
        data["rank"], data["rate_percent"], color=colour,
        s=theme.point_area(0.72), edgecolor="none", zorder=3,
    )
    if reference is not None and np.isfinite(reference):
        reference_percent = 100.0 * float(reference)
        ax.axhline(
            reference_percent, color=theme.colour("reference"),
            linewidth=theme.stroke("guide"), linestyle=(0, (5, 3)), zorder=1,
        )
        ax.text(
            0.99, reference_percent, f"{reference_label}: {reference_percent:.1f}%",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=theme.size("caption"), color=theme.colour("caption"),
        )
    ax.set_xlim(0.5, len(data) + 0.5)
    last_rank = int(data["rank"].iloc[-1])
    interior = np.linspace(1, last_rank, 5).round().astype(int)
    ax.set_xticks(sorted(set([1, last_rank, *interior.tolist()])))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    theme.set_value_ticks(
        ax, low=0.0, high=float(data["rate_percent"].max()),
        start=0.0, round_to=20.0,
    )
    return PanelResult(data=data, axes=ax)


# ---------------------------------------------------------- extended grammar

def phase_map(
    ax: Any,
    positions: Any,
    phases: Sequence[float],
    theme: Any,
    *,
    field: Mapping[str, float] | None = None,
    period: float = 24.0,
    phase_unit: str = "h",
    cmap: Any = None,
    labels: Sequence[Any] | None = None,
    point_scale: float = 1.8,
    invert_y: bool = True,
    x_label: str = "X position",
    y_label: str = "Y position",
) -> PanelResult:
    """Positions coloured by phase in any user-supplied cycle.

    ``period`` and ``phase_unit`` describe the values supplied by the caller;
    the panel only wraps and displays them.  It does not fit a rhythm.
    """
    import matplotlib

    points = np.asarray(positions, dtype=float)
    values = np.asarray(phases, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("phase_map positions must be an N by 2 array")
    if len(points) != len(values):
        raise ValueError("phase_map needs one phase value per position")
    if labels is not None and len(labels) != len(values):
        raise ValueError("phase_map needs one label per position")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("phase_map period must be finite and positive")
    if not np.isfinite(point_scale) or point_scale <= 0:
        raise ValueError("phase_map point_scale must be finite and positive")

    if cmap is None:
        chosen = matplotlib.colormaps["twilight_shifted"]
    elif isinstance(cmap, str):
        if cmap not in {"twilight", "twilight_shifted", "hsv"}:
            raise ValueError(f"phase_map needs a cyclic colour map, not {cmap!r}")
        chosen = matplotlib.colormaps[cmap]
    else:
        name = getattr(cmap, "name", "")
        if name not in {"twilight", "twilight_shifted", "hsv"}:
            raise ValueError(
                f"phase_map needs a cyclic colour map, not {name or 'an unnamed map'}"
            )
        chosen = cmap

    wrapped = np.mod(values, period)
    handle = ax.scatter(
        points[:, 0], points[:, 1], c=wrapped, cmap=chosen,
        vmin=0, vmax=period, s=theme.point_area(point_scale),
        edgecolor=theme.colour("page"), linewidth=theme.stroke("hairline"),
    )
    if field is not None:
        ax.set_xlim(0, float(field["width"]))
        ax.set_ylim((float(field["height"]), 0) if invert_y
                    else (0, float(field["height"])))
    elif invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    data = pd.DataFrame({
        "x": points[:, 0], "y": points[:, 1], "phase": wrapped,
        "period": float(period), "phase_unit": phase_unit,
    })
    if labels is not None:
        data.insert(0, "unit", list(labels))
    return PanelResult(data=data, axes=ax, extra={"handle": handle})

def rose(
    ax: Any,
    angles: Sequence[float],
    theme: Any,
    *,
    weights: Sequence[float] | None = None,
    bins: int = 24,
    period: float = 24.0,
    look: Look | None = None,
    role: str = "morphology",
    mean_vector: bool = True,
    label: str = "",
    tick_step: float | None = None,
    tick_values: Sequence[float] | None = None,
    tick_labels: Sequence[str] | None = None,
    phase_unit: str = "h",
) -> PanelResult:
    """A circular histogram on one wrapped period: one row per wedge."""
    if not np.isfinite(period) or period <= 0:
        raise ValueError("rose period must be finite and positive")
    values = np.asarray(angles, dtype=float)
    finite = np.isfinite(values)
    values = np.mod(values[finite], period)
    used_weights = None if weights is None else np.asarray(weights, dtype=float)[finite]
    edges = np.linspace(0.0, float(period), int(bins) + 1)
    counts, _ = np.histogram(values, bins=edges, weights=used_weights)
    theta = 2 * np.pi * edges / period
    centres = (theta[:-1] + theta[1:]) / 2
    colour = (look or Look(colour=theme.colour(role))).colour or theme.colour(role)
    ax.bar(
        centres, counts, width=np.diff(theta) * 0.92, bottom=0,
        color=colour, edgecolor=theme.colour("page"),
        linewidth=theme.stroke("hairline"), alpha=0.85, label=label or None,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if tick_values is None:
        step = float(period) / 4.0 if tick_step is None else float(tick_step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("rose tick_step must be finite and positive")
        ticks = np.arange(0.0, float(period), step)
    else:
        ticks = np.asarray(tick_values, dtype=float)
        if not np.isfinite(ticks).all():
            raise ValueError("rose tick_values must be finite")
        ticks = np.mod(ticks, period)
    if tick_labels is not None and len(tick_labels) != len(ticks):
        raise ValueError("rose needs one tick label per tick value")
    labels = (list(tick_labels) if tick_labels is not None else
              [f"{value:g} {phase_unit}".rstrip() for value in ticks])
    ax.set_xticks(2 * np.pi * ticks / period, labels=labels)
    if mean_vector and values.size:
        vector_weights = np.ones(values.size) if used_weights is None else used_weights
        radians = 2 * np.pi * values / period
        resultant = np.sum(vector_weights * np.exp(1j * radians))
        direction = float(np.angle(resultant) % (2 * np.pi))
        length = float(np.abs(resultant) / max(np.sum(np.abs(vector_weights)), 1e-12))
        ax.annotate(
            "", xy=(direction, max(float(np.max(counts)), 1.0) * length), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=theme.colour("ink"),
                            linewidth=theme.stroke("emphasis")),
        )
    return PanelResult(
        data=pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:],
                           "count": counts}),
        axes=ax, extra={"edges": edges, "counts": counts},
    )


def vectors(
    ax: Any,
    angles: Sequence[float],
    radii: Sequence[float],
    theme: Any,
    *,
    period: float = 24.0,
    colours: Sequence[str] | None = None,
    filled: Sequence[bool] | None = None,
    resultant: bool = True,
    role: str = "morphology",
    radius_label: str = "",
) -> PanelResult:
    """One polar arrow per row, with mean direction and concentration beside it."""
    angles = np.asarray(angles, dtype=float)
    radii = np.asarray(radii, dtype=float)
    usable = np.isfinite(angles) & np.isfinite(radii)
    angles, radii = np.mod(angles[usable], period), radii[usable]
    theta = 2 * np.pi * angles / period
    palette = list(colours) if colours is not None else [theme.colour(role)] * len(theta)
    if colours is not None:
        palette = [palette[index] for index, keep in enumerate(usable) if keep]
    fills = np.ones(len(theta), dtype=bool) if filled is None else np.asarray(filled, dtype=bool)[usable]
    for angle, radius, colour, is_filled in zip(theta, radii, palette, fills):
        ax.annotate(
            "", xy=(angle, radius), xytext=(angle, 0),
            arrowprops=dict(
                arrowstyle="-|>" if is_filled else "->", color=colour,
                linewidth=theme.stroke("line"), alpha=0.72,
            ),
        )
    total = float(np.sum(np.abs(radii)))
    complex_result = np.sum(radii * np.exp(1j * theta)) if len(theta) else 0j
    mean_radians = float(np.angle(complex_result) % (2 * np.pi)) if len(theta) else np.nan
    concentration = float(np.abs(complex_result) / total) if total else 0.0
    # Annotation endpoints do not contribute to Matplotlib's data limits.  Set
    # the radial scale explicitly or arrows longer than the default 0-1 range
    # are silently clipped, while short arrows sit inside a mostly empty dial.
    ceiling = float(np.nanmax(np.abs(radii))) if len(radii) else 1.0
    ax.set_ylim(0.0, ceiling * 1.08 if ceiling > 0 else 1.0)
    if resultant and len(theta):
        ax.annotate(
            "", xy=(mean_radians, ceiling * concentration), xytext=(mean_radians, 0),
            arrowprops=dict(arrowstyle="-|>", color=theme.colour("highlight"),
                            linewidth=theme.stroke("emphasis")),
        )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if radius_label:
        # Polar ``set_ylabel`` places the label through the western tick
        # labels.  Anchor it beyond the circular axes instead.
        ax.text(
            -0.24, 0.5, radius_label, transform=ax.transAxes, rotation=90,
            ha="center", va="center", color=theme.colour("ink"),
        )
    mean_period = mean_radians * period / (2 * np.pi) if np.isfinite(mean_radians) else np.nan
    return PanelResult(
        data=pd.DataFrame({"angle": angles, "radius": radii}),
        axes=ax,
        extra={"mean_period": float(mean_period),
               "concentration": float(concentration)},
    )


def matrix(
    ax: Any,
    values: Any,
    theme: Any,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    fmt: str = "{:.2f}",
    x_label: str = "",
    y_label: str = "",
    diagonal_mask: bool = False,
) -> PanelResult:
    """A categorical heatmap with optional editable annotations."""
    data = np.asarray(values, dtype=float).copy()
    if diagonal_mask:
        np.fill_diagonal(data, np.nan)
    handle = ax.imshow(
        data, cmap=theme["diverging_cmap"] if cmap is None else cmap,
        vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(list(column_labels), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=theme.colour("page"), linewidth=theme.stroke("hairline"))
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                if not np.isfinite(data[row, column]):
                    continue
                ax.text(
                    column, row, fmt.format(data[row, column]), ha="center", va="center",
                    fontsize=theme.size("annotation"), color=theme.colour("ink"),
                )
    rows, columns = np.indices(data.shape)
    return PanelResult(
        data=pd.DataFrame({
            "row": [list(row_labels)[index] for index in rows.ravel()],
            "column": [list(column_labels)[index] for index in columns.ravel()],
            "value": data.ravel(),
        }),
        axes=ax, extra={"handle": handle},
    )


def surface(
    ax: Any,
    values: Any,
    theme: Any,
    *,
    x: Sequence[float],
    y: Sequence[float],
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    contour_at: float | None = None,
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """A value over two continuous axes."""
    data = np.asarray(values, dtype=float)
    handle = ax.pcolormesh(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), data,
        shading="auto", cmap=theme["sequential_cmap"] if cmap is None else cmap,
        vmin=vmin, vmax=vmax,
    )
    if contour_at is not None and data.size:
        ax.contour(np.asarray(x, dtype=float), np.asarray(y, dtype=float), data,
                   levels=[float(contour_at)], colors=[theme.colour("ink")],
                   linewidths=theme.stroke("guide"))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    across, down = np.meshgrid(np.asarray(x, dtype=float),
                               np.asarray(y, dtype=float))
    return PanelResult(
        data=pd.DataFrame({"x": across.ravel(), "y": down.ravel(),
                           "value": data.ravel()}),
        axes=ax, extra={"handle": handle},
    )


def stacked_histogram(
    figure: Any,
    rect: tuple[float, float, float, float],
    groups: Sequence[Sequence[float]],
    theme: Any,
    *,
    labels: Sequence[str],
    bins: int = 30,
    look: Look | None = None,
    overlap: float = 0.55,
    labels_outside: bool = False,
    limits: tuple[float, float] | None = None,
    x_label: str = "",
) -> PanelResult:
    """Vertically stacked histograms for any collection of numeric groups.

    This is the generic panel used by Rhythm Strength. It knows nothing about
    rhythms: callers supply the groups, labels, bin count, colour, overlap,
    displayed limits and axis wording. The overlapping form is also commonly
    called a ridgeline plot.

    ``labels_outside`` moves each row's name into the left margin instead of
    sitting it on the row's own ridge. Inside is right for a handful of rows
    whose mass is to the right; outside is the only readable option for many
    rows, or for distributions that peak at their left edge, where an inside
    label lands squarely on the tallest part of the curve it is naming.

    ``limits`` fixes the range the bins cover instead of taking it from the
    data. Every row shares one axis, so one heavy tail sets the range for all
    of them and the other rows collapse into a spike at the middle - the shape
    the panel exists to show is then the one thing it cannot show. Values
    outside the range are left out of their histogram and counted in the
    returned table under ``outside``, so the caller can say how many.
    """
    groups = [np.asarray(group, dtype=float) for group in groups]
    if not groups:
        return PanelResult(
            data=pd.DataFrame(columns=["group", "bin_centre", "density"]), axes=[])
    finite = np.concatenate([group[np.isfinite(group)] for group in groups])
    if limits is not None:
        low, high = (float(value) for value in limits)
    elif finite.size:
        low, high = float(finite.min()), float(finite.max())
    else:
        low, high = 0.0, 1.0
    edges = np.linspace(low, high, int(bins) + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    left, bottom, width, height = rect
    each = height / (1 + (len(groups) - 1) * (1 - overlap))
    step = each * (1 - overlap)
    colour = (look or Look(colour=theme.colour("morphology"))).colour or theme.colour("morphology")
    axes = []
    drawn = []
    for index, (group, label) in enumerate(zip(groups, labels)):
        ax = figure.add_axes([left, bottom + (len(groups) - 1 - index) * step, width, each])
        inside = group[np.isfinite(group)]
        outside = int(((inside < low) | (inside > high)).sum())
        counts, _ = np.histogram(inside, bins=edges, density=True)
        drawn.append(pd.DataFrame({"group": str(label), "bin_centre": centres,
                                   "density": counts, "outside": outside}))
        ax.fill_between(centres, 0, counts, color=colour, alpha=0.65, linewidth=0)
        ax.plot(centres, counts, color=colour, linewidth=theme.stroke("line"))
        if labels_outside:
            ax.text(-0.012, 0.15, str(label), transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=theme.size("annotation"), fontweight="bold")
        else:
            ax.text(0.0, 0.1, str(label), transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=theme.size("annotation"), fontweight="bold")
        ax.set_yticks([])
        if index != len(groups) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(x_label)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        axes.append(ax)
    return PanelResult(data=pd.concat(drawn, ignore_index=True), axes=axes)


def ridgeline(
    figure: Any,
    rect: tuple[float, float, float, float],
    groups: Sequence[Sequence[float]],
    theme: Any,
    *,
    labels: Sequence[str],
    bins: int = 30,
    look: Look | None = None,
    overlap: float = 0.55,
    labels_outside: bool = False,
    limits: tuple[float, float] | None = None,
    x_label: str = "",
) -> PanelResult:
    """Compatibility name for :func:`stacked_histogram`."""
    return stacked_histogram(
        figure, rect, groups, theme, labels=labels, bins=bins, look=look,
        overlap=overlap, labels_outside=labels_outside, limits=limits,
        x_label=x_label,
    )


def stacked_area(
    ax: Any,
    x: Sequence[float],
    columns: Sequence[str],
    frame: Any,
    theme: Any,
    *,
    looks: dict[str, Look] | None = None,
    labels: Sequence[str] | None = None,
    normalise: bool = False,
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """Composition over time: the exact bands handed to stackplot."""
    x = np.asarray(x, dtype=float)
    names = list(columns)
    values = np.vstack([np.asarray(frame[name], dtype=float) for name in names]).T
    values = np.nan_to_num(values, nan=0.0)
    if normalise:
        totals = values.sum(axis=1)
        values = np.divide(values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] != 0)
    looks = looks or {}
    role_cycle = ("morphology", "reporter", "surveillance", "motility", "evidence")
    colours = [
        (looks.get(name) or Look(colour=theme.colour(role_cycle[index % len(role_cycle)]))).colour
        for index, name in enumerate(names)
    ]
    ax.stackplot(x, values.T, labels=list(labels or names), colors=colours, linewidth=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    result = pd.DataFrame(values, columns=names)
    result.insert(0, "x", x)
    result["total"] = values.sum(axis=1)
    return PanelResult(data=result, axes=ax)


def flow(
    figure: Any,
    rect: tuple[float, float, float, float],
    stages: Any,
    theme: Any,
    *,
    class_labels: Sequence[str],
    class_roles: Sequence[str] | None = None,
    stage_labels: Sequence[str] | None = None,
    min_width: float = 0.0,
) -> PanelResult:
    """An alluvial diagram from observation-by-stage class codes."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    data = np.asarray(stages)
    if data.ndim != 2:
        raise ValueError("flow stages must be an observation-by-stage matrix")
    if data.shape[0] < data.shape[1] and data.shape[0] == len(stage_labels or []):
        data = data.T
    n_observations, n_stages = data.shape
    labels = list(stage_labels or [f"Stage {index + 1}" for index in range(n_stages)])
    roles = list(class_roles or ["morphology"] * len(class_labels))
    ax = figure.add_axes(rect)
    ax.set_xlim(-0.1, n_stages - 0.9)
    ax.set_ylim(0, max(n_observations, 1))
    ax.set_axis_off()

    offsets: list[dict[int, float]] = []
    for stage in range(n_stages):
        counts = {klass: int(np.count_nonzero(data[:, stage] == klass)) for klass in range(len(class_labels))}
        cursor = 0.0
        stage_offsets = {}
        for klass in range(len(class_labels)):
            stage_offsets[klass] = cursor
            count = counts[klass]
            ax.add_patch(plt_rectangle(stage - 0.025, cursor, 0.05, count,
                                       theme.colour(roles[klass])))
            cursor += count
        offsets.append(stage_offsets)
        ax.text(stage, n_observations * 1.02, labels[stage], ha="center", va="bottom",
                fontsize=theme.size("annotation"), fontweight="bold")

    rows: list[dict] = []
    outgoing = [dict(offset) for offset in offsets]
    incoming = [dict(offset) for offset in offsets]
    for stage in range(n_stages - 1):
        for source in range(len(class_labels)):
            for target in range(len(class_labels)):
                count = int(np.count_nonzero((data[:, stage] == source) & (data[:, stage + 1] == target)))
                rows.append({
                    "from_stage": stage, "to_stage": stage + 1,
                    "from_class": source, "to_class": target, "count": count,
                })
                if count <= min_width:
                    continue
                y0a, y0b = outgoing[stage][source], outgoing[stage][source] + count
                y1a, y1b = incoming[stage + 1][target], incoming[stage + 1][target] + count
                outgoing[stage][source] = y0b
                incoming[stage + 1][target] = y1b
                x0, x1 = stage + 0.025, stage + 1 - 0.025
                c1, c2 = x0 + (x1 - x0) * 0.42, x1 - (x1 - x0) * 0.42
                path = Path(
                    [(x0, y0a), (c1, y0a), (c2, y1a), (x1, y1a),
                     (x1, y1b), (c2, y1b), (c1, y0b), (x0, y0b), (x0, y0a)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY],
                )
                ax.add_patch(PathPatch(path, facecolor=theme.colour(roles[source]),
                                       edgecolor="none", alpha=0.42))
    return PanelResult(data=pd.DataFrame(rows), axes=ax)


def plt_rectangle(x: float, y: float, width: float, height: float, colour: str) -> "Rectangle":
    """Local import keeps matplotlib out of module import time."""
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), width, height, facecolor=colour, edgecolor="none")


def chord(
    ax: Any,
    weights: Any,
    theme: Any,
    *,
    labels: Sequence[str],
    threshold: float = 0.0,
    role: str = "morphology",
    node_colours: Sequence[str] | None = None,
) -> PanelResult:
    """Nodes on a circle and weighted curved links between them."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    values = np.asarray(weights, dtype=float)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    colours = list(node_colours or [theme.colour(role)] * n)
    maximum = float(np.nanmax(values)) if values.size and np.isfinite(values).any() else 1.0
    for left in range(n):
        for right in range(left + 1, n):
            weight = float(values[left, right])
            if not np.isfinite(weight) or weight <= threshold:
                continue
            path = Path([points[left], (0.0, 0.0), points[right]],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colours[left],
                                   linewidth=theme.stroke("line") * max(weight / maximum, 0.15),
                                   alpha=0.55))
    ax.scatter(points[:, 0], points[:, 1], s=theme.point_area(1.8), c=colours, zorder=3)
    for point, label in zip(points, labels):
        ax.text(point[0] * 1.14, point[1] * 1.14, str(label), ha="center", va="center",
                fontsize=theme.size("annotation"))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    _bare_axes(ax)
    left_index, right_index = np.indices(values.shape)
    keep = left_index.ravel() < right_index.ravel()
    names = list(labels)
    return PanelResult(
        data=pd.DataFrame({
            "source": [names[index] for index in left_index.ravel()[keep]],
            "target": [names[index] for index in right_index.ravel()[keep]],
            "value": values.ravel()[keep],
        }),
        axes=ax,
    )


def paired_slopes(
    ax: Any,
    left: Sequence[float],
    right: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str] | None = None,
    left_label: str = "",
    right_label: str = "",
    role: str = "morphology",
    highlight: Sequence[bool] | None = None,
    y_label: str = "",
) -> PanelResult:
    """One line per subject joining its own two values."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    marked = np.zeros(len(left), dtype=bool) if highlight is None else np.asarray(highlight, dtype=bool)
    segments = np.stack([
        np.column_stack([np.zeros(len(left)), left]),
        np.column_stack([np.ones(len(right)), right]),
    ], axis=1)
    for index, segment in enumerate(segments):
        colour = theme.colour("highlight" if marked[index] else role)
        ax.plot(segment[:, 0], segment[:, 1], color=colour, alpha=0.55,
                linewidth=theme.stroke("line"), marker="o")
        if labels is not None:
            ax.text(1.03, right[index], str(labels[index]), va="center",
                    fontsize=theme.size("caption"), color=colour)
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylabel(y_label)
    drawn = pd.DataFrame({"left": left, "right": right, "highlight": marked})
    if labels is not None:
        drawn.insert(0, "label", list(labels))
    return PanelResult(data=drawn, axes=ax, extra={"segments": segments})


def forest(
    ax: Any,
    effects: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str],
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    significant: Sequence[bool] | None = None,
    refusals: Sequence[str] | None = None,
    no_effect: float = 0.0,
    refusal_width: int = 64,
    x_label: str = "",
) -> PanelResult:
    """One estimate per row, with its interval and a line at no effect.

    Three things this draws that a plain dotplot does not, each because leaving
    it out changes what the reader concludes:

    A row whose ``refusals`` entry is a non-empty sentence carries that sentence
    instead of a point. A refused comparison drawn as a blank row is
    indistinguishable from a comparison nobody asked for, and the reason a
    package writes a refusal row rather than skipping the contrast is that those
    two are different.

    A row with no interval - ``lo`` or ``hi`` missing, which is what a rank test
    without a bootstrap gives - draws its point with no whisker. Drawing a
    zero-width interval instead would claim the estimate was pinned exactly.

    Significance is taken, not derived. The caller passes what the statistics
    already decided, so the page cannot disagree with the file it came from.
    """
    effects = np.asarray(effects, dtype=float)
    labels = list(labels)
    rows = len(labels)
    lo = np.full(rows, np.nan) if lo is None else np.asarray(lo, dtype=float)
    hi = np.full(rows, np.nan) if hi is None else np.asarray(hi, dtype=float)
    flags = ([False] * rows if significant is None
             else [bool(value) for value in significant])
    notes = [""] * rows if refusals is None else ["" if value is None else str(value)
                                                 for value in refusals]

    ax.axvline(no_effect, color=theme.colour("reference"),
               linewidth=theme.stroke("guide"), zorder=1)
    drawn = []
    for row, label in enumerate(labels):
        y = rows - 1 - row
        if notes[row]:
            # Wrapped, because a refusal is a whole sentence and the saved page
            # is cropped tight: one unbroken line of it stretches the sheet to
            # its own width and squeezes every panel into a corner.
            ax.text(no_effect, y,
                    "  " + NEWLINE.join(textwrap.wrap(notes[row], refusal_width)),
                    va="center", ha="left", fontsize=theme.size("annotation"),
                    color=theme.colour("caption"), zorder=3)
            drawn.append({"label": label, "effect": np.nan, "lo": np.nan,
                          "hi": np.nan, "significant": False,
                          "refusal": notes[row], "drawn": False})
            continue
        colour = theme.colour("significant" if flags[row] else "not_significant")
        if np.isfinite(lo[row]) and np.isfinite(hi[row]):
            ax.plot([lo[row], hi[row]], [y, y], color=colour,
                    linewidth=theme.stroke("emphasis"), solid_capstyle="butt",
                    zorder=2)
        ax.scatter(effects[row], y, color=colour, s=theme.point_area(1.4), zorder=3)
        drawn.append({"label": label, "effect": float(effects[row]),
                      "lo": float(lo[row]), "hi": float(hi[row]),
                      "significant": flags[row], "refusal": "", "drawn": True})
    ax.set_yticks(list(range(rows)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_xlabel(x_label)
    return PanelResult(
        data=pd.DataFrame(drawn, columns=["label", "effect", "lo", "hi",
                                          "significant", "refusal", "drawn"]),
        axes=ax)


def forest_blocks(
    figure: Any,
    rect: tuple[float, float, float, float],
    effects: Sequence[float],
    theme: Any,
    *,
    labels: Sequence[str],
    scales: Sequence[str],
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    significant: Sequence[bool] | None = None,
    refusals: Sequence[str] | None = None,
    no_effect: float = 0.0,
    x_labels: dict | None = None,
) -> PanelResult:
    """A forest per scale, stacked, each with its own horizontal axis.

    The reason this exists rather than one axes: a forest is read by comparing
    distances to the no-effect line, and that only means something when every
    row is in the same units. A difference of 6.75 pixels and a difference of
    1300 camera units on one axis is not a comparison, it is two unrelated
    numbers sharing a ruler - and the larger one flattens the smaller to a dot
    on the line, which reads as "no effect" when it is nothing of the kind.

    ``scales`` labels each row with the scale it belongs to. Blocks appear in
    first-seen order and are given height in proportion to how many rows they
    hold, so no block is squeezed for being small.

    Rows are returned in one table with the scale on each, so the figure data
    still says what the page showed and in what units.
    """
    scales = [str(value) for value in scales]
    order = list(dict.fromkeys(scales))
    counts = [sum(1 for value in scales if value == name) for name in order]
    left, bottom, width, height = rect

    # Each block takes its own rows plus a fixed allowance of paper underneath
    # for its tick labels and axis title. Without the allowance a one-row block
    # is a sliver whose axis label lands on the block below it, which is how a
    # page ends up saying that two numbers share a scale when the whole point
    # of splitting them was that they do not.
    label_rows = 1.6
    weights = [count + label_rows for count in counts]
    total = sum(weights)

    made, drawn = [], []
    top = bottom + height
    for name, count, weight in zip(order, counts, weights):
        share = (weight / total) * height
        drawing = share * count / weight
        top -= share
        ax = figure.add_axes([left, top + (share - drawing), width, drawing])
        picked = [index for index, value in enumerate(scales) if value == name]
        result = forest(
            ax, [effects[i] for i in picked], theme,
            labels=[labels[i] for i in picked],
            lo=None if lo is None else [lo[i] for i in picked],
            hi=None if hi is None else [hi[i] for i in picked],
            significant=None if significant is None else [significant[i] for i in picked],
            refusals=None if refusals is None else [refusals[i] for i in picked],
            no_effect=no_effect,
            x_label=(x_labels or {}).get(name, name))
        table = result.data.copy()
        table.insert(0, "scale", name)
        drawn.append(table)
        made.append(ax)
    return PanelResult(data=pd.concat(drawn, ignore_index=True) if drawn
                       else pd.DataFrame(columns=["scale", "label", "effect", "lo",
                                                  "hi", "significant", "refusal",
                                                  "drawn"]),
                       axes=made)


def event_average(
    ax: Any,
    offsets: Sequence[float],
    curves: Any,
    theme: Any,
    *,
    role: str = "morphology",
    bootstrap: int = 1000,
    null_curves: Any = None,
    null_label: str = "Comparison interval",
    look: Look | None = None,
    x_label: str = "Frames from event",
    y_label: str = "",
    label: str = "",
) -> PanelResult:
    """Mean aligned trace with a deterministic bootstrap interval."""
    offsets = np.asarray(offsets, dtype=float)
    values = np.asarray(curves, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(offsets):
        raise ValueError("event_average curves must have one column per offset")
    mean = np.nanmean(values, axis=0)
    if len(values) and bootstrap > 0:
        generator = np.random.default_rng(0)
        sampled = np.empty((int(bootstrap), len(offsets)), dtype=float)
        for replicate in range(int(bootstrap)):
            picked = generator.integers(0, len(values), len(values))
            sampled[replicate] = np.nanmean(values[picked], axis=0)
        lo, hi = np.nanpercentile(sampled, [2.5, 97.5], axis=0)
    else:
        lo = hi = mean.copy()
    colour = (look or Look(colour=theme.colour(role))).colour or theme.colour(role)
    result = pd.DataFrame({
        "offset": offsets,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "n_events": np.sum(np.isfinite(values), axis=0),
    })
    if null_curves is not None:
        null = np.asarray(null_curves, dtype=float)
        if null.ndim == 1:
            null = null[None, :]
        null_mean = np.nanmean(null, axis=0)
        null_lo, null_hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
        ax.fill_between(offsets, null_lo, null_hi, color=theme.colour("reference"),
                        alpha=0.28, linewidth=0, label=null_label)
        result["null_mean"] = null_mean
        result["null_lo"] = null_lo
        result["null_hi"] = null_hi
        result["null_n_events"] = np.sum(np.isfinite(null), axis=0)
    ax.fill_between(offsets, lo, hi, color=colour, alpha=0.25, linewidth=0)
    ax.plot(offsets, mean, color=colour, linewidth=theme.stroke("emphasis"),
            label=label or None)
    ax.axvline(0, color=theme.colour("reference"), linewidth=theme.stroke("guide"))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return PanelResult(data=result, axes=ax)


def hexbin(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    theme: Any,
    *,
    gridsize: int = 24,
    cmap: Any = None,
    binned_mean: bool = True,
    mean_label: str = "Mean within x bin",
    reference_band: Any = None,
    xscale: str = "linear",
    yscale: str = "linear",
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """Observation density for any paired measurements, with an optional mean.

    ``xscale`` and ``yscale`` are passed to Matplotlib's hexagonal binning, so
    logarithmic axes are binned in logarithmic space instead of drawing linear
    bins and distorting them afterwards. The returned table is the optional
    mean of y in each x bin; the caller should retain the paired observations as
    its figure-data table because those are the inputs to the density counts.
    """
    if xscale not in ("linear", "log") or yscale not in ("linear", "log"):
        raise ValueError("hexbin scales must be 'linear' or 'log'")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if xscale == "log":
        usable &= x > 0
    if yscale == "log":
        usable &= y > 0
    x, y = x[usable], y[usable]
    handle = ax.hexbin(x, y, gridsize=int(gridsize), mincnt=1,
                       cmap=theme["sequential_cmap"] if cmap is None else cmap,
                       xscale=xscale, yscale=yscale)
    table = pd.DataFrame(
        columns=["bin_left", "bin_right", "x", "mean", "count"])
    if binned_mean and x.size:
        low, high = float(x.min()), float(x.max())
        if low == high:
            edges = np.asarray([low * 0.95, high * 1.05]) if low else np.asarray([-0.5, 0.5])
        elif xscale == "log":
            edges = np.geomspace(low, high, int(gridsize) + 1)
        else:
            edges = np.linspace(low, high, int(gridsize) + 1)
        which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        rows = []
        for index in range(len(edges) - 1):
            selected = y[which == index]
            if selected.size:
                centre = (np.sqrt(edges[index] * edges[index + 1])
                          if xscale == "log"
                          else (edges[index] + edges[index + 1]) / 2)
                rows.append({
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "x": float(centre),
                    "mean": float(np.mean(selected)),
                    "count": int(len(selected)),
                })
        table = pd.DataFrame(rows)
        if not table.empty:
            ax.plot(table["x"], table["mean"], color=theme.colour("ink"),
                    linewidth=theme.stroke("emphasis"),
                    label=mean_label or None)
    if reference_band is not None:
        band = np.asarray(reference_band, dtype=float)
        if band.ndim == 1 and len(band) == 2:
            ax.axhspan(band[0], band[1], color=theme.colour("reference"), alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    handle._binned_mean = table
    return PanelResult(data=table, axes=ax, extra={"handle": handle})


def dendrogram(
    ax: Any,
    linkage: Any,
    theme: Any,
    *,
    labels: Sequence[str] | None = None,
    orientation: str = "left",
    colour_threshold: float | None = None,
) -> PanelResult:
    """Draw the hierarchy and give back its leaf order."""
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    result = scipy_dendrogram(
        np.asarray(linkage, dtype=float), ax=ax, labels=None if labels is None else list(labels),
        orientation=orientation, color_threshold=colour_threshold,
        no_labels=labels is None,
        above_threshold_color=theme.colour("reference"),
        link_color_func=lambda _index: theme.colour("morphology"),
    )
    leaves = [int(index) for index in result["leaves"]]
    return PanelResult(
        data=pd.DataFrame({"position": range(len(leaves)), "leaf": leaves}),
        axes=ax, extra={"leaf_order": leaves},
    )
