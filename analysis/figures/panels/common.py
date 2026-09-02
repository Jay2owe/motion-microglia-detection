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
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ._contract import PanelResult

if TYPE_CHECKING:                       # names used only in annotations
    from matplotlib.colorbar import Colorbar
    from matplotlib.legend import Legend
    from matplotlib.patches import Rectangle

__all__ = [
    "Look", "Mark", "PanelResult", "resolve_look",
    "image_tile", "image_strip",
    "trace", "cosinor_curve", "trace_stack",
    "raster", "category_strip", "colour_bar", "inset_colour_bar", "semantic_legend",
    "histogram", "reference_lines",
    "scatter", "scatter_with_margins",
    "dumbbell", "lollipop",
    "rose", "vectors", "matrix", "surface", "ridgeline", "stacked_area",
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


def cosinor_curve(
    ax: Any,
    hours: Sequence[float],
    values: Sequence[float],
    theme: Any,
    *,
    period_hours: float = 24.0,
    role: str = "fit",
    linestyle: tuple = (0, (6, 3)),
) -> PanelResult:
    """A fixed-period cosine plus a straight line, drawn over a trace.

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

    slope, intercept = np.polyfit(hours[usable], values[usable], 1)
    trend = slope * hours + intercept
    omega = 2 * np.pi / float(period_hours)
    design = np.column_stack([np.ones(hours.size), np.cos(omega * hours), np.sin(omega * hours)])
    coefficients, *_ = np.linalg.lstsq(
        design[usable], (values - trend)[usable], rcond=None
    )
    fitted = design @ coefficients + trend
    ax.plot(
        hours, fitted, color=theme.colour(role),
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
) -> "Colorbar":
    """A compact scale key attached to the right of one axes."""
    colour_ax = ax.inset_axes([1.018, 0.12, 0.022, 0.76])
    bar = ax.figure.colorbar(mappable, cax=colour_ax, ticks=ticks)
    bar.set_label(label, fontsize=theme.size("caption"))
    bar.ax.tick_params(labelsize=theme.size("caption"), width=theme.stroke("hairline"))
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
        zorder=3, label=left_label or None,
    )
    ax.scatter(
        right, positions, s=theme.point_area(1.5),
        color=list(right_colours) if right_colours is not None else theme.colour("significant"),
        edgecolor="none", zorder=4, label=right_label or None,
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
        for y, (value, text, colour) in enumerate(zip(values, annotations, palette)):
            ax.text(
                value + 0.045, y, text, va="center", fontsize=theme.size("emphasis"),
                color=colour, fontweight="bold",
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=theme.size("panel"))
    ax.set_ylim(-0.65, len(labels) - 0.35)
    return PanelResult(
        data=pd.DataFrame({"row": positions, "label": list(labels),
                           "value": values, "baseline": float(baseline)}),
        axes=ax,
    )


# ---------------------------------------------------------- extended grammar

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
) -> PanelResult:
    """A circular histogram on one wrapped period: one row per wedge."""
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
    if resultant and len(theta):
        ceiling = float(np.nanmax(np.abs(radii))) if len(radii) else 1.0
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
            -0.42, 0.5, radius_label, transform=ax.transAxes, rotation=90,
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
    x_label: str = "",
) -> PanelResult:
    """Overlapping density-like histograms, one axes per group."""
    groups = [np.asarray(group, dtype=float) for group in groups]
    if not groups:
        return PanelResult(
            data=pd.DataFrame(columns=["group", "bin_centre", "density"]), axes=[])
    finite = np.concatenate([group[np.isfinite(group)] for group in groups])
    edges = np.linspace(float(finite.min()), float(finite.max()), int(bins) + 1) \
        if finite.size else np.linspace(0, 1, int(bins) + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    left, bottom, width, height = rect
    each = height / (1 + (len(groups) - 1) * (1 - overlap))
    step = each * (1 - overlap)
    colour = (look or Look(colour=theme.colour("morphology"))).colour or theme.colour("morphology")
    axes = []
    drawn = []
    for index, (group, label) in enumerate(zip(groups, labels)):
        ax = figure.add_axes([left, bottom + (len(groups) - 1 - index) * step, width, each])
        counts, _ = np.histogram(group[np.isfinite(group)], bins=edges, density=True)
        drawn.append(pd.DataFrame({"group": str(label), "bin_centre": centres,
                                   "density": counts}))
        ax.fill_between(centres, 0, counts, color=colour, alpha=0.65, linewidth=0)
        ax.plot(centres, counts, color=colour, linewidth=theme.stroke("line"))
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


def event_average(
    ax: Any,
    offsets: Sequence[float],
    curves: Any,
    theme: Any,
    *,
    role: str = "morphology",
    bootstrap: int = 1000,
    null_curves: Any = None,
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
    result = pd.DataFrame({"offset": offsets, "mean": mean, "lo": lo, "hi": hi})
    if null_curves is not None:
        null = np.asarray(null_curves, dtype=float)
        if null.ndim == 1:
            null = null[None, :]
        null_mean = np.nanmean(null, axis=0)
        null_lo, null_hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
        ax.fill_between(offsets, null_lo, null_hi, color=theme.colour("reference"),
                        alpha=0.28, linewidth=0, label="Matched-noise interval")
        result["null_mean"] = null_mean
        result["null_lo"] = null_lo
        result["null_hi"] = null_hi
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
    reference_band: Any = None,
    x_label: str = "",
    y_label: str = "",
) -> PanelResult:
    """Point density for a crowded scatter, optionally with a binned mean."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    handle = ax.hexbin(x, y, gridsize=int(gridsize), mincnt=1,
                       cmap=theme["sequential_cmap"] if cmap is None else cmap)
    table = pd.DataFrame(columns=["x", "mean", "count"])
    if binned_mean and x.size:
        edges = np.linspace(float(x.min()), float(x.max()), int(gridsize) + 1)
        which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        rows = []
        for index in range(len(edges) - 1):
            selected = y[which == index]
            if selected.size:
                rows.append({"x": (edges[index] + edges[index + 1]) / 2,
                             "mean": float(np.mean(selected)), "count": int(len(selected))})
        table = pd.DataFrame(rows)
        if not table.empty:
            ax.plot(table["x"], table["mean"], color=theme.colour("ink"),
                    linewidth=theme.stroke("emphasis"))
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
