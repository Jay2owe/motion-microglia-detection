"""One function per panel, so a page is a composition rather than a script.

A report card is not one figure. It is a strip of image tiles, a trace, a model
curve over a trace - four kinds of panel arranged on a page. Each is written
once here and takes an axes to draw on, so the same panel can appear on a report
card, on a page of its own, or in some arrangement nobody has thought of yet.

**The contract.** A panel takes an axes - or a figure and a rectangle, when it
has to make several axes of its own - then its numbers and the theme, and it
gives back a :class:`PanelResult`: the table it drew, the axes it made, and
anything the caller cannot recompute.

    drawn = common.histogram(ax, values, theme, bins=40)
    drawn.data          # bin_left, bin_right, count - exactly what is on screen
    drawn.axes          # the axes it drew on
    drawn.extra         # the counts and edges, for a caller that wants arrays

One return shape, because the caller's first question is always the same - what
exactly went on the page - and a builder that answers it by rebuilding the table
from the inputs it just passed in has written the same numbers twice. Hand the
result to ``ctx.drew("<panel key>", ...)`` and the figure's audit table comes
free.

Nothing in here reads a file, chooses a cell or decides a layout. A panel is
given the numbers and the rectangle, and draws. That is what makes it reusable:
the choices stay with the caller, which is where the user's options arrive.

Two kinds of function in these files are *not* panels and say so by their return
type. A **key** - ``colour_bar``, ``inset_colour_bar``, ``semantic_legend`` -
explains a mapping something else already recorded, and gives back the
matplotlib object. A **derivation** - ``path_segments``, ``detrended_z``,
``mechanism_colours``, ``breakout_events`` - takes no axes and draws nothing; it
is here because it belongs beside the panel it feeds.

**Where a panel lives.** ``common`` holds the chart grammar - a scatter with its
margins, a histogram, a raster, a dumbbell - anything that would draw the same
picture for any numbers. The other files are named after the measurement module
whose readouts they draw, and a panel belongs in one of them the moment it has
to *know* something about that measurement: which colours its states carry, what
has to be subtracted before it is honest, which points are allowed to be joined
by a line.

    from panels import common, motility, presence, rhythms, surveillance

    common.trace(ax, hours, values, theme)              # any measurement
    motility.trajectory_map(ax, groups, theme, ...)     # cuts paths at gaps
    presence.persistence_raster(ax, matrix, hours, ...) # three fixed states

The measurement modules themselves are in ``analysis/modules``; the names match
on purpose, so "which module wrote this number" and "which file draws it" have
one answer.
"""

from __future__ import annotations

from ._contract import PanelResult
from . import common, coupling, intensity, morphology, motility, presence, rhythms, surveillance, territory
from .common import (Look, Mark, category_strip, chord, colour_bar, cosinor_curve,
                     dendrogram, dumbbell, event_average, flow, hexbin, histogram,
                     image_strip, image_tile, lollipop, matrix, paired_slopes,
                     raster, reference_lines, resolve_look, ridgeline, rose,
                     scatter, scatter_with_margins, stacked_area, surface, trace,
                     trace_stack, vectors)

__all__ = [
    "common", "motility", "presence", "rhythms", "surveillance",
    "morphology", "intensity", "territory", "coupling",
    "Look", "Mark", "PanelResult", "resolve_look",
    "image_tile", "image_strip",
    "trace", "cosinor_curve", "trace_stack",
    "raster", "category_strip", "colour_bar",
    "histogram", "reference_lines",
    "scatter", "scatter_with_margins",
    "dumbbell", "lollipop",
    "rose", "vectors", "matrix", "surface", "ridgeline", "stacked_area",
    "flow", "chord", "paired_slopes", "event_average", "hexbin", "dendrogram",
]
