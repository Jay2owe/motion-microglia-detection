"""Command-line options a builder takes that are not figure text.

Every builder's one positional argument is the analysis run folder. Beyond that
it takes wording (see ``_text``) and a short list of options that change what is
*drawn* rather than what is written beside it::

    python analysis/figures/05_cell_report_card.py <run> --identity 7 --hour-ticks 12

They are declared here, in one place, for two reasons. ``_text`` can then skip
an option it does not own without losing its ability to refuse a typo - a
builder that quietly ignored ``--identiy`` would draw the wrong cell and say
nothing. And ``python <builder> --help`` has one list to print.

This module owns the **vocabulary**: what a flag is called, what it means in one
line, and how its text becomes a value. It does not own which figure accepts
which flag - that is declared on the figure itself, in ``_schema``. Splitting
the two is what lets one page refuse ``--bins`` while another accepts it,
without either page restating what ``--bins`` means.

An option is *not* a way to change a measurement. Nothing here reaches the
tables; a builder reads what ``python -m analysis run`` wrote and renders it.
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from analysis.theme import parse_hours_per_tick  # noqa: E402
from analysis.circadian import scientific_options  # noqa: E402

__all__ = ["OPTIONS", "PLACEMENTS", "SELECTION", "SWITCHES", "Length",
           "Vocabulary", "as_length", "commas", "exactly", "length", "numbers",
           "option", "raw_flag", "skip_tokens", "switch"]


@dataclass(frozen=True)
class Length:
    """A distance on the page, in inches or as a share of the page.

    Both units are needed and neither is the right one for everything, which is
    why this exists rather than a bare number. A title sits a fixed number of
    *inches* below the top edge, so that a stack of printed pages of different
    heights carries its titles at one distance from the paper's edge. A note
    beside a plot sits at a *share* of the page, so that it stays beside the
    plot when the sheet is made taller. Write either one in the other's unit and
    the placement drifts the moment the page changes size - which is exactly
    what happens when a figure grows to clear a long footnote.

    The unit travels with the number for the same reason a measurement carries
    its unit anywhere else: 0.2 of a page and 0.2 inches are both plausible
    placements and nothing downstream could tell them apart.
    """

    value: float
    #: ``"page"`` for a share of the page, ``"in"`` for inches.
    unit: str = "page"

    def __post_init__(self) -> None:
        if self.unit not in ("page", "in"):
            raise ValueError(f"unit must be 'page' or 'in', not {self.unit!r}")

    def fraction(self, span_inches: float) -> float:
        """This distance as a share of a page ``span_inches`` long.

        The span is the dimension the distance runs along - the width for
        anything measured from a side, the height for anything measured from
        the top or the bottom.
        """
        if self.unit == "page":
            return float(self.value)
        if span_inches <= 0:
            raise ValueError("a page with no extent has no fractions in it")
        return float(self.value) / float(span_inches)

    @property
    def in_inches(self) -> bool:
        return self.unit == "in"

    def __str__(self) -> str:
        return f"{self.value:g}in" if self.unit == "in" else f"{self.value * 100:g}%"


def as_length(value: Any) -> Length:
    """A placement as a ``Length``, letting a bare number mean a share of the page.

    Builders wrote ``header_x=0.085`` before there was a second unit, and that
    number is a share of the page. Reading a bare float that way keeps those
    declarations meaning what they have always meant, so adding the unit did not
    quietly move anything.
    """
    if isinstance(value, Length):
        return value
    return Length(float(value), "page")


def length(text: str) -> Length:
    """A typed placement: ``2%`` of the page, or ``0.25in``.

    The unit is required, with no default, because the two are not
    interchangeable and the defaults are not all in the same one - the title is
    placed in inches and the footnote as a share of the page. A bare ``0.2``
    would have to guess, and a guess here is a figure whose words move by an
    inch without anybody having asked for it.
    """
    cleaned = str(text).strip().lower().replace(" ", "")
    for suffix, unit, scale in (("%", "page", 0.01), ("in", "in", 1.0),
                                ('"', "in", 1.0)):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            body = cleaned[: -len(suffix)]
            try:
                return Length(float(body) * scale, unit)
            except ValueError:
                raise ValueError(f"{body!r} is not a number") from None
    raise ValueError(
        f"{text!r} needs a unit: '2%' for a fiftieth of the page, '0.25in' for "
        "a quarter of an inch. The two are not the same distance and the "
        "package will not guess which you meant"
    )


def commas(text: str) -> list[str]:
    """A comma-separated option as a list, with the blanks dropped.

    Its own function because every list-valued option has to treat a trailing
    comma and a stray space the same way, and because an empty string has to
    come back as an empty list rather than as one empty entry - that is how a
    user turns a whole block off.
    """
    return [piece.strip() for piece in str(text).split(",") if piece.strip()]


def numbers(text: str) -> list[float]:
    """A comma-separated option as floats, refusing a typo rather than dropping it."""
    values = []
    for piece in commas(text):
        try:
            values.append(float(piece))
        except ValueError:
            raise ValueError(f"{piece!r} is not a number") from None
    return values


def exactly(count: int, cast: Callable[[str], Any] = commas) -> Callable[[str], Any]:
    """A list-valued cast that refuses a list of the wrong length.

    ``--metrics`` on a scatter is the x column and then the y column, so a list
    of one is a typo rather than a choice, and a figure that reads it as a
    choice draws a page with an empty axis. Wrapping the cast puts the refusal
    where the value is made, which is early enough to name the flag.
    """

    def caster(text: str) -> Any:
        values = cast(text)
        if len(values) != count:
            raise ValueError(
                f"needs exactly {count} values, comma separated; got {len(values)}")
        return values

    caster.__name__ = f"exactly_{count}"
    return caster


@dataclass(frozen=True)
class Vocabulary:
    """One option name, documented once for every figure that accepts it.

    ``cast`` turns the typed text into the value a builder works with, and is
    the *usual* one rather than the only one: a figure that draws a single
    metric narrows ``metrics`` from a list to a string on its own spec. The help
    line is shared for the opposite reason - a flag that means two things on two
    pages is two flags with one name, and that is the confusion this file exists
    to prevent.
    """

    name: str
    help: str
    cast: Callable[[str], Any] = str
    metavar: str = "VALUE"

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")


#: name -> what it does. The name is spelled with underscores; on the command
#: line either ``--hour-ticks`` or ``--hour_ticks`` is accepted, because a user
#: who has just typed ``--footnote`` will type the hyphen.
OPTIONS: dict[str, Vocabulary] = {entry.name: entry for entry in (
    Vocabulary("spatial_metric", "numeric metric mapped through space; radial_occupancy is a derived shortcut"),
    Vocabulary("reference_metric", "numeric reference metric for within-cell timing comparisons"),
    Vocabulary("snapshot_hours", "recording hours to draw; nearest actual frames, no interpolation", cast=numbers),
    Vocabulary("snapshot_count", "number of snapshots: 1 shows the final frame (default); explicit snapshot_hours take precedence", cast=int),
    Vocabulary("show_history", "show the complete time trace instead of only selected snapshots", cast=json.loads),
    Vocabulary("spatial_axis", "physical coordinate used to order cells: x or y"),
    Vocabulary("spatial_centre", "position and optional displacement use soma or centroid"),
    Vocabulary("spatial_panel_inches", "minimum width of each spatial panel in inches", cast=float),
    Vocabulary("spatial_point_size", "area of a cell-position marker in points squared", cast=float),
    Vocabulary("spatial_annotate", "label cell-position markers with their identities", cast=json.loads),
    Vocabulary("spatial_arrows", "overlay displacement since the preceding consecutive frame", cast=json.loads),
    Vocabulary("spatial_tracks", "overlay recorded cell paths on the detected-period map", cast=json.loads),
    Vocabulary("spatial_track_cells", "which tracks to show: rhythmic (default) or all"),
    Vocabulary("spatial_track_width", "track line width in points", cast=float),
    Vocabulary("spatial_cmap", "colour map for the selected spatial values"),
    Vocabulary("phase_group_hours", "centres of period groups for separate peak-time maps; empty shows peak percentage for every significant cell with an available estimate", cast=numbers),
    Vocabulary("period_tolerance", "largest relative period difference for a timing comparison", cast=float),
    Vocabulary("timing_reference_hour", "recording hour at which relative phase is compared", cast=float),
    Vocabulary("spatial_neighbours", "nearest neighbours per cell; symmetric union defines the spatial graph", cast=int),
    Vocabulary("spatial_permutations", "whole-trace spatial permutations for the field-level comparison", cast=int),
    Vocabulary("spatial_seed", "reproducible seed for spatial permutation tests", cast=int),
    Vocabulary("max_inferred_fraction", "exclude cell-frame metrics above this tracker-reconstructed pixel fraction", cast=float),
    Vocabulary("stem",
               "which movie in the run to draw, when the run holds more than one",
               metavar="NAME"),
    Vocabulary("item",
               "which item of the run's plot plan to draw; "
               "see `python -m analysis plots <run> --dry-run` for the names",
               metavar="NAME"),
    Vocabulary("identity", "which cell to draw", cast=int, metavar="N"),
    Vocabulary("hour_ticks",
               "hours between ticks on a time axis; a factor or multiple of 24",
               cast=parse_hours_per_tick, metavar="HOURS"),
    Vocabulary("metrics",
               "which measured columns the figure draws, comma separated; see its docstring",
               cast=commas, metavar="COL,COL"),
    Vocabulary("display",
               "which representation of the selected measurement is drawn",
               metavar="NAME"),
    Vocabulary("hues",
               "colours for named groups, comma separated in the figure's displayed order",
               cast=commas, metavar="COLOUR,COLOUR"),
    Vocabulary("event_metric",
               "which measured column ranks the frames used as alignment events",
               metavar="COL"),
    Vocabulary("event_direction",
               "how event frames are ranked: high values, low values, or within-cell deviation",
               metavar="high|low|deviation"),
    Vocabulary("top_events", "number of top-ranked frames retained per cell",
               cast=int, metavar="N"),
    Vocabulary("events", "names for user-selected events, comma separated",
               cast=commas, metavar="NAME,NAME"),
    Vocabulary("event_times",
               "start time of each user-selected event window, in hours",
               cast=numbers, metavar="H,H"),
    Vocabulary("images", "how many image tiles across the top; 0 for none",
               cast=int, metavar="N"),
    Vocabulary("image_hours",
               "explicit hours for the image tiles, comma separated; beats --images",
               cast=numbers, metavar="H,H"),
    Vocabulary("cell_lut", "colour map for cell images, e.g. dluc_purple, magma, gray",
               metavar="NAME"),
    Vocabulary("image_filter",
               "display-only image denoising: auto-organotypic or none",
               metavar="NAME"),
    Vocabulary("display_black_percentile",
               "percentile of the mean image mapped to the display floor",
               cast=float, metavar="0-100"),
    Vocabulary("display_white_percentile",
               "percentile of the image stack mapped to the display ceiling",
               cast=float, metavar="0-100"),
    Vocabulary("display_range_scope",
               "frames used to choose the shared image display range",
               metavar="stack|displayed"),
    Vocabulary("display_gamma",
               "display tone-curve exponent; below 1 lifts dim detail",
               cast=float, metavar="N"),
    Vocabulary("display_gain",
               "display-only multiplier for frame departures from their mean",
               cast=float, metavar="N"),
    Vocabulary("display_spatial_sigma",
               "display-only Gaussian smoothing width in pixels",
               cast=float, metavar="PX"),
    Vocabulary("display_pool_px",
               "spatial pooling width for adaptive display denoising, in pixels",
               cast=float, metavar="PX"),
    Vocabulary("display_sharpness",
               "sharpness of the adaptive display denoising transition",
               cast=float, metavar="N"),
    Vocabulary("display_noise_multiple",
               "noise-floor multiplier used by adaptive display denoising",
               cast=float, metavar="N"),
    Vocabulary("display_pad_frames",
               "mirrored frames at each end of adaptive display denoising",
               cast=int, metavar="N"),
    Vocabulary("trace_luts",
               "colour or colour map per trace, comma separated, matched to --metrics",
               cast=commas, metavar="COLOUR,COLOUR"),
    Vocabulary("map_luts",
               "colour map per canonical spatial panel, comma separated in panel order",
               cast=commas, metavar="CMAP,CMAP"),
    Vocabulary("matrix_lut", "colour map used for numeric values in a matrix",
               metavar="CMAP"),
    Vocabulary("column_label_rotation",
               "rotation of matrix column labels in degrees",
               cast=float, metavar="DEGREES"),
    Vocabulary("column_label_wrap",
               "maximum characters per matrix column-label line; 0 disables wrapping",
               cast=int, metavar="N"),
    Vocabulary("map_summary",
               "how a per-frame metric becomes one value per cell: median, mean, minimum, maximum, first, or last",
               metavar="METHOD"),
    Vocabulary("map_assignment",
               "overlapping cell values: occupancy_weighted_mean (default), equal_mean, first, last, or most_frequent",
               metavar="METHOD"),
    Vocabulary("phase_coherence",
               "minimum circular agreement for a phase-map mixture or tissue reference, 0 to 1",
               cast=float, metavar="FRACTION"),
    Vocabulary("map_range",
               "colour limits for each spatial metric: robust percentiles or the full observed range",
               metavar="robust|full"),
    Vocabulary("density_lut",
               "colour map used for observation density",
               metavar="CMAP"),
    Vocabulary("density_summary",
               "summary over a density map: mean or none",
               metavar="mean|none"),
    Vocabulary("trace_layout", "trace arrangement: stack or overlay",
               metavar="stack|overlay"),
    Vocabulary("trace_view", "trace values shown: raw or detrended",
               metavar="raw|detrended"),
    Vocabulary("outline", "colour of the outline drawn over each image tile",
               metavar="COLOUR"),
    Vocabulary("fit",
               "which traces carry the fitted curve: a metric list, all, or empty for none",
               cast=commas, metavar="COL,COL"),
    *(Vocabulary(name, entry["description"] + " Units: " + entry["units"]
                 + (" " + entry["role"] if entry.get("role") else ""),
                 cast={"int": int, "float": float, "list": commas,
                       "object": json.loads}.get(entry["type"], str))
      for name, entry in scientific_options().items()),
    Vocabulary("multiple_testing",
               "correction across a family of rhythm tests: bh, bonferroni, sidak or none",
               metavar="METHOD"),
    Vocabulary("correction_scope",
               "multiple-testing family: every matrix cell together or each metric column separately",
               metavar="matrix|metric"),
    Vocabulary("min_observations",
               "fewest measured timepoints required before a trace is tested",
               cast=int, metavar="N"),
    Vocabulary("min_cycles",
               "fewest observed cycles required to treat a detected period as well constrained",
               cast=float, metavar="N"),
    Vocabulary("median_window_points",
               "odd number of adjacent observations in the median-filter sensitivity trace",
               cast=int, metavar="N"),
    Vocabulary("panels", "which panels of a multi-panel figure to draw, comma separated",
               cast=commas, metavar="NAME,NAME"),
    Vocabulary("bins", "how many bands a histogram panel divides its range into",
               cast=int, metavar="N"),
    Vocabulary("profile_bins",
               "how many equal positions a detected cycle is divided into for comparison",
               cast=int, metavar="N"),
    Vocabulary("size", "which column sets point size on a scatter; empty for one size",
               metavar="COL"),
    Vocabulary("line_width_metric",
               "which per-cell numeric column sets trajectory line width; empty for one width",
               metavar="COL"),
    Vocabulary("line_width_range",
               "smallest and largest widths as multiples of the default trajectory width",
               cast=numbers, metavar="LOW,HIGH"),
    Vocabulary("line_dash_metric",
               "which per-cell column sets trajectory dashes; rhythm_period_band uses rhythm results",
               metavar="COL|rhythm_period_band"),
    Vocabulary("cells",
               "which identities appear, comma separated, or how many longest-lived cells",
               metavar="N|IDS"),
    Vocabulary("crop_padding",
               "empty border around an image crop as a fraction of the measured object span",
               cast=float, metavar="0-1"),
    Vocabulary("quantile", "data quantile used to define an event threshold",
               cast=float, metavar="0-1"),
    Vocabulary("window", "frames either side of an aligned event",
               cast=int, metavar="N"),
    Vocabulary("max_lag", "largest lag drawn, in frames unless the figure says hours",
               cast=float, metavar="N"),
    Vocabulary("order", "row ordering used by a raster or ribbon", metavar="NAME"),
    Vocabulary("timing",
               "which precomputed timing column or figure alias is distributed",
               metavar="COL|NAME"),
    Vocabulary("period_hours",
               "cycle length used to wrap phase displays, in hours",
               cast=float, metavar="HOURS"),
    Vocabulary("period_bins",
               "period-band boundaries in hours, comma separated",
               cast=numbers, metavar="H,H,H"),
    Vocabulary("normalise", "matrix normalisation: row, column or none",
               metavar="row|column|none"),
    Vocabulary("shuffles", "number of shuffled null replicates",
               cast=int, metavar="N"),
    Vocabulary("regime", "regime number used for a within-cell contrast",
               cast=int, metavar="N"),
    Vocabulary("contour", "occupancy-frequency contour as a share of observed frames",
               cast=float, metavar="0-1"),
    Vocabulary("coverage_view", "coverage map shown: final or stages",
               metavar="final|stages"),
    Vocabulary("stages", "number of evenly spaced recording stages",
               cast=int, metavar="N"),
    Vocabulary("thresholds", "core and transient cut-offs, comma separated",
               cast=numbers, metavar="CORE,TRANSIENT"),
    Vocabulary("rings", "number of Sholl rings retained for display",
               cast=int, metavar="N"),
    Vocabulary("scaling", "which ring width to read: cell or global",
               cast=str, metavar="WHICH"),
    Vocabulary("annulus_support",
               "smallest observable fraction of every radial annulus required for a frame",
               cast=float, metavar="0-1"),
    Vocabulary("threshold", "data quantile used as a recurrence cut-off",
               cast=float, metavar="0-1"),
    Vocabulary("clusters", "number of sequence groups cut from a clustering tree",
               cast=int, metavar="N"),
    Vocabulary("dilation", "one or two contact radii in pixels, comma separated",
               cast=numbers, metavar="PX,PX"),
    Vocabulary("min_hours", "minimum contact duration retained, in hours",
               cast=float, metavar="HOURS"),
    Vocabulary("max_pairs", "maximum longest-contacting pairs shown in a pair ledger; 0 shows all",
               cast=int, metavar="N"),
    Vocabulary("rhythm_metric", "measurement whose rhythm result is compared between cells",
               metavar="COL"),
    Vocabulary("min_coverage",
               "least share of a window a cell must be observed in before its "
               "windowed summary is drawn",
               cast=float, metavar="0-1"),
    Vocabulary("min_contested",
               "minimum number of identities that must claim a contested pixel",
               cast=int, metavar="N"),
    Vocabulary("inferred_threshold",
               "share of an outline whose ownership must be reconstructed before the "
               "frame is drawn as reconstructed; 0 means any at all",
               cast=float, metavar="0-1"),
    # Placement. Every figure accepts these; see PLACEMENTS below.
    Vocabulary("header_x",
               "left edge of the title, subtitle and footnote",
               cast=length, metavar="N%|Nin"),
    Vocabulary("title_y", "title, measured down from the top edge",
               cast=length, metavar="N%|Nin"),
    Vocabulary("subtitle_y", "subtitle, measured down from the top edge",
               cast=length, metavar="N%|Nin"),
    Vocabulary("footnote_y", "footnote, measured up from the bottom edge",
               cast=length, metavar="N%|Nin"),
)}

#: Where the words go, in the order they appear down the page. Every figure
#: accepts all four without declaring them, because they are about the sheet
#: rather than about what is drawn on it - the same reason ``--stem`` is
#: universal. The default unit differs between them on purpose and each says
#: which: see ``FigureResult`` for why the title is in inches and the footnote
#: is not.
PLACEMENTS: tuple[str, ...] = ("header_x", "title_y", "subtitle_y", "footnote_y")

#: The options that say *which* figure is being drawn rather than anything
#: about what it draws. Every figure honours all three without declaring them,
#: and all three are read before a figure's context exists - the run folder to
#: read, the parts of the page to keep, and which item of the run's plot plan
#: this drawing is. Kept together because that is also what makes them the
#: three an item's own settings must not be checked as though they were the
#: figure's own options: a plan may set them, and no figure declares them.
SELECTION: tuple[str, ...] = ("stem", "panels", "item")

#: Options that are on or off and take no value. Kept apart from ``OPTIONS``
#: because everything that scans the command line has to know which tokens
#: swallow the next word and which do not - otherwise ``--draft <run>`` reads the
#: run folder as the value of ``--draft`` and then reports there is no run folder.
SWITCHES: dict[str, str] = {
    "draft": (
        "write only the figure, straight into <output root>/figures/, and skip "
        "the audit bundle - for deciding what a plot should look like"
    ),
    "overlay": "draw spatial classes over the raw image instead of alone",
}


def _flags(name: str) -> set[str]:
    return {f"--{name}", f"--{name.replace('_', '-')}"}


def switch(name: str, argv: list[str] | None = None) -> bool:
    """Whether an on-or-off option was given. It takes no value."""
    if name not in SWITCHES:
        raise KeyError(
            f"unknown builder switch {name!r}; declared switches are "
            f"{', '.join(sorted(SWITCHES))}"
        )
    tokens = list(sys.argv[1:] if argv is None else argv)
    return bool(_flags(name) & set(tokens))


def skip_tokens(token: str) -> int:
    """How many tokens ``token`` accounts for, itself included.

    One shared answer for the three scanners that walk the same argv - the run
    folder, the builder options and the figure wording. A switch is one token, an
    ``--option=value`` is one, an ``--option value`` is two.
    """
    head = token.partition("=")[0]
    if any(head in _flags(name) for name in SWITCHES):
        return 1
    return 1 if "=" in token else 2


def raw_flag(name: str, argv: list[str] | None = None) -> str | None:
    """The text a flag was given, or ``None`` if the flag is not there.

    The reading half of ``option`` without the casting half, because a figure
    that resolves flag, then what the run recorded, then a default has to know
    *whether* the flag was typed before it decides what to do with it. An empty
    ``--fit=`` is a value and comes back as ``""``, not as an absence.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    wanted = _flags(name)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        head, equals, inline = token.partition("=")
        if head in wanted:
            if equals:
                return inline
            if index + 1 < len(tokens):
                return tokens[index + 1]
            raise SystemExit(f"{head} needs a value")
        # Skip somebody else's option and its value, the same way ``run_folder``
        # does, so the scan never mistakes a value for a flag.
        index += skip_tokens(token)
    return None


def option(
    name: str,
    default: Any = None,
    cast: Callable[[str], Any] = str,
    argv: list[str] | None = None,
) -> Any:
    """The value of one option, or ``default`` if it was not given.

    ``cast`` converts the text; a value the cast rejects stops the build with a
    readable message rather than a traceback out of the middle of a figure.

    For the one flag that has to be read before a figure's context exists:
    ``--stem`` says which movie in the run is being drawn, and the context is
    built around that answer. Every other flag goes through
    ``_schema.FigureContext.option``, which knows which figure is asking and can
    therefore refuse one that page does not accept and read a value the run
    recorded. No builder calls this.
    """
    if name not in OPTIONS:
        raise KeyError(
            f"unknown builder option {name!r}; declared options are "
            f"{', '.join(sorted(OPTIONS))}"
        )
    raw = raw_flag(name, argv)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"--{name.replace('_', '-')} {raw!r}: {error}") from None
