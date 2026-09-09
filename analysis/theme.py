"""The aesthetic engine: one look, tunable, applied everywhere.

Every figure this package draws goes through a :class:`Theme`. Nothing writes a
hex value, a font size or a line width of its own, so a user who wants their
plots to look different changes one block of configuration and every figure
follows.

Three ideas, and the third is the one that matters:

**One contract.** The ``pyflash`` preset is the lab contract, defined once in
``analysis_kit.style`` and reproduced by ``PyFLASH.aesthetics``. It is vendored
here so an external user needs no extra install, and ``conformance_report``
asserts the two still agree wherever ``analysis_kit`` is importable. Every other
preset, including the ``house`` default, is derived from it by scaling rather
than by writing a second set of numbers.

**Roles, not colours.** A figure asks for ``reporter`` or ``surveillance``, not
for red or teal. The same readout is then the same colour in every figure in the
package, and a user can recolour a concept without hunting through five scripts.
This is also the only way a colourblind-safe or greyscale variant can exist at
all: swapping a palette works, swapping hard-coded hex does not.

**A theme is data.** It resolves to a plain dictionary that is written into the
run manifest, so the look of a figure is as reproducible as the numbers in it.

    from analysis.theme import load_theme

    theme = load_theme(config)            # or load_theme() for the house default
    theme.apply()
    ax.plot(hours, values, color=theme.colour("reporter"))
    theme.finish(ax)
    theme.save(fig, path)

Matplotlib is imported lazily, inside the methods that need it, so importing
``analysis`` stays as cheap as importing ``json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Theme",
    "load_theme",
    "PRESETS",
    "ROLES",
    "BASE_COLOURS",
    "lab_contract",
    "conformance_report",
    "parse_hours_per_tick",
    "sentence_case",
]


# ---------------------------------------------------------------- base colours

# Vendored from analysis_kit.style.palette. This is the only place in the
# package allowed to spell out a colour; everything else asks by name.
BASE_COLOURS: dict[str, str] = {
    # series slots
    "red": "#c0392b",
    "blue": "#4878A8",
    "teal": "#0e8f8f",
    "orange": "#d98a17",
    "dark": "#303030",
    # Reserved for test outcomes. Deliberately not one of the five family
    # slots above, so "passed the test" can never be read as "this is the
    # turnover series".
    "violet": "#7b52ab",
    # furniture
    "black": "#000000",
    "white": "#ffffff",
    "muted": "#B0B0B0",
    "grid": "#dfe4ea",
    "blank": "#e6e6e6",
    # Okabe-Ito, for the colourblind-safe preset
    "okabe_orange": "#E69F00",
    "okabe_yellow": "#F0E442",
    "okabe_sky_blue": "#56B4E9",
    "okabe_bluish_green": "#009E73",
    "okabe_blue": "#0072B2",
    "okabe_vermilion": "#D55E00",
    "okabe_reddish_purple": "#CC79A7",
    # greyscale, for the mono preset
    "grey_20": "#333333",
    "grey_40": "#5c5c5c",
    "grey_55": "#8c8c8c",
    "grey_70": "#b3b3b3",
    "grey_85": "#d9d9d9",
}


# ---------------------------------------------------------------------- roles

#: What each role is for. Shown by ``python -m analysis theme --list`` and used
#: to reject a typo in a user's configuration rather than silently ignore it.
ROLES: dict[str, str] = {
    # one per measurement family, so a readout keeps its colour across figures
    "reporter": "signal measured inside the outline",
    "morphology": "size and shape",
    "motility": "displacement and tracks",
    "surveillance": "pixel turnover, processes extending and retracting",
    "evidence": "the tracker's own motion-evidence channels",
    # The three evidence channels drawn apart, in the colours the evidence
    # stack itself encodes them with. A reader who has watched the QC movie has
    # already learned green-red-blue; recolouring them in a figure is a second
    # vocabulary for one thing.
    "evidence_held": "signal present on both sides of a transition",
    "evidence_gained": "signal that arrived - where a cell moved to",
    "evidence_lost": "signal that left - where a cell moved from",
    # An extra imaging channel measured through the same outlines. One role for
    # all of them, whatever they stain: how many there are and what they are
    # called is configuration, so a per-channel role could not be declared here
    # anyway. A figure drawing several tells them apart with the categorical
    # cycle and keeps this colour for the family.
    "channel": "an extra imaging channel measured through the outlines",
    # Shapes in the image that are not cells - vessels, plaques, a wound edge.
    # One role for all of them, for the same reason `channel` is one role: how
    # many sets there are and what they are called is configuration.
    "object": "a set of reference shapes the cells are measured against",
    # whether a name is on the picture at all, which is prior to any measurement
    "named": "signal, or a frame of a cell's life, that carries an identity",
    "inferred": "carries an identity the tracker reconstructed rather than saw",
    "unclaimed": "signal the pipeline declines to attribute to any identity",
    # states a figure needs to distinguish
    "rhythmic": "passed the rhythm tests",
    "arrhythmic": "did not pass",
    "significant": "passed whatever test the panel is about",
    "not_significant": "did not pass",
    "highlight": "the one thing the reader should look at",
    "reference": "median lines, null levels, guides",
    "invalid": "measured but flagged untrustworthy",
    "missing": "no measurement for this cell-frame",
    # Absence as the subject of the picture rather than as a hole in it. The
    # never-visited field is most of the frame, so drawing it in the same pale
    # grey that means "nothing here" makes the one thing the figure is about
    # the hardest thing on it to see.
    "unvisited": "ground no cell and no unclaimed signal ever covered",
    "visited": "ground a named cell or unclaimed signal covered at some point",
    "stable": "a measurement that identifies a cell across time",
    "variable": "a measurement that changes faster than it identifies",
    # drawing on images
    "outline": "cell boundary drawn over a raw frame",
    "fit": "a model curve over data",
    # furniture
    "ink": "axes, ticks and the title",
    "caption": "subtitles, footnotes and in-panel notes",
    "page": "figure background",
}

_ROLE_BASE: dict[str, str] = {
    "reporter": "red",
    "morphology": "blue",
    "motility": "orange",
    "surveillance": "teal",
    "evidence": "dark",
    "evidence_held": "teal",
    "evidence_gained": "red",
    "evidence_lost": "blue",
    # Borrowed from the Okabe-Ito block because it is the one hue no house
    # family already owns, and an extra channel is by definition a family the
    # house palette did not anticipate.
    "channel": "okabe_reddish_purple",
    # Okabe-Ito again, for the same reason `channel` is: no measurement family
    # owns this hue in any preset, and a reference shape is scenery - it must
    # not read as one of the readouts drawn beside it.
    "object": "okabe_sky_blue",
    "named": "dark",
    # A faded version of the name colour, not a fourth hue: a reconstructed
    # outline is still that cell, drawn with less to go on.
    "inferred": "grey_70",
    "unclaimed": "orange",
    "rhythmic": "violet",
    "arrhythmic": "muted",
    "significant": "violet",
    "not_significant": "muted",
    "highlight": "red",
    "reference": "muted",
    "invalid": "grey_55",
    "missing": "blank",
    "unvisited": "black",
    "visited": "black",
    "stable": "blue",
    "variable": "red",
    "outline": "teal",
    "fit": "dark",
    "ink": "black",
    "caption": "dark",
    "page": "white",
}

#: Order for unlabelled series. Blue leads so a single-series plot is not the
#: colour of an error - the same reasoning as the house categorical cycle.
_ROLE_CYCLE: tuple[str, ...] = ("morphology", "reporter", "surveillance", "motility", "evidence")


# -------------------------------------------------------------------- presets

def _contract() -> dict[str, Any]:
    """The lab contract, key for key with ``analysis_kit.style.THEMES['pyflash']``.

    Deliberately not a preset. It is a conformance reference, not a look anybody
    should pick: its name means nothing outside this lab, and offering it beside
    ``house`` would only ask a user to choose between two things they cannot
    tell apart from the name. Every preset is derived from it, and
    ``conformance_report`` asserts it still matches the installed
    ``analysis_kit``. Do not tune it.
    """
    return {
        "font_family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font_weight": "normal",
        "title_size": 20.0,
        "title_weight": "bold",
        "axis_size": 22.0,
        "axis_weight": "normal",
        "tick_size": 20.0,
        "legend_size": 15.0,
        "legend_frame": False,
        # Where a legend goes. "auto" leaves matplotlib to find the emptiest
        # corner inside the axes, which is fine until the data moves and the
        # legend jumps somewhere else on the next run. The outside placements
        # are stable: they depend on the axes, not on the data.
        "legend_location": "auto",
        "legend_pad": 0.02,      # gap between the axes and an outside legend
        "legend_columns": 0,     # 0 means one row for above/below
        "annotation_size": 16.0,
        "suptitle_size": 22.0,
        "spine_width": 2.0,
        "tick_mark_width": 2.0,
        "tick_mark_length": 11.0,
        "tick_direction": "out",
        "despine": True,
        "show_grid": False,
        "grid_width": 1.0,
        "line_width": 2.4,
        "legend_line_width": 3.0,
        "marker_size": 9.0,
        "scatter_alpha": 0.6,
        "figure_size": (7.0, 5.0),
        "save_dpi": 600,
        "save_bbox": "tight",
        "save_pad_inches": 0.1,
        "transparent": True,
        "diverging_cmap": "coolwarm",
        "sequential_cmap": "viridis",
        "image_cmap": "gray",
        "roles": dict(_ROLE_BASE),
    }


def _is_day_aligned(step: float, day: float = 24.0) -> bool:
    """True when every tick ``step`` hours apart falls at the same time of day.

    That is what "a factor or a multiple of 24" means once the number is
    allowed to be fractional: 1.5 h divides the day into 16, so it qualifies;
    10 h does not divide it at all. Compared with a tolerance because a step can
    arrive as 0.1 + 0.2.
    """
    if not step > 0:
        return False
    for whole in (day / step, step / day):
        if abs(whole - round(whole)) < 1e-9 and round(whole) >= 1:
            return True
    return False


def parse_hours_per_tick(value: Any) -> float:
    """A tick step in hours, or a readable refusal.

    Public because a builder validates its ``--hour-ticks`` flag with it before
    drawing anything: a bad number should stop the command, not surface as a
    traceback out of the middle of a figure that has already copied its sources.
    """
    try:
        step = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{value!r} is not a number of hours") from None
    if not _is_day_aligned(step):
        raise ValueError(
            f"must be a factor or a multiple of 24, not {step:g}; "
            "try 1, 2, 3, 4, 6, 8, 12, 24, 48 or 72"
        )
    return step


def sentence_case(text: str) -> str:
    """``text`` with its first letter raised, leaving deliberate case alone.

    Titles and axis labels start with a capital. The catch is that a lot of the
    words a biologist puts first carry their own capitalisation - pH, mCherry,
    iNOS - and forcing the first letter would corrupt the name. So the first
    word is returned exactly as written whenever it already contains a capital.

    A label that does not open on a plain ``a``-``z`` letter is left alone
    entirely: ``"24 h blocks"`` and a unit written with a micro sign are already
    correct, and raising the first letter they happen to contain would not be.

    Only the first character is ever changed, so the second line of a two-line
    label stays lower case - it is a continuation, not a new label.
    """
    stripped = text.lstrip()
    if not stripped or not ("a" <= stripped[0] <= "z"):
        return text
    if any(character.isupper() for character in stripped.split()[0]):
        return text
    index = len(text) - len(stripped)
    return text[:index] + text[index].upper() + text[index + 1 :]


def _scaled(base: dict[str, Any], type_scale: float, weight_scale: float) -> dict[str, Any]:
    """Scale every type size and every stroke width by one factor each."""
    out = dict(base)
    for key in ("title_size", "axis_size", "tick_size", "legend_size",
                "annotation_size", "suptitle_size"):
        out[key] = round(base[key] * type_scale, 2)
    for key in ("spine_width", "tick_mark_width", "tick_mark_length",
                "line_width", "legend_line_width"):
        out[key] = round(base[key] * weight_scale, 3)
    return out


def _base() -> dict[str, Any]:
    """The contract plus the choices this package makes that it does not cover.

    The contract covers type, strokes and export. It says nothing about where a
    legend goes, because PyFLASH leaves that to matplotlib's ``best``. This
    package does not leave it: ``best`` picks the emptiest corner of the *data*,
    so a legend moves when the data moves and two runs of the same figure can
    disagree about where it is. Above the axes it depends on the axes instead,
    and never covers a point.
    """
    style = _contract()
    style["legend_location"] = "above"
    # Hours per tick on a time axis. Matplotlib's locator counts in tens, so a
    # two-day recording gets ticks at 10, 20, 30, 40, 50 h - none of which is a
    # time of day. Circadian data is base 24, and a reader comparing a peak at
    # 21 h to one at 45 h needs the day boundaries marked, not round decimals.
    # Must stay a factor or a multiple of 24; see ``Theme.hour_ticks``.
    style["hours_per_tick"] = 24.0
    # Continuous value axes opt into this shared rule. It reproduces the useful
    # part of PyFLASH's convention: five ticks from a visible zero anchor to an
    # outer limit rounded up to a multiple of 5 in the second significant
    # figure. The three settings are data so a run can choose another density,
    # rounding structure or anchor without changing plotting code.
    style["value_tick_count"] = 5
    style["value_tick_round_to"] = 5.0
    style["value_tick_start"] = 0.0
    # Every continuous colour key uses one physical shape. Relative inset
    # widths turn a key beside a small multiple into an unreadable hairline;
    # nominal inches keep the same key legible beside every panel size.
    style["colour_bar_width_inches"] = 0.18
    style["colour_bar_height_inches"] = 2.2
    style["colour_bar_gap_inches"] = 0.16
    # Time ticks stay aligned to this origin. ``Theme.hour_ticks`` includes the
    # aligned tick immediately before the first observation, so a recording
    # beginning at 0.5 h still has a visible starting tick at 0 h.
    style["hours_tick_start"] = 0.0
    return style


def _house() -> dict[str, Any]:
    """The package default: the lab contract, sized up.

    The contract was drawn for one 7x5 panel. This package's figures are
    multi-panel and are usually read as a whole page, where contract-sized type
    comes out small. The default therefore runs a quarter larger with slightly
    heavier strokes. Use ``pyflash`` for a figure that has to sit beside
    existing lab panels.
    """
    style = _scaled(_base(), 1.25, 1.2)
    style["marker_size"] = 10.0
    return style


def _preset_talk() -> dict[str, Any]:
    """Bigger again, so it survives being projected across a room.

    The canvas deliberately does not grow. Growing both would leave the type the
    same size relative to the figure, which is the thing a talk needs changed.
    """
    style = _scaled(_base(), 1.55, 1.4)
    style["marker_size"] = 12.0
    style["transparent"] = False
    return style


def _preset_print() -> dict[str, Any]:
    """Sized for a single journal column, where 20 pt ticks are absurd."""
    style = _scaled(_base(), 0.42, 0.55)
    style["figure_size"] = (3.4, 2.6)
    style["tick_mark_length"] = 3.5
    style["marker_size"] = 4.5
    style["save_dpi"] = 1200
    return style


def _preset_colourblind() -> dict[str, Any]:
    style = _house()
    style["roles"] = {
        **_ROLE_BASE,
        "reporter": "okabe_vermilion",
        "morphology": "okabe_blue",
        "motility": "okabe_orange",
        "surveillance": "okabe_bluish_green",
        "evidence": "dark",
        "evidence_held": "okabe_bluish_green",
        "evidence_gained": "okabe_vermilion",
        "evidence_lost": "okabe_blue",
        "unclaimed": "okabe_orange",
        "rhythmic": "okabe_reddish_purple",
        "significant": "okabe_reddish_purple",
        "highlight": "okabe_vermilion",
        "invalid": "grey_55",
        "stable": "okabe_blue",
        "variable": "okabe_vermilion",
        "outline": "okabe_sky_blue",
    }
    style["diverging_cmap"] = "PuOr"
    return style


def _preset_mono() -> dict[str, Any]:
    """Greyscale, for a journal that charges for colour."""
    style = _house()
    style["roles"] = {
        **_ROLE_BASE,
        "reporter": "grey_20",
        "morphology": "grey_40",
        "motility": "grey_55",
        "surveillance": "grey_20",
        "evidence": "grey_55",
        "evidence_held": "grey_20",
        "evidence_gained": "grey_40",
        "evidence_lost": "grey_70",
        "named": "grey_20",
        "inferred": "grey_40",
        "unclaimed": "grey_70",
        "rhythmic": "grey_20",
        "arrhythmic": "grey_70",
        "significant": "grey_20",
        "not_significant": "grey_70",
        "highlight": "black",
        "reference": "grey_70",
        "stable": "grey_55",
        "variable": "grey_20",
        "outline": "white",
        "caption": "grey_20",
        "fit": "black",
    }
    style["diverging_cmap"] = "RdGy"
    style["sequential_cmap"] = "Greys"
    return style


PRESETS: dict[str, Any] = {
    "house": _house,
    "talk": _preset_talk,
    "print": _preset_print,
    "colourblind": _preset_colourblind,
    "mono": _preset_mono,
}

_PRESET_ALIASES = {
    "default": "house",
    "motion": "house",
    "poster": "talk",
    "slide": "talk",
    "slides": "talk",
    "paper": "print",
    "journal": "print",
    "colorblind": "colourblind",
    "cb_safe": "colourblind",
    "greyscale": "mono",
    "grayscale": "mono",
}


# ---------------------------------------------------------------------- theme

@dataclass(frozen=True)
class Theme:
    """A complete description of how this package's figures should look."""

    preset: str = "house"
    style: dict[str, Any] = field(default_factory=_house)
    palette: dict[str, str] = field(default_factory=lambda: dict(BASE_COLOURS))
    source: str = "default"
    #: Project vocabulary: experimental group to colour, from the conditions
    #: block rather than the theme block. Kept apart from ``palette`` and
    #: unreachable from :meth:`colour`, so declaring a condition called ``ink``
    #: recolours that condition and not the axes of every figure.
    conditions: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- colours

    def colour(self, role: str) -> str:
        """The value for a role. Accepts a literal colour too, for one-offs."""
        roles = self.style["roles"]
        if role in roles:
            value = roles[role]
            return self.palette.get(value, value)
        if role in self.palette:
            return self.palette[role]
        if isinstance(role, str) and (role.startswith("#") or role in ("none", "white", "black")):
            return role
        raise KeyError(
            f"unknown colour role {role!r}. Roles: {', '.join(sorted(ROLES))}. "
            f"Palette names: {', '.join(sorted(self.palette))}."
        )

    #: American spelling, for consumers whose surrounding code uses it.
    def color(self, role: str) -> str:
        return self.colour(role)

    def cycle(self, n: int | None = None) -> list[str]:
        """Colours for unlabelled series, in role order."""
        order = [self.colour(role) for role in _ROLE_CYCLE]
        if n is None:
            return order
        if n <= len(order):
            return order[:n]
        raise ValueError(
            f"the role cycle has {len(order)} colours and {n} were asked for. "
            "Repeating it would draw two series in one colour; name the series "
            f"by role, or sample {self.style['sequential_cmap']!r} instead."
        )

    # ------------------------------------------------------ conditions

    def with_conditions(self, mapping: dict[str, str]) -> "Theme":
        """The same theme, carrying this project's condition colours.

        The table arrives from the ``conditions`` block, not the ``theme``
        block, because which colour a treatment should be is a decision about
        the experiment rather than about the house style.
        """
        return replace(self, conditions={**self.conditions, **dict(mapping)})

    def condition_colour(self, name: str, index: int = 0) -> str:
        """The colour for one experimental group.

        Resolution order, first hit wins: the project's declared table, then a
        package palette name, then a ``#rrggbb`` literal. An unrecognised name
        falls back to a colour-blind-safe slot chosen by *index* rather than
        raising, because a group appearing in the data that nobody declared is
        a data problem the figure should still be able to draw and label.
        """
        declared = self.conditions.get(name)
        if declared is not None:
            return declared
        if name in self.palette:
            return self.palette[name]
        if isinstance(name, str) and name.startswith("#"):
            return name
        fallback = ("okabe_blue", "okabe_vermilion", "okabe_bluish_green",
                    "okabe_reddish_purple", "okabe_orange", "okabe_sky_blue")
        return self.palette[fallback[int(index) % len(fallback)]]

    #: American spelling, matching :meth:`color`.
    def condition_color(self, name: str, index: int = 0) -> str:
        return self.condition_colour(name, index)

    def __getitem__(self, key: str) -> Any:
        return self.style[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.style.get(key, default)

    # ---------------------------------------------------------------- text

    def size(self, step: str) -> float:
        """A named step on the type ladder, in points.

        Matplotlib's own rcParams cover axes, ticks and legends. Everything
        *else* a figure writes - a figure title, a subtitle, a footnote, a label
        over an image tile - is passed explicitly, and that is exactly where
        hard-coded numbers accumulate and stop responding to the theme. The
        ladder is one base size (the legend size) times a fixed set of ratios,
        so a preset that scales type scales all of it.
        """
        base = float(self.style["legend_size"])
        ladder = {
            "note": 0.90,        # footnotes and provenance lines
            "tile": 0.93,        # a label over an image tile
            "caption": 0.97,     # in-panel notes
            "subtitle": 1.00,    # the line under a figure title
            "emphasis": 1.03,    # a value the reader should read off
            "annotation": 1.07,  # free text inside the axes
            "panel": 1.13,       # a panel's own axis label
            "title": 1.40,       # the figure title
        }
        if step not in ladder:
            raise KeyError(
                f"unknown type step {step!r}; try one of {', '.join(ladder)}"
            )
        return round(base * ladder[step], 2)

    # -------------------------------------------------------------- geometry

    #: The canvas the type sizes were chosen against. A theme that asks for a
    #: different figure size scales every canvas by the same ratio.
    HOUSE_CANVAS_WIDTH = 7.0

    @property
    def canvas_scale(self) -> float:
        return float(self.style["figure_size"][0]) / self.HOUSE_CANVAS_WIDTH

    def canvas(self, width: float, height: float) -> tuple[float, float]:
        """Scale a builder's nominal canvas, in inches, to this theme.

        A multi-panel figure is laid out in fractional axes rectangles, so its
        proportions survive any canvas size; what does not survive is the ratio
        between the canvas and the type on it. Passing the nominal size through
        here keeps that ratio fixed, so ``print`` produces a small dense figure
        rather than a huge one with unreadably small labels.
        """
        # Deliberately not rounded: rounding each side independently shifts the
        # aspect ratio, and a panel laid out in fractional rectangles then no
        # longer lines up with itself.
        scale = self.canvas_scale
        return (float(width) * scale, float(height) * scale)

    def point_area(self, multiple: float = 1.0) -> float:
        """Scatter ``s=`` for a marker ``multiple`` times the theme's marker size."""
        return float(self.style["marker_size"] * multiple) ** 2

    def stroke(self, kind: str = "line") -> float:
        """A stroke width by purpose rather than by number."""
        widths = {
            "line": self.style["line_width"],
            "emphasis": self.style["legend_line_width"],
            "guide": self.style["spine_width"],
            "hairline": self.style["spine_width"] / 2.5,
        }
        if kind not in widths:
            raise KeyError(f"unknown stroke {kind!r}; try one of {', '.join(widths)}")
        return float(widths[kind])

    # ------------------------------------------------------------ value axes

    @staticmethod
    def _rounded_tick_span(value: float, round_to: float) -> float:
        """Round a positive span up on PyFLASH's significant-figure grid."""
        value = abs(float(value))
        if value == 0:
            return 0.0
        order = 10.0 ** (math.floor(math.log10(value)) - 1)
        quantum = float(round_to) * order
        return math.ceil(value / quantum - 1e-12) * quantum

    def value_ticks(
        self,
        low: float,
        high: float,
        *,
        count: int | None = None,
        round_to: float | None = None,
        start: float | None = None,
    ) -> list[float]:
        """Shared ticks for a continuous value axis.

        Positive values run from the configured start to a rounded upper
        limit; negative values end at the start. A range crossing the start is
        symmetric around it, which keeps zero visible in signed change plots.
        The default is five ticks and PyFLASH's multiple-of-5 rounding in the
        second significant figure.
        """
        count = int(self.style["value_tick_count"] if count is None else count)
        round_to = float(
            self.style["value_tick_round_to"] if round_to is None else round_to
        )
        start = float(self.style["value_tick_start"] if start is None else start)
        low, high = float(low), float(high)
        if not all(math.isfinite(value) for value in (low, high, round_to, start)):
            raise ValueError("value tick limits, start and rounding must be finite")
        if count < 2:
            raise ValueError("value_tick_count must be at least 2")
        if round_to <= 0:
            raise ValueError("value_tick_round_to must be positive")
        if high < low:
            low, high = high, low

        if low < start < high:
            span = self._rounded_tick_span(
                max(start - low, high - start), round_to
            )
            lower, upper = start - span, start + span
        elif high <= start:
            span = self._rounded_tick_span(start - low, round_to)
            lower, upper = start - span, start
        else:
            span = self._rounded_tick_span(high - start, round_to)
            lower, upper = start, start + span
        if lower == upper:
            return [round(start, 12)]
        step = (upper - lower) / (count - 1)
        ticks = [round(lower + index * step, 12) for index in range(count)]
        return [0.0 if abs(value) < 1e-12 else value for value in ticks]

    def set_value_ticks(
        self,
        ax: Any,
        *,
        low: float | None = None,
        high: float | None = None,
        axis: str = "y",
        count: int | None = None,
        round_to: float | None = None,
        start: float | None = None,
    ) -> list[float]:
        """Apply :meth:`value_ticks` and matching limits to one numeric axis."""
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
        ticks = self.value_ticks(
            limits[0] if low is None else low,
            limits[1] if high is None else high,
            count=count, round_to=round_to, start=start,
        )
        if axis == "x":
            ax.set_xlim(ticks[0], ticks[-1])
            ax.set_xticks(ticks)
        else:
            ax.set_ylim(ticks[0], ticks[-1])
            ax.set_yticks(ticks)
        return ticks

    # ------------------------------------------------------------- time axes

    def hour_ticks(
        self,
        low: float,
        high: float,
        step: float | None = None,
        start: float | None = None,
    ) -> list[float]:
        """Tick positions for an axis measured in hours.

        A day is the unit the experiment is about, so the ticks count in days,
        not in tens of hours. ``step`` must divide 24 evenly or be a whole
        number of days - 6 h, 12 h and 48 h all keep every tick at the same
        time of day, whereas 10 h puts the third tick at 06:00 and invites the
        reader to compare two peaks against a grid that means nothing.

        The aligned tick immediately before the first observation is retained,
        so every time axis has a visible starting tick. ``start`` sets the
        alignment origin and defaults to the theme's ``hours_tick_start``.
        Positions are returned rather than applied, so a caller may label them
        differently without inventing a second placement rule.
        """
        try:
            step = parse_hours_per_tick(self.style["hours_per_tick"] if step is None else step)
        except ValueError as error:
            raise ValueError(f"hours_per_tick {error}") from None
        start = float(self.style["hours_tick_start"] if start is None else start)
        if not math.isfinite(start):
            raise ValueError("hours_tick_start must be finite")
        low, high = float(low), float(high)
        if high < low:
            low, high = high, low
        # A hair of tolerance at each end: a tick exactly on the limit is inside
        # the axis, and floating-point hours from a frame interval rarely land
        # on an integer.
        edge = step * 1e-9
        first = start + math.floor((low + edge - start) / step) * step
        count = int(math.floor((high + edge - first) / step)) + 1
        return [round(first + index * step, 6) for index in range(max(count, 0))]

    # -------------------------------------------------------- matplotlib

    def rcparams(self, **overrides: Any) -> dict[str, Any]:
        """rcParams derived from this theme's own numbers, never listed twice."""
        style = self.style
        params: dict[str, Any] = {
            "axes.linewidth": style["spine_width"],
            "xtick.major.width": style["tick_mark_width"],
            "ytick.major.width": style["tick_mark_width"],
            "xtick.minor.width": style["tick_mark_width"] / 2,
            "ytick.minor.width": style["tick_mark_width"] / 2,
            "xtick.major.size": style["tick_mark_length"],
            "ytick.major.size": style["tick_mark_length"],
            "xtick.labelsize": style["tick_size"],
            "ytick.labelsize": style["tick_size"],
            "xtick.direction": style["tick_direction"],
            "ytick.direction": style["tick_direction"],
            "axes.labelsize": style["axis_size"],
            "axes.labelweight": style["axis_weight"],
            "axes.titlesize": style["title_size"],
            "axes.titleweight": style["title_weight"],
            "axes.spines.top": not style["despine"],
            "axes.spines.right": not style["despine"],
            "legend.frameon": style["legend_frame"],
            "legend.fontsize": style["legend_size"],
            "legend.title_fontsize": style["legend_size"],
            "font.weight": style["font_weight"],
            "font.family": "sans-serif",
            "font.sans-serif": list(style["font_family"]),
            "font.size": style["annotation_size"],
            "figure.titlesize": style["suptitle_size"],
            "figure.titleweight": "bold",
            "savefig.dpi": style["save_dpi"],
            "savefig.bbox": style["save_bbox"],
            "savefig.pad_inches": style["save_pad_inches"],
            "figure.figsize": tuple(style["figure_size"]),
            "lines.linewidth": style["line_width"],
            "axes.grid": style["show_grid"],
            "grid.color": self.colour("grid") if "grid" in self.palette else "#dfe4ea",
            "grid.linewidth": style["grid_width"],
            "image.cmap": style["sequential_cmap"],
            # Editable text in every vector format, not outlined glyph paths:
            # without this a figure cannot be relabelled without redrawing it.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
            "axes.edgecolor": self.colour("ink"),
            "xtick.color": self.colour("ink"),
            "ytick.color": self.colour("ink"),
            "legend.shadow": False,
            "patch.force_edgecolor": False,
            "figure.dpi": 110,
        }
        if style["transparent"]:
            params.update({
                "figure.facecolor": "none",
                "figure.edgecolor": "none",
                "axes.facecolor": "none",
                "savefig.facecolor": "none",
                "savefig.edgecolor": "none",
                "savefig.transparent": True,
            })
        else:
            page = self.colour("page")
            params.update({
                "figure.facecolor": page,
                "axes.facecolor": page,
                "savefig.facecolor": page,
                "savefig.transparent": False,
            })
        params.update(overrides)
        return params

    def apply(self, *, series_cycle: bool = True, **overrides: Any) -> dict[str, Any]:
        """Set this theme on the global matplotlib state; return what was set."""
        from matplotlib import cycler, pyplot as plt

        params = self.rcparams(**overrides)
        if series_cycle:
            params["axes.prop_cycle"] = cycler(color=self.cycle())
        plt.rcParams.update(params)
        return params

    def finish(self, *axes: Any, keep_spines: tuple[str, ...] = ()) -> None:
        """Re-assert spines, ticks and label capitals after a plotting call.

        Seaborn, pandas and several matplotlib helpers reset spine widths as a
        side effect of drawing. Call this last on each axes.

        Axis labels and axes titles are also raised to sentence case here, so
        the rule holds for a label written by a builder and for one that a
        plotting library invented from a column name. The text objects are
        edited in place rather than re-set, because ``set_xlabel`` would take
        the font size back to matplotlib's default.
        """
        style = self.style
        ink = self.colour("ink")
        drop = ("top", "right") if style["despine"] else ()
        for ax in axes:
            for text in (ax.xaxis.label, ax.yaxis.label, ax.title):
                text.set_text(sentence_case(text.get_text()))
            for side in drop:
                ax.spines[side].set_visible(side in keep_spines)
            for side in ("left", "bottom", *keep_spines):
                if ax.spines[side].get_visible():
                    ax.spines[side].set_linewidth(style["spine_width"])
                    ax.spines[side].set_color(ink)
            ax.tick_params(
                width=style["tick_mark_width"],
                length=style["tick_mark_length"],
                color=ink,
                labelcolor=ink,
            )

    # ---------------------------------------------------------------- legend

    #: Anchor and alignment for each placement, in axes coordinates. ``pad`` is
    #: added or subtracted at draw time so one setting moves them all.
    _LEGEND_PLACEMENTS = {
        "above": ("lower center", (0.5, 1.0), (0.0, +1.0)),
        "above left": ("lower left", (0.0, 1.0), (0.0, +1.0)),
        "above right": ("lower right", (1.0, 1.0), (0.0, +1.0)),
        "below": ("upper center", (0.5, 0.0), (0.0, -1.0)),
        "right": ("center left", (1.0, 0.5), (+1.0, 0.0)),
        "inside": ("best", None, (0.0, 0.0)),
        "auto": ("best", None, (0.0, 0.0)),
    }

    def legend(
        self,
        ax: Any,
        *args: Any,
        location: str | None = None,
        columns: int | None = None,
        pad: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Draw a legend where the theme says, not where matplotlib guesses.

        ``location="above"`` puts it in a single row just above the axes, which
        is the placement that never covers data and never moves between runs.
        Pass ``location`` to override the theme for one panel whose layout
        cannot take an outside legend.

        When the axes already carries a title, the title is pushed up by the
        legend's estimated height so the two do not overlap.
        """
        placement = (location or self.style["legend_location"]).strip().lower()
        if placement not in self._LEGEND_PLACEMENTS:
            raise KeyError(
                f"unknown legend location {placement!r}; try one of "
                f"{', '.join(sorted(self._LEGEND_PLACEMENTS))}"
            )
        loc, anchor, direction = self._LEGEND_PLACEMENTS[placement]
        gap = self.style["legend_pad"] if pad is None else float(pad)

        # An explicit `loc` from the caller is a manual placement and wins
        # outright. Honouring it *and* the theme's anchor would put the legend
        # somewhere neither asked for, which is the one outcome worth avoiding.
        if "loc" in kwargs:
            anchor = None

        options: dict[str, Any] = {
            "loc": loc,
            "frameon": self.style["legend_frame"],
            "fontsize": self.style["legend_size"],
            "borderaxespad": 0.0,
        }
        requested = columns if columns is not None else self.style["legend_columns"]
        if requested:
            # ``ncol`` is useful inside an axes too: a six-entry key should not
            # become one tall strip that hides the whole right-hand side.
            options["ncol"] = int(requested)
        if anchor is not None:
            options["bbox_to_anchor"] = (
                anchor[0] + direction[0] * gap,
                anchor[1] + direction[1] * gap,
            )
            handles, labels = ax.get_legend_handles_labels()
            entries = len(kwargs.get("labels", labels)) or 1
            if placement.startswith(("above", "below")):
                # One row unless asked otherwise: the point of putting it above
                # the axes is to spend height once, not once per entry.
                options["ncol"] = int(requested) if requested else entries
                options["columnspacing"] = 1.4
            elif requested:
                options["ncol"] = int(requested)

        options.update(kwargs)
        legend = ax.legend(*args, **options)

        title = ax.get_title()
        if title and placement.startswith("above"):
            entries = len(ax.get_legend_handles_labels()[1]) or 1
            columns_used = max(int(options.get("ncol", 1)), 1)
            rows = -(-entries // columns_used)          # ceiling division
            ax.set_title(
                title,
                pad=self.style["legend_size"] * (1.7 * rows + 0.6),
            )
        return legend

    def savefig_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs = {
            "dpi": self.style["save_dpi"],
            "bbox_inches": self.style["save_bbox"],
            "pad_inches": self.style["save_pad_inches"],
            "transparent": self.style["transparent"],
        }
        kwargs.update(overrides)
        return kwargs

    def save(self, figure: Any, svg_path: str | Path, preview: bool = True) -> Path:
        """Write the vector figure and, by default, an opaque PNG for visual QA."""
        svg_path = Path(svg_path)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(svg_path, format="svg", **self.savefig_kwargs())
        if preview:
            # Opaque white so overlapping labels and clipping are actually
            # visible when the preview is inspected.
            figure.savefig(
                svg_path.parent / "preview.png", format="png", dpi=200,
                transparent=False, facecolor="white", bbox_inches="tight",
            )
        return svg_path

    # ------------------------------------------------------------- record

    def stamp(self) -> dict[str, Any]:
        """What goes in the run manifest, so a look is as reproducible as a number."""
        return {
            "preset": self.preset,
            "source": self.source,
            "resolved_roles": {role: self.colour(role) for role in sorted(ROLES)},
            "condition_colours": dict(self.conditions),
            "type": {
                key: self.style[key]
                for key in ("title_size", "axis_size", "tick_size", "legend_size",
                            "annotation_size")
            },
            "strokes": {
                key: self.style[key]
                for key in ("spine_width", "tick_mark_width", "line_width")
            },
            "legend": {
                key: self.style[key]
                for key in ("legend_location", "legend_pad", "legend_columns")
            },
            "figure_size": list(self.style["figure_size"]),
            "hours_per_tick": self.style["hours_per_tick"],
            "hours_tick_start": self.style["hours_tick_start"],
            "value_ticks": {
                "count": self.style["value_tick_count"],
                "round_to": self.style["value_tick_round_to"],
                "start": self.style["value_tick_start"],
            },
            "colour_bar": {
                "width_inches": self.style["colour_bar_width_inches"],
                "height_inches": self.style["colour_bar_height_inches"],
                "gap_inches": self.style["colour_bar_gap_inches"],
            },
            "save_dpi": self.style["save_dpi"],
            "colormaps": {
                key: self.style[key]
                for key in ("diverging_cmap", "sequential_cmap", "image_cmap")
            },
        }


# --------------------------------------------------------------------- loading

def _resolve_preset(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    key = _PRESET_ALIASES.get(key, key)
    if key not in PRESETS:
        raise KeyError(
            f"unknown theme preset {name!r}; try one of "
            f"{', '.join(sorted(PRESETS))}"
        )
    return key


_TUNABLE = {
    "font_family", "font_weight", "title_size", "title_weight", "axis_size",
    "axis_weight", "tick_size", "legend_size", "legend_frame", "annotation_size",
    "suptitle_size", "spine_width", "tick_mark_width", "tick_mark_length",
    "tick_direction", "despine", "show_grid", "grid_width", "line_width",
    "legend_line_width", "marker_size", "scatter_alpha", "figure_size",
    "save_dpi", "save_bbox", "save_pad_inches", "transparent",
    "diverging_cmap", "sequential_cmap", "image_cmap", "hours_per_tick",
    "hours_tick_start", "value_tick_count", "value_tick_round_to",
    "value_tick_start", "colour_bar_width_inches", "colour_bar_height_inches",
    "colour_bar_gap_inches",
}


def load_theme(source: Any = None) -> Theme:
    """Build a theme from a configuration block, a file, a preset name, or nothing.

    Accepts, in order of convenience:

    * ``None`` - the house default;
    * ``"talk"`` - a preset name;
    * a ``dict`` - a ``theme`` block (see below), or a whole analysis
      configuration containing one;
    * a path to a JSON file holding either of those;
    * an :class:`AnalysisConfig`.

    The block::

        "theme": {
          "preset": "house",
          "roles": {"reporter": "#7a1fa2", "surveillance": "orange"},
          "palette": {"my_purple": "#7a1fa2"},
          "type_scale": 1.0,
          "stroke_scale": 1.0,
          "line_width": 2.4,
          "figure_size": [7.0, 5.0],
          "value_tick_count": 5,
          "value_tick_round_to": 5,
          "value_tick_start": 0,
          "hours_per_tick": 12,
          "hours_tick_start": 0,
          "transparent": true
        }

    A role may be given a palette name or a literal ``#rrggbb``. An unknown role
    name is an error rather than a silent no-op, because a typo in a theme is
    otherwise invisible until someone notices the wrong figure.
    """
    block, origin = _extract_block(source)

    preset = _resolve_preset(block.get("preset", "house"))
    style = PRESETS[preset]()
    palette = dict(BASE_COLOURS)

    extra_palette = block.get("palette") or {}
    if not isinstance(extra_palette, dict):
        raise TypeError("theme.palette must be a mapping of name to colour")
    palette.update({str(k): str(v) for k, v in extra_palette.items()})

    type_scale = float(block.get("type_scale", 1.0))
    stroke_scale = float(block.get("stroke_scale", 1.0))
    if type_scale != 1.0 or stroke_scale != 1.0:
        style = _scaled(style, type_scale, stroke_scale)

    unknown = set(block) - _TUNABLE - {"preset", "roles", "palette", "type_scale", "stroke_scale"}
    if unknown:
        raise KeyError(
            f"unknown theme setting(s): {', '.join(sorted(unknown))}. "
            f"Tunable: {', '.join(sorted(_TUNABLE))}, plus preset, roles, "
            "palette, type_scale, stroke_scale."
        )
    for key in _TUNABLE & set(block):
        style[key] = block[key]
    if "figure_size" in style:
        style["figure_size"] = tuple(style["figure_size"])
    if isinstance(style.get("font_family"), str):
        style["font_family"] = [style["font_family"]]
    # Caught here rather than when a figure draws, so a bad number stops the
    # configuration instead of a build that has already run the analysis.
    try:
        style["hours_per_tick"] = parse_hours_per_tick(style["hours_per_tick"])
    except ValueError as error:
        raise ValueError(f"hours_per_tick {error}") from None
    try:
        raw_count = float(style["value_tick_count"])
        if not raw_count.is_integer() or int(raw_count) < 2:
            raise ValueError("must be a whole number of at least 2")
        style["value_tick_count"] = int(raw_count)
        style["value_tick_round_to"] = float(style["value_tick_round_to"])
        style["value_tick_start"] = float(style["value_tick_start"])
        style["hours_tick_start"] = float(style["hours_tick_start"])
        for key in ("colour_bar_width_inches", "colour_bar_height_inches",
                    "colour_bar_gap_inches"):
            style[key] = float(style[key])
        if style["value_tick_round_to"] <= 0:
            raise ValueError("value_tick_round_to must be positive")
        if style["colour_bar_width_inches"] <= 0 or style["colour_bar_height_inches"] <= 0:
            raise ValueError("colour-bar width and height must be positive")
        if style["colour_bar_gap_inches"] < 0:
            raise ValueError("colour-bar gap must be non-negative")
        if not all(math.isfinite(style[key]) for key in (
            "value_tick_round_to", "value_tick_start", "hours_tick_start",
            "colour_bar_width_inches", "colour_bar_height_inches",
            "colour_bar_gap_inches",
        )):
            raise ValueError("tick settings must be finite")
    except (TypeError, ValueError) as error:
        raise ValueError(f"theme tick settings {error}") from None

    role_overrides = block.get("roles") or {}
    if not isinstance(role_overrides, dict):
        raise TypeError("theme.roles must be a mapping of role to colour")
    bad_roles = set(role_overrides) - set(ROLES)
    if bad_roles:
        raise KeyError(
            f"unknown colour role(s): {', '.join(sorted(bad_roles))}. "
            f"Roles: {', '.join(sorted(ROLES))}."
        )
    style["roles"] = {**style["roles"], **{str(k): str(v) for k, v in role_overrides.items()}}

    theme = Theme(preset=preset, style=style, palette=palette, source=origin)
    for role in ROLES:                        # fail now, not mid-figure
        theme.colour(role)
    return theme.with_conditions(_conditions_from(source))


def _conditions_from(source: Any) -> dict[str, str]:
    """Condition colours carried alongside a theme, when the source has them.

    A theme loaded from a whole analysis configuration, or from a run folder's
    ``theme.json``, picks up that project's condition colours too, so a figure
    never has to be handed the two separately. Imported here rather than at
    module scope: ``analysis.conditions`` reads this module's palette.
    """
    from analysis.conditions import ConditionSet

    block: Any = None
    if isinstance(source, dict):
        block = source.get("conditions")
    elif isinstance(source, (str, Path)):
        candidate = Path(source)
        if candidate.suffix.lower() == ".json" and candidate.exists():
            block = json.loads(candidate.read_text(encoding="utf-8")).get("conditions")
    else:
        block = getattr(source, "conditions", None)
        if isinstance(block, ConditionSet):
            return block.colours()

    if not block:
        return {}
    return ConditionSet.from_config(block).colours()


# Keys that mark a mapping as a whole analysis configuration rather than a bare
# theme block. A configuration with no theme block means "the default", not
# "every one of these keys is a theme setting".
_CONFIG_MARKERS = {"movies", "frame_interval_min", "dataset", "enabled_modules"}


def _block_from_mapping(data: dict, origin: str) -> tuple[dict[str, Any], str]:
    if "theme" in data:
        return dict(data["theme"] or {}), origin
    if _CONFIG_MARKERS & set(data):
        return {}, f"{origin} (no theme block; house default)"
    return dict(data), origin


def _extract_block(source: Any) -> tuple[dict[str, Any], str]:
    if source is None:
        return {}, "default"
    if isinstance(source, str) and source not in ("",):
        candidate = Path(source)
        if candidate.suffix.lower() == ".json" and candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return _block_from_mapping(data, f"file:{candidate.name}")
        return {"preset": source}, f"preset:{source}"
    if isinstance(source, Path):
        data = json.loads(source.read_text(encoding="utf-8"))
        return _block_from_mapping(data, f"file:{source.name}")
    if isinstance(source, dict):
        return _block_from_mapping(source, "config")
    block = getattr(source, "theme", None)          # e.g. an AnalysisConfig
    if isinstance(block, dict):
        return dict(block), "config"
    if block is None:
        return {}, "default"
    return _extract_block(block)


# ----------------------------------------------------------------- conformance

def lab_contract() -> dict[str, Any] | None:
    """The lab house rcParams from ``analysis_kit``, or None if it is not installed."""
    try:
        from analysis_kit.style import rcparams
    except Exception:
        return None
    return rcparams("pyflash")


def conformance_report() -> dict[str, Any]:
    """Check the vendored ``pyflash`` contract still matches ``analysis_kit``.

    The vendored copy exists so an external user needs no extra install. That is
    also how it drifts, so this is worth running whenever either side changes.
    Keys this package adds deliberately (annotation size, transparency, the
    colour cycle) are not part of the shared contract and are not compared.
    """
    contract = lab_contract()
    if contract is None:
        return {"available": False, "reason": "analysis_kit is not importable here"}

    ours = Theme(preset="contract", style=_contract(), source="internal").rcparams()
    differences = {}
    for key, expected in contract.items():
        if key not in ours:
            differences[key] = {"contract": expected, "ours": "<missing>"}
            continue
        mine = ours[key]
        if key == "font.family":                    # we use the sans-serif stack
            continue
        if isinstance(expected, float) and isinstance(mine, (int, float)):
            differs = abs(float(mine) - expected) > 1e-9
        elif isinstance(expected, (list, tuple)) and isinstance(mine, (list, tuple)):
            differs = tuple(expected) != tuple(mine)
        else:
            differs = expected != mine
        if differs:
            differences[key] = {"contract": expected, "ours": mine}

    return {
        "available": True,
        "keys_compared": len(contract),
        "matches": len(contract) - len(differences),
        "differences": differences,
        "conformant": not differences,
    }


# --------------------------------------------------------------------- preview

def write_swatch(theme: Theme, path: str | Path) -> Path:
    """Draw the theme so a user can see what they have chosen before running.

    Roles down the left with their colour and their purpose, a specimen line
    plot, and the three colour maps. Everything on this card is drawn with the
    theme itself, so a broken theme produces a broken card rather than a lie.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    theme.apply()
    path = Path(path)
    figure = plt.figure(figsize=(15.0, 9.5))

    figure.text(0.03, 0.955, f"Theme: {theme.preset}", fontsize=22, fontweight="bold",
                color=theme.colour("ink"))
    figure.text(0.03, 0.915, f"from {theme.source}", fontsize=15, color=theme.colour("ink"))

    ax_roles = figure.add_axes([0.03, 0.06, 0.40, 0.83])
    ax_roles.set_axis_off()
    ordered = list(ROLES)
    for index, role in enumerate(ordered):
        y = 1.0 - (index + 0.5) / len(ordered)
        ax_roles.add_patch(
            plt.Rectangle((0.0, y - 0.021), 0.085, 0.042, transform=ax_roles.transAxes,
                          facecolor=theme.colour(role), edgecolor=theme.colour("ink"),
                          linewidth=0.8, clip_on=False)
        )
        ax_roles.text(0.105, y, role, transform=ax_roles.transAxes, va="center",
                      fontsize=15, fontweight="bold", color=theme.colour("ink"))
        ax_roles.text(0.44, y, ROLES[role], transform=ax_roles.transAxes, va="center",
                      fontsize=13, color=theme.colour("ink"))

    hours = np.linspace(0, 48, 200)
    ax_demo = figure.add_axes([0.55, 0.46, 0.41, 0.40])
    for offset, role in enumerate(("reporter", "morphology", "surveillance")):
        ax_demo.plot(hours, np.sin(2 * np.pi * hours / 24 + offset) + offset * 2.4,
                     color=theme.colour(role), linewidth=theme["line_width"], label=role)
    # A fit sits over its data, so it is drawn over the reporter trace and
    # offset just enough that both are visible on the card.
    ax_demo.plot(hours, 0.75 * np.sin(2 * np.pi * hours / 24), color=theme.colour("fit"),
                 linewidth=theme["line_width"], linestyle=(0, (6, 3)), label="fit")
    ax_demo.axhline(-2.0, color=theme.colour("reference"), linewidth=theme["spine_width"])
    ax_demo.set_xlabel("Hours")
    ax_demo.set_ylabel("Specimen")
    theme.legend(ax_demo, columns=2)
    theme.finish(ax_demo)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    for index, key in enumerate(("diverging_cmap", "sequential_cmap", "image_cmap")):
        ax_map = figure.add_axes([0.55, 0.30 - index * 0.10, 0.41, 0.055])
        ax_map.imshow(gradient, aspect="auto", cmap=theme[key])
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_ylabel(sentence_case(key.replace("_cmap", "")), fontsize=13, rotation=0,
                          ha="right", va="center", labelpad=12)

    figure.text(
        0.55, 0.045,
        f"type {theme['tick_size']:g}/{theme['axis_size']:g}/{theme['title_size']:g} pt  "
        f"(tick/axis/title)   strokes {theme['spine_width']:g}/{theme['line_width']:g}   "
        f"{theme['figure_size'][0]:g}x{theme['figure_size'][1]:g} in at {theme['save_dpi']} dpi",
        fontsize=13, color=theme.colour("ink"),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, **theme.savefig_kwargs())
    figure.savefig(path.with_suffix(".png"), dpi=160, transparent=False,
                   facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path
