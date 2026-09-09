"""Every word on a figure, and where it comes from.

The package never decides what a figure means. A builder ships a plain
descriptive default - what is on the axes, nothing more - and every piece of
text is a parameter the user can replace.

Three places, most specific first:

1. **A command-line flag**, for a one-off::

       python analysis/figures/03_surveillance_not_translocation.py <run> \\
           --title "Microglia rebuild themselves without going anywhere"

2. **A ``figures`` block** in the analysis configuration, keyed by the figure's
   slug, which is the route that survives a rebuild::

       "figures": {
         "surveillance-not-translocation": {
           "title": "Microglia rebuild themselves without going anywhere",
           "claim": "The median cell moves 0.72 px per 30 min while replacing 18% ..."
         }
       }

3. **The builder's default**, which describes and does not conclude.

Builder-generated subtitles, footnotes and notes are retained in the bundle's
audit text but are not drawn on the canvas by default. Supply one of those
slots through the configuration or a command-line flag to draw it deliberately.
Setting a slot to ``""`` removes it entirely; ``null`` reads the same as ``""``.

A figure's *options* - what it draws rather than what is written beside it -
resolve the same three ways and live in an ``options`` sub-block of the same
entry. They are read by ``_schema``, not here; this module owns the five text
slots and hands the rest of the block over untouched.

A title is raised to sentence case whichever of the three it came from, so a
figure never opens on a lower-case letter. Nothing else about the wording is
touched, and a first word that carries its own capitals - pH, mCherry - is left
exactly as written.

``run`` copies the block into the run folder as ``figures.json``, so rebuilding
a run's figures later reproduces the wording that was used at the time rather
than whatever the configuration says now.

``claim`` is the one-sentence statement that goes in the bundle README and the
figure register. It has no default: an unset claim leaves the README saying the
figure has not been interpreted, which is accurate.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.theme import sentence_case

from _options import OPTIONS, SELECTION, SWITCHES, skip_tokens

__all__ = ["FigureText", "OPTIONS_KEY", "PLAN_KEY", "figure_text",
           "item_settings", "plan_items", "SLOTS"]

#: Where a run records the plan it was asked to draw, expanded into one entry
#: per figure. Written by ``analysis.run`` into ``figures.json`` beside the
#: per-figure blocks; see :mod:`analysis.plots`.
PLAN_KEY = "plots"

#: The text a builder can be given. Anything else in a ``figures`` block is a
#: typo, and is reported rather than ignored.
SLOTS: tuple[str, ...] = ("title", "subtitle", "footnote", "note", "claim")

#: Explanatory prose belongs in the audit bundle unless somebody deliberately
#: asks to put it on the canvas. Titles remain visible by default; claims were
#: never canvas text.
DETAIL_SLOTS: tuple[str, ...] = ("subtitle", "footnote", "note")

#: The one nested key a ``figures.<slug>`` block may carry beside the slots.
#: Everything else at that level is wording, which keeps the existing shape
#: working untouched and leaves exactly one place for a figure's settings.
OPTIONS_KEY = "options"


@dataclass(frozen=True)
class FigureText:
    """The resolved wording for one figure."""

    slug: str
    title: str = ""
    subtitle: str = ""
    footnote: str = ""
    #: A second free-text block some layouts place beside the axes.
    note: str = ""
    #: What the figure is claimed to show. Empty unless the user wrote one.
    claim: str = ""
    #: Where each slot came from: ``flag``, ``config`` or ``default``.
    source: dict[str, str] | None = None

    def __getitem__(self, slot: str) -> str:
        if slot not in SLOTS:
            raise KeyError(f"unknown text slot {slot!r}; slots are {', '.join(SLOTS)}")
        return getattr(self, slot)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            **{slot: getattr(self, slot) for slot in SLOTS},
            "source": dict(self.source or {}),
        }

    def on_canvas(self) -> "FigureText":
        """Return the wording deliberately selected for the visible figure.

        Builder defaults still travel with the bundle and remain available to
        the README. Only a flag or configuration entry promotes explanatory
        text onto the canvas, keeping the project-wide visual default to the
        title, axis labels and plot-native statistical annotations.
        """
        origin = self.source or {}
        values = {slot: getattr(self, slot) for slot in SLOTS}
        for slot in DETAIL_SLOTS:
            if origin.get(slot, "default") == "default":
                values[slot] = ""
        return FigureText(slug=self.slug, source=dict(origin), **values)


def _from_command_line(argv: list[str]) -> dict[str, str]:
    """``--title "..."`` and friends, in whatever order they appear."""
    found: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            name, equals, inline = token[2:].partition("=")
            name = name.replace("-", "_")
            if name in OPTIONS or name in SWITCHES:
                # Somebody else's option. Skip it, and its value if it takes one;
                # `_options.option` and `_options.switch` are what read them.
                index += skip_tokens(token)
                continue
            if name not in SLOTS:
                raise SystemExit(
                    f"unknown option --{token[2:]}; figure text options are "
                    + ", ".join(f"--{slot}" for slot in SLOTS)
                    + "; builder options are "
                    + ", ".join(f"--{name.replace('_', '-')}"
                                for name in sorted({**OPTIONS, **SWITCHES}))
                )
            # `--footnote=` is an empty value, not a missing one: it is how a
            # line is taken off the figure for a single build.
            if equals:
                found[name] = inline
                index += 1
                continue
            if index + 1 >= len(argv):
                raise SystemExit(f"--{name} needs a value")
            found[name] = argv[index + 1]
            index += 2
            continue
        index += 1
    return found


def plan_items(run: Path) -> list[dict]:
    """The expanded plot plan this run carries, in the order it was written.

    Two places, and the second wins. ``figures.json`` holds the plan the run was
    measured with, which is the record. ``figures/plan.json`` holds the plan a
    ``python -m analysis plots --plan <file>`` invocation drew instead, written
    beside the bundles it produced so that a bundle the run's own configuration
    does not mention still says where it came from.
    """
    found: list[dict] = []
    for stored in (Path(run) / "figures.json", Path(run) / "figures" / "plan.json"):
        if not stored.exists():
            continue
        data = json.loads(stored.read_text(encoding="utf-8"))
        entries = data.get(PLAN_KEY) or []
        if isinstance(entries, list):
            found = [entry for entry in entries if isinstance(entry, dict)] or found
    return found


def item_settings(run: Path, slug: str, item: str) -> dict[str, Any]:
    """One item of the run's plan, refusing a name the run does not have.

    The message lists the items the run *does* have, because this is the one a
    user meets most often while learning the feature: an item name is long, it
    is typed from a manifest, and a plan of forty is not something anybody
    remembers.
    """
    entries = plan_items(run)
    for entry in entries:
        if str(entry.get("name")) != item:
            continue
        figure = str(entry.get("figure", slug))
        if figure != slug:
            raise SystemExit(
                f"--item {item!r} is an item of {figure}, not of {slug}. An "
                "item names its own figure, so drawing it with another one "
                "would put one page's settings on a different page."
            )
        return dict(entry.get("options") or {})
    known = "\n  ".join(str(entry.get("name")) for entry in entries)
    raise SystemExit(
        f"--item {item!r} is not in this run's plot plan.\n"
        + (f"This run draws:\n  {known}" if known else
           "This run has no plot plan; add a `plots` block to the analysis "
           "configuration, or pass --plan to `python -m analysis plots`.")
    )


def _item_text(run: Path, slug: str, item: str) -> dict[str, str]:
    """The wording one item of the plan carries, if it carries any."""
    for entry in plan_items(run):
        if str(entry.get("name")) == item and str(entry.get("figure", slug)) == slug:
            block = entry.get("text") or {}
            return {key: "" if value is None else str(value)
                    for key, value in block.items() if key in SLOTS}
    return {}


def _from_run(run: Path, slug: str,
              item: str | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """The wording and the options a run recorded for one drawing of a figure.

    Two dicts rather than one because the two resolve against different
    vocabularies: a text slot is one of five names this module owns, an option
    is whatever that figure declared. Sharing a block but not a namespace is
    what lets a typo in either half still be refused by name - this function
    keeps refusing ``titel``, and ``_schema`` refuses ``bin`` against the
    figure that would have honoured ``bins``.

    Three layers, innermost last: the builder's own defaults (which this
    function never sees), the ``figures.<slug>`` block, and - when an item of
    the run's plot plan is named - that item's own settings. A plan may draw one
    figure many ways, so the per-slug block is what every drawing of it shares
    and the item is what makes this drawing different.

    An item's ``stem``, ``panels`` and ``item`` are held back. They say which
    figure is being drawn rather than what it draws, are read before a context
    exists, and no figure declares them - so letting them through here would
    have ``_schema`` refuse the plan's own settings as options this figure does
    not accept.
    """
    stored = Path(run) / "figures.json"
    if not stored.exists() and item is None:
        return {}, {}
    data = (json.loads(stored.read_text(encoding="utf-8"))
            if stored.exists() else {})
    block = dict(data.get("figures", data).get(slug) or {})
    options = block.pop(OPTIONS_KEY, {}) or {}
    if not isinstance(options, dict):
        raise SystemExit(
            f"figures.{slug}.{OPTIONS_KEY} must be a block of option names, "
            f"not {type(options).__name__}"
        )
    unknown = set(block) - set(SLOTS)
    if unknown:
        raise SystemExit(
            f"figures.{slug} has unknown key(s) {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(SLOTS)}, {OPTIONS_KEY}"
        )
    # An explicit key wins, empty string included: writing "footnote": "" is how
    # a user takes a line off a figure. Only an absent key falls back to the
    # builder's default, so omit the slot rather than emptying it to keep one.
    text = {key: "" if value is None else str(value) for key, value in block.items()}
    if item is not None:
        options = {**options, **{
            name: value
            for name, value in item_settings(run, slug, item).items()
            if name not in SELECTION
        }}
        text = {**text, **_item_text(run, slug, item)}
    return text, dict(options)


def figure_text(run: Path, slug: str, argv: list[str] | None = None,
                item: str | None = None, **defaults: str) -> FigureText:
    """The wording for one figure: flag, then configuration, then the default.

    Defaults are the builder's own and should describe the axes rather than
    interpret them. Nothing here inspects the data.

    When an item of the run's plot plan is named, its wording sits between the
    flag and the ``figures`` block: a plan that draws one figure six ways can
    put a different footnote on each of them without six configuration blocks.
    """
    unknown = set(defaults) - set(SLOTS)
    if unknown:
        raise KeyError(
            f"{slug}: unknown text slot(s) {', '.join(sorted(unknown))}; "
            f"slots are {', '.join(SLOTS)}"
        )

    flags = _from_command_line(list(sys.argv[1:] if argv is None else argv))
    configured, _ = _from_run(Path(run), slug, item)

    resolved: dict[str, str] = {}
    source: dict[str, str] = {}
    for slot in SLOTS:
        if slot in flags:
            resolved[slot], source[slot] = flags[slot], "flag"
        elif slot in configured:
            resolved[slot], source[slot] = configured[slot], "config"
        else:
            resolved[slot], source[slot] = defaults.get(slot, ""), "default"
    # A title starts with a capital wherever it came from. This is the only
    # thing done to a user's words, it changes one character, and it leaves a
    # name that carries its own capitals alone; see ``theme.sentence_case``.
    resolved["title"] = sentence_case(resolved["title"])
    return FigureText(slug=slug, source=source, **resolved)
