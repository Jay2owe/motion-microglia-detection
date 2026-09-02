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

Setting a slot to ``""`` removes that text from the figure entirely; leaving the
slot out keeps the default. ``null`` reads the same as ``""``.

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

from _options import OPTIONS, SWITCHES, skip_tokens

__all__ = ["FigureText", "OPTIONS_KEY", "figure_text", "SLOTS"]

#: The text a builder can be given. Anything else in a ``figures`` block is a
#: typo, and is reported rather than ignored.
SLOTS: tuple[str, ...] = ("title", "subtitle", "footnote", "note", "claim")

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


def _from_run(run: Path, slug: str) -> tuple[dict[str, str], dict[str, Any]]:
    """The wording and the options a run recorded for one figure.

    Two dicts rather than one because the two resolve against different
    vocabularies: a text slot is one of five names this module owns, an option
    is whatever that figure declared. Sharing a block but not a namespace is
    what lets a typo in either half still be refused by name - this function
    keeps refusing ``titel``, and ``_schema`` refuses ``bin`` against the
    figure that would have honoured ``bins``.
    """
    stored = Path(run) / "figures.json"
    if not stored.exists():
        return {}, {}
    data = json.loads(stored.read_text(encoding="utf-8"))
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
    return text, dict(options)


def figure_text(run: Path, slug: str, argv: list[str] | None = None, **defaults: str) -> FigureText:
    """The wording for one figure: flag, then configuration, then the default.

    Defaults are the builder's own and should describe the axes rather than
    interpret them. Nothing here inspects the data.
    """
    unknown = set(defaults) - set(SLOTS)
    if unknown:
        raise KeyError(
            f"{slug}: unknown text slot(s) {', '.join(sorted(unknown))}; "
            f"slots are {', '.join(SLOTS)}"
        )

    flags = _from_command_line(list(sys.argv[1:] if argv is None else argv))
    configured, _ = _from_run(Path(run), slug)

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
