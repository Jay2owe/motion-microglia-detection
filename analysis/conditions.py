"""Which experimental group each movie belongs to.

A *condition* is an experimental group: a treatment, a genotype, a light
schedule. It is the one piece of information this package cannot measure and
cannot guess, and it is also the one that, if it is wrong, makes every number
downstream wrong in a way no plot will reveal.

So there are exactly two ways a movie gets one, and both are written down:

**Declared.** The movie entry says ``"condition": "HCQ"``. Nothing is inferred.

**Derived from the name.** The configuration gives a regular expression per
condition, and the movie's stem is searched for it::

    "conditions": {"HCQ": "_A\\\\d", "vehicle": "_B\\\\d"}

A declared condition always wins over a derived one, so a single awkward file
is fixed by naming it rather than by contorting the pattern.

Three rules keep the derivation honest, and all three exist because the
alternative is silently mislabelled data:

1. **Two patterns matching one stem is an error, not a race.** There is no
   first-hit-wins ordering to remember. Fix the patterns.
2. **No pattern matching a stem leaves it unassigned**, and the run refuses to
   start. A movie quietly falling into a group it does not belong to is the
   failure this whole module exists to prevent.
3. **Nothing is matched until you have looked at it.** ``python -m analysis
   conditions --config <file>`` prints the assignment table without running
   anything.

Colours work the way :func:`analysis_kit.style.palette.declare_conditions`
does, and for the same reason: which colour a genotype should be is a decision
about the science, not about the house style, so a project names its own. The
override is confined to conditions - it is reached through
:meth:`analysis.theme.Theme.condition_colour` and never through
:meth:`~analysis.theme.Theme.colour` - so declaring a condition called ``ink``
recolours that condition and not the axes of every figure.

Crossed designs come free. Give each condition a ``factor`` and every factor is
resolved independently, then combined::

    "conditions": [
      {"name": "HCQ",     "factor": "treatment", "match": "_A\\\\d"},
      {"name": "vehicle", "factor": "treatment", "match": "_B\\\\d"},
      {"name": "young",   "factor": "age",       "match": "^3m"},
      {"name": "old",     "factor": "age",       "match": "^18m"}
    ]

A stem matching ``_A2`` and ``18m`` becomes ``HCQ_old``. Single-factor is the
default and needs no mention of factors at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from analysis.theme import BASE_COLOURS

__all__ = [
    "Condition",
    "ConditionSet",
    "Assignment",
    "resolve_colour",
]

#: The default factor name. A single-factor design never has to say it, and the
#: resulting column in every table is called ``condition``.
DEFAULT_FACTOR = "condition"

#: Auto-assigned colours, in order of declaration, for conditions nobody named
#: one for. Okabe-Ito, because this is the one case where the choice is
#: arbitrary and so should at least stay legible to a colourblind reader.
_AUTO_CYCLE: tuple[str, ...] = (
    "okabe_blue",
    "okabe_vermilion",
    "okabe_bluish_green",
    "okabe_reddish_purple",
    "okabe_orange",
    "okabe_sky_blue",
)

#: What an unassigned movie is called in a table. Never a group name, so a
#: pooled analysis cannot mistake it for one.
UNASSIGNED = "unassigned"

#: Joins names and labels of a crossed condition. Names use an underscore so
#: they stay safe in a filename or a column value; labels are only ever read.
_NAME_JOIN = "_"
_LABEL_JOIN = " / "


# ----------------------------------------------------------------- colours

def resolve_colour(value: Any, name: str) -> str:
    """One colour spec resolved to a value, for a condition.

    Accepts a package palette name (``"red"``, ``"okabe_blue"``), a
    ``#rrggbb`` literal, or any matplotlib colour name. Resolved once, here, so
    a typo surfaces when the configuration is read rather than halfway through
    drawing a figure.
    """
    if not isinstance(value, str):
        raise TypeError(f"condition {name!r} was given colour {value!r}, which is not a string")
    if value in BASE_COLOURS:
        return BASE_COLOURS[value]
    if value.startswith("#"):
        return value
    converted = _css_colour(value)
    if converted is not None:
        return converted
    raise KeyError(
        f"condition {name!r} was given colour {value!r}, which is neither a "
        f"package palette name ({', '.join(sorted(BASE_COLOURS))}), a #rrggbb "
        "value, nor a matplotlib colour name"
    )


def _css_colour(value: str) -> str | None:
    """A matplotlib colour name as hex, or None. Imported late so a machine
    without matplotlib can still read a configuration."""
    try:
        from matplotlib import colors as mcolors
    except Exception:
        return None
    try:
        return mcolors.to_hex(value)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------- condition

@dataclass(frozen=True)
class Condition:
    """One experimental group."""

    #: The token that appears in every table. Keep it filename-safe.
    name: str
    #: What a figure calls it. Defaults to the name.
    label: str = ""
    #: Which factor this is a level of. Levels of one factor are alternatives;
    #: levels of different factors are crossed.
    factor: str = DEFAULT_FACTOR
    #: Regular expressions. A stem matching **any** of them is this condition.
    match: tuple[str, ...] = ()
    #: The resolved colour, or None to auto-assign by declaration order.
    colour: str | None = None
    #: The reference group a later comparison should test the others against.
    control: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.name)
        # Compiling now means a bad pattern is a configuration error rather
        # than something that surfaces on the movie that happens to reach it.
        object.__setattr__(self, "_patterns", tuple(_compile(p, self.name) for p in self.match))

    @property
    def patterns(self) -> tuple[re.Pattern, ...]:
        return getattr(self, "_patterns")

    def matches(self, text: str) -> str | None:
        """The pattern that matched, or None. Returned rather than a bool so
        the audit can say *why* a movie landed where it did."""
        for pattern in self.patterns:
            if pattern.search(text):
                return pattern.pattern
        return None

    @classmethod
    def from_spec(cls, name: str, spec: Any) -> "Condition":
        """One condition from either shorthand form.

        ``"HCQ": "_A\\\\d"`` and the full mapping both arrive here.
        """
        if isinstance(spec, str):
            spec = {"match": spec}
        elif spec is None:
            spec = {}
        if not isinstance(spec, Mapping):
            raise TypeError(f"condition {name!r} must be a regex string or a mapping, not {spec!r}")

        raw_match = spec.get("match", spec.get("regex", ()))
        if isinstance(raw_match, str):
            raw_match = (raw_match,)
        patterns = tuple(str(p) for p in raw_match)

        colour = spec.get("colour", spec.get("color"))
        return cls(
            name=str(name),
            label=str(spec.get("label", "") or ""),
            factor=str(spec.get("factor", DEFAULT_FACTOR)),
            match=patterns,
            colour=resolve_colour(colour, str(name)) if colour is not None else None,
            control=bool(spec.get("control", False)),
            notes=str(spec.get("notes", "")),
        )


def _compile(pattern: str, name: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as error:
        raise ValueError(
            f"condition {name!r} has an invalid regular expression {pattern!r}: {error}"
        ) from error


# -------------------------------------------------------------- assignment

@dataclass(frozen=True)
class Assignment:
    """Where one movie landed, and how it got there."""

    stem: str
    #: The combined condition name across every factor, or ``unassigned``.
    condition: str
    label: str
    #: Level per factor. A factor with no match holds ``None``.
    factors: dict[str, str | None] = field(default_factory=dict)
    #: ``declared`` | ``derived`` | ``unassigned`` | ``ambiguous``
    source: str = "unassigned"
    #: Which pattern matched, per factor. Empty for a declared assignment.
    evidence: dict[str, str] = field(default_factory=dict)
    #: Why this is not usable, or None. The run refuses to start on any of these.
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.problem is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "condition": self.condition,
            "label": self.label,
            "factors": dict(self.factors),
            "source": self.source,
            "evidence": dict(self.evidence),
            "problem": self.problem,
        }


# ------------------------------------------------------------ condition set

@dataclass(frozen=True)
class ConditionSet:
    """Every declared condition, and the rules for assigning movies to them."""

    conditions: tuple[Condition, ...] = ()
    #: Which text a pattern is searched in: ``stem``, ``labels`` or ``raw``.
    #: The last two are the file *names*, not their full paths.
    match_against: str = "stem"
    #: These groups were invented to exercise the machinery, not observed. The
    #: flag travels into ``conditions.json`` and the manifest and is printed by
    #: every command that touches the design, because a fabricated group that
    #: reads as a real one is precisely the failure this module exists to stop.
    synthetic: bool = False

    # ------------------------------------------------------------- reading

    @classmethod
    def from_config(cls, data: Any) -> "ConditionSet":
        """Read the ``conditions`` block. Absent or empty means no conditions."""
        if data in (None, {}, []):
            return cls()

        match_against = "stem"
        synthetic = False
        if isinstance(data, Mapping) and "conditions" in data:
            match_against = str(data.get("match_against", data.get("derive_from", "stem")))
            synthetic = bool(data.get("synthetic", False))
            data = data["conditions"]

        entries: list[Condition] = []
        if isinstance(data, Mapping):
            for name, spec in data.items():
                if str(name).startswith("_"):        # a JSON comment key
                    continue
                entries.append(Condition.from_spec(str(name), spec))
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            for spec in data:
                if not isinstance(spec, Mapping) or "name" not in spec:
                    raise TypeError(
                        "each entry in a conditions list needs a 'name'; "
                        f"got {spec!r}"
                    )
                entries.append(Condition.from_spec(str(spec["name"]), spec))
        else:
            raise TypeError(
                "conditions must be a mapping of name to regex, or a list of "
                f"condition entries; got {type(data).__name__}"
            )

        if match_against not in ("stem", "labels", "raw"):
            raise ValueError(
                f"match_against must be stem, labels or raw; got {match_against!r}"
            )
        return cls(conditions=tuple(entries), match_against=match_against,
                   synthetic=synthetic)

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for entry in self.conditions:
            if entry.name in seen:
                raise ValueError(
                    f"condition {entry.name!r} is declared twice; names must be unique "
                    "because they are what appears in every table"
                )
            seen[entry.name] = entry.factor
        for factor, levels in self.by_factor().items():
            controls = [c.name for c in levels if c.control]
            if len(controls) > 1:
                raise ValueError(
                    f"factor {factor!r} names {len(controls)} controls ({', '.join(controls)}); "
                    "a comparison has one reference group"
                )

    # ------------------------------------------------------------ querying

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def __iter__(self) -> Iterable[Condition]:
        return iter(self.conditions)

    def __len__(self) -> int:
        return len(self.conditions)

    def factors(self) -> tuple[str, ...]:
        """Factor names in declaration order."""
        order: list[str] = []
        for entry in self.conditions:
            if entry.factor not in order:
                order.append(entry.factor)
        return tuple(order)

    def by_factor(self) -> dict[str, tuple[Condition, ...]]:
        return {
            factor: tuple(c for c in self.conditions if c.factor == factor)
            for factor in self.factors()
        }

    def get(self, name: str) -> Condition | None:
        for entry in self.conditions:
            if entry.name == name:
                return entry
        return None

    def control(self, factor: str | None = None) -> Condition | None:
        """The reference group, for a comparison that needs one."""
        factor = factor or (self.factors()[0] if self.factors() else None)
        for entry in self.conditions:
            if entry.control and (factor is None or entry.factor == factor):
                return entry
        return None

    def colours(self) -> dict[str, str]:
        """Condition name to colour, auto-assigning where none was named.

        Auto-assignment is by declaration order, so adding a condition at the
        end never recolours the ones before it - a figure redrawn after a new
        group joins the experiment keeps the colours the old groups had.
        """
        table: dict[str, str] = {}
        auto_index = 0
        for entry in self.conditions:
            if entry.colour is not None:
                table[entry.name] = entry.colour
                continue
            if auto_index >= len(_AUTO_CYCLE):
                raise ValueError(
                    f"{len(self.conditions)} conditions need colours and only "
                    f"{len(_AUTO_CYCLE)} can be auto-assigned before two would look "
                    "alike; give the later ones a colour of their own"
                )
            table[entry.name] = BASE_COLOURS[_AUTO_CYCLE[auto_index]]
            auto_index += 1
        return table

    def labels(self) -> dict[str, str]:
        return {entry.name: entry.label for entry in self.conditions}

    # ----------------------------------------------------------- assigning

    def assign(self, text: str, declared: str | None = None, stem: str | None = None) -> Assignment:
        """Where one movie belongs.

        *text* is what the patterns are searched in; *declared* is the
        movie's own ``condition`` entry, which always wins.
        """
        stem = stem if stem is not None else text

        if declared:
            return self._declared(stem, declared)
        if not self.conditions:
            # An experiment with no groups is a legitimate experiment. This is
            # only a problem once conditions exist and a movie falls outside
            # them - which is what ``_derived`` reports.
            return Assignment(
                stem=stem, condition=UNASSIGNED, label=UNASSIGNED,
                factors={DEFAULT_FACTOR: None}, source="unassigned",
            )
        return self._derived(stem, text)

    def _declared(self, stem: str, declared: str) -> Assignment:
        if not self.conditions:
            # Backwards compatible: a bare condition string with nothing to
            # check it against is taken at its word.
            return Assignment(
                stem=stem, condition=declared, label=declared,
                factors={DEFAULT_FACTOR: declared}, source="declared",
            )
        parts = [p for p in declared.split(_NAME_JOIN) if p]
        found = [self.get(p) for p in parts]
        if any(entry is None for entry in found):
            unknown = [p for p, entry in zip(parts, found) if entry is None]
            return Assignment(
                stem=stem, condition=declared, label=declared, source="declared",
                problem=(
                    f"condition {', '.join(unknown)} is not declared; "
                    f"known conditions are {', '.join(c.name for c in self.conditions)}"
                ),
            )
        entries = [entry for entry in found if entry is not None]
        return Assignment(
            stem=stem,
            condition=_NAME_JOIN.join(e.name for e in entries),
            label=_LABEL_JOIN.join(e.label for e in entries),
            factors={e.factor: e.name for e in entries},
            source="declared",
        )

    def _derived(self, stem: str, text: str) -> Assignment:
        factors: dict[str, str | None] = {}
        evidence: dict[str, str] = {}
        problems: list[str] = []
        ambiguous = False

        for factor, levels in self.by_factor().items():
            hits = [(entry, entry.matches(text)) for entry in levels]
            hits = [(entry, pattern) for entry, pattern in hits if pattern is not None]
            if len(hits) == 1:
                entry, pattern = hits[0]
                factors[factor] = entry.name
                evidence[factor] = pattern
            elif not hits:
                factors[factor] = None
                patterns = ", ".join(
                    f"{e.name}=/{p}/" for e in levels for p in e.match
                ) or "no patterns declared"
                problems.append(f"nothing matched {text!r} for factor {factor!r} ({patterns})")
            else:
                factors[factor] = None
                ambiguous = True
                named = ", ".join(f"{e.name} by /{p}/" for e, p in hits)
                problems.append(
                    f"{text!r} matched {len(hits)} conditions in factor {factor!r} "
                    f"({named}); patterns must not overlap"
                )

        assigned = [factors[f] for f in self.factors()]
        if any(level is None for level in assigned):
            source = "ambiguous" if ambiguous else "unassigned"
            return Assignment(
                stem=stem, condition=UNASSIGNED, label=UNASSIGNED, factors=factors,
                source=source, evidence=evidence, problem="; ".join(problems),
            )

        entries = [self.get(name) for name in assigned]
        return Assignment(
            stem=stem,
            condition=_NAME_JOIN.join(assigned),          # type: ignore[arg-type]
            label=_LABEL_JOIN.join(e.label for e in entries if e is not None),
            factors=factors,
            source="derived",
            evidence=evidence,
        )

    def text_for(self, movie: Any) -> str:
        """The string this set searches for one movie, per ``match_against``."""
        if self.match_against == "stem":
            return str(getattr(movie, "stem", movie))
        path = getattr(movie, self.match_against, None)
        if path is None:
            raise ValueError(
                f"match_against is {self.match_against!r} but movie "
                f"{getattr(movie, 'stem', movie)!r} has no such input"
            )
        from pathlib import Path

        return Path(str(path)).name

    # -------------------------------------------------------------- audit

    def audit(self, assignments: Sequence[Assignment]) -> list[str]:
        """Warnings a run should print but need not stop for.

        Separate from ``problem`` on purpose. A problem means a movie is in the
        wrong place and the run must not start. A warning means the design is
        thin - one group, one animal - which is the user's call to make, not
        this package's.
        """
        notes: list[str] = []
        usable = [a for a in assignments if a.ok and a.condition != UNASSIGNED]

        if self.synthetic:
            notes.append(
                "SYNTHETIC DESIGN - these groups were invented to exercise the "
                "pipeline. No number split by them means anything."
            )

        counted: dict[str, int] = {}
        for assignment in usable:
            counted[assignment.condition] = counted.get(assignment.condition, 0) + 1

        for entry in self.conditions:
            if not entry.match and not any(
                entry.name in (a.factors or {}).values() for a in usable
            ):
                notes.append(f"condition {entry.name!r} has no pattern and no movie declares it")

        for name, count in sorted(counted.items()):
            if count == 1:
                notes.append(f"condition {name!r} has one movie; nothing can be compared within it")

        if len(counted) == 1 and self.conditions:
            notes.append(
                f"every movie is {next(iter(counted))!r}; a comparison needs at least two groups"
            )

        if self.conditions and not self.control():
            notes.append(
                "no condition is marked as the control; a comparison will have no reference group"
            )
        return notes

    def as_dict(self) -> dict[str, Any]:
        """What goes in the run manifest and ``conditions.json``."""
        colours = self.colours() if self.conditions else {}
        return {
            "match_against": self.match_against,
            "synthetic": self.synthetic,
            "factors": list(self.factors()),
            "conditions": [
                {
                    "name": entry.name,
                    "label": entry.label,
                    "factor": entry.factor,
                    "match": list(entry.match),
                    "colour": colours.get(entry.name),
                    "colour_source": "declared" if entry.colour else "auto",
                    "control": entry.control,
                    "notes": entry.notes,
                }
                for entry in self.conditions
            ],
        }
