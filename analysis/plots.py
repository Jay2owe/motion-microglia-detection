"""What to draw, declared as a list rather than assumed to be all of it.

A **request** names a figure and, optionally, a setting to vary. **Expansion**
turns the list of requests into a flat list of **items**, one per figure that
will actually be drawn, each with a unique name and a complete set of settings.

Fan-out is a key of its own rather than a shape::

    {"figure": "rhythm-strength", "for_each": {"metrics": "@circadian"}}
    {"figure": "rhythm-strength", "options":  {"metrics": "@circadian"}}

The first is one page per circadian measurement; the second is one page with all
of them on it. The alternative - which the PyFLASH plot spec this is modelled on
takes - is to give one key three meanings distinguished by how deeply its value
is nested, which cannot be read at a glance and cannot be typed reliably.

Expansion happens once, here, before any builder runs. That is what lets a
builder keep the signature it has: one run folder, one resolved set of options,
one page. A builder never learns that a plan exists.

Nothing at module scope imports the drawing half of the package. Parsing a plan
is part of reading a configuration and must stay possible in an environment with
no matplotlib; only ``expand``, which needs to know what each figure declares,
reaches for the catalogue - and by then matplotlib is already loaded.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.metric_groups import (COLUMN_SETTINGS, COLUMN_SET_SETTINGS,
                                    is_reference, resolve_metrics, resolve_one)

__all__ = ["PlotItem", "PlotRequest", "REQUEST_KEYS", "TEXT_SLOTS", "expand",
           "parse", "problems", "slugify"]

#: The five text slots, mirrored from ``figures/_text.py`` so this module does
#: not import the drawing half at module scope. ``analysis/test_plot_plan.py``
#: asserts the two lists still agree, so the mirror cannot go stale.
TEXT_SLOTS: tuple[str, ...] = ("title", "subtitle", "footnote", "note", "claim")

#: Keys a request may carry, beyond the text slots.
REQUEST_KEYS: tuple[str, ...] = ("figure", "for_each", "options", "as")

#: Cast names that mean "one value of this setting is a list". Read off the
#: cast because the cast is the only thing that knows: a figure may narrow a
#: shared option on its own spec, so ``metrics`` is a list on one page, a single
#: column on another and exactly two on a third.
_LIST_CASTS = frozenset({"commas", "numbers"})

#: Cast names that mean "one value of this setting is the value itself".
#: Written out rather than inferred, so a cast nobody recognises is refused
#: instead of quietly treated as scalar - which would turn a fanned ``metrics``
#: item into a bare string that a builder then iterates one character at a time.
_SCALAR_CASTS = frozenset({"str", "int", "float", "length",
                           "parse_hours_per_tick"})

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


def slugify(value: object) -> str:
    """A setting's value as a piece of a folder name.

    Lower-cased, with every run of characters outside ``[a-z0-9]`` replaced by
    one ``-``. The name is a path on disk and a row in a manifest, so it has to
    survive both, and the value it came from is still in the item's ``varied``
    for anything that needs it exactly.
    """
    if isinstance(value, (list, tuple)):
        return "+".join(slugify(item) for item in value)
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower())
    return text.strip("-") or "none"


@dataclass(frozen=True)
class PlotRequest:
    """One entry of the plan, exactly as written."""

    figure: str
    #: setting -> the values it takes, in declaration order.
    for_each: dict = field(default_factory=dict)
    #: setting -> one value, applied to every item this request produces.
    options: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)
    #: The ``as`` template, if one was given.
    naming: str | None = None
    #: Position in the plan, so a refusal can say which entry it is about.
    index: int = 0

    @property
    def where(self) -> str:
        return f"plots[{self.index}] ({self.figure})"

    @classmethod
    def from_dict(cls, raw_dict: object, index: int, groups: dict) -> "PlotRequest":
        """Parse one entry. Nothing here needs the figure catalogue.

        What can be checked without knowing which figure this is, is checked
        here, at configuration-load time. Everything that needs the figure's own
        declaration - whether it accepts this setting, whether it is list-valued
        - waits for ``expand``.
        """
        where = f"plots[{index}]"
        if not isinstance(raw_dict, dict):
            raise ValueError(
                f"{where}: expected a request block, not "
                f"{type(raw_dict).__name__}")
        figure = str(raw_dict.get("figure", "")).strip()
        if not figure:
            raise ValueError(
                f"{where} does not say which figure it draws; every request "
                "names one, and this package does not guess it")
        where = f"{where} ({figure})"
        stray = sorted(set(raw_dict) - set(REQUEST_KEYS) - set(TEXT_SLOTS))
        if stray:
            raise ValueError(
                f"{where}: {', '.join(repr(key) for key in stray)} is not part "
                f"of a request; it takes {', '.join(REQUEST_KEYS)}, plus the "
                f"text slots {', '.join(TEXT_SLOTS)}")

        for_each = _for_each(raw_dict.get("for_each"), where, groups)
        options = _options(raw_dict.get("options"), where, groups)
        shared = sorted(set(for_each) & set(options))
        if shared:
            raise ValueError(
                f"{where}: {', '.join(repr(key) for key in shared)} is in both "
                "for_each and options. One says a page per value and the other "
                "says one page with the value on it; naming it twice says both.")
        naming = raw_dict.get("as")
        if naming is not None and not str(naming).strip():
            raise ValueError(f"{where}: `as` is empty; leave it out to get the "
                             "default name")
        text = {slot: "" if raw_dict[slot] is None else str(raw_dict[slot])
                for slot in TEXT_SLOTS if slot in raw_dict}
        return cls(figure=figure, for_each=for_each, options=options, text=text,
                   naming=None if naming is None else str(naming), index=index)


def _for_each(block: object, where: str, groups: dict) -> dict:
    """The settings to vary, with every group reference expanded."""
    if block in (None, {}):
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"{where}: for_each must be a block of setting -> values, not "
            f"{type(block).__name__}")
    varied: dict[str, tuple] = {}
    for name, value in block.items():
        setting = str(name)
        if is_reference(value):
            if setting not in COLUMN_SETTINGS:
                raise ValueError(
                    f"{where}: for_each.{setting} is {value!r}, but a metric "
                    f"group is a set of measured columns and {setting} does not "
                    f"take one; groups may be named for "
                    f"{', '.join(COLUMN_SETTINGS)}")
            values = list(resolve_one(value, groups))
        elif isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    f"{where}: for_each.{setting} is empty, so this request "
                    "draws nothing. A request that expands to nothing is a "
                    "request that was meant to do something.")
            values = list(resolve_metrics(value, groups,
                                          where=f"{where} for_each.{setting}")) \
                if any(is_reference(item) for item in value) else list(value)
        else:
            raise ValueError(
                f"{where}: for_each.{setting} is a single value. for_each is "
                f"one figure per value, so it takes a list - write "
                f"[{value!r}] to fan out over one, or move it to options to put "
                "it on every figure this request draws.")
        varied[setting] = tuple(values)
    return varied


def _options(block: object, where: str, groups: dict) -> dict:
    """The settings shared by every item, with every group reference expanded."""
    if block in (None, {}):
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"{where}: options must be a block of setting -> value, not "
            f"{type(block).__name__}")
    resolved: dict[str, Any] = {}
    for name, value in block.items():
        setting = str(name)
        holds_group = is_reference(value) or (
            isinstance(value, (list, tuple))
            and any(is_reference(item) for item in value))
        if holds_group and setting not in COLUMN_SET_SETTINGS:
            raise ValueError(
                f"{where}: options.{setting} names a metric group, but a group "
                f"is a set of columns and {setting} does not hold one; a whole "
                f"group can be given to {', '.join(COLUMN_SET_SETTINGS)}. To "
                f"draw one figure per column of a group, put {setting} in "
                "for_each instead.")
        if holds_group:
            entries = [value] if is_reference(value) else list(value)
            resolved[setting] = list(resolve_metrics(
                entries, groups, where=f"{where} options.{setting}"))
        else:
            resolved[setting] = value
    return resolved


def parse(entries: object, groups: dict) -> list[PlotRequest]:
    """Parse the ``plots`` block. Empty, or absent, is the normal case."""
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"plots must be a list of requests, not {type(entries).__name__}")
    return [PlotRequest.from_dict(entry, index, groups)
            for index, entry in enumerate(entries)]


@dataclass(frozen=True)
class PlotItem:
    """One figure that will actually be drawn."""

    name: str
    figure: str
    #: Only the settings that vary across the request, as the builder will get
    #: them. Kept apart from ``options`` for the name and the plan manifest: it
    #: is what makes this drawing different from its siblings.
    varied: dict = field(default_factory=dict)
    #: Every setting this item is drawn with: the request's ``options`` with
    #: its own ``varied`` laid over the top.
    options: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)
    #: Position in the plan of the request this came from.
    request: int = 0

    def as_dict(self) -> dict:
        """The form written into the run folder and read back by a builder."""
        return {"name": self.name, "figure": self.figure,
                "varied": dict(self.varied), "options": dict(self.options),
                "text": dict(self.text)}

    @classmethod
    def from_dict(cls, raw_dict: dict) -> "PlotItem":
        return cls(name=str(raw_dict["name"]), figure=str(raw_dict["figure"]),
                   varied=dict(raw_dict.get("varied") or {}),
                   options=dict(raw_dict.get("options") or {}),
                   text=dict(raw_dict.get("text") or {}))

    def described(self) -> str:
        """The varied settings on one line, for a manifest column or a log."""
        return ";".join(f"{key}={slugify(value)}"
                        for key, value in self.varied.items())


def vocabulary() -> dict:
    """The shared option vocabulary and the universal names, at call time.

    Imported here rather than at the top of this file because reaching
    ``_options`` pulls in the theme and therefore matplotlib. A configuration
    with no plan never calls this, and by the time anything does, the figure
    catalogue has been loaded anyway.
    """
    import sys

    figures_dir = Path(__file__).resolve().parent / "figures"
    if str(figures_dir) not in sys.path:
        sys.path.insert(0, str(figures_dir))
    from _options import OPTIONS, SWITCHES
    from _schema import UNIVERSAL, UNIVERSAL_SWITCHES

    return {"options": OPTIONS, "switches": SWITCHES, "universal": UNIVERSAL,
            "universal_switches": UNIVERSAL_SWITCHES}


def _is_list_valued(spec, name: str, shared: dict) -> bool:
    """Whether one value of this setting is a list of one, or the value itself.

    Read off the cast, which is the only thing that knows. A figure may narrow a
    shared option on its own spec - one page takes a single metric where the
    shared vocabulary gives a list - so the figure's own cast wins over the
    shared one when it sets one.
    """
    declared = next((o for o in spec.options if o.name == name), None)
    cast = (declared.cast if declared is not None and declared.cast is not None
            else shared[name].cast)
    label = getattr(cast, "__name__", "")
    if label in _LIST_CASTS or label.startswith("exactly_"):
        return True
    if label in _SCALAR_CASTS:
        return False
    raise ValueError(
        f"{spec.slug}: cannot tell whether one value of {name!r} is a list, "
        f"because it is read by a cast called {label!r} that this module does "
        f"not recognise. Add it to _LIST_CASTS or _SCALAR_CASTS in "
        f"analysis/plots.py - guessing scalar would turn a fanned list setting "
        f"into a bare string a builder then reads one character at a time.")


def _name_for(request: PlotRequest, raw: dict, single: bool) -> str:
    """What one item is called: its ``as`` template, or the default."""
    if request.naming:
        wanted = set(_PLACEHOLDER.findall(request.naming))
        unknown = sorted(wanted - set(request.for_each))
        if unknown:
            varies = ", ".join(request.for_each) or "nothing"
            raise ValueError(
                f"{request.where}: `as` names {{{unknown[0]}}}, which this "
                f"request does not vary; it varies {varies}")
        name = request.naming
        for key, value in raw.items():
            name = name.replace("{" + key + "}", slugify(value))
        return name
    if single:
        return request.figure
    varied = ",".join(f"{key}={slugify(value)}" for key, value in raw.items())
    return f"{request.figure}/{varied}"


def expand_one(request: PlotRequest, spec, shared: dict) -> list[PlotItem]:
    """One request as the list of figures it asks for, in a predictable order.

    Several ``for_each`` keys take the cross product in declaration order with
    the **last** key varying fastest, because that order is the order the
    figures appear in on disk and in the manifest, and an order that depended on
    a dictionary would shuffle a plan for no reason anybody could see.
    """
    accepted = {o.name for o in spec.options} | set(shared["universal"])
    switches = set(shared["switches"]) | set(spec.switches)
    for source, names in (("for_each", request.for_each), ("options", request.options)):
        for name in names:
            if name in switches:
                raise ValueError(
                    f"{request.where}: {source}.{name} is a switch - it is on "
                    f"or off and takes no value, so there is nothing to "
                    f"{'vary' if source == 'for_each' else 'set'} here. Pass "
                    f"--{name.replace('_', '-')} when you draw the plan.")
            if name not in accepted:
                raise ValueError(
                    f"{request.where}: {source}.{name} is not a setting of this "
                    f"figure; it accepts "
                    f"{', '.join(sorted(accepted)) or 'none'}")
    if spec.run_level and "stem" in request.for_each:
        raise ValueError(
            f"{request.where}: {spec.slug} reads nothing that belongs to one "
            "movie, so a page per stem would be the same page several times "
            "under different names")

    keys = list(request.for_each)
    if not keys:
        name = _name_for(request, {}, single=True)
        return [PlotItem(name=name, figure=request.figure, varied={},
                         options=dict(request.options), text=dict(request.text),
                         request=request.index)]

    listed = {key: _is_list_valued(spec, key, shared["options"]) for key in keys}
    combinations = list(itertools.product(*(request.for_each[key] for key in keys)))
    items = []
    for combination in combinations:
        raw = dict(zip(keys, combination))
        varied = {
            key: ([value] if listed[key] and not isinstance(value, (list, tuple))
                  else list(value) if listed[key] else value)
            for key, value in raw.items()
        }
        items.append(PlotItem(
            name=_name_for(request, raw, single=len(combinations) == 1),
            figure=request.figure,
            varied=varied,
            options={**request.options, **varied},
            text=dict(request.text),
            request=request.index,
        ))
    return items


def _spec_for(request: PlotRequest, specs: dict, unconverted) -> Any:
    """This request's figure declaration, or a refusal that says which kind."""
    spec = specs.get(request.figure)
    if spec is not None:
        return spec
    stems = {name[:-3] if name.endswith(".py") else name for name in unconverted}
    if request.figure in stems or f"{request.figure}.py" in set(unconverted):
        raise ValueError(
            f"{request.where}: that builder is not on the figure schema, so it "
            "declares no options and nothing about this request can be checked. "
            "Run it directly instead, or put it on the schema first.")
    known = ", ".join(sorted(specs)) or "none"
    trailer = (f" {len(unconverted)} builder(s) are not on the schema and "
               f"declare nothing." if unconverted else "")
    raise ValueError(
        f"{request.where}: no figure declares this slug.{trailer} "
        f"Known slugs: {known}")


def _refuse_repeats(items: list[PlotItem]) -> None:
    """Two items with one name, named by the requests that produced them.

    The single most valuable check here. Without it the second item silently
    overwrites the first item's bundle, and the plan reports two figures drawn
    where one folder exists.
    """
    seen: dict[str, PlotItem] = {}
    for item in items:
        first = seen.setdefault(item.name, item)
        if first is not item:
            raise ValueError(
                f"plots[{first.request}] and plots[{item.request}] both draw an "
                f"item called {item.name!r}. The second would overwrite the "
                f"first's bundle. Give one of them an `as` name, or vary a "
                f"setting the two disagree about.")


def expand(requests: list[PlotRequest], specs: dict,
           unconverted=()) -> list[PlotItem]:
    """Every request as a flat list of items, in declaration order.

    ``specs`` is ``{slug: FigureSpec}``, passed in rather than imported so this
    module never pulls matplotlib into a run that only measures.
    """
    if not requests:
        return []
    shared = vocabulary()
    items: list[PlotItem] = []
    for request in requests:
        items.extend(expand_one(request, _spec_for(request, specs, unconverted),
                                shared))
    _refuse_repeats(items)
    return items


def problems(requests: list[PlotRequest], specs: dict,
             unconverted=()) -> list[str]:
    """Every reason a plan is not what its author meant, one line each.

    Request by request, so a plan with three faults reports three lines. The
    doctor prints every fault it can find in one pass and the first must not
    hide the rest.
    """
    if not requests:
        return []
    try:
        shared = vocabulary()
    except Exception as error:                    # noqa: BLE001 - reported, not raised
        return [f"the figure catalogue could not be read: {error}"]
    found: list[str] = []
    items: list[PlotItem] = []
    for request in requests:
        try:
            items.extend(expand_one(
                request, _spec_for(request, specs, unconverted), shared))
        except (TypeError, ValueError, KeyError) as error:
            found.append(str(error).strip("'"))
    try:
        _refuse_repeats(items)
    except ValueError as error:
        found.append(str(error))
    return found
