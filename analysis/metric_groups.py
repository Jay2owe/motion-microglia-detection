"""Named sets of measured columns, declared once and used by name.

A figure and a contrast both take a list of columns, and both are usually given
the same list. Writing it twice is how the two drift apart; writing it by
substring match - "everything with ``cosinor`` in the name" - is how a renamed
column silently leaves the set with nothing failing. A group is a name, an
explicit membership, and a check that every member is a column some module says
it writes.

Two ways to declare one::

    "metric_groups": {
      "circadian": ["cosinor_amplitude", "relative_amplitude", "m10", "l5"],
      "drift":     {"module": "trend"}
    }

The list is taken literally. The selector block reads what the modules already
declare - ``role``, ``module``, or both, which are ANDed - so a group defined
that way follows the code rather than a list somebody has to remember to update.

Anywhere a list of columns is expected, ``"@circadian"`` stands for the group's
members. That is worth more than it sounds: ``metrics`` is a declared option on
26 of the 38 figures and on every contrast, so one group written here works
everywhere with no per-figure code.

One level only. A group may not be built from another group: nesting is a small
feature and a large class of cycle bugs, and the message says so rather than
letting the reference through as a column name nothing writes.

This module never imports anything under ``analysis/figures/``. A run that only
measures must stay possible in an environment with no matplotlib.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from analysis.registry import Column, declared_columns, list_derived, list_modules

__all__ = ["COLUMN_SETTINGS", "COLUMN_SET_SETTINGS", "MetricGroup", "PREFIX",
           "SELECTORS", "build", "group_name", "is_reference", "resolve_metrics",
           "resolve_one", "resolve_setting"]

#: A value that names a group rather than a column.
PREFIX = "@"

#: Builder options whose values are column names. Fanning a plan out over a
#: group is meaningful for all of these: one figure per column.
COLUMN_SETTINGS: tuple[str, ...] = ("metrics", "fit", "size")

#: The subset that holds a *set* of columns rather than one, so a whole group
#: can be dropped in as the value. ``size`` is missing on purpose: it sets point
#: size from one column, and a set of columns is not a thing it could draw.
COLUMN_SET_SETTINGS: tuple[str, ...] = ("metrics", "fit")

#: Selector keys a group may be defined by. Both are facts a module already
#: declares.
SELECTORS: tuple[str, ...] = ("role", "module")

#: Selector keys that read as though they should work and do not, with the
#: reason. Refused by name rather than as "unknown key", because the reason is
#: a fact about how the package declares things and a user cannot infer it.
REFUSED_SELECTORS: dict[str, str] = {
    "table": ("a module may write several tables and a column declaration does "
              "not say which one it lands in, so a group by table would be this "
              "package guessing"),
}

#: A group name is written into a configuration as ``"@circadian"`` and read
#: back out of one, so it is held to the same spelling as a window and a
#: contrast name.
_GROUP_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class MetricGroup:
    """One named set of columns, as declared and as resolved.

    Both halves are kept. The declaration is what the author wrote and is what
    a diff of the configuration shows; the resolved columns are what was
    actually drawn, and are what a run records - a selector group means
    something different after a module gains a column, and the run folder has
    to be able to say which meaning was used on the day.
    """

    name: str
    #: The declaration exactly as written, kept for the record.
    declared: Any
    #: The columns it resolved to, in a stable order.
    columns: tuple[str, ...]

    def as_list(self) -> list[str]:
        return list(self.columns)


def is_reference(value: object) -> bool:
    """Whether this value names a group rather than a column."""
    return isinstance(value, str) and value.startswith(PREFIX) and len(value) > 1


def group_name(value: str) -> str:
    return value[len(PREFIX):]


def _registered() -> tuple[dict[str, Column], dict[str, set[str]]]:
    """Every declared column, and which module(s) declared each one.

    ``declared_columns`` only knows about modules that have been imported, and
    importing them is what registers them. Done here rather than at the top of
    this file so that reading a configuration with no ``metric_groups`` block
    never pays for it.
    """
    import analysis.modules  # noqa: F401  - importing registers every module

    columns = declared_columns()
    owners: dict[str, set[str]] = {}
    for module in (*list_modules(), *list_derived()):
        for column in module.produces:
            owners.setdefault(column.name, set()).add(module.name)
    return columns, owners


def _nearest(name: str, known) -> str:
    close = difflib.get_close_matches(name, list(known), n=3, cutoff=0.6)
    return f"; did you mean {', '.join(close)}?" if close else ""


def _from_list(name: str, entries, columns: dict[str, Column]) -> tuple[str, ...]:
    """A list declaration, taken literally and checked member by member."""
    if not entries:
        raise ValueError(
            f"metric group {name!r} is empty. An empty group draws a blank "
            "figure and says nothing about why, so it is refused here instead."
        )
    members: list[str] = []
    for entry in entries:
        if is_reference(entry):
            raise ValueError(
                f"metric group {name!r} names {entry!r}: a group may not be "
                "built out of another group. One level only - write the columns "
                "out, or select them."
            )
        if not isinstance(entry, str):
            raise ValueError(
                f"metric group {name!r}: expected column names, not "
                f"{type(entry).__name__}"
            )
        if entry not in columns:
            raise ValueError(
                f"metric group {name!r} names {entry!r}, which no module says "
                f"it writes{_nearest(entry, columns)}"
            )
        if entry not in members:
            members.append(entry)
    return tuple(members)


def _from_selector(name: str, block: dict, columns: dict[str, Column],
                   owners: dict[str, set[str]]) -> tuple[str, ...]:
    """A selector declaration, resolved against what the modules declare."""
    for key in sorted(set(block) - set(SELECTORS)):
        reason = REFUSED_SELECTORS.get(key)
        if reason is not None:
            raise ValueError(
                f"metric group {name!r}: {key!r} is not a selector this package "
                f"offers, because {reason}"
            )
        raise ValueError(
            f"metric group {name!r}: {key!r} is not a selector; a group selects "
            f"by {' or '.join(SELECTORS)}"
        )
    role = block.get("role")
    module = block.get("module")
    if role is None and module is None:
        raise ValueError(
            f"metric group {name!r}: a selector block that selects nothing "
            f"would be every column ever measured; say "
            f"{' or '.join(SELECTORS)}"
        )
    members = sorted(
        column for column, declaration in columns.items()
        if (role is None or declaration.role == str(role))
        and (module is None or str(module) in owners.get(column, ()))
    )
    if not members:
        roles = sorted({declaration.role for declaration in columns.values()})
        modules = sorted({owner for names in owners.values() for owner in names})
        asked = ", ".join(f"{key}={block[key]!r}" for key in SELECTORS
                          if key in block)
        raise ValueError(
            f"metric group {name!r}: {asked} matches no declared column. An "
            f"empty group draws a blank figure and says nothing about why.\n"
            f"  declared roles: {', '.join(roles) or 'none'}\n"
            f"  declared modules: {', '.join(modules) or 'none'}"
        )
    return tuple(members)


def build(block: object) -> dict[str, MetricGroup]:
    """Parse the ``metric_groups`` block, resolving every group's membership.

    Membership is settled here, at configuration-load time, so a group naming a
    column no module writes is refused before anything is measured rather than
    discovered as a figure with nothing on it.
    """
    if not block:
        return {}
    if not isinstance(block, dict):
        raise TypeError(
            f"metric_groups must be a block of name -> members, not "
            f"{type(block).__name__}"
        )
    columns, owners = _registered()
    groups: dict[str, MetricGroup] = {}
    for name, declared in block.items():
        if not isinstance(name, str) or not _GROUP_NAME.fullmatch(name):
            raise ValueError(
                f"metric group name {name!r} is not a plain lower-case "
                f"identifier; it is written elsewhere as \"{PREFIX}circadian\", "
                "so it has to look like `circadian`"
            )
        if isinstance(declared, dict):
            members = _from_selector(name, declared, columns, owners)
        elif isinstance(declared, (list, tuple)):
            members = _from_list(name, declared, columns)
        else:
            raise ValueError(
                f"metric group {name!r}: expected a list of column names or a "
                f"selector block ({', '.join(SELECTORS)}), not "
                f"{type(declared).__name__}"
            )
        groups[name] = MetricGroup(
            name=name,
            declared=list(declared) if isinstance(declared, (list, tuple))
            else dict(declared),
            columns=members,
        )
    return groups


def resolve_one(value: object, groups: dict[str, MetricGroup]) -> list[str]:
    """One entry of a metrics list, expanded if it names a group.

    A plain column name comes back unchanged and in a list of one, so a caller
    can flatten a mixed list without asking which kind each entry was.
    """
    if not is_reference(value):
        return [str(value)]
    name = group_name(str(value))
    if name not in groups:
        known = ", ".join(sorted(groups)) or "none declared"
        raise ValueError(
            f"{value!r} names no metric group; declared groups: {known}")
    return groups[name].as_list()


def resolve_metrics(values, groups: dict[str, MetricGroup], *,
                    where: str) -> tuple[str, ...]:
    """A whole metrics list with every group reference expanded.

    Order is the order written, groups expanded in place. A duplicate that
    arrives because one column sits in two groups is dropped, keeping its first
    position: a figure drawing one column twice is never what was meant, and a
    contrast testing it twice would correct across it twice.

    ``where`` names the setting in any error - "contrast 'drift_in_area'
    metrics", "figures.rhythm-strength.options.metrics" - because the reference
    itself says nothing about which of forty places it was written in.
    """
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise ValueError(
            f"{where}: expected a list of column names, not "
            f"{type(values).__name__}"
        )
    resolved: list[str] = []
    for value in values:
        try:
            members = resolve_one(value, groups)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        for member in members:
            if member not in resolved:
                resolved.append(member)
    return tuple(resolved)


def resolve_setting(value, groups: dict[str, MetricGroup], *, where: str):
    """One configuration value that may name groups, with them expanded.

    Only a value that actually holds a reference is touched. Everything else
    comes back exactly as written, ``"a,b"`` included - that string is for the
    option's own cast to split, and resolving it here would be this module
    deciding what a comma means in somebody else's vocabulary.
    """
    if is_reference(value):
        return list(resolve_metrics([value], groups, where=where))
    if isinstance(value, (list, tuple)) and any(is_reference(v) for v in value):
        return list(resolve_metrics(value, groups, where=where))
    return value
