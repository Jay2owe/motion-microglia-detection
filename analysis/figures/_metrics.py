"""What each measured column is called, and what colour it belongs to.

A figure that lets the user choose what to plot has to be able to label a column
it has never been told about. This is the vocabulary: for every column the
package writes, the words to put on an axis, the unit, and the family it belongs
to - so choosing ``turnover_index`` gets the surveillance colour in every figure
that draws it, without any builder repeating the mapping.

Almost none of it is written here. A module declares the columns it writes on
its own ``@register(produces=...)``, next to the code that computes them, and
``METRICS`` is those declarations plus the ``RESIDUAL`` block below - what no
module produces, because a figure computes it on its way to drawing. Wording
kept away from the number it describes goes stale silently: nothing fails when
the two disagree, the axis just says something slightly wrong on every figure
that draws it.

A column that is not listed either way is still drawable. The fallback turns
``my_new_column`` into "My new column" and reads the unit off the suffix, which
is worse than a written label and better than a traceback. It stays for the
column a user derives themselves, which no module can have declared.

Units may name the frame interval as ``{interval}``: "pixels replaced per 30
min" has to say 30 min because the number is per frame, and hard-coding it makes
the label wrong the first time somebody images every 15 minutes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

__all__ = [
    "Metric", "METRICS", "RESIDUAL", "describe", "documented", "semantic_label",
    "axis_label", "role_for",
]


@dataclass(frozen=True)
class Metric:
    """One measured column, as a figure needs to talk about it."""

    column: str
    label: str
    unit: str = ""
    role: str = "morphology"

    def unit_text(self, interval_minutes: float | None = None) -> str:
        if not self.unit:
            return ""
        if "{interval}" not in self.unit:
            return self.unit
        if interval_minutes is None:
            return self.unit.replace("{interval}", "frame")
        return self.unit.replace("{interval}", f"{interval_minutes:.0f} min")


def _m(column: str, label: str, unit: str = "", role: str = "morphology") -> Metric:
    return Metric(column=column, label=label, unit=unit, role=role)


#: Columns no module produces, so there is nowhere else to put their wording.
#:
#: Empty, and that is the healthy state. Everything that lived here has gone the
#: same way. ``coverage_share`` was cumulative territory over eventual
#: territory, computed inside figure 26 - a number only one page could see;
#: ``territory_shape`` declares it now. The recurrence four -
#: ``recurrence_rate``, ``determinism``, ``laminarity`` and ``dtw_distance`` -
#: were quantified from distance matrices figures 32 and 33 built and never
#: wrote down; ``recurrence`` and ``sequence_distance`` declare them now, with
#: the same wording, so nothing on an axis changed when they moved.
#:
#: ``relative_amplitude`` left by retirement rather than by promotion. It was
#: only ever a short alias for ``rhythms``\' ``cosinor_relative_amplitude``, used
#: by two figures, and those figures now name the real column. The short name is
#: free, and ``rhythms`` claims it for the non-parametric measure:
#: ``(M10 - L5) / (M10 + L5)``, which is what the rest of the field means by
#: relative amplitude.
#:
#: The block stays. A figure that invents a number still needs somewhere to put
#: its wording, and the right response to finding an entry here is to ask which
#: module should own it - not to leave it.
#:
#: The other two groups this could have held are empty and should stay that way.
#: The index columns - ``identity``, ``frame_index``, ``hours`` - say which row
#: this is rather than what was measured, and the design columns ``stem``,
#: ``condition`` and ``subject`` say which movie, so none of them is ever an
#: axis. They are listed in ``registry.SHARED_COLUMNS`` instead.
#:
#: Anything else appearing here is a module that has not declared something it
#: writes, which ``test_column_declarations`` should have caught first.
RESIDUAL: dict[str, Metric] = {metric.column: metric for metric in ()}


class _Vocabulary(Mapping):
    """``RESIDUAL`` plus every column the registered modules declare.

    A mapping rather than a dict because building it means importing
    ``analysis.modules``, which pulls in scikit-image and scipy, and this module
    is imported by every figure and by ``_bundle`` at module scope. Deferring
    the import to the first lookup keeps ``import _metrics`` as cheap as it was
    when the table was a literal, and the result is memoised, so a figure build
    pays for it once.

    Reading the registry rather than a run's own record is the deliberate
    choice here. Making a bundle carry its vocabulary the way it carries
    ``theme.json`` would need either a mutable "current run" global in a module
    whose whole surface is free functions - ``describe``, ``axis_label``,
    ``role_for``, called from panels that are handed no run - or the run
    threaded through every one of those signatures. The second is a larger
    change than the vocabulary itself; the first is a global that quietly
    misleads the moment two runs are open at once. So a rebuild uses today's
    wording, and a wording change is a figure change, which is the same rule
    the rest of the package already applies to labels.
    """

    def __init__(self, residual: dict[str, Metric]) -> None:
        self._residual = residual
        self._merged: dict[str, Metric] | None = None

    def _entries(self) -> dict[str, Metric]:
        if self._merged is None:
            import analysis.modules  # noqa: F401  - registers every module
            from analysis.registry import declared_columns

            merged = dict(self._residual)
            for name, column in declared_columns().items():
                merged[name] = Metric(
                    column=name, label=column.label, unit=column.unit, role=column.role
                )
            self._merged = merged
        return self._merged

    def __getitem__(self, column: str) -> Metric:
        return self._entries()[column]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def __repr__(self) -> str:
        if self._merged is None:
            return f"<vocabulary, not yet built; {len(self._residual)} residual entries>"
        return f"<vocabulary of {len(self._merged)} columns>"


#: column -> how to talk about it, for every column the package writes.
METRICS: Mapping[str, Metric] = _Vocabulary(RESIDUAL)


#: Suffix -> unit, for a column nobody has written wording for yet.
_SUFFIX_UNITS = {
    "_px": "px",
    "_px2": "px^2",
    "_frames": "frames",
    "_hours": "h",
    "_fraction": "fraction",
    "_index": "",
    "_count": "count",
}


#: What ``cell_summary.csv`` does to a column name when it summarises a whole
#: cell. ``area_px_median`` is still an area, so it should be labelled like one.
_SUMMARY_SUFFIXES = (
    "_median", "_mean", "_iqr", "_sd", "_p10", "_p90", "_min", "_max",
    "_first", "_last", "_delta",
)


def describe(column: str) -> Metric:
    """The vocabulary for one column, invented from its name if it has none.

    A per-cell summary of a known measurement keeps that measurement's wording:
    ``turnover_index_median`` is pixels replaced, and how it was summarised is
    the builder's sentence to add, not a different quantity.
    """
    if column in METRICS:
        return METRICS[column]
    for suffix in _SUMMARY_SUFFIXES:
        if column.endswith(suffix) and column[: -len(suffix)] in METRICS:
            known = METRICS[column[: -len(suffix)]]
            return Metric(column=column, label=known.label, unit=known.unit,
                          role=known.role)
    unit = ""
    stem = column
    for suffix, guess in _SUFFIX_UNITS.items():
        if column.endswith(suffix):
            unit, stem = guess, column[: -len(suffix)]
            break
    words = stem.replace("_", " ").strip()
    label = (words[:1].upper() + words[1:]) if words else column
    return Metric(column=column, label=label, unit=unit, role="morphology")


def documented(column: str) -> bool:
    """Whether a plotted column has a deliberately written meaning."""
    if column in METRICS:
        return True
    return any(
        column.endswith(suffix) and column[: -len(suffix)] in METRICS
        for suffix in _SUMMARY_SUFFIXES
    )


def semantic_label(column: str) -> str:
    """Plain-language label for a plot, refusing an undocumented database name."""
    if not documented(column):
        raise ValueError(
            f"{column!r} has no documented plot label; declare it on the module "
            f"that writes it, as Column({column!r}, ...) in its PRODUCES, "
            "before putting it on a figure"
        )
    return describe(column).label


def axis_label(column: str, interval_minutes: float | None = None, wrap: bool = True) -> str:
    """The words for an axis: the name, then the unit on its own line.

    Two lines because these are stacked panels with a narrow left margin, and a
    one-line label there either shrinks the panel or overlaps the one above it.
    """
    metric = describe(column)
    unit = metric.unit_text(interval_minutes)
    if not unit:
        return metric.label
    separator = "\n" if wrap else " "
    bare = unit.startswith(("per ", "-", "0-")) or unit in {"CV", "count", "fraction", "ratio"}
    return f"{metric.label}{separator}{unit if bare else f'({unit})'}"


def role_for(column: str) -> str:
    """The colour role a column belongs to, so one metric keeps one colour."""
    return describe(column).role
