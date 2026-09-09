"""Module registry.

Every kind of analysis - morphology, intensity, motility, surveillance,
rhythms - is a self-contained module that registers itself here. Adding a new
readout means writing one file and importing it; nothing else in the package
changes, and the command line, the manifest and the report pick it up
automatically.

A module declares what image data it needs. If that data is not available for a
movie the module is skipped and the skip is recorded, rather than failing the
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from analysis.units import Scale


@dataclass(frozen=True)
class ChannelStack:
    """One extra imaging channel, already cropped and aligned to the labels.

    The package was built around a single signal stack, because the outlines
    were drawn from one. A microscope rarely records one. A second channel -
    a stain, a second reporter, a transmitted-light view - is measured through
    exactly the same outlines, and nothing about that is specific to which
    channel it is or what it stains, so channels are declared in the
    configuration and carried here by name rather than written into the
    package one at a time.

    ``values`` is float, not the source dtype, for one reason: aligning a
    channel to the labels can ask for a pixel the source does not have, at the
    edge of a frame the field drifted away from. That pixel is ``NaN`` rather
    than a wrapped-around neighbour or a zero, so a cell sitting half off the
    source is visibly missing instead of quietly dark. Every measurement is
    ``nan``-aware and reports how many valid pixels it had.

    ``saturation_value`` is the largest number the source dtype can hold, or
    ``None`` for a float source where there is no such ceiling. A pixel at
    that value was not measured, it was clipped, and a channel that spends
    much of its time there is reporting the camera rather than the sample.
    """

    name: str
    values: np.ndarray                  # (T, H, W) float32, NaN where unsampled
    source_dtype: str
    saturation_value: float | None
    path: str
    description: str = ""

    def frame(self, index: int) -> np.ndarray:
        return self.values[index]


@dataclass(frozen=True)
class ObjectStack:
    """One set of labelled reference shapes, already cropped and aligned.

    ``neighbours`` and ``contacts`` measure a cell against another cell, and
    that is the whole of what this package could relate a cell to. Blood
    vessels, plaques, a wound edge, a second cell type: none of it was
    expressible. An object set is that second thing, declared in the
    configuration and carried here by name, because what the shapes *are* is
    the user's to say and no module should need editing to accept a new kind.

    **Integer labels, and an unreachable pixel is 0 rather than NaN.** This is
    the one real departure from :class:`ChannelStack` and it is deliberate. A
    channel is a brightness, so a pixel the alignment could not reach has no
    value and ``NaN`` is the honest answer. An object stack is an identity, and
    "no object here" and "could not look here" are both the absence of an
    object; inventing a label number that meant "unknown" would put a shape in
    the tables that nobody drew. The count of unreachable pixels is kept in the
    load record instead, so the distinction survives where it can be acted on -
    in the manifest - rather than in an array that every module would have to
    remember to check.

    ``static`` is a map drawn once for the whole recording. It is stored as a
    single frame and handed out for every index, so the memory cost of a vessel
    tree does not multiply by the number of timepoints.
    """

    name: str
    values: np.ndarray                  # (T, H, W) or (1, H, W) int32; 0 = no object
    static: bool
    path: str
    description: str = ""

    def frame(self, index: int) -> np.ndarray:
        return self.values[0] if self.static else self.values[index]


@dataclass
class MeasurementContext:
    """Everything a module is allowed to read, already aligned in time.

    ``labels[i]``, ``raw[i]``, ``unclaimed[i]``, ``inferred[i]``,
    ``unresolved[i]`` and ``added[i]`` all describe the same moment.
    ``evidence[i]`` describes the transition from frame ``i`` to frame
    ``i + 1`` and its last entry is undefined, so modules must stop at
    ``n_frames - 1``.

    ``channels`` holds any extra imaging channels the configuration declared,
    keyed by name and already trimmed, cropped and shifted into the label
    field, so ``channels["dapi"].values[i]`` describes the same moment as
    ``labels[i]``.

    ``side`` holds any tables the user supplied - a stimulus log, a focus
    score, a per-cell call - already keyed into this movie's own numbering. A
    module may read them; nothing in the package derives anything from them,
    because their columns mean whatever the user meant by them.

    ``valid`` says where measuring can be trusted at all - the part of the
    field the microscope actually delivered in a given frame. It changes what a
    density or a share is divided by; it never filters a row, because whether a
    cell in a vignetted corner should be dropped is a scientific decision and
    this package does not make those.

    ``inferred``, ``unresolved`` and ``added`` say where the outlines came
    from, not what is in them. A true pixel in ``inferred`` is one whose owner
    the tracker reconstructed rather than saw; the far rarer true pixel in
    ``added`` is one the segmentation never saw at all, so the outline itself
    is the tracker's and not the microscope's. They change how a number should
    be weighed; they never change the number.
    """

    stem: str
    labels: np.ndarray                      # (T, H, W) uint16 identity labels
    raw: np.ndarray                         # (T, H, W) registered raw signal
    scale: Scale
    identities: list[int]
    unclaimed: np.ndarray | None = None     # (T, H, W) foreground with no name
    evidence: np.ndarray | None = None      # (T, C, H, W) motion evidence
    inferred: np.ndarray | None = None      # (T, H, W) bool: ownership reconstructed
    unresolved: np.ndarray | None = None    # (T, H, W) bool: tracker left undecided
    added: np.ndarray | None = None         # (T, H, W) bool: outline pixel never detected
    source_frame_offset: int = 0            # labels[i] == source frame i + offset
    #: Extra imaging channels, keyed by the name the configuration gave them.
    #: Empty when the movie declares none, which is why a module that needs one
    #: is skipped rather than failed - see ``AnalysisModule.available``.
    channels: dict[str, ChannelStack] = field(default_factory=dict)
    #: Where measuring can be trusted, in label space: ``(H, W)`` for one mask
    #: over the whole recording, ``(T, H, W)`` for one per frame, and ``None``
    #: when the movie declares none. Ask :meth:`valid_frame` and
    #: :meth:`valid_px` rather than reading it, so the undeclared case is
    #: handled once instead of in every module that divides by a field.
    valid: np.ndarray | None = None
    #: Sets of labelled reference shapes, keyed by the name the configuration
    #: gave them and already cropped and shifted into the label field, so
    #: ``objects["vessels"].frame(i)`` describes the same moment as
    #: ``labels[i]``. Empty when the movie declares none, which is why a module
    #: that needs one is skipped rather than failed.
    objects: dict[str, ObjectStack] = field(default_factory=dict)
    #: Tables the user supplied, keyed by the name the configuration gave them.
    #: Each holds its key column - ``frame_index`` or ``identity`` - and its own
    #: columns under that name as a prefix. Nothing here is measured and nothing
    #: is interpreted: the columns are the user's words, carried through, and
    #: the join attaches them to the tables that share the key.
    side: dict[str, pd.DataFrame] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def channel_names(self) -> list[str]:
        """The declared channels in configuration order, for stable output."""
        return list(self.channels)

    def valid_frame(self, index: int) -> np.ndarray:
        """The measurable field for one frame; all-``True`` when none was declared."""
        if self.valid is None:
            return np.ones(self.labels.shape[1:], dtype=bool)
        return self.valid[index] if self.valid.ndim == 3 else self.valid

    def valid_px(self, index: int) -> int:
        """How many pixels of one frame can be measured.

        With no mask this is the whole field, computed as the product of the
        two side lengths rather than by summing an array of ``True`` - the same
        arithmetic the modules did before a mask existed, so an undeclared mask
        cannot move a number in the last decimal place.
        """
        field = self.labels.shape[1:]
        if self.valid is None:
            return int(field[0] * field[1])
        return int(self.valid_frame(index).sum())

    @property
    def object_set_names(self) -> list[str]:
        """The declared object sets in configuration order, for stable output."""
        return list(self.objects)

    def side_keyed_on(self, key: str) -> list[str]:
        """The side tables joined on ``key``, in configuration order."""
        return [name for name, table in self.side.items() if key in table.columns]

    @property
    def n_frames(self) -> int:
        return int(self.labels.shape[0])

    def frame_table(self) -> pd.DataFrame:
        """The time axis every module and table shares."""
        index = np.arange(self.n_frames)
        return pd.DataFrame(
            {
                "frame_index": index,
                "imagej_frame": index + 1,
                "source_imagej_frame": index + 1 + self.source_frame_offset,
                "hours": self.scale.hours(index + self.source_frame_offset),
            }
        )

    def module_params(self, name: str) -> dict:
        return dict(self.params.get(name, {}))


@dataclass(frozen=True)
class Column:
    """One column a module writes, as the rest of the package needs to talk about it.

    Declared here rather than in ``figures/_metrics.py`` because the module is
    where the number is made, and a label kept somewhere else is a label that
    goes stale silently: nothing fails when they disagree, the axis just says
    something slightly wrong on every figure that draws it.

    ``unit`` may name the frame interval as ``{interval}``: "pixels replaced per
    30 min" has to say 30 min because the number is per frame, and hard-coding
    it makes the label wrong the first time somebody images every 15 minutes.

    ``role`` is a colour family the theme can resolve, so one measurement keeps
    one colour across every figure without any builder repeating the mapping.
    """

    name: str
    label: str
    unit: str = ""
    role: str = "morphology"


@dataclass(frozen=True)
class Output:
    """One CSV a module writes, as the run needs to talk about it.

    ``Column`` says what a number means; this says what a *row* is. Between
    them a module describes its own output completely, and the run writer stops
    having to know anything about particular modules.

    ``grain`` answers "one row per what": the columns whose combination is
    unique in this table. It is what lets the writer decide, without a
    hard-coded list, whether this table is a file of its own or columns of a
    shared table at the same grain.

    ``fold`` is deliberately separate from ``grain``. ``presence`` has the same
    grain as ``cell_frame`` and must NOT be folded into it: it carries a row
    for every cell in every frame including the frames where that cell is
    absent - 8217 rows against cell_frame's 6073 - and folding it would either
    lose the absences or turn cell_frame into a grid of mostly blank rows.
    Grain says what a row is; ``fold`` says whether these are the same rows.

    ``origin`` is ``"measured"`` for a number this package computed and
    ``"tracker"`` for a table copied verbatim out of a tracking run. It picks
    the folder the table is written into, so a reader can tell at a glance
    which half of a run folder this package is answerable for.

    ``optional`` marks a table that is not written for every movie, because the
    inputs it needs are not part of every tracking chain. An absent optional
    table is a skip, not a failure.
    """

    name: str
    grain: tuple[str, ...]
    fold: bool = False
    origin: str = "measured"
    optional: bool = False


#: The two origins an ``Output`` may claim. ``measured`` is this package's own
#: arithmetic; ``tracker`` is somebody else's CSV reproduced without comment.
ORIGINS = frozenset({"measured", "tracker"})


@dataclass
class AnalysisModule:
    name: str
    description: str
    requires: tuple[str, ...]
    measure: Callable[[MeasurementContext], dict[str, pd.DataFrame]]
    summarise: Callable[[dict[str, pd.DataFrame], MeasurementContext], dict] | None = None
    defaults: dict = field(default_factory=dict)
    produces: tuple[Column, ...] = ()
    writes: tuple[Output, ...] = ()

    def available(self, context: MeasurementContext) -> tuple[bool, str]:
        """Whether this movie carries what the module reads.

        An empty ``channels`` counts as absent, not as present-and-empty. The
        field is a dictionary so that a movie with no extra channels is the
        default rather than a special case, which means the missing-input test
        cannot be ``is None`` for it alone.
        """
        for need in self.requires:
            value = getattr(context, need, None)
            if value is None or (isinstance(value, dict) and not value):
                return False, f"{self.name} needs {need}, which this movie does not have"
        return True, ""


@dataclass
class DerivedModule:
    """A module that reads the joined measurement tables, not the pixels.

    Rhythm fitting, group contrasts and anything else that is arithmetic on
    already measured numbers lives here, so it runs in seconds and can be
    re-run without touching the image stacks again.
    """

    name: str
    description: str
    needs_columns: tuple[str, ...]
    derive: Callable[[pd.DataFrame, MeasurementContext], dict[str, pd.DataFrame]]
    defaults: dict = field(default_factory=dict)
    produces: tuple[Column, ...] = ()
    writes: tuple[Output, ...] = ()
    resolve_params: Callable[[MeasurementContext], dict] | None = None

    def parameters(self, context: MeasurementContext) -> dict:
        """The effective settings used by this module for one recording."""
        if self.resolve_params is not None:
            return dict(self.resolve_params(context))
        return {**self.defaults, **context.module_params(self.name)}


_REGISTRY: dict[str, AnalysisModule] = {}
_DERIVED: dict[str, DerivedModule] = {}


def register_derived(
    name: str,
    description: str,
    needs_columns: Iterable[str] = (),
    defaults: dict | None = None,
    produces: Iterable[Column] = (),
    writes: Iterable[Output] = (),
    resolve_params: Callable[[MeasurementContext], dict] | None = None,
) -> Callable:
    """Decorate a function that turns the joined cell-frame table into more tables."""

    def decorator(function: Callable) -> Callable:
        _DERIVED[name] = DerivedModule(
            name=name,
            description=description,
            needs_columns=tuple(needs_columns),
            derive=function,
            defaults=dict(defaults or {}),
            produces=tuple(produces),
            writes=tuple(writes),
            resolve_params=resolve_params,
        )
        return function

    return decorator


def list_derived() -> list[DerivedModule]:
    return [_DERIVED[key] for key in sorted(_DERIVED)]


def get_derived(name: str) -> DerivedModule:
    if name not in _DERIVED:
        known = ", ".join(sorted(_DERIVED)) or "none registered"
        raise KeyError(f"unknown derived module {name!r}; known modules: {known}")
    return _DERIVED[name]


def register(
    name: str,
    description: str,
    requires: Iterable[str] = ("labels",),
    defaults: dict | None = None,
    summarise: Callable | None = None,
    produces: Iterable[Column] = (),
    writes: Iterable[Output] = (),
) -> Callable:
    """Decorate a ``measure`` function to add a module to the package."""

    def decorator(function: Callable) -> Callable:
        _REGISTRY[name] = AnalysisModule(
            name=name,
            description=description,
            requires=tuple(requires),
            measure=function,
            summarise=summarise,
            defaults=dict(defaults or {}),
            produces=tuple(produces),
            writes=tuple(writes),
        )
        return function

    return decorator


def list_modules() -> list[AnalysisModule]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def get_module(name: str) -> AnalysisModule:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"unknown analysis module {name!r}; known modules: {known}")
    return _REGISTRY[name]


#: The columns that say which row this is rather than what was measured, so no
#: module declares them: ``identity`` and ``frame_index`` are the join keys
#: (``run._join_cell_frame``), the next three come from
#: ``MeasurementContext.frame_table``, and the last three are stamped onto every
#: table by ``run._stamp`` so a pooled analysis is a concatenation.
SHARED_COLUMNS = frozenset({
    "identity", "frame_index", "imagej_frame", "source_imagej_frame", "hours",
    "stem", "condition", "subject",
})


def declared_columns() -> dict[str, Column]:
    """Every column the registered modules say they write, keyed by name.

    Two modules may declare the same column - ``morphology`` and ``motility``
    both write a centroid, ``motility`` and ``surveillance`` both write
    ``gap_frames`` - and that is not a mistake: the join in
    ``run._join_cell_frame`` keeps one copy and drops the rest. What would be a
    mistake is the two disagreeing about what it means, because then the label
    on the axis depends on which module happened to sort first.
    """
    found: dict[str, Column] = {}
    owner: dict[str, str] = {}
    for module in (*list_modules(), *list_derived()):
        for column in module.produces:
            seen = found.get(column.name)
            if seen is not None and seen != column:
                raise ValueError(
                    f"{module.name} and {owner[column.name]} both write "
                    f"{column.name!r} but describe it differently:\n"
                    f"  {owner[column.name]}: {seen}\n"
                    f"  {module.name}: {column}\n"
                    "One column has one meaning, or the axis label depends on "
                    "which module sorted first."
                )
            found[column.name] = column
            owner.setdefault(column.name, module.name)
    return found


def declared_tables() -> dict[str, Output]:
    """Every table the registered modules say they write, keyed by name.

    Unlike a column, a table has exactly one author. Two modules writing one
    name is refused and both are named, because the second one silently
    overwrites the first in the dictionary the run collects results into, and
    the file that lands is whichever module happened to run last.

    A grain naming a column that no module produces and that is not in
    ``SHARED_COLUMNS`` is refused too: a grain is a promise that those columns
    identify a row, and a promise about a column that does not exist is a table
    nobody can join. An empty grain is allowed only for a copied table, where
    saying what one row is would be this package asserting the meaning of
    somebody else's CSV.
    """
    found: dict[str, Output] = {}
    owner: dict[str, str] = {}
    known = set(declared_columns()) | SHARED_COLUMNS
    for module in (*list_modules(), *list_derived()):
        for output in module.writes:
            if output.name in found:
                raise ValueError(
                    f"{module.name} and {owner[output.name]} both write the table "
                    f"{output.name!r}. One table has one author, or the file that "
                    "lands is whichever module ran last."
                )
            if output.origin not in ORIGINS:
                raise ValueError(
                    f"{module.name} declares {output.name!r} with origin "
                    f"{output.origin!r}; expected one of {sorted(ORIGINS)}"
                )
            if not output.grain and output.origin != "tracker":
                raise ValueError(
                    f"{module.name} declares {output.name!r} without a grain. "
                    "Only a table copied from the tracking pass may leave it "
                    "empty; for a table this package computes, not knowing what "
                    "one row is means not knowing what was written."
                )
            unknown = sorted(set(output.grain) - known)
            if unknown:
                raise ValueError(
                    f"{module.name} declares {output.name!r} at a grain of "
                    f"{', '.join(unknown)}, which no module produces. Declare "
                    "the column in a module's PRODUCES, or the table promises a "
                    "key nothing can join on."
                )
            found[output.name] = output
            owner[output.name] = module.name
    return found
