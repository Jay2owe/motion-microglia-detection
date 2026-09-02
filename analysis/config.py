"""Analysis configuration.

Deliberately separate from the tracking ``config.json``. Tracking configuration
describes how outlines are produced; this describes how an already accepted set
of outlines is measured. A user who never runs the tracker - who has outlines
from somewhere else - still needs this file and nothing else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from analysis.conditions import UNASSIGNED, Assignment, ConditionSet


def _declared_condition(value: object) -> str | None:
    text = str(value).strip() if value not in (None, "") else ""
    return None if text in ("", UNASSIGNED) else text


#: A channel name becomes a value in the ``channel`` column of three tables and
#: a series label on any figure that draws them, so it is held to the same
#: shape as a column name rather than accepted as free text. A side table's
#: name becomes the prefix on every column it contributes, so it is held to
#: the same shape for the same reason.
_CHANNEL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

#: Names the package already uses for a stack of its own. Reusing one would put
#: two different images under one word in the same manifest.
_RESERVED_CHANNEL_NAMES = frozenset(
    {"labels", "raw", "unclaimed", "evidence", "provenance", "history"}
)


@dataclass
class ChannelConfig:
    """One extra imaging channel, and how to line it up with the outlines.

    Everything here except ``name`` and ``path`` is a way of saying "this file
    is not already in the same space as the labels". They are separate settings
    because they are separate mistakes: a wrong ``channel_index`` measures the
    wrong dye, a wrong ``frame_offset`` measures the right dye at the wrong
    time, and a wrong ``crop_origin`` measures the right dye somewhere else in
    the dish. All three default to "no adjustment", so a channel already saved
    beside the raw stack needs two lines.

    The alignment contract is one sentence, and it is worth reading twice::

        analysed[i, y, x] == source[frame_offset + i,
                                    crop_origin[0] + y - shift_y[i],
                                    crop_origin[1] + x - shift_x[i]]

    ``shift_y``/``shift_x`` come from ``shifts``, a CSV with one row per source
    frame, read from the columns named in ``shift_columns`` and multiplied by
    ``shift_scale`` before rounding. A registration log that records the drift
    it *measured* rather than the correction it *applied* is the common case,
    and it is turned into the latter with ``shift_scale: -1``.
    """

    name: str
    path: Path
    #: Which channel of a multi-channel file. Required when the file has a
    #: channel axis, refused when it does not, because guessing either way
    #: measures the wrong dye without saying so.
    channel_index: int | None = None
    #: Source frame of label frame 0. ``None`` inherits the movie's
    #: ``source_frame_offset``, which is right whenever the channel comes from
    #: the same acquisition as the raw stack.
    frame_offset: int | None = None
    #: Where the analysed field sits inside this file, as (y, x). Zero when the
    #: channel is already cropped to the same field as the labels.
    crop_origin: tuple[int, int] = (0, 0)
    shifts: Path | None = None
    shift_columns: tuple[str, str] = ("shift_y", "shift_x")
    shift_scale: float = 1.0
    expected_sha256: str | None = None
    #: What this channel is, in the user's words. Carried into the manifest and
    #: available to figures; the package never guesses what a channel stains.
    description: str = ""

    @classmethod
    def from_dict(cls, raw_dict: dict, resolve) -> "ChannelConfig":
        name = str(raw_dict.get("name", "")).strip()
        if not _CHANNEL_NAME.match(name):
            raise ValueError(
                f"channel name {name!r} is not usable: a channel name becomes a "
                "value in the `channel` column and a label on a figure, so it "
                "must be lower case, start with a letter and hold only letters, "
                "digits and underscores"
            )
        if name in _RESERVED_CHANNEL_NAMES:
            raise ValueError(
                f"channel name {name!r} is already the name of a stack this "
                f"package loads; reserved names: "
                f"{', '.join(sorted(_RESERVED_CHANNEL_NAMES))}"
            )
        if not raw_dict.get("path"):
            raise ValueError(f"channel {name!r} has no path")
        origin = tuple(raw_dict.get("crop_origin", (0, 0)))
        if len(origin) != 2:
            raise ValueError(f"channel {name!r}: crop_origin must be [y, x]")
        columns = tuple(raw_dict.get("shift_columns", ("shift_y", "shift_x")))
        if len(columns) != 2:
            raise ValueError(
                f"channel {name!r}: shift_columns must name exactly two columns, "
                "the y one first"
            )
        index = raw_dict.get("channel_index")
        offset = raw_dict.get("frame_offset")
        return cls(
            name=name,
            path=resolve(raw_dict["path"]),
            channel_index=None if index is None else int(index),
            frame_offset=None if offset is None else int(offset),
            crop_origin=(int(origin[0]), int(origin[1])),
            shifts=resolve(raw_dict.get("shifts")),
            shift_columns=(str(columns[0]), str(columns[1])),
            shift_scale=float(raw_dict.get("shift_scale", 1.0)),
            expected_sha256=raw_dict.get("sha256"),
            description=str(raw_dict.get("description", "")),
        )


@dataclass
class MovieConfig:
    """One movie: its accepted outlines and the signal they were built from."""

    stem: str
    labels: Path
    raw: Path
    unclaimed: Path | None = None
    evidence: Path | None = None
    #: Per-pixel record of where the tracker decided ownership by
    #: reconstruction rather than by seeing that cell there, written beside the
    #: accepted labels and in the same frame space. Optional: a movie without
    #: one is measured exactly as before, it just cannot say how much of what
    #: it measured rests on a reconstructed attribution.
    provenance: Path | None = None
    #: The tracker's decision folder, holding why a name went missing, was
    #: hidden inside another or was renamed. Read by exactly one module.
    history: Path | None = None
    #: Where measuring can be trusted, as a boolean or 0/1 TIFF in label space:
    #: one frame applied to every label frame, or one per label frame. Several
    #: numbers divide by the size of the imaged field, and the field is not all
    #: measurable - registration leaves a margin blank in some frames and
    #: vignetting darkens the corners. Absent is the normal case and every
    #: number is then exactly what it was before this setting existed.
    valid_mask: Path | None = None
    #: Extra imaging channels measured through the same outlines. Empty is the
    #: normal case and costs nothing; see :class:`ChannelConfig`.
    channels: list["ChannelConfig"] = field(default_factory=list)
    #: Spreadsheets of the user's own, keyed on a timepoint or on a cell and
    #: joined onto the tables that share that key. See :class:`SideTableConfig`.
    side_tables: list["SideTableConfig"] = field(default_factory=list)
    #: Sets of labelled reference shapes - vessels, plaques, a wound edge -
    #: aligned to the outlines. See :class:`ObjectSetConfig`.
    objects: list["ObjectSetConfig"] = field(default_factory=list)
    #: Named stretches of *this* recording, replacing the shared block when set.
    #: Empty means "use whatever the top-level ``windows`` block says", which is
    #: the normal case; a movie declares its own only when its treatment went in
    #: at a different hour from everybody else's. See :class:`WindowConfig`.
    windows: list["WindowConfig"] = field(default_factory=list)
    source_frame_offset: int = 0
    #: The condition this movie was *declared* to be, or None to derive it from
    #: the name. Never read directly downstream - ask
    #: :meth:`AnalysisConfig.assignment`, which also handles derivation.
    condition: str | None = None
    #: The animal, dish or slice. Two movies from one subject are not two
    #: independent observations, and a later comparison has to know that.
    subject: str | None = None
    expected_sha256: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, raw_dict: dict, root: Path) -> "MovieConfig":
        def resolve(value):
            if value in (None, ""):
                return None
            path = Path(value)
            return path if path.is_absolute() else (root / path)

        return cls(
            stem=raw_dict["stem"],
            labels=resolve(raw_dict["labels"]),
            raw=resolve(raw_dict["raw"]),
            unclaimed=resolve(raw_dict.get("unclaimed")),
            evidence=resolve(raw_dict.get("evidence")),
            provenance=resolve(raw_dict.get("provenance")),
            history=resolve(raw_dict.get("history")),
            valid_mask=resolve(raw_dict.get("valid_mask")),
            channels=_channels(
                raw_dict.get("channels"), resolve, raw_dict.get("stem", "?")
            ),
            side_tables=_side_tables(
                raw_dict.get("side_tables"), resolve, raw_dict.get("stem", "?")
            ),
            objects=_object_sets(
                raw_dict.get("objects"), resolve, raw_dict.get("stem", "?")
            ),
            windows=_windows(raw_dict.get("windows"), raw_dict.get("stem", "?")),
            source_frame_offset=int(raw_dict.get("source_frame_offset", 0)),
            # "unassigned" is what a movie says when it has no group, not the
            # name of one, so it never counts as a declaration.
            condition=_declared_condition(raw_dict.get("condition")),
            subject=raw_dict.get("subject"),
            expected_sha256=dict(raw_dict.get("sha256", {})),
            notes=raw_dict.get("notes", ""),
        )


def _channels(entries: object, resolve, stem: str) -> list[ChannelConfig]:
    """Parse the per-movie ``channels`` block, refusing a repeated name.

    A repeat is refused here rather than left to the loader because the second
    one would silently replace the first in the context dictionary, and the
    tables would then carry one channel under two names without anything
    looking wrong.
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"{stem}: channels must be a list of channel blocks, not "
            f"{type(entries).__name__}"
        )
    parsed = [ChannelConfig.from_dict(entry, resolve) for entry in entries]
    seen: set[str] = set()
    for channel in parsed:
        if channel.name in seen:
            raise ValueError(f"{stem}: channel {channel.name!r} is declared twice")
        seen.add(channel.name)
    return parsed


@dataclass
class ObjectSetConfig:
    """One set of labelled reference shapes, and how to line it up with the outlines.

    Vessels, plaques, a wound edge, a scaffold, a second cell type. Everything
    in the package relates a cell to another cell or to nothing; this is the
    input that lets a cell be related to something that is not a cell.

    The alignment vocabulary is deliberately the same as
    :class:`ChannelConfig`'s, down to the setting names, because a user who has
    lined one input up already knows how to line this one up. The one addition
    is ``static``.

    ``static`` says the file holds a single map that applies to every frame - a
    vessel tree drawn once over the whole recording. It is held fixed rather
    than drifting with the registration, because such a map is drawn in the
    analysed field's own space; ``frame_offset`` and ``shifts`` are therefore
    refused alongside it rather than ignored, since a setting quietly dropped
    reads exactly like one that was honoured.
    """

    name: str
    path: Path
    #: One map for the whole recording, rather than one per frame.
    static: bool = False
    channel_index: int | None = None
    frame_offset: int | None = None
    crop_origin: tuple[int, int] = (0, 0)
    shifts: Path | None = None
    shift_columns: tuple[str, str] = ("shift_y", "shift_x")
    shift_scale: float = 1.0
    expected_sha256: str | None = None
    #: What these shapes are, in the user's words. The package never guesses,
    #: and never measures anything that depends on knowing.
    description: str = ""

    @classmethod
    def from_dict(cls, raw_dict: dict, resolve) -> "ObjectSetConfig":
        name = str(raw_dict.get("name", "")).strip()
        if not _CHANNEL_NAME.match(name):
            raise ValueError(
                f"object set name {name!r} is not usable: the name becomes a "
                "value in the `object_set` column and a label on a figure, so "
                "it must be lower case, start with a letter and hold only "
                "letters, digits and underscores"
            )
        if name in _RESERVED_CHANNEL_NAMES:
            raise ValueError(
                f"object set name {name!r} is already the name of a stack this "
                f"package loads; reserved names: "
                f"{', '.join(sorted(_RESERVED_CHANNEL_NAMES))}"
            )
        if not raw_dict.get("path"):
            raise ValueError(f"object set {name!r} has no path")
        origin = tuple(raw_dict.get("crop_origin", (0, 0)))
        if len(origin) != 2:
            raise ValueError(f"object set {name!r}: crop_origin must be [y, x]")
        columns = tuple(raw_dict.get("shift_columns", ("shift_y", "shift_x")))
        if len(columns) != 2:
            raise ValueError(
                f"object set {name!r}: shift_columns must name exactly two "
                "columns, the y one first"
            )
        static = bool(raw_dict.get("static", False))
        if static:
            for setting in ("frame_offset", "shifts"):
                if raw_dict.get(setting) not in (None, ""):
                    raise ValueError(
                        f"object set {name!r} is static and also sets "
                        f"{setting}, which has nothing to act on: a static set "
                        "is one map applied to every frame, drawn in the "
                        "analysed field's own space and held there. Drop "
                        f"{setting}, or drop static and supply one map per frame."
                    )
        index = raw_dict.get("channel_index")
        offset = raw_dict.get("frame_offset")
        return cls(
            name=name,
            path=resolve(raw_dict["path"]),
            static=static,
            channel_index=None if index is None else int(index),
            frame_offset=None if offset is None else int(offset),
            crop_origin=(int(origin[0]), int(origin[1])),
            shifts=resolve(raw_dict.get("shifts")),
            shift_columns=(str(columns[0]), str(columns[1])),
            shift_scale=float(raw_dict.get("shift_scale", 1.0)),
            expected_sha256=raw_dict.get("sha256"),
            description=str(raw_dict.get("description", "")),
        )


def _object_sets(entries: object, resolve, stem: str) -> list[ObjectSetConfig]:
    """Parse the per-movie ``objects`` block, refusing a repeated name.

    Refused for the same reason a repeated channel name is: the second would
    replace the first in the context dictionary, and the tables would then carry
    one set of shapes under two words without anything looking wrong.
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"{stem}: objects must be a list of object-set blocks, not "
            f"{type(entries).__name__}"
        )
    parsed = [ObjectSetConfig.from_dict(entry, resolve) for entry in entries]
    seen: set[str] = set()
    for objects in parsed:
        if objects.name in seen:
            raise ValueError(f"{stem}: object set {objects.name!r} is declared twice")
        seen.add(objects.name)
    return parsed


#: The two keys a side table may be joined on, and nothing else. Both already
#: exist as columns on the tables a side table joins into. A third key -
#: ``subject``, say - would need a roll-up at that grain, and this package has
#: no such table for it to join into.
_SIDE_KEYS = ("frame_index", "identity")

#: How a frame number is written in the user's file. ``label`` is already a
#: 0-based index into the analysed movie; ``source`` is a 1-based ImageJ frame
#: number in the acquisition the labels were cut out of, and is converted.
_SIDE_SPACES = ("label", "source")


@dataclass
class SideTableConfig:
    """One spreadsheet of the user's own, and which column of it is the key.

    Everything a microscope does not record ends up in a spreadsheet: which
    frames the stage jolted on, what the focus score was, which cell is which
    genotype, when a drug went in. None of it can reach a measurement today,
    because the only per-movie facts this package knows are ``condition`` and
    ``subject``, and they attach to a whole movie rather than to a cell.

    A side table is joined, never copied. The file stays where the user put it,
    it is fingerprinted, and its columns are attached to whichever tables share
    its key. Copying it into the run folder would be a second place for it to
    disagree with itself.

    ``keyed_on`` is the only choice that matters and it has two answers:

    ``frame_index``
        one row per timepoint. Reaches ``frame_summary`` and every row of
        ``cell_frame``.
    ``identity``
        one row per cell. Reaches ``cell_summary`` and every row of that cell
        in ``cell_frame``.

    ``key_space`` exists because frame numbering is the trap this package has
    already fallen into once. A log written in the microscope's own numbering
    is 1-based and counts frames the analysis discarded, so it is converted::

        frame_index = key - 1 - movie.source_frame_offset

    It means nothing for an identity key and is refused there rather than
    ignored, because a setting that is quietly dropped reads as one that was
    honoured.
    """

    name: str
    path: Path
    #: ``frame_index`` or ``identity``. Nothing else; see ``_SIDE_KEYS``.
    keyed_on: str = "frame_index"
    #: What the key column is called in the user's file. ``None`` means it is
    #: called whatever ``keyed_on`` is called here.
    key_column: str | None = None
    #: ``label`` or ``source``; frame keys only. See the class docstring.
    key_space: str = "label"
    #: An allow-list, so two columns of a fifty-column log can be attached
    #: without the other forty-eight. ``None`` takes every column but the key.
    columns: tuple[str, ...] | None = None
    expected_sha256: str | None = None
    #: What this table holds, in the user's words. Carried into the manifest;
    #: the package never reads it and never guesses what a column means.
    description: str = ""

    @property
    def key(self) -> str:
        """The column to look for in the user's file."""
        return self.key_column or self.keyed_on

    @classmethod
    def from_dict(cls, raw_dict: dict, resolve) -> "SideTableConfig":
        name = str(raw_dict.get("name", "")).strip()
        if not _CHANNEL_NAME.match(name):
            raise ValueError(
                f"side table name {name!r} is not usable: the name becomes the "
                "prefix on every column this table contributes, so it must be "
                "lower case, start with a letter and hold only letters, digits "
                "and underscores"
            )
        if name in _RESERVED_CHANNEL_NAMES:
            raise ValueError(
                f"side table name {name!r} is already the name of a stack this "
                f"package loads; reserved names: "
                f"{', '.join(sorted(_RESERVED_CHANNEL_NAMES))}"
            )
        if not raw_dict.get("path"):
            raise ValueError(f"side table {name!r} has no path")
        keyed_on = str(raw_dict.get("keyed_on", "frame_index"))
        if keyed_on not in _SIDE_KEYS:
            raise ValueError(
                f"side table {name!r}: keyed_on={keyed_on!r} is not a key this "
                f"package can join on; it accepts {' or '.join(_SIDE_KEYS)}"
            )
        key_space = str(raw_dict.get("key_space", "label"))
        if key_space not in _SIDE_SPACES:
            raise ValueError(
                f"side table {name!r}: key_space={key_space!r} is not a frame "
                f"numbering this package knows; it accepts "
                f"{' or '.join(_SIDE_SPACES)}"
            )
        if "key_space" in raw_dict and keyed_on != "frame_index":
            raise ValueError(
                f"side table {name!r} is keyed on {keyed_on!r} and also sets "
                "key_space, which only means something for a frame key. A cell "
                "identity is not numbered two ways; remove key_space."
            )
        columns = raw_dict.get("columns")
        if columns is not None:
            if not isinstance(columns, (list, tuple)):
                raise ValueError(
                    f"side table {name!r}: columns must be a list of column "
                    f"names, or null for all of them, not "
                    f"{type(columns).__name__}"
                )
            columns = tuple(str(column) for column in columns)
        key_column = raw_dict.get("key_column")
        return cls(
            name=name,
            path=resolve(raw_dict["path"]),
            keyed_on=keyed_on,
            key_column=None if key_column in (None, "") else str(key_column),
            key_space=key_space,
            columns=columns,
            expected_sha256=raw_dict.get("sha256"),
            description=str(raw_dict.get("description", "")),
        )


def _side_tables(entries: object, resolve, stem: str) -> list[SideTableConfig]:
    """Parse the per-movie ``side_tables`` block, refusing a repeated name.

    Refused here for the same reason a repeated channel name is: the second
    would replace the first in the context dictionary, and the columns of one
    table would then be missing without anything looking wrong.
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"{stem}: side_tables must be a list of side-table blocks, not "
            f"{type(entries).__name__}"
        )
    parsed = [SideTableConfig.from_dict(entry, resolve) for entry in entries]
    seen: set[str] = set()
    for table in parsed:
        if table.name in seen:
            raise ValueError(f"{stem}: side table {table.name!r} is declared twice")
        seen.add(table.name)
    return parsed


#: A window name has to be a plain lower-case identifier because it becomes a
#: value in a ``window`` column that people filter on and that figures put on an
#: axis. ``Baseline`` and ``baseline`` as two windows would be two groups nobody
#: meant to have.
_WINDOW_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class WindowConfig:
    """One named stretch of a recording.

    Half-open on purpose: ``from_hours <= h < to_hours``. Two windows that share
    an edge must not both claim the frame on it, or every paired difference is
    computed against a baseline that contains the first frame of the response.

    Declared in hours **or** in frames, never both. Hours are hours since the
    recording began - this package has no clock - so a window is a position in
    the movie rather than a time of day.
    """

    name: str
    from_hours: float | None = None
    to_hours: float | None = None
    from_frame: int | None = None
    to_frame: int | None = None
    #: Another window's name. When set, ``window_change`` carries this window's
    #: value against that one, per cell per metric, as a difference and a ratio.
    baseline: str | None = None
    description: str = ""

    @property
    def in_frames(self) -> bool:
        return self.from_frame is not None

    @classmethod
    def from_dict(cls, raw_dict: dict) -> "WindowConfig":
        name = str(raw_dict.get("name", "")).strip()
        if not _WINDOW_NAME.fullmatch(name):
            raise ValueError(
                f"window name {name!r} is not a plain lower-case identifier; "
                "it becomes a value in a `window` column that people filter on, "
                "so it must look like `baseline` or `after_drug`"
            )
        hours = ("from_hours" in raw_dict) or ("to_hours" in raw_dict)
        frames = ("from_frame" in raw_dict) or ("to_frame" in raw_dict)
        if hours and frames:
            raise ValueError(
                f"window {name!r} is declared in hours and in frames at once; "
                "give from_hours/to_hours or from_frame/to_frame, not both, "
                "because the two would have to agree and nothing checks that"
            )
        if not hours and not frames:
            raise ValueError(
                f"window {name!r} declares neither hours nor frames; give "
                "from_hours and to_hours, or from_frame and to_frame"
            )
        if hours:
            if "from_hours" not in raw_dict or "to_hours" not in raw_dict:
                raise ValueError(
                    f"window {name!r} needs both from_hours and to_hours; a "
                    "window with one open end would silently change length when "
                    "a longer recording was analysed"
                )
            start, stop = float(raw_dict["from_hours"]), float(raw_dict["to_hours"])
            if not stop > start:
                raise ValueError(
                    f"window {name!r} ends at {stop} h, which is not after its "
                    f"start at {start} h; a window is half-open, from <= h < to"
                )
            bounds = {"from_hours": start, "to_hours": stop}
        else:
            if "from_frame" not in raw_dict or "to_frame" not in raw_dict:
                raise ValueError(
                    f"window {name!r} needs both from_frame and to_frame; a "
                    "window with one open end would silently change length when "
                    "a longer recording was analysed"
                )
            start, stop = int(raw_dict["from_frame"]), int(raw_dict["to_frame"])
            if not stop > start:
                raise ValueError(
                    f"window {name!r} ends at frame {stop}, which is not after "
                    f"its start at frame {start}; a window is half-open, "
                    "from <= frame < to"
                )
            bounds = {"from_frame": start, "to_frame": stop}

        baseline = raw_dict.get("baseline")
        baseline = None if baseline in (None, "") else str(baseline)
        if baseline == name:
            raise ValueError(
                f"window {name!r} names itself as its own baseline, which would "
                "make every change zero and every ratio one"
            )
        return cls(name=name, baseline=baseline,
                   description=str(raw_dict.get("description", "")), **bounds)


def _windows(entries: object, where: str) -> list[WindowConfig]:
    """Parse a ``windows`` block, refusing a repeat and a dangling baseline.

    A repeated name is refused for the same reason a repeated channel name is:
    the second would quietly replace the first, and a summary would be missing a
    window without anything looking wrong. A baseline naming a window that does
    not exist is refused here rather than at use, because at use it would be a
    column of blanks that reads as "this cell had no baseline".
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"{where}: windows must be a list of window blocks, not "
            f"{type(entries).__name__}"
        )
    parsed = [WindowConfig.from_dict(entry) for entry in entries]
    seen: set[str] = set()
    for window in parsed:
        if window.name in seen:
            raise ValueError(f"{where}: window {window.name!r} is declared twice")
        seen.add(window.name)
    for window in parsed:
        if window.baseline is not None and window.baseline not in seen:
            raise ValueError(
                f"{where}: window {window.name!r} names {window.baseline!r} as "
                f"its baseline, but no window is called that; declared windows "
                f"are {sorted(seen)}"
            )
    return parsed


#: The units of replication this package will test at. No default anywhere:
#: eighty-three cells from one movie are not eighty-three independent samples,
#: and a package that guessed would find everything significant while looking
#: confident about it.
CONTRAST_UNITS = ("cell", "movie", "subject")

#: How several rows are reduced to one value for a unit. Also no default, and
#: for the same reason ``summarise.SUMMARY_METRICS`` writes four statistics
#: rather than one: which of these is used changes the answer, so the choice is
#: the reader's and it travels in the output row.
CONTRAST_AGGREGATES = ("median", "mean")


@dataclass(frozen=True)
class ContrastConfig:
    """One declared comparison.

    ``unit`` has no default on purpose. Eighty-three cells from one movie are
    not eighty-three independent samples, and a package that guessed would find
    everything significant while looking confident about it.

    There is deliberately no way to say "test every metric". Four hundred and
    forty-nine columns across a handful of contrasts is thousands of tests, and
    a correction across all of them destroys the power to find the twenty that
    were the point.
    """

    name: str
    table: str
    metrics: tuple[str, ...]
    group_by: str
    groups: tuple[str, ...]
    unit: str
    test: str
    #: ``median`` or ``mean``, and required whenever ``unit`` is coarser than a
    #: cell. Refused when ``unit`` is ``cell``, where there is nothing to
    #: aggregate and a setting would imply there was.
    aggregate: str | None = None
    family: str = "default"
    window: str | None = None
    correction: str = "benjamini_hochberg"
    alpha: float = 0.05
    description: str = ""

    @classmethod
    def from_dict(cls, raw_dict: dict, tests: tuple[str, ...],
                  paired_tests: tuple[str, ...],
                  corrections: tuple[str, ...]) -> "ContrastConfig":
        name = str(raw_dict.get("name", "")).strip()
        if not _WINDOW_NAME.fullmatch(name):
            raise ValueError(
                f"contrast name {name!r} is not a plain lower-case identifier; "
                "it names a family of results in statistics.csv, so it must "
                "look like `reporter_by_genotype`"
            )

        def required(key: str) -> object:
            if key not in raw_dict or raw_dict[key] in (None, ""):
                raise ValueError(
                    f"contrast {name!r} does not say {key!r}, and this package "
                    "does not guess it"
                )
            return raw_dict[key]

        metrics = required("metrics")
        if not isinstance(metrics, (list, tuple)) or not metrics:
            raise ValueError(
                f"contrast {name!r}: metrics must be a non-empty list of column "
                "names. There is no way to test every metric at once - a "
                "correction across a thousand tests nobody meant to run "
                "destroys the power to find the twenty that were the point."
            )
        groups = required("groups")
        if not isinstance(groups, (list, tuple)) or len(groups) < 2:
            raise ValueError(
                f"contrast {name!r}: groups must be a list of at least two "
                f"values of {raw_dict.get('group_by', 'the grouping column')} "
                "to compare"
            )
        groups = tuple(str(group) for group in groups)
        if len(set(groups)) != len(groups):
            raise ValueError(f"contrast {name!r}: the same group is named twice")

        unit = str(required("unit"))
        if unit not in CONTRAST_UNITS:
            raise ValueError(
                f"contrast {name!r}: unit={unit!r} is not a unit of replication "
                f"this package knows; it accepts {' or '.join(CONTRAST_UNITS)}"
            )
        test = str(required("test"))
        if test not in tests:
            raise ValueError(
                f"contrast {name!r}: test={test!r} is not one this package "
                f"offers; it accepts {', '.join(tests)}"
            )
        if test in paired_tests and len(groups) != 2:
            raise ValueError(
                f"contrast {name!r}: {test} is a paired test and compares two "
                f"groups, but {len(groups)} are named. A paired test matches "
                "each unit with itself in the other group, which only means "
                "something for two."
            )

        aggregate = raw_dict.get("aggregate")
        aggregate = None if aggregate in (None, "") else str(aggregate)
        if unit == "cell" and aggregate is not None:
            raise ValueError(
                f"contrast {name!r} is at unit={unit!r}, where each row is "
                "already one unit and there is nothing to aggregate; remove "
                "aggregate, which would otherwise imply there was."
            )
        if unit != "cell":
            if aggregate is None:
                raise ValueError(
                    f"contrast {name!r} is at unit={unit!r}, so several cells "
                    "are reduced to one value before testing, and this package "
                    "does not choose the statistic for you; set aggregate to "
                    f"{' or '.join(CONTRAST_AGGREGATES)}"
                )
            if aggregate not in CONTRAST_AGGREGATES:
                raise ValueError(
                    f"contrast {name!r}: aggregate={aggregate!r} is not one "
                    f"this package knows; it accepts "
                    f"{' or '.join(CONTRAST_AGGREGATES)}"
                )

        correction = str(raw_dict.get("correction", "benjamini_hochberg"))
        if correction not in corrections:
            raise ValueError(
                f"contrast {name!r}: correction={correction!r} is not one this "
                f"package offers; it accepts {', '.join(corrections)}"
            )
        alpha = float(raw_dict.get("alpha", 0.05))
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"contrast {name!r}: alpha={alpha} is not between 0 and 1")

        window = raw_dict.get("window")
        return cls(
            name=name,
            table=str(required("table")),
            metrics=tuple(str(metric) for metric in metrics),
            group_by=str(required("group_by")),
            groups=groups,
            unit=unit,
            test=test,
            aggregate=aggregate,
            family=str(raw_dict.get("family", "default")),
            window=None if window in (None, "") else str(window),
            correction=correction,
            alpha=alpha,
            description=str(raw_dict.get("description", "")),
        )


def _contrasts(entries: object) -> list[ContrastConfig]:
    """Parse the ``contrasts`` block, refusing a repeat and a split family.

    A family is the set of results one correction is applied across, so two
    contrasts in one family that disagree about the correction or about alpha
    are refused: whichever was read last would silently decide for both, and the
    corrected p-value is the number anyone quotes.
    """
    if not entries:
        return []
    if not isinstance(entries, list):
        raise TypeError(
            f"contrasts must be a list of contrast blocks, not "
            f"{type(entries).__name__}"
        )
    # Imported here rather than at the top: `contrasts` imports scipy, and a
    # caller that only wanted to read a configuration should not pay for it.
    from analysis.contrasts import CORRECTIONS, PAIRED_TESTS, TESTS

    parsed = [ContrastConfig.from_dict(entry, tuple(TESTS), tuple(PAIRED_TESTS),
                                       tuple(CORRECTIONS)) for entry in entries]
    seen: set[str] = set()
    for contrast in parsed:
        if contrast.name in seen:
            raise ValueError(f"contrast {contrast.name!r} is declared twice")
        seen.add(contrast.name)

    families: dict[str, ContrastConfig] = {}
    for contrast in parsed:
        first = families.setdefault(contrast.family, contrast)
        for setting in ("correction", "alpha"):
            if getattr(first, setting) != getattr(contrast, setting):
                raise ValueError(
                    f"contrasts {first.name!r} and {contrast.name!r} are both "
                    f"in family {contrast.family!r} but declare "
                    f"{setting}={getattr(first, setting)!r} and "
                    f"{setting}={getattr(contrast, setting)!r}. A family is the "
                    "set of results one correction is applied across, so it has "
                    "one correction and one alpha."
                )
    return parsed


@dataclass
class AnalysisConfig:
    dataset: str
    frame_interval_min: float
    movies: list[MovieConfig]
    output_root: Path
    microns_per_pixel: float | None = None
    calibration_tiffs: list[Path] = field(default_factory=list)
    enabled_modules: list[str] = field(default_factory=list)
    module_params: dict = field(default_factory=dict)
    verify_hashes: bool = True
    #: How figures should look. See ``analysis.theme.load_theme``; an empty
    #: block is the house default.
    theme: dict = field(default_factory=dict)
    #: The experimental groups. See ``analysis.conditions``; an empty set means
    #: every movie must declare its own condition or stay unassigned.
    conditions: ConditionSet = field(default_factory=ConditionSet)
    #: Wording per figure, keyed by the figure's slug. The package never writes
    #: a conclusion; anything a figure should say beyond describing its axes is
    #: written here. See ``analysis.figures._text``.
    figures: dict = field(default_factory=dict)
    #: Named stretches of a recording, shared by every movie that does not
    #: declare its own. Empty is the normal case and costs nothing: no windows
    #: means no windowed files, and a run behaves exactly as it did before
    #: windows existed. See :class:`WindowConfig`.
    windows: list[WindowConfig] = field(default_factory=list)
    #: What to compare, named explicitly. Empty is the normal case and writes no
    #: ``statistics.csv``; there is deliberately no "test everything" mode. See
    #: :class:`ContrastConfig`.
    contrasts: list[ContrastConfig] = field(default_factory=list)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        path = Path(path).resolve()
        root = path.parent
        data = json.loads(path.read_text(encoding="utf-8"))

        def resolve(value):
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (root / candidate)

        return cls(
            dataset=data.get("dataset", "unnamed dataset"),
            frame_interval_min=float(data["frame_interval_min"]),
            movies=[MovieConfig.from_dict(entry, root) for entry in data["movies"]],
            output_root=resolve(data.get("output_root", "outputs")),
            microns_per_pixel=data.get("microns_per_pixel"),
            calibration_tiffs=[resolve(p) for p in data.get("calibration_tiffs", [])],
            enabled_modules=list(data.get("enabled_modules", [])),
            module_params=dict(data.get("modules", {})),
            verify_hashes=bool(data.get("verify_hashes", True)),
            theme=dict(data.get("theme", {})),
            conditions=ConditionSet.from_config(data.get("conditions")),
            figures=dict(data.get("figures", {})),
            windows=_windows(data.get("windows"), "windows"),
            contrasts=_contrasts(data.get("contrasts")),
            source_path=path,
        )

    # --------------------------------------------------------------- windows

    def windows_for(self, movie: MovieConfig) -> list[WindowConfig]:
        """The windows that apply to one movie.

        A movie's own block *replaces* the shared one rather than adding to it.
        Merging the two would mean a movie that declared `baseline` at a
        different hour would end up with two windows called `baseline`, and the
        one that won would depend on which list was read first.
        """
        return list(movie.windows) if movie.windows else list(self.windows)

    # ------------------------------------------------------------ conditions

    def assignment(self, movie: MovieConfig) -> Assignment:
        """Which experimental group one movie belongs to, and how that was decided."""
        return self.conditions.assign(
            self.conditions.text_for(movie),
            declared=movie.condition,
            stem=movie.stem,
        )

    def assignments(self) -> list[Assignment]:
        return [self.assignment(movie) for movie in self.movies]

    def condition_problems(self) -> list[str]:
        """Every reason a run must not start. Empty means the design is readable."""
        return [
            f"{a.stem}: {a.problem}" for a in self.assignments() if a.problem is not None
        ]

    # --------------------------------------------------------------- figures

    def figure_problems(self) -> list[str]:
        """Every reason a ``figures`` block is not what its author meant.

        Checked here rather than at build time because the build that would
        catch it is the thirty-sixth one, run an hour later; and because a
        figure whose options are misspelled draws successfully with the
        defaults, which is the failure that looks like success.

        A figure not yet on the schema declares nothing, so nothing about its
        options can be checked. Its wording still can be, and is.
        """
        if not self.figures:
            return []
        # Imported here, not at the top: the measurement half of the package
        # has never imported the plotting half, and a run that only measures
        # must stay possible in an environment with no matplotlib. A
        # configuration with no `figures` block never reaches this line.
        import sys

        figures_dir = Path(__file__).resolve().parent / "figures"
        if str(figures_dir) not in sys.path:
            sys.path.insert(0, str(figures_dir))
        from _text import OPTIONS_KEY, SLOTS
        from _schema import catalogue

        rows = catalogue()
        known = {spec.slug: spec for _, _, spec in rows if spec is not None}
        unconverted = {path.name for _, path, spec in rows if spec is None}

        problems: list[str] = []
        for slug, block in self.figures.items():
            if not isinstance(block, dict):
                problems.append(f"figures.{slug}: expected a block of settings, "
                                f"not {type(block).__name__}")
                continue
            options = block.get(OPTIONS_KEY) or {}
            wording = {key: value for key, value in block.items() if key != OPTIONS_KEY}
            stray = sorted(set(wording) - set(SLOTS))
            if stray:
                problems.append(
                    f"figures.{slug}: {', '.join(repr(k) for k in stray)} is not a "
                    f"text slot; slots are {', '.join(SLOTS)}, plus {OPTIONS_KEY}")
            spec = known.get(slug)
            if spec is None:
                if options:
                    problems.append(
                        f"figures.{slug}.{OPTIONS_KEY}: no figure declares this slug"
                        + (f" yet; {len(unconverted)} builder(s) are not on the "
                           f"schema and declare nothing" if unconverted else "")
                        + f". Known slugs: {', '.join(sorted(known)) or 'none'}")
                continue
            if not isinstance(options, dict):
                problems.append(f"figures.{slug}.{OPTIONS_KEY}: expected a block of "
                                f"option names, not {type(options).__name__}")
                continue
            accepted = {declared.name for declared in spec.options}
            for name in sorted(set(options) - accepted):
                problems.append(
                    f"figures.{slug}.{OPTIONS_KEY}: {name!r} is not an option of this "
                    f"figure; it accepts {', '.join(sorted(accepted)) or 'none'}")
        return problems

    def movie(self, stem: str) -> MovieConfig:
        for entry in self.movies:
            if entry.stem == stem:
                return entry
        known = ", ".join(m.stem for m in self.movies)
        raise KeyError(f"stem {stem!r} is not in the configuration; known stems: {known}")
