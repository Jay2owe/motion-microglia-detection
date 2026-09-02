"""Loading, hashing and time-aligning the inputs.

Two rules hold everywhere in this package:

* nothing is ever written back into a pipeline run folder;
* an input whose fingerprint does not match the configuration stops the run,
  so a figure can never be built from a stack that quietly changed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from analysis.config import (AnalysisConfig, ChannelConfig, MovieConfig,
                             ObjectSetConfig, SideTableConfig)
from analysis.registry import ChannelStack, MeasurementContext, ObjectStack
from analysis.units import resolve_scale

_HASH_CHUNK = 1 << 20

#: Bit flags in the provenance sidecar written by the tracking pass. Kept in
#: step with the same three names in ``code/pipeline.py``; the two halves never
#: import each other. ``ADDED`` is the narrow one: an outline pixel the
#: segmentation never saw, as opposed to ``INFERRED``, which is an outline
#: pixel that was seen but whose owner the tracker worked out.
PROVENANCE_INFERRED = 0b001
PROVENANCE_UNRESOLVED = 0b010
PROVENANCE_ADDED = 0b100


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    return tifffile.imread(path)


def _align_to_labels(stack: np.ndarray, n_label_frames: int, offset: int, what: str) -> np.ndarray:
    """Trim a source-space stack so index ``i`` matches label frame ``i``."""
    if stack.shape[0] == n_label_frames and offset == 0:
        return stack
    end = offset + n_label_frames
    if stack.shape[0] < end:
        raise ValueError(
            f"{what} has {stack.shape[0]} frames but the labels need source frames "
            f"{offset + 1}-{end}; check source_frame_offset"
        )
    return stack[offset:end]


def _lead_axes(series, channel, kind: str = "channel") -> tuple[str, tuple[int, ...]]:
    """The axes in front of ``YX``, refusing anything this loader cannot index.

    Raised rather than guessed, in every case. A stack whose axes cannot be
    read is a stack whose channel cannot be identified, and picking one anyway
    measures a dye nobody asked for and reports it under the name they did.

    ``kind`` is the noun the refusal uses. One indexing routine serves extra
    channels and reference objects, and a message that called an object set a
    channel would send a reader to the wrong block of the settings file.
    """
    axes, shape = series.axes, tuple(series.shape)
    if len(axes) != len(shape) or not axes.endswith("YX"):
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.path} has axes {axes!r}, which "
            "this loader cannot index; it needs a stack whose last two axes are "
            "Y and X"
        )
    return axes[:-2], shape[:-2]


def _channel_pages(series, channel, kind: str = "channel") -> tuple[np.ndarray, int]:
    """Page number of each source frame, and how many source frames there are.

    Pages in a TIFF series run in C order over the axes in front of ``YX``, so
    fixing the channel coordinate and walking the time one gives the page for
    every frame without reading the file into memory first. A two-channel movie
    of a hundred frames is 700 MB; only the frames the labels cover are read,
    and only the channel that was asked for.
    """
    lead_axes, lead_shape = _lead_axes(series, channel, kind)

    if len(lead_axes) == 0:
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.path} is a single image, not "
            f"a stack, so it cannot supply one frame per label frame. A single "
            f"map that applies to the whole recording is an object set with "
            f"static set; a {kind} is measured frame by frame."
        )
    if len(lead_axes) == 1:
        if channel.channel_index is not None:
            raise ValueError(
                f"{kind} {channel.name!r} sets channel_index="
                f"{channel.channel_index}, but {channel.path} has axes "
                f"{series.axes!r} and so has no channel axis to index. Remove "
                "channel_index, or point at the multi-channel file."
            )
        return np.arange(lead_shape[0], dtype=np.int64), int(lead_shape[0])
    if len(lead_axes) > 2:
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.path} has axes {series.axes!r}. "
            "This loader indexes one time axis and at most one channel axis; "
            "save the plane of interest as a (frames, y, x) stack first."
        )

    if "C" not in lead_axes:
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.path} has axes {series.axes!r} "
            "and no C axis, so which of its two leading axes is the channel is "
            "not something this loader may decide. Save the channel as a "
            "(frames, y, x) stack first."
        )
    if channel.channel_index is None:
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.path} has axes {series.axes!r}, "
            f"so it holds {lead_shape[lead_axes.index('C')]} channels and "
            "channel_index says which one to measure. It has no default: the "
            "wrong guess measures the wrong dye and says nothing."
        )
    channel_axis = lead_axes.index("C")
    n_channels = int(lead_shape[channel_axis])
    if not 0 <= channel.channel_index < n_channels:
        raise ValueError(
            f"{kind} {channel.name!r}: channel_index={channel.channel_index} is "
            f"outside {channel.path}, which holds {n_channels} channel(s) "
            f"numbered 0 to {n_channels - 1}"
        )
    time_axis = 1 - channel_axis
    n_source = int(lead_shape[time_axis])
    coordinates = [None, None]
    coordinates[channel_axis] = np.full(n_source, channel.channel_index, dtype=np.int64)
    coordinates[time_axis] = np.arange(n_source, dtype=np.int64)
    pages = np.ravel_multi_index(tuple(coordinates), lead_shape)
    return np.asarray(pages, dtype=np.int64), n_source


def _channel_shifts(channel, n_source: int,
                    kind: str = "channel") -> tuple[np.ndarray, np.ndarray]:
    """Per-source-frame integer shifts, or zeros when none were configured."""
    if channel.shifts is None:
        zeros = np.zeros(n_source, dtype=np.int64)
        return zeros, zeros.copy()
    if not Path(channel.shifts).exists():
        raise FileNotFoundError(
            f"{kind} {channel.name!r}: shifts table not found: {channel.shifts}"
        )
    table = pd.read_csv(channel.shifts)
    missing = [c for c in channel.shift_columns if c not in table.columns]
    if missing:
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.shifts} has no column(s) "
            f"{', '.join(missing)}; it holds {', '.join(map(str, table.columns))}"
        )
    if len(table) < n_source:
        raise ValueError(
            f"{kind} {channel.name!r}: {channel.shifts} has {len(table)} rows but "
            f"{channel.path} has {n_source} frames; a shifts table carries one row "
            "per source frame"
        )
    y_column, x_column = channel.shift_columns
    scale = float(channel.shift_scale)
    # NaN is a frame the registration could not solve, not a frame that did not
    # move. It is carried as zero here and counted as unsampled below, so it
    # shows up as missing rather than as a confident measurement of the wrong
    # place. The first row of a cumulative drift table is normally blank.
    def integers(column: str) -> np.ndarray:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        return np.rint(np.nan_to_num(values[:n_source] * scale)).astype(np.int64)

    return integers(y_column), integers(x_column)


def _storage_dtype(dtype: np.dtype) -> np.dtype:
    """Float wide enough to hold this source exactly, and no wider.

    A channel is stored as float because pixels the alignment cannot reach have
    to be ``NaN``, and float32 holds every integer up to 2**24 exactly - which
    covers every 8- and 16-bit camera there is, at half the memory of float64.
    A wider source, or a float64 one, would silently lose its low bits, so it
    gets float64 instead. Measurements are computed in float64 either way; this
    is only what the stack costs to keep.
    """
    if np.issubdtype(dtype, np.floating) and dtype.itemsize > 4:
        return np.dtype(np.float64)
    if np.issubdtype(dtype, np.integer) and dtype.itemsize > 2:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _sample(frame: np.ndarray, origin_y: int, origin_x: int,
            height: int, width: int, dtype: np.dtype,
            fill: float = np.nan) -> np.ndarray:
    """The analysed field out of one source frame, ``fill`` where there is none.

    Reading the source at the shifted position rather than shifting the crop is
    the whole reason this is not two lines of ``np.roll``: a roll wraps the far
    edge of the field round to the near one, so a cell that drifted off the top
    would be measured against pixels from the bottom. Here it is measured
    against nothing, which is what actually happened.

    ``fill`` is ``NaN`` for a brightness, where there is no value to report,
    and 0 for a label image, where "no object" is the same absence.
    """
    out = np.full((height, width), fill, dtype=dtype)
    y0, y1 = max(origin_y, 0), min(origin_y + height, frame.shape[0])
    x0, x1 = max(origin_x, 0), min(origin_x + width, frame.shape[1])
    if y1 <= y0 or x1 <= x0:
        return out
    out[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x] = frame[y0:y1, x0:x1]
    return out


def _saturation_value(dtype: np.dtype) -> float | None:
    """The largest number this dtype holds, or ``None`` where there is no ceiling.

    A pixel sitting at the ceiling was clipped by the camera, not measured, and
    a channel that spends much of its time there is reporting the acquisition.
    A float source has no such value and gets ``None`` rather than a guess.
    """
    if dtype == np.bool_:
        return 1.0
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def load_channel(channel: ChannelConfig, movie: MovieConfig, n_frames: int,
                 height: int, width: int) -> tuple[ChannelStack, dict]:
    """One extra channel, cropped, trimmed and shifted into the label field.

    Returns the stack and a record of how it was lined up, because every one of
    those settings is a way to measure the right file in the wrong place, and a
    run that cannot say which alignment it used cannot be checked afterwards.
    """
    if not Path(channel.path).exists():
        raise FileNotFoundError(
            f"channel {channel.name!r} not found: {channel.path}"
        )
    offset = movie.source_frame_offset if channel.frame_offset is None else channel.frame_offset
    origin_y, origin_x = channel.crop_origin

    with tifffile.TiffFile(channel.path) as handle:
        series = handle.series[0]
        pages, n_source = _channel_pages(series, channel)
        source_height, source_width = int(series.shape[-2]), int(series.shape[-1])
        if offset + n_frames > n_source:
            raise ValueError(
                f"channel {channel.name!r}: {channel.path} has {n_source} frames "
                f"but the labels need source frames {offset + 1}-{offset + n_frames}; "
                "check frame_offset"
            )
        shift_y, shift_x = _channel_shifts(channel, n_source)
        if (origin_y < 0 or origin_x < 0
                or origin_y + height > source_height or origin_x + width > source_width):
            raise ValueError(
                f"channel {channel.name!r}: crop_origin {channel.crop_origin} puts the "
                f"{height}x{width} analysed field outside the {source_height}x"
                f"{source_width} frames of {channel.path}"
            )
        source_dtype = np.dtype(series.dtype)
        storage = _storage_dtype(source_dtype)
        values = np.empty((n_frames, height, width), dtype=storage)
        for index in range(n_frames):
            source_index = offset + index
            frame = np.asarray(series.asarray(key=int(pages[source_index])))
            if frame.ndim != 2:
                frame = np.squeeze(frame)
            values[index] = _sample(
                frame,
                origin_y - int(shift_y[source_index]),
                origin_x - int(shift_x[source_index]),
                height, width, storage,
            )

    used_y = shift_y[offset:offset + n_frames]
    used_x = shift_x[offset:offset + n_frames]
    unsampled = float(np.isnan(values).mean())
    record = {
        "path": str(channel.path),
        "name": channel.name,
        "description": channel.description,
        "channel_index": channel.channel_index,
        "frame_offset": offset,
        "crop_origin": [int(origin_y), int(origin_x)],
        "source_shape": [n_source, source_height, source_width],
        "source_dtype": str(source_dtype),
        "storage_dtype": str(storage),
        "shifts": None if channel.shifts is None else str(channel.shifts),
        "shift_y_range": [int(used_y.min()), int(used_y.max())],
        "shift_x_range": [int(used_x.min()), int(used_x.max())],
        # What fraction of the analysed field this channel could not supply. Not
        # a failure - drifting off the edge of the source is ordinary - but it
        # bounds how much of any number drawn from this channel is real.
        "unsampled_fraction": round(unsampled, 6),
    }
    stack = ChannelStack(
        name=channel.name,
        values=values,
        source_dtype=str(source_dtype),
        saturation_value=_saturation_value(source_dtype),
        path=str(channel.path),
        description=channel.description,
    )
    return stack, record


def load_valid_mask(movie: MovieConfig, n_frames: int, height: int,
                    width: int) -> tuple[np.ndarray, dict]:
    """Where measuring can be trusted, as a boolean array in label space.

    Label space, not source space: the mask says which pixels of the *analysed*
    field are usable, so it is aligned the way ``unclaimed`` and ``provenance``
    are, with no source offset. Using ``source_frame_offset`` on it would shift
    it by two frames on the pinned movie and produce a complete, wrong,
    plausible answer.

    Non-zero means valid. One frame applies to every label frame; a full stack
    is used frame by frame. Anything else is refused naming both shapes,
    because a mask of the wrong size is a mask of a different experiment.
    """
    mask = _read(Path(movie.valid_mask))
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 2:
        if mask.shape != (height, width):
            raise ValueError(
                f"valid_mask field {mask.shape} does not match label field "
                f"{(height, width)}"
            )
    elif mask.ndim == 3:
        mask = _align_to_labels(mask, n_frames, 0, "valid_mask")
        if mask.shape != (n_frames, height, width):
            raise ValueError(
                f"valid_mask is {mask.shape} but the labels are "
                f"{(n_frames, height, width)}"
            )
    else:
        raise ValueError(
            f"valid_mask must be a single (y, x) frame or a (frames, y, x) "
            f"stack matching the labels, got shape {mask.shape}"
        )

    valid = mask.astype(bool)
    total = int(valid.sum())
    if total == 0:
        # Refused at load rather than left to divide by zero eighty lines
        # later, where the message would be about a square root.
        raise ValueError(
            f"valid_mask {movie.valid_mask} marks no pixel as measurable, so "
            "every density and share would divide by zero. A mask says where "
            "measuring can be trusted; an empty one says nothing can."
        )
    per_frame = ([int(valid[index].sum()) for index in range(valid.shape[0])]
                 if valid.ndim == 3 else [total] * n_frames)
    field_px = height * width
    record = {
        "path": str(movie.valid_mask),
        "per_frame": valid.ndim == 3,
        "field_px": int(field_px),
        "valid_px_min": int(min(per_frame)),
        "valid_px_max": int(max(per_frame)),
        "valid_share_min": round(min(per_frame) / field_px, 6),
        "valid_share_max": round(max(per_frame) / field_px, 6),
    }
    return valid, record


def _object_pages(series, spec: ObjectSetConfig) -> tuple[np.ndarray, int]:
    """Which page of the file each frame of this object set comes from.

    A moving set indexes exactly as a channel does. A static one is a single
    map, so the only question is which page holds it, and anything with more
    than one is refused: a file with a time axis and ``static`` set means the
    two disagree about what the file is, and guessing which to believe would
    apply one arbitrary frame's shapes to the whole recording.
    """
    if not spec.static:
        return _channel_pages(series, spec, "object set")

    lead_axes, lead_shape = _lead_axes(series, spec, "object set")
    if len(lead_axes) == 0:
        if spec.channel_index is not None:
            raise ValueError(
                f"object set {spec.name!r} sets channel_index="
                f"{spec.channel_index}, but {spec.path} has axes "
                f"{series.axes!r} and so has no channel axis to index"
            )
        return np.zeros(1, dtype=np.int64), 1
    if len(lead_axes) == 1 and "C" in lead_axes:
        if spec.channel_index is None:
            raise ValueError(
                f"object set {spec.name!r}: {spec.path} has axes "
                f"{series.axes!r}, so it holds {lead_shape[0]} channels and "
                "channel_index says which one holds the shapes"
            )
        if not 0 <= spec.channel_index < int(lead_shape[0]):
            raise ValueError(
                f"object set {spec.name!r}: channel_index={spec.channel_index} "
                f"is outside {spec.path}, which holds {lead_shape[0]} channel(s)"
            )
        return np.array([spec.channel_index], dtype=np.int64), 1
    raise ValueError(
        f"object set {spec.name!r} is static, but {spec.path} has axes "
        f"{series.axes!r} and so holds more than one map. A static set is one "
        "drawing applied to every frame; drop static, or save the single map."
    )


def load_object_set(spec: ObjectSetConfig, movie: MovieConfig, n_frames: int,
                    height: int, width: int) -> tuple[ObjectStack, dict]:
    """One set of reference shapes, cropped, trimmed and shifted into the field.

    The same alignment as an extra channel, with two differences that both
    follow from a label being an identity rather than a brightness: the stack
    is integer, and a pixel the alignment could not reach is 0 with the count
    kept in the record rather than a value in the array.
    """
    if not Path(spec.path).exists():
        raise FileNotFoundError(f"object set {spec.name!r} not found: {spec.path}")
    offset = (0 if spec.static
              else (movie.source_frame_offset if spec.frame_offset is None
                    else spec.frame_offset))
    origin_y, origin_x = spec.crop_origin
    supplied = 1 if spec.static else n_frames

    with tifffile.TiffFile(spec.path) as handle:
        series = handle.series[0]
        pages, n_source = _object_pages(series, spec)
        source_height, source_width = int(series.shape[-2]), int(series.shape[-1])
        if offset + supplied > n_source:
            raise ValueError(
                f"object set {spec.name!r}: {spec.path} has {n_source} frames "
                f"but the labels need source frames {offset + 1}-"
                f"{offset + supplied}; check frame_offset"
            )
        shift_y, shift_x = _channel_shifts(spec, n_source, "object set")
        if (origin_y < 0 or origin_x < 0
                or origin_y + height > source_height
                or origin_x + width > source_width):
            raise ValueError(
                f"object set {spec.name!r}: crop_origin {spec.crop_origin} puts "
                f"the {height}x{width} analysed field outside the "
                f"{source_height}x{source_width} frames of {spec.path}"
            )
        values = np.zeros((supplied, height, width), dtype=np.int32)
        unreachable = 0
        for index in range(supplied):
            source_index = offset + index
            frame = np.asarray(series.asarray(key=int(pages[source_index])))
            if frame.ndim != 2:
                frame = np.squeeze(frame)
            top = origin_y - int(shift_y[source_index])
            left = origin_x - int(shift_x[source_index])
            values[index] = _sample(frame, top, left, height, width,
                                    np.dtype(np.int32), fill=0)
            # Counted rather than marked, because the array has no spare value
            # to mark it with - see ObjectStack. Computed from the overlap
            # rather than from the array so a genuine object 0 is not involved.
            covered_y = max(0, min(top + height, frame.shape[0]) - max(top, 0))
            covered_x = max(0, min(left + width, frame.shape[1]) - max(left, 0))
            unreachable += height * width - covered_y * covered_x

    if np.any(values < 0):
        raise ValueError(
            f"object set {spec.name!r}: {spec.path} holds negative values, so "
            "it is not a label image. An object set is a picture of which "
            "object is where, with 0 for none."
        )
    used_y = shift_y[offset:offset + supplied]
    used_x = shift_x[offset:offset + supplied]
    labels_present = sorted(int(value) for value in np.unique(values) if value)
    record = {
        "path": str(spec.path),
        "name": spec.name,
        "description": spec.description,
        "static": bool(spec.static),
        "channel_index": spec.channel_index,
        "frame_offset": None if spec.static else offset,
        "crop_origin": [int(origin_y), int(origin_x)],
        "source_shape": [n_source, source_height, source_width],
        "source_dtype": str(np.dtype(series.dtype)),
        "frames_supplied": int(supplied),
        "shifts": None if spec.shifts is None else str(spec.shifts),
        "shift_y_range": [int(used_y.min()), int(used_y.max())],
        "shift_x_range": [int(used_x.min()), int(used_x.max())],
        "object_labels": len(labels_present),
        # Pixels of the analysed field this file could not supply. They read as
        # empty ground, which is why the count is here: it is the only place the
        # difference between "no object" and "could not look" survives.
        "unreachable_px": int(unreachable),
        "unreachable_fraction": round(unreachable / float(supplied * height * width), 6),
    }
    stack = ObjectStack(
        name=spec.name, values=values, static=bool(spec.static),
        path=str(spec.path), description=spec.description,
    )
    return stack, record


def _side_prefixed(name: str, column: object) -> str:
    """A user column under the side table's own name.

    A log with a column called ``area_px`` is not hypothetical, and a user
    column silently replacing a measured one is the failure this package
    refuses everywhere else. The prefix makes the clash unlikely; the check in
    :func:`load_side_table` makes it impossible. A column the user already
    prefixed themselves is left alone rather than doubled.
    """
    text = str(column)
    return text if text.startswith(f"{name}_") else f"{name}_{text}"


def load_side_table(spec: SideTableConfig, movie: MovieConfig, n_frames: int,
                    identities: list[int],
                    taken: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """One user table, keyed and checked against the movie it describes.

    Returns the table - key column renamed to the key it joins on, every other
    column prefixed - and a record of what was read, what matched and what did
    not. Coverage is reported rather than enforced: a log covering 90 of 99
    frames is ordinary, a log covering none is a mistake, and the counts are
    how a reader tells the two apart afterwards.
    """
    # Imported here rather than at the top: the vocabulary is built by
    # importing every module, which pulls in scipy and scikit-image. A run
    # already has them loaded by the time this is called, and a caller that
    # only wanted to read a CSV should not pay for them at import time.
    from analysis.registry import SHARED_COLUMNS, declared_columns

    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(f"side table {spec.name!r} not found: {path}")
    table = pd.read_csv(path)
    key_column = spec.key
    if key_column not in table.columns:
        raise ValueError(
            f"side table {spec.name!r}: {path} has no column {key_column!r} to "
            f"key on; it holds {', '.join(map(str, table.columns))}"
        )

    keys = pd.to_numeric(table[key_column], errors="coerce")
    if spec.keyed_on == "frame_index" and spec.key_space == "source":
        # Label frame 0 is source ImageJ frame ``source_frame_offset + 1``, so
        # a log written in the microscope's numbering moves twice: once because
        # ImageJ counts from one, once for the frames the analysis starts after.
        keys = keys - 1 - int(movie.source_frame_offset)
    usable = keys.notna()
    keyed = table.loc[usable].copy()
    keyed[spec.keyed_on] = keys[usable].to_numpy(dtype=np.int64)

    # A merge against a repeated key is a cross product, not a join, and it
    # fails by silently multiplying rows rather than by raising. Refused here,
    # where the file is still in front of us and the message can be specific.
    duplicated = keyed[spec.keyed_on].duplicated(keep=False)
    if bool(duplicated.any()):
        repeats = sorted(set(keyed.loc[duplicated, spec.keyed_on].tolist()))
        shown = ", ".join(str(value) for value in repeats[:5])
        raise ValueError(
            f"side table {spec.name!r}: {path} repeats {len(repeats)} "
            f"{spec.keyed_on} value(s) over {int(duplicated.sum())} rows "
            f"({shown}{', ...' if len(repeats) > 5 else ''}). A side table is "
            "joined on its key, and a repeated key multiplies rows instead of "
            "attaching a column."
        )

    wanted = [column for column in table.columns if column != key_column]
    if spec.columns is not None:
        missing = [column for column in spec.columns if column not in table.columns]
        if missing:
            raise ValueError(
                f"side table {spec.name!r}: {path} has no column(s) "
                f"{', '.join(missing)}; it holds "
                f"{', '.join(map(str, table.columns))}"
            )
        wanted = [column for column in spec.columns if column != key_column]

    renamed = {column: _side_prefixed(spec.name, column) for column in wanted}
    reserved = set(declared_columns()) | set(SHARED_COLUMNS) | set(taken or ())
    clashes = sorted({new for new in renamed.values() if new in reserved})
    if clashes:
        raise ValueError(
            f"side table {spec.name!r} would contribute {', '.join(clashes)}, "
            "which this package already writes. Rename the column in your file, "
            "or the measured value and the supplied one end up under one name "
            "and whichever joined last wins."
        )
    repeated_here = sorted({new for new in renamed.values()
                            if list(renamed.values()).count(new) > 1})
    if repeated_here:
        raise ValueError(
            f"side table {spec.name!r}: {', '.join(repeated_here)} would be "
            "contributed twice, because two of its columns prefix to one name"
        )

    joined = keyed[[spec.keyed_on, *wanted]].rename(columns=renamed)
    joined = joined.reset_index(drop=True)

    movie_keys = (set(range(int(n_frames))) if spec.keyed_on == "frame_index"
                  else {int(value) for value in identities})
    present = {int(value) for value in joined[spec.keyed_on].tolist()}
    matched = present & movie_keys
    record = {
        "path": str(path),
        "name": spec.name,
        "description": spec.description,
        "keyed_on": spec.keyed_on,
        "key_space": spec.key_space if spec.keyed_on == "frame_index" else None,
        "key_column": key_column,
        "rows_read": int(len(table)),
        # A key that is not a number at all - a blank cell, a stray header - is
        # counted rather than guessed at. It cannot join, and saying how many
        # there were is the difference between a partial log and a broken one.
        "rows_unusable_key": int(len(table) - len(keyed)),
        "columns": list(renamed.values()),
        # The three coverage counts. ``keys_matched`` is how much of the file
        # landed, ``keys_unmatched`` is how much described something this movie
        # does not have, and ``movie_rows_covered`` is how much of the movie the
        # file speaks for - out of ``movie_rows``, which is the denominator a
        # reader would otherwise have to go and find.
        "keys_matched": int(len(matched)),
        "keys_unmatched": int(len(present - movie_keys)),
        "movie_rows": int(len(movie_keys)),
        "movie_rows_covered": int(len(matched)),
    }
    return joined, record


def load_movie(config: AnalysisConfig, movie: MovieConfig) -> tuple[MeasurementContext, dict]:
    """Load one movie into a :class:`MeasurementContext` plus a provenance record."""
    provenance: dict = {"stem": movie.stem, "condition": movie.condition, "inputs": {}}

    labels = _read(movie.labels)
    if labels.ndim != 3:
        raise ValueError(f"labels must be a (frames, y, x) stack, got shape {labels.shape}")
    n_frames, height, width = labels.shape

    raw_full = _read(movie.raw)
    if raw_full.ndim != 3:
        raise ValueError(f"raw signal must be a (frames, y, x) stack, got shape {raw_full.shape}")
    raw = _align_to_labels(raw_full, n_frames, movie.source_frame_offset, "registered raw")
    if raw.shape[1:] != labels.shape[1:]:
        raise ValueError(f"raw field {raw.shape[1:]} does not match label field {labels.shape[1:]}")

    unclaimed = None
    if movie.unclaimed and Path(movie.unclaimed).exists():
        unclaimed = _align_to_labels(_read(movie.unclaimed), n_frames, 0, "unclaimed")

    evidence = None
    if movie.evidence and Path(movie.evidence).exists():
        evidence_full = _read(movie.evidence)
        if evidence_full.ndim != 4:
            raise ValueError(
                f"motion evidence must be (transitions, channels, y, x), got {evidence_full.shape}"
            )
        # Evidence transition j covers source frames j -> j+1, so the transition
        # out of label frame i is source transition i + offset. The last label
        # frame has no outgoing transition, so only n_frames - 1 are needed.
        evidence = _align_to_labels(
            evidence_full, n_frames - 1, movie.source_frame_offset, "motion evidence"
        )

    # The sidecar is written in accepted-label space, like the unclaimed
    # ledger, so it is aligned with offset 0 and not with the raw stack's
    # source offset. Getting that wrong would shift every flag by two frames.
    inferred = unresolved = added = None
    if movie.provenance and Path(movie.provenance).exists():
        packed = _align_to_labels(_read(movie.provenance), n_frames, 0, "provenance")
        if packed.shape[1:] != labels.shape[1:]:
            raise ValueError(
                f"provenance field {packed.shape[1:]} does not match label field "
                f"{labels.shape[1:]}"
            )
        inferred = (packed & PROVENANCE_INFERRED).astype(bool)
        unresolved = (packed & PROVENANCE_UNRESOLVED).astype(bool)
        added = (packed & PROVENANCE_ADDED).astype(bool)

    # Label space, like the sidecar above and unlike the raw stack.
    valid = None
    valid_record = None
    if movie.valid_mask and Path(movie.valid_mask).exists():
        valid, valid_record = load_valid_mask(movie, n_frames, height, width)

    for key, path in (
        ("labels", movie.labels),
        ("raw", movie.raw),
        ("unclaimed", movie.unclaimed),
        ("evidence", movie.evidence),
        ("provenance", movie.provenance),
        ("valid_mask", movie.valid_mask),
    ):
        if not path or not Path(path).exists():
            continue
        digest = sha256_of(path)
        expected = movie.expected_sha256.get(key)
        record = {
            "path": str(path),
            "sha256": digest,
            "bytes": Path(path).stat().st_size,
        }
        if expected:
            record["expected_sha256"] = expected
            record["verified"] = digest == expected
            if config.verify_hashes and digest != expected:
                raise ValueError(
                    f"{key} for {movie.stem} has fingerprint {digest} but the configuration "
                    f"pins {expected}; refusing to analyse a changed input"
                )
        provenance["inputs"][key] = record
    if valid_record is not None:
        provenance["inputs"]["valid_mask"].update(valid_record)

    # Extra channels last, because every one of them is checked against the
    # label field that the lines above established.
    channels: dict[str, ChannelStack] = {}
    for spec in movie.channels:
        stack, record = load_channel(spec, movie, n_frames, height, width)
        digest = sha256_of(spec.path)
        record["sha256"] = digest
        if spec.expected_sha256:
            record["expected_sha256"] = spec.expected_sha256
            record["verified"] = digest == spec.expected_sha256
            if config.verify_hashes and digest != spec.expected_sha256:
                raise ValueError(
                    f"channel {spec.name!r} for {movie.stem} has fingerprint {digest} "
                    f"but the configuration pins {spec.expected_sha256}; refusing to "
                    "analyse a changed input"
                )
        channels[spec.name] = stack
        provenance["inputs"][f"channel:{spec.name}"] = record

    calibration_candidates = [movie.raw, *config.calibration_tiffs]
    scale = resolve_scale(
        minutes_per_frame=config.frame_interval_min,
        configured_microns_per_pixel=config.microns_per_pixel,
        candidate_tiffs=[Path(p) for p in calibration_candidates],
    )

    # Reference objects, checked against the label field the lines above
    # established, exactly as the channels were.
    objects: dict[str, ObjectStack] = {}
    for spec in movie.objects:
        stack, record = load_object_set(spec, movie, n_frames, height, width)
        digest = sha256_of(spec.path)
        record["sha256"] = digest
        if spec.expected_sha256:
            record["expected_sha256"] = spec.expected_sha256
            record["verified"] = digest == spec.expected_sha256
            if config.verify_hashes and digest != spec.expected_sha256:
                raise ValueError(
                    f"object set {spec.name!r} for {movie.stem} has fingerprint "
                    f"{digest} but the configuration pins {spec.expected_sha256}; "
                    "refusing to analyse a changed input"
                )
        objects[spec.name] = stack
        provenance["inputs"][f"objects:{spec.name}"] = record

    identities = sorted(int(v) for v in np.unique(labels) if v)

    # Side tables last of all, because the coverage counts are against the
    # frames and the identities the lines above established. ``taken`` grows as
    # they are read, so two tables cannot both contribute one column name.
    side: dict[str, pd.DataFrame] = {}
    taken: set[str] = set()
    for spec in movie.side_tables:
        table, record = load_side_table(spec, movie, n_frames, identities, taken)
        digest = sha256_of(spec.path)
        record["sha256"] = digest
        if spec.expected_sha256:
            record["expected_sha256"] = spec.expected_sha256
            record["verified"] = digest == spec.expected_sha256
            if config.verify_hashes and digest != spec.expected_sha256:
                raise ValueError(
                    f"side table {spec.name!r} for {movie.stem} has fingerprint "
                    f"{digest} but the configuration pins {spec.expected_sha256}; "
                    "refusing to analyse a changed input"
                )
        side[spec.name] = table
        taken.update(record["columns"])
        provenance["inputs"][f"side:{spec.name}"] = record

    # ``DerivedModule.derive`` is handed the tables and the context, never the
    # configuration, so a per-movie folder has to travel as a module parameter.
    params = dict(config.module_params)
    history = {**params.get("history", {}),
               # Some decision tables are written beside the accepted labels
               # rather than into the decision folder, so the module is told
               # where the labels came from as well.
               "labels_directory": str(Path(movie.labels).parent)}
    if movie.history:
        history["directory"] = str(movie.history)
        provenance["inputs"]["history"] = {
            "path": str(movie.history),
            "is_directory": Path(movie.history).is_dir(),
        }
    params["history"] = history

    context = MeasurementContext(
        stem=movie.stem,
        labels=labels,
        raw=raw,
        scale=scale,
        identities=identities,
        unclaimed=unclaimed,
        evidence=evidence,
        inferred=inferred,
        unresolved=unresolved,
        added=added,
        source_frame_offset=movie.source_frame_offset,
        channels=channels,
        objects=objects,
        side=side,
        valid=valid,
        params=params,
    )

    provenance["scale"] = scale.describe()
    provenance["field"] = {"frames": n_frames, "height": height, "width": width}
    provenance["identity_count"] = len(identities)
    provenance["hours_covered"] = scale.hours(n_frames - 1)
    return context, provenance
