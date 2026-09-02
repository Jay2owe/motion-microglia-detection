"""Lining an extra channel up with the outlines, checked pixel by pixel.

Every other test in this package checks what a number means. These check where
a number came *from*, which for an extra channel is the whole problem: the file
is right, the arithmetic is right, and the answer is still about the wrong dye
in the wrong frame at the wrong end of the dish, with nothing anywhere looking
broken.

So each setting is given a file where the correct answer is known by
construction, and asserted exactly rather than approximately. The contract
under test is one line::

    analysed[i, y, x] == source[frame_offset + i,
                                crop_origin[0] + y - shift_y[i],
                                crop_origin[1] + x - shift_x[i]]

The refusals are tested as carefully as the successes. A loader that guesses a
channel index when the file has two channels is worse than one that fails,
because the guess is reported with the same confidence as a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from analysis.config import ChannelConfig, MovieConfig
from analysis.io import load_channel

SOURCE_FRAMES, SOURCE_HEIGHT, SOURCE_WIDTH = 8, 30, 40
FIELD_HEIGHT, FIELD_WIDTH = 10, 12


def _ramp(frames: int = SOURCE_FRAMES, channels: int | None = None) -> np.ndarray:
    """A stack in which every pixel says where and when it is.

    ``value = 10000*channel + 1000*frame + 10*y + x`` so a single wrong number
    names the mistake: a slip in time, in space or in the channel each land in
    their own digits.
    """
    y, x = np.mgrid[0:SOURCE_HEIGHT, 0:SOURCE_WIDTH]
    plane = (10 * y + x).astype(np.uint32)
    stack = np.stack([plane + 1000 * t for t in range(frames)])
    if channels is None:
        return stack.astype(np.uint16)
    # uint16, not uint32: the ImageJ hyperstack format this writes carries only
    # 8-bit, 16-bit and float, and a TCYX page order is the thing under test.
    return np.stack(
        [np.stack([stack[t] + 10000 * c for c in range(channels)]) for t in range(frames)]
    ).astype(np.uint16)


def _write(path, array, axes: str | None = None):
    if axes:
        tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})
    else:
        tifffile.imwrite(path, array)
    return path


def _movie(offset: int = 0) -> MovieConfig:
    return MovieConfig(stem="t", labels="labels.tif", raw="raw.tif",
                       source_frame_offset=offset)


def _load(channel: ChannelConfig, movie: MovieConfig | None = None, frames: int = 4):
    return load_channel(channel, movie or _movie(), frames, FIELD_HEIGHT, FIELD_WIDTH)


# ------------------------------------------------------------------ the happy path

def test_a_plain_stack_already_in_the_label_field_needs_two_settings(tmp_path):
    """Name and path. Everything else defaults to "no adjustment"."""
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, record = _load(ChannelConfig(name="green", path=path))

    assert stack.values.shape == (4, FIELD_HEIGHT, FIELD_WIDTH)
    source = _ramp()
    assert np.array_equal(stack.values, source[:4, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["unsampled_fraction"] == 0.0
    assert record["shift_y_range"] == [0, 0]


def test_the_crop_picks_the_field_out_of_a_larger_frame(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, _ = _load(ChannelConfig(name="green", path=path, crop_origin=(7, 13)))

    source = _ramp()
    assert np.array_equal(
        stack.values, source[:4, 7:7 + FIELD_HEIGHT, 13:13 + FIELD_WIDTH])


def test_the_frame_offset_is_inherited_from_the_movie_and_overridable(tmp_path):
    """A channel from the same acquisition as the raw stack needs nothing said."""
    path = _write(tmp_path / "plain.tif", _ramp())
    source = _ramp()

    inherited, record = _load(ChannelConfig(name="green", path=path), _movie(offset=3))
    assert np.array_equal(inherited.values[0], source[3, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 3

    overridden, record = _load(
        ChannelConfig(name="green", path=path, frame_offset=1), _movie(offset=3))
    assert np.array_equal(overridden.values[0], source[1, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 1


def test_the_named_channel_of_a_hyperstack_is_the_one_measured(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    source = _ramp(channels=2)

    for index in (0, 1):
        stack, record = _load(
            ChannelConfig(name="green", path=path, channel_index=index))
        assert np.array_equal(
            stack.values, source[:4, index, :FIELD_HEIGHT, :FIELD_WIDTH]), index
        assert record["channel_index"] == index


def test_the_alignment_contract_holds_pixel_for_pixel(tmp_path):
    """The one line in the docstring, asserted against the file it describes.

    ``shift_scale`` is negative here on purpose: a registration log usually
    records the drift it measured rather than the correction it applied, and
    getting that sign backwards moves every frame twice as far in the wrong
    direction while still producing a full set of numbers.
    """
    path = _write(tmp_path / "plain.tif", _ramp())
    drift = pd.DataFrame({
        "t": range(SOURCE_FRAMES),
        "cum_dy": [0.0, -1.4, -2.6, 1.0, 0.0, 0.0, 0.0, 0.0],
        "cum_dx": [0.0, 3.2, -0.4, 2.5, 0.0, 0.0, 0.0, 0.0],
    })
    shifts = tmp_path / "drift.csv"
    drift.to_csv(shifts, index=False)

    channel = ChannelConfig(
        name="green", path=path, crop_origin=(9, 11), shifts=shifts,
        shift_columns=("cum_dy", "cum_dx"), shift_scale=-1.0)
    stack, record = _load(channel)

    source = _ramp()
    shift_y = np.rint(-drift["cum_dy"].to_numpy()).astype(int)
    shift_x = np.rint(-drift["cum_dx"].to_numpy()).astype(int)
    for index in range(4):
        top = 9 - shift_y[index]
        left = 11 - shift_x[index]
        expected = source[index, top:top + FIELD_HEIGHT, left:left + FIELD_WIDTH]
        assert np.array_equal(stack.values[index], expected), index

    assert record["shift_y_range"] == [int(shift_y[:4].min()), int(shift_y[:4].max())]


def test_a_pixel_the_alignment_cannot_reach_is_blank_not_borrowed(tmp_path):
    """A wrapped edge would measure the far side of the field and say nothing.

    This is the one place a roll would have been two lines and wrong: a cell
    that drifted off the top would come back measured against the bottom, at
    full confidence, with no column anywhere recording it.
    """
    path = _write(tmp_path / "plain.tif", _ramp())
    shifts = tmp_path / "drift.csv"
    pd.DataFrame({"shift_y": [5] * SOURCE_FRAMES,
                  "shift_x": [0] * SOURCE_FRAMES}).to_csv(shifts, index=False)

    stack, record = _load(
        ChannelConfig(name="green", path=path, crop_origin=(0, 0), shifts=shifts))

    # Sampling starts five rows above the top of the source, so five rows are
    # missing and everything below them is real.
    assert np.isnan(stack.values[:, :5, :]).all()
    assert np.isfinite(stack.values[:, 5:, :]).all()
    assert record["unsampled_fraction"] == pytest.approx(5 / FIELD_HEIGHT)


def test_the_stack_is_kept_narrow_when_the_source_allows_it(tmp_path):
    """float32 is exact up to 2**24, which is every ordinary camera."""
    narrow = _write(tmp_path / "narrow.tif", _ramp())
    wide = _write(tmp_path / "wide.tif", _ramp().astype(np.uint32))

    small, record = _load(ChannelConfig(name="green", path=narrow))
    assert small.values.dtype == np.float32
    assert small.saturation_value == float(np.iinfo(np.uint16).max)
    assert record["storage_dtype"] == "float32"

    large, record = _load(ChannelConfig(name="green", path=wide))
    assert large.values.dtype == np.float64
    assert large.saturation_value == float(np.iinfo(np.uint32).max)
    assert record["storage_dtype"] == "float64"


# ----------------------------------------------------------------- the refusals

def test_a_hyperstack_without_a_channel_index_is_refused(tmp_path):
    """No default, because the wrong guess measures the wrong dye in silence."""
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    with pytest.raises(ValueError, match="channel_index"):
        _load(ChannelConfig(name="green", path=path))


def test_a_channel_index_on_a_single_channel_file_is_refused(tmp_path):
    """Naming a plane a file does not have is a mistake, not a no-op."""
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="no channel axis"):
        _load(ChannelConfig(name="green", path=path, channel_index=1))


def test_a_channel_index_past_the_end_is_refused(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    with pytest.raises(ValueError, match="outside"):
        _load(ChannelConfig(name="green", path=path, channel_index=4))


def test_a_crop_that_leaves_the_source_is_refused(tmp_path):
    """Refused rather than padded: the field is simply not in this file."""
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="crop_origin"):
        _load(ChannelConfig(name="green", path=path,
                            crop_origin=(SOURCE_HEIGHT - 2, 0)))


def test_a_channel_too_short_for_the_labels_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="frame_offset"):
        _load(ChannelConfig(name="green", path=path, frame_offset=6), frames=4)


def test_a_shifts_table_that_does_not_cover_the_movie_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    shifts = tmp_path / "short.csv"
    pd.DataFrame({"shift_y": [0, 0], "shift_x": [0, 0]}).to_csv(shifts, index=False)
    with pytest.raises(ValueError, match="one row per source frame"):
        _load(ChannelConfig(name="green", path=path, shifts=shifts))


def test_a_shifts_table_missing_its_columns_names_what_it_has(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    shifts = tmp_path / "wrong.csv"
    pd.DataFrame({"dy": [0] * SOURCE_FRAMES,
                  "dx": [0] * SOURCE_FRAMES}).to_csv(shifts, index=False)
    with pytest.raises(ValueError, match="dy, dx"):
        _load(ChannelConfig(name="green", path=path, shifts=shifts))


# -------------------------------------------------------------- the declaration

def _resolve(value):
    return None if value in (None, "") else value


@pytest.mark.parametrize("name", ["Green", "green channel", "2nd", "", "raw", "labels"])
def test_a_channel_name_that_cannot_be_a_column_or_a_series_is_refused(name):
    """It becomes a value in the ``channel`` column and a label on a figure."""
    with pytest.raises(ValueError):
        ChannelConfig.from_dict({"name": name, "path": "x.tif"}, _resolve)


def test_two_channels_with_one_name_are_refused():
    """The second would replace the first and the tables would look fine."""
    from analysis.config import _channels

    entries = [{"name": "green", "path": "a.tif"}, {"name": "green", "path": "b.tif"}]
    with pytest.raises(ValueError, match="declared twice"):
        _channels(entries, _resolve, "stem")


def test_the_defaults_are_no_adjustment():
    """A channel already beside the raw stack is two lines of configuration."""
    channel = ChannelConfig.from_dict({"name": "green", "path": "x.tif"}, _resolve)
    assert channel.channel_index is None
    assert channel.frame_offset is None
    assert channel.crop_origin == (0, 0)
    assert channel.shifts is None
    assert channel.shift_scale == 1.0


def test_a_channel_that_reaches_nothing_measures_nothing_rather_than_failing():
    """An alignment pointed at the wrong corner of a bigger file is a real state.

    Every pixel comes back blank, so the columns that would have held a
    brightness are never created. That has to arrive as an empty result, not as
    a ``KeyError`` from the roll-up three lines later - the run should say the
    channel measured nothing, which is a fact about the alignment.
    """
    from analysis.modules.channels import measure
    from analysis.registry import ChannelStack
    from analysis.test_column_declarations import _movie

    context = _movie()
    blank = np.full_like(context.channels["extra"].values, np.nan)
    context.channels = {"extra": ChannelStack(
        name="extra", values=blank, source_dtype="uint16",
        saturation_value=65535.0, path="synthetic")}

    tables = measure(context)
    assert tables["channel_tracks"].empty
    assert not tables["channels"].empty                 # the cells are still there
    assert tables["channels"]["channel_px"].sum() == 0
    assert tables["channels"]["channel_missing_px"].sum() > 0
    assert tables["channel_frame"]["channel_field_valid_px"].sum() == 0
