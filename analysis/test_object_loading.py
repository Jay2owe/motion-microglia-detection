"""Lining a set of reference shapes up with the outlines, checked pixel by pixel.

The same problem as an extra channel and the same contract, with one thing
added: an object set is a *label image*, so a mistake does not show up as a
brightness that looks slightly wrong. It shows up as a shape in the wrong
place, measured confidently, with the right number of objects and the right
areas. Nothing about the output would look broken.

So the alignment is asserted against a file whose pixels say where they are::

    analysed[i, y, x] == source[frame_offset + i,
                                crop_origin[0] + y - shift_y[i],
                                crop_origin[1] + x - shift_x[i]]

and every refusal is tested as carefully as every success.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from analysis.config import MovieConfig, ObjectSetConfig, _object_sets
from analysis.io import load_object_set

SOURCE_FRAMES, SOURCE_HEIGHT, SOURCE_WIDTH = 8, 30, 40
FIELD_HEIGHT, FIELD_WIDTH = 10, 12


def _ramp(frames: int = SOURCE_FRAMES, channels: int | None = None) -> np.ndarray:
    """A label stack in which every pixel says where and when it is.

    ``value = 10000*channel + 1000*frame + 10*y + x``, so one wrong number
    names its own mistake: a slip in time, in space or in the channel each land
    in their own digits. They are not plausible object numbers and are not
    meant to be - this file is under test as a picture, not as a set of shapes.
    """
    y, x = np.mgrid[0:SOURCE_HEIGHT, 0:SOURCE_WIDTH]
    plane = (10 * y + x).astype(np.uint32)
    stack = np.stack([plane + 1000 * t for t in range(frames)])
    if channels is None:
        return stack.astype(np.uint16)
    # uint16, because the ImageJ hyperstack format this writes carries 8-bit,
    # 16-bit and float only, and the page order is the thing under test.
    return np.stack(
        [np.stack([stack[t] + 10000 * c for c in range(channels)]) for t in range(frames)]
    ).astype(np.uint16)


def _blobs(frames: int = SOURCE_FRAMES) -> np.ndarray:
    """Two numbered shapes: one that stays put and one that moves."""
    stack = np.zeros((frames, SOURCE_HEIGHT, SOURCE_WIDTH), dtype=np.uint16)
    for frame in range(frames):
        stack[frame, 2:6, 2:5] = 1                      # 12 px, fixed, mid-field
        stack[frame, 0:3, 8 + frame:11 + frame] = 2     # 9 px, drifting, at the top
    return stack


def _write(path, array, axes: str | None = None):
    if axes:
        tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})
    else:
        tifffile.imwrite(path, array)
    return path


def _movie(offset: int = 0) -> MovieConfig:
    return MovieConfig(stem="t", labels="labels.tif", raw="raw.tif",
                       source_frame_offset=offset)


def _load(spec: ObjectSetConfig, movie: MovieConfig | None = None, frames: int = 4):
    return load_object_set(spec, movie or _movie(), frames, FIELD_HEIGHT, FIELD_WIDTH)


# ------------------------------------------------------------------ the happy path

def test_a_plain_stack_already_in_the_label_field_needs_two_settings(tmp_path):
    """Name and path. Everything else defaults to "no adjustment"."""
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, record = _load(ObjectSetConfig(name="vessels", path=path))

    assert stack.values.shape == (4, FIELD_HEIGHT, FIELD_WIDTH)
    assert stack.values.dtype == np.int32
    assert stack.static is False
    assert np.array_equal(stack.values,
                          _ramp()[:4, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["unreachable_px"] == 0
    assert record["frames_supplied"] == 4


def test_the_crop_picks_the_field_out_of_a_larger_frame(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    stack, _ = _load(ObjectSetConfig(name="vessels", path=path, crop_origin=(7, 13)))
    assert np.array_equal(
        stack.values, _ramp()[:4, 7:7 + FIELD_HEIGHT, 13:13 + FIELD_WIDTH])


def test_the_alignment_contract_holds_pixel_for_pixel(tmp_path):
    """The one line in the docstring, asserted against the file it describes.

    ``shift_scale`` is negative on purpose: a registration log usually records
    the drift it measured rather than the correction it applied, and the wrong
    sign moves every frame twice as far the wrong way while still producing a
    complete set of shapes.
    """
    path = _write(tmp_path / "plain.tif", _ramp())
    drift = pd.DataFrame({
        "t": range(SOURCE_FRAMES),
        "cum_dy": [0.0, -1.4, -2.6, 1.0, 0.0, 0.0, 0.0, 0.0],
        "cum_dx": [0.0, 3.2, -0.4, 2.5, 0.0, 0.0, 0.0, 0.0],
    })
    shifts = tmp_path / "drift.csv"
    drift.to_csv(shifts, index=False)

    stack, record = _load(ObjectSetConfig(
        name="vessels", path=path, crop_origin=(9, 11), shifts=shifts,
        shift_columns=("cum_dy", "cum_dx"), shift_scale=-1.0))

    source = _ramp()
    shift_y = np.rint(-drift["cum_dy"].to_numpy()).astype(int)
    shift_x = np.rint(-drift["cum_dx"].to_numpy()).astype(int)
    for index in range(4):
        top, left = 9 - shift_y[index], 11 - shift_x[index]
        expected = source[index, top:top + FIELD_HEIGHT, left:left + FIELD_WIDTH]
        assert np.array_equal(stack.values[index], expected), index
    assert record["shift_y_range"] == [int(shift_y[:4].min()), int(shift_y[:4].max())]


def test_the_frame_offset_is_inherited_from_the_movie_and_overridable(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    source = _ramp()

    inherited, record = _load(ObjectSetConfig(name="vessels", path=path),
                              _movie(offset=3))
    assert np.array_equal(inherited.values[0], source[3, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 3

    overridden, record = _load(
        ObjectSetConfig(name="vessels", path=path, frame_offset=1), _movie(offset=3))
    assert np.array_equal(overridden.values[0], source[1, :FIELD_HEIGHT, :FIELD_WIDTH])
    assert record["frame_offset"] == 1


def test_the_named_channel_of_a_hyperstack_is_the_one_read(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    source = _ramp(channels=2)
    for index in (0, 1):
        stack, record = _load(
            ObjectSetConfig(name="vessels", path=path, channel_index=index))
        assert np.array_equal(
            stack.values, source[:4, index, :FIELD_HEIGHT, :FIELD_WIDTH]), index
        assert record["channel_index"] == index


# ---------------------------------------------------------------------- static

def test_a_static_map_is_stored_once_and_handed_out_for_every_frame(tmp_path):
    """A vessel tree drawn once should not cost a hundred copies of itself."""
    single = _blobs(frames=1)[0]
    path = _write(tmp_path / "map.tif", single)
    stack, record = _load(ObjectSetConfig(name="vessels", path=path, static=True))

    assert stack.static is True
    assert stack.values.shape == (1, FIELD_HEIGHT, FIELD_WIDTH)
    assert record["frames_supplied"] == 1
    assert record["frame_offset"] is None
    expected = single[:FIELD_HEIGHT, :FIELD_WIDTH]
    for index in range(4):
        assert np.array_equal(stack.frame(index), expected), index


def test_a_static_map_is_held_fixed_and_does_not_inherit_the_movies_offset(tmp_path):
    """It was drawn in the analysed field's own space, so it stays there."""
    single = _blobs(frames=1)[0]
    path = _write(tmp_path / "map.tif", single)
    stack, _ = _load(ObjectSetConfig(name="vessels", path=path, static=True),
                     _movie(offset=3))
    assert np.array_equal(stack.frame(0), single[:FIELD_HEIGHT, :FIELD_WIDTH])


def test_a_static_map_may_be_one_plane_of_a_multi_channel_image(tmp_path):
    plane = _blobs(frames=1)[0]
    both = np.stack([plane, plane * 7])
    path = _write(tmp_path / "map.tif", both, axes="CYX")
    stack, _ = _load(ObjectSetConfig(name="vessels", path=path, static=True,
                                     channel_index=1))
    assert np.array_equal(stack.frame(0), (plane * 7)[:FIELD_HEIGHT, :FIELD_WIDTH])


@pytest.mark.parametrize("setting,value", [("frame_offset", 2), ("shifts", "d.csv")])
def test_static_together_with_a_time_setting_is_refused(setting, value):
    """Refused rather than ignored: a dropped setting reads like an honoured one."""
    def resolve(item):
        return None if item in (None, "") else item

    with pytest.raises(ValueError, match=setting):
        ObjectSetConfig.from_dict(
            {"name": "vessels", "path": "x.tif", "static": True, setting: value},
            resolve)


def test_a_static_set_pointed_at_a_whole_movie_is_refused(tmp_path):
    """The file and the setting disagree about what the file is."""
    path = _write(tmp_path / "many.tif", _blobs())
    with pytest.raises(ValueError, match="more than one map"):
        _load(ObjectSetConfig(name="vessels", path=path, static=True))


# ----------------------------------------------------------------- the refusals

def test_a_pixel_the_alignment_cannot_reach_is_empty_ground_and_is_counted(tmp_path):
    """A label image has no spare value for "could not look", so it is counted.

    The array cannot say it, which is the whole reason the record does. Five
    rows of the field are sampled from above the top of the source; they come
    back as 0 - empty ground - and ``unreachable_px`` is what tells a reader
    that they were never looked at rather than looked at and found empty.
    """
    path = _write(tmp_path / "plain.tif", _blobs())
    shifts = tmp_path / "drift.csv"
    pd.DataFrame({"shift_y": [5] * SOURCE_FRAMES,
                  "shift_x": [0] * SOURCE_FRAMES}).to_csv(shifts, index=False)

    stack, record = _load(ObjectSetConfig(name="vessels", path=path,
                                          crop_origin=(0, 0), shifts=shifts))
    assert (stack.values[:, :5, :] == 0).all()
    assert record["unreachable_px"] == 4 * 5 * FIELD_WIDTH
    assert record["unreachable_fraction"] == pytest.approx(5 / FIELD_HEIGHT)


def test_a_crop_that_leaves_the_source_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="crop_origin"):
        _load(ObjectSetConfig(name="vessels", path=path,
                              crop_origin=(SOURCE_HEIGHT - 2, 0)))


def test_a_set_too_short_for_the_labels_is_refused(tmp_path):
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="frame_offset"):
        _load(ObjectSetConfig(name="vessels", path=path, frame_offset=6), frames=4)


def test_a_hyperstack_without_a_channel_index_is_refused(tmp_path):
    path = _write(tmp_path / "two.tif", _ramp(channels=2), axes="TCYX")
    with pytest.raises(ValueError, match="channel_index"):
        _load(ObjectSetConfig(name="vessels", path=path))


def test_a_negative_value_means_this_is_not_a_label_image(tmp_path):
    path = _write(tmp_path / "signed.tif",
                  (_blobs().astype(np.int16) - 1))
    with pytest.raises(ValueError, match="not a label image"):
        _load(ObjectSetConfig(name="vessels", path=path))


def test_a_missing_file_says_which_set_it_belongs_to(tmp_path):
    with pytest.raises(FileNotFoundError, match="vessels"):
        _load(ObjectSetConfig(name="vessels", path=tmp_path / "absent.tif"))


def test_the_refusal_says_object_set_rather_than_channel(tmp_path):
    """One indexing routine serves both inputs; the message must not.

    A refusal that called an object set a channel would send a reader to the
    wrong block of the settings file, which on a bad afternoon is an hour.
    """
    path = _write(tmp_path / "plain.tif", _ramp())
    with pytest.raises(ValueError, match="object set 'vessels'"):
        _load(ObjectSetConfig(name="vessels", path=path,
                              crop_origin=(SOURCE_HEIGHT - 2, 0)))


# -------------------------------------------------------------- the declaration

def _resolve(value):
    return None if value in (None, "") else value


@pytest.mark.parametrize("name", ["Vessels", "blood vessels", "2nd", "", "raw"])
def test_a_set_name_that_cannot_be_a_column_value_is_refused(name):
    with pytest.raises(ValueError):
        ObjectSetConfig.from_dict({"name": name, "path": "x.tif"}, _resolve)


def test_two_object_sets_with_one_name_are_refused():
    entries = [{"name": "vessels", "path": "a.tif"},
               {"name": "vessels", "path": "b.tif"}]
    with pytest.raises(ValueError, match="declared twice"):
        _object_sets(entries, _resolve, "stem")


def test_the_defaults_are_no_adjustment():
    spec = ObjectSetConfig.from_dict({"name": "vessels", "path": "x.tif"}, _resolve)
    assert spec.static is False
    assert spec.channel_index is None
    assert spec.frame_offset is None
    assert spec.crop_origin == (0, 0)
    assert spec.shifts is None


# --------------------------------------------------------------- the module

def test_the_module_describes_the_shapes_and_counts_them(tmp_path):
    """Areas checked against the blobs by counting them, not by trusting it."""
    from analysis.modules.objects import measure
    from analysis.registry import MeasurementContext, ObjectStack
    from analysis.units import Scale

    path = _write(tmp_path / "blobs.tif", _blobs())
    stack, _ = load_object_set(ObjectSetConfig(name="vessels", path=path),
                               _movie(), 4, SOURCE_HEIGHT, SOURCE_WIDTH)
    labels = np.zeros((4, SOURCE_HEIGHT, SOURCE_WIDTH), dtype=np.uint16)
    labels[:, 20:24, 20:24] = 1
    context = MeasurementContext(
        stem="t", labels=labels, raw=labels.astype(float), scale=Scale(30.0),
        identities=[1], objects={"vessels": stack})

    table = measure(context)["objects"]
    assert len(table) == 8                      # two objects, four frames
    assert set(table["object_set"]) == {"vessels"}
    assert sorted(set(table["object"])) == [1, 2]
    source = _blobs()
    for _, row in table.iterrows():
        counted = int((source[int(row["frame_index"])] == int(row["object"])).sum())
        assert row["object_area_px"] == counted
    # Object 2 runs along the top row of the source, so it is at the edge and
    # object 1 is not.
    assert set(table.loc[table["object"] == 1, "object_touches_border"]) == {False}
    assert set(table.loc[table["object"] == 2, "object_touches_border"]) == {True}
    assert set(table["object_count"]) == {2}
    assert set(table["object_frames_present"]) == {4}
    share = table["object_set_px"] / (SOURCE_HEIGHT * SOURCE_WIDTH)
    assert np.allclose(table["object_field_share"], share)


def test_a_movie_with_no_object_set_skips_the_module_rather_than_failing():
    """An empty ``objects`` is absent, not present-and-empty.

    The behaviour most likely to regress, because the field is a dictionary and
    the availability check for every other optional input is ``is None``.
    """
    from analysis.registry import get_module
    from analysis.test_column_declarations import _movie as fixture

    context = fixture()
    assert get_module("objects").available(context)[0]
    context.objects = {}
    available, reason = get_module("objects").available(context)
    assert not available and "objects" in reason
