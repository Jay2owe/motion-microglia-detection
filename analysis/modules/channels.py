"""Any extra imaging channel, measured through the same outlines.

The rest of the package measures one signal stack, because that is the one the
outlines were drawn from. A microscope almost never records only one. This
module takes whatever other channels the configuration declares - a stain, a
second reporter, a transmitted-light view, a dye nobody has invented yet - and
runs each of them through the same measurements, inside the same cells, in the
same frames.

The everyday version is a shop till receipt. The outlines say which items
belong to which customer; the channels are the different things you can total
per customer - price, weight, calories. Working out who bought what is the hard
part, and it is already done. Adding another total is not another shop.

Nothing here knows what any channel is. That is deliberate and it is the point:
a module that knew would have to be edited for every new experiment, and the
first dataset it had not been edited for would be measured wrongly or not at
all. What a channel stains is the user's to say, in the configuration's
``description``, and it travels into the manifest untouched.

**One row per cell per frame per channel.** The tables are long rather than
wide - ``channel`` is a column, not a suffix on twenty column names - so
declaring a second channel changes how many rows come out and never how many
columns. A wide table would mean the schema depended on the configuration, and
every figure, test and roll-up that names a column would break the first time
somebody imaged a third dye.

Three tables, because they answer three questions that cannot be derived from
each other:

``channels``
    what each cell looked like in each channel in each frame.
``channel_frame``
    what the whole field looked like, which is the only thing that separates a
    cell getting brighter from the lamp getting brighter.
``channel_tracks``
    what each cell did over the whole recording, with the field-wide trend
    taken out.

Every measurement is ``NaN``-aware. Lining a channel up with the labels can ask
for a pixel the source frame does not have, at the edge the field drifted away
from, and ``analysis.io`` fills those with ``NaN`` rather than a wrapped-around
neighbour. ``channel_px`` and ``channel_missing_px`` say how much of each cell
was actually there to measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import disk

from analysis.registry import (ChannelStack, Column, MeasurementContext, Output,
                               register)

DEFAULTS = {
    # The same ring as ``intensity`` uses, and for the same reason: a cell
    # compared to the field beside it rather than to one number for the whole
    # image is a cell whose measurement survives uneven illumination. Separate
    # settings, though, because a channel with a wide halo may need a wider one.
    "background_ring_inner_px": 4,
    "background_ring_outer_px": 12,
    "min_background_px": 25,
    "bright_threshold_multiplier": 2.0,
}


#: All one colour family. Which channel a row is about is a value in the
#: ``channel`` column, so a figure separates them by series and not by hue -
#: there is no fixed number of channels for a palette to have an entry per.
PRODUCES = (
    Column("channel", "Channel", "", "channel"),
    Column("channel_px", "Channel pixels measured", "px", "channel"),
    Column("channel_missing_px", "Channel pixels off the source", "px", "channel"),
    Column("channel_mean", "Channel signal", "camera units", "channel"),
    Column("channel_median", "Channel signal, median", "camera units", "channel"),
    Column("channel_sd", "Channel signal, s.d.", "camera units", "channel"),
    Column("channel_p10", "Channel signal, 10th percentile", "camera units", "channel"),
    Column("channel_p90", "Channel signal, 90th percentile", "camera units", "channel"),
    Column("channel_min", "Channel dimmest pixel", "camera units", "channel"),
    Column("channel_max", "Channel brightest pixel", "camera units", "channel"),
    Column("channel_integrated", "Channel total", "camera units", "channel"),
    Column("channel_background_median", "Channel local background", "camera units", "channel"),
    Column("channel_background_px", "Channel local background ring", "px", "channel"),
    Column("channel_background_is_zero", "Channel background read exactly zero", "0 or 1", "channel"),
    Column("channel_corrected_mean", "Channel intensity", "camera units", "channel"),
    Column("channel_corrected_integrated", "Channel total, corrected", "camera units", "channel"),
    Column("channel_signal_to_background", "Channel over background", "ratio", "channel"),
    Column("channel_cv", "Channel variability", "CV", "channel"),
    Column("channel_punctate_fraction", "Channel punctate fraction", "fraction", "channel"),
    Column("channel_punctateness", "Channel punctateness", "ratio", "channel"),
    Column("channel_saturated_px", "Channel pixels at the ceiling", "px", "channel"),
    Column("channel_saturated_fraction", "Channel clipped fraction", "fraction", "channel"),
    Column("channel_dff", "Channel dF/F", "fraction", "channel"),
    # whole field, one row per frame per channel
    Column("channel_field_mean", "Channel field signal", "camera units", "channel"),
    Column("channel_field_median", "Channel field signal, median", "camera units", "channel"),
    Column("channel_field_sd", "Channel field signal, s.d.", "camera units", "channel"),
    Column("channel_inside_mean", "Channel signal inside outlines", "camera units", "channel"),
    Column("channel_outside_mean", "Channel signal outside outlines", "camera units", "channel"),
    Column("channel_inside_over_outside", "Channel contrast at the outlines", "ratio", "channel"),
    Column("channel_field_valid_px", "Channel field pixels measured", "px", "channel"),
    Column("channel_field_saturated_px", "Channel field pixels at the ceiling", "px", "channel"),
    # one row per cell per channel
    Column("channel_frames", "Frames measured in this channel", "frames", "channel"),
    Column("channel_level_median", "Channel level", "camera units", "channel"),
    Column("channel_level_iqr", "Channel level, IQR", "camera units", "channel"),
    Column("channel_trend_per_hour", "Channel trend", "camera units per hour", "channel"),
    Column("channel_trend_r2", "Channel trend fit", "r squared", "channel"),
    Column("channel_detrended_median", "Channel level against the field", "camera units", "channel"),
    Column("channel_detrended_sd", "Channel wobble against the field", "camera units", "channel"),
)


#: None of the three folds. A roll-up is keyed on a cell, a frame or both, and
#: every table here is additionally keyed on which channel it is about, so
#: folding one in would need a column per channel - the wide schema this module
#: exists to avoid.
WRITES = (
    Output("channels", grain=("identity", "frame_index", "channel")),
    Output("channel_frame", grain=("frame_index", "channel")),
    Output("channel_tracks", grain=("identity", "channel")),
)


def _finite(values: np.ndarray) -> np.ndarray:
    """The measurable pixels, widened to float64 before anything is summed.

    The stack is kept as float32 where the source allows it, which is exact for
    every 8- and 16-bit camera and half the memory. Accumulating in float32 is
    not exact: summing a few hundred pixels of a 16-bit image moves a standard
    deviation in the seventh significant figure. Every statistic here is
    therefore computed in float64, so this module and ``intensity`` return the
    same number for the same pixels rather than nearly the same one.
    """
    return values[np.isfinite(values)].astype(np.float64)


def _trend(hours: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Least-squares slope per hour and how much of the variance it explains.

    Two numbers rather than one because a slope with no fit behind it is the
    classic way to report a trend that is not there: three noisy points always
    have a slope. ``r2`` is what says whether to believe it.
    """
    keep = np.isfinite(hours) & np.isfinite(values)
    if keep.sum() < 3:
        return np.nan, np.nan
    x, y = hours[keep], values[keep]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(1.0 - (residual ** 2).sum() / total) if total > 0 else np.nan


def _cell_rows(context: MeasurementContext, stack: ChannelStack, params: dict) -> list[dict]:
    """Every cell in every frame, measured in one channel."""
    inner_disk = disk(int(params["background_ring_inner_px"]))
    outer_disk = disk(int(params["background_ring_outer_px"]))
    pad = int(params["background_ring_outer_px"]) + 2
    multiplier = float(params["bright_threshold_multiplier"])
    ceiling = stack.saturation_value
    rows: list[dict] = []

    for frame_index in range(context.n_frames):
        labels = context.labels[frame_index]
        values_frame = stack.values[frame_index]
        occupied = labels > 0
        if context.unclaimed is not None:
            occupied = occupied | (context.unclaimed[frame_index] > 0)

        for zero_based, window in enumerate(ndi.find_objects(labels)):
            if window is None:
                continue
            identity = zero_based + 1
            rows_slice = slice(max(window[0].start - pad, 0),
                               min(window[0].stop + pad, labels.shape[0]))
            cols_slice = slice(max(window[1].start - pad, 0),
                               min(window[1].stop + pad, labels.shape[1]))
            local_labels = labels[rows_slice, cols_slice]
            local_values = values_frame[rows_slice, cols_slice]
            local_occupied = occupied[rows_slice, cols_slice]

            mask = local_labels == identity
            inside = local_values[mask]
            measured = _finite(inside)
            del inside
            row = {
                "identity": identity,
                "frame_index": frame_index,
                "channel": stack.name,
                "channel_px": int(measured.size),
                "channel_missing_px": int(int(mask.sum()) - measured.size),
            }
            if measured.size == 0:
                # The cell is here and the channel is not: every measurement is
                # blank rather than zero, because zero is a brightness.
                rows.append(row)
                continue

            near = ndi.binary_dilation(mask, structure=inner_disk)
            far = ndi.binary_dilation(mask, structure=outer_disk)
            ring = _finite(local_values[far & ~near & ~local_occupied])
            if ring.size >= int(params["min_background_px"]):
                background, background_px = float(np.median(ring)), int(ring.size)
            else:
                elsewhere = _finite(local_values[~local_occupied])
                background = float(np.median(elsewhere)) if elsewhere.size else 0.0
                background_px = int(elsewhere.size)

            corrected = measured - background
            median = float(np.median(measured))
            mean = float(measured.mean())
            background_is_zero = background <= 0
            saturated = 0 if ceiling is None else int((measured >= ceiling).sum())

            row.update({
                "channel_mean": mean,
                "channel_median": median,
                "channel_sd": float(measured.std(ddof=0)),
                "channel_p10": float(np.percentile(measured, 10)),
                "channel_p90": float(np.percentile(measured, 90)),
                "channel_min": float(measured.min()),
                "channel_max": float(measured.max()),
                "channel_integrated": float(measured.sum()),
                "channel_background_median": background,
                "channel_background_px": background_px,
                "channel_background_is_zero": bool(background_is_zero),
                "channel_corrected_mean": float(corrected.mean()),
                "channel_corrected_integrated": float(corrected.sum()),
                "channel_signal_to_background": (
                    float(mean / background) if not background_is_zero else np.nan
                ),
                "channel_cv": float(measured.std(ddof=0) / mean) if mean > 0 else np.nan,
                # How much of the cell is far brighter than the rest of the same
                # cell. What that means depends on the channel and this module
                # does not say; for a punctate stain it is the puncta.
                "channel_punctate_fraction": float((measured > median * multiplier).mean()),
                "channel_punctateness": (
                    float(np.percentile(measured, 90) / median) if median > 0 else np.nan
                ),
                "channel_saturated_px": saturated,
                "channel_saturated_fraction": float(saturated / measured.size),
            })
            rows.append(row)
    return rows


def _frame_rows(context: MeasurementContext, stack: ChannelStack) -> list[dict]:
    """The whole field, frame by frame, in one channel.

    Without this table a channel that fades cannot be told from cells that fade,
    and every per-cell trend in ``channel_tracks`` would be a measurement of the
    lamp. It is deliberately not derived from the per-cell table: the cells are
    where the answer is not, so the outside has to be measured directly.
    """
    ceiling = stack.saturation_value
    rows: list[dict] = []
    for frame_index in range(context.n_frames):
        values = stack.values[frame_index]
        finite = np.isfinite(values)
        named = context.labels[frame_index] > 0
        measured = values[finite].astype(np.float64)
        inside = values[finite & named].astype(np.float64)
        outside = values[finite & ~named].astype(np.float64)
        outside_mean = float(outside.mean()) if outside.size else np.nan
        inside_mean = float(inside.mean()) if inside.size else np.nan
        rows.append({
            "frame_index": frame_index,
            "channel": stack.name,
            "channel_field_valid_px": int(measured.size),
            "channel_field_mean": float(measured.mean()) if measured.size else np.nan,
            "channel_field_median": float(np.median(measured)) if measured.size else np.nan,
            "channel_field_sd": float(measured.std(ddof=0)) if measured.size else np.nan,
            "channel_inside_mean": inside_mean,
            "channel_outside_mean": outside_mean,
            # A contrast at the outlines, not a cell-against-empty-field
            # ratio: "outside" is every other measurable pixel, foreground
            # that carries no name included.
            "channel_inside_over_outside": (
                float(inside_mean / outside_mean)
                if np.isfinite(outside_mean) and outside_mean != 0
                else np.nan
            ),
            "channel_field_saturated_px": (
                0 if ceiling is None else int((measured >= ceiling).sum())
            ),
        })
    return rows


def _track_rows(cells: pd.DataFrame, frames: pd.DataFrame,
                context: MeasurementContext) -> list[dict]:
    """One row per cell per channel: its level, its trend, and its wobble.

    ``channel_detrended_*`` is the pair that earns this table. Subtracting the
    field's own mean for that frame leaves what the cell did that the field did
    not, which is the only version of "this cell got brighter" that survives a
    channel fading by a third over two days - and one that fades by a third is
    the ordinary case, not the broken one.
    """
    # A channel that lands entirely off the source measures nothing, so the
    # corrected column never gets created. That is a real state - an alignment
    # pointed at the wrong corner of a bigger file - and it has to come back as
    # an empty table rather than a KeyError three lines later.
    if cells.empty or "channel_corrected_mean" not in cells.columns:
        return []
    hours = context.frame_table().set_index("frame_index")["hours"]
    field = frames.set_index(["channel", "frame_index"])["channel_field_mean"]
    rows: list[dict] = []
    for (channel, identity), group in cells.groupby(["channel", "identity"], sort=True):
        group = group.sort_values("frame_index")
        level = group["channel_corrected_mean"].to_numpy(dtype=float)
        time = hours.reindex(group["frame_index"]).to_numpy(dtype=float)
        reference = field.reindex(
            pd.MultiIndex.from_product([[channel], group["frame_index"]])
        ).to_numpy(dtype=float)
        detrended = level - reference
        finite_level = level[np.isfinite(level)]
        finite_detrended = detrended[np.isfinite(detrended)]
        slope, r2 = _trend(time, level)
        rows.append({
            "identity": int(identity),
            "channel": channel,
            "channel_frames": int(np.isfinite(level).sum()),
            "channel_level_median": (
                float(np.median(finite_level)) if finite_level.size else np.nan
            ),
            "channel_level_iqr": (
                float(np.percentile(finite_level, 75) - np.percentile(finite_level, 25))
                if finite_level.size else np.nan
            ),
            "channel_trend_per_hour": slope,
            "channel_trend_r2": r2,
            "channel_detrended_median": (
                float(np.median(finite_detrended)) if finite_detrended.size else np.nan
            ),
            "channel_detrended_sd": (
                float(finite_detrended.std(ddof=0)) if finite_detrended.size else np.nan
            ),
        })
    return rows


@register(
    name="channels",
    description="Every extra imaging channel measured through the same outlines",
    requires=("labels", "channels"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("channels")}
    cell_rows: list[dict] = []
    frame_rows: list[dict] = []

    # Configuration order, not alphabetical: the first channel a user lists is
    # the one they care about most, and it should be the first series drawn.
    for name in context.channel_names:
        stack = context.channels[name]
        cell_rows.extend(_cell_rows(context, stack, params))
        frame_rows.extend(_frame_rows(context, stack))

    cells = pd.DataFrame(cell_rows)
    frames = pd.DataFrame(frame_rows)
    if not cells.empty:
        cells = cells.sort_values(["channel", "identity", "frame_index"]).reset_index(drop=True)
    # dF/F against each cell's own median in that channel, so cells whose
    # absolute brightness differs for reasons that are not biological can
    # still be compared. Grouped by channel as well as by cell: one cell has
    # a different baseline in every channel.
    if "channel_corrected_mean" in cells.columns:
        baseline = cells.groupby(["channel", "identity"])["channel_corrected_mean"].transform("median")
        cells["channel_dff"] = np.where(
            baseline.abs() > 0,
            (cells["channel_corrected_mean"] - baseline) / baseline,
            np.nan,
        )
    tracks = pd.DataFrame(_track_rows(cells, frames, context))
    return {"channels": cells, "channel_frame": frames, "channel_tracks": tracks}
