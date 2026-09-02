"""Reporter signal inside each cell in each frame.

For this dataset the registered raw stack is the mCherry channel driven by the
CD68 promoter, so signal inside an outline is phagolysosomal reporter load.
The package makes no assumption about that: it measures whatever single-channel
signal the outlines were built from.

Every cell gets its own local background, taken from a ring of empty field just
outside it that belongs to no other cell. Comparing a cell to the field beside
it rather than to a single number for the whole image is what stops slow
illumination drift being read as biology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import disk

from analysis.registry import Column, MeasurementContext, Output, register

DEFAULTS = {
    "background_ring_inner_px": 4,
    "background_ring_outer_px": 12,
    "min_background_px": 25,
    "bright_threshold_multiplier": 2.0,
}


def _ring_footprints(inner: int, outer: int) -> tuple[np.ndarray, np.ndarray]:
    return disk(inner), disk(outer)


#: Everything here is in camera units, not calibrated intensity, and the
#: ``reporter`` role keeps all of it one colour family across the figure set.
PRODUCES = (
    Column("signal_mean", "Reporter signal", "camera units", "reporter"),
    Column("signal_median", "Reporter signal, median", "camera units", "reporter"),
    Column("signal_std", "Reporter signal, s.d.", "camera units", "reporter"),
    Column("signal_p10", "Reporter signal, 10th percentile", "camera units", "reporter"),
    Column("signal_p90", "Reporter signal, 90th percentile", "camera units", "reporter"),
    Column("signal_max", "Brightest pixel", "camera units", "reporter"),
    Column("integrated_density", "Reporter total", "camera units", "reporter"),
    Column("background_median", "Local background", "camera units", "reporter"),
    Column("background_px", "Local background ring", "px", "reporter"),
    Column("background_is_zero", "Background read exactly zero", "0 or 1", "reporter"),
    Column("corrected_mean", "Reporter intensity", "camera units", "reporter"),
    Column("corrected_integrated_density", "Reporter total, corrected", "camera units", "reporter"),
    Column("signal_to_background", "Signal over background", "ratio", "reporter"),
    Column("bright_fraction_over_background", "Bright fraction", "fraction", "reporter"),
    Column("punctate_fraction", "Punctate fraction", "fraction", "reporter"),
    Column("punctate_px", "Punctate pixels", "px", "reporter"),
    Column("punctateness", "Punctateness", "ratio", "reporter"),
    Column("signal_cv", "Signal variability", "CV", "reporter"),
    Column("dff", "dF/F", "fraction", "reporter"),
)


WRITES = (
    Output("intensity", grain=("identity", "frame_index"), fold=True),
)


@register(
    name="intensity",
    description="Reporter signal inside each outline with a local background ring",
    requires=("labels", "raw"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("intensity")}
    inner_disk, outer_disk = _ring_footprints(
        int(params["background_ring_inner_px"]), int(params["background_ring_outer_px"])
    )
    pad = int(params["background_ring_outer_px"]) + 2
    rows: list[dict] = []

    for frame_index in range(context.n_frames):
        labels = context.labels[frame_index]
        raw = context.raw[frame_index].astype(np.float64)
        occupied = labels > 0
        if context.unclaimed is not None:
            occupied = occupied | (context.unclaimed[frame_index] > 0)

        objects = ndi.find_objects(labels)
        for zero_based, window in enumerate(objects):
            if window is None:
                continue
            identity = zero_based + 1

            rows_slice = slice(max(window[0].start - pad, 0), min(window[0].stop + pad, labels.shape[0]))
            cols_slice = slice(max(window[1].start - pad, 0), min(window[1].stop + pad, labels.shape[1]))
            local_labels = labels[rows_slice, cols_slice]
            local_raw = raw[rows_slice, cols_slice]
            local_occupied = occupied[rows_slice, cols_slice]

            mask = local_labels == identity
            values = local_raw[mask]
            if values.size == 0:
                continue

            near = ndi.binary_dilation(mask, structure=inner_disk)
            far = ndi.binary_dilation(mask, structure=outer_disk)
            ring = far & ~near & ~local_occupied
            ring_values = local_raw[ring]
            if ring_values.size >= int(params["min_background_px"]):
                background = float(np.median(ring_values))
                background_px = int(ring_values.size)
            else:
                background = float(np.median(local_raw[~local_occupied])) if (~local_occupied).any() else 0.0
                background_px = int((~local_occupied).sum())

            corrected = values - background
            multiplier = float(params["bright_threshold_multiplier"])
            cell_median = float(np.median(values))

            # Two different questions, and which one is answerable depends on the
            # input. If the stack has already been background-subtracted the
            # field outside the cells is exactly zero, so "brighter than the
            # background" is undefined and only the within-cell measure means
            # anything. Both are emitted; the flag says which to trust.
            background_is_zero = background <= 0
            bright_over_background = (
                float((values > background * multiplier).mean()) if not background_is_zero else np.nan
            )
            punctate_cut = cell_median * multiplier

            row = {
                "identity": identity,
                "frame_index": frame_index,
                "signal_mean": float(values.mean()),
                "signal_median": float(np.median(values)),
                "signal_std": float(values.std(ddof=0)),
                "signal_p10": float(np.percentile(values, 10)),
                "signal_p90": float(np.percentile(values, 90)),
                "signal_max": float(values.max()),
                "integrated_density": float(values.sum()),
                "background_median": background,
                "background_px": background_px,
                "background_is_zero": bool(background_is_zero),
                "corrected_mean": float(corrected.mean()),
                "corrected_integrated_density": float(corrected.sum()),
                "signal_to_background": (
                    float(values.mean() / background) if not background_is_zero else np.nan
                ),
                "bright_fraction_over_background": bright_over_background,
                # How much of the cell is much brighter than the rest of the same
                # cell: a proxy for punctate reporter, which for CD68 is
                # phagolysosomal accumulation rather than diffuse expression.
                "punctate_fraction": float((values > punctate_cut).mean()),
                "punctate_px": int((values > punctate_cut).sum()),
                "punctateness": (
                    float(np.percentile(values, 90) / cell_median) if cell_median > 0 else np.nan
                ),
                "signal_cv": (
                    float(values.std(ddof=0) / values.mean()) if values.mean() > 0 else np.nan
                ),
            }
            rows.append(row)

    cell_frame = pd.DataFrame(rows).sort_values(["identity", "frame_index"]).reset_index(drop=True)

    # dF/F against each cell's own median, the standard way to compare cells
    # whose absolute brightness differs for reasons that are not biological.
    baseline = cell_frame.groupby("identity")["corrected_mean"].transform("median")
    cell_frame["dff"] = np.where(
        baseline.abs() > 0, (cell_frame["corrected_mean"] - baseline) / baseline, np.nan
    )
    return {"intensity": cell_frame}
