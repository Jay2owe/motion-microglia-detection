"""Where each cell went.

Two centroids are tracked. The plain centroid is the middle of the whole
outline, which a single extending branch can drag several pixels. The
intensity-weighted centroid sits near the brightest mass, which for these
reporters is the soma, so it is the better estimate of whether the cell body
itself relocated.

Steps are measured between *observed* frames only. When a cell is missing for
three frames the next step is divided by three rather than pretended to be one
frame long, and the gap is carried in a column so those rows can be dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from analysis.registry import Column, MeasurementContext, Output, register

DEFAULTS = {
    "max_msd_lag_frames": 24,
    "min_frames_for_track_stats": 5,
}


def _nan_safe(function, values: np.ndarray) -> float:
    """Apply a nan-aware reduction, returning NaN instead of warning on all-NaN."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(function(values))


def _convex_hull_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(points).volume)  # 2-D volume is area
    except Exception:
        return float("nan")


def _mean_squared_displacement(
    frames: np.ndarray,
    coords: np.ndarray,
    max_lag: int,
    *,
    return_curve: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    """MSD per lag, then a log-log slope.

    A slope near 1 is a random walk, below 1 is confined wandering, above 1 is
    directed travel. Microglial somata usually sit well below 1.
    """
    lookup = {int(f): coords[i] for i, f in enumerate(frames)}
    lags, values, pairs = [], [], []
    span = int(frames.max() - frames.min())
    for lag in range(1, min(max_lag, span) + 1):
        squared = [
            float(np.sum((lookup[f + lag] - lookup[f]) ** 2))
            for f in lookup
            if f + lag in lookup
        ]
        if len(squared) >= 3:
            lags.append(lag)
            values.append(float(np.mean(squared)))
            pairs.append(len(squared))
    if len(lags) < 3:
        summary = {
            "msd_alpha": np.nan,
            "msd_lag1_px2": values[0] if values else np.nan,
            "msd_points": len(lags),
        }
    else:
        slope = float(np.polyfit(np.log(lags), np.log(np.maximum(values, 1e-12)), 1)[0])
        summary = {"msd_alpha": slope, "msd_lag1_px2": values[0], "msd_points": len(lags)}
    if not return_curve:
        return summary
    curve = pd.DataFrame({"lag_frames": lags, "msd_px2": values, "pairs": pairs})
    return summary, curve


#: Three tables' worth: the per-frame steps, the one-row-per-cell track summary
#: and the MSD curve. Declared together because they are one module's account of
#: the same thing, and figure 8 lets a reader colour paths by any track column,
#: so every one of them can end up on an axis. The micron columns appear only on
#: a calibrated run; declaring them costs nothing and means they arrive labelled.
PRODUCES = (
    # motility - one row per cell per frame
    Column("centroid_y", "Centroid, y", "px", "morphology"),
    Column("centroid_x", "Centroid, x", "px", "morphology"),
    Column("soma_y", "Soma, y", "px", "motility"),
    Column("soma_x", "Soma, x", "px", "motility"),
    Column("gap_frames", "Frames missing", "frames", "reference"),
    Column("step_px", "Centroid step", "px", "motility"),
    Column("step_px_gapless", "Centroid step", "px per {interval}", "motility"),
    Column("soma_step_px", "Soma step", "px", "motility"),
    Column("soma_step_px_gapless", "Soma step", "px per {interval}", "motility"),
    Column("step_px_per_frame", "Centroid step", "px per frame", "motility"),
    Column("speed", "Speed", "px per min", "motility"),
    Column("cumulative_path_px", "Path travelled", "px", "motility"),
    Column("displacement_from_first_px", "Displacement from start", "px", "motility"),
    # motility_tracks - one row per cell
    Column("first_frame_index", "First frame held", "frame", "reference"),
    Column("last_frame_index", "Last frame held", "frame", "reference"),
    Column("observed_frames", "Frames the cell was seen in", "frames", "reference"),
    Column("span_frames", "Frames from first to last", "frames", "reference"),
    Column("total_path_px", "Path travelled in total", "px", "motility"),
    Column("net_displacement_px", "Net displacement", "px", "motility"),
    Column("straightness", "Path straightness", "0-1", "motility"),
    Column("consecutive_steps", "Steps across consecutive frames", "count", "motility"),
    Column("median_step_px", "Centroid step, median", "px", "motility"),
    Column("p95_step_px", "Centroid step, 95th percentile", "px", "motility"),
    Column("max_step_px", "Centroid step, largest", "px", "motility"),
    Column("median_soma_step_px", "Soma step, median", "px", "motility"),
    Column("median_step_px_including_gaps", "Centroid step, median across gaps", "px", "motility"),
    Column("territory_hull_px2", "Convex hull of the path", "px^2", "motility"),
    Column("mean_speed", "Speed, mean", "px per min", "motility"),
    Column("msd_alpha", "Motion scaling exponent, α", "", "motility"),
    Column("msd_lag1_px2", "Mean squared displacement at one frame", "px²", "motility"),
    Column("msd_points", "Lags the exponent was fitted over", "count", "motility"),
    Column("total_path_um", "Path travelled in total", "µm", "motility"),
    Column("net_displacement_um", "Net displacement", "µm", "motility"),
    Column("territory_hull_um2", "Convex hull of the path", "µm²", "motility"),
    # msd_curves - one row per cell per lag
    Column("lag_frames", "Lag", "frames", "motility"),
    Column("lag_hours", "Lag", "h", "motility"),
    Column("msd_px2", "Mean squared displacement", "px²", "motility"),
    Column("msd", "Mean squared displacement", "px²", "motility"),
    Column("pairs", "Frame pairs averaged", "count", "motility"),
)


#: ``motility_tracks`` is one row per cell and every one of its columns is
#: already inside ``cell_summary`` - it is where they come from. It is declared
#: as a fold rather than a file so there is one copy of those numbers rather
#: than two, which is one place for them to be wrong rather than two places to
#: disagree. The module still builds the table; ``build_cell_summary`` and
#: ``build_movie_summary`` both read it out of the results dictionary.
WRITES = (
    Output("motility",        grain=("identity", "frame_index"), fold=True),
    Output("motility_tracks", grain=("identity",),               fold=True),
    Output("msd_curves",      grain=("identity", "lag_frames")),
)


@register(
    name="motility",
    description="Centroid displacement, speed, path straightness and explored territory",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("motility")}
    scale = context.scale
    rows: list[dict] = []

    for frame_index in range(context.n_frames):
        labels = context.labels[frame_index]
        raw = context.raw[frame_index].astype(np.float64) if context.raw is not None else None
        present = [int(v) for v in np.unique(labels) if v]
        if not present:
            continue
        centroids = ndi.center_of_mass(labels > 0, labels, present)
        weighted = (
            ndi.center_of_mass(raw, labels, present) if raw is not None else [(np.nan, np.nan)] * len(present)
        )
        for identity, centre, wcentre in zip(present, centroids, weighted):
            rows.append(
                {
                    "identity": identity,
                    "frame_index": frame_index,
                    "centroid_y": float(centre[0]),
                    "centroid_x": float(centre[1]),
                    "soma_y": float(wcentre[0]),
                    "soma_x": float(wcentre[1]),
                }
            )

    cell_frame = pd.DataFrame(rows).sort_values(["identity", "frame_index"]).reset_index(drop=True)

    track_rows: list[dict] = []
    step_columns: list[pd.DataFrame] = []
    msd_rows: list[pd.DataFrame] = []

    for identity, group in cell_frame.groupby("identity", sort=True):
        group = group.sort_values("frame_index")
        frames = group["frame_index"].to_numpy()
        coords = group[["centroid_y", "centroid_x"]].to_numpy(float)
        soma = group[["soma_y", "soma_x"]].to_numpy(float)

        gap = np.diff(frames, prepend=frames[0])
        step = np.full(len(frames), np.nan)
        soma_step = np.full(len(frames), np.nan)
        step[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        soma_step[1:] = np.linalg.norm(np.diff(soma, axis=0), axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            per_frame_step = step / np.where(gap > 0, gap, np.nan)

        # A step measured across a 72-frame absence is not comparable with one
        # measured across 30 minutes. Consecutive-frame steps are kept in their
        # own column and are the only ones the track statistics use.
        consecutive = gap == 1
        gapless_step = np.where(consecutive, step, np.nan)

        step_columns.append(
            pd.DataFrame(
                {
                    "identity": identity,
                    "frame_index": frames,
                    "gap_frames": gap,
                    "step_px": step,
                    "step_px_gapless": gapless_step,
                    "soma_step_px": soma_step,
                    "soma_step_px_gapless": np.where(consecutive, soma_step, np.nan),
                    "step_px_per_frame": per_frame_step,
                    "speed": [scale.speed(v) for v in per_frame_step],
                    "cumulative_path_px": np.nancumsum(np.nan_to_num(step, nan=0.0)),
                    "displacement_from_first_px": np.linalg.norm(coords - coords[0], axis=1),
                }
            )
        )

        if len(frames) < int(params["min_frames_for_track_stats"]):
            continue

        path = float(np.nansum(step))
        net = float(np.linalg.norm(coords[-1] - coords[0]))
        observed = int(len(frames))
        span = int(frames[-1] - frames[0] + 1)
        track = {
            "identity": identity,
            "first_frame_index": int(frames[0]),
            "last_frame_index": int(frames[-1]),
            "observed_frames": observed,
            "span_frames": span,
            "gap_frames": span - observed,
            "total_path_px": path,
            "net_displacement_px": net,
            "straightness": (net / path) if path > 0 else np.nan,
            "consecutive_steps": int(np.count_nonzero(consecutive)),
            "median_step_px": _nan_safe(np.nanmedian, gapless_step),
            "p95_step_px": _nan_safe(lambda v: np.nanpercentile(v, 95), gapless_step),
            "max_step_px": _nan_safe(np.nanmax, gapless_step),
            "median_soma_step_px": _nan_safe(
                np.nanmedian, np.where(consecutive, soma_step, np.nan)
            ),
            "median_step_px_including_gaps": _nan_safe(np.nanmedian, step),
            "territory_hull_px2": _convex_hull_area(coords),
            "mean_speed": scale.speed(_nan_safe(np.nanmean, gapless_step)),
        }
        msd_summary, msd_curve = _mean_squared_displacement(
            frames, coords, int(params["max_msd_lag_frames"]), return_curve=True
        )
        track.update(msd_summary)
        if not msd_curve.empty:
            msd_curve.insert(0, "identity", int(identity))
            msd_curve["lag_hours"] = [scale.hours(v) for v in msd_curve["lag_frames"]]
            # ``msd`` is the run-scale value figures draw; ``msd_px2`` remains
            # beside it so the raw measurement is never lost on calibrated runs.
            msd_curve["msd"] = [scale.area(v) for v in msd_curve["msd_px2"]]
            msd_rows.append(msd_curve)
        if scale.calibrated:
            track["total_path_um"] = scale.length(path)
            track["net_displacement_um"] = scale.length(net)
            track["territory_hull_um2"] = scale.area(track["territory_hull_px2"])
        track_rows.append(track)

    if step_columns:
        steps = pd.concat(step_columns, ignore_index=True)
        cell_frame = cell_frame.merge(steps, on=["identity", "frame_index"], how="left")

    return {
        "motility": cell_frame,
        "motility_tracks": pd.DataFrame(track_rows),
        "msd_curves": (
            pd.concat(msd_rows, ignore_index=True)
            if msd_rows else
            pd.DataFrame(columns=["identity", "lag_frames", "lag_hours", "msd", "msd_px2", "pairs"])
        ),
    }
