"""Reusable, file-free measurements for spatial time-series figures.

Coordinates remain in the recording's coordinate system. No spatial smoothing,
phase sorting, gap interpolation or inference of physical communication is done.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import circadian as workbench

RADIAL_METRIC = "mean_relative_radial_occupancy"


def radial_occupancy(sholl: pd.DataFrame, *, scaling="cell", support=0.5,
                     length_per_pixel=1.0) -> pd.DataFrame:
    """Mean of all equally spaced occupancy bands, never crossing-count AUC.

Require every ring and sufficient geometric support in every cell-frame. A
missing outer ring must not make a truncated cell appear more densely occupied.
"""
    if not 0 < support <= 1 or length_per_pixel <= 0:
        raise ValueError("annulus support must be in (0, 1]; length scale must be positive")
    frame = sholl.loc[sholl.scaling.eq(scaling)].copy()
    if frame.empty:
        raise ValueError(f"no radial profiles with scaling={scaling!r}")
    keys = ["identity", "frame_index"]
    if frame.duplicated(keys + ["ring"]).any():
        raise ValueError("duplicate radial band in a cell-frame")
    expected = np.pi * ((frame.radius_outer / length_per_pixel) ** 2
                        - (frame.radius_inner / length_per_pixel) ** 2)
    frame["usable"] = (expected.gt(0) & (frame.annulus_px / expected).ge(support)
                       & np.isfinite(frame.occupancy))
    total = frame.ring.nunique()
    grouped = frame.groupby(keys, sort=True)
    out = grouped.agg(hours=("hours", "first"), valid_rings=("usable", "sum"),
                      recorded_rings=("ring", "nunique"),
                      **{RADIAL_METRIC: ("occupancy", "mean")}).reset_index()
    complete = out.valid_rings.eq(total) & out.recorded_rings.eq(total)
    out.loc[~complete, RADIAL_METRIC] = np.nan
    out["required_rings"] = total
    return out


def display_traces(frame: pd.DataFrame, metric: str, params: dict,
                   display="standardized") -> pd.DataFrame:
    """Preserve missing frames; detrend with all shared Workbench controls."""
    if metric not in frame or not pd.api.types.is_numeric_dtype(frame[metric]):
        raise ValueError(f"{metric!r} is not a numeric cell-frame metric")
    if display not in {"raw", "residual", "standardized"}:
        raise ValueError("display must be raw, residual or standardized")
    result = frame.copy()
    if result.duplicated(["identity", "frame_index"]).any():
        raise ValueError("one row per cell-frame is required")
    result["value"] = np.nan
    for _, group in result.groupby("identity"):
        valid = group[np.isfinite(group[metric]) & np.isfinite(group.hours)].sort_values("hours")
        if not len(valid):
            continue
        values = valid[metric].to_numpy(float)
        if display != "raw":
            if len(valid) < 3:
                continue
            transformed = workbench.detrend_trace(valid.hours, values, params)
            residual = np.asarray(transformed["values"], float)
            values = (workbench.scale_detrended(residual, values)
                      if display == "standardized" else residual)
        result.loc[valid.index, "value"] = values
    return result


def positions(frame: pd.DataFrame, *, centre="soma", length_per_pixel=1.0) -> pd.DataFrame:
    """Median measured position per cell (not its first-occupant territory)."""
    if centre not in {"soma", "centroid"}:
        raise ValueError("centre must be soma or centroid")
    cols = [f"{centre}_x", f"{centre}_y"]
    if any(col not in frame for col in cols):
        raise ValueError(f"missing {centre} coordinates")
    out = frame.groupby("identity")[cols].median().rename(
        columns={cols[0]: "x", cols[1]: "y"}).reset_index()
    out[["x", "y"]] *= length_per_pixel
    return out


def track_segments(frame: pd.DataFrame, *, centre="soma", length_per_pixel=1.0) -> pd.DataFrame:
    """Actual consecutive-frame path segments; never bridge missing tracking."""
    if centre not in {"soma", "centroid"}:
        raise ValueError("centre must be soma or centroid")
    if not np.isfinite(length_per_pixel) or length_per_pixel <= 0:
        raise ValueError("length_per_pixel must be finite and positive")
    columns = ["identity", "frame_index", "hours", f"{centre}_x", f"{centre}_y"]
    if set(columns) - set(frame):
        raise ValueError("track segments need identity, frame/time and the selected position coordinates")
    if frame.duplicated(["identity", "frame_index"]).any():
        raise ValueError("track segments need one observation per cell-frame")
    ordered = frame[columns].sort_values(["identity", "frame_index"])
    previous = ordered.groupby("identity").shift()
    coordinates = [f"{centre}_x", f"{centre}_y"]
    valid = ((ordered.frame_index - previous.frame_index).eq(1)
             & (ordered.hours > previous.hours)
             & np.isfinite(ordered[coordinates]).all(axis=1)
             & np.isfinite(previous[coordinates]).all(axis=1))
    current = ordered[valid]
    return pd.DataFrame(dict(identity=current.identity,
        from_frame_index=previous.loc[valid, "frame_index"].astype(int),
        frame_index=current.frame_index, from_hours=previous.loc[valid, "hours"], hours=current.hours,
        x=previous.loc[valid, coordinates[0]] * length_per_pixel,
        y=previous.loc[valid, coordinates[1]] * length_per_pixel,
        x2=current[coordinates[0]] * length_per_pixel,
        y2=current[coordinates[1]] * length_per_pixel)).reset_index(drop=True)


def snapshot_indices(hours, requested=None, count=1):
    """Final frame by default; explicit hours override an evenly spaced count.

    Selected frames are unique and chronological. A count requests exactly that
    many actual frames, even on an irregular time axis. Nothing is interpolated
    or replaced by a cell's last non-missing observation.
    """
    hours = np.asarray(hours, float)
    if hours.ndim != 1 or len(hours) < 1 or not np.isfinite(hours).all() or np.any(np.diff(hours) <= 0):
        raise ValueError("frame hours must be finite and strictly increasing")
    requested = np.asarray([] if requested is None else requested, float)
    if requested.ndim != 1 or not np.isfinite(requested).all():
        raise ValueError("snapshot times must be finite and nonempty")
    if not len(requested):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("snapshot_count must be a positive integer")
        if count > len(hours):
            raise ValueError("snapshot_count exceeds the number of recorded frames")
        if count == 1:
            return np.array([len(hours) - 1], dtype=int)
        targets = np.linspace(hours[0], hours[-1], count)
        indices = []
        for j, target in enumerate(targets):
            start = indices[-1] + 1 if indices else 0
            stop = len(hours) - (count - j - 1)
            indices.append(start + int(np.argmin(np.abs(hours[start:stop] - target))))
        return np.asarray(indices, dtype=int)
    if requested.min() < hours[0] or requested.max() > hours[-1]:
        raise ValueError("snapshot times lie outside the recording")
    return np.unique([int(np.argmin(np.abs(hours - t))) for t in requested])


def rhythm_fits(frame, metrics, params, *, method="lomb", significance_method=None,
                correction="bh", min_observations=24, min_cycles=3):
    long = frame.melt(id_vars=["identity", "hours"], value_vars=metrics,
                      var_name="measurement", value_name="measurement_value")
    fits = workbench.estimate_grouped_rhythms(
        long, group_columns=["identity", "measurement"], value_column="measurement_value",
        params=params, method=method, significance_method=significance_method,
        min_observations=min_observations, correction=correction, min_cycles=min_cycles)
    fits = fits.drop(columns="metric").rename(columns={"measurement": "metric"})
    fits["supported"] = (fits.significant & fits.period_available
                         & ~fits.period_underdetermined & ~fits.period_at_search_edge)
    fits["display_status"] = np.select(
        [fits.test_status.ne("ok"), ~fits.significant, ~fits.period_available,
         fits.period_underdetermined | fits.period_at_search_edge],
        ["not tested", "not significant", "period unavailable", "exploratory period"],
        default="supported rhythm")
    fits["correction_family"] = "all selected cell-metric traces"
    return fits


def timing_difference(fits: pd.DataFrame, metric: str, reference: str,
                      *, period_tolerance=0.1, reference_hour=0.0):
    """Shortest peak delay at a named recording hour, not modulo 24.

For differing periods this offset changes over time. Also report its drift
across a common observation span rather than implying a stable phase lock.
"""
    if not 0 <= period_tolerance < 1 or metric == reference:
        raise ValueError("use different metrics and a period tolerance in [0, 1)")
    a = fits[fits.metric.eq(metric)].set_index("identity")
    b = fits[fits.metric.eq(reference)].set_index("identity")
    rows = []
    for identity in a.index.intersection(b.index):
        x, y = a.loc[identity], b.loc[identity]
        p, q = float(x.period_hours), float(y.period_hours)
        compatible = bool(np.isfinite(p + q) and p > 0 and q > 0
                          and abs(p - q) / ((p + q) / 2) <= period_tolerance)
        eligible = bool(x.supported and y.supported and compatible
                        and np.isfinite(x.get("phase_hours", np.nan))
                        and np.isfinite(y.get("phase_hours", np.nan)))
        delay = drift = np.nan
        if eligible:
            # Relative peak positions at one common recording-time origin.
            phase_a = (float(x.phase_hours) - reference_hour) / p
            phase_b = (float(y.phase_hours) - reference_hour) / q
            delay = ((phase_a - phase_b + .5) % 1 - .5) * q
            drift = (1 / q - 1 / p) * min(x.span_hours, y.span_hours) * q
        rows.append(dict(identity=identity, value=delay, supported=eligible,
                         period_hours=p, reference_period_hours=q,
                         compatible_periods=compatible, phase_drift_hours=drift,
                         timing_reference_hour=reference_hour,
                         display_status="compatible rhythms" if eligible else "not comparable"))
    return pd.DataFrame(rows)


def neighbour_coordination(frame, pos, *, neighbours=3, min_overlap=24,
                           permutations=999, seed=20260907):
    """Zero-delay Pearson correlation, with a whole-trace identity permutation.

The symmetric k-nearest-neighbour graph is specified from measured positions,
not correlations. The null permutes complete traces amongst positions within
coverage strata, preserving temporal autocorrelation and pair dependence.
This is a within-recording spatial association test, not an animal-level test.
"""
    if neighbours < 1 or min_overlap < 3 or permutations < 19:
        raise ValueError("need neighbours >= 1, overlap >= 3 and permutations >= 19")
    matrix = frame.pivot(index="identity", columns="hours", values="value")
    pos = pos.dropna(subset=["x", "y"]).set_index("identity")
    ids = pos.index.intersection(matrix.index)
    ids = ids[matrix.reindex(ids).notna().sum(axis=1).ge(min_overlap)]
    if len(ids) < neighbours + 3:
        raise ValueError("too few sufficiently observed cells for neighbours and non-neighbours")
    xy = pos.loc[ids, ["x", "y"]].to_numpy(float)
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    graph = np.zeros(distance.shape, bool)
    graph[np.arange(len(ids))[:, None], np.argsort(distance, axis=1)[:, :neighbours]] = True
    graph |= graph.T
    values = matrix.loc[ids].to_numpy(float)
    corr = np.full(distance.shape, np.nan)
    count = np.zeros(distance.shape, int)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            valid = np.isfinite(values[i]) & np.isfinite(values[j])
            count[i, j] = count[j, i] = valid.sum()
            if valid.sum() >= min_overlap and np.std(values[i, valid]) > 1e-9 and np.std(values[j, valid]) > 1e-9:
                corr[i, j] = corr[j, i] = np.corrcoef(values[i, valid], values[j, valid])[0, 1]
    upper = np.triu(np.ones(graph.shape, bool), 1)
    def contrast(array):
        near, far = array[upper & graph], array[upper & ~graph]
        if not np.isfinite(near).any() or not np.isfinite(far).any():
            return np.nan
        return float(np.nanmean(near) - np.nanmean(far))
    observed = contrast(corr)
    rng = np.random.default_rng(seed)
    # Fixed coverage quartiles, not one bin per rank: equal-coverage cells stay exchangeable.
    coverage = np.isfinite(values).sum(axis=1) / values.shape[1]
    strata = np.minimum((coverage * 4).astype(int), 3)
    groups = [np.flatnonzero(strata == key) for key in np.unique(strata)]
    null = []
    for _ in range(permutations):
        order = np.arange(len(ids))
        for group in groups:
            order[group] = rng.permutation(group)
        null.append(contrast(corr[np.ix_(order, order)]))
    null = np.asarray(null, float)
    valid_null = null[np.isfinite(null)]
    p = ((1 + np.count_nonzero(np.abs(valid_null) >= abs(observed))) / (len(valid_null) + 1)
         if np.isfinite(observed) and len(valid_null) else np.nan)
    pairs = [dict(identity_a=int(ids[i]), identity_b=int(ids[j]),
                  x=xy[i, 0], y=xy[i, 1], x2=xy[j, 0], y2=xy[j, 1],
                  distance=distance[i, j], value=corr[i, j], observations=count[i, j],
                  neighbour=bool(graph[i, j])) for i, j in zip(*np.where(upper))]
    near_n = int(np.isfinite(corr[upper & graph]).sum())
    far_n = int(np.isfinite(corr[upper & ~graph]).sum())
    statistics = pd.DataFrame([dict(
        test="two-sided coverage-stratified whole-trace spatial permutation",
        estimate=observed, p_value=p, q_value=p, correction="none (one field contrast)",
        alpha=.05, cells=len(ids), neighbour_pairs=near_n, non_neighbour_pairs=far_n,
        permutations=len(valid_null), seed=seed,
        null_mean=float(np.mean(valid_null)) if len(valid_null) else np.nan)])
    return pd.DataFrame(pairs), pd.DataFrame({"iteration": np.arange(len(null)), "value": null}), statistics


def coverage_gaps(occupied, hours, valid=None):
    """Elapsed time since last observed occupancy; observation breaks reset history.

The fixed denominator is pixels occupied at least once while measurable.
Unobserved frames are neither empty nor occupied; gap durations are sampled
    upper bounds on vacancy duration, not exact departure/return times.
"""
    occupied = np.asarray(occupied, bool)
    hours = np.asarray(hours, float)
    if occupied.ndim != 3 or len(hours) != len(occupied) or np.any(np.diff(hours) <= 0):
        raise ValueError("occupancy must be frame x y x x with increasing frame hours")
    valid = np.ones(occupied.shape, bool) if valid is None else np.broadcast_to(valid, occupied.shape)
    domain = (occupied & valid).any(axis=0)
    last = np.full(occupied.shape[1:], np.nan)
    age = np.full(occupied.shape, np.nan, dtype=np.float32)
    records = []
    for i, hour in enumerate(hours):
        last[~valid[i]] = np.nan
        present = occupied[i] & valid[i] & domain
        vacant = ~occupied[i] & valid[i] & domain
        last[present] = hour
        age[i, present] = 0
        age[i, vacant] = hour - last[vacant]
        denom = int((domain & valid[i]).sum())
        records.append(dict(hours=hour, frame_index=i, measurable_pixels=denom,
                            occupied_pixels=int(present.sum()), vacant_pixels=int(vacant.sum()),
                            value=float(present.sum() / denom) if denom else np.nan))
    return domain, age, pd.DataFrame(records)
