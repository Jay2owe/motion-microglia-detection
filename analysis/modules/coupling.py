"""Temporal coupling within cells and phase relationships between cells."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.modules.rhythms import _detrend, cosinor
from analysis.registry import Column, MeasurementContext, Output, register_derived


DEFAULTS = {
    "max_lag_frames": 24,
    "metric_pairs": [
        ["turnover_index", "corrected_mean"],
        ["area_px", "turnover_index"],
    ],
    "between_metrics": ["corrected_mean"],
    "min_overlap_frames": 60,
    "surrogates": 200,
    "random_state": 20260825,
    "fixed_period_hours": 24.0,
    "differenced_columns": [
        "gained_px", "lost_px", "turnover_index", "extension_bias", "area_change_px"
    ],
}


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3:
        return float("nan")
    a, b = left[usable], right[usable]
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ar1(values: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centred = values - np.nanmean(values)
    if len(values) < 3 or np.allclose(centred, 0):
        return generator.permutation(values)
    denominator = float(np.sum(centred[:-1] ** 2))
    phi = float(np.sum(centred[1:] * centred[:-1]) / denominator) if denominator else 0.0
    phi = float(np.clip(phi, -0.98, 0.98))
    sigma = float(np.nanstd(centred) * np.sqrt(max(1.0 - phi * phi, 1e-6)))
    out = np.empty(len(values), dtype=float)
    out[0] = generator.normal(0, np.nanstd(centred) or 1.0)
    for index in range(1, len(out)):
        out[index] = phi * out[index - 1] + generator.normal(0, sigma)
    return out + np.nanmean(values)


def _phase_randomised(values: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centred = values - np.nanmean(values)
    spectrum = np.fft.rfft(centred)
    if len(spectrum) > 2:
        phases = generator.uniform(0, 2 * np.pi, len(spectrum) - 2)
        spectrum[1:-1] = np.abs(spectrum[1:-1]) * np.exp(1j * phases)
    return np.fft.irfft(spectrum, n=len(values)) + np.nanmean(values)


def _lag_values(
    frames: np.ndarray, left: np.ndarray, right: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs where a negative lag means the first series leads."""
    right_by_frame = {int(frame): right[index] for index, frame in enumerate(frames)}
    a, b = [], []
    for frame, value in zip(frames, left):
        partner = int(frame - lag)
        if partner in right_by_frame:
            a.append(value)
            b.append(right_by_frame[partner])
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def _circular_difference(left: float, right: float, period: float) -> float:
    raw = abs((left - right) % period)
    return float(min(raw, period - raw))


#: Two tables that answer different questions with the same arithmetic: whether
#: two measurements of one cell move together at a lag, and whether two cells
#: peak at the same time. ``metric``, ``metric_a`` and ``metric_b`` hold column
#: names rather than numbers, so their "unit" is the vocabulary itself.
#:
#: The ``surrogate_`` columns are the null the correlation is read against, and
#: they take the ``reference`` role for exactly that reason: a correlation drawn
#: in the same colour as the band it has to clear invites the reader to compare
#: it with nothing.
PRODUCES = (
    # lag_profiles - one row per metric pair per lag
    Column("metric_a", "First measurement of the pair", "column name", "reference"),
    Column("metric_b", "Second measurement of the pair", "column name", "reference"),
    Column("lag_frames", "Lag", "frames", "motility"),
    Column("lag_hours", "Lag", "h", "motility"),
    Column("correlation", "Correlation at this lag", "-1 to 1", "morphology"),
    Column("pairs", "Frame pairs averaged", "count", "motility"),
    Column("surrogate_mean", "Correlation expected by chance", "-1 to 1", "reference"),
    Column("surrogate_lo", "Chance correlation, 2.5th percentile", "-1 to 1", "reference"),
    Column("surrogate_hi", "Chance correlation, 97.5th percentile", "-1 to 1", "reference"),
    Column("surrogate_model", "How the null series were made", "name", "reference"),
    # coupling - one row per cell pair per metric
    Column("identity_a", "First cell of the pair", "", "reference"),
    Column("identity_b", "Second cell of the pair", "", "reference"),
    Column("metric", "Which measurement this row is about", "column name", "reference"),
    Column("distance", "Distance between the two cells", "px", "motility"),
    Column("phase_difference", "Peak-to-peak offset between the two cells", "h", "rhythmic"),
    Column("overlap_frames", "Frames both cells were seen in", "frames", "reference"),
)


WRITES = (
    Output("lag_profiles", grain=("identity", "metric_a", "metric_b", "lag_frames")),
    Output("coupling",     grain=("identity_a", "identity_b", "metric")),
)


def _check_pairs(pairs: list[tuple], cell_frame: pd.DataFrame,
                 context: MeasurementContext) -> None:
    """Refuse a pair this module cannot honour, rather than skipping it.

    A *measured* column that is absent is skipped, and deliberately: it means a
    module was switched off, and a run with fewer modules should produce fewer
    lag profiles rather than stopping. A *side* column that is absent means
    something else entirely - the user asked for a computation on their own
    data, and the only reason it is not here is that this module was not granted
    it. Skipping that is the failure this whole feature exists to remove: a run
    that finishes, with the requested result quietly missing.

    A non-numeric side column is refused for the same reason. A genotype call is
    a string, and correlating one against a time series either fails obscurely
    inside numpy or coerces to something meaningless.
    """
    from pandas.api.types import is_numeric_dtype

    from analysis.summarise import side_column_names

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                f"coupling: metric_pairs entry {list(pair)} has {len(pair)} "
                "members; each entry is the two series to correlate against "
                "each other, so it must have exactly two"
            )

    supplied = side_column_names(context)
    for metric_a, metric_b in pairs:
        for metric in (metric_a, metric_b):
            if metric not in supplied:
                continue                    # a measured column: the skip is right
            if metric not in cell_frame.columns:
                raise ValueError(
                    f"coupling: metric_pairs names the side column {metric!r}, "
                    "which this module was not granted. Add it to this module's "
                    'own settings: "coupling": {"side_columns": '
                    f'["{metric}"]}}. A derived module reads a user column only '
                    "where the configuration names it, so that nothing picks one "
                    "up by scanning the table."
                )
            if not is_numeric_dtype(cell_frame[metric]):
                raise ValueError(
                    f"coupling: the side column {metric!r} is "
                    f"{cell_frame[metric].dtype}, and a lag profile is a "
                    "correlation between two numbers. Give it as a number, or "
                    "pair a different column."
                )


@register_derived(
    name="coupling",
    description="Gap-respecting lag profiles and between-cell phase similarity",
    needs_columns=("identity", "frame_index", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("coupling")}
    maximum_lag = int(params["max_lag_frames"])
    minimum = int(params["min_overlap_frames"])
    n_surrogates = int(params["surrogates"])
    generator = np.random.default_rng(int(params["random_state"]))
    differenced = set(params["differenced_columns"])
    lag_rows: list[dict] = []

    pairs = [tuple(pair) for pair in params["metric_pairs"]]
    _check_pairs(pairs, cell_frame, context)
    for identity, group in cell_frame.groupby("identity", sort=True):
        group = group.sort_values("frame_index")
        frames = group["frame_index"].to_numpy(int)
        for metric_a, metric_b in pairs:
            if metric_a not in group or metric_b not in group:
                continue
            usable = group[["frame_index", metric_a, metric_b]].dropna()
            if len(usable) < minimum:
                continue
            pair_frames = usable["frame_index"].to_numpy(int)
            left = usable[metric_a].to_numpy(float)
            right = usable[metric_b].to_numpy(float)
            null_kind = "ar1" if metric_a in differenced or metric_b in differenced else "phase_randomised"
            null_profiles = np.full((n_surrogates, maximum_lag * 2 + 1), np.nan)
            for replicate in range(n_surrogates):
                fake = _ar1(right, generator) if null_kind == "ar1" else _phase_randomised(right, generator)
                for column, lag in enumerate(range(-maximum_lag, maximum_lag + 1)):
                    a, b = _lag_values(pair_frames, left, fake, lag)
                    null_profiles[replicate, column] = _correlation(a, b)
            for column, lag in enumerate(range(-maximum_lag, maximum_lag + 1)):
                a, b = _lag_values(pair_frames, left, right, lag)
                null_values = null_profiles[:, column]
                finite_null = null_values[np.isfinite(null_values)]
                lag_rows.append(
                    {
                        "identity": int(identity),
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "lag_frames": lag,
                        "lag_hours": context.scale.hours(lag),
                        "correlation": _correlation(a, b),
                        "pairs": int(np.count_nonzero(np.isfinite(a) & np.isfinite(b))),
                        "surrogate_mean": float(np.mean(finite_null)) if finite_null.size else np.nan,
                        "surrogate_lo": float(np.percentile(finite_null, 2.5)) if finite_null.size else np.nan,
                        "surrogate_hi": float(np.percentile(finite_null, 97.5)) if finite_null.size else np.nan,
                        "surrogate_model": null_kind,
                    }
                )

    between_rows: list[dict] = []
    period = float(params["fixed_period_hours"])
    between_metrics = [metric for metric in params["between_metrics"] if metric in cell_frame]
    identities = sorted(int(value) for value in cell_frame["identity"].dropna().unique())
    for metric in between_metrics:
        prepared: dict[int, pd.DataFrame] = {}
        phase: dict[int, float] = {}
        position: dict[int, np.ndarray] = {}
        for identity in identities:
            columns = ["frame_index", "hours", metric]
            position_columns = [column for column in ("centroid_y", "centroid_x") if column in cell_frame]
            group = cell_frame[cell_frame["identity"] == identity][columns + position_columns].dropna(
                subset=[metric]
            ).sort_values("frame_index")
            if len(group) < minimum:
                continue
            prepared[identity] = group
            values = group[metric].to_numpy(float)
            hours = group["hours"].to_numpy(float)
            fitted = cosinor(hours, _detrend(hours, values, "linear"), period, np.mean(np.abs(values)))
            if fitted:
                phase[identity] = float(fitted["cosinor_peak_hour"])
            if len(position_columns) == 2:
                position[identity] = group[position_columns].median().to_numpy(float)

        available = sorted(prepared)
        for left_index, identity_a in enumerate(available):
            for identity_b in available[left_index + 1:]:
                merged = prepared[identity_a][["frame_index", metric]].merge(
                    prepared[identity_b][["frame_index", metric]], on="frame_index",
                    suffixes=("_a", "_b"), how="inner",
                )
                if len(merged) < minimum:
                    continue
                distance = (
                    float(np.linalg.norm(position[identity_a] - position[identity_b]))
                    if identity_a in position and identity_b in position else np.nan
                )
                between_rows.append(
                    {
                        "identity_a": identity_a,
                        "identity_b": identity_b,
                        "metric": metric,
                        "distance": context.scale.length(distance) if np.isfinite(distance) else np.nan,
                        "correlation": _correlation(
                            merged[f"{metric}_a"].to_numpy(float),
                            merged[f"{metric}_b"].to_numpy(float),
                        ),
                        "phase_difference": (
                            _circular_difference(phase[identity_a], phase[identity_b], period)
                            if identity_a in phase and identity_b in phase else np.nan
                        ),
                        "overlap_frames": int(len(merged)),
                    }
                )

    lag_columns = [
        "identity", "metric_a", "metric_b", "lag_frames", "lag_hours", "correlation",
        "pairs", "surrogate_mean", "surrogate_lo", "surrogate_hi", "surrogate_model",
    ]
    coupling_columns = [
        "identity_a", "identity_b", "metric", "distance", "correlation",
        "phase_difference", "overlap_frames",
    ]
    return {
        "lag_profiles": pd.DataFrame(lag_rows, columns=lag_columns),
        "coupling": pd.DataFrame(between_rows, columns=coupling_columns),
    }
