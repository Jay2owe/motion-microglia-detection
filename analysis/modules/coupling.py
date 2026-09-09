"""Temporal coupling within cells and phase relationships between cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analysis import circadian as workbench
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
    **workbench.PERIOD_ANALYSIS_DEFAULTS,
    **workbench.DETREND_DEFAULTS,
    "differenced_columns": [
        "gained_px", "lost_px", "turnover_index", "extension_bias", "area_change_px"
    ],
}


def _parameters(context: MeasurementContext) -> dict:
    """Coupling settings after inheriting the main rhythm analysis choices."""
    params = {
        **DEFAULTS,
        **context.module_params("rhythms"),
        **context.module_params("coupling"),
    }
    return {**params, **workbench.detrend_settings(params)}


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


def _pair_differences(left: np.ndarray, right: np.ndarray,
                      circular_period: float | None) -> np.ndarray:
    """Absolute pair differences, optionally wrapped around a circular scale."""
    difference = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    if circular_period is not None:
        period = float(circular_period)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("circular_period must be finite and positive")
        difference = np.minimum(np.mod(difference, period),
                                period - np.mod(difference, period))
    return difference


def _row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlations between one vector and each row of a matrix."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_centre = left - np.mean(left)
    right_centre = right - np.mean(right, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centre ** 2) * np.sum(right_centre ** 2, axis=1)
    )
    numerator = np.sum(right_centre * left_centre[None, :], axis=1)
    return np.divide(
        numerator, denominator,
        out=np.full(len(right), np.nan, dtype=float), where=denominator > 0,
    )


def contact_metric_permutation_test(
    contacts: pd.DataFrame,
    values: pd.Series,
    *,
    circular_period: float | None = None,
    shuffles: int = 10_000,
    random_state: int = 24_051_986,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Compare contact-pair differences with values shuffled over the same network.

    The contact edges and their repeated-cell structure never move. Only the
    cell-level measurement is reassigned, so a highly connected cell remains a
    highly connected node in every null replicate. The returned duration test
    is Spearman's rank correlation between contact hours and pair difference;
    negative values mean longer-contacting pairs are more alike.
    """
    required = {"identity_a", "identity_b", "hours_in_contact"}
    missing = required - set(contacts.columns)
    if missing:
        raise KeyError(f"contacts are missing {sorted(missing)}")
    shuffles = int(shuffles)
    if shuffles < 100:
        raise ValueError("shuffles must be at least 100 for a useful null interval")

    series = pd.to_numeric(values, errors="coerce").dropna()
    if not series.index.is_unique:
        series = series.groupby(level=0).median()
    identities = series.index.to_numpy()
    position = {int(identity): index for index, identity in enumerate(identities)}
    eligible = contacts[
        contacts["identity_a"].isin(identities)
        & contacts["identity_b"].isin(identities)
    ].copy()
    eligible = eligible.sort_values(
        ["hours_in_contact", "identity_a", "identity_b"],
        ascending=[False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    columns = [
        "identity_a", "identity_b", "hours_in_contact", "value_a", "value_b",
        "pair_difference", "random_mean_difference", "difference_ratio",
    ]
    if len(eligible) < 3 or len(series) < 3:
        return pd.DataFrame(columns=columns), {
            "status": "too_few_pairs", "cells_available": int(len(series)),
            "pairs": int(len(eligible)), "shuffles": shuffles,
            "random_state": int(random_state),
        }

    left_index = np.asarray([position[int(value)] for value in eligible["identity_a"]], dtype=int)
    right_index = np.asarray([position[int(value)] for value in eligible["identity_b"]], dtype=int)
    cell_values = series.to_numpy(dtype=float)
    observed_differences = _pair_differences(
        cell_values[left_index], cell_values[right_index], circular_period,
    )

    generator = np.random.default_rng(int(random_state))
    # Sorting independent uniform draws yields independent uniform permutations
    # and makes all replicates one vectorised operation rather than a slow loop.
    orders = np.argsort(
        generator.random((shuffles, len(cell_values))), axis=1, kind="quicksort"
    )
    shuffled_values = cell_values[orders]
    shuffled_differences = _pair_differences(
        shuffled_values[:, left_index], shuffled_values[:, right_index],
        circular_period,
    )
    null_means = np.mean(shuffled_differences, axis=1)
    observed_mean = float(np.mean(observed_differences))
    null_mean = float(np.mean(null_means))
    difference_ratio = observed_mean / null_mean if null_mean > 0 else np.nan

    lower_tail = (1 + np.count_nonzero(null_means <= observed_mean)) / (shuffles + 1)
    upper_tail = (1 + np.count_nonzero(null_means >= observed_mean)) / (shuffles + 1)
    similarity_p = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))

    contact_ranks = rankdata(eligible["hours_in_contact"].to_numpy(dtype=float))
    observed_rank = rankdata(observed_differences)
    duration_rho = float(_row_correlations(contact_ranks, observed_rank[None, :])[0])
    shuffled_ranks = rankdata(shuffled_differences, axis=1)
    null_rho = _row_correlations(contact_ranks, shuffled_ranks)
    finite_rho = null_rho[np.isfinite(null_rho)]
    duration_p = (
        float((1 + np.count_nonzero(np.abs(finite_rho) >= abs(duration_rho)))
              / (len(finite_rho) + 1))
        if np.isfinite(duration_rho) and len(finite_rho) else np.nan
    )

    pair_data = eligible[["identity_a", "identity_b", "hours_in_contact"]].copy()
    pair_data["value_a"] = cell_values[left_index]
    pair_data["value_b"] = cell_values[right_index]
    pair_data["pair_difference"] = observed_differences
    pair_data["random_mean_difference"] = null_mean
    pair_data["difference_ratio"] = (
        observed_differences / null_mean if null_mean > 0 else np.nan
    )
    null_ratio = null_means / null_mean if null_mean > 0 else np.full(shuffles, np.nan)
    return pair_data[columns], {
        "status": "ok",
        "cells_available": int(len(series)),
        "pairs": int(len(eligible)),
        "observed_mean_difference": observed_mean,
        "random_mean_difference": null_mean,
        "difference_ratio": float(difference_ratio),
        "similarity_null_lo": float(np.nanquantile(null_ratio, 0.025)),
        "similarity_null_hi": float(np.nanquantile(null_ratio, 0.975)),
        "similarity_p_value": similarity_p,
        "duration_spearman_rho": duration_rho,
        "duration_null_lo": float(np.nanquantile(finite_rho, 0.025)),
        "duration_null_hi": float(np.nanquantile(finite_rho, 0.975)),
        "duration_p_value": duration_p,
        "shuffles": shuffles,
        "random_state": int(random_state),
    }


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
    Column("phase_difference_fraction", "Peak-to-peak offset as a fraction of a cycle", "0-0.5 cycle", "rhythmic"),
    Column("period_a_hours", "Selected period for the first cell", "h", "rhythmic"),
    Column("period_b_hours", "Selected period for the second cell", "h", "rhythmic"),
    Column("pair_phase_reference_period_hours", "Mean selected period used to express the pair offset in hours", "h", "reference"),
    Column("period_estimation_method", "Method used to estimate period and phase", "method", "reference"),
    Column("workbench_version", "Circadian Workbench version", "version", "reference"),
    Column("overlap_frames", "Frames both cells were seen in", "frames", "reference"),
    Column("detrend", "How the trend was removed", "name", "reference"),
    Column("detrend_window_hours", "Baseline window used for detrending", "h", "reference"),
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
    resolve_params=_parameters,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    # A coupling block may override either shared value when this analysis
    # genuinely needs a different baseline.
    params = _parameters(context)
    detrending = workbench.detrend_settings(params)
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
            usable = group[["frame_index", "hours", metric_a, metric_b]].dropna()
            if len(usable) < minimum:
                continue
            pair_frames = usable["frame_index"].to_numpy(int)
            pair_hours = usable["hours"].to_numpy(float)
            left = np.asarray(workbench.detrend_trace(
                pair_hours, usable[metric_a].to_numpy(float), detrending,
            )["values"], dtype=float)
            right = np.asarray(workbench.detrend_trace(
                pair_hours, usable[metric_b].to_numpy(float), detrending,
            )["values"], dtype=float)
            finite_pair = np.isfinite(left) & np.isfinite(right)
            pair_frames = pair_frames[finite_pair]
            left, right = left[finite_pair], right[finite_pair]
            if len(pair_frames) < minimum:
                continue
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
                        "detrend": detrending["detrend"],
                        "detrend_window_hours": detrending["detrend_window_hours"],
                    }
                )

    between_rows: list[dict] = []
    estimator_method = str(params.get(
        "period_estimation_method", params.get("primary_rhythm_test", "lomb")
    ))
    between_metrics = [metric for metric in params["between_metrics"] if metric in cell_frame]
    identities = sorted(int(value) for value in cell_frame["identity"].dropna().unique())
    for metric in between_metrics:
        prepared: dict[int, pd.DataFrame] = {}
        phase_fraction: dict[int, float] = {}
        periods: dict[int, float] = {}
        position: dict[int, np.ndarray] = {}
        for identity in identities:
            columns = ["frame_index", "hours", metric]
            position_columns = [column for column in ("centroid_y", "centroid_x") if column in cell_frame]
            group = cell_frame[cell_frame["identity"] == identity][columns + position_columns].dropna(
                subset=[metric]
            ).sort_values("frame_index")
            if len(group) < minimum:
                continue
            values = group[metric].to_numpy(float)
            hours = group["hours"].to_numpy(float)
            detrended = np.asarray(
                workbench.detrend_trace(hours, values, detrending)["values"],
                dtype=float,
            )
            finite = np.isfinite(detrended)
            group = group.copy()
            group["_detrended"] = detrended
            prepared[identity] = group
            fitted = workbench.estimate_one(
                hours[finite], detrended[finite], params,
                estimator_method, detrend="none",
            )
            fitted_period = fitted.get("period_hours")
            fitted_phase = fitted.get("phase_hours")
            if (fitted.get("status") == "ok"
                    and fitted_period is not None and np.isfinite(fitted_period)
                    and float(fitted_period) > 0
                    and fitted_phase is not None and np.isfinite(fitted_phase)):
                periods[identity] = float(fitted_period)
                phase_fraction[identity] = (
                    float(fitted_phase) % float(fitted_period)
                ) / float(fitted_period)
            if len(position_columns) == 2:
                position[identity] = group[position_columns].median().to_numpy(float)

        available = sorted(prepared)
        for left_index, identity_a in enumerate(available):
            for identity_b in available[left_index + 1:]:
                merged = prepared[identity_a][["frame_index", "_detrended"]].merge(
                    prepared[identity_b][["frame_index", "_detrended"]], on="frame_index",
                    suffixes=("_a", "_b"), how="inner",
                )
                if len(merged) < minimum:
                    continue
                distance = (
                    float(np.linalg.norm(position[identity_a] - position[identity_b]))
                    if identity_a in position and identity_b in position else np.nan
                )
                pair_period = (
                    float(np.mean([periods[identity_a], periods[identity_b]]))
                    if identity_a in periods and identity_b in periods else np.nan
                )
                phase_offset = (
                    _circular_difference(
                        phase_fraction[identity_a], phase_fraction[identity_b], 1.0
                    )
                    if identity_a in phase_fraction and identity_b in phase_fraction
                    else np.nan
                )
                between_rows.append(
                    {
                        "identity_a": identity_a,
                        "identity_b": identity_b,
                        "metric": metric,
                        "distance": context.scale.length(distance) if np.isfinite(distance) else np.nan,
                        "correlation": _correlation(
                            merged["_detrended_a"].to_numpy(float),
                            merged["_detrended_b"].to_numpy(float),
                        ),
                        "phase_difference": (
                            phase_offset * pair_period
                            if np.isfinite(phase_offset) and np.isfinite(pair_period)
                            else np.nan
                        ),
                        "phase_difference_fraction": phase_offset,
                        "period_a_hours": periods.get(identity_a, np.nan),
                        "period_b_hours": periods.get(identity_b, np.nan),
                        "pair_phase_reference_period_hours": pair_period,
                        "period_estimation_method": estimator_method,
                        "workbench_version": workbench.WORKBENCH_VERSION,
                        "overlap_frames": int(len(merged)),
                        "detrend": detrending["detrend"],
                        "detrend_window_hours": detrending["detrend_window_hours"],
                    }
                )

    lag_columns = [
        "identity", "metric_a", "metric_b", "lag_frames", "lag_hours", "correlation",
        "pairs", "surrogate_mean", "surrogate_lo", "surrogate_hi", "surrogate_model",
        "detrend", "detrend_window_hours",
    ]
    coupling_columns = [
        "identity_a", "identity_b", "metric", "distance", "correlation",
        "phase_difference", "phase_difference_fraction", "period_a_hours",
        "period_b_hours", "pair_phase_reference_period_hours", "period_estimation_method",
        "overlap_frames", "detrend", "detrend_window_hours", "workbench_version",
    ]
    return {
        "lag_profiles": pd.DataFrame(lag_rows, columns=lag_columns),
        "coupling": pd.DataFrame(between_rows, columns=coupling_columns),
    }
