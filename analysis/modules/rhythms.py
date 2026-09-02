"""Does each cell cycle, and when does it peak?

Two independent tests are run on every requested measurement, because they fail
in different ways and agreement between them is the evidence:

* **Cosinor** fits a single cosine wave and reports how tall it is (amplitude),
  when it peaks (acrophase) and how much of the variation it explains. It is
  strong when a rhythm is close to a clean sine and weak when it is not.
* **Lomb-Scargle** scans a range of periods and reports which one carries the
  most power. It handles missing frames without interpolation, which matters
  because cells drop out of the outlines for a frame or two.

Both answer "is there a rhythm and how big". Neither answers *when*, which is
how a circadian result is usually reported, so every row also carries the
**non-parametric block**: the busiest ten hours of the average day and the
quietest five (M10 and L5), how far apart they are (relative amplitude), when
the active phase starts and ends, how much one day resembles the next
(interdaily stability) and how broken up the daily pattern is (intradaily
variability). Those are transcribed from the lab's own CircadianWorkbench
rather than derived here; see the block above ``PRODUCES``.

None of this is specific to brightness. The module fits whatever ``metrics``
names, so cell size, branching, footprint turnover and step size get the same
readouts as the reporter does.

Honest limit for this dataset: 99 frames at 30 minutes is 49.5 hours, which is
just over two cycles of a 24-hour rhythm. Two cycles can show that something
oscillates; they cannot pin the period tightly. Every row therefore carries
``cycles_covered``, and a fit with fewer than three cycles is flagged so it can
never be quoted as a precise period without the caveat travelling with it. The
same limit bites harder on interdaily stability, which compares one day with
the next and has one comparison to work with here: ``days_covered`` and
``stability_underdetermined`` are what say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from analysis.registry import Column, MeasurementContext, Output, register_derived

DEFAULTS = {
    "metrics": [
        "area_px",
        "corrected_mean",
        "solidity",
        "ramification_index",
        "turnover_index",
        "step_px",
    ],
    "period_search_hours": [16.0, 32.0],
    "period_step_hours": 0.1,
    "fixed_period_hours": 24.0,
    "detrend": "linear",
    "min_observations": 24,
    "min_cycles_for_confident_period": 3.0,
    "rhythmic_alpha": 0.05,
    # Surrogate control. Slowly drifting noise passes a cosinor fit far more
    # often than a p-value suggests, so the package measures its own false
    # positive rate rather than assuming it. "ar1" builds surrogates with the
    # same variance and the same frame-to-frame correlation as the real trace
    # but no rhythm; "shuffle" destroys that correlation and is therefore too
    # easy to beat. Set to 0 to skip.
    "null_surrogates_per_cell": 5,
    "null_model": "ar1",
    "null_seed": 20260824,
    # --- the non-parametric block ---------------------------------------
    # Window lengths are the actigraphy conventions and are settings rather
    # than constants: a 30-minute frame interval makes M10 twenty frames, and
    # a dataset imaged every 5 minutes would make it 120.
    "bin_hours": 1.0,               # the folded day's resolution
    "high_window_hours": 10.0,      # M10
    "low_window_hours": 5.0,        # L5
    # ClockLab's onset template: how much quiet has to sit before a candidate
    # hour and how much activity after it, and where the line between the two
    # is drawn. 80 means the threshold sits at the 20th percentile.
    "onset_off_hours": 6.0,
    "onset_on_hours": 6.0,
    "onset_threshold_percent": 80.0,
    "min_days_for_stability": 3.0,
}


def _detrend(hours: np.ndarray, values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values
    if method == "linear":
        slope, intercept = np.polyfit(hours, values, 1)
        return values - (slope * hours + intercept)
    if method == "mean":
        return values - values.mean()
    raise ValueError(f"unknown detrend method {method!r}")


def cosinor(
    hours: np.ndarray,
    values: np.ndarray,
    period_hours: float,
    reference_level: float | None = None,
) -> dict:
    """Least-squares fit of ``mesor + amplitude * cos(2*pi*t/period - acrophase)``.

    ``values`` is normally detrended, which puts its own mesor at roughly zero.
    Relative amplitude must therefore be taken against ``reference_level`` - the
    mean of the trace before detrending - or it divides by nothing and returns
    nonsense.
    """
    n = len(hours)
    if n < 4:
        return {}
    omega = 2.0 * np.pi / period_hours
    design = np.column_stack([np.ones(n), np.cos(omega * hours), np.sin(omega * hours)])
    coefficients, residual_sum, *_ = np.linalg.lstsq(design, values, rcond=None)
    mesor, beta, gamma = coefficients
    fitted = design @ coefficients
    residuals = values - fitted
    ss_residual = float(np.sum(residuals ** 2))
    ss_total = float(np.sum((values - values.mean()) ** 2))
    amplitude = float(np.hypot(beta, gamma))
    acrophase = float(np.arctan2(-gamma, beta))
    peak_hour = float((acrophase % (2 * np.pi)) / omega)

    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else np.nan
    degrees = n - 3
    if degrees > 0 and ss_residual > 0 and ss_total > ss_residual:
        f_statistic = ((ss_total - ss_residual) / 2.0) / (ss_residual / degrees)
        p_value = float(stats.f.sf(f_statistic, 2, degrees))
    else:
        f_statistic, p_value = np.nan, np.nan

    # Relative amplitude error: standard error of the amplitude over the
    # amplitude itself. Below about 0.3 is conventionally called rhythmic.
    if degrees > 0 and ss_residual > 0:
        sigma_squared = ss_residual / degrees
        covariance = sigma_squared * np.linalg.pinv(design.T @ design)
        amplitude_variance = (
            beta ** 2 * covariance[1, 1]
            + gamma ** 2 * covariance[2, 2]
            + 2 * beta * gamma * covariance[1, 2]
        ) / max(amplitude ** 2, 1e-12)
        relative_amplitude_error = float(np.sqrt(max(amplitude_variance, 0.0)) / amplitude) if amplitude > 0 else np.nan
    else:
        relative_amplitude_error = np.nan

    denominator = reference_level if reference_level is not None else mesor
    return {
        "cosinor_mesor": float(mesor),
        "cosinor_amplitude": amplitude,
        "cosinor_relative_amplitude": (
            float(amplitude / abs(denominator)) if denominator else np.nan
        ),
        "cosinor_peak_hour": peak_hour,
        "cosinor_r_squared": float(r_squared),
        "cosinor_p_value": p_value,
        "cosinor_relative_amplitude_error": relative_amplitude_error,
    }


def lomb_scargle(hours: np.ndarray, values: np.ndarray, periods: np.ndarray) -> dict:
    """Periodogram over a period grid, with an analytic false-alarm probability."""
    centred = values - values.mean()
    if np.allclose(centred, 0):
        return {}
    angular = 2.0 * np.pi / periods
    power = signal.lombscargle(hours, centred, angular, normalize=True)
    best = int(np.argmax(power))
    peak_power = float(power[best])
    independent = max(len(periods), 2)
    false_alarm = float(1.0 - (1.0 - np.exp(-peak_power * len(hours) / 2.0)) ** independent)
    return {
        "lombscargle_period_hours": float(periods[best]),
        "lombscargle_power": peak_power,
        "lombscargle_false_alarm": min(max(false_alarm, 0.0), 1.0),
    }


def rayleigh(peak_hours: np.ndarray, period_hours: float) -> dict:
    """Do the cells peak at the same time of day, or at random times?

    Each cell's peak is a direction on a 24-hour clock face. If the cells are
    independent oscillators the arrows point everywhere and cancel; if the
    tissue shares a clock they add up. ``vector_length`` is how much they add
    up, from 0 (scattered) to 1 (identical), and the Rayleigh p-value is the
    chance of seeing that much agreement from random directions.
    """
    angles = 2.0 * np.pi * np.asarray(peak_hours, dtype=float) / period_hours
    angles = angles[np.isfinite(angles)]
    n = len(angles)
    if n < 3:
        return {}
    cosine, sine = float(np.cos(angles).sum()), float(np.sin(angles).sum())
    resultant = float(np.hypot(cosine, sine))
    # Zar (1999) approximation to the Rayleigh probability.
    p_value = float(
        np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n ** 2 - resultant ** 2)) - (1.0 + 2.0 * n))
    )
    mean_hour = float((np.arctan2(sine, cosine) % (2 * np.pi)) * period_hours / (2 * np.pi))
    return {
        "cells": n,
        "vector_length": resultant / n,
        "mean_peak_hour": mean_hour,
        "rayleigh_p_value": min(p_value, 1.0),
    }


def _surrogate(values: np.ndarray, model: str, generator: np.random.Generator) -> np.ndarray:
    """A trace with the same statistics as this cell but no rhythm."""
    if model == "shuffle":
        return generator.permutation(values)
    if model == "ar1":
        centred = values - values.mean()
        if len(centred) < 3 or np.allclose(centred, 0):
            return generator.permutation(values)
        denominator = float(np.sum(centred[:-1] ** 2))
        phi = float(np.sum(centred[1:] * centred[:-1]) / denominator) if denominator > 0 else 0.0
        phi = float(np.clip(phi, -0.98, 0.98))
        sigma = float(np.std(centred) * np.sqrt(max(1.0 - phi ** 2, 1e-6)))
        out = np.empty(len(centred))
        out[0] = generator.normal(0.0, np.std(centred))
        noise = generator.normal(0.0, sigma, len(centred) - 1)
        for i in range(1, len(centred)):
            out[i] = phi * out[i - 1] + noise[i - 1]
        return out + values.mean()
    raise ValueError(f"unknown null model {model!r}")


# ---------------------------------------------------------------------------
# The non-parametric block: when the active phase runs, rather than how big it
# is. Everything below is transcribed from the lab's own CircadianWorkbench
# (v0.5.0, MIT, `circadian-workbench` on PyPI), `src/circadian_workbench/
# analysis.py`: `_circular_window` at line 1362, `nonparametric_metrics` at
# 1375, `_template_scores` at 420 and `_template_marker` at 438.
#
# Copied rather than imported. The functions are eighty lines and hold no state,
# and the package they live in pulls a web stack - FastAPI, uvicorn, statsmodels
# - into what is meant to ship as one measurement tool.
# `test_nonparametric_circadian` runs the same trace through both and requires
# the same answer, so the copy cannot drift silently; the version copied from is
# recorded here so the two can be diffed.
#
# Three things had to be adapted, and each is marked where it happens:
#
# 1. **The workbench knows the time of day and this package does not.** Every
#    function there is written against real timestamps. `cell_frame` carries
#    `hours` - hours since the recording started - and nothing else, so the day
#    is folded on `hours % 24`. The arithmetic is identical; what the answer
#    means is not. An onset of 7.5 is "seven and a half hours in", not "half
#    past seven", and two movies started at different times of day cannot have
#    their onsets compared at all. Every affected column says "h from start" in
#    its unit. Adding a recording start time to the configuration is what would
#    fix it, and it is a decision nobody has made.
# 2. **A bin is a mean, not a sum.** Activity counts add up over a bin;
#    brightness, area and solidity do not.
# 3. **Nothing is excluded as a structural zero.** The workbench takes its
#    threshold over non-zero bins, because a zero count means the beam was never
#    broken - an absence rather than a low value. A cell's area is never
#    structurally zero and a turnover of zero is a real measurement, so the
#    threshold here is taken over every measured bin.


def _binned(hours: np.ndarray, values: np.ndarray, bin_hours: float) -> np.ndarray:
    """Mean value per bin, anchored at hour zero of the *recording*.

    Anchored at the recording rather than at each cell's own first frame, which
    is the whole reason two cells' onsets can be compared with each other. A
    per-cell anchor would fold a cell that appeared five hours in onto a day
    starting five hours later, and every hour reported for it would be measured
    on a different clock from its neighbour's - silently, and only visibly wrong
    once somebody compared two cells. The workbench anchors on the first
    midnight for the same reason; this package has no midnight, so it uses the
    only shared origin it has.

    The leading bins of a late-arriving cell are therefore ``NaN``, which is
    exactly what they are: not looked at. Gap-aware throughout - a cell not
    measured anywhere in a bin gives ``NaN`` rather than zero, because zero is a
    value and "not looked at" is not, and every measure below treats the two
    differently.
    """
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    if not len(hours):
        return np.zeros(0)
    index = np.floor(hours / bin_hours).astype(int)
    binned = np.full(int(index.max()) + 1, np.nan)
    for position in range(len(binned)):
        chosen = values[index == position]
        if len(chosen):
            binned[position] = float(np.mean(chosen))
    return binned


def _folded_day(binned: np.ndarray, bins_per_day: int) -> np.ndarray:
    """Every day of the recording averaged into one, bin by bin.

    This is the step an independent implementation gets wrong. L5 and M10 are
    not the quietest and busiest stretches *somewhere in the recording*; they
    are stretches of the average day, so a cell quiet at the same hour on both
    days scores lower than one quiet for five hours once.
    """
    profile = np.full(bins_per_day, np.nan)
    for position in range(bins_per_day):
        chosen = binned[position::bins_per_day]
        if np.isfinite(chosen).any():
            profile[position] = float(np.nanmean(chosen))
    return profile


def _circular_window(profile: np.ndarray, window_bins: int, mode: str) -> tuple[int, float]:
    """Best or worst run of bins on the folded day, wrapping past the end.

    Circular because the quietest five hours of a day very often straddle
    midnight, and a window that stopped at the edge would miss them. A window
    is scored on the bins it has, so a gap narrows it rather than voiding it -
    ``hours_binned`` is what says how much of the day was measured at all.
    """
    scores = np.full(len(profile), np.nan)
    for start in range(len(profile)):
        indices = np.mod(np.arange(start, start + window_bins), len(profile))
        window = profile[indices]
        if np.isfinite(window).any():
            scores[start] = float(np.nanmean(window))
    if not np.isfinite(scores).any():
        return 0, np.nan
    index = int(np.nanargmin(scores) if mode == "min" else np.nanargmax(scores))
    return index, float(scores[index])


def _intradaily_variability(binned: np.ndarray) -> float:
    """How choppy the trace is hour to hour, against how much it varies overall.

    Only pairs where both hours were measured contribute, and the denominator
    counts those pairs rather than assuming every hour has a predecessor. High
    means the rhythm is broken into fragments rather than one rise and fall.
    """
    valid = binned[np.isfinite(binned)]
    if len(valid) <= 2:
        return np.nan
    adjacent = np.isfinite(binned[1:]) & np.isfinite(binned[:-1])
    denominator = float(np.sum((valid - np.mean(valid)) ** 2))
    if denominator <= 0 or not adjacent.any():
        return np.nan
    return float(
        len(valid) * np.sum((binned[1:][adjacent] - binned[:-1][adjacent]) ** 2)
        / (adjacent.sum() * denominator)
    )


def _interdaily_stability(binned: np.ndarray, bins_per_day: int) -> float:
    """How much one day resembles the next, from 0 to 1.

    Needs more than one day of measured bins, which is the workbench's guard and
    the reason a one-day recording returns nothing rather than a number. Two
    days is one comparison and is barely a measurement either, which is what
    ``stability_underdetermined`` is for.
    """
    valid = binned[np.isfinite(binned)]
    if len(valid) <= bins_per_day:
        return np.nan
    grand = float(np.mean(valid))
    denominator = float(bins_per_day * np.sum((valid - grand) ** 2))
    if denominator <= 0:
        return np.nan
    profile = _folded_day(binned, bins_per_day)
    return float(len(valid) * np.nansum((profile - grand) ** 2) / denominator)


def _template_scores(thresholded: np.ndarray, off_bins: int, on_bins: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """ClockLab-style circular onset and offset template scores.

    For each candidate bin, how much quiet sits in the window before it and how
    much activity follows. The best-matching bin is the onset. Both windows wrap
    circularly, so an onset near the end of the day is found rather than lost at
    the edge.
    """
    count = len(thresholded)
    onset = np.full(count, np.nan)
    offset = np.full(count, np.nan)
    for candidate in range(count):
        before_off = thresholded[np.mod(np.arange(candidate - off_bins, candidate), count)]
        after_on = thresholded[np.mod(np.arange(candidate, candidate + on_bins), count)]
        before_on = thresholded[np.mod(np.arange(candidate - on_bins, candidate), count)]
        after_off = thresholded[np.mod(np.arange(candidate, candidate + off_bins), count)]
        onset[candidate] = (np.sum(after_on) - np.sum(before_off)) / (on_bins + off_bins)
        offset[candidate] = (np.sum(before_on) - np.sum(after_off)) / (on_bins + off_bins)
    return onset, offset


def _template_marker(scores: np.ndarray, bin_hours: float) -> float:
    """Hour of the best template match, or ``NaN`` when the day carries no phase.

    The guard is the point of this function and it is copied with its reason
    intact. A day whose bins all fall on the same side of the threshold scores
    identically everywhere, and ``argmax`` on a flat curve returns bin 0, which
    reads downstream as a perfectly regular onset at the start of the recording.
    The workbench shipped that: an all-zero record produced a period of exactly
    24.0 hours with zero fit error, described in its own comment as "the most
    convincing-looking result in the app, and entirely fabricated".

    ``nanmax - nanmin`` rather than ``ptp``, because ``ptp`` returns ``NaN`` if
    any element is ``NaN`` and ``nan <= 0.0`` is ``False``, which would bypass
    the guard exactly when it is needed.
    """
    if not np.isfinite(scores).any():
        return np.nan
    if float(np.nanmax(scores) - np.nanmin(scores)) <= 0.0:
        return np.nan
    return float(int(np.nanargmax(scores)) * bin_hours)


def nonparametric(hours: np.ndarray, values: np.ndarray, params: dict) -> dict:
    """L5, M10, relative amplitude, onset, offset and the two stability measures.

    One row's worth of the non-parametric readouts, for one cell and one
    measurement. Interdaily stability and intradaily variability are always
    computed on hourly bins whatever ``bin_hours`` is, because that is how they
    are defined and how the workbench computes them; ``bin_hours`` sets the
    resolution of the folded day that L5, M10 and the onset template are read
    off.
    """
    bin_hours = float(params["bin_hours"])
    per_day = 24.0 / bin_hours if bin_hours > 0 else 0.0
    if bin_hours <= 0 or abs(per_day - round(per_day)) > 1e-9:
        raise ValueError(
            f"rhythms bin_hours must divide a 24-hour day, and {bin_hours!r} does not. "
            "A day that does not hold a whole number of bins cannot be folded, and "
            "rounding it would make interdaily stability subtly wrong rather than absent."
        )
    bins_per_day = int(round(per_day))

    hours = np.asarray(hours, dtype=float)
    hourly = _binned(hours, values, 1.0)
    binned = hourly if bin_hours == 1.0 else _binned(hours, values, bin_hours)
    profile = _folded_day(binned, bins_per_day)

    low_bins = max(1, int(round(float(params["low_window_hours"]) / bin_hours)))
    high_bins = max(1, int(round(float(params["high_window_hours"]) / bin_hours)))
    l5_index, l5 = _circular_window(profile, low_bins, "min")
    m10_index, m10 = _circular_window(profile, high_bins, "max")
    total = m10 + l5
    amplitude = (m10 - l5) / total if np.isfinite(total) and total != 0 else np.nan

    # The threshold is taken over the whole record rather than over the folded
    # day, which is the workbench's choice: a stretch is called active relative
    # to the recording it sits in, not relative to itself.
    measured = binned[np.isfinite(binned)]
    threshold = (float(np.percentile(measured, 100.0 - float(params["onset_threshold_percent"])))
                 if measured.size else np.nan)
    thresholded = np.where(np.isfinite(profile) & (profile >= threshold), 1.0, -1.0)
    off_bins = max(1, int(round(float(params["onset_off_hours"]) / bin_hours)))
    on_bins = max(1, int(round(float(params["onset_on_hours"]) / bin_hours)))
    onset_scores, offset_scores = _template_scores(thresholded, off_bins, on_bins)
    onset = _template_marker(onset_scores, bin_hours)
    offset = _template_marker(offset_scores, bin_hours)

    span = float(hours.max() - hours.min()) if len(hours) else np.nan
    days = span / 24.0
    duration = (offset - onset) % 24.0 if np.isfinite(onset) and np.isfinite(offset) else np.nan
    return {
        "m10": m10,
        "m10_onset_hour": m10_index * bin_hours,
        "l5": l5,
        "l5_onset_hour": l5_index * bin_hours,
        "relative_amplitude": amplitude,
        "onset_hour": onset,
        "offset_hour": offset,
        "active_duration_hours": duration,
        "interdaily_stability": _interdaily_stability(hourly, 24),
        "intradaily_variability": _intradaily_variability(hourly),
        "days_covered": days,
        "hours_binned": int(np.isfinite(hourly).sum()),
        "stability_underdetermined": days < float(params["min_days_for_stability"]),
        "onset_found": bool(np.isfinite(onset)),
    }


#: Three tables. The wide one is long by metric, not wide by metric - one row
#: per cell per measurement - so these column names are fixed rather than
#: data-dependent, and every one of them can be declared.
#:
#: Two prefixes carry the whole argument of the module and the labels have to
#: keep them apart. ``cosinor_`` is the fit at the period the analysis was told
#: to assume, usually 24 h; ``free_cosinor_`` is the fit at the period
#: Lomb-Scargle actually found. Quoting one while meaning the other is the
#: easiest mistake to make with this table, which is why no label here says
#: "amplitude" without saying which fit it came from.
#:
#: **Three amplitudes, and they are three different numbers.**
#:
#: ===============================  ==================================================
#: ``relative_amplitude``           ``(M10 - L5) / (M10 + L5)``: the busiest ten hours
#:                                  of the average day against the quietest five. What
#:                                  the field means by the term, and no fit involved.
#: ``cosinor_relative_amplitude``   fitted amplitude over the mean, at the period the
#:                                  fit was told to assume.
#: ``free_cosinor_relative_amplitude``  the same ratio at the period the search found.
#: ===============================  ==================================================
#:
#: The labels have to keep them apart on an axis as well as in this schema, so
#: none of the three is labelled "relative amplitude": each says what it divides.
PRODUCES = (
    # rhythms - one row per cell per measurement
    Column("metric", "Which measurement this row is about", "column name", "reference"),
    Column("observations", "Frames the fit used", "frames", "reference"),
    Column("span_hours", "Time from first to last frame used", "h", "reference"),
    Column("cycles_covered", "Cycles the recording covers at the assumed period", "count", "reference"),
    Column("mean_level", "Mean of the trace before detrending", "", "reference"),
    Column("detrend", "How the trend was removed", "name", "reference"),
    Column("fixed_period_hours", "Period the fit was told to assume", "h", "reference"),
    Column("cosinor_mesor", "Fitted mean at the assumed period", "", "rhythmic"),
    Column("cosinor_amplitude", "Fitted amplitude at the assumed period", "", "rhythmic"),
    Column("cosinor_relative_amplitude", "Fitted amplitude over the mean, assumed period", "ratio", "rhythmic"),
    Column("cosinor_peak_hour", "Time of day the fit peaks, assumed period", "h", "rhythmic"),
    Column("cosinor_r_squared", "Variation the assumed-period fit explains", "fraction", "fit"),
    Column("cosinor_p_value", "Chance of the assumed-period fit from noise", "p", "significant"),
    Column("cosinor_relative_amplitude_error", "Uncertainty on the relative amplitude", "ratio", "reference"),
    Column("lombscargle_period_hours", "Period carrying the most power", "h", "rhythmic"),
    Column("lombscargle_power", "Power at that period", "", "rhythmic"),
    Column("lombscargle_false_alarm", "Chance of that power from noise", "p", "significant"),
    Column("free_cosinor_mesor", "Fitted mean at the found period", "", "rhythmic"),
    Column("free_cosinor_amplitude", "Fitted amplitude at the found period", "", "rhythmic"),
    Column("free_cosinor_relative_amplitude", "Fitted amplitude over the mean, found period", "ratio", "rhythmic"),
    Column("free_cosinor_peak_hour", "Time of day the fit peaks, found period", "h", "rhythmic"),
    Column("free_cosinor_r_squared", "Variation the found-period fit explains", "fraction", "fit"),
    Column("free_cosinor_p_value", "Chance of the found-period fit from noise", "p", "significant"),
    Column("free_cosinor_relative_amplitude_error", "Uncertainty on the found-period relative amplitude", "ratio", "reference"),
    Column("cycles_covered_at_free_period", "Cycles the recording covers at the found period", "count", "reference"),
    Column("rhythmic_cosinor", "Called rhythmic by the cosinor fit", "0 or 1", "rhythmic"),
    Column("rhythmic_lombscargle", "Called rhythmic by Lomb-Scargle", "0 or 1", "rhythmic"),
    Column("rhythmic_both", "Called rhythmic by both tests", "0 or 1", "rhythmic"),
    Column("period_underdetermined", "Too few cycles to pin the period", "0 or 1", "invalid"),
    # The non-parametric block, on the same row as the fits it sits beside.
    #
    # Every hour here is hours *since the recording started*, not a time of day,
    # which is why the unit says so rather than saying "h". This package has no
    # clock: `cell_frame` carries hours since the first frame and the
    # configuration holds no recording start time. The arithmetic is unaffected
    # and the meaning is: 7.5 is "seven and a half hours in", and two movies
    # begun at different times of day cannot have their onsets compared at all.
    Column("m10", "Busiest ten hours", "", "rhythmic"),
    Column("m10_onset_hour", "Start of the busiest stretch", "h from start", "rhythmic"),
    Column("l5", "Quietest five hours", "", "rhythmic"),
    Column("l5_onset_hour", "Start of the quietest stretch", "h from start", "rhythmic"),
    Column("relative_amplitude", "Busiest against quietest", "ratio", "rhythmic"),
    Column("onset_hour", "Active phase starts", "h from start", "rhythmic"),
    Column("offset_hour", "Active phase ends", "h from start", "rhythmic"),
    Column("active_duration_hours", "Active phase length", "h", "rhythmic"),
    Column("interdaily_stability", "How alike one day is to the next", "ratio", "rhythmic"),
    Column("intradaily_variability", "How broken up the daily pattern is", "ratio", "rhythmic"),
    Column("days_covered", "Days of recording behind these", "days", "reference"),
    Column("hours_binned", "Hourly bins with a measurement in them", "bins", "reference"),
    Column("stability_underdetermined",
           "Too few days for the stability measure", "0 or 1", "reference"),
    Column("onset_found", "A template match was found", "0 or 1", "reference"),
    # rhythms_population - one row per measurement per group of cells
    Column("population", "Which group of cells", "name", "reference"),
    Column("cells", "Cells in the group", "count", "reference"),
    Column("vector_length", "How tightly the peaks agree", "0-1", "rhythmic"),
    Column("mean_peak_hour", "Time of day the group peaks", "h", "rhythmic"),
    Column("rayleigh_p_value", "Chance of that agreement from scattered peaks", "p", "significant"),
    # rhythms_null - one row per measurement
    Column("surrogates", "Noise traces tested per cell", "count", "reference"),
    Column("null_model", "How the noise traces were made", "name", "reference"),
    Column("cells_tested", "Cells the null was run on", "count", "reference"),
    Column("false_positive_rate_cosinor", "Noise traces the cosinor test called rhythmic", "fraction", "arrhythmic"),
    Column("false_positive_rate_lombscargle", "Noise traces Lomb-Scargle called rhythmic", "fraction", "arrhythmic"),
    Column("false_positive_rate_both", "Noise traces both tests called rhythmic", "fraction", "arrhythmic"),
    Column("observed_rate_cosinor", "Real cells the cosinor test called rhythmic", "fraction", "rhythmic"),
    Column("observed_rate_both", "Real cells both tests called rhythmic", "fraction", "rhythmic"),
    Column("excess_over_null_both", "How far the real rate beats matched noise", "fraction", "significant"),
)


#: The two population tables are optional because they only exist when there
#: was something to summarise: no cell fitted, no population row.
WRITES = (
    Output("rhythms",            grain=("identity", "metric")),
    Output("rhythms_population", grain=("metric", "population"), optional=True),
    Output("rhythms_null",       grain=("metric",),              optional=True),
)


@register_derived(
    name="rhythms",
    description="Cosinor and Lomb-Scargle rhythm fits per cell, per measurement",
    needs_columns=("identity", "frame_index", "hours"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("rhythms")}
    low, high = params["period_search_hours"]
    periods = np.arange(float(low), float(high) + 1e-9, float(params["period_step_hours"]))
    metrics = [m for m in params["metrics"] if m in cell_frame.columns]
    fixed_period = float(params["fixed_period_hours"])
    min_observations = int(params["min_observations"])

    generator = np.random.default_rng(int(params["null_seed"]))
    surrogates_per_cell = int(params["null_surrogates_per_cell"])
    alpha_for_null = float(params["rhythmic_alpha"])

    rows: list[dict] = []
    null_rows: list[dict] = []
    for metric in metrics:
        for identity, group in cell_frame.groupby("identity", sort=True):
            usable = group[["hours", metric]].dropna()
            if len(usable) < min_observations:
                continue
            hours = usable["hours"].to_numpy(float)
            values = usable[metric].to_numpy(float)
            if np.allclose(values, values[0]):
                continue
            detrended = _detrend(hours, values, params["detrend"])
            span = float(hours.max() - hours.min())

            for replicate in range(surrogates_per_cell):
                fake = _surrogate(detrended, params["null_model"], generator)
                null_fit = {**cosinor(hours, fake, fixed_period), **lomb_scargle(hours, fake, periods)}
                if not null_fit:
                    continue
                null_rows.append(
                    {
                        "metric": metric,
                        "identity": int(identity),
                        "replicate": replicate,
                        "null_model": params["null_model"],
                        "cosinor_p_value": null_fit.get("cosinor_p_value", np.nan),
                        "lombscargle_false_alarm": null_fit.get("lombscargle_false_alarm", np.nan),
                        "lombscargle_period_hours": null_fit.get("lombscargle_period_hours", np.nan),
                    }
                )

            row = {
                "identity": int(identity),
                "metric": metric,
                "observations": int(len(usable)),
                "span_hours": span,
                "cycles_covered": span / fixed_period,
                "mean_level": float(values.mean()),
                "detrend": params["detrend"],
                "fixed_period_hours": fixed_period,
            }
            reference = float(np.mean(np.abs(values)))
            # On the trace itself, not the detrended one: L5 and M10 are levels
            # and their ratio is the whole measure, so removing the trend first
            # would divide two numbers straddling zero.
            row.update(nonparametric(hours, values, params))
            row.update(cosinor(hours, detrended, fixed_period, reference))
            row.update(lomb_scargle(hours, detrended, periods))
            if "lombscargle_period_hours" in row:
                row.update(
                    {
                        f"free_{key}": value
                        for key, value in cosinor(
                            hours, detrended, row["lombscargle_period_hours"], reference
                        ).items()
                    }
                )
                row["cycles_covered_at_free_period"] = span / row["lombscargle_period_hours"]
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return {"rhythms": table}

    alpha = float(params["rhythmic_alpha"])
    minimum_cycles = float(params["min_cycles_for_confident_period"])
    table["rhythmic_cosinor"] = table["cosinor_p_value"] < alpha
    table["rhythmic_lombscargle"] = table.get("lombscargle_false_alarm", 1.0) < alpha
    table["rhythmic_both"] = table["rhythmic_cosinor"] & table["rhythmic_lombscargle"]
    table["period_underdetermined"] = table["cycles_covered"] < minimum_cycles

    population_rows: list[dict] = []
    for metric, group in table.groupby("metric"):
        for label, subset in (
            ("all_tested", group),
            ("rhythmic_by_both_tests", group[group["rhythmic_both"]]),
        ):
            result = rayleigh(subset["cosinor_peak_hour"].to_numpy(), fixed_period)
            if result:
                population_rows.append({"metric": metric, "population": label, **result})

    output = {"rhythms": table}
    if population_rows:
        output["rhythms_population"] = pd.DataFrame(population_rows)
    if null_rows:
        null = pd.DataFrame(null_rows)
        null["rhythmic_cosinor"] = null["cosinor_p_value"] < alpha_for_null
        null["rhythmic_lombscargle"] = null["lombscargle_false_alarm"] < alpha_for_null
        null["rhythmic_both"] = null["rhythmic_cosinor"] & null["rhythmic_lombscargle"]
        summary = (
            null.groupby("metric")
            .agg(
                surrogates=("replicate", "size"),
                false_positive_rate_cosinor=("rhythmic_cosinor", "mean"),
                false_positive_rate_lombscargle=("rhythmic_lombscargle", "mean"),
                false_positive_rate_both=("rhythmic_both", "mean"),
            )
            .reset_index()
        )
        summary["null_model"] = params["null_model"]
        observed = (
            table.groupby("metric")
            .agg(
                cells_tested=("identity", "size"),
                observed_rate_cosinor=("rhythmic_cosinor", "mean"),
                observed_rate_both=("rhythmic_both", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(observed, on="metric", how="outer")
        # The number that matters: how much more often the real traces are
        # called rhythmic than matched noise with the same drift and smoothness.
        summary["excess_over_null_both"] = (
            summary["observed_rate_both"] - summary["false_positive_rate_both"]
        )
        output["rhythms_null"] = summary
    return output
