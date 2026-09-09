"""Figure 20: does a significant rhythm repeat in the next detected cycle?

Only cells called rhythmic by the configured primary Circadian Workbench test
enter the comparison. Each cell supplies its own detected period: its first and
second sufficiently observed cycles are scaled to 0--100%, so an ultradian cell and a
circadian-like cell can be compared without pretending that either is 24 h.

    python analysis/figures/20_second_verse.py <run>
    ... --metrics area_px        compare a different fitted measurement
    ... --cells 7,11,16          show named eligible cells only
    ... --profile-bins 36        sample each scaled cycle more finely
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from _metrics import describe
from _options import commas
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import common


METHOD_LABELS = {
    "lomb": "Lomb-Scargle periodogram",
    "chi_square": "Enright-Sokolove periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}


def _as_bool(values: pd.Series) -> pd.Series:
    """Read old CSV booleans as strictly as current boolean columns."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _period_class(period: pd.Series, params: dict) -> pd.Series:
    low, high = map(float, params.get("circadian_band_hours", [20.0, 28.0]))
    return pd.Series(
        np.select(
            [period < low, period <= high, np.isfinite(period)],
            ["ultradian", "circadian-like", "infradian"],
            default="unknown",
        ),
        index=period.index,
    )


def _primary_rows(fits: pd.DataFrame, params: dict) -> pd.DataFrame:
    """One selected period and one rhythm-test result per cell, including legacy runs."""
    source = fits.copy()
    modern = "best_period_hours" in source
    primary = str(params.get("primary_rhythm_test", "lomb"))
    estimator = str(params.get("period_estimation_method", primary))
    primary_label = workbench.PERIOD_METHODS.get(
        primary, {"label": METHOD_LABELS.get(primary, primary)})["label"]
    if modern:
        period = pd.to_numeric(source["best_period_hours"], errors="coerce")
        p_value = pd.to_numeric(source.get("best_p_value"), errors="coerce")
        alpha = pd.to_numeric(
            source.get("best_alpha", pd.Series(params["rhythmic_alpha"], index=source.index)),
            errors="coerce",
        )
        rhythmic = _as_bool(source["rhythmic"])
        status = source.get(
            "rhythm_status",
            pd.Series(np.where(rhythmic, "rhythmic", "arrhythmic"), index=source.index),
        )
        estimator_label = source.get(
            "best_method_label",
            pd.Series(
                workbench.PERIOD_METHODS.get(
                    estimator, {"label": METHOD_LABELS.get(estimator, estimator)}
                )["label"],
                index=source.index,
            ),
        )
        at_edge = _as_bool(source.get(
            "best_period_at_search_edge", pd.Series(False, index=source.index)
        ))
        period_class = source.get("best_period_class", _period_class(period, params))
        cycles = pd.to_numeric(
            source.get("cycles_covered_at_best_period", source.get("span_hours") / period),
            errors="coerce",
        )
    else:
        if primary != "lomb" or estimator != "lomb":
            raise SystemExit(
                "This older run only stored the Lomb-Scargle period and verdict; rerun rhythms "
                "before selecting another estimator or rhythm test."
            )
        period = pd.to_numeric(source.get("lombscargle_period_hours"), errors="coerce")
        p_value = pd.to_numeric(source.get("lombscargle_false_alarm"), errors="coerce")
        alpha = pd.Series(float(params.get("rhythmic_alpha", 0.05)), index=source.index)
        rhythmic = _as_bool(source.get(
            "rhythmic_lombscargle", pd.Series(False, index=source.index)
        ))
        status = pd.Series(np.where(rhythmic, "rhythmic", "arrhythmic"), index=source.index)
        estimator_label = pd.Series(METHOD_LABELS["lomb"], index=source.index)
        low, high = map(float, params["period_search_hours"])
        tolerance = max(0.1, 0.005 * (high - low))
        at_edge = (period <= low + tolerance) | (period >= high - tolerance)
        period_class = _period_class(period, params)
        cycles = pd.to_numeric(source.get("span_hours"), errors="coerce") / period

    return pd.DataFrame({
        "identity": pd.to_numeric(source["identity"], errors="coerce").astype("Int64"),
        "metric": source["metric"].astype(str),
        "primary_method": primary,
        "primary_method_label": primary_label,
        "period_estimation_method": estimator,
        "period_estimation_method_label": estimator_label,
        "rhythm_status": status,
        "rhythmic": rhythmic,
        "detected_period_hours": period,
        "period_class": period_class,
        "p_value": p_value,
        "alpha": alpha,
        "period_at_search_edge": at_edge,
        "cycles_in_recording": cycles,
        "detrend": source.get("detrend", pd.Series("unknown", index=source.index)),
        "workbench_version": source.get(
            "workbench_version", pd.Series("legacy run", index=source.index)
        ),
    })


def _fallback_traces(
    frame: pd.DataFrame,
    metric: str,
    detrend: str,
    detrend_window_hours: float,
    detrend_options: dict | None = None,
) -> pd.DataFrame:
    """Recreate a pre-table run through the same Circadian Workbench detrend."""
    if metric not in frame:
        raise SystemExit(f"--metrics {metric} is not in cell_frame.csv")
    rows = []
    for identity, group in frame[["identity", "hours", metric]].dropna().groupby(
        "identity", sort=True
    ):
        group = group.sort_values("hours")
        hours = group["hours"].to_numpy(float)
        values = group[metric].to_numpy(float)
        scaled = rhythm_panels.detrended_z(
            hours, values, method=detrend,
            window_hours=detrend_window_hours,
            detrend_options=detrend_options,
        )
        rows.extend({
            "identity": int(identity), "metric": metric, "hours": float(hour),
            "detrended_z": float(value), "trace_source": "legacy trace reconstructed for display",
        } for hour, value in zip(hours, scaled))
    return pd.DataFrame(rows)


def _cycle_profile(phases: np.ndarray, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Circular linear interpolation onto shared within-cycle positions."""
    order = np.argsort(phases)
    phases, values = phases[order], values[order]
    unique, inverse = np.unique(phases, return_inverse=True)
    if len(unique) != len(phases):
        totals = np.bincount(inverse, weights=values)
        values = totals / np.bincount(inverse)
        phases = unique
    return np.interp(
        positions,
        np.concatenate([phases - 1.0, phases, phases + 1.0]),
        np.tile(values, 3),
    )


def _peak_fraction(profile: np.ndarray, positions: np.ndarray) -> float:
    """Peak of a three-bin circular moving average, without a sinusoid fit."""
    smooth = (np.roll(profile, 1) + profile + np.roll(profile, -1)) / 3.0
    return float(positions[int(np.nanargmax(smooth))])


def _selected_identities(data: pd.DataFrame, requested: str) -> list[int]:
    text = str(requested).strip().lower()
    if text in {"", "all"}:
        return [int(value) for value in data["identity"]]
    try:
        count = int(text)
    except ValueError:
        wanted = [int(value) for value in commas(requested)]
        missing = sorted(set(wanted) - set(data["identity"].astype(int)))
        if missing:
            raise SystemExit(
                "--cells includes identities that are not eligible for this comparison: "
                + ", ".join(map(str, missing))
            )
        return wanted
    if count < 1:
        raise SystemExit("--cells needs a positive count, comma-separated identities, or all")
    return [int(value) for value in data.sort_values(
        ["minimum_cycle_coverage", "frames_cycle_1"], ascending=False
    )["identity"].head(count)]


@figure(
    number=20,
    slug="second-verse",
    summary="repeatability of significant rhythms at each cell's own detected period",
    title="Does each rhythmic cell repeat its detected cycle?",
    reads=(
        Table("rhythms.csv", module="rhythms"),
        Table("rhythm_traces.csv", module="rhythms", optional=True),
        Table("cell_frame.csv", module="rhythms", optional=True),
    ),
    panels=(
        Panel("profiles", common.trace,
              min_width_inches=14.0, min_height_inches=5.2,
              title="Median waveform after aligning every cell to its cycle-1 peak\n"
                    "Cycle 2 stays aligned only when the detected pattern repeats"),
        Panel("peak_agreement", common.scatter,
              min_width_inches=7.0, min_height_inches=5.2,
              title="Repeatability across detected periods\n"
                    "Zero shift and correlation near 1 mean the cycle repeats"),
        Panel("peak_shift", common.histogram,
              min_width_inches=7.0, min_height_inches=5.2,
              title="How far does the peak move between cycles?"),
    ),
    options=(
        Option("metrics", default="corrected_mean", cast=str, metavar="COL",
               help="which rhythm-tested measurement is compared cycle to cycle"),
        Option("cells", default="all", cast=str),
        Option("bins", default=12),
        Option("profile_bins", default=24),
        Option("min_coverage", default=0.8),
    ),
    grammar="paired cycle profiles phase-repeat scatter and shift histogram",
)
def build(ctx: FigureContext) -> FigureResult:
    metric = ctx.option("metrics")
    params = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    fitted = ctx.table("rhythms.csv")
    fitted = fitted[fitted["metric"] == metric].copy()
    if fitted.empty:
        available = ", ".join(sorted(set(ctx.table("rhythms.csv")["metric"])))
        raise SystemExit(f"--metrics {metric} was not rhythm-tested; available: {available}")
    tests = _primary_rows(fitted, params)

    stored_traces = ctx.optional_table("rhythm_traces.csv")
    if stored_traces is not None:
        traces = stored_traces[stored_traces["metric"] == metric].copy()
        if traces.empty:
            raise SystemExit(f"rhythm_traces.csv has no rows for --metrics {metric}")
        traces["trace_source"] = "Circadian Workbench detrended trace"
    else:
        frame = ctx.optional_table("cell_frame.csv")
        if frame is None:
            raise SystemExit(
                "This run has neither rhythm_traces.csv nor cell_frame.csv; rerun rhythms."
            )
        detrends = tests["detrend"].dropna().astype(str)
        detrend = detrends.mode().iloc[0] if not detrends.empty else str(params["detrend"])
        traces = _fallback_traces(
            frame, metric, detrend,
            float(params["detrend_window_hours"]),
            detrend_options=params,
        )

    profile_bins = int(ctx.option("profile_bins"))
    min_coverage = float(ctx.option("min_coverage"))
    if profile_bins < 4:
        raise SystemExit("--profile-bins needs at least 4 positions")
    if not 0 < min_coverage <= 1:
        raise SystemExit("--min-coverage must be greater than 0 and no more than 1")
    positions = (np.arange(profile_bins, dtype=float) + 0.5) / profile_bins
    aligned_positions = (
        np.arange(profile_bins, dtype=float) - profile_bins // 2
    ) / profile_bins

    eligibility = tests.copy()
    eligibility["included_in_plot"] = False
    eligibility["exclusion_reason"] = "not significantly rhythmic by the primary test"
    invalid_period = ~np.isfinite(eligibility["detected_period_hours"]) | (
        eligibility["detected_period_hours"] <= 0
    )
    eligibility.loc[invalid_period, "exclusion_reason"] = "no valid detected period"
    eligibility.loc[
        eligibility["rhythmic"] & eligibility["period_at_search_edge"], "exclusion_reason"
    ] = "detected period touches the search boundary"
    initial = eligibility[
        eligibility["rhythmic"] & ~eligibility["period_at_search_edge"] & ~invalid_period
    ]

    cell_rows: list[dict] = []
    profile_rows: list[dict] = []
    for test in initial.itertuples(index=False):
        group = traces[traces["identity"] == test.identity].dropna(
            subset=["hours", "detrended_z"]
        ).sort_values("hours")
        if group.empty:
            eligibility.loc[
                eligibility["identity"] == test.identity, "exclusion_reason"
            ] = "no detrended trace"
            continue
        hours = group["hours"].to_numpy(float)
        values = group["detrended_z"].to_numpy(float)
        period = float(test.detected_period_hours)
        relative = hours - float(hours.min())
        intervals = np.diff(np.unique(hours))
        intervals = intervals[intervals > 0]
        interval = float(np.median(intervals)) if len(intervals) else np.nan
        expected = period / interval if np.isfinite(interval) and interval > 0 else np.nan
        profiles = []
        counts = []
        coverages = []
        for cycle in (0, 1):
            keep = (relative >= cycle * period) & (relative < (cycle + 1) * period)
            counts.append(int(np.count_nonzero(keep)))
            coverage = min(counts[-1] / expected, 1.0) if np.isfinite(expected) else 0.0
            coverages.append(float(coverage))
            if counts[-1] < 4 or coverage < min_coverage:
                profiles.append(None)
                continue
            phase = (relative[keep] - cycle * period) / period
            profiles.append(_cycle_profile(phase, values[keep], positions))
        if profiles[0] is None or profiles[1] is None:
            eligibility.loc[
                eligibility["identity"] == test.identity, "exclusion_reason"
            ] = f"fewer than two cycles with {min_coverage:.0%} measurement coverage"
            continue

        peak_1 = _peak_fraction(profiles[0], positions)
        peak_2 = _peak_fraction(profiles[1], positions)
        shift = abs(((peak_2 - peak_1 + 0.5) % 1.0) - 0.5)
        signed_shift = ((peak_2 - peak_1 + 0.5) % 1.0) - 0.5
        correlation = float(np.corrcoef(profiles[0], profiles[1])[0, 1])
        peak_index_1 = int(np.argmin(np.abs(positions - peak_1)))
        alignment = profile_bins // 2 - peak_index_1
        aligned_profiles = [np.roll(profile, alignment) for profile in profiles]
        cell_rows.append({
            "identity": int(test.identity), "metric": metric,
            "primary_method": test.primary_method,
            "primary_method_label": test.primary_method_label,
            "period_estimation_method": test.period_estimation_method,
            "period_estimation_method_label": test.period_estimation_method_label,
            "detected_period_hours": period, "period_class": test.period_class,
            "p_value": test.p_value, "alpha": test.alpha,
            "cycle_1_peak_percent": peak_1 * 100.0,
            "cycle_2_peak_percent": peak_2 * 100.0,
            "signed_peak_shift_percent": signed_shift * 100.0,
            "absolute_peak_shift_percent": shift * 100.0,
            "waveform_correlation": correlation,
            "frames_cycle_1": counts[0], "frames_cycle_2": counts[1],
            "cycle_1_coverage": coverages[0], "cycle_2_coverage": coverages[1],
            "minimum_cycle_coverage": min(coverages),
            "cycle_origin_hour": float(hours.min()),
        })
        for cycle, profile in enumerate(aligned_profiles, start=1):
            profile_rows.extend({
                "identity": int(test.identity), "metric": metric,
                "detected_period_hours": period, "cycle": cycle,
                "position_from_cycle_1_peak_percent": float(position * 100.0),
                "detrended_signal_sd": float(value),
            } for position, value in zip(aligned_positions, profile))
        mask = eligibility["identity"] == test.identity
        eligibility.loc[mask, "included_in_plot"] = True
        eligibility.loc[mask, "exclusion_reason"] = ""

    cells = pd.DataFrame(cell_rows)
    profiles = pd.DataFrame(profile_rows)
    if cells.empty:
        raise SystemExit(
            "No cells passed the primary rhythm test and supplied two sufficiently observed "
            "detected cycles."
        )
    selected = _selected_identities(cells, ctx.option("cells"))
    cells = cells[cells["identity"].isin(selected)].copy()
    profiles = profiles[profiles["identity"].isin(selected)].copy()
    eligibility["selected_by_cells_option"] = eligibility["identity"].isin(selected)

    population = profiles.groupby(
        ["metric", "cycle", "position_from_cycle_1_peak_percent"], as_index=False
    ).agg(
        median_signal_sd=("detrended_signal_sd", "median"),
        lower_quartile_signal_sd=("detrended_signal_sd", lambda values: values.quantile(0.25)),
        upper_quartile_signal_sd=("detrended_signal_sd", lambda values: values.quantile(0.75)),
        cells=("identity", "nunique"),
    )
    panels = ctx.panels()
    if panels.keys == ["profiles", "peak_agreement", "peak_shift"]:
        fig, axes = ctx.grid_layout(
            panels,
            {
                "profiles": (0, 0, 1, 2),
                "peak_agreement": (1, 0, 1, 1),
                "peak_shift": (1, 1, 1, 1),
            },
            top_inches=2.55,
            bottom_inches=2.20,
            horizontal_gap_inches=2.1,
            vertical_gap_inches=1.55,
        )
    else:
        fig, axes = ctx.layout(panels, gap_inches=1.3)

    if "profiles" in axes:
        colours = {1: "morphology", 2: "reporter"}
        for cycle in (1, 2):
            part = population[population["cycle"] == cycle]
            colour = ctx.theme.colour(colours[cycle])
            axes["profiles"].fill_between(
                part["position_from_cycle_1_peak_percent"],
                part["lower_quartile_signal_sd"],
                part["upper_quartile_signal_sd"],
                color=colour, alpha=0.18, linewidth=0,
            )
            common.trace(
                axes["profiles"], part["position_from_cycle_1_peak_percent"],
                part["median_signal_sd"],
                ctx.theme, look=common.Look(colour=colour), label=f"Cycle {cycle}",
                x_label="Position from cycle-1 peak (% of own detected period)",
                y_label="Detrended signal (within-cell SD)",
            )
        axes["profiles"].axvline(
            0, color=ctx.theme.colour("reference"),
            linewidth=ctx.theme.stroke("guide"), linestyle=(0, (5, 3)),
        )
        axes["profiles"].set_xlim(-50, 50)
        axes["profiles"].set_xticks([-50, -25, 0, 25, 50])

    if "peak_agreement" in axes:
        import matplotlib

        look = common.Look(cmap=matplotlib.colormaps[ctx.theme["diverging_cmap"]])
        result = common.scatter(
            axes["peak_agreement"], cells["detected_period_hours"],
            cells["signed_peak_shift_percent"], ctx.theme, look=look,
            colours=cells["waveform_correlation"], alpha=0.78,
        )
        result.extra["handle"].set_clim(-1.0, 1.0)
        axes["peak_agreement"].axhline(
            0, color=ctx.theme.colour("reference"),
            linewidth=ctx.theme.stroke("guide"), linestyle=(0, (5, 3)),
        )
        axes["peak_agreement"].set(
            ylim=(-50, 50),
            xlabel="Detected period (h)",
            ylabel="Cycle-2 peak shift (% of own period)",
        )
        axes["peak_agreement"].set_yticks([-50, -25, 0, 25, 50])
        common.inset_colour_bar(
            axes["peak_agreement"], result.extra["handle"], ctx.theme,
            label="Cycle waveform correlation", ticks=[-1, 0, 1],
        )
        axes["peak_agreement"]._semantic_legend_handled = True

    if "peak_shift" in axes:
        edges = np.linspace(0.0, 50.0, int(ctx.option("bins")) + 1)
        common.histogram(
            axes["peak_shift"], cells["absolute_peak_shift_percent"], ctx.theme,
            bins=edges, role="reporter",
            x_label="Absolute peak shift (% of own period)", y_label="Cells",
        )
        axes["peak_shift"].set_xlim(0, 50)
        median_shift = float(cells["absolute_peak_shift_percent"].median())
        common.reference_lines(
            axes["peak_shift"],
            [common.Mark(median_shift, f"Median {median_shift:.0f}%", role="ink")],
            ctx.theme,
        )

    tested = len(tests)
    significant = int(tests["rhythmic"].sum())
    eligible = int(eligibility["included_in_plot"].sum())
    method = str(tests["primary_method_label"].dropna().mode().iloc[0])
    estimator = str(tests["period_estimation_method_label"].dropna().mode().iloc[0])
    alpha = tests["alpha"].dropna()
    threshold = f"p < {float(alpha.iloc[0]):g}" if not alpha.empty else "the configured threshold"
    period_low = float(cells["detected_period_hours"].min())
    period_high = float(cells["detected_period_hours"].max())
    selected_suffix = f"; {len(cells)} shown" if len(cells) != eligible else ""
    subtitle = (
        f"{significant} of {tested} cells were significant by {method} ({threshold}); "
        f"{eligible} supplied two {period_low:g}-{period_high:g} h cycles estimated by "
        f"{estimator}, each at least "
        f"{min_coverage:.0%} observed"
        f"{selected_suffix}."
    )
    metric_label = describe(metric).label
    statistics = eligibility[[
        "identity", "metric", "primary_method", "primary_method_label",
        "period_estimation_method", "period_estimation_method_label", "rhythm_status",
        "rhythmic", "detected_period_hours", "period_class", "p_value", "alpha",
        "period_at_search_edge", "cycles_in_recording", "included_in_plot",
        "selected_by_cells_option", "exclusion_reason", "detrend", "workbench_version",
    ]].copy()
    return FigureResult(
        figure=fig,
        axes=list(axes.values()),
        figure_data=population,
        auxiliary={
            "cell_repeatability.csv": cells,
            "cycle_profiles.csv": profiles,
            "statistics.csv": statistics,
        },
        subtitle=subtitle,
        footnote=(
            "Cycle 1 starts at each cell's first usable observation; both cycles are scaled to "
            "that cell's detected period and aligned to its cycle-1 peak. Shaded bands are cell "
            "quartiles. Peaks are "
            "the maxima of three-bin circular moving averages, not cosinor fits. Search-boundary "
            f"periods and cycles below {min_coverage:.0%} measurement coverage are excluded. "
            "This is a descriptive repeatability check; the rhythm test and period use the full record."
        ),
        title_fields={"metric": metric_label},
        readme=(
            "The primary Circadian Workbench rhythm test decides which cells are eligible. "
            "No fixed day length is used: every included cell is folded at its own significant "
            "detected period, and only its first two sufficiently observed cycles are compared. "
            "The source rhythm test is inferential; cycle repeatability is descriptive."
        ),
        console=(
            f"Compared {len(cells)} cells at their own significant {period_low:g}-{period_high:g} h periods."
        ),
    )


if __name__ == "__main__":
    run_figure("second-verse")
